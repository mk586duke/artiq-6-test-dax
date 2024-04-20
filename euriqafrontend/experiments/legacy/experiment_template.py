"""Full Template.
"""
import logging
import time
from collections import defaultdict

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import numpy as np
import oitg.fitting as fit
from artiq.experiment import TerminationRequested
from artiq.language.core import delay
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import parallel
from artiq.language.core import rpc
from artiq.language.types import TNone
from artiq.language.units import MHz
from artiq.language.units import ms
from artiq.language.units import us

from euriqafrontend.modules.cw_lasers import DopplerCooling
from euriqafrontend.modules.cw_lasers import PumpDetect
from euriqafrontend.modules.cw_lasers import SSCooling
from euriqafrontend.modules.dac import SandiaDAC
from euriqafrontend.modules.mw import Microwave
from euriqafrontend.modules.pmt import PMTArray
from euriqafrontend.modules.raman import GlobalDDS
from euriqafrontend.modules.raman import IndCollectiveDDS
from euriqafrontend.modules.raman import IndSwitchnetDDS

_LOGGER = logging.getLogger(__name__)


class BasicExperimentTemplate(artiq_env.EnvExperiment):
    """Experiment Template (Full)
    Will loop over some parameter and do doppler -> SS cooling -> pump -> YOUR EXPERIMENT -> detect
    Includes basic data logging, plotting, and fitting
    """

    _DATA_FOLDER = "template"
    _APPLET_NAME = "Insert Applet Name Here"
    _APPLET_GROUP_NAME = "Insert SubGroup Name for Applet Here"
    _FIT_TYPE = fit.cos_fft

    kernel_invariants = {
        "scan_values",
        "num_shots",
        "detect_time",
        "detect_thresh",
        "optical_pump_time",
        "doppler_cooling_time",
        "ss_cooling_time",
        "pmt_array",
        "num_pmts",
    }

    applet_stream_cmd = (
        "$python -m euriqafrontend.applets.plot_multi" + " "
    )  # White space is required

    def build(self):
        """Initialize experiment & variables."""
        # devices
        self.setattr_device("core")
        self.setattr_device("ccb")
        self.setattr_device("oeb")
        self.setattr_device("scheduler")

        # arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0 * us,
                    stop=20 * ms,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="us",
                global_min=0,
            ),
        )

        self.setattr_argument(
            "num_shots", artiq_env.NumberValue(default=100, step=1, ndecimals=0)
        )
        self.setattr_argument(
            "detect_time", artiq_env.NumberValue(default=200 * us, unit="us")
        )
        self.setattr_argument(
            "detect_thresh",
            artiq_env.NumberValue(default=1, unit="counts", ndecimals=0, scale=1),
        )
        self.setattr_argument(
            "optical_pump_time", artiq_env.NumberValue(default=0.5 * ms, unit="ms")
        )
        self.setattr_argument(
            "doppler_cooling_time", artiq_env.NumberValue(default=2 * ms, unit="ms")
        )
        self.setattr_argument(
            "ss_cooling_time", artiq_env.NumberValue(default=2 * ms, unit="ms")
        )

        self.doppler_cooling = DopplerCooling(self)
        self.ss_cooling = SSCooling(self)
        self.pump_detect = PumpDetect(self)
        self.pmt_array = PMTArray(self)
        self.mw = Microwave(self)
        # self.sandia_box = SandiaDAC(self)
        self.ind_collective_dds = IndCollectiveDDS(self)
        self.ind_switchnetwork_dds = IndSwitchnetDDS(self)
        self.global_raman_dds = GlobalDDS(self)

    def prepare(self):
        """Pre-calculate any values."""
        # Run prepare method of all the imported modules
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        # calculate scan parameters
        self.scan_values = [t for t in self.scan_range]
        self.num_steps = len(self.scan_values)
        self.num_pmts = self.pmt_array.num_active

    @host_only
    def run(self):
        """Start the experiment on the host."""
        raw_counts_empty = np.full(
            (self.num_pmts, self.num_shots, self.num_steps), np.nan
        )
        avg_thresh_empty = np.full((self.num_pmts, self.num_steps), np.nan)

        self.set_dataset(self._DATA_FOLDER + ".x_values", self.scan_values)
        self.set_dataset(self._DATA_FOLDER + ".raw_counts", raw_counts_empty)
        self.set_dataset(
            self._DATA_FOLDER + ".avg_thresh", avg_thresh_empty, broadcast=True
        )
        self.set_dataset(self._DATA_FOLDER + ".fit_x", np.nan, broadcast=True)
        self.set_dataset(
            self._DATA_FOLDER + ".fit_y", np.full(self.num_pmts, np.nan), broadcast=True
        )

        self.ccb.issue(
            "create_applet",
            name=self._APPLET_NAME,
            command=self.applet_stream_cmd
            + "--x "
            + self._DATA_FOLDER
            + ".x_values "
            + "--y-names "
            + self._DATA_FOLDER
            + ".avg_thresh "
            + "--x-fit "
            + self._DATA_FOLDER
            + ".fit_x "
            + "--y-fits "
            + self._DATA_FOLDER
            + ".fit_y ",
            group=self._APPLET_GROUP_NAME,
        )

        # RUN EXPERIMENT
        try:
            self.kn_initialize()
            self.kn_run()
        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")
            self.kn_idle()

    @kernel
    def kn_initialize(self):
        """Initialize the core and all devices."""
        self.core.reset()
        self.core.break_realtime()
        self.oeb.off()
        self.doppler_cooling.init()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.pump_detect.init_to_detect()

        ###################################
        # INSERT CUSTOM INITIALIZATIONS HERE
        ###################################

    @kernel
    def kn_run(self):
        """Run the experiment on the core device."""

        # Loop over main experimental scan parameters
        for istep in range(len(self.scan_values)):
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()

            # Loop over to gather statistics
            for ishot in range(self.num_shots):

                # Doppler Cooling
                self.doppler_cooling.pulse(self.doppler_cooling_time)

                # Second Stage Cooling
                self.ss_cooling.prepare_ss_cool()
                self.ss_cooling.pulse(self.ss_cooling_time)

                # Slow pump to 0 state
                self.pump_detect.switch_to_slow_pump()
                self.pump_detect.pulse(self.optical_pump_time)

                ##############################
                # MAIN EXPERIMENT HERE
                # use self.scan_values[istep] as your iteration value
                ##############################

                # Detect Ions
                counts = [0] * self.num_pmts
                counts = self.kn_detect(counts)
                self.save_counts(ishot=ishot, istep=istep, counts=counts)

            # threshold raw counts, save data, and push it to the plot applet
            self.threshold_data(istep)

    @kernel
    def kn_detect(self, buffer):
        """Readout PMT counts and return the value."""
        self.pump_detect.switch_to_detect()

        with parallel:
            stopcounter_mu = self.pmt_array.gate_rising(self.detect_time)
            self.pump_detect.pulse(self.detect_time)

        delay(50 * us)
        buffer = self.pmt_array.count(up_to_time_mu=stopcounter_mu, buffer=buffer)
        return buffer

    @rpc(flags={"async"})
    def save_counts(self, ishot, istep, counts):
        np_counts = np.array(counts, ndmin=3).T
        self.mutate_dataset(
            self._DATA_FOLDER + ".raw_counts",
            (
                (0, self.num_pmts, 1),
                (ishot, None, self.num_shots),
                (istep, None, self.num_steps),
            ),
            np_counts,
        )

    @rpc(flags={"async"})
    def threshold_data(self, istep):
        counts = np.array(self.get_dataset(self._DATA_FOLDER + ".raw_counts"))
        thresh = np.mean(counts[:, :, istep] > self.detect_thresh, axis=1).tolist()
        np_thresh = np.array(thresh, ndmin=2).T
        self.mutate_dataset(
            self._DATA_FOLDER + ".avg_thresh",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
            np_thresh,
        )

    @kernel
    def kn_idle(self) -> TNone:
        """Reset all used devices to default state."""
        self.core.break_realtime()
        self.doppler_cooling.idle()
        self.pump_detect.off()

        ##################################
        # INSERT CUSTOM IDLE COMMANDS HERE
        #################################

    def analyze(self):
        """Analyze and Fit data"""
        """Threshold values and analyze."""
        fit_plot_points = 500

        x = np.array(self.get_dataset(self._DATA_FOLDER + ".x_values"))
        y = self.get_dataset(self._DATA_FOLDER + ".avg_thresh")

        p_all = defaultdict(list)
        p_error_all = defaultdict(list)
        y_fit_all = np.full((y.shape[0], fit_plot_points), np.nan)

        for iy in range(y.shape[0]):
            p, p_error, x_fit, y_fit = self._FIT_TYPE.fit(
                x, y[iy, :], evaluate_function=True, evaluate_n=fit_plot_points
            )
            for ip in p:
                p_all[ip].append(p[ip])
                p_error_all[ip].append(p_error[ip])

            y_fit_all[iy, :] = y_fit

        for ip in p:
            self.set_dataset(
                self._DATA_FOLDER + "." + ip + "_fit", p_all[ip], broadcast=True
            )
        self.set_dataset(self._DATA_FOLDER + ".fit_x", x_fit, broadcast=True)
        self.set_dataset(self._DATA_FOLDER + ".fit_y", y_fit_all, broadcast=True)

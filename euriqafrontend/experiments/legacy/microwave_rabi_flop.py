"""Initial Rabi flop experiment.

TODO:
    * tweak windows
    * record more values
    * slice the dataset better?
    * Write SBC (from Sandia)
    * Breakout cooling into more functions
    * Basic experiment to cool & graph ion counts
"""
import logging
import time
from collections import defaultdict

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import numpy as np
import oitg.fitting as fit
from artiq.language.core import delay
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import parallel
from artiq.language.core import rpc
from artiq.language.types import TNone
from artiq.language.units import MHz
from artiq.language.units import ms
from artiq.language.units import us

from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend.modules.cw_lasers import DopplerCooling
from euriqafrontend.modules.cw_lasers import PumpDetect
from euriqafrontend.modules.cw_lasers import SSCooling
from euriqafrontend.modules.mw import Microwave
from euriqafrontend.modules.pmt import PMTArray

_LOGGER = logging.getLogger(__name__)


class BasicMicrowaveRabiFlop(artiq_env.EnvExperiment):
    """Perform a microwave Rabi Flop on the ions.

    One of the first ARTIQ experiments attempted on ions at EURIQA.
    Definitely not best practices, just a test sort of thing.
    Written by Drew Risinger 3/8/19
    Modified by Laird Egan 4/17/19 with updated modules.
    """

    kernel_invariants = {
        "wait_times",
        "pmt_array",
        "doppler_cooling_time",
        "optical_pump_time",
        "ss_cooling_time",
        "detect_time",
        "num_pmts",
    }

    applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi "

    def build(self):
        """Initialize experiment & variables."""
        # devices
        self.setattr_device("core")
        self.setattr_device("ccb")
        self.setattr_device("oeb")

        # arguments
        self.setattr_argument(
            "time_range",
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

    def prepare(self):
        """Pre-calculate any values."""

        # calculate scan parameters
        self.wait_times = [t for t in self.time_range]
        self.num_steps = len(self.wait_times)
        self.num_pmts = self.pmt_array.num_active

        # Run prepare method of all my imported modules
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        raw_counts_empty = np.zeros(
            (self.num_pmts, self.num_shots, self.num_steps), dtype=np.int32
        )
        avg_thresh_empty = np.zeros((self.num_pmts, self.num_steps), dtype=np.int32)
        self.set_dataset("raw_counts", raw_counts_empty)
        self.set_dataset("avg_thresh", avg_thresh_empty, broadcast=True)
        self.set_dataset("x_values", self.wait_times)

        self.ccb.issue(
            "create_applet",
            name="Rabi MW",
            command=self.applet_stream_cmd + "--y-names avg_thresh ",
            group="Rabi",
        )

        # RUN EXPERIMENT
        self.kn_initialize()
        self.kn_run()
        self.kn_idle()

    @kernel
    def kn_initilize(self):
        """Initialize the core and all devices."""
        self.core.reset()
        self.core.break_realtime()
        self.oeb.off()
        self.doppler_cooling.init()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.mw.init()
        self.pump_detect.init_to_detect()

    @kernel
    def kn_run(self):
        """Run the experiment on the core device."""

        for t in range(len(self.wait_times)):
            self.core.break_realtime()

            # run experiment
            for i in range(self.num_shots):

                # Doppler Cooling
                self.doppler_cooling.pulse(self.doppler_cooling_time)

                # Second Stage Cooling
                self.ss_cooling.prepare_ss_cool()
                self.ss_cooling.pulse(self.ss_cooling_time)

                # Slow pump to 0 state
                self.pump_detect.switch_to_slow_pump()
                self.pump_detect.pulse(self.optical_pump_time)

                # Rabi Flop
                self.mw.pulse(self.wait_times[t])

                # Detect Ions
                counts = [0] * self.num_pmts
                counts = self.kn_detect(counts)
                self.mutate_dataset(
                    "raw_counts",
                    ((0, self.num_pmts, 1), (i, None, None), (t, None, None)),
                    counts,
                )

            # save data
            thresh = self.calc_avg_thresh(self.per_step_array)
            self.mutate_dataset(
                "avg_thresh", ((0, self.num_pmts, 1), (t, None, None)), thresh
            )

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

    @rpc
    def calc_avg_thresh(self, counts):
        thresh = np.mean(counts > self.detect_thresh, axis=1)
        return thresh

    @kernel
    def kn_idle(self) -> TNone:
        """Reset all used devices to default state."""
        self.core.break_realtime()
        self.doppler_cooling.idle()
        self.pump_detect.off()
        self.mw.off()

    def analyze(self):
        """Threshold values and analyze."""
        x = np.array(self.get_dataset("x_values"))
        y = np.array(self.get_dataset("avg_thresh"))

        p_all = defaultdict(list)
        p_error_all = defaultdict(list)
        y_fit_all = np.zeros(y.shape)

        for iy in y.shape[0]:
            p, p_error, x_fit, y_fit = fit.cos.fit(x, y[iy, :], evaluate_function=True)

            for ip in p:
                p_all[ip].append(p[ip])
                p_error_all[ip].append(p_error[ip])

            y_fit_all[iy, :] = y_fit

        for ip in p:
            self.set_dataset(ip + "_fit", p_all[ip])
        self.set_dataset("fit_curve", y_fit, broadcast=True)

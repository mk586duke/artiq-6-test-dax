import logging

import artiq.language.environment as artiq_env
import numpy as np
from artiq.language.core import delay
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import rpc
from artiq.language.core import TerminationRequested
from artiq.language.types import TFloat
from artiq.language.types import TInt32
from artiq.language.types import TList
from artiq.language.types import TNone
from artiq.language.units import ms
from oitg import fitting

from euriqafrontend.modules.cw_lasers import DopplerCooling
from euriqafrontend.modules.dac import SandiaDAC
from euriqafrontend.modules.pmt import PMTArray

_LOGGER = logging.getLogger(__name__)


class CalibrateDX(artiq_env.EnvExperiment):
    """Calibrate.DX.

    Iteratively Relax X2 and center DX to maximize cooling counts on the center PMT channel.

    Note: Assumes that there is just one ion loaded
    """

    kernel_invariants = {"detect_time", "pmt_array", "num_pmts"}

    applet_stream_cmd = (
        "$python -m euriqafrontend.applets.plot_multi" + " "
    )  # White space is required

    _DATA_FOLDER = "calibrate.DX"
    _APPLET_NAME = "DX Calibration"
    _APPLET_GROUP_NAME = "Calibration"

    def build(self):
        """Initialize experiment & variables."""

        # arguments
        self.setattr_argument(
            "detect_time", artiq_env.NumberValue(default=50 * ms, unit="ms", min=0)
        )
        self.setattr_argument("center_pmt_number", artiq_env.StringValue(default="5"))

        # Load core devices
        self.setattr_device("core")
        self.setattr_device("scheduler")
        self.setattr_device("oeb")
        self.setattr_device("ccb")

        # Load other devices
        # Get PMT Settings
        self.pmt_array = PMTArray(self)
        self.sandia_box = SandiaDAC(self)
        self.doppler_cooling = DopplerCooling(self)

    def prepare(self):
        # overwrite the input to the PMT
        self.pmt_array.pmt_input_s = self.center_pmt_number
        # Run prepare method of all my imported modules
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        # pre-calculation
        self.detect_window_mu = self.core.seconds_to_mu(self.detect_time)
        self.num_pmts = self.pmt_array.num_active
        self.num_fit_points = 100

        self.num_steps = 15
        self.dac_resolution = 20 / (2 ** 18)

        self.x2_val = np.array([0.025, 0.01, 0.001, 0])
        self.stepsize_val = np.array([30, 10, 3, 1])

        self.dx_best_guess = self.get_dataset("global.Voltages.DX") / 1000

        if np.mod(self.num_steps, 2) != 1:
            self.num_steps = np.round(self.num_steps / 2) * 2 + 1

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.kn_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")

            self.calibrate_dx()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")
            self.kn_idle()

    @host_only
    def experiment_initialize(self):
        """Start the experiment on the host."""
        self.set_dataset(
            "data." + self._DATA_FOLDER + ".x_values",
            np.full((len(self.x2_val), self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data." + self._DATA_FOLDER + ".y_values",
            np.full((len(self.x2_val), self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data." + self._DATA_FOLDER + ".fit_x",
            np.full((len(self.x2_val), self.num_fit_points), np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data." + self._DATA_FOLDER + ".fit_y",
            np.full((len(self.x2_val), self.num_fit_points), np.nan),
            broadcast=True,
        )

        # self.ccb.issue("create_applet",
        #                name=self._APPLET_NAME,
        #                command=self.applet_stream_cmd +
        #                        "--x " + self._DATA_FOLDER + ".x_values" + " " +
        #                        "--y-names " + self._DATA_FOLDER + ".y_values" + " " +
        #                        "--x-fit " + self._DATA_FOLDER + ".fit_x" + " " +
        #                        "--y-fits " + self._DATA_FOLDER + ".fit_y " +
        #                        "--transpose",
        #                group=self._APPLET_GROUP_NAME)

        self.sandia_box.dac_pc.send_voltage_lines_to_fpga()
        self.sandia_box.dac_pc.send_shuttling_lookup_table()
        self.sandia_box.dac_pc.apply_line_async(853)

    @kernel
    def kn_initialize(self) -> TNone:
        """Initialize the core and all devices."""
        self.core.reset()
        self.oeb.off()
        self.doppler_cooling.init()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.pmt_array.clear_buffer()

    @kernel
    def kn_idle(self):
        self.core.break_realtime()
        self.doppler_cooling.idle()

    @kernel
    def calibrate_dx(self):

        for iX2 in range(len(self.x2_val)):

            self.core.break_realtime()
            scan_range = self.calc_scan_range(
                self.dx_best_guess, self.stepsize_val[iX2]
            )
            self.mutate_dataset(
                "data." + self._DATA_FOLDER + ".x_values",
                ((iX2, None, len(self.x2_val)), (0, self.num_steps, 1)),
                scan_range,
            )

            for iDX in range(len(scan_range)):
                self.update_DAC(x2=self.x2_val[iX2], dx=scan_range[iDX])
                delay(100 * ms)
                counts = [0]
                stopcounter_mu = self.pmt_array.gate_rising(self.detect_time)
                counts = self.pmt_array.count(
                    up_to_time_mu=stopcounter_mu, buffer=counts
                )

                self.mutate_dataset(
                    "data." + self._DATA_FOLDER + ".y_values",
                    ((iX2, None, len(self.x2_val)), (iDX, None, self.num_steps)),
                    counts,
                )

            self.dx_best_guess = self.fit_scan(iX2)

    @rpc
    def calc_scan_range(
        self, dx_best_guess: TFloat, step_size: TInt32
    ) -> TList(TFloat):
        scan_range = np.linspace(
            start=dx_best_guess
            - step_size * self.dac_resolution * (self.num_steps - 1) / 2,
            stop=dx_best_guess
            + step_size * self.dac_resolution * (self.num_steps - 1) / 2,
            num=self.num_steps,
        )
        scan_list = scan_range.tolist()
        return scan_list

    @rpc(flags={"async"})
    def update_DAC(self, x2, dx):
        self.sandia_box.dac_pc.adjustment_dictionary["X2"]["adjustment_gain"] = x2
        self.sandia_box.dac_pc.adjustment_dictionary["DX"]["adjustment_gain"] = dx
        self.sandia_box.dac_pc.apply_line_async(853)

    @rpc
    def fit_scan(self, iX2: TInt32) -> TFloat:
        x = self.get_dataset("data." + self._DATA_FOLDER + ".x_values")
        x = x[iX2, :]
        y = self.get_dataset("data." + self._DATA_FOLDER + ".y_values")
        y = y[iX2, :]
        p, p_error, x_fit, y_fit = fitting.gaussian.fit(
            x, y, evaluate_function=True, evaluate_n=self.num_fit_points
        )
        self.mutate_dataset(
            "data." + self._DATA_FOLDER + ".fit_x",
            ((iX2, None, len(self.x2_val)), (0, self.num_fit_points, 1)),
            x_fit,
        )
        self.mutate_dataset(
            "data." + self._DATA_FOLDER + ".fit_y",
            ((iX2, None, len(self.x2_val)), (0, self.num_fit_points, 1)),
            y_fit,
        )
        print(p["x0"])
        return p["x0"]

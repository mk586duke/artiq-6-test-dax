"""Idle kernel to maintain the ion and display PMT counts on the master.

TODO:
    * pull config values (i.e. DDS frequencies) from `dataset_db.pyon`
"""
import logging

import artiq.language.environment as artiq_env
import numpy as np
from artiq.language.core import delay
from artiq.language.core import delay_mu
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import TerminationRequested
from artiq.language.types import TNone
from artiq.language.units import MHz
from artiq.language.units import ms
from artiq.language.units import us

from euriqafrontend.modules.cw_lasers import DopplerCooling
from euriqafrontend.modules.dac import SandiaDAC
from euriqafrontend.modules.pmt import PMTArray

_LOGGER = logging.getLogger(__name__)


class DisplayCoolingCounts(artiq_env.EnvExperiment):
    """Display Cooling Counts.

    Cool ion and display counts from the PMT's.
    """

    kernel_invariants = set(
        ["detect_window_mu", "pmt_array", "scale_factor", "num_pmts"]
    )

    applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi_timestream "

    def build(self):
        """Initialize experiment & variables."""
        # devices
        self.setattr_device("core")
        self.setattr_device("oeb")

        # display
        self.setattr_device("ccb")

        # scheduler
        self.setattr_device("scheduler")

        # arguments
        self.setattr_argument(
            "detect_time", artiq_env.NumberValue(default=1 * ms, unit="ms", min=0)
        )
        self.units = self.get_argument(
            "Plot Units", artiq_env.EnumerationValue(["Counts", "kHz"], default="kHz")
        )
        self.sample_rate = self.get_argument(
            "Plot Update Rate",
            artiq_env.NumberValue(default=10, unit="Hz", min=1, max=20),
        )
        self.num_points = self.get_argument(
            "Number of Points to Keep",
            artiq_env.NumberValue(
                default=200, unit="samples", min=1, scale=1, ndecimals=0
            ),
        )

        self.load_on = self.get_argument(
            "Load Solution On", artiq_env.BooleanValue(default=False))



        # modules
        self.doppler_cooling = DopplerCooling(self)
        self.pmt_array = PMTArray(self)
        self.dac = SandiaDAC(self)

        self.setattr_device("magfield_x")
        self.setattr_device("magfield_y")
        self.setattr_device("magfield_z")

        _LOGGER.debug("Done Building Experiment")

    def prepare(self):

        # Run prepare method of all my imported modules
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        self._RID = self.scheduler.rid

        # pre-calculation
        self.detect_window_mu = self.core.seconds_to_mu(self.detect_time)
        self.num_pmts = self.pmt_array.num_active
        self.active_pmts = self.pmt_array.active_pmts

        if self.units == "Counts":
            self.scale_factor = 1
        elif self.units == "kHz":
            self.scale_factor = 1 / self.detect_time / 1000

        Ix = self.get_dataset("global.B_coils.Ix")
        Iy = self.get_dataset("global.B_coils.Iy")
        Iz = self.get_dataset("global.B_coils.Iz")
        Vx = self.get_dataset("global.B_coils.Vx")
        Vy = self.get_dataset("global.B_coils.Vy")
        Vz = self.get_dataset("global.B_coils.Vz")

        self.magfield_x.set_voltage(Vx)
        self.magfield_x.set_current(Ix)
        self.magfield_y.set_voltage(Vy)
        self.magfield_y.set_current(Iy)
        self.magfield_z.set_voltage(Vz)
        self.magfield_z.set_current(Iz)

        if self.load_on:
            print("Here!")
            self.dac.dac_pc.line_gain = 1.0
        else:
            self.dac.dac_pc.line_gain = 0.0

        self.dac.dac_pc.send_voltage_lines_to_fpga()
        self.dac.dac_pc.send_shuttling_lookup_table()
        if self.load_on is True:
            self.dac.dac_pc.apply_line_async("QuantumRelaxed_LoadOn")
        else:
            self.dac.dac_pc.apply_line_async("Start")

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def run(self):
        """Run the idle kernel."""

        self.set_dataset(
            "data.display.active_pmts", np.array(self.active_pmts), broadcast=True
        )
        self.set_dataset("data.display.rid", self._RID, broadcast=True)
        self.ccb.issue(
            "create_applet",
            name="pmt_stream_display",
            command=self.applet_stream_cmd
            + "--y-names data.display.pmt_counts "
            + "--update-delay "
            + str(np.round(1 / self.sample_rate, 2))
            + " --transpose "
            + "--retain-points "
            + str(int(self.num_points))
            + " --active-pmts "
            + "data.display.active_pmts"
            + " --rid {0}".format("data.display.rid"),

        group="pmt",
        )
        self.set_dataset(
            "data.display.pmt_counts",
            np.full(self.num_pmts, np.nan),
            broadcast=True,
            archive=False,
        )
        try:
            self.kn_run()

        except TerminationRequested:
            # self.ccb.issue("disable_applet", name="pmt_stream_display", group="pmt")
            _LOGGER.info("Termination Requested. Ending Experiment")

        finally:
            self.kn_idle()

            # end of experiment

    @kernel
    def kn_run(self) -> TNone:
        """Run the cooling setup and PMT measurement on the core."""
        self.core.reset()
        self.kn_setup()

        # NOTE: scheduler might not work, but give it a shot. not perfect on timing
        while not self.scheduler.check_pause():
            # while True:
            delay(100 * ms)
            for i in range(100):
                i += 1
                self.kn_run_once()

    @kernel
    def kn_setup(self) -> TNone:
        """Initialize the core and all devices."""
        self.oeb.off()
        self.doppler_cooling.init()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.pmt_array.clear_buffer()
        self.core.break_realtime()

    @kernel
    def kn_run_once(self) -> TNone:
        """Run the experiment on the core device."""
        delay(1000 * us)
        buffer = [0] * self.num_pmts
        scaled_counts = [0] * self.num_pmts
        stopcounter_mu = self.pmt_array.gate_rising_mu(self.detect_window_mu)
        delay(10 * us)
        counts = self.pmt_array.count(up_to_time_mu=stopcounter_mu, buffer=buffer)
        for i in range(self.num_pmts):
            scaled_counts[i] = counts[i] #* self.scale_factor
        delay(5000 * us)

        self.mutate_dataset(
            "data.display.pmt_counts", (0, self.num_pmts), scaled_counts
        )

    @kernel
    def kn_idle(self):
        self.core.break_realtime()
        self.doppler_cooling.idle()

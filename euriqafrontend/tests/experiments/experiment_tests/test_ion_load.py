"""Try to load an ion and shuttle it into the central load slot."""
import logging
import time

import artiq
from artiq.experiment import *
from artiq.language import NumberValue
from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.core import sequential
from artiq.language.units import MHz
from artiq.language.units import ms
from artiq.language.units import us

from euriqabackend.coredevice.ad9912 import freq_to_mu

_LOGGER = logging.getLogger(__name__)


class IonLoader(artiq.language.environment.EnvExperiment):
    """Test Ion Loading."""

    # pylint: disable=no-member

    def build(self):
        """Get any ARTIQ arguments."""
        self.loops = self.get_argument(
            "loops",
            NumberValue(
                default=10.0, unit="loops", scale=1, step=1, min=1, ndecimals=0
            ),
        )

        self.setattr_argument(
            "cool_1_freq",
            NumberValue(default=180e6, unit="MHz", min=100e6, max=300e6),
            group="Cool 1",
        )
        self.setattr_argument(
            "cool_2_freq",
            NumberValue(default=195e6, unit="MHz", min=100e6, max=300e6),
            group="Cool 2",
        )
        self.setattr_argument(
            "cool_1_amp", NumberValue(default=1000, min=0, max=1000), group="Cool 1"
        )
        self.setattr_argument(
            "cool_2_amp", NumberValue(default=1000, min=0, max=1000), group="Cool 2"
        )

        self.setattr_device("scheduler")
        self.setattr_device("core")
        self.setattr_device("oeb")

        self.setattr_device("electrodes")
        self.dac = (
            self.electrodes
        )  # type: artiq.devices.sandia_dac.mediator.CompoundDACFPGAHardware

        # self.setattr_device("w_dds")

        self.dds_cool_1 = self.get_device("dds6")
        self.dds_cool_2 = self.get_device("dds7")
        self.dds_cool_1_switch = self.get_device("dds6_switch")
        self.dds_cool_2_switch = self.get_device("dds7_switch")
        self.reset_switch = self.get_device("reset67")

        self.frequency = [freq_to_mu(i * MHz) for i in [180, 195]]
        self.power_cool_1 = self.get_device("power_cool_1")
        self.power_cool_2 = self.get_device("power_cool_2")
        self.shutter_399 = self.get_device("shutter_399")
        self.shutter_370ion = self.get_device("shutter_370ion")
        _LOGGER.debug("Done Building Ion Loading Experiment")

    def prepare(self):
        """Load Dac Files."""
        self.dac.load_pin_map_file(
            "C:/Users/EuriqaControl/Documents/IonControlProjects/EURIQA/config/EURIQA_socket_map.txt"
        )
        self.dac.load_voltage_file(
            "Z:/CompactTrappedIonModule/Voltage Solutions/Translated Solutions/ionQ_merge_quantumLRswap_translate_cp2_PAD.txt"
        )
        self.dac.load_global_adjust_file(
            "Z:/CompactTrappedIonModule/Voltage Solutions/Compensation Solutions/EURIQA_central_11_pairs_compensation_plus_loadsol.txt"
        )
        self.dac.load_shuttling_definitions_file(
            "Z:/CompactTrappedIonModule/Voltage Solutions/Translated Solutions/ionQ_merge_quantumLRswap_translate_cp2_PAD_shuttling.xml"
        )

        self.dac.send_voltage_lines_to_fpga()
        self.dac.send_shuttling_lookup_table()

        self.dac.apply_line(7.0)

        _LOGGER.debug("Done setting up Sandia DAC x100")

        _LOGGER.debug("Done Preparing Ion Loading Experiment")

    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()
        self.initialize_shutters()
        # self.dac.apply_line(7)
        delay(1.0)
        # time.sleep(1.0)
        with sequential:
            for i in range(self.loops):
                # self.load_lasers_on(True)
                # self.dac.apply_line(10.0)
                # delay(1.0)
                # time.sleep(1.0)
                # self.wait(1.0)
                self.load_lasers_on(False)
                self.dac.apply_line(7.0)
                delay(1.0)
                # time.sleep(1.0)
                # self.wait(1.0)

            # newline = self.dac.shuttle("Center")
            # self.wait(2.0)
            # print(newline)
            # self.wait(5.0)
            # newline = self.dac.shuttle("Loading")
            # self.wait(2.0)
            # print(newline)

            # print("Loading Loop %i Completed", i+1)

        # _LOGGER.info("Finished")

    @kernel
    def initialize_shutters(self):
        """Set up shutters for cooling."""
        # self.core.reset()
        # self.core.break_realtime()
        self.reset_switch.off()
        self.oeb.off()

        self.dds_cool_1.setup_bus(write_div=6)
        delay(10 * us)
        self.dds_cool_1.set(frequency=self.frequency[0], amplitude=999)

        self.dds_cool_2.setup_bus(write_div=6)
        delay(10 * us)
        self.dds_cool_2.set(frequency=self.frequency[1], amplitude=999)

        self.dds_cool_1_switch.on()
        self.dds_cool_2_switch.on()
        self.power_cool_1.on()
        self.power_cool_2.on()

    @kernel
    def load_lasers_on(self, on):
        if on is True:
            with parallel:
                self.shutter_399.on()
                self.shutter_370ion.on()
        elif on is False:
            with parallel:
                self.shutter_399.off()
                self.shutter_370ion.off()

    @kernel
    def wait(self, sec):
        delay(sec)

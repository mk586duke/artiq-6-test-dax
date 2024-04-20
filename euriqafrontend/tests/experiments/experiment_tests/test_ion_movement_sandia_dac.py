"""Test that you can move an ion around using the Sandia DAC.

Assumes ion is pre-loaded, and tries to shuttle it and move it.
Requires manual viewing on camera, no detection implemented.

Changes the potential well (via DC voltage) of an ion trap to move an ion around.
"""
import logging
import time

import artiq
from artiq.language.units import ms

_LOGGER = logging.getLogger(__name__)


class SandiaDACIonMover(artiq.language.environment.EnvExperiment):
    """Sandia DAC ion movement test.

    ARTIQ Experiment to move an ion around by changing DC voltages.
    """

    # pylint: disable=no-member

    def build(self):
        """Get any ARTIQ arguments."""
        self.loops = self.get_argument(
            "loops",
            artiq.language.NumberValue(
                default=10.0, unit="loops", scale=1, step=1, min=1, ndecimals=0
            ),
        )
        self.setattr_device("dac_pc_interface")
        self.dac = (
            self.dac_pc_interface
        )  # type: euriqabackend.devices.sandia_dac.interface.CompoundDACFPGAHardware

    @artiq.language.core.host_only
    def manual_setup(self):
        """Prompts to user to manually setup experiment.

        Just a test.
        """
        input(
            "Please switch on the camera to manually verify ion movement. "
            "Press any key to continue..."
        )

    def prepare(self):
        """Confirm experiment is manually set up correctly with operator."""
        # TODO: fill in file paths
        self.dac.load_pin_map_file("Q:/PATH_TO_PIN_MAP")
        self.dac.load_voltage_file("Q:/PATH_TO_VOLTAGE_SOLUTION")
        self.dac.load_global_adjust_file("Q:/PATH_TO_GLOBAL_ADJUST_FILE")
        self.dac.load_shuttling_definitions_file("Q:/PATH_TO_SHUTTLING_GRAPH_FILE")
        # TODO: set adjustments
        self.dac.set_adjustments({}, 1.0)
        _LOGGER.debug("Done setting up Sandia DAC x100")
        # self.manual_setup()

    def run(self):
        """Move the ion around visibly on a camera."""
        self.dac.send_voltage_lines_to_fpga()
        for i in range(self.loops):
            _LOGGER.debug("Starting Sandia DAC Output loop #%i", i)
            # TODO: time how long calls take?
            self.tweak_ion()

            time.sleep(i * 100 * ms)
            # use time.sleep because not running on kernel, don't know how long to delay
            self.apply_lines()
            self.shuttle_ion()

    def tweak_ion(self):
        """Move the ion a little bit by applying "tweaks"/adjustments."""
        starting_dac_state = self.dac.save_state()

        # TODO: define adjustment values
        self.dac.set_adjustments({}, 1.0)
        self.dac.set_adjustments({}, 1.0)

        self.dac.load_saved_state(starting_dac_state)

    def apply_lines(self):
        """Jump to different lines in the voltage solution."""
        self.dac.apply_line_async(line_number=1.3, line_gain=1.0, global_gain=1.0)
        # TODO: delay?
        self.dac.apply_line_async(
            line_number=(len(self.dac.lines) - 1.2), line_gain=1.0, global_gain=1.0
        )

    def shuttle_ion(self):
        """Shuttle the ion between different locations."""
        self.dac.shuttle_async("loading")
        self.dac.shuttle_async("quantum")
        self.dac.apply_line_async(7.0)
        self.dac.shuttle_async("quantum")

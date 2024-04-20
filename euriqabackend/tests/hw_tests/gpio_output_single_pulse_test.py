"""Experiment to test GPIO values can change.

Test plan:
    * output data via this script
    * check outputs occur on scope
    * if not on scope, check outputs on the input side of the line drivers/buffers
    * if not in line driver, check the ARTIQ core analyzer to check outputs are toggled
        * also check the rtio_log "strb" channel to see if it's toggling in time.
"""
import logging

from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.environment import EnvExperiment
from artiq.language.units import us

_LOGGER = logging.getLogger(__name__)


class TestOutputs(EnvExperiment):
    """Experiment to test GPIO/TTL outputs are working as expected."""

    def build(self):
        """Build the experiment."""
        self.setattr_device("core")
        self.setattr_device("oeb")

        self.outputs = [self.get_device("out2_2"), self.get_device("out2_4")]

    @kernel
    def run(self):
        """Strobe all GPIO/TTL outputs to check FPGA output."""
        self.core.reset()
        self.core.break_realtime()
        self.oeb.on()  # works
        delay(10 * us)
        self.oeb.off()
        # doesn't work (following)
        for _ in range(10):
            self.outputs[0].on()
            self.outputs[1].on()
            delay(20 * us)
            self.outputs[0].off()
            self.outputs[1].off()
            delay(20 * us)

"""Experiment to test GPIO values can change.

Test plan:
    * output data via this script
    * check outputs occur on scope
    * if not on scope, check outputs on the input side of the line drivers/buffers
    * if not in line driver, check the ARTIQ core analyzer to check outputs are toggled
        * also check the rtio_log "strb" channel to see if it's toggling in time.
"""
import itertools
import logging

from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.environment import EnvExperiment
from artiq.language.units import us

_LOGGER = logging.getLogger(__name__)


class TestOutputs(EnvExperiment):
    """Experiment to test GPIO/TTL outputs are working as expected."""

    kernel_invariants = {"outputs", "core", "oeb"}  # avoid RPC for outputs local var

    def build(self):
        """Build the experiment."""
        self.setattr_device("core")
        self.setattr_device("oeb")

        output_nums = list(
            itertools.product(range(1, 5), range(8))
        )  # list because reused
        _LOGGER.debug(
            ", ".join(["out{}_{}".format(bank, chan) for bank, chan in output_nums])
        )
        self.outputs = [
            self.get_device("out{}_{}".format(bank, chan)) for bank, chan in output_nums
        ]
        _LOGGER.debug("outputs[%i]: %s", len(self.outputs), self.outputs)

    @kernel
    def run(self):
        """Strobe all GPIO/TTL outputs to check FPGA output."""
        self.core.reset()
        self.core.break_realtime()
        self.oeb.off()
        delay(0.1 * us)
        for _ in range(10):
            # ! following code DOESN'T WORK. Generates SequenceError.
            # ! Can only change ~8 at once (number of lanes in your gateware)
            for out in self.outputs:
                out.on()
            delay(50 * us)
            for out in self.outputs:
                out.off()
            delay(50 * us)

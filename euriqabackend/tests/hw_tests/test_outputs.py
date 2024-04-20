"""Experiment to manually set GPIO/TTL outputs."""
import logging

from artiq.language.core import kernel
from artiq.language.core import parallel
from artiq.language.environment import EnvExperiment
from artiq.language.types import TInt32
from artiq.language.types import TList
from artiq.language.units import us

_LOGGER = logging.getLogger(__name__)


def input_state() -> TList(TInt32):
    """Get input from user about output value of chosen channel."""
    return list(
        map(int, input("Enter bank (1-4), channel (0-7), state (0/1): ").split())
    )


class TestOutputs(EnvExperiment):
    """Experiment to test GPIO/TTL outputs are working as expected."""

    kernel_invariants = {"outputs", "marker_out"}

    def build(self):
        """Build the experiment."""
        self.setattr_device("core")
        self.setattr_device("oeb")

        _LOGGER.debug(
            ["out{}_{}".format(bank, chan) for bank in range(1, 5) for chan in range(8)]
        )
        self.outputs = [
            self.get_device("out{}_{}".format(bank, chan))
            for bank in range(1, 5)
            for chan in range(8)
        ]

        self.marker_out = self.get_device("out2_4")  # arbitrary

    @kernel
    def run(self):
        """Request and set state of chosen channels infinitely (until ctrl-c)."""
        self.core.reset()
        self.oeb.off()
        while True:
            set_state = input_state()
            bank, channel, state = set_state[0], set_state[1], set_state[2]
            _LOGGER.info("Setting Output %i_%i to %i", bank, channel, state)
            self.core.break_realtime()

            self.marker_out.pulse(1 * us)
            try:
                output_chan = self.outputs[8 * (bank - 1) + channel]
                if state:
                    output_chan.on()
                else:
                    output_chan.off()
            except IndexError:
                # set all outputs high as diagnosis.
                with parallel:
                    for outchan in self.outputs:
                        outchan.pulse(1 * us)
        # NOTE: tried using getattr() and 2D array
        #   (i.e. index var like output[bank][chan]),
        #   But they DIDN'T WORK in ARTIQ core code for some reason. Gave up debugging.

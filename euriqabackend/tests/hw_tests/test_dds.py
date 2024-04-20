"""Tests that the DDS's are working.

TODO:
    * rewrite to check all DDS channels.
    * Update for ARTIQ 4
"""
import logging

import numpy
from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.environment import EnvExperiment
from artiq.language.types import TInt64
from artiq.language.types import TList
from artiq.language.units import MHz
from artiq.language.units import us
from more_itertools import sliced

from euriqabackend.coredevice.ad9912 import freq_to_mu

# from euriqabackend.coredevice.ad9912 import phase_to_mu

_LOGGER = logging.getLogger(__name__)


class Test_DDS(EnvExperiment):
    """Test that the Duke DDS (based on ad9912 chip) are working properly."""

    kernel_invariants = {
        "core",
        "oeb",
        "frequency_mu",
        "frequency2_mu",
        "marker",
        "dds_switch",
        "dds",
        "reset_switch",
    }

    def build(self):
        """Declare devices for experiment."""
        self.setattr_device("core")
        self.setattr_device("oeb")

        # dds_nums = range(2) # for testing only
        dds_nums = range(10)

        self.dds = [self.get_device("dds{}".format(i)) for i in dds_nums]
        self.dds_switch = [self.get_device("dds{}_switch".format(i)) for i in dds_nums]
        self.reset_switch = [
            self.get_device("reset{}{}".format(a, b)) for a, b in sliced(dds_nums, 2)
        ]  # reset01, reset23, ..., reset89

        self.marker = self.get_device("out2_4")

        self.frequency_mu = [
            numpy.int64(freq_to_mu((100 + 10 * i) * MHz)) for i in dds_nums
        ]
        self.frequency2_mu = [
            numpy.int64(freq_to_mu((200 + 10 * i) * MHz)) for i in dds_nums
        ]
        _LOGGER.debug("f1: %s, f2: %s", self.frequency_mu, self.frequency2_mu)

    @kernel
    def run(self):
        """Run the DDS boards. Alternate frequencies & amplitudes."""
        # ** SETUP
        self.core.reset()
        self.core.break_realtime()

        self.oeb.off()
        self.marker.on()
        for sw in self.dds_switch:
            sw.on()
            delay(0.1 * us)

        for rst in self.reset_switch:
            # pulse/on sets default freq of ~155 MHz, ~half amplitude, and phase=0
            rst.off()
            delay(0.1 * us)

        self.marker.off()
        self.core.break_realtime()

        # ** ACTIONS
        for _ in range(100):
            self.write_frequency(self.frequency_mu)
            delay(100 * us)  # must be 80-90 us, otherwise RTIOUnderflow
            self.write_frequency(self.frequency2_mu)
            delay(100 * us)

    @kernel
    def write_frequency(self, frequencies_mu: TList(TInt64)):
        """Write a specified frequency to the DDS."""
        self.marker.on()
        # can't program two DDS on the same board at the same time. SPI bus conflict
        # solved by breaking into two sections. Could probably reduce the delay.
        for i in range(len(frequencies_mu)):
            if i % 2 == 0:
                self.dds[i].set_mu(frequencies_mu[i], preset=True)
        delay(5 * us)  # seems to be sweet spot, can't be reduced lower w/o seq error
        for i in range(len(frequencies_mu)):
            if i % 2 == 1:
                self.dds[i].set_mu(frequencies_mu[i], preset=True)
        self.marker.off()
        # actually finished programming before 5 us when preset=True

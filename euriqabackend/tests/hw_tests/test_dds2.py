"""Tests that the DDS's are working.

TODO:
    * rewrite to check all DDS channels.
    * Update for ARTIQ 4 (limited number of outputs)
"""
import logging

from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.core import parallel
from artiq.language.environment import EnvExperiment
from artiq.language.units import MHz
from artiq.language.units import ms
from artiq.language.units import us
from more_itertools import sliced

from euriqabackend.coredevice.ad9912 import freq_to_mu

_LOGGER = logging.getLogger(__name__)

# from euriqabackend.coredevice.ad9912 import phase_to_mu


class Test_DDS(EnvExperiment):
    """Test that the Duke DDS (based on ad9912 chip) are working properly."""

    def build(self):
        """Declare devices for experiment."""
        self.setattr_device("core")
        self.setattr_device("oeb")

        dds_nums = range(8)

        self.dds = [self.get_device("dds{}".format(i)) for i in dds_nums]
        self.dds_switch = [self.get_device("dds{}_switch".format(i)) for i in dds_nums]
        self.reset_switch = [
            self.get_device("reset{}{}".format(a, b)) for a, b in sliced(dds_nums, 2)
        ]  # reset01, reset23, ..., reset89

        self.marker = self.get_device("out2_4")

        self.frequency = [freq_to_mu(100 * MHz + i * 10 * MHz) for i in dds_nums]

    @kernel
    def run(self):
        """Run the DDS boards. Alternate frequencies & amplitudes."""
        # ** SETUP
        self.core.reset()
        self.core.break_realtime()

        self.oeb.off()
        self.marker.on()
        with parallel:
            for sw in self.dds_switch:
                sw.on()

            for rst in self.reset_switch:
                # pulse/on sets default freq of ~155 MHz, ~half amplitude, and phase=0
                rst.off()
        self.marker.off()

        # ** ACTIONS
        for _ in range(1):
            self.dds[6].setup_bus(write_div=6)
            # self.write_frequency(self.frequency[6], 6)
            delay(10 * us)
            self.dds[6].set(frequency=self.frequency[6])
            # self.dds[6].set(amplitude=500)

            delay(100 * ms)
            self.dds[7].setup_bus(write_div=6)
            self.write_frequency(self.frequency[7], 7)
            delay(100 * ms)

    @kernel
    def write_frequency(self, freq_mu, ind):
        """Write a specified frequency to the DDS."""
        # freq_mu = freq_to_mu(frequency_raw)
        self.marker.on()
        delay(10 * us)
        self.dds[ind].set(frequency=freq_mu)
        self.marker.off()
        #
        # self.dds1.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds1.set(frequency=self.frequency1)
        #
        # self.dds2.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds2.set(frequency=self.frequency2)
        #
        # self.dds3.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds3.set(frequency=self.frequency3)
        #
        # self.dds4.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds4.set(frequency=self.frequency4)
        #
        # self.dds5.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds5.set(frequency=self.frequency5)
        #
        # self.dds6.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds6.set(frequency=self.frequency6)
        #
        # self.dds7.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds7.set(frequency=self.frequency7)
        #
        # self.dds8.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds8.set(frequency=self.frequency8)
        #
        # self.dds9.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds9.set(frequency=self.frequency9)

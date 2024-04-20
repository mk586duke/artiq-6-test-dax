import time

import numpy as np
from artiq.experiment import *


class TestOutputs(EnvExperiment):
    """Scan CW Frequency (single channel)."""

    def build(self):
        """Build the experiment."""
        self.setattr_device("core")
        self.setattr_device("led0")
        self.setattr_device("led1")
        self.setattr_device("oeb")

        for i in range(10):
            self.setattr_device("io_update{}".format(i))
            self.setattr_device("dds{}_switch".format(i))
            self.setattr_device("dds{}".format(i))

        for i in range(5):
            self.setattr_device("reset" + str(i * 2) + str(i * 2 + 1))

        for i in range(1, 5):
            for j in range(8):
                self.setattr_device("out{}_{}".format(i, j))

        self.setattr_argument(
            "num_dds",
            EnumerationValue(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
        )
        self.setattr_argument(
            "freq_scan", Scannable(default=LinearScan(37.47 - 3.25, 37.47 - 3.1, 3))
        )
        self.setattr_argument(
            "freq_hold_time_s",
            NumberValue(default=1, unit="s", scale=1, step=0.001, ndecimals=3),
        )
        self.qubit_ampl = 1023

    def prepare(self):
        """Prepare the experiment."""
        self.freq_scan_bin = [self.freq_to_bin(f * MHz) for f in self.freq_scan]
        self.qubit_dds = int(self.num_dds)
        self.OddsOrEvens = np.mod(int(self.num_dds), 2)
        # print(self.OddsOrEvens)

    @staticmethod
    def freq_to_bin(freq):
        """Convert a frequency to a binned value."""
        dds_clk_freq = np.int64(10 ** 9)  # 1 GHz clock
        bin_width = np.int64(2 ** 48)  # 48 bit frequency tuning word
        bin_freq = np.int64((freq / dds_clk_freq) * bin_width)
        return bin_freq

    def run(self):
        self.core.reset()
        self.run_exp()

    @kernel
    def run_exp(self):
        """Run the experiment."""
        self.core.break_realtime()
        self.exp_init()
        delay(1000 * us)
        self.out1_0.pulse(1 * us)

        self.dds_switch_on(self.qubit_dds)
        self.dds_freq(self.qubit_dds, self.freq_scan_bin[0])
        self.dds_ampl(self.qubit_dds, self.qubit_ampl)
        delay(1 * s)
        for i in self.freq_scan_bin:
            self.dds_freq(self.qubit_dds, i)
            delay(self.freq_hold_time_s)

    @kernel
    def exp_init(self):
        """Initialize the experiment."""
        self.oeb.off()
        for i in range(10):
            self.dds_switch_off(i)

        self.reset01.off()
        self.reset23.off()
        self.reset45.off()
        self.reset67.off()
        self.reset89.off()

        # Set up DDS bus for the DDS's we are
        # actively using. Since 0/1, 2/3, etc share
        # a bus, setting up the other DDS will prevent
        # changes to these.
        delay(100 * us)
        with parallel:
            if self.OddsOrEvens == 0:  # even
                self.dds0.setup_bus(write_div=6)
                self.dds2.setup_bus(write_div=6)
                self.dds4.setup_bus(write_div=6)
                self.dds6.setup_bus(write_div=6)
                self.dds8.setup_bus(write_div=6)
            else:
                self.dds1.setup_bus(write_div=6)
                self.dds3.setup_bus(write_div=6)
                self.dds5.setup_bus(write_div=6)
                self.dds7.setup_bus(write_div=6)
                self.dds9.setup_bus(write_div=6)

        delay(20 * us)  # DDS programming is slow
        self.dds_freq(self.qubit_dds, self.freq_scan_bin[0])  # Zero 'time'
        delay(10 * us)  # Need to give some room for bus to clear
        self.dds_ampl(self.qubit_dds, self.qubit_ampl)  # Zero 'time'
        pass

    @portable
    @staticmethod
    def check_value_range(val, max, min=0):
        """Raise error if the input is not in the given range."""
        if val > max or val < min:
            raise ValueError("Value {} is not in range [{}, {}]".format(val, min, max))

    @kernel
    def dds_freq(self, idx, freq):
        """Change the DDS output frequency."""
        self.check_value_range(idx, 9)
        getattr(self, "dds" + str(idx)).set(frequency=freq)

    @kernel
    def dds_phase(self, idx, phase):
        """Change the DDS output phase."""
        self.check_value_range(idx, 9)
        getattr(self, "dds" + str(idx)).phase(phase)

    @kernel
    def dds_ampl(self, idx, ampl):
        """Change the DDS output amplitude."""
        self.check_value_range(idx, 9)
        getattr(self, "dds" + str(idx)).set(amplitude=ampl)

    @kernel
    def dds_switch_on(self, idx):
        """Switch on the DDS output."""
        getattr(self, "dds" + str(idx) + "_switch").on()

    @kernel
    def dds_switch_off(self, idx):
        """Switch off the DDS output."""
        self.check_value_range(idx, 9)
        getattr(self, "dds" + str(idx) + "_switch").off()

import random

import numpy as np
from artiq.experiment import *


def input_state() -> TBool:
    return input("Enter desired TTL state: ") == "1"


def oeb_state() -> TBool:
    return input("Output on?: ") == "1"


class test_EURIQA_hardware(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("led0")
        self.setattr_device("oeb")
        self.setattr_device("out1_0")
        self.setattr_device("dds0_switch")
        self.setattr_device("dds0")
        self.setattr_device("dds1_switch")
        self.setattr_device("dds1")
        # self.setattr_device("sma0")
        self.setattr_device("io_update0")
        self.setattr_device("reset01")
        # self.frequency = self.freq_to_bin(92.407226562499 * MHz)
        # self.frequency = self.freq_to_bin(92.407226562500 * MHz)
        # self.frequency = self.freq_to_bin(100 * MHz)
        self.frequency = np.int64(0x0000FFFFFFFFFFFF)

        self.qubit_count = 8  # qubits under test
        self.PMTs = [0] * self.qubit_count  # PMTs measured = qubit count
        self.PMT_counts = [
            random.randint(0, 9) for _ in range(100 * self.qubit_count)
        ]  # long list of random "measurements"
        self.index = 0  # index through the list of measurements
        self.threshold = 5.5  # binary threshold

    def freq_to_bin(self, freq):
        dds_clk_freq = np.int64(10 ** 9)  # 1 GHz clock
        bin_width = np.int64(2 ** 48)  # 48 bit frequency tuning word
        bin_freq = np.int64((freq / dds_clk_freq) * bin_width)
        # print(hex(bin_freq))
        return bin_freq

    def func(self, freq):
        print(freq)
        print(type(freq))

    @kernel
    def run(self):
        self.core.reset()
        self.out1_0.off()
        self.oeb.off()
        self.dds0_switch.on()
        self.reset01.off()
        # self.dds0.setup_bus(write_div=8)
        # self.dds0.set(amplitude=1023)
        # delay(200*us)
        # self.func(self.frequency)
        # self.core.break_realtime()
        # self.dds0.set(frequency=self.frequency, amplitude=1023)
        # self.dds0.set(frequency=int(26010321944575, width=64), amplitude=1023) # bad
        # self.dds0.set(frequency=26010321944575)
        # self.dds0.set(frequency = self.frequency)
        # self.dds0.set(frequency=int(26010321944576, width=64), amplitude=1023) # good
        # self.dds0.set(frequency=28147497671066, amplitude=1023)
        # s1 = input_state()
        # self.core.break_realtime()
        # if s1:
        #     self.led0.on()
        #     self.out1_0.on()
        # else:
        #     self.led0.off()
        #     self.out1_0.off()
        # if s2:
        #     self.oeb.off()
        # else:
        #     self.oeb.on()

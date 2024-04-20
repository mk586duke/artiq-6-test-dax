import time

import numpy as np
from artiq.experiment import *


def input_state() -> TList(TInt32):
    return list(map(int, input("Enter bank, output, state: ").split()))


class TestOutputs(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("led0")
        self.setattr_device("led1")
        self.setattr_device("oeb")

        self.setattr_device("io_update0")
        self.setattr_device("io_update1")
        self.setattr_device("io_update2")
        self.setattr_device("io_update3")
        self.setattr_device("io_update4")
        self.setattr_device("io_update5")
        self.setattr_device("io_update6")
        self.setattr_device("io_update7")
        self.setattr_device("io_update8")
        self.setattr_device("io_update9")

        self.setattr_device("reset01")
        self.setattr_device("reset23")
        self.setattr_device("reset45")
        self.setattr_device("reset67")
        self.setattr_device("reset89")

        self.setattr_device("dds0_switch")
        self.setattr_device("dds1_switch")
        self.setattr_device("dds2_switch")
        self.setattr_device("dds3_switch")
        self.setattr_device("dds4_switch")
        self.setattr_device("dds5_switch")
        self.setattr_device("dds6_switch")
        self.setattr_device("dds7_switch")
        self.setattr_device("dds8_switch")
        self.setattr_device("dds9_switch")

        self.setattr_device("dds0")
        self.setattr_device("dds1")
        self.setattr_device("dds2")
        self.setattr_device("dds3")
        self.setattr_device("dds4")
        self.setattr_device("dds5")
        self.setattr_device("dds6")
        self.setattr_device("dds7")
        self.setattr_device("dds8")
        self.setattr_device("dds9")

        self.setattr_device("out1_0")
        self.setattr_device("out1_1")
        self.setattr_device("out1_2")
        self.setattr_device("out1_3")
        self.setattr_device("out1_4")
        self.setattr_device("out1_5")
        self.setattr_device("out1_6")
        self.setattr_device("out1_7")

        self.setattr_device("out2_0")
        self.setattr_device("out2_1")
        self.setattr_device("out2_2")
        self.setattr_device("out2_3")
        self.setattr_device("out2_4")
        self.setattr_device("out2_5")
        self.setattr_device("out2_6")
        self.setattr_device("out2_7")

        self.setattr_device("out3_0")
        self.setattr_device("out3_1")
        self.setattr_device("out3_2")
        self.setattr_device("out3_3")
        self.setattr_device("out3_4")
        self.setattr_device("out3_5")
        self.setattr_device("out3_6")
        self.setattr_device("out3_7")

        self.setattr_device("out4_0")
        self.setattr_device("out4_1")
        self.setattr_device("out4_2")
        self.setattr_device("out4_3")
        self.setattr_device("out4_4")
        self.setattr_device("out4_5")
        self.setattr_device("out4_6")
        self.setattr_device("out4_7")

        # Note: Since dds0/dds1 share a single SPI channel we cannot program them simultaneously
        # Same for dds2/dds3, dds4/dds5, etc. To simplify things, the code will only work for
        # a qubit dds set (below) with even numbers, so none of the dds's share SPI lines.
        self.qubit_count = 2
        self.qubit_dds = [0, 2]  # dds numbers for the qubits in the sequence
        self.qubit_freq = [self.freq_to_bin(f * MHz) for f in [1, 1]]
        self.f = np.arange(37.470 - 3.250, 37.470 - 3.100, 0.005)
        self.stupidlist = np.arange(len(self.f))
        self.freqlist = [self.freq_to_bin((f) * MHz) for f in self.f]
        self.qubit_ampl = [1023, 1023]

    def freq_to_bin(self, freq):
        dds_clk_freq = np.int64(10 ** 9)  # 1 GHz clock
        bin_width = np.int64(2 ** 48)  # 48 bit frequency tuning word
        bin_freq = np.int64((freq / dds_clk_freq) * bin_width)
        return bin_freq

    def run(self):
        self.core.reset()
        self.run_exp()
        print("Done")

    @kernel
    def run_exp(self):
        self.core.break_realtime()
        self.exp_init()
        delay(1000 * us)
        self.out1_0.pulse(1 * us)
        # self.out1_4.off() #this is the switch for the optical pumping EOM frequency source
        self.dds0_switch.on()
        self.dds2_switch.on()

        self.dds0.set(frequency=self.freqlist[0], amplitude=self.qubit_ampl[0])
        self.dds2.set(frequency=self.freqlist[0], amplitude=self.qubit_ampl[1])
        delay(1 * s)
        for i in self.stupidlist:
            self.dds0.set(frequency=self.freqlist[i])
            self.dds2.set(frequency=self.freqlist[i])

            # print(self.f[i])
            delay(1.0 * s)

    @kernel
    def exp_init(self):

        self.oeb.off()
        self.dds0_switch.off()
        self.dds1_switch.off()
        self.dds2_switch.off()
        self.dds3_switch.off()
        self.dds4_switch.off()
        self.dds5_switch.off()
        self.dds6_switch.off()
        self.dds7_switch.off()
        self.dds8_switch.off()
        self.dds9_switch.off()
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
            self.dds0.setup_bus(write_div=6)
            self.dds2.setup_bus(write_div=6)
            self.dds4.setup_bus(write_div=6)
            self.dds6.setup_bus(write_div=6)
            self.dds8.setup_bus(write_div=6)

        delay(self.qubit_count * 20 * us)  # DDS programming is slow
        for idx in range(self.qubit_count):
            self.dds_freq(self.qubit_dds[idx], self.qubit_freq[idx])  # Zero 'time'
        delay(10 * us)  # Need to give some room for bus to clear
        for idx in range(self.qubit_count):
            self.dds_ampl(self.qubit_dds[idx], self.qubit_ampl[idx])  # Zero 'time'
        pass

    @kernel
    def dds_freq(self, idx, freq):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        if idx < 4:
            if idx < 2:
                if idx == 0:
                    self.dds0.set(frequency=freq)
                else:
                    self.dds1.set(frequency=freq)
            else:
                if idx == 2:
                    self.dds2.set(frequency=freq)
                else:
                    self.dds3.set(frequency=freq)
        else:
            if idx < 6:
                if idx == 4:
                    self.dds4.set(frequency=freq)
                else:
                    self.dds5.set(frequency=freq)
            else:
                if idx == 6:
                    self.dds6.set(frequency=freq)
                else:
                    self.dds7.set(frequency=freq)

    @kernel
    def dds_phase(self, idx, phase):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        if idx < 4:
            if idx < 2:
                if idx == 0:
                    self.dds0.phase(phase)
                else:
                    self.dds1.phase(phase)
            else:
                if idx == 2:
                    self.dds2.phase(phase)
                else:
                    self.dds3.phase(phase)
        else:
            if idx < 6:
                if idx == 4:
                    self.dds4.phase(phase)
                else:
                    self.dds5.phase(phase)
            else:
                if idx == 6:
                    self.dds6.phase(phase)
                else:
                    self.dds7.phase(phase)

    @kernel
    def dds_ampl(self, idx, ampl):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        if idx < 4:
            if idx < 2:
                if idx == 0:
                    self.dds0.set(amplitude=ampl)
                else:
                    self.dds1.set(amplitude=ampl)
            else:
                if idx == 2:
                    self.dds2.set(amplitude=ampl)
                else:
                    self.dds3.set(amplitude=ampl)
        else:
            if idx < 6:
                if idx == 4:
                    self.dds4.set(amplitude=ampl)
                else:
                    self.dds5.set(amplitude=ampl)
            else:
                if idx == 6:
                    self.dds6.set(amplitude=ampl)
                else:
                    self.dds7.set(amplitude=ampl)

    @kernel
    def dds_switch_on(self, idx):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        if idx < 4:
            if idx < 2:
                if idx == 0:
                    self.dds0_switch.on()
                else:
                    self.dds1_switch.on()
            else:
                if idx == 2:
                    self.dds2_switch.on()
                else:
                    self.dds3_switch.on()
        else:
            if idx < 6:
                if idx == 4:
                    self.dds4_switch.on()
                else:
                    self.dds5_switch.on()
            else:
                if idx == 6:
                    self.dds6_switch.on()
                else:
                    self.dds7_switch.on()

    @kernel
    def dds_switch_off(self, idx):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        if idx < 4:
            if idx < 2:
                if idx == 0:
                    self.dds0_switch.off()
                else:
                    self.dds1_switch.off()
            else:
                if idx == 2:
                    self.dds2_switch.off()
                else:
                    self.dds3_switch.off()
        else:
            if idx < 6:
                if idx == 4:
                    self.dds4_switch.off()
                else:
                    self.dds5_switch.off()
            else:
                if idx == 6:
                    self.dds6_switch.off()
                else:
                    self.dds7_switch.off()

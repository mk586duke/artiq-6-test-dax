import random
import time

import numpy as np
from artiq.experiment import *

from euriqafrontend.experiments.GST import GST  # local


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
        self.qubit_freq = [self.freq_to_bin(f * MHz) for f in [0.5, 0.5]]
        self.qubit_ampl = [1023, 1023]
        self.qubit_pi2_mu = [
            seconds_to_mu(t * us, self.core) for t in [5, 5]
        ]  # List of ion pi/2-times

        # Constants
        self.DDS_pi2 = 4096  # Need to figure this out
        self.gate_pre_mu = seconds_to_mu(1 * us, self.core)
        self.gate_post_mu = seconds_to_mu(1 * us, self.core)

        # Sequence information
        Gi = 0
        Gx = 1
        Gy = 2
        gst = GST.Parser(["Gi", "Gx", "Gy"])
        gst_set = gst.read("GST/MyDataTemplate.txt")
        max_len = max([len(x) for x in gst_set])
        gst_pad = [x + (max_len - len(x)) * [[Gi]] for x in gst_set]
        gst_a = gst_pad.copy()
        random.shuffle(gst_a)
        gst_b = gst_pad.copy()
        random.shuffle(gst_b)
        self.GST = [[d + e for d, e in zip(a, b)] for a, b in zip(gst_a, gst_b)]
        # for g in self.GST:
        #    print(g)

        # self.GST = [] # list of sequences
        # self.GST.append([[Gy, Gx]])
        # self.GST.append([[Gy, Gx], [Gx, Gy]])
        # self.GST.append([[Gy, Gx], [Gx, Gy], [Gy, Gy]])
        # self.GST.append([[Gy, Gx], [Gx, Gy], [Gy, Gy], [Gx, Gx]])
        # self.GST.append([[Gy, Gx], [Gx, Gy], [Gy, Gy], [Gx, Gx], [Gy, Gx]])
        # self.r = []
        # self.r += [[Gy, Gx], [Gx, Gy], [Gy, Gy], [Gx, Gx], [Gy, Gx]]
        # self.r += [[Gy, Gx], [Gx, Gy], [Gy, Gy], [Gx, Gx], [Gy, Gx]]
        # self.r += [[Gy, Gx], [Gx, Gy], [Gy, Gy], [Gx, Gx], [Gy, Gx]]
        # self.GST.append(self.r)

    def freq_to_bin(self, freq):
        dds_clk_freq = np.int64(10 ** 9)  # 1 GHz clock
        bin_width = np.int64(2 ** 48)  # 48 bit frequency tuning word
        bin_freq = np.int64((freq / dds_clk_freq) * bin_width)
        return bin_freq

    def run(self):
        self.core.reset()  # Clear out PHYS bufs, just in case.
        t0 = time.time()
        for k in range(1):
            self.run_step(10)
        t1 = time.time()
        print("Done")
        print(t1 - t0)

    @kernel
    def exp_setup(self):
        pass

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

    # @kernel
    # def dds_phase(self,idx, phase):
    #     # Use a binary matching to reduce the
    #     # number of comparisons to log2(# dds's)
    #     if idx<4:
    #         if idx<2:
    #             if idx==0:
    #                 self.dds0.set(phase=phase)
    #             else:
    #                 self.dds1.set(phase=phase)
    #         else:
    #             if idx==2:
    #                 self.dds2.set(phase=phase)
    #             else:
    #                 self.dds3.set(phase=phase)
    #     else:
    #         if idx<6:
    #             if idx==4:
    #                 self.dds4.set(phase=phase)
    #             else:
    #                 self.dds5.set(phase=phase)
    #         else:
    #             if idx==6:
    #                 self.dds6.set(phase=phase)
    #             else:
    #                 self.dds7.set(phase=phase)
    #     delay_mu(self.DDS_delay)

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

    @kernel
    def Gi(self, ion):
        delay_mu(self.gate_pre_mu)
        delay_mu(self.qubit_pi2_mu[ion])
        delay_mu(self.gate_post_mu)
        pass

    @kernel
    def Gx(self, ion):
        dds = self.qubit_dds[ion]
        self.dds_phase(dds, 0)
        delay_mu(self.gate_pre_mu)
        self.dds_switch_on(dds)
        delay_mu(self.qubit_pi2_mu[ion])
        self.dds_switch_off(dds)
        delay_mu(self.gate_post_mu)
        pass

    @kernel
    def Gy(self, ion):
        dds = self.qubit_dds[ion]
        self.dds_phase(dds, self.DDS_pi2)
        delay_mu(self.gate_pre_mu)
        self.dds_switch_on(dds)
        delay_mu(self.qubit_pi2_mu[ion])
        self.dds_switch_off(dds)
        delay_mu(self.gate_post_mu)
        pass

    @kernel
    def run_step(self, n):
        self.core.break_realtime()
        self.exp_setup()
        for s in self.GST:
            for rep in range(n):
                # t=0 starts here (with a slight delay = 125 us)
                # Using break_realtime in case this point is
                # reached before the previous experiment's PHYS
                # entries are complete.
                self.core.break_realtime()
                # Run experiment
                self.run_seq1(s)
        pass

    @kernel
    def timing(self):
        self.core.break_realtime()
        phase = 0
        t0 = self.core.get_rtio_counter_mu()
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        # self.dds0.phase(phase)
        delay_mu(10000)
        t1 = self.core.get_rtio_counter_mu()
        print(mu_to_seconds(t1 - t0))

    @kernel
    def run_seq1(self, seq):
        # Initialize hardware to known state
        self.exp_init()

        # Startup delay
        delay(100 * us)

        # For some reason, this single line speeds up processing by ~25% (7 us vs 9 us per ion per step)
        self.dds0.phase(0)

        # Give enough delay for the sequence to program
        s_delay_mu = seconds_to_mu(len(seq) * self.qubit_count * 7 * us)
        delay_mu(s_delay_mu)  # Headway per step

        # Trigger scope
        self.out1_0.pulse(1 * us)

        # Loop through each step in the sequence
        t0 = self.core.get_rtio_counter_mu()
        gxy_count = 0
        for s in seq:
            # Parallel operations on each ion.
            # The "with parallel" construction does not work across the interior of a for loop.
            # We need to mimic this effect. The following assumes that all the G's take the same
            # real-time.
            t = now_mu()
            ion = 0
            for g in s:
                at_mu(t)
                if g == 0:
                    self.Gi(ion)
                    gxy_count += 1
                elif g == 1:
                    self.Gx(ion)
                    gxy_count += 1
                elif g == 2:
                    self.Gy(ion)
                else:
                    self.Gi(ion)
                ion += 1
        t1 = self.core.get_rtio_counter_mu()
        print(seq)
        print(gxy_count, mu_to_seconds(t1 - t0), mu_to_seconds(t1 - t0) / gxy_count)
        # print(tb-ta)
        pass

    @kernel
    def Gi0(self):
        delay_mu(self.gate_pre_mu)
        delay_mu(self.qubit_pi2_mu[0])
        delay_mu(self.gate_post_mu)
        pass

    @kernel
    def Gx0(self):
        self.dds0.set(phase=0)
        delay_mu(self.gate_pre_mu)
        self.dds0_switch.on()
        delay_mu(self.qubit_pi2_mu[0])
        self.dds0_switch.off()
        delay_mu(self.gate_post_mu)
        pass

    @kernel
    def Gy0(self):
        self.dds0.set(phase=self.DDS_pi2)
        delay_mu(self.gate_pre_mu)
        self.dds0_switch.on()
        delay_mu(self.qubit_pi2_mu[0])
        self.dds0_switch.off()
        delay_mu(self.gate_post_mu)

    @kernel
    def Gi1(self):
        delay_mu(self.gate_pre_mu)
        delay_mu(self.qubit_pi2_mu[0])
        delay_mu(self.gate_post_mu)
        pass

    @kernel
    def Gx1(self):
        self.dds2.set(phase=0)
        delay_mu(self.gate_pre_mu)
        self.dds2_switch.on()
        delay_mu(self.qubit_pi2_mu[1])
        self.dds2_switch.off()
        delay_mu(self.gate_post_mu)
        pass

    @kernel
    def Gy1(self):
        self.dds2.set(phase=self.DDS_pi2)
        delay_mu(self.gate_pre_mu)
        self.dds2_switch.on()
        delay_mu(self.qubit_pi2_mu[1])
        self.dds2_switch.off()
        delay_mu(self.gate_post_mu)

    @kernel
    def run_seq2(self, seq):
        # Initialize hardware to known state
        self.exp_init()

        # Trigger scope
        self.out1_0.pulse(1 * us)

        # Startup delay
        delay(100 * us)

        # Loop through each step in the sequence
        s_delay_mu = seconds_to_mu(self.qubit_count * 9 * us)
        t0 = self.core.get_rtio_counter_mu()
        for s in seq:
            delay_mu(s_delay_mu)  # 150 us for two ions
            with parallel:
                with sequential:  # Probably not needed
                    g = s[0]
                    if g == 0:
                        self.Gi0()
                    elif g == 1:
                        self.Gx0()
                    elif g == 2:
                        self.Gy0()
                    else:
                        self.Gi0()
                with sequential:  # Probably not needed
                    g = s[1]
                    if g == 0:
                        self.Gi1()
                    elif g == 1:
                        self.Gx1()
                    elif g == 2:
                        self.Gy1()
                    else:
                        self.Gi1()
        t1 = self.core.get_rtio_counter_mu()
        print(seq)  # This should eat up some of the cursor lead on the realtime clock
        print(mu_to_seconds(t1 - t0))
        # print(mu_to_seconds(125000))

        pass

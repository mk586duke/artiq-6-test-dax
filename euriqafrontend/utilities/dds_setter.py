import numpy as np
from artiq.experiment import *


class dds_setter(EnvExperiment):
    """Program DDS Channels"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device(
            "oeb"
        )  # what is this TTL doing? Just copied from GST code...

        self.setattr_device("reset01")  # what do these reset?
        self.setattr_device("reset23")
        self.setattr_device("reset45")
        self.setattr_device("reset67")
        self.setattr_device("reset89")

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

        self.setattr_argument(
            "dds0_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 0",
        )
        self.setattr_argument(
            "dds1_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 1",
        )
        self.setattr_argument(
            "dds2_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 2",
        )
        self.setattr_argument(
            "dds3_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 3",
        )
        self.setattr_argument(
            "dds4_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 4",
        )
        self.setattr_argument(
            "dds5_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 5",
        )
        self.setattr_argument(
            "dds6_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 6",
        )
        self.setattr_argument(
            "dds7_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 7",
        )
        self.setattr_argument(
            "dds8_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 8 -- Does Not Work",
        )
        self.setattr_argument(
            "dds9_frequency",
            NumberValue(default=200 * MHz, unit="MHz", ndecimals=3, step=0.001),
            "Channel 9 -- Does Not Work",
        )

        self.setattr_argument(
            "dds0_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 0",
        )
        self.setattr_argument(
            "dds1_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 1",
        )
        self.setattr_argument(
            "dds2_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 2",
        )
        self.setattr_argument(
            "dds3_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 3",
        )
        self.setattr_argument(
            "dds4_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 4",
        )
        self.setattr_argument(
            "dds5_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 5",
        )
        self.setattr_argument(
            "dds6_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 6",
        )
        self.setattr_argument(
            "dds7_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 7",
        )
        self.setattr_argument(
            "dds8_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 8 -- Does Not Work",
        )
        self.setattr_argument(
            "dds9_amplitude",
            NumberValue(default=500, unit="", ndecimals=0, step=1, min=0),
            "Channel 9 -- Does Not Work",
        )

        self.setattr_argument("dds0_state", BooleanValue(False), "Channel 0")
        self.setattr_argument("dds1_state", BooleanValue(False), "Channel 1")
        self.setattr_argument("dds2_state", BooleanValue(False), "Channel 2")
        self.setattr_argument("dds3_state", BooleanValue(False), "Channel 3")
        self.setattr_argument("dds4_state", BooleanValue(False), "Channel 4")
        self.setattr_argument("dds5_state", BooleanValue(False), "Channel 5")
        self.setattr_argument("dds6_state", BooleanValue(False), "Channel 6")
        self.setattr_argument("dds7_state", BooleanValue(False), "Channel 7")
        self.setattr_argument(
            "dds8_state", BooleanValue(False), "Channel 8 -- Does Not Work"
        )
        self.setattr_argument(
            "dds9_state", BooleanValue(False), "Channel 9 -- Does Not Work"
        )

        # parameters
        self.dds_update_time_us = (
            100 * us
        )  # DDS wait time after programming frequency -- check

    def prepare(self):
        self.dds_frequency = [
            self.freq_to_bin(self.dds0_frequency),
            self.freq_to_bin(self.dds1_frequency),
            self.freq_to_bin(self.dds2_frequency),
            self.freq_to_bin(self.dds3_frequency),
            self.freq_to_bin(self.dds4_frequency),
            self.freq_to_bin(self.dds5_frequency),
            self.freq_to_bin(self.dds6_frequency),
            self.freq_to_bin(self.dds7_frequency),
            self.freq_to_bin(self.dds8_frequency),
            self.freq_to_bin(self.dds9_frequency),
        ]
        self.dds_amplitude = [
            np.int(i)
            for i in [
                self.dds0_amplitude,
                self.dds1_amplitude,
                self.dds2_amplitude,
                self.dds3_amplitude,
                self.dds4_amplitude,
                self.dds5_amplitude,
                self.dds6_amplitude,
                self.dds7_amplitude,
                self.dds8_amplitude,
                self.dds9_amplitude,
            ]
        ]
        self.dds_state = [
            self.dds0_state,
            self.dds1_state,
            self.dds2_state,
            self.dds3_state,
            self.dds4_state,
            self.dds5_state,
            self.dds6_state,
            self.dds7_state,
            self.dds8_state,
            self.dds9_state,
        ]

    def freq_to_bin(self, freq):
        freq = np.int64(freq)
        dds_clk_freq = np.int64(10 ** 9)  # 1 GHz clock
        bin_width = np.int64(2 ** 48)  # 48 bit frequency tuning word
        bin_freq = np.int64((freq / dds_clk_freq) * bin_width)
        return bin_freq

    @kernel
    def run(self):
        self.core.reset()
        self.exp_init()

    @kernel
    def exp_init(self):
        self.oeb.off()

        self.reset01.off()
        self.reset23.off()
        self.reset45.off()
        self.reset67.off()
        self.reset89.off()

        for i in range(10):
            self.dds_switch_off(i)

        # Set up DDS bus for the DDS's we are actively using. Since 0/1, 2/3, etc share
        # a bus, setting up the other DDS will prevent changes to these.
        delay(10 * us)
        with parallel:
            self.dds0.setup_bus(write_div=6)
            self.dds2.setup_bus(write_div=6)
            self.dds4.setup_bus(write_div=6)
            self.dds6.setup_bus(write_div=6)
            self.dds8.setup_bus(write_div=6)

        delay(self.dds_update_time_us)
        for i in [0, 2, 4, 6, 8]:
            self.dds_freq(i, self.dds_frequency[i])
        delay(self.dds_update_time_us)
        for i in [0, 2, 4, 6, 8]:
            self.dds_ampl(i, self.dds_amplitude[i])
        delay(self.dds_update_time_us)
        for i in [0, 2, 4, 6, 8]:
            self.dds_phase(i, 0)

        delay(20 * us)
        with parallel:
            self.dds1.setup_bus(write_div=6)
            self.dds3.setup_bus(write_div=6)
            self.dds5.setup_bus(write_div=6)
            self.dds7.setup_bus(write_div=6)
            self.dds9.setup_bus(write_div=6)

        delay(self.dds_update_time_us)
        for i in [1, 3, 5, 7, 9]:
            self.dds_freq(i, self.dds_frequency[i])

        delay(self.dds_update_time_us)
        for i in [1, 3, 5, 7, 9]:
            self.dds_ampl(i, self.dds_amplitude[i])

        delay(self.dds_update_time_us)
        for i in [1, 3, 5, 7, 9]:
            self.dds_phase(i, 0)

        for i in range(10):
            self.dds_switch(i, self.dds_state[i])
        pass

    @kernel
    def dds_freq(self, idx, freq):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        getattr(self, "dds" + str(idx)).set(freq)

    @kernel
    def dds_ampl(self, idx, ampl):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        if idx < 8:
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
        else:
            if idx == 8:
                self.dds8.set(amplitude=ampl)
            else:
                self.dds9.set(amplitude=ampl)

    @kernel
    def dds_phase(self, idx, phase):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        if idx < 8:
            if idx < 4:
                if idx < 2:
                    if idx == 0:
                        self.dds0.set(phase=phase)
                    else:
                        self.dds1.set(phase=phase)
                else:
                    if idx == 2:
                        self.dds2.set(phase=phase)
                    else:
                        self.dds3.set(phase=phase)
            else:
                if idx < 6:
                    if idx == 4:
                        self.dds4.set(phase=phase)
                    else:
                        self.dds5.set(phase=phase)
                else:
                    if idx == 6:
                        self.dds6.set(phase=phase)
                    else:
                        self.dds7.set(phase=phase)
        else:
            if idx == 8:
                self.dds8.set(phase=phase)
            else:
                self.dds9.set(phase=phase)

    @kernel
    def dds_switch(self, idx, state):
        # Use a binary matching to reduce the
        # number of comparisons to log2(# dds's)
        if state == True:
            if idx < 8:
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
            else:
                if idx == 8:
                    self.dds8_switch.on()
                else:
                    self.dds9_switch.on()
        else:
            if idx < 8:
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
            else:
                if idx == 8:
                    self.dds8_switch.off()
                else:
                    self.dds9_switch.off()

    @kernel
    def dds_switch_off(self, idx):
        self.dds_switch(idx, False)

    @kernel
    def dds_switch_on(self, idx):
        self.dds_switch(idx, True)

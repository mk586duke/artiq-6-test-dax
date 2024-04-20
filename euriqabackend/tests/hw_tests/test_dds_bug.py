import numpy as np
from artiq.experiment import *


def input_state() -> TList(TInt32):
    return list(map(int, input("Enter bank, output, state: ").split()))


class Test_DDS_Bug(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("led0")
        self.setattr_device("led1")
        self.setattr_device("oeb")

        self.setattr_device("io_update0")
        self.setattr_device("io_update1")

        self.setattr_device("reset01")

        self.setattr_device("dds0_switch")
        self.setattr_device("dds1_switch")

        self.setattr_device("dds0")
        self.setattr_device("dds1")

        self.setattr_device("out1_0")

        # self.frequency = np.int64(0x7FFFFFFF7FFFFFFF)

        self.frequency = np.int64(0x000FFFFFFFFFFFFE)

    @kernel
    def run(self):
        self.core.reset()
        self.oeb.off()

        self.dds0_switch.on()
        self.dds1_switch.on()

        self.reset01.off()

        self.dds0.setup_bus(write_div=6)
        # delay(10 * us)
        # self.dds0.set(amplitude=1023)
        delay(10 * us)
        self.dds0.set(frequency=self.frequency)

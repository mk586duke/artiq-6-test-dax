from artiq.experiment import *


class testDAC(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("ttl3")
        self.setattr_device("dac0")

    @kernel
    def run(self):
        self.dac0.setup_bus(write_div=3)
        delay(50 * us)
        self.dac0.set([253])
        self.ttl3.pulse(50 * ns)

    # self.dac0.write_channel(0, 21352)

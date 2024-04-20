from artiq.experiment import *


class testSPI(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("oeb")
        self.setattr_device("testSPI")

    @kernel
    def run(self):
        self.core.reset()
        self.oeb.off()
        self.testSPI.setup_bus(write_div=8)
        delay(200 * us)
        self.testSPI.set()

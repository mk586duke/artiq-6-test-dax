from artiq.experiment import *


class HelloControllerTest(EnvExperiment):
    """RPC Tool Test"""

    def build(self):
        self.setattr_device("scheduler")
        self.setattr_device("core")
        self.hello_control = self.get_device("hello_control")
        self.setattr_device("artiq_test_ttl")

        self.setattr_argument("string_val", StringValue("print me!"), "String to print")

    def run(self):
        self.hello_control.message(self.string_val)

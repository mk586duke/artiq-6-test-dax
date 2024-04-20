from artiq.experiment import *


class hello(EnvExperiment):
    """Hello Message Tool (Tutorial) Tester"""

    def build(self):
        self.setattr_device("hello")
        self.setattr_argument("message", StringValue(default=""))
        self.setattr_argument(
            "send_to", EnumerationValue(["direct_print", "remote_print", "log"])
        )

    def run(self):
        if self.send_to == "direct_print":
            print(self.message)
        elif self.send_to == "remote_print":
            self.hello.message(self.message)
        else:
            self.hello.logger_message(self.message)

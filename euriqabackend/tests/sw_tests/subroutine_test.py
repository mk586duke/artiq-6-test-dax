from artiq.experiment import *


class _MessageInASubroutine(EnvExperiment):
    """print the message"""

    def build(self):
        self.setattr_device("core")

    def run(self):
        print("a message in a subroutine")


class _VariableMessageInASubroutine(EnvExperiment):
    """choose the message"""

    def build(self):
        self.setattr_device("core")
        self.setattr_argument("secret_message", BooleanValue(True))

    def run(self):
        if self.secret_message:
            print(
                "here's a secret - you taste something different every time you try new flavor of quantum ice cream."
            )
        else:
            print("there's no secret - it's just a message in a subroutine")


class _DatasetMessageInASubroutine(EnvExperiment):
    """choose the message from dataset"""

    def build(self):
        self.setattr_device("core")

    def prepare(self):
        self.secret_message = self.get_dataset("secret_message")

    def run(self):
        if self.secret_message:
            print(
                "here's a secret - you taste something different every time you try new flavor of quantum ice cream."
            )
        else:
            print("there's no secret - it's just a message in a subroutine")


class _EchoVariable(EnvExperiment):
    """Echo Variable Value"""

    def build(self):
        self.setattr_device("core")
        self.setattr_argument("echo_variable", StringValue(""))

    def run(self):
        print(self.echo_variable)


class _EchoParameter(EnvExperiment):
    """Echo Parameter Value"""

    def build(self):
        self.setattr_device("core")
        self.echo_variable = "blah blah blah"

    def run(self):
        print(self.echo_variable)


class _EchoPassedParameter(EnvExperiment, HasEnvironment):
    """Echo Passed Parameter Value"""

    def build(self, **kwargs):
        self.setattr_device("core")
        self.setattr_argument("echo_variable", StringValue(kwargs["echo_variable"]))

    def run(self):
        print(self.echo_variable)

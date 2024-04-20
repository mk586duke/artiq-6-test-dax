from artiq.experiment import *

from .subroutine_test import _DatasetMessageInASubroutine
from .subroutine_test import _EchoParameter
from .subroutine_test import _EchoPassedParameter
from .subroutine_test import _EchoVariable
from .subroutine_test import _MessageInASubroutine
from .subroutine_test import _VariableMessageInASubroutine


class ImportSubroutineTest(EnvExperiment):
    """Import Subroutine Test"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("out1_2")

        self.setattr_argument("secret_message", BooleanValue(True))
        self.setattr_argument("echo_variable", StringValue(""))
        self.setattr_argument(
            "choose_subroutine",
            EnumerationValue(
                [
                    "Fixed",
                    "Variable",
                    "Saved Data",
                    "Echo Variable",
                    "Echo Parameter",
                    "Echo Passed Parameter",
                ]
            ),
        )

    def prepare(self):
        if self.choose_subroutine == "Fixed":
            self.mt = _MessageInASubroutine(self)
        elif self.choose_subroutine == "Variable":
            self.mt = _VariableMessageInASubroutine(self)
        elif self.choose_subroutine == "Saved Data":
            self.mt = _DatasetMessageInASubroutine(self)
        elif self.choose_subroutine == "Echo Variable":
            self.mt = _EchoVariable(self)
        elif self.choose_subroutine == "Echo Parameter":
            self.mt = _EchoParameter(self)
        elif self.choose_subroutine == "Echo Passed Parameter":
            self.mt = _EchoPassedParameter(self, echo_variable="pass me")
        self.mt.prepare()

    def run(self):
        self.mt.run()


class UpdateSecret(EnvExperiment):
    """Update dataset info"""

    def build(self):
        self.setattr_argument("stored_secret_message", BooleanValue(True))

    def prepare(self):
        self.set_dataset(
            "secret_message", self.stored_secret_message, persist=True, archive=False
        )
        self.set_dataset("secret_message_value", 42, persist=True, archive=False)

    def run(self):
        pass

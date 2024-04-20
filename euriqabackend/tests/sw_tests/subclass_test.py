from artiq.experiment import *


class _DerivedClass(EnvExperiment, HasEnvironment):
    def build_me(self, **kwargs):
        for key, value in kwargs.items():
            print(key, " : ", value, " : ", type(value))
        if "subroutines" in kwargs:
            if "Variable" in kwargs["subroutines"]:
                self.setattr_argument("secret_message", BooleanValue(True))
            if "Echo Variable" in kwargs["subroutines"]:
                self.setattr_argument("echo_variable", StringValue(""))
            self.setattr_argument(
                "choose_subroutine", EnumerationValue(kwargs["subroutines"])
            )

    def message_in_a_subclass(self):
        print("a message in a subclass.")

    def kernel_message_in_a_subclass(self, x):
        if x:
            print("a secret message in a subclass.")
        else:
            print("a not-so-secret message in a subclass.")

    def variable_message_in_a_subclass(self):
        self.kernel_message_in_a_subclass(self.secret_message)

    def dataset_message_in_a_subclass(self):
        self.kernel_message_in_a_subclass(self.get_dataset("secret_message"))

    def echo_message_in_a_subclass(self):
        print(self.echo_variable)


class tryThis(_DerivedClass):
    """Derived Class Test"""

    def build(self):
        self.setattr_device("core")
        self.build_me(
            subroutines=[
                "Fixed",
                "Variable",
                "Saved Data",
                "Echo Variable",
                "Test Kernel",
            ]
        )

    def run(self):
        if self.choose_subroutine == "Fixed":
            self.message_in_a_subclass()
        elif self.choose_subroutine == "Variable":
            self.variable_message_in_a_subclass()
        elif self.choose_subroutine == "Saved Data":
            self.dataset_message_in_a_subclass()
        elif self.choose_subroutine == "Echo Variable":
            self.echo_message_in_a_subclass()
        elif self.choose_subroutine == "Test Kernel":
            self.kernel_message_in_a_subclass(False)

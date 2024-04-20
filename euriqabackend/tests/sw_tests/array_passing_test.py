"""Test passing arrays to & from kernel.

All methods work unless otherwise specified.
"""
import artiq
from artiq.language import host_only
from artiq.language import kernel


class ArrayReturnTest(artiq.language.environment.EnvExperiment):
    """Check if 2-D arrays can be created & manipulated on core."""

    def build(self):
        """Declare needed devices."""
        self.setattr_device("core")

    @host_only
    def run(self):
        """Run tests on 2D arrays."""
        self.buffer = [[0 for i in range(10)] for j in range(5)]

        print("Pre-buffer modification", self.buffer)
        self.array_local_test()
        print("Post-buffer modification", self.buffer)

        # reset buffer
        self.buffer = [[0 for i in range(10)] for j in range(5)]
        self.buffer = self.array_as_arg_test(self.buffer)
        print("Post-argument result", self.buffer)

    @kernel
    def array_local_test(self):
        """Test modifying a local (within class) array."""
        counter = 0
        for i in range(len(self.buffer)):
            for j in range(len(self.buffer[0])):
                self.buffer[i][j] = counter
                counter += 1

        print("Kernel modifying local buffer", self.buffer)

    @kernel
    def array_as_arg_test(self, array):
        """Test modifying and returning an array."""
        counter = 0
        for i in range(len(array)):
            for j in range(len(array[0])):
                array[i][j] = counter
                counter += 1

        print("Kernel modified parameter array", array)

        return array

"""Test creating & manipulating 2+ Dimensional arrays on ARTIQ core.

Most methods work, unless otherwise specified.

TODO:
    * Write some sort of wrapper class around numpy arrays to convert them to
    lists for use on the core.
"""
import artiq
import numpy as np
from artiq.language import kernel


class TwoDimensionalArrayTest(artiq.language.environment.EnvExperiment):
    """Check if 2-D arrays can be created & manipulated on core."""

    def build(self):
        """Declare needed devices."""
        self.setattr_device("core")

    def prepare(self):
        """Prepare and pre-allocate arrays."""
        # Must convert numpy arrays to lists b/c core/compiler can't handle arrays
        self.np_array = np.zeros((10, 30), dtype=np.int32).tolist()
        self.np_array_3d = np.zeros((10, 5, 30), dtype=np.int32).tolist()
        # core can't handle tuples, so need to convert to list
        # also can't do (on kernel): `list(self.np_array_3d.shape)`
        self.np_array_3d_shape = list((10, 5, 30))

    @kernel
    def run(self):
        """Run tests on 2D arrays."""
        self.test_numpy_array()
        self.test_python_array()  # DOES NOT WORK
        self.test_python_array_2()
        self.test_numpy_array_3d()

    @kernel
    def test_numpy_array(self):
        """Test using a converted numpy array (WORKS).

        NOTE: to use a numpy array easily, MUST be pre-allocated/declared in
        :meth:`prepare`/:meth:`init`, b/c the ARTIQ compiler doesn't like numpy
        arrays in `@kernel` code
        """
        arr = self.np_array

        for i in range(len(arr)):
            for j in range(len(arr[0])):
                arr[i][j] = i * len(arr) + j

        print("Numpy solution:")
        print(arr)

    @kernel
    def test_python_array(self):
        """Test using an allocated multiplied array (DOES NOT WORK).

        This seems to have some sort of problem with referencing.
        """
        # ALLOCATION DOES NOT WORK
        x_len = 10
        y_len = 5
        arr = [[0] * x_len] * y_len

        counter = 0
        for i in range(y_len):
            for j in range(x_len):
                arr[i][j] = counter
                counter += 1

        print("Python method 1 (BROKEN)")
        print(arr)

    @kernel
    def test_python_array_2(self):
        """Test using an explicitly allocated array (WORKS)."""
        x_len = 10
        y_len = 5
        arr = [[0 for i in range(x_len)] for j in range(y_len)]

        counter = 0
        for i in range(y_len):
            for j in range(x_len):
                arr[i][j] = counter
                counter += 1

        print("Python method 2")
        print(arr)

    @kernel
    def test_numpy_array_3d(self):
        """Test using a pre-allocated numpy array (WORKS).

        Assumes array is not jagged.
        """
        arr = self.np_array_3d

        counter = 0
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                for k in range(len(arr[0][0])):
                    arr[i][j][k] = counter
                    counter += 1

        print("3d numpy array:")
        print(arr)

        arr = self.np_array_3d
        arr_shape = self.np_array_3d_shape

        counter = 0
        for i in range(arr_shape[0]):
            for j in range(arr_shape[1]):
                for k in range(arr_shape[2]):
                    arr[i][j][k] = counter
                    counter += 1

        print("3d numpy array (with shape):")
        print(arr)

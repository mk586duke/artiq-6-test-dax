"""Low-dependency ARTIQ functions.

Meant to replace basic Python functions that are missing.
They aren't implemented quickly or using smart algorithms, just quick code because
they might be useful.
"""

import numpy as np
from artiq.language import TInt32, TList, TNone, kernel


@portable
def copy_array(from_array, to_array) -> TNone:
    """Copy values from an array to another array.

    Stops after exhausting from_array, does not check lengths match.
    """
    for i in range(len(from_array)):
        to_array[i] = from_array[i]

@portable
def list_max_int32(array: TList(TInt32)) -> TInt32:
    """Find the maximum value in an array of int32.

    Dumb algorithm, but should work.
    """
    max_value = np.int32(-1<<31)    # most negative value
    for i in range(len(array)):
        if array[i] > max_value:
            max_value = array[i]
    return max_value

@portable
def list_max_int64(array: TList(TInt64)) -> TInt64:
    """Find the maximum value in an array of int64.

    Somewhat dumb algorithm, but should work.
    """
    max_value = np.int64(-1<<63‬)   # most negative value
    for i in range(len(array)):
        if array[i] > max_value:
            max_value = array[i]
    return max_value

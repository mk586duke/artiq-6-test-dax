"""Shared utilities that many modules (or experiments) need access to."""
import logging


_LOGGER = logging.getLogger(__name__)


def parse_ion_input(input_str: str, python_ranges: bool = True):
    """Parse a string containing a list of ion indices (in center index notation).

    Args:
        input_str (str): A comma separated string of integers.
        python_ranges (bool): Whether Python range notation should be used by default.
            If True, then Python notation i:j can be used to indicate ranges, which are
            NOT inclusive of the right side (in keeping with standard Python).
            If False, then ``i:j`` will include both i & j. This is NOT in keeping
            with standard Python slicing/ranges, but might be more user-intuitive.


    Returns:
        List[int]: A sorted list of integers generated from the input string

    Raises:
        ValueError: if the string cannot be parsed for any reason.

    Example:
        >>> parse_ion_input("-9,0:4,-5,", python_ranges=True)
        [-9, -5, 0, 1, 2, 3]
        >>> parse_ion_input("-9,0:4,-5", python_ranges=False)
        [-9, -5, 0, 1, 2, 3, 4]
        >>> parse_ion_input("") # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        AssertionError: Must provide some input to the parser
    """
    _LOGGER.debug("Attempting to parse ion string %s", input_str)
    assert len(input_str) >= 1, "Must provide some input to the parser"
    indices = []
    # split string at ",", then split each substr at ":"
    partitioned_str = (
        s.strip().partition(":") for s in input_str.lstrip(",").rstrip(",").split(",")
    )
    # TODO: check that no invalid characters are contained in string (e.g. " ", ";")
    for l, partition, r in partitioned_str:
        # partition breaks everything into 3-tuples.
        # If finds ":", then partition & r != "",
        # otherwise they are ":" & the second str
        if partition != "":
            if python_ranges:
                indices.extend(range(int(l), int(r)))
            else:
                indices.extend(range(int(l), int(r) + 1))
        else:
            indices.append(int(l))
    return sorted(indices)


def _parse_ion_input_old(s):
    """
    Old, unimproved version of the ion input parsing.

    SHOULD NOT BE USED FOR NEW CODE. This code promises to use Python-like slice/range
    notation, but is not equivalent due to including endpoints, while Python does not.


    Args:
        s (str): A Comma seperated string of inters.
            Python notation i:j can be used to indicate ranges

    Returns:
        A list of integers generated from the input string

        Example:
            >>> _parse_ion_input_old("-9,-5,0:4")
            [-9, -5, 0, 1, 2, 3, 4]
    """
    nums = []
    for x in map(str.strip, s.split(",")):
        try:
            i = int(x)
            nums.append(i)
        except ValueError:
            if ":" in x:
                xr = list(map(str.strip, x.split(":")))
                nums.extend(range(int(xr[0]), int(xr[1]) + 1))
            else:
                _LOGGER.warning("Unknown string format for ion input: %s", x)
    return nums

"""Test :mod:`euriqafrontend.modules.utilities`."""
import typing

import pytest

import euriqafrontend.modules.utilities as utils


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("-3:3", [-3, -2, -1, 0, 1, 2]),
        ("5", [5]),
        ("5,", [5]),
        ("-100,-20,50,1:4", [-100, -20, 1, 2, 3, 50]),
        (" -20 , 0, 10\t", [-20, 0, 10]),
    ],
)
def test_parse_ion_string_valid(input_str: str, expected: typing.List[int]):
    assert utils.parse_ion_input(input_str) == expected


@pytest.mark.parametrize("input_str", [":", "1;2", "\n", "1::2", "1.0", ""])
def test_parse_ion_input_string_fails(input_str: str):
    with pytest.raises((ValueError, AssertionError)):
        utils.parse_ion_input(input_str)

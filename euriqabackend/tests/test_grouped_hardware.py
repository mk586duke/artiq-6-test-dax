"""Test that grouping hardware together functions properly."""
import os
import pathlib

import pytest
from artiq.language.units import Hz
from artiq.language.units import s
from artiq.master.databases import DeviceDB
from artiq.master.worker_db import DeviceManager

import euriqabackend.utilities.grouped_devices as grouper

# pylint: disable=redefined-outer-name

ARTIQ_ROOT_DIR = os.getenv("ARTIQ_ROOT")


def test_basic_grouping():
    """Test that basic (built-in) objects are grouped properly."""
    list_range = list(range(3))
    num_test_lists = 5

    # model device database as dictionary, and use range as keys
    device_dictionary = {i: list(list_range) for i in range(num_test_lists)}
    # test_list_input = [list_range for i in range(num_test_lists)]
    grouped_list = grouper.VectorGrouper(device_dictionary, device_dictionary.keys())
    assert num_test_lists in grouped_list.argument_length

    grouped_list.append(5)
    for wo in grouped_list.wrapped_obj_list:
        assert len(wo) == len(list_range) + 1


def test_invalid_grouping():
    """Test that errors are raised when doing invalid operations."""
    device_manager = {1: [1, 2, 3], 2: "abc"}

    with pytest.raises(TypeError):
        # raise error on construction, different types
        grouped = grouper.VectorGrouper(device_manager, device_manager.keys())

    # test passing bad key/device
    with pytest.raises(KeyError):
        grouped = grouper.VectorGrouper(device_manager, ["fake_key"])

    # test not passing enough arguments
    device_manager_2 = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9]}
    grouped = grouper.VectorGrouper(device_manager_2, device_manager_2.keys())
    with pytest.raises(ValueError):
        grouped.append([[3, 4, 5], [3]])

    # test too many arguments
    with pytest.raises(ValueError):
        grouped.append([[3, 4, 5, 6], [3]])


@pytest.fixture(scope="module")
def vector_grouped_device_factory():
    """Pytest Fixture to yield a grouped device for testing."""
    if ARTIQ_ROOT_DIR is None:
        pytest.fail(
            "ARTIQ_ROOT environment variable is not present. Must have ARTIQ_ROOT for device_db and other databases"
        )

    # Get available devices
    device_database = DeviceDB(pathlib.Path(ARTIQ_ROOT_DIR, "device_db.pyon"))
    device_manager = DeviceManager(device_database)

    # yield function to start requested device
    def _start_artiq_device(device_name: str):
        try:
            return device_manager.get(device_name)
        except KeyError:
            pytest.fail(
                "VectorGrouped device '{}' not found. Can't continue test".format(
                    device_name
                )
            )

    yield _start_artiq_device

    # cleanup and close devices.
    device_manager.close_devices()


@pytest.mark.parametrize("input_device_name", ["pmt_vectorgroup_test"])
def test_vectorgrouper_input(input_device_name, vector_grouped_device_factory):
    """Test input functionality of :class:`VectorGrouper` on ARTIQ devices."""
    # Test parameters
    OBSERVE_TIME = 1 * s
    PMT_DARK_COUNT_RATE = 100 * Hz

    # Get device
    grouped_input_device = vector_grouped_device_factory(input_device_name)

    # Run test
    grouped_input_device.gate_both(OBSERVE_TIME)
    input_pmt_count_rates = grouped_input_device.count() / OBSERVE_TIME
    for count_rate in input_pmt_count_rates:
        assert count_rate >= PMT_DARK_COUNT_RATE * 2

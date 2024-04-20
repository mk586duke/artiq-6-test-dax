"""
Unit test to test Sandia DAC driver & interface. Meant to be run with `pytest`.

Author: Drew Risinger, UMD JQI

Revision History:
    7/19/18:
        Write tests for Sandia DAC Mediator.

    7/18/18:
        Write tests for Sandia DAC Driver, and test
"""
import logging
import pathlib
import random
import sys
import time
import unittest.mock as mock
from typing import Dict
from typing import Tuple

import artiq.coredevice.core as core
import artiq.master.databases as db
import numpy
import pandas
import pytest

import euriqabackend.devices.sandia_dac.driver as driver
import euriqabackend.devices.sandia_dac.interface as interface

_LOGGER = logging.getLogger(__name__)

test_file_directory = pathlib.Path(__file__).resolve().parent

skip_if_no_fpga = pytest.mark.skipif(
    "ok" not in sys.builtin_module_names,
    reason="OpalKelly board interface not available",
)


# todo: add options for manual tests, specifying bitfile path, and slow tests
# https://docs.pytest.org/en/latest/example/simple.html


def pytest_addoption(parser):
    """Add option to run manual tests.

    Controls skipping tests by command-line option
    """
    parser.addoption("--runmanual", action="store_true", help="Run Manual tests")


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `manual` by default."""
    # to use this skip, use @pytest.mark.manual decorator
    if not config.getoption("--runmanual"):
        # --runmanual NOT given in cli, do not skip manual tests
        skip_manual = pytest.mark.skip(
            reason="Need --runmanual option to run. Needs manual verification"
        )
        for item in items:
            if "manual" in item.keywords:
                item.add_marker(skip_manual)


# **** FIXTURES ****
@pytest.fixture(
    scope="class",
    params=[
        pytest.param({"fpga": "SimDACFPGA", "simulation": True}, marks=pytest.mark.sim),
        pytest.param(
            {
                "fpga": "TrapVoltFPGA",
                "bitfile": "../devices/sandia_opalkelly/dac/DAC-firmware.bit",
            },
            marks=[skip_if_no_fpga, pytest.mark.hardware],
        ),
    ],
)
def sandia_dac_driver(request):
    """Yield a generic Sandia DAC Driver."""
    try:
        # Use bitfile as relative to current test path
        if request.param["bitfile"].startswith("."):
            request.param["bitfile"] = pathlib.Path(
                test_file_directory, request.param["bitfile"]
            ).resolve()
    except KeyError:
        pass  # bitfile key not found, just didn't pass as parameter
    dac = driver.SandiaDACDriverLow(**request.param)
    yield dac
    try:
        dac.close()
    except AttributeError:
        _LOGGER.warning("Already closed DAC manually.")


@pytest.fixture(scope="class")
def sandia_dac_interface(sandia_dac_driver: driver.SandiaDACDriverLow):
    """Generate a connected & initialized Sandia DAC Mediator.

    Uses a simulated device database.
    """
    # todo: fill in with live test to check device DB w/o mocking works,
    # by adding another interface fixture to params

    # Setup mocks to force the device DB to yield desired (fake) devices
    # without making full Device DB
    def fake_device_db_get(device_name: str):
        if device_name == sandia_dac_driver.fpga_name:
            return sandia_dac_driver
        elif device_name == "core":
            return mock.NonCallableMagicMock(spec=core.Core)
        else:
            raise ValueError("{} is not a permitted device".format(device_name))

    mock_device_db = mock.NonCallableMagicMock(spec=db.DeviceDB)
    mock_device_db.get.side_effect = fake_device_db_get

    yield interface.SandiaDACInterface(mock_device_db, sandia_dac_driver.fpga_name)


def gen_global_tweaks_contents(num_tweaks: int, num_pads: int) -> str:
    """
    Generate the contents of a global tweaks file.

    Args:
        num_tweaks: Number of total tweaks to generate
        num_pads: Number of pads to tweak

    Return:
        String formatted as contents of fake global tweaks file

    """
    retval = ""
    for i in range(num_tweaks):
        retval += "Line{0}={0}\n".format(i)

    for i in range(num_pads):
        retval += "PAD_{0}\t".format(i)
    retval += "\n"

    for i in range(num_tweaks):
        retval += "{}\t".format(i) * num_pads
        retval += "\n"
    return retval


@pytest.fixture(scope="session", params=[[2, 4], [3, 1], [3, 3]])
def dac_global_tweaks_file(
    request, tmpdir_factory
) -> Tuple[pathlib.Path, Tuple[int, int]]:
    """
    Pytest fixture to generate global tweaks file for Sandia DAC.

    Global tweaks are "scalings"/gains that should be applied to each individual pin.

    Args:
        request: pytest fixture.
            Should include request.param as two ints:
            * number of tweaks
            * number of channels to tweak
        tmpdir_factory: pytest fixture

    Returns:
        Tuple: path to tweaks file, and the parameters
        (see request arg) used to construct it

    """
    glob_tweaks_content = gen_global_tweaks_contents(*request.param)
    glob_tweaks_file_path = tmpdir_factory.mktemp("data").join("test_global.txt")
    glob_tweaks_file_path.write(glob_tweaks_content)
    return glob_tweaks_file_path, request.param


def gen_mapping_file_contents(num_pads: int) -> str:
    r"""
    Generate a pin mapping file contents, given number of pads.

    Essentially mapping between channel names and which physical pin they correspond to.

    Args:
         num_pads: number of pads to output

    Returns:
        Contents of fake pin mapping file.
        Formatted like: ``PAD_0    0   0\nPAD_1  1   1``

    """
    retval = ""
    for i in range(num_pads):
        retval += "PAD_{0}\t{0}\t{0}\n".format(i)
    return retval


@pytest.fixture(scope="session", params=[3, 10, 100, 112, 200])
def dac_pin_map_file(request, tmpdir_factory) -> Tuple[pathlib.Path, int]:
    """
    Generate a temporary pin map file.

    Args:
        request: pytest fixture. Should give one int in request.param
        tmpdir_factory: pytest fixture

    Returns:
        Tuple: a path to the pin map file and the number of pins

    """
    pin_map_file_content = gen_mapping_file_contents(request.param)
    pin_map_file_path = tmpdir_factory.mktemp("data").join("test_mapping.txt")
    pin_map_file_path.write(pin_map_file_content)
    return pin_map_file_path, request.param


def gen_voltage_file_contents(
    num_lines: int, num_channels: int, reverse_pad_order: bool = False
) -> str:
    """
    Generate a fake voltage file.

    Args:
        num_channels: number of pads (i.e. channels) to output voltages for
        num_lines: number of lines to create
        reverse_pad_order: whether the pads should go from least-> greatest (
            left->right), or greatest->least

    Returns:
        Voltage file in format:
            * Pad headers (channel names)
            * Many lines of voltages, each line being the same but monotonically
                increasing

    """
    retval = ""
    for i in range(num_channels):
        if not reverse_pad_order:
            retval += "PAD_{}\t".format(i)
        else:
            retval += "PAD_{}\t".format(num_channels - i)
    retval += "\n"
    for i in range(num_lines):
        retval += (str(i) + "\t") * num_channels
        retval += "\n"
    return retval


@pytest.fixture(scope="session", params=[[2, 4], [3, 1], [1000, 112], [10000, 200]])
def dac_voltage_lines_file(
    request, tmpdir_factory
) -> Tuple[pathlib.Path, Tuple[int, int]]:
    """
    Generate. a temporary voltage line file.

    Corresponds to voltages that should be applied at different times.

    Args:
        request: pytest fixture. Should give two ints in request.param
            (number of lines and number of channels)
        tmpdir_factory: pytest fixture

    Returns:
        path to temporary voltage file and the parameters used to create it.

    """
    voltage_lines_file_content = gen_voltage_file_contents(*request.param)
    voltage_file_path = tmpdir_factory.mktemp("data").join("test_voltages.txt")
    voltage_file_path.write(voltage_lines_file_content)
    return voltage_file_path, request.param


def gen_shuttle_graph_file(num_edges: int) -> str:  # pylint: disable=unused-argument
    """Generate a file that denotes the shuttling graph."""
    # todo: automatically generate return value
    return """<?xml version="1.0" ?>
    <VoltageAdjust>
      <ShuttlingGraph currentPosition="0.0" currentPositionName="Loading">
        <ShuttleEdge _startType="" _stopType="" direction="0" idleCount="0.0"
            startLength="0" startLine="0.0" startName="Zero" steps="5.0"
            stopLength="0" stopLine="1" stopName="PadNum" wait="0"/>
        <ShuttleEdge _startType="" _stopType="" direction="0" idleCount="0.0"
            startLength="0" startLine="1.0" startName="PadNum" steps="5.0"
            stopLength="0" stopLine="2" stopName="Const" wait="0"/>
      </ShuttlingGraph>
    </VoltageAdjust>
    """


@pytest.fixture(scope="session", params=[100, 200, 300])
def dac_shuttle_graph_file(request, tmpdir_factory):
    """
    Generate a temporary shuttling graph file.

    The shuttling graph corresponds to paths that the ion(s) can take while
    shuttling between two nodes, literally the voltages that are used to move
    the ion(s) between locations.

    Args:
        request: pytest fixture.
        tmpdir_factory: pytest fixture

    Returns:
        (tuple): path to temporary shuttling graph file and the parameters
            used to create it.

    """
    shuttle_graph_content = gen_shuttle_graph_file(request.param)
    shuttle_file_path = tmpdir_factory.mktemp("data").join("test_graph.xml")
    shuttle_file_path.write(shuttle_graph_content)
    return shuttle_file_path, request.param


# ***** TESTS *****


class TestSandiaDACDriver:
    """Tests of the Sandia DAC driver by itself."""

    @staticmethod
    @pytest.mark.parametrize("name", [None, "test_name"])
    @pytest.mark.parametrize("bitfile_path", [None, "path_to_test_bitfile"])
    def test_sandia_dac_init(name, bitfile_path):
        """Tests if Sandia DAC initializes correctly."""
        if name is not None and bitfile_path is not None:
            test_device = driver.SandiaDACDriverLow(name, bitfile_path, simulation=True)
        elif name is None and bitfile_path is not None:
            test_device = driver.SandiaDACDriverLow(
                bitfile=bitfile_path, simulation=True
            )
        elif name is not None and bitfile_path is None:
            test_device = driver.SandiaDACDriverLow(fpga=name, simulation=True)
        else:
            test_device = driver.SandiaDACDriverLow(simulation=True)

        # Test parameters used correctly
        if name is not None:
            assert test_device.fpga_name == name
        else:
            assert test_device.fpga_name != name
        if bitfile_path is not None:
            assert test_device.fpga_bitfile_path == bitfile_path
        else:
            assert test_device.fpga_bitfile_path is None

        assert test_device.simulation is True
        assert test_device.dacController.isOpen() is True
        # ^^ can test on both simulation & real hardware. no way to distinguish here

        # test DAC properties/constant methods
        assert test_device.ping() is True
        assert test_device.channel_count() == 112

        # test that you cannot access channel_count like a property
        assert callable(test_device.channel_count)

        return

    wrapped_functions = {
        "shuttle": {"wrapped_calls": "shuttle", "args": [10]},
        "shuttle_along_path": {"wrapped_calls": "shuttlePath", "args": ["fakePath"]},
        "write_voltage_lines": {
            "wrapped_calls": ["writeVoltages", "verifyVoltages"],
            "args": [1, numpy.ones((10, 112)), 129346],
        },
        "write_shuttling_lookup_table": {
            "wrapped_calls": "writeShuttleLookup",
            "args": [["fake shuttle edges"], 10],
        },
        "trigger_shuttling": {"wrapped_calls": "triggerShuttling", "args": None},
    }

    close_function = {"close": {"wrapped_calls": "close", "args": None}}

    @pytest.mark.parametrize(
        "method,properties",
        [
            (method, properties)
            for method, properties in wrapped_functions.items()  # pylint: disable=E0602
        ],
    )
    def test_sandia_dac_wrapping(
        self, caplog, sandia_dac_driver: driver.SandiaDACDriverLow, method, properties
    ):
        """Check wrapped function calls in the simulated DAC work properly."""
        if sandia_dac_driver.simulation:
            caplog.set_level(logging.INFO)
            self._call_and_check_mock(
                sandia_dac_driver, "dacController", {method: properties}
            )
        else:
            pytest.skip("Cannot check function calls on non-simulated DAC")
        return

    @staticmethod
    def test_dac_settings_hash(caplog, sandia_dac_driver: driver.SandiaDACDriverLow):
        """Tests memory of DAC settings, and comparison against remote."""
        caplog.set_level(logging.INFO)
        fake_settings = (1, 2, 3)
        num_lines = 10
        bad_shape_data = numpy.array([[1, 2], [3, 4]])
        bad_data = numpy.arange(num_lines * sandia_dac_driver.channel_count()).reshape(
            num_lines, sandia_dac_driver.channel_count()
        )
        good_data = numpy.ones((num_lines, sandia_dac_driver.channel_count()))
        fake_hash = 1
        assert sandia_dac_driver.needs_new_data(fake_hash)
        assert sandia_dac_driver.needs_new_data(None)

        with pytest.raises(ValueError):
            sandia_dac_driver.write_voltage_lines(
                1, bad_shape_data, hash(fake_settings)
            )

        if not sandia_dac_driver.simulation:
            # unfortunately has dependency on Sandia OpalKelly code,
            # could try to catch Exception??
            from euriqabackend.devices.sandia_opalkelly.dac.DACController import (
                DACControllerException,
            )

            with pytest.raises(
                DACControllerException, message="Should raise DACControllerException"
            ):
                sandia_dac_driver.write_voltage_lines(1, bad_data, hash(fake_settings))

        sandia_dac_driver.write_voltage_lines(1, good_data, hash(fake_settings))
        assert sandia_dac_driver.settings_hash == hash(fake_settings)
        assert sandia_dac_driver.needs_new_data(hash(fake_settings)) is False
        assert sandia_dac_driver.needs_new_data(fake_hash) is True
        return

    @pytest.mark.parametrize(
        "close_fn_name,close_fn_properties",
        [
            (name, props)
            for name, props in close_function.items()  # pylint: disable=E0602
        ],
    )
    def test_sandia_dac_closing(
        self,
        caplog,
        sandia_dac_driver: driver.SandiaDACDriverLow,
        close_fn_name,
        close_fn_properties,
    ):
        """Check the DAC closes properly, MUST BE RUN LAST."""
        if sandia_dac_driver.simulation:
            caplog.set_level(logging.INFO)
            self._call_and_check_mock(
                sandia_dac_driver, "dacController", {close_fn_name: close_fn_properties}
            )
            with pytest.raises(AttributeError):
                # check that DAC was actually deleted
                assert sandia_dac_driver.dacController is not None
        else:
            pytest.skip("Cannot count function calls on non-simulated DAC.")

    @staticmethod
    @pytest.mark.manual
    def test_dac_manual():
        """Manually check the DAC."""
        # todo: force real DAC only, no sim
        pytest.fail("Test not finished")

    @staticmethod
    def _call_and_check_mock(
        device_under_test, mock_name: str, function_dict: Dict
    ) -> None:
        """
        Call every given method with a set of parameters.

        Check the mocked object has increased the number of calls to the wrapped
        method by 1.

        Args:
            device_under_test: whatever device is being tested
            mock_name: the instance property of device_under_test that is a
                Mock (standard library) object
            function_dict: Dictionary of function names
        """
        mocked_device = getattr(device_under_test, mock_name)
        for method, properties in function_dict.items():
            if isinstance(properties["wrapped_calls"], str):
                # if string, homogenize by converting to list
                properties["wrapped_calls"] = [properties["wrapped_calls"]]

            # Figure out number of calls the function should generate
            target_count_wrapped_function_calls = [
                getattr(mocked_device, fn).call_count + 1
                for fn in properties["wrapped_calls"]
            ]

            # Calls the functions
            # todo: change to use inspect and generate arguments automatically.
            # todo: check that wrapped methods use the given parameters SOMEHOW
            if properties["args"] is not None:
                getattr(device_under_test, method)(*properties["args"])
            else:
                getattr(device_under_test, method)()

            # Check that all wrapped functions have been called one more time
            # than before
            assert [
                getattr(mocked_device, fn).call_count
                for fn in properties["wrapped_calls"]
            ] == target_count_wrapped_function_calls
        return


class TestSandiaDACInterface:
    """Pytest Class to check the main shuttling interface code."""

    # Instantiation Tests
    @staticmethod
    def test_interface_init(sandia_dac_interface: interface.SandiaDACInterface):
        """Check started interface properly.

        Assumes no state loaded.
        """
        # Check started driver properly
        assert isinstance(sandia_dac_interface.hardware, driver.SandiaDACDriverLow)
        assert sandia_dac_interface.hardware.ping()

        # Check vars initialized to None
        for attr in [
            "pin_map_file_path",
            "_pin_names",
            "_analog_output_channel_numbers",
            "_dsub_pin_numbers",
            "uploaded_data_hash",
            "shuttling_definition_file_path",
            "shuttling_graph",
        ]:
            assert getattr(sandia_dac_interface, attr) is None

        # Check it gets a core device
        assert isinstance(sandia_dac_interface.core, core.Core)

        # Check lists
        for attr in ["lines", "table_header"]:
            assert isinstance(getattr(sandia_dac_interface, attr), list)
            assert not getattr(
                sandia_dac_interface, attr
            )  # empty lists return as False

        # Check numpy arrays
        for attr in ["line_adjustment_values"]:
            assert isinstance(getattr(sandia_dac_interface, attr), numpy.ndarray)
            assert numpy.all(getattr(sandia_dac_interface, attr) == pytest.approx(0))

        # Check Dicts
        for attr in ["adjustment_dictionary"]:
            assert isinstance(getattr(sandia_dac_interface, attr), dict)
            assert not getattr(sandia_dac_interface, attr)

        # Check state variables
        for attr in ["can_shuttle"]:
            assert isinstance(getattr(sandia_dac_interface, attr), bool)
            assert getattr(sandia_dac_interface) is False

        # Check numbers
        for var in ["line_number"]:
            assert getattr(sandia_dac_interface, var) == pytest.approx(0)

        for var in ["line_gain", "adjustment_gain", "global_gain"]:
            assert getattr(sandia_dac_interface, var) == pytest.approx(1.0)

    @staticmethod
    def _check_line_valid(line: numpy.ndarray, line_number: int, valid_cols: int):
        for v in line[:valid_cols]:
            assert line_number == v
        for v in line[valid_cols:]:
            assert v == 0.0
        return

    # Functional Tests
    @staticmethod
    def test_interface_state_load_and_save(
        sandia_dac_interface: interface.SandiaDACInterface
    ):
        """
        Tests saving & loading interface state.

        Flow:
            * Save current state
            * Set new random state
            * Load original state

        Args:
            sandia_dac_interface: pytest fixture generating a sandia_dac interface
        """
        # pylint: disable=protected-access
        saved_state = sandia_dac_interface.save_state()
        for key, value in saved_state.items():
            assert key in sandia_dac_interface._state_variables
            if isinstance(getattr(sandia_dac_interface, key), numpy.ndarray):
                assert numpy.array_equal(getattr(sandia_dac_interface, key), value)
            elif isinstance(getattr(sandia_dac_interface, key), pandas.Index):
                assert value.equals(getattr(sandia_dac_interface, key))
            else:
                assert getattr(sandia_dac_interface, key) == value

        for var in sandia_dac_interface._state_variables:
            random_val = random.random()
            setattr(sandia_dac_interface, var, random_val)
            assert getattr(sandia_dac_interface, var) == random_val

        new_saved_state = sandia_dac_interface.save_state()
        assert new_saved_state != saved_state

        sandia_dac_interface.load_saved_state(saved_state)
        for var in sandia_dac_interface._state_variables:
            if isinstance(getattr(sandia_dac_interface, var), numpy.ndarray):
                assert numpy.array_equal(
                    getattr(sandia_dac_interface, var), saved_state[var]
                )
            elif isinstance(getattr(sandia_dac_interface, var), pandas.Index):
                assert saved_state[var].equals(getattr(sandia_dac_interface, var))
            else:
                assert getattr(sandia_dac_interface, var) == saved_state[var]

        # todo: check other members (not in _state_variables),
        #   make sure they don't change???
        return

    @staticmethod
    def test_interface_pin_mapping(
        dac_pin_map_file, sandia_dac_interface: interface.SandiaDACInterface
    ):
        """Test that pin mapping is loaded correctly."""
        # pylint: disable=protected-access
        path, pads = dac_pin_map_file
        if pads > sandia_dac_interface.hardware.channel_count():
            with pytest.raises(ValueError):
                sandia_dac_interface.load_pin_map_file(path.strpath)
            return
        sandia_dac_interface.load_pin_map_file(path.strpath)

        pad_list = ["PAD_{}".format(i) for i in range(pads)]
        ao_list = list(range(pads))
        dsub_list = [int(i) for i in range(pads)]
        for i, pad in enumerate(sandia_dac_interface._pin_names):
            assert pad == pad_list[i]

        for i, ao in enumerate(sandia_dac_interface._analog_output_channel_numbers):
            assert ao == ao_list[i]

        for i, dsub in enumerate(sandia_dac_interface._dsub_pin_numbers):
            assert dsub == dsub_list[i]

    @staticmethod
    def test_interface_global_tweak_set(
        dac_global_tweaks_file,
        dac_pin_map_file,
        sandia_dac_interface: interface.SandiaDACInterface,
    ):
        """Checks the global tweaks are read correctly from a file."""
        path, (lines, pads) = dac_global_tweaks_file
        if dac_pin_map_file[1] > sandia_dac_interface.hardware.channel_count():
            with pytest.raises(ValueError):
                sandia_dac_interface.load_pin_map_file(dac_pin_map_file[0].strpath)
            return
        sandia_dac_interface.load_pin_map_file(dac_pin_map_file[0].strpath)
        sandia_dac_interface.load_global_adjust_file(path.strpath)

        assert len(sandia_dac_interface.adjustment_dictionary) == lines
        assert len(sandia_dac_interface.line_adjustment_values) == lines

        valid_channels = min(pads, dac_pin_map_file[1])

        for i in range(lines):
            line_name = "Line{}".format(i)
            assert (
                sandia_dac_interface.adjustment_dictionary[line_name]["name"]
                == line_name
            )
            assert sandia_dac_interface.adjustment_dictionary[line_name]["line"] == i
            assert sandia_dac_interface.adjustment_dictionary[line_name][
                "adjustment_gain"
            ] == pytest.approx(0)
            TestSandiaDACInterface._check_line_valid(
                sandia_dac_interface.line_adjustment_values[i], i, valid_channels
            )
            # normally do lookup by adjustment_dictionary, but here it's equiv to i
            assert (
                len(sandia_dac_interface.line_adjustment_values[i])
                == sandia_dac_interface.hardware.channel_count()
            )

    @staticmethod
    def test_interface_voltages_load(
        dac_voltage_lines_file,
        dac_pin_map_file,
        sandia_dac_interface: interface.SandiaDACInterface,
    ):
        """Checks the global tweaks are read correctly from a file."""
        v_path, (v_lines, voltage_file_channels) = dac_voltage_lines_file
        pin_path, pin_file_channels = dac_pin_map_file
        # assert voltage_file_channels == pin_file_channels
        chans = min([voltage_file_channels, pin_file_channels])
        if pin_file_channels > sandia_dac_interface.hardware.channel_count():
            with pytest.raises(ValueError):
                sandia_dac_interface.load_pin_map_file((pin_path.strpath))
            return
        sandia_dac_interface.load_pin_map_file(pin_path.strpath)
        # if voltage_file_channels > sandia_dac_interface.hardware.channel_count():
        #     with pytest.raises(AttributeError):
        #         sandia_dac_interface.load_voltage_file(v_path)
        #     return

        sandia_dac_interface.load_voltage_file(v_path)

        num_chans = sandia_dac_interface.hardware.channel_count()

        assert len(sandia_dac_interface.lines) == v_lines

        for i, line in enumerate(sandia_dac_interface.lines):
            assert len(line) == num_chans
            TestSandiaDACInterface._check_line_valid(line, i, chans)

        voltage_df = pandas.read_csv(v_path, delim_whitespace=True, header=0)
        assert (
            sandia_dac_interface.table_header.tolist()
            <= voltage_df.columns.values.tolist()
        )
        assert (
            len(sandia_dac_interface.table_header)
            <= sandia_dac_interface.hardware.channel_count()
        )
        return

    @staticmethod
    def test_dac_control(
        sandia_dac_interface: interface.SandiaDACInterface,
        dac_pin_map_file,
        dac_global_tweaks_file,
        dac_voltage_lines_file,
        dac_shuttle_graph_file,
    ):
        """Functional test to make sure that can control voltages to FPGA."""
        # pylint: disable=protected-access
        sandia_dac_interface.hardware.settings_hash = (
            None
        )  # hack to invalidate settings on driver, rather than
        # re-instantiate driver every time

        voltage_file_lines, voltage_file_channels = dac_voltage_lines_file[1]
        max_shuttle_line_used = (
            2
        )  # todo: replace with logic from dac_shuttle_graph_file
        pin_file_path, pin_pads_num = dac_pin_map_file
        if pin_pads_num > sandia_dac_interface.hardware.channel_count():
            with pytest.raises(ValueError):
                sandia_dac_interface.load_pin_map_file(pin_file_path.strpath)
            return
        sandia_dac_interface.load_pin_map_file(dac_pin_map_file[0].strpath)
        sandia_dac_interface.load_global_adjust_file(dac_global_tweaks_file[0].strpath)

        if voltage_file_channels > sandia_dac_interface.hardware.channel_count():
            # test greater number of channels in file than on hardware
            num_valid_columns = len(
                [
                    col
                    for col in pandas.read_csv(
                        dac_voltage_lines_file[0].strpath
                    ).columns.values
                    if col in sandia_dac_interface._pin_names
                ]
            )
            if num_valid_columns > sandia_dac_interface.hardware.channel_count():
                with pytest.raises(RuntimeError):
                    sandia_dac_interface.load_voltage_file(
                        dac_voltage_lines_file[0].strpath
                    )
                return

        sandia_dac_interface.load_voltage_file(dac_voltage_lines_file[0].strpath)
        sandia_dac_interface.load_shuttling_definitions_file(
            dac_shuttle_graph_file[0].strpath
        )
        assert sandia_dac_interface.hardware.needs_new_data(
            sandia_dac_interface._shuttling_data_hash()
        )

        # send data to DAC FPGA
        if max_shuttle_line_used >= voltage_file_lines:
            with pytest.raises(IndexError):
                # get IndexError when calculating lines.
                # Reference line that doesn't exist in too-small voltage file
                sandia_dac_interface.send_voltage_lines_to_fpga()
            return
        sandia_dac_interface.send_voltage_lines_to_fpga()
        sandia_dac_interface.set_shuttling_lookup_table()

        # test sent data properly
        assert (
            sandia_dac_interface.hardware.needs_new_data(
                sandia_dac_interface._shuttling_data_hash()
            )
            is False
        )

        return

    @staticmethod
    def test_sandia_dac_interface_functions(
        # sandia_dac_interface: interface.SandiaDACInterface
    ):
        """
        Test Mediator functionality.

        Args:
            sandia_dac_interface:
        """
        pytest.fail("Not finished writing test")
        # todo: fill in
        # funcs to test: setAdjust, applyLine, calculateLine, shuttle,
        #   adjustLine, blendLines, set_shuttling_lookup_table,
        #   load_shuttling_definitions_file

        # todo: test alignment of pin outputs
        #   (i.e. channel 1 will always output to correct pin)
        # Needs to be done on hardware

    @staticmethod
    def test_sandia_dac_interface_set_adjustments(
        sandia_dac_interface: interface.SandiaDACInterface,
        dac_pin_map_file,
        dac_global_tweaks_file,
        dac_voltage_lines_file,
    ):
        """Test :meth:`~interface.SandiaDACInterface.set_adjustments`."""
        # Set to some random output, and get the output voltages
        if sandia_dac_interface.hardware.simulation:
            pytest.skip("Cannot read voltages back from simulated hardware.")
        if dac_pin_map_file[1] > sandia_dac_interface.hardware.channel_count():
            with pytest.raises(ValueError):
                sandia_dac_interface.load_pin_map_file(dac_pin_map_file[0].strpath)
            return
        sandia_dac_interface.load_pin_map_file(dac_pin_map_file[0].strpath)
        sandia_dac_interface.load_voltage_file(dac_voltage_lines_file[0].strpath)
        sandia_dac_interface.load_global_adjust_file(dac_global_tweaks_file[0].strpath)

        test_line_number = 0.6
        sandia_dac_interface.apply_line_async(test_line_number, 1.0, 1.0)
        initial_voltage = sandia_dac_interface.hardware.dacController.readVoltage(0)

        # Change output voltage, and check that the DAC has changed.
        sandia_dac_interface.set_adjustments(
            {"test_adjust_1": {"line": 1, "adjustment_gain": random.random()}},
            adjustment_gain=1.0,
        )
        assert not numpy.array_equal(
            sandia_dac_interface.hardware.dacController.readVoltage(0), initial_voltage
        )

    @staticmethod
    def test_sandia_dac_interface_apply_line(
        # sandia_dac_interface: interface.SandiaDACInterface
    ):
        """Test :meth:`~interface.SandiaDACInterface.apply_line`."""
        pytest.fail("Test not finished")

    @staticmethod
    def test_sandia_dac_interface_calculate_line(
        # sandia_dac_interface: interface.SandiaDACInterface
    ):
        """Test :meth:`~interface.SandiaDACInterface._calculate_line`."""
        pytest.fail("Test not finished")

    @staticmethod
    def test_sandia_dac_interface_adjust_line(
        # sandia_dac_interface: interface.SandiaDACInterface
    ):
        """Test :meth:`~interface.SandiaDACInterface._adjust_line`."""
        pytest.fail("Test not finished")

    @staticmethod
    def test_sandia_dac_interface_blend_lines(
        # sandia_dac_interface: interface.SandiaDACInterface
    ):
        """Test :meth:`~interface.SandiaDACInterface._blend_lines`."""
        pytest.fail("Test not finished")

    @staticmethod
    @pytest.mark.manual
    def test_sandia_dac_interface_voltage_on_oscilloscope(
        sandia_dac_interface: interface.SandiaDACInterface,
        dac_pin_map_file,
        dac_global_tweaks_file,
        dac_voltage_lines_file,
    ):
        """Check that the interface is properly functioning on an oscilloscope."""
        if dac_voltage_lines_file[1][0] <= 2:
            pytest.skip("Not enough voltage lines")
        if dac_pin_map_file[1] > sandia_dac_interface.hardware.channel_count():
            with pytest.raises(ValueError):
                sandia_dac_interface.load_pin_map_file(dac_pin_map_file[0].strpath)
            return

        sandia_dac_interface.load_pin_map_file(dac_pin_map_file[0].strpath)
        sandia_dac_interface.load_global_adjust_file(dac_global_tweaks_file[0].strpath)
        sandia_dac_interface.load_voltage_file(dac_voltage_lines_file[0].strpath)

        for _ in range(10):
            sandia_dac_interface.apply_line_async(
                1, line_gain=1.0, global_gain=5.0, update_hardware=True
            )
            time.sleep(0.5)
            sandia_dac_interface.apply_line_async(
                1, line_gain=1.0, global_gain=1.0, update_hardware=True
            )
            time.sleep(0.2)
            sandia_dac_interface.apply_line_async(
                1, line_gain=0.75, global_gain=2.0, update_hardware=True
            )
            time.sleep(0.2)

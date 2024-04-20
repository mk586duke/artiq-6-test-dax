# Copyright 2020 Drew Risinger, Chris Monroe Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
High-level interface to the Sandia x100 DAC for shuttling ions.

This software is designed to run the "Sandia" 100x DAC.

Mimics :mod:`pdq2.mediator` for synchronous logic/real-time control
Based on :mod:`artiq.devices.pdq2.mediator` and the Sandia pyIonControl software.

Directly uses Peter Maunz's pulser/DACController and OpalKelly wrappers
pulser/OKBase, pulser/bitfileHeader), and adapts the initialization sequence
from ExperimentUi to implement FPGAHardware class in
voltageControl/VoltageBlender for the ARTIQ platform.

Designed to be run on Host PC, i.e. the one running artiq_master to provide
interface to Device Database

Revision History:
    9/14/18:
        Author: Drew Risinger.
        **BROKE API**
        Add set_global_gain() function.
        Reorganize to clarify logical function groupings.
        Rename functions & internal variables to obey snake_case and clarify usage.

    7/12/18:
        Author: Drew Risinger
        Add type hints and docstrings. Not 100% sure correct, but it's a start.
        Remove shell that should run this, b/c not meant to run as script.

Note:
    EMIT means original Sandia code called some FPGA function, needs tob e back-filled.

    The asynchronous commands use the SandiaDACDriverLower class from driver.py,
    which adapts Peter Maunz's controller for the ARTIQ platform.
"""
import contextlib
import logging
import math
import pathlib
import pprint
import typing
import warnings
import xml.etree.ElementTree as xml_tree
from itertools import permutations
import os

import artiq.master.worker_db as mgr
import numpy
import pandas
import sipyco.packed_exceptions as aqexc
from artiq.language.types import TInt32
from artiq.language.types import TNone
from PyQt5 import QtCore

import euriqabackend.devices.sandia_opalkelly.dac.ShuttlingDefinition as shuttle_defs
import euriqabackend.utilities.hashing as special_hash


class RPCWarning(RuntimeWarning):
    """Warning to denote that an RPC failed."""

    pass


def pc_rpc_warning_filter(message: str) -> typing.Tuple[bool, Warning, str]:
    """Filter log messages to determine if should raise an RPCWarning.

    Returns:
        (bool, str): Whether filter matches and what message to raise warning with.

    """
    pc_rpc_warning_keywords = ["rpc", "connection", "to", "failed", "re", "background"]
    if all(keyword in message.lower() for keyword in pc_rpc_warning_keywords):
        return True, RPCWarning, "ARTIQ PC RPC Warning: RPC to device failed"
    else:
        return False, None, ""


_WARNING_FILTERS = [pc_rpc_warning_filter]


# HACK: patch logging.warning() to use warnings module
# Better: modify ARTIQ to use warnings module instead of logging.warning()
def log_warning_to_warning_module(self: logging.Logger, message: str, *args, **kws):
    """Hack logger to generate RPC warnings."""
    if self.isEnabledFor(logging.WARNING):
        self._log(logging.WARNING, message, args, **kws)  # pylint: disable=W0212

        for w_filter in _WARNING_FILTERS:
            filter_true, warning_type, warning_text = w_filter(message)
            if filter_true:
                warnings.warn(warning_text, warning_type, stacklevel=2)


# patch this to overwrite warning (might need to be before ARTIQ imports?)
logging.Logger.warning = log_warning_to_warning_module

_LOGGER = logging.getLogger(__name__)
logging.captureWarnings(True)


@contextlib.contextmanager
def warnings_raise_exceptions(category: BaseException = Warning):
    """Warnings raise exceptions in this context.

    Can optionally specify the category of warnings to catch.

    Example:
    >>> with warnings_raise_exceptions(Warning):
    ...     try:
    ...         warnings.warn()
    ...     except Warning:
    ...         print("Caught Warning")
    "Caught Warning"
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=category)
        yield


class SandiaDACInterface(QtCore.QObject):
    """
    Command a remote Sandia DAC via the `artiq_master` PC..

    Interfaces with a remote PC connected to a Sandia DAC
    (which is running the :mod:`artiq.frontend.aqctl_sandia_dac_100x` program)
    """

    _numpy_print_options = {"suppress": True, "precision": 3}

    # Think these are the indices of the table to change.
    voltages_changed = QtCore.pyqtSignal(
        int, int, int, int
    )  # notifier that output voltages changed.
    voltage_errors = QtCore.pyqtSignal(list)

    def __init__(
        self,
        device_manager: mgr.DeviceManager,
        dac_device: str,
        saved_state: typing.Dict = None,
    ):
        """
        Connect to DAC and start internal state.

        Args:
            device_manager (Device Manager/Database (
                :class:`~artiq.master.worker_db.DeviceManager`)): for retrieving
                accessible instruments/network devices
            dac_device (str): Name of the DAC to connect to (from the device_manager)
            saved_state (Dict): A saved state of the mediator to load from
        """
        # , trigger_in_device, trigger_out_device, serial_device):
        super().__init__()
        self.hardware = device_manager.get(dac_device)

        # digital lines for synchronous control of shuttling
        # self.triggerOut = device_manager.get(trigger_in_device)
        # self.serial = device_manager.get(serial_device)
        # self.triggerIn = device_manager.get(trigger_out_device)
        if not self.hardware.ping():
            raise ConnectionError(
                "Could not connect to remote DAC hardware. Make sure remote "
                "driver is started, and the network is configured properly."
            )

        self.lines = list()  # a list of lines with numpy arrays
        self.adjustment_dictionary = dict()  # names of the possible adjustments
        self.tweak_dictionary = dict()
        self.line_adjustment_values = numpy.zeros((1, 1))
        self.line_gain = 1.0
        self.global_gain = 1.0
        self.adjustment_gain = 1.0
        self.line_number = 0
        self.pin_map_file_path = None
        self._pin_names = None
        self._analog_output_channel_numbers = None
        self._dsub_pin_numbers = None
        self._pin_map_df = None
        self.table_header = list()
        self.uploaded_data_hash = None
        self.file_line_num_to_address_dict = dict()  # acts as flag for vals on FPGA
        self.shuttling_definition_file_path = None
        self.shuttling_graph = shuttle_defs.ShuttlingGraph()
        self.can_shuttle = False  # Unable to shuttle until table written
        if saved_state is not None:
            self.load_saved_state(saved_state)
        self.channels = self.hardware.channel_count()
        self.num_line = None

    _state_variables = [
        "lines",
        "adjustment_dictionary",
        "line_adjustment_values",
        "line_gain",
        "global_gain",
        "adjustment_gain",
        "line_number",
        "pin_map_file_path",
        "_pin_names",
        "_analog_output_channel_numbers",
        "_dsub_pin_numbers",
        "table_header",
        "uploaded_data_hash",
        "shuttling_definition_file_path",
        "shuttling_graph",
        "can_shuttle",
    ]

    # *** CHANGE/SAVE MEDIATOR STATE ***
    def load_saved_state(self, saved_state: typing.Dict[str, typing.Any]) -> TNone:
        """
        Reload state from a given saved state.

        Args:
            saved_state(Dict): Dictionary containing {variable,value} pairs

        Raises:
            KeyError if given key in saved_state is not a valid variable

        """
        assert saved_state is not None
        for key in saved_state:
            if key in self._state_variables:
                setattr(self, key, saved_state[key])
            else:
                raise KeyError(
                    "Saved state key {} is not a valid variable in {}".format(
                        key, self.__class__
                    )
                )

    # todo: save DAC name so you can reconnect to the same later?
    # todo: factory to create Interface & connection from saved_state
    def save_state(self) -> typing.Dict[str, typing.Any]:
        """
        Save the state of the class instance into a dictionary.

        Returns:
             Dictionary of (key, value) pairs which correspond to this
             instance's variables.

        """
        # todo: replace with pickling (python pickle library), maybe???
        state_variables = dict()
        for field in self._state_variables:
            state_variables.update({field: getattr(self, field)})
        return state_variables

    def current_data(self):
        """Return the current data stored about the DAC.

        Not really sure what this is for. Not used, legacy code.
        Returned data doesn't seem terribly useful.
        """
        return (
            self._pin_names,
            self._analog_output_channel_numbers,
            self._dsub_pin_numbers,
            self.current_output_voltage,
        )

    # *** INITIALIZE DAC/MEDIATOR ***
    def load_pin_map_file(
        self, pin_map_file: typing.Union[str, pathlib.Path, typing.TextIO]
    ) -> TNone:
        """
        Load a pin mapping file.

        A mapping file maps "pin names" to the actual pin that will output
        the desired voltage.
        i.e. if you set "Pad_10" to 1.0 V, the mapping file will translate
        that to the correct pad.

        Args:
            pin_map_file(str): path where the pin mapping file is on computer

        Raises:
            ValueError: If pass a file with more pins than are in hardware.

        """
        self.pin_map_file_path = pin_map_file
        pin_map_headers = [
            # "Pin Name",   # used as index
            "Analog Output Number",
            "DSub Connector Pin Number",
        ]
        self._pin_map_df = pandas.read_csv(
            pin_map_file,
            delim_whitespace=True,
            header=None,
            names=pin_map_headers,
            index_col=0,
        )
        if len(self._pin_map_df.index) > self.channels:
            raise ValueError("Number of pins is greater than hardware supports.")
        self._pin_names = self._pin_map_df.index.values
        self._analog_output_channel_numbers, self._dsub_pin_numbers = (
            self._pin_map_df[col_name] for col_name in self._pin_map_df
        )

        self.can_shuttle = False
        # EMIT
        return

    def load_voltage_file(
        self, voltage_file: typing.Union[str, pathlib.Path, typing.TextIO]
    ) -> TNone:
        """
        Load a particular voltage file of output voltages.

        This voltage file contains many "lines" which each correspond
        to a set of output voltages to be output simultaneously.

        Args:
            voltage_file(str): file voltage_file to the voltage file
        """
        if not self.pin_map_file_path or self._pin_names is None:
            raise RuntimeError(
                "Must load mapping (load_pin_map_file()) before loading voltages"
            )

        voltage_file_df = pandas.read_csv(voltage_file, delim_whitespace=True, header=0)
        voltage_file_df = self._sanitize_dataframe(voltage_file_df)
        with numpy.printoptions(**self._numpy_print_options):
            _LOGGER.debug("Read voltages from file: \n%s", voltage_file_df.values)
        voltage_file_df = self._sort_dataframe_by_pin(voltage_file_df)
        self.table_header = voltage_file_df.columns.values
        self.lines = self._pad_array_zeros(voltage_file_df.values, self.channels)
        self.num_line = numpy.shape(self.lines)[0]
        self.can_shuttle = False
        self.uploaded_data_hash = None
        return

    def _sort_dataframe_by_pin(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """Return sorted (not in-place) dataframe by corresponding analog_output."""
        if any((name not in self._pin_names for name in dataframe.columns)):
            _LOGGER.error(
                "Invalid pin name. %s is not in %s",
                dataframe.columns.values,
                self._pin_names,
            )
            raise ValueError(
                "Invalid name pin name given. {} is not in {}".format(
                    dataframe.columns.values, self._pin_names
                )
            )
        if self._pin_map_df is None:
            raise RuntimeError("Should run load_pin_map_file() first")

        sorted_columns = dataframe.columns[
            dataframe.columns.map(self._analog_output_channel_numbers)
        ]
        return dataframe.reindex(sorted_columns, axis="columns")

    def load_voltage_dict(
        self, voltage_lines: typing.Dict[str, numpy.ndarray]
    ) -> TNone:
        """
        Load voltages from a dictionary instead of a file.

        Args:
            voltage_lines (typing.Dict[str, numpy.ndarray]):
                A mapping between electrode names, and pin numbers
                (analog out/dsub connector)

        Raises:
            ValueError: keys must be an electrode, and all vectors must be the
                same size.

        Returns:
            None.

        """
        if any((arr.ndim != 1 for arr in voltage_lines.values())):
            raise ValueError("Provided voltages are not one-dimensional arrays.")

        if any(
            (a.shape != b.shape for a, b in permutations(voltage_lines.values(), 2))
        ):
            raise ValueError("Provided voltage arrays are not all the same length.")

        if any((name not in self._pin_names for name in voltage_lines.keys())):
            raise ValueError("Provided incorrect voltage keys.")

        voltage_dataframe = pandas.DataFrame.from_dict(voltage_lines)
        voltage_dataframe = self._sort_dataframe_by_pin(voltage_dataframe)

        self.lines = self._pad_array_zeros(voltage_dataframe, self.channels)
        self.uploaded_data_hash = None

    def _sanitize_dataframe(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Remove invalid columns not present in `self._pin_names`.

        Does not modify in-place.
        """
        df = df.fillna(0)
        invalid_columns = [col for col in df.columns if col not in self._pin_names]
        if any(invalid_columns):
            sanitized_df = df.drop(invalid_columns, axis="columns")
            _LOGGER.debug(
                "Sanitized %s columns from df: \n%s", invalid_columns, sanitized_df
            )
            return sanitized_df
        else:
            return df

    @staticmethod
    def _pad_array_zeros(arr: numpy.ndarray, num_columns: int) -> numpy.ndarray:
        """
        Pad values with 0's to reach desired number of columns/channels.

        Does not operate in-place

        Args:
            arr: array to be padded
            num_columns: desired number of columns

        Return:
            Array with num_columns columns, and the original number of rows.

        Raises:
            RuntimeError if number of original columns is higher than
            number of desired columns

        """
        if arr.shape[1] > num_columns:
            raise RuntimeError(
                "Can't shrink the array. Num_columns ({}) < "
                "array columns ({})".format(num_columns, arr.shape[1])
            )
        temp = numpy.zeros((arr.shape[0], num_columns))
        temp[:, : arr.shape[1]] = arr
        return temp

    def load_global_adjust_file(
        self, global_adjustments_file: typing.Union[str, pathlib.Path, typing.TextIO]
    ) -> TNone:
        """
        Load a file with global adjustments to the voltage outputs.

        Global adjustments/tweaks can be switched at runtime.

        Args:
             global_adjustments_file(str): file global_adjustments_file to file with
             the global adjustments
        """
        if not self.pin_map_file_path or self._pin_names is None:
            raise RuntimeError(
                "Must run load_pin_map_file() before loading global adjustments"
            )
        channelCount = self.channels
        self.line_adjustment_values = numpy.zeros(self.line_adjustment_values.shape)
        self.adjustment_dictionary = dict()  # SequenceDict()

        self.can_shuttle = False
        with open(global_adjustments_file, "r") as file:
            skiplines = 0
            line = file.readline()
            while "=" in line:
                skiplines += 1
                name, value = line.split("=")
                self.adjustment_dictionary[name] = {
                    # "name": name,
                    "line": int(value),
                    "adjustment_gain": 0,
                }
                line = file.readline()

        global_adjust_df = pandas.read_csv(
            global_adjustments_file,
            delim_whitespace=True,
            header="infer",
            skiprows=skiplines,
        )
        global_adjust_df = self._sanitize_dataframe(global_adjust_df)
        self.line_adjustment_values = self._pad_array_zeros(
            global_adjust_df.values, channelCount
        )

        # sanitize invalid lines from self.adjustment_dictionary
        for k in self.adjustment_dictionary.keys():
            if self.adjustment_dictionary[k]["line"] > len(global_adjust_df):
                del self.adjustment_dictionary[k]
        # EMIT
        return

    def load_shuttling_definitions_file(
        self, filename: typing.Union[str, typing.IO]
    ) -> TNone:
        """
        Load a shuttling definition file defining shuttling graph/paths.

        A shuttling graph defines the paths that you can take to transition
        between one steady voltage state and another.

        Args:
             filename(str): Path to the file containing the shuttling path edges.
                Should be an XML file
        """
        self.shuttling_graph = self.get_shuttling_graph_from_xml(filename)
        self.can_shuttle = False

    def get_shuttling_graph_from_xml(
        self, filename: typing.Union[str, typing.IO]
    ):
        """
        Generate a shuttling definition file defining shuttling graph/paths.

        A shuttling graph defines the paths that you can take to transition
        between one steady voltage state and another.

        Args:
             filename(str): Path to the file containing the shuttling path edges.
                Should be an XML file
        """
        self.shuttling_definition_file_path = filename
        tree = xml_tree.parse(filename)
        root = tree.getroot()
        shuttling_graph_xml_element = root.find("ShuttlingGraph")
        return shuttle_defs.ShuttlingGraph.from_xml_element(
            shuttling_graph_xml_element
        )

    def send_shuttling_lookup_table(
        self, shuttling_graph: shuttle_defs.ShuttlingGraph = None
    ) -> TNone:
        """
        Write the shuttling graph to the DAC.

        Calls writeShuttleLookup on FPGA. See
        :meth:`euriqabackend.devices.sandia_dac.driver.SandiaDACDriverLow.
        writeShuttleLookup` for more info.

        Args:
            shuttling_graph (ShuttlingGraph): A sequence of edges, linked together into
                a graph. These denote parts of the shuttling solution that can
                be joined together to move ions/ion chains around a chip trap.
        """
        if shuttling_graph is None and self.shuttling_graph is not None:
            shuttling_graph = self.shuttling_graph
        else:
            raise RuntimeError("Must generate ShuttlingGraph first")

        # convert shuttling_graph to a format that PYON can encode to send over network
        shuttle_lookup = tuple(
            tuple(entry)
            for entry in shuttle_defs.shuttle_edges_to_fpga_entry(
                shuttling_graph, self.channels
            )
        )
        _LOGGER.debug("Shuttle lookup table for FPGA: %s", shuttle_lookup)

        try:
            self.hardware.write_shuttling_lookup_table(shuttle_lookup, 0)
            self.can_shuttle = True
        except Exception as e:
            raise aqexc.GenericRemoteException(
                "Remote DAC is not accessible or open."
            ) from e

    def send_voltage_lines_to_fpga(
        self, shuttling_graph: shuttle_defs.ShuttlingGraph = None, force_full_update:bool = True
    ) -> TNone:
        """
        Write voltage output data onto the FPGA.

        Data is sent as voltages, and is in the order that the shuttling graph
        expects.

        Args:
            shuttling_graph:

        Raises:
            RuntimeError if it doesn't recognize any valid ShuttlingGraphs

        """
        voltage_outputs_to_write = list()
        if shuttling_graph is None and self.shuttling_graph is not None:
            shuttling_graph = self.shuttling_graph
            if shuttling_graph is None:
                raise RuntimeError(
                    "No shuttling graph provided. "
                    "Either must give argument to method, or call "
                    "load_shuttling_definition_file first"
                )

        # prep data for writing to FPGA,
        self.file_line_num_to_address_dict.clear()
        current_line = start_line = 1

        # Edge iterates based on the order in the .xml file
        for edge in shuttling_graph:
            # add each calculated line to pending towrite object
            line_number_and_value = [
                (
                    line_num,
                    index,
                    self._calculate_line(line_num, self.line_gain, self.global_gain),
                )
                for index, line_num in enumerate(edge.line_number_iterator())
            ]
            line_number_to_address_dict = {
                line_num: (index + current_line)
                for line_num, index, _ in line_number_and_value
            }
            voltage_outputs_to_write.extend(
                [voltages for _, _, voltages in line_number_and_value]
            )
            edge.memory_start_address = current_line
            edge.memory_stop_address = current_line = start_line + len(
                voltage_outputs_to_write
            )

            # # For checking tweak of dumped voltages
            # print("write", len(voltage_outputs_to_write))
            # for k in range(len(line_number_and_value)):
            #     # if line_number_and_value[k][0] == 6964:
            #     if line_number_and_value[k][0] == 4863:
            #         print("11-0 index: ", k + len(voltage_outputs_to_write))
            # print(len(line_number_and_value))

            self.file_line_num_to_address_dict.update(line_number_to_address_dict)

        # prep values to send to FPGA
        self.uploaded_data_hash = self._shuttling_data_hash()
        write_np_array = numpy.array(voltage_outputs_to_write)

        # debug/value checking
        with numpy.printoptions(**self._numpy_print_options):
            _LOGGER.debug(
                "Writing (lines,length)=%s data to FPGA:\n%s",
                write_np_array.shape,
                write_np_array,
            )
            if _LOGGER.isEnabledFor(logging.DEBUG):
                # guard against long pprint time
                line_print_interval = 5  # interval of consecutive lines to print
                line_num_lookup = {
                    line: address
                    for line, address in self.file_line_num_to_address_dict.items()
                    if (line / line_print_interval).is_integer()
                }
                _LOGGER.debug(
                    "Line # -> Address dictionary:\n%s", pprint.pformat(line_num_lookup)
                )
            if numpy.isnan(write_np_array).any():
                _LOGGER.error(
                    "Found NaN value(s) in values being sent to DAC. "
                    "Check your global adjustments."
                )
                _LOGGER.debug(
                    "NaN value indices: %s", numpy.argwhere(numpy.isnan(write_np_array))
                )

        # To test if swap waveforms are applied
        #numpy.savetxt("/media/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/2023-12-22/write_np_array.txt", write_np_array)
        # with open("/home/euriqa/git/euriqa-artiq/euriqabackend/voltage_solutions/Voltages_to_write.csv", "a+") as f:
        #    f.write(write_np_array)
        # send values to controller PC/FPGA
        with warnings_raise_exceptions(RPCWarning):
            chunk_size = 1000
            try:
                # Disallow silent fail (warning) if PYON can't encode/send the data.
                # Other code ensures data goes through
                # _LOGGER.debug("Chunking voltages for sending across TCP/PYON.")
                self._write_voltages_in_chunks(write_np_array, start_line, chunk_size, force_full_update=force_full_update)
            except (RPCWarning, ConnectionResetError, BrokenPipeError):
                _LOGGER.debug(
                    "Issue while trying to write large chunks of voltages. "
                    "Breaking into smaller chunks for sending across TCP/PYON."
                )
                try:
                    self._write_voltages_in_chunks(
                        write_np_array, start_line, chunk_size / 2, force_full_update=force_full_update
                    )
                except Exception as err:
                    raise aqexc.GenericRemoteException(
                        "Sending smaller voltage chunks didn't work. "
                        "Try decreasing chunk_size or increasing ARTIQ rpc data sizes."
                    ) from err
            else:
                _LOGGER.debug("Writing voltages to DAC succeeded.")

    def _write_voltages_in_chunks(
        self, voltage_array: numpy.ndarray, start_line: int, chunk_size: int = 1000, force_full_update: bool = True
    ):
        """Write voltages to DAC in chunks."""
        num_chunks = math.ceil(float(len(voltage_array)) / chunk_size)
        _LOGGER.debug(
            "Sending %d voltage chunks to DAC (across TCP/PYON, size limited)",
            num_chunks,
        )

        chunk_data_filepath = os.path.dirname(os.path.abspath(__file__))+"/chunk_data/"
        if not os.path.exists(chunk_data_filepath):
            os.mkdir(chunk_data_filepath)
        _LOGGER.debug(f"Loading chunk data from {chunk_data_filepath}")
        # Load old chunk data
        prev_chunk_data = []
        for j in range(num_chunks):
            try:
                prev_chunk_data += [numpy.load(f"{chunk_data_filepath}chunk_{j}.npy")]
            except:
                _LOGGER.debug(f"Chunk File {j} Not Found, generating empty")
                prev_chunk_data += [numpy.zeros((chunk_size, 112))]

        # Append chunk_data until it is the same length as the new data
        while len(prev_chunk_data) < num_chunks:
            prev_chunk_data += [numpy.zeros((chunk_size, 112))]

        for i in range(num_chunks):
            start_address = start_line + (i * chunk_size)
            if i == (num_chunks - 1):
                chunk_min, chunk_max = (i * chunk_size), len(voltage_array)
            else:
                chunk_min, chunk_max = (i * chunk_size, (i + 1) * chunk_size)
            new_chunk = voltage_array[chunk_min:chunk_max]
            # If chunk data different
            if (not numpy.array_equal(new_chunk, prev_chunk_data[i])) or force_full_update:
                _LOGGER.debug(f"Writing DAC Chunk {i}/{num_chunks}")
                # Write to DAC
                self.hardware.write_voltage_lines(
                    start_address,
                    new_chunk,
                    self.uploaded_data_hash,
                )
                # Store chunk update.
                prev_chunk_data[i] = new_chunk

        # write chunk data store
        _LOGGER.debug(f"Saving Chunk Data to {chunk_data_filepath}")
        for j in range(num_chunks):
            numpy.save(f"{chunk_data_filepath}chunk_{j}.npy", prev_chunk_data[j])

        return chunk_size

    def _check_can_shuttle(self):
        if not self.can_shuttle:
            raise RuntimeError(
                "Cannot shuttle until you write the shuttling graph "
                "to the FPGA (run send_shuttling_lookup_table)"
            )
        self._check_can_output()

    def _check_can_output(self) -> bool:
        if self.uploaded_data_hash is None:
            raise RuntimeError("Must update DAC voltage values before outputting")

    # *** ASYNC CONTROL METHODS ***
    # CANNOT be called from @kernel code, but useful when running from host
    def apply_line_async(
        self,
        to_line_or_name: typing.Union[str, int],
        line_gain: float = 1.0,
        global_gain: float = 1.0,
        trigger_immediate: bool = True,
    ) -> TNone:
        """
        Write a line (set of output voltages) to the DAC outputs.

        Args:
            to_line_or_name(Union[str, int]): node name or line number that will
                be applied
            line_gain (float): the gain to apply to the particular line
                (1.0 for no gain)
            global_gain (float): the global gain to apply to the whole system
                (1.0 for no gain)
            trigger_immediate (bool): whether the DAC should be updated immediately to
                match this line
        """
        # self._check_can_output()
        line_number = self._normalize_to_line(to_line_or_name)

        _LOGGER.debug(
            "Applying voltage: (lineNum, lineGain, globGain) = (%.1f, %.2f, %.2f)",
            line_number,
            line_gain,
            global_gain,
        )

        # Check if current data and settings match what's on the FPGA.
        # If they match, don't need to write a full line and can just jump to line.
        update_succeeded = False
        if (
            numpy.any(self.lines)
            and line_number in self.file_line_num_to_address_dict.keys()
            and self.is_shuttling_data_valid
            # TODO: Track down the bug of FPGA not uploading
            # never reuse FPGA lines before bug is tracked down
            and False
        ):
            _LOGGER.debug("Reusing line stored on FPGA")
            # TODO: avoid hardware calls in checking, because network lag.
            self.hardware.apply_line_address(
                self.file_line_num_to_address_dict[line_number],
                immediate_trigger=trigger_immediate,
            )
            update_succeeded = True
        else:
            line = self._calculate_line(
                float(line_number), float(line_gain), float(global_gain)
            )
            _LOGGER.debug("Writing line NOT stored on FPGA: \n%s", line)
            # print([l for l in line])
            try:
                self.hardware.apply_line(line) #Todo: , line_num=line_number)
            except aqexc.GenericRemoteException:
                _LOGGER.error("Remote exception in apply_line()")
            else:
                update_succeeded = True

        if update_succeeded:
            self.line_number = line_number
            if self.shuttling_graph is not None:
                self.shuttling_graph.set_position(line_number)
            else:
                _LOGGER.warning(
                    "Need to create shuttling graph to keep track of position"
                )
            self._notify_output_changed()
        else:
            _LOGGER.error("Applying line to FPGA failed.")

    def _normalize_to_name(
        self, line_num_or_name: typing.Union[str, int], none_allowed: bool = False
    ):
        if line_num_or_name is None and not none_allowed:
            raise ValueError("Cannot handle None as line number or name")
        if line_num_or_name is None:
            # Try to use current positions
            if self.shuttling_graph.current_position is not None:
                curr_node = self.shuttling_graph.get_node_name(
                    self.shuttling_graph.current_position
                )
                if curr_node is None:
                    raise RuntimeError(
                        "Current position is not a shuttling graph node. "
                        "Please change line to a node then retry."
                    )
                else:
                    return curr_node
            else:
                # if graph current position not initialized.
                raise RuntimeError(
                    "Current position has not been initialized. "
                    "No idea where to shuttle from. "
                    "Either use a node/line # or apply_line first"
                )
        elif isinstance(line_num_or_name, (int, float)):
            return self.shuttling_graph.get_node_name(line_num_or_name)
        elif hasattr(line_num_or_name, "magnitude"):
            # handle pint quantities, from Sandia GUI boxes.
            return self.shuttling_graph.get_node_name(int(line_num_or_name.magnitude))
        elif isinstance(line_num_or_name, str):
            name = line_num_or_name.lower()
            assert name in self.shuttling_graph.nodes()
            return name
        else:
            raise TypeError(
                "Invalid type provided: type({}) = {}".format(
                    line_num_or_name, type(line_num_or_name)
                )
            )

    def _normalize_to_line(
        self, line_num_or_name: typing.Union[str, int], none_allowed: bool = False
    ):
        if line_num_or_name is None and not none_allowed:
            raise ValueError("Cannot handle None as line number or name")
        if line_num_or_name is None:
            # Try to use current position
            if self.line_number is not None:
                line = self.line_number
            else:
                # if graph current position not initialized.
                raise RuntimeError(
                    "Current position has not been initialized. "
                    "No idea where to shuttle from. "
                    "Either use a node/line # or apply_line first"
                )
        elif isinstance(line_num_or_name, (int, float)):
            line = int(line_num_or_name)
        elif hasattr(line_num_or_name, "magnitude"):
            # pint quantity, can be converted to int with .magnitude
            line = int(line_num_or_name.magnitude)
        elif isinstance(line_num_or_name, str):
            line = self.shuttling_graph.get_node_line(line_num_or_name)
            if line is None:
                raise ValueError(
                    "Could not find node name in shuttling graph: {}".format(
                        line_num_or_name
                    )
                )
        else:
            raise TypeError(
                "Invalid type provided: type({}) = {}".format(
                    line_num_or_name, type(line_num_or_name)
                )
            )

        return line

    def get_shuttle_path(
        self,
        to_line_or_name: typing.Union[str, int],
        from_line_or_name: typing.Optional[typing.Union[str, int]] = None,
    ) -> typing.Sequence[shuttle_defs.ShuttlePathEdgeDescriptor]:
        """Return a shuttling path from one point on the graph to another."""
        # standardize input types
        to_node_name = self._normalize_to_name(to_line_or_name)
        from_node_name = self._normalize_to_name(from_line_or_name, none_allowed=True)

        _LOGGER.debug(
            "Obtaining shuttle path from `%s` to `%s`", from_node_name, to_node_name
        )
        return self.shuttling_graph.get_path_from_node_to_node(
            from_name=from_node_name, to_name=to_node_name
        )

    def shuttle_async(
        self,
        to_line_or_name: typing.Union[str, int],
        from_line_or_name: typing.Optional[typing.Union[str, int]] = None,
    ) -> TInt32:
        """
        Shuttle to a different point in the trap.

        Note: if called from prepare(), or any @kernel function, will run without
        waiting for trigger from ARTIQ. So essentially will execute all shuttling
        at beginning of program. Better way: use
        :mod:`euriqabackend.coredevice.sandia_dac_core` with this. Use
        :meth:`get_shuttle_path` to get the path, convert to serial commands
        (:meth:`euriqabackend.coredevice.sandia_dac_core.DACSerialCoredevice.
        path_to_data`), then execute those commands in realtime ARTIQ with
        :meth:`euriqabackend.coredevice.sandia_dac_core.DACSerialCoredevice.
        output_command_and_data` and
        :meth:`euriqabackend.coredevice.sandia_dac_core.DACSerialCoredevice.
        trigger_shuttling`

        Args:
            to_line_or_name(Union[str, int]): destination node that the
                ion/chain is being shuttled to
            from_line_or_name(Optional[Union[str, int]]): Origination node that
                the ion/chain is shuttling FROM. Defaults to current node.

        Returns:
            (int): line that the shuttling ends on (last line output from DAC)

        """
        self._check_can_shuttle()
        to_node_name = self._normalize_to_name(to_line_or_name)
        from_node_name = self._normalize_to_name(from_line_or_name, none_allowed=True)

        if from_node_name == to_node_name:
            _LOGGER.warning(
                "You're already at %s, you can't shuttle there!", from_node_name
            )
            raise ValueError()
            # return None

        # Determine path to shuttle along
        shuttle_path = self.get_shuttle_path(to_node_name, from_node_name)

        # execute shuttling on hardware
        _LOGGER.debug(
            "Starting finite shuttling from %s to %s", from_node_name, to_node_name
        )
        return self.shuttle_along_path_async(shuttle_path)

    def shuttle_along_path_async(
        self, shuttle_path: typing.Sequence[shuttle_defs.ShuttlePathEdgeDescriptor]
    ) -> int:
        """Execute a provided shuttling path.

        See :meth:`shuttle_async` for more details.
        """
        self._check_can_shuttle()
        _LOGGER.debug("Shuttling along path: %s", shuttle_path)
        # global_adjust = [0] * len(self.lines[0])
        # self._adjust_line(global_adjust)
        edge_descriptors = [
            tuple(desc)
            for desc in shuttle_defs.path_descriptor_to_fpga_edge_descriptor(
                shuttle_path
            )
        ]
        self.hardware.shuttle_along_path(edge_descriptors)

        # sync up last line
        last_edge = shuttle_path[-1]
        last_edge_reversed = last_edge.from_node_name != last_edge.edge.start_name
        self.line_number = (
            last_edge.edge.stop_line
            if not last_edge_reversed
            else last_edge.edge.start_line
        )
        self.shuttling_graph.set_position(self.line_number)
        _LOGGER.debug(
            "Shuttled to %s (line=%i)", last_edge.to_node_name, self.line_number
        )
        return self.line_number

    def trigger_shuttling_async(self) -> TNone:
        """
        Triggers the DAC to output the queued voltage output.

        Only executes remote call, :meth:`euriqabackend.devices.sandia_dac.driver.
        SandiaDACDriverLow.triggerShuttling`
        """
        self._check_can_shuttle()
        self.hardware.trigger_shuttling()
        self._notify_output_changed()

    def set_global_gain_async(self, global_gain: float, update_hardware: bool = True):
        """
        Set the global gain of the DAC.

        Args:
            global_gain (float): Multiplicative global gain to apply
            update_hardware (bool, optional): Defaults to True.
                Should the gain be applied to the hardware immediately?
        """
        self.global_gain = global_gain
        if update_hardware:
            self.apply_line_async(self.line_number, self.line_gain, self.global_gain)
            self._notify_output_changed()

    def dac_on_async(self) -> TNone:
        """Turn DAC on (gain = 1)."""
        self.set_global_gain_async(1.0, update_hardware=True)

    def dac_off_async(self) -> TNone:
        """Turn DAC off (gain = 0)."""
        self.set_global_gain_async(0.0, update_hardware=True)

    # VOLTAGE OUTPUT CALCULATION UTILITIES
    _hash_variables = (
        "line_gain",
        "global_gain",
        "adjustment_gain",
        "adjustment_dictionary",
    )

    def _shuttling_data_hash(self) -> TInt32:
        """Hashes the gain states and adjustments to verify settings."""
        settings_hash = special_hash.make_hash(
            (tuple(getattr(self, field) for field in self._hash_variables),)
        )
        _LOGGER.debug("Shuttling Settings Hash: %X", settings_hash)
        return settings_hash

    @property
    def is_shuttling_data_valid(self) -> bool:
        """Return true if the shuttling data is equivalent to the uploaded data."""
        return self._shuttling_data_hash() == self.uploaded_data_hash

    def set_adjustments(
        self, adjustments: typing.Dict, adjustment_gain: float
    ) -> TNone:
        """
        Apply some adjustment to the current line.

        Uses the adjust and adjustment_gain for future, but applies the changes
        immediately.

        Args:
            adjustments(Dict): a dictionary of adjustments.
                Example: {"adjustment_1": {"line": 10, "adjustment_gain": 1.0}}
                `line` is the line number in the adjustment file that is referenced.
                `adjustment_gain` is the multiplied scale factor.
            adjustment_gain(float): number to be multiplied by every adjustment value.
        """
        _LOGGER.debug("Setting new adjustments: %s", adjustments)
        if not set(self.adjustment_dictionary.keys()).issuperset(
            set(adjustments.keys())
        ):
            _LOGGER.warning(
                "New adjustments contain adjustment(s) not in current adjustments: %s",
                set(adjustments.keys()).difference(
                    set(self.adjustment_dictionary.keys())
                ),
            )
        self.adjustment_dictionary = adjustments
        self.adjustment_gain = float(adjustment_gain)
        self.apply_line_async(self.line_number, self.line_gain, self.global_gain)
        return

    def _calculate_line(
        self, line_number: float, line_gain: float, global_gain: float
    ) -> typing.Sequence[float]:
        """
        Calculate the output voltages for a particular line, given gain/adjustments.

        Args:
            line_number(float): the line to be calculated.
                Can be a float (e.g. for interpolating between line 0 and
                line 1)
            line_gain(float): the gain to apply to a particular (set to 1.0 for no
                effect)
            global_gain(float): gain to apply to every line
        """
        # Modify here to change the global scaling of line voltage (Need to change it back during reloading!!!)
        line = self._blend_lines(line_number, line_gain)
        #print("linenum", line_number)
        #print("linegain", line_gain)
        # line = self._blend_lines(line_number, line_gain)
        self.line_gain = line_gain
        self.global_gain = global_gain
        line = line + self._get_tweak_offset(line_number) # self._adjust_line(line) * self.global_gain # This commented line applies a global adjustment
        return line

    def _adjust_line(self, line: numpy.ndarray) -> numpy.ndarray:
        """
        Offset the magnitude of a voltage line according to a dictionary of adjustments.

        Calculates the magnitude of each adjustment, sums all adjustments, and then
        adds that to the current line (set of output voltages) as a DC offset.
        """
        offset = numpy.zeros(len(line))
        for adjust in self.adjustment_dictionary:
            offset += self.line_adjustment_values[
                int(self.adjustment_dictionary[adjust]["line"])
            ] * float(self.adjustment_dictionary[adjust]["adjustment_gain"])
        offset *= self.adjustment_gain
        return line + offset

    def _get_tweak_offset(self, line_number) -> numpy.ndarray:
        """
        Offsets the shuttling line <line_number> according to a dictionary of tweaks
        (self.<tweak_dictionary>)

        The keys of the tweak dictionary are the line numbers to which the corresponding
        tweaks will be applied. The tweaks (dict. values) are specified as dictionaries
        (<compensation_name> e.g. "DX", "DY"..., value)
        """
        line_number=int(math.floor(line_number))
        offset = numpy.zeros(len(self.lines[line_number]))

        if line_number in self.tweak_dictionary:
            for adjust in self.tweak_dictionary[line_number]:
                offset += self.line_adjustment_values[
                              int(self.adjustment_dictionary[adjust]["line"])
                          ] * float(self.tweak_dictionary[line_number][adjust])
        return offset

    def _blend_lines(self, line_number: float, line_gain: float) -> numpy.ndarray:
        """
        Interpolates between two voltage output lines, then applies a gain.

        Uses linear interpolation, the factor is determined by the decimal
        portion of lineNumber

        Args:
            line_number(float): line number in the voltages file to interpolate between
            line_gain(float): number to multiply result by

        Returns:
            (numpy.ndarray): Interpolated line, in numpy vector format

        """

        if self.lines is not None:
            left = int(math.floor(line_number))
            right = int(math.ceil(line_number))
            convexc = line_number - left
            leftLine = self.lines[left]
            rightLine = self.lines[right]
            try:
                return (
                    leftLine * (1 - convexc) + rightLine * convexc
                ) * line_gain
            except IndexError as err:
                raise IndexError(
                    "Provided an out-of-bounds line number: "
                    "{} not in [0, {}]".format(line_number, len(self.lines) - 1)
                ) from err
        else:
            raise RuntimeError(
                "Need to load lines before you can blend. "
                "Try calling load_voltage_file() first."
            )

    @property
    def current_output_voltage(self):
        """Get the current output voltage of the DAC."""
        # changed to calculating this on the fly cause it's not used often, and
        # it's hard to keep it in sync. Better to worry about syncing fewer things
        # (just line_number here)
        return self._calculate_line(self.line_number, self.line_gain, self.global_gain)

    def _notify_output_changed(self):
        """Send PyQt notification that the output voltages changed.

        Checks if any invalid voltages and sends those too.
        """
        self.voltages_changed.emit(0, 0, len(self._pin_names) - 1, 3)
        current_line = numpy.array(self.current_output_voltage)
        bad_voltages = numpy.less(current_line, -10) | numpy.greater(current_line, 10)
        if numpy.any(bad_voltages):
            self.voltage_errors.emit(bad_voltages.tolist())

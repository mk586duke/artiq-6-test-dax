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
Run the "Sandia" 100x DAC (DAC with ~100 output lines) via a PC.

Based on :mod:`artiq.devices.pdq2.driver` and the Sandia pyIonControl software.

Directly uses Peter Maunz's pulser/DACController and OpalKelly wrappers
(sandia_opalkelly/OKBase, sandia_opalkelly/bitfileHeader).

Note:
    * OpalKelly (ok) drivers need installed on the local machine to run this
        file, because DACController requires OpalKelly drivers to run. **Can
        "fix" this** by moving DACController import into just
        :class:`.SandiaDACFPGAHardware`, but that's not good practice.

    * If you just want to run a simulation, you do not need OpalKelly
        drivers.
"""
import logging
import os
import unittest.mock as mock
from typing import Sequence

import numpy

from ..sandia_opalkelly.dac.ShuttlingDefinition import ShuttleEdgeDescriptor
from ..sandia_opalkelly.dac.ShuttlingDefinition import ShuttleEdgeFPGAEntry

_LOGGER = logging.getLogger(__name__)

try:
    from ..sandia_opalkelly.dac.DACController import DACController
except ImportError:
    _LOGGER.warning(
        "Warning: DACController could not be found. OpalKelly Python "
        "interface ('ok') module is probably not available. "
        "You will ONLY be able to run simulations"
    )


class SandiaDACDriverLow(object):
    """
    Low-level driver for Sandia 100-channel DAC.

    Acts as a wrapper for Sandia's OpalKelly DAC drivers.
    Low-level because it interfaces directly with hardware.
    """

    _numpy_print_options = {"suppress": True, "precision": 3}

    def __init__(
        self, fpga: str = "DAC Test", bitfile: str = None, simulation: bool = False
    ):
        """Connect to the Sandia DAC and upload bitfile to it."""
        if fpga is None or not isinstance(fpga, str):
            raise ValueError("Bad FPGA name passed: {}".format(fpga))
        self.fpga_name = fpga
        self.fpga_bitfile_path = bitfile

        # validate bitfile input
        if bitfile is not None and not simulation:
            if not os.path.exists(bitfile):
                raise FileExistsError(
                    "Bitfile {} does not exist."
                    "Make sure that you are using the proper file, "
                    "or that this driver is being run from proper computer".format(
                        bitfile
                    )
                )

        self.simulation = simulation
        self.settings_hash = None

        if self.simulation:
            # insert monkeypatches here

            # use a Mock. This allows you to call a "fake" object (i.e. for
            # simulation), without having to create a separate class. This
            # means that the interface and logging and calls will still be
            # the same without any overhead, and you can track all calls made
            # to the mocked object (dacController in this case), and make
            # sure that they are all valid. Ideally would set
            # spec=ClassToBeMocked, but if can't import DACController then
            # can't do that.

            # Spec is the list of available methods. Set other attributes (
            # properties) in constructor (e.g. channelCount)
            self.dacController = mock.NonCallableMock(
                spec=[
                    "listBoards",
                    "openBySerial",
                    "uploadBitfile",
                    "writeVoltage",
                    "readVoltage",
                    "shuttleDirect",
                    "triggerShuttling",
                    "shuttle",
                    "close",
                    "writeShuttleLookup",
                    "shuttleDirect",
                    "verifyVoltages",
                    "writeVoltages",
                    "shuttlePath",
                    "memToVoltage",
                ],
                channelCount=112,
                name="dacController",
            )
            board_mock = mock.NonCallableMagicMock(
                modelName="mock_model_name", serial="mock_serial_port"
            )
            self.dacController.listBoards.return_value = {
                fpga: board_mock,
                fpga + str(2): board_mock,
            }

            # set default return values
            type(self.dacController).isOpen = mock.PropertyMock(return_value=True)

            print("Beginning simulation.")
        else:
            try:
                self.dacController = DACController()
            except NameError as err:
                raise ImportError(
                    "DACController has not imported properly "
                    "because ok (OpalKelly) python interface not "
                    "installed. Please install it to use {"
                    "}".format(self.__class__.__name__)
                ) from err

        self.opalkelly_fpga_devices = self.dacController.listBoards()
        _LOGGER.debug(
            "Opal Kelly Devices found: %s",
            {k: v.modelName for k, v in self.opalkelly_fpga_devices.items()},
        )

        try:
            device = self.opalkelly_fpga_devices[self.fpga_name]
        except KeyError as err:
            _LOGGER.error(
                "Sandia DAC FPGA '%s' is not accessible. Try closing any other "
                "OpalKelly software (like FrontPanel), check if "
                "there is another instance of this driver running, "
                "or check that you have the right name",
                self.fpga_name,
            )
            raise ValueError(
                "Could not connect to FPGA '{}'. "
                "Try closing OpalKelly FrontPanel or Sandia.".format(self.fpga_name)
            ) from err

        self.dacController.openBySerial(device.serial)

        if self.fpga_bitfile_path:
            # NOTE: uploading bitfile turns off DAC outputs, unknown ion location
            self.dacController.uploadBitfile(self.fpga_bitfile_path)
            _LOGGER.debug(
                "Uploaded file '%s' to %s (Opal Kelly FPGA %s)",
                self.fpga_bitfile_path,
                self.fpga_name,
                device.modelName,
            )

    def apply_line(self, voltage_line: numpy.ndarray, line_num:int = 0) -> None:
        """
        Apply a particular set of voltages to the DAC.

        Functionally this means that we set the DC voltage levels to some
        particular value, essentially creating some potential well(s) in the
        trap that control ion locations.

        Args:
            line: a set of voltages to IMMEDIATELY apply to the DACs.
            line_num: the line number in memory to overwrite

        Example:
            >>> import numpy as np
            >>> dac = SandiaDACFPGAHardware(simulation=True)
            >>> dac.apply_line(2 * np.ones((112)))   # Apply 2V to all outputs
        """
        if len(voltage_line) != self.channel_count():
            _LOGGER.warning(
                "Only applying voltages to %i/%i channels",
                len(voltage_line),
                self.channel_count(),
            )
        self.dacController.writeVoltage(line_num, voltage_line)
        self.dacController.readVoltage(line_num, voltage_line)
        self.dacController.shuttleDirect(line_num, line_num+1, idleCount=0, immediateTrigger=True)
        self.dacController.triggerShuttling()
        with numpy.printoptions(**self._numpy_print_options):
            _LOGGER.debug("Outputting voltages \n%s", voltage_line)

    def apply_line_address(self, line_address: int, immediate_trigger=True):
        """
        Jump directly to a line in memory.

        Differs from :meth:`apply_line` because that one writes a new set of
        voltages to the DAC, so it has higher communication overhead.

        Args:
            line_address (int): Which line in FPGA memory to jump to for
                outputting voltages.
            immediate_trigger (bool, optional): Defaults to True. Whether the
                output should be changed immediately or wait for a trigger.
        """
        _LOGGER.debug("Outputting voltages at memory address %i", line_address)
        self.dacController.shuttleDirect(
            line_address,
            line_address + 1,
            idleCount=0,
            immediateTrigger=immediate_trigger,
        )
        # TODO: extra trigger for surety?
        # self.dacController.triggerShuttling()

    def read_voltage_line_at_address(self, line_address: int) -> None:
        """Read the output voltages at a particular address in memory.

        Note that address != line: if you have steps per line > 1, then this
        will change the address where a line is stored.
        """
        return self.dacController.readVoltage(line_address)

    def jump_to_line(self, line_address: int) -> None:
        """Jump directly to a specified line in the memory.

        Does not check to see if that address is valid, so could move to
        garbage/old voltage outputs.
        """
        voltages_at_address = self.dacController.memToVoltage(
            self.read_voltage_line_at_address(line_address)
        )
        with numpy.printoptions(**self._numpy_print_options):
            _LOGGER.debug("Line %i data:\n%s", line_address, voltages_at_address)
        return self.apply_line(voltages_at_address)

    def shuttle(
        self,
        lookupIndex: int,
        reverseEdge: bool = False,
        immediateTrigger: bool = False,
    ) -> None:
        """
        Shuttle ions between two locations that are joined by an edge.

        Edges are defined in the shuttling definition file.

        TODO: CHECK THESE
        Args:
            lookupIndex: shuttling solution edge to apply. Shuttle should
            fail if you start on a line that is not a beginning/ending node
            of the edge.
            reverseEdge: if the edge should be traversed in reverse
            immediateTrigger: begin shuttling immediately if ``True``
        """
        self.dacController.shuttle(lookupIndex, reverseEdge, immediateTrigger)
        if immediateTrigger:
            _LOGGER.info(
                "Shuttled to %i in %s order",
                lookupIndex,
                "forward" if not reverseEdge else "reverse",
            )
        else:
            _LOGGER.info(
                "Set up to shuttle to %i in %s order",
                lookupIndex,
                "forward" if not reverseEdge else "reverse",
            )

    def shuttle_along_path(self, path: Sequence[ShuttleEdgeDescriptor]) -> None:
        """
        Shuttle the ion along a particular path.

        Path is made of one or more edges in the shuttling lookup table
        that share beginning/ending nodes.

        Args:
            path: a set of edges in the pre-written shuttling lookup table
                to traverse between an initial and final voltage solution.
                Functionally, these define a set of voltage levels to apply to
                the trap in sequential order.
                Each value in path is a tuple of the index in the lookup table,
                which order the edge should be traversed in, and if the edge
                should be started immediately or wait for a trigger
        """
        self.dacController.shuttlePath(path)
        _LOGGER.info("Shuttled via path %s", path)

    def close(self) -> None:
        """Close and shut down the control system."""
        self.dacController.close()
        if self.simulation:
            _LOGGER.info(
                "Simulator received following method calls: \n%s",
                self.dacController.method_calls,
            )
        del self.dacController
        _LOGGER.info("Deleted dacController and shut down Sandia DAC control")

    def write_voltage_lines(
        self, address: int, line_list: Sequence[numpy.ndarray], settings_hash: int
    ) -> None:
        """
        Store voltage outputs in memory on the FPGA for shuttling.

        Store voltage solution table in memory on the FPGA controller for
        retrieval during shuttling. Can write a partial table.

        Args:
            address: first address of the voltage solution table to be updated
                (seems to be the same as line number in its usage)
            lineList: voltage solutions to be uploaded (seems to be a list
                of voltages.)
            settings_hash: hash of settings like gain to check that global
                settings are the same as on FPGA
        """
        if numpy.array(line_list).size % self.channel_count():
            raise ValueError(
                "Invalid size/shape of lineList: {}."
                "Should be a multiple of {}".format(
                    numpy.array(line_list).shape, self.channel_count()
                )
            )
        data = self.dacController.writeVoltages(address, line_list)
        self.dacController.verifyVoltages(address, data)
        self.settings_hash = settings_hash
        _LOGGER.debug(
            "Wrote voltages to lines %s starting at address %i", line_list, address
        )

    def write_shuttling_lookup_table(
        self, shuttling_edges: Sequence[ShuttleEdgeFPGAEntry], start_address: int = 0
    ) -> None:
        """
        Store shuttling graph/table on the FPGA for retrieval during shuttling.

        Args:
            shuttling_edges (Sequence[ShuttleEdgeFPGAEntry]): representations of
                shuttling edges to be stored in the FPGA memory.
            start_address (int): first address of the shuttling lookup table to be
                updated (seems to be the same as lookupIndex in usage).
        """
        self.dacController.writeShuttleLookup(shuttling_edges, start_address)
        _LOGGER.info(
            "Wrote shuttle lookup table with %i edges %s starting at address %i",
            len(shuttling_edges),
            shuttling_edges,
            start_address,
        )

    def trigger_shuttling(self) -> None:
        """
        Trigger shuttling (i.e. changing DAC output) to begin.

        Should have been pre-loaded with a particular shuttling solution to
        execute.
        """
        self.dacController.triggerShuttling()
        _LOGGER.info("Triggered shuttling")

    def channel_count(self) -> int:
        """
        Return the number of channels that the DAC can output.

        Returns:
             (int) number of output DAC channels.

        """
        return self.dacController.channelCount

    def is_open(self) -> bool:
        """
        Return true if the serial connection to the FPGA (i.e. the DAC) is open.

        Returns:
             True if connected to FPGA, False otherwise. Always True in simulation.

        """
        return self.dacController.isOpen

    def ping(self) -> bool:
        """
        Test to check that the driver exists & is running.

        Returns:
            Always ``True``.

        """
        return True

    def needs_new_data(self, remote_settings_hash: int) -> bool:
        """
        Check if data currently on FPGA is valid with current (remote) settings.

        If settings are invalid, will need to reload data.
        If settings match (returns `False`), no data needs to be reloaded

        Args:
            remote_settings_hash: Hash of the remote settings, generated by
                :meth:`euriqabackend.devices.sandia_dac.interface
                .SandiaDACInterface._shuttlingDataHash`

        Returns:
            True if the data needs to be refreshed on device, i.e. local
            settings hash != remote_settings_hash or no data yet. Else False.

        """
        if self.settings_hash is None:
            return True
        return self.settings_hash != remote_settings_hash

"""
Low-level control of the Sandia DAC.

Copyright
=========
IonControl:  Copyright 2016 Sandia Corporation
This Software is released under the GPL license detailed
in the file "license.txt" in the top-level IonControl directory
"""
import logging
import struct
import typing
from itertools import chain

import numpy

from ..OKBase import check
from ..OKBase import OKBase
from .ShuttlingDefinition import ShuttleEdgeDescriptor
from .ShuttlingDefinition import ShuttleEdgeFPGAEntry


class DACControllerException(Exception):
    """Exception from the low-level DAC Controller."""

    pass


_LOGGER = logging.getLogger(__name__)


class DACController(OKBase):
    """Controller class built on Opalkelly software to control the Sandia x100 DAC."""

    channelCount = 112  # Number of output DAC analog channels.

    # TODO: redo this method
    @staticmethod
    def edge_to_lookup_encode(edge: ShuttleEdgeFPGAEntry):
        """Encode an edge in the shuttling graph into FPGA memory.

        Includes the start and stop addresses in memory,
        and the number of cycles to idle (between outputs??).
        """
        edge = tuple(edge)
        # 0x0 is the entry terminator
        return struct.pack("=4I", *edge, 0x0)

    def toInteger(self, iterable):
        """Convert an array of output voltages to the representation on the FPGA."""
        result = list()
        for value in chain(
            iterable[0::4], iterable[1::4], iterable[2::4], iterable[3::4]
        ):
            if not -10 <= value < 10:
                raise DACControllerException(
                    "voltage {0} out of range -10V <= V < 10V".format(value)
                )
            result.append(int(value / 10.0 * 0x7FFF))
        return result
        # list(chain(range(96)[0::4], range(96)[1::4], range(96)[2::4],
        # range(96)[3::4])) # list( [0x000 for _ in range(96)]) #result #

    def memToVoltage(self, iterable: typing.Iterable[int]) -> numpy.ndarray:
        """Convert an array of memory values (from FPGA) to voltages.

        Performs no bounds checking. Should be in range (unsigned) [0, 2^16 - 1].
        """
        return numpy.array([x * 10 / 0x7FFF for x in iterable])

    def writeVoltage(self, address, line):
        """Write a voltage `line` to the FPGA memory at a given `address`.

        Line is in units of volts.
        """
        if self.xem:
            if len(line) < self.channelCount:
                line = numpy.append(
                    line, [0.0] * (self.channelCount - len(line))
                )  # extend the line to the channel count
            startaddress = (
                address * 2 * self.channelCount
            )  # 2 bytes per channel, 96 channels
            # set the host write address
            self.xem.WriteToPipeIn(
                0x84, bytearray(struct.pack("=HQ", 0x4, startaddress))
            )  # write start address to extended wire 2
            check(self.xem.ActivateTriggerIn(0x43, 6), "HostSetWriteAddress")

            data = bytearray(
                numpy.array(self.toInteger(line), dtype=numpy.int16).view(
                    dtype=numpy.int8
                )
            )
            _LOGGER.debug(
                "Writing %i bytes to DAC: %s", len(data), self.toInteger(line)
            )
            # check( self.xem.ActivateTriggerIn( 0x40, 2), 'ActivateTrigger' )
            return self.xem.WriteToPipeIn(0x83, data)

    def writeVoltages(self, address, lineList):
        """Write a list of voltage `lines` to the FPGA memory, starting at `address`.

        `linelist` is in units of Volts.
        """
        if self.xem:
            startaddress = (
                address * 2 * self.channelCount
            )  # 2 bytes per channel, 96 channels
            # set the host write address
            self.xem.WriteToPipeIn(
                0x84, bytearray(struct.pack("=HQ", 0x4, startaddress))
            )  # write start address to extended wire 2
            check(self.xem.ActivateTriggerIn(0x43, 6), "HostSetWriteAddress")

            odata = (
                numpy.array(lineList)
                .reshape((len(lineList), 28, 4))
                .swapaxes(1, 2)
                .flatten()
            )
            maximum = numpy.amax(odata)
            minimum = numpy.amin(odata)
            if maximum >= 10.0:
                raise DACControllerException(
                    "voltage {0} out of range V >= 10V".format(maximum)
                )
            if minimum < -10:
                raise DACControllerException(
                    "voltage {0} out of range V < -10V".format(minimum)
                )
            odata *= 0x7FFF / 10.0
            outdata = bytearray(odata.astype(numpy.int16).view(dtype=numpy.int8))
            _LOGGER.debug(
                "uploading %i bytes to DAC controller, %i voltage samples",
                len(outdata),
                len(outdata) / self.channelCount / 2,
            )
            self.xem.WriteToPipeIn(0x83, outdata)
            return outdata
        return bytearray()

    def verifyVoltages(self, address, data):
        """Verify the voltages in memory against a given set of voltages."""
        if self.xem:
            startaddress = (
                address * 2 * self.channelCount
            )  # 2 bytes per channel, 96 channels
            # set the host write address
            self.xem.WriteToPipeIn(
                0x84, bytearray(struct.pack("=HQ", 0x3, startaddress))
            )  # write start address to extended wire 2
            check(self.xem.ActivateTriggerIn(0x43, 7), "HostSetWriteAddress")

            returndata = bytearray(len(data))
            self.xem.ReadFromPipeOut(0xA3, returndata)
            matches = data == returndata
            if not matches:
                _LOGGER.error("Data verification failure")
            else:
                _LOGGER.debug("Data verified")
            return returndata
        return bytearray()

    def readVoltage(self, address, line=None):
        """Read voltages at `address` back from the FPGA."""
        if self.xem:
            startaddress = (
                address * 2 * self.channelCount
            )  # 2 bytes per channel, 96 channels
            # set the host write address
            self.xem.WriteToPipeIn(
                0x84, bytearray(struct.pack("=HQ", 0x3, startaddress))
            )  # write start address to extended wire 2
            check(self.xem.ActivateTriggerIn(0x43, 7), "HostSetReadAddress")

            data = bytearray(2 * self.channelCount)
            self.xem.ReadFromPipeOut(0xA3, data)
            result = numpy.array(data, dtype=numpy.int8).view(dtype=numpy.int16)
            if line is not None:
                if len(line) < self.channelCount:
                    line = numpy.append(
                        line, [0.0] * (self.channelCount - len(line))
                    )  # extend the line to the channel count
                matches = numpy.array_equal(result, numpy.array(self.toInteger(line)))
                if not matches:
                    _LOGGER.warning(
                        "%i %s", len(self.toInteger(line)), list(self.toInteger(line))
                    )
                    _LOGGER.warning("%i %s", len(result), list(result))
                    # raise DACControllerException(
                    #   "Data read from memory does not match data written"
                    # )
                    _LOGGER.warning("Data written and read does NOT match")
                else:
                    _LOGGER.debug("Data written and read matches")
            return result
        return bytearray()

    def writeShuttleLookup(
        self,
        shuttle_edges_fpga: typing.Sequence[ShuttleEdgeFPGAEntry],
        startAddress: int = 0,
    ):
        """Write shuttling edge lookup table to the FPGA."""
        if self.xem:
            data = bytearray()
            for shuttle_edge_fpga in shuttle_edges_fpga:
                data.extend(self.edge_to_lookup_encode(shuttle_edge_fpga))
            self.xem.SetWireInValue(0x3, startAddress << 3)
            self.xem.UpdateWireIns()
            self.xem.ActivateTriggerIn(0x40, 2)
            written = self.xem.WriteToPipeIn(0x85, data)
            _LOGGER.debug(
                "Wrote ShuttleLookup table %i bytes, %i entries", written, written / 16
            )
            self.xem.SetWireInValue(0x3, startAddress << 3)
            self.xem.UpdateWireIns()
            self.xem.ActivateTriggerIn(0x40, 2)
            mybuffer = bytearray(len(data))
            self.xem.ReadFromPipeOut(0xA4, mybuffer)
            if data == mybuffer:
                _LOGGER.debug("Written and read lookup data matches")
            else:
                _LOGGER.error("Written and read lookup data do NOT match")

    def shuttleDirect(
        self,
        startLine: int,
        beyondEndLine: int,
        idleCount: int = 0,
        immediateTrigger: bool = False,
    ):
        """
        Shuttle directly from a starting line up to but not including the end line.

        Does not follow the shuttling graph, just sequentially outputs the lines
        between [start_line_address, end_line_address) (not including end_line_address).

        Args:
            start_line_address (int): What address in the FPGA to start at
            end_line_address (int): What line in the FPGA to stop BEFORE
            idle_counts (int, optional): Defaults to 0. Number of clock cycles
                (20 ns) to wait between outputting sequential voltage lines
            immediate_trigger (bool, optional): Defaults to False. If the output
                should be started immediately.
        """
        if self.xem:
            self.xem.WriteToPipeIn(
                0x86,
                bytearray(
                    struct.pack(
                        "=IIII",
                        (0x01000000 | self.boolToCode(immediateTrigger)),
                        idleCount,
                        startLine * 2 * self.channelCount,
                        beyondEndLine * 2 * self.channelCount,
                    )
                ),
            )

    @staticmethod
    def boolToCode(b: bool, bit: int = 0):
        """Convert a boolean to a flag register mask."""
        return 1 << bit if b else 0

    def shuttle(
        self,
        lookupIndex: int,
        reverseEdge: bool = False,
        immediateTrigger: bool = False,
    ):
        """
        Prepare to execute one edge of the shuttling graph.

        Args:
            lookupIndex (int): Index of the desired edge in the shuttling
                lookup table.
            reverseEdge (bool, optional): Defaults to False. Whether the
                selected edge should be output in reverse order.
            immediateTrigger (bool, optional): Defaults to False. If the
                shuttling should be executed immediately, or wait for a
                different trigger.
        """
        if self.xem:
            self.xem.WriteToPipeIn(
                0x86,
                bytearray(
                    struct.pack(
                        "=IIII",
                        0x03000000,
                        0x0,
                        self.boolToCode(reverseEdge, 1)
                        | self.boolToCode(immediateTrigger),
                        lookupIndex,
                    )
                ),
            )

    def shuttlePath(self, path: typing.Sequence[ShuttleEdgeDescriptor]):
        """Shuttle along a given path."""
        if self.xem:
            data = bytearray()
            for lookupIndex, reverseEdge, immediateTrigger in path:
                data.extend(
                    struct.pack(
                        "=IIII",
                        0x03000000,
                        0x0,
                        self.boolToCode(reverseEdge, 1)
                        | self.boolToCode(immediateTrigger),
                        lookupIndex,
                    )
                )
            self.xem.WriteToPipeIn(0x86, data)

    def triggerShuttling(self):
        """Start the queued shuttling."""
        if self.xem:
            check(self.xem.ActivateTriggerIn(0x40, 0), "ActivateTrigger")

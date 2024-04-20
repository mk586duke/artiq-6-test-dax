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

"""Core device driver for the Sandia DAC to allow real-time-ish shuttling.

Basically sends RS232/UART-type commands to the DAC to trigger ion shuttling.
Assumes that the DAC has been pre-loaded with voltage lines and a lookup table via
:mod:`.driver` & :mod:`.interface`.

NOTE: For shuttling to be valid & work properly, requires that the DAC NOT be written to
during an experiment. Thus, any
:meth:`.interface.SandiaDACInterface.send_shuttling_lookup_table` or
meth:`.interface.SandiaDACInterface.send_voltage_lines_to_fpga` calls should only
happen from :meth:`~artiq.language.environment.EnvExperiment.run`.
NOT from :meth:`~artiq.language.environmentEnvExperiment.prepare`

Each packet sent is 72 bits, with a HIGH start bit and a LOW stop bit (in addition).
    * 8 command bits
    * 64 data bits

See "Serial Protocol README.md" for full details (excerpted here).

## DAC Serial Protocol Overview

* RS232 (UART) like.
* Comm Format: 1 start bit, 72 data bits, 1 stop bit
    * *Start bit:* 1
    * *Stop bit:* 0
* Default line to 0
* MSB sent first.
* Data layout: {8 command bits, 64 data bits}
* Samples at 1/10 the FPGA clock freq = 5 MHz. Max data = 5 Mbps =
    67k packets per second, ~ 150 us per packet/shuttle edge.
    * TODO: maybe revise the Verilog to speed up shuttling. Only an issue if
    trying to shuttle > 1 edge every 150 us, ~ 100 blended lines

## Serial Protocol Shuttling Commands

### Drew's interpretation from source code

* command[1:0] = 0b01: Execute lines directly.
    * data[63:32]: start output address in memory to output
    * data[31:0]: end address in memory to output
    * command[2]: trigger_immediate (1 for immediate, 0 for wait)
    * TODO: does this use idle counts??
* command[1:0] = 0b10: Set the idle count for the current output sequence
    * data[15:0] = Number of idle clock cycles to wait (clk cycle = 20 ns)
* command[1:0] = 0b11: Execute ONE shuttle edge
    * data[7:0] = shuttle lookup
    * data[32]: trigger mode: 1 to trigger immediately, 0 to wait
        (or imm if trigger high through transaction).
    * data[33]: reverse_edge. execute edge forward (0) or reverse (1)

"""
import enum
import logging
import typing

import artiq
import artiq.coredevice.spi2 as spi  # ARTIQ 4
import artiq.language.units as aqunits
import artiq.master.worker_db as db
import numpy as np
from artiq.language.core import at_mu
from artiq.language.core import delay_mu
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import now_mu
from artiq.language.core import portable
from artiq.language.types import TBool
from artiq.language.types import TInt32
from artiq.language.types import TInt64
from artiq.language.types import TList
from artiq.language.types import TNone

import euriqabackend.devices.sandia_opalkelly.dac.ShuttlingDefinition as shuttle_defs

# import artiq.coredevice.spi as spi  # ARTIQ 2


_LOGGER = logging.getLogger(__name__)
_ARTIQ_VERSION = int(artiq.__version__[0])


class _DACSerialCommand(enum.IntEnum):
    # Only 8 bits, but 32 for typing reasons
    # NOTE: disabled because non-working
    # direct_lines_wait_trigger = np.int32(0x1)
    # direct_lines_immediate = np.int32(0b101)
    set_idle_counts = np.int32(0x2)
    shuttle_edge_from_lookup = np.int32(0x3)


class DACSerialCoredevice:
    """
    Trigger and queue real-time-ish shuttling via Sandia 100x DAC.

    Driver to send data via custom one-wire serial protocol to Sandia 100x DAC.
    Exposes shuttle_edge, set_idle_counts, and shuttle_direct commands.

    NOTE: can only queue one output edge at a time. If the device is already
    outputting, this will set the next edge to execute.
    """

    kernel_invariants = {
        "data_period_mu",
        "trigger",
        "trigger_time_mu",
        "serial_out",
        "core",
        "_SPI_CONFIG",
        "spi_clk_div_mu",
    }

    _SPI_CONFIG = (
        0 & spi.SPI_OFFLINE
        | 0 & spi.SPI_CS_POLARITY
        | 0 & spi.SPI_CLK_POLARITY
        | 0 & spi.SPI_CLK_PHASE
        | 0 & spi.SPI_LSB_FIRST
        | 0 & spi.SPI_HALF_DUPLEX
    )
    TRIGGER_TIME = 400 * aqunits.ns

    def __init__(
        self,
        device_manager: db.DeviceManager,
        serial_device: str,
        trigger_line: str,
        data_period: float = 200 * aqunits.ns,
    ):
        """Initialize the DAC controller core device.

        Args:
            serial_device (str): name of SPI serial bus in device database
            trigger_line (str): name of trigger line (TTL/GPIO) in device database
            data_period (float): interval between data outputs on serial line in seconds
        """
        self.core = device_manager.get("core")
        self.serial_out = device_manager.get(serial_device)
        self.trigger = device_manager.get(trigger_line)
        if _ARTIQ_VERSION >= 4:
            self.data_period_mu = self.core.seconds_to_mu(data_period)
            self.trigger_time_mu = self.core.seconds_to_mu(self.TRIGGER_TIME)
        else:
            import artiq.language.core as aqcore

            self.data_period_mu = aqcore.seconds_to_mu(data_period, core=self.core)
            self.trigger_time_mu = aqcore.seconds_to_mu(
                self.TRIGGER_TIME, core=self.core
            )
        spi_clk_freq = 1 / data_period  # about 5 MHz
        self.spi_clk_div_mu = int(125 * aqunits.MHz / spi_clk_freq)
        # min time between sending commands, only used if preset=True
        # some slack, but close enough. Doesn't account for time the remote FPGA
        # spends outputting data to the DACs (>100 us, typically)
        self.min_wait_time_mu = 75 * self.data_period_mu

    @portable
    @staticmethod
    def _generate_packet(
        command: TInt32, data_upper: TInt32, data_lower: TInt32
    ) -> TList(TInt32):
        """Generate a Sandia DAC data packet: List of 3x 32-bit ints.

        Only lowest 8 bits of first int are used, total = 72 bits.

        Args:
            command (Union[TInt32, _DACSerialCommand]): The type of the serial command.
                Either change the idle counts or select an edge to execute.
            data_upper (TInt32): Upper 4 bytes of the data to make into a packet.
            data_lower (TInt32): Lower 4 bytes of the data to make into a packet.

        Returns:
            TList(TInt32): List of 32-bit integers denoting the serial
            data packet to be sent to the 100-channel Sandia DAC.

        Example:
            >>> [hex(i) for i in _generate_packet(0xFFF, 0xDEADDEAD, 0xBEEFBEEF)]
                ['0xff', '0xdeaddead', '0xbeefbeef']
        """
        # TODO: might need to change to add bordering [1, data, 0] bits.
        return [np.int32(command & 0xFF), np.int32(data_upper), np.int32(data_lower)]

    @portable
    @staticmethod
    def get_serial_shuttle_packet(
        edge_index: TInt32,
        reverse_edge: TBool = False,
        trigger_immediate: TBool = False,
    ) -> TList(TInt32):
        """Compute or precompute the data in a shuttle packet."""
        if reverse_edge:
            reverse_command = np.int32(0b10)
        else:
            reverse_command = np.int32(0b00)
        if trigger_immediate:
            trigger = 0b1
        else:
            trigger = 0b0
        return DACSerialCoredevice._generate_packet(
            _DACSerialCommand.shuttle_edge_from_lookup,
            data_upper=reverse_command | trigger,
            data_lower=edge_index & 0xFF,
        )

    @kernel
    def serial_shuttle(
        self,
        edge_index: TInt32,
        reverse_edge: TBool = False,
        trigger_immediate: TBool = False,
    ) -> TNone:
        """Queue or execute a shuttle edge on the 100x DAC.

        Can optionally reverse the edge or start executing the edge immediately
        versus waiting for a TTL/GPIO trigger.

        NOTE: can only queue/execute one shuttle edge at a time. You will need
        to trigger this edge before you can start queueing the next one.
        """
        self._output_serial_data(
            self.get_serial_shuttle_packet(edge_index, reverse_edge, trigger_immediate)
        )

    @portable
    @staticmethod
    def get_serial_idle_packet(idle_counts: TInt32) -> TList(TInt32):
        """Compute the data in an idle counts command packet."""
        return DACSerialCoredevice._generate_packet(
            _DACSerialCommand.set_idle_counts, 0x0, idle_counts & 0xFFFF
        )

    @kernel
    def serial_set_idle_counts(self, idle_counts: TInt32) -> TNone:
        """Set the idle counts for the current edge.

        Idle counts are the number of clock cycles to wait before outputting
        the next memory line (in addition to time to update the DAC values).
        """
        self._output_serial_data(self.get_serial_idle_packet(idle_counts))

    @portable
    @staticmethod
    def get_serial_output_memory_packet(
        start_line: TInt32, end_line: TInt32, trigger_immediate: TBool = False
    ) -> TList(TInt32):
        """Compute the data in a packet to set output of specified line range."""
        if trigger_immediate:
            command = _DACSerialCommand.direct_lines_immediate
        else:
            command = _DACSerialCommand.direct_lines_wait_trigger
        return DACSerialCoredevice._generate_packet(command, start_line, end_line)

    # TODO: not working!! Might be due to DAC needing memory ADDRESSES,
    # not memory LINE NUMBERS (i.e. line 7 ~ memory line 30 ~ address 30 * 112)
    # @kernel
    # def serial_output_memory(
    #     self, start_line: TInt32, end_line: TInt32, trigger_immediate: TBool = False
    # ) -> TNone:
    #     """Output a sequence of lines from memory (uses memory address, not line #).

    #     These should be the "blended" version of the lines, so linenum != address.
    #     See :attr:`euriqabackend.devices.sandia_dac.interface.SandiaDACInterface.
    #     file_line_num_to_address_dict`

    #     Can optionally start immediately, or wait for GPIO/TTL trigger.

    #     NOTE: NOT WORKING!!
    #     """
    #     self._output_serial_data(
    #         self.get_serial_output_memory_packet(
    #             start_line, end_line, trigger_immediate
    #         )
    #     )

    @kernel
    def _output_serial_data(
        self, data_list: TList(TInt32), preset: TBool = False
    ) -> TNone:
        """
        Output the data over serial port to the 100x DAC.

        Args:
            data_list (TList[TInt32]): a list of 3x 32-bit integers, in MSB->LSB order.
                Upper 24 bits of the first int in the list are ignored.
            preset (aqtype.TBool, optional): Defaults to True. If True, will
                put the time cursor in the past to output data so that it is
                ready at current time cursor value.
                Otherwise, increments time cursor from this position.
        """
        assert len(data_list) == 3
        num_outputs = 0
        start_time = now_mu()
        if preset:
            # rewind time cursor so all changes happen in past, are ready now()
            delay_mu(-74 * self.data_period_mu)
        for i in range(len(data_list)):
            data = data_list[i]
            if i == 0:
                num_bits = 9
                data = data | 0x100
            else:
                num_bits = 32
            num_outputs += self._output_int32_over_serial(data, num_bits)

        # stop bit, then reset line to 0
        num_outputs += self._output_int32_over_serial(0b10, 2)

        # check packet was sized properly.
        assert num_outputs == 75
        if preset:
            assert now_mu() <= start_time
            at_mu(start_time)

    @kernel
    def trigger_shuttling(self, preset: TBool = False) -> TNone:
        """
        Triggers shuttling with a digital pulse for pulse_time.

        Args:
            pulse_time (float): Defaults to 400ns.
                Recommended >= 100 ns
            preset (aqtype.TBool, optional): Defaults to False. If True,
                rewinds time cursor so this is ready at current time.
                Otherwise increases time cursor. Should probably leave at False
                to prevent race conditions with 100x DAC's internal logic
                processing the serial command.
        """
        start_time = now_mu()
        if preset:
            delay_mu(-self.trigger_time_mu)
        self.trigger.pulse_mu(self.trigger_time_mu)
        if preset:
            assert now_mu() == start_time

    @host_only
    def path_to_data(
        self,
        shuttle_path: typing.Sequence[shuttle_defs.ShuttlePathEdgeDescriptor],
        trigger_type: str = "first_only",
    ) -> [TList(TInt64), TList(TInt64)]:
        """Convert a shuttling path to a list of serial data packets to the DAC box.

        Args:
            shuttle_path (Sequence[ShuttlePathEdgeDescriptor]):
                A shuttle path generated by :meth:`euriqabackend.devices.sandia_dac
                .interface.SandiaDACInterface.get_shuttle_path`.
            trigger_type (str): What external triggers are needed to start shuttling.

        trigger types: "first_only", "all", "imm"/"immediate"
            * "first_only": send a trigger to start first waveform then
                every other executes immediately. Issue is that
                subsequent edges will overwrite the first edge if the
                first edge has not been triggered before next received.
            * "all": every edge needs a trigger
            * "imm"/"immediate": start every edge immediately.

        Returns:
            Tuple[Sequence[Int64], Sequence[Int64]]: *First item*: All edges in the
            shuttling path converted to FPGA-readable values. These values can be
            directly output in the data field of a serial command to the 100x DAC.
            *Second item*: the time (*in mu*) that it takes for each edge in
            the path to execute on the core device. When executing, need to
            delay at least this long before the next edge (doesn't count the
            time to output a new shuttle edge serial data packet).

        """
        data_list = list()
        edge_descriptors = list()
        trigger_type = trigger_type.lower().strip()
        if trigger_type == "first_only":
            edge_descriptors.append(
                shuttle_defs.path_descriptor_to_fpga_edge_descriptor(
                    shuttle_path[0], trigger_immediate=False
                )
            )
            edge_descriptors.extend(
                shuttle_defs.path_descriptor_to_fpga_edge_descriptor(
                    shuttle_path[1:], trigger_immediate=True
                )
            )
        elif trigger_type == "all" or trigger_type.startswith("imm"):
            immediate_trigger = True if trigger_type.startswith("imm") else False
            edge_descriptors.extend(
                shuttle_defs.path_descriptor_to_fpga_edge_descriptor(
                    shuttle_path, trigger_immediate=immediate_trigger
                )
            )
        else:
            raise ValueError("Invalid Trigger type: {}".format(trigger_type))

        for i, edge in enumerate(edge_descriptors):
            packet = DACSerialCoredevice.get_serial_shuttle_packet(*edge)
            upper, lower = packet[1:]
            # effectively int64(upper,lower). << was failing
            data = np.int64((upper * 2 ** 32) | lower)
            _LOGGER.debug("Data packet[%i]: %s; converted = %x", i, packet, data)
            data_list.append(data)

        edge_times_seconds = [path_edge.edge.total_time for path_edge in shuttle_path]
        edge_times_mu = [self.core.seconds_to_mu(t) for t in edge_times_seconds]
        assert len(data_list) == len(shuttle_path)
        assert len(edge_times_mu) == len(shuttle_path)
        return data_list, edge_times_mu

    @kernel
    def _output_int32_over_serial(self, val: TInt32, num_bits: TInt32 = 32) -> TInt32:
        """Output to the Sandia DAC via SPI.

        Requires a SPI line from the gateware. Only use MOSI line, other signals can
        be any pin b/c unused. Needed b/c toggling pins w/ TTL.on()/off()
        computes too slow, so underflow errors.

        Args:
            val (TInt32): value to output over serial. Should be right-aligned,
                MSB on left
            num_bits (TInt32, optional): Defaults to 32. number of bits of val to output
                over serial connection.

        Returns:
            TInt32: number of bits output over serial. Useful for counting outputs.
        """
        # if _ARTIQ_VERSION >= 4:
        self.serial_out.set_config_mu(
            self._SPI_CONFIG, num_bits, self.spi_clk_div_mu, 0
        )
        self.serial_out.write(val << (32 - num_bits))
        # cancel any delay b/w calls
        delay_mu(-26 * self.serial_out.ref_period_mu)  # empirical.
        return num_bits

    @kernel
    def output_command_and_data(
        self,
        command: TInt32,
        data: TInt64,
        preset: TBool = False,
        trigger: TBool = False,
    ) -> TNone:
        """Output a command and pre-computed packet data via serial to 100x DAC.

        data is the packet data. If preset is True, rewinds the timeline so output
        is complete by current time cursor.

        NOTE: trigger is not compatible with preset. Trigger will execute after
        presetting the output, and then execute. So not perfectly parallel timing.
        """
        num_outputs = 0
        start_time = now_mu()
        if preset:
            # rewind time cursor so all changes happen in past, are ready now()
            delay_mu(-(74 * self.data_period_mu + 5 * self.serial_out.ref_period_mu))
        num_outputs += self._output_int32_over_serial(0x100 | command, 9)
        num_outputs += self._output_int32_over_serial(np.int32(data >> 32))
        num_outputs += self._output_int32_over_serial(np.int32(data))
        # stop bit, then reset line to 0
        num_outputs += self._output_int32_over_serial(0b10, 2)

        if preset:
            assert now_mu() <= start_time
            at_mu(start_time)
        # check packet was sized properly.
        assert num_outputs == 75

        if trigger:
            # TODO: DOES NOT work with **PRESET**
            self.trigger_shuttling(preset)

    @kernel
    def shuttle_path_sync(
        self,
        data_packets: TList(TInt64),
        delay_times_mu: TList(TInt64),
        auto_trigger: TBool = True,
        preset: TBool = False,
    ):
        """Shuttle ion along specified path sequentially and as fast as possible.

        Args:
            data_packets (TList(TInt64)): Data packets describing each shuttling
                edge. Should be pre-generated on the host in :meth:`prepare` by
                :meth:`path_to_data`[0]
            delay_times_mu (TList(TInt64)): Machine units denoting how long to
                wait before outputting the next edge. This should be AT LEAST
                the amount of time that each edge takes to output, as returned
                by :meth:`path_to_data`[1] or :attr:`euriqabackend.devices.
                sandia_opalkelly.dac.ShuttlingDefinition.ShuttleEdge.total_time`
            auto_trigger (TBool, optional): Defaults to True. Whether the DAC
                shuttle edge should be auto-triggered as soon as serial edge
                is sent. This shouldn't be needed if you used "immediate"
                trigger in :meth:`path_to_data`, and can save some (small)
                amount of time.
            preset (TBool, optional): Defaults to False. If shuttling should be
                retroactive (happen in past), so that it is finished at current
                time cursor.
        """
        start_time = now_mu()
        path_time_mu = 0
        shuttle_cmd = _DACSerialCommand.shuttle_edge_from_lookup
        if len(delay_times_mu) != len(data_packets):
            raise ValueError("Properly-sized delay times array not provided")

        # calculate total delay time for preset, including trigger
        if preset:
            single_output_cmd_time_mu = self.min_wait_time_mu
            if auto_trigger and not preset:
                single_output_cmd_time_mu += self.core.seconds_to_mu(
                    self.trigger_time_mu
                )
            for t in delay_times_mu:
                path_time_mu += t
            path_time_mu += len(data_packets) * single_output_cmd_time_mu

            delay_mu(-path_time_mu)

        for i in range(len(data_packets)):
            self.output_command_and_data(
                shuttle_cmd, data_packets[i], trigger=auto_trigger
            )
            delay_mu(delay_times_mu[i])

        if preset:
            assert now_mu() <= start_time
            at_mu(start_time)

"""Test that you can move an ion around using the Sandia DAC.

Assumes ion is pre-loaded, and tries to shuttle it and move it.
Requires manual viewing on camera, no detection implemented.

Changes the potential well (via DC voltage) of an ion trap to move an ion around.
"""
import logging
import os

import artiq
import numpy
from artiq.language.core import delay
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.units import ms
from artiq.language.units import s
from artiq.language.units import us

import euriqabackend.voltage_solutions.translated_solutions as voltage_solutions
from euriqabackend.coredevice.sandia_dac_core import _DACSerialCommand

# from artiq.language.units import MHz
# from euriqabackend.coredevice.ad9912 import freq_to_mu


_LOGGER = logging.getLogger(__name__)


class SandiaDACSyncShuttler(artiq.language.environment.EnvExperiment):
    """Sandia DAC ion movement test.

    ARTIQ Experiment to move an ion around by changing DC voltages.

    Example Test 1:
        1. Load 1 ion with Sandia.
        2. Set voltages with Sandia.
        3. Load ARTIQ on box 2 using same voltage/graph/adjustments.
        4. Test 1: Move ion b/w nearby locations: i.e. line 7 -> line 10
        5. Test 2: shuttle ion b/w loading & quantum (or loading -> junction, wait 1s, & back)

    Test 2 (full ARTIQ):
        1. Load DAC with voltages & file
        2. Load camera & view
        3. "Direct shuttle" (hardware timed) from line 5->10 ish, every 1s (repeat 10x)
        4. Verify movement
        5. Shuttle along one edge & back
    """

    kernel_invariants = {
        "dac_pc",
        "dac_realtime",
        "load_to_quantum_path",
        "quantum_to_load_path",
        "shuttle_command",
        "data_edge1",
        "data_edge2",
    }
    # pylint: disable=no-member

    def build(self):
        """Get any ARTIQ arguments."""
        self.setattr_device("core")
        self.dac_pc = self.get_device("dac_pc_interface")
        self.dac_realtime = self.get_device("realtime_sandia_dac")

        # basic cooling devices. For trapping ion in loading region
        # self.setattr_device("oeb")
        # self.dds_cool_1 = self.get_device("dds6")
        # self.dds_cool_2 = self.get_device("dds7")
        # self.setattr_device("dds6_switch")
        # self.setattr_device("dds7_switch")
        # self.dds_reset = self.get_device("reset67")
        # self.cool_frequencies = [freq_to_mu(i * MHz) for i in [180, 210]]
        # self.setattr_device("power_cool_1")
        # self.setattr_device("power_cool_2")

    def prepare(self):
        """Confirm experiment is manually set up correctly with operator."""
        # TODO: fill in file paths
        base_voltages_path = voltage_solutions.__path__._path[0]

        self.dac_pc.load_pin_map_file(
            os.path.join(base_voltages_path, "EURIQA_socket_map.txt")
        )
        self.dac_pc.load_voltage_file(
            os.path.join(base_voltages_path, "tilted_merge_relaxedLoading_PAD.txt")
        )
        self.dac_pc.load_global_adjust_file(
            os.path.join(base_voltages_path, "EURIQA_MHz_units_plus_load_duke.txt")
        )
        self.dac_pc.load_shuttling_definitions_file(
            os.path.join(
                base_voltages_path, "tilted_merge_relaxedLoading_PAD_shuttling.xml"
            )
        )
        _LOGGER.debug("Done setting up Sandia DAC x100")
        # self.manual_setup()

    @host_only
    def run(self):
        """Move the ion around visibly on a camera."""
        self.dac_pc.apply_line_async(7)
        self.dac_pc.send_voltage_lines_to_fpga()
        self.dac_pc.send_shuttling_lookup_table()
        self.dac_pc.apply_line_async(7)

        # Calculate paths
        path1 = self.dac_pc.get_shuttle_path("center", "loading")
        _LOGGER.debug("path1: %s", path1)
        self.quantum_to_load_path, self.q_to_load_times = self.dac_realtime.path_to_data(
            path1, "imm"
        )
        _LOGGER.debug("Path 1 (FPGA): %s", self.quantum_to_load_path)

        self.load_to_quantum_path, self.load_to_q_times = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path("loading", "center"), "imm"
        )
        _LOGGER.debug("Path2 (FPGA): %s", self.load_to_quantum_path)

        self.shuttle_command = _DACSerialCommand.shuttle_edge_from_lookup

        # calculate direct shuttling for specific lines.
        # line_to_mem = self.dac_pc.file_line_num_to_address_dict
        # start_mem, end_mem = line_to_mem[849], line_to_mem[650]
        # _LOGGER.info("Line 650 = mem %i, Line 849 = mem %i", end_mem, start_mem)
        # edge1 = self.dac_realtime.get_serial_output_memory_packet(
        #     start_mem, end_mem, False
        # )
        # self.data_edge1 = numpy.int64(edge1[1] << 32 | edge1[2])
        # edge2 = self.dac_realtime.get_serial_output_memory_packet(
        #     end_mem, start_mem, False
        # )
        # self.data_edge2 = numpy.int64(edge2[1] << 32 | edge2[2])
        # _LOGGER.info("Edge1data: %x, edge2data: %x", self.data_edge1, self.data_edge2)

        # JUMP: TO LOADING (for jumping to load solution)
        self.jump_to_load_path, timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="LoadingJump", to_line_or_name="Loading"
            ),
            "immediate",
        )
        self.jump_to_load_timing = [self.core.seconds_to_mu(t) for t in timing]

        # execute test
        # self.initialize_shutters()
        for i in range(1):
            _LOGGER.info("Starting Sandia DAC Shuttle loop #%i", i)

            self.test_self_shuttle()
            self.test_shuttle_sequence()  # WORKS
            self.test_serial_shuttle_command()  # WORKS
            # doesn't work. maybe issue with Verilog code
            # self.test_serial_output_memory_command()
            self.test_idle_count_command()  # WORKS. not sure what it does on DAC
            self.test_shuttle_sync_command()

    @kernel
    def test_self_shuttle(self):
        self.core.reset()
        self.core.break_realtime()
        self.dac_realtime.shuttle_path_sync(
            self.jump_to_load_path, self.jump_to_load_timing
        )

    @kernel
    def test_idle_count_command(self):
        """Test that the `set_idle_count` serial command doesn't crash FPGA.

        This doesn't crash, but working hasn't been confirmed.
        """
        self.core.break_realtime()
        self.dac_realtime.output_command_and_data(
            _DACSerialCommand.set_idle_counts, 500
        )

    @kernel
    def test_output_shuttling_edge(self):
        """Output a single shuttling edge from the shuttling path."""
        self.core.reset()
        self.core.break_realtime()
        delay(100 * ms)
        self.dac_realtime.output_command_and_data(
            self.shuttle_command, self.load_to_quantum_path[0]
        )
        self.dac_realtime.trigger_shuttling()
        delay(100 * ms)
        self.dac_realtime.output_command_and_data(
            self.shuttle_command, self.quantum_to_load_path[-1]
        )
        self.dac_realtime.trigger_shuttling()
        delay(100 * ms)

    @kernel
    def test_shuttle_sequence(self):
        """Test outputting the entire shuttling sequence from a shuttling path."""
        self.core.break_realtime()
        for _ in range(10):
            # shuttle quantum to load region
            for edge_data in self.quantum_to_load_path:
                # for i in range(len(self.quantum_to_load_path)):
                self.dac_realtime.output_command_and_data(
                    self.shuttle_command, edge_data
                )
                # self.dac_realtime.output_command_and_data(
                #   self.shuttle_command, self.quantum_to_load_path[i]
                # )
                delay(1 * us)
                self.dac_realtime.trigger_shuttling()
                # delay_mu(self.dac_realtime.min_wait_time_mu)
                delay(25 * ms)  # delay to wait for edge output

            delay(2 * s)  # long to be visible on camera

            # shuttle load to quantum region
            for edge_data in self.load_to_quantum_path:
                self.dac_realtime.output_command_and_data(
                    self.shuttle_command, edge_data
                )
                delay(1 * us)
                self.dac_realtime.trigger_shuttling()
                # delay_mu(self.dac_realtime.min_wait_time_mu)
                delay(25 * ms)  # wait for edge to finish outputting

            delay(2 * s)

    @kernel
    def test_serial_output_memory_command(self):
        """Test serial_output_memory(): output sequential addresses from memory.

        NOT WORKING. Corrupts FPGA memory when run. Might be due to memory address vs
        memory line, or something. Also seems to hang FPGA so it doesn't respond.
        """
        self.core.break_realtime()
        for _ in range(10):
            self.core.break_realtime()
            self.dac_realtime.output_command_and_data(
                _DACSerialCommand.direct_lines_wait_trigger, self.data_edge1
            )
            self.dac_realtime.trigger_shuttling()
            delay(1 * s)

            self.core.break_realtime()
            self.dac_realtime.output_command_and_data(
                _DACSerialCommand.direct_lines_wait_trigger, self.data_edge2
            )
            self.dac_realtime.trigger_shuttling()
            delay(1 * s)

    @kernel
    def test_serial_shuttle_command(self):
        """Test serial_shuttle(): execute shuttling edge from FPGA lookup table."""
        self.core.break_realtime()
        for _ in range(10):
            self.dac_realtime.output_command_and_data(
                _DACSerialCommand.shuttle_edge_from_lookup,
                numpy.int64(0x0000000200000005),
            )
            # self.dac_realtime.serial_shuttle(
            #     5, reverse_edge=False, trigger_immediate=False
            # )
            self.dac_realtime.trigger_shuttling()
            delay(1 * s)

            self.dac_realtime.output_command_and_data(
                _DACSerialCommand.shuttle_edge_from_lookup,
                numpy.int64(0x0000000000000005),
            )
            # self.dac_realtime.serial_shuttle(
            #     5, reverse_edge=True, trigger_immediate=False
            # )
            self.dac_realtime.trigger_shuttling()
            delay(1 * s)

    @kernel
    def test_shuttle_sync_command(self):
        """Test the shuttle_path_sync command works as expected.

        Should simplify shuttling & its timing.
        """
        self.core.break_realtime()
        # test standard shuttling.
        for _ in range(10):
            self.dac_realtime.shuttle_path_sync(
                self.load_to_quantum_path, self.load_to_q_times
            )
            delay(0.5 * s)
            self.dac_realtime.shuttle_path_sync(
                self.quantum_to_load_path, self.q_to_load_times
            )
            delay(0.5 * s)

        # test preset
        self.dac_realtime.shuttle_path_sync(
            self.load_to_quantum_path, self.load_to_q_times, preset=True
        )
        delay(0.5 * s)
        self.dac_realtime.shuttle_path_sync(
            self.quantum_to_load_path, self.q_to_load_times, preset=True
        )

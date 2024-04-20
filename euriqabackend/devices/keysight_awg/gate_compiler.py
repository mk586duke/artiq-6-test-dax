"""The GateCompiler class is used by the RFCompiler to compile lists of gates.  It can
either compile a single sequence of gates, which is its normal mode of operation, or it
can implement circuit scan mode, in which multiple circuits are generated and run in
parallel.  In circuit scan mode, ARTIQ scans through the various circuits using the
ScanInt lines, just as it would scan through, for example, the length of a Rabi scan or
the phase of an analysis pulse.

This class will also generate waveform files based on these gate arrays.
"""
import logging
import time

import euriqabackend.devices.keysight_awg.common_types as common_types
import euriqabackend.devices.keysight_awg.RFCompiler as rfc
from euriqabackend.devices.keysight_awg import gate
from euriqabackend.devices.keysight_awg import sequence as seq


_LOGGER = logging.getLogger(__name__)


class GateList:
    """The GateList class is contains a single sequence of gates, which it compiles into
    a single sequence of waveforms. It also calls on its sequence object to write itself
    to a series of waveform files on disk.

    This class is used by the GateCompiler object, which can construct either one
    GateList object or an array of GateList objects.  We would generate an array of
    GateList objects to implement circuit scan mode, wherein ARTIQ scans through the
    various circuits using the ScanInt lines, just as it would scan through, for
    example, the length of a Rabi scan or the phase of an analysis pulse.
    """

    def __init__(self, rf_compiler: rfc.RFCompiler):

        self.rf_compiler = rf_compiler
        self.gate_array = []
        self.N_gates = 0
        self.suppress_circuit_scan_array = []
        self.sequence = None
        self.compiled = False
        self.total_duration = 0
        self.timestep_times = []

    def add(self, gate_to_add: gate.Gate, suppress_circuit_scan: bool = False):
        """This function adds one gate to the gate array.

        Args:
            gate_to_add: The gate to add
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        # if self.compiled:
        #     _LOGGER.error(
        #         "Error: Cannot add an additional gate to a circuit
        #           that has already been compiled")
        #     raise Exception(
        #         "Error: Cannot add an additional gate to a circuit
        #           that has already been compiled")

        self.gate_array.append(gate_to_add)
        self.suppress_circuit_scan_array.append(suppress_circuit_scan)
        self.N_gates += 1

    def generate_sequence(self):

        # First, prepend a reference pulse to the experiment,
        # but only if this is the first time compiling the sequence
        if not self.compiled:
            self.gate_array = [
                gate.ReferencePulse(self.rf_compiler.physical_params)
            ] + self.gate_array
            self.suppress_circuit_scan_array = [True] + self.suppress_circuit_scan_array
            self.N_gates += 1

        # Generate the waveform lists in each gate
        for g in self.gate_array:
            g.set_scan()
            g.compile_waveform()

        # Accumulate phase gates, write phase gate values to waveforms,
        # and remove phase gates
        prev_phase_gates = [0.0] * 33
        non_phase_gates = []
        for i, g in enumerate(self.gate_array):
            if g.gate_type == common_types.GateType.Phase:
                prev_phase_gates[g.slot_array[0]] += g.phase
                # Must remove RZs from this array too. Set them to None and remove later
                self.suppress_circuit_scan_array[i] = None
            else:
                prev_phase_gates = g.assign_phase_gates(prev_phase_gates)
                non_phase_gates.append(g)
        self.gate_array = non_phase_gates

        # Filter out all the Nones we set above
        self.suppress_circuit_scan_array = list(filter(lambda x: x is not None,self.suppress_circuit_scan_array))

        # Extract the various arrays from the list of gates
        wav_array = [x.wav_array for x in self.gate_array]
        slot_array = [x.slot_array for x in self.gate_array]
        twoQ_gate_array = [x.twoQ_gate for x in self.gate_array]
        wait_after_array = [x.wait_after for x in self.gate_array]

        # The LO frequencies are f_carrier for slot 0 (the global beam)
        # and f_ind for slots 1-32 (the individual beams)
        PA_dark_freqs = [self.rf_compiler.physical_params.f_carrier] + [
            self.rf_compiler.physical_params.f_ind
        ] * 32

        # Compile and phase-forward the complete experimental sequence
        self.sequence = seq.Sequence(
            self.rf_compiler.calibration,
            wav_array,
            slot_array,
            twoQ_gate_array,
            self.rf_compiler.waveform_dir + self.rf_compiler.name,
            PA_dark_freqs,
            wait_after_array=wait_after_array,
            wait_after_time=self.rf_compiler.wait_after_time,
        )

        self.sequence.compile()
        self.total_duration, self.timestep_times = self.sequence.phase_forward()

        # Now that we know the total duration of the sequence,
        # we do a bit of a HACK to set the reference pulse length
        # to this value plus some padding.
        # We do this by modifying the length of the first waveform output on Ch D as
        # well as the duration of the first (and only) segment in that waveform.
        # This only works because we just prepended the ReferencePulse gate
        # during this function, so we know the location of of the reference pulse.
        monitor_padding = 10
        self.sequence.wav_array[0][3].length = (
            self.sequence.total_duration + monitor_padding
        )
        self.sequence.wav_array[0][3].segments[0].duration = (
            self.sequence.total_duration + monitor_padding
        )

        self.compiled = True

        return self.total_duration, self.timestep_times

    def write_waveforms(self, circuit_scan: bool, circuit_index: int):

        # Load the calibration parameters from disk into memory
        self.rf_compiler.calibration.load_params()

        # Write the experimental sequence to disk
        circuit_indices = (
            [-1 if scs else circuit_index for scs in self.suppress_circuit_scan_array]
            if circuit_scan
            else [-1] * self.N_gates
        )
        # print(circuit_indices)
        return self.sequence.write(circuit_indices)


class GateCompiler:
    """The GateCompiler class is used by the RFCompiler to compile lists of gates.

    It can either compile a single sequence of gates, which is its normal mode of
    operation, or it can implement circuit scan mode, in which multiple circuits are
    generated and run in parallel.  In circuit scan mode, ARTIQ scans through the
    various circuits using the ScanInt lines, just as it would scan through, for
    example, the length of a Rabi scan or the phase of an analysis pulse.

    This class will also generate waveform files based on these gate arrays.
    """

    def __init__(self, rf_compiler: rfc.RFCompiler):

        self.rf_compiler = rf_compiler
        self.gate_list_array = []
        self.circuit_scan = False

    def clear_gate_array(self):
        """Ths function clears the gate array."""

        self.gate_list_array = []
        self.circuit_scan = False

    def add(
        self,
        gate_to_add: gate.Gate,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This function adds one gate to the specified gate list object.

        Args:
            gate_to_add: The gate to add
            circuit_index: The index of the gate list to which this gate will be added
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        N_gate_lists = len(self.gate_list_array)

        # First, make sure that we've initialized enough gate lists
        if N_gate_lists - 1 < circuit_index:
            for _ in range(circuit_index + 1 - N_gate_lists):
                self.gate_list_array.append(GateList(self.rf_compiler))

        # Add the gate to the specified gate list
        self.gate_list_array[circuit_index].add(gate_to_add, suppress_circuit_scan)

        self.circuit_scan = len(self.gate_list_array) > 1

    def compile_circuit(self, circuit_index: int):
        """This function enables you to compile one circuit at a time.

        This is useful if, for instance, you want to scan through circuits
        using different values of physical constants.
        We make sure that each circuit is compiled down to a
        sequence of waveforms (which have not yet been written to disk) before changing
        the physical constants and generating the next circuit.

        Args:
            circuit_index: The index of the circuit to be compiled
        """

        self.gate_list_array[circuit_index].generate_sequence()

    def fill_in_blank_gates(self):
        """This function fills in shorter gate lists so that all gate lists are the same
        length.

        We're going to avoid using this for now.
        Instead, we'll explicitly insert blank gates in place of actual gates
        when we construct the circuits.  This is necessary when we, for example,
        apply a series of XX gates followed by analysis SK1 pulses.
        The blank filler gates need to come between the XX gates and the SK1 pulses,
        not at the end of the gate array.
        """

        list_lengths = [gl.N_gates for gl in self.gate_list_array]
        _LOGGER.debug("List lengths in gate_list_array: %s", list_lengths)
        longest_index = list_lengths.index(max(list_lengths))

        for i, gl in enumerate(self.gate_list_array):
            for j in range(list_lengths[i], list_lengths[longest_index]):
                gl.add(
                    gate.Blank(
                        physical_params=self.rf_compiler.physical_params,
                        slots=self.gate_list_array[longest_index].sequence.slot_array[
                            j
                        ],
                        twoQ_gate=self.gate_list_array[
                            longest_index
                        ].sequence.twoQ_gate_array[j],
                        wait_after=self.gate_list_array[
                            longest_index
                        ].sequence.wait_after_array[j],
                    )
                )
            self.compile_circuit(i)

    def check_gate_list_consistency(self):
        """This function checks that all of the gate list objects are consistent.

        In order to enable us to scan through the different waveform sequences generated
        by the different gate list objects, all of the sequences need to have certain
        characteristics in common.  For instance, the 1Q and 2Q gate lengths need to be
        consistent across all gate lists, and each gate within a given gate list needs
        to be the same type (either 1Q or 2Q) and be directed to the same slots as the
        corresponding gates in other gate lists."""
        # for i in range(len(self.gate_list_array)):
        #     print(self.gate_list_array[i].N_gates)

        for i in range(len(self.gate_list_array)):
            if self.gate_list_array[i].sequence.N_scan_values > 0:
                _LOGGER.error("Waveforms cannot be scanned in parallel circuit mode")
                raise Exception("Waveforms cannot be scanned in parallel circuit mode")

        for i in range(len(self.gate_list_array) - 1):
            if (
                self.gate_list_array[i].sequence.slot_array
                != self.gate_list_array[i + 1].sequence.slot_array
            ):
                _LOGGER.error(
                    "Corresponding gates in parallel circuit mode must be "
                    "assigned to the same slots"
                )
                raise Exception(
                    "Corresponding gates in parallel circuit mode must be "
                    "assigned to the same slots"
                )
            if (
                self.gate_list_array[i].sequence.twoQ_gate_array
                != self.gate_list_array[i + 1].sequence.twoQ_gate_array
            ):
                _LOGGER.error(
                    "Corresponding gates in parallel circuit mode must be the "
                    "same type (1Q vs 2Q)"
                )
                raise Exception(
                    "Corresponding gates in parallel circuit mode must be the "
                    "same type (1Q vs 2Q)"
                )
            if (
                self.gate_list_array[i].sequence.wait_after_array
                != self.gate_list_array[i + 1].sequence.wait_after_array
            ):
                _LOGGER.error("Wait times in parallel circuit mode must be equal")
                raise Exception("Wait times in parallel circuit mode must be equal")
            if (
                self.gate_list_array[i].sequence.wait_after_time
                != self.gate_list_array[i + 1].sequence.wait_after_time
            ):
                _LOGGER.error(
                    "Parallel circuits must have the same wait_after_time values"
                )
                raise Exception(
                    "Parallel circuits must have the same wait_after_time values"
                )
            if (
                self.gate_list_array[i].sequence.oneQ_length
                != self.gate_list_array[i + 1].sequence.oneQ_length
            ):
                _LOGGER.error("Parallel circuits must have the same 1Q gate length")
                raise Exception("Parallel circuits must have the same 1Q gate length")
            if (
                self.gate_list_array[i].sequence.twoQ_length
                != self.gate_list_array[i + 1].sequence.twoQ_length
            ):
                _LOGGER.error("Parallel circuits must have the same 2Q gate length")
                raise Exception("Parallel circuits must have the same 2Q gate length")
            if (
                self.gate_list_array[i].sequence.total_duration
                != self.gate_list_array[i + 1].sequence.total_duration
            ):
                _LOGGER.error("Parallel circuits must have the same total duration")
                raise Exception("Parallel circuits must have the same total duration")

    def generate_waveforms(self):

        start_time = time.time()

        total_duration = 0

        # First, we compile any circuits that have not yet been compiled
        for gl in self.gate_list_array:
            if gl.compiled:
                total_duration, timestep_times = gl.total_duration, gl.timestep_times
            else:
                total_duration, timestep_times = gl.generate_sequence()
        print("Experiment duration: {0} us".format(total_duration))
        _LOGGER.info("Experiment duration: %.3f us", total_duration)

        # If we're in circuit scan mode, check that the various circuits
        # are consistent with each other
        if self.circuit_scan:
            # self.fill_in_blank_gates()  See note in function description
            # about why we're not using this.
            self.check_gate_list_consistency()

        # Finally, we write all circuits to disk
        N_files_written = 0
        total_data_written = 0
        for i, gl in enumerate(self.gate_list_array):
            write_result = gl.write_waveforms(self.circuit_scan, i)
            N_files_written += write_result[0]
            total_data_written += write_result[1]

        result_string = "Total of {0} us of waveforms written to {1} files".format(
            total_data_written / 1000.0, N_files_written
        )
        _LOGGER.debug(result_string)

        end_time = time.time()
        _LOGGER.debug(
            "Compilation time elapsed: {0:.3f} s".format(end_time - start_time)
        )

        self.clear_gate_array()

        return total_duration,timestep_times

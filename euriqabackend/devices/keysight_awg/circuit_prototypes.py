import typing

import numpy as np

from euriqabackend.devices.keysight_awg import circuit_interpreter as ci
from euriqabackend.devices.keysight_awg import physical_parameters as pp
from euriqabackend.devices.keysight_awg import RFCompiler as rfc


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector."""
    return np.flip(np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8))


class CircuitPrototypes:
    def __init__(
        self,
        RFCompiler: rfc.RFCompiler,
        physical_params: pp.PhysicalParams,
        N_qubits_present: int,
        N_qubits_used: int,
        qubit_offset: int = 0,
        use_SK1_AM: bool = True,
        print_circuit: bool = False,
        print_gate_list: bool = False,
    ):

        self.N_qubits_present = N_qubits_present
        self.N_qubits_used = N_qubits_used
        self.qubit_offset = qubit_offset
        self.use_SK1_AM = use_SK1_AM
        self.print_circuit = print_circuit
        self.print_gate_list = print_gate_list

        self.circuit_interpreter = ci.CircuitInterpreter(RFCompiler, physical_params)

    def prep_in_X_basis(
        self,
        prep_state: int,
        qubits_to_exclude: typing.List[int] = None,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
        suppress_print: bool = False,
    ):
        """This circuit prepares all qubits that we are using, with the exception of
        those listed in qubits_to_exclude, in the X basis by applying a pi/2 pulse along
        +Y or -Y.  It prepares them in either +X or -X according to the bitwise
        representation of the prep_state int, which enables us to scan many initial
        states of the circuit.

        Args:
            prep_state: An integer whose bitwise representation determines
                whether the qubit is prepared in +X or -X
            qubits_to_exclude: A list of qubits (out of our list of used qubits)
                to exclude from rotation
            circuit_index: The circuit to which these gates should be appended,
                for use in circuit_scan mode
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding (i.e.,
                only one waveform file will be written)
            suppress_print: Suppresses printing the circuit and causes the
                return string to be empty
        """

        gate_name = "Prepare in X basis"

        builder = ci.CircuitBuilder(
            self.N_qubits_present, qubit_offset=self.qubit_offset
        )

        if qubits_to_exclude is None:
            qubits_to_exclude = list()

        qubits_to_initialize = list(range(self.N_qubits_used))
        for i in sorted(qubits_to_exclude):
            del qubits_to_initialize[i]

        prep_state_binary = bin_array(prep_state, len(qubits_to_initialize))

        for i, qi in enumerate(qubits_to_initialize):
            rot_phase = +1 if prep_state_binary[i] == 1 else -1
            builder.RY(qi, rot_phase)

        circuit_str = ""
        if (self.print_circuit or self.print_gate_list) and not suppress_print:
            circuit_str += "\n\n" + gate_name + ":\n"
            print("\n" + gate_name + ":")
        circuit_str += self.circuit_interpreter.compile_circuit(
            circuit=builder.get_circuit(),
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
            use_SK1_AM=self.use_SK1_AM,
            print_circuit=(self.print_circuit and not suppress_print),
            print_gate_list=(self.print_gate_list and not suppress_print),
        )

        return "" if suppress_print else circuit_str

    def single_stabilizer_readout(
        self,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
        suppress_print: bool = False,
    ):
        """This circuit transfers the XXXXXX stabilizer from the six data qubits to one
        ancilla (qubit 3)

        Args:
            circuit_index: The circuit to which these gates should be appended,
                for use in circuit_scan mode
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding (i.e.,
                only one waveform file will be written)
            suppress_print: Suppresses printing the circuit and causes the
                return string to be empty
        """

        gate_name = "Single stabilizer readout"

        builder = ci.CircuitBuilder(
            self.N_qubits_present, qubit_offset=self.qubit_offset
        )

        builder.XX([0, 3], +1)
        builder.XX([2, 3], -1)
        builder.XX([1, 3], +1)
        builder.XX([4, 3], -1)
        builder.XX([6, 3], +1)
        builder.XX([5, 3], -1)

        circuit_str = ""
        if (self.print_circuit or self.print_gate_list) and not suppress_print:
            circuit_str += "\n\n" + gate_name + ":\n"
            print("\n" + gate_name + ":")
        circuit_str += self.circuit_interpreter.compile_circuit(
            circuit=builder.get_circuit(),
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
            use_SK1_AM=self.use_SK1_AM,
            print_circuit=(self.print_circuit and not suppress_print),
            print_gate_list=(self.print_gate_list and not suppress_print),
        )

        return "" if suppress_print else circuit_str

    def readout_X_basis(
        self,
        qubits_to_exclude: typing.List[int] = None,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
        suppress_print: bool = False,
    ):
        """This circuit rotates all qubits from the X basis to the Z basis to be read
        out.

        Args:
            qubits_to_exclude: A list of qubits (out of our list of used qubits)
                to exclude from rotation
            circuit_index: The circuit to which these gates should be appended,
                for use in circuit_scan mode
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding (i.e.,
                only one waveform file will be written)
            suppress_print: Suppresses printing the circuit and causes the
                return string to be empty
        """

        gate_name = "Readout in X basis"

        builder = ci.CircuitBuilder(
            self.N_qubits_present, qubit_offset=self.qubit_offset
        )

        if qubits_to_exclude is None:
            qubits_to_exclude = list()

        qubits_to_read_out = list(range(self.N_qubits_used))
        for i in sorted(qubits_to_exclude):
            del qubits_to_read_out[i]

        for i in qubits_to_read_out:
            builder.RY(i, -1)

        circuit_str = ""
        if (self.print_circuit or self.print_gate_list) and not suppress_print:
            circuit_str += "\n\n" + gate_name + ":\n"
            print("\n" + gate_name + ":")
        circuit_str += self.circuit_interpreter.compile_circuit(
            circuit=builder.get_circuit(),
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
            use_SK1_AM=self.use_SK1_AM,
            print_circuit=(self.print_circuit and not suppress_print),
            print_gate_list=(self.print_gate_list and not suppress_print),
        )

        return "" if suppress_print else circuit_str

    def global_phase_shift(
        self,
        phase_shift: float,
        qubits_to_exclude: typing.List[int] = None,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
        suppress_print: bool = False,
    ):
        """This circuit applied a phase shift to all qubits.

        Args:
            phase_shift: The phase shift to apply to the qubits
            qubits_to_exclude: A list of qubits (out of our list of used qubits)
                to exclude from rotation
            circuit_index: The circuit to which these gates should be appended,
                for use in circuit_scan mode
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding (i.e.,
                only one waveform file will be written)
            suppress_print: Suppresses printing the circuit and causes the
                return string to be empty
        """

        gate_name = "Phase shift"

        builder = ci.CircuitBuilder(
            self.N_qubits_present, qubit_offset=self.qubit_offset
        )

        if qubits_to_exclude is None:
            qubits_to_exclude = list()

        qubits_to_read_out = list(range(self.N_qubits_used))
        for i in sorted(qubits_to_exclude):
            del qubits_to_read_out[i]

        for i in qubits_to_read_out:
            builder.RZ(i, +1, phase_shift)

        circuit_str = ""
        if (self.print_circuit or self.print_gate_list) and not suppress_print:
            circuit_str += "\n\n" + gate_name + ":\n"
            print("\n" + gate_name + ":")
        circuit_str += self.circuit_interpreter.compile_circuit(
            circuit=builder.get_circuit(),
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
            use_SK1_AM=self.use_SK1_AM,
            print_circuit=(self.print_circuit and not suppress_print),
            print_gate_list=(self.print_gate_list and not suppress_print),
        )

        return "" if suppress_print else circuit_str

    def phase_shifts(
        self,
        phase_shifts: typing.List[float],
        qubits_to_exclude: typing.List[int] = None,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
        suppress_print: bool = False,
    ):
        """This circuit applied a phase shift to all qubits.

        Args:
            phase_shifts: The phase shifts to apply to the qubits
            qubits_to_exclude: A list of qubits (out of our list of used qubits)
                to exclude from rotation
            circuit_index: The circuit to which these gates should be appended,
                for use in circuit_scan mode
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding (i.e.,
                only one waveform file will be written)
            suppress_print: Suppresses printing the circuit and causes the
                return string to be empty
        """

        gate_name = "Phase shift"

        builder = ci.CircuitBuilder(
            self.N_qubits_present, qubit_offset=self.qubit_offset
        )

        if qubits_to_exclude is None:
            qubits_to_exclude = list()

        qubits_to_read_out = list(range(self.N_qubits_used))
        for i in sorted(qubits_to_exclude):
            del qubits_to_read_out[i]

        for i, qi in enumerate(qubits_to_read_out):
            builder.RZ(qi, +1, phase_shifts[i])

        circuit_str = ""
        if (self.print_circuit or self.print_gate_list) and not suppress_print:
            circuit_str += "\n\n" + gate_name + ":\n"
            print("\n" + gate_name + ":")
        circuit_str += self.circuit_interpreter.compile_circuit(
            circuit=builder.get_circuit(),
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
            use_SK1_AM=self.use_SK1_AM,
            print_circuit=(self.print_circuit and not suppress_print),
            print_gate_list=(self.print_gate_list and not suppress_print),
        )

        return "" if suppress_print else circuit_str

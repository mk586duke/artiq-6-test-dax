"""Convert Cirq circuits to the equivalent Qiskit circuit."""
import math
import logging
import typing

import cirq
import packaging.version as version
import qiskit
import qiskit.circuit as q_circ


_ALLOWED_QISKIT_TERRA_VERSION_RANGE = (version.parse("0.16.0"), version.parse("0.17.4"))
_qiskit_terra_ver = version.parse(qiskit.__qiskit_version__["qiskit-terra"])
assert _qiskit_terra_ver >= _ALLOWED_QISKIT_TERRA_VERSION_RANGE[0]
assert _qiskit_terra_ver <= _ALLOWED_QISKIT_TERRA_VERSION_RANGE[1]

_ALLOWED_CIRQ_VERSION_RANGE = (version.parse("0.9.0"), version.parse("0.9.1"))
_cirq_ver = version.parse(cirq.__version__)
assert _cirq_ver >= _ALLOWED_CIRQ_VERSION_RANGE[0]
assert _cirq_ver <= _ALLOWED_CIRQ_VERSION_RANGE[1]

_LOGGER = logging.getLogger(__name__)


def _validate_qubits(
    circuit: cirq.Circuit, num_qubits: int, center_index: bool = True
) -> bool:
    """Check that all qubit indices & types are valid.

    If ``center_index==None``, then will attempt to guess based on the indices.

    If center_index, valid indices are (e.g. odd, even):
        * 4 qubits: (-2, -1,    1, 2)
        * 5 qubits: (-2, -1, 0, 1, 2)
    """
    qbs = circuit.all_qubits()
    if any(map(lambda qb: not isinstance(qb, cirq.LineQubit), qbs)):
        raise TypeError(
            f"Circuit has unsupported qubit type(s): {qbs} is not all LineQubits"
        )
    if center_index is None:
        center_index = any(map(lambda qb: qb.x < 0, qbs))
    idx_range = (
        range(num_qubits)
        if not center_index
        else range(-num_qubits // 2, math.ceil((num_qubits + 0.5) / 2))
    )
    if center_index and num_qubits % 2 == 0:
        idx_range = filter(lambda i: i != 0, idx_range)
    qb_idxs = set(idx_range)
    invalid_idxs = set(
        filter(lambda idx: idx not in qb_idxs, map(lambda qb: qb.x, qbs))
    )
    if len(invalid_idxs) > 0:
        raise ValueError(
            f"Qubit indices {invalid_idxs} not supported in {num_qubits} qb chain"
        )
    return True


def _validate_operations(
    circuit: cirq.Circuit, valid_op_func: typing.Callable[[cirq.GateOperation], bool]
):
    return all(map(valid_op_func, circuit.all_operations()))


def _exponent_gate(cirq_op: cirq.GateOperation) -> q_circ.Gate:
    qiskit_gate = _cirq_to_qiskit_gate_map[type(cirq_op.gate)]
    try:
        new_gate = qiskit_gate(cirq_op.gate.exponent * math.pi)
        assert new_gate.label is None  # false for e.g. HGate, which don't take args
        return new_gate
    except (TypeError, AssertionError):
        return qiskit_gate()


_cirq_to_qiskit_gate_map = {
    cirq.XPowGate: q_circ.library.RXGate,
    cirq.XXPowGate: q_circ.library.RXXGate,
    cirq.YPowGate: q_circ.library.RYGate,
    cirq.YYPowGate: q_circ.library.RYYGate,
    cirq.ZPowGate: q_circ.library.RZGate,
    cirq.ZZPowGate: q_circ.library.RZZGate,
    cirq.ops.pauli_gates._PauliX: q_circ.library.XGate,
    cirq.ops.pauli_gates._PauliY: q_circ.library.YGate,
    cirq.ops.pauli_gates._PauliZ: q_circ.library.ZGate,
    cirq.HPowGate: q_circ.library.HGate,
    cirq.ion.ion_gates.MSGate: q_circ.library.RXXGate,  # TODO: q_circ.MSGate deprecated
}


def _qiskit_equivalent(cirq_op: cirq.GateOperation) -> q_circ.Gate:
    if hasattr(cirq_op.gate, "exponent"):
        return _exponent_gate(cirq_op)
    else:
        raise NotImplementedError(
            f"Gate {cirq_op} conversion to Qiskit has not yet been defined"
        )


def convert_cirq_to_qiskit(
    cirq_circuit: cirq.Circuit, try_qasm_ir: bool = True, validate: bool = True, num_ions: int = 15
) -> q_circ.QuantumCircuit:
    """Convert a cirq Circuit to a roughly equivalent Qiskit circuit.

    Caveats:
      * Loses the concept of cirq "Moments"
      * Some of the converted gates might not have the exact same unitary.
        Needs more testing. E.g. cirq.XXPowGate != qiskit.RXXGate
      * Discards measurements, all qubits assumed to be measured at the end
    """
    assert isinstance(cirq_circuit, cirq.Circuit)
    assert cirq_circuit.are_all_measurements_terminal()
    _LOGGER.warning(
        "Converting Cirq -> Qiskit can cause phase mismatches in the unitary. "
        "It is experimental & temporary, use with caution!"
    )

    if try_qasm_ir:
        # Try to convert via built-in cirq methods (cirq -> QASM -> QuantumCircuit)
        # Check that the number of instructions & qubits matches
        qasm_qiskit_circ = q_circ.QuantumCircuit.from_qasm_str(cirq_circuit.to_qasm())
        if qasm_qiskit_circ.num_qubits == len(cirq_circuit.all_qubits()) and len(
            qasm_qiskit_circ
        ) == len(cirq_circuit):
            return qasm_qiskit_circ

    # Custom cirq -> qiskit conversion
    if validate:
        print(num_ions)
        assert num_ions >= len(cirq_circuit.all_qubits())

        # _validate_qubits(cirq_circuit, num_qubits=num_ions)#num_qubits=len(cirq_circuit.all_qubits()))
        # _validate_operations(
        #     cirq_circuit,
        #     lambda gate_op: type(gate_op.gate) in set(_cirq_to_qiskit_gate_map.keys()),
        # )

    qreg = q_circ.QuantumRegister(len(cirq_circuit.all_qubits()))
    qb_to_qreg = dict(zip(sorted(cirq_circuit.all_qubits()), qreg))

    def _qiskit_qubits(*qbs) -> typing.List[q_circ.Qubit]:
        return list(map(lambda qb: qb_to_qreg[qb], qbs))

    qiskit_circ = q_circ.QuantumCircuit(qreg)

    for op in cirq_circuit.all_operations():
        qiskit_circ.append(_qiskit_equivalent(op), _qiskit_qubits(*op.qubits))

    return qiskit_circ

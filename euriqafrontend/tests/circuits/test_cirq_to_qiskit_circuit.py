"""Tests conversion from Cirq -> Qiskit circuits."""
import cirq
import numpy as np
import pytest
import qiskit.circuit as q_circ
import qiskit
from qiskit import Aer

import euriqafrontend.circuits.cirq_converter as circuit_conv


@pytest.fixture
def sample_cirq_circuit() -> cirq.Circuit:
    qb0 = cirq.LineQubit(0)
    qb1 = cirq.LineQubit(1)
    c = cirq.Circuit()
    c.append(cirq.rx(np.pi / 2).on(qb0))
    c.append(cirq.X.on(qb1))
    c.append(cirq.Y.on(qb1))
    c.append(cirq.Z.on(qb0))
    c.append(cirq.H.on(qb0))
    c.append(cirq.XX(qb0, qb1))
    c.append(cirq.ms(np.pi / 4).on(qb0, qb1))
    return c


@pytest.mark.xfail(
    reason="Cirq decomposes ion gates when writing QASM",
    raises=AssertionError,
    strict=True,
)
def test_circuit_to_qasm(sample_cirq_circuit):
    circuit_qasm = sample_cirq_circuit.to_qasm()
    assert len(q_circ.QuantumCircuit.from_qasm_str(circuit_qasm)) == len(
        sample_cirq_circuit
    )


def test_cirq_circuit_to_qiskit(sample_cirq_circuit):
    qiskit_circuit = circuit_conv.convert_cirq_to_qiskit(sample_cirq_circuit)
    assert len(qiskit_circuit) == len(tuple(sample_cirq_circuit.all_operations()))
    assert len(qiskit_circuit.qubits) == len(sample_cirq_circuit.all_qubits())
    reg = qiskit_circuit._qubits[0].register

    expected_qiskit_circuit = q_circ.QuantumCircuit(reg)
    expected_qiskit_circuit.rx(0.5 * np.pi, 0)
    expected_qiskit_circuit.x(1)
    expected_qiskit_circuit.y(1)
    expected_qiskit_circuit.z(0)
    expected_qiskit_circuit.h(0)
    expected_qiskit_circuit.rxx(np.pi, 0, 1)
    expected_qiskit_circuit.rxx(0.5 * np.pi, 0, 1)  # maximally entangling @ pi/2
    assert qiskit_circuit == expected_qiskit_circuit


def test_cirq_converter_validate_qubit_failure():
    c0 = cirq.Circuit()
    qb0 = cirq.NamedQubit("q0")
    c0.append(cirq.rx(np.pi / 2).on(qb0))
    with pytest.raises(TypeError):
        circuit_conv.convert_cirq_to_qiskit(c0, try_qasm_ir=False)

    circuit_conv.convert_cirq_to_qiskit(c0, try_qasm_ir=False, validate=False)

    c1 = cirq.Circuit()
    qb1 = cirq.LineQubit(5)
    c1.append(cirq.rx(np.pi / 2).on(qb1))
    with pytest.raises(ValueError):
        circuit_conv.convert_cirq_to_qiskit(c1, try_qasm_ir=False)

    circuit_conv.convert_cirq_to_qiskit(c1, try_qasm_ir=False, validate=False)


@pytest.mark.xfail(
    reason="Phase (?) difference between Cirq & Qiskit unitaries. Magnitudes same",
    raises=AssertionError,
    strict=True,
)
def test_cirq_to_qiskit_same_unitary(sample_cirq_circuit):
    qiskit_circuit = circuit_conv.convert_cirq_to_qiskit(sample_cirq_circuit)
    backend = Aer.get_backend("unitary_simulator")
    transpiled_circuit = qiskit.transpile(qiskit_circuit, backend)
    job = backend.run(qiskit.assemble(transpiled_circuit, backend))
    qiskit_unitary = job.result().get_unitary()
    cirq_unitary = cirq.unitary(sample_cirq_circuit)
    np.testing.assert_array_almost_equal(np.abs(cirq_unitary), np.abs(qiskit_unitary))
    assert cirq.linalg.allclose_up_to_global_phase(cirq_unitary, qiskit_unitary)


@pytest.mark.parametrize(
    "input_op,qiskit_op",
    [
        (
            # Identity
            cirq.GateOperation(
                cirq.XXPowGate(exponent=0), (cirq.LineQubit(0), cirq.LineQubit(1))
            ),
            q_circ.library.RXXGate(0),
        ),
        (
            # Fully entangling gate
            cirq.GateOperation(
                cirq.XXPowGate(exponent=0), (cirq.LineQubit(0), cirq.LineQubit(1))
            ),
            q_circ.library.RXXGate(0),
        ),
        (
            # anti-diagonal matrix
            cirq.GateOperation(
                cirq.XXPowGate(exponent=1), (cirq.LineQubit(0), cirq.LineQubit(1))
            ),
            q_circ.library.RXXGate(np.pi),
        ),
        (cirq.GateOperation(cirq.X, (cirq.LineQubit(0),)), q_circ.library.XGate()),
        (cirq.GateOperation(cirq.Y, (cirq.LineQubit(0),)), q_circ.library.YGate()),
        (cirq.GateOperation(cirq.Z, (cirq.LineQubit(0),)), q_circ.library.ZGate()),
        (
            cirq.GateOperation(cirq.XPowGate(exponent=1), (cirq.LineQubit(0),)),
            q_circ.library.RXGate(np.pi),
        ),
        (
            cirq.GateOperation(cirq.XPowGate(exponent=0.5), (cirq.LineQubit(0),)),
            q_circ.library.RXGate(np.pi / 2),
        ),
        (
            cirq.GateOperation(cirq.HPowGate(exponent=1), (cirq.LineQubit(0),)),
            q_circ.library.HGate(),
        ),
        (
            cirq.GateOperation(cirq.YPowGate(exponent=1), (cirq.LineQubit(0),)),
            q_circ.library.RYGate(np.pi),
        ),
        (
            cirq.GateOperation(cirq.ZPowGate(exponent=1), (cirq.LineQubit(0),)),
            q_circ.library.RZGate(np.pi),
        ),
    ],
)
def test_cirq_common_gate_conversion(
    input_op: cirq.GateOperation, qiskit_op: q_circ.Gate
):
    """Spot check some gate conversions."""
    converted_gate = circuit_conv._qiskit_equivalent(input_op)
    assert converted_gate == qiskit_op
    assert cirq.linalg.allclose_up_to_global_phase(
        converted_gate.to_matrix(), cirq.unitary(input_op)
    )

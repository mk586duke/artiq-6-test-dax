"""Test :mod:`euriqabackend.waveforms.instruction_schedule`."""
import itertools

import pytest
import qiskit.circuit.random as q_circ_rand
import qiskit.pulse as qp
from qiskit.compiler import transpile, schedule
from pulsecompiler.qiskit.transforms.gate_barrier import BarrierBetweenGates

import euriqabackend.waveforms.instruction_schedule as eur_insts


# pylint: disable=redefined-outer-name
@pytest.fixture
def euriqa_instruction_schedule(
    qiskit_backend_with_gate_solutions,
) -> eur_insts.EURIQAInstructionScheduleMap:
    return eur_insts.EURIQAInstructionScheduleMap(
        qiskit_backend_with_gate_solutions,
        eur_insts.SINGLE_QUBIT_FUNCTION_MAP,
        eur_insts.MULTI_QUBIT_FUNCTION_MAP,
    )


def test_inst_schedule_type(rfsoc_qiskit_backend):
    assert isinstance(
        eur_insts.EURIQAInstructionScheduleMap(
            rfsoc_qiskit_backend,
            eur_insts.SINGLE_QUBIT_FUNCTION_MAP,
            eur_insts.MULTI_QUBIT_FUNCTION_MAP,
        ),
        qp.InstructionScheduleMap,
    )


@pytest.mark.backend_zero_index(True)
def test_inst_schedule_has_1q_gates(euriqa_instruction_schedule):
    # pylint: disable=protected-access
    for qb in range(euriqa_instruction_schedule._backend.configuration().n_qubits):
        euriqa_instruction_schedule.assert_has("rx", qb)
        euriqa_instruction_schedule.assert_has("rz", qb)
        euriqa_instruction_schedule.assert_has("id", qb)
        euriqa_instruction_schedule.assert_has("r", qb)
        euriqa_instruction_schedule.assert_has("ry", qb)


@pytest.mark.backend_zero_index(True)
def test_inst_schedule_has_2q_gates(
    euriqa_instruction_schedule, qiskit_backend_with_gate_solutions
):
    n_qubits = qiskit_backend_with_gate_solutions.configuration().n_qubits

    for qbs in itertools.combinations(list(range(n_qubits)), 2):
        euriqa_instruction_schedule.assert_has("rxx", qubits=qbs)


# @pytest.mark.todo
# def test_inst_schedule_gate_equality(
#     euriqa_instruction_schedule, qiskit_backend_with_gate_solutions
# ):
#     # TODO: check the gate functions
#     n_qubits = qiskit_backend_with_gate_solutions.configuration().n_qubits
#     for qb in range(n_qubits):
#         assert euriqa_instruction_schedule.get("rx", qb) ==
#         assert euriqa_instruction_schedule.get("rz", qb) ==
#         assert euriqa_instruction_schedule.assert_has("id", qb) ==

#     for qbs in itertools.combinations(list(range(n_qubits)), 2):
#         euriqa_instruction_schedule.get("rxx", qubits=qbs) ==


def test_inst_schedule_has_2q_gates_all_indices(
    euriqa_instruction_schedule, qiskit_backend_with_gate_solutions
):
    for qbs in qiskit_backend_with_gate_solutions.configuration().coupling_map:
        euriqa_instruction_schedule.assert_has("rxx", qubits=qbs)


def test_inst_schedule_has_1q_gates_all_indices(
    euriqa_instruction_schedule, qiskit_backend_with_gate_solutions
):
    for qb in qiskit_backend_with_gate_solutions.configuration().all_qubit_indices_iter:
        euriqa_instruction_schedule.assert_has("rx", qb)
        euriqa_instruction_schedule.assert_has("rz", qb)
        euriqa_instruction_schedule.assert_has("id", qb)
        euriqa_instruction_schedule.assert_has("r", qb)
        euriqa_instruction_schedule.assert_has("ry", qb)


@pytest.mark.backend_zero_index(True)
@pytest.mark.timeout(10)
def test_inst_schedule_circuit_compilation(
    euriqa_instruction_schedule, qiskit_backend_with_gate_solutions
):
    num_qubits = qiskit_backend_with_gate_solutions.configuration().num_qubits
    if num_qubits > 13:
        pytest.xfail("Gate solutions not defined for > 13 qubits (15 ions)")
    circuit = q_circ_rand.random_circuit(num_qubits, 5)
    transpiled_circuit = transpile(circuit, qiskit_backend_with_gate_solutions)
    schedule(
        BarrierBetweenGates().run_circuit(transpiled_circuit),
        qiskit_backend_with_gate_solutions,
        inst_map=euriqa_instruction_schedule,
    )

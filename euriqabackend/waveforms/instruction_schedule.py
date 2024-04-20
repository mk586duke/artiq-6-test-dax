"""Create a Qiskit ``InstructionScheduleMap`` with EURIQA's default gates.

All the gates will be partially specified functions (returning Qiskit Pulse Schedules),
to allow them to be generated as close to runtime as possible with the latest
calibration values.
"""
import functools

import numpy as np
import pulsecompiler.qiskit.backend as qbe
import pulsecompiler.qiskit.instruction_schedule as pc_inst

import euriqabackend.waveforms.single_qubit as single_qb_wf
import euriqabackend.waveforms.multi_qubit as multi_qb_wf

SINGLE_QUBIT_FUNCTION_MAP = {
    "id": single_qb_wf.id_gate,
    "r": single_qb_wf.sk1_gaussian,
    "rx": functools.partial(single_qb_wf.sk1_gaussian, phi=0),
    "ry": functools.partial(single_qb_wf.sk1_gaussian, phi=np.pi / 2),
    "rz": single_qb_wf.rz,
}
MULTI_QUBIT_FUNCTION_MAP = {
    "rxx": multi_qb_wf.xx_am_gate,
}
GATE_FUNCTION_MAP = SINGLE_QUBIT_FUNCTION_MAP.copy().update(MULTI_QUBIT_FUNCTION_MAP)


class EURIQAInstructionScheduleMap(pc_inst.IonInstructionScheduleMap):
    """Instruction Schedule Map for EURIQA ion trap system.

    Mapping between instruction types & the functions defining the pulses for
    that instruction.

    Mostly a thin wrapper around
    :class:`pulsecompiler.qiskit.instruction_schedule.IonInstructionScheduleMap`,
    but with the backend argument explicitly passed to every function.
    """

    def __init__(
        self, backend: qbe.MinimalQiskitIonBackend, one_qubit_gates, multi_qb_gates
    ):
        # fill in backend arg on functions
        one_qubit_gates_backend = {
            g: functools.partial(f, backend=backend) for g, f in one_qubit_gates.items()
        }
        two_qubit_gates_backend = {
            g: functools.partial(f, backend=backend) for g, f in multi_qb_gates.items()
        }
        super().__init__(backend, one_qubit_gates_backend, two_qubit_gates_backend)

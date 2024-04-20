# %%
import cirq
import sys
import numpy as np
# temporary workarounds b/c jaqalpaw doesn't seem to work happily w/ the jupyter shell in the nix scripts...
sys.path.insert(0, "/home/euriqa/git/euriqa-artiq")
sys.path.insert(0, "/tmp/pulsecompiler-master-mts-100cd1ae")
sys.path.insert(0, "/tmp/jaqalpaw-mirror/src")

import sipyco.pyon as pyon
import euriqafrontend.circuits.cirq_converter as cirq_conv
import qiskit.circuit.parameter as qiskit_param
cirq_circuit = cirq.Circuit()

for i in range(19):
    cirq_circuit.append(cirq.rz(0).on(cirq.LineQubit(i)))

# qb0 = cirq.LineQubit(9)
# qb1 = cirq.LineQubit(10)
# cirq_circuit.append(cirq.ms(np.pi/2).on(qb0,qb1))
# cirq_circuit.append(cirq.rx(np.pi/2).on(qb1))
num_ions = 23
qiskit_circuit = cirq_conv.convert_cirq_to_qiskit(cirq_circuit,num_ions=num_ions)

# %%
print(cirq_circuit)
print(qiskit_circuit)


# %%
import pulsecompiler.qiskit.transforms.gate_barrier as barrier
print(barrier.BarrierBetweenGates().run_circuit(qiskit_circuit))

# %%
import euriqafrontend.interactive.rfsoc.qiskit_backend as rfsoc_qiskit
from pulsecompiler.qiskit.configuration import QuickConfig
import euriqafrontend.modules.rfsoc as rfsoc_mod

rfsoc_map = rfsoc_qiskit.get_default_rfsoc_map()
master_ip = "192.168.78.152"

# qiskit_backend = rfsoc_qiskit.get_default_qiskit_backend("192.168.78.152", num_ions=num_ions, with_2q_gate_solutions=False)

qiskit_backend = rfsoc_qiskit.get_default_qiskit_backend(master_ip, num_ions)

# %%
import pulsecompiler.qiskit.transforms.gate_barrier as barrier
import qiskit.compiler as q_compile
import euriqabackend.waveforms.instruction_schedule as eur_inst_sched
import euriqabackend.waveforms.single_qubit as wf_sq

single_qubit_gates = eur_inst_sched.SINGLE_QUBIT_FUNCTION_MAP
# def simple_square_rabi_gate(ion_index, phase, backend=None):
#     return wf_sq.square_rabi_by_rabi_frequency(ion_index=17,duration: float,
#     rabi_frequency: float,
#     backend: qbe.Backend,
#     phase_insensitive: bool = False,
#     **kwargs,
# ):

# single_qubit_gates["rx"] = simple_square_rabi_gate

instruction_schedule = eur_inst_sched.EURIQAInstructionScheduleMap(
        qiskit_backend,
        single_qubit_gates,
        eur_inst_sched.MULTI_QUBIT_FUNCTION_MAP,
    )

scheduled_scan_circuits = []
for theta_val in np.linspace(0, 2 * np.pi, 11):
    circuit_copy = qiskit_circuit.copy()
    # circuit_copy.delay(1000)
    # circuit_copy.r(np.pi/2,theta_val, 8)
    circuit_copy.r(np.pi/2,theta_val, 9)
    # circuit_copy.rx(np.pi/2, theta_val,0)
    # circuit_copy.rx(np.pi/2, theta_val,1)

    # circuit_copy.rx(np.pi/2, 1)
    # circuit_copy.rx(-np.pi/2, 0)
    circuit_sequential = barrier.BarrierBetweenGates().run_circuit(circuit_copy)
    print(circuit_sequential)
    circuit_scheduled = q_compile.schedule(
        circuit_sequential, qiskit_backend, inst_map=instruction_schedule
    )
    scheduled_scan_circuits.append(circuit_scheduled)


# %%
# submit converted qiskit circuit
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit

rfsoc_submit.submit_schedule(
        scheduled_scan_circuits,
        master_ip,
        qiskit_backend,
        experiment_kwargs={
            "xlabel": "red and blue detuning",
            # "x_values": pyon.encode(w_k_vec),
            # "xlabel": "rred blue spin phase",
            # "x_values": pyon.encode(rel_phase_vec),
            "default_sync": False,
            "num_shots": 200,
            "PMT Input String": "13:16",
            "lost_ion_monitor": False,
            "schedule_transform_aom_nonlinearity": False,
            "schedule_transform_pad_schedule": True,
            "do_sbc": True,
            "priority": 0,
        })

# rfsoc_submit.submit_schedule(scheduled_scan_circuits, master_ip, qiskit_backend, experiment_kwargs={"lost_ion_monitor": False})

# %%
import cirq
import sys
import numpy as np
# temporary workarounds b/c jaqalpaw doesn't seem to work happily w/ the jupyter shell in the nix scripts...
sys.path.insert(0, "/home/euriqa/git/euriqa-artiq")
sys.path.insert(0, "/tmp/pulsecompiler-master-mts-100cd1ae")
sys.path.insert(0, "/tmp/jaqalpaw-mirror/src")

import euriqafrontend.circuits.cirq_converter as cirq_conv
import qiskit.circuit.parameter as qiskit_param
import qiskit

cirq_circuit = cirq.Circuit()
qb0 = cirq.LineQubit(-1)
qb1 = cirq.LineQubit(1)
cirq_circuit.append(cirq.ms(np.pi/4).on(qb0, qb1))

qiskit_circuit = cirq_conv.convert_cirq_to_qiskit(cirq_circuit)
mycircuit = qiskit.QuantumCircuit(7,7)
mycircuit.rxx(np.pi/2,-1,1)

# %%
#print(cirq_circuit)
print(mycircuit)#qiskit_circuit)


# %%
import pulsecompiler.qiskit.transforms.gate_barrier as barrier
print(barrier.BarrierBetweenGates().run_circuit(mycircuit))


# %%
import euriqafrontend.interactive.rfsoc.qiskit_backend as rfsoc_qiskit
import euriqafrontend.interactive.artiq_clients as artiq_clients
import euriqafrontend.modules.rfsoc as rfsoc_module

master_ip = "192.168.78.152"
num_ions = 7
calibrations = rfsoc_qiskit.get_calibration_box(
    rfsoc_qiskit.default_rf_calibration_path(), artiq_clients.get_artiq_dataset_db(master_ip)
)
# hack dataset -> 15 ions
# calibrations.
gate_solutions_15_ions = "/media/euriqa-nas/CompactTrappedIonModule/Data/gate_solutions/2022_7_15/15ions_interpolated_127us_to_7ions.h5"
calibrations.merge_update(rfsoc_module.RFSOC._load_gate_solutions(gate_solutions_15_ions, num_ions))

from pulsecompiler.qiskit.configuration import QuickConfig
qiskit_backend = rfsoc_qiskit.get_qiskit_backend(num_ions, rfsoc_qiskit.get_default_rfsoc_map(), calibrations, use_zero_index=False)
channel_map = rfsoc_qiskit.get_default_rfsoc_map()
#three_ion_config = QuickConfig(3,channel_map,{-1:1,0:3,1:5})
#qiskit_backend._config = three_ion_config
print("Print out the channel map")
print(channel_map)

# %%
import pulsecompiler.qiskit.transforms.gate_barrier as barrier
import qiskit.compiler as q_compile
import euriqabackend.waveforms.instruction_schedule as eur_inst_sched
import euriqabackend.waveforms.single_qubit as wf_sq

single_qubit_gates = eur_inst_sched.SINGLE_QUBIT_FUNCTION_MAP
def simple_square_rabi_gate(ion_index, phase, backend=None):
    return wf_sq.square_rabi(ion_index, duration=2e-6, individual_amp=0.1, global_amp=0.63, phase=phase, backend=backend)

single_qubit_gates["rx"] = simple_square_rabi_gate

instruction_schedule = eur_inst_sched.EURIQAInstructionScheduleMap(
        qiskit_backend,
        single_qubit_gates,
        eur_inst_sched.MULTI_QUBIT_FUNCTION_MAP,
    )

scheduled_scan_circuits = []
for theta_val in np.linspace(0, 2 * np.pi, 11):
    circuit_copy = mycircuit.copy()
    #circuit_copy.delay(1000)
    circuit_copy.rz(theta_val, 0)
    #circuit_copy.rx(-np.pi/2, 0)
    circuit_sequential = barrier.BarrierBetweenGates().run_circuit(circuit_copy)
    circuit_scheduled = q_compile.schedule(
        circuit_sequential, qiskit_backend, inst_map=instruction_schedule
    )
    scheduled_scan_circuits.append(circuit_scheduled)


print(scheduled_scan_circuits[0])
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
scheduled_scan_circuits[-1].draw()
plt.savefig("./tmp/schedule_0_2q_gate_circuit.png")

# %%
# submit converted qiskit circuit
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit


# rfsoc_submit.submit_schedule(scheduled_scan_circuits, master_ip, qiskit_backend, experiment_kwargs={"lost_ion_monitor": False})

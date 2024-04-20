import pickle

import cirq

from euriqabackend.devices.keysight_awg import circuit_interpreter as ci


def sample_clifford_circuit(N_qubits: int, print_circuit: bool = False):

    qubit_list = cirq.LineQubit.range(N_qubits)

    # make your ion trap device with desired gate times and qubits
    us = 1000 * cirq.Duration(nanos=1)
    ion_device = cirq.ion.IonDevice(
        measurement_duration=100 * us,
        twoq_gates_duration=200 * us,
        oneq_gates_duration=10 * us,
        qubits=qubit_list,
    )

    # some clifford gates
    circuit = cirq.Circuit()
    for i in [0, 3, 6]:
        circuit.append([cirq.H(qubit_list[i])])
        circuit.append([cirq.CNOT(qubit_list[i], qubit_list[i + 1])])
        circuit.append([cirq.CNOT(qubit_list[i], qubit_list[i + 2])])
    # for i in range(len(qubit_list)):
    # circuit.append([cirq.H(qubit_list[i])])
    for i in range(len(qubit_list)):
        circuit.append([cirq.measure(qubit_list[i])])
    # circuit.append([cirq.H(qubit_list[1])])

    if print_circuit:
        print("Clifford Circuit: \n", circuit, "\n")

    # convert the clifford circuit into circuit with ion trap native gates
    ion_circuit = ion_device.decompose_circuit(circuit)

    # if print_circuit:
    # print("Iontrap Circuit: \n", ion_circuit, "\n")

    return ion_circuit


def sample_ion_circuit(N_qubits: int):

    qubit_offset = 4

    builder = ci.CircuitBuilder(N_qubits, qubit_offset=qubit_offset)

    builder.XX([0, 3], +1)
    builder.XX([2, 3], -1)
    builder.XX([1, 3], +1)
    builder.XX([4, 3], -1)
    builder.XX([6, 3], +1)
    builder.XX([5, 3], -1)
    builder.RX(0, +1)
    builder.RX(1, +1)
    builder.RX(2, -1)
    builder.RX(4, -1)
    builder.RX(5, -1)
    builder.RX(6, +1)
    # The gates below are just for testing
    # builder.RX(1, +1, theta=4/4*np.pi)
    # builder.RY(2, +1)
    # builder.RX(4, -1, theta=np.pi)
    # builder.RX(5, -1, theta=0)
    # builder.RX(6, +1)
    # builder.RY(1, +1, theta=4/4*np.pi)
    # builder.RX(2, +1)

    return builder.get_circuit()


def write_cirq_file(cirq_file: str, print_circuit: bool = False):

    N_qubits = 15
    compile_from_Cliffords = False

    if compile_from_Cliffords:
        circuit = sample_clifford_circuit(N_qubits, print_circuit)
    else:
        circuit = sample_ion_circuit(N_qubits)

    with open(cirq_file, "wb") as c_file:
        pickle.dump(circuit, c_file)

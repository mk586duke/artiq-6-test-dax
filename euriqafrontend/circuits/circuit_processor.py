"""Tools for processing & converting cirq circuits into ones runnable on Breadboard."""
import logging
import pathlib
import pickle
import typing

import cirq
import numpy as np
import networkx as nx


_LOGGER = logging.getLogger(__name__)


def map_to_zero_basis(circuit: cirq.Circuit) -> cirq.Circuit:
    n = len(circuit.all_qubits())

    def zero_map_qubit(qubit: cirq.LineQubit) -> cirq.LineQubit:
        # latest cirq: return qubit - int(n/2)
        newx = qubit.x - int(n / 2)
        return cirq.LineQubit(newx)

    circuit = circuit.with_device(circuit.device, zero_map_qubit)
    circuit = strip_measurement(circuit)

    circuit.append(cirq.measure(*circuit.all_qubits()))
    return circuit


def map_matrix(matrix: np.matrix, num_ions: int):
    temp_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            temp_matrix[i - int(num_ions / 2)][j - int(num_ions / 2)] = matrix[i][j]

    return temp_matrix


def add_qubits(circuit: cirq.Circuit, num_qubits: int) -> cirq.Circuit:
    n = len(circuit.all_qubits())
    new_qubits = [cirq.LineQubit(i) for i in range(n, num_qubits)]

    my_qubits = [*circuit.all_qubits()] + new_qubits
    circuit.append(cirq.measure(*my_qubits))

    return circuit


def add_endcap_ions(circuit: cirq.Circuit, num_ions_total: int) -> cirq.Circuit:
    def move_qubits_up(qubit: cirq.LineQubit) -> cirq.LineQubit:
        newx = qubit.x + 1
        return cirq.LineQubit(newx)

    circuit = circuit.with_device(circuit.device, move_qubits_up)

    new_qubits = [cirq.LineQubit(0), cirq.LineQubit(num_ions_total - 1)]
    all_qubits = [*circuit.all_qubits()] + new_qubits

    circuit = strip_measurement(circuit)
    circuit.append(cirq.measure(*all_qubits))

    return circuit


def count_xx(circuit: cirq.Circuit) -> np.matrix:
    num_qubits = len(circuit.all_qubits())
    num_xx = np.zeros((num_qubits, num_qubits))
    for imoment in circuit:
        for op in imoment.operations:
            if op.gate.num_qubits() != 2:
                pass
            else:
                ion_index = list(sorted([op.qubits[0].x, op.qubits[1].x]))
                num_xx[ion_index[0]+7, ion_index[1]+7] += 1
    num_xx += num_xx.T
    return num_xx


def count_xx_dict(circuit: cirq.Circuit) -> typing.Dict[str, int]:
    num_xx = {}
    for imoment in circuit:
        for op in imoment.operations:
            if op.gate.num_qubits() != 2:
                pass
            else:
                ion_index = list(sorted([op.qubits[0].x, op.qubits[1].x]))
                bin_indices = str(ion_index[0]+8) + ", " + str(ion_index[1]+8)
                if bin_indices not in num_xx.keys():
                    num_xx[bin_indices] = 1
                else:
                    num_xx[bin_indices] = num_xx[bin_indices] + 1
    return num_xx


def find_gates_needed(circuit: cirq.Circuit) -> typing.KeysView[str]:
    xx_dict = count_xx_dict(circuit)
    return xx_dict.keys()
    # gates_needed = 1+np.transpose(np.unravel_index(np.flatnonzero(xx_matrix),xx_matrix.shape))
    # return gates_needed


def minimize_crosstalk(
    circuit: cirq.Circuit, cost_matrix: np.matrix, num_ions: int
) -> cirq.Circuit:
    xx_matrix = count_xx(circuit)

    new_map = {}
    total_gates = sum(xx_matrix.flatten())

    while total_gates > 0:

        [max_x, max_y] = np.unravel_index(np.argmax(xx_matrix), xx_matrix.shape)
        [min_x, min_y] = np.unravel_index(np.argmin(cost_matrix), xx_matrix.shape)
        # if not already re-mapped go ahead and re-map
        if max_x not in new_map.keys() and min_x not in new_map.values():
            new_map[max_x] = min_x
        if max_y not in new_map.keys() and min_y not in new_map.values():
            new_map[max_y] = min_y
        # update
        print(new_map)
        xx_matrix[max_x, max_y] = 0
        cost_matrix[min_x, min_y] = 1
        total_gates = sum(xx_matrix.flatten())

    for x in range(num_ions):
        if x not in new_map.keys():
            y = x
            while y in new_map.values():
                y = (y + 1) % num_ions
            new_map[x] = y

    def map_qubit_cross(qubit):
        newx = new_map[qubit.x]
        return cirq.LineQubit(newx)

    circuit = circuit.with_device(circuit.device, map_qubit_cross)

    return circuit


def random_map(circuit: cirq.Circuit, num_ions: int) -> cirq.Circuit:
    num_qubits = len(circuit.all_qubits())
    new_map = {}
    for qubit in circuit.all_qubits():
        if qubit.x not in new_map.keys():
            y = np.random.randint(0, num_qubits)
            while y in new_map.values():
                y = (y - 1) % num_ions
            new_map[qubit.x] = y

    def map_qubit_random(qubit):
        newx = new_map[qubit.x]
        return cirq.LineQubit(newx)

    circuit = circuit.with_device(circuit.device, map_qubit_random)

    return circuit


def strip_measurement(circuit: cirq.Circuit) -> cirq.Circuit:
    """Remove measurements from the circuit, if they exist."""
    # has_measurements() not in v0.5
    if any(circuit.findall_operations(cirq.protocols.is_measurement)):
        assert (
            circuit.are_all_measurements_terminal()
        ), "Mid-circuit measurements not supported"
        no_measure_circuit = circuit.copy()
        measurements_to_remove = list(
            (i, op)
            for i, op, _ in no_measure_circuit.findall_operations_with_gate_type(
                cirq.MeasurementGate
            )
        )
        no_measure_circuit.batch_remove(measurements_to_remove)
        return no_measure_circuit
    else:
        _LOGGER.debug("No measurements found to strip.")
        return circuit


def get_circuit_from_pickle(
    filename: typing.Union[str, pathlib.Path],
    N_ions: int,
    hardware_map_fn: typing.Callable[[cirq.LineQubit], cirq.LineQubit] = lambda q: q,
    save_output_txt_diagram: bool = False,
    out_pickle_fname: typing.Union[str, pathlib.Path] = None,
) -> cirq.Circuit:
    """
    Load a pickle file and convert it to run on the EURIQA breadboard system.

    Args:
        filename (typing.Union[str, pathlib.Path]): pickled cirq.Circuit filename
            to load
        N_ions (int): Number of ions currently in trap to work on.
        hardware_map_fn (typing.Callable[[cirq.LineQubit], cirq.LineQubit], optional):
            Mapping function from input qubit numbering to hardware qubit numbering.
            Given a cirq.LineQubit, should return a LineQubit.
            Defaults to doing no mapping/conversion.
        save_output_txt_diagram (bool, optional): Whether the processed circuit
            should be saved as a text file to record what will actually be run.
            Defaults to False.
        out_pickle_fname (str, pathlib.Path): name of output converted Cirq pickle file.
            If not set, input filename must include "raw" in the filename.

    Returns:
        [cirq.Circuit]: Unpickled circuit that has been processed to run on the
        EURIQA system with the current number of ions.
    """
    filename = pathlib.Path(filename)  # convert strs to proper type
    assert filename.resolve().is_file(), "file does not exist"
    circuit = pickle.loads(filename.read_bytes())

    circuit = strip_measurement(circuit)
    circuit = circuit.with_device(circuit.device, hardware_map_fn)

    endcap_index = int(N_ions / 2)
    all_ions = set(map(cirq.LineQubit, range(-endcap_index, endcap_index + 1)))
    # Add unused qubits by measuring any qubits not in circuit already
    # circuit.append(cirq.measure(*(all_ions - set(circuit.all_qubits()))))

    # Measure all ions
    circuit.append(cirq.measure(*all_ions))

    ##crosstalk minimization
    # xx_matrix = count_xx(circuit1)

    # total_crosstalk_mat = xx_matrix*cost_matrix
    # total_crosstalk = sum(total_crosstalk_mat.flatten())

    # Save modified circuit to FILENAME-RUN.pickle
    if out_pickle_fname is None:
        out_pickle_fname = filename.with_name(filename.name.replace("raw", "RUN"))
    else:
        out_pickle_fname = pathlib.Path(out_pickle_fname)
    out_pickle_fname.write_bytes(pickle.dumps(circuit))

    if save_output_txt_diagram:
        out_pickle_fname.with_suffix(".txt").write_text(
            "Circuit {orig_fname}:\n"
            "(Converted to run on EURIQA Breadboard in {out_fname})\n\n"
            "{circuit}".format(
                orig_fname=str(filename),
                out_fname=str(out_pickle_fname),
                circuit=str(circuit),
            ),
            encoding="utf-8",
        )

    return circuit

def check_gate_angles(circuit: cirq.Circuit)-> bool:
    for moment in circuit[:-1]:
        for op in moment.operations:
            if len(op.qubits) < 2:
                if abs(op.gate.exponent) > 1:
                    print("Single qubit gate larger than pi")
                    return False
            else:
                if op.gate.exponent >0.5:
                    print('MS gate angle larger than pi/2')
                    return False
    return True

class TransformSmallAngles(cirq.PointOptimizer):
    def optimization_at(self,circuit: cirq.Circuit, index: int, op: cirq.Operation):
        name = op.gate.__class__.__name__
        if name=='YPowGate' and abs(op.gate.exponent) % 0.5 != 0 :
            _LOGGER.info('changing a Y gate with exp {}'.format(op.gate.exponent))
            return cirq.PointOptimizationSummary(clear_span=1,
                                                new_operations=[cirq.rx(np.pi/2).on(op.qubits[0]),
                                                                cirq.rz(np.pi*op.gate.exponent).on(op.qubits[0]),
                                                                cirq.rx(-np.pi/2).on(op.qubits[0])],
                                                clear_qubits=[op.qubits[0]])
        elif name=='XPowGate' and abs(op.gate.exponent) % 0.5 != 0:
            return cirq.PointOptimizationSummary(clear_span=1,
                                                new_operations=[cirq.ry(-np.pi/2).on(op.qubits[0]),
                                                                cirq.rz(np.pi*op.gate.exponent).on(op.qubits[0]),
                                                                cirq.ry(np.pi/2).on(op.qubits[0])],
                                                clear_qubits=[op.qubits[0]])
        elif name == 'XPowGate' or name == 'YPowGate':
            num_ops = int(abs(op.gate.exponent/0.5))
            op_sign = np.sign(op.gate.exponent)
            if name=='YPowGate':
                myop = cirq.ry(op_sign*np.pi/2).on(op.qubits[0])
            else:
                myop = cirq.rx(op_sign*np.pi/2).on(op.qubits[0])
            new_operations = [myop]*num_ops
            return cirq.PointOptimizationSummary(clear_span=1,
                                                new_operations=new_operations,
                                                clear_qubits=[op.qubits[0]])
                    
def convert_1Qgates(circuit: cirq.Circuit) -> cirq.Circuit:
    return TransformSmallAngles().optimize_circuit(circuit)
    

import numpy as np

# from numpy import genfromtxt
# from euriqafrontend import EURIQA_NAS_DIR
# base_dir = EURIQA_NAS_DIR / "CompactTrappedIonModule"
# save_path = base_dir / "Data" / "gate_solutions" / "2019_12_10"
# xx_crosstalk = genfromtxt(str(save_path/"XXcrosstalk_total.csv"), delimiter=',')
# cost_matrix = xx_crosstalk

#### THIS IS NOT MATURE CODE, ADDING IT HERE SO IT IS NOT LOST!!!!
def optimize_ion_mapping(num_qubits,circuit, cost_matrix):
    raise NotImplementedError("THIS CODE IS NOT MATURE YET")

    num_xx = np.zeros((num_qubits - 2, num_qubits - 2))
    for imoment in circuit:
        for op in imoment.operations:
            if op.gate.num_qubits() != 2:
                pass
            else:
                ion_index = list(
                    sorted([op.qubits[0].x + 6, op.qubits[1].x + 6]))  # TODO add ion indexing to register object as a dict
                num_xx[ion_index[0], ion_index[1]] += 1
    num_xx += num_xx.T

    method = "greedy"
    if method == "brute":
        pass

    elif method == "greedy":

        num_iterations = int(10)
        best_map = np.copy(num_xx)

        for itr in range(num_iterations):

            mapping = []
            map_qubits = np.arange(num_qubits - 2)
            most_gates = np.flip(np.argsort(np.sum(best_map, axis=0)))
            perm = np.arange(num_qubits - 2).tolist()

            for i_from in most_gates:
                score = np.zeros(len(map_qubits))

                for i, i_to in enumerate(map_qubits):
                    test_map = np.zeros(best_map.shape)
                    test_map[i_to, :] = best_map[i_from, :]
                    test_map[:, i_to] = best_map[:, i_from]
                    score[i] = np.sum(test_map * cost_matrix)

                i_to_best = map_qubits[np.argmin(score)]
                mapping.append(tuple([i_from, i_to_best]))
                map_qubits = map_qubits[map_qubits != i_to_best]
                perm[i_to_best] = i_from

            new_best_map = np.copy(best_map)
            new_best_map = new_best_map[perm, :]
            new_best_map = new_best_map[:, perm]
            best_map = np.copy(new_best_map)

import os
import pathlib
import platform
import typing
from datetime import date

import numpy as np
import cirq
import pickle
import yaml
from scipy.stats import binom
import networkx as nx
import matplotlib.pyplot as plt

import euriqafrontend.circuits.circuit_plotter as plotter
from euriqafrontend.circuits.circuit_scanner import CircuitSubmitter
from euriqabackend.databases.device_db_main_box import CONTROL_PC_IP_ADDRESS
from euriqafrontend import EURIQA_NAS_DIR


NAS_SERVER_PATH = EURIQA_NAS_DIR / "CompactTrappedIonModule"
import euriqafrontend.circuits.circuit_processor as processor


class CircuitInterface:
    """Utility class to facilitate circuit-specific experiment control & simulation."""

    def __init__(
        self,
        circuit_path: pathlib.Path,
        project_name: str,
        default_mapping: typing.List[int],
        master_ip: str = CONTROL_PC_IP_ADDRESS,
        server: pathlib.Path = NAS_SERVER_PATH,
        prototype_path: pathlib.Path = pathlib.Path(
            "ARTIQ", "experiment_prototypes", "Circuit_15ions.h5"
        ),
        data_path: pathlib.Path = pathlib.Path("Data", "artiq_data"),
        all_ions: np.array = np.arange(-7, 8, 1),
        result_path: pathlib.Path = pathlib.Path("circuit_result"),
    ):
        """
        Create an interface for submitting & retrieving results from Cirq circuits.

        Args:
            circuit_path (pathlib.Path): Path where the circuits are locally cached.
            project_name (str): used for naming folders
            default_mapping (typing.List[int]): default ordering of qubits.
            master_ip (str, optional): IP address of the ARTIQ master PC.
                Defaults to the value in the Device Database (device_db_main_box).
            server (pathlib.Path, optional): Path to which the NAS server is mapped
                on local PC. Defaults to "CompactTrappedIonModule" folder
                on the EURIQA NAS.
            prototype_path (pathlib.Path, optional): Location of ARTIQ experiment
                prototypes. Defaults to pathlib.Path("ARTIQ", "experiment_prototypes",
                "Circuit_15ions.h5").
            data_path (pathlib.Path, optional): Relative to ``server``, where the
                HDF5 experiment result files are stored.
                Defaults to pathlib.Path("Data", "artiq_data").
            all_ions (np.array, optional): All ions available in the current
                configuration. Uses center-indexed numbering scheme.
                Defaults to np.arange(-7, 8, 1).
            result_path (pathlib.Path, optional): Local path where results should be stored. Defaults to pathlib.Path("circuit_result").
        """
        # specify local info : where the circuits are stored locally (circuit_path),
        # name of the circuit batches to be saved on server (project_name),
        # qubit mapping,
        # and where are the processed result saved locally

        save_path = pathlib.Path("Data", "circuits", project_name)
        self.circuit_path = circuit_path
        self.result_path = result_path
        self.default_mapping = default_mapping
        self.qubits = [cirq.LineQubit(i) for i in default_mapping]
        self.data_path = server / data_path
        self.all_ions = all_ions.tolist()

        circuit_path.mkdir(parents=True, exist_ok=True)
        result_path.mkdir(parents=True, exist_ok=True)

        self.submit = CircuitSubmitter(
            local_server=server,
            prototype_path=prototype_path,
            save_path=save_path,
            master_ip=master_ip,
        )

    def measurement_all(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Add measurement for all available qubits (all ions)."""
        temp = cirq.Circuit()
        temp.append(circuit)
        temp2 = cirq.Circuit()
        for i in self.all_ions:
            temp2.append(cirq.measure(cirq.LineQubit(i)))
        temp.append(temp2)
        return temp

    def measurement_used(self, circuit: cirq.Circuit, qubit_mapping: typing.List[int] = None) -> cirq.Circuit:
        """Add measurement for all used qubits in the circuit."""
        temp = cirq.Circuit()
        temp.append(circuit)
        temp2 = cirq.Circuit()

        if qubit_mapping is None:
            qubit_mapping = self.default_mapping

        for i in qubit_mapping:
            temp2.append(cirq.measure(cirq.LineQubit(i)))
        temp.append(temp2)
        return temp

    def validate_result(
        self,
        result: np.array,
        circuit: cirq.Circuit = None,
        qubit_mapping: typing.List[int] = None,
        tolerance: float = 1,
    ) -> bool:
        """Automatically validate the data.

        By default, check if the result population is normalized.
        cherry_picking=True: directly comparing the circuit with simulation

        This needs some more work. This function does nothing by default,
        and the default circuit just errors.

        Args:
            result (np.array): TODO
            circuit (cirq.Circuit, optional): The circuit that the result should
                be compared against. Defaults to None.
            qubit_mapping (typing.List[int], optional): TODO. Defaults to None.
            tolerance (float, optional): TODO. Defaults to 1.

        Raises:
            RuntimeError: if no circuit has been provided

        Returns:
            bool: ``True`` if the circuit is close to simulation, ``False`` otherwise.
        """
        if abs(np.sum(result) - 1) > 0.01:
            return False

        if tolerance > 0.99:
            return True

        if circuit is None:
            raise RuntimeError("circuit is missing for fidelity test")

        if qubit_mapping is None:
            qubit_mapping = self.default_mapping

        simulated_histo = self.simulate(circuit=processor.strip_measurement(circuit), qubit_mapping=qubit_mapping)

        similarity = np.inner(result, simulated_histo) / np.sqrt(
            np.inner(result) * np.inner(simulated_histo)
        )
        return similarity > tolerance

    def kl_divergence(self, measured_histo,target_histo):
        if len(target_histo)!=len(measured_histo):
            raise RuntimeError(
                "the probability distribution doesn't have the same measure"
            )
        kld=0
        for i in range(len(target_histo)):
            if measured_histo[i]<0.0001:
                temp=0.0001
            else:
                temp=measured_histo[i]

            if target_histo[i]>0:
                kld+=target_histo[i]*np.log2(target_histo[i]/temp)
        return kld

    def inver_bino_lower(self, x, n, x1, x2):
        temp = (x1 + x2) / 2
        if abs(temp - x1) < 0.000000001:
            return temp
        elif binom.cdf(x, n, temp) > 0.84135:
            return self.inver_bino_lower(x, n, temp, x2)
        else:
            return self.inver_bino_lower(x, n, x1, temp)

    def inver_bino_upper(self,x, n, x1, x2):
        temp = (x1 + x2) / 2
        if abs(temp - x1) < 0.000000001:
            return temp
        elif binom.cdf(x, n, temp) > 0.15865:
            return self.inver_bino_upper(x, n, temp, x2)
        else:
            return self.inver_bino_upper(x, n, x1, temp)

    # x here is taken as probability of geting a specific state configuration from measurement.
    def binomial_error(self,x, n=2000):
        return self.inver_bino_lower(round(x * n), n, 0, 1), self.inver_bino_upper(round(x * n), n, 0, 1)


    def kld_test(self,
        result: np.array,
        circuit: cirq.Circuit = None,
        qubit_mapping: typing.List[int] = None):
        """compute the K-L divergence of a experimental result with the simulation result"""

        if qubit_mapping is None:
            qubit_mapping = self.default_mapping

        temp_circuit=processor.strip_measurement(circuit)

        simulated_histo = np.square(np.absolute(self.simulate(circuit=temp_circuit, qubit_mapping=qubit_mapping)))

        return self.kl_divergence(result,simulated_histo)


    def experiment(
        self,
        circuit: cirq.Circuit,
        shots: int = 500,
        repetition: int = 4,
        scan=[],
        qubit_mapping: typing.List[int] = None,
        result_format: str = "histogram",
        has_measurements: bool = False,
        max_attempts: int = 10,
        wait_time_ms: float = 0,
        wait_before_time_ms: float = 0,
        wait_after_time_ms: float = 0,
        num_interactions: int = 0,
        do_validation: bool=True

    ) -> typing.Dict[str, typing.Any]:
        """
        Run experiment on the EURIQA system.

        Args:
            circuit (cirq.Circuit): Circuit to run.
            shots (int, optional): Number of shots to run (per repetition).
                Total shots = ``num_shots * repetitions``. Defaults to 500.
            repetition (int, optional): Number of repetition blocks that the
                circuit is divided into. Defaults to 4.
            qubit_mapping (typing.List[int], optional): TODO. Defaults to None.
            result_format (str, optional): Different ways of returning the results.
                Only "histogram" currently supported. Defaults to "histogram".
            has_measurements (bool, optional): Whether the given circuit has
                measurements. If not, measurements will be added to it.
                Defaults to False.
            max_attempts (int, optional): How many times to try resubmitting
                the circuit before giving up/needing to check the machine.
                Defaults to 10.

        Raises:
            RuntimeError: if the circuit has failed to complete too many times,
                if the data was corrupted with NaN values, or
                if the population was not normalized.

        Returns:
            dict: contains RID, result population, and number of shots.
            Keys: ["results", "shots", "rid"]
        """
        if qubit_mapping is None:
            qubit_mapping = self.default_mapping

        self.submit.args.num_shots = shots

        if not has_measurements:
            circuit_to_submit=self.measurement_all(circuit)
        else:
            circuit_to_submit=circuit.copy()

        cirq.DropEmptyMoments().optimize_circuit(circuit_to_submit)

        counter = 0

        if len(scan)==0:
            scan=range(repetition)
            tot_shots=repetition*shots
        else:
            tot_shots=len(scan)*shots

        while True:
            counter += 1
            if counter >= max_attempts:
                raise RuntimeError("Too many failed attempt: please debug the system")

            rid = self.submit.submit_circuit_scan(
                circuit=circuit_to_submit, scan=scan, wait_after_time_ms=wait_after_time_ms,
                wait_before_time_ms=wait_before_time_ms, wait_time_ms=wait_time_ms,num_interactions=num_interactions
            )

            self.submit.wait_for_completion(rid)


            try:
                temp = plotter.get_binary_data(self.data_path, rid, date=date.today())
                bin_data = temp.reshape(temp.shape[0], -1)

                if np.isnan(np.sum(bin_data)):
                    print("Data corrupted: result contains NAN values")
                    raise RuntimeError("Data corrupted: result contains NAN values")

                if result_format == "histogram":
                    hist_data = np.around(
                        np.divide(
                            plotter.hist_states(
                                bin_data,
                                [self.all_ions.index(i) for i in qubit_mapping],
                            )[0],
                            tot_shots,
                        ),
                        5,
                    ).tolist()
                    if do_validation and (not self.validate_result(hist_data)):
                        print("Data corrupted: the population is not normalized.")
                        raise RuntimeError(
                            "Data corrupted: the population is not normalized."
                        )
                break
            except:
                print("circuit failed: make another attempt")

        result = dict()

        if result_format == "histogram":
            result["result"] = hist_data
        elif result_format == "binary":
            result["result"] = bin_data

        result["shots"] = shots * repetition
        result["qubit_mapping"]=qubit_mapping
        result["rid"] = rid
        if (result_format=="histogram") and (do_validation):
            result["KLD"]=float(self.kld_test(result=hist_data,circuit=circuit,qubit_mapping=qubit_mapping))
        return result

    def mapping_fn(self,qubit_mapping):

        def temp(i):
            return cirq.LineQubit(qubit_mapping[i.x])

        return temp

    def remap_circuit(self,circuit: cirq.Circuit,qubit_mapping: typing.List[int] = None):
        if qubit_mapping is None:
            qubit_mapping = self.default_mapping
        return circuit.with_device(circuit.device,self.mapping_fn(qubit_mapping))

    def emulate(
        self,
        circuit: cirq.Circuit,
        shots: int = 2000,
        qubit_mapping: typing.List[int] = None,
        result_format: str = "histogram",
    ) -> typing.Dict[str, typing.Any]:
        """
        Emulate a circuit as running on the EURIQA machine.

        Mimics the behavior of :meth:`experiment`. See that for explanation of arguments.
        NOTE: the shots argument is different here and not multiplied, because
        ``repetitions`` is a backend limitation in :meth:`experiment`.
        """
        if qubit_mapping is None:
            qubit_mapping = self.default_mapping
        # Emulate the circuit, mimicking the experiment
        simulator = cirq.Simulator()
        result = simulator.run(self.measurement_used(circuit,qubit_mapping=qubit_mapping), repetitions=shots)
        processed_result = np.array(
            [result.measurements[str(i)].astype(int) for i in qubit_mapping]
        )[:, :, 0]
        result = dict()
        if result_format == "histogram":
            result["result"] = np.around(
                np.divide(
                    plotter.hist_states(processed_result, range(len(qubit_mapping)))[0],
                    shots,
                ),
                4,
            ).tolist()
        elif result_format == "binary":
            result["result"] = processed_result
        result["shots"] = shots
        return result

    def simulate(self, circuit: cirq.Circuit, qubit_mapping: typing.List[int] = None,printing=False):
        """Simulate the circuit, give the state vector as the result."""
        if qubit_mapping is None:
            qubit_order = self.qubits
        else:
            qubit_order = [cirq.LineQubit(i) for i in qubit_mapping]

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit, qubit_order=qubit_order)
        if printing:
            print(result)
        return result.final_state

    def gates_count(self, circuit: cirq.Circuit) -> typing.Dict[str, typing.Any]:
        """count the number of single-qubit gates and two-qubit gates used"""
        count = dict()
        count["single-qubit"] = 0
        count["two-qubit"] = 0
        for moment in circuit:
            for gate in moment:
                if len(gate.qubits) == 1:
                    count["single-qubit"] += 1
                else:
                    count["two-qubit"] += 1
        count["total"] = count["single-qubit"] + count["two-qubit"]
        return count

    def measurement_basis(self, circuit: cirq.Circuit) -> typing.List[int]:
        """Retrieve measurement basis of a circuit.

        For the classical shadow experiment.

        Returns the "basis" for each qubit in the circuit. 0 is no operation.
        Others need clarified.
        """
        last_moment = circuit[-1]
        basis = [0] * len(self.qubits)
        for operation in last_moment.operations:
            for j in range(len(self.qubits)):
                if operation.qubits[0] == self.qubits[j]:
                    index = j
            if operation.gate == cirq.rx(np.pi / 2):
                basis[index] = 2
            elif operation.gate == cirq.ry(-np.pi / 2):
                basis[index] = 1
            elif operation.gate == cirq.rz(0):
                basis[index] = 3
            else:
                print("warning: failed to recover measurement basis")
        return basis

    def save_circuit(self, circuit: cirq.Circuit, filename: str, path: str = None):
        """Save the given circuit as a pickle file.

        If no path is given, defaults to ``circuit_path``.
        """
        if path is None:
            path = self.circuit_path
        else:
            path = pathlib.Path(path)

        with open(str(path / filename), "wb") as file:
            pickle.dump(circuit, file)
        return

    def all_circuits(
        self, path: str = None, keyword: typing.Optional[str] = ".pickle"
    ) -> typing.List[pathlib.Path]:
        """Retrieve list of all circuits in the circuit directory.

        If no path is provided, defaults to ``circuit_path``.
        All file names in path are searched for ``keyword`` if given, and only
        matches will be returned.
        """
        if path is None:
            path = self.circuit_path

        file_list = [
            x
            for x in os.listdir(str(path))
            if (x.find(".pickle") > -1 and x.find(keyword) > -1)
        ]
        return file_list

    def load_circuit(self, filename: str, path: str = None):
        """Load a Cirq circuit from a pickle file."""
        if path is None:
            path = self.circuit_path
        else:
            path = pathlib.Path(path)

        with open(str(path / filename), "rb") as file:
            circuit = pickle.load(file)
        return circuit

    def save_result(self, file_name: str, **kwargs):
        """Save an experiment/emulation result as a yaml file."""
        with open(str(self.result_path / file_name), "w") as file:
            yaml.dump(kwargs, file)

    def load_result(self, file_name: str) -> typing.Dict:
        """Load a yaml file that contains an experiment result."""
        with open(str(self.result_path / file_name), "r") as file:
            result = yaml.load(file, Loader=yaml.FullLoader)
        return result

    def calibration_sequence(self,circuit: cirq.Circuit):
        """return a string that can be used to control calibrations"""
        temp = processor.find_gates_needed(circuit)
        sequence = str()
        for pair in temp:
            sequence += pair + ";"

        return sequence.replace(" ", "")

    def count_xx_dict(self, circuit: cirq.Circuit):
        num_xx = {}
        for imoment in circuit:
            for op in imoment.operations:
                if op.gate.num_qubits() != 2:
                    pass
                else:
                    ion_index = list(sorted([op.qubits[0].x, op.qubits[1].x]))
                    bin_indices = str(ion_index[0]) + ", " + str(ion_index[1])
                    if bin_indices not in num_xx.keys():
                        num_xx[bin_indices] = 1
                    else:
                        num_xx[bin_indices] = num_xx[bin_indices] + 1
        return num_xx

    def xx_graph(self, circuit: cirq.Circuit):
        graph = nx.Graph()
        xx_dict = self.count_xx_dict(circuit)
        for key in xx_dict.keys():
            nodes = key.split(",")
            graph.add_edge(int(nodes[0]), int(nodes[1]), weight=xx_dict[key])
        return graph

    def plot_xx_graph(self,circuit:cirq.Circuit,file_name="xx_graph"):
        G = self.xx_graph(circuit)
        pos = nx.circular_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.savefig(file_name+".svg")
        return G



import logging
import pathlib
import pickle
import re
import time
import typing
import copy
from datetime import datetime

import cirq
import h5py
import numpy as np
import sympy
import sipyco.pyon as pyon

import euriqafrontend.interactive.artiq_clients as artiq_clients
from euriqabackend.databases.device_db_main_box import CONTROL_PC_IP_ADDRESS

_LOGGER = logging.getLogger(__name__)

class ExperimentArgs:

    def __init__(self, expid: dict):
        """The class is used to manage setting and updating circuit_scan experiment arguments

        Args:
            expid: expid dictionary loaded from the .h5 file of a previous experiment
        """
        self.name_map = dict()
        for key in expid["arguments"].keys():
            clean_key = re.sub(' ', '_', key)
            clean_key = re.sub('[^0-9a-zA-Z_]', '', clean_key)
            self.name_map[clean_key] = key
            setattr(self, clean_key, expid["arguments"][key])

    def update_args(self,expid):
        for key in self.name_map:
            expid["arguments"][self.name_map[key]] = getattr(self,key)
        return expid


class CircuitSubmitter:
    def __init__(self,
                 local_server: pathlib.Path,
                 prototype_path: pathlib.Path,
                 save_path: pathlib.Path,
                 master_ip: str = CONTROL_PC_IP_ADDRESS,
                 master_port: int = 3251
                 ):
        """ This class is used to manage submitting circuits through a jupyter notebook interface

        Args:
            local_server: a pathlib Path to the EURIQA Q server on your local machine
                          (Should end in  .../CompactTrappedIonModule)
            prototype_path: pathLib Path to .h5 file of a circuit_scan experiment prototype (relative to server)
            save_path:  pathlib Path of where to save circuits (relative to server)
            master_ip: IP address of the master
            master_port: port of the master
        """
        if type(local_server) is str:
            local_server = pathlib.Path(local_server)

        if type(prototype_path) is str:
            prototype_path = pathlib.Path(prototype_path)
        self.prototype_path_local = local_server / prototype_path
        if not self.prototype_path_local.exists():
            _LOGGER.error("Prototype .h5 file for circuit submission not found")

        if type(save_path) is str:
            save_path = pathlib.Path(save_path)
        self.save_path_relative = save_path
        self.save_path_local = local_server / save_path
        self.save_path_local.mkdir(parents=True, exist_ok=True)

        # let's connect to the master
        self.master_ip = master_ip
        self.master_port = master_port
        self.schedule = artiq_clients.get_artiq_scheduler(master_ip, master_port)
        self.exps = artiq_clients.get_artiq_experiment_db(master_ip, master_port)
        self.datasets = artiq_clients.get_artiq_dataset_db(master_ip, master_port)

        file = h5py.File(str(self.prototype_path_local), "r")
        self.expid = pyon.decode(file["expid"][()])
        self.args = ExperimentArgs(self.expid)


    def submit_circuit_scan(self,
                            circuit: cirq.Circuit,
                            scan: typing.Union[int, list, np.array] = 0,
                            scan_parameters=None,
                            priority:int=0,
                            wait_time_ms:float=0,
                            wait_before_time_ms:float=0,
                            wait_after_time_ms:float=0,
                            num_interactions:int=0):
        """ Set the value of the symbols. Vector if scan, int if static
        # Finds symbols in the circuit and return their locations, names, and values


        Args:
            circuit: a single cirq circuit that may have symbolic representations in it
            scan: a list, np array or float - values over which to scan the symbols in the circuit.
                    All unresolved symbols will be scanned together.
            scan_parameters: a dictionary output from categorize_symbols. Optional if you want to only scan
                            certain symbols. Default is to scan all symbols in the circuit together.
            priority: sets the experiment priority in the queue
            wait_time_ms: For experiments with a wait gate, see awg_raman.AWGRamanCircuitWait
            wait_before_time_ms: For experiments with a wait gate, see awg_raman.AWGRamanCircuitWait

        Returns:

        """
        if scan_parameters is None:
            scan_parameters = categorize_symbols(circuit)

        for sym in scan_parameters:
            if scan_parameters[sym]["val"] is None:
                scan_parameters[sym]["val"] = scan

        if hasattr(circuit, "__name__"):
            save_name = circuit.__name__ + ".pickle"
        else:
            now = datetime.now()
            save_name = now.strftime("%Y_%m_%d_%H%M%S") + ".pickle"

        circuit_file = str(self.save_path_local / save_name)

        if type(scan) == int:
            num_scan = 1
        else:
            num_scan = len(scan)

        # If there are no symbols just copy the circuit over the length of the scan
        if len(scan_parameters) == 0:
            circuit_scan = [circuit] * num_scan
            static_gate_array = True

        # If there are symbols in the circuit, try to resolve them
        else:
            circuit_scan, static_gate_array = make_circuit_scan(circuit, scan_parameters)

        # Write circuit to file so RFCompiler can read it in
        with open(circuit_file, "wb") as fname:
            pickle.dump(circuit_scan, fname)

        self.args.suppress_circuit_scan = static_gate_array
        self.args.path_to_circuit = str(self.save_path_relative / save_name)
        self.expid = self.args.update_args(self.expid)
        expid = copy.deepcopy(self.expid)


        if num_interactions !=0:
            expid["class_name"] = "AWGInteraction"
            expid["arguments"]["wait_time"] = wait_time_ms*1e-3
            expid["arguments"]["wait_time_before"] = wait_before_time_ms*1e-3
            expid["arguments"]["wait_time_after"] = wait_after_time_ms*1e-3
            expid["arguments"]["num_interactions"] = num_interactions


        elif wait_before_time_ms!=0 or wait_time_ms!=0:
            expid["class_name"] = "AWGRamanCircuitWait"
            expid["arguments"]["wait_time"] = wait_time_ms*1e-3
            expid["arguments"]["wait_time_before"] = wait_before_time_ms*1e-3



        rid = self.schedule.submit(pipeline_name="main",
                              expid=expid,
                              priority=priority,
                              due_date=None,
                              flush=False)

        return rid

    def wait_for_completion(self, rid):
        while rid in self.schedule.get_status():
            time.sleep(0.1)
        return True


def categorize_symbols(circuit: cirq.Circuit) -> dict:
    """ Scans through a circuit and pulls out all the parameterized symbols.

    Args:
        circuit: a Cirq circuit object

    Returns: All the symbols found in the circuit and some additional information about them. A dict of dicts.
        e.g. scan_paramters = {"symbol_1": {inds:, List of tuples (moment, operation) where gate is found in circuit
                                            val:, Value of the symbol. Defaults to None, must be overwritten later
                                            sym: sympy object with symbol_1 name},
                               "symbol_2": ...... }
    """

    scan_parameters = dict()
    for imoment, moment in enumerate(circuit):
        for iop, op in enumerate(moment):
            try:
                exp = circuit[imoment].operations[iop].gate.exponent
                if isinstance(exp, sympy.Basic):
                    for var in exp.free_symbols:
                        sym_name = var.name
                        try:
                            scan_parameters[sym_name]["inds"].append((imoment,iop))
                        except KeyError:
                            scan_parameters[sym_name] = {"inds": [(imoment,iop)], "val": None, "sym":var}
            except AttributeError:
                pass
                # Found a gate without an exponent. Assume this is a destructive measurement gate with no symbols
    return scan_parameters


def make_circuit_scan(circuit: cirq.Circuit, scan_parameters: dict) -> (typing.List[cirq.Circuit], np.ndarray):
    """ Takes a parameterized circuit and generates a scan based on values provided in scan_parameters dictionary

    Args:
        circuit: a Cirq circuit object
        scan_parameters: see output of categorize_symbols(). All symbols must have some "val". Multiple scans must be
                         same length and will be stepped through simultaneously.

    Returns:
        circuit_scan: List of circuits. Each circuit in the list has been populated with values. No more symbols.
        static_gate_array: A boolean numpy array where index [i,j] indicates whether the gate in the i-th moment,
                           j-th operation is static across all the circuit scan.

    """

    num_moments = len(circuit)
    max_ops = max([len(moment) for moment in circuit])

    static_gate_array = np.full((num_moments, max_ops), True)
    scan_length = None
    scanned_parameters = dict()

    for isym in scan_parameters:
        val = scan_parameters[isym]["val"]
        assert val is not None, "Parameterized symbols must be given values before building a circuit scan"
        if type(val) == int or type(val) == float or len(val) == 1:
            resolver = cirq.ParamResolver({isym: val})
            circuit = cirq.resolve_parameters(circuit, resolver)
        else:
            if scan_length is None:
                scan_length = len(val)
            else:
                assert len(val) == scan_length, "Cannot scan multiple parameters that have different lengths"

            scanned_parameters[isym] = val
            for inds in scan_parameters[isym]["inds"]:
                static_gate_array[inds[0], inds[1]] = False

    circuit_scan = []
    if scan_length is None:
        circuit_scan.append(circuit)
    else:
        for iscan in np.arange(scan_length):
            current_scan = dict()
            for isym in scanned_parameters:
                current_scan[isym] = scanned_parameters[isym][iscan]
            resolver = cirq.ParamResolver(current_scan)
            circuit_scan.append(cirq.resolve_parameters(circuit,resolver))

    return circuit_scan, static_gate_array

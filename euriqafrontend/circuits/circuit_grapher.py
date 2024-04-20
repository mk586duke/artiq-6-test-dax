"""Tools for processing circuits as graphs or networks."""
import logging
import pathlib
import pickle
import typing

import cirq
import numpy as np
import networkx as nx

from euriqafrontend.circuits.circuit_processor import *


_LOGGER = logging.getLogger(__name__)


def make_line_graph(G: nx.Graph):
    """this function takes a graph and makes the edges into nodes and vice versa
		also known asan adjoint/conjugate"""
    test = nx.Graph()
    test.add_nodes_from(G.edges)
    for (ion1, ion2) in test.nodes:
        for (ion1b, ion2b) in test.nodes:
            if ion1 == ion1b or ion2 == ion1b:
                test.add_edge((ion1, ion2), (ion1b, ion2b), weight=ion1b)
            elif ion2 == ion2b or ion1 == ion2b:
                test.add_edge((ion1, ion2), (ion1b, ion2b), weight=ion2b)
    return test


def inverse_line_graph(G: nx.Graph):
    """this function takes a graph with nodes labeled as edges and inverts it"""
    return nx.Graph().add_edges_from(G.nodes)


def remap(
    G: nx.graph, J: nx.graph,
):
    """We want to see if J can be remapped as a subgraph of G and return the remapped graph
	this function is hand-written and slower, but more tested"""
    G_line = make_line_graph(G)
    J_line = make_line_graph(J)
    ## GraphMatcher is define as (large graph, desired subgraph)
    gm = nx.algorithms.isomorphism.GraphMatcher(G_line, J_line)
    if not gm.subgraph_is_isomorphic():
        return None
    ## the mapping that is returned is from G to J, so we need to invert it
    inv_map = {v: k for k, v in gm.mapping.items()}
    J_line_re = nx.relabel_nodes(J_line, inv_map)
    J_re = inverse_line_graph(J_line_re)
    return J_re


def remap_mono(
    G: nx.graph, J: nx.graph,
):
    """We want to see if J can be remapped as a subgraph of G and return the remapped graph
	this funciton is faster than remap"""
    gm = nx.algorithms.isomorphism.GraphMatcher(G, J)
    if not gm.subgraph_is_monomorphic():
        return None
    inv_map = {v: k for k, v in gm.mapping.items()}
    J_re = nx.relabel_nodes(J, inv_map)
    return J_re


def add_weights(
    G: nx.graph, cost_matrix: np.matrix, N: int = 13,
):
    """the function adds weights to the edges in a graph corresponding to their cost (crosstalk for now)
	NOTE: this is an active function on a graph - no copy is made"""
    N = cost_matrix.shape[0]
    for i in range(N):
        for j in range(N):
            if i != j:
                G[i - int(N / 2)][j - int(N / 2)]["weight"] = cost_matrix[i][j]


def make_machine_graph(
    cost_matrix: np.matrix, N_qubits: int = 13,
):
    """this makes a fully connected graph with the given cost matrix
	returns the graph and the edges sorted by weight"""
    H = nx.complete_graph(N_qubits)
    J = nx.convert_node_labels_to_integers(H, first_label=-int(N_qubits / 2))
    add_weights(J, cost_matrix, N_qubits)
    edges = sorted(J.edges.data("weight"), key=lambda t: t[2])
    return J, edges


def optimize_for_crosstalk(
    paths_to_circuits: list,
    save_path: pathlib.Path,
    cost_matrix: np.matrix,
    N: int = 13,
    threshold: float = 0.1,
):
    """ given a set of circuits and an acceptable crosstalk threshold, make a graph of all gates in all circuits (without any remapping)
	 	iterate by adding gates in order of least to most cost until an embedded mapping, or subgraph monomorphism is found"""
    J, edges = make_machine_graph(cost_matrix, N)

    ###find all the gates needed
    all_gates = []
    for single_circuit_path in paths_to_circuits:
        single_circuit = pickle.loads(single_circuit_path.read_bytes())
        if isinstance(single_circuit, list):
            single_circuit = single_circuit[0]
        gates_needed = find_gates_needed(single_circuit)
        for pair in gates_needed:
            if pair not in all_gates:
                all_gates.append(pair)

    ## create the graph from the imported circuits

    gate_edges = [tuple(x.split(",")) for x in all_gates]
    G = nx.Graph()
    for (u, v) in gate_edges:
        G.add_edge(int(u), int(v))

    ## iterate through thresholds to find subgraphisomorphism
    threshold_values = np.linspace(0.02, threshold)
    for val in threshold_values:
        crosstalk_threshold = val
        # J is the machine fully connected graph
        # first we want to make a graph with only those edges that meet the crosstalk criteria
        esmall = [
            (u, v)
            for (u, v, d) in J.edges(data=True)
            if d["weight"] <= crosstalk_threshold
        ]
        J_reduced = nx.Graph()
        J_reduced.add_edges_from(esmall)

        G_run = remap_mono(J_reduced, G)
        if G_run is not None:
            print("threshold is ", crosstalk_threshold)
            break
        if crosstalk_threshold == threshold:
            raise RuntimeError("No isomorphism found, raise your threshold")

    gm_nodes = nx.algorithms.isomorphism.GraphMatcher(G, G_run)
    gm_nodes.is_isomorphic()
    low_crosstalk_map = gm_nodes.mapping

    def low_cross_fcn(qubit):
        return cirq.LineQubit(low_crosstalk_map[qubit.x])

    for single_circuit_path in paths_to_circuits:
        filename_save = pathlib.Path(save_path, single_circuit_path.name)
        single_circuit = get_circuit_from_pickle(
            single_circuit_path,
            15,
            hardware_map_fn=low_cross_fcn,
            out_pickle_fname=filename_save,
        )

    nx.draw_circular(G_run, with_labels=True)

    print("max degree = ", str(len(nx.degree_histogram(G_run)) - 1))
    print("total num of gates = ", str(G_run.number_of_edges()))

    #     qubits_needed = len(max(nx.connected_components(G_run), key=len))
    print("qubits needed = ", str(G_run.number_of_nodes()))
    print("gates needed ", str(G_run.edges))

    return low_crosstalk_map,G_run.edges


def optimize_for_crosstalk_iter(
    paths_to_circuits: list,
    save_path: pathlib.Path,
    cost_matrix: np.matrix,
    N: int = 13,
    threshold: float = 0.1,
):
    """ given a set of circuits and an acceptable crosstalk threshold, for each circuit - iterate by adding gates in order
		of least to most cost until an embedded mapping, or subgraph monomorphism is found"""
    J, edges = make_machine_graph(cost_matrix, N)
    all_gates = []
    all_thresholds = []
    ###find all the gates needed
    for single_circuit_path in paths_to_circuits:
        single_circuit = pickle.loads(single_circuit_path.read_bytes())
        if isinstance(single_circuit, list):
            single_circuit = single_circuit[0]
        gates_needed = find_gates_needed(single_circuit)

        ## create the graph from the imported circuit

        gate_edges = [tuple(x.split(",")) for x in gates_needed]
        G = nx.Graph()
        for (u, v) in gate_edges:
            G.add_edge(int(u), int(v))

        ## iterate through thresholds to find subgraphisomorphism
        threshold_values = np.linspace(0.02, threshold)
        for val in threshold_values:
            crosstalk_threshold = val
            # J is the machine fully connected graph
            # first we want to make a graph with only those edges that meet the crosstalk criteria
            esmall = [
                (u, v)
                for (u, v, d) in J.edges(data=True)
                if d["weight"] <= crosstalk_threshold
            ]
            J_reduced = nx.Graph()
            J_reduced.add_edges_from(esmall)

            G_run = remap_mono(J_reduced, G)
            if G_run is not None:
                all_thresholds.append(crosstalk_threshold)
                break
            if crosstalk_threshold == threshold:
                print("no isomorphism found, raise your threshold")
                return
        gm_nodes = nx.algorithms.isomorphism.GraphMatcher(G, G_run)
        gm_nodes.is_isomorphic()
        low_crosstalk_map = gm_nodes.mapping

        def low_cross_fcn(qubit):
            return cirq.LineQubit(low_crosstalk_map[qubit.x])

        filename_save = pathlib.Path(save_path, single_circuit_path.name)
        single_circuit = get_circuit_from_pickle(
            single_circuit_path,
            15,
            hardware_map_fn=low_cross_fcn,
            out_pickle_fname=filename_save,
        )
        gates_needed = find_gates_needed(single_circuit)
        for pair in gates_needed:
            if pair not in all_gates:
                all_gates.append(pair)

    G_total = nx.Graph()
    gate_edges = [tuple(x.split(",")) for x in all_gates]
    for (u, v) in gate_edges:
        G_total.add_edge(int(u), int(v))

    nx.draw_circular(G_total, with_labels=True)

    # print("max degree = " + str(len(nx.degree_histogram(G_total)) - 1))
    print("total num of gates = " + str(G_total.number_of_edges()))

    #     qubits_needed = len(max(nx.connected_components(G_run), key=len))
    print("qubits needed = " + str(G_total.number_of_nodes()))

    print("gates needed " + str(G_total.edges))
    print("average crosstalk threshold = {:.3f}".format(np.mean(all_thresholds)))


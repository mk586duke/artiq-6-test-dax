# *****************************************************************
# IonControl:  Copyright 2016 Sandia Corporation
# This Software is released under the GPL license detailed
# in the file "license.txt" in the top-level IonControl directory
# *****************************************************************
"""
Define edges and graphs that represent shuttling ions around an ion trap.

Edges are paths that the ion(s) can take, represented by sequences of voltage
potentials that define the preferred position of the ion.

The graph represents the ways that you can shuttle an ion (or chain) around a
trap using the pre-calculated voltages.

Originally written by Sandia National Labs, updated by UMD Euriqa/LogiQ team.
"""
import logging
import operator
import xml.etree.ElementTree as ElementTree
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union

import more_itertools
import networkx as nx
import numpy as np
from artiq.language.units import us

_LOGGER = logging.getLogger(__name__)


class ShuttleEdge(object):
    """An (indivisible) edge between two steady states for the ion trap.

    Edge represents indivisible path that the ion (chain) takes while moving between
    one steady state (location) and another steady state (location).

    Concretely, these are represented by lines in a voltage solution file.
    This Edge is used to give those lines meaning, such as one line being designated
    the potential used to load ions, and another potential as the region where
    operations are performed on the ions/qubits.
    """

    _state_fields = [
        "start_line",
        "stop_line",
        "idle_count",
        "direction",
        "wait",
        "start_name",
        "stop_name",
        "steps",
        "start_length",
        "stop_length",
        "soft_trigger",
    ]

    def __init__(
        self,
        startName="start",
        stopName="stop",
        startLine=0.0,
        stopLine=1.0,
        idleCount=0,
        direction=0,
        wait=0,
        soft_trigger=0,
    ):
        """Create a ShuttleEdge."""
        self.start_line = startLine
        self.stop_line = stopLine
        self.idle_count = idleCount
        self.direction = direction
        self.wait = wait
        self.start_name = startName.lower()
        self.stop_name = stopName.lower()
        self.soft_trigger = soft_trigger
        self.steps = 0
        self.start_length = 0
        self.stop_length = 0
        self.memory_start_address = -1
        self.memory_stop_address = -1

    def to_xml_element(self, root: ElementTree.Element):
        """Create an XML element from the given ShuttleEdge."""
        state_dictionary = {key: str(getattr(self, key)) for key in self._state_fields}
        edge_element = ElementTree.SubElement(root, "ShuttleEdge", state_dictionary)
        return edge_element

    @staticmethod
    def from_xml_element(element: ElementTree.Element):
        """Create a ShuttleEdge from the given XML element."""
        a = element.attrib
        edge = ShuttleEdge(
            startName=a.get("startName", "start"),
            stopName=a.get("stopName", "stop"),
            startLine=float(a.get("startLine", 0.0)),
            stopLine=float(a.get("stopLine", 1.0)),
            idleCount=float(a.get("idleCount", 0.0)),
            direction=int(a.get("direction", 0)),
            wait=int(a.get("wait", 0)),
            soft_trigger=int(a.get("softTrigger", 0)),
        )
        edge.start_length = int(a.get("startLength", 0))
        edge.stop_length = int(a.get("stopLength", 0))
        edge.steps = float(a.get("steps", 0))
        return edge

    @property
    def weight(self):
        """Get the 'weight' of an edge.

        Equivalent to the number of lines in a file that it takes to traverse this edge.
        """
        return abs(self.stop_line - self.start_line)

    @property
    def time_per_sample(self):
        """Time (in seconds) to output a single sample (set of voltages) to the DAC.

        Can be increased by changing `self.idle_count`, cannot be decreased.
        """
        return 2.06 * us + self.idle_count * 0.02 * us

    @property
    def total_sample_count(self):
        """Get the total number of samples (addresses) used to represent this edge."""
        return abs(self.stop_line - self.start_line) * self.steps + 1

    @property
    def total_time(self):
        """Total time that it takes to execute this edge (in seconds)."""
        return self.total_sample_count * self.time_per_sample

    def line_number_iterator(self):
        """Iterate through all the interpolated line numbers in this edge.

        Example:
            if start_line = 100, stop_line = 102, steps=10:
                yields: [100.0, 100.1, 100.2, ..., 101.9, 102.0]
        """
        yield from np.linspace(
            self.start_line, self.stop_line, int(self.total_sample_count)
        )


class ShuttlingGraphException(Exception):
    """Exception to show that something went wrong with the :class:`ShuttlingGraph`."""

    pass


# High-level edge description, describes shuttling edge
ShuttlePathEdgeDescriptor = NamedTuple(
    "ShuttlePathEdgeDescriptor",
    [
        ("from_node_name", str),
        ("to_node_name", str),
        ("edge", ShuttleEdge),
        ("edge_index", int),
    ],
)
ShuttlePathEdgeDescriptor.__doc__ = (
    """Describe one edge of a path between shuttle nodes."""
)

# Used to describe edges when commanding FPGA to shuttle
ShuttleEdgeDescriptor = NamedTuple(
    "ShuttleEdgeDescriptor",
    [("lookup_index", int), ("reverse_edge", bool), ("immediate_trigger", bool)],
)
ShuttleEdgeDescriptor.__doc__ = """Describe how to execute a shuttling edge.

This refers to an edge in the shuttling graph stored in FPGA memory."""


def path_descriptor_to_fpga_edge_descriptor(
    path_desc: Iterable[ShuttlePathEdgeDescriptor], trigger_immediate: bool = True
) -> Iterable[ShuttleEdgeDescriptor]:
    """Convert a high-level description of shuttling to a FPGA-readable description.

    Converts :class:`ShuttlePathEdgeDescriptor` to :class:`ShuttleEdgeDescriptor`.
    """
    for start, _, edge, index in path_desc:
        yield ShuttleEdgeDescriptor(
            lookup_index=index,
            reverse_edge=start != edge.start_name,
            immediate_trigger=trigger_immediate,
        )


ShuttleEdgeFPGAEntry = NamedTuple(
    "ShuttleEdgeFPGAEntry",
    [("stop_address", int), ("start_address", int), ("idle_counts", int)],
)
ShuttleEdgeFPGAEntry.__doc__ = """Binary representation of a shuttling edge in FPGA memory.

One entry of the shuttling graph table. Comprises the start and stop memory
addresses in an edge, as well as the number of idle counts
(idle counts are between individual output lines comprising the edge??
or idle after executing that edge??)."""


def shuttle_edges_to_fpga_entry(
    edges: Iterable[ShuttleEdge], fpga_number_of_channels: int
) -> Iterable[ShuttleEdgeFPGAEntry]:
    """
    Convert :class:`ShuttleEdge` to be stored on the shuttling FPGA.

    Args:
        edges (Iterable[ShuttleEdge]): ShuttleEdges to convert
        fpga_number_of_channels (int): Number of output DAC channels that
            the FPGA supports. Used to generate the memory stride, i.e. the
            interval in memory addresses between sequential output lines.

    Returns:
        Iterable[ShuttleEdgeFPGAEntry]: Shuttling edges that will be
        stored in a lookup table on the FPGA to represent the shuttling graph.

    """
    address_stride = 2 * fpga_number_of_channels  # interval between lines in FPGA mem
    for edge in edges:
        yield ShuttleEdgeFPGAEntry(
            start_address=edge.memory_start_address * address_stride,
            stop_address=edge.memory_stop_address * address_stride,
            idle_counts=int(edge.idle_count),
        )


class ShuttlingGraph(list):
    """Create a graph that denotes the possible shuttling paths.

    Notes:
        * Must include some sort of sequencing (i.e. iter/list), b/c
            ordering matters, and Edges refer to each other in the graph
            when put on the FPGA. Need to know what edge is in which position
            in memory

    """

    # TODO: trim unused functions, there's too many here that have no real purpose
    def __init__(self, shuttling_edges: Optional[Sequence[ShuttleEdge]] = None):
        """Create the ShuttlingGraph from given :class:`ShuttleEdge`'s."""
        if shuttling_edges is None:
            shuttling_edges = list()
        super().__init__(shuttling_edges)
        self.current_position = None
        self.current_position_name = None
        self.node_lookup = dict()
        self._initialize_graph()
        self._has_changed = True

    def _initialize_graph(self) -> None:
        """Initialize the graph with all nodes and edges."""
        self.shuttling_graph = nx.MultiGraph()
        edge_tuples = (
            (
                edge.start_name,
                edge.stop_name,
                hash(edge),
                {"edge": edge, "weight": edge.weight},
            )
            for edge in self
        )  # Generate edge-tuples from start->stop, keyed with hash, extra data in dict
        self.shuttling_graph.add_edges_from(edge_tuples)
        self.regenerate_node_lookup()

    def regenerate_node_lookup(self) -> None:
        """Recalculate the lookup between lines to node name."""
        self.node_lookup.clear()
        for edge in self:
            self.node_lookup[edge.start_line] = edge.start_name
            self.node_lookup[edge.stop_line] = edge.stop_name

    @property
    def has_changed(self) -> bool:
        """Determine if the graph has changed."""
        return self._has_changed

    @has_changed.setter
    def has_changed(self, value: bool) -> None:
        """Set whether the graph has changed."""
        self._has_changed = value

    def get_node_name(self, line: int) -> str:
        """Get the name of the node at given node."""
        return self.node_lookup.get(line)

    def get_node_line(self, node_name: str) -> int:
        """Get the line number at given node."""
        for line, name in self.node_lookup.items():
            if node_name.lower() == name.lower():
                return line

    def set_position(self, line: int) -> None:
        """Set position in shuttling graph to the output line the DAC is outputting."""
        if self.current_position != line:
            self.current_position = line
            self.current_position_name = self.get_node_name(line)

    def get_matching_position(self, graph: "ShuttlingGraph") -> int:
        """Match node name/position to `graph.current_position`."""
        if not graph:
            return self.current_position  # no change
        # Matching node name. Need to set the corresponding position
        for edge in self:
            if edge.start_name == graph.current_position_name:
                return edge.start_line
            if edge.stop_name == graph.current_position_name:
                return edge.stop_line
        # if graph.currentPosition:
        #    return graph.currentPosition #just use the graph's position
        return self.current_position

    def add_edge(self, edge: ShuttleEdge) -> None:
        """Add an edge to the Shuttling Graph."""
        self._has_changed = True
        self.append(edge)
        self.shuttling_graph.add_edge(
            edge.start_name,
            edge.stop_name,
            key=hash(edge),
            edge=edge,
            weight=edge.weight,
        )
        self.node_lookup[edge.start_line] = edge.start_name
        self.node_lookup[edge.stop_line] = edge.stop_name
        self.set_position(self.current_position)

    def is_valid_edge(self, edge: ShuttleEdge) -> bool:
        """Check if a given edge is in the ShuttlingGraph."""
        return (
            edge.start_line not in self.node_lookup
            or self.node_lookup[edge.start_line] == edge.start_name
        ) and (
            edge.stop_line not in self.node_lookup
            or self.node_lookup[edge.stop_line] == edge.stop_name
        )

    def get_valid_edge(self) -> ShuttleEdge:
        """Return an edge."""
        index = 0
        while self.shuttling_graph.has_node("Start_{0}".format(index)):
            index += 1
        start_name = "Start_{0}".format(index)
        index = 0
        while self.shuttling_graph.has_node("Stop_{0}".format(index)):
            index += 1
        stop_name = "Stop_{0}".format(index)
        index = 0
        start_line = (max(self.node_lookup.keys()) + 1) if self.node_lookup else 1
        stop_line = start_line + 1
        return ShuttleEdge(start_name, stop_name, start_line, stop_line, 0, 0, 0, 0)

    def remove_edge(self, edge_number: int) -> None:
        """Remove edge from the ShuttlingGraph."""
        self._has_changed = True
        edge = self.pop(edge_number)
        self.shuttling_graph.remove_edge(edge.start_name, edge.stop_name, hash(edge))
        if self.shuttling_graph.degree(edge.start_name) == 0:
            self.shuttling_graph.remove_node(edge.start_name)
        if self.shuttling_graph.degree(edge.stop_name) == 0:
            self.shuttling_graph.remove_node(edge.stop_name)
        self.regenerate_node_lookup()
        self.set_position(self.current_position)

    def set_start_name(self, edge_number: int, start_name: str) -> bool:
        """Set the node name of an edge, and corrects the graph to match."""
        self._has_changed = True
        start_name = str(start_name)
        edge = self[edge_number]
        if edge.start_name != start_name:
            self.shuttling_graph.remove_edge(
                edge.start_name, edge.stop_name, key=hash(edge)
            )
            if self.shuttling_graph.degree(edge.start_name) == 0:
                self.shuttling_graph.remove_node(edge.start_name)
            edge.start_name = start_name
            self.shuttling_graph.add_edge(
                edge.start_name,
                edge.stop_name,
                key=hash(edge),
                edge=edge,
                weight=edge.weight,
            )
            self.regenerate_node_lookup()
            self.set_position(self.current_position)
        return True

    def set_stop_name(self, edge_number: int, stop_name: str) -> bool:
        """Set the name of the node that an edge stops at, and correct graph."""
        self._has_changed = True
        stop_name = str(stop_name)
        edge = self[edge_number]
        if edge.stop_name != stop_name:
            self.shuttling_graph.remove_edge(
                edge.start_name, edge.stop_name, key=hash(edge)
            )
            if self.shuttling_graph.degree(edge.stop_name) == 0:
                self.shuttling_graph.remove_node(edge.stop_name)
            edge.stop_name = stop_name
            self.shuttling_graph.add_edge(
                edge.start_name,
                edge.stop_name,
                key=hash(edge),
                edge=edge,
                weight=edge.weight,
            )
            self.regenerate_node_lookup()
            self.set_position(self.current_position)
        return True

    def set_start_line(self, edge_number: int, start_line: int) -> bool:
        """Set the starting line that the given edge starts at.

        Returns `True` if succeeds, and `False` otherwise.
        """
        self._has_changed = True
        edge = self[edge_number]
        if start_line != edge.start_line and (
            start_line not in self.node_lookup
            or self.node_lookup[start_line] == edge.start_name
        ):
            self.node_lookup.pop(edge.start_line)
            edge.start_line = start_line
            self.shuttling_graph.adj[edge.start_name][edge.stop_name][hash(edge)][
                "weight"
            ] = edge.weight
            self.regenerate_node_lookup()
            self.set_position(self.current_position)
            return True
        return False

    def set_stop_line(self, edge_number: int, stop_line: int) -> bool:
        """Set the starting line that the given edge starts at.

        Returns `True` if succeeds, and `False` otherwise.
        """
        self._has_changed = True
        edge = self[edge_number]
        if stop_line != edge.stop_line and (
            stop_line not in self.node_lookup
            or self.node_lookup[stop_line] == edge.stop_name
        ):
            self.node_lookup.pop(edge.stop_line)
            edge.stop_line = stop_line
            self.shuttling_graph.adj[edge.start_name][edge.stop_name][hash(edge)][
                "weight"
            ] = edge.weight
            self.regenerate_node_lookup()
            self.set_position(self.current_position)
            return True
        return False

    def set_idle_count(self, edge_number: int, idle_count: int) -> bool:
        """Set the number of idle clock cycles between DAC outputs for an edge."""
        self._has_changed = True
        self[edge_number].idle_count = idle_count
        return True

    def set_steps(self, edge_number: int, steps: int) -> bool:
        """Set the number of interpolation steps between lines for a given edge.

        Returns `True` if succeeds, and `False` otherwise.
        """
        self._has_changed = True
        self[edge_number].steps = steps
        return True

    def get_path_from_node_to_node(
        self, from_name: Union[str, None], to_name: str
    ) -> Sequence[ShuttlePathEdgeDescriptor]:
        """Return a shortest-possible path between nodes on the shuttling graph."""
        from_name = (
            from_name if from_name else self.get_node_name(float(self.current_position))
        )
        if from_name not in self.shuttling_graph:
            raise ShuttlingGraphException(
                "Shuttling failed, origin '{0}' is not a valid shuttling node".format(
                    from_name
                )
            )
        if to_name not in self.shuttling_graph:
            raise ShuttlingGraphException(
                "Shuttling failed, target '{0}' is not a valid shuttling node".format(
                    to_name
                )
            )
        sp = nx.shortest_path(self.shuttling_graph, from_name, to_name)
        path = list()
        for a, b in more_itertools.pairwise(sp):
            # find the least-weight edge between nodes in the shortest path
            edge = sorted(
                self.shuttling_graph.adj[a][b].values(),
                key=operator.itemgetter("weight"),
            )[0]["edge"]
            path.append(ShuttlePathEdgeDescriptor(a, b, edge, self.index(edge)))

        return path

    def nodes(self) -> Sequence[str]:
        """Return the nodes in the shuttling graph."""
        return self.shuttling_graph.nodes()

    def to_xml_element(self, root: ElementTree.Element) -> ElementTree.Element:
        """Convert the Shuttling Graph to XML format."""
        state_dict = dict(
            (
                (key, str(getattr(self, key)))
                for key in ("current_position", "current_position_name")
                if getattr(self, key) is not None
            )
        )
        graph_root_xml_element = ElementTree.SubElement(
            root, "ShuttlingGraph", attrib=state_dict
        )
        for edge in self:
            edge.to_xml_element(graph_root_xml_element)
        return graph_root_xml_element

    def set_start_length(self, edge_number: int, length: int) -> bool:
        """Set the number of points used to ramp (startup) an given edge.

        Returns `True` if succeeds, and `False` otherwise.
        """
        edge = self[edge_number]
        if length != edge.start_length:
            if length + edge.stop_length < edge.total_sample_count:
                self._has_changed = True
                edge.start_length = int(length)
            else:
                return False
        return True

    def set_stop_length(self, edge_number: int, length: int) -> bool:
        """Set the number of points used to ramp down (stop) an given edge.

        Returns `True` if succeeds, and `False` otherwise.
        """
        edge = self[edge_number]
        if length != edge.stop_length:
            if edge.start_length + length < edge.total_sample_count:
                self._has_changed = True
                edge.stop_length = int(length)
            else:
                return False
        return True

    @staticmethod
    def from_xml_element(element: ElementTree.Element) -> "ShuttlingGraph":
        """Create a :class:`ShuttlingGraph` from an XML element."""
        edgeElementList = element.findall("ShuttleEdge")
        edgeList = [ShuttleEdge.from_xml_element(e) for e in edgeElementList]
        return ShuttlingGraph(edgeList)

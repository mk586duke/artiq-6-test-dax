# *****************************************************************
# IonControl:  Copyright 2016 Sandia Corporation
# This Software is released under the GPL license detailed
# in the file "license.txt" in the top-level IonControl directory
# *****************************************************************
"""GUI to adjust voltages on the Sandia DAC.

The main interface that you use for changing lines and shuttling.
This includes the apply_line type functions, and the shuttle-to functions,
as well as gains.
"""
import logging
import os
import re
import sys
import typing
from xml.dom import minidom
from xml.etree import ElementTree

import more_itertools
import PyQt5.uic
from PyQt5 import QtCore

import euriqabackend.devices.sandia_opalkelly.dac.ShuttlingDefinition as shuttle_def
from .data_models import ShuttleEdgeTableModel
from .modules.ComboBoxDelegate import ComboBoxDelegate
from .modules.DataChanged import DataChangedS
from .modules.Expression import Expression
from .modules.ExpressionValue import ExpressionValue
from .modules.firstNotNone import firstNotNone
from .modules.GuiAppearance import restoreGuiState
from .modules.GuiAppearance import saveGuiState
from .modules.PyqtUtility import updateComboBoxItems
from .modules.quantity import Q
from euriqabackend.devices.sandia_dac.interface import SandiaDACInterface

sys.path.append(os.path.dirname(__file__))

uipath = os.path.join(os.path.dirname(__file__), "VoltageAdjust.ui")
VoltageAdjustForm, VoltageAdjustBase = PyQt5.uic.loadUiType(uipath)
uipath = os.path.join(os.path.dirname(__file__), "ShuttlingEdge.ui")
ShuttlingEdgeForm, ShuttlingEdgeBase = PyQt5.uic.loadUiType(uipath)

_LOGGER = logging.getLogger(__name__)


class Adjust:
    """Window for displaying all adjustments applied to the solution."""

    expression = Expression()
    dataChangedObject = DataChangedS()
    dataChanged = dataChangedObject.dataChanged

    def __init__(self, globalDict=None):
        """Create the Adjustment window."""
        if globalDict is None:
            globalDict = dict()
        self._globalDict = globalDict
        self._line = ExpressionValue(name="line", globalDict=globalDict, value=Q(0.0))
        self._lineValue = self._line.value
        self._lineGain = ExpressionValue(
            name="lineGain", globalDict=globalDict, value=Q(1.0)
        )
        self._globalGain = ExpressionValue(
            name="globalGain", globalDict=globalDict, value=Q(1.0)
        )
        self._line.valueChanged.connect(self.onLineExpressionChanged)
        self._lineGain.valueChanged.connect(self.onLineExpressionChanged)
        self._globalGain.valueChanged.connect(self.onLineExpressionChanged)

    @property
    def globalDict(self):
        """Return the dictionary of global variables."""
        return self._globalDict

    @globalDict.setter
    def globalDict(self, globalDict):
        self._globalDict = globalDict
        self._lineGain.globalDict = globalDict
        self._globalGain.globalDict = globalDict
        self._line.globalDict = globalDict

    @property
    def line(self):
        """Which line the GUI is displaying in the ."""
        return self._lineValue

    @line.setter
    def line(self, value):
        self._lineValue = value

    @property
    def lineGain(self):
        """Return the gain applied to each line."""
        return self._lineGain.value

    @lineGain.setter
    def lineGain(self, value):
        self._lineGain.value = value

    @property
    def globalGain(self):
        """Return the gain of all calculated values (post adjustment & line gain)."""
        return self._globalGain.value

    @globalGain.setter
    def globalGain(self, value):
        self._globalGain.value = value

    @property
    def lineString(self):
        """Return the line value (??) as a string."""
        return self._line.string

    @lineString.setter
    def lineString(self, s):
        self._line.string = s

    @property
    def lineGainString(self):
        """Return a string representation of the gain applied to each line."""
        return self._lineGain.string

    @lineGainString.setter
    def lineGainString(self, s):
        self._lineGain.string = s

    @property
    def globalGainString(self):
        """Return the string value of the global gain applied."""
        return self._globalGain.value

    @globalGainString.setter
    def globalGainString(self, s):
        self._globalGain.string = s

    def __getstate__(self):
        """Save the class state, for pickling."""
        dictcopy = dict(self.__dict__)
        dictcopy.pop("_globalDict", None)
        return dictcopy

    def __setstate__(self, state):
        """Restore the class state from pickling."""
        state.setdefault("_globalDict", dict())
        state.pop("line", None)
        self.__dict__ = state
        if not isinstance(self._line, ExpressionValue):
            self._line = ExpressionValue(
                name="line", globalDict=self._globalDict, value=Q(self._line)
            )
        if not isinstance(self._lineGain, ExpressionValue):
            self._lineGain = ExpressionValue(
                name="lineGain", globalDict=self._globalDict, value=Q(self._lineGain)
            )
        if not isinstance(self._globalGain, ExpressionValue):
            self._globalGain = ExpressionValue(
                name="globalGain",
                globalDict=self._globalDict,
                value=Q(self._globalGain),
            )
        self._lineValue = self._line.value
        self._line.valueChanged.connect(self.onLineExpressionChanged)
        self._lineGain.valueChanged.connect(self.onLineExpressionChanged)
        self._globalGain.valueChanged.connect(self.onLineExpressionChanged)

    def onLineExpressionChanged(self, name, value, string, origin):
        """Call when the expression for a line is change."""
        # pylint: disable=unused-argument
        if name == "line":
            self._lineValue = value
        self.dataChanged.emit(self)


class Settings:
    """Store the shuttling settings."""

    def __init__(self):
        """Wrap the shuttling settings."""
        self.adjust = Adjust()
        self.shuttlingRoute = ""
        self.shuttlingRepetitions = 1

    def __setstate__(self, state):
        """Set the settings state from pickling."""
        self.__dict__ = state
        self.__dict__.setdefault("shuttlingRoute", "")
        self.__dict__.setdefault("shuttlingRepetitions", 1)


class ShuttlingException(Exception):
    """An exception to indicate issue with shuttling."""

    pass


def triplet_iterator(iterable: typing.Iterable):
    """Yield elements in a triplet."""
    # TODO: replace with more_itertools
    i = 0
    while i + 2 < len(iterable):
        yield iterable[i : i + 3]  # noqa: E203
        i += 2


class VoltageAdjust(VoltageAdjustForm, VoltageAdjustBase):
    """Voltage Adustment window."""

    updateOutput = QtCore.pyqtSignal(object, object)
    shuttleOutput = QtCore.pyqtSignal(object)

    def __init__(
        self, config, dac_interface: SandiaDACInterface, globalDict: dict, parent=None
    ):
        """Initialize the settings & values for a Voltage Adjustment window."""
        VoltageAdjustForm.__init__(self)
        VoltageAdjustBase.__init__(self, parent)
        self.config = config
        self.configname = "VoltageAdjust.Settings"
        self.settings = self.config.get(self.configname, Settings())
        self.settings.adjust.globalDict = globalDict
        self.adjust = self.settings.adjust
        self.dac_interface = dac_interface
        self.shuttlingDefinitionFile = None

    @property
    def shuttling_graph(self):
        """Return the current internal shuttling graph."""
        return self.dac_interface.shuttling_graph

    def setupUi(self, parent):
        """Set up and connect the User Interface for Voltage Adjustment window."""
        # pylint: disable=protected-access
        VoltageAdjustForm.setupUi(self, parent)
        self.lineBox.globalDict = self.settings.adjust.globalDict
        self.lineGainBox.globalDict = self.settings.adjust.globalDict
        self.globalGainBox.globalDict = self.settings.adjust.globalDict
        self.lineBox.setExpression(self.adjust._line)
        self.currentLineDisplay.setText(str(self.dac_interface.line_number))
        self.lineGainBox.setExpression(self.adjust._lineGain)
        self.globalGainBox.setExpression(self.adjust._globalGain)
        self.adjust.dataChanged.connect(self.onExpressionChanged)
        self.triggerButton.clicked.connect(self.onTrigger)
        # Shuttling
        self.addEdgeButton.clicked.connect(self.addShuttlingEdge)
        self.removeEdgeButton.clicked.connect(self.removeShuttlingEdge)
        self.shuttleEdgeTableModel = ShuttleEdgeTableModel(
            self.config, self.dac_interface.shuttling_graph
        )
        self.delegate = ComboBoxDelegate()
        self.edgeTableView.setModel(self.shuttleEdgeTableModel)
        self.edgeTableView.setItemDelegateForColumn(8, self.delegate)
        self.edgeTableView.setItemDelegateForColumn(10, self.delegate)
        self.currentPositionLabel.setText(
            firstNotNone(self.dac_interface.shuttling_graph.current_position_name, "")
        )
        # self.dac_interface.shuttling_graph.currentPositionObservable.subscribe(
        #     self.onCurrentPositionEvent
        # )
        # self.dac_interface.shuttling_graph.graphChangedObservable.subscribe(
        #     self.setupGraphDependent
        # )
        self.setupGraphDependent()
        self.uploadDataButton.clicked.connect(self.onUploadData)
        self.uploadEdgesButton.clicked.connect(self.onUploadEdgesButton)
        restoreGuiState(self, self.config.get("VoltageAdjust.GuiState"))
        self.destinationComboBox.currentIndexChanged[str].connect(
            self.onShuttleSequence
        )
        self.shuttlingRouteEdit.setText(" ".join(self.settings.shuttlingRoute))
        self.shuttlingRouteEdit.editingFinished.connect(self.onSetShuttlingRoute)
        self.shuttlingRouteButton.clicked.connect(self.onShuttlingRoute)
        self.repetitionBox.setValue(self.settings.shuttlingRepetitions)
        self.repetitionBox.valueChanged.connect(self.onShuttlingRepetitions)

    def onTrigger(self):
        """Handle a trigger event."""
        self.dac_interface.trigger_shuttling_async()

    def onShuttlingRepetitions(self, value: typing.Union[int, str]):
        """Handle updating the number of shuttling repetitions."""
        self.settings.shuttlingRepetitions = int(value)

    def onSetShuttlingRoute(self):
        """Parse the shuttling route from a string.

        Split the string by -, |, ','.
        """
        self.settings.shuttlingRoute = re.split(
            r"\s*(-|,)\s*", str(self.shuttlingRouteEdit.text()).strip()
        )

    def onShuttlingRoute(self):
        """Shuttle the FPGA DAC along a given route."""
        self.synchronize()
        if self.settings.shuttlingRoute:
            path = list()
            for start, transition, stop in triplet_iterator(
                self.settings.shuttlingRoute
            ):
                if transition == "-":
                    path.extend(
                        self.dac_interface.get_shuttle_path(
                            from_line_or_name=start, to_line_or_name=stop
                        )
                    )
            if path:
                self.shuttleOutput.emit(path * self.settings.shuttlingRepetitions)

    def onUploadData(self):
        """Upload voltage values to FPGA on button press."""
        self.dac_interface.send_voltage_lines_to_fpga()

    def onUploadEdgesButton(self):
        """Upload shuttling graph to FPGA on button press."""
        self.writeShuttleLookup()

    def writeShuttleLookup(self):
        """Write shuttling graph to FPGA."""
        self.dac_interface.send_shuttling_lookup_table()

    def setupGraphDependent(self):
        """Update combo box with all the shuttling nodes."""
        updateComboBoxItems(self.destinationComboBox, self.shuttlingNodes())

    def shuttlingNodes(self):
        """Return all nodes in the shuttling graph."""
        return sorted(self.dac_interface.shuttling_graph.nodes())

    def currentShuttlingPosition(self):
        """Return current position in the shuttling graph."""
        return self.dac_interface.shuttling_graph.current_position_name

    # def onCurrentPositionEvent(self, event):
    #     """Update the current position."""
    #     self.adjust.line = event.line
    #     self.currentLineDisplay.setText(str(self.adjust.line))
    #     self.currentPositionLabel.setText(firstNotNone(event.text, ""))
    #     self.updateOutput.emit(self.adjust, False)

    def onShuttleSequence(self, destination: str, instant=False):
        """Shuttle to a new position  on shuttling graph (`destination`)."""
        self.synchronize()
        destination = str(destination)
        _LOGGER.debug("ShuttleSequence")
        try:
            path = self.dac_interface.get_shuttle_path(
                from_line_or_name=None, to_line_or_name=destination
            )
            if path:
                if instant:
                    # TODO: path not tested. might not work?
                    edge = path[-1].edge
                    toLine = (
                        edge.stop_line
                        if path[-1].from_node_name == edge.start_name
                        else edge.start_line
                    )
                    self.adjust.line = toLine
                    self.currentLineDisplay.setText(str(self.dac_interface.line_number))
                    self.updateOutput.emit(self.adjust, True)
                else:
                    self.shuttleOutput.emit(path)
        except shuttle_def.ShuttlingGraphException:
            _LOGGER.error(
                "Need to initialize DAC & shuttling location before shuttling",
                exc_info=True,
            )
        except RuntimeError:
            _LOGGER.error(
                "Error when shuttling to new location. "
                "Check your current line is a node.",
                exc_info=True,
            )
        else:
            self.adjust.line = self.dac_interface.line_number
            return bool(path)

    def onShuttlingDone(self, currentline):
        """Update current line when shuttling is done."""
        self.currentLineDisplay.setText(str(self.dac_interface.line_number))
        self.adjust.line = currentline
        self.updateOutput.emit(self.adjust, False)

    def addShuttlingEdge(self):
        """Add an edge to the shuttling graph."""
        edge = self.dac_interface.shuttling_graph.get_valid_edge()
        self.shuttleEdgeTableModel.add(edge)

    def removeShuttlingEdge(self):
        """Remove an edge from the shuttling graph."""
        for index in sorted(
            more_itertools.unique_everseen(
                (i.row() for i in self.edgeTableView.selectedIndexes())
            ),
            reverse=True,
        ):
            self.shuttleEdgeTableModel.remove(index)

    def onExpressionChanged(self, value):
        """Update output when a line expression is changed."""
        # pylint: disable=unused-argument
        self.updateOutput.emit(self.adjust, True)

    def onValueChanged(self, attribute, value):
        """Update output when adjustment is changed."""
        setattr(self.adjust, attribute, float(value))
        self.updateOutput.emit(self.adjust, True)

    def setLine(self, line):
        """Set the position in the shuttling graph."""
        self.dac_interface.shuttling_graph.set_position(line)

    def setCurrentPositionLabel(self, event):
        """Update current position text."""
        self.currentPositionLabel.setText(event.text)

    def saveConfig(self):
        """Save adjustments & configuration state."""
        self.config[self.configname] = self.settings
        self.config["VoltageAdjust.GuiState"] = saveGuiState(self)
        root = ElementTree.Element("VoltageAdjust")
        self.dac_interface.shuttling_graph.from_xml_element(root)
        if self.shuttlingDefinitionFile:
            with open(self.shuttlingDefinitionFile, "w") as f:
                f.write(self.prettify(root))

    def prettify(self, elem):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ElementTree.tostring(elem, "utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def loadShuttleDef(self, filename):
        """Load the shuttling graph definitions from a file."""
        # TODO: replace with call to dac_interface.load_shuttle...
        if filename is not None:
            self.shuttlingDefinitionFile = filename
            if os.path.exists(filename):
                tree = ElementTree.parse(filename)
                root = tree.getroot()

                # load pulse definition
                ShuttlingGraphElement = root.find("ShuttlingGraph")
                newGraph = shuttle_def.ShuttlingGraph.from_xml_element(
                    ShuttlingGraphElement
                )
                matchingPos = newGraph.get_matching_position(
                    self.dac_interface.shuttling_graph
                )  # Try to match previous graph node/position
                self.dac_interface.shuttling_graph = newGraph
                self.shuttleEdgeTableModel.setShuttlingGraph(self.shuttling_graph)
                self.currentPositionLabel.setText(
                    firstNotNone(self.shuttling_graph.current_position_name, "")
                )
                # self.shuttling_graph.currentPositionObservable.subscribe(
                #     self.onCurrentPositionEvent
                # )
                # self.shuttling_graph.graphChangedObservable.subscribe(
                #     self.setupGraphDependent
                # )
                self.setupGraphDependent()
                self.setPosition(matchingPos)
                self.updateOutput.emit(
                    self.adjust, True
                )  # Update the output voltages by setting updateHardware to True

    def setPosition(self, line):
        """Set the current position in the shuttling graph.

        Provides a link for pyqtSignal connections to access the shuttlingGraph
        even after loadShuttleDef replaces it.
        """
        # TODO: remove? redundant. setCurrentPosition??
        self.dac_interface.shuttling_graph.set_position(line)

    def synchronize(self):
        """Update the data on the FPGA."""
        if (
            self.dac_interface.shuttling_graph.has_changed
            or not self.dac_interface.is_shuttling_data_valid
        ) and self.dac_interface.hardware.is_open():
            _LOGGER.debug("Synchronizing DAC shuttling & voltage data")
            self.dac_interface.send_voltage_lines_to_fpga()
            self.writeShuttleLookup()
            self.shuttling_graph.has_changed = False

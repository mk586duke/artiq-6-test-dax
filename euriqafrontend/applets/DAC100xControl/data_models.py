# *****************************************************************
# IonControl:  Copyright 2016 Sandia Corporation
# This Software is released under the GPL license detailed
# in the file "license.txt" in the top-level IonControl directory
# *****************************************************************
"""
Table & Data Models used when constructing GUI.

Mostly data tables that are shown in the GUI.

Taken liberally from Peter Maunz/Sandia's PyIonControl software. Mostly edited
to sync with UMD code, and for formatting/syntax.
"""
import logging
import operator
import sys
import typing

from PyQt5 import QtCore
from PyQt5 import QtGui

import euriqabackend.devices.sandia_dac.interface as dac_interface
import euriqabackend.devices.sandia_opalkelly.dac.ShuttlingDefinition as shuttle_defs

_LOGGER = logging.getLogger(__name__)


class VoltageTableModel(QtCore.QAbstractTableModel):
    def __init__(
        self, dac_interface: dac_interface.SandiaDACInterface, *args, parent=None
    ):
        super().__init__(parent, *args)
        self.dac_interface = dac_interface
        self.orderLookup = None
        # self.electrodes, self.aoNums, self.dsubNums, self.outputVoltage
        # arrange the ones in the voltage file first and in the same order
        # self.orderAsVoltageFile()
        self.lastElectrodeOrder = 0
        self.lastLength = 0
        self.voltagesOutOfRange = list()
        self.allVoltagesOkay = False

    def sort(self, column, order):
        if self.dac_interface._pin_names:
            if column == 0:
                self.lastElectrodeOrder = (self.lastElectrodeOrder + 1) % 4
                if self.lastElectrodeOrder == 0:
                    self.orderLookup = list(range(len(self.dac_interface._pin_names)))
                elif self.lastElectrodeOrder == 1:
                    self.orderAsVoltageFile()
                elif self.lastElectrodeOrder in [2, 3]:
                    d = enumerate(self.dac_interface._pin_names)
                    d = sorted(
                        d,
                        key=operator.itemgetter(1),
                        reverse=True if self.lastElectrodeOrder == 3 else False,
                    )
                    self.orderLookup = [operator.itemgetter(0)(t) for t in d]
            else:
                d = enumerate(
                    {
                        0: self.dac_interface._pin_names,
                        1: self.dac_interface.current_output_voltage,
                        2: self.dac_interface._analog_output_channel_numbers,
                        3: [int(val) for val in self.dac_interface._dsub_pin_numbers],
                    }[column]
                )
                d = sorted(
                    d,
                    key=operator.itemgetter(1),
                    reverse=True if order == QtCore.Qt.DescendingOrder else False,
                )
                self.orderLookup = [operator.itemgetter(0)(t) for t in d]
            self.dataChanged.emit(
                self.index(0, 0), self.index(len(self.dac_interface._pin_names) - 1, 3)
            )

    def electrodeIndex(self, name):
        try:
            index = self.dac_interface._pin_names.index(name)
        except Exception as e:
            raise type(e)(str(e) + " for x='{0}'".format(name)).with_traceback(
                sys.exc_info()[2]
            )
        return index

    def orderAsVoltageFile(self):
        if self.dac_interface._pin_names:
            self.orderLookup = list()
            allindices = [False] * len(self.dac_interface._pin_names)
            for name in self.dac_interface.table_header:
                index = self.electrodeIndex(name)
                self.orderLookup.append(index)
                allindices[index] = True
            for index, included in enumerate(allindices):
                if not included:
                    self.orderLookup.append(index)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return (
            len(self.dac_interface._pin_names)
            if self.dac_interface._pin_names is not None
            else 0
        )

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 4

    def onDataChanged(self, x1, y1, x2, y2):
        _LOGGER.debug("VoltageTableModel.onDataChanged %s %s %s %s", x1, y1, x2, y2)
        newLength = len(self.dac_interface._pin_names)
        if newLength > self.lastLength:
            self.beginInsertRows(QtCore.QModelIndex(), self.lastLength, newLength - 1)
            self.endInsertRows()
            self.lastLength = newLength
        if self.orderLookup is None:
            self.orderLookup = list(range(newLength))
        self.voltagesOutOfRange = [False] * newLength
        self.allVoltagesOkay = True
        self.dataChanged.emit(
            self.index(0, y1), self.index(len(self.dac_interface._pin_names) - 1, y2)
        )

    def onDataError(self, boolarray: typing.Sequence[bool]):
        self.voltagesOutOfRange = boolarray
        self.allVoltagesOkay = False
        self.dataChanged.emit(
            self.index(1, 0), self.index(1, len(self.dac_interface._pin_names) - 1)
        )

    def data(self, index, role):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return {
                    0: str(self.dac_interface._pin_names[self.orderLookup[index.row()]])
                    if self.dac_interface._pin_names is not None
                    else None,
                    1: str(
                        self.dac_interface.current_output_voltage[
                            self.orderLookup[index.row()]
                        ]
                    )
                    if self.dac_interface.current_output_voltage is not None
                    else None,
                    2: self.dac_interface._analog_output_channel_numbers[
                        self.orderLookup[index.row()]
                    ]
                    if self.dac_interface._analog_output_channel_numbers is not None
                    else None,
                    3: self.dac_interface._dsub_pin_numbers[
                        self.orderLookup[index.row()]
                    ]
                    if self.dac_interface._dsub_pin_numbers is not None
                    else None,
                }.get(index.column(), None)
            elif role == QtCore.Qt.BackgroundColorRole:
                return {
                    0: QtGui.QColor(QtCore.Qt.white)
                    if self.allVoltagesOkay
                    else QtGui.QColor(QtCore.Qt.red),
                    1: QtGui.QColor(QtCore.Qt.red)
                    if self.voltagesOutOfRange[self.orderLookup[index.row()]]
                    else QtGui.QColor(QtCore.Qt.white),
                }.get(index.column(), None)
        return None

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return {
                    0: "Electrode/Pin Name",
                    1: "Voltage (Current)",
                    2: "Analog Output channel",
                    3: "DSub Connector pin",
                }[section]
            elif orientation == QtCore.Qt.Vertical:
                return section
        return None  # QtCore.QVariant()

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled


class ShuttleEdgeTableModel(QtCore.QAbstractTableModel):
    def __init__(
        self, config, shuttlingGraph: shuttle_defs.ShuttlingGraph, *args, parent=None
    ):
        _LOGGER.debug(
            "ShuttleEdgeTableModel init args: %s, %s, %s, %s",
            config,
            shuttlingGraph,
            args,
            parent,
        )
        super().__init__(parent)
        self.config = config
        self.shuttlingGraph = shuttlingGraph
        self.columnHeaders = [
            "From Name",
            "From Line",
            "To Name",
            "To Line",
            "Steps per line",
            "Idle count",
            "time per sample (s)",
            "total time (s)",
            "Start length",
            "Stop length",
        ]
        self.dataLookup = {
            (QtCore.Qt.DisplayRole, 0): lambda row: self.shuttlingGraph[row].start_name,
            (QtCore.Qt.DisplayRole, 1): lambda row: self.shuttlingGraph[row].start_line,
            (QtCore.Qt.DisplayRole, 2): lambda row: self.shuttlingGraph[row].stop_name,
            (QtCore.Qt.DisplayRole, 3): lambda row: self.shuttlingGraph[row].stop_line,
            (QtCore.Qt.DisplayRole, 4): lambda row: self.shuttlingGraph[row].steps,
            (QtCore.Qt.DisplayRole, 5): lambda row: self.shuttlingGraph[row].idle_count,
            (QtCore.Qt.DisplayRole, 6): lambda row: str(
                self.shuttlingGraph[row].time_per_sample
            ),
            (QtCore.Qt.ToolTipRole, 6): lambda row: str(
                1 / self.shuttlingGraph[row].time_per_sample
            ),
            (QtCore.Qt.DisplayRole, 7): lambda row: str(
                self.shuttlingGraph[row].total_time
            ),
            (QtCore.Qt.DisplayRole, 8): lambda row: str(
                self.shuttlingGraph[row].start_length
            ),
            (QtCore.Qt.DisplayRole, 9): lambda row: str(
                self.shuttlingGraph[row].stop_length
            ),
            (QtCore.Qt.EditRole, 0): lambda row: self.shuttlingGraph[row].start_name,
            (QtCore.Qt.EditRole, 1): lambda row: self.shuttlingGraph[row].start_line,
            (QtCore.Qt.EditRole, 2): lambda row: self.shuttlingGraph[row].stop_name,
            (QtCore.Qt.EditRole, 3): lambda row: self.shuttlingGraph[row].stop_line,
            (QtCore.Qt.EditRole, 4): lambda row: self.shuttlingGraph[row].steps,
            (QtCore.Qt.EditRole, 5): lambda row: self.shuttlingGraph[row].idle_count,
            (QtCore.Qt.EditRole, 8): lambda row: self.shuttlingGraph[row].start_length,
            (QtCore.Qt.EditRole, 9): lambda row: self.shuttlingGraph[row].stop_length,
        }
        self.setDataLookup = {
            (QtCore.Qt.EditRole, 0): shuttle_defs.ShuttlingGraph.set_start_name,
            (QtCore.Qt.EditRole, 1): shuttle_defs.ShuttlingGraph.set_start_line,
            (QtCore.Qt.EditRole, 2): shuttle_defs.ShuttlingGraph.set_stop_name,
            (QtCore.Qt.EditRole, 3): shuttle_defs.ShuttlingGraph.set_stop_line,
            (QtCore.Qt.EditRole, 4): shuttle_defs.ShuttlingGraph.set_steps,
            (QtCore.Qt.EditRole, 5): shuttle_defs.ShuttlingGraph.set_idle_count,
            (QtCore.Qt.EditRole, 8): shuttle_defs.ShuttlingGraph.set_start_length,
            (QtCore.Qt.EditRole, 9): shuttle_defs.ShuttlingGraph.set_stop_length,
        }

    def setShuttlingGraph(self, shuttlingGraph: shuttle_defs.ShuttlingGraph):
        self.beginResetModel()
        self.shuttlingGraph = shuttlingGraph
        self.endResetModel()

    def add(self, edge):
        if self.shuttlingGraph.is_valid_edge(edge):
            self.beginInsertRows(
                QtCore.QModelIndex(), len(self.shuttlingGraph), len(self.shuttlingGraph)
            )
            self.shuttlingGraph.add_edge(edge)
            self.endInsertRows()

    def remove(self, index):
        self.beginRemoveRows(QtCore.QModelIndex(), index, index)
        self.shuttlingGraph.remove_edge(index)
        self.endRemoveRows()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.shuttlingGraph) if self.shuttlingGraph else 0

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.columnHeaders)

    def data(self, index, role):
        if index.isValid():
            return self.dataLookup.get((role, index.column()), lambda row: None)(
                index.row()
            )
        return None

    def setData(self, index, value, role):
        return self.setDataLookup.get(
            (role, index.column()), lambda g, row, value: False
        )(self.shuttlingGraph, index.row(), value)

    def flags(self, index):
        return (
            QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEnabled
            | (
                QtCore.Qt.ItemIsEditable
                if index.column() < 6 or index.column() > 7
                else 0
            )
        )

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.columnHeaders[section]
            else:
                return str(section)
        return None  # QtCore.QVariant()

    def setValue(self, index, value):
        return self.setData(index, value, QtCore.Qt.EditRole)


class VoltageGlobalAdjustTableModel(QtCore.QAbstractTableModel):
    headerDataLookup = ["Adjustment", "Amplitude"]

    def __init__(self, globalAdjustDict: dict, globalDict: dict, *args, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self._global_adjustments_dict = None
        self._adjustment_keys_sorted_by_file_line = None
        self.globalAdjustDict = globalAdjustDict
        _LOGGER.debug(
            "Sorted global adjustment names (#=%i): %s",
            len(self._adjustment_keys_sorted_by_file_line),
            self._adjustment_keys_sorted_by_file_line,
        )
        # scanNames are given as a SortedDict
        defaultBG = QtGui.QColor(QtCore.Qt.white)
        textBG = QtGui.QColor(QtCore.Qt.green).lighter(175)
        self.backgroundLookup = {True: textBG, False: defaultBG}
        self.dataLookup = {
            (
                QtCore.Qt.DisplayRole,
                0,
            ): lambda row: self._adjustment_keys_sorted_by_file_line[row],
            (QtCore.Qt.DisplayRole, 1): lambda row: str(
                self.globalAdjustDict[self._adjustment_keys_sorted_by_file_line[row]][
                    "adjustment_gain"
                ]
            ),
            (QtCore.Qt.EditRole, 1): lambda row: self.globalAdjustDict[
                self._adjustment_keys_sorted_by_file_line[row]
            ]["adjustment_gain"],
        }

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.globalAdjustDict)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2

    def data(self, index, role):
        if index.isValid():
            # _LOGGER.debug("Row/Column: %i, %i", index.row(), index.column())
            return self.dataLookup.get((role, index.column()), lambda row: None)(
                index.row()
            )
        return None

    def setData(self, index, value, role):
        _LOGGER.debug("Setting globaladjust data: %s, %s, %s", index, value, role)
        if index.column() == 1:
            retval = None
            if role == QtCore.Qt.EditRole:
                self._global_adjustments_dict[
                    self._adjustment_keys_sorted_by_file_line[index.row()]
                ]["adjustment_gain"] = (
                    float(value) if not callable(value.m) else value.m
                )
                retval = True
            if role == QtCore.Qt.UserRole:
                self._global_adjustments_dict[
                    self._adjustment_keys_sorted_by_file_line[index.row()]
                ]["adjustment_gain"] = value
                retval = True
            if value is not None:
                # _LOGGER.debug("Emitting change signal: (ind, value) = %s, %s", index, value)
                self.dataChanged.emit(index, index)
            if retval is not None:
                return retval
        return False

    def setValue(self, index, value):
        _LOGGER.debug(
            "setValue: %s. (ind, val) = %s, %s", self.__class__.__name__, index, value
        )
        self.setData(index, value, QtCore.Qt.EditRole)
        self.dataChanged.emit(index, index)

    def flags(self, index):
        return (
            QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
            if index.column() == 0
            else QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEnabled
            | QtCore.Qt.ItemIsEditable
        )

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.headerDataLookup[section]
        return None  # QtCore.QVariant()

    def setGlobalAdjust(self, globalAdjustDict: dict):
        self.beginResetModel()
        self.globalAdjustDict = globalAdjustDict
        self.endResetModel()

    @property
    def globalAdjustDict(self):
        return self._global_adjustments_dict

    @globalAdjustDict.setter
    def globalAdjustDict(self, new_adjusts: dict):
        self._global_adjustments_dict = new_adjusts
        self._adjustment_keys_sorted_by_file_line = sorted(
            self._global_adjustments_dict.keys(),
            key=lambda k: self._global_adjustments_dict[k]["line"],
        )

    def valueRecalculated(self, name):
        _LOGGER.debug("Value recalculated: %s", name)
        index = self.createIndex(self.globalAdjustDict.index(name), 1)
        self.dataChanged.emit(index, index)

    def sort(self, column, order):
        if column == 0 and self.globalAdjustDict:
            self.globalAdjustDict.sort(reverse=order == QtCore.Qt.DescendingOrder)
            self.dataChanged.emit(
                self.index(0, 0), self.index(len(self.globalAdjustDict) - 1, 1)
            )


class VoltageLocalAdjustTableModel(QtCore.QAbstractTableModel):
    filesChanged = QtCore.pyqtSignal(object, object)
    voltageChanged = QtCore.pyqtSignal()
    headerDataLookup = ["Solution", "Amplitude", "Filepath"]

    def __init__(self, localAdjustList, channelDict, globalDict, *args, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self.localAdjustList = localAdjustList
        # scanNames are given as a SortedDict
        self.channelDict = channelDict
        defaultBG = QtGui.QColor(QtCore.Qt.white)
        textBG = QtGui.QColor(QtCore.Qt.green).lighter(175)
        self.backgroundLookup = {True: textBG, False: defaultBG}
        self.dataLookup = {
            (QtCore.Qt.DisplayRole, 0): lambda row: self.localAdjustList[row].name,
            (QtCore.Qt.DisplayRole, 1): lambda row: str(
                self.localAdjustList[row].gainValue
            ),
            (QtCore.Qt.DisplayRole, 2): lambda row: str(self.localAdjustList[row].path),
            (QtCore.Qt.EditRole, 2): lambda row: str(self.localAdjustList[row].path),
            (QtCore.Qt.EditRole, 1): lambda row: self.localAdjustList[row].gain.string,
            (QtCore.Qt.BackgroundColorRole, 1): lambda row: self.backgroundLookup[
                self.localAdjustList[row].gain.hasDependency
            ],
        }

    def add(self, record):
        offset = 0
        while record.name in self.channelDict:
            offset += 1
            record.name = "Adjust_{0}".format(len(self.localAdjustList) + offset)
        self.beginInsertRows(
            QtCore.QModelIndex(), len(self.localAdjustList), len(self.localAdjustList)
        )
        self.localAdjustList.append(record)
        self.channelDict[record.name] = record
        self.endInsertRows()
        return record

    def drop(self, row):
        self.beginRemoveRows(QtCore.QModelIndex(), row, row)
        del self.channelDict[self.localAdjustList[row].name]
        del self.localAdjustList[row]
        self.endRemoveRows()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.localAdjustList)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 3

    def data(self, index, role):
        if index.isValid():
            return self.dataLookup.get((role, index.column()), lambda row: None)(
                index.row()
            )
        return None

    def setData(self, index, value, role):
        if index.column() == 1:
            if role == QtCore.Qt.EditRole:
                self.localAdjustList[index.row()].gain.value = (
                    float(value) if not callable(value.m) else value.m
                )
                return True
            if role == QtCore.Qt.UserRole:
                self.localAdjustList[index.row()].gain.string = value
                return True
        if index.column() == 0:
            if role == QtCore.Qt.EditRole:
                newname = value
                oldname = self.localAdjustList[index.row()].name
                if newname == oldname:
                    return True
                if newname not in self.channelDict:
                    self.localAdjustList[index.row()].name = newname
                    self.channelDict[newname] = self.localAdjustList[index.row()]
                return True
        if index.column() == 2:
            if role == QtCore.Qt.EditRole:
                self.localAdjustList[index.row()].path = value
                self.filesChanged.emit(self.localAdjustList, list())
                return True

        return False

    def setValue(self, index, value):
        self.setData(index, value, QtCore.Qt.EditRole)

    def flags(self, index):
        return (
            QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEnabled
            | QtCore.Qt.ItemIsEditable
        )

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.headerDataLookup[section]
        return None  # QtCore.QVariant()

    def setLocalAdjust(self, localAdjustList):
        self.beginResetModel()
        self.localAdjustList = localAdjustList
        self.endResetModel()

    def valueRecalculated(self, record):
        index = self.createIndex(self.localAdjustList.index(record), 1)
        self.dataChanged.emit(index, index)

    def sort(self, column, order):
        if column == 0:
            self.localAdjustList.sort(reverse=order == QtCore.Qt.DescendingOrder)
            self.dataChanged.emit(
                self.index(0, 0), self.index(len(self.localAdjustList) - 1, 1)
            )

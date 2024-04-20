# *****************************************************************
# IonControl:  Copyright 2016 Sandia Corporation
# This Software is released under the GPL license detailed
# in the file "license.txt" in the top-level IonControl directory
# *****************************************************************
import copy
import logging
import os
from collections import defaultdict
from collections import OrderedDict

import PyQt5.uic
from PyQt5 import QtCore

from .data_models import VoltageGlobalAdjustTableModel
from .modules.MagnitudeSpinBoxDelegate import MagnitudeSpinBoxDelegate
from .modules.Observable import Observable
from .modules.VoltageOutputChannel import VoltageOutputChannel

uipath = os.path.join(os.path.dirname(__file__), "VoltageGlobalAdjust.ui")
VoltageGlobalAdjustForm, VoltageGlobalAdjustBase = PyQt5.uic.loadUiType(uipath)

_LOGGER = logging.getLogger(__name__)


class Settings:
    def __init__(self):
        self.gain = 1.0
        self.gainCache = dict()

    def __setstate__(self, state):
        self.__dict__ = state
        self.__dict__.setdefault("gainCache", dict())


class VoltageGlobalAdjust(VoltageGlobalAdjustForm, VoltageGlobalAdjustBase):
    updateOutput = QtCore.pyqtSignal(object, object)
    outputChannelsChanged = QtCore.pyqtSignal(object)
    _channelParams = {}

    def __init__(self, config, globalDict, parent=None):
        VoltageGlobalAdjustForm.__init__(self)
        VoltageGlobalAdjustBase.__init__(self, parent)
        self.config = config
        self.configname = "VoltageGlobalAdjust.Settings"
        self.settings = self.config.get(self.configname, Settings())
        self.globalAdjustDict = OrderedDict()
        self.myLabelList = list()
        self.myBoxList = list()
        self.historyCategory = "VoltageGlobalAdjust"
        self.adjustHistoryName = None
        self.globalDict = globalDict
        self.adjustCache = self.config.get(self.configname + ".cache", dict())
        self.savedValue = defaultdict(lambda: None)
        self.displayValueObservable = defaultdict(lambda: Observable())
        self._outputChannels = None

    def setupUi(self, parent):
        VoltageGlobalAdjustForm.setupUi(self, parent)
        self.gainBox.setValue(self.settings.gain)
        self.gainBox.valueChanged.connect(self.onGainChanged)
        self.enableGlobalAdjustAutoUpdate.setChecked(False)
        self.tableModel = VoltageGlobalAdjustTableModel(
            self.globalAdjustDict, self.globalDict
        )
        self.tableModel.dataChanged.connect(self.onGlobalAdjustGainChange)
        # On disable, update output voltages.
        self.enableGlobalAdjustAutoUpdate.toggled.connect(self.onGlobalAdjustGainChange)
        self.tableView.setModel(self.tableModel)
        self.tableView.setSortingEnabled(True)  # triggers sorting
        self.delegate = MagnitudeSpinBoxDelegate(self.globalDict)
        self.tableView.setItemDelegateForColumn(1, self.delegate)

    def onGainChanged(self, gain):
        _LOGGER.debug("Global adjust gain changed: %f", gain)
        self.settings.gain = gain
        self.updateOutput.emit(self.globalAdjustDict, self.settings.gain)

    def setupGlobalAdjust(self, name, adjustDict):
        # _LOGGER.debug("setupGlobalAdjust inputs: %s, %s", name, adjustDict)
        # _LOGGER.debug("globalAdjust values: %s", self.globalAdjustDict.values())
        if name != self.adjustHistoryName:
            # Save old values to caches on change.
            self.adjustCache[self.adjustHistoryName] = self.globalAdjustDict.copy()
            self.settings.gainCache[self.adjustHistoryName] = self.settings.gain
            self.settings.gain = self.settings.gainCache.get(name, self.settings.gain)
            if name in self.adjustCache:
                for data in self.adjustCache[name]:
                    if data[0] in adjustDict:
                        adjustDict[data[0]].data = data
            self.adjustHistoryName = name
        self.globalAdjustDict = adjustDict
        for name, adjust in self.globalAdjustDict.items():
            try:
                adjust.valueChanged.connect(
                    self.onValueChanged, QtCore.Qt.UniqueConnection
                )
            except:
                pass
        self.tableModel.setGlobalAdjust(adjustDict)
        self.outputChannelsChanged.emit(self.outputChannels())
        self.gainBox.setValue(self.settings.gain)
        # self.updateOutput.emit(self.globalAdjustDict, self.settings.gain)

    def onValueChanged(self, name, value, string, origin):
        _LOGGER.debug("Value Changed: %s, %s, %s, %s", name, value, string, origin)
        if origin == "recalculate":
            self.tableModel.valueRecalculated(name)
        self.globalAdjustDict[name]._value = float(self.globalAdjustDict[name]._value)
        if self.enableGlobalAdjustAutoUpdate.isChecked():
            self.updateOutput.emit(self.globalAdjustDict, self.settings.gain)
        else:
            _LOGGER.debug("Won't update output. Disabled by checkbox.")

    def onGlobalAdjustGainChange(self):
        """Output a signal with current adjustments & gain.

        Meant to activate when ONE of the adjustment gains changes, can be
        blocked by the checkbox :attr:`enableGlobalAdjustAutoUpdate`.
        """
        _LOGGER.debug("onGlobalAdjustGainChange")
        if self.enableGlobalAdjustAutoUpdate.isChecked():
            _LOGGER.debug("Global adjust auto-update enabled. Updating output")
            self.updateOutput.emit(self.globalAdjustDict, self.settings.gain)
        else:
            _LOGGER.debug("Global adjust auto-update disabled")

    def saveConfig(self):
        self.config[self.configname] = self.settings
        self.adjustCache[self.adjustHistoryName] = [
            v.data for v in list(self.globalAdjustDict.values())
        ]
        self.config[self.configname + ".cache"] = self.adjustCache

    def setValue(self, channel, value):
        _LOGGER.debug("Set value: %s, %s", channel, value)
        self.globalAdjustDict[channel].value = value
        self.globalAdjustDict[channel].setDefaultFunc()
        return value

    def getValue(self, channel):
        return self.globalAdjustDict[channel].value

    def currentValue(self, channel):
        return self.globalAdjustDict[channel].value

    def saveValue(self, channel):
        self.savedValue[channel] = copy.deepcopy(self.globalAdjustDict[channel])

    def restoreValue(self, channel):
        if self.savedValue[channel] is not None:
            self.globalAdjustDict[channel] = self.savedValue[channel]
        return True

    def strValue(self, channel):
        adjust = self.globalAdjustDict[channel]
        return adjust.string if adjust.hasDependency else None

    def setStrValue(self, channel, value):
        pass

    def outputChannels(self):
        self._outputChannels = dict(
            (
                (
                    channelName,
                    VoltageOutputChannel(self, None, channelName, self.globalDict),
                )
                for channelName in self.globalAdjustDict.keys()
            )
        )
        return self._outputChannels

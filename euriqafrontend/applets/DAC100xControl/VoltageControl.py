# *****************************************************************
# IonControl:  Copyright 2016 Sandia Corporation
# This Software is released under the GPL license detailed
# in the file "license.txt" in the top-level IonControl directory
# *****************************************************************
"""Master GUI window for the DAC Voltage control.

Wraps together the sub-elements like VoltageAdjust, VoltageFiles, etc into
one window, and connects all their elements/notifications together.

NOTE: stripped out local adjustments because wasn't used in my new dac_interface code.
    Could maybe add them back in if needed, but I wasn't clear how they're supposed
    to be used.
"""
import logging
import os
import pathlib

import PyQt5.uic
from PyQt5 import QtWidgets

import euriqabackend.devices.sandia_dac.interface as dac_if
from .data_models import VoltageTableModel
from .VoltageAdjust import VoltageAdjust
from .VoltageFiles import VoltageFiles
from .VoltageGlobalAdjust import VoltageGlobalAdjust

# NOTE: local adjustments disabled because not currently
# supported in dac_if.SandiaDACInterface
# from .VoltageLocalAdjust import VoltageLocalAdjust

uipath = os.path.join(os.path.dirname(__file__), "VoltageControl.ui")
VoltageControlForm, VoltageControlBase = PyQt5.uic.loadUiType(uipath)
_LOGGER = logging.getLogger(__name__)


class Settings:
    pass


class VoltageControl(VoltageControlForm, VoltageControlBase):
    def __init__(
        self,
        config,
        dac_interface: dac_if.SandiaDACInterface,
        globalDict: dict = None,
        parent=None,
    ):
        VoltageControlForm.__init__(self)
        VoltageControlBase.__init__(self, parent)
        self.config = config
        self.configname = "VoltageControl.Settings"
        self.settings = self.config.get(self.configname, Settings())

        # Interface should be local class, not remote ARTIQ wrapper.
        assert isinstance(dac_interface, dac_if.SandiaDACInterface)
        self.dac_interface = dac_interface
        self.globalDict = globalDict

    def setupUi(self, parent):
        VoltageControlForm.setupUi(self, parent)
        self.voltageFilesUi = VoltageFiles(self.config)
        self.voltageFilesUi.setupUi(self.voltageFilesUi)
        self.voltageFilesDock.setWidget(self.voltageFilesUi)
        self.adjustUi = VoltageAdjust(self.config, self.dac_interface, self.globalDict)
        self.adjustUi.updateOutput.connect(self.onUpdate)
        self.adjustUi.setupUi(self.adjustUi)
        self.adjustDock.setWidget(self.adjustUi)
        self.globalAdjustUi = VoltageGlobalAdjust(self.config, self.globalDict)
        self.globalAdjustUi.setupUi(self.globalAdjustUi)
        self.globalAdjustUi.updateOutput.connect(self.dac_interface.set_adjustments)
        self.globalAdjustDock.setWidget(self.globalAdjustUi)
        # self.localAdjustUi = VoltageLocalAdjust(self.config, self.globalDict)
        # self.localAdjustUi.setupUi(self.localAdjustUi)
        # # TODO
        # self.localAdjustUi.updateOutput.connect(self.dac_interface.setLocalAdjust)
        # self.localAdjustDock = QtWidgets.QDockWidget("Local Adjust")
        # self.localAdjustDock.setObjectName("_LocalAdjustDock")
        # self.localAdjustDock.setWidget(self.localAdjustUi)
        # self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.localAdjustDock)
        if hasattr(self.settings, "state"):
            self.restoreState(self.settings.state)
        self.voltageFilesUi.loadMapping.connect(self.dac_interface.load_pin_map_file)
        self.voltageFilesUi.loadDefinition.connect(self.onLoadVoltage)
        self.voltageFilesUi.loadGlobalAdjust.connect(self.onLoadGlobalAdjust)
        self.voltageTableModel = VoltageTableModel(self.dac_interface)
        self.tableView.setModel(self.voltageTableModel)
        self.tableView.resizeColumnsToContents()
        self.tableView.resizeRowsToContents()
        # self.localAdjustUi.filesChanged.connect(self.dac_interface.loadLocalAdjust)
        self.dac_interface.voltages_changed.connect(
            self.voltageTableModel.onDataChanged
        )
        # TODO: no dataerror/dataChanged
        self.dac_interface.voltage_errors.connect(self.voltageTableModel.onDataError)
        self.tableView.setSortingEnabled(True)
        self.voltageFilesUi.reloadAll()
        adjust = self.adjustUi.adjust
        # self.dac_interface.loadLocalAdjust(self.localAdjustUi.localAdjustList, list())
        try:
            self.dac_interface.apply_line_async(
                adjust.line, adjust.lineGain, adjust.globalGain
            )
            self.adjustUi.setLine(adjust.line)
        except Exception as e:
            _LOGGER.warning(
                "cannot apply voltages on setup. Ignored for now. Exception: %s",
                e,
                exc_info=True,
            )
        self.adjustUi.shuttleOutput.connect(self.dac_interface.shuttle_along_path_async)
        # self.dac_interface.shuttlingOnLine.connect(self.adjustUi.setPosition)

    def onLoadVoltage(self, voltage_file_path: str, shuttle_graph_path: str):
        if (
            pathlib.Path(voltage_file_path).is_file()
            and pathlib.Path(shuttle_graph_path).is_file()
        ):
            _LOGGER.debug("Loading voltage and shuttling graph files")
            self.dac_interface.load_voltage_file(voltage_file_path)
            self.adjustUi.loadShuttleDef(shuttle_graph_path)
        else:
            _LOGGER.warning(
                "One of (voltage_file, shuttle_graph_file) = (%s, %s) are invalid",
                voltage_file_path,
                shuttle_graph_path,
            )

    def shuttleTo(self, destination, onestep=False):
        _LOGGER.debug("Shuttling to: %s (onestep=%s)", destination, onestep)
        return self.adjustUi.onShuttleSequence(destination)

    # Not enabled in shuttlingGraph
    # def shuttlingNodesObservable(self):
    #     return self.adjustUi.shuttlingGraph.graphChangedObservable

    def currentShuttlingPosition(self):
        _LOGGER.debug("Retrieving current shuttling position")
        return self.adjustUi.currentShuttlingPosition()

    def shuttlingNodes(self):
        _LOGGER.debug("retrieving shuttling nodes")
        return self.adjustUi.shuttlingNodes()

    def synchronize(self):
        _LOGGER.debug("Voltagecontrol synchronizing")
        self.adjustUi.synchronize()

    def onUpdate(self, adjust, updateHardware=True):
        _LOGGER.debug("Updating DAC outputs: %s", adjust)
        try:
            self.dac_interface.apply_line_async(
                float(adjust.line),
                float(adjust.lineGain),
                float(adjust.globalGain),
                updateHardware,
            )
        except ValueError as e:
            _LOGGER.warning("%s", str(e))
        self.adjustUi.setLine(float(adjust.line))

    def onLoadGlobalAdjust(self, path):
        _LOGGER.debug("onloadglobaladjust path: %s, type=%s", path, type(path))
        if not pathlib.Path(path).is_file():
            _LOGGER.warning("global adjust file path is invalid: %s", path)
        try:
            self.dac_interface.load_global_adjust_file(path)
        except RuntimeError:
            _LOGGER.error(
                "Must load pin mapping first. "
                "Load mapping then reload the global adjustments to continue.",
                exc_info=True,
            )
        self.globalAdjustUi.setupGlobalAdjust(
            path, self.dac_interface.adjustment_dictionary
        )

    def saveConfig(self):
        self.settings.state = self.saveState()
        self.settings.isVisible = self.isVisible()
        self.config[self.configname] = self.settings
        self.adjustUi.saveConfig()
        self.globalAdjustUi.saveConfig()
        self.voltageFilesUi.saveConfig()
        # self.localAdjustUi.saveConfig()

    def onClose(self):
        pass

    def closeEvent(self, e):
        self.onClose()

    def onShuttleSequence(self, cont=False):
        self.adjustUi.onShuttleEdge(0)


if __name__ == "__main__":

    class MyMainWindow(QtWidgets.QMainWindow):
        def setCentralWidget(self, widget):
            self.myCentralWidget = widget
            super(MyMainWindow, self).setCentralWidget(widget)

        def closeEvent(self, e):
            self.myCentralWidget.onClose()
            super(MyMainWindow, self).closeEvent(e)

    # NOTE: not allowed because persist not defined/imported.
    # import sys
    # from persist import configshelve

    # with configshelve.configshelve("VoltageControl-test") as config:
    #     app = QtWidgets.QApplication(sys.argv)
    #     MainWindow = MyMainWindow()
    #     ui = VoltageControl(config)
    #     ui.setupUi(ui)
    #     MainWindow.setCentralWidget(ui)
    #     MainWindow.show()
    #     sys.exit(app.exec_())

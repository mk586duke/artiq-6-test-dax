# *****************************************************************
# IonControl:  Copyright 2016 Sandia Corporation
# This Software is released under the GPL license detailed
# in the file "license.txt" in the top-level IonControl directory
# *****************************************************************
"""Load voltages, pin maps, and adjustments from files.

Contains both GUI code and the data structures needed to keep track of these files.
Had to break some dependencies and such to get this working on ARTIQ, mostly
long-term storage of the directories/files used.

TODO:
    * store files/directories used to ARTIQ datasets? (and restore them too)
"""
import logging
import os
import pathlib
import typing

import PyQt5.uic
from PyQt5 import QtCore, QtWidgets

from euriqabackend import _EURIQA_LIB_DIR

from .modules.firstNotNone import firstNotNone
from .modules.PyqtUtility import BlockSignals

uipath = os.path.join(os.path.dirname(__file__), "VoltageFiles.ui")
VoltageFilesForm, VoltageFilesBase = PyQt5.uic.loadUiType(uipath)
DEFAULT_SOLUTIONS_PATH = pathlib.Path(
    _EURIQA_LIB_DIR, "euriqabackend", "voltage_solutions", "dac"
)
DEFAULT_SOLUTIONS_PATH_STR = str(DEFAULT_SOLUTIONS_PATH)

_LOGGER = logging.getLogger(__name__)


class Scan:
    pass


class Files:
    def __init__(self):
        self.mappingFile = None
        self.definitionFile = None
        self.globalFile = None
        self.localFile = None
        self.mappingHistory = dict()
        self.definitionHistory = dict()
        self.globalHistory = dict()
        self.localHistory = dict()


class VoltageFiles(VoltageFilesForm, VoltageFilesBase):
    loadMapping = QtCore.pyqtSignal(str)
    loadDefinition = QtCore.pyqtSignal(str, str)
    loadGlobalAdjust = QtCore.pyqtSignal(str)
    loadLocalAdjust = QtCore.pyqtSignal(str)

    def __init__(self, config, parent=None):
        VoltageFilesForm.__init__(self)
        VoltageFilesBase.__init__(self, parent)
        self.config = config
        self.configname = "VoltageFiles.Files"
        self.files = self.config.get(self.configname, Files())
        # self.lastDir = getProject().configDir
        self.lastDir = DEFAULT_SOLUTIONS_PATH_STR
        self.default_paths = {
            "pin_map": str(DEFAULT_SOLUTIONS_PATH / "reference_files"),
            "voltage_solutions": str(DEFAULT_SOLUTIONS_PATH / "translated_solutions"),
            "global_compensations": str(
                DEFAULT_SOLUTIONS_PATH / "compensation_solutions"
            ),
        }

        def _update_file_dict_from_path(
            hist_dict: dict, path: pathlib.Path, suffix: str
        ) -> typing.Iterable[pathlib.Path]:
            for file_path in path.glob("*{}".format(suffix)):
                hist_dict.update({file_path.name: str(file_path)})

        # Populate history values with default files from given paths.
        for key, path in self.default_paths.items():
            path = pathlib.Path(path)
            if path.exists():
                if key == "pin_map":
                    _update_file_dict_from_path(self.files.mappingHistory, path, ".txt")
                if key == "voltage_solutions":
                    _update_file_dict_from_path(
                        self.files.definitionHistory, path, ".txt"
                    )
                if key == "global_compensations":
                    _update_file_dict_from_path(self.files.globalHistory, path, ".txt")
            else:
                _LOGGER.warning("%s directory does not exist: %s", key, path)
                self.default_paths[key] = self.lastDir

    def setupUi(self, parent):
        VoltageFilesForm.setupUi(self, parent)
        self.mappingCombo.addItems(list(self.files.mappingHistory.keys()))
        self.loadMappingButton.clicked.connect(self.onLoadMapping)
        self.loadDefinitionButton.clicked.connect(self.onLoadDefinition)
        self.loadGlobalButton.clicked.connect(self.onLoadGlobal)
        if self.files.mappingFile is not None:
            _, filename = os.path.split(self.files.mappingFile)
            self.mappingCombo.setCurrentIndex(self.mappingCombo.findText(filename))
        self.definitionCombo.addItems(list(self.files.definitionHistory.keys()))
        if self.files.definitionFile is not None:
            _, filename = os.path.split(self.files.definitionFile)
            self.definitionCombo.setCurrentIndex(
                self.definitionCombo.findText(filename)
            )
        self.globalCombo.addItems(list(self.files.globalHistory.keys()))
        if self.files.globalFile is not None:
            _, filename = os.path.split(self.files.globalFile)
            self.globalCombo.setCurrentIndex(self.globalCombo.findText(filename))
        self.mappingCombo.currentIndexChanged["QString"].connect(self.onMappingChanged)
        self.definitionCombo.currentIndexChanged["QString"].connect(
            self.onDefinitionChanged
        )
        self.globalCombo.currentIndexChanged["QString"].connect(self.onGlobalChanged)
        self.reloadDefinition.clicked.connect(self.onReloadDefinition)
        self.removeDefinition.clicked.connect(self.onRemoveDefinition)
        self.reloadMapping.clicked.connect(self.onReloadMapping)
        self.removeMapping.clicked.connect(self.onRemoveMapping)
        self.reloadGlobal.clicked.connect(self.onReloadGlobal)
        self.removeGlobal.clicked.connect(self.onRemoveGlobal)
        self.reloadAll()

    def onReloadDefinition(self):
        if self.files.definitionFile:
            self.loadDefinition.emit(
                self.files.definitionFile, self.shuttlingDefinitionPath()
            )
            _LOGGER.debug("onReloadDefinition %s", self.files.definitionFile)
        else:
            init_file = self.definitionCombo.currentText()
            _LOGGER.debug("Reloading initial definitions: %s", init_file)
            if init_file in self.files.definitionHistory.keys():
                self.files.definitionFile = self.files.definitionHistory[init_file]
                _LOGGER.debug("onReloadDefinition %s", self.files.definitionFile)
                self.loadDefinition.emit(
                    self.files.definitionFile, self.shuttlingDefinitionPath()
                )
            else:
                _LOGGER.warning("Invalid voltage definition file on startup.")

    def onRemoveDefinition(self):
        self.removeComboUtility(
            self.files.definitionHistory,
            self.files.definitionFile,
            self.definitionCombo,
        )
        _LOGGER.debug("onRemoveMapping %s", self.files.globalFile)

    def onReloadMapping(self):
        if self.files.mappingFile:
            self.loadMapping.emit(self.files.mappingFile)
            _LOGGER.debug("onReloadMapping %s", self.files.mappingFile)
        else:
            # handle startup
            init_mapping_file = self.mappingCombo.currentText()
            _LOGGER.debug("Reloading initial mapping: %s", init_mapping_file)
            if init_mapping_file in self.files.mappingHistory.keys():
                self.files.mappingFile = self.files.mappingHistory[init_mapping_file]
                _LOGGER.debug("onReloadMapping %s", self.files.mappingFile)
                self.loadMapping.emit(self.files.mappingFile)
            else:
                _LOGGER.warning("Invalid pin mapping file on startup.")

    def onRemoveMapping(self):
        self.removeComboUtility(
            self.files.mappingHistory, self.files.mappingFile, self.mappingCombo
        )
        _LOGGER.debug("onRemoveMapping %s", self.files.mappingFile)

    def onReloadGlobal(self):
        if self.files.globalFile:
            self.loadGlobalAdjust.emit(self.files.globalFile)
            _LOGGER.debug("onReloadGlobal %s", self.files.globalFile)
        else:
            init_file = self.globalCombo.currentText()
            _LOGGER.debug("Reloading initial global adjusts: %s", init_file)
            if init_file in self.files.globalHistory.keys():
                self.files.globalFile = self.files.globalHistory[init_file]
                _LOGGER.debug("onReloadglobal %s", self.files.globalFile)
                self.loadGlobalAdjust.emit(self.files.globalFile)
            else:
                _LOGGER.warning("Invalid global adjusts file on startup.")

    def onRemoveGlobal(self):
        self.removeComboUtility(
            self.files.globalHistory, self.files.globalFile, self.globalCombo
        )
        _LOGGER.debug("onRemoveGlobal %s", self.files.globalFile)

    def removeComboUtility(self, history, file, combo):
        """update the combo boxes and remove the entry from the file history.

        Used by onRemoveMapping/Definition/etc.
        """
        for v, k in list(zip(history.values(), history.keys())):
            if v is file:
                history.pop(k)
        # updateComboBoxItems(self.definitionCombo,
        #   list(self.files.definitionHistory.keys()))
        # Cannot use updateComboBox because it would display the first item
        # in the list. We want the combo box to display a blank until the user
        # switches the selection.
        # TODO: incorporate this as an option into updateComboBox
        with BlockSignals(combo):
            combo.clear()
            if history:
                combo.addItems(history)
            combo.setCurrentIndex(-1)

    def reloadAll(self):
        _LOGGER.debug("Reloading all voltage files.")
        self.onReloadMapping()
        self.onReloadDefinition()
        self.onReloadGlobal()

    def onMappingChanged(self, value):
        self.files.mappingFile = self.files.mappingHistory[str(value)]
        _LOGGER.debug("onMappingChanged %s", self.files.mappingFile)
        self.loadMapping.emit(self.files.mappingFile)

    def onDefinitionChanged(self, value):
        self.files.definitionFile = self.files.definitionHistory[str(value)]
        self.loadDefinition.emit(
            self.files.definitionFile, self.shuttlingDefinitionPath()
        )
        _LOGGER.debug("onDefinitionChanged %s", self.files.definitionFile)

    def onGlobalChanged(self, value):
        if value is not None:
            value = str(value)
            if value in self.files.globalHistory:
                self.files.globalFile = self.files.globalHistory[value]
            _LOGGER.debug("onGlobalChanged %s", self.files.globalFile)
            self.loadGlobalAdjust.emit(self.files.globalFile)

    def onLocalChanged(self, value):
        pass

    def onLoadMapping(self):
        _LOGGER.debug("onLoadMapping")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open mapping file:", self.default_paths["pin_map"]
        )
        if path != "":
            filedir, filename = os.path.split(path)
            self.lastDir = filedir
            if filename not in self.files.mappingHistory:
                self.files.mappingHistory[filename] = path
                self.mappingCombo.addItem(filename)
            else:
                self.files.mappingHistory[filename] = path
            self.mappingCombo.setCurrentIndex(self.mappingCombo.findText(filename))
            self.files.mappingFile = path
            self.loadMapping.emit(path)

    def onLoadDefinition(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open definition file:", self.default_paths["voltage_solutions"]
        )
        self.loadVoltageDef(path)

    def loadVoltageDef(self, path):
        # Load the voltage definition given by path.
        if path != "":
            filedir, filename = os.path.split(path)
            self.lastDir = filedir
            if filename not in self.files.definitionHistory:
                self.files.definitionHistory[filename] = path
                self.definitionCombo.addItem(filename)
            else:
                self.files.definitionHistory[filename] = path
            self.definitionCombo.setCurrentIndex(
                self.definitionCombo.findText(filename)
            )
            self.files.definitionFile = path
            self.loadDefinition.emit(path, self.shuttlingDefinitionPath(path))

    def shuttlingDefinitionPath(self, definitionpath=None):
        path = firstNotNone(definitionpath, self.files.definitionFile)
        path = pathlib.Path(path)
        file_dir = path.parent
        if path.with_suffix(".xml").exists():
            return str(path.with_suffix(".xml"))
        else:
            return str(pathlib.Path(file_dir, path.stem + "_shuttling.xml"))

    def onLoadGlobal(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open global adjust file:", self.default_paths["global_compensations"]
        )
        if path != "":
            _LOGGER.debug("selected globaladjust file: %s", path)
            self.files.globalFile = path
            filedir, filename = os.path.split(path)
            self.lastDir = filedir
            if filename not in self.files.globalHistory:
                self.globalCombo.addItem(filename)
            self.files.globalHistory[filename] = path
            self.globalCombo.setCurrentIndex(self.globalCombo.findText(filename))
            self.loadGlobalAdjust.emit(path)
        else:
            _LOGGER.warning("Invalid globaladjust file chosen")

    def onLoadLocal(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open local adjust file:", self.lastDir
        )
        if path != "":
            filedir, filename = os.path.split(path)
            self.lastDir = filedir
            if filename not in self.files.localHistory:
                self.localCombo.addItem(filename)
            self.files.localHistory[filename] = path
            self.localCombo.setCurrentIndex(self.localCombo.find(filename))
            self.files.localFile = path
            self.loadLocalAdjust.emit(path)

    def saveConfig(self):
        self.config[self.configname] = self.files

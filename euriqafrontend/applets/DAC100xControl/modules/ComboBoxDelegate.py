# *****************************************************************
# IonControl:  Copyright 2016 Sandia Corporation
# This Software is released under the GPL license detailed
# in the file "license.txt" in the top-level IonControl directory
# *****************************************************************
from functools import partial

from PyQt5 import QtCore
from PyQt5 import QtWidgets

from .PyqtUtility import BlockSignals


class ComboBoxDelegateMixin(object):
    def createEditor(self, parent, option, index):
        """Create the combo box editor."""
        editor = QtWidgets.QComboBox(parent)
        if hasattr(index.model(), "comboBoxEditable"):
            editor.setEditable(index.model().comboBoxEditable(index))
        choice = (
            index.model().choice(index) if hasattr(index.model(), "choice") else None
        )
        if choice:
            editor.addItems(choice)
        editor.currentIndexChanged["QString"].connect(
            partial(index.model().setValue, index)
        )
        return editor

    def setEditorData(self, editor, index):
        """Set the data in the editor based on the model"""
        value = index.model().data(index, QtCore.Qt.EditRole)
        if value:
            with BlockSignals(editor) as e:
                e.setCurrentIndex(e.findText(value))

    def setModelData(self, editor, model, index):
        """Set the data in the model based on the editor"""
        value = editor.currentText()
        model.setData(index, value, QtCore.Qt.EditRole)


class ComboBoxDelegate(QtWidgets.QStyledItemDelegate, ComboBoxDelegateMixin):
    """Class for combo box editors in models"""

    def __init__(self):
        QtWidgets.QStyledItemDelegate.__init__(self)

    createEditor = ComboBoxDelegateMixin.createEditor
    setEditorData = ComboBoxDelegateMixin.setEditorData
    setModelData = ComboBoxDelegateMixin.setModelData

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

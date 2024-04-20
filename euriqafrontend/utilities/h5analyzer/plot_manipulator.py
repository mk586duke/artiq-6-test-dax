import re
import sys

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import euriqafrontend.fitting as umd_fit
from .h5plotter import h5plotter
from .h5wrapper import h5wrapper


class plot_manipulator(QtWidgets.QMainWindow):
    """Plot Manipulator
    Is passed an h5plotter instance and creates a GUI for a user to
    re-fit data among other things (to be implemented)
    grid layout's layout (rows, cols):
          0      ;       1     ;      2     ;       3      ;     4      ;     5      ;
    xs label     ;        xs text box       ;   ys label   ;        ys text box      ;
    file label   ;        file chooser      ;    space     ; remove file;  add file  ;
    pmt label    ;         pmt chooser      ;fit type label;    fit type chooser     ;
    fit name     ; fit input  ;   fit name  ;   fit input  ;  fit name  ; fit input  ;
    fit name     ; fit input  ;   fit name  ;   fit input  ;  fit name  ; fit input  ;
    fit name     ; fit input  ;   fit name  ;   fit input  ;  fit name  ; fit input  ;
    ;                                     plot                                       ;
                                        ;fits checkbox; re fit button; re plot button;

    """

    # tuple of start and end of rows containing changable fits widgets
    _FITS_WIDGETS_RANGE = (3, 6)
    _NUM_COLS = 6

    def __init__(self, plotter):
        super(plot_manipulator, self).__init__()
        self.plotter = plotter
        self.current_fits = None
        self.selected_pmt = None
        self.selected_filename = None
        self.__param_widgets = None
        self.__param_inputs = {}
        self.__pmt_to_index = {}

        # Initialize window
        self.setWindowTitle("Plot Manipulator")
        self.resize(800, 800)
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        self.grid_layout = QtWidgets.QGridLayout()
        central_widget.setLayout(self.grid_layout)

        # Palettes for setting text color to black and green
        self.__text_color_black = QtGui.QPalette()
        self.__text_color_black.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
        self.__text_color_green = QtGui.QPalette()
        self.__text_color_green.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGreen)

        # Labels and input boxes for setting x dataset name and axis label
        xinput_label = QtWidgets.QLabel("Xs Dataset/Axis Name:")
        self.__xinput_box = QtWidgets.QLineEdit()
        self.__xinput_box.setText("x_values")
        self.__xinput_box.setPalette(self.__text_color_green)
        self.__xinput_box.returnPressed.connect(self.xinput_on_enter)
        self.__xinput_box.textChanged.connect(self.xinput_on_change)

        self.__xlabel_input = QtWidgets.QLineEdit()
        self.__xlabel_input.setText("x label")
        self.__xlabel_input.returnPressed.connect(self.xlabel_on_enter)

        # Labels and input boxes for setting y dataset name and axis label
        yinput_label = QtWidgets.QLabel("Ys Dataset/Axis Name:")
        self.__yinput_box = QtWidgets.QLineEdit()
        self.__yinput_box.setText("avg_thresh")
        self.__yinput_box.setPalette(self.__text_color_green)
        self.__yinput_box.returnPressed.connect(self.yinput_on_enter)
        self.__yinput_box.textChanged.connect(self.yinput_on_change)

        self.__ylabel_input = QtWidgets.QLineEdit()
        self.__ylabel_input.setText("y label")
        self.__ylabel_input.returnPressed.connect(self.ylabel_on_enter)

        # Putting x and y input boxes in place
        self.grid_layout.addWidget(xinput_label, 0, 0)
        self.grid_layout.addWidget(self.__xinput_box, 0, 1)
        self.grid_layout.addWidget(self.__xlabel_input, 0, 2)
        self.grid_layout.addWidget(yinput_label, 0, 3)
        self.grid_layout.addWidget(self.__yinput_box, 0, 4)
        self.grid_layout.addWidget(self.__ylabel_input, 0, 5)

        # Label and Drowpdown Menu for selecting File
        file_select_label = QtWidgets.QLabel("Select File/PMT: ")
        self.__file_select = QtWidgets.QComboBox()
        # self.__file_select.addItem("") # default selection of No Value
        for filename in self.plotter.wrappers:
            self.__file_select.addItem(filename)
        self.__file_select.activated[str].connect(self.on_file_select)
        self.grid_layout.addWidget(file_select_label, 1, 0)
        self.grid_layout.addWidget(self.__file_select, 1, 1, 1, 2)

        # Dropdown Menu for selecting PMT Number
        self.pmt_select = QtWidgets.QComboBox()
        self.pmt_select.activated[str].connect(self.on_pmt_select)
        self.pmt_select.addItem("global")
        self.selected_pmt = "global"
        self.grid_layout.addWidget(self.pmt_select, 1, 3)

        # Add file and remove file buttons
        add_file_button = QtWidgets.QPushButton("Add File", self)
        add_file_button.clicked.connect(self.on_add_file)
        remove_file_button = QtWidgets.QPushButton("Remove File", self)
        remove_file_button.clicked.connect(self.on_remove_file)
        self.grid_layout.addWidget(remove_file_button, 1, 4)
        self.grid_layout.addWidget(add_file_button, 1, 5)

        # Button that updates the plot when clicked
        update_button = QtWidgets.QPushButton("Re-Plot", self)
        update_button.resize(update_button.minimumSizeHint())
        update_button.clicked.connect(plotter.plot)
        self.grid_layout.addWidget(update_button, 7, 5, QtCore.Qt.AlignCenter)

        # Button that re-calculates fits when pressed
        fits_button = QtWidgets.QPushButton("Re-Fit", self)
        self.grid_layout.addWidget(fits_button, 7, 4)
        fits_button.clicked.connect(self.re_fit)

        # Checkbox that enables/disables the plotting of fits
        fits_checkbox = QtWidgets.QCheckBox("Plot Fits")
        fits_checkbox.toggle()
        fits_checkbox.stateChanged.connect(self.fits_checkbox_callback)
        self.grid_layout.addWidget(fits_checkbox, 7, 3, QtCore.Qt.AlignCenter)

        # Add h5plotter object and prioritize it when resizing
        self.grid_layout.addWidget(self.plotter, 6, 0, 1, 6)
        self.grid_layout.setRowStretch(6, 1)

        if len(self.plotter.wrappers) > 0:
            self.plotter.plot()
        self.show()

    # Re fits the selected file using oitg fitting
    def re_fit(self):
        fit_type = umd_fit.positive_gaussian

        if self.selected_pmt == "global":
            # Do fitting with new paramters on all pmts
            print("a")
        else:
            # Fit selected PMT
            print("A")

        print("unimplemented")

    # Adds a new file to wrappers
    def on_add_file(self):
        file_chooser = QtWidgets.QFileDialog()
        file_chooser.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file_chooser.setNameFilter("HDF% Files (*.h5)")
        if file_chooser.exec():
            filenames = file_chooser.selectedFiles()
            wrapper = h5wrapper(filenames[0])
            h5plotter.add_file(self.plotter, wrapper)
            self.__file_select.addItem(wrapper.filename)
            self.on_file_select(wrapper.filename)

    def on_remove_file(self):
        if self.selected_filename is None or self.selected_filename == "":
            return

        self.plotter.remove_file(self.selected_filename)

        # Update file dropdown menu
        self.__file_select.clear()
        if len(self.plotter.wrappers) == 0:
            self.on_file_select("")
            self.selected_filename = None
        else:
            for file_name in self.plotter.wrappers:
                self.__file_select.addItem(file_name)
            self.selected_filename = list(self.plotter.wrappers.keys())[0]
            self.on_file_select(self.selected_filename)

    # Handler for when a file is selected from dropdown menu
    # Deletes old widgets, updates pmts, and
    # adds fits widgets, which are the labels and input boxes for re fitting
    def on_file_select(self, filename):
        if self.current_fits == filename:
            return

        # Delete all previous fits widgets
        for row in range(*self._FITS_WIDGETS_RANGE):
            for col in range(self._NUM_COLS):
                if self.grid_layout.itemAtPosition(row, col) is not None:
                    self.grid_layout.itemAtPosition(row, col).widget().deleteLater()

        # Only display new widgets if filename is not None
        if filename is "":
            self.selected_filename = None
            self.current_fits = None
        else:
            self.selected_filename = filename
            wrapper = self.plotter.wrappers[filename]
            self.current_fits = wrapper.fit_params

            # Loop through all fit params and add widgets for them
            # Prerequisite: Num params <= 9 (max space allocated for fit widgets)
            # Can allocate more space if necessary
            row = self._FITS_WIDGETS_RANGE[0]
            col = 0
            for name in self.current_fits:
                if col >= self._NUM_COLS:
                    col = 0
                    row += 1

                # Add label
                fit_label = QtWidgets.QLabel(name)
                self.grid_layout.addWidget(fit_label, row, col)

                # Create corresponding input text box with default value

                input_box = QtWidgets.QLineEdit()

                # Default value is mean of all values for all pmts since global is selected
                # by default
                input_box.setText(str(np.mean(self.current_fits[name])))

                # Add input box to dictionary and to layout
                self.__param_inputs[name] = input_box
                self.grid_layout.addWidget(input_box, row, col + 1)

                col += 2

        self.update_pmt_menu()

    # Updates pmt menu with contents of new file
    def update_pmt_menu(self):
        self.pmt_select.clear()
        self.selected_pmt = "global"
        self.pmt_select.addItem("global")
        self.__pmt_to_index = {}

        if len(self.plotter.wrappers) > 0:
            for index, pmt in enumerate(
                self.plotter.wrappers[self.selected_filename].pmt_nums
            ):
                pmt_text = "pmt {0}".format(pmt)
                self.pmt_select.addItem(pmt_text)
                self.__pmt_to_index[pmt_text] = index

            self.on_pmt_select("global")

    # Updates values in fit widget text boxes when new pmt is selected
    def on_pmt_select(self, selection):
        self.selected_pmt = selection

        if self.selected_filename is None or self.selected_filename is "":
            return

        # current fits is a dictionary of str : array[np.float64]  =>
        # param_name : array of param values for every pmt
        self.current_fits = self.plotter.wrappers[self.selected_filename].fit_params

        for name, values in self.current_fits.items():
            if self.selected_pmt == "global":
                np.set_printoptions(precision=3)
                val = np.around(np.mean(values), decimals=10)
                if val == 0:
                    val = np.mean(values)

                self.__param_inputs[name].setText(str(val))
            else:
                val = np.around(
                    values[self.__pmt_to_index[self.selected_pmt]], decimals=10
                )
                if val == 0:
                    val = values[self.__pmt_to_index[self.selected_pmt]]

                self.__param_inputs[name].setText(str(val))

    # Handles the event where the fits checkbox's state changes
    def fits_checkbox_callback(self, state):
        if state == QtCore.Qt.Checked:
            self.plotter.plot_fits = True
        else:
            self.plotter.plot_fits = False

        self.plotter.plot()

    # Changes text back to black from green after changing dataset name
    def xinput_on_change(self):
        self.__xinput_box.setPalette(self.__text_color_black)

    # Tries to extract dataset by the name in the text box. Turns text green if successful
    def xinput_on_enter(self):
        try:
            self.plotter.set_xs(dataset_name=self.__xinput_box.text())
            self.__xinput_box.setPalette(self.__text_color_green)
            self.plotter.plot()
        except Exception as e:
            QtWidgets.QMessageBox.about(self, "Error", str(e))

    # Changes x axis label on enter key press
    def xlabel_on_enter(self):
        self.plotter.x_label = self.__xlabel_input.text()
        self.plotter.plot()

    # Changes text back to black from green after changing dataset name
    def yinput_on_change(self):
        self.__yinput_box.setPalette(self.__text_color_black)

    # Tries to extract dataset by the name in the text box. Turns text green if successful
    def yinput_on_enter(self):
        try:
            self.plotter.set_ys(dataset_name=self.__yinput_box.text())
            self.__yinput_box.setPalette(self.__text_color_green)
            self.plotter.plot()
        except Exception as e:
            QtWidgets.QMessageBox.about(self, "Error", str(e))

    # Changes x axis label on enter key press
    def ylabel_on_enter(self):
        self.plotter.y_label = self.__ylabel_input.text()
        self.plotter.plot()

    # Handles window closing event. Exits process if window is closed
    def closeEvent(self, QCloseEvent):
        sys.exit(1)

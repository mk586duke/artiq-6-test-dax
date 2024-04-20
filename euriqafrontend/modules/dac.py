# Copyright 2020 Laird Egan, Chris Monroe Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper for controlling the chip trap voltages.

Provides real-time control of the Sandia 100x DAC.
"""

import logging
import os
import pathlib

import numpy as np
from artiq.language.units import ms, us
from artiq.experiment import BooleanValue
from artiq.experiment import NumberValue
from artiq.experiment import StringValue
from artiq.language.core import kernel, delay
from artiq.language.environment import HasEnvironment

import euriqabackend.voltage_solutions as voltage_solutions

import csv as csv

_LOGGER = logging.getLogger(__name__)

class shuttle_data():
    paths=[]
    timings=[]

class SandiaDAC(HasEnvironment):

    kernel_invariants = {
        "load_to_inspect_path",
        "load_to_inspect_timing",
#        "inspect_to_center_path",
#        "inspect_to_center_timing",
#        "load_to_center_path",
#        "load_to_center_timing",
        "center_to_relaxed_path",
        "center_to_relaxed_timing",
        "relaxed_to_center_path",
        "relaxed_to_center_timing",
        "relaxed_to_boil_load_path",
        "relaxed_to_boil_load_timing",
        "boil_load_to_relaxed_path",
        "boil_load_to_relaxed_timing",
        "relaxed_to_dump_load_path",
        "relaxed_to_dump_load_timing",
        "dump_load_to_relaxed_path",
        "dump_load_to_relaxed_timing",
        "relaxed_to_dump_quantum_path",
        "relaxed_to_dump_quantum_timing",
        "dump_quantum_to_relaxed_path",
        "dump_quantum_to_relaxed_timing",
        "relaxed_to_no_load_path",
        "relaxed_to_no_load_timing",
        "no_load_to_relaxed_path",
        "no_load_to_relaxed_timing",
        "rotated_to_upright_path",
        "rotated_to_upright_timing",
        "upright_to_rotated_path",
        "upright_to_rotated_timing",
        "jump_to_load_path",
        "jump_to_load_timing",
        # "default_to_catcher_path",
        # "default_to_catcher_timing",
        # "catcher_to_default_path",
        # "catcher_to_default_timing",
        # "default_to_edgeimage_path",
        # "default_to_edgeimage_timing",
        # "edgeimage_to_default_path",
        # "edgeimage_to_default_timing",
        "center_to_m100p100_path",
        "center_to_m100p100_timing",
        "n_ions",
        "n_steps_per_swap",
        "shuttle_data"
    }

    DX_compensations = []

    current_ion_swap=0
    current_swap_step=0

    # ion from, step from, ion to, step to
    shuttle_data=[]

    _to_swap_data=[]
    _from_swap_data=[]

    # def build(self, default_voltage_file="new_Trans_Hold_tilted_merge_relaxedLoading_PAD.txt", default_shuttle_file="new_Trans_Hold_tilted_merge_relaxedLoading_PAD.xml")
    def build(self, default_voltage_file="sort_solutions_23.txt",
              default_shuttle_file="sort_solutions_23.xml"):
        # Pull globals for GUI, don't archive in build.

        self.pull_globals(archive=False)

        self.pin_file = self.get_argument(
            "Pin Map File Name",
            StringValue(default="EURIQA_socket_map.txt"),
            group="Sandia DAC",
        )
        self.voltage_file = self.get_argument(
            "Voltage Definition File Name",
            StringValue(default=default_voltage_file),
            group="Sandia DAC",
        )
        self.global_file = self.get_argument(
            "Global Compensation File Name",
            StringValue(default="EURIQA_MHz_units_plus_load.txt"),
            group="Sandia DAC",
        )
        self.shuttle_file = self.get_argument(
            "Shuttling Graph File",
            StringValue(default=default_shuttle_file),
            group="Sandia DAC",
        )

        self.setattr_argument(
            "use_global_voltages", BooleanValue(bool(True)), group="Sandia DAC"
        )

        self.DX_GUI = self.get_argument(
            "DX (V/m)",
            NumberValue(
                default=self.DX_global, unit="", min=-100.0, max=+100.0, ndecimals=3
            ),
            group="Sandia DAC",
        )
        self.DY_GUI = self.get_argument(
            "DY (V/m)",
            NumberValue(
                default=self.DY_global, unit="", min=-100.0, max=+100.0, ndecimals=3
            ),
            group="Sandia DAC",
        )
        self.DZ_GUI = self.get_argument(
            "DZ (V/m)",
            NumberValue(
                default=self.DZ_global, unit="", min=-2000.0, max=+2000.0, ndecimals=3
            ),
            group="Sandia DAC",
        )
        self.X2_offset_GUI = self.get_argument(
            "X2 offset (MHz^2)",
            NumberValue(
                default=self.X2_offset_global, unit="", min=-6.0, max=+6.0, ndecimals=4
            ),
            group="Sandia DAC",
        )
        self.QXZ_offset_GUI = self.get_argument(
            "QXZ offset (MHz^2)",
            NumberValue(
                default=self.QXZ_offset_global, unit="", min=-6.0, max=+6.0, ndecimals=4
            ),
            group="Sandia DAC",
        )
        self.QZZ_offset_GUI = self.get_argument(
            "QZZ offset (MHz^2)",
            NumberValue(
                default=self.QZZ_offset_global, unit="", min=-6.0, max=+6.0, ndecimals=4
            ),
            group="Sandia DAC",
        )
        self.QZY_offset_GUI = self.get_argument(
            "QZY offset (MHz^2)",
            NumberValue(
                default=self.QZY_offset_global, unit="", min=-6.0, max=+6.0, ndecimals=4
            ),
            group="Sandia DAC",
        )
        self.X4_offset_GUI = self.get_argument(
            "X4 offset (MHz^2 / (2.74 um)^2)",
            NumberValue(
                default=self.X4_offset_global, unit="", min=-6e-3, max=+6e-3, ndecimals=5
            ),
            group="Sandia DAC",
        )
        self.QXZ_GUI = self.get_argument(
            "QXZ (MHz^2)",
            NumberValue(default=self.QXZ_global, unit="", min=-6, max=+6, ndecimals=4),
            group="Sandia DAC",
        )
        self.QZZ_GUI = self.get_argument(
            "QZZ (MHz^2)",
            NumberValue(default=self.QZZ_global, unit="", min=-6, max=+6, ndecimals=4),
            group="Sandia DAC",
        )
        self.QZY_GUI = self.get_argument(
            "QZY (MHz^2)",
            NumberValue(default=self.QZY_global, unit="", min=-6, max=+6, ndecimals=4),
            group="Sandia DAC",
        )
        self.QZY_Swap_GUI = self.get_argument(
            "QZY Swap (MHz^2)",
            NumberValue(default=self.QZY_Swap_global, unit="", min=-6, max=+6, ndecimals=4),
            group="Sandia DAC",
        )
        self.X1_GUI = self.get_argument(
            "X1 (2.74 um * MHz^2)",
            NumberValue(default=self.X1_global, unit="", min=-0.2, max=+0.2, ndecimals=4),
            group="Sandia DAC",
        )
        self.X2_GUI = self.get_argument(
            "X2 (MHz^2)",
            NumberValue(default=self.X2_global, unit="", min=-6, max=+6, ndecimals=4),
            group="Sandia DAC",
        )
        self.X3_GUI = self.get_argument(
            "X3 (MHz^2 / (2.74 um))",
            NumberValue(
                default=self.X3_global, unit="", min=-0.014, max=+0.014, ndecimals=4
            ),
            group="Sandia DAC",
        )
        self.X4_GUI = self.get_argument(
            "X4 (MHz^2 / (2.74 um)^2)",
            NumberValue(
                default=self.X4_global, unit="", min=-6e-3, max=+6e-3, ndecimals=5
            ),
            group="Sandia DAC",
        )
        self.X4_Catcher_GUI = self.get_argument(
            "X4 (MHz^2 / (2.74 um)^2)",
            NumberValue(
                default=self.X4_Catcher_global, unit="", min=-6e-3, max=+6e-3, ndecimals=5
            ),
            group="Sandia DAC",
        )
        self.X4_EdgeImage_GUI = self.get_argument(
            "X4 (MHz^2 / (2.74 um)^2)",
            NumberValue(
                default=self.X4_EdgeImage_global, unit="", min=-6e-3, max=+6e-3, ndecimals=5
            ),
            group="Sandia DAC",
        )
        self.center_GUI = self.get_argument(
            "center (um)",
            NumberValue(
                default=self.center_global, unit="", min=-50, max=+50, ndecimals=2
            ),
            group="Sandia DAC",
        )
        self.catcher_holdtime_GUI = self.get_argument(
            "catcher holdtime (ms)",
            NumberValue(
                default=self.catcher_holdtime_global, unit="ms", min=0, max=+1000, ndecimals=2
            ),
            group="Sandia DAC",
        )
        self.force_full_update = self.get_argument(
            "Force DAC Update",
            BooleanValue(default=bool(False)),
            group="Sandia DAC",
        )

        self.DX_file = self.get_argument(
            "DX Compensation File",
            StringValue(default="sort_23_DX.csv"),
            group="Split_Swap",
        )

        self.Swap_file = self.get_argument(
            "Swap waveform File",
            StringValue(default="sort_23_swap.csv"),
            group="Split_Swap",
        )

        self.n_ions = self.get_argument(
            "Number of Ions",
            NumberValue(default=23, unit="", ndecimals=0, scale=1, step=1),
            group="Split_Swap",
        )

        self.n_steps_per_swap = self.get_argument(
            "Number of Steps per Swap",
            NumberValue(default=4, unit="", ndecimals=0, scale=1, step=1),
            group="Split_Swap",
        )

        self.DX_n_lines = self.get_argument(
            "Number of lines to apply DX following split",
            NumberValue(default=20, unit="", ndecimals=0, scale=1, step=1),
            group="Split_Swap",
        )

        self.update_swap_lines = self.get_argument(
            "Update Swap Lines",
            BooleanValue(default=True),
            group="Split_Swap",
        )
        # True: use the swap shuttling file; False: use the autoloader shuttling file.
        self.shuttle_flag = True

        self.module_path = pathlib.Path(voltage_solutions.__file__).resolve().parent

        self.dac_pc = self.get_device("dac_pc_interface")
        self.dac_realtime = self.get_device("realtime_sandia_dac")
        self.setattr_device("core")

        _LOGGER.debug("Done building DAC")

    def prepare(self):
        _LOGGER.debug("Loading Files")
        try:
            fname_pin_map = str(next(self.module_path.glob("**/" + self.pin_file)))
            self.dac_pc.load_pin_map_file(fname_pin_map)
            _LOGGER.debug(f"loaded pin map file {fname_pin_map}")
        except StopIteration:
            raise RuntimeError("Sandia DAC Pin Map File not found")

        try:
            fname_voltages = str(next(self.module_path.glob("**/" + self.voltage_file)))
            self.dac_pc.load_voltage_file(fname_voltages)
            _LOGGER.debug(f"loaded voltages file {fname_voltages}")
        except StopIteration:
            raise RuntimeError("Sandia DAC Voltage File not found")

        try:
            fname_shuttle = str(next(self.module_path.glob("**/" + self.shuttle_file)))
            self.dac_pc.load_shuttling_definitions_file(fname_shuttle)
            _LOGGER.debug(f"loaded shuttle file {fname_shuttle}")
        except StopIteration:
            raise RuntimeError("Sandia DAC Shuttling File not found")

        try:
            fname_compensation = str(next(self.module_path.glob("**/" + self.global_file)))
            self.dac_pc.load_global_adjust_file(fname_compensation)
            _LOGGER.debug(f"loaded compensation file {fname_compensation}")
        except StopIteration:
            raise RuntimeError("Sandia DAC Compensation File not found")

        # Load DX values
        try:
            # trouble in finding this file (07/11)
            fname_DX = str(next(self.module_path.glob("**/" + self.DX_file)))
            # print("DX file name:", fname_DX)
            # fname_DX = "/home/euriqa/git/euriqa-artiq/euriqabackend/voltage_solutions/split_swap_solutions/split_23_DX.csv"
            _LOGGER.debug(f"Loading DX File {fname_DX}")
            self.DX_compensations = []
            with open(fname_DX, 'r') as DX_csv:
                reader = csv.reader(DX_csv, quoting=csv.QUOTE_NONNUMERIC)
                for row in reader:
                    self.DX_compensations.append(row)
                    _LOGGER.info(f"Loaded DX Compensation {row}")
            fname_Swap = str(next(self.module_path.glob("**/" + self.Swap_file)))
            _LOGGER.debug(f"Loading Swap File {fname_Swap}")
            self.Swap = []
            with open(fname_Swap, 'r') as Swap_csv:
                reader = csv.reader(Swap_csv, quoting=csv.QUOTE_NONNUMERIC)
                for row in reader:
                    self.Swap.append(row)
                    _LOGGER.debug(f"Loaded Swap {row}")

        except StopIteration:
            self.DX_compensations = []
            raise RuntimeError("Split DX file not found")

        # in autoloader, we want to skip preparing for swap related paths
        self.load_compensations()
        self.calculate_compensations()
        if self.shuttle_flag:
            self.load_catcher_and_edgeimage()
            self.get_swap_paths()
            self.tweak_default()
            self.tweak_catcher()
            self.tweak_edgeimage()
            if self.update_swap_lines:
                self.tweak_split_swap_global()
                self.tweak_swap()
                if not self.DX_compensations == []:
                    self.tweak_split_DX()
            else:
                _LOGGER.info("Not tweaking split lines.")
        else: # Load solution tweaks are generated in the load experiments
            self.load_path_data()


        _LOGGER.debug("Done preparing DACs")

    def pull_globals(self, archive: bool=True):
        self.DX_global = self.get_dataset("global.Voltages.Offsets.DX", archive=archive)
        self.DY_global = self.get_dataset("global.Voltages.Offsets.DY", archive=archive)
        self.DZ_global = self.get_dataset("global.Voltages.Offsets.DZ", archive=archive)
        self.X2_offset_global = self.get_dataset("global.Voltages.Offsets.X2", archive=archive)
        self.X4_offset_global = self.get_dataset("global.Voltages.Offsets.X4", archive=archive)
        self.QXZ_offset_global = self.get_dataset("global.Voltages.Offsets.QXZ", archive=archive)
        self.QZZ_offset_global = self.get_dataset("global.Voltages.Offsets.QZZ", archive=archive)
        self.QZY_offset_global = self.get_dataset("global.Voltages.Offsets.QZY", archive=archive)
        self.X1_global = self.get_dataset("global.Voltages.X1", archive=archive)
        self.X2_global = self.get_dataset("global.Voltages.X2", archive=archive)
        self.X3_global = self.get_dataset("global.Voltages.X3", archive=archive)
        self.X4_Catcher_global = self.get_dataset("global.Voltages.X4_Catcher", archive=archive)
        self.X4_EdgeImage_global = self.get_dataset("global.Voltages.X4_EdgeImage", archive=archive)
        self.X4_global = self.get_dataset("global.Voltages.X4", archive=archive)
        self.X6_global = self.get_dataset("global.Voltages.X6", archive=archive)
        self.QXZ_global = self.get_dataset("global.Voltages.QXZ", archive=archive)
        self.QZZ_global = self.get_dataset("global.Voltages.QZZ", archive=archive)
        self.QZY_global = self.get_dataset("global.Voltages.QZY", archive=archive)
        self.QZY_Swap_global = self.get_dataset("global.Voltages.QZY_Swap", archive=archive)
        self.center_global = self.get_dataset("global.Voltages.center", archive=archive)
        self.catcher_holdtime_global = self.get_dataset("global.Voltages.Catcher_Holdtime", archive=archive)

    def load_compensations(self):
        self.X6 = 0
        if self.use_global_voltages:
            _LOGGER.debug("Updating compensation voltages using the globals.")
            self.pull_globals(archive=True)
            self.DX = self.DX_global
            self.DY = self.DY_global
            self.DZ = self.DZ_global
            self.X2_offset = self.X2_offset_global
            self.X4_offset = self.X4_offset_global
            self.QZY_offset = self.QZY_offset_global
            self.QZZ_offset = self.QZZ_offset_global
            self.QXZ_offset = self.QXZ_offset_global
            self.X1 = self.X1_global
            self.X2 = self.X2_global
            self.X3 = self.X3_global
            self.X4 = self.X4_global
            self.X4_Catcher = self.X4_Catcher_global
            self.X4_EdgeImage = self.X4_EdgeImage_global
            self.X6 = self.X6_global
            self.QXZ = self.QXZ_global
            self.QZY = self.QZY_global
            self.QZY_Swap = self.QZY_Swap_global
            self.QZZ = self.QZZ_global
            self.QXZ = self.QXZ_global
            self.center = self.center_global
            self.catcher_holdtime = self.catcher_holdtime_global
        else:
            _LOGGER.debug("Updating compensation voltages from the locals.")
            self.DX = self.DX_GUI
            self.DY = self.DY_GUI
            self.DZ = self.DZ_GUI
            self.X2_offset = self.X2_offset_GUI
            self.X4_offset = self.X4_offset_GUI
            self.QZY_offset = self.QZY_offset_GUI
            self.QZZ_offset = self.QZZ_offset_GUI
            self.QXZ_offset = self.QXZ_offset_GUI
            self.X1 = self.X1_GUI
            self.X2 = self.X2_GUI
            self.X3 = self.X3_GUI
            self.X4 = self.X4_GUI
            self.X4_Catcher = self.X4_Catcher_GUI
            self.X4_EdgeImage = self.X4_EdgeImage_GUI
            self.X6 = 0.0
            self.QXZ = self.QXZ_GUI
            self.QZY = self.QZY_GUI
            self.QZY_Swap = self.QZY_Swap_GUI
            self.QZZ = self.QZZ_GUI
            self.QXZ = self.QXZ_GUI
            self.center = self.center_GUI
            self.catcher_holdtime = self.catcher_holdtime_GUI

    def calculate_compensations(self):
        # natural distance unit corresponding to
        # MHz^2 units for the quadrupoles(see OneNote)
        dunit = 2.74  # (um)

        self.DX_adjustment_gain = self.DX / 1e3
        self.DY_adjustment_gain = self.DY / 1e3
        self.DZ_adjustment_gain = -self.QXZ*0.124 + self.DZ / 1e3

        shiftedX1 = self.X1 - (self.X2) * (self.center / dunit) + \
                    self.X3 * (self.center / dunit) ** 2 / 2.0 - \
                    (self.X4) * (self.center / dunit) ** 3 / 6.0

        shiftedX2 = self.X2 - self.X3 * (self.center / dunit) \
                    + (self.X4) * (self.center / dunit) ** 2 / 2.0

        shiftedX3 = self.X3 - (self.X4 * (self.center / dunit))

        self.X1_adjustment_gain = shiftedX1 + self.QXZ * 0.1713
        self.X2_adjustment_gain = shiftedX2 + self.X2_offset
        self.X3_adjustment_gain = shiftedX3
        self.X4_adjustment_gain = self.X4 + self.X4_offset
        self.X4_Catcher_adjustment_gain = self.X4_Catcher + self.X4_offset
        self.X4_EdgeImage_adjustment_gain = self.X4_EdgeImage + self.X4_offset
        self.X6_adjustment_gain = self.X6

        self.QXZ_adjustment_gain = (self.QXZ + self.QXZ_offset)
        self.QZZ_adjustment_gain = (self.QZZ + self.QZZ_offset)
        self.QZY_adjustment_gain = (self.QZY + self.QZY_offset)

    def tweak_split_DX(self):
        _LOGGER.debug(f"Tweaking DX from {self.DX}")
        for i in range(self.n_ions-1):
            for s in range(self.n_steps_per_swap-1):
                for j in range(self.DX_n_lines):
                    line_to_tweak=int(self.dac_pc.shuttling_graph.get_node_line(f'{i}-{s}')+j)
                    # print("Tweak line DX, ", line_to_tweak)
                    if line_to_tweak not in self.dac_pc.tweak_dictionary:
                        self.dac_pc.tweak_dictionary[line_to_tweak] = {}
                    # if(i==5):
                    #     _LOGGER.info(f"Tweaking DX {line_to_tweak}:{i}-{s}:{self.DX_compensations[i][s]}->{(self.DX + self.DX_compensations[i][s]) / 1e3}")
                    # # if (i == 0 and s == 1):
                    #     print("DX comp: ", self.DX_compensations[i][s])
                    #     print("Total DX to be tweaked:", (self.DX + self.DX_compensations[i][s]))
                    self.dac_pc.tweak_dictionary[line_to_tweak]["DX"] = (self.DX + self.DX_compensations[i][s]) / 1e3
                    # Tweak the line with the global X2 offset to compensate X2
                    self.dac_pc.tweak_dictionary[line_to_tweak]["X2"] = self.DX_compensations[i][4]

    def tweak_swap(self):
        _LOGGER.debug(f"Tweaking SWAP lines")
        n_lines_in_swap = 20 # Number of appended lines to use for swapping

        for i in range(self.n_ions -1):
            swap_starting_linenum = int(self.dac_pc.shuttling_graph.get_node_line(f'{i}-3'))
            X2init = self.Swap[i][0]
            A = self.Swap[i][1]
            B = self.Swap[i][2]
            # A = 0
            # B = 0
            for j in range(1, n_lines_in_swap):
                line_to_tweak = swap_starting_linenum + j
                theta = 2*np.pi*j/(n_lines_in_swap-1)

                if line_to_tweak not in self.dac_pc.tweak_dictionary:
                    self.dac_pc.tweak_dictionary[line_to_tweak] = {}

                self.dac_pc.tweak_dictionary[line_to_tweak]["QZY"] = self.QZY_Swap
                self.dac_pc.tweak_dictionary[line_to_tweak]["QXY"] = np.sin(theta) * A
                self.dac_pc.tweak_dictionary[line_to_tweak]["X2_tight"] = X2init-np.cos(theta)*B
                # 2.256-np.cos(theta)*A/1.8

    def tweak_default(self):
        self.tweak_line_all_compensations('Start')

    def tweak_catcher(self):
        # Before tweaking, Catcher line is intialized to be all zeros just like Default line
        # Change the X4 in the tweakdictionary of Catcher without shifting
        self.tweak_line_all_compensations("Catcher")
        line_num = int(self.dac_pc.shuttling_graph.get_node_line("Catcher"))
        self.dac_pc.tweak_dictionary[line_num]['X4'] = self.X4_Catcher_adjustment_gain

    def tweak_edgeimage(self):
        self.tweak_line_all_compensations("EdgeImage")
        line_num = int(self.dac_pc.shuttling_graph.get_node_line("EdgeImage"))
        self.dac_pc.tweak_dictionary[line_num]['X4'] = self.X4_EdgeImage_adjustment_gain

    def tweak_split_swap_global(self):
        shuttle_start_linenum = int(self.dac_pc.shuttling_graph.get_node_line('0-0'))
        shuttle_end_linenum = int(self.dac_pc.shuttling_graph.get_node_line('Start'))
        for line_num in range(shuttle_start_linenum, shuttle_end_linenum):
            if line_num not in self.dac_pc.tweak_dictionary:
                self.dac_pc.tweak_dictionary[line_num] = {}

            self.dac_pc.tweak_dictionary[line_num]['DY'] = self.DY_adjustment_gain
            self.dac_pc.tweak_dictionary[line_num]['DZ'] = self.DZ_adjustment_gain
            self.dac_pc.tweak_dictionary[line_num]['QZY'] = self.QZY_Swap

    def tweak_line_all_compensations(self, line):
        line_num = int(self.dac_pc.shuttling_graph.get_node_line(line))
        self.tweak_line_all_compensations_linenum(line_num)

    def tweak_line_all_compensations_linenum(self, line_num):
        # if line_num < 6966:
        #     print("tweak_line_all_compensations_linenum being applied in the swap region.")
        #
        # else:
        #     print("Global tweaking line ", line_num)

        if line_num not in self.dac_pc.tweak_dictionary:
            self.dac_pc.tweak_dictionary[line_num] = {}

        self.dac_pc.tweak_dictionary[line_num]['DX'] = self.DX_adjustment_gain
        self.dac_pc.tweak_dictionary[line_num]['DY'] = self.DY_adjustment_gain
        self.dac_pc.tweak_dictionary[line_num]['DZ'] = self.DZ_adjustment_gain

        self.dac_pc.tweak_dictionary[line_num]['X1'] = self.X1_adjustment_gain
        self.dac_pc.tweak_dictionary[line_num]['X2'] = self.X2_adjustment_gain
        self.dac_pc.tweak_dictionary[line_num]['X3'] = self.X3_adjustment_gain
        self.dac_pc.tweak_dictionary[line_num]['X4'] = self.X4_adjustment_gain
        self.dac_pc.tweak_dictionary[line_num]['X6'] = self.X6_adjustment_gain

        self.dac_pc.tweak_dictionary[line_num]['QXZ'] = self.QXZ_adjustment_gain
        self.dac_pc.tweak_dictionary[line_num]['QZZ'] = self.QZZ_adjustment_gain
        self.dac_pc.tweak_dictionary[line_num]['QZY'] = self.QZY_adjustment_gain

        # # test swap waveform
        # self.dac_pc.tweak_dictionary[line_num]['QXY'] = 0.6

    def tweak_global_all_compensations(self):
        # Note: must be run after loading voltage file
        num_line = self.dac_pc.num_line
        if(num_line == None):
            _LOGGER.error("Tweak Globals must be called after loading file, Global Compensations not applied")
        else:
            for line_num in range(num_line):
                self.tweak_line_all_compensations_linenum(line_num)

    def get_shuttle_data_stepnum(self, from_ion, from_step, to_ion, to_step):
        from_name = f'{from_ion}-{from_step}'
        to_name = f'{to_ion}-{to_step}'
        return self.get_shuttle_data(from_name, to_name)

    def get_shuttle_data(self, from_name, to_name):
        _LOGGER.debug(f'Path {from_name}->{to_name}')
        shuttle_path = self.dac_pc.get_shuttle_path(from_line_or_name=from_name,
                                                    to_line_or_name=to_name)
        paths, timings = self.dac_realtime.path_to_data(shuttle_path, "immediate")

        data = shuttle_data()
        data.paths = paths
        data.timings = timings
        return data

    def get_swap_paths(self):
        _LOGGER.debug("Getting swap paths")

        for ion_swap in range(self.n_ions-1):
            # #for swap testing with additional node
            # self._to_swap_data += [self.get_shuttle_data('Start', f'{ion_swap}-5')]
            self._to_swap_data += [self.get_shuttle_data('Start', f'{ion_swap}-4')]
            self._from_swap_data += [self.get_shuttle_data(f'{ion_swap}-3', 'Start')]

    @kernel
    def swap(self, swap_ion_index):
        # check ion_number passed in is valid
        if 0 <= swap_ion_index <= self.n_ions - 1:
            # Shuttle to swap
            self.shuttle_path(self._to_swap_data[swap_ion_index])
            # Shuttle from swap
            self.shuttle_path(self._from_swap_data[swap_ion_index])

    @kernel
    def shuttle_path(self, data):
        self.dac_realtime.shuttle_path_sync(data.paths, data.timings)

    @kernel
    def shuttle_to_step(self, ion, step):
        # Verify valid input
        if 0 <= ion < self.n_ions and 0 <= step < self.n_steps_per_swap:
            self.shuttle_path(shuttle_data[self.current_ion_swap][self.current_swap_step][ion][step])
            self.current_ion_swap=ion
            self.current_swap_step=step

            #If you shuttle to the post-swap step, move the cursor to the pre-swap step
            if self.current_swap_step==(self.n_steps_per_swap-1):
                self.current_swap_step = self.n_steps_per_swap-2


    @kernel
    def hw_boil_load(self, boil=True):
        if boil:
            self.dac_realtime.shuttle_path_sync(
                self.relaxed_to_boil_load_path, self.relaxed_to_boil_load_timing
            )
        else:
            self.dac_realtime.shuttle_path_sync(
                self.boil_load_to_relaxed_path, self.boil_load_to_relaxed_timing
            )

    @kernel
    def hw_dump_load(self, dump=True):
        if dump:
            self.dac_realtime.shuttle_path_sync(
                self.relaxed_to_dump_load_path, self.relaxed_to_dump_load_timing
            )
        elif not dump:
            self.dac_realtime.shuttle_path_sync(
                self.dump_load_to_relaxed_path, self.dump_load_to_relaxed_timing
            )

    @kernel
    def hw_dump_quantum(self, dump=True):
        if dump:
            self.dac_realtime.shuttle_path_sync(
                self.relaxed_to_dump_quantum_path, self.relaxed_to_dump_quantum_timing
            )
        elif not dump:
            self.dac_realtime.shuttle_path_sync(
                self.dump_quantum_to_relaxed_path, self.dump_quantum_to_relaxed_timing
            )

    @kernel
    def hw_tilt_quadrupole(self, upright=True):
        if upright:
            self.dac_realtime.shuttle_path_sync(
                self.rotated_to_upright_path, self.rotated_to_upright_timing
            )
        elif not upright:
            self.dac_realtime.shuttle_path_sync(
                self.upright_to_rotated_path, self.upright_to_rotated_timing
            )

    @kernel
    def hw_idle(self, load=False):
        if load:
            self.dac_realtime.shuttle_path_sync(
                self.no_load_to_relaxed_path, self.no_load_to_relaxed_timing
            )
        elif not load:
            self.dac_realtime.shuttle_path_sync(
                self.relaxed_to_no_load_path, self.relaxed_to_no_load_timing
            )

    @kernel
    def hw_shuttle_inspect_to_relaxed(self):
        self.dac_realtime.shuttle_path_sync(
            self.inspect_to_m100p100_path, self.inspect_to_m100p100_timing
        )
        self.dac_realtime.shuttle_path_sync(
            self.m100p100_to_center_path, self.m100p100_to_center_timing
        )
        self.dac_realtime.shuttle_path_sync(
            self.center_to_relaxed_path, self.center_to_relaxed_timing
        )
    @kernel
    def hw_shuttle_relaxed_to_inspect(self):
        self.dac_realtime.shuttle_path_sync(
            self.relaxed_to_center_path, self.relaxed_to_center_timing
        )
        self.dac_realtime.shuttle_path_sync(
            self.center_to_m100p100_path, self.center_to_m100p100_timing
        )
        self.dac_realtime.shuttle_path_sync(
            self.m100p100_to_inspect_path, self.m100p100_to_inspect_timing
        )

    @kernel
    def hw_shuttle_inspect_to_m100p100(self):
        self.dac_realtime.shuttle_path_sync(
            self.inspect_to_m100p100_path, self.inspect_to_m100p100_timing
        )

    @kernel
    def hw_shuttle_m100p100_to_inspect(self):
        self.dac_realtime.shuttle_path_sync(
            self.m100p100_to_inspect_path,self.m100p100_to_inspect_timing
        )

    @kernel
    def hw_shuttle_inspect_to_center(self):
        self.dac_realtime.shuttle_path_sync(
            self.inspect_to_center_path, self.inspect_to_center_timing,
        )

    @kernel
    def hw_dump_inspect(self):
        """Dump the ion that is currently being inspected pre-merge w/ chain."""
        self.dac_realtime.shuttle_path_sync(
            self.inspect_to_junction_path, self.inspect_to_junction_timing
        )
        delay(10*ms)
        self.dac_realtime.shuttle_path_sync(
           self.junction_dump_path, self.junction_dump_timing
        )

    @kernel
    def hw_shuttle_load_to_inspect(self):
        self.dac_realtime.shuttle_path_sync(
            self.relaxed_to_center_path, self.relaxed_to_center_timing
        )
        self.dac_realtime.shuttle_path_sync(
            self.jump_to_load_path, self.jump_to_load_timing
        )
        self.dac_realtime.shuttle_path_sync(
            self.load_to_inspect_path, self.load_to_inspect_timing
        )

    @kernel
    def hw_shuttle_load_to_m100p100(self):
        self.dac_realtime.shuttle_path_sync(
            self.relaxed_to_center_path, self.relaxed_to_center_timing
        )
        self.dac_realtime.shuttle_path_sync(
            self.jump_to_load_path, self.jump_to_load_timing
        )
        self.dac_realtime.shuttle_path_sync(
            self.load_to_m100p100_path, self.load_to_m100p100_timing
        )

    @kernel
    def hw_shuttle_m100p100_to_relaxed(self):
        self.dac_realtime.shuttle_path_sync(
            self.m100p100_to_relaxed_path, self.m100p100_to_relaxed_timing
        )

    @kernel
    def hw_shuttle_center_to_relaxed(self):
        self.dac_realtime.shuttle_path_sync(
            self.center_to_relaxed_path, self.center_to_relaxed_timing
        )

    @kernel
    def default_to_catcher(self):
        self.dac_realtime.shuttle_path_sync(
            self.default_to_catcher_path, self.default_to_catcher_timing
        )

    @kernel
    def catcher_to_default(self):
        self.dac_realtime.shuttle_path_sync(
            self.catcher_to_default_path, self.catcher_to_default_timing
        )

    @kernel
    def default_to_edgeimage(self):
        self.dac_realtime.shuttle_path_sync(
            self.default_to_edgeimage_path, self.default_to_edgeimage_timing
        )

    @kernel
    def edgeimage_to_default(self):
        self.dac_realtime.shuttle_path_sync(
            self.edgeimage_to_default_path, self.edgeimage_to_default_timing
        )

    @kernel
    def catcher(self):
        self.hw_default_to_catcher()
        delay(self.catcher_holdtime)
        self.hw_catcher_to_default()
        delay(10 * us)

    def load_catcher_and_edgeimage(self):
        # CATCHER: START -> CATCHER
        self.default_to_catcher_path, self.default_to_catcher_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Start",
                to_line_or_name="Catcher",
            ),
            "immediate",
        )

        _LOGGER.debug(
            "SHUTTLE: START -> CATCHER --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.default_to_catcher_timing)),
        )

        # CATCHER: CATCHER -> DEFAULT
        self.catcher_to_default_path, self.catcher_to_default_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Catcher",
                to_line_or_name="Start",
            ),
            "immediate",
        )

        _LOGGER.debug(
            "SHUTTLE: CATCHER -> START --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.catcher_to_default_timing)),
        )

        # EdgeImage: START -> EdgeImage
        self.default_to_edgeimage_path, self.default_to_edgeimage_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Start",
                to_line_or_name="EdgeImage",
            ),
            "immediate",
        )

        _LOGGER.debug(
            "SHUTTLE: START -> EDGEIMAGE --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.default_to_edgeimage_timing)),
        )

        # EdgeImage: EdgeImage -> START
        self.edgeimage_to_default_path, self.edgeimage_to_default_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="EdgeImage",
                to_line_or_name="Start",
            ),
            "immediate",
        )

        _LOGGER.debug(
            "SHUTTLE: EDGEIMAGE -> START --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.edgeimage_to_default_timing)),
        )

    def load_path_data(self):

        # SHUTTLE: LOAD -> M100P100
        self.load_to_m100p100_path, self.load_to_m100p100_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Loading", to_line_or_name="Pos.m100.p100"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: LOADING -> M100P100 --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.load_to_m100p100_timing)),
        )

        # SHUTTLE: M100P100 -> RELAXED
        self.m100p100_to_relaxed_path, self.m100p100_to_relaxed_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Pos.m100.p100.2", to_line_or_name="QuantumRelaxed_LoadOn"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: M100P100 -> RELAXED --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.m100p100_to_relaxed_timing)),
        )

        # SHUTTLE: LOAD -> INSPECT
        self.load_to_inspect_path, self.load_to_inspect_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Loading", to_line_or_name="Pos.m200.p0"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: LOADING -> INSPECT --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.load_to_inspect_timing)),
        )

        # SHUTTLE: INSPECT -> M100P100
        self.inspect_to_m100p100_path, self.inspect_to_m100p100_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Pos.m200.p0", to_line_or_name="Pos.m100.p100"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: INSPECT -> M100P100 --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.inspect_to_m100p100_timing)),
        )

        # SHUTTLE: M100P100 -> INSPECT
        self.m100p100_to_inspect_path, self.m100p100_to_inspect_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Pos.m100.p100", to_line_or_name="Pos.m200.p0"
            ),
            "immediate"
        )
        _LOGGER.debug(
            "SHUTTLE: LOADING -> INSPECT --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.m100p100_to_inspect_timing)),
        )

        # SHUTTLE: Inspect -> Center
        # self.inspect_to_center_path, self.inspect_to_center_timing = self.dac_realtime.path_to_data(
        #     self.dac_pc.get_shuttle_path(
        #         from_line_or_name="Pos.m200.p0", to_line_or_name="Center"
        #     ),
        #     "immediate",
        # )
        # _LOGGER.debug(
        #     "SHUTTLE: INSPECT -> CENTER --- Total time = %f ms",
        #     1000 * self.core.mu_to_seconds(np.sum(self.inspect_to_center_timing)),
        # )

        # SHUTTLE: M100P100 -> CENTER
        self.m100p100_to_center_path, self.m100p100_to_center_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Pos.m100.p100.2", to_line_or_name="Center"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: M100P100 -> CENTER --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.m100p100_to_center_timing)),
        )
        # SHUTTLE: CENTER -> M100P100
        self.center_to_m100p100_path, self.center_to_m100p100_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Center", to_line_or_name="Pos.m100.p100.2"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: M100P100 -> CENTER --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.m100p100_to_center_timing)),
        )

        # SHUTTLE: INSPECT -> JUNCTION
        self.inspect_to_junction_path, self.inspect_to_junction_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Pos.m200.p0", to_line_or_name="JunctionS"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: INSPECT -> JUNCTIONS --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.inspect_to_junction_timing)),
        )

        # SHUTTLE: JUNCTION -> INSPECT
        self.junction_to_inspect_path, self.junction_to_inspect_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="JunctionS", to_line_or_name="Pos.m200.p0"
            ),
            "immediate",
        )

        # SHUTTLE: JUNCTION DUMP
        self.junction_dump_path, self.junction_dump_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="JunctionS.Eject.Start", to_line_or_name="JunctionS.Eject.End"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: INSPECT -> JUNCTIONS --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.junction_dump_timing)),
        )

        # SHUTTLE: LOAD -> CENTER
        # self.load_to_center_path, self.load_to_center_timing = self.dac_realtime.path_to_data(
        #     self.dac_pc.get_shuttle_path(
        #         from_line_or_name="Loading", to_line_or_name="Center"
        #     ),
        #     "immediate",
        # )
        # _LOGGER.debug(
        #     "SHUTTLE: LOADING -> CENTER --- Total time = %f ms",
        #     1000 * self.core.mu_to_seconds(np.sum(self.load_to_center_timing)),
        # )

        # SHUTTLE: CENTER -> RELAXED
        self.center_to_relaxed_path, self.center_to_relaxed_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Center", to_line_or_name="QuantumRelaxed_LoadOn"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: CENTER -> RELAXED --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.center_to_relaxed_timing)),
        )

        # SHUTTLE: RELAXED -> CENTER
        self.relaxed_to_center_path, self.relaxed_to_center_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_LoadOn", to_line_or_name="Center"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: RELAXED -> CENTER --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.relaxed_to_center_timing)),
        )

        # SHUTTLE: RELAXED -> BOIL_LOAD
        self.relaxed_to_boil_load_path, self.relaxed_to_boil_load_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_LoadOn",
                to_line_or_name="QuantumRelaxed_LoadBoiloff",
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: RELAXED -> BOIL_LOAD --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.relaxed_to_boil_load_timing)),
        )

        # SHUTTLE: BOIL_LOAD -> RELAXED
        self.boil_load_to_relaxed_path, self.boil_load_to_relaxed_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_LoadBoiloff",
                to_line_or_name="QuantumRelaxed_LoadOn",
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: BOIL_LOAD -> RELAXED --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.boil_load_to_relaxed_timing)),
        )

        # SHUTTLE: RELAXED -> DUMP LOAD (for dumping chains in load slot)
        self.relaxed_to_dump_load_path, self.relaxed_to_dump_load_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_LoadOn",
                to_line_or_name="QuantumRelaxed_LoadEject",
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: RELAXED -> DUMP QUANTUM --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.relaxed_to_dump_load_timing)),
        )

        # SHUTTLE: DUMP_LOAD -> RELAXED (for dumping chains in load slot)
        self.dump_load_to_relaxed_path, self.dump_load_to_relaxed_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_LoadEject",
                to_line_or_name="QuantumRelaxed_LoadOn",
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: DUMP_QUANTUM -> RELAXED --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.dump_load_to_relaxed_timing)),
        )

        # SHUTTLE: RELAXED -> DUMP QUANTUM (for dumping chains in quantum)
        self.relaxed_to_dump_quantum_path, self.relaxed_to_dump_quantum_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_LoadOn", to_line_or_name="Eject"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: RELAXED -> DUMP QUANTUM --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.relaxed_to_dump_quantum_timing)),
        )

        # SHUTTLE: DUMP_QUANTUM -> RELAXED (for dumping chains in quantum)
        self.dump_quantum_to_relaxed_path, self.dump_quantum_to_relaxed_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="Eject", to_line_or_name="QuantumRelaxed_LoadOn"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: DUMP_QUANTUM -> RELAXED --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.dump_quantum_to_relaxed_timing)),
        )

        # SHUTTLE: RELAXED LOAD ON -> RELAXED LOAD OFF
        self.relaxed_to_no_load_path, self.relaxed_to_no_load_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_LoadOn",
                to_line_or_name="QuantumRelaxed_ZeroLoad",
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: RELAXED LOAD ON -> RELAXED LOAD OFF --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.relaxed_to_no_load_timing)),
        )

        # SHUTTLE: RELAXED LOAD OFF -> RELAXED LOAD ON
        self.no_load_to_relaxed_path, self.no_load_to_relaxed_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_ZeroLoad",
                to_line_or_name="QuantumRelaxed_LoadOn",
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: RELAXED LOAD OFF -> RELAXED LOAD ON --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.no_load_to_relaxed_timing)),
        )

        ########################
        # SHUTTLE: ROTATED -> UPRIGHT
        self.rotated_to_upright_path, self.rotated_to_upright_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxed_ZeroLoad",
                to_line_or_name="QuantumRelaxedUpright_ZeroLoad",
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: ROTATED -> UPRIGHT --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.rotated_to_upright_timing)),
        )

        # SHUTTLE: UPRIGHT -> ROTATED
        self.upright_to_rotated_path, self.upright_to_rotated_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="QuantumRelaxedUpright_ZeroLoad",
                to_line_or_name="QuantumRelaxed_ZeroLoad",
            ),
            "immediate",
        )
        _LOGGER.debug(
            "SHUTTLE: UPRIGHT -> ROTATED --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.upright_to_rotated_timing)),
        )

        # JUMP TO: LOADING (for jumping to load solution)
        self.jump_to_load_path, self.jump_to_load_timing = self.dac_realtime.path_to_data(
            self.dac_pc.get_shuttle_path(
                from_line_or_name="LoadingJump", to_line_or_name="Loading"
            ),
            "immediate",
        )
        _LOGGER.debug(
            "JUMP TO: LOADING --- Total time = %f ms",
            1000 * self.core.mu_to_seconds(np.sum(self.jump_to_load_timing)),
        )

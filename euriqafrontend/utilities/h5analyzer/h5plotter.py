import sys
from typing import List

import matplotlib.colors as clr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .h5wrapper import h5wrapper

# Flattens 3D Array to 2D (pmts, steps, shots) => (pmt, steps & shots) for example
def _flatten3d(arr):
    flattened = np.empty((arr.shape[0], arr.shape[1] * arr.shape[2]))
    for i in range(arr.shape[0]):
        flattened[i] = arr[i].flatten()
    return flattened


class h5plotter(FigureCanvas):
    """ H5Plotter
    is passed up to 4 h5wrappers and plots specified data and fits in them on one plot.
    Defaults to plotting x = data.x_values and y = data.avg_thresh, all parameters are changed
    by the user through the plot_manipulator object/window
    TODO:
        * add plot updating functions with manipulator
        * add legend creation
    """

    def __init__(self, wrappers: List[h5wrapper]) -> None:
        super(h5plotter, self).__init__(plt.figure())
        if len(wrappers) > 4:
            raise Exception("Too Many Files")

        # Store wrappers in dictionary as filename : wrapper_obj
        self.wrappers = {}
        for wrapper in wrappers:
            self.wrappers[wrapper.filename] = wrapper

        self.x_label = None
        self.y_label = None
        self.colors = None
        self.set_xs(dataset_name="x_values")
        self.set_ys(dataset_name="avg_thresh")
        self.plot_fits = True
        self.style = dict(marker=".", ms=9, linestyle="None", markeredgecolor="k")
        self.figure = plt.figure()
        plt.axes().ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))

        # Define the color groups. This is used when multiple files are being plotted
        # Each file is assigned a color group, which is a range of RGB values that
        # All the lines/points in that file will fit in
        self.color_groups = [
            [(0.957, 0.057, 0.057), (0.898, 0.783, 0.057)],  # red to orange
            [(0.051, 0.862, 0.957), (0.086, 0.055, 0.957)],  # cyan to blue
            [(0.580, 0.957, 0.051), (0.051, 0.957, 0.427)],  # yellow to green
            [(0.533, 0.051, 0.957), (0.957, 0.051, 0.518)],  # violet to pink
        ]

        plt.ion()
        plt.tight_layout()
        self.figure.tight_layout()

    def add_file(self, wrapper):
        self.wrappers[wrapper.filename] = wrapper
        self.set_xs(dataset_name="x_values")
        self.set_ys(dataset_name="avg_thresh")
        self.plot()

    def remove_file(self, filename):
        self.wrappers.pop(filename)
        self.set_xs(dataset_name="x_values")
        self.set_ys(dataset_name="avg_thresh")
        self.plot()

    # sets y values to be plotted. Can either be passed a List of np arrays with x values for each file being plotted
    # or a dataset_name to be extracted from the hdf5 files. Can also be set to auto, in which case
    # x values will automatically be generated.
    def set_xs(
        self, xs: List[np.ndarray] = None, dataset_name: str = None, auto: bool = False
    ) -> None:
        if auto:
            self.xs = None

        if dataset_name is None and xs is None:
            print("Not enough Arguments. Xs have not been set")
        elif dataset_name is None:
            self.xs = xs
            self.x_label = "Custom Xs"
        else:
            xs = []
            for wrapper in self.wrappers.values():
                x = wrapper.extract_dataset(dataset_name)
                if len(x.shape) == 3:
                    _flatten3d(x)
                xs.append(x)
            self.xs = xs
            self.x_label = dataset_name

    # sets y values to be plotted. Can either be passed a List of np arrays with y values for each file being plotted
    # or a dataset_name to be extracted from the hdf5 files.
    def set_ys(self, ys: List[np.ndarray] = None, dataset_name: str = None) -> None:
        if dataset_name is None and ys is None:
            print("Not enough Arguments. Ys have not been set")
        elif dataset_name is None:
            self.ys = ys
            self.y_label = "Custom Ys"
        else:
            ys = []
            for wrapper in self.wrappers.values():
                y = wrapper.extract_dataset(dataset_name)
                if len(y.shape) == 3:
                    _flatten3d(y)
                ys.append(y)
            self.ys = ys
            self.y_label = dataset_name

    def plot(self) -> None:
        plt.clf()

        # If no files are open, just create axis and return
        if len(self.wrappers) == 0:
            plt.xlabel("")
            plt.ylabel("")
            return

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        self._generate_colors()

        if self.plot_fits is True:
            self._plot_fits()
            self.style["linestyle"] = "None"
        else:
            self.style["linestyle"] = "-"

        for wrapper_num, (x_set, y_set) in enumerate(zip(self.xs, self.ys)):
            if len(x_set.shape) == 1 and len(y_set.shape) != 1:
                for i, ys in enumerate(y_set):
                    plt.plot(
                        x_set,
                        ys,
                        **self.style,
                        color=self.colors[wrapper_num][i],
                        label="a"
                    )
            else:
                if x_set.shape[0] != y_set.shape[0]:
                    raise Exception(
                        "Mismatch in number of rows in datasets. x shape:{0} , y shape:{1}".format(
                            x_set.shape, y_set.shape
                        )
                    )

                for i in range(x_set.shape[0]):
                    plt.plot(
                        x_set[i],
                        y_set[i],
                        **self.style,
                        color=self.colors[wrapper_num][i]
                    )

        self._create_legend()
        self.draw()

    def _plot_fits(self) -> None:
        for wrapper_num, wrapper in enumerate(self.wrappers.values()):
            fit_x = wrapper.fit_x
            fit_y = wrapper.fit_y
            if fit_x is None or fit_y is None:
                continue

            if len(fit_x.shape) == 1 and len(fit_y.shape) != 1:
                for i, ys in enumerate(fit_y):
                    plt.plot(fit_x, ys, color=self.colors[wrapper_num][i])
            else:
                if fit_x.shape[0] != fit_y.shape[0]:
                    raise Exception(
                        "Mismatch in number of rows in datasets. x shape:{0} , y shape:{1}".format(
                            fit_x.shape, fit_y.shape
                        )
                    )

                for i in range(fit_x.shape[0]):
                    plt.plot(fit_x[i], fit_y[i], color=self.colors[wrapper_num][i])

    # Create and display legend with correct colors for corresponding pmts
    def _create_legend(self):
        entries = []
        if len(self.wrappers) > 1:
            for index, wrapper in enumerate(self.wrappers.values()):
                entries.append(
                    mpatches.Patch(
                        color=self.colors[index][len(self.colors[index]) // 2],
                        label="RID: {}".format(wrapper.rid),
                    )
                )
        else:
            for wrapper in self.wrappers.values():
                for index, pmt_num in enumerate(wrapper.pmt_nums):
                    entries.append(
                        mpatches.Patch(
                            color=self.colors[0][index], label="pmt {}".format(pmt_num)
                        )
                    )
        plt.legend(handles=entries, loc="best")

    # generate color scheme. Want to distinguish neighboring PMTs
    def _generate_colors(self):
        self.colors = []
        # Generate colors for multiple files
        if len(self.ys) > 1:
            # A 'set' here represents a y dataset from a single file
            for set_num, y_set in enumerate(self.ys):
                color_assignment = np.empty(y_set.shape[0], dtype=object)
                start_color = self.color_groups[set_num][0]
                end_color = self.color_groups[set_num][1]

                red_progression = np.linspace(
                    start_color[0], end_color[0], color_assignment.shape[0]
                )
                green_progression = np.linspace(
                    start_color[1], end_color[1], color_assignment.shape[0]
                )
                blue_progression = np.linspace(
                    start_color[2], end_color[2], color_assignment.shape[0]
                )
                for i in range(color_assignment.shape[0]):
                    color_assignment[i] = clr.to_hex(
                        (red_progression[i], green_progression[i], blue_progression[i])
                    )
                self.colors.append(color_assignment)

        # Default color Generation for one file to distinguish adjacent PMTS
        # Will need updating if PMTS adjacencies change
        else:
            num_lines = self.ys[0].shape[
                0
            ]  # The number of rows in ys, meaning the number of lines to be plotted
            color_assignment = np.empty(num_lines, dtype=object)
            color_progressions = np.empty(
                (len(self.color_groups), 3, int(np.ceil(num_lines / 4)))
            )
            for i, color_range in enumerate(self.color_groups):
                color_progressions[i] = np.array(
                    [
                        np.linspace(
                            color_range[0][0], color_range[1][0], np.ceil(num_lines / 4)
                        ),
                        np.linspace(
                            color_range[0][1], color_range[1][1], np.ceil(num_lines / 4)
                        ),
                        np.linspace(
                            color_range[0][2], color_range[1][2], np.ceil(num_lines / 4)
                        ),
                    ]
                )

            color_progression_counter = 0
            for i in range(num_lines):
                if i % 4 == 0 and i != 0:
                    color_progression_counter += 1
                color_assignment[i] = clr.to_hex(
                    (
                        color_progressions[i % 4][0][color_progression_counter],
                        color_progressions[i % 4][1][color_progression_counter],
                        color_progressions[i % 4][2][color_progression_counter],
                    )
                )

            self.colors.append(color_assignment)

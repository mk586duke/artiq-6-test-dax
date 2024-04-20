#!/usr/bin/env python3
"""Plot multiple inputs on the same plot.

Optionally either plot multiple datasets or multiple points from the same dataset.
"""
import argparse
import logging
import os
import typing

import artiq.applets as applets
import artiq.applets.plot_xy as plot
import more_itertools
import numpy as np
import pyqtgraph

_LOGGER = logging.getLogger(__name__)
ModList = typing.Sequence[typing.Dict]
DataDict = typing.Dict[str, typing.Tuple[bool, typing.Any]]
OptionalXYPair = typing.Tuple[typing.Optional[typing.Sequence[int]], np.ndarray]

# better background colors
pyqtgraph.setConfigOption("background", "w")
pyqtgraph.setConfigOption("foreground", "k")


class MultiYPlot(plot.XYPlot):
    """Plot multiple Y values on the same axis."""

    def __init__(self, args: argparse.Namespace, **kwargs):
        """Create a plot with multiple Y axes/lines."""
        # allow kwargs to control widget construction
        # pylint: disable=non-parent-init-called,super-init-not-called
        pyqtgraph.PlotWidget.__init__(self, **kwargs)
        self.args = args
        self.num_inputs = len(self.args.y_names)
        self._first_run = True
        self._plot_data_next = True
        self.fit = False
        self.legend_labels = []

    def data_changed(self, data: DataDict, mods: ModList, title: str) -> None:
        """Update method, called when data is changed.

        Processes a list of changes (`mods`), and then plots the data.
        """

        # build legend if first run (handles case where applet is opened mid experiment)
        if self._first_run:
            if self.args.active_pmts is None:
                self.init_legend(pmts=None, labels=self.args.y_names)
            else:
                self.init_legend(
                    pmts=data.get(self.args.active_pmts, None), labels=None
                )

        # Checks if any datasets are being initialized and sets first run to true if so
        # the length of mods is generally 1
        for mod in mods:
            if "key" in mod.keys() and type(mod["key"]) == str and "rid" in mod["key"]:
                self._first_run = True
                # Initalize the legend if it is the first run for experiment
                if self.args.active_pmts is None:
                    self.init_legend(pmts=None, labels=self.args.y_names)
                else:
                    self.init_legend(
                        pmts=data.get(self.args.active_pmts, None), labels=None
                    )

        if self.args.rid is not None:
            rid_string = "RID: {}".format(data.get(self.args.rid, (None, "MISSING"))[1])
            if title is not None:
                title += " {}".format(rid_string)
            else:
                title = rid_string
        elif title is None:
            title = ""  # adds padding at top

        try:
            x_data, y_datas = self._process_data(data, mods)
            x_fit, y_fits = self._process_fits(data, mods)
        except RuntimeWarning:
            _LOGGER.debug("Invalid data. Ignore if starting up.", exc_info=True)
            return
        except KeyError:
            _LOGGER.warning(
                "Dataset key not found. Check the name (ignore on startup)",
                exc_info=True,
            )
            return

        try:
            self.plot_data(x_data, y_datas, x_fit, y_fits, title)
            if (
                self.args.error_bars_bottom is not None
                and self.args.error_bars_top is not None
            ):
                self.plot_error_bars(
                    x_data,
                    y_datas,
                    data[self.args.error_bars_top][1],
                    data[self.args.error_bars_bottom][1],
                )
        except (RuntimeWarning, ValueError) as e:
            _LOGGER.debug(e)

    def init_legend(
        self,
        pmts: typing.Optional[typing.Iterable[int]] = None,
        labels: typing.Optional[typing.Iterable[str]] = None,
    ):
        """Initialize the legend into the list self.legend_labels

        @param: pmts. ndarray of the active pmts numbers. if none, use labels
        @param: labels.  the y names to be used if pmts aren't being used

        PMTs & labels arguments are mutually exclusive.
        """
        self.legend_labels = []
        if pmts is not None:
            for n in pmts[1]:
                self.legend_labels.append("pmt {0}".format(n))
        elif labels is not None:
            for l in labels:
                if "." in l:
                    # Take segment after last period
                    self.legend_labels.append(list(l.split("."))[-1])
                else:
                    self.legend_labels.append(l)

    def _plot_first_run(self, title: str) -> None:
        """Initialize the plot.

        Need to only specify legend once so that it doesn't grow continuously.
        """
        # TODO: enable this on pyqtgraph > 0.11.0
        # num_curves_guess = len(self.legend_labels)
        # expected_columns = (num_curves_guess // 8) + 1
        self.clear()
        self.setTitle(title)
        self.enableAutoRange()
        legend = self.addLegend()
        # legend = self.addLegend(colCount=expected_columns)
        for item in legend.items:
            legend.removeItem(item)
        # legend.clear()
        self.curves = []
        self.fits = []

        if self.args.y_label is not None:
            self.setLabel("left", self.args.y_label, **{"font-size": "12pt"})

        if self.args.x_label is not None:
            self.setLabel("bottom", self.args.x_label, **{"font-size": "12pt"})

        # Adds box around plot area and padding on right side
        self.setLabel("right", "")
        self.getAxis("right").setStyle(showValues=False)
        self.showAxis("right")
        self.getAxis("top").setStyle(showValues=False)
        self.showAxis("top")

        for (i, name) in enumerate(self.legend_labels):
            line_color = pyqtgraph.intColor(i, hues=4, values=len(self.legend_labels))
            plot_item = self.plot(
                pen=line_color,
                symbol=None,
                symbolPen=line_color,
                symbolBrush=line_color,
                name=name,
            )
            self.curves.append(plot_item)
            self.fits.append(
                self.plot(
                    pen=line_color,
                    symbol="o",
                    symbolPen=line_color,
                    symbolBrush=line_color,
                )
            )
            legend.addItem(plot_item, name)

        self._first_run = False
        self._plot_data_next = True

    def _plot_data(self, num_curves: int, title: str) -> None:
        """Switch to "data mode".

        Data points are connected by a line and updated as pushed by Artiq.
        """
        self.clear()
        self.setTitle(title)
        self.enableAutoRange()
        self.curves = []
        self.fits = []

        for i in range(num_curves):

            line_color = pyqtgraph.intColor(i, hues=4, values=num_curves)
            self.curves.append(
                self.plot(
                    pen=line_color,
                    symbol="o",
                    symbolPen=line_color,
                    symbolBrush=line_color,
                )
            )

            self.fits.append(
                self.plot(pen=None, symbol=None, symbolPen=None, symbolBrush=None)
            )
            # Hack to clear fits
            self.fits[i].setData(x=np.array([0]), y=np.array([0]))

        self._plot_data_next = False

    def _plot_fits(self, num_curves: int, title: str) -> None:
        # Switch to "fit mode" where data points scatter-like and fits are a line
        # Assume we fit once and then switch back to "data_mode"
        self.clear()
        self.setTitle(title)
        self.enableAutoRange()
        self.curves = []
        self.fits = []
        for i in range(num_curves):

            line_color = pyqtgraph.intColor(i, hues=4, values=num_curves)
            self.curves.append(
                self.plot(
                    pen=None, symbol="o", symbolPen=line_color, symbolBrush=line_color
                )
            )
            self.fits.append(
                self.plot(
                    pen=line_color,
                    symbol=None,
                    symbolPen=line_color,
                    symbolBrush=line_color,
                )
            )
        self._plot_data_next = True

    def plot_error_bars(self, x, ys, bars_top, bars_bottom):
        """Show error bars on the plot."""
        # Validate error bar items
        if bars_bottom.shape != bars_top.shape or (bars_top.shape[-1] != len(x)):
            message = (
                "Invalid error bar data array shapes.\n"
                "Bars should have same shape, and length = # x points.\n"
                "Top shape: {}, bottom shape: {}, X shape: {}".format(
                    bars_top.shape, bars_bottom.shape, x.shape
                )
            )
            _LOGGER.error(message)
            raise ValueError(message)

        num_curves = len(ys)
        width = 0.05 * (
            (x[len(x) - 1] - x[0]) / len(x)
        )  # adds a small cross over the point, helps visibility
        for i, y in enumerate(ys):
            line_color = pyqtgraph.intColor(i, hues=4, values=num_curves)
            self.addItem(
                pyqtgraph.ErrorBarItem(
                    x=x,
                    y=y,
                    top=np.nan_to_num(bars_top[i]),
                    bottom=np.nan_to_num(bars_bottom[i]),
                    width=width,
                    beam=0,  # .006 * x[len(x) - 1] for beams at top/bottom
                    pen={"color": line_color, "width": 4},
                )
            )

    def plot_data(
        self,
        x: typing.Optional[typing.Sequence[int]],
        ys: np.ndarray,
        x_fit: typing.Optional[typing.Sequence[int]],
        y_fits: np.ndarray,
        title: str,
    ) -> None:
        """Plot multiple y axes on the same X axis."""
        if self._first_run:
            self._plot_first_run(title)
        if self._plot_data_next:
            self._plot_data(len(ys), title)
        if self.fit:
            self._plot_fits(len(ys), title)

        if self.fit and x_fit is None:
            x_fit = np.linspace(x[0], x[-1], y_fits.shape[1])

        for i, y_data in enumerate(ys):
            y_name = "channel {}".format(i)
            _LOGGER.debug("Plotting %s: %i points", y_name, len(y_data))
            # _LOGGER.debug("%s data: %s", y_name, y_data)
            self.curves[i].setData(x=x, y=y_data)
            if self.fit:
                self.fits[i].setData(x=x_fit, y=y_fits[i])

        # TODO: Plot Error Bars

        self.setTitle(title, size="20pt")

    def _process_data(self, data: DataDict, mods: ModList) -> OptionalXYPair:
        """Retrieve x and y data from the input datasets and validate."""
        # pylint: disable=unused-argument
        # _LOGGER.debug("Mods: %s", mods)

        # Retrieve
        if len(self.args.y_names) == 1:
            # just one dataset, but multi-dimensional
            y = np.array(data[self.args.y_names[0]][1], ndmin=2)
        else:
            y = np.array([data[y_data_name][1] for y_data_name in self.args.y_names])

        if self.args.transpose:
            y = y.T

        # If x exists, this return the data, other it returns the tuple (False, None)
        x = np.array(data.get(self.args.x, (False, None))[1])
        if x is None:  # If x doesnt exist
            x = np.linspace(0, y.shape[1] - 1, y.shape[1])
        elif x is not None:
            if x.size > 1 and np.array_equal(
                x, [x[0]] * x.size
            ):  # check to see if step in place scan
                x = np.linspace(0, y.shape[1] - 1, y.shape[1])
            else:
                try:
                    sort = np.argsort(x)
                    x = x[sort]
                    y = y[:, sort]
                except:
                    _LOGGER.debug("Sort Failed")

        if self.args.units is not None:
            x = x / float(self.args.units)

        # Validate
        if y.shape[1] != len(x):
            _LOGGER.debug("Array sizes do not match")
            _LOGGER.debug("Y size: %s. X size: %s", y.shape, len(x))

        if np.all(np.isnan(y)):
            _LOGGER.debug("Datasets are just NaN")

        return x, y

    def _process_fits(self, data: DataDict, mods: ModList) -> OptionalXYPair:
        """Retrieve x and y fits from the input datasets and validate.

        Example data value (from ARTIQ subscriber dataset thing):
        {
            "dataset_name": (True, dataset_value),
        }
        """
        # pylint: disable=unused-argument
        # _LOGGER.debug("Mods: %s", mods)

        x_fit = data.get(self.args.x_fit, (False, None))[1]
        if x_fit is not None and np.all(np.isnan(x_fit)):
            x_fit = None

        if self.args.units is not None and x_fit is not None:
            x_fit = x_fit / float(self.args.units)

        if self.args.y_fits is not None:
            y_fits = np.array(
                [
                    data.get(y_fit_name, (False, None))[1]
                    for y_fit_name in self.args.y_fits
                ]
            )
        else:
            y_fits = None

        # Retrieve
        if y_fits is not None:
            if len(self.args.y_fits) == 1:
                # just one dataset, but multi-dimensional
                y_fits = np.array(data[self.args.y_fits[0]][1])
            else:
                y_fits = np.array(
                    [data[y_fit_name][1] for y_fit_name in self.args.y_fits]
                )

            self.fit = True
            if self.args.transpose:
                y_fits = y_fits.T

        else:
            self.fit = False

        if self.fit and np.all(np.isnan(y_fits)):
            self.fit = False

        # Validate
        if x_fit is not None and self.fit and (y_fits.shape[1] != len(x_fit)):
            _LOGGER.error("Fit array sizes do not match")
            _LOGGER.error("Y size: %s. X size: %s", y_fits.shape, len(x_fit))
            raise RuntimeError("Dataset array sizes do not match")

        return x_fit, y_fits


class MultiDataApplet(applets.simple.TitleApplet):
    """Specialized applet to handle receiving multiple datasets in an argument."""

    def __init__(self, *args, **kwargs):
        """Start an Applet to handle plotting multiple sets of data on y axis."""
        super().__init__(*args, **kwargs)

        # Add logging argument
        self.argparser.add_argument("-v", "--verbose", action="count", default=0)

        # Add multi-data argparsing
        self.multi_plot_args = self.argparser.add_argument_group(
            "MultiPlot Args", "Arguments to control how multiple plots are displayed"
        )
        self.add_dataset_arg(
            "x", required=False, help="Name of the X axis dataset (ONE, optional)"
        )
        self.add_dataset_arg(
            "y-names",
            help="Name(s) of the Y axis datasets",
            nargs="+",
            type=str,
            required=True,
        )
        self.add_dataset_arg(
            "active-pmts",
            help="Numpy array with the numbers for the active pmts",
            required=False,
        )
        self.add_dataset_arg(
            "x-fit",
            required=False,
            help="Name of the X axis for the fits (ONE, optional). "
            "Expects data as 1-D numpy array, shape: (num_points,)",
        )
        self.add_dataset_arg(
            "y-fits",
            help="Dataset name(s) for y-fits (same number of curves as y-names). "
            "Expects data as numpy array, shape: (num_curves, num_points). "
            "NOTE: num_points is same as in --x-fits",
            nargs="+",  # require at least one fit
            type=str,
            required=False,
        )
        self.add_dataset_arg(
            "error-bars-top",
            help="Array with the lengths of the top of each point's error bar. "
            "Expected numpy array shape: (num_y_curves, num_x_points)",
            required=False,
        )
        self.add_dataset_arg(
            "error-bars-bottom",
            help="Array with the lengths of the bottom of each point's error bar. "
            "Values should be positive. "
            "Expected numpy array shape: (num_y_curves, num_x_points).",
            required=False,
        )
        self.add_dataset_arg(
            "rid", help="Dataset holding RID for plot title", type=str, required=False
        )
        self.multi_plot_args.add_argument(
            "--transpose",
            "-t",
            action="store_true",
            help="Flag to transpose the y data",
        )
        self.multi_plot_args.add_argument(
            "--units",
            "-u",
            type=str,
            help="Units for Scan Axis (alias from artiq.language.units)",
            required=False,
        )
        self.multi_plot_args.add_argument(
            "--x-label", type=str, help="X-label for plot", required=False
        )
        self.multi_plot_args.add_argument(
            "--y-label", type=str, help="Y-label for plot", required=False
        )

    def add_dataset_arg(self, argname: str, required: bool = True, **kwargs) -> None:
        """Add dataset to command-line arguments & subscribe to changes to its dataset.

        `kwargs` are passed directly to :meth:`argparse.ArgumentParser.add_argument`.
        """
        kwargs["required"] = required
        # HACK: ARTIQ didn't like to pass all arguments to argparse
        self._arggroup_datasets.add_argument("--" + argname, **kwargs)
        self.dataset_args.add(argname)

    def args_init(self) -> None:
        """Parse arguments and setup initial values."""
        self.args = self.argparser.parse_args()
        self.embed = os.getenv("ARTIQ_APPLET_EMBED")
        # _LOGGER.info("args: %s", self.args)
        # _LOGGER.info("ds args: %s", self.dataset_args)

        # Record which datasets are used, based on CLI arguments.
        # HACK: ARTIQ couldn't handle multiple datasets from 1 arg, so modified
        self.datasets = set(
            more_itertools.collapse(
                getattr(self.args, arg.replace("-", "_")) for arg in self.dataset_args
            )
        )
        # _LOGGER.info("datasets: %s", self.datasets)

    def run(self) -> None:
        """Run the main loop of the applet."""
        try:
            return super().run()
        except ConnectionRefusedError as err:
            raise RuntimeError("ARTIQ master not started") from err


def main() -> None:
    """Start the multi plot applet."""
    # Declare applet
    applet = MultiDataApplet(MultiYPlot)
    # Default to WARNING
    logging.basicConfig(
        level=(logging.WARNING - 10 * applet.argparser.parse_args().verbose)
    )

    # args handled in __init__

    applet.run()


if __name__ == "__main__":
    main()

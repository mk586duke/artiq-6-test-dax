#!/usr/bin/env python3
"""
Code is built on Artiq's plot_xy_hist applet

TODO:
    * Get X Values to work with corresponding PMT Numbers or add legend
    * Add Light/Dark plots in analyze phase
"""
import numpy as np
import pyqtgraph
from artiq.applets.simple import SimpleApplet
from PyQt5 import QtWidgets


# Computes y values for each x val in the left plot
def _compute_ys(histograms_counts):
    ys = np.empty(len(histograms_counts))
    for i in range(len(histograms_counts)):
        ys[i] = 1  # i % 3 for light, dark
    return ys


# Computes bins for a histogram. 25 is default max
def _compute_bins(arr):
    if np.isnan(np.nanmax(arr)):
        return np.arange(26)
    elif np.nanmax(arr) <= 2:
        return np.arange(10)
    else:
        return np.arange(int(np.nanmax(arr)) + 2)

    return bins


# Flattens 3D Array to 2D (pmts, steps, shots) => (pmt, steps & shots)
def _flatten3d(arr):
    flattened = np.empty((arr.shape[0], arr.shape[1] * arr.shape[2]))
    for i in range(arr.shape[0]):
        flattened[i] = arr[i].flatten()
    return flattened


# pyqtgraph.GraphicsWindow fails to behave like a regular Qt widget
# and breaks embedding. Do not use as top widget.
class XYHistPlot(QtWidgets.QSplitter):
    def __init__(self, args):
        # setting color options
        pyqtgraph.setConfigOption("background", "w")
        pyqtgraph.setConfigOption("foreground", 0.0)

        # allows for dual windows
        QtWidgets.QSplitter.__init__(self)
        self.resize(1000, 600)
        self.setWindowTitle("XY/Histogram")

        # plot widget on left window (index 0)
        self.xy_plot = pyqtgraph.PlotWidget()
        self.xy_plot.getPlotItem().setLabel("left", text="Both, Light, Dark")
        self.xy_plot.getPlotItem().setLabel("bottom", text="PMT Number")
        self.insertWidget(0, self.xy_plot)
        self.xy_plot_data = None
        self.arrow = None
        self.selected_index = None

        # histogram widget on right window (index 1)
        self.hist_plot = pyqtgraph.PlotWidget()
        self.hist_plot.getPlotItem().setLabel("left", text="# times Count Observed")
        self.hist_plot.getPlotItem().setLabel("bottom", text="Count Value")
        self.insertWidget(1, self.hist_plot)
        self.hist_plot_data = None

        self.args = args
        self.first_run = True

    # Initializes/re-plots all plots. Called on first plot
    def _set_full_data(self, xs, histograms_counts):
        self.first_run = False
        self.xy_plot.clear()
        self.hist_plot.clear()
        self.xy_plot_data = None
        self.hist_plot_data = None
        self.arrow = None
        self.selected_index = None

        ys = _compute_ys(histograms_counts)
        self.xy_plot_data = self.xy_plot.plot(
            x=xs, y=ys, pen=None, symbol="x", symbolSize=20
        )
        self.xy_plot_data.sigPointsClicked.connect(self._point_clicked)

        # assigns all the points/spot items their indicies and counts
        counter = 0
        points = self.xy_plot_data.scatter.points()
        while counter < len(points):
            points[counter].histogram_index = counter
            points[counter].histogram_counts = histograms_counts[counter]
            counter += 1

        self.hist_plot_data = self.hist_plot.plot(
            stepMode=True, fillLevel=0, brush=(0, 0, 255, 255)
        )

    # called whenever a new point is added to left graph
    def _set_partial_data(self, xs, histograms_counts):
        # Live updating the selected histogram
        if self.selected_index != None:
            bins = _compute_bins(histograms_counts[self.selected_index])
            hist, _ = np.histogram(histograms_counts[self.selected_index], bins)
            self.hist_plot_data.setData(x=bins, y=hist)

    # handles clicking points on left graph
    def _point_clicked(self, data_item, spot_items):
        spot_item = spot_items[0]
        position = spot_item.pos()
        # initalizes/updates arrow
        if self.arrow is None:
            self.arrow = pyqtgraph.ArrowItem(
                angle=-120,
                tipAngle=30,
                baseAngle=20,
                headLen=40,
                tailLen=40,
                tailWidth=8,
                pen=None,
                brush="b",
            )
            self.arrow.setPos(position)
            # NB: temporary glitch if addItem is done before setPos
            self.xy_plot.addItem(self.arrow)
        else:
            self.arrow.setPos(position)
        # updates histogram plot
        self.selected_index = spot_item.histogram_index
        bins = _compute_bins(spot_item.histogram_counts)
        hist, _ = np.histogram(spot_item.histogram_counts, bins)
        self.hist_plot_data.setData(x=bins, y=hist)

    def data_changed(self, data, mods):
        try:
            # When raw counts is stored as a local variable, it fixes a bug where the counts are lost when running is over
            self.raw_counts = data[self.args.pmt_counts][1]

            # Flatten raw counts if it is a 3d array
            if len(self.raw_counts.shape) == 3:
                raw_counts = _flatten3d(self.raw_counts)

            """
            # Filter raw counts into light, dark, and both
            filtered_raw_counts = []
            for pmt in raw_counts:
                filtered_raw_counts.append(pmt) # Both
                bright = []
                dark = []
                for count in pmt.flatten():
                    if count > 1: # Hard coded pmt threshold
                        bright.append(count)
                    else:
                        dark.append(count)

                filtered_raw_counts.append(np.array(bright))
                filtered_raw_counts.append(np.array(dark))
            """

            # Setting x values given an array/dataset of active pmt numbers
            xs = np.empty(data[self.args.pmt_numbers][1].shape[0])
            for i in range(xs.shape[0]):
                xs[i] = data[self.args.pmt_numbers][1][i]

        except KeyError:
            return

        if self.first_run:
            self._set_full_data(xs, self.raw_counts)
        else:
            self._set_partial_data(xs, self.raw_counts)


def main():
    applet = SimpleApplet(XYHistPlot)
    applet.add_dataset(
        "pmt_counts", "Array of raw counts for every PMT", required=False
    )
    applet.add_dataset("pmt_numbers", "Array of currently active PMT's", required=False)
    applet.run()


if __name__ == "__main__":
    main()

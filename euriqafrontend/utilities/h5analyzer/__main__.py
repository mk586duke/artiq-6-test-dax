"""Entry point for h5analyzer command line tool.

Parses arguments passed to it and creates the plot_manipulator, plotter, and h5reader objects
"""
import argparse
import sys

from PyQt5 import QtWidgets

from .h5plotter import h5plotter
from .h5wrapper import h5wrapper
from .plot_manipulator import plot_manipulator


def main():
    parser = argparse.ArgumentParser(
        description="Plots and helps to analyse given list filepaths for hdf5 data files"
    )
    parser.add_argument("filepaths", nargs="+")
    args = parser.parse_args()
    wrappers = []
    for filepath in args.filepaths:
        wrappers.append(h5wrapper(filepath))

    plotter = h5plotter(wrappers)
    app = QtWidgets.QApplication(sys.argv)
    ui = plot_manipulator(plotter)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

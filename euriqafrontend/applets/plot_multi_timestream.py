"""Appends new data to the plot and then displays.

Useful for datasets with unknown lengths, i.e. running forever.

Use with :mod:`euriqafrontend.experiments.repository.cool_and_display`.
"""
import datetime
import logging
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import numpy as np
import pyqtgraph

import euriqafrontend.applets.plot_multi as multi
import euriqafrontend.applets.plot_multi_stream as stream

_LOGGER = logging.getLogger(__name__)


class TimeAxisItem(pyqtgraph.AxisItem):
    """Modifies axis to plot floating-point timestamps as human-readable strings."""

    def tickStrings(self, values: Sequence[float], scale, spacing):
        """Change UTC timestamps (floats) into time-strings."""
        return [
            datetime.datetime.utcfromtimestamp(value).strftime("%M:%S.%f")
            for value in values
        ]


class MultiYTimeStreamPlot(stream.MultiYStreamPlot):
    """Updates data in real time based on modified datasets.

    Useful for plotting data that is streaming in, and don't know end size originally.
    Timestamps ever data point that arrives. Time interval is determined by
    ``update-delay`` arg, with this grabbing the most recent data every ``update-delay``
    seconds. So you should generate data into the array AT LEAST every ``update-delay``
    s.
    """

    def __init__(self, args, **kwargs):
        """Create a multiple-y axis plot with timestamps as strings on X axis."""
        kwargs["axisItems"] = {"bottom": TimeAxisItem(orientation="bottom")}
        super().__init__(args, **kwargs)

    def _process_data(
        self, data: Dict[str, Tuple[bool, Any]], mods: Sequence[Dict]
    ) -> Tuple[Sequence[datetime.datetime], np.ndarray]:
        """Append new data from the dataset(s) to an internal array."""
        # Get new data & add to arrays
        # _LOGGER.info("New data received")
        # _LOGGER.info("Mods: %s", mods)

        for mod in mods:
            try:
                if mod["path"] == []:
                    self.x_data_array = None
                    self.y_data_array = None
                    break
            except Exception:
                ()

        new_x_data = data.get(self.args.x, (False, [datetime.datetime.utcnow()]))[1]
        new_y_data = np.array(
            [data[y_data_name][1] for y_data_name in self.args.y_names]
        )

        if len(new_y_data.shape) > 2:
            _LOGGER.error(
                "ONLY ACCEPTS 1-D streaming data arrays (1D per dataset). Shape: %s",
                new_y_data.shape,
            )

        if self.x_data_array is None:
            self.x_data_array = new_x_data
        else:
            if isinstance(self.x_data_array, list):
                self.x_data_array.extend(new_x_data)
            elif isinstance(self.x_data_array, np.ndarray):
                self.x_data_array = np.concatenate(
                    (self.x_data_array, new_x_data), axis=0
                )
            else:
                raise ValueError("Incorrect datatype for x")

        # Make sure np-arrays are oriented correctly to allow for concatenation
        if self.y_data_array is None:
            self.y_data_array = np.array(new_y_data, ndmin=2)
            if self.y_data_array.shape[0] < self.y_data_array.shape[1]:
                self.y_data_array = self.y_data_array.T
        else:
            self.y_data_array = np.hstack(
                (self.y_data_array, np.array(new_y_data, ndmin=2).T)
            )

        # Trim per retention policy:
        if self.args.retention == "last":
            if self.x_data_array is not None:
                self.x_data_array = self.x_data_array[-self.args.retain_points :]
            self.y_data_array = self.y_data_array[:, -self.args.retain_points :]

        # Validate
        if self.x_data_array is not None and (
            not len(self.y_data_array)
            or self.y_data_array.shape[1] != len(self.x_data_array)
        ):
            _LOGGER.error("Array sizes do not match")
            _LOGGER.error(
                "Y size: %s. X size: %s", self.y_data_array.size, len(self.x_data_array)
            )
            raise RuntimeError("Dataset array sizes do not match")

        if np.all(np.isnan(self.y_data_array)):
            pass
            #raise RuntimeError("Y Datasets are just NaN")
        if self.x_data_array is not None and np.all(
            np.array(self.x_data_array) == np.nan
        ):
            raise RuntimeError("X datasets are just NaN")

        # TODO: commit data changes
        timestamp = [
            (np.datetime64(t) - np.datetime64("1970-01-01T00:00:00Z"))
            / np.timedelta64(1, "s")  # pylint: disable=too-many-function-args
            for t in self.x_data_array
        ]

        return timestamp, self.y_data_array


def main() -> None:
    """Start the multi plot applet."""
    # Declare applet
    logging.basicConfig(level=logging.WARNING)
    applet = multi.MultiDataApplet(MultiYTimeStreamPlot, default_update_delay=0.1)

    # Add CLI arguments
    applet.multi_plot_args.add_argument(
        "--retention",
        "-r",
        choices=["all", "last"],
        default="last",
        help="How to retain points in the 'append' applet: all, or some subset",
    )
    applet.multi_plot_args.add_argument(
        "--retain-points",
        "-rp",
        type=int,
        default=200,
        help="Number of points to retain (only if used with --retention != 'all'",
    )

    applet.run()


if __name__ == "__main__":
    main()

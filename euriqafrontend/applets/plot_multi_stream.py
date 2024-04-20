"""Appends new data to the plot and then displays.

Useful for datasets with unknown lengths, i.e. running forever.

Use with :mod:`euriqafrontend.experiments.repository.cool_and_display`.
"""
import argparse
import logging
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

import euriqafrontend.applets.plot_multi as multi

_LOGGER = logging.getLogger(__name__)


class MultiYStreamPlot(multi.MultiYPlot):
    """Updates data in real time based on modified datasets.

    Useful for plotting data that is streaming in, and don't know end size originally.
    """

    def __init__(self, args: argparse.Namespace, **kwargs):
        """Create a streaming data plot with multiple Y axes/lines."""
        super().__init__(args, **kwargs)
        self.y_data_array = None
        self.x_data_array = None

    def _process_data(
        self, data: Dict[str, Tuple[bool, Any]], mods: Sequence[Dict]
    ) -> Tuple[Union[None, Sequence[int]], np.ndarray]:
        """Append new data from the dataset(s) to an internal array."""
        # Get new data & add to arrays
        # _LOGGER.info("New data received")
        # _LOGGER.info("Mods: %s", mods)

        new_x_data = data.get(self.args.x, (False, None))[1]
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
        if (
            self.x_data_array is not None
            and not self.args.timestamp_x
            and np.all(np.isnan(self.x_data_array))
        ):
            raise RuntimeError("X datasets are just NaN")

        return self.x_data_array, self.y_data_array


def main() -> None:
    """Start the multi plot applet."""
    # Declare applet
    logging.basicConfig(level=logging.WARNING)
    applet = multi.MultiDataApplet(MultiYStreamPlot)

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
        help="Number of points to retain (only if used with --retention != 'all')",
    )

    applet.run()


if __name__ == "__main__":
    main()

"""Test for indexing with mutate_dataset.
"""
import logging
import time
from collections import defaultdict

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import numpy as np
import oitg.fitting as fit
from artiq.experiment import TerminationRequested
from artiq.language.core import host_only
from artiq.language.units import kHz
from artiq.language.units import ms
from artiq.language.units import us

_LOGGER = logging.getLogger(__name__)


class TestIndexing(artiq_env.EnvExperiment):
    """Test for figuring out indexing with mutate_dateset"""

    _EXPERIMENT_NAME = "Test Fits"
    _EXPERIMENT_GROUP_NAME = "Tests"
    _FIT_TYPE = fit.cos_fft

    kernel_invariants = {"scan_values", "num_pmts"}

    applet_stream_cmd = (
        "$python -m euriqafrontend.applets.plot_multi" + " "
    )  # White space is required

    def build(self):
        """Initialize experiment & variables."""
        # devices
        self.setattr_device("core")
        self.setattr_device("ccb")
        self.setattr_device("oeb")
        self.setattr_device("scheduler")

        # arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0 * us,
                    stop=2 * ms,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="us",
                global_min=0,
            ),
        )
        self.setattr_argument(
            "num_shots", artiq_env.NumberValue(default=100, step=1, ndecimals=0)
        )

    def prepare(self):
        """Pre-calculate any values."""
        # Run prepare method of all the imported modules
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        # calculate scan parameters
        self.scan_values = [t for t in self.scan_range]
        self.num_steps = len(self.scan_values)
        self.rabi_frequency = np.arange(1000, 6000, 1000)
        self.num_pmts = len(self.rabi_frequency)

        self.counts_temp = np.full((self.num_pmts, 1, 1), np.nan)
        self.thresh_temp = np.full((self.num_pmts, 1), np.nan)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        raw_counts_empty = np.full(
            (self.num_pmts, self.num_shots, self.num_steps), np.nan
        )
        avg_thresh_empty = np.full((self.num_pmts, self.num_steps), np.nan)
        self.set_dataset("test.raw_counts", raw_counts_empty)
        self.set_dataset("test.avg_thresh", avg_thresh_empty, broadcast=True)
        self.set_dataset("test.x_values", self.scan_values, broadcast=True)
        self.set_dataset("test.fit_x", np.nan, broadcast=True)
        self.set_dataset("test.fit_y", np.full(self.num_pmts, np.nan), broadcast=True)

        self.ccb.issue(
            "create_applet",
            name=self._EXPERIMENT_NAME,
            command=self.applet_stream_cmd
            + "--x test.x_values "
            + "--y-names test.avg_thresh "
            + "--x-fit test.fit_x "
            + "--y-fits test.fit_y ",
            group=self._EXPERIMENT_GROUP_NAME,
        )

        # RUN EXPERIMENT
        try:
            for istep in range(len(self.scan_values)):
                if self.scheduler.pause():
                    break

                # Loop over to gather statistics
                for ishot in range(self.num_shots):
                    self.counts_temp[:, 0, 0] = np.random.random(self.num_pmts)
                    self.mutate_dataset(
                        "test.raw_counts",
                        (
                            (0, self.num_pmts, 1),
                            (ishot, None, self.num_shots),
                            (istep, None, self.num_steps),
                        ),
                        self.counts_temp,
                    )

                time.sleep(0.25)

                self.thresh_temp[:, 0] = (
                    np.cos(self.rabi_frequency * 2 * np.pi * self.scan_values[istep])
                    + (np.random.random(self.num_pmts) * 2 - 1) / 3
                )
                self.mutate_dataset(
                    "test.avg_thresh",
                    ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
                    self.thresh_temp,
                )
        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    def analyze(self):
        """Threshold values and analyze."""
        fit_plot_points = 500
        x = np.array(self.get_dataset("test.x_values"))
        y = self.get_dataset("test.avg_thresh")
        p_all = defaultdict(list)
        p_error_all = defaultdict(list)
        y_fit_all = np.full((y.shape[0], fit_plot_points), np.nan)
        for iy in range(y.shape[0]):
            p, p_error, x_fit, y_fit = self._FIT_TYPE.fit(
                x, y[iy, :], evaluate_function=True, evaluate_n=fit_plot_points
            )
            for ip in p:
                p_all[ip].append(p[ip])
                p_error_all[ip].append(p_error[ip])

            y_fit_all[iy, :] = y_fit

        for ip in p:
            self.set_dataset("test." + ip + "_fit", p_all[ip], broadcast=True)
        self.set_dataset("test.fit_x", x_fit, broadcast=True)
        self.set_dataset("test.fit_y", y_fit_all, broadcast=True)

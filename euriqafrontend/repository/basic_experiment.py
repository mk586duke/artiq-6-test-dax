import logging
import time
import numpy as np

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import oitg.fitting as fit
import euriqafrontend.fitting as umdfit
from artiq.experiment import TerminationRequested
from artiq.language.core import delay, host_only,  kernel, rpc
from artiq.language import TInt32
from artiq.language.units import ms

from euriqafrontend.repository.basic_environment import BasicEnvironment

_LOGGER = logging.getLogger(__name__)


class _BasicExperiment(artiq_env.Experiment):
    """Experiment Template.

    Inherit from BasicEnvironment which handles most of the infrastructure.
    Underscore prevents ARTIQ from reading/erroring on this, shouldn't be used otherwise
    """

    data_folder = "template"
    applet_name = "Insert Applet Name Here"
    applet_group = "Insert SubGroup Name for Applet Here"
    fit_type = fit.sin_fft

    def build(self):
        """Initialize experiment & variables."""
        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0 * ms,
                    stop=20 * ms,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="ms",
                global_min=0,
            ),
        )
        super().build()

    def prepare(self):
        """Save & precalculate experiment-tracking variables."""
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.scan_values = [t for t in self.scan_range]
        self.num_steps = len(self.scan_values)
        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")

            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        """Runs before each step in the experiment. Generally used to update sweep parameters"""
        pass

    @kernel
    def main_experiment(self, istep: TInt32, ishot: TInt32):
        """Run main experiment."""
        delay(self.scan_values[istep])

    @rpc(flags={"async"})
    def custom_proc_data(self, istep: TInt32):
        """ Custom processing code that runs on the host at the end of every step point in the scan (after num_shots
        exps are taken). This function should be an async RPC on the host"""
        pass

    @rpc(flags={"async"})
    def custom_proc_counts(self, ishot: TInt32, istep: TInt32):
        """ Custom processing code that runs on the host at the end of every experiment (a single shot). This function
         should be an async RPC on the host"""
        pass

    @host_only
    def analyze(self):
        """Add custom analysis. May include print statements or global value updates"""
        super().analyze()


import logging
import time

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import artiq.language.units as artiq_units
import numpy as np
import oitg.fitting as fit
from artiq.experiment import NumberValue
from artiq.experiment import TerminationRequested
from artiq.language.core import delay
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import parallel
from artiq.language.core import rpc
from artiq.language.types import TBool
from artiq.language.types import TFloat
from artiq.language.types import TInt32
from artiq.language.types import TInt64
from artiq.language.types import TList
from artiq.language.types import TNone
from artiq.language.units import MHz
from artiq.language.units import us

import euriqabackend.coredevice.dac8568 as dac8568
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend.repository.basic_environment import BasicEnvironment

_LOGGER = logging.getLogger(__name__)


class StarkShift(BasicEnvironment, artiq_env.Experiment):
    """AWG.StarkShift
    """

    data_folder = "raman_ramsey"
    applet_name = "Stark Shift Echo"
    applet_group = "Calibration"
    fit_type = fit.cos
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Wait Time (us)"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0 * us,
                    stop=20 * us,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="us",
                global_min=0,
            ),
        )
        self.setattr_argument(
            "detuning",
            NumberValue(
                0,
                unit="MHz",
                min=-10 * MHz,
                max=10 * MHz,
                step=0.0025 * MHz,
                ndecimals=6,
            ),
        )
        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("lock_active", artiq_env.BooleanValue(default=False))
        super().build()

    def prepare(self):

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )
        self.scan_values = [t_s for t_s in self.scan_range]
        self.scan_values_mu = [self.core.seconds_to_mu(t_s) for t_s in self.scan_range]
        self.num_steps = len(self.scan_values)
        super().prepare()
        self.detuning_mu = freq_to_mu(self.detuning)
        self.global_amp_mu = np.int32(self.rabi_global_amp)
        self.ind_amp_mu = np.int32(self.rabi_ind_amp)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_init()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def kn_custom_init(self):
        self.raman.collective_dds.enabled = False

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        #
        # self.raman.global_dds.update_amp(self.global_amp_mu)
        # delay(5 * us)
        # self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        # delay(5 * us)
        # self.raman.set_detuning_mu(self.detuning_mu)
        # delay(120 * us)

        self.calib_stark_shift(self.scan_values_mu[istep])

        # delay(self.scan_values[istep])

    def analyze(self):
        """Analyze and Fit data"""
        """Threshold values and analyze."""
        super().analyze()
        meanTpi = 0.0
        num_active_pmts = len(self.p_all["t_period"])
        for ifit in range(num_active_pmts):
            meanTpi += self.p_all["t_period"][ifit] * 0.5
            print(
                "Fit",
                ifit,
                ": ",
                "Tpi = (",
                self.p_all["t_period"][ifit] * 0.5 * 1e6,
                " +- ",
                self.p_error_all["t_period"][ifit] * 0.5 * 1e6,
                ") us",
            )
            print(
                "Fit",
                ifit,
                ": ",
                "tau = (",
                self.p_all["tau_decay"][ifit] * 1e6,
                "+-",
                self.p_error_all["tau_decay"][ifit] * 1e6,
                ") us",
            )
        meanTpi /= num_active_pmts

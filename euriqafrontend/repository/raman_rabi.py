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
from artiq.language.types import TInt32
from artiq.language.units import MHz
from artiq.language.units import us, ms

import euriqafrontend.fitting as umd_fit
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend.repository.basic_environment import BasicEnvironment
from euriqafrontend.modules.spec_analyzer import Spec_Analyzer

_LOGGER = logging.getLogger(__name__)


class RamanRabi(BasicEnvironment, artiq_env.Experiment):
    """Raman.Rabi"""

    data_folder = "raman_rabi"
    applet_name = "Raman Rabi"
    applet_group = "Raman"
    fit_type = umd_fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

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
                max=40 * MHz,
                step=0.0025 * MHz,
                ndecimals=6,
            ),
        )
        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=1000))

        self.setattr_argument(
            "c8",
            artiq_env.BooleanValue(default=True),
            tooltip="True -> 8dB attentuation on ind",
        )
        self.setattr_argument(
            "c16",
            artiq_env.BooleanValue(default=True),
            tooltip="True -> 16dB attentuation on ind",
        )

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
        # self.scan_values = [self.core.seconds_to_mu(t_s) for t_s in self.scan_range]
        self.scan_values = [t_s for t_s in self.scan_range]
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
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        pass
        # self.calib_wait_time_mu = self.scan_values[istep]

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_ind_attenuation(c8=self.c8, c16=self.c16)
        self.raman.global_dds.update_amp(self.global_amp_mu)
        delay(5 * us)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_global_detuning_mu(self.detuning_mu)

        delay(120 * us)
        self.raman.pulse(self.scan_values[istep])

    def analyze(self):
        super().analyze()
        meanTpi = 0.0
        num_active_pmts = len(self.p_all["t_period"])
        for ifit in range(num_active_pmts):
            meanTpi += self.p_all["t_period"][ifit] * 0.5
            print("Fit {:d} : Tpi = ({:.3f}  +- {:.3f}) us".format(
                ifit,
                self.p_all["t_period"][ifit] * 0.5 * 1e6,
                self.p_error_all["t_period"][ifit] * 0.5 * 1e6
            )
            )
            print("Fit {:d} : tau = ({:.3f}  +- {:.3f}) us".format(
                ifit,
                self.p_all["tau_decay"][ifit] * 0.5 * 1e6,
                self.p_error_all["tau_decay"][ifit] * 0.5 * 1e6
            )
            )


        meanTpi /= num_active_pmts
        print([self.p_all["t_period"][ifit] * 0.5 * 1e6 for ifit in range(num_active_pmts)])


class RamanMicromotionRabi(BasicEnvironment, artiq_env.Experiment):
    """Raman.Micromotion_Rabi"""

    data_folder = "raman_rabi"
    applet_name = "Raman Rabi"
    applet_group = "Raman"
    fit_type = fit.rabi_flop

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0 * us,
                    stop=500 * us,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="us",
                global_min=0,
            ),
        )
        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=1000))
        self.detuning_mu = np.int64(self.get_dataset("global.Ion_Freqs.RF_Freq"))
        super().build()

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        # self.scan_values = [self.core.seconds_to_mu(t_s) for t_s in self.scan_range]
        self.scan_values = [t_s for t_s in self.scan_range]
        self.num_steps = len(self.scan_values)
        super().prepare()
        self.detuning_mu = freq_to_mu(self.get_dataset("global.Ion_Freqs.RF_Freq"))
        self.global_amp_mu = np.int32(self.rabi_global_amp)
        self.ind_amp_mu = np.int32(self.rabi_ind_amp)

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
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.global_dds.update_amp(self.global_amp_mu)
        delay(5 * us)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_global_detuning_mu(self.detuning_mu)
        delay(15 * us)
        self.raman.pulse(self.scan_values[istep])

    def analyze(self):
        """Analyze and Fit data"""
        """Threshold values and analyze."""
        super().analyze()
        meanTpi = 0.0
        # num_active_pmts = len(self.p_all["t_period"])
        for ifit in range(self.num_pmts):
            meanTpi += self.p_all["t_period"][ifit] * 0.5
            _LOGGER.info(
                "Fit %i:\n\tTpi = (%f +- %f) us\n\ttau = (%f +- %f) us",
                ifit,
                self.p_all["t_period"][ifit] * 0.5e6,
                self.p_error_all["t_period"][ifit] * 0.5e6,
                self.p_all["tau_decay"][ifit] * 1e6,
                self.p_error_all["tau_decay"][ifit] * 1e6,
            )
        meanTpi /= self.num_pmts  # num_active_pmts


class RamanRabiSpec(BasicEnvironment, artiq_env.Experiment):
    """Raman.RabiSpec
    """

    data_folder = "raman_rabi_spec"
    applet_name = "Raman Rabi Spectroscopy"
    applet_group = "Raman"
    fit_type = umd_fit.positive_gaussian
    fit_type = None
    xlabel = "detuning (MHz)"
    ylabel = "population transfer"

    def get_lower_com_freq(self):
        frf_sec = self.get_dataset("global.Ion_Freqs.frf_sec")
        qzy = self.get_dataset("global.Voltages.QZY")
        qzz = self.get_dataset("global.Voltages.QZZ")
        x2 = self.get_dataset("global.Voltages.X2")
        return 1e6 * np.sqrt(
            (frf_sec / 1e6) ** 2 - x2 / 2 - np.sqrt(qzy * qzy + qzz * qzz)
        )

    def get_upper_com_freq(self):
        frf_sec = self.get_dataset("global.Ion_Freqs.frf_sec")
        qzy = self.get_dataset("global.Voltages.QZY")
        qzz = self.get_dataset("global.Voltages.QZZ")
        x2 = self.get_dataset("global.Voltages.X2")
        return frf_sec - x2 / 2 + np.sqrt(qzy * qzy + qzz * qzz)

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=self.get_lower_com_freq() - 0.030 * MHz,
                    stop=self.get_lower_com_freq() + 0.030 * MHz,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                ndecimals=4,
                unit="MHz",
                global_min=-5e6,
            ),
        )

        self.setattr_argument(
            "sideband_order",
            NumberValue(1, scale=1, min=-3, max=3, step=1, ndecimals=0),
        )

        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=250 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument(
            "c8",
            artiq_env.BooleanValue(default=True),
            tooltip="True -> 8dB attentuation on ind",
        )
        self.setattr_argument(
            "c16",
            artiq_env.BooleanValue(default=True),
            tooltip="True -> 16dB attentuation on ind",
        )

        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=0))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=0))

        super().build()

        self.spec_analyzer = Spec_Analyzer(self)

    def prepare(self):
        self.set_variables(
            self.data_folder,
            self.applet_name,
            self.applet_group,
            self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )

        self.scan_values = [f * self.sideband_order for f in self.scan_range]
        self.scan_values_mu = [freq_to_mu(f) for f in self.scan_values]
        self.rabi_time_mu = self.core.seconds_to_mu(self.rabi_time)
        self.num_steps = len(self.scan_values)
        super().prepare()

        self.global_amp_mu = np.int32(self.rabi_global_amp)
        self.ind_amp_mu = np.int32(self.rabi_ind_amp)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()

            self.spec_analyzer.module_init(data_folder=self.data_folder, num_steps=self.num_steps)

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_ind_attenuation(c8=self.c8, c16=self.c16)
        self.raman.global_dds.update_amp(self.global_amp_mu)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_ind_detuning_mu(np.int64(0))
        self.raman.set_global_detuning_mu(self.scan_values_mu[istep])
        delay(120 * us)
        self.raman.pulse_mu(self.rabi_time_mu)

    def analyze(self):
        """Analyze and Fit data."""
        super().analyze()
        temp = self.spec_analyzer.fit_multiple_peaks()
        print("peaks are:")
        print(temp)
        if self.fit_type is not None:
            for ifit in range(len(self.p_all["x0"])):
                _LOGGER.info(
                    "Fit %i: x0= (%f +- %f) MHz",
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                )


class RamanRamsey(BasicEnvironment, artiq_env.Experiment):
    """Raman.Ramsey"""

    data_folder = "raman_ramsey"
    applet_name = "Raman Ramsey"
    applet_group = "Raman"
    fit_type = fit.cos_fft

    def build(self):
        """Initialize experiment & variables."""

        self.setattr_argument(
            "phase_scan",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0,
                    stop=360,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                ndecimals=3,
                scale=1,
                global_min=0,
            ),
        )

        self.setattr_argument(
            "tau_scan",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0 * us,
                    stop=1000 * us,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                ndecimals=3,
                global_min=0,
                unit="us",
            ),
        )

        self.setattr_argument(
            "scan_type",
            artiq_env.EnumerationValue(["phase_scan", "tau_scan"], default="tau_scan"),
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

        self.setattr_argument(
            "pi_half_time", artiq_env.NumberValue(default=1 * us, unit="us", ndecimals=4)
        )

        self.setattr_argument(
            "wait_time", artiq_env.NumberValue(default=100 * us, unit="us")
        )
        self.setattr_argument(
            "analysis_phase", artiq_env.NumberValue(default=0, unit="degrees", scale=1)
        )
        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=1000))

        super().build()

        # scan arguments

    def prepare(self):

        if self.scan_type == "tau_scan":
            self.tau = [t for t in self.tau_scan]
            self.scan_values = self.tau.copy()
            self.phase = [self.analysis_phase for _ in self.tau_scan]

        elif self.scan_type == "phase_scan":
            self.tau = [self.wait_time for _ in self.phase_scan]
            self.phase = [p for p in self.phase_scan]
            self.scan_values = self.phase.copy()

        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.num_steps = len(self.scan_values)
        self.detuning_mu = freq_to_mu(self.detuning)
        self.global_amp_mu = np.int32(self.rabi_global_amp)
        self.ind_amp_mu = np.int32(self.rabi_ind_amp)

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
    def main_experiment(self, istep, ishot):
        self.raman.global_dds.update_amp(self.global_amp_mu)
        delay(5 * us)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_global_detuning_mu(self.detuning_mu)
        self.raman.global_dds.update_phase(phase_deg=0.0)
        delay(120 * us)

        self.raman.pulse(self.pi_half_time)

        with parallel:
            self.raman.global_dds.update_phase(phase_deg=self.phase[istep])
            delay(self.tau[istep])
        self.raman.pulse(self.pi_half_time)

    def analyze(self):
        """Analyze and Fit data."""
        if self.scan_type == "phase_scan":
            super().analyze(constants={"period": 360})
        else:
            super().analyze()

        num_active_pmts = len(self.p_all["period"])
        for ifit in range(num_active_pmts):
            print("Fit {:d} : Amplitude = ({:.3f}  +- {:.3f}) ".format(
                ifit,
                self.p_all["a"][ifit],
                self.p_error_all["a"][ifit]
            ))
            print("Fit {:d} : Phase = ({:.3f}  +- {:.3f}) rad".format(
                ifit,
                self.p_all["x0"][ifit]/360*2*np.pi,
                self.p_error_all["x0"][ifit]/360*2*np.pi
            ))
            print("Fit {:d} : Period = ({:.3f}  +- {:.3f}) ms".format(
                ifit,
                self.p_all["period"][ifit] * 1e3,
                self.p_error_all["period"][ifit]*1e3
            ))

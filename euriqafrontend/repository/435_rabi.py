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
from artiq.language.types import TInt32, TInt64, TFloat
from artiq.language.units import MHz
from artiq.language.units import us

import euriqafrontend.fitting as umd_fit
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend.repository.basic_environment import BasicEnvironment
from euriqafrontend.modules.spec_analyzer import Spec_Analyzer
from artiq.language.units import ms, MHz, us, kHz
from euriqabackend.coredevice.ad9912 import phase_to_mu, freq_to_mu
from artiq.language.environment import HasEnvironment

_LOGGER = logging.getLogger(__name__)

class QuadrupoleRamsey(BasicEnvironment, artiq_env.Experiment):
    """Quadrupole.Ramsey"""

    data_folder = "quadrupole_ramsey"
    applet_name = "Quadrupole Ramsey"
    applet_group = "Quadrupole"
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
            "pi_half_time", artiq_env.NumberValue(default=1 * us, unit="us")
        )

        self.setattr_argument(
            "wait_time", artiq_env.NumberValue(default=100 * us, unit="us")
        )
        self.setattr_argument(
            "analysis_phase", artiq_env.NumberValue(default=0, unit="degrees", scale=1)
        )
        self.setattr_argument(
            "aom_freq",
            NumberValue(
                0,
                unit="MHz",
                min=160 * MHz,
                max=240 * MHz,
                step=0.0025 * MHz,
                ndecimals=6,
            ),
        )
        self.setattr_argument(
            "uwave_detuning", artiq_env.NumberValue(default=-6.48687 * kHz, unit="kHz")
        )
        self.setattr_argument(
            "uwave_pi_time", artiq_env.NumberValue(default=0.6833 * ms, unit="ms")
        )

        self.setattr_argument("quadrupole_amp", artiq_env.NumberValue(default=1000))

        super().build()

        # scan arguments

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        if self.scan_type == "tau_scan":
            self.tau = [t for t in self.tau_scan]
            self.scan_values = self.tau.copy()
            self.phase = [self.analysis_phase for _ in self.tau_scan]

        elif self.scan_type == "phase_scan":
            self.tau = [self.wait_time for _ in self.phase_scan]
            self.phase = [p for p in self.phase_scan]
            self.scan_values = self.phase.copy()

        self.num_steps = len(self.scan_values)
        self.detuning_mu = freq_to_mu(self.detuning)
        #self.global_amp_mu = np.int32(self.rabi_global_amp)
        #self.ind_amp_mu = np.int32(self.rabi_ind_amp)

        self.uwave_resonance = self.get_dataset("global.Ion_Freqs.MW_Freq")
        self.uwave_freq = self.uwave_resonance + self.uwave_detuning

        super().prepare()
        self.aom_freq_mu = freq_to_mu(self.aom_freq)
        self.quadrupole_amp_mu = np.int32(self.quadrupole_amp)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            # self.custom_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def custom_kn_init(self):
        self.core.break_realtime()
        self.mw.freq = freq_to_mu(self.uwave_freq)
        self.mw.phase = phase_to_mu(0.0)
        self.mw.init()

    @kernel
    def main_experiment(self, istep, ishot):
        self.SDDDS.update_amp1(self.quadrupole_amp_mu)
        delay(5 * us)
        self.SDDDS.update_freq1_mu(self.aom_freq_mu)

        delay(120 * us)
        self.eom_935_3ghz.on()
        self.SDDS.update_phase1(phase_deg=0.0)
        self.SDDS.pulse1(self.pi_half_time)
        self.eom_935_3ghz.on()
        with parallel:
            self.SDDDS.update_phase1(phase_deg=self.phase[istep])
            delay(self.tau[istep])
        self.eom_935_3ghz.on()
        self.SDDDS.pulse1(self.pi_half_time)

        #self.eom_935_3ghz.on()
        self.mw.pulse(self.uwave_pi_time)

    def analyze(self):
        """Analyze and Fit data."""
        super().analyze()
        for ifit in range(len(self.p_all["a"])):
            print(
                "Fit %i:\n\tC = (%.3f)) us\n\tphi0 = (%.2f) deg\n\tperiod = (%f)" %
                (ifit,
                self.p_all["a"][ifit] * 2,
                self.p_all["x0"][ifit],
                self.p_all["period"][ifit])
            )

class QuadrupoleRabi(BasicEnvironment, artiq_env.Experiment):
    """Quadrupole.Rabi"""

    data_folder = "quadrupole_rabi"
    applet_name = "Quadrupole Rabi"
    applet_group = "Quadrupole"
    fit_type = fit.rabi_flop
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
            "aom_freq",
            NumberValue(
                0,
                unit="MHz",
                min=160 * MHz,
                max=240 * MHz,
                step=0.0025 * MHz,
                ndecimals=6,
            ),
        )
        self.setattr_argument(
            "uwave_detuning", artiq_env.NumberValue(default=-6.48687*kHz, unit="kHz")
        )
        self.setattr_argument(
            "uwave_pi_time", artiq_env.NumberValue(default=0.6833 * ms, unit="ms")
        )

        self.setattr_argument("quadrupole_amp", artiq_env.NumberValue(default=1000))

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
        self.uwave_resonance = self.get_dataset("global.Ion_Freqs.MW_Freq")
        self.uwave_freq = self.uwave_resonance + self.uwave_detuning

        super().prepare()
        self.aom_freq_mu = freq_to_mu(self.aom_freq)
        self.quadrupole_amp_mu = np.int32(self.quadrupole_amp)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            # self.custom_initialize()
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
    def custom_kn_init(self):
        """Add custom initialization steps before experiment execution."""
        self.core.break_realtime()
        self.mw.freq = freq_to_mu(self.uwave_freq)
        # self.mw.phase = phase_to_mu(0.0)
        self.mw.init()

    @kernel
    def main_experiment(self, istep, ishot):
        self.SDDDS.update_amp1(self.quadrupole_amp_mu)
        delay(5 * us)
        self.SDDDS.update_freq1_mu(self.aom_freq_mu)

        delay(120 * us)
        #self.mw.pulse(self.scan_values[istep])
        #self.mw.pulse(1*ms)
        #self.raman.collective_dds.pulse(self.scan_values[istep])
        #self.mw.on()#.collective_dds.pulse(self.scan_values[istep])
        self.eom_935_3ghz.on()
        delay(5*us)
        self.SDDDS.on1()
        delay(self.scan_values[istep])
        self.SDDDS.off1()
        self.eom_935_3ghz.on()
        self.mw.pulse(self.uwave_pi_time)

    def analyze(self):
        super().analyze()
        meanTpi = 0.0
        num_active_pmts = len(self.p_all["t_period"])
        print(num_active_pmts)
        buf = "{"
        for ifit in range(num_active_pmts):
            meanTpi += self.p_all["t_period"][ifit] * 0.5
            ret = self.p_all["t_period"][ifit] * 0.5 * 1e6
            buf = buf + "%f," % ret
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
        buf = buf + "}"
        print(buf)
#        meanTpi /= num_active_pmts

class QuadrupoleRabiSpec(BasicEnvironment, artiq_env.Experiment):
    """Quadrupole.RabiSpec
    """

    data_folder = "quadrupole_rabi_spec"
    applet_name = "Quadrupole Rabi Spectroscopy"
    applet_group = "Quadrupole"
    fit_type = umd_fit.negative_gaussian
    xlabel = "AOM frequency (MHz)"
    ylabel = "population transfer"

    # def get_lower_com_freq(self):
    #     frf_sec = self.get_dataset("global.Ion_Freqs.frf_sec")
    #     qzy = self.get_dataset("global.Voltages.QZY")
    #     qzz = self.get_dataset("global.Voltages.QZZ")
    #     x2 = self.get_dataset("global.Voltages.X2")
    #     return 1e6 * np.sqrt(
    #         (frf_sec / 1e6) ** 2 - x2 / 2 - np.sqrt(qzy * qzy + qzz * qzz)
    #     )

    # def get_upper_com_freq(self):
    #     frf_sec = self.get_dataset("global.Ion_Freqs.frf_sec")
    #     qzy = self.get_dataset("global.Voltages.QZY")
    #     qzz = self.get_dataset("global.Voltages.QZZ")
    #     x2 = self.get_dataset("global.Voltages.X2")
    #     return frf_sec - x2 / 2 + np.sqrt(qzy * qzy + qzz * qzz)

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    #start=self.get_lower_com_freq() - 0.030 * MHz,
                    #stop=self.get_lower_com_freq() + 0.030 * MHz,
                    start=160*MHz,
                    stop=240*MHz,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                ndecimals=5,
                unit="MHz",
                global_min=-5e6,
            ),
        )
        self.setattr_argument(
            "uwave_detuning", artiq_env.NumberValue(default=-6.48687*kHz, unit="kHz")
        )
        self.setattr_argument(
            "uwave_pi_time", artiq_env.NumberValue(default=0.6833 * ms, unit="ms")
        )
        # self.setattr_argument(
        #     "sideband_order",
        #     NumberValue(1, scale=1, min=-3, max=3, step=1, ndecimals=0),
        # )

        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=250 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument("quadrupole_amp", artiq_env.NumberValue(default=0))

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

        self.scan_values = [f for f in self.scan_range]
        self.scan_values_mu = [freq_to_mu(f) for f in self.scan_values]
        self.rabi_time_mu = self.core.seconds_to_mu(self.rabi_time)
        self.num_steps = len(self.scan_values)
        super().prepare()

        self.quadrupole_amp_mu = np.int32(self.quadrupole_amp)

        self.uwave_resonance = self.get_dataset("global.Ion_Freqs.MW_Freq")
        self.uwave_freq = self.uwave_resonance + self.uwave_detuning

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            # self.custom_initialize()
            self.spec_analyzer.module_init(data_folder=self.data_folder, num_steps=self.num_steps)

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")


    @kernel
    def custom_kn_init(self):
        """Add custom initialization steps before experiment execution."""
        self.core.break_realtime()
        self.mw.freq = freq_to_mu(self.uwave_freq)
        #self.mw.phase = phase_to_mu(0.0)
        self.mw.init()

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def main_experiment(self, istep, ishot):

        self.SDDDS.update_amp1(self.quadrupole_amp_mu)
        delay(200 * us)
        self.SDDDS.update_freq1_mu(self.scan_values_mu[istep])
        delay(5 * us)

        self.eom_935_3ghz.on()
        self.SDDDS.on()
        delay(self.rabi_time)
        self.SDDDS.off()

        self.mw.pulse(self.uwave_pi_time)

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

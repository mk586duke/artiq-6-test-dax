import logging
import time
import numpy as np
import math

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import oitg.fitting as fit
import euriqafrontend.fitting as umdfit
from artiq.experiment import TerminationRequested
from artiq.language.core import kernel, delay, host_only, parallel, delay_mu
from artiq.language.units import ms, MHz, us, kHz
from artiq.language import TInt32
from euriqabackend.coredevice.ad9912 import phase_to_mu, freq_to_mu

from euriqafrontend.repository.basic_environment import BasicEnvironment

_LOGGER = logging.getLogger(__name__)


class MWRabi(BasicEnvironment, artiq_env.Experiment):
    """MW.Rabi

    Inherit from BasicEnvironment which handles most of the infrastructure
    """

    data_folder = "mw_rabi"
    applet_name = "MW Rabi"
    applet_group = "MW"
    fit_type = umdfit.rabi_flop

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

        self.setattr_argument(
            "detuning", artiq_env.NumberValue(default=0*kHz, unit="kHz")
        )

        self.setattr_argument(
            "mw_amplitude",
            artiq_env.NumberValue(default=999, min=0, scale=1, max=1000, ndecimals=0),
        )

        super().build()

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )

        self.resonance = self.get_dataset("global.Ion_Freqs.MW_Freq")
        self.freq = self.resonance + self.detuning

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
    def custom_kn_init(self):
        """Add custom initialization steps before experiment execution."""
        self.core.break_realtime()
        self.mw.freq = freq_to_mu(self.freq)
        self.mw.phase = 0 # phase_to_mu(0.0)
        self.mw.init()
        delay(250*us)
        self.mw.update_amp(np.int32(self.mw_amplitude))
        delay(200 * us)

    @kernel
    def main_experiment(self, istep, ishot):
        self.mw.pulse(self.scan_values[istep])

    def analyze(self):
        super().analyze(constants={'t_dead':0.0})
        meanTpi = 0.0
        num_active_pmts = len(self.p_all["t_period"])
        buf = "{"
        for ifit in range(num_active_pmts):
            meanTpi += self.p_all["t_period"][ifit] * 0.5
            ret = self.p_all["t_period"][ifit] * 0.5 * 1e6
            buf += "{:f},".format(ret)
            _LOGGER.info(
                "Fit %i:\n\tTpi = (%f +- %f) us\n\ttau = (%f +- %f) us",
                ifit,
                ret,
                self.p_error_all["t_period"][ifit] * 0.5e6,
                self.p_all["tau_decay"][ifit] * 1e6,
                self.p_error_all["tau_decay"][ifit] * 1e6,
            )
        buf += "}"
        print(buf)
        print(self.p_all["t_dead"])
        meanTpi /= num_active_pmts


class MW_Fscan(BasicEnvironment, artiq_env.Experiment):
    """MW.Freq Scan

    Inherit from BasicEnvironment which handles most of the infrastructure
    """

    data_folder = "mw_fscan"
    applet_name = "MW Freq Scan"
    applet_group = "MW"
    fit_type = umdfit.positive_gaussian
    # fit_type = fit.sinc_2


    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=42.820 * MHz,
                    stop=42.821 * MHz,
                    npoints=25,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="MHz",
                global_min=0,
                ndecimals=9,
            ),
        )
        self.setattr_argument(
            "mw_amplitude",
            artiq_env.NumberValue(default=999, min=0, scale=1, max=1000, ndecimals=0),
        )
        self.setattr_argument(
            "pi_time", artiq_env.NumberValue(default=5 * ms, unit="ms")
        )

        self.setattr_argument(
            "set_globals",
            artiq_env.BooleanValue(default=False)
        )
        super().build()

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.scan_values = [t for t in self.scan_range]
        self.num_steps = len(self.scan_values)
        self.mw_amplitude = np.int32(self.mw_amplitude)
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
    def custom_kn_init(self):
        """Add custom initialization steps before experiment execution."""
        self.core.break_realtime()
        self.mw.init()
        self.mw.update_amp(self.mw_amplitude)
        delay(200 * us)


    @kernel
    def prepare_step(self, istep: TInt32):
        self.mw.update_freq(self.scan_values[istep])

    @kernel
    def main_experiment(self, istep, ishot):
        self.mw.pulse(self.pi_time)

    def analyze(self):
        super().analyze()
        num_active_pmts = len(self.pmt_array.active_pmts)
        buf = "{"
        all_fits = []
        for ifit in range(num_active_pmts):
            ret = self.p_all["x0"][ifit]
            all_fits.append(ret)
            buf = buf + "%f," % ret
            print(
                "Fit",
                ifit,
                ": ",
                "x0 = (",
                self.p_all["x0"][ifit],
                " +- ",
                self.p_error_all["x0"][ifit],
                ")",
            )
            # print(
            #     "Fit",
            #     ifit,
            #     ": ",
            #     "sigma = (",
            #     self.p_all["sigma"][ifit],
            #     " +- ",
            #     self.p_error_all["sigma"][ifit],
            #     ")",
            # )
        buf = buf + "}"
        print(buf)

        avg_val = np.mean(all_fits)

        if self.set_globals:
            print("setting global MW Frequency to {}".format(avg_val))
            self.set_dataset(
                "global.Ion_Freqs.MW_Freq",
                avg_val,
                persist=True,
            )


class MW_Ramsey(BasicEnvironment, artiq_env.Experiment):
    """MW.Ramsey
    Inherit from BasicEnvironment which handles most of the infrastructure
    """

    data_folder = "mw_ramsey"
    applet_name = "MW Ramsey"
    applet_group = "MW"

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
                    start=0 * ms,
                    stop=1000 * ms,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                ndecimals=3,
                global_min=0,
                unit="ms",
            ),
        )

        self.setattr_argument(
            "scan_type",
            artiq_env.EnumerationValue(["phase_scan", "tau_scan"], default="tau_scan"),
        )

        self.setattr_argument(
            "wait_time", artiq_env.NumberValue(default=10* ms, unit="ms")
        )

        self.setattr_argument(
            "num_pulses", artiq_env.NumberValue(default=5, unit="pulses", scale=1, ndecimals=0)
        )
        self.setattr_argument(
            "pi_time", artiq_env.NumberValue(default=5 * ms, unit="ms", ndecimals=3)
        )
        self.setattr_argument(
            "detuning", artiq_env.NumberValue(default=0*kHz, unit="kHz",ndecimals=6)
        )

        self.setattr_argument(
            "analysis_phase", artiq_env.NumberValue(default=0, unit="degrees", scale=1)
        )
        self.setattr_argument(
            "set_global_MWfreq", artiq_env.BooleanValue(default=False)
        )

        super().build()

    def prepare(self):
        # Make sure the user selects tau scans if setting the global MW freq
        assert not(self.set_global_MWfreq and self.scan_type == "phase_scan"),\
            "Currently only tau scans support setting the global MW Freq"

        if self.scan_type == "tau_scan":
            self.tau = [t for t in self.tau_scan]
            self.scan_values = self.tau.copy()
            self.phase = [self.analysis_phase for _ in self.tau_scan]
            self.fit_type = fit.cos_fft


        elif self.scan_type == "phase_scan":
            self.tau = [self.wait_time for _ in self.phase_scan]
            self.phase = [p for p in self.phase_scan]
            self.scan_values = self.phase.copy()
            self.fit_type = fit.cos

        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )

        self.resonance = self.get_dataset("global.Ion_Freqs.MW_Freq")
        self.freq = self.resonance + self.detuning
        #self.freq = self.resonance
        # self.num_pulses = int(self.num_pulses)
        # self.t = self.tau - ((self.num_pulses+1)*self.pi_time)
        # if self.t <= 0:
        #     _LOGGER.error("Pulse sequence is longer than all of the scan ranges. Cannot Execute")
        #
        #
        self.num_steps = len(self.scan_values)
        #self.twait = self.t/(2*self.num_pulses)
        #self.twait_mu = self.core.seconds_to_mu(self.tau)
        #

        self.phase_mu = [phase_to_mu(2*np.pi/360*p) for p in self.phase]
        self.x_phase_mu = phase_to_mu(0.0)
        self.y_phase_mu = phase_to_mu(90/360*2*math.pi)
        #
        self.pi_time_mu = self.core.seconds_to_mu(self.pi_time)
        self.pihalf_time_mu = self.core.seconds_to_mu(self.pi_time/2)
        # self.detuning_freq_hz = self.detuning*1e3
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
    def custom_kn_init(self):
        """Add custom initialization steps before experiment execution."""
        self.core.break_realtime()
        self.mw.freq = freq_to_mu(self.freq)
        #self.mw.phase = self.x_phase_mu
        self.mw.phase = 0
        self.mw.init()

    @kernel
    def main_experiment(self, istep, ishot):
        self.mw.update_phase_mu(self.x_phase_mu)
        delay(5*us)
        self.mw.pulse_mu(self.pihalf_time_mu)

        # p = 1  # Start with a Y echo
        # for ipulse in range(self.num_pulses):
        #
        #     # Alternate between X and Y echos
        #     if p == 1:
        #         self.mw.update_phase_mu(self.y_phase_mu)
        #         p = 0
        #     elif p == 0:
        #         self.mw.update_phase_mu(self.x_phase_mu)
        #         p = 1
        #
        #     delay_mu(self.twait_mu)
        #     self.mw.pulse_mu(self.pi_time_mu)
        #     delay_mu(self.twait_mu)
        with parallel:
            self.mw.update_phase_mu(self.phase_mu[istep])
            delay(self.tau[istep])

        self.mw.pulse_mu(self.pihalf_time_mu)

    def analyze(self):
        """Analyze and Fit data."""
        if self.scan_type == "tau_scan":
            constants = {}
        else:
            constants = {"period": 360}

        super().analyze(constants=constants)

        # Hardcode the center PMT to 8
        center_pmt_idx = self.pmt_array.active_pmts.index(8)
        num_active_pmts = len(self.pmt_array.active_pmts)

        for ifit in range(num_active_pmts):
            phase = -self.p_all["x0"][ifit]/self.p_all["period"][ifit] * 360
            print(
                "Fit {:d}: \t C = {:3.3f} \t phi = {:3.2f} deg \t period = {:3.2f} ms".format(
                    ifit,
                    self.p_all["a"][ifit] * 2,
                    phase,
                    self.p_all["period"][ifit]*1e3)
            )
        # p['a']*np.cos(2*np.pi*(x-p['x0'])/p['period'])
        center_period = self.p_all["period"][center_pmt_idx]
        center_phase  = -2*np.pi/center_period*self.p_all["x0"][center_pmt_idx]
        print("Center phase = ", center_phase)
        if self.p_all["a"][center_pmt_idx] < 0:
            center_period = -center_period
        if np.cos(center_phase) < 0:
            center_period = -center_period
        print("abs freq shift = ", 1/center_period)
        freq_shift = self.detuning + 1/center_period
        print("net freq shift = ", freq_shift)
        if self.set_global_MWfreq:
            new_freq = self.resonance+freq_shift
            print("Setting global.Ion_Freqs.MW_Freq = {:9.7f}".format(new_freq))
            self.set_dataset(
                "global.Ion_Freqs.MW_Freq",
                new_freq,
                persist=True,
            )


class MW_T2XYXY(BasicEnvironment, artiq_env.Experiment):
    """MW.T2_XYXY
    Inherit from BasicEnvironment which handles most of the infrastructure
    """

    data_folder = "mw_t2"
    applet_name = "MW T2"
    applet_group = "MW"
    fit_type = fit.sin

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0,
                    stop=360,
                    npoints=8,
                    randomize=True,
                    seed=int(time.time()),
                ),
                unit="degrees",
                global_min=0,
                global_max=360,
                scale=1
            ),
        )

        self.setattr_argument(
            "tau", artiq_env.NumberValue(default=1000*ms, unit="ms")
        )

        self.setattr_argument(
            "num_pulses", artiq_env.NumberValue(default=5, unit="pulses", scale=1, ndecimals=0)
        )
        self.setattr_argument(
            "pi_time", artiq_env.NumberValue(default=5 * ms, unit="ms", ndecimals=3)
        )
        self.setattr_argument(
            "detuning", artiq_env.NumberValue(default=0*kHz, unit="kHz")
        )

        super().build()

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )

        self.scan_values = [t for t in self.scan_range]

        self.resonance = self.get_dataset("global.Ion_Freqs.MW_Freq")
        self.freq = self.resonance + self.detuning
        self.num_pulses = int(self.num_pulses)
        self.t = self.tau - ((self.num_pulses+1)*self.pi_time)
        if self.t <= 0:
            _LOGGER.error("Pulse sequence is longer than all of the scan ranges. Cannot Execute")


        self.num_steps = len(self.scan_values)
        self.twait = self.t/(2*self.num_pulses)
        self.twait_mu = self.core.seconds_to_mu(self.t/(2*self.num_pulses))

        self.phase_scan = [phase_to_mu(ideg/360*2*math.pi) for ideg in self.scan_values]
        self.x_phase_mu = phase_to_mu(0.0)
        self.y_phase_mu = phase_to_mu(90/360*2*math.pi)

        self.pi_time_mu = self.core.seconds_to_mu(self.pi_time)
        self.pihalf_time_mu = self.core.seconds_to_mu(self.pi_time/2)

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
    def custom_kn_init(self):
        """Add custom initialization steps before experiment execution."""
        self.core.break_realtime()
        self.mw.freq = freq_to_mu(self.freq)
        self.mw.phase = self.x_phase_mu
        self.mw.init()

    @kernel
    def main_experiment(self, istep, ishot):
        self.mw.update_phase_mu(self.x_phase_mu)
        self.mw.pulse_mu(self.pihalf_time_mu)

        p = 1  # Start with a Y echo
        for ipulse in range(self.num_pulses):

            # Alternate between X and Y echos
            if p == 1:
                self.mw.update_phase_mu(self.y_phase_mu)
                p = 0
            elif p == 0:
                self.mw.update_phase_mu(self.x_phase_mu)
                p = 1

            delay_mu(self.twait_mu)
            self.mw.pulse_mu(self.pi_time_mu)
            delay_mu(self.twait_mu)

        self.mw.update_phase_mu(self.phase_scan[istep])
        self.mw.pulse_mu(self.pihalf_time_mu)

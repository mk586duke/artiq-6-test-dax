import logging
import time
import math

import artiq.language.environment as artiq_env
from artiq.language.environment import HasEnvironment
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
from artiq.language.types import TFloat, TInt32, TInt64
from artiq.language.units import A, MHz, ms, s, us,kHz


import euriqafrontend.fitting as umd_fit
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend.repository.basic_environment import BasicEnvironment

from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqabackend.coredevice.ad9912 import phase_to_mu

_LOGGER = logging.getLogger(__name__)

class DDS_MW_StarkShift(BasicEnvironment, artiq_env.Experiment):
    """Stark shift - single beam
    """

    data_folder = "stark_shift"
    applet_name = "Stark Shift"
    applet_group = "Raman"
    fit_type = fit.sin_fft
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
                    start=0,
                    stop=1000,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="",
                global_min=-20,
            ),
        )

        self.setattr_argument(
            "scan_parameter",
            artiq_env.EnumerationValue(
                [
                    "raman_time",
                    "MW Phase Y (rad)",
                    "raman_detuning"
                ],
                default="raman_time",
            ),
        )

        self.setattr_argument("enable_pi_pulse", artiq_env.BooleanValue(default=True))

        self.setattr_argument(
            "beam_on",
            artiq_env.EnumerationValue(
                [
                    "global",
                    "individual",
                    "neither"
                ],
                default="global",
            ),
        )

        self.setattr_argument(
            "raman_detuning",
            NumberValue(
                0,
                unit="MHz",
                min=-20 * MHz,
                max=40 * MHz,
                step=0.0025 * MHz,
                ndecimals=6,
            ),
        )
        self.setattr_argument(
            "mw_detuning", artiq_env.NumberValue(default=0*kHz, unit="kHz", ndecimals=6)
        )

        self.setattr_argument("dds_global_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("dds_ind_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("mw_phase", artiq_env.NumberValue(default=0.0))
        self.setattr_argument(
            "mw_pi_time", artiq_env.NumberValue(default=5 * ms, unit="ms", ndecimals=3)
        )

        self.setattr_argument(
            "raman_time", artiq_env.NumberValue(default=5 * us, unit="us", ndecimals=3)
        )

        super().build()

    def prepare(self):

        if self.scan_parameter == "raman_time":
            self.xlabel = "Duration (us)"
        elif self.scan_parameter == "raman_detuning":
            self.xlabel = "Raman Detuning (MHz)"
        else:
            self.xlabel = "Phase (rad)"

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
        self.raman_detuning_mu = freq_to_mu(self.raman_detuning)
        self.global_amp_mu = np.int32(self.dds_global_amp)
        self.ind_amp_mu = np.int32(self.dds_ind_amp)

        # microwave related things
        self.resonance = self.get_dataset("global.Ion_Freqs.MW_Freq")
        self.freq = self.resonance + self.mw_detuning

        self.x_phase_mu = phase_to_mu(0.0)
        self.y_phase_mu = phase_to_mu(90/360*2*math.pi)

        self.pi_time_mu = self.core.seconds_to_mu(self.mw_pi_time)
        self.pihalf_time_mu = self.core.seconds_to_mu(self.mw_pi_time/2)


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

        # MW stuff
        self.core.break_realtime()
        self.mw.freq = freq_to_mu(self.freq)
        self.mw.phase = self.x_phase_mu
        self.mw.init()

        #
        # if self.beam_on == "global":
        #     self.ind_amp_mu = 0
        # else:
        #     self.global_amp_mu = 0

    @kernel
    def prepare_step(self, istep: TInt32):
        pass
        # self.calib_wait_time_mu = self.scan_values[istep]

    @kernel
    def main_experiment(self, istep, ishot):

        self.mw.update_phase_mu(self.x_phase_mu)  # this advances the timeline

        if self.scan_parameter == "raman_time":
            raman_time = self.scan_values[istep]*us
            yphase = self.y_phase_mu
        elif self.scan_parameter == "raman_detuning":
            raman_time = self.raman_time
            self.raman_detuning_mu = freq_to_mu(self.scan_values[istep]*1e6)
            yphase = self.y_phase_mu
        else:
            raman_time = self.raman_time
            yphase = phase_to_mu(self.scan_values[istep])

        self.raman.set_global_detuning_mu(self.raman_detuning_mu)

        # regular raman rabi pulse
        self.raman.global_dds.update_amp(self.global_amp_mu)
        delay(5 * us)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_global_detuning_mu(self.raman_detuning_mu)
        delay(120 * us)

        # microwave  - initially set to 0.0 phase (X)
        self.mw.pulse_mu(self.pihalf_time_mu)
        # self.raman.pulse(0.5*us)

        # dds A with scanned time
        if self.beam_on == "global":
            self.raman.global_dds.pulse(raman_time)
        elif self.beam_on == "individual":
            self.raman.switchnet_dds.pulse(raman_time)
        else:
            delay(raman_time)
        delay(5*us)
        # microwave pi x
        if self.enable_pi_pulse:
            self.mw.pulse_mu(self.pi_time_mu)
            #self.raman.pulse(1.1*us)

        # ##dds B with scanned time
        # if self.beam_on == "global":
        #     self.raman.global_dds.pulse(raman_time)
        # else:
        #     self.raman.switchnet_dds.pulse(raman_time)
        # delay(5*us)

        delay(raman_time)
        with parallel:
            self.mw.update_phase_mu(yphase)  # this advances the timeline
            delay(5*us)

        # final mw pulse pi/2 y
        self.mw.pulse_mu(self.pihalf_time_mu)
        # self.raman.pulse(0.5 * us)

    def analyze(self):
        """Analyze and Fit data"""
        """Threshold values and analyze."""
        if self.scan_parameter == "raman_time":
            super().analyze()
        else:
            super().analyze(constants={"n_periods": 1})

        # if self.fit_type is not None:
        #     for ifit in range(len(self.p_all["x0"])):
        #         _LOGGER.info(
        #             "period = %f, x0 = %f, a = %f",
        #             self.p_all["period"][ifit],
        #             self.p_all["x0"][ifit],
        #             self.p_all["a"][ifit],
        #         )
        num_active_pmts = len(self.p_all["period"])
        buf = "{"
        for ifit in range(num_active_pmts):
            ret = self.p_all["period"][ifit]
            buf += "{:f},".format(ret)
            _LOGGER.info(
                "Fit %i:\n\tTpi = (%f +- %f) us",
                ifit,
                ret,
                self.p_error_all["period"][ifit]
            )
        buf += "}"
        print(buf)




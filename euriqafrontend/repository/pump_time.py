import logging
import time

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import numpy as np
import oitg.fitting as fit
from artiq.experiment import TerminationRequested
from artiq.language.core import delay
from artiq.language.core import host_only
from artiq.language.core import kernel, now_mu, rpc
from artiq.language.units import us, MHz, ms
from artiq.language.types import TInt32

import euriqafrontend.fitting as umd_fit
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend.repository.basic_environment import BasicEnvironment

_LOGGER = logging.getLogger(__name__)


class PumpTime(BasicEnvironment, artiq_env.Experiment):
    """PumpDetect.PumpTime
    """

    data_folder = "pump_detect"
    applet_name = "Pump"
    applet_group = "Raman Calib"
    fit_type = fit.exponential_decay

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

        super().build()
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )

    def prepare(self):
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
    def experiment_loop(self):
        """Run the experiment on the core device."""
        calib_counts = [0] * self.num_pmts
        doppler_counts = [0] * self.num_pmts
        ss_counts = [0] * self.num_pmts
        detect_counts = [0] * self.num_pmts

        # Loop over main experimental scan parameters
        for istep in range(self.num_steps):
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()

            # Loop over to gather statistics
            for ishot in range(self.num_shots):

                # Doppler Cooling
                stopcounter_doppler_mu = self.doppler_cool(monitor=False)

                # Second Stage Cooling
                if self.do_SS_cool:
                    stopcounter_ss_mu = self.second_stage_cool(monitor=True)
                    delay(20 * us)
                else:
                    stopcounter_ss_mu = now_mu()

                delay(1*us)#

                self.pump_detect.prepare_pump()
                delay(200*us) # Wait for attenuators on DDS from SS cooling

                # Pump to the 0 state
                self.pump_detect.pulse(self.scan_values[istep])
                delay(10*us)

                self.pump_detect.prepare_detect()
                delay(10*us)

                # Detect Ions
                stopcounter_detect_mu = self.detect()

                doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)
                ss_counts = self.pmt_array.count(up_to_time_mu=stopcounter_ss_mu, buffer=ss_counts)
                detect_counts = self.pmt_array.count(up_to_time_mu=stopcounter_detect_mu, buffer=detect_counts)

                self.save_counts(ishot=ishot,
                                 istep=istep,
                                 calib_counts=calib_counts,
                                 doppler_counts=doppler_counts,
                                 ss_counts=ss_counts,
                                 ss_coolant_counts=ss_counts,
                                 ss_all_counts=ss_counts,
                                 detect_counts=detect_counts,
                                 tlock_feedback=self.calib_tlock_int,
                                 xlock_feedback=self.calib_xlock_int)
                self.core.break_realtime()

            self.threshold_data(istep)



    @kernel
    def custom_kn_init(self):
        self.core.break_realtime()
        delay(10 * us)

    @kernel
    def custom_kn_idle(self):
        self.core.break_realtime()
        self.raman.idle()
        # save ourselves from collisions by setting Doppler cooling power very low
        self.doppler_cooling.set_power(0b01)

    def analyze(self):
        """Analyze and Fit data"""
        """Threshold values and analyze."""
        super().analyze()
        num_active_pmts = len(self.p_all["tau"])
        for ifit in range(num_active_pmts):
            print("Fit {:d} : tau = ({:.3f}  +- {:.3f}) us".format(
                ifit,
                self.p_all["tau"][ifit]*1e6,
                self.p_error_all["tau"][ifit]*1e6,
            )
            )

class SweepResonance(BasicEnvironment, artiq_env.Experiment):
    """PumpDetect.FreqSweep
    """

    data_folder = "pump_detect"
    applet_name = "Pump"
    applet_group = "Raman Calib"
    xlabel = "Detuning"
    ylabel = "Avg. Detection Counts"
    fit_type = umd_fit.positive_gaussian

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "detuning",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=-20*MHz,
                    stop=20*MHz,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="MHz"
            ),
        )

        self.setattr_argument("mw_tpi", artiq_env.NumberValue(default=20*ms, unit="ms"))

        super().build()
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type, xlabel=self.xlabel, ylabel=self.ylabel
        )

    def prepare(self):

        self._detect_1_freq_global =  + 1 / 2 * self.get_dataset("global.Detect.Yb.Detuning")
        self._detect_2_freq_global =  + 1 / 2 * self.get_dataset("global.Detect.Yb.Detuning")

        self.scan_values = [t for t in self.detuning]

        base_freq_1 = self.get_dataset("global.CW_Lasers.Yb.Resonance_PumpDetectAOM1_Freq")
        base_freq_2 = self.get_dataset("global.CW_Lasers.Yb.Resonance_PumpDetectAOM2_Freq")

        self.detect_1_freq_mu = [freq_to_mu(base_freq_1 + 1/2*idetuning) for idetuning in self.scan_values]
        self.detect_2_freq_mu = [freq_to_mu(base_freq_2 + 1/2*idetuning) for idetuning in self.scan_values]

        # print([base_freq_1 + 1/2*idetuning for idetuning in self.scan_values])
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
    def experiment_loop(self):
        """Run the experiment on the core device."""
        calib_counts = [0] * self.num_pmts
        doppler_counts = [0] * self.num_pmts
        ss_counts = [0] * self.num_pmts
        detect_counts = [0] * self.num_pmts

        # Loop over main experimental scan parameters
        for istep in range(self.num_steps):
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()

            # Loop over to gather statistics
            for ishot in range(self.num_shots):

                # Doppler Cooling
                stopcounter_doppler_mu = self.doppler_cool(monitor=False)

                # Second Stage Cooling
                if self.do_SS_cool:
                    stopcounter_ss_mu = self.second_stage_cool(monitor=False)
                else:
                    stopcounter_ss_mu = now_mu()

                delay(1*us)#

                self.pump_detect.prepare_pump()
                delay(200*us) # Wait for attenuators on DDS from SS cooling

                # Pump to the 0 state
                self.pump_detect.pump()

                #Overwrite detection scan frequency before detuning
                self.pump_detect.detect_frequency_mu[0] = self.detect_1_freq_mu[istep]
                self.pump_detect.detect_frequency_mu[1] = self.detect_2_freq_mu[istep]
                self.pump_detect.prepare_detect()
                delay(10*us)

                self.mw.pulse(self.mw_tpi)

                # Detect Ions
                stopcounter_detect_mu = self.detect()

                doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)
                ss_counts = self.pmt_array.count(up_to_time_mu=stopcounter_ss_mu, buffer=ss_counts)
                detect_counts = self.pmt_array.count(up_to_time_mu=stopcounter_detect_mu, buffer=detect_counts)

                self.save_counts(ishot=ishot,
                                 istep=istep,
                                 calib_counts=calib_counts,
                                 doppler_counts=doppler_counts,
                                 ss_counts=ss_counts,
                                 ss_coolant_counts=ss_counts,
                                 ss_all_counts=ss_counts,
                                 detect_counts=detect_counts,
                                 tlock_feedback=self.calib_tlock_int,
                                 xlock_feedback=self.calib_xlock_int)
                self.core.break_realtime()

            self.threshold_data(istep)

    @rpc
    def threshold_data(self, istep: TInt32):

        counts = np.array(self.get_experiment_data("raw_counts"))
        avg_counts = np.mean(counts[:, :, istep], axis=1).tolist()
        np_thresh = np.array(avg_counts,ndmin=2).T

        self.mutate_experiment_data(
            "avg_thresh",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
            np_thresh,
        )

    @kernel
    def custom_kn_init(self):
        self.core.break_realtime()
        delay(10 * us)

    @kernel
    def custom_kn_idle(self):
        self.core.break_realtime()
        self.raman.idle()
        # save ourselves from collisions by setting Doppler cooling power very low
        self.doppler_cooling.set_power(0b01)

    def analyze(self):
        """Analyze and Fit data"""
        """Threshold values and analyze."""
        pass
        # super().analyze()
        # for ifit in range(self.y_fit_all.shape[0]):
        #     print(
        #         "Fit",
        #         ifit,
        #         ": ",
        #         "tau = (",
        #         self.p_all["tau"][ifit] * 1e6,
        #         " +- ",
        #         self.p_error_all["tau"][ifit] * 1e6,
        #         ") us",
        #     )
        #     print(
        #         "Fit",
        #         ifit,
        #         ": ",
        #         "offset = ",
        #         self.p_all["y_inf"][ifit],
        #         "+-",
        #         self.p_error_all["y_inf"][ifit],
        #     )

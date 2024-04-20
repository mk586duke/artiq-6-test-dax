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


class RamanPiezoLock(BasicEnvironment, artiq_env.Experiment):
    """Raman.PiezoLock."""

    data_folder = "raman_rabi"
    applet_name = "Raman Rabi"
    applet_group = "Raman"
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
        # dac8568.vout_to_mu(
        #     self.get_dataset("global.Raman.Piezos.Ind_FinalX"),
        #     self.raman.ind_final_x_piezo.SandiaSerialDAC.V_OUT_MAX,
        # )

        # self.piezo_mu = self.get_dataset("monitor.Stark_Detuning")

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
        self.piezo_lock_int = 0
        self.ind_final_x_piezo_0_mu = self.raman.ind_final_x_piezo.value_mu

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
    def experiment_loop(self):
        """Run the experiment on the core device."""
        # Create input buffers
        calib_counts = [0] * self.num_pmts
        doppler_counts = [0] * self.num_pmts
        ss_counts = [0] * self.num_pmts
        detect_counts = [0] * self.num_pmts

        self.core.break_realtime()
        # Set switch network to sequence mode
        self.raman.setRunningSN(True)
        delay(10 * us)

        # Loop over main experimental scan parameters
        for istep in range(len(self.scan_values)):

            # check for pause/abort from the host
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()

            # do any initialization that needs to be done for each step
            # such as variable hardware-shuttling, variable piezo setting, etc.
            self.prepare_step(istep)

            # Loop over to gather statistics
            for ishot in range(self.num_shots):

                # Reset Switch Network
                self.raman.resetSN()
                delay(10 * us)
                # Advance Switch Network to its first line
                self.raman.advanceSN()

                if self.do_calib:
                    # prepare the pump/detect path for pumping
                    with parallel:
                        with sequential:
                            self.pump_detect.set_param_mu(
                                freq_mu=self.pump_detect.pump_frequency_mu,
                                amp_int=self.pump_detect.pump_amp,
                            )
                            self.pump_detect.write_to_dds()
                        # while, in parallel, preparing the pump EOM
                        self.pump_detect.pump_eom(pump=True)
                        # while, in parallel, doppler-cooling
                        self.doppler_cool()
                    # pump
                    self.pump_detect.pulse_mu(2 * self.pump_detect.pump_duration_mu)
                    # apply the Raman pulse for intensity-detection
                    self.calib_stark_shift(self.calib_wait_time_mu)
                    # discard counts up to this time
                    stopcounter_dummy_mu = now_mu()
                    # detect
                    stopcounter_calib_mu = self.detect()
                    delay(5 * us)
                    self.raman.advanceSN()
                    delay(5 * us)

                else:
                    stopcounter_dummy_mu = now_mu()
                    stopcounter_calib_mu = now_mu()

                # prepare the Raman system for SBC
                # do this early to give time to the DDS attenuators to the target level
                if self.do_SBC:
                    self.sbc_prepare_raman()

                # Doppler Cooling
                stopcounter_doppler_mu = self.doppler_cool(monitor=True)

                # Second Stage Cooling
                if self.do_SS_cool:
                    stopcounter_ss_mu = self.second_stage_cool(monitor=True)
                else:
                    stopcounter_ss_mu = now_mu()

                if self.do_pump or self.do_SBC:
                    # prepare pumping
                    with parallel:
                        with sequential:
                            self.pump_detect.set_param_mu(
                                freq_mu=self.pump_detect.pump_frequency_mu,
                                amp_int=self.pump_detect.pump_amp,
                            )
                            self.pump_detect.write_to_dds()
                        self.pump_detect.pump_eom(pump=True)

                # Sideband Cooling
                if self.do_SBC:
                    self.sbc_sweep()

                # Slow pump to the 0 state
                if self.do_pump:
                    self.doppler_cooling.off()
                    self.pump_detect.pump_eom(pump=True)
                    self.pump()
                    self.pump_detect.pump_eom(pump=False)

                # self.pump_detect.pump_eom(pump=True)

                # self.pump_detect.update_amp(amp_int=[np.int32(500), np.int32(500)])
                # self.pmt_array.clear_buffer()
                # delay(150 * us)
                # self.pump_detect.pulse(100*us)

                # delay(2000*us)
                self.pmt_array.clear_buffer()

                delay(5 * us)
                self.raman.advanceSN()
                delay(5 * us)
                # Do not advance the switch network if using the AWG
                # if not self.use_AWG:
                #    self.raman.advanceSN()
                #    delay(5*us)

                # Perform the main experiment
                self.main_experiment(istep, ishot)

                # Detect Ions
                stopcounter_detect_mu = self.detect()

                # Readout out counts and save the data
                if self.do_calib:
                    calib_counts = self.pmt_array.count(
                        up_to_time_mu=stopcounter_dummy_mu, buffer=calib_counts
                    )
                    calib_counts = self.pmt_array.count(
                        up_to_time_mu=stopcounter_calib_mu, buffer=calib_counts
                    )
                # doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)
                # ss_counts = self.pmt_array.count(up_to_time_mu=stopcounter_ss_mu, buffer=ss_counts)
                detect_counts = self.pmt_array.count(
                    up_to_time_mu=stopcounter_detect_mu, buffer=detect_counts
                )

                self.save_counts(
                    ishot=ishot,
                    istep=istep,
                    calib_counts=calib_counts,
                    doppler_counts=doppler_counts,
                    ss_counts=ss_counts,
                    detect_counts=detect_counts,
                )

                self.core.break_realtime()  # TODO do we need this?
                if self.lock_active:
                    if detect_counts[0] > 1:
                        self.piezo_lock_int += 1
                    else:
                        self.piezo_lock_int -= 1
                    self.raman.ind_final_x_piezo.value_mu = (
                        self.ind_final_x_piezo_0_mu - (self.piezo_lock_int << 5)
                    )
                    self.raman.ind_final_x_piezo.update_value()

            # threshold raw counts, save data, and push it to the plot applet
            self.threshold_data(istep)

    @kernel
    def main_experiment(self, istep, ishot):

        # self.raman.global_dds.update_amp(self.global_amp_mu)
        # delay(5 * us)
        # self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        # delay(5 * us)
        # self.raman.set_global_detuning_mu(self.detuning_mu)
        # delay(120 * us)
        # # delay_mu(self.scan_values[istep])

        #        self.raman.switchnet_dds.on()
        #        delay(self.scan_values[istep])
        #        self.raman.global_dds.on()
        #        delay(3.6 * us)  # self.scan_values[istep])
        #        self.raman.switchnet_dds.off()
        #        self.raman.global_dds.off()
        # self.raman.pulse_mu(self.scan_values[istep])
        self.calib_stark_shift(self.calib_wait_time_mu)

    #        self.raman.pulse(self.scan_values[istep])
    # delay(self.scan_values[istep])

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        meanTpi = 0.0
        num_active_pmts = len(self.p_all["t_period"])
        for ifit in range(num_active_pmts):
            meanTpi += self.p_all["t_period"][ifit] * 0.5
            _LOGGER.info(
                "Fit %i:\n\tTpi = (%f +- %f) us\n\ttau = (%f +- %f) us",
                ifit,
                self.p_all["t_period"][ifit] * 0.5 * 1e6,
                self.p_error_all["t_period"][ifit] * 0.5 * 1e6,
                self.p_all["tau_decay"][ifit] * 1e6,
                self.p_error_all["tau_decay"][ifit] * 1e6,
            )
        meanTpi /= num_active_pmts

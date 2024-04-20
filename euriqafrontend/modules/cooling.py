import numpy as np
from abc import abstractmethod, ABC
from artiq.experiment import TBool
from artiq.experiment import TInt64
from artiq.language import delay
from artiq.language import kernel
from artiq.language import parallel
from artiq.language.core import delay_mu
from artiq.language.core import now_mu
from artiq.language.environment import HasEnvironment

from artiq.language.units import us

from euriqafrontend.modules.awg import AWG
from euriqafrontend.modules.cw_lasers import DopplerCooling
from euriqafrontend.modules.cw_lasers import PumpDetect
from euriqafrontend.modules.cw_lasers import SSCooling_Settings
from euriqafrontend.modules.cw_lasers import DopplerCoolingCoolant
from euriqafrontend.modules.cw_lasers import SDCooling_Settings
from euriqafrontend.modules.cw_lasers import BaseDoubleDDS

from euriqafrontend.modules.pmt import PMTArray
from euriqafrontend.modules.raman import Raman
from euriqafrontend.modules.raman import SBC_DDS


class Cooling(HasEnvironment, ABC):
    """This super-module contains subroutines for cooling that span many different hardware modules including SBC,
    Second stage cooling and doppler cooling.

    THIS CLASS SHOULD BE INHERITED, NOT INSTANTIATED.

    If you instantiate this class you will end up with multiple copies of the same hardware which will lead to a lot
    of confusion.
    """

    @abstractmethod
    def build(self):
        """Initialize experiment & variables."""
        # basic ARTIQ devices
        self.setattr_device("core")

        self.pmt_array = PMTArray(self)
        self.doppler_cooling = DopplerCooling(self)
        self.doppler_cooling_coolant = DopplerCoolingCoolant(self)
        self.pump_detect = PumpDetect(self)
        self.ss_cooling = SSCooling_Settings(self)
        self.SDDDS = BaseDoubleDDS(
            self,
            dds_name1 = "sd_435_dds1",
            gui_freq_name1 = "SD_DDS1_freq",
            gui_amp_name1 = "SD_DDS1_amp",
            gui_phase_name1="SD_DDS1_phase",
            gui_default_freq1 = 198.2e6,
            gui_default_amp1 = 1000,
            dds_name2 = "sd_435_dds2",
            gui_freq_name2="SD_DDS2_freq",
            gui_amp_name2="SD_DDS2_amp",
            gui_phase_name2="SD_DDS2_phase",
            gui_default_freq2=200e6,
            gui_default_amp2=1000,
            gui_group_name="Quadrupole DDSs"
        )
        self.sd_cooling = SDCooling_Settings(self)
        #self.sbc = SBC_DDS(self)
        self.raman = Raman(self)
        self.setattr_device("eom_935_3ghz")
        self.setattr_device("freq_shift_935")

    def prepare(self):
        """Pre-calculate any values before addressing hardware."""
        # Run prepare method of all the imported modules
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

    @kernel
    def doppler_cool(self, monitor: TBool=False) -> TInt64:
        # Doppler cooling of 171 and 172 are simultaneously turned on
        # Doppler counts are from both isotopes simultaneously
        self.doppler_cooling.update_freq_mu(freq_mu=self.doppler_cooling.freq_mu)
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        # TODO: Add a DDS for 370(172) aom
        self.doppler_cooling_coolant.cool(True)

        if monitor is True:
            stopcounter_doppler_mu = self.pmt_array.gate_rising_mu(self.doppler_cooling.doppler_cooling_monitor_duration_mu)
            delay_mu(self.doppler_cooling.doppler_cooling_duration_mu-self.doppler_cooling.doppler_cooling_monitor_duration_mu)
        else:
            delay_mu(self.doppler_cooling.doppler_cooling_duration_mu)
            stopcounter_doppler_mu = now_mu()

        self.doppler_cooling.off()
        self.doppler_cooling_coolant.cool(False)

        return stopcounter_doppler_mu

    # @kernel
    # def doppler_cool_coolant(self, monitor: TBool=False) -> TInt64:
    #     self.doppler_cooling_coolant.cool(True)
    #
    #     if monitor:
    #         # for now it is set to be the same as 171
    #         stopcounter_doppler_mu = self.pmt_array.gate_rising_mu(self.doppler_cooling.doppler_cooling_monitor_duration_mu)
    #         delay_mu(self.doppler_cooling.doppler_cooling_duration_mu-self.doppler_cooling.doppler_cooling_monitor_duration_mu)
    #     else:
    #         delay_mu(self.doppler_cooling.doppler_cooling_duration_mu)
    #         stopcounter_doppler_mu = now_mu()
    #
    #     self.doppler_cooling_coolant.cool(False)
    #
    #     return stopcounter_doppler_mu

    @kernel
    def second_stage_cool(self, monitor: TBool=False) -> TInt64:
        self.doppler_cooling.update_amp(self.ss_cooling.cool_amp)
        delay(10*us)
        with parallel:
            self.doppler_cooling.update_freq_mu(freq_mu=self.ss_cooling.cool_frequency_mu)
            self.doppler_cooling.set_power(0b01)
            self.pump_detect.set_param_mu(freq_mu=self.ss_cooling.detect_frequency_mu,
                                          amp_int=self.ss_cooling.detect_amp)
            self.pump_detect.pump_eom(pump=False)
        self.pump_detect.write_to_dds()
        delay(50 * us) # MC: 5/12/19: what is this for?

        self.doppler_cooling.on()
        self.pump_detect.on()

        if monitor is True:
            stopcounter_ss_mu = self.pmt_array.gate_rising_mu(self.ss_cooling.ss_cooling_monitor_duration_mu)
            delay_mu(self.ss_cooling.ss_cooling_duration_mu - self.ss_cooling.ss_cooling_monitor_duration_mu)
        else:
            delay_mu(self.ss_cooling.ss_cooling_duration_mu)
            stopcounter_ss_mu = now_mu()

        self.doppler_cooling.off()
        self.pump_detect.off()

        return stopcounter_ss_mu

    @kernel
    def second_stage_cool_coolant(self, monitor: TBool = False) -> TInt64:
        self.doppler_cooling_coolant.cool(True)

        if monitor is True:
            stopcounter_ss_coolant_mu = self.pmt_array.gate_rising_mu(self.ss_cooling.ss_cooling_coolant_monitor_duration_mu)
            delay_mu(self.ss_cooling.ss_cooling_coolant_duration_mu - self.ss_cooling.ss_cooling_coolant_monitor_duration_mu)
        else:
            delay_mu(self.ss_cooling.ss_cooling_coolant_duration_mu)
            stopcounter_ss_coolant_mu = now_mu()

        self.doppler_cooling_coolant.cool(False)

        return stopcounter_ss_coolant_mu

    @kernel
    def second_stage_cool_all(self, monitor: TBool = False) -> TInt64:
        with parallel:
            #171
            self.doppler_cooling.update_freq_mu(freq_mu=self.ss_cooling.cool_frequency_mu)
            self.doppler_cooling.set_power(0b01)
            self.pump_detect.set_param_mu(freq_mu=self.ss_cooling.detect_frequency_mu,
                                          amp_int=self.ss_cooling.detect_amp)
            self.pump_detect.pump_eom(pump=False)

        self.pump_detect.write_to_dds()
        delay(50 * us)  # MC: 5/12/19: what is this for?

        self.doppler_cooling.on()
        self.pump_detect.on()
        # 172
        self.doppler_cooling_coolant.cool(True)

        if monitor is True:
            stopcounter_ss_all_mu = self.pmt_array.gate_rising_mu(self.ss_cooling.ss_cooling_all_monitor_duration_mu)
            delay_mu(self.ss_cooling.ss_cooling_all_duration_mu - self.ss_cooling.ss_cooling_all_monitor_duration_mu)
        else:
            delay_mu(self.ss_cooling.ss_cooling_all_duration_mu)
            stopcounter_ss_all_mu = now_mu()

        self.doppler_cooling.off()
        self.pump_detect.off()
        self.doppler_cooling_coolant.cool(False)

        return stopcounter_ss_all_mu

    @kernel
    def sbc_prepare_raman(self):

        if self.sbc.always_on:
            # Set the switch network DDS amplitude to minimum value
            # in order to prepare it for the gradual rampup at the beginning
            # of the SBC
            # The expectation is that this will settle during the SS cooling
            # IF SS COOLING IS OFF, THE SN DDS MAY NOT START AT MIN VALUE
            # AT THE BEGINNING OF ITS RAMPUP IN SBC
            ind_amp_init = np.int32(0)
            global_amp_init = np.int32(0)
        else:
            ind_amp_init = self.sbc.sn_amp
            global_amp_init = self.sbc.global_amp

        # Make sure attenuation is set to 0
        self.raman.set_ind_attenuation(c8=False, c16=False)
        self.raman.switchnet_dds.update_amp(ind_amp_init)

        self.raman.global_dds.update_amp(global_amp_init)
        self.raman.set_global_aom_source(dds=True)
        self.raman.off()

        if self.sbc.parallel_sbc:
            self.sbc.update_amp(ind_amp_init)
            self.sbc.dds_off()

    @kernel
    def sbc_sweep(self):

        if self.sbc.sweep_on:
            sweep_loop = self.sbc.sweep_max
        else:
            sweep_loop = 1

        # For parallel SBC, we define frequency with the global
        self.raman.set_global_detuning_mu(self.sbc.detuned_mode_mu)
        self.raman.set_ind_detuning_mu(np.int64(0))

        self.sbc_riffle_beams(on=True)

        while sweep_loop > 0:

            for imode in range(4):
                if self.sbc.modes_active[imode] == 1:
                    sbc_loops_remaining = self.sbc.loops_max[imode]

                    sbc_rsb_time = self.sbc.mode_t0s_mu[imode]
                    sbc_dtime = self.sbc.mode_dts_mu[imode]

                    while sbc_loops_remaining > 0:

                        self.sbc_mode(global_detuning_mu=self.sbc.mode_detunings_mu[imode], duration_mu=sbc_rsb_time)
                        sbc_rsb_time += sbc_dtime
                        sbc_loops_remaining -= 1

            sweep_loop -= 1

        self.sbc_riffle_beams(on=False)

    @kernel
    def sbc_sweep_parallel(self):

        if self.sbc.sweep_on:
            sweep_loop = self.sbc.sweep_max
        else:
            sweep_loop = 1

        # Map tones onto the DDS. This must match what the switch network is expecting in riffle command.
        # For parallel SBC we define frequency with the individuals
        # Current configuration (10/17/19). In order of increasing frequency of the tone
        # Channel e -> switchnet_dds -> addresses the bottom third of the spectrum -> Mode 1
        # Channel d -> sbc2_dds -> addresses the middle third of the spectrum -> Mode 2
        # Channel c -> sbc1_dds -> addresses the top third of the spectrum -> Mode 3
        self.raman.set_global_detuning_mu(self.sbc.detuned_mode_mu)
        self.raman.set_ind_detuning_mu(self.sbc.mode_detunings_mu[0])
        self.sbc.set_scb2_detuning_mu(self.sbc.mode_detunings_mu[1])
        self.sbc.set_scb1_detuning_mu(self.sbc.mode_detunings_mu[2])

        self.sbc_riffle_beams(on=True)

        while sweep_loop > 0:
            sbc_loops_remaining = self.sbc.loops_max[0]
            sbc_rsb_time = self.sbc.mode_t0s_mu[0]
            sbc_dtime = self.sbc.mode_dts_mu[0]

            while sbc_loops_remaining > 0:

                self.sbc_mode(global_detuning_mu=np.int64(0), duration_mu=sbc_rsb_time)
                sbc_rsb_time += sbc_dtime
                sbc_loops_remaining -= 1

            sweep_loop -= 1

        self.sbc_riffle_beams(on=False)

    @kernel
    def sbc_mode(self, global_detuning_mu: TInt64, duration_mu: TInt64):

        if self.sbc.always_on:
            # Put global back on resonance
            self.raman.set_global_detuning_mu(global_detuning_mu)

        else:
            # Can either pulse the global or individual. Here we choose the global
            self.raman.global_dds.on()

        if duration_mu > 0:
            delay_mu(duration_mu)

        if self.sbc.always_on:
            # detune global instead of turning off during SBC to avoid prompt charge response
            self.raman.set_global_detuning_mu(self.sbc.detuned_mode_mu)
        else:
            # Can either pulse the global or individual. Here we choose the global
            self.raman.global_dds.off()

        # Pump back to 0
        self.pump_detect.pulse_mu(self.pump_detect.pump_duration_mu)

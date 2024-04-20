"""ARTIQ module for simplifying access & control of the RF chain.

Uses Mike Goldman's "RF Compiler" device to manage compiling & sending RF waveforms.
Also manages hardware devices to trigger/control the switch network output.
For example, you can select which scan point you are on and then output
the appropriate waveform.
"""
import logging

import numpy as np
from artiq.experiment import BooleanValue
from artiq.experiment import TBool
from artiq.experiment import TInt32
from artiq.language import delay
from artiq.language import host_only
from artiq.language import kernel
from artiq.language.environment import HasEnvironment
from artiq.language.units import MHz
from artiq.language.units import us

import euriqabackend.devices.keysight_awg.interpolation_functions as intfn
from euriqafrontend.modules.raman import Raman

_LOGGER = logging.getLogger(__name__)


class AWG(HasEnvironment):
    """ARTIQ module to prepare & control the switch network."""

    _MONITOR_AMP = 0
    _MONITOR_DETUNING = 0.0  # MHz
    # 10/10/19:
    # WARNING:
    # Setting _MONITOR_IND to False in order to monitor the global AWG output
    # will cause the AWG to output a monitoring tone on its another channel.
    # This tone will be detuned by _MONITOR_DETUNING from the carrier
    # For long gate sequences, VERY low amplitudes of crosstalk from this channel
    # will cause rotations on the qubits.
    # This means that,
    # WHEN RUNNING GATES, _MONITOR_IND NEEDS TO BE SET TO True AND _MONITOR_AMP SET TO 0
    _MONITOR_IND = True  #
    _USE_SHAPED_CALIBRATION = True

    def build(self):
        """Add arguments & devices used by the AWG/Switch network module."""
        # self.amp_ind_Rabi = self.get_argument(
        #     "amp_ind_Rabi",
        #     NumberValue(default=1000, min=0, scale=1, max=1000, ndecimals=0),
        #     group="AWG",
        # )
        # self.amp_global_Rabi = self.get_argument(
        #     "amp_global_Rabi",
        #     NumberValue(default=1000, min=0, scale=1, max=1000, ndecimals=0),
        #     group="AWG",
        # )
        # self.stark_shift_Rabi = self.get_argument(
        #     "stark_shift_Rabi",
        #     NumberValue(default=1000, min=0, scale=1, max=1000, ndecimals=0),
        #     group="AWG",
        # )
        self.setattr_argument(
            "use_calibrated_Tpis", BooleanValue(default=True), group="AWG"
        )
        self.setattr_argument(
            "use_linearity_correction", BooleanValue(default=True), group="AWG"
        )

        self.setattr_device("sn_enable")
        self.setattr_device("sn_advance")
        self.setattr_device("sn_reset")
        self.setattr_device("sn_running")
        self.setattr_device("awg_trigger")

        self.awg_bits = 7
        self.awg_register = [
            self.get_device("awg_bit{:d}".format(i)) for i in range(self.awg_bits)
        ]

        self.raman = Raman(self)  # Needs the Raman module for frequency calculations
        self.setattr_device("rf_compiler")

    def prepare(self):
        """Load the parameters and calibration values that will be used.

        DO NOT SEND THEM TO RF COMPILER. Doing so will override whatever state
        the RF compiler is in, and it could be preparing/running some other experiment.
        """
        # Load in Physical Params
        self.raman.init_raman_param(calc_mu=True) # Make sure the raman frequency parameters are up to date
        self.f_carrier = self.raman._CARRIER_FREQ / MHz
        self.f_ind = self.raman._IND_BASE / MHz

        self.N_ions = np.int32(self.get_dataset("global.AWG.N_ions"))
        self.t_delay = self.raman._GLOBAL_DELAY / us  # pylint: disable=protected-access
        self.Rabi_max = self.get_dataset("global.AWG.Rabi_max")
        self.PI_center_freq_1Q = self.f_ind
        self.PI_center_freq_2Q = self.f_ind
        self.use_shaped_calibration = self._USE_SHAPED_CALIBRATION

        # Load in Monitor Params
        self.monitor_ind = self._MONITOR_IND
        self.monitor_detuning = self._MONITOR_DETUNING
        self.amp_monitor = self._MONITOR_AMP

        # Load in Rabi Params
        self.amp_ind_Rabi = 1000
        self.amp_global_Rabi = 1000
        self.Stark_shift_Rabi = 0

        # Load in SK1 Params
        self.amp_ind_SK1 = 1000
        self.amp_global_SK1 = 1000 #set to match what the amplitude is during XX gates
        self.Tpi_multiplier_SK1 = 1  # 1.2 * 5
        self.Stark_shift_SK1 = 0

        # Load in Calibration Params
        self.amp_ind_calib = 354/4 # set so that, when run uncalibrated, individual amplitude
                                    # matches the amplitude during calibrated SK1s
        self.amp_global_calib = 525 # set to match the amplitude during SK1 pulses

        # Load in SK1 AM Params
        # self.amp_ind_SK1_AM = 345
        # self.amp_ind_SK1_AM = 300
        self.amp_ind_SK1_AM = 425/4 # need to calibrate with total length of pulse - keep it to about 30-40 us
        self.amp_global_SK1_AM = 525 #set to match what the amplitude is during XX gates
        self.theta_SK1_AM = np.pi / 2
        self.envelope_type_SK1_AM = intfn.InterpFunction.FunctionType.full_Gaussian
        self.envelope_scale_SK1_AM = self.get_dataset(
            "global.AWG.SK1_AM.envelope_scale_SK1_AM"
        )
        self.rotation_pulse_length_SK1_AM = self.get_dataset(
            "global.AWG.SK1_AM.rotation_pulse_length_SK1_AM"
        )
        self.correction_pulse_1_length_SK1_AM = self.get_dataset(
            "global.AWG.SK1_AM.correction_pulse_1_length_SK1_AM"
        )
        self.correction_pulse_2_length_SK1_AM = self.get_dataset(
            "global.AWG.SK1_AM.correction_pulse_2_length_SK1_AM"
        )
        self.tweak_motional_frequency_adjust = self.get_dataset(
            "global.gate_tweak.motional_frequency_adjust")
        self.tweak_stark_shift = self.get_dataset(
            "global.gate_tweak.stark_shift")
        self.tweak_sideband_amplitude_imbalance = self.get_dataset(
            "global.gate_tweak.sideband_amplitude_imbalance")

        # Units in MHz
        self.Stark_shift_SK1_AM = 0

        # Units in radians, wraps the SK1 in RZs each with angle half this value
        # self.phase_correction_SK1_AM = -0.304  # Calibrated 6/10/2020
        #self.phase_correction_SK1_AM = -0.2068+0.1871  # Calibrated 6/10/2020
        #self.phase_correction_SK1_AM = -0.035 # Calibrated 11/28/20
        self.phase_correction_SK1_AM = -0.01 # Calibrated 07/11/22

        # Units radians, this provide a constant offset of the XX phase to align it to the SK1 phase;. Globally applied
        # to all of the XX gates
        self.XX_phase_offset = 0.02#0.1

        # Load in Calibration Params
        # run this to re-initialize the calibrated_Tpi dataset
        # We hard-code the "calibrated" Tpis to be just faster than we need them to be for ions 1 and 15 (indices 9 and
        # 23), as well as all currently unused (15-ion chain) channels (indices 0-8 and 24-31)
        Tpi_adequate = 0.99 * 1 / (2 * self.Rabi_max)
        #self.set_dataset("global.AWG.calibrated_Tpi", [Tpi_adequate] * 32, persist=True)

        self.calibrated_Tpi = self.get_dataset("global.AWG.calibrated_Tpi")

        for i in range(10):
            self.calibrated_Tpi[i] = Tpi_adequate
        for i in range(9):
            self.calibrated_Tpi[23 + i] = Tpi_adequate
        _LOGGER.debug("Got calibrated Tpis = %s", self.calibrated_Tpi)

        # mult = 1.0
        # self.calibrated_Tpi[9 + 6] *= mult
        # self.calibrated_Tpi[9 + 7] *= mult
        # self.calibrated_Tpi[9 + 8] *= mult
        # self.calibrated_Tpi[9 + 9] *= mult
        # self.calibrated_Tpi[9 + 10] *= mult

        # These are from the nonlinearity calibration we performed on 2021/05/21
        self.AOM_saturation_params = [1.4] * 32
        # index into this list by the AWG slot index
        self.AOM_saturation_params[11] = 0.786
        self.AOM_saturation_params[12] = 0.806
        self.AOM_saturation_params[13] = 0.801
        self.AOM_saturation_params[14] = 0.798
        self.AOM_saturation_params[15] = 0.841
        self.AOM_saturation_params[16] = 0.797
        self.AOM_saturation_params[17] = 0.999#0.864 #
        self.AOM_saturation_params[18] = 0.835
        self.AOM_saturation_params[19] = 0.843
        self.AOM_saturation_params[20] = 0.841
        self.AOM_saturation_params[21] = 0.845
        self.AOM_saturation_params[22] = 0.867
        self.AOM_saturation_params[23] = 0.846
        # NOTE: use the following line to 'turn off' non-linearity correction
        # self.AOM_saturation_params = [1e6] * 32
        # use the nonlinearity calibration for the middle channel from 6/18/22
        # for all the channels
        self.AOM_saturation_params = [0.670] * 32

    @host_only
    def experiment_initialize(self):
        """Create & calculate the AWG waveforms for the experiment."""
        self.rf_compiler.clear_gates()

        self.rf_compiler.set_params(
            f_carrier=self.f_carrier,
            f_ind=self.f_ind,
            N_ions=self.N_ions,
            t_delay=self.t_delay,
            Rabi_max=self.Rabi_max,
            PI_center_freq_1Q=self.PI_center_freq_1Q,
            PI_center_freq_2Q=self.PI_center_freq_2Q,
        )

        self.rf_compiler.set_monitor_params(
            monitor_ind=self.monitor_ind,
            detuning=self.monitor_detuning,
            amp=self.amp_monitor,
        )

        self.rf_compiler.set_Rabi_params(
            amp_ind=self.amp_ind_Rabi,
            amp_global=self.amp_global_Rabi,
            Stark_shift=self.Stark_shift_Rabi,
        )

        self.rf_compiler.set_SK1_params(
            amp_ind=self.amp_ind_SK1,
            amp_global=self.amp_global_SK1,
            Tpi_multiplier=self.Tpi_multiplier_SK1,
            Stark_shift=self.Stark_shift_SK1,
        )

        self.rf_compiler.set_SK1_AM_params(
            amp_ind=self.amp_ind_SK1_AM,
            amp_global=self.amp_global_SK1_AM,
            theta=self.theta_SK1_AM,
            envelope_type=int(self.envelope_type_SK1_AM),
            envelope_scale=self.envelope_scale_SK1_AM,
            rotation_pulse_length=self.rotation_pulse_length_SK1_AM,
            correction_pulse_1_length=self.correction_pulse_1_length_SK1_AM,
            correction_pulse_2_length=self.correction_pulse_2_length_SK1_AM,
            Stark_shift=self.Stark_shift_SK1_AM,
            phase_correction=self.phase_correction_SK1_AM,
        )

        self.rf_compiler.set_XX_params(
            phase_offset=self.XX_phase_offset
        )

        # NOTE: these paths are left as Q reference b/c this is sent to the RFControl PC
        if self.N_ions == 15:
            self.XX_sols_dir = (
                #"Q:/CompactTrappedIonModule/Data/gate_solutions/2019_12_10/15ions_fullset_interpolated_225us.h5"
                #"Q:/CompactTrappedIonModule/Data/gate_solutions/2020_01_28_pi8gates/15ions_fullset.h5"
                #"Q:/CompactTrappedIonModule/Data/gate_solutions/2022_6_26/2ions_interpolated_127us.h5"
                "Q:/CompactTrappedIonModule/Data/gate_solutions/2022_6_30/15ions_interpolated_127us.h5"
            )
        elif self.N_ions == 23:
            self.xx_sols_dir = (
                #"Q:/CompactTrappedIonModule/Data/gate_solutions/2022_10_20/23ions_interpolated_190us.h5"
                "Q:/CompactTrappedIonModule/Data/gate_solutions/2023_01_25/23ions_interpolated_230us.h5"
            )
        elif self.N_ions == 25:
            self.XX_sols_dir = ("Q:/CompactTrappedIonModule/Data/gate_solutions/2020_04_28/25ions_interpolated_500us.h5")
        else:
            self.XX_sols_dir = None

        if self.XX_sols_dir is not None:
            self.rf_compiler.set_XX_solution_from_file(self.XX_sols_dir)
            self.rf_compiler.set_XX_tweak("global",
                                          **{"motional_frequency_adjust": self.tweak_motional_frequency_adjust})
            self.rf_compiler.set_XX_tweak("global",
                                          **{"sideband_amplitude_imbalance": self.tweak_sideband_amplitude_imbalance})
            self.rf_compiler.set_XX_tweak("global",
                                          **{"stark_shift": self.tweak_stark_shift})

        if self.use_linearity_correction:
            self.rf_compiler.set_AOM_saturation_params(self.AOM_saturation_params)
        if self.use_calibrated_Tpis:
            self.rf_compiler.set_AOM_levels(calibrated_Tpi=self.calibrated_Tpi, used_shaped=self.use_shaped_calibration)
        else:
            self.rf_compiler.disable_calibration(level_active=False,linearity_active=self.use_linearity_correction)

    @kernel
    def init(self):
        """Initialize the switch network outputs."""
        # if the previous run failed
        self.setRunningSN(False)
        self.sn_enable.on()
        self.sn_reset.off()

    @kernel
    def idle(self):
        """Set the switch network & AWG to its idle state."""
        self.setRunningSN(False)
        # reset switch network
        self.sn_reset.on()

    @kernel
    def resetSN(self):
        """Reset switch network state."""
        self.sn_reset.pulse(5 * us)

    @kernel
    def advanceSN(self):
        """Move the switch network to the next output."""
        self.sn_advance.pulse(5 * us)

    @kernel
    def setRunningSN(self, running: TBool):
        """Set the switch network to active mode."""
        if running is True:
            self.sn_running.on()
        else:
            self.sn_running.off()

    @kernel
    def set_scan_state(self, index: TInt32):
        """Output the current point in the scan to the switch network/AWG."""
        loc_index = index

        # for i in range(self.awg_bits):
        #     self.awg_register[i].off()

        for i in range(self.awg_bits):
            on = loc_index & 1
            loc_index = loc_index >> 1
            if on:
                self.awg_register[i].on()
            else:
                self.awg_register[i].off()

        delay(100 * us)

    @host_only
    def upload_data_to_awg(self):
        """Upload the RF waveforms to the AWG."""
        self.rf_compiler.select_exp_to_program()
        completed = self.rf_compiler.program_AWG()
        return completed

    @host_only
    def convert_tpi_to_min(self,
                           tpi: float,
                           ind_amp: int,
                           global_amp: int,
                           slot: int,
                           ):
        """Converts the measured Tpi with given individual amplitude and global amplitude
           to the Tpi with individual amplitude = 1000, assuming that the linearity correction is done correctly,
           and with global amplitude = 1000, using the measurements of Rabi freq. vs. global amplitude
        """
        ### adjust level for the ind amp and use the calibrated global adjustment from 11/24/2020
        # calibrated on 7/9/22 by M.C.
        scale_global_param = np.sin(global_amp/494.5) / np.sin(1000./494.5)
        min_tpi = tpi*(ind_amp/1000)*scale_global_param
        return min_tpi

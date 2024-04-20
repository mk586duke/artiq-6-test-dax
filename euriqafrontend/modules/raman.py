import numpy as np
from artiq.experiment import BooleanValue
from artiq.experiment import NumberValue
from artiq.experiment import TBool
from artiq.experiment import TFloat
from artiq.experiment import TInt32
from artiq.experiment import TInt64
from artiq.experiment import TList
from artiq.language import delay
from artiq.language import kernel
from artiq.language import parallel
from artiq.language import sequential
from artiq.language.core import delay_mu
from artiq.language.environment import HasEnvironment
from artiq.language.units import A
from artiq.language.units import MHz, Hz
from artiq.language.units import ms
from artiq.language.units import s
from artiq.language.units import us

import euriqabackend.coredevice.dac8568 as dac8568
import euriqafrontend.settings.calibration_box as calibrations
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqabackend.coredevice.ad9912 import phase_to_mu
from euriqafrontend.settings import RF_CALIBRATION_PATH
from euriqafrontend.modules.artiq_dac import SinglePiezo, DualPiezo


class BaseRamanDDS(HasEnvironment):

    _CONST_PI = np.pi

    def build(self, **kwargs):

        self.dds_name = kwargs["dds_name"]
        self.gui_freq_name = kwargs["gui_freq_name"]
        self.gui_amp_name = kwargs["gui_amp_name"]
        self.gui_phase_name = kwargs["gui_phase_name"]
        self.gui_default_freq = kwargs["gui_default_freq"]
        self.gui_default_amp = kwargs["gui_default_amp"]

        self.freq_input = self.get_argument(
            self.gui_freq_name,
            NumberValue(
                default=self.gui_default_freq,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=7,
            ),
            group="Raman DDSs",
        )
        self.amp_input = self.get_argument(
            self.gui_amp_name,
            NumberValue(
                default=self.gui_default_amp, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Raman DDSs",
        )
        self.phase_input = self.get_argument(
            self.gui_phase_name,
            NumberValue(default=0, unit="Degrees", min=0, scale=1, max=360),
            group="Raman DDSs",
        )

        self.dds = self.get_device(self.dds_name)

    def prepare(self):
        self.enabled = False
        self.freq = freq_to_mu(self.freq_input)
        self.amp = np.int32(self.amp_input)
        self.phase = phase_to_mu(self.phase_input / 360 * 2 * self._CONST_PI)

    def set_param(self, freq=0, amp=0, phase=0):
        self.freq = freq_to_mu(freq)
        self.amp = np.int32(amp)
        self.phase = phase_to_mu(phase / 360 * 2 * self._CONST_PI)

    @kernel
    def update_freq(self, freq_hz: TFloat):
        self.freq = freq_to_mu(freq_hz)
        self.dds.set_mu(bus_group=1, frequency_mu=self.freq)

    @kernel
    def update_freq_mu(self, freq_mu: TInt64):
        self.freq = freq_mu
        self.dds.set_mu(bus_group=1, frequency_mu=self.freq)

    @kernel
    def update_amp(self, amp_int: TInt32):
        self.amp = np.int32(amp_int)
        self.dds.set_mu(bus_group=1, amplitude_mu=self.amp)

    @kernel
    def update_phase(self, phase_deg: TFloat):
        phase_rad = phase_deg / 360.0 * 2.0 * self._CONST_PI
        self.phase = phase_to_mu(phase_radians=phase_rad)
        self.dds.dds.phase_mu(phase=self.phase)

    @kernel
    def init(self):
        self.dds.init()
        self.write_to_dds()

    @kernel
    def write_to_dds(self):
        self.dds.set_mu(
            bus_group=1,
            frequency_mu=self.freq,
            amplitude_mu=self.amp,
            phase_mu=self.phase,
        )

    @kernel
    def on(self):
        self.dds.on()

    @kernel
    def off(self):
        self.dds.off()

    @kernel
    def pulse(self, time_s: TFloat):
        self.dds.pulse(time_s)

    @kernel
    def pulse_mu(self, time_mu: TInt64):
        self.dds.pulse_mu(time_mu)

    @kernel
    def idle(self):
        self.off()

    @kernel
    def load(self):
        self.dds.dds.load()

class BaseRamanDDS_noGUI(BaseRamanDDS):

    def build(self, **kwargs):
        self.dds_name = kwargs["dds_name"]
        self.freq_input = kwargs["freq_init"]
        self.amp_input = kwargs["amp_init"]
        self.phase_input = kwargs["phase_init"]

        self.dds = self.get_device(self.dds_name)

class Raman(HasEnvironment):

    _CONST_PI = np.pi

    def init_raman_param(self, calc_mu: bool = False):
        """Calculate Raman parameters based on constants."""
        rf_calib = calibrations.CalibrationBox.from_json(filename=RF_CALIBRATION_PATH, dataset_dict=self._HasEnvironment__dataset_mgr)

        self._GLOBAL_DELAY = rf_calib.delays.global_aom_to_individual_aom.value

        self._FEEDFORWARD_BASE = rf_calib.frequencies.feedforward_frequency.value

        self._IND_BASE = rf_calib.frequencies.individual_frequency.value

        # Rep Rate Setpoint
        self._REPRATE_SP = rf_calib.frequencies.repetition_rate_setpoint.value
        self._MW_FREQ = rf_calib.frequencies.microwave_frequency.value

        self._CARRIER_FREQ = rf_calib.frequencies.global_carrier_frequency.value

        # Calculate machine units, cannot be run in build()
        if calc_mu:
            self._GLOBAL_DELAY_MU = self.core.seconds_to_mu(self._GLOBAL_DELAY)
            self._FEEDFORWARD_BASE_MU = freq_to_mu(self._FEEDFORWARD_BASE)
            self._IND_BASE_MU = freq_to_mu(self._IND_BASE)
            self._CARRIER_FREQ_MU = freq_to_mu(self._CARRIER_FREQ)

    def build(self):
        # Get DDS Values from globals, don't archive in build
        archive = False
        global_amp_init = self.get_dataset("global.Raman.Raman_Global_Amp", archive=archive)
        ind_switchnet_amp_init = self.get_dataset("global.Raman.Raman_SN_Amp", archive=archive)
        ind_collective_amp_init = self.get_dataset("global.Raman.Raman_Coll_Amp", archive=archive)
        ind_final_x_init = self.get_dataset("global.Raman.Piezos.Ind_FinalX", archive=archive)
        ind_final_y_init = self.get_dataset("global.Raman.Piezos.Ind_FinalY", archive=archive)
        self.init_raman_param(calc_mu=False)
        self.setattr_argument(
            "use_global_globalAOM_values",
            BooleanValue(default=True),
            group="Raman DDSs",
        )
        self.setattr_argument(
            "use_global_SN_values", BooleanValue(default=True), group="Raman DDSs"
        )
        self.setattr_argument(
            "use_global_collective_values",
            BooleanValue(default=True),
            group="Raman DDSs",
        )
        self.setattr_argument(
            "use_global_piezo_values", BooleanValue(default=True), group="Raman DDSs"
        )

        self.global_dds = BaseRamanDDS(
            self,
            dds_name="global_raman_dds",
            gui_freq_name="global_freq",
            gui_amp_name="global_amp",
            gui_phase_name="global_phase",
            gui_default_freq=self._CARRIER_FREQ,
            gui_default_amp=global_amp_init,
        )

        self.switchnet_dds = BaseRamanDDS(
            self,
            dds_name="switchnet_dds",
            gui_freq_name="ind_switchnet_freq",
            gui_amp_name="ind_switchnet_amp",
            gui_phase_name="ind_switchnet_phase",
            gui_default_freq=self._IND_BASE,
            gui_default_amp=ind_switchnet_amp_init,
        )

        # self.ind_final_x_piezo = SinglePiezo(self,
        #                                  gui_name="ind_final_x_piezo",
        #                                  gui_default_value=ind_final_x_init,
        #                                  device_name="dac8568_1",
        #                                  device_channel=dac8568.AOut.Out5,
        #                                  gui_group="Raman DDSs"
        #                                  )
        # self.ind_final_y_piezo = SinglePiezo(self,
        #                                      gui_name="ind_final_y_piezo",
        #                                      gui_default_value=ind_final_y_init,
        #                                      device_name="dac8568_1",
        #                                      device_channel=dac8568.AOut.Out2,
        #                                      gui_group="Raman DDSs"
        #                                      )
        self.ind_final_piezos = DualPiezo(
            self,
            gui_name1="ind_final_x_piezo",
            gui_name2="ind_final_y_piezo",
            gui_default_value1=ind_final_x_init,
            gui_default_value2=ind_final_y_init,
            device_name="dac8568_1",
            device_channel1=dac8568.AOut.Out2,
            device_channel2=dac8568.AOut.Out1,
            gui_group="Raman DDSs",
        )
        self.setattr_device("core")
        self.setattr_device("global_aom_source")  # on for AWG, off for DDS
        self.setattr_device("attenuator_c8")
        self.setattr_device("attenuator_c16")

    def prepare(self):
        self.init_raman_param(calc_mu=True)

        self.global_dds.prepare()
        if self.use_global_globalAOM_values is True:
            self.global_dds.set_param(
                freq=self._CARRIER_FREQ,
                amp=self.get_dataset("global.Raman.Raman_Global_Amp"),
                phase=0,
            )

        self.switchnet_dds.prepare()
        if self.use_global_SN_values is True:
            self.switchnet_dds.set_param(
                freq=self._IND_BASE,
                amp=self.get_dataset("global.Raman.Raman_SN_Amp"),
                phase=0,
            )

        # self.ind_final_x_piezo.prepare()
        # self.ind_final_y_piezo.prepare()
        self.ind_final_piezos.feedforward_1_to_2 = self.get_dataset(
            "global.Raman.Piezos.XtoY_feedforward"
        )
        self.ind_final_piezos.feedforward_2_to_1 = self.get_dataset(
            "global.Raman.Piezos.YtoX_feedforward"
        )
        self.ind_final_piezos.prepare()

        if self.use_global_piezo_values is True:
            # self.ind_final_x_piezo.set_value(self.get_dataset("global.Raman.Piezos.Ind_FinalX"))
            # self.ind_final_y_piezo.set_value(self.get_dataset("global.Raman.Piezos.Ind_FinalY"))
            self.ind_final_piezos.set_value(
                self.get_dataset("global.Raman.Piezos.Ind_FinalX"),
                self.get_dataset("global.Raman.Piezos.Ind_FinalY"),
            )

        self.global_source = str()

    @kernel
    def set_global_aom_source(self, dds: TBool):
        if dds is True:
            self.global_aom_source.on()  # off for DDS
            self.global_dds.enabled = True
            self.global_source = "dds"
        elif dds is False:
            self.global_aom_source.off()  # on for AWG
            self.global_dds.enabled = False
            self.global_source = "awg"

    @kernel
    def set_ind_attenuation(self, c8: TBool=False, c16: TBool=False):
        """
        Args:
            c8 (bool): True -> applies 8dB of attenuation
            c16 (bool): True -> applies 16dB of attenuation
        """
        self.attenuator_c8.on() if c8 else self.attenuator_c8.off()
        self.attenuator_c16.on() if c16 else self.attenuator_c16.off()

    @kernel
    def set_ind_detuning(self, detuning_freq_hz: TFloat):
        freq_hz = self._IND_BASE - detuning_freq_hz
        self.switchnet_dds.update_freq(freq_hz=freq_hz)

    @kernel
    def set_ind_detuning_mu(self, detuning_freq_mu: TInt64):
        freq_mu = self._IND_BASE_MU - detuning_freq_mu
        self.switchnet_dds.update_freq_mu(freq_mu=freq_mu)

    @kernel
    def set_global_detuning(self, detuning_freq_hz: TFloat):
        freq_hz = self._CARRIER_FREQ + detuning_freq_hz
        self.global_dds.update_freq(freq_hz=freq_hz)

    @kernel
    def set_global_detuning_mu(self, detuning_freq_mu: TInt64):
        freq_mu = self._CARRIER_FREQ_MU + detuning_freq_mu
        self.global_dds.update_freq_mu(freq_mu=freq_mu)

    @kernel
    def init(self):
        # self.ind_final_x_piezo.init()
        # self.ind_final_y_piezo.init()
        # self.ind_final_x_piezo.update_value()
        # self.ind_final_y_piezo.update_value()
        self.ind_final_piezos.init()
        self.ind_final_piezos.update_value()

        self.switchnet_dds.enabled = True
        # MC: 7/7/19
        # parallel programming causing the SN freq to be set incorrectly or to not be set
        # as evidenced by Raman Ramsey on one ion
        # fixed by changing to sequential
        # with parallel:
        with sequential:
            self.global_dds.init()
            self.set_global_aom_source(dds=True)

            with sequential:  # Individual DDSs are on the same chip and must be written sequentially.
                self.switchnet_dds.init()

    @kernel
    def on(self):
        assert (
            self.global_source == "dds"
        ), "Point global at the DDS before turning on. "
        with parallel:
            if self.switchnet_dds.enabled is True:
                self.switchnet_dds.on()
            with sequential:
                delay_mu(-self._GLOBAL_DELAY_MU)
                self.global_dds.on()
                delay_mu(self._GLOBAL_DELAY_MU)

    @kernel
    def off(self):
        with parallel:
            self.switchnet_dds.off()
            with sequential:
                delay_mu(-self._GLOBAL_DELAY_MU)
                self.global_dds.off()
                delay_mu(self._GLOBAL_DELAY_MU)

    @kernel
    def pulse(self, time_s: TFloat):
        assert (
            self.global_source == "dds"
        ), "Point global at the DDS before turning on. "
        with parallel:
            if self.switchnet_dds.enabled is True:
                self.switchnet_dds.pulse(time_s=time_s)
            with sequential:
                delay_mu(-self._GLOBAL_DELAY_MU)
                self.global_dds.pulse(time_s=time_s)
                delay_mu(self._GLOBAL_DELAY_MU)

    @kernel
    def pulse_mu(self, time_mu: TInt64):
        assert (
            self.global_source == "dds"
        ), "Point global at the DDS before turning on. "
        with parallel:
            if self.switchnet_dds.enabled is True:
                self.switchnet_dds.pulse_mu(time_mu=time_mu)
            with sequential:
                delay_mu(-self._GLOBAL_DELAY_MU)
                self.global_dds.pulse_mu(time_mu=time_mu)
                delay_mu(self._GLOBAL_DELAY_MU)

    @kernel
    def idle(self):
        self.global_dds.idle()
        self.switchnet_dds.idle()
        self.set_ind_attenuation(c8=False, c16=False)

class SBC_DDS(HasEnvironment):

    def build(self):
        # Needs the Raman module for frequency calculations
        self.raman = Raman(self)
        self.raman.init_raman_param(calc_mu=False)
        self._CARRIER_FREQ = self.raman._CARRIER_FREQ
        self._FEEDFORWARD_BASE = self.raman._FEEDFORWARD_BASE
        self._IND_BASE = self.raman._IND_BASE

        self.setattr_device("core")

        self._get_sbc_vars_from_globals(archive=False)

        self.setattr_argument(
            "use_globals", BooleanValue(default=True), group="Sideband Cooling"
        )

        self.setattr_argument(
            "Parallel_SBC", BooleanValue(default=self.parallel_sbc), group="Sideband Cooling"
        )

        self.setattr_argument(
            "Global_Amp",
            NumberValue(default=self.global_amp, step=10, ndecimals=0),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "SN_Amp",
            NumberValue(default=self.sn_amp, step=10, ndecimals=0),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Mode1_Active_Control",
            BooleanValue(default=self.modes_active[0]),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Mode2_Active_Control",
            BooleanValue(default=self.modes_active[1]),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Mode3_Active_Control",
            BooleanValue(default=self.modes_active[2]),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Mode4_Active_Control",
            BooleanValue(default=self.modes_active[3]),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Ind_Always_On_Control",
            BooleanValue(default=self.always_on),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Sweep_On_Control",
            BooleanValue(default=self.sweep_on),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Loops1_Max_Control",
            NumberValue(default=self.loops_max[0], step=1, ndecimals=0),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Loops2_Max_Control",
            NumberValue(default=self.loops_max[1], step=1, ndecimals=0),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Loops3_Max_Control",
            NumberValue(default=self.loops_max[2], step=1, ndecimals=0),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Loops4_Max_Control",
            NumberValue(default=self.loops_max[3], step=1, ndecimals=0),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Sweep_Max_Control",
            NumberValue(default=self.sweep_max, step=1, ndecimals=0),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Mode1_Detuning_Control",
            NumberValue(
                default=self.mode_detunings[0],
                unit="MHz",
                min=-7 * MHz,
                max=7 * MHz,
                ndecimals=5,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode2_Detuning_Control",
            NumberValue(
                default=self.mode_detunings[1],
                unit="MHz",
                min=-7 * MHz,
                max=7 * MHz,
                ndecimals=4,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode3_Detuning_Control",
            NumberValue(
                default=self.mode_detunings[2],
                unit="MHz",
                min=-7 * MHz,
                max=7 * MHz,
                ndecimals=4,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode4_Detuning_Control",
            NumberValue(
                default=self.mode_detunings[3],
                unit="MHz",
                min=-7 * MHz,
                max=7 * MHz,
                ndecimals=4,
            ),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Mode1_t0_Control",
            NumberValue(
                default=self.mode_t0s[0],
                unit="us",
                min=0 * us,
                max=100 * us,
                ndecimals=2,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode2_t0_Control",
            NumberValue(
                default=self.mode_t0s[1],
                unit="us",
                min=0 * us,
                max=100 * us,
                ndecimals=2,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode3_t0_Control",
            NumberValue(
                default=self.mode_t0s[2],
                unit="us",
                min=0 * us,
                max=100 * us,
                ndecimals=2,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode4_t0_Control",
            NumberValue(
                default=self.mode_t0s[3],
                unit="us",
                min=0 * us,
                max=100 * us,
                ndecimals=2,
            ),
            group="Sideband Cooling",
        )

        self.setattr_argument(
            "Mode1_dt_Control",
            NumberValue(
                default=self.mode_dts[0],
                unit="us",
                min=0 * us,
                max=100 * us,
                ndecimals=2,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode2_dt_Control",
            NumberValue(
                default=self.mode_dts[1],
                unit="us",
                min=0 * us,
                max=100 * us,
                ndecimals=2,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode3_dt_Control",
            NumberValue(
                default=self.mode_dts[2],
                unit="us",
                min=0 * us,
                max=100 * us,
                ndecimals=2,
            ),
            group="Sideband Cooling",
        )
        self.setattr_argument(
            "Mode4_dt_Control",
            NumberValue(
                default=self.mode_dts[3],
                unit="us",
                min=0 * us,
                max=100 * us,
                ndecimals=2,
            ),
            group="Sideband Cooling",
        )

    def prepare(self):
        if self.use_globals:
            self._get_sbc_vars_from_globals(archive=True)
        else:
            self._get_sbc_vars_from_controls()

        # Make sure the raman frequency parameters are up to date
        self.raman.init_raman_param(calc_mu=True)
        self._CARRIER_FREQ_MU = self.raman._CARRIER_FREQ_MU
        self._FEEDFORWARD_BASE_MU = self.raman._FEEDFORWARD_BASE_MU
        self._IND_BASE_MU = self.raman._IND_BASE_MU

        # initialize various other machine-unit time variables
        # this has to be done here since build does not have the core device available
        self.mode_t0s_mu = [self.core.seconds_to_mu(t) for t in self.mode_t0s]
        self.mode_dts_mu = [self.core.seconds_to_mu(t) for t in self.mode_dts]

        self.start_riffle_wait_mu = self.core.seconds_to_mu(self.start_riffle_wait)
        self.end_riffle_wait_mu = self.core.seconds_to_mu(self.end_riffle_wait)

    def _get_sbc_vars_from_globals(self, archive: bool = True):

        self.global_amp = np.int32(self.get_dataset("global.SBC.Global_Amp", archive=archive))
        self.sn_amp = np.int32(self.get_dataset("global.SBC.SN_Amp", archive=archive))

        self.start_riffles = np.int32(self.get_dataset("global.SBC.riffle.start_riffles", archive=archive))
        self.end_riffles = np.int32(self.get_dataset("global.SBC.riffle.end_riffles", archive=archive))
        self.start_riffle_wait = self.get_dataset("global.SBC.riffle.start_riffle_wait", archive=archive)
        self.end_riffle_wait = self.get_dataset("global.SBC.riffle.end_riffle_wait", archive=archive)

        self.modes_active = [
            np.int32(self.get_dataset("global.SBC.Mode1_Active", archive=archive)),
            np.int32(self.get_dataset("global.SBC.Mode2_Active", archive=archive)),
            np.int32(self.get_dataset("global.SBC.Mode3_Active", archive=archive)),
            np.int32(self.get_dataset("global.SBC.Mode4_Active", archive=archive)),
        ]

        self.always_on = np.int32(self.get_dataset("global.SBC.Ind_Always_On", archive=archive))
        self.sweep_on = np.int32(self.get_dataset("global.SBC.Sweep_On", archive=archive) != 0)

        self.mode_detunings = [
            (self.get_dataset("global.SBC.Mode1_Detuning", archive=archive)),
            (self.get_dataset("global.SBC.Mode2_Detuning", archive=archive)),
            (self.get_dataset("global.SBC.Mode3_Detuning", archive=archive)),
            (self.get_dataset("global.SBC.Mode4_Detuning", archive=archive)),
        ]

        self.mode_detunings_mu = [freq_to_mu(imode) for imode in self.mode_detunings]

        self.detuned_mode_mu = freq_to_mu(self.get_dataset("global.SBC.Detuned_Mode", archive=archive))

        self.parallel_sbc = bool(self.get_dataset("global.SBC.Parallel_SBC_On", archive=archive))

        self.mode_t0s = [
            self.get_dataset("global.SBC.Mode1_t0", archive=archive),
            self.get_dataset("global.SBC.Mode2_t0", archive=archive),
            self.get_dataset("global.SBC.Mode3_t0", archive=archive),
            self.get_dataset("global.SBC.Mode4_t0", archive=archive),
        ]

        self.mode_dts = [
            self.get_dataset("global.SBC.Mode1_dt", archive=archive),
            self.get_dataset("global.SBC.Mode2_dt", archive=archive),
            self.get_dataset("global.SBC.Mode3_dt", archive=archive),
            self.get_dataset("global.SBC.Mode4_dt", archive=archive),
        ]

        self.loops_max = [
            np.int32(self.get_dataset("global.SBC.Loops1", archive=archive)),
            np.int32(self.get_dataset("global.SBC.Loops2", archive=archive)),
            np.int32(self.get_dataset("global.SBC.Loops3", archive=archive)),
            np.int32(self.get_dataset("global.SBC.Loops4", archive=archive)),
        ]

        self.sweep_max = np.int32(self.get_dataset("global.SBC.Sweep_Max", archive=archive))

    def _get_sbc_vars_from_controls(self):

        self.global_amp = np.int32(self.Global_Amp)
        self.sn_amp = np.int32(self.SN_Amp)

        self.modes_active = [
            np.int32(self.Mode1_Active_Control),
            np.int32(self.Mode2_Active_Control),
            np.int32(self.Mode3_Active_Control),
            np.int32(self.Mode4_Active_Control),
        ]

        self.always_on = np.int32(self.Ind_Always_On_Control)
        self.sweep_on = np.int32(self.Sweep_On_Control)
        self.parallel_sbc = bool(self.Parallel_SBC)


        self.mode_detunings_ = [
            self.Mode1_Detuning_Control,
            self.Mode2_Detuning_Control,
            self.Mode3_Detuning_Control,
            self.Mode4_Detuning_Control,
        ]

        self.mode_detunings_mu = [
            freq_to_mu(self.Mode1_Detuning_Control),
            freq_to_mu(self.Mode2_Detuning_Control),
            freq_to_mu(self.Mode3_Detuning_Control),
            freq_to_mu(self.Mode4_Detuning_Control),
        ]

        self.mode_t0s = [
            self.Mode1_t0_Control,
            self.Mode2_t0_Control,
            self.Mode3_t0_Control,
            self.Mode4_t0_Control,
        ]

        self.mode_dts = [
            self.Mode1_dt_Control,
            self.Mode2_dt_Control,
            self.Mode3_dt_Control,
            self.Mode4_dt_Control,
        ]

        self.loops_max = [
            self.Loops1_Max_Control,
            self.Loops2_Max_Control,
            self.Loops3_Max_Control,
            self.Loops4_Max_Control,
        ]

        self.sweep_max = np.int32(self.Sweep_Max_Control)

    @kernel
    def init(self):
        pass

    @kernel
    def set_sbc1_detuning(self, detuning_freq_hz: TFloat):
        freq_hz = (self._IND_BASE + self._FEEDFORWARD_BASE) - detuning_freq_hz

    @kernel
    def set_scb1_detuning_mu(self, detuning_freq_mu: TInt64):
        freq_mu = (self._IND_BASE_MU + self._FEEDFORWARD_BASE_MU) - detuning_freq_mu

    @kernel
    def set_sbc2_detuning(self, detuning_freq_hz: TFloat):
        freq_hz = (self._IND_BASE + self._FEEDFORWARD_BASE) - detuning_freq_hz

    @kernel
    def set_scb2_detuning_mu(self, detuning_freq_mu: TInt64):
        freq_mu = (self._IND_BASE_MU + self._FEEDFORWARD_BASE_MU) - detuning_freq_mu

    @kernel
    def update_amp(self, amp_int: TInt32):
        pass

    @kernel
    def dds_on(self):
        pass

    @kernel
    def dds_off(self):
        pass

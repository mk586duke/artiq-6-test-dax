import numpy as np
import logging
from artiq.experiment import BooleanValue
from artiq.experiment import NumberValue
from artiq.experiment import TBool
from artiq.experiment import TFloat
from artiq.experiment import TInt32
from artiq.experiment import TInt64
from artiq.experiment import TList
from artiq.language import at_mu
from artiq.language import delay
from artiq.language import kernel
from artiq.language import now_mu
from artiq.language import parallel
from artiq.language import portable
from artiq.language.core import sequential
from artiq.language.environment import HasEnvironment
from artiq.language.units import A
from artiq.language.units import MHz
from artiq.language.units import ms
from artiq.language.units import s
from artiq.language.units import us

from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqabackend.coredevice.ad9912 import phase_to_mu

_LOGGER = logging.getLogger(__name__)

class BaseDoubleDDS(HasEnvironment):
    def build(self, **kwargs):

        self.dds_name_1 = kwargs["dds_name1"]
        self.gui_freq_name_1 = kwargs["gui_freq_name1"]
        self.gui_amp_name_1 = kwargs["gui_amp_name1"]
        self.gui_phase_name_1 = kwargs["gui_phase_name1"]
        self.gui_default_freq_1 = kwargs["gui_default_freq1"]
        self.gui_default_amp_1 = kwargs["gui_default_amp1"]

        self.dds_name_2 = kwargs["dds_name2"]
        self.gui_freq_name_2 = kwargs["gui_freq_name2"]
        self.gui_amp_name_2 = kwargs["gui_amp_name2"]
        self.gui_phase_name_2 = kwargs["gui_phase_name2"]
        self.gui_default_freq_2 = kwargs["gui_default_freq2"]
        self.gui_default_amp_2 = kwargs["gui_default_amp2"]

        self.gui_group_name = kwargs["gui_group_name"]

        self.freq_input_1 = self.get_argument(
            self.gui_freq_name_1,
            NumberValue(
                default=self.gui_default_freq_1,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=7,
            ),
            group=self.gui_group_name,
        )
        self.amp_input_1 = self.get_argument(
            self.gui_amp_name_1,
            NumberValue(
                default=self.gui_default_amp_1, min=0, scale=1, max=1000, ndecimals=0
            ),
            group=self.gui_group_name,
        )
        self.phase_input_1 = self.get_argument(
            self.gui_phase_name_1,
            NumberValue(default=0, unit="Degrees", min=0, scale=1, max=360),
            group=self.gui_group_name
        )

        self.freq_input_2 = self.get_argument(
            self.gui_freq_name_2,
            NumberValue(
                default=self.gui_default_freq_2,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=7,
            ),
            group=self.gui_group_name,
        )
        self.amp_input_2 = self.get_argument(
            self.gui_amp_name_2,
            NumberValue(
                default=self.gui_default_amp_2, min=0, scale=1, max=1000, ndecimals=0
            ),
            group=self.gui_group_name,
        )
        self.phase_input_2 = self.get_argument(
            self.gui_phase_name_2,
            NumberValue(default=0, unit="Degrees", min=0, scale=1, max=360),
            group=self.gui_group_name
        )

        self.dds1 = self.get_device(self.dds_name_1)
        self.dds2 = self.get_device(self.dds_name_2)

    def prepare(self):
        self.dds = [self.dds1, self.dds2]
        self.freq_mu = [freq_to_mu(i) for i in [self.freq_input_1, self.freq_input_2]]
        self.amp = [np.int32(i) for i in [self.amp_input_1, self.amp_input_2]]
        self.phase_mu = [phase_to_mu(i) for i in [self.phase_input_1, self.phase_input_2]]

    @portable
    def set_param(self, freq_hz: TList(TFloat), amp_int: TList(TInt32), phases: TList(TFloat) = [0,0]):
        assert len(freq_hz) == 2, "Please specify frequencies for both DDSs"
        assert len(amp_int) == 2, "Please specify amplitudes for both DDSs"
        self.freq_mu = [freq_to_mu(i) for i in freq_hz]
        self.amp = amp_int
        self.phase_mu = [phase_to_mu(i) for i in phases]

    def set_param1(self, freq_hz: TFloat, amp_int: TInt32, phase: TFloat = 0):
        self.freq_mu[0] = freq_to_mu(freq_hz)
        self.amp[0] = amp_int
        self.phase_mu[0] = phase_to_mu(phase)

    def set_param2(self, freq_hz: TFloat, amp_int: TInt32, phase: TFloat = 0):
        self.freq_mu[1] = freq_to_mu(freq_hz)
        self.amp[1] = amp_int
        self.phase_mu[1] = phase_to_mu(phase)

    @portable
    def set_param_mu(self, freq_mu: TList(TInt64), amp_int: TList(TInt32), phases_mu: TList(TInt32) = [0,0]):
        assert len(freq_mu) == 2, "Please specify frequencies for both DDSs"
        assert len(amp_int) == 2, "Please specify amplitudes for both DDSs"
        assert len(phases_mu) == 2, "Please specify amplitudes for both DDSs"
        self.freq_mu = freq_mu
        self.amp = amp_int
        self.phase_mu = phases_mu

    @portable
    def set_param_mu1(self, freq_mu: TInt64, amp_int: TInt32, phase_mu: TInt32 = 0):
        self.freq_mu[0] = freq_mu
        self.amp[0] = amp_int
        self.phase_mu[0] = phase_mu

    @portable
    def set_param_mu2(self, freq_mu: TInt64, amp_int: TInt32, phase_mu: TInt32 = 0):
        self.freq_mu[1] = freq_mu
        self.amp[1] = amp_int
        self.phase_mu[1] = phase_mu

    @kernel
    def init(self):
        # manual parallel init. Equiv to "with parallel" block
        parallel_start_mu = now_mu()
        for idds in range(len(self.dds)):
            at_mu(parallel_start_mu)
            self.dds[idds].init()
        self.write_to_dds()

    @kernel
    def init1(self):
        self.dds[0].init()
        self.write_to_dds1()

    @kernel
    def init2(self):
        self.dds[1].init()
        self.write_to_dds2()

    @kernel
    def write_to_dds(self):
        with sequential:
            for idds in range(len(self.dds)):
                ret = self.dds[idds].set_mu(
                    bus_group=1,
                    frequency_mu=self.freq_mu[idds],
                    amplitude_mu=self.amp[idds],
                )
                # assert ret is True
    @kernel
    def write_to_dds1(self):
        ret = self.dds[0].set_mu(
            bus_group=1,
            frequency_mu=self.freq_mu[0],
            amplitude_mu=self.amp[0],
        )
        # assert ret is True

    @kernel
    def write_to_dds2(self):
        ret = self.dds[1].set_mu(
            bus_group=1,
            frequency_mu=self.freq_mu[1],
            amplitude_mu=self.amp[1],
        )
        # assert ret is True

    @kernel
    def update_freq(self, freq_hz: TList(TFloat)):
        assert len(freq_hz) == 2, "Please specify frequencies for both DDSs"
        self.freq_mu = [freq_to_mu(i) for i in freq_hz]
        with sequential:
            for idds in range(len(self.dds)):
                ret = self.dds[idds].set_mu(
                    bus_group=1, frequency_mu=self.freq_mu[idds]
                )
                assert ret is True

    @kernel
    def update_freq1(self, freq_hz: TList(TFloat)):
        self.freq_mu = freq_to_mu(freq_hz)
        ret = self.dds[0].set_mu(
            bus_group=1, frequency_mu=self.freq_mu[0]
        )
        assert ret is True

    @kernel
    def update_freq2(self, freq_hz: TList(TFloat)):
        self.freq_mu = freq_to_mu(freq_hz)
        ret = self.dds[1].set_mu(
            bus_group=1, frequency_mu=self.freq_mu[1]
        )
        assert ret is True

    @kernel
    def update_freq_mu(self, freq_mu: TList(TInt64)):
        assert len(freq_mu) == 2, "Please specify frequencies for both DDSs"
        self.freq_mu = freq_mu
        with sequential:
            for idds in range(len(self.dds)):
                ret = self.dds[idds].set_mu(
                    bus_group=1, frequency_mu=self.freq_mu[idds]
                )
                assert ret is True

    @kernel
    def update_freq1_mu(self, freq_mu: TInt64):
        self.freq_mu[0] = freq_mu
        ret = self.dds[0].set_mu(
            bus_group=1, frequency_mu=self.freq_mu[0]
        )

    @kernel
    def update_freq2_mu(self, freq_mu: TInt64):
        self.freq_mu[1] = freq_mu
        ret = self.dds[1].set_mu(
            bus_group=1, frequency_mu=self.freq_mu[1]
        )

    @kernel
    def update_center_freq(self, freq_hz : TFloat):
        self.update_center_freq_mu(freq_to_mu(freq_hz))

    @kernel
    def update_center_freq_mu(self, freq_mu : TInt64):
        cur_center_freq = (self.freq_mu[0] + self.freq_mu[1]) // 2
        freq_offset = freq_mu - cur_center_freq
        self.freq_mu[0] += freq_offset
        self.freq_mu[1] += freq_offset
        with sequential:
            for idds in range(len(self.dds)):
                ret = self.dds[idds].set_mu(
                    bus_group=1, frequency_mu=self.freq_mu[idds]
                )
                assert ret is True

    @kernel
    def update_diff_freq(self, diff_freq_hz : TFloat):
        self.update_diff_freq_mu(freq_to_mu(diff_freq_hz))

    @kernel
    def update_diff_freq_mu(self, diff_freq_mu: TInt64):
        cur_diff_freq_mu = (self.freq_mu[1] - self.freq_mu[1])
        dshift_mu = (diff_freq_mu - cur_diff_freq_mu) // 2
        self.freq_mu[0] -= dshift_mu
        self.freq_mu[1] += dshift_mu
        with sequential:
            for idds in range(len(self.dds)):
                ret = self.dds[idds].set_mu(
                    bus_group=1, frequency_mu=self.freq_mu[idds]
                )
                assert ret is True

    @kernel
    def update_amp(self, amp_int: TList(TInt32)):
        assert len(amp_int) == 2, "Please specify amplitudes for both DDSs"
        self.amp = amp_int
        with sequential:
            for idds in range(len(self.dds)):
                ret = self.dds[idds].set_mu(bus_group=1, amplitude_mu=self.amp[idds])
                assert ret is True

    @kernel
    def update_amp1(self, amp_int: TInt32):
        self.amp[0] = amp_int
        ret = self.dds[0].set_mu(bus_group=1, amplitude_mu=self.amp[0])
        assert ret is True

    @kernel
    def update_amp2(self, amp_int: TInt32):
        self.amp[1] = amp_int
        ret = self.dds[1].set_mu(bus_group=1, amplitude_mu=self.amp[1])
        assert ret is True

    @kernel
    def update_phase(self, phases: TList(TFloat)):
        assert len(phases) == 2, "Please specify amplitudes for both DDSs"
        self.phase_mu = [phase_to_mu(phase_radials = ph / 360.0 * 2.0 * self._CONST_PI) for ph in phases]
        with sequential:
            for idds in range(len(self.dds)):
                ret = self.dds[idds].phase_mu(phase=self.phase_mu[idds])
                assert ret is True

    @kernel
    def update_phase1(self, phase_deg: TFloat):
        phase_rad = phase_deg / 360.0 * 2.0 * self._CONST_PI
        self.phase_mu[0] = phase_to_mu(phase_radians=phase_rad)
        self.dds[0].phase_mu(phase=self.phase_mu[0])

    @kernel
    def update_phase2(self, phase_deg: TFloat):
        phase_rad = phase_deg / 360.0 * 2.0 * self._CONST_PI
        self.phase_mu[1] = phase_to_mu(phase_radians=phase_rad)
        self.dds[1].phase_mu(phase=self.phase_mu[1])

    @kernel
    def on(self):
        # manual parallel. Equiv to "with parallel" block
        parallel_start_mu = now_mu()
        for idds in range(len(self.dds)):
            at_mu(parallel_start_mu)
            self.dds[idds].on()

    @kernel
    def on1(self):
        self.dds[0].on()

    @kernel
    def on2(self):
        self.dds[0].on()

    @kernel
    def off(self):
        # manual parallel. Equiv to "with parallel" block
        parallel_start_mu = now_mu()
        for idds in range(len(self.dds)):
            at_mu(parallel_start_mu)
            self.dds[idds].off()

    @kernel
    def off1(self):
        self.dds[0].off()

    @kernel
    def off2(self):
        self.dds[1].off()

    @kernel
    def pulse(self, time_s: TFloat):
        # manual parallel. Equiv to "with parallel" block
        parallel_start_mu = now_mu()
        for idds in range(len(self.dds)):
            at_mu(parallel_start_mu)
            self.dds[idds].pulse(time_s)

    @kernel
    def pulse1(self, time_s: TFloat):
        self.dds[0].pulse(time_s)

    @kernel
    def pulse2(self, time_s: TFloat):
        self.dds[1].pulse(time_s)

    @kernel
    def pulse_mu(self, time_mu: TInt64):
        # manual parallel. Equiv to "with parallel" block
        parallel_start_mu = now_mu()
        for idds in range(len(self.dds)):
            at_mu(parallel_start_mu)
            self.dds[idds].pulse_mu(time_mu)

    @kernel
    def pulse1_mu(self, time_mu: TInt64):
        self.dds[0].pulse_mu(time_mu)

    @kernel
    def pulse2_mu(self, time_mu: TInt64):
        self.dds[1].pulse_mu(time_mu)

    @kernel
    def load(self):
        # manual parallel. Equiv to "with parallel" block
        parallel_start_mu = now_mu()
        for idds in range(len(self.dds)):
            at_mu(parallel_start_mu)
            self.dds[idds].dds.load()

    @kernel
    def load1(self):
        self.dds[0].dds.load()

    @kernel
    def load2(self):
        self.dds[1].dds.load()

    @kernel
    def idle(self):
        self.off()


class DopplerCooling(BaseDoubleDDS, HasEnvironment):
    def load_globals(self, archive: bool=True):

        self._resonance_coolAOM1_freq = self.get_dataset("global.CW_Lasers.Yb.Resonance_CoolAOM1_Freq", archive=archive)
        self._resonance_coolAOM2_freq = self.get_dataset("global.CW_Lasers.Yb.Resonance_CoolAOM2_Freq", archive=archive)
        cool_detuning =self.get_dataset("global.Doppler_Cool.Yb.Detuning", archive=archive)

        self._coolaom1_freq_global = self._resonance_coolAOM1_freq + 1 / 2 * cool_detuning
        self._coolaom2_freq_global = self._resonance_coolAOM2_freq + 1 / 2 * cool_detuning

        self._coolaom1_amp_global = self.get_dataset("global.Doppler_Cool.Yb.CoolAOM1_Amp", archive=archive)
        self._coolaom2_amp_global = self.get_dataset("global.Doppler_Cool.Yb.CoolAOM2_Amp", archive=archive)

        self._doppler_cooling_time_global = self.get_dataset("global.Doppler_Cool.Duration", archive=archive)
        self._doppler_cooling_monitor_time_global = self.get_dataset("global.Doppler_Cool.Monitor_Duration",
                                                                     archive=archive
                                                                     )
        self._loaded_globals = True

    def build(self):
        archive = False
        self.load_globals(archive=archive)

        self.setattr_argument(
            "_cool_1_freq",
            NumberValue(
                default=self._coolaom1_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Doppler Cooling",
        )
        self.setattr_argument(
            "_cool_2_freq",
            NumberValue(
                default=self._coolaom2_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Doppler Cooling",
        )
        self.setattr_argument(
            "_cool_1_amp",
            NumberValue(
                default=self._coolaom1_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Doppler Cooling",
        )
        self.setattr_argument(
            "_cool_2_amp",
            NumberValue(
                default=self._coolaom2_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Doppler Cooling",
        )
        self.setattr_argument(
            "_doppler_cooling_time",
            NumberValue(default=self._doppler_cooling_time_global, unit="ms"),
            group="Doppler Cooling",
        )
        self.setattr_argument(
            "_doppler_cooling_monitor_time",
            NumberValue(default=self._doppler_cooling_monitor_time_global, unit="ms"),
            group="Doppler Cooling",
        )

        self.setattr_argument(
            "_use_global_values", BooleanValue(default=True), group="Doppler Cooling"
        )

        self.setattr_device("core")

        self.setattr_device("cooling_dds1")
        self.setattr_device("cooling_dds2")

        self.setattr_device("power_cool_1")
        self.setattr_device("power_cool_2")

    def set_detuning(self, detuning):
        if not self._loaded_globals:
            _LOGGER.error("Setting Doppler cooling detuning without having loaded the resonance setting datasets!")

        self._coolaom1_freq_global = self._resonance_coolAOM1_freq + 1 / 2 * detuning
        self._coolaom2_freq_global = self._resonance_coolAOM2_freq + 1 / 2 * detuning

        self.freq_mu = [
            freq_to_mu(i)
            for i in [self._coolaom1_freq_global, self._coolaom2_freq_global]
        ]

    def prepare(self):
        self._loaded_globals = False
        self.dds = [self.cooling_dds1, self.cooling_dds2]
        self.amp = [np.int32(i) for i in [0, 0]]

        if self._use_global_values:
            # The global variables may have changed since the build()
            # Therefore, reload them into local variables
            self.load_globals(archive=True)

            self.freq_mu = [
                freq_to_mu(i)
                for i in [self._coolaom1_freq_global, self._coolaom2_freq_global]
            ]
            self.cool_amp = [
                np.int32(i)
                for i in [self._coolaom1_amp_global, self._coolaom2_amp_global]
            ]
            self.doppler_cooling_duration = self._doppler_cooling_time_global
            self.doppler_cooling_duration_mu = self.core.seconds_to_mu(
                self._doppler_cooling_time_global
            )
            self.doppler_cooling_monitor_duration = (
                self._doppler_cooling_monitor_time_global
            )
            self.doppler_cooling_monitor_duration_mu = self.core.seconds_to_mu(
                self._doppler_cooling_monitor_time_global
            )

        else:
            self.freq_mu = [
                freq_to_mu(i) for i in [self._cool_1_freq, self._cool_2_freq]
            ]
            self.cool_amp = [np.int32(i) for i in [self._cool_1_amp, self._cool_2_amp]]
            self.doppler_cooling_duration = self._doppler_cooling_time
            self.doppler_cooling_duration_mu = self.core.seconds_to_mu(
                self._doppler_cooling_time
            )
            self.doppler_cooling_monitor_duration = self._doppler_cooling_monitor_time
            self.doppler_cooling_monitor_duration_mu = self.core.seconds_to_mu(
                self._doppler_cooling_monitor_time
            )

        assert self.doppler_cooling_duration >= self.doppler_cooling_monitor_duration
        assert (
            self.doppler_cooling_duration_mu >= self.doppler_cooling_monitor_duration_mu
        )

    @kernel
    def init(self):
        # manual parallel. Equiv to "with parallel" block
        parallel_start_mu = now_mu()
        for idds in range(len(self.dds)):
            at_mu(parallel_start_mu)
            self.dds[idds].init()
        self.write_to_dds()
        self.set_power(0b01)
        self.update_amp(self.cool_amp)
        self.on()

    @kernel
    def idle(self):
        self.update_amp(self.cool_amp)
        self.on()
        self.set_power(0b01)

    @kernel
    def set_power(self, state):
        if state == 0b00:
            self.power_cool_1.off()
            self.power_cool_2.off()
        elif state == 0b01:
            self.power_cool_1.off()
            self.power_cool_2.on()
        elif state == 0b10:
            self.power_cool_1.on()
            self.power_cool_2.off()
        elif state == 0b11:
            self.power_cool_1.on()
            self.power_cool_2.on()


class PumpDetect(BaseDoubleDDS, HasEnvironment):
    def load_globals(self, archive: bool=True):
        # Initialize the UI elements using the corresponding global variables
        resonance_pumpdetectAOM1_freq = self.get_dataset("global.CW_Lasers.Yb.Resonance_PumpDetectAOM1_Freq",
                                                    archive=archive)
        resonance_pumpdetectAOM2_freq = self.get_dataset("global.CW_Lasers.Yb.Resonance_PumpDetectAOM2_Freq",
                                                    archive=archive)

        pump_detuning = self.get_dataset("global.Pump.Yb.Detuning", archive=archive)
        self._pump_1_freq_global = resonance_pumpdetectAOM1_freq + 1 / 2 * pump_detuning
        self._pump_2_freq_global = resonance_pumpdetectAOM2_freq + 1 / 2 * pump_detuning

        self._pump_1_amp_global = self.get_dataset("global.Pump.Yb.PumpDetectAOM1_Amp", archive=archive)
        self._pump_2_amp_global = self.get_dataset("global.Pump.Yb.PumpDetectAOM2_Amp", archive=archive)

        self._slow_pump_1_amp_global = self.get_dataset("global.Slow_Pump.Yb.PumpDetectAOM1_Amp", archive=archive)
        self._slow_pump_2_amp_global = self.get_dataset("global.Slow_Pump.Yb.PumpDetectAOM2_Amp", archive=archive)

        detect_detuning = self.get_dataset("global.Detect.Yb.Detuning", archive=archive)
        self._detect_1_freq_global = resonance_pumpdetectAOM1_freq + 1 / 2 * detect_detuning
        self._detect_2_freq_global = resonance_pumpdetectAOM2_freq + 1 / 2 * detect_detuning

        self._detect_1_amp_global = self.get_dataset("global.Detect.Yb.PumpDetectAOM1_Amp", archive=archive)
        self._detect_2_amp_global = self.get_dataset("global.Detect.Yb.PumpDetectAOM2_Amp", archive=archive)

        self._optical_pump_time_global = self.get_dataset("global.Pump.Duration", archive=archive)
        self._slow_pump_time_global = self.get_dataset("global.Slow_Pump.Duration", archive=archive)
        self._detect_time_global = self.get_dataset("global.Detect.Duration", archive=archive)

        self._slow_pump_on_global = bool(self.get_dataset("global.Slow_Pump.Slow_Pump_On", archive=archive))

    def build(self):
        # Get global Cooling DDS Values, don't archive in build
        archive = False
        self.load_globals(archive=archive)

        self.setattr_argument(
            "_pump_1_freq",
            NumberValue(
                default=self._pump_1_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Pump",
        )
        self.setattr_argument(
            "_pump_2_freq",
            NumberValue(
                default=self._pump_2_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Pump",
        )
        self.setattr_argument(
            "_pump_1_amp",
            NumberValue(
                default=self._pump_1_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Pump",
        )
        self.setattr_argument(
            "_pump_2_amp",
            NumberValue(
                default=self._pump_2_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Pump",
        )
        self.setattr_argument(
            "_optical_pump_time",
            NumberValue(default=self._optical_pump_time_global, unit="us"),
            group="Pump",
        )

        self.setattr_argument(
            "_slow_pump_1_amp",
            NumberValue(
                default=self._slow_pump_1_amp_global,
                min=0,
                scale=1,
                max=1000,
                ndecimals=0,
            ),
            group="Pump",
        )
        self.setattr_argument(
            "_slow_pump_2_amp",
            NumberValue(
                default=self._slow_pump_2_amp_global,
                min=0,
                scale=1,
                max=1000,
                ndecimals=0,
            ),
            group="Pump",
        )
        self.setattr_argument(
            "_slow_pump_time",
            NumberValue(default=self._slow_pump_time_global, unit="us"),
            group="Pump",
        )

        self.setattr_argument(
            "_slow_pump_on",
            BooleanValue(default=self._slow_pump_on_global),
            group="Pump",
        )

        self.setattr_argument(
            "_use_global_pump_values", BooleanValue(default=True), group="Pump"
        )

        self.setattr_argument(
            "_detect_1_freq",
            NumberValue(
                default=self._detect_1_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Detect",
        )
        self.setattr_argument(
            "_detect_2_freq",
            NumberValue(
                default=self._detect_2_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Detect",
        )
        self.setattr_argument(
            "_detect_1_amp",
            NumberValue(
                default=self._detect_1_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Detect",
        )
        self.setattr_argument(
            "_detect_2_amp",
            NumberValue(
                default=self._detect_2_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Detect",
        )
        self.setattr_argument(
            "_detect_time",
            NumberValue(default=self._detect_time_global, unit="us"),
            group="Detect",
        )

        self.setattr_argument(
            "_use_global_detect_values", BooleanValue(default=True), group="Detect"
        )

        self.setattr_device("core")
        self.setattr_device("pump_det_dds1")
        self.setattr_device("pump_det_dds2")
        self.setattr_device("optical_pumping")

    def prepare(self):

        self.dds = [self.pump_det_dds1, self.pump_det_dds2]
        self.freq_mu = [np.int64(i) for i in [0, 0]]
        self.amp = [np.int32(i) for i in [0, 0]]
        self.phase_mu = [np.int32(i) for i in [0,0]]

        # The global variables may have changed. Reload them into local variables before proceeding.
        if self._use_global_pump_values or self._use_global_detect_values:
            self.load_globals(archive=True)

        if self._use_global_pump_values:
            self.pump_frequency_mu = [
                freq_to_mu(i)
                for i in [self._pump_1_freq_global, self._pump_2_freq_global]
            ]
            self.pump_amp = [
                np.int32(i) for i in [self._pump_1_amp_global, self._pump_2_amp_global]
            ]
            self.slow_pump_amp = [
                np.int32(i)
                for i in [self._slow_pump_1_amp_global, self._slow_pump_2_amp_global]
            ]
            self.pump_duration = self._optical_pump_time_global
            self.pump_duration_mu = self.core.seconds_to_mu(
                self._optical_pump_time_global
            )
            self.slow_pump_duration = self._slow_pump_time_global
            self.slow_pump_duration_mu = self.core.seconds_to_mu(
                self._slow_pump_time_global
            )
            self.do_slow_pump = bool(self._slow_pump_on_global)
        else:
            self.pump_frequency_mu = [
                freq_to_mu(i) for i in [self._pump_1_freq, self._pump_2_freq]
            ]
            self.pump_amp = [np.int32(i) for i in [self._pump_1_amp, self._pump_2_amp]]
            self.slow_pump_amp = [
                np.int32(i) for i in [self._slow_pump_1_amp, self._slow_pump_2_amp]
            ]
            self.pump_duration = self._optical_pump_time
            self.pump_duration_mu = self.core.seconds_to_mu(self._optical_pump_time)
            self.slow_pump_duration = self._slow_pump_time
            self.slow_pump_duration_mu = self.core.seconds_to_mu(self._slow_pump_time)
            self.do_slow_pump = bool(self._slow_pump_on)

        if self._use_global_detect_values:
            self.detect_frequency_mu = [
                freq_to_mu(i)
                for i in [self._detect_1_freq_global, self._detect_2_freq_global]
            ]
            self.detect_amp = [
                np.int32(i)
                for i in [self._detect_1_amp_global, self._detect_2_amp_global]
            ]
            self.detect_duration = self._detect_time_global
            self.detect_duration_mu = self.core.seconds_to_mu(self._detect_time_global)
        else:
            self.detect_frequency_mu = [
                freq_to_mu(i) for i in [self._detect_1_freq, self._detect_2_freq]
            ]
            self.detect_amp = [
                np.int32(i) for i in [self._detect_1_amp, self._detect_2_amp]
            ]
            self.detect_duration = self._detect_time
            self.detect_duration_mu = self.core.seconds_to_mu(self._detect_time)

    @kernel
    def init(self):
        raise NotImplementedError

    @kernel
    def prepare_pump(self):
        with parallel:
            with sequential:
                self.set_param_mu(freq_mu=self.pump_frequency_mu, amp_int=self.pump_amp)
                self.write_to_dds()
            self.pump_eom(pump=True)

    @kernel
    def prepare_detect(self):
        with parallel:
            with sequential:
                self.set_param_mu(freq_mu=self.detect_frequency_mu,
                                  amp_int=self.detect_amp)
                self.write_to_dds()
            self.pump_eom(pump=False)

    @kernel
    def pump_eom(self, pump: TBool):
        if pump is True:
            self.optical_pumping.off()
        elif pump is False:
            self.optical_pumping.on()
        # on 5/4/19, observed that pumping is incomplete unless
        # a delay is inserted here
        delay(80 * us)

    @kernel
    def pump(self):
        if self.do_slow_pump:
            self.pulse_mu(self.slow_pump_duration_mu)
        else:
            self.pulse_mu(self.pump_duration_mu)


class SSCooling_Settings(HasEnvironment):
    def load_globals(self, archive: bool=True):
        # Initialize the UI elements using the corresponding global variables

        # These two variables are also loaded in PumpDetect.load_globals(), which supersedes SSCooling
        resonance_pumpdetectAOM1_freq = self.get_dataset("global.CW_Lasers.Yb.Resonance_PumpDetectAOM1_Freq",
                                                    archive=False)
        resonance_pumpdetectAOM2_freq = self.get_dataset("global.CW_Lasers.Yb.Resonance_PumpDetectAOM2_Freq",
                                                    archive=False)

        ss_pumpdetect_detuning = self.get_dataset("global.SS_Cool.Yb.PumpDetect_Detuning", archive=archive)

        self._pumpdetect_1_freq_global = resonance_pumpdetectAOM1_freq + 1 / 2 * ss_pumpdetect_detuning

        self._pumpdetect_2_freq_global = resonance_pumpdetectAOM2_freq + 1 / 2 * ss_pumpdetect_detuning

        self._pumpdetect_1_amp_global = self.get_dataset("global.SS_Cool.Yb.PumpDetectAOM1_Amp", archive=archive)
        self._pumpdetect_2_amp_global = self.get_dataset("global.SS_Cool.Yb.PumpDetectAOM2_Amp", archive=archive)

        # These two variables are also loaded in DopplerCooling.load_globals(), which supersedes SSCooling
        resonance_coolAOM1_freq = self.get_dataset("global.CW_Lasers.Yb.Resonance_CoolAOM1_Freq", archive=False)
        resonance_coolAOM2_freq = self.get_dataset("global.CW_Lasers.Yb.Resonance_CoolAOM2_Freq", archive=False)

        ss_cool_detuning = self.get_dataset("global.SS_Cool.Yb.Cool_Detuning", archive=archive)

        self._cool_1_freq_global = resonance_coolAOM1_freq + 1 / 2 * ss_cool_detuning
        self._cool_2_freq_global = resonance_coolAOM2_freq + 1 / 2 * ss_cool_detuning

        self._cool_1_amp_global = self.get_dataset("global.SS_Cool.Yb.CoolAOM1_Amp", archive=archive)
        self._cool_2_amp_global = self.get_dataset("global.SS_Cool.Yb.CoolAOM2_Amp", archive=archive)

        self._ss_cooling_time_global = self.get_dataset("global.SS_Cool.Duration", archive=archive)
        self._ss_cooling_monitor_time_global = self.get_dataset("global.SS_Cool.Monitor_Duration", archive=archive)

        # 172
        self._ss_cooling_coolant_time_global = self.get_dataset("global.SS_Cool.Duration_Coolant", archive=archive)
        self._ss_cooling_coolant_monitor_time_global = self.get_dataset("global.SS_Cool.Monitor_Duration_Coolant", archive=archive)

        # All
        self._ss_cooling_all_time_global = self.get_dataset("global.SS_Cool.Duration_All", archive=archive)
        self._ss_cooling_all_monitor_time_global = self.get_dataset("global.SS_Cool.Monitor_Duration_All", archive=archive)

    def build(self):
        # Load Globals for GUI args but dont archive in build
        archive = False
        self.load_globals(archive=archive)

        # Get Cooling DDS Values
        self.setattr_argument(
            "_ss_cool_1_freq",
            NumberValue(
                default=self._cool_1_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_cool_2_freq",
            NumberValue(
                default=self._cool_2_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_cool_1_amp",
            NumberValue(
                default=self._cool_1_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_cool_2_amp",
            NumberValue(
                default=self._cool_2_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_detect_1_freq",
            NumberValue(
                default=self._pumpdetect_1_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_detect_2_freq",
            NumberValue(
                default=self._pumpdetect_2_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_detect_1_amp",
            NumberValue(
                default=self._pumpdetect_1_amp_global,
                min=0,
                scale=1,
                max=1000,
                ndecimals=0,
            ),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_detect_2_amp",
            NumberValue(
                default=self._pumpdetect_2_amp_global,
                min=0,
                scale=1,
                max=1000,
                ndecimals=0,
            ),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_cooling_time",
            NumberValue(default=self._ss_cooling_time_global, unit="ms"),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_cooling_monitor_time",
            NumberValue(default=self._ss_cooling_monitor_time_global, unit="ms"),
            group="Second Stage Cooling",
        )

        # for 172
        self.setattr_argument(
            "_ss_cooling_coolant_time",
            NumberValue(default=self._ss_cooling_coolant_time_global, unit="ms"),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_cooling_coolant_monitor_time",
            NumberValue(default=self._ss_cooling_coolant_monitor_time_global, unit="ms"),
            group="Second Stage Cooling",
        )

        # for all
        self.setattr_argument(
            "_ss_cooling_all_time",
            NumberValue(default=self._ss_cooling_all_time_global, unit="ms"),
            group="Second Stage Cooling",
        )
        self.setattr_argument(
            "_ss_cooling_all_monitor_time",
            NumberValue(default=self._ss_cooling_all_monitor_time_global, unit="ms"),
            group="Second Stage Cooling",
        )

        self.setattr_argument(
            "_use_global_ss_values",
            BooleanValue(default=True),
            group="Second Stage Cooling",
        )

        self.setattr_device("core")

    def prepare(self):

        if self._use_global_ss_values:
            # The global variables may have changed. Reload them into local variables before proceeding.
            self.load_globals(archive=True)

            self.cool_frequency_mu = [
                freq_to_mu(i)
                for i in [self._cool_1_freq_global, self._cool_2_freq_global]
            ]
            self.detect_frequency_mu = [
                freq_to_mu(i)
                for i in [
                    self._pumpdetect_1_freq_global,
                    self._pumpdetect_2_freq_global,
                ]
            ]
            self.cool_amp = [
                np.int32(i) for i in [self._cool_1_amp_global, self._cool_2_amp_global]
            ]
            self.detect_amp = [
                np.int32(i)
                for i in [self._pumpdetect_1_amp_global, self._pumpdetect_2_amp_global]
            ]
            self.ss_cooling_duration = self._ss_cooling_time_global
            self.ss_cooling_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_time_global
            )
            self.ss_cooling_monitor_duration = self._ss_cooling_monitor_time_global
            self.ss_cooling_monitor_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_monitor_time_global
            )
            # for 172
            self.ss_cooling_coolant_duration = self._ss_cooling_coolant_time_global
            self.ss_cooling_coolant_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_coolant_time_global
            )
            self.ss_cooling_coolant_monitor_duration = self._ss_cooling_coolant_monitor_time_global
            self.ss_cooling_coolant_monitor_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_coolant_monitor_time_global
            )
            # for all
            self.ss_cooling_all_duration = self._ss_cooling_all_time_global
            self.ss_cooling_all_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_all_time_global
            )
            self.ss_cooling_all_monitor_duration = self._ss_cooling_all_monitor_time_global
            self.ss_cooling_all_monitor_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_all_monitor_time_global
            )

        else:
            self.cool_frequency_mu = [
                freq_to_mu(i) for i in [self._ss_cool_1_freq, self._ss_cool_2_freq]
            ]
            self.detect_frequency_mu = [
                freq_to_mu(i) for i in [self._ss_detect_1_freq, self._ss_detect_2_freq]
            ]
            self.cool_amp = [
                np.int32(i) for i in [self._ss_cool_1_amp, self._ss_cool_2_amp]
            ]
            self.detect_amp = [
                np.int32(i) for i in [self._ss_detect_1_amp, self._ss_detect_2_amp]
            ]
            self.ss_cooling_duration = self._ss_cooling_time
            self.ss_cooling_duration_mu = self.core.seconds_to_mu(self._ss_cooling_time)
            self.ss_cooling_monitor_duration = self._ss_cooling_monitor_time
            self.ss_cooling_monitor_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_monitor_time
            )

            # for 172
            self.ss_cooling_coolant_duration = self._ss_cooling_coolant_time
            self.ss_cooling_coolant_duration_mu = self.core.seconds_to_mu(self._ss_cooling_coolant_time)
            self.ss_cooling_coolant_monitor_duration = self._ss_cooling_coolant_monitor_time
            self.ss_cooling_coolant_monitor_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_coolant_monitor_time
            )

            # for all
            self.ss_cooling_all_duration = self._ss_cooling_all_time
            self.ss_cooling_all_duration_mu = self.core.seconds_to_mu(self._ss_cooling_all_time)
            self.ss_cooling_all_monitor_duration = self._ss_cooling_all_monitor_time
            self.ss_cooling_all_monitor_duration_mu = self.core.seconds_to_mu(
                self._ss_cooling_all_monitor_time
            )

        assert self.ss_cooling_duration >= self.ss_cooling_monitor_duration
        assert self.ss_cooling_duration_mu >= self.ss_cooling_monitor_duration_mu
        assert self.ss_cooling_coolant_duration >= self.ss_cooling_coolant_monitor_duration
        assert self.ss_cooling_coolant_duration_mu >= self.ss_cooling_coolant_monitor_duration_mu
        assert self.ss_cooling_all_duration >= self.ss_cooling_all_monitor_duration
        assert self.ss_cooling_all_duration_mu >= self.ss_cooling_all_monitor_duration_mu

class DopplerCoolingCoolant(HasEnvironment):
    # Measured with fluorescence, resonance is at 348 MHz beatnote freq at 200 MHz AOM RF input;
    def load_globals(self, archive: bool=True):
        # Initialize the UI elements using the corresponding global variables
        self._coolant_dopplercool_freq_global = self.get_dataset("global.Doppler_Cool.Yb.Coolant.AOM_Freq", archive=archive)
        self._coolant_dopplercool_amp_global = self.get_dataset("global.Doppler_Cool.Yb.Coolant.AOM_Amp", archive=archive)
        self._coolant_dopplercool_monitor_duration_global = self.get_dataset("global.Doppler_Cool.Yb.Coolant.Monitor_Duration", archive=archive)

    def build(self):
        # Get global Cooling DDS Values, don't archive in build
        archive = False
        self.load_globals(archive=archive)

        self.setattr_argument(
            "_coolant_doppler_monitor_duration",
            NumberValue(
                default=self._coolant_dopplercool_monitor_duration_global,
                unit="us",
            ),
            group="Coolant",
        )

        self.setattr_argument(
            "_coolant_doppler_freq",
            NumberValue(
                default=self._coolant_dopplercool_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9
            ),
            group="Coolant",
        )

        self.setattr_argument(
            "_coolant_doppler_amp",
            NumberValue(
                default=1000,
                min=0,
                scale=1,
                max=1000,
                ndecimals=0
            ),
            group="Coolant",
        )

        self.setattr_argument(
            "_use_global_coolant_values", BooleanValue(default=True), group="Coolant",
        )

        # Add devices
        self.setattr_device("core")
        self.setattr_device("eom_935_172")
        self.setattr_device("aom_172_369")
        self.setattr_device("cool_172_dds")

    def prepare(self):
        # The global variables may have changed. Reload them into local variables before proceeding.
        if self._use_global_coolant_values:
            self.load_globals(archive=True)
            self.freq_mu = freq_to_mu(self._coolant_dopplercool_freq_global)
            self.cool_amp_coolant = np.int32(self._coolant_dopplercool_amp_global)
            self.coolant_monitor_duration = self._coolant_dopplercool_monitor_duration_global
            self.coolant_monitor_duration_mu = self.core.seconds_to_mu(
                self._coolant_dopplercool_monitor_duration_global
            )
        else:
            self.freq_mu = freq_to_mu(self._coolant_doppler_freq)
            self.cool_amp_coolant = np.int32(self._coolant_doppler_amp)
            self.coolant_monitor_duration = self._coolant_doppler_monitor_duration
            self.coolant_monitor_duration_mu = self.core.seconds_to_mu(self._coolant_doppler_monitor_duration)

    @kernel
    def init(self):
        self.cool_172_dds.set_mu(bus_group=1, amplitude_mu=self.cool_amp_coolant)
        self.cool_172_dds.init()
        self.write_to_dds()
        self.cool_172_dds.on()

    @kernel
    def idle_source(self):
        self.cool_172_dds.off()

    @kernel
    def pump_sawtooth(self, pump=True):
        # True: on; False: off. (only the sawtooth despite what status 3ghz is in)
        if pump:
            self.eom_935_172.on()
            delay(20 * us)
        else:
            self.eom_935_172.off()
            delay(20 * us)

    @kernel
    def cool(self, cool=True):
        if cool:
            self.aom_172_369.on()
            delay(20 * us)
        else:
            self.aom_172_369.off()
            delay(20 * us)


    @kernel
    def idle(self):
        self.idle_source()
        self.aom_172_369.off()
        delay(20 * us)

    @kernel
    def write_to_dds(self):
        ret = self.cool_172_dds.set_mu(
            bus_group=1,
            frequency_mu=self.freq_mu,
            amplitude_mu=self.cool_amp_coolant,
        )

class SDCooling_Settings(HasEnvironment):

    def load_globals(self, archive: bool=True):

        isotope = self.get_dataset("global.SD_Cool.Isotope", archive=archive)
        assert(isotope==171 or isotope==172)

        if isotope==171:
            cavity_offset = self.get_dataset("global.Quadrupole.Cavity_Drift_Offset", archive=archive)
            self._SD_DDS_resonance_freq_global = self.get_dataset("global.Quadrupole.171.Resonance", archive=archive)
            self._SD_DDS_resonance_freq_global += cavity_offset
            self._SD_DDS_diff_freq_global = 0

        else:
            self._SD_DDS_resonance_freq_global = self.get_dataset("global.Quadrupole.172.Resonance", archive=archive) + self.get_dataset("global.Quadrupole.Cavity_Drift_Offset", archive=archive)
            self._SD_DDS_diff_freq_global = self.get_dataset("global.Quadrupole.172.Zeeman_Splitting", archive=archive)

        self._cool_detuning_global =self.get_dataset("global.SD_Cool.Detuning", archive=archive)

        self._SD_DDS_mean_freq_global = self._SD_DDS_resonance_freq_global + self._cool_detuning_global

        self._SD_DDS1_amp_global = self.get_dataset("global.SD_Cool.Tone1_Amp", archive=archive)
        self._SD_DDS2_amp_global = self.get_dataset("global.SD_Cool.Tone2_Amp", archive=archive)

        self._cooling_duration_global = self.get_dataset("global.SD_Cool.Duration", archive=archive)
        self._loaded_globals = True

    def build(self):
        archive = False
        self.load_globals(archive=archive)

        self.setattr_argument(
            "_SD_DDS_mean_freq",
            NumberValue(
                default=self._SD_DDS_mean_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="SD Cooling"
        )
        self.setattr_argument(
            "_SD_DDS_diff_freq",
            NumberValue(
                default=self._SD_DDS_diff_freq_global,
                unit="MHz",
                min=0 * MHz,
                max=50 * MHz,
                ndecimals=9,
            ),
            group="SD Cooling"
        )
        self.setattr_argument(
            "_SD_DDS1_amp",
            NumberValue(
                default=self._SD_DDS1_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="SD Cooling"
        )
        self.setattr_argument(
            "_SD_DDS2_amp",
            NumberValue(
                default=self._SD_DDS2_amp_global, min=0, scale=1, max=1000, ndecimals=0
            ),
            group="SD Cooling"
        )
        self.setattr_argument(
            "_SD_cooling_duration",
            NumberValue(default=self._cooling_duration_global, unit="ms"),
            group="SD Cooling"
        )

        self.setattr_argument(
            "_use_global_values", BooleanValue(default=True), group="SD Cooling"
        )

        self.setattr_device("core")

    def prepare(self):
        self._loaded_globals = False

        if self._use_global_values:
            # The global variables may have changed since the build()
            # Therefore, reload them into local variables
            self.load_globals(archive=True)

            self.freq_mu = [
                freq_to_mu(i)
                for i in [self._SD_DDS_mean_freq_global - self._SD_DDS_diff_freq_global/2, self._SD_DDS_mean_freq_global + self._SD_DDS_diff_freq_global/2]
            ]
            self.amp = [
                np.int32(i)
                for i in [self._SD_DDS1_amp_global, self._SD_DDS2_amp_global]
            ]
            self.cooling_duration = self._cooling_duration_global
            self.cooling_duration_mu = self.core.seconds_to_mu(
                self.cooling_duration
            )

        else:
            self.freq_mu = [
                freq_to_mu(i) for i in [self._SD_DDS_mean_freq - self._SD_DDS_diff_freq/2, self._SD_DDS_mean_freq + self._SD_DDS_diff_freq/2]
            ]
            self.amp = [np.int32(i) for i in [self._SD_DDS1_amp, self._SD_DDS2_amp]]

            self.sd_cooling_duration = self._cooling_duration
            self.sd_cooling_duration_mu = self.core.seconds_to_mu(
                self.sd_cooling_duration
            )

    def set_detuning(self, detuning):
        if not self._loaded_globals:
            _LOGGER.error("Setting Doppler cooling detuning without having loaded the resonance setting datasets!")

        self._cooldds1_freq_global = self._resonance_coolAOM1_freq + 1 / 2 * detuning
        self._cooldds2_freq_global = self._resonance_coolAOM2_freq + 1 / 2 * detuning

        self.freq_mu = [
            freq_to_mu(i)
            for i in [self._coolaom1_freq_global, self._coolaom2_freq_global]
        ]

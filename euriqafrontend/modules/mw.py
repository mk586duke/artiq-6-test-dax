import numpy as np
from artiq.experiment import NumberValue
from artiq.experiment import TFloat, TInt32, TInt64
from artiq.language import kernel, parallel
from artiq.language.environment import HasEnvironment
from artiq.language.units import A, MHz, ms, s, us

from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqabackend.coredevice.ad9912 import phase_to_mu


class Microwave(HasEnvironment):

    _CONST_PI = np.pi

    def build(self):
        # Get Cooling DDS Values
        self.setattr_argument(
            "mw_freq",
            NumberValue(
                default=self.get_dataset("global.Ion_Freqs.MW_Freq"),
                unit="MHz",
                min=0 * MHz,
                max=450 * MHz,
                ndecimals=9,
            ),
            group="Microwave",
        )
        self.setattr_argument(
            "mw_amp",
            NumberValue(default=999, min=0, scale=1, max=1000, ndecimals=0),
            group="Microwave",
        )
        self.setattr_argument(
            "mw_phase",
            NumberValue(default=0, unit="Degrees", min=0, scale=1, max=360),
            group="Microwave",
        )

        self.setattr_device("shutter_mw")
        self.setattr_device("microwave_dds")

    def prepare(self):
        self.dds = self.microwave_dds
        self.freq = freq_to_mu(self.mw_freq)
        self.amp = np.int32(self.mw_amp)
        self.phase = phase_to_mu(self.mw_phase / 360 * 2 * self._CONST_PI)

    @kernel
    def update_freq(self, freq_hz: TFloat):
        self.freq = freq_to_mu(freq_hz)
        self.dds.set_mu(bus_group=1, frequency_mu=self.freq)

    @kernel
    def update_amp(self, amp_int: TInt32):
        self.amp = np.int32(amp_int)
        self.dds.set_mu(bus_group=1, amplitude_mu=self.amp)

    @kernel
    def update_phase(self, phase_deg: TFloat):
        self.phase = phase_to_mu(phase_deg / 360 * 2 * self._CONST_PI)
        self.dds.dds.phase_mu(phase=self.phase)

    @kernel
    def update_phase_mu(self, phase_mu: TInt32):
        self.phase = phase_mu
        self.dds.dds.phase_mu(phase=self.phase)

    @kernel
    def init(self):
        self.dds.init()
        self.write_to_dds()
        self.shutter_mw.off()

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
        with parallel:
            self.dds.on()
            self.shutter_mw.on()

    @kernel
    def off(self):
        with parallel:
            self.dds.off()
            self.shutter_mw.off()

    @kernel
    def pulse(self, time_s: TFloat):
        with parallel:
            self.dds.pulse(time_s)
            self.shutter_mw.pulse(time_s)

    @kernel
    def pulse_mu(self, time_mu: TInt64):
        with parallel:
            self.dds.pulse_mu(time_mu=time_mu)
            self.shutter_mw.pulse_mu(time_mu)

    @kernel
    def idle(self):
        self.off()

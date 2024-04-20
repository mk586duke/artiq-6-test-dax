"""Basic experiment to test that ONE DDS is working properly."""
from artiq.experiment import delay
from artiq.experiment import EnvExperiment
from artiq.experiment import kernel
from artiq.language.units import ns
from artiq.language.units import us


class testDDS(EnvExperiment):
    """Experiment to test setting amplitude & frequency of one DDS."""

    def build(self):
        """Declare arguments and devices."""
        self.setattr_device("core")
        self.marker = self.get_device("out2_4")
        self.setattr_device("dds0")
        self.frequency_machine_units = int(self.freq_bin_val(10 ** 8))

    def freq_bin_val(self, freq):
        """Convert frequency to corresponding binary value for DDS."""
        return freq * 2 ** 48 / 1e9

    @kernel
    def run(self):
        """Set DDS to 100 MHz."""
        self.core.reset()
        self.core.break_realtime()
        # self.dds0.setup_bus(write_div=4)
        delay(50 * us)
        self.dds0.set(frequency=self.frequency_machine_units, amplitude=1023)
        # self.dds0.set(frequency=int(14073748835532, width=64))
        self.marker.pulse(50 * ns)

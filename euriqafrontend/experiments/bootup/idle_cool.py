"""Idle kernel to maintain the ion.

Cools the ion and keeps all outputs at default (correct) values.

Load with:
```
$ artiq_compile ./euriqafrontend/experiments/bootup/idle_cool.py
... --device-db=./euriqabackend/databases/device_db_main_box.py
... -o ./idle_cool.elf
$ artiq_coremgmt -D 192.168.78.105 config write -f idle_kernel idle_cool.elf
$ artiq_coremgmt -D 192.168.78.105 reboot
```

TODO:
    * pull config values (i.e. DDS frequencies) from `dataset_db.pyon`
"""
import artiq.language.environment as artiq_env
from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.core import now_mu
from artiq.language.types import TNone
from artiq.language.units import MHz
from artiq.language.units import ms
from artiq.language.units import us

from euriqabackend.coredevice.ad9912 import freq_to_mu


class IdleKernel(artiq_env.EnvExperiment):
    """Idle kernel to initialize lines & cool ion."""

    kernel_invariants = {"cooling_dds_1_freq_mu", "cooling_dds_2_freq_mu"}

    def build(self):
        """Initialize experiment & variables."""
        # devices
        self.setattr_device("core")
        self.setattr_device("oeb")
        self.setattr_device("cooling_dds1")
        self.setattr_device("cooling_dds2")
        self.setattr_device("power_cool_1")
        self.setattr_device("power_cool_2")

        # real-time Sandia x100 DAC
        self.setattr_device("dac_trigger")
        self.setattr_device("dac_serial")

        self.setattr_device("led0")

        # arguments
        self.cooling_dds_1_freq_mu = freq_to_mu(180 * MHz)
        self.cooling_dds_2_freq_mu = freq_to_mu(195 * MHz)

    @kernel
    def run(self) -> TNone:
        """Run the idle kernel."""
        self.kn_wait_for_finish()
        self.kn_setup()
        self.kn_run_forever()

    @kernel
    def kn_wait_for_finish(self) -> TNone:
        """Wait for any previously executing kernels to finish, then reset."""
        start_time = now_mu() + self.core.seconds_to_mu(1 * ms)
        while self.core.get_rtio_counter_mu() < start_time:
            pass
        self.core.reset()

    @kernel
    def kn_setup(self) -> TNone:
        """Initialize the core and all devices."""
        self.oeb.off()
        self.cooling_dds1.init()
        self.cooling_dds2.init()
        delay(1 * us)

        # cooling power setup (minimal power)
        self.power_cool_1.off()
        self.power_cool_2.off()

        # initialize DAC serial output to avoid it hanging up (doesn't like all 1's)
        self.dac_trigger.off()
        self.dac_serial.set_config_mu(0, 1, 4, 0)
        self.dac_serial.write(0)

        # pre-program DDS (relies on OEB off, so must happen after)
        self.cooling_dds1.set_mu(
            1, frequency_mu=self.cooling_dds_1_freq_mu, amplitude_mu=600
        )
        self.cooling_dds2.set_mu(
            1, frequency_mu=self.cooling_dds_2_freq_mu, amplitude_mu=600
        )

    @kernel
    def kn_run_forever(self) -> TNone:
        """Run the experiment on the core device."""
        self.cooling_dds1.on()
        self.cooling_dds2.on()

        while True:
            # first stage (far-detuned) doppler cooling
            self.led0.on()
            delay(500 * ms)
            self.led0.off()
            delay(500 * ms)

"""Idle kernel to maintain the ion.

Cools the ion and keeps all outputs at default (correct) values.

Load with:
```
$ artiq_compile ./euriqafrontend/experiments/bootup/startup_kernel.py
... --device-db=./euriqabackend/databases/device_db_main_box.py
... -o ./startup_kernel.elf
$ artiq_coremgmt -D 192.168.78.105 config write -f startup_kernel startup_kernel.elf
$ artiq_coremgmt -D 192.168.78.105 reboot
```
"""
import artiq.language.environment as artiq_env
from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.types import TNone
from artiq.language.units import us


class StartupKernel(artiq_env.EnvExperiment):
    """Startup kernel to initialize lines."""

    def build(self):
        """Initialize experiment & variables."""
        # devices
        self.setattr_device("core")
        self.setattr_device("oeb")

        # real-time Sandia x100 DAC
        self.setattr_device("dac_trigger")
        self.setattr_device("dac_serial")

    @kernel
    def run(self) -> TNone:
        """Initialize the core and all devices."""
        self.core.reset()
        self.oeb.on()
        self.dac_serial.set_config_mu(0, 1, 4, 0)

        delay(0.1 * us)

        # initialize DAC serial output to avoid it hanging up (doesn't like all 1's)
        # the SPI line will default to High (all 1's), so minimize this to avoid hanging
        self.dac_trigger.off()
        self.dac_serial.write(0)

        delay(50 * us)
        # for changing TTL lines outputs in the GUI,
        # or just so that every experiment doesn't have to do `self.oeb.off()`
        # (though they still should)
        # self.oeb.off()

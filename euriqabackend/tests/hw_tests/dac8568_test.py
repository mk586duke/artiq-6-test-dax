"""Test the DAC8568 chip onboard the Duke breakout board.

This is an 8-output DAC, single-ended. Can be used for driving piezos,
locks, or other analog lab devices.

Running this test successfully demonstrates that the DAC coredevice driver
compiles and has all required functionality.
"""
import logging

import artiq
from artiq.language import delay
from artiq.language import kernel
from artiq.language.units import ms
from artiq.language.units import us

import euriqabackend.coredevice.dac8568 as dac

_LOGGER = logging.getLogger(__name__)


def pause_input():
    """Pause program until hit ENTER (only works in `artiq_run`)."""
    input("Press ENTER to continue")


class DAC8568Test(artiq.language.environment.EnvExperiment):
    """Test that the DAC8568 driver compiles and works."""

    # pylint: disable=attribute-defined-outside-init

    def build(self):
        """Define devices used."""
        self.setattr_device("core")
        self.setattr_device("oeb")
        self.marker = self.get_device("out2_4")
        self.dac = self.get_device("dac8568_1")

    def prepare(self):
        """Initialize the DAC."""
        # self.dac = dac.DAC8568(
        #     self._HasEnvironment__device_mgr,
        #     spi_device="spi5",
        #     chip_select=1,
        #     ldac_trigger="load_dac_1",
        # )
        self.test_out_mu = dac.vout_to_mu(1.0, 5.0)
        _LOGGER.info("DAC prepared properly.")

    @kernel
    def run(self):
        """Run a test on the DAC8568.

        Mostly tests that everything compiles properly in ARTIQ.
        """
        _LOGGER.info("Beginning tests")
        self.core.reset()
        self.core.break_realtime()
        self.test_init()
        pause_input()

        _LOGGER.info("Testing voltages")
        self.core.break_realtime()
        self.test_voltages()
        pause_input()

        _LOGGER.info("Testing settings")
        self.core.break_realtime()
        self.test_settings()

    @kernel
    def test_init(self):
        """Test that the DAC can initialize itself properly."""
        # self.core.reset()
        self.oeb.off()
        delay(1 * us)
        self.marker.pulse(1 * us)
        self.dac.reset()
        self.dac.init()
        # check all voltages are at 0/disconnected

    @kernel
    def test_voltages(self):
        """Test that you can set voltages correctly."""
        test_channel = dac.AOut.Out4
        self.core.break_realtime()
        self.marker.pulse(2 * us)
        self.dac.set_voltage(dac.DACChannel.ALL, 2.0)

        # Delay update until LDAC pin toggled
        self.core.break_realtime()
        self.marker.on()
        self.dac.set_voltage(test_channel, 4.999, update_immediate=False)
        delay(10 * us)
        self.dac.update_outputs()
        delay(5 * us)
        self.marker.off()

        # update two channels simultaneously, no LDAC
        self.core.break_realtime()
        self.marker.pulse(3 * us)
        self.dac.set_voltage(
            test_channel, 0.8, update_immediate=False
        )  # delay until update_all_channels
        delay(50 * us)
        self.dac.set_voltage(test_channel, 0.5, update_all_channels=True)

        # set set_mu
        self.core.break_realtime()
        self.marker.pulse(5 * us)
        self.dac.set_voltage_mu(
            dac.DACCommands.write_and_update_all, dac.DACChannel.ALL, 32000
        )
        self.dac.set_voltage_mu(
            dac.DACCommands.write_and_update_all, dac.DACChannel.ALL, 0xFFFF
        )
        self.dac.set_voltage_mu(
            dac.DACCommands.write_and_update_all, dac.DACChannel.ALL, self.test_out_mu
        )
        return

    @kernel
    def test_settings(self):
        """Test that the DAC settings compile & (theoretically) work correctly."""
        # test setting the clear register
        self.core.break_realtime()
        self.dac.settings_clear_register(dac.ClearOptions.clear_to_middle)
        delay(50 * ms)

        # test changing which DACs are controlled by LDAC pin
        self.dac.settings_load_dac_sync([1, 2, 5])
        self.dac.settings_load_dac_sync([0xF])
        delay(50 * ms)

        # set internal reference
        self.dac.settings_internal_reference(
            static_reference=False, enable=True, always_on=True
        )
        delay(50 * ms)
        self.dac.settings_internal_reference(
            static_reference=False, enable=True, always_on=False
        )
        delay(50 * ms)
        self.dac.settings_internal_reference(static_reference=False, enable=False)
        delay(50 * ms)
        self.dac.settings_internal_reference(
            static_reference=True, enable=True, always_on=True, flex2stat=True
        )  # always_on does nothing
        delay(50 * ms)
        self.dac.settings_internal_reference(
            static_reference=True, enable=False, flex2stat=True
        )
        delay(50 * ms)

        # power down then power up channels
        self.dac.settings_power_down(0xF, dac.PowerDownImpedance.high_z)
        delay(1 * us)
        self.dac.settings_power_down(5, dac.PowerDownImpedance.dac_on)

        self.dac.settings_sync_mode(False)
        delay(1 * us)
        self.dac.settings_sync_mode(True)

        delay(50 * ms)
        # fix all modified settings.
        self.dac.reset()

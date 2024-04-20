"""Coredevice driver in ARTIQ for the TI DAC8568 (8-channel 16-bit DAC).

This DAC is on the Duke breakout board, and can be used for real-time feedback
and control of analog devices. Examples include real-time piezo feedback for
tuning laser direction.

This DAC communicates via SPI, so there must be gateware support for this
device.

Originally written by Bichen Zhang, updated/modified by Drew Risinger.
"""
import enum
import logging

import numpy as np
from artiq.coredevice import spi2 as spi
from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.core import portable
from artiq.language.types import TBool
from artiq.language.types import TFloat
from artiq.language.types import TInt32
from artiq.language.types import TList
from artiq.language.types import TNone
from artiq.language.units import MHz
from artiq.language.units import ns
from artiq.master.worker_db import DeviceManager

_LOGGER = logging.getLogger(__name__)


@portable
def vout_to_mu(v_out: TFloat, v_out_max: TFloat) -> TInt32:
    """Convert desired floating-point output voltage to DAC machine units.

    Note: This is a floating-point calculation, and so relatively time-costly.
    To mitigate this, you can pre-calculate these values in :meth:`prepare`.

    With 16 bits of binary precision, you get ~4 decimal places of DAC precision.

    >>> vout_to_mu(4.9999, 5.0)
    65534
    >>> vout_to_mu(2.49995, 2.5)
    65534
    >>> vout_to_mu(1.0, 5.0)
    13107
    """
    d_out = v_out / v_out_max * (2 ** 16)
    if d_out > 65535 or d_out < 0:
        raise DAC8568Exception("Voltage out of range")
    return np.int32(d_out)


@portable
def mu_to_vout(v_mu: TInt32, v_out_max: TFloat) -> TFloat:
    """Convert desired floating-point output voltage to DAC machine units.

    Note: This is a floating-point calculation, and so relatively time-costly.
    To mitigate this, you can pre-calculate these values in :meth:`prepare`.

    With 16 bits of binary precision, you get ~4 decimal places of DAC precision.

    >>> mu_to_vout(10000, 5.0)
    0.762939453125
    >>> mu_to_vout(20000, 2.5)
    0.762939453125
    >>> import math
    >>> mu_to_vout(math.ceil(65534/5), 5.0)
    0.9999847412109375
    """
    v_out = (v_mu * v_out_max) / (2 ** 16)
    if v_out > v_out_max or v_out < 0:
        raise DAC8568Exception("Voltage out of range")
    return v_out


class DAC8568Exception(Exception):
    """Exception to show that something went wrong with the DAC."""

    pass


class DACChannel(enum.IntEnum):
    """Denotes which channel of the DAC you are trying to control.

    Note: A-H are not necessarily equal to 1-8 on your PCB.
    """

    A = 0x0 << 20
    B = 0x1 << 20
    C = 0x2 << 20
    D = 0x3 << 20
    E = 0x4 << 20
    F = 0x5 << 20
    G = 0x6 << 20
    H = 0x7 << 20
    ALL = 0xF << 20


class AOut(enum.IntEnum):
    """Which Analog Output on the Duke Breakout board you're using.

    Mapped differently than :class:`DACChannel` on PCB. Use this!!
    """

    Out1 = DACChannel.A
    Out2 = DACChannel.C
    Out3 = DACChannel.E
    Out4 = DACChannel.G
    Out5 = DACChannel.H
    Out6 = DACChannel.F
    Out7 = DACChannel.D
    Out8 = DACChannel.B
    ALL = DACChannel.ALL


class DACCommands(enum.IntEnum):
    """Commands available to run on the TI DAC8568.

    Standard command format (name[bits]):
        `prefix[4]-control[4]-address[4]-data[16]-feature[4]`
    """

    clear = 0b0101 << 24  # clear code register
    # Control the LDAC (Load DAC) functionality (i.e. update DAC outputs)
    load_dac_pin_ignore = 0b0110 << 24  # setting to 1 overrides the LDAC pin
    write = 0b0000 << 24  # write to one data buffer
    update = 0b0001 << 24  # update selected register
    write_and_update = (
        0b0011 << 24
    )  # write to one data buffer and load corresponding DAC
    write_and_update_all = 0b0010 << 24  # write to one data buffer and update all DACs
    power_down_channel = 0b0100 << 24  # power down a channel
    static_internal_reference = 0b1000 << 24  # set internal reference to static output
    flexible_internal_reference = (
        0b1001 << 24
    )  # set output state of flexible internal reference
    software_reset = 0x07 << 24  # power-on reset


class ClearOptions(enum.IntEnum):
    """Options for what happens to ALL outputs when the clear pin is toggled."""

    clear_to_zero = 0b00
    clear_to_middle = 0b01
    clear_to_full = 0b10
    ignore_clear_pin = 0b11


class PowerDownImpedance(enum.IntEnum):
    """Settings for channel output impedance when powered down."""

    dac_on = 0b00 << 8
    ohms_1k = 0b01 << 8
    ohms_100k = 0b10 << 8
    high_z = 0b11 << 8


class InternalReferenceCommands(enum.IntEnum):
    """'Flexible' internal reference will power up when needed, if enabled here.

    This sets the power-up settings for the flexible internal voltage reference.
    """

    flex_reference_on = 0b100 << 17
    flex_reference_always_on = 0b101 << 17
    flex_reference_off = 0b110 << 17
    flexible_to_static = 0b000 << 17  # convert to statically-on internal reference


class DAC8568:
    """ARTIQ Coredevice driver for the TI DAC8568C (16-bit 8-channel DAC).

    This DAC is used on Duke Breakout Board.

    This should also work for DAC7568, DAC8168, but :func:`vout_to_mu` will need
    adjusted for fewer bits of precision.
    """

    kernel_invariants = {
        "_SPI_CONFIG",
        "_spi_clk_div_mu",
        "_SPI_FREQ",  # unused, but here just in case it's used.
        "MIN_DELAY_TIME",
        "MIN_DELAY_TIME_MU",
        "core",
        "spi",
        "load_dac",
        "chip_select",
        "V_OUT_MAX",
    }
    _SPI_CONFIG = (
        0 * spi.SPI_OFFLINE
        | 0 * spi.SPI_CS_POLARITY
        | 0 * spi.SPI_CLK_POLARITY
        | 1 * spi.SPI_CLK_PHASE  # MOSI latched on Falling clk edge (changes on rising)
        | 0 * spi.SPI_LSB_FIRST
        | 0 * spi.SPI_HALF_DUPLEX
        | 1 * spi.SPI_END  # only allow sending one command at once.
    )
    _SPI_FREQ = 20 * MHz  # Max = 50 MHz
    MIN_DELAY_TIME = 80 * ns

    def __init__(
        self,
        device_manager: DeviceManager,
        spi_device: str,
        ldac_trigger: str,
        chip_select: int,
        v_out_max: float = 5.0,
        core_device: str = "core",
    ) -> None:
        """
        Control an 8-output DAC on the Duke Breakout Board.

        Args:
            device_manager (DeviceManager): Device manager to provide & launch devices
            ldac_trigger (str): Trigger line (TTL) device name to load dac
            chip_select (int): SPI chip select setting for this particular DAC
            v_out_max (float, optional): Defaults to 5.0. This is a hardware-set
                value that is DAC8568-variant dependent.
                The C/D variant (used in UMD EURIQA) uses 5.0V, while A/B use 2.5V.
                See Datasheet for more info.
            core_device (str, optional): Defaults to "core". Core device that connects
                to and controls this DAC.
        """
        self.core = device_manager.get(core_device)
        self.spi = device_manager.get(spi_device)
        self.load_dac = device_manager.get(ldac_trigger)
        self.chip_select = chip_select
        self._init_v_ref_internal = True  # default to internal reference
        self._v_ref_internal = self._init_v_ref_internal
        self.V_OUT_MAX = v_out_max  # Max DAC output voltage
        self._spi_clk_div_mu = self.spi.frequency_to_div(self._SPI_FREQ)
        self.sync_mode = True
        self.MIN_DELAY_TIME_MU = self.core.seconds_to_mu(self.MIN_DELAY_TIME)
        self.seed = np.random.rand()

    @kernel
    def _spi_send(self, data: TInt32) -> TNone:
        """Write data to SPI bus."""
        self.spi.set_config_mu(
            self._SPI_CONFIG, 32, self._spi_clk_div_mu, self.chip_select
        )
        self.spi.write(data)

    @kernel
    def init(self, v_ref_internal: TBool = True) -> TNone:
        """Initialize the DAC settings, default to using internal voltage reference."""
        self.settings_sync_mode(sync_mode=True, force_update=True)
        self.settings_internal_reference(static_reference=True, enable=v_ref_internal)

    @kernel
    def set_voltage_mu(
        self,
        channel: TInt32,
        v_out_mu: TInt32,
        update_immediate: TBool = True,
        update_all_channels: TBool = False,
        channel_as_analog_out: TBool = True,
    ) -> TNone:
        """
        Set the output voltage of a DAC channel.

        If `update_immediate` is not set, then you need to manually call
        :meth:`load_dac`.

        Args:
            channel (TInt32): Integer describing DAC channel. Can either be
                one of :class:`DACChannel` (required for ALL channels),
                one of :class:`AOut`,
                or 1-8 denoting :class:`AOut` channels Out1-Out8
                (if channel_as_analog_out == True),
                or 0-7 denoting :class:`DACChannel` channels A-H
                (if channel_as_analog_out == False).
            v_out_mu (TInt32): Desired output voltage (in machine units).
            update_immediate (TBool, optional): Defaults to True. Update the
                output voltage as soon as this command is received.
            update_all_channels (TBool, optional): Defaults to False. Update
                all channels' output, even if only writing to one channel.
            channel_as_analog_out (TBool, optional): Defaults to True.
                If True, interprets the channel number given (1-8) as an Analog
                Output number from the Duke Breakout board v3, selecting
                correct DAC channel.
        """
        if update_immediate and update_all_channels:
            command = DACCommands.write_and_update_all
        elif update_immediate and not update_all_channels:
            command = DACCommands.write_and_update
        else:
            command = DACCommands.write
        if 1 < channel <= 8 and channel_as_analog_out:
            # convert Aout number (1-8) on Duke Breakout to DAC Channel Number (A-H)
            # Basically same conversion at AOut class above.
            bit2 = ((channel - 1) & 0b100) >> 2
            bit1 = ((channel - 1) & 0b010) >> 1
            bit0 = (channel - 1) & 0b001
            channel = ((bit2 ^ bit1) << 2) | ((bit0 | bit2) << 1) | (bit2)
        if 0 < channel <= 7:  # executes if pass a channel number or AOut Num
            address_bits = channel << 20
        else:
            address_bits = channel
        self._set_voltage_mu(command, address_bits, v_out_mu)

    @kernel
    def _set_voltage_mu(
        self, update_type: TInt32, channel: TInt32, v_out_mu: TInt32
    ) -> TNone:
        """Set the output voltage to one that has already been pre-calculated.

        `update_type` must be one of :class:`DACCommands`, either
        `write_and_update_all`, `write_and_update`, or `write`.
        Similarly, `channel` must be one of :class:`DACChannel`.
        """
        self._spi_send(update_type | channel | (v_out_mu << 4))

    @kernel
    def update_outputs(self) -> TNone:
        """Update the DAC outputs with the queued values.

        Useful for synchronizing multiple DAC output changes.
        """
        # falling edge-triggered, >= 80 ns
        if not self.sync_mode:
            # async update to DAC output
            delay(40 * ns)  # from datasheet
            self.load_dac.off()
            delay(self.MIN_DELAY_TIME)
            self.load_dac.on()
        else:
            # otherwise does nothing. check it's off
            self.load_dac.off()

    # *** DAC configuration settings ***
    @kernel
    def settings_internal_reference(
        self,
        static_reference: TBool,
        enable: TBool = False,
        always_on: TBool = True,
        flex2stat: TBool = False,
    ) -> TNone:
        """
        Set internal voltage reference to either always-on (static) or as-needed (flex).

        The DAC defaults on power-on to flexible (only turning on when there is a valid
        output on one of the channels), but this can be set to static to prevent noise/
        timing issues.

        WARNING: do not use both the internal reference on and an external reference
        at the same time.

        Args:
            static_reference (TBool): Whether the device should be set to use a
                static, always-on Reference, or a flexible as-needed reference.
            enable (TBool, optional): Defaults to False. Enable the output from
                the reference source. E.g. if using an external reference,
                can power-down the internal reference.
            always_on (TBool, optional): Defaults to True. FLEXIBLE ONLY. Once an
                output is set, the flexible DAC reference will always remain on
                (without powering down when channels disable), until the next
                power-on reset.
            flex2stat (TBool, optional): Defaults to False. MUST be set when
                switching from flexible reference to static reference.
        """
        if flex2stat and static_reference:
            self._spi_send(
                DACCommands.flexible_internal_reference
                | InternalReferenceCommands.flexible_to_static
            )
        if static_reference:
            data = DACCommands.static_internal_reference
            if enable:
                data |= 1
                self._v_ref_internal = True
            else:
                # data |= 0     # does nothing
                self._v_ref_internal = False
        else:
            data = DACCommands.flexible_internal_reference
            if enable and always_on:
                data |= InternalReferenceCommands.flex_reference_always_on
                self._v_ref_internal = True
            elif enable and not always_on:
                data |= InternalReferenceCommands.flex_reference_on
                self._v_ref_internal = True
            else:
                data |= InternalReferenceCommands.flex_reference_off
                self._v_ref_internal = False
        self._spi_send(data)

    @kernel
    def settings_clear_register(self, mode: TInt32) -> TNone:
        """Set what happens when the DAC `CLR` pin is toggled.

        Can clear the DAC to 0, full-scale, or mid-scale, or just ignore the clear
        pin toggle. Should pass in one of :class:`ClearOptions`.
        """
        self._spi_send(DACCommands.clear | mode)

    @kernel
    def settings_load_dac_sync(self, sync_channels: TList(TInt32)) -> TNone:
        """
        Choose which channels will be updated immediately when written to.

        This acts in parallel with `update_immediate` from :meth:`set_voltage`,
        and operates the specified channels in 'synchronous' mode, i.e. they update
        at the falling edge of the last clock in the SPI command, vs waiting for
        an external LDAC trigger.

        NOTE: THIS OVERWRITES PREVIOUS SETTINGS. If you want to keep your previous
        synchronous loading channels, you must explicitly add them to the argument list.

        Args:
            sync_channels (TList(TInt32)): List of integers denoting channels (0-7).
                Use 0xF to set ALL channels to synchronous.
        """
        one_hot = np.int32(0)
        for chan in sync_channels:
            if chan == 0xF:
                one_hot = 0xFF
                break
            else:
                one_hot |= 1 << chan
        self._spi_send(DACCommands.load_dac_pin_ignore | one_hot)

    @kernel
    def reset(self) -> TNone:
        """Perform software-triggered reset on the DAC.

        THIS WILL CHANGE YOUR SETTINGS ON THE DAC. YOU WILL NEED TO RE-INIT.
        """
        # prevent the DAC channel to automatically initialize when the RF is ramped down
        # self._spi_send(DACCommands.software_reset)
        self.settings_sync_mode(True)

    @kernel
    def settings_power_down(self, channel: TInt32, impedance: TInt32) -> TNone:
        """
        Set individual DAC channels to power-up/down.

        If powering down, set their impedance to ground to: 100k Ohms, 1k, or High-Z.
        One of :class:`PowerDownImpedance`

        Args:
            channel (TInt32): Integer denoting which channel to power-up/down.
                Either 0-7 (A-H), 0xF (ALL), or one of DACChannel.
            impedance (PowerDownImpedance): The setting for either turning on
                or which power-down impedance.
        """
        if channel > 7 and channel != 0xF:
            channel = channel >> 20
            if channel == 0:
                raise DAC8568Exception("Invalid channel number")

        if channel == 0xF:
            power_down_channel = 0xFF
        else:
            power_down_channel = 1 << channel

        self._spi_send(DACCommands.power_down_channel | impedance | power_down_channel)

    @kernel
    def settings_sync_mode(self, sync_mode: TBool, force_update: TBool = False):
        """Set the DAC to synchronous or asynchronous mode.

        Should use Synchronous (True) by default.
        Synchronous means updates occur immediately after writing.
        In Asynchronous mode, updates occur after setting LDAC/Load DAC line to low.
        """
        if sync_mode != self.sync_mode or force_update:
            self.sync_mode = sync_mode
            if sync_mode:
                self.load_dac.off()
            else:
                self.load_dac.on()

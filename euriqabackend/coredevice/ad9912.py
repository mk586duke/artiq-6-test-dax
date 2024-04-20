"""
Core device driver for the AD9912 DDS.

Based on `./ad5360.py`. Written by Jonathan Mizrahi (and others), then Drew Risinger.

This driver works on ARTIQ4, and has been tested (partially, for freq/amp, not phase).

TODO:
    * confirm all phase/freq/amp adjustments work as expected.
    * mimic ARTIQ 4's "urukul" DDS ad9912.py closer (get inspiration)
    * add set non-mu
    * remove set_phase()
"""
import logging

import numpy as np
from artiq.coredevice import spi2 as spi
from artiq.language.core import at_mu
from artiq.language.core import delay_mu
from artiq.language.core import kernel
from artiq.language.core import now_mu
from artiq.language.core import portable
from artiq.language.types import TBool
from artiq.language.types import TFloat
from artiq.language.types import TInt32
from artiq.language.types import TInt64
from artiq.language.types import TNone
from artiq.language.units import us
from artiq.master.worker_db import DeviceManager

_LOGGER = logging.getLogger(__name__)


# ** UTILITY FUNCTIONS **
@portable
def freq_to_mu(frequency_raw: TFloat) -> TInt64:
    """
    Convert raw frequency to value used by DDS to denote frequency.

    Args:
        frequency_raw (float): Frequency in Hertz to set DDS to.

    Returns:
        Int64: DDS "frequency tuning word" equivalent to given frequency. Max of 48 bits

    Examples:
        >>> freq_to_mu(int(100E6))
        28147497671065
        >>> freq_to_mu(int(100.01E6))
        28150312420832
        >>> freq_to_mu(int(200E6))
        56294995342131
        >>> hex(freq_to_mu(int(100E6)))
        '0x199999999999'

    """
    dds_clk_freq_sample = 1000000000.0  # 1 GHz sampling clock
    binary_freq_tune_word_max = 281474976710656.0  # 1 << 48 = 2^48

    # Note: np.int64(round(float())) syntax required to make float -> int conversion work on both host and core
    freq_mu = np.int64(
        round(float(frequency_raw / dds_clk_freq_sample * binary_freq_tune_word_max))
    )
    return freq_mu


@portable
def phase_to_mu(phase_radians: TFloat) -> TInt32:
    """
    Convert the phase in radians to the DDS phase offset.

    Args:
        phase_radians (float): desired phase offset in radians

    Raises:
        ValueError: `phase_radians` must be in range [0, 2pi)

    Returns:
        Int32: Phase in DDS units to set. Max of 14 bits.

    Examples:
        >>> import math
        >>> phase_to_mu(math.pi)
        8192
        >>> phase_to_mu(7 * math.pi / 4)
        14336
        >>> phase_to_mu(-1)
        Traceback (most recent call last):
            ...
        ValueError: Invalid phase: -1 not in [0, 2pi)

    """
    assert phase_radians <= 2 * np.pi, "Invalid phase: not in [0, 2pi)"
    assert phase_radians >= 0, "Invalid phase: not in [0, 2pi)"
    if phase_radians == 2 * np.pi:
        phase_radians = 0.0  # Deal with degeneracy of the edge case
    phase_bit_width = 14

    # Note: np.int32(round(float())) syntax required to make float -> int conversion work on both host and core
    phase_mu = np.int32(
        round(float(phase_radians / (2 * np.pi) * (2 ** phase_bit_width)))
    )
    return phase_mu


class AD9912Error(Exception):
    """Error in the AD9912 DDS."""

    pass


class AD9912:
    """Simple output control of the Analog Devices AD9912 DDS.

    AD9912 DDS is connected over SPI.
    """

    kernel_invariants = set(
        [
            "_SPI_CONFIG",
            "_AD9912_ADDR_FTW",
            "_AD9912_ADDR_PHASE",
            "_AD9912_ADDR_AMPLITUDE",
            "_AD9912_WRITE_ONE_CMD",
            "_AD9912_WRITE_TWO_CMD",
            "_AD9912_WRITE_THREE_CMD",
            "_AD9912_WRITE_STREAM_CMD",
            "_SPI_CLK_DIVIDER",
            "core",
            "bus",
            "io_update",
            "chip_select",
            "safety_delay_mu",
        ]
    )
    # Flags to configure the SPI bus, based on the AD9912 datasheet
    # todo: set flags properly. Not 0's?
    _SPI_CONFIG = (
        0 * spi.SPI_OFFLINE
        | 0 * spi.SPI_CS_POLARITY
        | 0 * spi.SPI_CLK_POLARITY
        | 0 * spi.SPI_CLK_PHASE
        | 0 * spi.SPI_LSB_FIRST
        | 0 * spi.SPI_HALF_DUPLEX
    )

    # From AD9912 datasheet pages 31-32
    # Addresses in memory of the frequency (called ftw), phase, and amplitude
    # Amplitude = DAC output current
    _AD9912_ADDR_FTW = 0x01AB << 16
    _AD9912_ADDR_PHASE = 0x01AD << 16
    _AD9912_ADDR_AMPLITUDE = 0x040C << 16

    # From AD9912 datasheet page 28
    # Commands to write 1, 2, 3 bytes, or a continuous stream.
    _AD9912_WRITE_ONE_CMD = 0b000 << 29
    _AD9912_WRITE_TWO_CMD = 0b001 << 29
    _AD9912_WRITE_THREE_CMD = 0b010 << 29
    _AD9912_WRITE_STREAM_CMD = 0b011 << 29

    _SPI_CLK_DIVIDER = 4  # value to divide RTIO clk by to get SPI clk. Low = fast.
    # SPI CLK approx (base clk ~120 MHz) / divider

    def __init__(
        self,
        device_manager: DeviceManager,
        spi_device: str,
        io_update: str,
        chip_select: int = 0b1,
        core_device: str = "core",
    ):
        """
        Initialize an AD9912 instance without connecting or writing any data.

        Args:
            device_manager (DeviceManager): Set of all available devices.
            spi_device (str): Name of the SPI bus this DDS is on.
            io_update (str): Name of the TTL device that io_update is connected to.
            chip_select (int, optional): Defaults to 0b1.
                Bitmask of values to drive on the chip select lines during transactions.
                If there is only one chip select, this should be set to 1.
            core_device (str, optional): Defaults to "core". Name of the core device
                in device_db
        """
        self.core = device_manager.get(core_device)
        self.bus = device_manager.get(spi_device)  # type: spi.SPIMaster
        self.io_update = device_manager.get(io_update)
        self.chip_select = chip_select
        self.safety_delay_mu = self.core.seconds_to_mu(2.5 * us)

    @kernel
    def _output_spi(
        self, data: TInt32, data_len_bytes: TInt32 = 4, more_data: TBool = False
    ):
        """
        Output data over SPI bus.

        Simplifies output, but less control over SPI chip-select pin.

        Args:
            data (TInt32): Right-aligned binary data to write out over SPI.
                Max of 32 bits.
            data_len_bytes (TInt32): Defaults to 4. Number of bytes of ``data``
                to write out. Defaults to outputting all 32 bits.
            more_data (bool): Whether there is more data that should be grouped
                together (i.e. forces chip select to remain asserted between
                transfers) in this transfer.
        """
        output_bits = data_len_bytes * 8
        # _LOGGER.debug("Writing SPI: %x (%i bytes)", data, data_len_bytes)
        if more_data:
            spi_config = self._SPI_CONFIG
        else:
            spi_config = self._SPI_CONFIG | spi.SPI_END
        self.bus.set_config_mu(
            spi_config, output_bits, self._SPI_CLK_DIVIDER, self.chip_select
        )
        # rtio_log("spi", "dat", data, "len_b", output_bits, "cs#", more_data)
        # _LOGGER.debug("SPI Transfer duration (mu): %i", self.bus.xfer_duration_mu)
        self.bus.write(data)

    @portable
    def maximum_programming_time(self) -> TFloat:
        """Return the maximum wait time (seconds) before new output is ready."""
        return self.core.mu_to_seconds(
            4 * (self.bus.xfer_duration_mu + self.bus.ref_period_mu)  # SPI data time
            + self.bus.ref_period_mu  # io_update time
            + self.safety_delay_mu
        )

    @kernel
    def load(self):
        """Pulse the io_update line, updating the DDS output.

        This method advances the timeline by two SPI clock periods.
        """
        self.io_update.pulse_mu(2 * self.bus.ref_period_mu)

    @kernel
    def set_mu(
        self,
        frequency: TInt64 = -1,
        phase: TInt32 = -1,
        amplitude: TInt32 = -1,
        preset: TBool = False,
    ) -> TNone:
        """
        Write the frequency, phase, and/or amplitude to the DDS.

        This method does not advance the timeline. Write events are scheduled
        in the past. The DDS will synchronously start changing its output `now`.

        Values are set in "machine units", i.e. the units expected by the
        AD9912 chip. Use the :func:`freq_to_mu` and :func:`phase_to_mu`
        functions to convert.

        Args:
            frequency (int, optional): Defaults to -1. Frequency to set (if any)
            phase (int, optional): Defaults to -1. Phase to set (if any)
            amplitude (int, optional): Defaults to -1. Amplitude to set (if any)
            preset (bool, optional): Defaults to False. If True, rewinds the
                time cursor so that all changes here are ready by ``now``.

        Raises:
            AD9912Error: Didn't provide a valid frequency/phase/amplitude.

        """
        # determine values to write
        write_frequency = 1 if frequency >= 0 else 0
        write_phase = 1 if phase >= 0 else 0
        write_amplitude = 1 if amplitude >= 0 else 0

        start_time = now_mu()

        if (not write_frequency) and (not write_phase) and (not write_amplitude):
            # _LOGGER.error(
            #     "Invalid freq/phase/amp: %i, %i, %i", frequency, phase, amplitude
            # )
            raise AD9912Error(
                "AD9912 set called with no valid frequency, phase, or amplitude: "
            )

        if preset:
            # compensate all delays that will be applied
            delay_mu(
                -(
                    (write_phase + write_amplitude + 2 * write_frequency)
                    * (self.bus.xfer_duration_mu + self.bus.ref_period_mu)
                    + self.safety_delay_mu
                )
            )

        if write_frequency:
            # rtio_log("dds", "wrtfreq")
            freqFirst16 = np.int32(0xFFFF & (frequency >> 32))
            freqLast32 = np.int32(frequency)

            self._output_spi(
                self._AD9912_WRITE_STREAM_CMD | self._AD9912_ADDR_FTW | freqFirst16,
                more_data=True,
            )
            # TODO: (optional) Rewind time cursor from setting SPI config
            # delay_mu(-self.bus.xfer_duration_mu)
            self._output_spi(freqLast32)
            # delay_mu(self.bus.xfer_duration_mu)
        if write_phase:
            # rtio_log("dds", "wrtphs")
            phase = phase & 0x3FFF  # phase is 14 bits
            self._output_spi(
                self._AD9912_WRITE_TWO_CMD | self._AD9912_ADDR_PHASE | phase
            )
        if write_amplitude:
            # rtio_log("dds", "wrtamp")
            amplitude = amplitude & 0x3FF  # amplitude is 10 bits
            self._output_spi(
                self._AD9912_WRITE_TWO_CMD | self._AD9912_ADDR_AMPLITUDE | amplitude
            )
        # else:
        # _LOGGER.debug("SPI programming time (mu): %i", now_mu() - start_time)

        if preset:
            end_time = now_mu()
            assert end_time < start_time
            at_mu(start_time)  # reset to start time
            assert now_mu() == start_time

        if write_frequency or write_phase:
            if preset:
                delay_mu(-2 * self.bus.ref_period_mu)
            self.load()

    # Development code
    @kernel
    def phase_mu(self, phase: TInt32):
        """Write the phase to the DDS. Optimized for kernel processing time.

        This method does not advance the timeline. Write events are scheduled
        in the past. The DDS will synchronously start changing its output `now`

        Args:
            phase(int): Phase to write.
        """
        # Start programming DDS so that it is ready to change at the current cursor time
        # delay_mu(
        #     -(
        #         self.bus.xfer_duration_mu
        #         + self.bus.ref_period_mu
        #         + self.safety_delay
        #     )
        # )
        phase = phase & 0x3FFF  # phase is 14 bits
        self._output_spi(self._AD9912_WRITE_TWO_CMD | self._AD9912_ADDR_PHASE | phase)
        # delay_mu(self.safety_delay)
        self.load()
        # delay_mu(-2 * self.bus.ref_period_mu)  # undo load() delay
        # If all is OK, should result in no change in time cursor.

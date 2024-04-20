#!/usr/bin/env python3
"""Build ARTIQ for EURIQA's hardware based on the Xilinx KC705 FPGA.

Uses hardware (DAC/ADC/GPIO/etc) built by Duke. Located in blue "pulser" box.
"""
import argparse
import itertools
import logging

import misoc.integration.builder as misoc_builder
import misoc.targets.kc705 as misoc_kc705
from artiq.build_soc import build_artiq_soc
from artiq.gateware import rtio
from artiq.gateware.rtio.phy import spi2
from artiq.gateware.rtio.phy import ttl_serdes_7series
from artiq.gateware.rtio.phy import ttl_simple
from artiq.gateware.targets.kc705 import _StandaloneBase

from . import euriqa

_LOGGER = logging.getLogger(__name__)


class EURIQA(_StandaloneBase):
    """EURIQA pulser setup."""

    def __init__(self, **kwargs):
        """Declare hardware available on Euriqa's KC705 & Duke Breakout."""
        add_sandia_dac_spi = kwargs.pop("sandia_dac_spi", False)
        _StandaloneBase.__init__(self, **kwargs)
        unused_count = itertools.count()

        platform = self.platform
        platform.add_extension(euriqa.fmc_adapter_io)
        if add_sandia_dac_spi:
            # segment to prevent accidentally adding x100 DAC comm/pins
            platform.add_extension(euriqa.x100_dac_spi)

        rtio_channels = list()

        # Output GPIO/TTL Banks
        for bank, i in itertools.product(["out1", "out2", "out3", "out4"], range(8)):
            # out1-1, out1-2, ..., out4-7
            if add_sandia_dac_spi and bank == "out2" and i == 7:
                # add unused dummy channel. to keep channel #s same.
                # Won't output to useful digital line
                phy = ttl_serdes_7series.Output_8X(
                    platform.request("unused", next(unused_count))
                )
            else:
                phy = ttl_serdes_7series.Output_8X(platform.request(bank, i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending output GPIO chan: %i", len(rtio_channels))

        # Input GPIO/TTL Banks
        for bank, i in itertools.product(["in1", "in2", "in3"], range(8)):
            # in1-1, in1-2, ..., in3-7
            phy = ttl_serdes_7series.InOut_8X(platform.request(bank, i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))
        _LOGGER.debug("Ending input GPIO chan: %i", len(rtio_channels))

        # User-controlled LED's? Not in device_db
        for i in range(2, 4):
            phy = ttl_simple.Output(platform.request("user_led", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending output LED chan: %i", len(rtio_channels))

        # Tri-state buffer to disable the TTL/GPIO outputs (out1, ...)
        phy = ttl_simple.Output(platform.request("oeb", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("OEB GPIO chan: %i", len(rtio_channels))

        # TODO: figure out usage/what this is
        for i in range(9):
            phy = ttl_serdes_7series.Output_8X(platform.request("sma", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending sma chan: %i", len(rtio_channels))

        # TODO: update name for io_update everywhere
        # Update triggers for DDS. Edge will trigger output settings update
        for i in range(10):
            phy = ttl_serdes_7series.Output_8X(platform.request("io_update", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending io_update GPIO chan: %i", len(rtio_channels))

        # Reset lines for the DDS boards.
        for i in range(5):
            phy = ttl_serdes_7series.Output_8X(platform.request("reset", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending DDS Reset GPIO chan: %i", len(rtio_channels))

        # SPI interfaces to control the DDS board outputs
        for i in range(5):
            spi_bus = self.platform.request("spi", i)
            phy = spi2.SPIMaster(spi_bus)
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))
            odd_channel_sdio = platform.request("odd_channel_sdio", i)
            self.comb += odd_channel_sdio.eq(spi_bus.mosi)
        _LOGGER.debug("Ending SPI chan: %i", len(rtio_channels))

        # SPI & Load DAC (LDAC) pins for Controlling 8x DAC (DAC 8568)
        spi_bus = self.platform.request("spi", 5)
        phy = spi2.SPIMaster(spi_bus)
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))
        _LOGGER.debug("DAC8568 SPI RTIO channel: %i", len(rtio_channels))
        phy = ttl_simple.Output(platform.request("ldac", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("DAC8568 LDAC GPIO channel: %i", len(rtio_channels))

        # SPI for Coredevice serial comm to Sandia DAC
        if add_sandia_dac_spi:
            print("Adding SPI for Sandia DAC comms")
            spi_bus = self.platform.request("spi", 6)
            phy = spi2.SPIMaster(spi_bus)
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))

        self.config["HAS_RTIO_LOG"] = None
        self.config["RTIO_LOG_CHANNEL"] = len(rtio_channels)
        _LOGGER.debug("RTIO log chan: %i", len(rtio_channels))
        rtio_channels.append(rtio.LogChannel())

        _LOGGER.debug("Euriqa KC705 RTIO channels: %s", list(enumerate(rtio_channels)))
        self.add_rtio(rtio_channels)


class EuriqaSandiaDAC(EURIQA):
    """A variant of the EURIQA SOC for controlling ONE Sandia 100x DAC.

    Replaces one of the GPIO/TTL Out lines with an SPI data line to output the
    SPI-like timed signal expected by the real-time-triggering Sandia 100x DAC.
    """

    def __init__(self, **kwargs):
        """Create a EURIQA variant with hardware support for Sandia 100x DAC."""
        kwargs["sandia_dac_spi"] = True
        super().__init__(**kwargs)


VARIANTS = {cls.__name__.lower(): cls for cls in [EURIQA, EuriqaSandiaDAC]}


def get_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for kc705 gateware builder."""
    parser = argparse.ArgumentParser(
        description="KC705 gateware and firmware builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    misoc_builder.builder_args(parser)
    misoc_kc705.soc_kc705_args(parser)
    parser.add_argument(
        "-V",
        "--variant",
        choices=VARIANTS.keys(),
        default="euriqasandiadac",
        help="variant: %(choices)s (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase logging verbosity level (default=WARNING)",
    )
    return parser


def main() -> None:
    """Build gateware for specified KC705 FPGA variant."""
    args = get_argument_parser().parse_args()
    logging.basicConfig(level=logging.WARNING - args.verbosity)

    variant = args.variant.lower()
    try:
        cls = VARIANTS[variant]
    except KeyError:
        raise SystemExit("Invalid variant (-V/--variant)")

    soc = cls(**misoc_kc705.soc_kc705_argdict(args))
    build_artiq_soc(soc, misoc_builder.builder_argdict(args))
    # NOTE: if you get a XILINX license error,
    #   check you have the proper license in ~/.Xilinx/


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ARTIQ controller server to command the Harris Multichannel AOM Amplifier."""
import argparse

import sipyco.common_args as artiq_args
from sipyco.pc_rpc import simple_server_loop

from ..devices.harris_aom.driver import HarrisMultichannelAOM


def get_argparser():
    """Generate command-line arguments for Harris Multichannel AOM server."""
    parser = argparse.ArgumentParser(
        description="ARTIQ controller for Harris Multichannel AOM Amplifier over TCP/IP"
    )
    artiq_args.simple_network_args(parser, 3273)
    parser.add_argument(
        "-d", "--devport", default=2101, help="The port number of the device."
    )
    parser.add_argument(
        "-a", "--address", default="192.168.1.160", help="The address of the device."
    )
    parser.add_argument(
        "-s",
        "--simulation",
        action="store_true",
        help="Put the driver in simulation mode.",
    )
    artiq_args.verbosity_args(parser)
    return parser


def main():
    """Run the Harris Multichannel AOM server."""
    args = get_argparser().parse_args()
    artiq_args.init_logger_from_args(args)
    if args.simulation:
        aom = HarrisMultichannelAOM(simulation=True)
    else:
        aom = HarrisMultichannelAOM(
            port=args.devport, ipaddr=args.address, simulation=False
        )
    try:
        simple_server_loop(
            {"MultichannelAOM": aom}, artiq_args.bind_address_from_args(args), args.port
        )
    finally:
        aom.close()


if __name__ == "__main__":
    main()

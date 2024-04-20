#!/usr/bin/env python3
"""ARTIQ controller for the Harris Global AOM amplifier."""
import argparse

import sipyco.common_args as artiq_args
from sipyco.pc_rpc import simple_server_loop

from ..devices.harris_aom.driver import HarrisGlobalAOM


def get_argparser():
    """Generate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ARTIQ controller for Harris Global AOM Amplifier over TCP/IP"
    )
    artiq_args.simple_network_args(parser, 3272)
    parser.add_argument(
        "-a", "--address", default="192.168.1.162", help="The address of the device."
    )
    parser.add_argument(
        "-d", "--devport", default=2101, help="The port number of the device."
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
    """Run a server to connect to the Harris Global AOM Amplifier."""
    args = get_argparser().parse_args()
    artiq_args.init_logger_from_args(args)
    if args.simulation:
        aom = HarrisGlobalAOM(simulation=True)
    else:
        aom = HarrisGlobalAOM(port=args.devport, ipaddr=args.address, simulation=False)
    try:
        simple_server_loop(
            {"GlobalAOM": aom}, artiq_args.bind_address_from_args(args), args.port
        )
    finally:
        aom.close()


if __name__ == "__main__":
    main()

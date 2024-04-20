#!/usr/bin/env python3
"""ARTIQ server for the Conex linear positioning devices."""
import argparse

import sipyco.common_args as artiq_args
from sipyco.pc_rpc import simple_server_loop

from ..devices.conex.driver import ConexBox


def get_argparser():
    """Generate command-line arguments for conex controller."""
    parser = argparse.ArgumentParser(description="ARTIQ controller for the Conex Boxes")
    parser.add_argument(
        "-comx",
        "-x",
        help="'COMi' where i designates the serial port of the x-axis Conex Box",
    )
    parser.add_argument(
        "-comy",
        "-y",
        help="'COMj' where j designates the serial port of the y-axis Conex Box",
    )
    parser.add_argument(
        "--simulation",
        "-s",
        action="store_true",
        help="True puts the device in simulation mode",
    )
    artiq_args.simple_network_args(parser, 3274)
    artiq_args.verbosity_args(parser)
    return parser


def main():
    """Run the conex controller ARTIQ server."""
    args = get_argparser().parse_args()
    artiq_args.init_logger_from_args(args)

    if args.simulation:
        box = ConexBox(args.comx, args.comy, simulation=True)
    else:
        box = ConexBox(args.comx, args.comy, simulation=False)
    try:
        simple_server_loop(
            {"ConexBox": box}, artiq_args.bind_address_from_args(args), args.port
        )
    finally:
        box.close()


if __name__ == "__main__":
    main()

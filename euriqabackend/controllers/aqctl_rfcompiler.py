#!/usr/bin/env python3
"""ARTIQ Controller to control the Keysight M3202A AWG.

This calls the RF Compiler program, which compiles circuit elements into RF waveforms.
"""
import argparse
import logging

import sipyco.common_args as artiq_args
from sipyco.pc_rpc import simple_server_loop

from euriqabackend.devices.keysight_awg import RFCompiler

_LOGGER = logging.getLogger(__name__)


def get_argparser() -> argparse.ArgumentParser:
    """
    Return command line arguments used in this driver.

    Basic network arguments, simulation, and setup.
    """
    parser = argparse.ArgumentParser(
        description="ARTIQ Controller for Keysight M3202A AWG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    artiq_args.simple_network_args(parser, 3277)
    artiq_args.verbosity_args(parser)

    # awg_settings = parser.add_argument_group("AWG Settings")

    return parser


def main() -> None:
    """
    Launch driver to interface ARTIQ with the RF Compiler.

    The RF Compiler controls a Keysight M3202A AWG.
    Exposes the AWG to network control. Simulation not enabled yet.
    """
    args = get_argparser().parse_args()
    artiq_args.init_logger_from_args(args)

    # Start Compiler
    try:
        rfc = RFCompiler.RFCompiler()
    except Exception as err:
        raise ValueError(
            "Could not initialize RF Compiler. "
            "Please provide valid command-line arguments"
        ) from err

    # Open RPC server
    _LOGGER.info(
        "Starting RF Compiler. "
        "Use another tool like `artiq_rpctool` to control this device."
    )
    print("Starting RF Compiler")

    try:
        simple_server_loop(
            {"RF Controller": rfc}, artiq_args.bind_address_from_args(args), args.port
        )
    except OSError as err:
        raise OSError(
            "Network port is probably already in use. Check if you "
            "have any other drivers running on port {}.".format(args.port)
        ) from err


if __name__ == "__main__":
    main()

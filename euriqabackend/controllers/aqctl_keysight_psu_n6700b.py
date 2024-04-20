#!/usr/bin/env python3
"""ARTIQ Controller to control the Ion's Magnetic Field via a power supply."""
import argparse
import logging
import socket

import sipyco.common_args as artiq_args
from sipyco.pc_rpc import simple_server_loop

from euriqabackend.devices.keysight_psu.n6700b import N6700bPowerSupply

_LOGGER = logging.getLogger(__name__)


def get_argparser() -> argparse.ArgumentParser:
    """
    Return command line arguments used in this driver.

    Basic network arguments, simulation, and setup.
    """
    parser = argparse.ArgumentParser(
        description="ARTIQ Controller for the Ion Magnetic field, "
        "using the Keysight N6700b.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    artiq_args.simple_network_args(parser, 3280)
    artiq_args.verbosity_args(parser)

    psu_settings = parser.add_argument_group("PSU Settings")
    psu_settings.add_argument(
        "--psu-ip",
        "-i",
        required=True,
        type=str,
        help="IP Address for Keysight N6700b",
    )
    psu_settings.add_argument(
        "--psu-port",
        "-pp",
        default=5025,
        help="Port for TCP control of Keysight N6700b",
    )
    psu_settings.add_argument(
        "--channel",
        "-c",
        required=True,
        type=int,
        help="Channel/slot in the Keysight N6700b chassis for the PSU control.",
    )
    psu_settings.add_argument(
        "--watchdog",
        "-w",
        type=int,
        metavar="TIME_MS",
        default=None,
        help="Enable watchdog timer on ALL channels (not just channel you are using). "
        "If connection lost to device, ALL CHANNELS will shut off."
        "Defaults to no watchdog (watchdog could be enabled by other controllers)",
    )
    return parser


def main() -> None:
    """
    Launch driver for interfacing ARTIQ with Keysight N6700b.

    Used for controlling the magnetic (B) field of the ion, i.e. polarization.
    """
    args = get_argparser().parse_args()
    artiq_args.init_logger_from_args(args)
    try:
        if args.simulate:
            raise NotImplementedError("Simulation not yet implemented.")
    except AttributeError:
        pass
    # Start PSU
    try:
        psu = N6700bPowerSupply(
            ip_address=args.psu_ip, port=args.psu_port, instrument_channel=args.channel,
        )
    except Exception as err:
        raise ValueError(
            "Could not initialize PSU. Please provide valid command-line arguments"
        ) from err

    # Open RPC server
    _LOGGER.info(
        "Starting PSU (%s:%i (#%i)) at IP: `%s:%i.` "
        "Use another tool like `artiq_rpctool` to control this device",
        args.psu_ip,
        args.psu_port,
        args.channel,
        socket.gethostbyname(socket.gethostname()),
        args.port,
    )
    print(
        "Starting PSU '{}:{}' (#{}) at IP: {}:{}.".format(
            args.psu_ip,
            args.psu_port,
            args.channel,
            socket.gethostbyname(socket.gethostname()),
            args.port,
        )
    )

    try:
        simple_server_loop(
            {"PSU": psu}, artiq_args.bind_address_from_args(args), args.port
        )
    except OSError as err:
        raise OSError(
            "Network port is probably already in use. Check if you "
            "have any other drivers running on port {}.".format(args.port)
        ) from err
    finally:
        # does weird logic internally, but this is correct
        psu._close_connection()
        print(
            "Shut down controller. Closed connection to PSU #{}.".format(args.channel)
        )


if __name__ == "__main__":
    main()

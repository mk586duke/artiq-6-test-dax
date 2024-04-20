#!/usr/bin/env python3
"""
ARTIQ Controller to control an Ytterbium/Barium oven via a power supply.

Oven basically acts as light bulb, and by heating a filament via a power supply,
some amount of neutral atoms will be expelled. This is mostly a current-controlled
device.

**BE CAREFUL.** This has some safeguards built in, but you could still ruin
your experiment by heating the oven too much and dispelling all your atoms,
thus requiring you to break vacuum and cost lots of time/money.

Safe operating point determined to be ~~ 2V, 2A.
"""
import argparse
import logging
import socket

import sipyco.common_args as artiq_args
from sipyco.pc_rpc import simple_server_loop

from euriqabackend.devices.keysight_psu.n6700b import OvenPowerSupply

_LOGGER = logging.getLogger(__name__)


def get_argparser() -> argparse.ArgumentParser:
    """
    Return command line arguments used in this driver.

    Basic network arguments, simulation, and setup.
    """
    parser = argparse.ArgumentParser(
        description="ARTIQ Controller for the Yb or Ba ovens, "
        "using the Keysight N6700b.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    artiq_args.simple_network_args(parser, 3276)
    artiq_args.verbosity_args(parser)

    oven_settings = parser.add_argument_group("Oven Settings")
    oven_settings.add_argument(
        "--oven-ip",
        "-i",
        required=True,
        type=str,
        help="IP Address for Oven/Keysight N6700b",
    )
    oven_settings.add_argument(
        "--oven-port",
        "-op",
        default=5025,
        help="Port for TCP control of Keysight N6700b",
    )
    oven_settings.add_argument(
        "--channel",
        "-c",
        required=True,
        type=int,
        help="Channel/slot in the Keysight N6700b chassis for the oven control.",
    )
    oven_settings.add_argument(
        "--watchdog",
        "-w",
        type=int,
        metavar="TIME_MS",
        default=None,
        help="Enable watchdog timer on ALL channels (not just channel you are using). "
        "If connection lost to device, ALL CHANNELS will shut off.",
    )
    oven_settings.add_argument(
        "--no-turnoff-on-shutdown",
        action="store_false",
        help="Turn off the output when controller is shut down. (Default to turnoff)",
    )
    # oven_settings.add_argument(
    #     "--simulate",
    #     "-s",
    #     "--sim",
    #     action="store_true",
    #     help="Simulate connecting to the Oven. Just dump received data",
    # )

    return parser


def main() -> None:
    """
    Launch driver for interfacing ARTIQ with Oven (Keysight N6700b).

    Exposes the Oven to network control. Simulation not enabled yet
    """
    args = get_argparser().parse_args()
    artiq_args.init_logger_from_args(args)
    try:
        if args.simulate:
            raise NotImplementedError("Simulation not yet implemented.")
    except AttributeError:
        pass
    # Start Oven
    try:
        oven = OvenPowerSupply(
            ip_address=args.oven_ip,
            port=args.oven_port,
            instrument_channel=args.channel,
        )
    except Exception as err:
        raise ValueError(
            "Could not initialize Oven. Please provide valid command-line arguments"
        ) from err

    # Open RPC server
    _LOGGER.info(
        "Starting Oven (%s:%i (#%i)) at IP: `%s:%i.` "
        "Use another tool like `artiq_rpctool` to control this device",
        args.oven_ip,
        args.oven_port,
        args.channel,
        socket.gethostbyname(socket.gethostname()),
        args.port,
    )
    print(
        "Starting Oven '{}:{}' (#{}) at IP: {}:{}.".format(
            args.oven_ip,
            args.oven_port,
            args.channel,
            socket.gethostbyname(socket.gethostname()),
            args.port,
        )
    )

    try:
        simple_server_loop(
            {"Oven": oven}, artiq_args.bind_address_from_args(args), args.port
        )
    except OSError as err:
        raise OSError(
            "Network port is probably already in use. Check if you "
            "have any other drivers running on port {}.".format(args.port)
        ) from err
    finally:
        # does weird logic internally, but this is correct
        turnoff = args.no_turnoff_on_shutdown
        oven.disconnect(turnoff)
        print(
            "Shut down controller. Closed connection to Oven #{}.".format(args.channel)
        )


if __name__ == "__main__":
    main()

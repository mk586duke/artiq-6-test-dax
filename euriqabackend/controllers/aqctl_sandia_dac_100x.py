#!/usr/bin/env python3

# Copyright 2020 Drew Risinger, Chris Monroe Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Controller for communicating with a

Shuttling solutions. Has some global tweaks. PAD & _PAD_shuttling.xml are used directly,
non-pad are from Sandia direct.
`smb://euriqa-nas/lab/CompactTrappedIonModule/Voltage%20Solutions/Translated%20Solutions`

Asynchronous: changes shuttle solution (mostly line in single table) &
    global tweaks over USB, non-deterministic timing

Synchronous shuttling uses 3 logic lines sending serial & digital handshakes.
This is used for partial readout and autoloading.

"""
import argparse
import logging
import socket

import sipyco.common_args as artiq_args
from sipyco.pc_rpc import simple_server_loop

from euriqabackend.devices.sandia_dac.driver import SandiaDACDriverLow

_LOGGER = logging.getLogger(__name__)


def get_argparser() -> argparse.ArgumentParser:
    """
    Return command line arguments used in this driver.

    Basic network arguments, simulation, specifying FPGA name & bitfile.
    """
    parser = argparse.ArgumentParser(
        description="ARTIQ Controller for the Sandia 100-channel DAC. "
        "Allows the DAC to accept remote commands, to move the ions around a chip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    artiq_args.simple_network_args(parser, 3270)
    artiq_args.verbosity_args(parser)

    parser.add_argument(
        "-f",
        "--fpga",
        required=True,
        help="FPGA name (find in OpalKelly FrontPanel). (example: 'DAC box')",
    )
    parser.add_argument(
        "-b",
        "--bitfile",
        type=str,
        default=None,
        help="Path to DAC FPGA bitfile (*.bit) on host device. "
        "If specified, will reload bitfile. "
        "Reloading turns DAC output off, could lose ion.",
    )
    parser.add_argument(
        "--simulate",
        "-s",
        "--sim",
        action="store_true",
        help="simulate connecting to the DAC. Just dump received data",
    )

    return parser


def main() -> None:
    """
    Launch driver for interfacing ARTIQ with Sandia DAC x100.

    Exposes the DAC to network control. Should be auto-started. Allows simulation.
    """
    args = get_argparser().parse_args()
    artiq_args.init_logger_from_args(args)

    # Start DAC
    try:
        dac = SandiaDACDriverLow(
            fpga=args.fpga, bitfile=args.bitfile, simulation=args.simulate
        )
    except (FileExistsError, ValueError) as err:
        raise ValueError(
            "Could not initialize DAC. Please provide valid command-line arguments"
        ) from err

    # Open RPC server
    _LOGGER.info(
        "Starting Sandia DAC '%s' at IP: %s:%i. "
        "Use another tool like `artiq_rpctool` to control this device",
        args.fpga,
        socket.gethostbyname(socket.gethostname()),
        args.port,
    )
    print(
        "Starting DAC '{}' at IP: {}:{}.".format(
            args.fpga, socket.gethostbyname(socket.gethostname()), args.port
        )
    )
    try:
        simple_server_loop(
            {"SandiaDAC": dac}, artiq_args.bind_address_from_args(args), args.port
        )
    except OSError as err:
        raise OSError(
            "Network port is probably already in use. Check if you "
            "have any other drivers running on port {}.".format(args.port)
        ) from err
    finally:
        dac.close()
        print("Shut down controller. Closed connection to DAC {}.".format(args.fpga))


if __name__ == "__main__":
    main()

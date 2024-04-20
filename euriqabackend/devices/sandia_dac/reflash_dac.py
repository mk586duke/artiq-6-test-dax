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

"""Utility to reflash the Sandia 100x DAC when it is not responding.

Ideally would figure out what is causing the Sandia 100x DAC to hang,
but haven't spent the time.
"""
import logging
import subprocess

import click

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument(
    "dac_bitfile", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option("--serial-number", "-s", type=int, default=None, help="Serial")
@click.option(
    "-v", "--verbosity", count=True, help="Logging level (default to Warn)", default=0
)
def cli(dac_bitfile: str, serial_number: int, verbosity: int) -> None:
    """Execute a command to reflash the Sandia DAC when it stops responding."""
    logging.basicConfig(level=logging.WARNING - 10 * verbosity)
    flash_cmd = ["FrontPanel.exe", "--load-bitfile={}".format(dac_bitfile)]

    if serial_number is None:
        try:
            click.confirm(
                "No serial number provided.\n"
                "Do you want to flash the first FPGA OpalKelly finds?",
                abort=True,
                default=False,  # in non-interactive mode, abort
            )
        except click.Abort as err:
            print(
                "FYI: Open the OpalKelly FrontPanel program to find the serial number."
            )
            raise err
    else:
        flash_cmd.insert(1, "--device-serial={}".format(serial_number))

    _LOGGER.debug("Passing control to OpalKelly FrontPanel to flash the bitfile")
    _LOGGER.debug("Command passed: %s", flash_cmd)
    try:
        subprocess.run(
            flash_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as err:
        if "FrontPanel.exe" in str(err):
            raise FileNotFoundError(
                "OpalKelly FrontPanel is not on your path. Check that it is installed"
            ) from err
        else:
            raise err
    else:
        print("Flashing DAC succeeded.")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()

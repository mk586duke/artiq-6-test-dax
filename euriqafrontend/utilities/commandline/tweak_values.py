"""Get the XX gate calibrations applied to a set of ions."""
import click
import prettytable
import sipyco.pc_rpc as rpc

import euriqabackend.devices.keysight_awg.common_types as rf_common
from euriqabackend.databases.device_db_main_box import device_db


# TODO: change default of "num_ions" in future, this is just quick solution.
@click.command()
@click.argument("ion1")  # String for setting global tweaks, int for ion number
@click.argument("ion2", type=int)
@click.option(
    "--num-ions-in-chain",
    "--ions",
    "-n",
    type=int,
    default=15,
    show_default=True,
    help="Number of ions in the chain (for ion->slot conversion)",
)
@click.option(
    "--rfcompiler-ip",
    "--ip",
    "-i",
    type=str,
    default=device_db["rf_compiler"]["host"],
    show_default=True,
    help="IP Address of the RF Compiler PC (ARTIQ controller)",
)
@click.option(
    "--rfcompiler-port",
    "--port",
    "-p",
    type=int,
    default=device_db["rf_compiler"]["port"],
    show_default=True,
    help="IP port where the RF Compiler ARTIQ server is",
)
@click.option(
    "--set-tweak",
    "-s",
    nargs=2,
    type=(str, float),
    default=(None, 0.0),
    help="Set specified tweak value. Example: `-s [TWEAK NAME] [VALUE]`",
)
def cli(
    num_ions_in_chain: int,
    rfcompiler_ip: str,
    rfcompiler_port: int,
    ion1: str,
    ion2: int,
    set_tweak: (str, float),
):
    """Retrieve the gate tweaks applied to a specific ion pair.

    Simple access & print CLI tool, connects to RF Compiler ARTIQ server,
    retrieves the individual & global tweaks for a set of ions, and then
    calculates the sum (total tweak) applied to the pair, and prints.
    """

    rfcompiler = rpc.Client(rfcompiler_ip, rfcompiler_port)

    if ion1.lower() == "global":
        slot_pair = "global"
    else:
        # Setup & conversion
        ion1 = int(ion1)
        ion1, ion2 = sorted((ion1, ion2))
        slot_pair = (
            rf_common.ion_to_slot(ion1, num_ions_in_chain),
            rf_common.ion_to_slot(ion2, num_ions_in_chain),
        )

    # If setting tweak values, send them to RFCompiler
    if set_tweak[0] is not None:
        tweak = dict()
        tweak[set_tweak[0]] = set_tweak[1]
        rfcompiler.set_XX_tweak(slot_pair, **tweak)

    # Pull Data
    tweaks = dict()
    tweaks[slot_pair] = rfcompiler.get_XX_tweak(slot_pair)
    # tweaks[slot_pair]["ions"] = (ion1, ion2)
    tweaks["global"] = rfcompiler.get_XX_tweak("global")

    # Don't show some of the less useful tweaks
    omit = [
        "XX_amplitudes",
        "XX_detuning",
        "XX_duration_us",
        "XX_gate_type",
        "XX_sign",
        "individual_amplitude_imbalance",
    ]

    # Print Data
    table = prettytable.PrettyTable()
    table.field_names = [
        "Tweak Name",
        "Individual Tweaks",
        "Global Tweaks",
        "(Ind + Glob)",
    ]
    table.sortby = "Tweak Name"
    for k in sorted(set(tweaks["global"].keys()) | set(tweaks[slot_pair].keys())):
        ind_tweak = tweaks[slot_pair][k]
        glob_tweak = tweaks["global"][k]
        if k not in omit:
            table.add_row([k, ind_tweak, glob_tweak, ind_tweak + glob_tweak])

    print(table)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter

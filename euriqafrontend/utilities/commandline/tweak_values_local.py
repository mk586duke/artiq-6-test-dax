"""Get the XX gate calibrations applied to a set of ions."""
import pathlib
import typing

import click
from euriqabackend.devices.keysight_awg.gate_parameters import GateCalibrations
import prettytable

import euriqabackend.devices.keysight_awg.common_types as rf_common


# TODO: change default of "num_ions" in future, this is just quick solution.
@click.command()
@click.argument("gate_tweak_path", type=str)
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
    "--set-tweak",
    "-s",
    nargs=2,
    type=(str, float),
    default=(None, 0.0),
    help="Set specified tweak value. Example: `-s [TWEAK NAME] [VALUE]`",
)
def cli(
    num_ions_in_chain: int,
    gate_tweak_path: str,
    ion1: str,
    ion2: int,
    set_tweak: typing.Tuple[str, float],
):
    """Retrieve the gate tweaks applied to a specific ion pair.

    Simple access & print CLI tool. Loads the gate calibrations (tweaks),
    retrieves the individual & global tweaks for a set of ions, and then
    calculates the sum (total tweak) applied to the pair, and prints.
    """
    gate_tweak_file_path = pathlib.Path(gate_tweak_path)
    assert gate_tweak_file_path.is_file()
    gate_tweaks_struct = GateCalibrations.from_h5(gate_tweak_path)

    if ion1.lower() == "global":
        slot_pair = GateCalibrations.GLOBAL_CALIBRATION_SLOT
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
        set_tweak(
            gate_tweaks_struct,
            tweak_name=set_tweak[0],
            tweak_value=set_tweak[1],
            index=slot_pair,
        )

    # Pull Data
    tweaks_to_print = dict()
    tweaks_to_print[slot_pair] = get_all_tweaks(gate_tweaks_struct, slot_pair)
    # tweaks[slot_pair]["ions"] = (ion1, ion2)
    tweaks_to_print["global"] = get_all_tweaks(
        gate_tweaks_struct, GateCalibrations.GLOBAL_CALIBRATION_SLOT
    )

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
    for k in sorted(
        set(tweaks_to_print["global"].keys()) | set(tweaks_to_print[slot_pair].keys())
    ):
        ind_tweak = tweaks_to_print[slot_pair][k]
        glob_tweak = tweaks_to_print["global"][k]
        if k not in omit:
            table.add_row([k, ind_tweak, glob_tweak, ind_tweak + glob_tweak])

    print(table)

    # save the updated tweaks back to disk for later use
    gate_tweaks_struct.to_h5(gate_tweak_file_path)


def set_tweak(
    tweak_struct: GateCalibrations,
    tweak_name: str,
    tweak_value: typing.Any,
    index: typing.Optional[typing.Tuple[int, int]] = None,
) -> None:
    if index is None:
        index = slice()
    tweak_struct.loc[index, tweak_name] = tweak_value


def get_all_tweaks(
    tweak_struct: GateCalibrations, index: typing.Tuple[int, int]
) -> typing.Dict[str, typing.Any]:
    return tweak_struct.loc[index, :].to_dict()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter

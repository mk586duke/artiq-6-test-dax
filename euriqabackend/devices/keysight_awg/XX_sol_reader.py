"""This code generally handles reading the XX gate solution parameters from file.

Notice that this is the only place in the Keysight code where we index by ion rather
that slot.  This follows from how the solution solver indexes qubits, as specific ions
in an N-ion chain, not as ions located in specific slots.
"""
import collections
import itertools
import logging
import pathlib
import pickle
import typing

import euriqabackend.utilities.hashing as eur_hash
from . import common_types


_LOGGER = logging.getLogger(__name__)


def _XX_sol_filepath(
    XX_sols_dir: typing.Union[str, pathlib.Path],
    N_ions: int,
    sol_name: str,
    ions: typing.Tuple[int, int],
) -> pathlib.Path:
    """Construct a filepath for an XX gate solution file.

    The directory structure will have a top-level folder, followed by folders
    for different ion numbers, followed by folders for specifically named solutions,
    followed by an array of files for each ion combination.

    Args:
        XX_sols_dir: The top-level directory for XX gate solutions
        N_ions: The total number of ions in the chain
        sol_name: The name given to a specific solution
        ions: The two ions being addressed. Order doesn't matter.

    Returns:
        file path (pathlib.Path)

    Examples:
        >>> str(_XX_sol_filepath("/", 15, "XX_GateSolutions", (8,9)))
        '/15-ion solutions/XX_GateSolutions/08-09.sol'
        >>> str(_XX_sol_filepath("/", 15, "XX_GateSolutions", (9, 8)))
        '/15-ion solutions/XX_GateSolutions/08-09.sol'

    """
    assert len(ions) == 2 and min(ions) != max(ions), "Invalid ion parameters passed"
    XX_sols_dir = pathlib.Path(XX_sols_dir)
    num_specific_dir = "{:d}-ion solutions".format(N_ions)
    filename = "{:02d}-{:02d}.sol".format(*sorted(ions))
    solution_dir = XX_sols_dir / num_specific_dir / sol_name / filename
    return solution_dir


def write_XX_params(
    XX_sols_dir: str,
    N_ions: int,
    sol_name: str,
    ions: typing.List[int],
    modulation_type: common_types.XXModulationType,
    detuning: float,
    sign: int,
    duration: float,
    amplitudes: typing.List[float],
):
    """Write an XX gate solution to the appropriately named and placed file.

    This function does not have dependencies from elsewhere in the ARTIQ
    distribution because it is meant to be incorporated into the gate solver.

    Args:
        XX_sols_dir: The top-level directory into which all XX gate solutions
            will be placed
        N_ions: The total number of ions in the chain
        sol_name: The name of this specific solution
        ions: The specific ions being addressed
        modulation_type: The type of gate (AM, FM, AM/FM and segmented or interpolated)
            that we are implementing
        detuning: The sideband detuning from the carrier
        sign: The sign (+1 or -1) of the geometric phase acquired during the gate
        duration: The total gate duration in us
        amplitudes: The amplitudes of the gate segments, normalized to +-1
    """
    filepath = _XX_sol_filepath(XX_sols_dir, N_ions, sol_name, ions)
    XX_sol_params = [modulation_type.value, detuning, sign, duration, amplitudes]
    filepath.mkdir(parents=True, exist_ok=True)

    with filepath.open(mode="wb") as sol_file:
        pickle.dump(XX_sol_params, sol_file)


def read_XX_params(
    XX_sols_dir: str, N_ions: int, sol_name: str, slots: typing.List[int]
):
    """Read XX gate parameters (detuning, sign, duration, and amplitudes) from file.

    Args:
        XX_sols_dir: The top-level directory where all XX gate solutions are found
        N_ions: The total number of ions in the chain
        sol_name: The name of this specific solution
        slots: The two slots that are being addressed

    Returns:
        Return a list containing
        [XX_gate_type, XX_detuning, XX_sign, XX_duration, [XX_amplitudes]]

    """
    ions = [common_types.slot_to_ion(s, N_ions) for s in slots]
    filepath = _XX_sol_filepath(XX_sols_dir, N_ions, sol_name, ions)

    assert filepath.exists(), "Solution file does not exist"
    with filepath.open(mode="rb") as sol_file:
        XX_sol_params = pickle.load(sol_file)

    # The modulation type is saved to file as an int, so we recast it as its proper enum
    XX_sol_params[0] = common_types.XXModulationType(XX_sol_params[0])

    # print(XX_sol_params)
    return XX_sol_params


def read_all_XX_gate_params(
    XX_sols_dir: str, N_ions: int, sol_name: str
) -> typing.Dict[common_types.IonPair, common_types.XXGateParams]:
    """Read all XX Gate Parameters from a directory into a dictionary.

    Args:
        XX_sols_dir (str): Directory where XX Gate solutions can be found.
        N_ions (int): Number of ions that solutions were designed for
        sol_name (str): Name of the particular solution used (e.g. "interpolated")

    Returns:
        typing.Dict[common_types.IonPair, common_types.XXGateParams]:
        A pruned dictionary of solutions.
        Formatted as a dictionary from a tuple of ions (e.g. (8, 9))
        (always in sorted order) to a set of :attr:`common_types.XXGateParams`.
        Only contains the solutions that were found in files.
        Also has an attr ("_hash") that represents the hash of the solutions read,
        to uniquely identify the solutions.

    """
    solutions = collections.OrderedDict(
        zip(
            common_types.gen_all_IonPairs(num_ions=N_ions),
            itertools.repeat(None, N_ions ** 2),
        )
    )  # dictionary of gate solutions

    # Get directory of all XX Gate solutions
    gate_solutions_dir = _XX_sol_filepath(
        XX_sols_dir, N_ions, sol_name, (1, 2)
    ).parent.resolve()

    solutions_hash = eur_hash.hashdir(gate_solutions_dir)
    for sol_file in gate_solutions_dir.glob(
        common_types.SOLUTION_FILENAME_GLOB_PATTERN
    ):
        _LOGGER.debug("Parsing gate solutions file '%s'", sol_file)
        ions = tuple(map(int, sol_file.stem.split("-")))  # "08-09.sol" -> [8, 9]
        assert len(ions) == 2, "Incorrect filename format"
        # slots = (ion_to_slot(i, N_ions) for i in ions)

        with sol_file.open(mode="rb") as f:
            gate_sol_params = pickle.load(f)
            gate_sol_params[0] = common_types.XXModulationType(gate_sol_params[0])

        _LOGGER.debug("Ions %s Gate Params: %s", ions, gate_sol_params)
        solutions[ions] = common_types.XXGateParams(*gate_sol_params)

    assert solutions_hash == eur_hash.hashdir(
        gate_solutions_dir
    ), "solutions changed while loading"
    solutions["_hash"] = solutions_hash
    missing_ion_solutions = [k for k, v in solutions.items() if v is None]
    for k in missing_ion_solutions:
        solutions.pop(k)
    _LOGGER.info(
        "Solutions for the following ions were not found (removed from dict): %s",
        missing_ion_solutions,
    )

    return solutions

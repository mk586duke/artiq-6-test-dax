"""
Shared data types in the RF Compilation chain.

Used by multiple modules, so located in shared file w/ few dependencies.
"""
import enum
import functools
import itertools
import typing

import more_itertools
import numpy as np

# TODO: python 3.6 / 3.7: use auto() for enums

# *** XX Gate Solution Types ***


class XXModulationType(enum.IntEnum):
    """Type of the gate modulation in the gate solution."""

    AM_segmented = 0
    AM_interp = 1
    FM_segmented = 2
    FM_interp = 3
    AMFM_segmented = 4
    AMFM_interp = 5,
    AMFM_spline = 6


IonPair = typing.Tuple[int, int]
SlotPair = typing.Tuple[int, int]

# If NOT interpolated, then a series of amplitudes.
# Otherwise, a series of (start, stop) interpolation amplitude points
RabiPoint = typing.Union[float, typing.Tuple[float, float]]


class XXGateParams(typing.NamedTuple):
    """Specify what parameters of an XX gate are stored for nominal gate solutions.

    Deprecated, replaced by euriqabackend.waveforms.types.XXGateParams."""
    XX_gate_type: XXModulationType
    XX_detuning: float
    XX_sign: int
    XX_duration_us: float
    XX_amplitudes: typing.Sequence[RabiPoint]
    XX_angle: float = np.pi / 4

# *** Ion Gate Types ***


class GateType(enum.IntEnum):
    """Gate types recognized by :class:`~.RFCompiler.RFCompiler`."""

    Phase = 0
    ReferencePulse = 1
    Blank = 2
    Rabi = 3
    Rabi_AM = 4
    Rabi_PI = 5
    WindUnwind = 6
    FastEcho = 7
    SK1 = 8
    SK1_AM = 9
    XX = 10
    Bichromatic = 11
    CrosstalkCalib = 12


# *** Scan Parameters from RFCompiler, can't be there b/c import issues ***
class CrosstalkCalibScanParameter(enum.IntEnum):
    static = -1
    duration = 0
    phase_weak = 1
    ind_amp_weak = 2

class CrossSK1ScanParameter(enum.IntEnum):
    static = -1
    theta = 0
    phi = 1
    ind_amplitude = 2

class StabReadoutScanParameter(enum.IntEnum):
    static = -1
    initial_state = 0
    post_prepare_phase = 1
    post_XX_phase = 2
    correction_duration = 3
    correction_phase = 4

class ChargeResponseProbeScanParameter(enum.IntEnum):
    static = -1
    push_duration = 0
    probe_duration = 1
    push_detuning = 2
    probe_detuning = 3
    push_ind_amp = 4

# TODO: convert this to subclass of gate.XX.ScanParameter - Drew.
# Issue is that you'd have to import it to get that, which I DON'T LIKE.
# TODO: break out Scan Parameters into a separate unified file.
# This enum has the same elements as XXScanParameter,
# but with echo_phase tacked on at the end.
# There's probably a better way to do this, but I (Mike) wasn't sure to how do this
# programmatically with a static enum.
class XXCrosstalkScanParameter(enum.IntEnum):
    static = -1
    duration_adjust = 0
    Stark_shift = 1
    Stark_shift_diff = 2
    motional_freq_adjust = 3
    phi_ind1 = 4
    phi_ind2 = 5
    phi_ind_com = 6
    phi_ind_diff = 7
    phi_global = 8
    phi_motional = 9
    phi_b = 10
    phi_r = 11
    ind_amplitude_multiplier = 12
    ind_amplitude_imbalance = 13
    global_amplitude = 14
    sb_amplitude_imbalance = 15
    t_delay = 16
    N_gates = 17
    echo_phase_single_absolute = 18
    echo_phase_all_relative = 19
    analysis_phase = 20

# *** Ion <-> Slot mappings, convert between ion # & physical hardware ***


def gen_all_IonPairs(num_ions: int) -> typing.Iterator[IonPair]:
    """Generate all possible ion gate pairs, no repetitions.

    Always in order (lower_ion, upper_ion). 1-indexed.

    Example:
        >>> list(gen_all_IonPairs(3))
        [(1, 2), (1, 3), (2, 3)]
    """
    yield from itertools.combinations(range(1, num_ions + 1), 2)


# Center slot num for the EURIQA breadboard system. 32 channels / 2 (half) + 1
CENTER_SLOT = 17


def slot_to_ion(slot: int, N_ions: int, center_slot: int = CENTER_SLOT):
    """Convert a slot number to an ion number based on the total number of ions.

    Args:
        slot: The AOM/fiber slot, running from 1 to 32
        N_ions: The total number of ions in the chain
        center_slot (int): the slot that corresponds to the center ion in the chain

    Returns:
        The ion number, running from 1 to N_ions

    Examples:
        >>> slot_to_ion(17, 32)
        16
        >>> slot_to_ion(17, 1)
        1
        >>> slot_to_ion(17, 32, 15)
        18

    """
    center_ion = int(np.ceil(N_ions / 2))
    return slot - center_slot + center_ion


def gen_all_SlotPairs(
    num_ions: int, center_slot: int = CENTER_SLOT
) -> typing.Iterator[SlotPair]:
    """Generate all possible slot pairs, no repetitions.

    Always in order (lower_ion, upper_ion), 1-indexed.

    Example:
        >>> list(gen_all_SlotPairs(3, 17))
        [(16, 17), (16, 18), (17, 18)]
    """
    slot_convert_func = functools.partial(
        ion_to_slot, N_ions=num_ions, center_slot=center_slot
    )
    # yield tuples of slot pairs in same ordering as ion pairs
    yield from map(
        tuple,  # coerce to tuples vs list
        # First flatten to do the mapping, and then re-chunk in groups of 2
        more_itertools.chunked(
            # Apply ion -> slot conversion on all ion pairs
            map(slot_convert_func, more_itertools.flatten(gen_all_IonPairs(num_ions))),
            2,
        ),
    )


def ion_to_slot(ion: int, N_ions: int, center_slot: int = CENTER_SLOT, one_indexed: bool = True):
    """Convert an ion number to a slot number based on the total number of ions.

    Args:
        ion: The ion number, running from 1 to N_ions
        N_ions: The total number of ions in the chain
        center_slot (int): the slot that corresponds to the center ion in the chain
        one_indexed (bool): One-indexed ions are labeled like 1,2,3,4....,15
                            Zero-indexed ions are labeled like -7,-6,...-1,0,1,...6,7

    Returns:
        The AOM/fiber slot, running from 1 to 32

    Examples:
        >>> ion_to_slot(1, 1)
        17
        >>> ion_to_slot(16, 32)
        16
        >>> ion_to_slot(16, 31, 16)
        16
        >>> ion_to_slot(-5,15,17,False)
        12

    """
    if one_indexed:
        center_ion = int(np.floor(N_ions / 2)) + 1
        return ion - center_ion + center_slot
    else:
        return ion + center_slot



SOLUTION_FILENAME_GLOB_PATTERN = "??-??.sol"

"""Reconstruct timing information based on a schedule."""
import enum
import typing

import more_itertools
import numpy as np
import qiskit.pulse as qp


class EdgeType(enum.Enum):
    RISING = enum.auto()
    FALLING = enum.auto()
    DELTA = enum.auto()
    """Instantaneous edge, for timeslot that has no duration"""


def channel_timing(
    schedule: qp.Schedule,
    channel: qp.MeasureChannel,
    dt: float,
    desired_instructions: typing.Set[typing.Type[qp.Instruction]] = None,
    initial_state=EdgeType.FALLING,
) -> typing.Tuple[typing.Sequence[float], typing.Sequence[EdgeType]]:
    """Extract timing information from a schedule.

    Useful for retrieving timings that are stored in a schedule,
    e.g. for SBC pump timings during a schedule.
    """
    if desired_instructions is None:
        desired_instructions = {qp.Play}
    timing_dt = []
    state = []
    filtered_schedule = schedule.filter(
        channels=[channel], instruction_types=desired_instructions
    )
    filtered_timeslots = filtered_schedule.timeslots[channel]

    if filtered_timeslots[0][0] != 0:
        timing_dt.append(0)
        state.append(initial_state)

    for ts_start, ts_stop in filtered_timeslots:
        if ts_start == ts_stop:
            timing_dt.append(ts_start)
            state.append(EdgeType.DELTA)
        else:
            timing_dt.append(ts_start)
            state.append(EdgeType.RISING)
            timing_dt.append(ts_stop)
            state.append(EdgeType.FALLING)

    timing_seconds = list(time * dt for time in timing_dt)

    return timing_seconds, state


def timing_to_differential(
    timing_seconds: typing.Sequence[float],
) -> typing.Sequence[float]:
    """Convert a sequence of timestamps to the differences between the timestamps."""
    return np.diff(np.array(timing_seconds))


def edges_to_bool(states: typing.Sequence[EdgeType]) -> typing.Sequence[bool]:
    """Convert a sequence of edge transitions to the logic levels in between.

    Useful for turning a sequence of events into TTL On/Off's for ARTIQ.
    """
    return list(False if state == EdgeType.FALLING else True for state in states)

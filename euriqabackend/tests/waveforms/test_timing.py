"""Test :mod:`euriqabackend.waveforms.timing`."""
import numpy as np
import pytest
import qiskit.pulse as qp

import euriqabackend.waveforms.timing as wf_timing


@pytest.fixture
def schedule():
    with qp.build() as schedule:
        ch0 = qp.DriveChannel(0)
        ch1 = qp.ControlChannel(0)
        qp.set_frequency(100e6, ch0)
        qp.play(qp.Constant(100, 1.0), ch0)
        qp.delay(250, ch0)
        qp.play(qp.Constant(100, 1.0), ch0)

        qp.play(qp.Constant(500, 1.0), ch1)

    return schedule


def test_channel_timing(schedule: qp.Schedule):
    dt = 1e-9
    # Check handling of instantaneous commands
    timing, states = wf_timing.channel_timing(
        schedule,
        qp.DriveChannel(0),
        dt=dt,
        desired_instructions={qp.Play, qp.SetFrequency},
    )
    np.testing.assert_allclose(np.array(timing), np.array([0, 0, 100, 350, 450]) * dt)
    assert states == [
        wf_timing.EdgeType.DELTA,
        wf_timing.EdgeType.RISING,
        wf_timing.EdgeType.FALLING,
        wf_timing.EdgeType.RISING,
        wf_timing.EdgeType.FALLING,
    ]

    # Check handling of commands that are not instantaneous
    timing, states = wf_timing.channel_timing(
        schedule, qp.DriveChannel(0), dt=dt, desired_instructions={qp.Play}
    )
    np.testing.assert_allclose(np.array(timing), np.array([0, 100, 350, 450]) * dt)
    assert states == [
        wf_timing.EdgeType.RISING,
        wf_timing.EdgeType.FALLING,
        wf_timing.EdgeType.RISING,
        wf_timing.EdgeType.FALLING,
    ]

    # Check handling of initial state
    timing, states = wf_timing.channel_timing(
        schedule.shift(5), qp.DriveChannel(0), dt=dt, desired_instructions={qp.Play}
    )
    np.testing.assert_allclose(np.array(timing), np.array([0, 5, 105, 355, 455]) * dt)
    assert states == [
        wf_timing.EdgeType.FALLING,
        wf_timing.EdgeType.RISING,
        wf_timing.EdgeType.FALLING,
        wf_timing.EdgeType.RISING,
        wf_timing.EdgeType.FALLING,
    ]


def test_timing_to_differential():
    timestamps = [0.0, 1e-9, 2e-9, 5e-9]
    timestamps_diff = wf_timing.timing_to_differential(timestamps)
    np.testing.assert_allclose(timestamps_diff, np.diff(timestamps))


def test_states_to_differential():
    states = [
        wf_timing.EdgeType.FALLING,
        wf_timing.EdgeType.RISING,
        wf_timing.EdgeType.RISING,
        wf_timing.EdgeType.FALLING,
        wf_timing.EdgeType.RISING,
    ]
    assert wf_timing.edges_to_bool(states) == [False, True, True, False, True]

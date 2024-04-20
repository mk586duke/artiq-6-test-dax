"""Test :mod:`euriqabackend.waveforms.sideband_cooling`."""
import itertools
import random
import typing

import numpy as np
import pulsecompiler.qiskit.backend as pc_backend
import pulsecompiler.qiskit.configuration as pc_config
import pytest
import qiskit.pulse as qp

import euriqabackend.waveforms.sideband_cooling as wf_sbc


@pytest.fixture
def num_sbc_ions(rfsoc_qiskit_backend) -> int:
    return rfsoc_qiskit_backend.configuration().num_qubits


@pytest.fixture
def sbc_detunings(num_sbc_ions) -> np.ndarray:
    cooling_detuning_freq = -3e6
    num_cooling_steps = num_sbc_ions
    detuning_array = np.empty((num_cooling_steps, num_sbc_ions))
    detuning_array[:] = np.NaN
    center_idx = int(np.floor(num_sbc_ions / 2))
    for i in range(num_cooling_steps):
        if i <= num_cooling_steps / 2:
            detuning_array[
                i, center_idx - i : center_idx + i + 1
            ] = cooling_detuning_freq
        else:
            start_index = np.abs(center_idx - i)
            stop_index = num_sbc_ions - np.abs(center_idx - i)
            detuning_array[i, start_index:stop_index] = cooling_detuning_freq
    return detuning_array


@pytest.fixture
def sbc_frequencies(
    rfsoc_qiskit_backend: pc_backend.MinimalQiskitIonBackend,
) -> typing.Dict[typing.Any, float]:
    return {
        c: 200e6 if isinstance(c, qp.DriveChannel) else 150e6
        for c in rfsoc_qiskit_backend.configuration().all_channels
    }


@pytest.fixture
def sbc_timings(sbc_detunings):
    timings = np.ones(sbc_detunings.shape) * 1e-6
    for row in range(len(timings)):
        timings[row] *= row + 1

    return timings


def test_sbc_schedule_creation(
    sbc_detunings: np.ndarray,
    sbc_timings: np.ndarray,
    sbc_frequencies: typing.Dict[typing.Any, float],
    rfsoc_qiskit_backend: pc_backend.MinimalQiskitIonBackend,
):
    num_ions = sbc_detunings.shape[-1]
    if rfsoc_qiskit_backend.configuration().num_qubits < sbc_detunings.shape[0]:
        pytest.xfail("Not enough qubits specified for SBC sequence")

    backend_config = rfsoc_qiskit_backend.configuration()
    config_zero_index = (
        backend_config
        if type(backend_config) == pc_config.BackendConfigZeroIndex
        else backend_config.to_zero_index()
    )
    index_channel_mapping = {i: config_zero_index.drive(i) for i in range(num_ions)}
    sbc_schedules = wf_sbc.multiple_schedule_parallel_sbc(
        num_ions,
        sbc_detunings,
        sbc_timings,
        sbc_frequencies,
        rfsoc_qiskit_backend,
        index_to_channel_map=index_channel_mapping,
    )

    for s in sbc_schedules:
        assert isinstance(s, qp.Schedule)
        # assert


def test_generate_detunings():
    mode_arr = np.array([[0, 1], [1, 0]])
    modes = [1e6, 2e6]
    np.testing.assert_array_equal(
        wf_sbc.generate_detunings(mode_arr, modes), np.array([[1e6, 2e6], [2e6, 1e6]])
    )


def test_generate_timing_increment(sbc_detunings):
    initial_time = 1e-6
    time_increment = 5e-6
    timings = wf_sbc.generate_timings_increment(
        sbc_detunings.shape, initial_time=1e-6, time_increment=5e-6
    )

    assert timings.shape == sbc_detunings.shape
    for step in range(timings.shape[0]):
        assert (timings[step] == (initial_time + step * time_increment)).all()


@pytest.mark.parametrize(
    "riffle_string,expected_array",
    [
        (
            "e 10-24\ne 17\nd 16;e 17;d 18",
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        1,
                        2,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ]
            ),
        )
    ],
)
def test_riffle_string_to_array(riffle_string: str, expected_array: np.array):
    mode_array = wf_sbc.awg_riffle_to_modes(
        riffle_string, min_slot=10, max_slot=24, min_channel="c"
    )

    np.testing.assert_array_equal(mode_array, expected_array)


def test_single_schedule_sbc(
    rfsoc_qiskit_backend: pc_backend.MinimalQiskitIonBackend, tmp_path
):
    ion_freqs = {
        idx: random.uniform(190e6, 195e6)
        for idx in rfsoc_qiskit_backend.configuration().all_qubit_indices_iter
    }
    num_main_pulses = 5
    main_pulse_durations = np.linspace(1e-6, 50e-6, num=num_main_pulses)
    relative_ion_durations = np.array(
        [random.uniform(0.1, 1.0) for _idx in sorted(ion_freqs.keys())]
    )
    pump_duration = 1e-6
    timing_array = wf_sbc.relative_timings_to_absolute(
        main_pulse_durations, relative_ion_durations
    )
    amplitude_array = wf_sbc.expand_relative_amplitudes(
        np.array([1.0] * rfsoc_qiskit_backend.configuration().num_qubits),
        num_repeats=num_main_pulses,
    )
    sbc_schedule, (pump_durations, pump_states) = wf_sbc.single_schedule_parallel_sbc(
        ion_freqs,
        timing_array,
        amplitude_array,
        pump_duration,
        200e6,
        rfsoc_qiskit_backend,
    )

    assert (len(pump_durations) + 1) == len(pump_states) == (num_main_pulses * 2 + 1)
    assert all((isinstance(d, float) for d in pump_durations))
    assert all((isinstance(state, bool) for state in pump_states))

    global_channel = qp.ControlChannel(0)

    sbc_schedule.draw().savefig(tmp_path / "sbc_single_schedule.png")

    with qp.build(rfsoc_qiskit_backend):
        # check that the global beam is not on during pump.
        # This is an approximation to checking that the timing of the pump beam is correct
        pump_timings_accumulated_dt = list(
            map(qp.seconds_to_samples, itertools.accumulate(pump_durations))
        )
        pump_timings_iter = iter(pump_timings_accumulated_dt)
        pump_states_iter = iter(pump_states)
        assert len(pump_timings_accumulated_dt) == len(pump_durations)
        assert sbc_schedule.duration >= pump_timings_accumulated_dt[-1]

        current_pump_start_dt = next(pump_timings_iter)
        current_pump_state = next(pump_states_iter)
        # compare timing against play/delay instructions only (ignore frequencies)
        for start_time, instr in sbc_schedule.filter(
            channels=[global_channel], instruction_types=[qp.Delay, qp.Play]
        ).instructions:
            while start_time > current_pump_start_dt:
                current_pump_start_dt = next(pump_timings_iter)
                current_pump_state = next(pump_states_iter)
            # Check that instr is only playing IFF pump state is True
            if isinstance(instr, qp.Play):
                assert not current_pump_state
            elif isinstance(instr, qp.Delay) and start_time >= current_pump_start_dt:
                assert current_pump_state


def test_relative_timings_to_absolute():
    num_durations = 50
    overall_durations = np.linspace(1e-6, 50e-6, num=num_durations)
    relative_durations = np.array([0.2, 0.5, 0.8, 1.0])
    timing_array = wf_sbc.relative_timings_to_absolute(
        overall_durations, relative_durations
    )
    assert timing_array.shape == (num_durations, len(relative_durations))
    np.testing.assert_allclose(
        timing_array[0, :], min(overall_durations) * relative_durations
    )
    np.testing.assert_allclose(
        timing_array[-1, :], max(overall_durations) * relative_durations
    )

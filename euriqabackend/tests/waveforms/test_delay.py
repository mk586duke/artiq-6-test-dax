"""Test :mod:`euriqabackend.waveforms.delay`."""
import itertools

import qiskit.pulse as qp
import more_itertools

import pulsecompiler.qiskit.schedule_converter as sched_converter
import euriqabackend.waveforms.delay as wf_delay


def test_wait_gate_channels_basic(rfsoc_qiskit_backend):
    """Test :func:`wait_gate_channels`."""
    with qp.build(rfsoc_qiskit_backend) as test_sched:
        delay_time = 100e-6

        # Play a tone on a random channel, and check that all have same end point
        qp.play(qp.Constant(100, 1.0), qp.DriveChannel(0))
        qp.call(
            wf_delay.wait_gate_channels(
                delay_time, {qp.DriveChannel(0), qp.ControlChannel(0)},
            )
        )
        delay_time_dt = qp.seconds_to_samples(delay_time)

    assert more_itertools.all_equal(map(test_sched.ch_stop_time, test_sched.channels))
    assert set(test_sched.channels) == {qp.DriveChannel(0), qp.ControlChannel(0)}
    assert test_sched.duration == 100 + delay_time_dt


def test_wait_gate_channels_all_channels(rfsoc_qiskit_backend):
    with qp.build(rfsoc_qiskit_backend) as test_sched:
        delay_time = 1e-6
        delay_2_time = 2 * delay_time
        delay_time_dt = qp.seconds_to_samples(delay_time)
        delay_2_time_dt = qp.seconds_to_samples(delay_2_time)

        qp.play(qp.Constant(100, amp=0.5), qp.DriveChannel(0))
        qp.call(wf_delay.wait_gate_channels(delay_time))
        # Add new channel, check that it starts after the first
        qp.play(qp.Constant(10, amp=0.5), qp.DriveChannel(2))
        qp.call(wf_delay.wait_gate_channels(delay_time * 2))

    assert test_sched.ch_start_time(qp.DriveChannel(2)) == 100
    assert (
        test_sched.ch_stop_time(qp.DriveChannel(2))
        == 100 + (delay_time_dt + delay_2_time_dt) + 10
    )
    assert test_sched.duration == 100 + (delay_time_dt + delay_2_time_dt) + 10
    all_rfsoc_qiskit_channels = rfsoc_qiskit_backend.configuration().all_channels
    assert set(test_sched.channels) == all_rfsoc_qiskit_channels

    # Test compilation to PulseCompiler, b/c had issues in past w/ odd start times
    rfsoc_schedule = sched_converter.OpenPulseToOctetConverter.schedule_to_octet(
        test_sched, {c: 200e6 for c in all_rfsoc_qiskit_channels}
    )
    assert set(rfsoc_schedule.keys()) == all_rfsoc_qiskit_channels
    # Choose a non-specified drive channel, make sure that it makes sense.
    # Extra time is for the prepare & measure pulses
    assert sum(
        map(lambda td: td.duration_cycles, rfsoc_schedule[qp.DriveChannel(1)])
    ) == test_sched.duration + 4 + (122 * 2)


def test_wait_gate_ions_all_ions(rfsoc_qiskit_backend):
    with qp.build(rfsoc_qiskit_backend) as test_sched:
        delay_time = 1e-6
        delay_time_dt = qp.seconds_to_samples(delay_time)

        # play a fake schedule where a "gate" is played on one channel, then all delayed
        individual_channel = qp.drive_channel(0)
        qp.play(qp.Constant(100, amp=0.5), individual_channel)
        wf_delay.wait_gate_ions(delay_time)

    assert test_sched.duration == 100 + delay_time_dt
    assert more_itertools.all_equal(map(test_sched.ch_stop_time, test_sched.channels))
    assert test_sched.ch_stop_time(individual_channel) == 100 + delay_time_dt
    # get all output channels for all qubit channels
    all_qubit_output_channels = set(
        itertools.chain.from_iterable(
            map(
                rfsoc_qiskit_backend.configuration().get_qubit_channels,
                rfsoc_qiskit_backend.configuration().all_qubit_indices_iter,
            )
        )
    )
    assert set(test_sched.channels) == all_qubit_output_channels

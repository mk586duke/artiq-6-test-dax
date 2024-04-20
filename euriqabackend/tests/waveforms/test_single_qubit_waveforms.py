"""Test :mod:`euriqabackend.waveforms.single_qubit`."""
import numpy as np
import pulsecompiler.qiskit.pulses as pc_pulses
import pytest
import qiskit.pulse as qp

import euriqabackend.waveforms.single_qubit as oneq


def test_single_qubit_square_pulse(rfsoc_qiskit_backend):
    with qp.build(rfsoc_qiskit_backend) as test_sched:
        qp.call(oneq.square_pulse(qp.drive_channel(0), 100e-6, detuning=100))

    with qp.build(rfsoc_qiskit_backend) as correct_sched:
        out_chan = qp.drive_channel(0)
        qp.shift_frequency(100, out_chan)
        qp.play(qp.Constant(40960, amp=1.0), out_chan)
        qp.shift_frequency(-100, out_chan)

    assert test_sched == correct_sched


def test_single_qubit_square_rabi(rfsoc_qiskit_backend):
    with qp.build(rfsoc_qiskit_backend) as test_sched:
        qp.call(
            oneq.square_rabi_by_amplitude(
                ion_index=0,
                duration=50e-6,
                phase=0.763,
                detuning=1e6,
                sideband_order=2,
                individual_amp=0.6,
                global_amp=0.8,
            )
        )

    with qp.build(rfsoc_qiskit_backend) as correct_sched:
        ind_chan = qp.drive_channel(0)
        global_chan = qp.ControlChannel(0)
        qp.play(qp.Constant(20480, amp=0.6), ind_chan)
        qp.shift_phase(0.763, global_chan)
        qp.shift_frequency(2e6, global_chan)
        qp.play(qp.Constant(20480, amp=0.8), global_chan)
        qp.shift_frequency(-2e6, global_chan)
        qp.shift_phase(-0.763, global_chan)

    assert test_sched == correct_sched


def test_single_qubit_square_rabi_phase_insensitive(rfsoc_qiskit_backend):
    with pytest.raises(NotImplementedError):
        with qp.build(rfsoc_qiskit_backend):
            qp.call(
                oneq.square_rabi_by_amplitude(
                    0,
                    duration=1e-6,
                    phase_insensitive=True,
                    individual_amp=0.5,
                    global_amp=0.5,
                )
            )


@pytest.mark.todo  # remove once remove hack for modifying phase in gate func
def test_single_qubit_sk1_constant(qiskit_backend_with_real_cals):
    """Test a single-qubit SK1 pulse matches something reasonable."""
    theta = np.pi / 2
    phi = np.pi / 4
    with qp.build(qiskit_backend_with_real_cals) as test_sched:
        qp.call(
            oneq.sk1_square_by_amplitude(
                0, theta=theta, phi=phi, individual_amplitude=1.0, global_amplitude=1.0
            )
        )

    global_channel = qp.ControlChannel(0)
    with qp.build(qiskit_backend_with_real_cals) as correct_sched:
        individual_channel = qp.drive_channel(0)
        qp.shift_frequency(-0, individual_channel)

        # Initial shift to put the global in the frame of the gate phi
        qp.shift_phase(phi, global_channel)

        # First pulse
        qp.play(qp.Constant(243, amp=1.0), individual_channel)
        qp.shift_phase(0, global_channel)
        qp.play(qp.Constant(243, amp=1.0), global_channel)
        qp.shift_phase(0, global_channel)

        # First correction pulse
        qp.play(qp.Constant(975, amp=1.0), individual_channel)
        qp.shift_phase(4.587061149216624, global_channel)
        qp.play(qp.Constant(975, amp=1.0), global_channel)
        qp.shift_phase(-4.587061149216624, global_channel)

        # second correction pulse
        qp.play(qp.Constant(975, amp=1.0), individual_channel)
        qp.shift_phase(1.696124157962962, global_channel)
        qp.play(qp.Constant(975, amp=1.0), global_channel)
        qp.shift_phase(-1.696124157962962, global_channel)

        # Unshift frequency & phi phase
        qp.shift_phase(-phi, global_channel)
        qp.shift_frequency(0, individual_channel)

    assert test_sched == correct_sched


def test_single_qubit_sk1_constant_no_correction(qiskit_backend_with_real_cals):
    """Test a single-qubit SK1 pulse of theta=0.0 returns nothing."""
    theta = 0
    phi = np.pi / 4
    with qp.build(qiskit_backend_with_real_cals) as test_sched:
        qp.call(
            oneq.sk1_square_by_amplitude(
                0, theta=theta, phi=phi, individual_amplitude=1.0, global_amplitude=1.0
            )
        )

    # no theta rotation -> produce no output
    assert test_sched == qp.Schedule()


@pytest.mark.todo  # remove once remove hack for modifying phase in gate func
def test_single_qubit_sk1_custom_durations(qiskit_backend_with_real_cals):
    """Test custom durations for the rotations of an SK1 gate are respected."""
    theta = np.pi / 2
    phi = np.pi / 4
    rotation_durations = (2e-6, 5e-6, 10e-6)
    with qp.build(qiskit_backend_with_real_cals) as test_sched:
        # check errors with no durations specified, but no auto-calculate
        with pytest.raises(AssertionError):
            qp.call(
                oneq.sk1_square_by_amplitude(
                    0,
                    theta=theta,
                    phi=phi,
                    calculate_durations=False,
                    individual_amplitude=1.0,
                    global_amplitude=1.0,
                )
            )
        # check errors with auto-calculation & a duration specified
        with pytest.raises(AssertionError):
            qp.call(
                oneq.sk1_square_by_amplitude(
                    0,
                    theta=theta,
                    phi=phi,
                    rotation_duration=1e-6,
                    individual_amplitude=1.0,
                    global_amplitude=1.0,
                )
            )
        qp.call(
            oneq.sk1_square_by_amplitude(
                0,
                theta=theta,
                phi=phi,
                calculate_durations=False,
                rotation_duration=rotation_durations[0],
                correction_duration_0=rotation_durations[1],
                correction_duration_1=rotation_durations[2],
                individual_amplitude=1.0,
                global_amplitude=1.0,
            )
        )

    global_channel = qp.ControlChannel(0)
    with qp.build(qiskit_backend_with_real_cals) as correct_sched:
        individual_channel = qp.drive_channel(0)
        qp.shift_frequency(-0, individual_channel)
        rot_durations_dt = list(map(qp.seconds_to_samples, rotation_durations))

        # Initial shift to put the global in the frame of the gate phi
        qp.shift_phase(phi, global_channel)

        # First pulse
        qp.play(qp.Constant(rot_durations_dt[0], amp=1.0), individual_channel)
        qp.shift_phase(0, global_channel)
        qp.play(qp.Constant(rot_durations_dt[0], amp=1.0), global_channel)
        qp.shift_phase(0, global_channel)

        # First correction pulse
        qp.play(qp.Constant(rot_durations_dt[1], amp=1.0), individual_channel)
        qp.shift_phase(4.587061149216624, global_channel)
        qp.play(qp.Constant(rot_durations_dt[1], amp=1.0), global_channel)
        qp.shift_phase(-4.587061149216624, global_channel)

        # second correction pulse
        qp.play(qp.Constant(rot_durations_dt[2], amp=1.0), individual_channel)
        qp.shift_phase(1.696124157962962, global_channel)
        qp.play(qp.Constant(rot_durations_dt[2], amp=1.0), global_channel)
        qp.shift_phase(-1.696124157962962, global_channel)

        # Unshift frequency & phi phase
        qp.shift_phase(-phi, global_channel)
        qp.shift_frequency(0, individual_channel)

    assert test_sched == correct_sched


def test_rz(rfsoc_qiskit_backend):
    with qp.build(rfsoc_qiskit_backend) as test_schedule:
        qp.call(oneq.rz(0, 0.7123))

    with qp.build(rfsoc_qiskit_backend) as correct_schedule:
        individual_channel = qp.drive_channel(0)
        qp.shift_phase(0.7123, individual_channel)

    assert test_schedule == correct_schedule


def test_single_qubit_sk1_gaussian_runs(qiskit_backend_with_real_cals):
    with qp.build(qiskit_backend_with_real_cals):
        qp.call(
            oneq.sk1_gaussian_by_amplitude(
                0,
                theta=np.pi / 2,
                phi=np.pi / 2,
                individual_amplitude=0.5,
                global_amplitude=0.5,
            )
        )


@pytest.mark.todo
def test_single_qubit_sk1_gaussian(qiskit_backend_with_real_cals):
    theta = np.pi / 2
    phi = np.pi / 2
    individual_amplitude = 0.425
    global_amplitude = 0.63

    with qp.build(qiskit_backend_with_real_cals) as test_schedule:
        qp.call(
            oneq.sk1_gaussian_by_amplitude(
                0,
                theta=theta,
                phi=phi,
                individual_amplitude=individual_amplitude,
                global_amplitude=global_amplitude,
            )
        )

    with qp.build(qiskit_backend_with_real_cals) as correct_schedule:
        individual_channel = qp.drive_channel(0)
        global_channel = qp.control_channels()[0]
        qp.shift_frequency(-0, individual_channel)

        # Initial shift to put the global in the frame of the gate phi
        qp.shift_phase(phi, global_channel)

        # individual is gaussian for full duraiton
        qp.play(
            pc_pulses.LinearGaussian(243 + 975 + 975, amp=individual_amplitude),
            individual_channel,
        )

        # First pulse
        qp.shift_phase(0, global_channel)
        qp.play(pc_pulses.LinearGaussian(243, amp=global_amplitude), global_channel)
        qp.shift_phase(0, global_channel)

        # First correction pulse
        qp.shift_phase(4.587061149216624, global_channel)
        qp.play(pc_pulses.LinearGaussian(975, amp=global_amplitude), global_channel)
        qp.shift_phase(-4.587061149216624, global_channel)

        # second correction pulse
        qp.shift_phase(1.696124157962962, global_channel)
        qp.play(pc_pulses.LinearGaussian(975, amp=global_amplitude), global_channel)
        qp.shift_phase(-1.696124157962962, global_channel)

        # Unshift frequency & phi phase
        qp.shift_frequency(0, individual_channel)
        qp.shift_phase(-phi, global_channel)

    assert test_schedule == correct_schedule


def test_single_qubit_sk1_gaussian_no_rotation(qiskit_backend_with_real_cals):
    with qp.build(qiskit_backend_with_real_cals) as test_schedule:
        qp.call(
            oneq.sk1_gaussian_by_amplitude(
                0,
                theta=0,
                phi=np.pi / 2,
                individual_amplitude=1.0,
                global_amplitude=1.0,
            )
        )

    assert test_schedule == qp.Schedule()


def test_single_qubit_sk1_gaussian_custom_durations(qiskit_backend_with_real_cals):
    """Test custom durations for the rotations of an SK1 gaussian gate are respected."""
    theta = np.pi / 2
    phi = np.pi / 4
    rotation_durations = (2e-6, 5e-6, 10e-6)
    individual_amplitude = 0.425
    global_amplitude = 0.525
    with qp.build(qiskit_backend_with_real_cals) as test_sched:
        # check errors with no durations specified, but no auto-calculate
        with pytest.raises(AssertionError):
            qp.call(
                oneq.sk1_gaussian_by_amplitude(
                    0,
                    theta=theta,
                    phi=phi,
                    calculate_durations=False,
                    individual_amplitude=individual_amplitude,
                    global_amplitude=1.09,
                )
            )
        # check errors with auto-calculation & a duration specified
        with pytest.raises(AssertionError):
            qp.call(
                oneq.sk1_gaussian_by_amplitude(
                    0,
                    theta=theta,
                    phi=phi,
                    rotation_duration=1e-6,
                    individual_amplitude=individual_amplitude,
                    global_amplitude=global_amplitude,
                )
            )
        qp.call(
            oneq.sk1_gaussian_by_amplitude(
                0,
                theta=theta,
                phi=phi,
                calculate_durations=False,
                rotation_duration=rotation_durations[0],
                correction_duration_0=rotation_durations[1],
                correction_duration_1=rotation_durations[2],
                individual_amplitude=individual_amplitude,
                global_amplitude=global_amplitude,
            )
        )

    with qp.build(qiskit_backend_with_real_cals) as correct_sched:
        individual_channel = qp.drive_channel(0)
        global_channel = qp.control_channels()[0]

        qp.shift_frequency(-0, individual_channel)
        rot_durations_dt = list(map(qp.seconds_to_samples, rotation_durations))

        # Initial shift to put the global in the frame of the gate phi
        qp.shift_phase(phi, global_channel)

        # individual channel pulse
        qp.play(
            pc_pulses.LinearGaussian(
                qp.seconds_to_samples(sum(rotation_durations)), amp=individual_amplitude
            ),
            individual_channel,
        )
        # First pulse
        qp.shift_phase(0, global_channel)
        qp.play(
            pc_pulses.LinearGaussian(rot_durations_dt[0], amp=global_amplitude),
            global_channel,
        )
        qp.shift_phase(0, global_channel)

        # First correction pulse
        qp.shift_phase(4.587061149216624, global_channel)
        qp.play(
            pc_pulses.LinearGaussian(rot_durations_dt[1], amp=global_amplitude),
            global_channel,
        )
        qp.shift_phase(-4.587061149216624, global_channel)

        # second correction pulse
        qp.shift_phase(1.696124157962962, global_channel)
        qp.play(
            pc_pulses.LinearGaussian(rot_durations_dt[2], amp=global_amplitude),
            global_channel,
        )
        qp.shift_phase(-1.696124157962962, global_channel)

        # Unshift frequency & phi phase
        qp.shift_phase(-phi, global_channel)
        qp.shift_frequency(0, individual_channel)

    assert test_sched == correct_sched


@pytest.mark.parametrize(
    "gate_function,args",
    [
        (oneq.sk1_gaussian_by_rabi_frequency, (0, np.pi / 2, 0)),
        (oneq.sk1_square_by_rabi_frequency, (0, np.pi / 2, 0)),
        (oneq.square_rabi_by_rabi_frequency, (0, 10e-6, 100e3)),
    ],
)
def test_gate_functions_by_rabi_frequency(
    gate_function, args, qiskit_backend_with_real_cals
):
    with qp.build(qiskit_backend_with_real_cals):
        qp.call(gate_function(*args, backend=qiskit_backend_with_real_cals))

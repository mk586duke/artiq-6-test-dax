"""Waveforms to be used in EURIQA.

This file only contains single-qubit gates.

Currently, most parameters are passed by hand.
I would like to pull them from the configuration of the active backend with
:func:`qiskit.pulse.active_backend`.

These functions are all supposed to run within the post-qiskit-terra v0.15.1
:func:`qiskit.pulse.build` pulse-builder contexts, and will cause unexpected
errors if not properly called.

They can be called e.g.

.. code-block:: python

    import qiskit.pulse as qp
    from euriqabackend.waveforms.single_qubit import square_rabi

    output_channel = qp.ControlChannel(0)
    with qp.build(backend=QiskitBackend()) as schedule:
        qp.call(square_rabi(output_channel, duration=100e-6, detuning=10e6, amp=1.0))

    print(schedule)

Notes:
    * Parameters that are supposed to be filled in from calibrations should
      default to ``None``, even if there is a sensible value. This is so that
      if the calibrations are not available/passed, then a loud error will be raised,
      instead of a silent failure.
    * The ``@default_args_from_calibration()`` decorator pulls the default arguments
      from the :class:`CalibrationBox` attached to the backend.
      However, any default calibration arguments can be overridden for a particular
      instance of the gate function, e.g. if you are calibrating an amplitude, you
      can manually specify that when calling the gate function, and it will override
      the (out-of-date) calibrated value.
"""
import logging
import typing

import numpy as np
import pulsecompiler.qiskit.pulses as pc_pulses
import qiskit.providers.backend as qbe
import qiskit.pulse as qp
from qiskit.pulse import channels as chans

import euriqabackend.waveforms.conversions as wf_convert
import euriqabackend.waveforms.decorators as wf_dec


_LOGGER = logging.getLogger(__name__)


def square_pulse(
    channel: typing.Union[qp.DriveChannel, qp.ControlChannel],
    duration: float,
    detuning: float = 0.0,
    amp: float = 1.0,
    backend: qbe.Backend = None,
) -> qp.Schedule:
    """Add a Schedule containing a square pulse on the given channel.

    This ONLY plays it on a single channel, and not also on e.g. the global channel.
    Meant to be used within a Qiskit Pulse Builder context.
    See :mod:`qiskit.pulse.builder` for more details.

    This provides a bit finer control over the pulses vs
    :func:`square_rabi_by_amplitude`,
    because it doesn't auto-assume control of the global channel,
    allowing performing operations on multiple ions in parallel.

    Args:
        channel (Union[DriveChannel, ControlChannel]): Qiskit-language channel to play
            the pulse on. Corresponds to one tone of an RFSoC channel.
        duration (float): duration of the rabi pulse in seconds.
        detuning (float): frequency offset of the channel from its nominal value
            (i.e. the detuning from the nominal qubit frequency). Units of Hz.
            Defaults to 0.0.
        amp (float): Amplitude of the pulse to play on ``channel``.
            Valid in ranges [-1.0, 1.0]. Defaults to 1.0.

    Returns:
        Schedule: Qiskit Schedule comprised of a constant-amplitude pulse on the given
        channel, detuned (shifted) from the nominal (i.e. previous) frequency by
        ``detuning``.
    """
    if not (-1.0 <= amp <= 1.0):
        raise ValueError(f"Amplitude {amp} out of range -1.0 <= amp <= 1.0")
    if backend is None:
        backend = qp.active_backend()
    with qp.build(backend=backend) as schedule:
        # NOTE: set compensate_phase=True to reset back to phase at original frequency
        with qp.frequency_offset(detuning, channel, compensate_phase=False):
            qp.play(qp.Constant(qp.seconds_to_samples(duration), amp), channel)

    return schedule

@wf_dec.check_all_channels_same_duration
def bichromatic_rabi_by_amplitude(
    ion_index: typing.Union[int,list],
    duration: float,
    individual_amp: typing.Union[float,list],
    global_amp: float,
    phase: float = 0.0,
    motional_detuning: float = 0.0,
    carrier_detuning: float = 0.0,
    sideband_order: int = 0,
    phase_insensitive: bool = False,
    sideband_amplitude_imbalance: float = 0,
    stark_shift : typing.Union[float,list] = [0.0],
    backend: qbe.Backend = None,
) -> qp.Schedule:
    """Play a bichromatic Rabi on a given ion.

    For performing e.g. a Rabi flop on a given ion.
    Also plays the appropriate signal on the global channel.

    Args:
        ion_index (int): ion index, in center-index notation.
        duration (float): duration of the square pulse, in seconds.
        phase (float, optional): phase of the square pulse, relative to the current
            oscillator frame. Phase is applied to the global channel. Units of radians.
            Defaults to 0.0.
        detuning (float, optional): How much the global channel is detuned by.
            Units of Hz. Defaults to 0.0.
        sideband_order (int, optional): Which sideband order that you are
            detuning from. Multiplied by the detuning. Positive integers
            correspond to the Blue Sideband, Negative to the Red Sideband.
            This must be non-default for the detuning to be applied.
            Defaults to 0.
        individual_amp (float, optional): Amplitude of the individual beam
            RF output. In range of [-1.0, 1.0]. Defaults to 1.0.
        global_amp (float, optional): Amplitude of the global beam.
            See ``individual_amp`` for details.
        phase_insensitive (bool, optional): If the square pulse should be
            output in phase-insensitive configuration.
            Defaults to False (i.e. phase-sensitive).

    Returns:
        Schedule: Qiskit schedule describing the square pulse played on the
        given ion(s) and the global beam.
    """

    if isinstance(ion_index, int):
        ion_index = [ion_index]

    if phase_insensitive:
        raise NotImplementedError("Phase-insensitive pulses not yet supported.")

    if motional_detuning != 0.0 and sideband_order == 0.0:
        _LOGGER.warning(
            "Set detuning but didn't set sideband order, the detuning will not be used!"
        )

    if backend is None:
        backend = qp.active_backend()
    if isinstance(individual_amp,float):
        individual_amp = [individual_amp for x in stark_shift]

    global_blue_amp = (1 + sideband_amplitude_imbalance) / 2 * global_amp
    global_red_amp = (1 - sideband_amplitude_imbalance) / 2 * global_amp

    with qp.build(backend=backend) as out_sched:
        global_channel_blue, global_channel_red = qp.control_channels()

        duration_dt = qp.seconds_to_samples(duration)
        for i,shift,amp in zip(ion_index,stark_shift,individual_amp):
            individual_channel = qp.drive_channel(i)
            with qp.frequency_offset(-shift, individual_channel):
                with qp.phase_offset(-phase, individual_channel):
                    qp.play(qp.Constant(duration_dt, amp=amp), individual_channel)

        with qp.frequency_offset(carrier_detuning + motional_detuning * sideband_order, global_channel_blue):
            qp.play(qp.Constant(duration_dt, amp=global_blue_amp), global_channel_blue)

        with qp.frequency_offset(carrier_detuning - motional_detuning * sideband_order, global_channel_red):
            qp.play(qp.Constant(duration_dt, amp=global_red_amp), global_channel_red)

    return out_sched

@wf_dec.check_all_channels_same_duration
def square_rabi_by_amplitude(
    ion_index: typing.Union[int,list],
    duration: float,
    individual_amp: typing.Union[float,list],
    global_amp: float,
    phase: float = 0.0,
    detuning: float = 0.0,
    sideband_order: int = 0,
    phase_insensitive: bool = False,
    stark_shift : typing.Union[float,list] = [0.0],
    backend: qbe.Backend = None,
) -> qp.Schedule:
    """Play a square Rabi on a given ion.

    For performing e.g. a Rabi flop on a given ion.
    Also plays the appropriate signal on the global channel.

    Args:
        ion_index (int): ion index, in center-index notation.
        duration (float): duration of the square pulse, in seconds.
        phase (float, optional): phase of the square pulse, relative to the current
            oscillator frame. Phase is applied to the global channel. Units of radians.
            Defaults to 0.0.
        detuning (float, optional): How much the global channel is detuned by.
            Units of Hz. Defaults to 0.0.
        sideband_order (int, optional): Which sideband order that you are
            detuning from. Multiplied by the detuning. Positive integers
            correspond to the Blue Sideband, Negative to the Red Sideband.
            This must be non-default for the detuning to be applied.
            Defaults to 0.
        individual_amp (float, optional): Amplitude of the individual beam
            RF output. In range of [-1.0, 1.0]. Defaults to 1.0.
        global_amp (float, optional): Amplitude of the global beam.
            See ``individual_amp`` for details.
        phase_insensitive (bool, optional): If the square pulse should be
            output in phase-insensitive configuration.
            Defaults to False (i.e. phase-sensitive).

    Returns:
        Schedule: Qiskit schedule describing the square pulse played on the
        given ion(s) and the global beam.
    """

    if isinstance(ion_index, int):
        ion_index = [ion_index]

    if detuning != 0.0 and sideband_order == 0.0:
        _LOGGER.warning(
            "Set detuning but didn't set sideband order, the detuning will not be used!"
        )

    if backend is None:
        backend = qp.active_backend()

    if isinstance(individual_amp,float):
        individual_amp = [individual_amp for x in stark_shift]

    if phase_insensitive:
        indf0 = backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value
        globalf0 = backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
        with qp.build(backend=backend) as out_sched:
            duration_dt = qp.seconds_to_samples(duration)
            for i, shift, amp in zip(ion_index, stark_shift, individual_amp):
                individual_channel1 = qp.drive_channel(i)
                individual_channel2 = chans.DriveChannel(qp.drive_channel(i).index + 1)
                with qp.frequency_offset(-shift + (detuning * sideband_order + globalf0 - indf0)/2, individual_channel1):
                    with qp.phase_offset(-phase, individual_channel1):
                        qp.play(qp.Constant(duration_dt, amp=amp), individual_channel1)
                with qp.frequency_offset(-(detuning * sideband_order + globalf0 - indf0)/2, individual_channel2):
                    qp.play(qp.Constant(duration_dt, amp=amp), individual_channel2)

        return out_sched

    with qp.build(backend=backend) as out_sched:
        global_channel = qp.control_channels()[0]
        duration_dt = qp.seconds_to_samples(duration)
        for i,shift,amp in zip(ion_index,stark_shift,individual_amp):
            individual_channel = qp.drive_channel(i)
            with qp.frequency_offset(-shift, individual_channel):
                with qp.phase_offset(-phase, individual_channel):
                    qp.play(qp.Constant(duration_dt, amp=amp), individual_channel)

        with qp.frequency_offset(detuning * sideband_order, global_channel):
            qp.play(qp.Constant(duration_dt, amp=global_amp), global_channel)

    return out_sched


def square_rabi_by_rabi_frequency(
    ion_index: int,
    duration: float,
    rabi_frequency: float,
    backend: qbe.Backend,
    phase_insensitive: bool = False,
    **kwargs,
):
    """Play a square rabi pulse on the given ion.

    The difference between this and :func:`square_rabi_by_amplitude` is that
    the amplitudes are specified in rabi frequencies.

    This is a wrapper around :func:`square_rabi_by_amplitude`, so see that for
    details on arguments/kwargs.
    """
    if phase_insensitive:
        raise NotImplementedError("Phase insensitive rabi's have not yet been defined.")
    rf_calib = backend.properties().rf_calibration
    rabi_global_amp = rf_calib.rabi.global_amplitude_single_tone.value
    with qp.build(backend):
        individual_channel = qp.drive_channel(ion_index)
        global_channel = qp.control_channels()[0]
        rabi_individual_amp = wf_convert.rabi_frequency_to_amplitude(
            rabi_frequency, individual_channel, backend
        )

    return square_rabi_by_amplitude(
        ion_index,
        duration,
        individual_amp=rabi_individual_amp,
        global_amp=rabi_global_amp,
        backend=backend,
        **kwargs,
    )


def rz(ion_index: int, angle: float, backend: qbe.Backend = None) -> qp.Schedule:
    """
    Apply a RZ(theta) gate to a given ion.

    Args:
        ion_index (int): Index of the ion to apply the RZ to.
        angle (float): Angle of the RZ to apply, in radians.

    Returns:
        qp.Schedule: [description]
    """
    if backend is None:
        backend = qp.active_backend()
    with qp.build(backend=backend) as out_schedule:
        qp.shift_phase(-angle, qp.drive_channel(ion_index))
    return out_schedule


def id_gate(*args, **kwargs) -> qp.Schedule:  # pylint: disable=unused-argument
    """Does nothing (Identity Gate).

    No delay needed for RFSoC.
    """
    return qp.Schedule()  # pylint: disable=unnecessary-pass


def _normalize_theta(theta: float) -> float:
    """Normalize ``theta`` to [0, 2pi)."""
    if not isinstance(theta, float):
        theta = float(theta)
    assert np.abs(theta) <= 4 * np.pi, "SK1 Rotation angles must be <= 4 pi"
    return np.mod(theta, 2 * np.pi)  # normalize to [0, 2pi]. could be negative


def _sk1_phase_calculation(theta_norm: float) -> typing.Sequence[float]:
    """Calculate the phases of each segment of an SK1 pulse, relative to phi.

    Arguments:
        theta_norm (float): theta, normalized to range [0, 2pi).
            See :func:`_normalize_theta`.
    """
    phi_correction_1 = np.remainder(-np.arccos(theta_norm / (-4 * np.pi)), 2 * np.pi)
    phi_correction_2 = np.remainder(np.arccos(theta_norm / (-4 * np.pi)), 2 * np.pi)
    # these phases are relative to phi, so the first phase will always be 0
    pulse_phases = (0, phi_correction_1, phi_correction_2)
    return pulse_phases

def _sk1_duration_calculation(
    max_rabi_frequency: float, pi_time_multiplier: float, theta_norm: float,
) -> typing.Sequence[float]:
    """Calculate the durations of each segment of an SK1 pulse.

    Arguments:
        max_rabi_frequency (float): maximum rabi frequency at full power output.
        pi_time_multiplier (float): amount to increase the pi time by (multiplied).
        theta_norm (float): normalized theta value. See :func:`_normalize_theta`.
    """
    tpi = 1 / (2 * max_rabi_frequency) * pi_time_multiplier
    rotation_time = tpi * (theta_norm / np.pi)
    t_correction = 2 * tpi
    return (rotation_time, t_correction, t_correction)


@wf_dec.check_all_channels_same_duration
@wf_dec.default_args_from_calibration(
    {
        "stark_shift": "sk1_square.stark_shift",
        "max_rabi_frequency": "sk1_square.individual_rabi_frequency",
        "pi_time_multiplier": "sk1_square.pi_time_multiplier",
    }
)
def sk1_square_by_amplitude(
    ion_index: int,
    theta: float,
    phi: float,
    individual_amplitude: float,
    global_amplitude: float,
    stark_shift: float = None,
    max_rabi_frequency: float = None,
    pi_time_multiplier: float = None,
    calculate_durations: bool = True,
    rotation_duration: float = None,
    correction_duration_0: float = None,
    correction_duration_1: float = None,
    backend: qbe.Backend = None,
) -> qp.Schedule:
    """Generate the pulses for a single-qubit SK1 gate of arbitrary angle.

    Equation:
    .. math::
        \\phi_SK1 = arccos(\\frac{-\\theta, 4*\\pi})
        R(\\theta, \\phi) = exp(-i\\theta(cos(\\phi H_x) + sin(\\phi H_y)))

        M_SK1(\\theta, \\phi) = M(2\\pi, \\phi - \\phi_SK1)R(2\\pi, \\phi + \\phi_SK1)R(\\theta, \\phi)

    Where H_x and H_y are the X & Y axes of the Bloch sphere, and together define the equator plane.
    M is the imperfect rotation actually applied, and can be thought of as R(\\theta, \\phi).

    References:
        * https://arxiv.org/pdf/1203.6392.pdf (section 4.1)
        * https://doi.org/10.1103/PhysRevA.70.052318
        * https://doi.org/10.1103/PhysRevA.72.039905

    Arguments:
        ion_index (int): index of the ion in the chain to apply the gate to.
        theta (float): the gate angle, roughly the amplitude on the Bloch sphere.
            Full transfer at ``theta = pi``. Sometimes called the polar angle.
        phi (float): the gate "phase" angle, roughly the rotation on the XY plane of
            the Bloch sphere. Sometimes called the azimuthal angle.

    Keyword Arguments:
        These are automatically filled from the RF calibrations, but can be
            overridden by specifying them as a keyword argument.
        individual_amplitude (float): amplitude of the individual beam. Scaled [-1, 1].
        global_amplitude (float): amplitude of the global beam. Scaled [-1, 1].
        stark_shift (float): The sum of the 4-photon and 2-photon stark shift
            applied during this gate, in Hertz.
            The individual oscillator will be adjusted by this frequency during the gate.
        max_rabi_frequency (float): Fastest rabi frequency time that any ion can
            experience (in Hz).
        pi_time_multiplier (float): multiplier applied to the pi time.
            Used for slowing down a gate as desired. Values should be >= 1.0.

        The duration arguments are typically automatically calculated,
        but they can be manually specified.
        To manually specify, you must set ``calculate_durations=False``,
        and then specify ``rotation_duration``, ``correction_duration_0``,
        & ``correction_duration_1``. All durations are specified in seconds.
    """
    if backend is None:
        backend = qp.active_backend()

    theta_norm = _normalize_theta(theta)
    if theta_norm == 0.0:
        # do nothing if no rotation commanded
        return qp.Schedule()
    if calculate_durations:
        # check that these values were not specified, b/c they won't be used
        assert rotation_duration is None
        assert correction_duration_0 is None
        assert correction_duration_1 is None
        pulse_durations = _sk1_duration_calculation(
            max_rabi_frequency=max_rabi_frequency,
            pi_time_multiplier=pi_time_multiplier,
            theta_norm=theta_norm,
        )
    else:
        # check durations are all specified
        assert rotation_duration is not None and rotation_duration > 0
        assert correction_duration_0 is not None
        assert correction_duration_1 is not None
        pulse_durations = (
            rotation_duration,
            correction_duration_0,
            correction_duration_1,
        )

    pulse_phases = _sk1_phase_calculation(theta_norm)
    with qp.build(backend, name="sk1 gate") as schedule:
        individual_channel = qp.drive_channel(ion_index)
        global_channel = qp.control_channels()[0]
        # Shift the frequency by the stark shift
        # pylint: disable=E1130
        with qp.frequency_offset(-stark_shift, individual_channel):
            # Do gates in the frame of the phi currently being applied
            with qp.phase_offset(-phi, individual_channel):
                for time, phase in zip(pulse_durations, pulse_phases):
                    time_dt = qp.seconds_to_samples(time)
                    qp.play(
                        qp.Constant(time_dt, global_amplitude), global_channel
                    )
                    with qp.phase_offset(-phase, individual_channel):
                        qp.play(qp.Constant(time_dt, individual_amplitude), individual_channel)
    return schedule


@wf_dec.default_args_from_calibration(
    {
        "individual_rabi_frequency": "sk1_square.individual_rabi_frequency",
    }
)
def sk1_square_by_rabi_frequency(
    ion_index: int,
    theta: float,
    phi: float,
    backend: qbe.Backend,
    ind_amp_multiplier: float = 1.0,
    individual_rabi_frequency: float = None,
    **kwargs,
):
    """Square single-qubit SK1 gate controlled by rabi frequencies.

    This is a wrapper around :func:`sk1_square_by_amplitude`,
    with the argument Rabi frequencies here converted to amplitudes.
    """

    rf_calib = backend.properties().rf_calibration
    rabi_global_amp = rf_calib.rabi.global_amplitude_single_tone.value
    with qp.build(backend):
        # convert Rabi frequencies -> corresponding amplitudes
        individual_channel = qp.drive_channel(ion_index)
        rabi_individual_amp = wf_convert.rabi_frequency_to_amplitude(
            individual_rabi_frequency, individual_channel, backend
        )

    return sk1_square_by_amplitude(
        ion_index,
        theta,
        phi,
        individual_amplitude=rabi_individual_amp*ind_amp_multiplier,
        global_amplitude=rabi_global_amp,
        backend=backend,
        **kwargs,
    )

import pulsecompiler.qiskit.pulses as euriqa_pulses

@wf_dec.check_all_channels_same_duration
@wf_dec.default_args_from_calibration(
    {
        "stark_shift": "sk1_gaussian.stark_shift",
        "sub_pulse_duration": "sk1_gaussian.sub_pulse_duration",
        "amplitude_feedforward_step": "gate_tweaks.amplitude_feedforward_step",
        "ind_amp_multiplier": "sk1_gaussian.ind_amp_multiplier"
    }
)
def sk1_gaussian(
    ion_index: int,
    theta: float,
    phi: float,
    sub_pulse_duration: float,
    stark_shift: float = None,
    ind_amp_multiplier: float = 1.0,
    amplitude_feedforward_step: float = 0.0,
    common_freq_offset: float = 0.0,
    do_global_sync: bool = True,
    backend: qbe.Backend = None
) -> qp.Schedule:
    """Generate the pulses for a single-qubit SK1 gate of arbitrary angle.

    Equation:
    .. math::
        \\phi_SK1 = arccos(\\frac{-\\theta, 4*\\pi})
        R(\\theta, \\phi) = exp(-i\\theta(cos(\\phi H_x) + sin(\\phi H_y)))

        M_SK1(\\theta, \\phi) = M(2\\pi, \\phi - \\phi_SK1)R(2\\pi, \\phi + \\phi_SK1)R(\\theta, \\phi)

    Where H_x and H_y are the X & Y axes of the Bloch sphere, and together define the equator plane.
    M is the imperfect rotation actually applied, and can be thought of as R(\\theta, \\phi).

    References:
        * https://arxiv.org/pdf/1203.6392.pdf (section 4.1)
        * https://doi.org/10.1103/PhysRevA.70.052318
        * https://doi.org/10.1103/PhysRevA.72.039905

    Arguments:
        ion_index (int): index of the ion in the chain to apply the gate to.
        theta (float): the gate angle, roughly the amplitude on the Bloch sphere.
            Full transfer at ``theta = pi``. Sometimes called the polar angle.
        phi (float): the gate "phase" angle, roughly the rotation on the XY plane of
            the Bloch sphere. Sometimes called the azimuthal angle.

    Keyword Arguments:
        These are automatically filled from the RF calibrations, but can be
            overridden by specifying them as a keyword argument.
        individual_amplitude (float): amplitude of the individual beam. Scaled [-1, 1].
        global_amplitude (float): amplitude of the global beam. Scaled [-1, 1].
        stark_shift (float): The sum of the 4-photon and 2-photon stark shift
            applied during this gate, in Hertz.
            The individual oscillator will be adjusted by this frequency during the gate.
        max_rabi_frequency (float): Fastest rabi frequency time that any ion can
            experience (in Hz).
        pi_time_multiplier (float): multiplier applied to the pi time.
            Used for slowing down a gate as desired. Values should be >= 1.0.

        The duration arguments are typically automatically calculated,
        but they can be manually specified.
        To manually specify, you must set ``calculate_durations=False``,
        and then specify ``rotation_duration``, ``correction_duration_0``,
        & ``correction_duration_1``. All durations are specified in seconds.
    """
    if backend is None:
        backend = qp.active_backend()

    myphi = phi
    theta_div = wf_convert.return_hidden_digit(theta)
    hacked_ind_amp_multiplier = ind_amp_multiplier * (1 + theta_div*amplitude_feedforward_step)
    theta_norm = np.mod(theta,2*np.pi)

    if theta_norm>np.pi:
        theta_norm = 2*np.pi - theta_norm
        myphi+=np.pi

    if theta_norm == 0.0:
        return qp.Schedule()

    rf_calib = backend.properties().rf_calibration
    global_amplitude = rf_calib.rabi.global_amplitude_single_tone.value
    rot_pulse_amplitude_multiplier = rf_calib.sk1_gaussian.rotation_pulse_multiplier.value

    t_correction = sub_pulse_duration
    pulse_durations = [t_correction, t_correction, t_correction]
    # convert Rabi frequencies -> corresponding amplitudes
    peak_rabi_frequency = 1/sub_pulse_duration * 1.5791219573725075
    with qp.build(backend):
        individual_channel = qp.drive_channel(ion_index)
        amp_correction = wf_convert.rabi_frequency_to_amplitude(
            peak_rabi_frequency, individual_channel, backend
        )
    amp_rotation = theta_norm / (np.pi*2) * amp_correction
    pulse_amplitudes = [ hacked_ind_amp_multiplier * amp_rotation * rot_pulse_amplitude_multiplier,
                        hacked_ind_amp_multiplier * amp_correction ,
                        hacked_ind_amp_multiplier * amp_correction]

    pulse_phases = _sk1_phase_calculation(theta_norm)
    fcarrier = backend.properties().rf_calibration.frequencies.global_carrier_frequency.value

    with qp.build(backend, name="sk1_gaussian gate") as schedule:
        individual_channel = qp.drive_channel(ion_index)
        global_channel = qp.control_channels()[0]
        pulse_durations_mu = list(map(qp.seconds_to_samples, pulse_durations))
        qp.play(
                euriqa_pulses.ToneDataPulse(
                    sum(pulse_durations_mu),
                    frequency_hz=common_freq_offset+fcarrier,
                    amplitude=global_amplitude,
                    output_enable=False,
                    sync=do_global_sync,
                    phase_rad=0
                ),
                global_channel
            )
        qp.delay(10, global_channel)
        with qp.frequency_offset(-stark_shift + common_freq_offset, individual_channel):
            # Do gates in the frame of the phi currently being applied
            with qp.phase_offset(-myphi, individual_channel):
                for time_dt, phase, ind_amp in zip(pulse_durations_mu, pulse_phases, pulse_amplitudes):
                    with qp.phase_offset(-phase, individual_channel):
                        qp.play(
                            pc_pulses.LinearGaussian(time_dt, ind_amp),
                            individual_channel,
                        )
        qp.delay(10, individual_channel)

    return schedule

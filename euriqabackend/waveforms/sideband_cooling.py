"""Waveforms for executing sideband cooling on a set of ions."""
from itertools import count
import typing

import numpy as np
import pulsecompiler.qiskit.backend as pc_qb
import qiskit.pulse as qp

import euriqabackend.waveforms.timing as wf_timing


def multiple_schedule_parallel_sbc(
    number_of_ions: int,
    detunings: np.ndarray,
    durations: np.ndarray,
    sbc_frequencies: typing.Dict[
        typing.Union[qp.DriveChannel, qp.ControlChannel], float
    ],
    backend: pc_qb.MinimalQiskitIonBackend,
    cooling_amplitude_global: float = 1.0,
    cooling_amplitude_individual: float = 1.0,
    index_to_channel_map: typing.Optional[typing.Dict[int, qp.DriveChannel]] = None,
) -> typing.Sequence[qp.Schedule]:
    """Generate a sequence of Parallel SBC cooling schedules.

    Each schedule corresponds to one 'pulse' of SBC. The ion(s) will be pumped back
    to the ground state with a cooling pulse between each schedule. However,
    each schedule will roughly be compiled as a separately-triggered sequence, so the
    duration of that pumping time does not need specified.

    Args:
        number_of_ions (int): number of ions to generate the sequence for
        detunings (np.ndarray): Array of the mode detunings to put into the sequence.
            Each row is a different schedule/SBC step, and the columns are ion indices
            (0-indexed) to apply the detuned pulse to.
        durations (np.ndarray): Array of the duration of the detuned pulse to play on
            a given ion. See ``detunings`` for meaning of row/column. ``durations[i][j]``
            corresponds to ``detunings[i][j]``
        sbc_frequencies (Dict[Channel, float]): Dictionary between Qiskit channels and
            the default (non-detuned) frequency that it runs at.
        backend (MinimalQiskitIonBackend): The RFSoC backend that the produced Qiskit
            :class:`Schedules` are intended to run on.
        cooling_amplitude_global (float, optional): The amplitude of the sine wave
            to apply on the global channel. Defaults to 1.0
        cooling_amplitude_individual (float, optional): The amplitude of the sine wave
            to apply. Constant for all channels. Defaults to 1.0.
        index_to_channel_map (Dict[int, DriveChannel]): Mapping between indices
            (columns) of ``detunings/durations`` and the output channel that the pulse
            plays on. Optional. If None, will use :func:`qiskit.pulse.drive_channel`
            to look up the output channel for the ion index.

    Returns:
        Sequence[Schedule]: Sideband cooling schedules constructed from the
        given detunings and amplitudes. The number of schedules corresponds to the
        number of rows in ``detunings/durations``.
    """
    assert number_of_ions >= 1
    assert len(detunings.shape) == 2
    assert detunings.shape[-1] == number_of_ions
    assert durations.shape == detunings.shape

    sbc_schedules = []
    for i, detunings_for_step in enumerate(detunings):
        # cooling_duration_s = mode_cooling_duration_initial + (i * mode_cooling_duration_increment)
        with qp.build(backend) as sbc_sched_step:
            # initialize channels to default frequency.
            for c, freq in sbc_frequencies.items():
                qp.set_frequency(freq, c)
                # delay min # of cycles to avoid simultaneous set/shift frequency
                qp.delay(4, c)
            for chan_idx, d in enumerate(detunings_for_step):
                if index_to_channel_map is not None:
                    channel = index_to_channel_map[chan_idx]
                else:
                    # assumes chan_idx is the qubit index (zero-indexed)
                    channel = qp.drive_channel(chan_idx)
                cooling_duration = durations[i, chan_idx]
                if not np.isnan(d):
                    # NaN detuning denotes nothing to play
                    qp.shift_frequency(d, channel)
                    qp.play(
                        qp.Constant(
                            qp.seconds_to_samples(cooling_duration),
                            cooling_amplitude_individual,
                        ),
                        channel,
                    )
            qp.play(
                qp.Constant(
                    qp.seconds_to_samples(max(durations[i])), cooling_amplitude_global
                ),
                qp.control_channels()[0],
            )
        sbc_schedules.append(sbc_sched_step)

    return sbc_schedules


def single_schedule_parallel_sbc(
    ion_frequencies: typing.Dict[int, float],
    config_ion_mode_map: np.ndarray,
    pulse_durations: np.ndarray,
    pulse_amplitudes_individual: np.ndarray,
    pump_duration: float,
    global_frequency: float,
    backend: pc_qb.MinimalQiskitIonBackend,
    global_amplitude: float = 1.0,
    off_detuning: float = -10e6,
    pump_buffer_duration: float = 500e-9,
) -> typing.Tuple[
    qp.Schedule, typing.Tuple[typing.Sequence[float], typing.Sequence[bool]]
]:
    """
    Generate a sideband-cooling schedule for a given set of ions.

    Args:
        ion_frequencies (typing.Dict[int, float]): Dictionary between ion index and
            the frequency that it should play at to do a red sideband pulse.
            Ion index should match the indices in the backend (zero- vs center-index).
        pulse_durations (np.ndarray): Durations of the red sideband pulse on
            each ion. The shape of this determines the number of times that
            these modes (frequencies) are cooled.
            The first index is the timestep (i.e. all with same index are started
            simultaneously), and the second index indicates the ion index (0-indexed).
            Values indicate the duration of the pulse.
        pulse_amplitudes_individual (np.ndarray): Amplitudes of the red sideband pulse
            on each ion. The shape of this determines the number of times that
            these modes (frequencies) are cooled.
            This is indexed the same as ``pulse_amplitudes``. The values are the
            amplitude played on each ion, in the range abs(amp) <= 1.0
        pump_duration (float): Duration of the pumping (cooling) to extract the
            phonon as a photon from the ion.
        global_frequency (float): Frequency of the global beam during the cooling
            sequence.
        backend (pc_qb.MinimalQiskitIonBackend): Qiskit backend that this schedule
            will be built against. Contains e.g. mapping from ions -> channels.
        individual_amplitude (float, optional): Amplitude of the individual ion
            beams, in [0.0, 1.0]. Defaults to 1.0.
        global_amplitude (float, optional): Amplitude of the global beam,
            in [0.0, 1.0]. Defaults to 1.0.
        pump_buffer_duration (float, optional): duration of the buffer between the
            end of the red sideband pulse & the pump pulse.
            The purpose of this is to give some slack for timing misalignment,
            turn-on delays, and rounding errors when converting b/w
            RFSoC & ARTIQ clocks. Units are in seconds, defaults to 500 ns (500e-9).
        off_detuning (float, optional): detuning of the global beam from resonance
            while "off" (it will actually play the whole time, just detuned).
            This is to have a steady charge environment, insteady of toggling the
            global beam on/off. Defaults to 1 MHz.

    Returns:
        typing.Tuple[
            qp.Schedule, typing.Tuple[typing.Sequence[float], typing.Sequence[bool]]
        ]: Tuple of schedule, pump timings. Pump timings are defined as two equal-length
        lists, with the first list being the duration of that timestep, and the second
        list being whether the pump should be on (i.e. pulsed) for that duration, or off
        (i.e. delayed) for the duration.
    """
    assert set(range(len(ion_frequencies.keys()))) == set(
        range(pulse_durations.shape[1])
    )
    assert pulse_amplitudes_individual.shape == pulse_durations.shape
    assert np.all(pulse_durations >= 0.0)
    assert np.all(np.abs(pulse_amplitudes_individual) <= 1.0)
    config = backend.configuration()
    ion_channels = {idx: config.individual_channel(idx) for idx in ion_frequencies}
    # TODO: FIX, hacky way of getting center index.
    # Slightly easier than checking if the config is zero-indexed or center-indexed.
    # This WILL NOT work if ion indices are missing in one of the two arrays.
    ion_zero_index = {
        center_index: i for i, center_index in enumerate(sorted(ion_frequencies.keys()))
    }
    global_channel = config.global_channels()[0]
    pump_channel = qp.MeasureChannel(
        100
    )  # add dummy channel for specifying pump channel timings

    with qp.build(backend=backend) as cooling_schedule:
        frequencies_unordered = np.zeros(len(ion_frequencies))
        counter = 0
        for ion_index, channel in ion_channels.items():
            frequencies_unordered[counter] = ion_frequencies[ion_index]
            counter +=1

        qp.set_frequency(global_frequency, global_channel)
        # for c in all_channels:
        #     # delay by 4 clock cycles to avoid simultaneous set/shift frequencies
        #     qp.delay(4, c)
        # TODO: which channels are supposed to be on-resonant and which are not?
        # play amplitude ramp
        # shift_all_resonance(on_resonance=False)

        # for each main_pulse_duration, play pi-ish pulses on each mode
        # shift_all_resonance(on_resonance=True)
        for timestep_index, ion_rsb_pulse_durations in enumerate(pulse_durations):
            for ion_index, channel in ion_channels.items():
                qp.set_frequency(frequencies_unordered[config_ion_mode_map[timestep_index,ion_index]], channel)

            max_duration = np.max(ion_rsb_pulse_durations)
            # Calculate all durations in cycles (int), to prevent float rounding
            max_duration_cycles = qp.seconds_to_samples(max_duration)
            pump_buffer_cycles = qp.seconds_to_samples(pump_buffer_duration)
            pump_duration_cycles = qp.seconds_to_samples(pump_duration)
            cooling_cycle_duration = (
                max_duration_cycles + pump_buffer_cycles * 2 + pump_duration_cycles
            )
            with qp.align_sequential():
                # wait on all channels until the end of the pump.
                with qp.align_left():
                    # play global pulse
                    qp.play(
                        qp.Constant(max_duration_cycles, global_amplitude),
                        global_channel,
                    )
                    with qp.frequency_offset(off_detuning, global_channel):
                        qp.play(
                            qp.Constant(
                                (pump_buffer_cycles * 2 + pump_duration_cycles),
                                global_amplitude,
                            ),
                            global_channel,
                        )

                    qp.delay(
                        (max_duration_cycles + pump_buffer_cycles), pump_channel,
                    )
                    qp.play(
                        qp.Constant(pump_duration_cycles, amp=1.0), pump_channel,
                    )
                    qp.delay(pump_buffer_cycles, pump_channel)

                    for ion_idx, chan in ion_channels.items():
                        idx = ion_zero_index[ion_idx]
                        # TODO: remove dead code
                        # b/c don't actually allow per-cooling-cycle pulse duration now.
                        qp.play(
                            qp.Constant(
                                cooling_cycle_duration,
                                pulse_amplitudes_individual[timestep_index][idx],
                            ),
                            chan,
                        )

    pump_timestamps, pump_edges = wf_timing.channel_timing(
        cooling_schedule, pump_channel, dt=backend.configuration().dt
    )
    pump_timestamps_diff = wf_timing.timing_to_differential(pump_timestamps)
    pump_state_bools = wf_timing.edges_to_bool(pump_edges)
    # Should be one shorter because the timestamps and edges are the same length
    # The remaining pump state is the ending state of the pump beam
    assert (len(pump_timestamps_diff) + 1) == len(pump_state_bools)

    # pad the schedule to make sure that it lasts as long as the cooling pulses.
    # prevents other outputs from starting while SBC is still happening.
    schedule_without_pump = qp.transforms.pad(
        cooling_schedule.exclude(channels=[pump_channel]),
        until=cooling_schedule.duration,
        inplace=True,
    )

    return schedule_without_pump, (pump_timestamps_diff, pump_state_bools)


def generate_detunings(
    mode_array: np.ndarray, mode_detunings: typing.List[float]
) -> np.ndarray:
    """Given an array of mode indices (``mode_array``), select the mode frequencies.

    Essentially, each element in ``mode_array`` is an index in ``mode_detunings``,
    and the returned value the de-indexed array of correct frequencies.

    Example:
        >>> generate_detunings([1, 2, 0], [2.9e6, 3.0e6, 3.1e6])
        array([3000000., 3100000., 2900000.])
    """
    assert np.nanmax(mode_array) < len(mode_detunings)
    assert np.nanmin(mode_array) >= 0

    mode_detunings_by_idx = dict(enumerate(mode_detunings))
    detunings_array = np.vectorize(mode_detunings_by_idx.get)(mode_array)

    return detunings_array


def generate_timings_increment(
    array_shape: typing.Tuple[int, int], initial_time: float, time_increment: float
) -> np.ndarray:
    assert len(array_shape) == 2

    timing_array = np.empty(array_shape)
    for step in range(array_shape[0]):
        timing_array[step, :] = initial_time + time_increment * step

    return timing_array


def relative_timings_to_absolute(
    overall_durations: np.ndarray, relative_durations: np.ndarray,
) -> np.ndarray:
    """Convert relative timings with an overall duration to full timing specs.

    This is useful for specifying the durations of each SBC pulse by the
    mode participation of that mode, so that you don't over-/under-rotate b/c
    the pulse was the wrong duration.
    """
    assert len(np.atleast_1d(np.squeeze(overall_durations)).shape) == 1
    assert len(np.atleast_1d(np.squeeze(relative_durations)).shape) == 1
    assert np.all((relative_durations >= 0.0) & (relative_durations <= 1.0))
    assert np.all(overall_durations >= 0.0)
    return np.outer(overall_durations, relative_durations)


def expand_relative_amplitudes(
    relative_amplitudes: np.ndarray,
    num_repeats: int,
    scaled_amplitude: float = 1.0,
    max_amplitude: float = 1.0,
) -> np.ndarray:
    """Given a vector of ``relative_amplitudes``, expands it to an array and scales by ``scaled_amplitude``.

    For use with ``pulse_amplitudes_individual`` argument of
    :func:`single_schedule_parallel_sbc`
    ``max_amplitude`` is defined as the maximum possible value of
    ``relative_amplitudes``.
    """
    # assert len(np.atleast_1d(np.squeeze(relative_amplitudes)).shape) == 1  # 1-D-like
    # row = repeat, column = relative amplitude
    a_repeated = np.outer(relative_amplitudes, np.ones(num_repeats)).T
    return a_repeated / max_amplitude * scaled_amplitude


def awg_riffle_to_modes(
    riffle_string: str, min_slot: int, max_slot: int, min_channel: str = "c"
):
    """Function to convert legacy AWG-style Riffle descriptions to SBC modes here.

    Args:
        riffle_string (str): AWG-style string describing the steps of the riffle.
            Example: "e 10-24\\n d 17;e 18;d 19"
        min_slot (int): the minimum slot that the ``riffle_string`` uses in its output
            sequence.
        max_slot (int): the maximum slot that the ``riffle_string`` uses in its output
            sequence.
        min_channel (str): The minimum channel that is used in ``riffle_string``.
            Modes are indexed to this. That is, if the channel is "d", then the
            index will be 1. Defaults to "c".
    """
    lines = riffle_string.strip().splitlines()
    output_array = np.full((len(lines), max_slot - min_slot + 1), np.NaN)
    for step, line in enumerate(lines):
        for out in line.strip().split(";"):
            chan, slots = out.split(" ")
            if "-" in slots:
                lower, upper = slots.split("-", maxsplit=1)
                assert int(upper) <= max_slot
                assert int(lower) >= min_slot
                idx_range = slice(int(lower) - min_slot, int(upper) - min_slot + 1)
            else:
                idx_range = int(slots) - min_slot
                assert idx_range <= max_slot

            assert len(chan) == 1
            chan_int = ord(chan.lower()) - ord(min_channel.lower())

            output_array[step, idx_range] = chan_int

    return output_array

import typing

import numpy as np

from euriqabackend.devices.keysight_awg import interpolation_functions as intfn
from euriqabackend.devices.keysight_awg import waveform as wf


def blank():
    """Constructs a blank waveform, which has no data."""
    return wf.Waveform("Blank")


def sine(
    amplitude: float,
    freq: float,
    phase: float,
    duration: float,
    PA_freq: float = None,
    t_delay: float = 0,
    delay_PA_freq: float = 0,
    name: str = "Default",
    scan_parameter: wf.ScanParameter = wf.ScanParameter.static,
    scan_values: typing.List[float] = None,
):
    """
    Constructs a simple sinusoidal waveform with given frequency, amplitude, phase, and
    duration.

    Args:
        amplitude: Amplitude (scaled to 1000)
        freq: Frequency in MHz
        phase: Phase in rad
        duration: Duration in us
        PA_freq: phase-advance frequency, at which the phase of the ion
            advances during this waveform
        t_delay: We prepend a delay time of null data to the beginning of this waveform
        delay_PA_freq: The phase-advance frequency to use during the pre-waveform delay
        name: The name of this specific sine waveform, which is set to include the
            duration, frequency, and amplitude by default
        scan_parameter: The parameter of the sine wave to scan
        scan_values: The values that the scan parameter will scan over
    """
    if scan_values is None:
        scan_values = list()
    N_scan_values = len(scan_values)
    if PA_freq is None:
        PA_freq = freq

    # The t_delay scan parameter is a special case of seg_duration where we
    # only scan the duration of the first segment
    wav_scan_parameter = (
        wf.ScanParameter.seg_duration
        if scan_parameter == wf.ScanParameter.t_delay
        else scan_parameter
    )

    # If we're scanning some parameters, the scan_values should be an array
    # of length-one arrays
    if (
        wav_scan_parameter == wf.ScanParameter.amplitude
        or scan_parameter == wf.ScanParameter.frequency
        or scan_parameter == wf.ScanParameter.phase
        or scan_parameter == wf.ScanParameter.PA_prefactor
    ):
        scan_values = [[x] for x in scan_values]

    # Name and create the waveform
    waveform_name = (
        "{0} us {1} MHz {2} V".format(duration, freq, amplitude)
        if name == "Default"
        else name
    )
    waveform = wf.Waveform(waveform_name, scan_parameter=wav_scan_parameter)

    # Prepend a delay segment of null data.
    # We need to make sure that the required duration and amplitude are not
    # impacted when we scan those parameters for the actual waveform,
    # but scanning the frequency and phase are fine.
    # We also need to make sure that the durations of the actual waveform
    # segments are not impacted by sweeping t_delay.
    if t_delay > 0:
        delay_scan_values = scan_values
        if scan_parameter == wf.ScanParameter.amplitude:
            delay_scan_values = [[0]] * N_scan_values
        elif scan_parameter == wf.ScanParameter.seg_duration:
            delay_scan_values = [t_delay] * N_scan_values
        waveform.add_segment(
            wf.Segment(
                t_delay,
                [0],
                [0],
                [0],
                [1],
                delay_PA_freq,
                scan_values=delay_scan_values,
            )
        )
    if scan_parameter == wf.ScanParameter.t_delay:
        scan_values = [duration] * N_scan_values

    # Add the segment containing the actual waveform
    waveform.add_segment(
        wf.Segment(
            duration,
            [amplitude],
            [freq],
            [phase],
            [1],
            PA_freq,
            scan_values=scan_values,
        )
    )
    return waveform


def multitone(
    amplitudes: typing.List[float],
    freqs: typing.List[float],
    phases: typing.List[float],
    duration: float,
    PA_freq: float = None,
    t_delay: float = 0,
    delay_PA_freq: float = 0,
    name: str = "Default",
    scan_parameter: wf.ScanParameter = wf.ScanParameter.static,
    scan_values: typing.List[float] = None,
):
    """
    Constructs a sum of sinusoidal waveforms with the given frequencies,
    amplitudes, phases, and duration

    Args:
        amplitudes: Amplitudes (scaled to 1000)
        freqs: Frequencies in MHz
        phases: Phases in rad
        duration: Duration in us
        PA_freq: phase-advance frequency, at which the phase of the ion advances
            during this waveform
        t_delay: We prepend a delay time of null data to the beginning of this
            waveform
        delay_PA_freq: The phase-advance frequency to use during the
            pre-waveform delay
        name: The name of this specific waveform, which is set to include the
            duration, frequency, and amplitude
            by default
        scan_parameter: The parameter of the sine wave to scan
        scan_values: The values that the scan parameter will scan over
    """
    if scan_values is None:
        scan_values = list()
    N_scan_values = len(scan_values)
    if PA_freq is None:
        PA_freq = np.mean(freqs)
    N_tones = len(amplitudes)

    # The t_delay scan parameter is a special case of seg_duration where we
    # only scan the duration of the first segment
    wav_scan_parameter = (
        wf.ScanParameter.seg_duration
        if scan_parameter == wf.ScanParameter.t_delay
        else scan_parameter
    )

    # Name and create the waveform
    waveform_name = (
        "{0} us {1} MHz {2} V".format(duration, freqs, amplitudes)
        if name == "Default"
        else name
    )
    waveform = wf.Waveform(waveform_name, scan_parameter=wav_scan_parameter)

    # Prepend a delay segment of null data.  We need to make sure that the required
    # duration and amplitude are not impacted when we scan those parameters for the
    # actual waveform, but scanning the frequency and phase are fine.
    # We also need to make sure that the durations of the actual waveform segments
    # are not impacted by sweeping t_delay.
    if t_delay > 0:
        delay_scan_values = scan_values
        if scan_parameter == wf.ScanParameter.amplitude:
            delay_scan_values = [[0] * N_tones] * N_scan_values
        elif scan_parameter == wf.ScanParameter.seg_duration:
            delay_scan_values = [t_delay] * N_scan_values
        waveform.add_segment(
            wf.Segment(
                t_delay,
                [0] * N_tones,
                [0] * N_tones,
                [0] * N_tones,
                [1] * N_tones,
                delay_PA_freq,
                scan_values=delay_scan_values,
            )
        )
    if scan_parameter == wf.ScanParameter.t_delay:
        scan_values = [duration] * N_scan_values

    # Add the segment containing the actual waveform
    waveform.add_segment(
        wf.Segment(
            duration,
            amplitudes,
            freqs,
            phases,
            [1] * N_tones,
            PA_freq,
            scan_values=scan_values,
        )
    )
    return waveform


def multisegment(
    amplitudes: typing.List[float],
    freqs: typing.List[float],
    phases: typing.List[float],
    durations: typing.List[float],
    PA_freqs: typing.List[float] = None,
    t_delay: float = 0,
    delay_PA_freq: float = 0,
    name: str = "Default",
    scan_parameter: wf.ScanParameter = wf.ScanParameter.static,
    scan_values: typing.List[typing.List[float]] = None,
):
    """
    Constructs a series of sinusoidal waveforms with the given frequencies,
    amplitudes, phases, and durations

    Args:
        amplitudes: Amplitudes (scaled to 1000)
        freqs: Frequencies in MHz
        phases: Phases in rad
        durations: Durations in us
        PA_freqs: phase-advance frequencies, at which the phase of the ion advances
            during this waveform
        t_delay: We prepend a delay time of null data to the beginning of this waveform
        delay_PA_freq: The phase-advance frequency to use during the pre-waveform delay
        name: The name of this specific waveform, which is set to include the
            duration, frequency, and amplitude by default
        scan_parameter: The parameter of the sine wave to scan
        scan_values: The values that the scan parameter will scan over
    """
    if scan_values is None or len(scan_values) == 0:
        scan_values = list()
        N_scan_values = 0
    else:
        N_scan_values = len(scan_values[0])
    if PA_freqs is None:
        PA_freqs = freqs

    # The t_delay scan parameter is a special case of seg_duration where we only
    # scan the duration of the first segment
    wav_scan_parameter = (
        wf.ScanParameter.seg_duration
        if scan_parameter == wf.ScanParameter.t_delay
        else scan_parameter
    )

    # The scan_values list as it comes in has dimensions (N_seg, N_opt) for the number
    # of segments in the waveform and the number of options in the scan.
    # Because each segment is expecting, for each option, to receive a list of
    # frequencies, amplitudes, etc. to apply to the different tones that could be
    # applied during that segment, we need to change the dimensionality of this
    # list to (N_seg, N_opt, 1).  We did something similar in the sine prototype.
    if (
        wav_scan_parameter == wf.ScanParameter.amplitude
        or scan_parameter == wf.ScanParameter.frequency
        or scan_parameter == wf.ScanParameter.phase
        or scan_parameter == wf.ScanParameter.PA_prefactor
    ):
        scan_values = [[[x] for x in y] for y in scan_values]

    # Name and create the waveform
    waveform_name = (
        "{0} us {1} MHz {2} V multisegment".format(durations, freqs, amplitudes)
        if name == "Default"
        else name
    )
    waveform = wf.Waveform(waveform_name, scan_parameter=wav_scan_parameter)

    # Prepend a delay segment of null data.  We need to make sure that the required
    # duration and amplitude are not impacted when we scan those parameters for the
    # actual waveform, but scanning the frequency and phase are fine.
    # We also need to make sure that the durations of the actual waveform
    # segments are not impacted by sweeping t_delay.
    if t_delay > 0:
        delay_scan_values = scan_values[0]
        if scan_parameter == wf.ScanParameter.amplitude:
            delay_scan_values = [[0]] * N_scan_values
        elif scan_parameter == wf.ScanParameter.seg_duration:
            delay_scan_values = [t_delay] * N_scan_values
        waveform.add_segment(
            wf.Segment(
                t_delay,
                [0],
                [0],
                [0],
                [1],
                delay_PA_freq,
                scan_values=delay_scan_values,
            )
        )
    if scan_parameter == wf.ScanParameter.t_delay:
        scan_values = [[d] * N_scan_values for d in durations]

    # Add the segments containing the actual waveform
    for i in range(len(amplitudes)):
        waveform.add_segment(
            wf.Segment(
                durations[i],
                [amplitudes[i]],
                [freqs[i]],
                [phases[i]],
                [1.0],
                PA_freqs[i],
                scan_values=scan_values[i],
            )
        )
    return waveform


def fastecho(
    mean_amplitude: float,
    amplitude_imbalance: float,
    center_freq: float,
    detuning: float,
    echo_duration: float,
    N_echo_cycles: int,
    PA_freq: float = 0,
    t_delay: float = 0,
    delay_PA_freq: float = 0,
    name: str = "Default",
    scan_parameter: wf.ScanParameter = wf.ScanParameter.static,
    scan_values_blue: typing.List[float] = None,
    scan_values_red: typing.List[float] = None,
):
    """
    This is a more specific waveform that we designed on 03 January 2019 to
    measure the Stark shift while driving on or close to the qubit transition.
    We drive on one sideband for a short period of time, then drive on the other
    sideband with an instantaneous pi phase shift between the two.
    We need to take some care at the boundary to ensure that the waveform actually
    changes by pi, regardless of the frequency difference between the adjacent segments.

    Args:
        mean_amplitude: The mean amplitudes of the two sidebands (scaled to 1000)
        amplitude_imbalance: The amplitude imbalance between the two sidebands
        center_freq: The central frequency in MHz
        detuning: The detuning of the sidebands from the center_freq
        echo_duration: The total time of one red or blue sideband pulse
        N_echo_cycles: The number of red-blue cycles to perform
        PA_freq: phase-advance frequencies, at which the phase of the ion
            advances during this waveform
        t_delay: We prepend a delay time of null data to the beginning of this waveform
        delay_PA_freq: The phase-advance frequency to use during the pre-waveform delay
        name: The name of this specific waveform, which is set to
            include the duration, frequency, and amplitude by default
        scan_parameter: The parameter of the sine wave to scan
        scan_values_blue: The values that the scan parameter will scan over in the
            blue sideband portions
        scan_values_red: The values that the scan parameter will scan over in the
            red sideband portions
    """
    if scan_values_blue is None:
        scan_values_blue = list()
        scan_values_red = list()
    N_scan_values = len(scan_values_blue)
    if PA_freq == 0:
        PA_freq = center_freq

    # The t_delay scan parameter is a special case of seg_duration where we only
    # scan the duration of the first segment
    wav_scan_parameter = (
        wf.ScanParameter.seg_duration
        if scan_parameter == wf.ScanParameter.t_delay
        else scan_parameter
    )

    # The scan_values list as it comes in has dimensions (N_opt) for the number of
    # options in the scan.
    # Because each segment is expecting, for each option, to receive a list of
    # frequencies, amplitudes, etc. to apply to the different tones that could be
    # applied during that segment, we need to change the dimensionality of this list to
    # (N_opt, 1).  We did something similar in the sine prototype.
    if (
        wav_scan_parameter == wf.ScanParameter.amplitude
        or scan_parameter == wf.ScanParameter.frequency
        or scan_parameter == wf.ScanParameter.phase
        or scan_parameter == wf.ScanParameter.PA_prefactor
    ):
        scan_values_blue = [[x] for x in scan_values_blue]
        scan_values_red = [[x] for x in scan_values_red]

    # This is somewhat of a hack.  We would like to scan the length of this waveform, but the length is quantized
    # in terms of echo cycles.  We therefore cannot scan the waveform length by scanning the segment duration; the
    # segments themselves are close to the 10 ns AWG quantization time, so that would not work.  Instead, we will scan
    # the waveform length by zeroing out the amplitudes of some of the segments depending on where they fall in the
    # waveform and what waveform length we're trying to achieve.  We therefore change the scan parameter to amplitude
    # and make functions that construct scan_values lists for the ith segment.
    wav_scan_parameter = wf.ScanParameter.amplitude if scan_parameter == wf.ScanParameter.seg_duration \
        else scan_parameter

    def scan_values_blue_fn(ii):
        if scan_parameter == wf.ScanParameter.seg_duration:
            return [([mean_amplitude+amplitude_imbalance] if ii < n else [0]) for n in scan_values_blue]
        else:
            return scan_values_blue

    def scan_values_red_fn(ii):
        if scan_parameter == wf.ScanParameter.seg_duration:
            return [([mean_amplitude-amplitude_imbalance] if ii < n else [0]) for n in scan_values_red]
        else:
            return scan_values_red

    # Name and create the waveform
    waveform_name = (
        "2x{0}x{1} us fast echo".format(N_echo_cycles, echo_duration)
        if name == "Default"
        else name
    )
    waveform = wf.Waveform(waveform_name, scan_parameter=wav_scan_parameter)

    # Prepend a delay segment of null data.
    # We need to make sure that the required duration and amplitude are not
    # impacted when we scan those parameters for the actual waveform, but scanning
    # the frequency and phase are fine.
    # We also need to make sure that the durations of the actual waveform segments
    # are not impacted by sweeping t_delay.
    if t_delay > 0:
        delay_scan_values = scan_values_blue[0]
        if scan_parameter == wf.ScanParameter.amplitude:
            delay_scan_values = [[0]] * N_scan_values
        elif scan_parameter == wf.ScanParameter.seg_duration:
            delay_scan_values = [t_delay] * N_scan_values
        waveform.add_segment(
            wf.Segment(
                t_delay,
                [0],
                [0],
                [0],
                [1],
                delay_PA_freq,
                scan_values=delay_scan_values,
            )
        )
    if scan_parameter == wf.ScanParameter.t_delay:
        scan_values_blue = [[echo_duration] * N_scan_values]
        scan_values_red = scan_values_blue

    # Add the segments containing the actual waveform
    # This is where the echoing magic happens.  
    # See the detailed description in the 03 January 2019 notebook page.
    # 10/18/19: We're changing from a one-sided echo (0-pi-0-pi-0-pi) scheme 
    # to a symmetric (0-pi-pi-0-0-pi) scheme,
    # which comes at the expense of being confined to the carrier
    symmetric = 1
    for i in range(N_echo_cycles):
        waveform.add_segment(wf.Segment(duration=echo_duration,
                                        amplitudes=[mean_amplitude+amplitude_imbalance],
                                        freqs=[center_freq+detuning],
                                        # phases=[-i * 4*np.pi * detuning * echo_duration],
                                        phases=[((symmetric*i) % 2)*np.pi],
                                        PA_prefactors=[1.],
                                        PA_freq=PA_freq,
                                        scan_values=scan_values_blue_fn(i)))
        waveform.add_segment(wf.Segment(duration=echo_duration,
                                        amplitudes=[mean_amplitude-amplitude_imbalance],
                                        freqs=[center_freq-detuning],
                                        # phases=[(i+1) * 4*np.pi * detuning * echo_duration + np.pi],
                                        phases=[((symmetric*i + 1) % 2) * np.pi],
                                        PA_prefactors=[1.],
                                        PA_freq=PA_freq,
                                        scan_values=scan_values_red_fn(i)))

    return waveform


def multisegment_AM(
    amplitude_fns: typing.List[intfn.InterpFunction],
    amplitudes: typing.List[float],
    freqs: typing.List[float],
    phases: typing.List[float],
    durations: typing.List[float],
    PA_freqs: typing.List[float] = None,
    t_delay: float = 0,
    delay_PA_freq: float = 0,
    name: str = "Default",
    scan_parameter: wf.ScanParameter = wf.ScanParameter.static,
    scan_values: typing.List[typing.List[float]] = None,
):
    """
    Constructs a series of sinusoidal waveforms with the given frequencies,
    amplitudes, phases, and durations

    Args:
        amplitudes: Amplitudes (scaled to 1000) by which the amplitude_fns are scaled
        amplitude_fns: Interpolation functions determining the waveform amplitude
            (scaled to 1000)
        freqs: Frequencies in MHz
        phases: Phases in rad
        durations: Durations in us
        PA_freqs: phase-advance frequencies, at which the phase of the ion advances
            during this waveform
        t_delay: We prepend a delay time of null data to the beginning of this waveform
        delay_PA_freq: The phase-advance frequency to use during the pre-waveform delay
        name: The name of this specific waveform, which is set to include the duration,
            frequency, and amplitude by default
        scan_parameter: The parameter of the sine wave to scan
        scan_values: The values that the scan parameter will scan over
    """
    if scan_values is None or len(scan_values) == 0:
        scan_values = list()
        N_scan_values = 0
    else:
        N_scan_values = len(scan_values[0])
    if PA_freqs is None:
        PA_freqs = freqs

    # The t_delay scan parameter is a special case of seg_duration where we only
    # scan the duration of the first segment
    wav_scan_parameter = (
        wf.ScanParameter.seg_duration
        if scan_parameter == wf.ScanParameter.t_delay
        else scan_parameter
    )

    # The scan_values list as it comes in has dimensions (N_seg, N_opt) for the
    # number of segments in the waveform and
    # the number of options in the scan.
    # Because each segment is expecting, for each option, to receive a list of
    # frequencies, amplitudes, etc. to apply to the different tones that could be
    # applied during that segment, we need
    # to change the dimensionality of this list to (N_seg, N_opt, 1).
    # We did something similar in the sine prototype.
    if (
        wav_scan_parameter == wf.ScanParameter.amplitude
        or scan_parameter == wf.ScanParameter.frequency
        or scan_parameter == wf.ScanParameter.phase
        or scan_parameter == wf.ScanParameter.PA_prefactor
    ):
        scan_values = [[[x] for x in y] for y in scan_values]

    # Name and create the waveform
    waveform_name = (
        "{0} us {1} MHz interp multisegment".format(durations, freqs)
        if name == "Default"
        else name
    )
    waveform = wf.Waveform(waveform_name, scan_parameter=wav_scan_parameter)

    # Prepend a delay segment of null data.
    # We need to make sure that the required duration and amplitude are not
    # impacted when we scan those parameters for the actual waveform, but scanning the
    # frequency and phase are fine.
    # We also need to make sure that the durations of the actual waveform segments are
    # not impacted by sweeping t_delay.
    if t_delay > 0:
        delay_scan_values = scan_values[0]
        if scan_parameter == wf.ScanParameter.amplitude:
            delay_scan_values = [[0]] * N_scan_values
        elif scan_parameter == wf.ScanParameter.seg_duration:
            delay_scan_values = [t_delay] * N_scan_values
        waveform.add_segment(
            wf.Segment(
                t_delay,
                [0],
                [0],
                [0],
                [1],
                delay_PA_freq,
                scan_values=delay_scan_values,
            )
        )
    if scan_parameter == wf.ScanParameter.t_delay:
        scan_values = [[d] * N_scan_values for d in durations]

    # Add the segments containing the actual waveform
    for i in range(len(amplitude_fns)):
        if isinstance(amplitudes[i],list):
            waveform.add_segment(
                wf.SegmentAM(
                    durations[i],
                    [amplitude_fns[i]]*len(amplitudes[i]),
                    amplitudes[i],
                    freqs[i],
                    phases[i],
                    [1.0]*len(amplitudes[i]),
                    PA_freqs[i],
                    scan_values=scan_values[i],
                )
            )
        else:
            waveform.add_segment(
                wf.SegmentAM(
                    durations[i],
                    [amplitude_fns[i]],
                    [amplitudes[i]],
                    [freqs[i]],
                    [phases[i]],
                    [1.0],
                    PA_freqs[i],
                    scan_values=scan_values[i],
                )
            )
    return waveform


def phase_advance(
    amplitude: float,
    freq: float,
    phase: float,
    duration: float,
    t_delay: float = 0,
    delay_PA_freq: float = 0,
    name: str = "Default",
    scan_parameter: wf.ScanParameter = wf.ScanParameter.static,
    scan_values: typing.List[float] = None,
):
    """
    Constructs a sinusoidal waveform with the given frequency, amplitude, phase,
    and duration that steps its phase from 0 to 2pi in steps of pi/2

    Args:
        amplitude: Amplitude (scaled to 1000)
        freq: Frequency in MHz
        phase: Phase in rad
        duration: Duration in us
        t_delay: We prepend a delay time of null data to the beginning of this waveform
        delay_PA_freq: The phase-advance frequency to use during the pre-waveform delay
        name: The name of this specific waveform, which is set to include the duration,
            frequency, and amplitude by default
        scan_parameter: The parameter of the sine wave to scan
        scan_values: The values that the scan parameter will scan over
    """
    if scan_values is None:
        scan_values = list()
    N_scan_values = len(scan_values)

    # The t_delay scan parameter is a special case of seg_duration where we only
    # scan the duration of the first segment
    wav_scan_parameter = (
        wf.ScanParameter.seg_duration
        if scan_parameter == wf.ScanParameter.t_delay
        else scan_parameter
    )

    # If we're scanning some parameters, the scan_values should be an array of
    # length-one arrays
    if (
        wav_scan_parameter == wf.ScanParameter.amplitude
        or scan_parameter == wf.ScanParameter.frequency
        or scan_parameter == wf.ScanParameter.phase
        or scan_parameter == wf.ScanParameter.PA_prefactor
    ):
        scan_values = [[x] for x in scan_values]

    # Name and create the waveform
    waveform_name = (
        "{0} us {1} MHz phase-advance".format(amplitude, freq)
        if name == "Default"
        else name
    )
    waveform = wf.Waveform(waveform_name, scan_parameter=wav_scan_parameter)

    N_segments = 5

    # Prepend a delay segment of null data.
    # We need to make sure that the required duration and amplitude are not
    # impacted when we scan those parameters for the actual waveform, but scanning
    # the frequency and phase are fine.
    # We also need to make sure that the durations of the actual waveform segments
    # are not impacted by sweeping t_delay.
    if t_delay > 0:
        delay_scan_values = scan_values
        if scan_parameter == wf.ScanParameter.amplitude:
            delay_scan_values = [[0]] * N_scan_values
        elif scan_parameter == wf.ScanParameter.seg_duration:
            delay_scan_values = [t_delay] * N_scan_values
        waveform.add_segment(
            wf.Segment(
                t_delay,
                [0],
                [0],
                [0],
                [1],
                delay_PA_freq,
                scan_values=delay_scan_values,
            )
        )
    if scan_parameter == wf.ScanParameter.t_delay:
        scan_values = [duration / N_segments] * N_scan_values

    # Add the segments containing the actual waveform
    Stark_shift = 0
    phases = np.linspace(0, 2 * np.pi, N_segments)
    for ph in phases:
        if scan_parameter == wf.ScanParameter.phase:
            waveform.add_segment(
                wf.Segment(
                    duration / N_segments,
                    [amplitude],
                    [freq],
                    [phase + ph],
                    [1],
                    freq + Stark_shift,
                    scan_values=scan_values + ph,
                )
            )
        else:
            waveform.add_segment(
                wf.Segment(
                    duration / N_segments,
                    [amplitude],
                    [freq],
                    [phase + ph],
                    [1],
                    freq + Stark_shift,
                    scan_values=scan_values,
                )
            )
    return waveform

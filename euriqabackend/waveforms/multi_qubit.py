"""Waveforms to be used in EURIQA.

This file only contains multi-qubit gates.

Currently, most parameters are passed by hand.
I would like to pull them from the configuration of the active backend with
``qiskit.pulse.active_backend()``.

These functions are all supposed to run within the post-qiskit-terra v0.15.1
:func:`qiskit.pulse.build` pulse-builder contexts, and will cause unexpected
errors if not properly called.
"""
import typing

import numpy as np
import qiskit.providers.backend as qbe
import qiskit.pulse as qp
import more_itertools
import pulsecompiler.qiskit.pulses as euriqa_pulses

import euriqabackend.devices.keysight_awg.common_types as common_types
import euriqabackend.waveforms.conversions as wf_convert
import euriqabackend.waveforms.decorators as wf_dec

# import euriqabackend.devices.keysight_awg.gate_parameters as gate_params

# TwoIons = typing.Tuple[int, int]
AmpSegmentList = typing.Sequence[
    typing.Tuple[float, typing.Union[typing.Tuple[float, float], float]]
]
"""Segment describing a duration & amplitude for an Amplitude-Modulated XX Gate.

The amplitudes are specified in terms of Rabi frequencies.

E.g.: [(50e-6, (0, 200e3)), (70e-6, 200e3), (100e-6, (200e3, 400e3))]
describes a stair-stepped amplitude sequence, roughly:
```
      /
  ___/
 /
/
```
"""

def xx_gate(
    ions: typing.Sequence[int],
    *args,
    **kwargs,
) -> qp.Schedule:
    @wf_dec.get_gate_solution(
        {
            common_types.XXModulationType.AM_interp,
            common_types.XXModulationType.AM_segmented,
        },
        {"type" : "type"},
        convert_ions_to_slots=True,
        convert_solution_units=True,
    )
    def _fetch_soln_type_fnc(ions: typing.Sequence[int], type = common_types.XXModulationType.AM_interp) -> common_types.XXModulationType:
        return type
    mytype = _fetch_soln_type_fnc(ions)

    if mytype == common_types.XXModulationType.AM_interp or mytype == common_types.XXModulationType.AM_segmented:
        return xx_am_gate(ions, *args, **kwargs)

    elif mytype == common_types.XXModulationType.AMFM_spline:
        return xx_am_fm_gate(ions, args, kwargs)

    else:
        assert("Wrong gate type.")

@wf_dec.check_all_channels_same_duration
@wf_dec.get_gate_solution(
    {
        common_types.XXModulationType.AM_interp,
        common_types.XXModulationType.AM_segmented,
    },
    {"nominal_rabi_segments": "segments", "nominal_detuning": "detuning", "sign": "sign"},
    convert_ions_to_slots=True,
    convert_solution_units=True,
)
@wf_dec.get_gate_parameters(
    {
        "sideband_amplitude_imbalance": "sideband_amplitude_imbalance",
        "individual_amplitude_imbalance": "individual_amplitude_imbalance",
        "individual_amplitude_multiplier": "individual_amplitude_multiplier",
        "global_amplitude_multiplier": "global_amplitude",
        "stark_shift": "stark_shift",
        "stark_shift_differential": "stark_shift_differential",
        "motional_frequency_adjustment": "motional_frequency_adjust",
    },
    convert_ions_to_slots=True,
    rescale_amplitude=True,
)
@wf_dec.default_args_from_calibration(
    {
    "global_amplitude" : "rabi.global_amplitude_two_tone",
    "amplitude_feedforward_step": "gate_tweaks.amplitude_feedforward_step"
    }
)
def xx_am_gate(
    ions: typing.Sequence[int],
    # specified in terms of rabi frequencies
    nominal_rabi_segments: AmpSegmentList,
    nominal_detuning: float,
    *,
    theta: float = np.pi/4,
    sign : float = 1,
    phi_individual_0: float = 0.0,
    phi_individual_1: float = 0.0,
    phi_global: float = 0.0,
    phi_motion: float = 0.0,
    positive_gate: bool = True,
    sideband_amplitude_imbalance: float = 0.0,
    individual_amplitude_imbalance: float = 0.0,
    individual_amplitude_multiplier: float = 0.0,
    global_amplitude: float = 1.0,
    global_amplitude_multiplier : float = 1.0,
    stark_shift: float = 0.0,
    stark_shift_differential: float = 0.0,
    motional_frequency_adjustment: float = 0.0,
    amplitude_feedforward_step: float = 0.0,
    backend: qbe.Backend = None,
) -> qp.Schedule:
    r"""Generate an amplitude-modulated XX gate.

    All parameters must be explicitly passed, and are not auto-pulled as they
    were in the RFCompiler.

    Takes in a series of amplitude segments & their duration, and plays
    those on the individual channels.

    If this is going to be extended, then a new function should probably be
    created.

    Approximate waveform on global tones:
    ```
     ____________
    /            \
    ```
    Example waveform on individual tones (depends on exact ``nominal_rabi_segments``):
    ```
          __
      ___/  \___
    _/          \_
    ```

    Returns:
        Schedule: Qiskit Pulse Schedule denoting the actions needed to perform a
        2-qubit XX AM gate
    """
    # *** Calculate parameters ***
    ions = tuple(more_itertools.always_iterable(ions))
    if len(ions) != 2:
        raise ValueError(
            f"Amplitude-modulated XX gate can only operate on 2 ions: got {ions}"
        )
    ion0, ion1 = ions

    if backend is None:
        backend = qp.active_backend()

    # account for the sign of the physical gate
    individual_1_invert_factor = sign if positive_gate else -sign

    # flip the gate sign if the gate angle is negative
    theta_div = wf_convert.return_hidden_digit(theta)
    mytheta = theta if theta!=None else np.pi/2
    mytheta *= (1 + theta_div * amplitude_feedforward_step)
    mytheta *= (1 + theta_div * amplitude_feedforward_step)

    if mytheta<0:
        individual_1_invert_factor = -individual_1_invert_factor

    # gate_rabi_amplitudes = gate_solution.get_gate_solution((ion0, ion1))
    durations, rabi_freqs = more_itertools.unzip(nominal_rabi_segments)

    def _normalize_rabi_freq(rabi_frequency):
        if not isinstance(rabi_frequency, (tuple, list)):
            return (rabi_frequency, rabi_frequency)
        else:
            assert len(rabi_frequency) == 2
            return rabi_frequency

    rabi_freqs = np.array([_normalize_rabi_freq(r) for r in rabi_freqs])

    # convert from rabi frequencies -> amplitudes, and normalize factors
    # from old XX gate code:
    # The gate solutions define the gate amplitude of a segment as the rabi frequency
    # of EACH sideband frequency when brought into resonance.
    # Whereas the RFCompiler defines the gate amplitude of a segment as the rabi
    # frequency when BOTH sidebands are brought into resonance.
    # We multiply the Rabi freqs by 2 at this interface to resolve the discrepancy.
    rabi_freqs_normalized = rabi_freqs*2

    durations = np.array(list(durations))

    # Calculate global parameters

    global_blue_amp = (1 + sideband_amplitude_imbalance) / 2 * global_amplitude * global_amplitude_multiplier
    global_red_amp = (1 - sideband_amplitude_imbalance) / 2 * global_amplitude * global_amplitude_multiplier

    # print("Nominal_detuning = ", nominal_detuning)
    # print("Sideband imbalance = ", sideband_amplitude_imbalance)
    # print("ind amplitude multiplier = ", individual_amplitude_multiplier)
    # print("Motional freq adjust = ", motional_frequency_adjustment)
    # print("Stark shift = ", stark_shift)
    # print("Stark shift differential = ", stark_shift_differential)
    # print("global amp = ", global_amplitude)
    # print("global amp multiplier = ", global_amplitude_multiplier)
    motional_shift = 1e6*nominal_detuning + motional_frequency_adjustment
    fcarrier = backend.properties().rf_calibration.frequencies.global_carrier_frequency.value

    global_blue_phi = +phi_motion / 2
    global_red_phi = -phi_motion / 2
    phi_global -= 0.03
    # *** Generate schedule from parameters ***
    with qp.build(backend, name="two-qubit XX gate (constant segments)") as schedule:
        # finish defining parameters using backend properties/config
        individual_0_channel = qp.drive_channel(ion0)
        individual_1_channel = qp.drive_channel(ion1)
        global_channel_blue, global_channel_red = qp.control_channels()

        # fully entangling gate <=> \theta = \pi/4
        normalized_xx_angle = np.abs(mytheta/(np.pi/4))

        # scale the stark shifts by |\theta / (\pi/2)|
        individual_0_detuning = -normalized_xx_angle * (stark_shift + stark_shift_differential / 2)
        individual_1_detuning = -normalized_xx_angle * (stark_shift - stark_shift_differential / 2)

        individual_0_amps_raw = wf_convert.get_amplitude_from_rabi_frequency_vectorized(
            rabi_freqs_normalized, wf_convert.channel_rabi_maximum_frequency(individual_0_channel,backend)
        )
        individual_1_amps_raw = wf_convert.get_amplitude_from_rabi_frequency_vectorized(
            rabi_freqs_normalized, wf_convert.channel_rabi_maximum_frequency(individual_1_channel,backend)
        )
        individual_0_amplitudes = (
            np.sqrt(normalized_xx_angle) * individual_amplitude_multiplier
            * (1 + individual_amplitude_imbalance)
            * individual_0_amps_raw
        )
        individual_1_amplitudes = (
            np.sqrt(normalized_xx_angle) * individual_amplitude_multiplier
            * (1 - individual_amplitude_imbalance)
            * individual_1_invert_factor
            * individual_1_amps_raw
        )
        if np.max(np.abs(np.array(individual_0_amplitudes))) > 0.3 or np.max(np.abs(np.array(individual_1_amplitudes))) > 0.3:
            print("over-amplitude: ", ion0, ",", ion1)
            assert False
        # temporary workaround for https://github.com/Qiskit/qiskit-terra/issues/6209
        durations_dt = np.array(list(map(qp.seconds_to_samples, durations)))
        xx_duration_dt = sum(durations_dt)

        # Global Beam, no "shaping" (gaussian??)
        # Ramp global beam up
        # qp.play(
        #     euriqa_pulses.CubicSplinePulse(global_ramp_duration_dt, 0, global_blue_amp),
        #     global_channel_blue,
        # )
        # qp.play(
        #     euriqa_pulses.CubicSplinePulse(global_ramp_duration_dt, 0, global_red_amp),
        #     global_channel_red,
        # )

        # TODO: convert global beam to a gaussian square??
        # qp.GaussianSquare(
        #   (2 * global_ramp_duration_dt + xx_duration_dt),
        #   amp=global_(red, blue)_amp, sigma=??, width=xx_duration_dt
        # )

        def freq_to_mu(val):
            precision_bits = 40
            CLOCK_FREQUENCY = float(819.2e6)
            # from in pulsecompiler
            # CLOCK_FREQUENCY = 819.2e6
            # lshift = 0
            # retval = (((val / max_value) * ((2.0 ** precision_bits) - 1)) * (2 ** lshift))
            return int( (float(val) / CLOCK_FREQUENCY) * ((2 ** precision_bits) - 1 ) )

        def mu_to_freq(val):
            precision_bits = 40
            CLOCK_FREQUENCY = float(819.2e6)
            return CLOCK_FREQUENCY * (float(val) / ((float(2 ** precision_bits)) - 1))

        fblue = mu_to_freq(freq_to_mu(fcarrier) + freq_to_mu(motional_shift))
        fred = mu_to_freq(freq_to_mu(fcarrier) - freq_to_mu(motional_shift))

        sumerr = int(freq_to_mu(fblue)+freq_to_mu(fred)-2*freq_to_mu(fcarrier))
        if sumerr!=0:
            fred = mu_to_freq(freq_to_mu(fcarrier) - freq_to_mu(motional_shift) - sumerr)

        if int(freq_to_mu(fblue)+freq_to_mu(fred)-2*freq_to_mu(fcarrier))!=0:
            print("Roundoff error in computing global frequencies.")

        qp.set_frequency(fblue, global_channel_blue)
        with qp.phase_offset(global_blue_phi, global_channel_blue):
            qp.play(
                qp.Constant(xx_duration_dt, global_blue_amp), global_channel_blue,
            )

        qp.set_frequency(fred, global_channel_red)
        with qp.phase_offset(global_red_phi, global_channel_red):
            qp.play(
                qp.Constant(xx_duration_dt, global_red_amp), global_channel_red,
            )

        qp.set_frequency(fcarrier,global_channel_red)
        qp.set_frequency(fcarrier,global_channel_blue)
        qp.delay(10, global_channel_blue)
        qp.delay(10, global_channel_red)

        # Ramp global beam down
        # qp.play(
        #     euriqa_pulses.CubicSplinePulse(
        #         global_ramp_duration_dt, global_blue_amp, -global_blue_amp
        #     ),
        #     global_channel_blue,
        # )
        # qp.play(
        #     euriqa_pulses.CubicSplinePulse(
        #         global_ramp_duration_dt, global_red_amp, -global_red_amp
        #     ),
        #     global_channel_red,
        # )

        # Individual beams
        # Wait for global beam to ramp up
        # qp.delay(global_ramp_duration_dt, individual_0_channel)
        # qp.delay(global_ramp_duration_dt, individual_1_channel)
        with qp.frequency_offset(-individual_0_detuning, individual_0_channel):
            with qp.frequency_offset(-individual_1_detuning, individual_1_channel):
                with qp.phase_offset(-phi_global - phi_individual_0, individual_0_channel):
                    with qp.phase_offset(-phi_global - phi_individual_1, individual_1_channel):
                        for dur_dt, ind0_amps, ind1_amps in zip(
                            durations_dt,
                            individual_0_amplitudes,
                            individual_1_amplitudes,
                        ):
                            ind0_ramp = ind0_amps[1] - ind0_amps[0]
                            ind1_ramp = ind1_amps[1] - ind1_amps[0]
                            qp.play(
                                euriqa_pulses.CubicSplinePulse(
                                    dur_dt, ind0_amps[0], ind0_ramp
                                ),
                                individual_0_channel,
                            )
                            qp.play(
                                euriqa_pulses.CubicSplinePulse(
                                    dur_dt, ind1_amps[0], ind1_ramp
                                ),
                                individual_1_channel,
                            )
        qp.delay(10, individual_0_channel)
        qp.delay(10, individual_1_channel)
        # qp.delay(global_ramp_duration_dt, individual_0_channel)
        # qp.delay(global_ramp_duration_dt, individual_1_channel)

    return schedule

"""Segment describing a duration & amplitude for an Amplitude-Modulated XX Gate.

The amplitudes are specified in terms of Rabi frequencies.

E.g.: [(50e-6, (0, 200e3)), (70e-6, 200e3), (100e-6, (200e3, 400e3))]
describes a stair-stepped amplitude sequence, roughly:
```
      /
  ___/
 /
/
```
"""
@wf_dec.check_all_channels_same_duration
@wf_dec.get_gate_solution(
    {
        common_types.XXModulationType.AMFM_spline
    },
    {"nominal_segments": "segments", "nominal_detuning": "detuning", "sign": "sign"},
    convert_ions_to_slots=True,
    convert_solution_units=True,
)
@wf_dec.get_gate_parameters(
    {
        "sideband_amplitude_imbalance": "sideband_amplitude_imbalance",
        "individual_amplitude_imbalance": "individual_amplitude_imbalance",
        "individual_amplitude_multiplier": "individual_amplitude_multiplier",
        "global_amplitude_multiplier": "global_amplitude",
        "stark_shift": "stark_shift",
        "stark_shift_differential": "stark_shift_differential",
        "motional_frequency_adjustment": "motional_frequency_adjust",
    },
    convert_ions_to_slots=True,
    rescale_amplitude=True,
)
@wf_dec.default_args_from_calibration(
    {"global_amplitude" : "rabi.global_amplitude_two_tone"}
)
def xx_am_fm_gate(
    ions: typing.Sequence[int],
    # specified in terms of rabi frequencies
    nominal_segments: AmpSegmentList,
    nominal_detuning: float,
    *,
    theta: float = np.pi/4,
    sign : float = 1,
    phi_individual_0: float = 0.0,
    phi_individual_1: float = 0.0,
    phi_global: float = 0.0,
    phi_motion: float = 0.0,
    positive_gate: bool = True,
    sideband_amplitude_imbalance: float = 0.0,
    individual_amplitude_imbalance: float = 0.0,
    individual_amplitude_multiplier: float = 0.0,
    global_amplitude: float = 1.0,
    global_amplitude_multiplier : float = 1.0,
    stark_shift: float = 0.0,
    stark_shift_differential: float = 0.0,
    motional_frequency_adjustment: float = 0.0,
    backend: qbe.Backend = None,
) -> qp.Schedule:
    r"""Generate an FM+AM phase-sensitive XX gate.

    Takes in a series of equispaced in time nodal points for amplitude and phase
    Computes a cubic spline interpolation of the nodal points.

    Programs the individual beams on two ions with identical amplitude data.

    Returns:
        Schedule: Qiskit Pulse Schedule denoting the actions needed to perform a
        2-qubit XX FM+AM gate
    """
    # *** Calculate parameters ***
    ions = tuple(more_itertools.always_iterable(ions))
    if len(ions) != 2:
        raise ValueError(
            f"AM+FM XX gate currently supports only 2 ions: got {ions}"
        )
    ion0, ion1 = ions

    if backend is None:
        backend = qp.active_backend()

    # account for the sign of the physical gate
    individual_1_invert_factor = sign if positive_gate else -sign

    # flip the gate sign if the gate angle is negative
    mytheta = theta if theta!=None else np.pi/2
    if mytheta<0:
        individual_1_invert_factor = -individual_1_invert_factor

    # gate_rabi_amplitudes = gate_solution.get_gate_solution((ion0, ion1))
    rabi_freqs = np.array([r[1][0] for r in nominal_segments])
    #print("rabi_freqs = ", rabi_freqs)
    df = 0
    phases = 1e-6*np.array([-r[1][1] for r in nominal_segments])

    for i in range(phases.size):
        phases[i] -= df*0.220*2*np.pi*i/phases.size

    #print(phases)
    durations = np.array([r[0] for r in nominal_segments])

    # convert from rabi frequencies -> amplitudes, and normalize factors
    # from old XX gate code:
    # The gate solutions define the gate amplitude of a segment as the rabi frequency
    # of EACH sideband frequency when brought into resonance.
    # Whereas the RFCompiler defines the gate amplitude of a segment as the rabi
    # frequency when BOTH sidebands are brought into resonance.
    # We multiply the Rabi freqs by 2 at this interface to resolve the discrepancy.
    rabi_freqs_normalized = rabi_freqs*2

    durations = np.array(list(durations))
    assert(np.size(durations) == np.size(phases) and np.size(durations) == np.size(rabi_freqs))

    # Calculate global parameters

    global_blue_amp = (1 + sideband_amplitude_imbalance) / 2 * global_amplitude * global_amplitude_multiplier
    global_red_amp = (1 - sideband_amplitude_imbalance) / 2 * global_amplitude * global_amplitude_multiplier

    # print("Nominal_detuning = ", nominal_detuning)
    # print("Sideband imbalance = ", sideband_amplitude_imbalance)
    # print("ind amplitude multiplier = ", individual_amplitude_multiplier)
    # print("Motional freq adjust = ", motional_frequency_adjustment)
    # print("Stark shift = ", stark_shift)
    # print("Stark shift differential = ", stark_shift_differential)
    # print("global amp = ", global_amplitude)
    # print("global amp multiplier = ", global_amplitude_multiplier)

    motional_shift = 1e6*nominal_detuning + motional_frequency_adjustment
    print("adjustment = ", motional_frequency_adjustment)
    fcarrier = backend.properties().rf_calibration.frequencies.global_carrier_frequency.value

    global_blue_phi = +phi_motion
    global_red_phi = -phi_motion

    # given a list of nodal points at x = 1...n
    # and assuming 0 node values elsewhere
    # computes the spline coefficients in the segments [0,1],[1,2]...[n-1,n]
    def nodes_to_spline_coeffs(lst):
        #c0 = np.append(lst,[0])
        c0 = lst
        n = np.size(c0)
        cp = np.roll(c0,-1)
        cp[-1] = 0
        cm = np.roll(c0,1)
        cm[0] = 0

        cpp = np.roll(c0,-2)
        cpp[-1] = 0
        if n>1:
            cpp[-2] = 0

        c0 = 1.0/6.0*cm + 2.0/3.0*lst + 1.0/6.0 * cp
        c1 = -0.5*cm + 0.5*cp
        c2 = (cm-2*lst+cp)/2.0
        c3 = (-cm+3*lst-3*cp+cpp)/6.0

        return np.transpose(np.array([c0,c1,c2,c3]))

    #print(nodes_to_spline_coeffs(rabi_freqs_normalized))
    #print("Phases = ",phases)
    # fully entangling gate <=> \theta = \pi/4
    normalized_xx_angle = np.abs(mytheta/(np.pi/4))

    # scale the stark shifts by |\theta / (\pi/2)|
    individual_0_detuning = -normalized_xx_angle * (stark_shift + stark_shift_differential / 2)
    individual_1_detuning = -normalized_xx_angle * (stark_shift - stark_shift_differential / 2)

    blue_phase_coeffs = nodes_to_spline_coeffs(global_blue_phi + phases)
    red_phase_coeffs = nodes_to_spline_coeffs(global_red_phi - phases)
    #print(blue_phase_coeffs)
    def freq_to_mu(val):
        precision_bits = 40
        CLOCK_FREQUENCY = float(819.2e6)
        # from in pulsecompiler
        # CLOCK_FREQUENCY = 819.2e6
        # lshift = 0
        # retval = (((val / max_value) * ((2.0 ** precision_bits) - 1)) * (2 ** lshift))
        return int( (float(val) / CLOCK_FREQUENCY) * ((2 ** precision_bits) - 1 ) )

    def mu_to_freq(val):
        precision_bits = 40
        CLOCK_FREQUENCY = float(819.2e6)
        return CLOCK_FREQUENCY * (float(val) / ((float(2 ** precision_bits)) - 1))

    fblue = mu_to_freq(freq_to_mu(fcarrier) + freq_to_mu(motional_shift))
    fred = mu_to_freq(freq_to_mu(fcarrier) - freq_to_mu(motional_shift))

    sumerr = int(freq_to_mu(fblue)+freq_to_mu(fred)-2*freq_to_mu(fcarrier))
    if sumerr!=0:
        fred = mu_to_freq(freq_to_mu(fcarrier) - freq_to_mu(motional_shift) - sumerr)

    if int(freq_to_mu(fblue)+freq_to_mu(fred)-2*freq_to_mu(fcarrier))!=0:
        print("Roundoff error in computing global frequencies.")

    # *** Generate schedule from parameters ***
    with qp.build(backend, name="two-qubit XX gate (constant segments)") as schedule:
        # finish defining parameters using backend properties/config
        individual_0_channel = qp.drive_channel(ion0)
        individual_1_channel = qp.drive_channel(ion1)
        global_channel_blue, global_channel_red = qp.control_channels()

        individual_0_amps_raw = wf_convert.get_amplitude_from_rabi_frequency_vectorized(
            rabi_freqs_normalized, wf_convert.channel_rabi_maximum_frequency(individual_0_channel,backend)
        )
        individual_1_amps_raw = wf_convert.get_amplitude_from_rabi_frequency_vectorized(
            rabi_freqs_normalized, wf_convert.channel_rabi_maximum_frequency(individual_1_channel,backend)
        )

        individual_0_a_coeffs = nodes_to_spline_coeffs(
            normalized_xx_angle * individual_amplitude_multiplier
            * (1 + individual_amplitude_imbalance)
            * individual_0_amps_raw
        )
        individual_1_a_coeffs = nodes_to_spline_coeffs(
            normalized_xx_angle * individual_amplitude_multiplier
            * (1 - individual_amplitude_imbalance)
            * individual_1_invert_factor
            * individual_1_amps_raw
        )


        # temporary workaround for https://github.com/Qiskit/qiskit-terra/issues/6209
        durations_dt = np.array(list(map(qp.seconds_to_samples, durations)))
        xx_duration_dt = sum(durations_dt)
        print(xx_duration_dt)

        first=True
        for dur_dt, phr, phb in zip(
            durations_dt,
            blue_phase_coeffs,
            red_phase_coeffs
        ):
            qp.play(
                euriqa_pulses.ToneDataPulse(
                    dur_dt,
                    frequency_hz=fblue,
                    amplitude=global_blue_amp,
                    output_enable=False,
                    phase_rad=phb,
                    sync=first
                ),
                global_channel_blue
            )
            qp.play(
                euriqa_pulses.ToneDataPulse(
                    dur_dt,
                    frequency_hz=fred,
                    amplitude=global_red_amp,
                    output_enable=False,
                    phase_rad=phr,
                    sync=first
                ),
                global_channel_red
            )
            first=False

        qp.set_frequency(fcarrier,global_channel_red)
        qp.set_frequency(fcarrier,global_channel_blue)
        qp.delay(10, global_channel_blue)
        qp.delay(10, global_channel_red)

        # Individual beams
        with qp.frequency_offset(-individual_0_detuning, individual_0_channel):
            with qp.frequency_offset(-individual_1_detuning, individual_1_channel):
                with qp.phase_offset(-phi_global - phi_individual_0, individual_0_channel):
                    with qp.phase_offset(-phi_global - phi_individual_1, individual_1_channel):
                        for dur_dt, ind0_amps, ind1_amps in zip(
                            durations_dt,
                            individual_0_a_coeffs,
                            individual_1_a_coeffs
                        ):
                            qp.play(
                                euriqa_pulses.CubicSplinePulse(
                                    dur_dt, ind0_amps[0], ind0_amps[1], ind0_amps[2]
                                ),
                                individual_0_channel,
                            )
                            qp.play(
                                euriqa_pulses.CubicSplinePulse(
                                    dur_dt, ind1_amps[0], ind1_amps[1], ind1_amps[2]
                                ),
                                individual_1_channel,
                            )
        qp.delay(10, individual_0_channel)
        qp.delay(10, individual_1_channel)

    return schedule

@wf_dec.check_all_channels_same_duration
# @wf_dec.default_args_from_calibration(
#     {"max_rabi_frequency": "rabi.maximum_frequency_individual"}
# )
@wf_dec.ignore_arguments({"theta"})
def bichromatic_drive(
    ions: typing.Sequence[int],
    duration : float,
    ind_amps : float,
    stark_shifts,
    phis,
    phi_global: float = 0.0,
    phi_motion: float = 0.0,
    sideband_amplitude_imbalance: float = 0.0,
    global_amplitude: float = 1.0,
    motional_frequency_adjustment: float = 0.0,
    backend: qbe.Backend = None,
) -> qp.Schedule:
    r"""Generate a constant bichromatic drive

    All parameters must be explicitly passed, and are not auto-pulled as they
    were in the RFCompiler.

    Returns:
        Schedule: Qiskit Pulse Schedule denoting the actions needed to perform a
        the bichromatic drive
    """
    # *** Calculate parameters ***
    # ions = tuple(more_itertools.always_iterable(ions))
    if backend is None:
        backend = qp.active_backend()

    # Calculate global parameters
    global_blue_amp = (1 + sideband_amplitude_imbalance) / 2 * global_amplitude
    global_red_amp = (1 - sideband_amplitude_imbalance) / 2 * global_amplitude
    #freq_shift = nominal_detuning + motional_frequency_adjustment
    #print("Nominal_detuning = ", nominal_detuning)
    #print("Sideband imbalance = ", sideband_amplitude_imbalance)
    #print("Motional freq adjust = ", motional_frequency_adjustment)
    #print("Stark shift = ", stark_shift)
    global_blue_phi = phi_global + phi_motion / 2
    global_red_phi = phi_global - phi_motion / 2

    # *** Generate schedule from parameters ***
    with qp.build(backend, name="two-qubit XX gate (constant segments)") as schedule:
        fcarrier = backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
        global_channel_blue, global_channel_red = qp.control_channels()

        # temporary workaround for https://github.com/Qiskit/qiskit-terra/issues/6209
        duration_dt = qp.seconds_to_samples(duration)

        #######################################################

        def freq_to_mu(val):
            precision_bits = 40
            CLOCK_FREQUENCY = float(819.2e6)
            # from in pulsecompiler
            # CLOCK_FREQUENCY = 819.2e6
            # lshift = 0
            # retval = (((val / max_value) * ((2.0 ** precision_bits) - 1)) * (2 ** lshift))
            return int( (float(val) / CLOCK_FREQUENCY) * ((2 ** precision_bits) - 1 ) )

        def mu_to_freq(val):
            precision_bits = 40
            CLOCK_FREQUENCY = float(819.2e6)
            return CLOCK_FREQUENCY * (float(val) / ((float(2 ** precision_bits)) - 1))

        fblue = mu_to_freq(freq_to_mu(fcarrier) + freq_to_mu(motional_frequency_adjustment))
        fred = mu_to_freq(freq_to_mu(fcarrier) - freq_to_mu(motional_frequency_adjustment))

        sumerr = int(freq_to_mu(fblue)+freq_to_mu(fred)-2*freq_to_mu(fcarrier))
        if sumerr!=0:
            fred = mu_to_freq(freq_to_mu(fcarrier) - freq_to_mu(motional_frequency_adjustment) - sumerr)

        if int(freq_to_mu(fblue)+freq_to_mu(fred)-2*freq_to_mu(fcarrier))!=0:
            print("Roundoff error in computing global frequencies.")

        qp.set_frequency(fblue, global_channel_blue)

        with qp.phase_offset(global_blue_phi, global_channel_blue):
            qp.play(
                qp.Constant(duration_dt, global_blue_amp), global_channel_blue,
            )
        qp.set_frequency(fred, global_channel_red)
        with qp.phase_offset(global_red_phi, global_channel_red):
            qp.play(
                qp.Constant(duration_dt, global_red_amp), global_channel_red,
            )

        qp.set_frequency(fcarrier,global_channel_red)
        qp.set_frequency(fcarrier,global_channel_blue)
        qp.delay(10, global_channel_blue)
        qp.delay(10, global_channel_red)

        #######################################################
        # Individual beams
        for i,shift,amp,phase in zip(ions,stark_shifts,ind_amps,phis):
            individual_channel = qp.drive_channel(i)
            with qp.frequency_offset(-shift, individual_channel):
                with qp.phase_offset(-phase, individual_channel):
                    qp.play(qp.Constant(duration_dt, amp=amp), individual_channel)
            qp.delay(10, individual_channel)


    return schedule

import logging

import numpy as np
import sipyco.pyon as pyon
import qiskit.pulse as qp

import euriqafrontend.interactive.rfsoc.qiskit_backend as qbe
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit
import pulsecompiler.qiskit.pulses as pc_pulses
import pulsecompiler.rfsoc.structures.splines as pc_splines
import pulsecompiler.qiskit.schedule_converter as schedule_converter
import pulsecompiler.rfsoc.tones.record as record


# logging.basicConfig(level=logging.DEBUG)

rabi_pi_time = 2.2e-6 * 2

master_ip: str = "192.168.78.152"
schedules = []
backend = qbe.get_default_qiskit_backend(master_ip, 1, with_2q_gate_solutions=False)
x_values = np.linspace(0.0, 1.0, num=30)
for amp in x_values:
    # "args" to sk1 gate
    theta = np.pi * (1 / 2)
    theta_pi_radians = theta / np.pi
    pi_time_multiplier = 5.0
    ion_index = 0
    phi = 0.0
    global_frequency = backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
    global_amp = 0.3
    individual_frequency = backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value
    individual_amp = amp

    # settings for this experiment
    use_frame_param = True
    sync = False

    pulse_phases = (
        phi,
        4.587061149216624,
        1.696124157962962
    )
    print(f"Used pulse phases: {pulse_phases}")
    if use_frame_param:
        frame_params = (phi, pulse_phases[1] - phi, (pulse_phases[2] - pulse_phases[1]))
        # frame_params = (phi, pulse_phases[1] - phi, (pulse_phases[2] - 0*pulse_phases[1]))
        #frame_params = (0.0, -np.pi/2, -np.pi/2)
        phase_params = (0.0, 0.0, 0.0)
    else:
        frame_params = (0.0, 0.0, 0.0)
        phase_params = pulse_phases

    with qp.build(backend) as schedule:
        pi_time_samples = qp.seconds_to_samples(rabi_pi_time)
        individual_channel = qp.drive_channel(ion_index)
        global_channel = qp.control_channels()[0]
        qp.play(
            pc_pulses.ToneDataPulse(
                duration_cycles=int(pi_time_samples * theta_pi_radians * pi_time_multiplier),
                frequency_hz=pc_splines.CubicSpline(
                    order0=global_frequency, order1=0, order2=0, order3=0
                ),
                amplitude=pc_splines.CubicSpline(
                    order0=global_amp, order1=0, order2=0, order3=0
                ),
                phase_rad=pc_splines.CubicSpline(
                    order0=phase_params[0], order1=0, order2=0, order3=0
                ),
                frame_rotation_rad=pc_splines.CubicSpline(
                    order0=frame_params[0], order1=0, order2=0, order3=0
                ),
                wait_trigger=False,
                sync=sync,
                output_enable=False,
                feedback_enable=False,
                frame_rotate_at_end=False,
                reset_frame=False,
                use_frame_a=True,
                use_frame_b=None,
                invert_frame_a=False,
                invert_frame_b=False,
                bypass_lookup_tables=True,
                _name="Constant",
            ),
            global_channel,
        )
        # qp.play(
        #     pc_pulses.ToneDataPulse(
        #         duration_cycles=int(pi_time_samples * 2 * pi_time_multiplier),
        #         frequency_hz=pc_splines.CubicSpline(
        #             order0=global_frequency, order1=0, order2=0, order3=0
        #         ),
        #         amplitude=pc_splines.CubicSpline(
        #             order0=global_amp, order1=0, order2=0, order3=0
        #         ),
        #         phase_rad=pc_splines.CubicSpline(
        #             order0=phase_params[1], order1=0, order2=0, order3=0
        #         ),
        #         frame_rotation_rad=pc_splines.CubicSpline(
        #             order0=frame_params[1], order1=0, order2=0, order3=0
        #         ),
        #         wait_trigger=False,
        #         sync=sync,
        #         output_enable=False,
        #         feedback_enable=False,
        #         frame_rotate_at_end=False,
        #         reset_frame=False,
        #         use_frame_a=True,
        #         use_frame_b=None,
        #         invert_frame_a=False,
        #         invert_frame_b=False,
        #         bypass_lookup_tables=True,
        #         _name="Constant",
        #     ),
        #     global_channel,
        # )
        # qp.play(
        #     pc_pulses.ToneDataPulse(
        #         duration_cycles=int(pi_time_samples * pi_time_multiplier),
        #         frequency_hz=pc_splines.CubicSpline(
        #             order0=global_frequency, order1=0, order2=0, order3=0
        #         ),
        #         amplitude=pc_splines.CubicSpline(
        #             order0=global_amp, order1=0, order2=0, order3=0
        #         ),
        #         phase_rad=pc_splines.CubicSpline(
        #             order0=phase_params[2], order1=0, order2=0, order3=0
        #         ),
        #         frame_rotation_rad=pc_splines.CubicSpline(
        #             order0=frame_params[2], order1=0, order2=0, order3=0
        #         ),
        #         wait_trigger=False,
        #         sync=sync,
        #         output_enable=False,
        #         feedback_enable=False,
        #         frame_rotate_at_end=False,
        #         reset_frame=False,
        #         use_frame_a=True,
        #         use_frame_b=None,
        #         invert_frame_a=True,
        #         invert_frame_b=False,
        #         bypass_lookup_tables=True,
        #         _name="Constant",
        #     ),
        #     global_channel,
        # )
        # individual channel
        qp.play(
            pc_pulses.ToneDataPulse(
                duration_cycles=int(pi_time_samples * theta_pi_radians * pi_time_multiplier),
                frequency_hz=pc_splines.CubicSpline(
                    order0=individual_frequency, order1=0, order2=0, order3=0
                ),
                amplitude=pc_splines.CubicSpline(
                    order0=individual_amp, order1=0, order2=0, order3=0
                ),
                phase_rad=pc_splines.CubicSpline(
                    order0=0.0, order1=0, order2=0, order3=0
                ),
                frame_rotation_rad=pc_splines.CubicSpline(
                    order0=0.0, order1=0, order2=0, order3=0
                ),
                wait_trigger=False,
                sync=sync,
                output_enable=False,
                feedback_enable=False,
                frame_rotate_at_end=False,
                reset_frame=False,
                use_frame_a=None,
                use_frame_b=None,
                invert_frame_a=False,
                invert_frame_b=False,
                bypass_lookup_tables=True,
                _name="Constant",
            ),
            individual_channel,
        )
        # qp.play(
        #     pc_pulses.ToneDataPulse(
        #         duration_cycles=int(pi_time_samples * 2 * pi_time_multiplier),
        #         frequency_hz=pc_splines.CubicSpline(
        #             order0=individual_frequency, order1=0, order2=0, order3=0
        #         ),
        #         amplitude=pc_splines.CubicSpline(
        #             order0=individual_amp, order1=0, order2=0, order3=0
        #         ),
        #         phase_rad=pc_splines.CubicSpline(
        #             order0=0.0, order1=0, order2=0, order3=0
        #         ),
        #         frame_rotation_rad=pc_splines.CubicSpline(
        #             order0=0.0, order1=0, order2=0, order3=0
        #         ),
        #         wait_trigger=False,
        #         sync=sync,
        #         output_enable=False,
        #         feedback_enable=False,
        #         frame_rotate_at_end=False,
        #         reset_frame=False,
        #         use_frame_a=None,
        #         use_frame_b=None,
        #         invert_frame_a=False,
        #         invert_frame_b=False,
        #         bypass_lookup_tables=True,
        #         _name="Constant",
        #     ),
        #     individual_channel,
        # )
        # qp.play(
        #     pc_pulses.ToneDataPulse(
        #         duration_cycles=int(pi_time_samples * 2 * pi_time_multiplier),
        #         frequency_hz=pc_splines.CubicSpline(
        #             order0=individual_frequency, order1=0, order2=0, order3=0
        #         ),
        #         amplitude=pc_splines.CubicSpline(
        #             order0=individual_amp, order1=0, order2=0, order3=0
        #         ),
        #         phase_rad=pc_splines.CubicSpline(
        #             order0=0.0, order1=0, order2=0, order3=0
        #         ),
        #         frame_rotation_rad=pc_splines.CubicSpline(
        #             order0=0.0, order1=0, order2=0, order3=0
        #         ),
        #         wait_trigger=False,
        #         sync=sync,
        #         output_enable=False,
        #         feedback_enable=False,
        #         frame_rotate_at_end=False,
        #         reset_frame=False,
        #         use_frame_a=None,
        #         use_frame_b=None,
        #         invert_frame_a=False,
        #         invert_frame_b=False,
        #         bypass_lookup_tables=True,
        #         _name="Constant",
        #     ),
        #     individual_channel,
        # )

    schedules.append(schedule)


rfsoc_submit.submit_schedule(
    schedules,
    master_ip,
    backend,
    experiment_kwargs={
        "xlabel": "Amplitude",
        "x_values": pyon.encode(x_values),
        "default_sync": True,
        "num_shots": 200,
        "lost_ion_monitor": False,
        "schedule_transform_aom_nonlinearity": False,
    },
)
print(f"Using frame? {use_frame_param}. Sync? {sync}")
# print(schedules[-1])

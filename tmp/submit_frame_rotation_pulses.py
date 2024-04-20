import logging

import sipyco.pyon as pyon
import numpy as np
import qiskit.pulse as qp
import euriqabackend.waveforms.single_qubit as oneq
import euriqabackend.waveforms.delay as delay_gate

import euriqafrontend.interactive.rfsoc.qiskit_backend as qbe
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit
import pulsecompiler.qiskit.schedule_converter as schedule_converter
import pulsecompiler.rfsoc.tones.record as record
import pulsecompiler.qiskit.pulses as pc_pulses
import pulsecompiler.rfsoc.structures.splines as pc_splines


# logging.basicConfig(level=logging.DEBUG)

rabi_pi_time = 5e-6
global_amplitude = 0.3
individual_amplitude = 0.2

master_ip: str = "192.168.78.152"
schedules = []
backend = qbe.get_default_qiskit_backend(master_ip, 1, with_2q_gate_solutions=False)
x_values = np.linspace(-np.pi, np.pi, num=11)
theta = np.pi * (1 / 2)
pi_time_multiplier = 4.25
ion_index = 0
phi = 0.0
global_frequency = backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
global_amp = 0.3
individual_frequency = backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value

for frame_rotation in x_values:
    with qp.build(backend) as schedule:
        global_channel = qp.control_channels()[0]
        individual_channel = qp.drive_channel(0)
        # qp.play(
        #     pc_pulses.ToneDataPulse(
        #         duration_cycles=4,
        #         frequency_hz=0.0,
        #         amplitude=0.0,
        #         phase_rad=0.0,
        #         frame_rotation_rad=0*frame_rotation,
        #         wait_trigger=False,
        #         sync=True,
        #         output_enable=False,
        #         feedback_enable=False,
        #         frame_rotate_at_end=False,
        #         reset_frame=False,
        #         use_frame_a=None,
        #         use_frame_b=None,
        #         invert_frame_a=False,
        #         invert_frame_b=False,
        #         bypass_lookup_tables=True,
        #     ),
        #     global_channel,
        # )
        # qp.call(
        #     oneq.square_rabi(
        #         0,
        #         0.8*rabi_pi_time / 4,
        #         individual_amp=individual_amplitude,
        #         global_amp=global_amplitude,
        #     )
        # )
        mysync = False
        qp.play(
            pc_pulses.ToneDataPulse(
                duration_cycles=int(102 * pi_time_multiplier),
                frequency_hz=pc_splines.CubicSpline(
                    order0=global_frequency, order1=0, order2=0, order3=0
                ),
                amplitude=pc_splines.CubicSpline(
                    order0=global_amp, order1=0, order2=0, order3=0
                ),
                phase_rad=pc_splines.CubicSpline(
                    order0=0, order1=0, order2=0, order3=0
                ),
                frame_rotation_rad=pc_splines.CubicSpline(
                    order0=0, order1=0, order2=0, order3=0
                ),
                wait_trigger=False,
                sync=False,
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
        qp.play(
            pc_pulses.ToneDataPulse(
                duration_cycles=int(102 * pi_time_multiplier),
                frequency_hz=pc_splines.CubicSpline(
                    order0=individual_frequency, order1=0, order2=0, order3=0
                ),
                amplitude=pc_splines.CubicSpline(
                    order0=individual_amplitude, order1=0, order2=0, order3=0
                ),
                phase_rad=pc_splines.CubicSpline(
                    order0=0.0, order1=0, order2=0, order3=0
                ),
                frame_rotation_rad=pc_splines.CubicSpline(
                    order0=0.0, order1=0, order2=0, order3=0
                ),
                wait_trigger=False,
                sync=False,
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
        qp.play(
            pc_pulses.ToneDataPulse(
                duration_cycles=4,
                frequency_hz=0.0,
                amplitude=0.0,
                phase_rad=0,
                frame_rotation_rad=frame_rotation,
                wait_trigger=False,
                sync=False,
                output_enable=False,
                feedback_enable=False,
                frame_rotate_at_end=False,
                reset_frame=False,
                use_frame_a=True,
                use_frame_b=None,
                invert_frame_a=False,
                invert_frame_b=False,
                bypass_lookup_tables=True,
            ),
            global_channel,
        )
        # qp.call(
        #     oneq.square_rabi(
        #         0,
        #         0.8*rabi_pi_time / 4,
        #         phase=0,
        #         individual_amp=individual_amplitude,
        #         global_amp=global_amplitude,
        #     )
        # )
        qp.play(
            pc_pulses.ToneDataPulse(
                duration_cycles=int(102 * pi_time_multiplier),
                frequency_hz=pc_splines.CubicSpline(
                    order0=global_frequency, order1=0, order2=0, order3=0
                ),
                amplitude=pc_splines.CubicSpline(
                    order0=global_amp, order1=0, order2=0, order3=0
                ),
                phase_rad=pc_splines.CubicSpline(
                    order0=0, order1=0, order2=0, order3=0
                ),
                frame_rotation_rad=pc_splines.CubicSpline(
                    order0=frame_rotation, order1=0, order2=0, order3=0
                ),
                wait_trigger=False,
                sync=False,
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
        qp.play(
            pc_pulses.ToneDataPulse(
                duration_cycles=int(102 * pi_time_multiplier),
                frequency_hz=pc_splines.CubicSpline(
                    order0=individual_frequency, order1=0, order2=0, order3=0
                ),
                amplitude=pc_splines.CubicSpline(
                    order0=individual_amplitude, order1=0, order2=0, order3=0
                ),
                phase_rad=pc_splines.CubicSpline(
                    order0=0.0, order1=0, order2=0, order3=0
                ),
                frame_rotation_rad=pc_splines.CubicSpline(
                    order0=0.0, order1=0, order2=0, order3=0
                ),
                wait_trigger=False,
                sync=False,
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

    schedules.append(schedule)


rfsoc_submit.submit_schedule(
    schedules,
    master_ip,
    backend,
    experiment_kwargs={
        "xlabel": "Cancellation Phase (rad / pi)",
        "x_values": pyon.encode(x_values / np.pi),
        "default_sync": True,
        "num_shots": 200,
        "lost_ion_monitor": False,
        "schedule_transform_aom_nonlinearity": False,
    },
)
print("Submitted")

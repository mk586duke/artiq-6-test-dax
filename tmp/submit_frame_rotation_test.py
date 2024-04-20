import logging

import sipyco.pyon as pyon
import numpy as np
import qiskit.pulse as qp
import pulsecompiler.qiskit.pulses as pc_pulses

import euriqafrontend.interactive.rfsoc.qiskit_backend as qbe
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit
import pulsecompiler.qiskit.schedule_converter as schedule_converter
import pulsecompiler.rfsoc.tones.record as record


logging.basicConfig(level=logging.WARNING)

master_ip: str = "192.168.78.152"
schedules = []
backend = qbe.get_default_qiskit_backend(master_ip, 1, with_2q_gate_solutions=False)
x_values = np.linspace(0.0, 2 * np.pi, num=21)
global_amp = 0.3
individual_amp = 1.0
pi_time = 2.2e-6
pi_half_duration = pi_time / 2
individual_freq = backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value
global_freq = backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
for theta in x_values:
    with qp.build(backend) as schedule:
        global_channel = qp.control_channels()[0]
        individual_channel = qp.drive_channel(0)

        def _square_gate(duration, phase, use_frame: bool):
            duration_dt = qp.seconds_to_samples(duration)
            tonedata_settings = {
                "sync": False,
                "output_enable": False,
                "feedback_enable": False,
                "frame_rotate_at_end": True,
                "reset_frame": False,
                "use_frame_a": None,
                "use_frame_b": None,
                "invert_frame_a": False,
                "invert_frame_b": False,
            }
            qp.play(
                pc_pulses.ToneDataPulse(
                    duration_dt,
                    amplitude=individual_amp,
                    phase_rad=0.0,
                    frequency_hz=individual_freq,
                    **tonedata_settings
                ),
                individual_channel,
            )
            if use_frame:
                qp.play(
                    pc_pulses.ToneDataPulse(
                        duration_dt,
                        amplitude=global_amp,
                        frequency_hz=global_freq,
                        phase_rad=0.0,
                        frame_rotation_rad=phase,
                        **tonedata_settings
                    ),
                    global_channel,
                )
            else:
                qp.play(
                    pc_pulses.ToneDataPulse(
                        duration_dt,
                        amplitude=global_amp,
                        frequency_hz=global_freq,
                        phase_rad=phase,
                        frame_rotation_rad=0.0,
                        **tonedata_settings
                    ),
                    global_channel,
                )

        # seq 1: all phase on frame param
        _square_gate(pi_half_duration, 0.0, True)
        _square_gate(10e-9, np.pi/2, True)
        _square_gate(pi_half_duration, theta, True)

        # seq 2: all phase on phase param
        # _square_gate(pi_half_duration, 0.0, False)
        # _square_gate(pi_time, np.pi/2, False)
        # _square_gate(pi_half_duration, theta, False)

    schedules.append(schedule)

rfsoc_submit.submit_schedule(
    schedules,
    master_ip,
    backend,
    experiment_kwargs={
        "xlabel": "Second pulse phase",
        "x_values": pyon.encode(x_values),
        "default_sync": True,
        "num_shots": 100,
        "lost_ion_monitor": False,
        "schedule_transform_aom_nonlinearity": False,
        "do_sbc": False,
    },
)
print("Submitted")
# print(schedules[0])
# seq = schedule_converter.OpenPulseToOctetConverter.schedule_to_octet(schedules[-1], default_lo_freq_hz={c: 100e6 for c in schedules[-1].channels})
# print(record.text_channel_sequence(seq[qp.ControlChannel(0)]))

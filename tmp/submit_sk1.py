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


logging.basicConfig(level=logging.DEBUG)

master_ip: str = "192.168.78.152"
schedules = []
backend = qbe.get_default_qiskit_backend(master_ip, 1, with_2q_gate_solutions=False)
breakpoint()
x_values = np.linspace(0.0, 1.0, num=40)
max_rabi_freq = 1/(2.2e-6)
for amp in x_values:
    with qp.build(backend) as schedule:
        qp.call(
            oneq.sk1_square(
                0,
                np.pi / 2,
                0.0,
                individual_amplitude=amp,
                global_amplitude=0.3,
                max_rabi_frequency=max_rabi_freq,
                pi_time_multiplier=5.0,
                calculate_durations=True,
                # rotation_duration=0.0,
                # correction_duration_0=10e-9,
                # correction_duration_1=10e-9,
            )
        )

    schedules.append(schedule)

rfsoc_submit.submit_schedule(schedules, master_ip, backend, experiment_kwargs={
        "xlabel": "Individual Amplitude",
        "x_values": pyon.encode(x_values),
        "default_sync": True,
        "num_shots": 200,
        "lost_ion_monitor": False,
        "schedule_transform_aom_nonlinearity": False,
        "do_sbc": False,
    },)
print("Submitted")
# print(schedules[-1])

# seq = schedule_converter.OpenPulseToOctetConverter.schedule_to_octet(schedules[-1], default_lo_freq_hz={c: 100e6 for c in schedules[-1].channels})
# print(record.text_channel_sequence(seq))


# ramsey_schedules = []
# for phase in np.linspace(-np.pi, np.pi, num=10):
#     with qp.build(backend) as ramsey_schedule:
#         rabi_duration = 12e-6
#         ind_amp = 1.0
#         global_amp = 0.3
#         qp.call(oneq.square_rabi(0, rabi_duration, individual_amp=ind_amp, global_amp=global_amp))
#         delay_gate.wait_gate_ions(100e-9, [0])
#         qp.call(oneq.square_rabi(0, rabi_duration, phase=phase, individual_amp=ind_amp, global_amp=global_amp))
#     ramsey_schedules.append(ramsey_schedule)

# rfsoc_submit.submit_schedule(ramsey_schedules, master_ip, backend, experiment_kwargs={"default_sync": True})
# seq = schedule_converter.OpenPulseToOctetConverter.schedule_to_octet(ramsey_schedules[-1], default_lo_freq_hz={c: 100e6 for c in ramsey_schedule.channels})
# print(record.text_channel_sequence(seq))


# equivalent of the two sk1 square correction pulses, without the initial phase.
# sk1_square_corrections_only = []
# for amp in np.linspace(0.0, 1.0, num=20):
#     theta = np.pi/2
#     max_rabi_frequency=33e3
#     pulse_durations = oneq._sk1_duration_calculation(max_rabi_frequency=max_rabi_frequency, pi_time_multiplier=1.0, theta_norm=theta)
#     pulse_phases = oneq._sk1_phase_calculation(theta)
#     individual_amp = 1.0
#     global_amp = 0.3
#     ion = 0
#     with qp.build(backend) as sched:
#         individual_channel = qp.drive_channel(ion)
#         global_channel = qp.control_channels()[0]
#         with qp.phase_offset(pulse_phases[1], global_channel):
#             qp.play(qp.Constant(qp.seconds_to_samples(pulse_durations[1]), individual_amp), individual_channel)
#             qp.play(qp.Constant(qp.seconds_to_samples(pulse_durations[1]), global_amp), global_channel)
#         oneq.rz(ion, np.pi)
#         with qp.phase_offset(pulse_phases[2], global_channel):
#             qp.play(qp.Constant(qp.seconds_to_samples(pulse_durations[2]), individual_amp), individual_channel)
#             qp.play(qp.Constant(qp.seconds_to_samples(pulse_durations[2]), global_amp), global_channel)
#     sk1_square_corrections_only.append(sched)

# rfsoc_submit.submit_schedule(sk1_square_corrections_only, master_ip, backend, experiment_kwargs={"default_sync": True, "num_shots": 200,})

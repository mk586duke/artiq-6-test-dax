import logging

import numpy as np
import sipyco.pyon as pyon
import qiskit.pulse as qp
import euriqabackend.waveforms.single_qubit as oneq
import euriqabackend.waveforms.delay as delay_gate

import euriqafrontend.interactive.rfsoc.qiskit_backend as qbe
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit
import pulsecompiler.qiskit.schedule_converter as schedule_converter
import pulsecompiler.rfsoc.tones.record as record


# logging.basicConfig(level=logging.DEBUG)

rabi_pi_time = 5.6e-6
global_amplitude = 0.3
individual_amplitude = 1.0

master_ip: str = "192.168.78.152"
schedules = []
backend = qbe.get_default_qiskit_backend(master_ip, 1, with_2q_gate_solutions=False)
x_values = np.linspace(0.0, 0.3, num=40)
for amp in x_values:
    # "args" to sk1 gate
    theta = np.pi * (1 / 2)
    max_rabi_frequency = 1 / (2 * rabi_pi_time)
    pi_time_multiplier = 5.0
    ion_index = 0
    calculate_durations = False
    stark_shift = 0.0
    phi = 0.0

    rotation_duration = rabi_pi_time / 2 * pi_time_multiplier
    correction_duration_0 = rabi_pi_time * 2 * pi_time_multiplier
    correction_duration_1 = rabi_pi_time * 2 * pi_time_multiplier
    # rotation_duration = 0.0
    # correction_duration_0 = 0.0
    # correction_duration_1 = 0.0


    # body here is copied from waveforms/single_qubit.py::sk1_square
    theta_norm = oneq._normalize_theta(theta)
    pulse_durations_nominal = oneq._sk1_duration_calculation(max_rabi_frequency, pi_time_multiplier, theta_norm)
    print(f"Nominal pulse durations (used={calculate_durations}): {pulse_durations_nominal}")

    # if theta_norm == 0.0:
    #     schedules.append(qp.Schedule())
    #     continue
    if calculate_durations:
        pulse_durations = pulse_durations_nominal
    else:
        pulse_durations = (
            rotation_duration,
            correction_duration_0,
            correction_duration_1,
        )

    # pulse_phases = oneq._sk1_phase_calculation(theta_norm)
    def _custom_sk1_phase_calc(theta_norm, phi):
        """Calculate the phases of each segment of an SK1 pulse, relative to phi."""
        phi_correction_1 = np.remainder(phi - np.arccos(theta_norm / (-4 * np.pi)), 2 * np.pi)
        phi_correction_2 = np.remainder(phi + np.arccos(theta_norm / (-4 * np.pi)), 2 * np.pi)
        # these phases are relative to phi, so the first phase will always be 0
        pulse_phases = (phi, phi_correction_1, phi_correction_2)
        return pulse_phases

    pulse_phases = _custom_sk1_phase_calc(theta_norm, phi)
    print(f"Used pulse phases: {pulse_phases}")

    with qp.build(backend) as schedule:

        individual_channel = qp.drive_channel(ion_index)
        global_channel = qp.control_channels()[0]
        # Shift the frequency by the stark shift
        # pylint: disable=E1130
        with qp.frequency_offset(-stark_shift, individual_channel):
            for time, phase in zip(pulse_durations, pulse_phases):
                if time == 0.0:
                    continue
                time_dt = qp.seconds_to_samples(time)
                qp.play(
                    qp.Constant(time_dt, amp), individual_channel
                )
                with qp.phase_offset(phase, global_channel):
                    qp.play(qp.Constant(time_dt, global_amplitude), global_channel)

    schedules.append(schedule)


rfsoc_submit.submit_schedule(schedules, master_ip, backend, experiment_kwargs={
    "xlabel": "Amplitude",
    "x_values": pyon.encode(x_values),
    "default_sync": True,
    "num_shots": 200,
    "lost_ion_monitor": False,
    "schedule_transform_aom_nonlinearity": False,
    "do_sbc": False,
})

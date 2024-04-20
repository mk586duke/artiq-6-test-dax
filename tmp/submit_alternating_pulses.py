import logging

import numpy as np
import qiskit.pulse as qp
import euriqabackend.waveforms.single_qubit as oneq
import euriqabackend.waveforms.delay as delay_gate

import euriqafrontend.interactive.rfsoc.qiskit_backend as qbe
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit
import pulsecompiler.qiskit.schedule_converter as schedule_converter
import pulsecompiler.rfsoc.tones.record as record


# logging.basicConfig(level=logging.DEBUG)

rabi_pi_time = 290e-9 * 2
global_amplitude = 0.3
individual_amplitude = 1.0

master_ip: str = "192.168.78.152"
schedules = []
backend = qbe.get_default_qiskit_backend(master_ip, 1, with_2q_gate_solutions=False)
for amp in np.linspace(0.0, 1.0, num=20):
    with qp.build(backend) as schedule:
        qp.call(
            oneq.square_rabi(
                0,
                rabi_pi_time / 2,
                individual_amp=amp,
                global_amp=global_amplitude
            )
        )
        qp.call(oneq.rz(0, np.pi))
        qp.call(
            oneq.square_rabi(
                0,
                rabi_pi_time / 2,
                phase=0,
                individual_amp=amp,
                global_amp=global_amplitude,
            )
        )
        qp.call(oneq.rz(0, np.pi))
        qp.call(
            oneq.square_rabi(
                0,
                rabi_pi_time / 2,
                individual_amp=amp,
                global_amp=global_amplitude
            )
        )

    schedules.append(schedule)


rfsoc_submit.submit_schedule(schedules, master_ip, backend, experiment_kwargs={"default_sync": True, "num_shots": 100, "lost_ion_monitor": False,})

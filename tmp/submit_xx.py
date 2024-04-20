import logging

import sipyco.pyon as pyon
import numpy as np
import qiskit.pulse as qp
import pulsecompiler.qiskit.schedule_converter as schedule_converter
import pulsecompiler.rfsoc.tones.record as record

import euriqafrontend.interactive.rfsoc.qiskit_backend as qbe
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit
import euriqabackend.waveforms.single_qubit as oneq
import euriqabackend.waveforms.multi_qubit as multiqb
import euriqabackend.waveforms.delay as delay_gate
import euriqafrontend.modules.rfsoc as rfsoc_mod
from pulsecompiler.qiskit.configuration import QuickConfig

master_ip: str = "192.168.78.152"
schedules = []

rfsoc_map = qbe.get_default_rfsoc_map()
num_ions = 15
# config = QuickConfig(num_ions, rfsoc_map, {11: 14, 10: 13, 9: 12, 8: 11, 7: 0, 6: 10, 5: 7, 4: 9,
#                             3: 8, 2: 2, 1: 1, 0: 3, -1: 5, -2: 4, -3: 16, -4: 17, -5: 15,
#                             -6: 18, -7: 6, -8: 19, -9: 20, -10: 21, -11: 22})
config = QuickConfig(num_ions, rfsoc_map, {
                7: 6, # bd0_01 # PMT1
                6: 18,# bd1_03 # PMT2
                5: 15, # bd1_00 # PMT3
                4: 17, # bd1_02 # PMT4
                3: 16, # bd1_01 # PMT5
                2: 4, # bd0_03 # PMT6
                1: 5, # bd0_02 # PMT7
                0: 3, # bd0_04 # PMT8
                -1: 1, # bd0_06 # PMT9
                -2: 2, # bd0_05 # PMT10
                -3: 8, # bd2_01 # PMT11
                -4: 9, # bd2_02 # PMT12
                -5: 7, # bd2_00 # PMT13
                -6: 10, # bd2_03 # PMT14
                -7: 0 # bd0_07 PMT15
                }
                )

backend = qbe.get_default_qiskit_backend(master_ip, 15, with_2q_gate_solutions=False)
backend._config = config

backend.properties().rf_calibration.merge_update(rfsoc_mod.RFSOC._load_gate_solutions(\
    "/media/euriqa-nas/CompactTrappedIonModule/Data/gate_solutions/2022_6_30/15ions_interpolated_127us.h5", 15))
#    "/media/euriqa-nas/CompactTrappedIonModule/Data/gate_solutions/2022_7_15/15ions_interpolated_127us_to_7ions.h5", 7))

# x_values = np.linspace(0.4, 0.75, num=15)
# for amp in x_values:
#     with qp.build(backend) as schedule:
#         qp.call(
#             multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=amp, global_amplitude=0.8)
#         )

#     schedules.append(schedule)

x_values = np.linspace(start=-12000, stop=10000, num=26)
for detuning in x_values:
    with qp.build(backend) as schedule:
        qp.call(
            multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, motional_frequency_adjustment=detuning)
        )
        # qp.call(
        #     multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, motional_frequency_adjustment=detuning)
        # )
        # qp.call(
        #     multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, motional_frequency_adjustment=detuning)
        # )
        # qp.call(
        #     multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, motional_frequency_adjustment=detuning)
        # )
        # qp.call(
        #     multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, motional_frequency_adjustment=detuning)
        # )
        # qp.call(
        #     multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, motional_frequency_adjustment=detuning)
        # )

    schedules.append(schedule)

# x_values = np.linspace(start=-1000, stop=1000, num=21)
# for detuning in x_values:
#     with qp.build(backend) as schedule:
#         qp.call(
#             multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, stark_shift=detuning)
#         )
#         qp.call(
#             multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, stark_shift=detuning,positive_gate=False)
#         )
#         qp.call(
#             multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, stark_shift=detuning)
#         )
#         qp.call(
#             multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, stark_shift=detuning,positive_gate=False)
#         )
#         qp.call(
#             multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, stark_shift=detuning)
#         )
#         qp.call(
#             multiqb.xx_am_gate((-6, -5), np.pi/4, individual_amplitude_multiplier=0.48, global_amplitude=0.8, stark_shift=detuning,positive_gate=False)
#         )

#     schedules.append(schedule)

rfsoc_submit.submit_schedule(schedules, master_ip, backend, experiment_kwargs={
        "xlabel": "Individual Amplitude Multiple",
        "x_values": pyon.encode(x_values),
        "default_sync": True,
        "num_shots": 125,
        "lost_ion_monitor": False,
        "schedule_transform_aom_nonlinearity": True,
        "do_sbc": True,
    },)
print("Submitted")

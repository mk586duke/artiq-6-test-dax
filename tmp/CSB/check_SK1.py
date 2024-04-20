import pathlib
import os
import json

import sipyco.pyon as pyon
import qiskit.pulse as qp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from pulsecompiler.rfsoc.structures.channel_map import RFSoCChannelMapping
from pulsecompiler.qiskit.backend import MinimalQiskitIonBackend
from pulsecompiler.qiskit.configuration import QuickConfig
from pulsecompiler.qiskit.schedule_converter import OpenPulseToOctetConverter
from pulsecompiler.qiskit.pulses import CubicSplinePulse, ToneDataPulse
from pulsecompiler.rfsoc.tones.upload import upload_channel_sequence
import pulsecompiler.rfsoc.tones.record as record
import pulsecompiler.rfsoc.structures.splines as spl
import euriqabackend.waveforms.single_qubit as single_qubit
import pulsecompiler.rfsoc.structures.channel_map as rfsoc_mapping

import euriqafrontend.interactive.rfsoc.qiskit_backend as qbe
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit
import euriqafrontend.interactive.artiq_clients as artiq_clients


def initial_sync(ch_global_vec,ch_ind_vec,freq_global,freq_ind_list,is_sync_global=True,dt_sync_mu = 4):

    if(is_sync_global):
        qp.play(ToneDataPulse(dt_sync_mu, frequency_hz = freq_global[0], amplitude = 0,
            phase_rad=0,output_enable=False,sync=True,_name="sync_global"), ch_global_vec[0])

        qp.play(ToneDataPulse(dt_sync_mu, frequency_hz = freq_global[1], amplitude = 0,
            phase_rad=0,output_enable=False,sync=True,_name="sync_global"), ch_global_vec[1])

    counter = 0
    for ch_ind in ch_ind_vec:
        qp.play(ToneDataPulse(dt_sync_mu, frequency_hz = freq_ind_list[counter], amplitude = 0,
            phase_rad=0,output_enable=False,sync=True,_name="sync_ind"), ch_ind)
        counter += 1

def zero_padding_ind(ch_ind_vec,freq_ind_list,dt_wait_mu,sync_Status=False):

    counter = 0
    for ch_ind in ch_ind_vec:
        qp.play(ToneDataPulse(dt_wait_mu, frequency_hz = freq_ind_list[counter], amplitude = 0,
            phase_rad=0,output_enable=False,sync=sync_Status), ch_ind)
        # qp.play(ToneDataPulse(dt_wait_mu, frequency_hz = freq_ind_list[counter], amplitude = 0,
            # phase_rad=0,output_enable=False,sync=sync_Status), ch_ind)
        counter += 1


def indices_pad_array(N,ind_vec_remove):
    indices_pad_list =list(range(N))
    for ind in ind_vec_remove:
        indices_pad_list.remove(ind)
    indices_pad = np.array(indices_pad_list,int)
    return indices_pad

def chan_vec_pad(chan_ind_vec,indices_pad):
    ch_vec_pad = []
    for counter in range(len(chan_ind_vec)):
        if(counter in indices_pad):
            ch_vec_pad = ch_vec_pad+[chan_ind_vec[counter]]
    return ch_vec_pad

def Rabi_flop_SK1(chan_global,freq_glob,chan_ind,freq_ind,chan_ind_pad_vec,freq_ind_pad,Amp_norm_global,
                Tpi,theta,phi,tanh_scale_factor,detuning_Hz=0, zero_pad_flag = True):
    ### This function flips at a given time the single qubit associated with chan_ind
    # 3 segments, all segments equal duration
    scale_time =2
    duration_vec = [int(scale_time*1*251),int(scale_time*1*1237),int(scale_time*1*251)]
    # duration_vec = [int(scale_time*4*251),int(scale_time*4*1237),int(scale_time*4*251)]
    # theta = np.remainder(theta + np.pi, 2 * np.pi)
    duration_total = sum(duration_vec)
    pulse_phases = single_qubit._sk1_phase_calculation(theta)
    rotation_angle = (theta/np.pi) * (Tpi / 1e6) / qp.samples_to_seconds(sum(duration_vec))
    # rect_to_tanh_scale_correction = 0.1*1.57
    rect_to_tanh_scale_correction = tanh_scale_factor

    ## rotation pulse
    spline_coeffs = np.array([[    0.0124,    2.1229,   -1.4524,    0.2862], [0.9692 , 0, 0, 0], [0.9692,   -0.0768,   -0.5937,   -0.2862]])
    spline_coeffs_scaled = np.copy(spline_coeffs) * rotation_angle * rect_to_tanh_scale_correction
    rotation_splines_ind = [spl.CubicSpline(*coeffs) for coeffs in spline_coeffs_scaled]
    # assert len(rotation_splines_ind) == 3
    qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
        phase_rad=phi + pulse_phases[0],output_enable=False,sync=False, _name="sk1_rotate_global"), chan_global)
    for spline_ind, duration in zip(rotation_splines_ind, duration_vec):
        qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
            phase_rad=0,output_enable=False,sync=False, _name="sk1_rotate_individual"), chan_ind)

    # if(theta>0.0):
    #     ## correction pulse 0 (the first 2pi pulse)
    #     correction_amplitude = 2 * (Tpi/1e6) / qp.samples_to_seconds(sum(duration_vec))
    #     correction_coeffs_scaled = spline_coeffs * correction_amplitude * rect_to_tanh_scale_correction
    #     correction_splines_ind = [spl.CubicSpline(*coeffs) for coeffs in correction_coeffs_scaled]
    #     qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
    #         phase_rad=phi + pulse_phases[1],output_enable=False,sync=False, _name="sk1_correct1_global"), chan_global)
    #     for spline_ind, duration in zip(correction_splines_ind, duration_vec):
    #         qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
    #             phase_rad=0,output_enable=False,sync=False, _name="sk1_correct1_individual"), chan_ind)

    #     # # # correction pulse 1  (the second 2pi pulse)
    #     qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
    #         phase_rad=phi + pulse_phases[2],output_enable=False,sync=False, _name="sk1_correct2_global"), chan_global)
    #     for spline_ind, duration in zip(correction_splines_ind, duration_vec):
    #         qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
    #             phase_rad=0,output_enable=False,sync=False, _name="sk1_correct2_individual"), chan_ind)

    #     ##padding all other channels
    #     if(zero_pad_flag):
    #         zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(3*duration_total))

    # else:
    #     if(zero_pad_flag):
    #         zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(1*duration_total))
    # return 3*duration_total

    if(zero_pad_flag):
        zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(1*duration_total))
    return 1*duration_total

### Run flags
num_ions=15
scan_type = 'Angle'   #### 'Angle'  |  'Ngates'  |' SK1_amp'
is_submit_via_python = False
is_debug_sequence = False ### only for debugging
is_SK1 = [True] *num_ions
is_get_dataset_vals =True


### channels and parameters kept fixed for the 3 ions experiment
master_ip: str = "192.168.78.152"
rfsoc_map = qbe.get_default_rfsoc_map()#rfsoc_mapping.RFSoCChannelMapping(self.rfsoc_board_description)
config = QuickConfig(num_ions, rfsoc_map, {11: 14, 10: 13, 9: 12, 8: 11, -7: 0, -6: 10, -5: 7, -4: 9,
                                        -3: 8, -2: 2, -1: 1, 0: 3, 1: 5, 2: 4, 3: 16, 4: 17, 5: 15,
                                        6: 18, 7: 6, -8: 19, -9: 20, -10: 21, -11: 22})
qp_backend = qbe.get_default_qiskit_backend(master_ip, num_ions, with_2q_gate_solutions=False)
channel_map = qbe.get_default_rfsoc_map()
qp_backend._config = config


Amp_norm_global = 0.45
Amp_norm_global_SK1 = 0.45
SK1_nominal_duration = 5217
freq_offset = 0
if(Amp_norm_global>=0.5):
    print("error!! amplitude of two tones signal is saturated above 0.5")

##### calibrated at amp global 0.45 and am ind=0.06. then we get nominal values of reference T_pi_vec_microsec =  np.array([7.22658878e-06, 5.43244127e-06, 5.44120399e-06, 4.93449343e-06, 4.48604338e-06, 4.26321191e-06, 4.24458272e-06, 3.77700894e-06, 4.32514211e-06, 4.09906662e-06, 4.58169481e-06, 5.18988991e-06, 5.08513471e-06, 5.10058139e-06, 1.11003692e-05])
dataset_db = artiq_clients.get_artiq_dataset_db(master_ip)
pi_time_full_array = dataset_db.get("global.RFSOC.rabi_pi_time_individual")
if(np.mod(num_ions,2)==1):
    ind_pi_time_min = int(16-0.5*(num_ions-1))
    ind_pi_time_max = int(16+0.5*(num_ions-1)+1)
T_pi_vec_microsec = (1e6)*pi_time_full_array[ind_pi_time_min:ind_pi_time_max]
print(T_pi_vec_microsec)
# T_pi_vec_microsec =  (1e6)*np.array([7.32938498e-06, 5.16611436e-06, 5.19397383e-06, 4.83398479e-06, 4.41082791e-06, 4.20976995e-06, 4.23389503e-06, 3.77215628e-06, 4.35971590e-06, 4.06041126e-06, 4.55460724e-06, 5.31376336e-06, 5.26995719e-06, 5.10301175e-06, 1.01712217e-05])
### Main Schedule
schedules = []

N_gate_rep_vec = [1]
theta_vec = [np.pi]
amp_scale_factor_vec = [1]

if(scan_type == 'Angle'):
    theta_vec = np.linspace(0,2.5*np.pi,21)
    # theta_vec = np.array([0,np.pi/2,np.pi])
elif(scan_type == 'SK1_amp'):
    amp_scale_factor_vec = np.linspace(0.6,1.3,21)

counter_scan = 0
for N_gate_rep in N_gate_rep_vec:
    for theta in theta_vec:
        for amp_scale_factor in amp_scale_factor_vec:
            with qp.build(backend=qp_backend) as out_sched:
                # tanh_scale_factor = 1*amp_scale_factor/np.array([1,1.010864143352493, 1.0155739246100977, 0.9758973924694272, 0.9874321962148809, 0.9442526136868277, 0.9259506107007508, 0.9710440381630175, 0.9366137613959852, 0.9451632797230591, 0.9442722156408533, 0.953916605542164, 0.9171838005255846, 0.9375556730141229,1])# measured values for the 4 times longer pulse
                tanh_scale_factor = 1*amp_scale_factor*np.array([1.3600201237400071, 1.1950996355428771, 1.2113753731022272, 1.2383676179536554, 1.2062325219228924, 1.2508120271835235, 1.2412135840340144, 1.2143770591832472, 1.2269968285219433, 1.2339312143322876, 1.268860667538022, 1.320941271693219, 1.3328926890614001, 1.2779522378964572, 1.2193051873794316])*np.array([1.0321351157103864, 0.9425569580481474, 0.9058591732980279, 0.886550587279146, 0.9718776331008022, 0.9704353340062185, 1.0055884673663404, 0.9923371692029463, 0.9928503626525393, 1.006524837920474, 0.9749731393511611, 0.9882763500231294, 0.9946508580543696, 0.993671093059356, 0.842488702724964])/np.array([1,1.0758834270405784, 1.0379432377036002, 1.0319228604546027, 1.1142506099801377, 1.112166824676328, 1.1348525811951264, 1.107487987172318, 1.1334525543846805, 1.1199029511533098, 1.1627625154216001, 1.235940862253538, 1.2644554996186386, 1.2017094803718202,1])# measured values for the short pulse

                ### carrier
                freq_carrier_glob = qp_backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
                freq_global_vec_SB0 = np.array([freq_carrier_glob,freq_carrier_glob]) # red and blue tones' frequencies in Hz for carrier
                freq_ind_vec = np.array([qp_backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value]*num_ions)

                ## assign channels
                chan_glob_tone_red = qp.control_channels()[0]
                chan_glob_tone_blue = qp.control_channels()[1]
                chan_glob_vec = [chan_glob_tone_red, chan_glob_tone_blue]
                chan_ind_vec = []
                for ion_idx in range(-int((num_ions-1)/2),1+int((num_ions-1)/2)):
                    chan_ind_vec = chan_ind_vec + [qp_backend.configuration().individual_channel(ion_idx, 0)]
                ##########################################



                #########   Useful functions    ######
                def sequential_sk1(applied_gates, pi_times, thetas, phis):
                    duration_SK_sequential = 0
                    for i, do_gate in enumerate(applied_gates):
                        if do_gate:
                            indices_pad_Rabi= indices_pad_array(len(freq_ind_vec),[i])
                            channels_pad_Rabi = chan_vec_pad(chan_ind_vec,indices_pad_Rabi)+[chan_glob_tone_blue]
                            freq_pad_Rabi = np.hstack((freq_ind_vec[indices_pad_Rabi],freq_global_vec_SB0[1]))
                            duration_SK1 = Rabi_flop_SK1(
                                chan_glob_tone_red,
                                freq_carrier_glob,
                                chan_ind_vec[i], freq_ind_vec[i], channels_pad_Rabi, freq_pad_Rabi,
                                Amp_norm_global=Amp_norm_global_SK1,
                                Tpi=pi_times[i],
                                theta=thetas[i],
                                phi=phis[i],
                                detuning_Hz=0,
                                zero_pad_flag=True,
                                tanh_scale_factor=tanh_scale_factor[i],
                            )
                            duration_SK_sequential = duration_SK_sequential + duration_SK1
                    return duration_SK_sequential

                def delay_all(T_delay):
                    ### T_delay is given in machiine units
                    channels_pad_delay = chan_ind_vec+chan_glob_vec
                    freq_pad_delay = np.hstack((freq_ind_vec,freq_global_vec_SB0))
                    zero_padding_ind(channels_pad_delay,freq_pad_delay,int(T_delay))
                ##########################################



                #########   Experimental sequence   ######
                ## Initial sync of phases (b.c. sync is off)
                print('')
                print('Exp no.' +str(counter_scan+1))
                initial_sync(ch_global_vec = chan_glob_vec, ch_ind_vec = chan_ind_vec,freq_global = freq_global_vec_SB0,freq_ind_list=freq_ind_vec)

                theta_vec_SK1 = [theta]*num_ions
                phi_vec = [0]*num_ions
                for n in range(N_gate_rep):
                    T_ramsey1 = sequential_sk1(applied_gates=is_SK1, pi_times=T_pi_vec_microsec,
                                    thetas=np.array(theta_vec_SK1), phis=np.array(phi_vec))
                    # delay_all(T_delay=tau_wait)
                    # print('duration of waiting in mu' +str(tau_wait))
            # print(out_sched)
            schedules.append(out_sched)
            counter_scan = counter_scan+1



default_freqs = {chan: 0 for chan in out_sched.channels}
if(is_submit_via_python):
    ### submits via the python terminal. HOwever then is limited by unnecessary communication steps of the RFSOC data
    rfsoc_submit.submit_schedule(
        schedules,
        master_ip,
        qp_backend,
        experiment_kwargs={
            "xlabel": "red and blue detuning",
            "x_values": pyon.encode(w_k_vec),
            # "xlabel": "rred blue spin phase",
            # "x_values": pyon.encode(rel_phase_vec),
            "default_sync": False,
            "num_shots": 200,
            "PMT Input String": "7:9",
            "lost_ion_monitor": False,
            "schedule_transform_aom_nonlinearity": False,
            "schedule_transform_pad_schedule": True,
            "do_sbc": True,
        },
    )
    print("Submitted")

if(is_debug_sequence):
    print(schedules[0].filter(channels=[qp.ControlChannel(0)]).instructions[:10])
    matplotlib.use("agg")
    out_sched.draw()
    print(default_freqs)
    plt.savefig('/home/euriqa/git/euriqa-artiq/tmp/CSB/schedule_displace.png')
    # compiled_schedule = OpenPulseToOctetConverter.schedule_to_octet(out_sched, default_lo_freq_hz=default_freqs)
    # compiled_schedule_global_only = {k: v for k, v in compiled_schedule.items() if isinstance(k, qp.ControlChannel)}
    # print(record.text_channel_sequence(compiled_schedule_global_only))
    # record.save_text_channel_sequence(compiled_schedule, "./tmp/Nbody_gate/NOT_working_phase_func_commands.tones")

### The plots presented after submission via the GUI in  artiq
if(scan_type=='Angle'):
        x_values = theta_vec
        x_label = "rotation angle"
elif(scan_type=='SK1_amp'):
    x_values = amp_scale_factor_vec
    x_label = "relative amp"



print("check_SK1 experiment was submitted")

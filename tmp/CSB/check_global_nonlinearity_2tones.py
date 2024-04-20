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
        counter += 1
    # print(ch_ind_vec)


def calibrate_J_pair(chan_glob_vec,f_carrier_glob,chan_ind_vec,f_carrier_ind_vec,chan_ind_pad_vec,freq_ind_pad,
    duration,LS_vec,amp_scaling_pi_time,Amp_norm_ind =0.5,Amp_norm_global_cm =0.45,amp_imbal_ratio_b2r = 1,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,n_corner = 0,):
    ### This function illuminates a pair of spins and measures the J oscillations
    ### chan_ind_vec is the list of channels of the ions that are nonzero

    ### load splines
    this_dir = pathlib.Path(__file__).parent.resolve()

    amp_blue = Amp_norm_global_cm*amp_imbal_ratio_b2r
    amp_red =  Amp_norm_global_cm*(2-amp_imbal_ratio_b2r)

    freq_red = f_carrier_glob[0]
    freq_blue = f_carrier_glob[1]

    phase_ind = 0
    phase_red_ofst = phi_spin-phi_motion_offset
    phase_blue_ofst = phi_spin+phi_motion_offset


    ### Channel assignments
    chan_glob_tone_red = chan_glob_vec[0]
    chan_glob_tone_blue = chan_glob_vec[1]
    round_fact = 1

    ### Sequence
    duration_tot_actual = int(0)
    N_pts_max_seg = 1e10
    N_segments=int(duration/N_pts_max_seg)+1
    for n_seg in range(0,N_segments):
        if(n_seg<N_segments-1):
            duration_act = N_pts_max_seg
        else:
            duration_act = int(np.mod(duration,N_pts_max_seg))
        ### GLOBAL BEAM - constant
        qp.play(ToneDataPulse(int(round_fact*duration_act), frequency_hz = freq_red, amplitude =1*amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(round_fact*duration_act), frequency_hz = freq_blue, amplitude = 0*amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            freq_ind = f_carrier_ind_vec[counter]+LS_vec[counter]

            qp.play(ToneDataPulse(int(round_fact*duration_act), frequency_hz = freq_ind, amplitude = 0.06,
                phase_rad=phase_ind,output_enable=False,sync=False), chan_ind_tone_0)

        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(round_fact*duration_act))
        duration_tot_actual = duration_tot_actual+int(round_fact*duration_act)

    return [int(duration_tot_actual)]


def Hamiltonian_evolution(chan_glob_vec,f_carrier_glob,chan_ind_vec,f_carrier_ind_vec,chan_ind_pad_vec,freq_ind_pad,
    amp_scaling_pi_time,Amp_norm_ind =0.5,Amp_norm_global_cm =0.3,amp_imbal_ratio_b2r = 1,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,LS = 0):
    ### This function generates a squeezing pulse, addressing multiple ions simultaneously by modulating the amplitude of the individual beams only.
    ### Global's amplitude & phase are assumed to be constant
    ### chan_ind_vec is the list of channels of the ions that are modulated

    ### load splines
    this_dir = pathlib.Path(__file__).parent.resolve()
    json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz106_sign-1_N13.json' #"splines_squeeze_compact__mode3_pulsedur029.json"
    # json_file = this_dir /'splines_squeeze_compact__mode3_pulsedur073.json' #"splines_squeeze_compact__mode3_pulsedur029.json"
    data_splines = json.loads(json_file.read_text())

    duration_tot = data_splines["ramp_time_mu"] #### This is given in MU for entire pulse
    # print(duration_tot)
    duration_dt = data_splines["t_seg_mu"] #### This is given in MU for one segments
    N_segments_vec = data_splines["N_segments"]
    N_segments= int(np.sum(N_segments_vec))

    #### Values for the Splines for individual beams amplitudes (without relative scaling of each channel)
    amp_ind_c0 = data_splines["splines_glob_amp"]["c0"]
    amp_ind_c1 = data_splines["splines_glob_amp"]["c1"]
    amp_ind_c2 = data_splines["splines_glob_amp"]["c2"]
    amp_ind_c3 = data_splines["splines_glob_amp"]["c3"]

    amp_scaling_ind_vec = data_splines["amp_ind_relative_scale"]  ### This is a relative scaling vector w. respect to the middle ion that is with amplitude 1

    #### Values for the Splines for individual beams frequencies
    kHz = 1e3
    freq_ind_odd_c0 = data_splines["freqlist_ind"][0]["splines"]["c0"] #### Splines for odd signed ind beam in KHz
    freq_ind_odd_c1 = data_splines["freqlist_ind"][0]["splines"]["c1"] #### Splines for odd signed ind beam in KHz
    freq_ind_odd_c2 = data_splines["freqlist_ind"][0]["splines"]["c2"] #### Splines for odd signed ind beam in KHz
    freq_ind_odd_c3 = data_splines["freqlist_ind"][0]["splines"]["c3"] #### Splines for odd signed ind beam in KHz

    freq_ind_even_c0 = data_splines["freqlist_ind"][1]["splines"]["c0"] #### Splines for odd signed ind beam in KHz
    freq_ind_even_c1 = data_splines["freqlist_ind"][1]["splines"]["c1"] #### Splines for odd signed ind beam in KHz
    freq_ind_even_c2 = data_splines["freqlist_ind"][1]["splines"]["c2"] #### Splines for odd signed ind beam in KHz
    freq_ind_even_c3 = data_splines["freqlist_ind"][1]["splines"]["c3"] #### Splines for odd signed ind beam in KHz



    ### Set amplitudes frequencies and phases
    amp_blue = Amp_norm_global_cm*amp_imbal_ratio_b2r
    amp_red =  Amp_norm_global_cm*(2-amp_imbal_ratio_b2r)

    freq_red = f_carrier_glob[0]
    freq_blue = f_carrier_glob[1]

    phase_ind = 0
    phase_red_ofst = phi_spin-phi_motion_offset
    phase_blue_ofst = phi_spin+phi_motion_offset


    ### Channel assignments
    chan_glob_tone_red = chan_glob_vec[0]
    chan_glob_tone_blue = chan_glob_vec[1]
    round_fact = 1

    ### Sequence
    duration_tot_actual = int(0)

    ### GLOBAL BEAM - constant
    for n_seg in range(0,N_segments):
        ### GLOBAL BEAM - constant
        qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_red, amplitude =1*amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_blue, amplitude = 1*amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM (Jij control) and FM (Stark shift control)
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            freq_ind_carrier = f_carrier_ind_vec[counter]+LS
            if(np.mod(counter,2)==1):
                #### odd beams
                freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_odd_c0[n_seg]),kHz*freq_ind_odd_c1[n_seg],kHz*freq_ind_odd_c2[n_seg],kHz*freq_ind_odd_c3[n_seg])
            else:
                #### even beams
                freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_even_c0[n_seg]),kHz*freq_ind_even_c1[n_seg],kHz*freq_ind_even_c2[n_seg],kHz*freq_ind_even_c3[n_seg])

            amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter]*amp_scaling_ind_vec[counter] #### Amp_norm_ind is input by the user and is ideally 1
            amp_spline_ind = spl.CubicSpline(amp_scaling_ind*amp_ind_c0[n_seg],amp_scaling_ind*amp_ind_c1[n_seg],amp_scaling_ind*amp_ind_c2[n_seg],amp_scaling_ind*amp_ind_c3[n_seg])

            qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_spline_ind, amplitude = amp_spline_ind,
                phase_rad=phase_ind,output_enable=False,sync=False), chan_ind_tone_0)

        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(round_fact*duration_dt))
        duration_tot_actual = duration_tot_actual+int(round_fact*duration_dt)

    return [int(duration_tot_actual)]


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
    duration_vec = [int(scale_time*251),int(scale_time*1237),int(scale_time*251)]
    # theta = np.remainder(theta + np.pi, 2 * np.pi)
    duration_total = sum(duration_vec)
    pulse_phases = single_qubit._sk1_phase_calculation(theta)
    rotation_angle = (theta/np.pi) * (Tpi / 1e6) / qp.samples_to_seconds(sum(duration_vec))
    # rect_to_tanh_scale_correction = 0.1*1.57
    rect_to_tanh_scale_correction = tanh_scale_factor

    ### rotation pulse
    spline_coeffs = np.array([[    0.0124,    2.1229,   -1.4524,    0.2862], [0.9692 , 0, 0, 0], [0.9692,   -0.0768,   -0.5937,   -0.2862]])
    spline_coeffs_scaled = np.copy(spline_coeffs) * rotation_angle * rect_to_tanh_scale_correction
    rotation_splines_ind = [spl.CubicSpline(*coeffs) for coeffs in spline_coeffs_scaled]
    # assert len(rotation_splines_ind) == 3
    qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
        phase_rad=phi + pulse_phases[0],output_enable=False,sync=False, _name="sk1_rotate_global"), chan_global)
    for spline_ind, duration in zip(rotation_splines_ind, duration_vec):
        qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
            phase_rad=0,output_enable=False,sync=False, _name="sk1_rotate_individual"), chan_ind)

    if(theta>0.0):
        ## correction pulse 0 (the first 2pi pulse)
        correction_amplitude = 2 * (Tpi/1e6) / qp.samples_to_seconds(sum(duration_vec))
        correction_coeffs_scaled = spline_coeffs * correction_amplitude * rect_to_tanh_scale_correction
        correction_splines_ind = [spl.CubicSpline(*coeffs) for coeffs in correction_coeffs_scaled]
        qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
            phase_rad=phi + pulse_phases[1],output_enable=False,sync=False, _name="sk1_correct1_global"), chan_global)
        for spline_ind, duration in zip(correction_splines_ind, duration_vec):
            qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
                phase_rad=0,output_enable=False,sync=False, _name="sk1_correct1_individual"), chan_ind)

        # # # correction pulse 1  (the second 2pi pulse)
        qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
            phase_rad=phi + pulse_phases[2],output_enable=False,sync=False, _name="sk1_correct2_global"), chan_global)
        for spline_ind, duration in zip(correction_splines_ind, duration_vec):
            qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
                phase_rad=0,output_enable=False,sync=False, _name="sk1_correct2_individual"), chan_ind)

        ##padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(3*duration_total))

    else:
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(1*duration_total))
    return 3*duration_total

### Run flags
num_ions=15
scan_type = 'calibrate_J'   #### 'calibrate_J'  |  'motional_freq'   | 'phi_SK1'    |'imbl' |  'LS'
is_submit_via_python = False
is_debug_sequence = False ### only for debugging
is_get_dataset_vals =True
is_SK1 = [False] *num_ions

#### board 0 ip 102
# is_SK1[5] = True
# is_SK1[6] = True
is_SK1[7] = False
# is_SK1[8] = True
# is_SK1[9] = True
is_SK1[0] = False
is_SK1[14] = False

#### board 1 ip 103
# is_SK1[1] = True
# is_SK1[2] = True
# is_SK1[3] = True
# is_SK1[4] = True

#### board 2 ip 104
# is_SK1[10] = True
# is_SK1[11] = True
# is_SK1[12] = True
# is_SK1[13] = True



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
SK1_nominal_duration = 5217*2
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

T_pi_vec_microsec_ref_J =  0.33143372#np.array([0.92891288, 0.26185544, 0.26274342, 0.25277198, 0.24154937, 0.36214298, 0.35827388, 0.33143372, 0.38263681, 0.35127014, 0.25426646, 0.28588474, 0.26714632, 0.28037115, 1.25087551])
amp_scaling_pi_time_J = 1+0*(T_pi_vec_microsec/T_pi_vec_microsec_ref_J)
### Main Schedule
schedules = []
freq_com = 3.315e6
detuning_vec = [3.1e4]

phi_offset_2ndSK1 = -0.06 ### measured in RID 293253
phi_offset_XX_int = 0.0 ### measured in RID 293253
phi_SK1_vec = [0]
time_J_mu_vec = [1.37e6]
amp_ind_scale = 0.06### we measured J of 3.8765 for amplitude of 0.24 (including the Tpi relative calibration)
amp_imbal_ratio_b2r_vec = [1.015]
LS_scan_vec = [0]
calibration_pair =[0]##[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]#
calibration_pair_offset =  np.array([int((num_ions-1)*0.5)],int)+np.array(calibration_pair,int)

LS_vec_calib = np.zeros(num_ions)
LS_vec_calib[1] = -450
LS_vec_calib[2] = -100
LS_vec_calib[3] = -50
LS_vec_calib[7] = 0

if(scan_type == 'calibrate_J'):
    T_final =int(7.5e3)
    Npts = 15
    time_J_mu_vec =np.array([*range(100, T_final, np.int(T_final/Npts))])
    print(time_J_mu_vec)
elif(scan_type == 'motional_freq'):
    detuning_vec = 3e4+np.linspace(-16e3,16e3,61)
    print(detuning_vec)
elif(scan_type == 'phi_SK1'):
    phi_SK1_vec = np.linspace(0,2*np.pi,17)
    phi_SK1_vec = np.linspace(-0.4,0.4,21)
    print(phi_SK1_vec)
elif(scan_type == 'imbl'):
    amp_imbal_ratio_b2r_vec = 1.014+np.linspace(-0.007,0.004,21)
    print(phi_SK1_vec)
elif('LS'):
    LS_scan_vec = np.linspace(-8e2,0e2,13)
    print(LS_scan_vec)


counter_scan = 0
for time_J_mu in time_J_mu_vec:
    for detuning in detuning_vec:
        for phi_SK1 in phi_SK1_vec:
            for amp_imbal_ratio_b2r in amp_imbal_ratio_b2r_vec:
                for LS_scan in  LS_scan_vec:
                    with qp.build(backend=qp_backend) as out_sched:
                        ### carrier
                        freq_carrier_glob = qp_backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
                        freq_global_vec_SB0 = np.array([freq_carrier_glob,freq_carrier_glob]) # red and blue tones' frequencies in Hz for carrier
                        freq_ind_vec = np.array([qp_backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value]*num_ions)
                        phase_correction0 = 0 ## correction of carrier phase w.respect to the current frequency of sideband 1

                        ### 1st sidebands
                        w_k = freq_com + detuning
                        freq_global_vec_SB1 = np.array([freq_carrier_glob,freq_carrier_glob+w_k]) # red and blue tones' frequencies in Hz for first sideband
                        phase_correction1 = 0 ## correction of motion_phase_offset_squzcarrier phase w.respect to the current frequency of sideband 1

                        LS_vec_calib_exp = LS_vec_calib[calibration_pair_offset]
                        LS_vec_calib_exp[0] = LS_vec_calib_exp[0]+LS_scan
                        print(LS_vec_calib_exp)
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
                            # tanh_scale_factor = 1/np.array([1,1.010864143352493, 1.0155739246100977, 0.9758973924694272, 0.9874321962148809, 0.9442526136868277, 0.9259506107007508, 0.9710440381630175, 0.9366137613959852, 0.9451632797230591, 0.9442722156408533, 0.953916605542164, 0.9171838005255846, 0.9375556730141229,1])#np.ones(15)
                            tanh_scale_factor = np.array([1.3600201237400071, 1.1950996355428771, 1.2113753731022272, 1.2383676179536554, 1.2062325219228924, 1.2508120271835235, 1.2412135840340144, 1.2143770591832472, 1.2269968285219433, 1.2339312143322876, 1.268860667538022, 1.320941271693219, 1.3328926890614001, 1.2779522378964572, 1.2193051873794316])*np.array([1.0321351157103864, 0.9425569580481474, 0.9058591732980279, 0.886550587279146, 0.9718776331008022, 0.9704353340062185, 1.0055884673663404, 0.9923371692029463, 0.9928503626525393, 1.006524837920474, 0.9749731393511611, 0.9882763500231294, 0.9946508580543696, 0.993671093059356, 0.842488702724964])
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

                        ## initial SK1 pulse ##
                        theta_SK1_vec = [np.pi]*num_ions
                        phi0_SK1_vec = [0]*num_ions
                        T_SK1_tot = sequential_sk1(applied_gates=is_SK1, pi_times=T_pi_vec_microsec,
                                        thetas=np.array(theta_SK1_vec), phis=np.array(phi0_SK1_vec)+phi_SK1)
                        phase_correction1 = phase_correction1 +2*np.pi*qp.samples_to_seconds(T_SK1_tot)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0]) ### we put this form because it is a correction on the motional frequency
                        print('duration of SK1 in mu' +str(T_SK1_tot))


                        # delay_all(time_J_mu)

                        ## J_matrix calibration ##
                        print(amp_scaling_pi_time_J[calibration_pair_offset])
                        indices_pad_calib = indices_pad_array(len(freq_ind_vec),calibration_pair_offset)
                        [duration_calibrate] = calibrate_J_pair(chan_glob_vec,f_carrier_glob = freq_global_vec_SB1,chan_ind_vec = chan_vec_pad(chan_ind_vec,calibration_pair_offset),
                                                                f_carrier_ind_vec = freq_ind_vec[calibration_pair_offset],chan_ind_pad_vec = chan_vec_pad(chan_ind_vec,indices_pad_calib),
                                                                freq_ind_pad = freq_ind_vec[indices_pad_calib],duration = time_J_mu,LS_vec= LS_vec_calib_exp,
                                                                amp_scaling_pi_time = amp_scaling_pi_time_J[calibration_pair_offset],Amp_norm_ind =amp_ind_scale,
                                                                Amp_norm_global_cm =Amp_norm_global,amp_imbal_ratio_b2r = amp_imbal_ratio_b2r,phi_spin = phi_offset_XX_int,
                                                                phi_motion_offset = phase_correction1,zero_pad_flag = True)
                        phase_correction0 = phase_correction0 + 2*np.pi*qp.samples_to_seconds(duration_calibrate)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0])
                        print('duration of J calibration in mu' +str(duration_calibrate))

                        # Hamiltonian_evolution(chan_glob_vec,f_carrier_glob = freq_global_vec_SB1,chan_ind_vec = chan_vec_pad(chan_ind_vec,calibration_pair_offset),
                        #                                         f_carrier_ind_vec = freq_ind_vec[calibration_pair_offset],chan_ind_pad_vec = chan_vec_pad(chan_ind_vec,indices_pad_calib),
                        #                                         freq_ind_pad = freq_ind_vec[indices_pad_calib],Amp_norm_ind =amp_ind_scale,
                        #                                         Amp_norm_global_cm =Amp_norm_global,amp_imbal_ratio_b2r = amp_imbal_ratio_b2r,phi_spin = 0,
                        #                                         phi_motion_offset = phase_correction1,zero_pad_flag = True,LS = 0)

                        ##Final SK1 pulse ##
                        T_SK1_tot2 = sequential_sk1(applied_gates=is_SK1, pi_times=T_pi_vec_microsec,
                                        thetas=np.array(theta_SK1_vec), phis=np.array(phi0_SK1_vec)+np.pi+phase_correction0+phi_offset_2ndSK1+phi_SK1)
                        phase_correction1 = phase_correction1 +2*np.pi*qp.samples_to_seconds(T_SK1_tot2)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0]) ### we put this form because it is a correction on the motional frequency
                        print('duration of SK1 in mu' +str(T_SK1_tot2))


                    # print(out_sched)
                    schedules.append(out_sched)
                    counter_scan = counter_scan+1


default_freqs = {chan: 0 for chan in out_sched.channels}
# print(schedules[0].filter(channels=[qp.ControlChannel(0)]).instructions[:10])
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
if(scan_type=='calibrate_J'):
    x_values = (1e-6)*(1000/409)*np.array(time_J_mu_vec)
    x_label = "time_J [ms]"
    # x_values = (1e-6)*np.array(time_J_mu_vec)
    # x_label = "time_J [Mmu]"
elif(scan_type=='motional_freq'):
    x_values = np.array(detuning_vec/1e3)
    x_label = "detuning [kHz]"
elif(scan_type=='phi_SK1'):
    x_values = phi_SK1_vec
    x_label = "phi_SK1 [rad]"
elif(scan_type=='imbl'):
    x_values = amp_imbal_ratio_b2r_vec
    x_label = "global imbalance"
elif(scan_type=='LS'):
    x_values = LS_scan_vec
    x_label = "light shift [Hz]"

print("CSB_evolution_v1 experiment was submitted")

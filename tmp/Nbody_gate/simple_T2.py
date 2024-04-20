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


def edge(chan_glob_vec,f_carrier_glob,chan_ind,f_carrier_ind,chan_ind_pad_vec,freq_ind_pad,
    Amp_norm_ind,Amp_norm_global_cm,amp_imbal_ratio_b2r=1,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,n_edge = 0):
    ### This function generates a single displacement pulse for one ion by modulating the amplitude of the individual beam and modulating the relative phase between the red and blue tones of the global beam
    ### Assumes that chan_ind is a single channel

    ### load splines
    this_dir = pathlib.Path(__file__).parent.resolve()
    # json_file = this_dir / "splines_displace_calib_xxNN_mode3_pulsedur026_waitdur1.json"
    # json_file = this_dir / "splines_displace_xxNN_mode3_pulsedur026_isAMPM0.json"
    # json_file = this_dir / "splines_displace_allrad_xxNN_mode3_pulsedur026_isAMPM0.json"
    json_file = this_dir / "new_splines_displace_d_xxNN_mode3_pulsedur026_isAMPM0.json"

    data_splines = json.loads(json_file.read_text())

    duration_tot = data_splines["edges"][n_edge]["edge_time_mu"] #### This is given in MU for entire pulse
    duration_dt_vec = data_splines["edges"][n_edge]["segment_time_mu"] #### This is given in MU for one segments
    N_segments = data_splines["edges"][n_edge]["number_spline_segments"]

    amp_ind_c0 = data_splines["edges"][n_edge]["amplitude_ind_spline_coeff"]["c0"] #### Splines for individual beams amplitudes
    amp_ind_c1 = data_splines["edges"][n_edge]["amplitude_ind_spline_coeff"]["c1"]
    amp_ind_c2 = data_splines["edges"][n_edge]["amplitude_ind_spline_coeff"]["c2"]
    amp_ind_c3 = data_splines["edges"][n_edge]["amplitude_ind_spline_coeff"]["c3"]

    phi_motion_c0 = data_splines["edges"][n_edge]["phase_motional_global_spline_coeff"]["c0"] #### Splines for Global beam relative red-blue phase
    phi_motion_c1 = data_splines["edges"][n_edge]["phase_motional_global_spline_coeff"]["c1"]
    phi_motion_c2 = data_splines["edges"][n_edge]["phase_motional_global_spline_coeff"]["c2"]
    phi_motion_c3 = data_splines["edges"][n_edge]["phase_motional_global_spline_coeff"]["c3"]
    ### Set amplitudes frequencies and phases
    amp_blue = Amp_norm_global_cm*amp_imbal_ratio_b2r
    amp_red =  Amp_norm_global_cm*(2-amp_imbal_ratio_b2r)

    freq_red = f_carrier_glob[0]
    freq_blue = f_carrier_glob[1]
    freq_ind = f_carrier_ind

    phase_ind = 0
    phase_red_ofst = phi_spin-phi_motion_offset
    phase_blue_ofst = phi_spin+phi_motion_offset


    ### Channel assignments
    chan_glob_tone_red = chan_glob_vec[0]
    chan_glob_tone_blue = chan_glob_vec[1]
    chan_ind1_tone_0 = chan_ind

    round_fact = 1.0

    ### Sequence
    for n_seg in range(0,N_segments):
        ### GLOBAL BEAM - PM
        is_PM = 1
        phase_spline_red =(phase_red_ofst-is_PM*0.5*phi_motion_c0[n_seg],-is_PM*0.5*phi_motion_c1[n_seg],-is_PM*0.5*phi_motion_c2[n_seg],-is_PM*0.5*phi_motion_c3[n_seg]) # (phase_red_ofst,0,0,0) #
        phase_spline_blue = (phase_blue_ofst+is_PM*0.5*phi_motion_c0[n_seg],+is_PM*0.5*phi_motion_c1[n_seg],+is_PM*0.5*phi_motion_c2[n_seg],+is_PM*0.5*phi_motion_c3[n_seg])#(phase_blue_ofst,0,0,0)#

        qp.play(ToneDataPulse(int(round_fact*duration_dt_vec[n_seg]), frequency_hz = freq_red, amplitude = 0*amp_red,
            phase_rad=phase_spline_red,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(round_fact*duration_dt_vec[n_seg]), frequency_hz = freq_blue, amplitude = 1*amp_blue,
            phase_rad=phase_spline_blue,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beam - AM
        amp_spline_ind = spl.CubicSpline(Amp_norm_ind*amp_ind_c0[n_seg],Amp_norm_ind*amp_ind_c1[n_seg],Amp_norm_ind*amp_ind_c2[n_seg],Amp_norm_ind*amp_ind_c3[n_seg])
        qp.play(ToneDataPulse(int(round_fact*duration_dt_vec[n_seg]), frequency_hz = freq_ind, amplitude = amp_spline_ind,
            phase_rad=phase_ind,output_enable=False,sync=False), chan_ind1_tone_0)

        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(round_fact*duration_dt_vec[n_seg]))

    return [int(round_fact*duration_tot)]

def corner(chan_glob_vec,f_carrier_glob,chan_ind_vec,f_carrier_ind_vec,chan_ind_pad_vec,freq_ind_pad,
    Amp_norm_ind =0.5,Amp_norm_global_cm =0.3,amp_imbal_ratio_b2r = 1,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,n_corner = 0):
    ### This function generates a squeezing pulse, addressing multiple ions simultaneously by modulating the amplitude of the individual beams only.
    ### Global's amplitude & phase are assumed to be constant
    ### chan_ind_vec is the list of channels of the ions that are modulated

    ### load splines
    this_dir = pathlib.Path(__file__).parent.resolve()
    json_file = this_dir / "splines_squeeze_compact__mode3_pulsedur024.json"
    data_splines = json.loads(json_file.read_text())

    duration_tot = data_splines["corner"][n_corner]["corner_time_mu"] #### This is given in MU for entire pulse
    duration_dt_vec = data_splines["corner"][n_corner]["segment_time_mu"] #### This is given in MU for one segments
    N_segments_vec = data_splines["corner"][n_corner]["number_spline_segments"]
    N_segments= int(np.sum(N_segments_vec))

    amp_ind_c0 = data_splines["corner"][n_corner]["amplitude_ind_spline_coeff"]["c0"] #### Splines for individual beams amplitudes
    amp_ind_c1 = data_splines["corner"][n_corner]["amplitude_ind_spline_coeff"]["c1"]
    amp_ind_c2 = data_splines["corner"][n_corner]["amplitude_ind_spline_coeff"]["c2"]
    amp_ind_c3 = data_splines["corner"][n_corner]["amplitude_ind_spline_coeff"]["c3"]

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
    for n_seg in range(0,N_segments):
        ### GLOBAL BEAM - constant
        qp.play(ToneDataPulse(int(round_fact*duration_dt_vec[n_seg]), frequency_hz = freq_red, amplitude = amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(round_fact*duration_dt_vec[n_seg]), frequency_hz = freq_blue, amplitude = amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            freq_ind = f_carrier_ind_vec[counter]
            amp_spline_ind = spl.CubicSpline(Amp_norm_ind*amp_ind_c0[n_seg],Amp_norm_ind*amp_ind_c1[n_seg],Amp_norm_ind*amp_ind_c2[n_seg],Amp_norm_ind*amp_ind_c3[n_seg])

            qp.play(ToneDataPulse(int(round_fact*duration_dt_vec[n_seg]), frequency_hz = freq_ind, amplitude = amp_spline_ind,
                phase_rad=phase_ind,output_enable=False,sync=False), chan_ind_tone_0)

        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(round_fact*duration_dt_vec[n_seg]))
        duration_tot_actual = duration_tot_actual+int(round_fact*duration_dt_vec[n_seg])

    return [int(duration_tot_actual)]

def pad_end_streaming_mode(chan_glob_vec,f_carrier_glob,chan_ind_pad_vec,freq_ind_pad,
    Amp_norm_global_cm,amp_imbal_ratio_b2r,lightshift_freq,phi_spin = 0,phi_motion_offset = 0):
    ### Unlike in the memory limited LUT mode, In streaming mode the last 2 microseconds are output weird stuff on the global. This function ensures that this garbage happens while the individual is set to 0

    duration_dt = 100
    N_segments = 10
    duration_tot = duration_dt*N_segments

    ### Set amplitudes frequencies and phases
    amp_blue = Amp_norm_global_cm*amp_imbal_ratio_b2r
    amp_red =  Amp_norm_global_cm*(2-amp_imbal_ratio_b2r)

    freq_red = f_carrier_glob[0]+lightshift_freq
    freq_blue = f_carrier_glob[1]+lightshift_freq

    phase_red_ofst = phi_spin-phi_motion_offset
    phase_blue_ofst = phi_spin+phi_motion_offset

    ### Channel assignments
    chan_glob_tone_red = chan_glob_vec[0]
    chan_glob_tone_blue = chan_glob_vec[1]

    ### Sequence
    for n_seg in range(0,N_segments):
        ### GLOBAL BEAM constant amplitude
        qp.play(ToneDataPulse(duration_dt, frequency_hz = freq_red, amplitude = amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(duration_dt, frequency_hz = freq_blue, amplitude = amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Padding all other channels
        zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,duration_dt)

    return [duration_tot]

def Rabi_flop_SK1(chan_global,freq_glob,chan_ind,freq_ind,chan_ind_pad_vec,freq_ind_pad,Amp_norm_global,
                Tpi,theta,phi,tanh_scale_factor,detuning_Hz=0, zero_pad_flag = True):
    ### This function flips at a given time the single qubit associated with chan_ind
    # 3 segments, all segments equal duration
    scale_time =1
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
        ## correction pulse 0
        correction_amplitude = 2 * (Tpi/1e6) / qp.samples_to_seconds(sum(duration_vec))
        correction_coeffs_scaled = spline_coeffs * correction_amplitude * rect_to_tanh_scale_correction
        correction_splines_ind = [spl.CubicSpline(*coeffs) for coeffs in correction_coeffs_scaled]
        qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
            phase_rad=phi + pulse_phases[1],output_enable=False,sync=False, _name="sk1_correct1_global"), chan_global)
        for spline_ind, duration in zip(correction_splines_ind, duration_vec):
            qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
                phase_rad=0,output_enable=False,sync=False, _name="sk1_correct1_individual"), chan_ind)

        # # # correction pulse 1
        qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
            phase_rad=phi + pulse_phases[2],output_enable=False,sync=False, _name="sk1_correct2_global"), chan_global)

        for spline_ind, duration in zip(correction_splines_ind, duration_vec):
            qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
                phase_rad=0,output_enable=False,sync=False, _name="sk1_correct2_individual"), chan_ind)

        ###padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(3*duration_total))

    else:
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(1*duration_total))

    return 3*duration_total

### Run flags
is_submit_via_python = False
is_debug_sequence = False ### only for debugging
seq_type1 = 'Both' #### 'Displacement'  |  'Squeezing'  |  'Both'  |  'Sqz_Clos'  |
seq_type2 = 'Gate' #### 'Closure'  |  'Gate'  |
scan_type = 'Phi12' #### 'Freq'  |  'ImbGlob'   |  'IndAmp'  | 'Tdelay'  |  'phiSK1'  | 'Phi12'  | 'Parity'
is_Ramsey = True
is_Echo = True
is_initial_pi_pulse = False
N_stg = 4
spin_ramsey = [True, True, True]
spin_echo = [True, True, True]
is_get_dataset_vals =True


### channels and parameters kept fixed for the 3 ions experiment
master_ip: str = "192.168.78.152"
schedules = []
qp_backend = qbe.get_default_qiskit_backend(master_ip, 3, with_2q_gate_solutions=False)
channel_map = qbe.get_default_rfsoc_map()
three_ion_config = QuickConfig(3,channel_map,{-1:1,0:3,1:5})
qp_backend._config = three_ion_config
Ions_displacement_ind = np.array([0,2],int)
Ions_squeezing_ind = np.array([1],int)
Amp_norm_global_cm_disp = 0.45
Amp_norm_global_cm_squz = 0.45
tanh_scale_factor = [0.0689,0.068,0.0681]# SK1 calibration parameters these values were calibrated for Tpi=np.array([4.556,4.062,4.624]) at 0.06 amp individual and 0.45 amp global and will be used in the SK1 as is
SK1_nominal_duration = 5217
phi_per_ramsey = 0*(0.5372+0.026) #### phase accumulated for displacement or squeezing per SK1 pulse
freq_offset = 0
if((Amp_norm_global_cm_disp>=0.5) or (Amp_norm_global_cm_squz>=0.5)):
    print("error!! amplitude of two tones signal is saturated above 0.5")

##### calibrated at amp global 0.45 and am ind=0.06. then we get nominal values of T_pi_vec_microsec = np.array([4.556,4.062,4.624])
T_pi_vec_microsec_ref_lightshift =  np.array([3.643,3.17,3.662])##np.array([4.411,3.823,4.429])#this should be
T_pi_vec_microsec_ref_disp =  np.array([3.859,3.307,3.856])##np.array([4.411,3.823,4.429])#this should be
T_pi_vec_microsec =  np.array([3.864,3.396,3.947])##np.array([4.411,3.823,4.429])#this should be

tau_wait_vec = [0]#np.linspace(1e4,3e6,9)
Phi12_vec = np.linspace(-0.1,0.1,13)


### Main Schedule
schedules = []


amp_imbal_ratio_b2r_disp_vec = np.array([1.0095])#np.array([1.01])#
amp_imbal_ratio_b2r_squz_vec = [1.016]# at 0.7 ind amp it is 1.018

phiSK1_vec_bare = np.array([0.0,0.18,0.0])#np.array([0.0,np.pi/2,0.0])
phiSK1 = np.array([0*np.pi/2]);#[+1.21+np.pi]#[np.pi+1.21]#
phi_Parity_vec = [-0.1]
# phi_light_shift_squeezing = -0.14*0
N_gate_rep_vec = [0]

phi_disp_per_ramsey = phi_per_ramsey*np.mean([T_pi_vec_microsec_ref_lightshift[0]/T_pi_vec_microsec[0],T_pi_vec_microsec_ref_lightshift[2]/T_pi_vec_microsec[2]])
phi_squeez_per_ramsey = phi_per_ramsey*T_pi_vec_microsec_ref_lightshift[1]/T_pi_vec_microsec[1]

counter_scan = 0
for tau_wait in tau_wait_vec:
    for Phi12 in Phi12_vec:
        with qp.build(backend=qp_backend) as out_sched:

            #########   frequency & phases assignment   ######
            ### carrier
            freq_carrier_glob = qp_backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
            freq_global_vec_SB0 = np.array([freq_carrier_glob,freq_carrier_glob]) # red and blue tones' frequencies in Hz for carrier
            freq_ind_vec = np.array([qp_backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value]*3)
            T_echo = 0

            ## assign channels
            chan_glob_tone_red = qp.control_channels()[0]
            chan_glob_tone_blue = qp.control_channels()[1]
            chan_glob_vec = [chan_glob_tone_red, chan_glob_tone_blue]
            chan_ind_vec = []
            for ion_idx in range(-int((-1+freq_ind_vec.shape[0])/2),1+int((-1+freq_ind_vec.shape[0])/2)):
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
                            Amp_norm_global=0.45,
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

            for n in range(30):
                T_ramsey1 = sequential_sk1(applied_gates=spin_ramsey, pi_times=T_pi_vec_microsec,
                                thetas=np.array([np.pi/2,np.pi/2,np.pi/2]), phis=np.array([0,0, 0])+phiSK1_vec_bare+phiSK1)

                T_ramsey2 = sequential_sk1(applied_gates=spin_ramsey, pi_times=T_pi_vec_microsec,
                                thetas=np.array([np.pi/2,np.pi/2,np.pi/2]), phis=Phi12+np.array([np.pi,np.pi, np.pi])+phiSK1_vec_bare+phiSK1)

            if(tau_wait>0):
                delay_all(T_delay=tau_wait)
                print('duration of waiting in mu' +str(tau_wait))

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
    plt.savefig('tmp/Nbody_gate/schedule_displace.png')
    # compiled_schedule = OpenPulseToOctetConverter.schedule_to_octet(out_sched, default_lo_freq_hz=default_freqs)
    # compiled_schedule_global_only = {k: v for k, v in compiled_schedule.items() if isinstance(k, qp.ControlChannel)}
    # print(record.text_channel_sequence(compiled_schedule_global_only))
    # record.save_text_channel_sequence(compiled_schedule, "./tmp/Nbody_gate/NOT_working_phase_func_commands.tones")

### The plots presented after submission via the GUI in  artiq
if(scan_type=='Freq'):
    if(seq_type1=='Displacement'):
        x_values = w_k_disp_vec*1e-6
        x_label = "red and blue detuning displacements [MHz]"
    elif(seq_type1=='Squeezing'):
        x_values = w_k_squz_vec*1e-6
        x_label = "red and blue detuning squeezing [MHz]"
    if((seq_type1=='Both')or (seq_type1=='Sqz_Clos')):
        x_values = w_k_disp_vec*1e-6
        x_label = "red and blue detuning 1 and 2 bands [MHz]"

elif(scan_type=='ImbGlob'):
    if((seq_type1=='Displacement') or ((seq_type1=='Both')or (seq_type1=='Sqz_Clos'))):
        x_values = amp_imbal_ratio_b2r_disp_vec
        x_label = "global B2R ratio 1st sidebands"
    elif((seq_type1=='Squeezing')):
        x_values = amp_imbal_ratio_b2r_squz_vec
        x_label = "global B2R ratio  2nd sidebands"

elif(scan_type=='IndAmp'):
    if((seq_type1=='Displacement')or ((seq_type1=='Both')or (seq_type1=='Sqz_Clos'))):
        x_values = Amp_norm_ind_disp_vec
        x_label = "amp ind displacements"
    elif(seq_type1=='Squeezing'):
        x_values = Amp_norm_ind_squz_vec
        x_label = "amp ind squeezing"

elif(scan_type=='phiSK1'):
    x_values = phiSK1_vec
    x_label = "phiSK1"

elif(scan_type=='Tdelay'):
    x_values = tau_wait_vec
    x_label = "delay time [rfsoc units]"

elif(scan_type=='Phi12'):
    x_values = Phi12_vec
    x_label = "relative motional phase squeezing - displacements"
elif(scan_type=='Parity'):
    x_values = phi_Parity_vec
    x_label = "Parity phase scan"

print("N-Gate experiment was submitted")

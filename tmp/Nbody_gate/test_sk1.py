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



def initial_sync(ch_global_vec,ch_ind_vec,freq_global,freq_ind_list,is_sync_global=True,dt_sync_mu = 6):

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
    Amp_norm_ind,Amp_norm_global_cm,amp_imbal_ratio_b2r=1,lightshift_freq=0,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,n_edge = 0):
    ### This function generates a single displacement pulse for one ion by modulating the amplitude of the individual beam and modulating the relative phase between the red and blue tones of the global beam
    ### Assumes that chan_ind is a single channel

    ### load splines
    this_dir = pathlib.Path(__file__).parent.resolve()
    json_file = this_dir / "splines_displace_calib_xxNN_mode3_pulsedur026_waitdur1.json"
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

    freq_red = f_carrier_glob[0]+lightshift_freq
    freq_blue = f_carrier_glob[1]+lightshift_freq
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
        phase_spline_red =(phase_red_ofst-0.5*phi_motion_c0[n_seg],-0.5*phi_motion_c1[n_seg],-0.5*phi_motion_c2[n_seg],-0.5*phi_motion_c3[n_seg]) # (phase_red_ofst,0,0,0) #
        phase_spline_blue = (phase_blue_ofst+0.5*phi_motion_c0[n_seg],+0.5*phi_motion_c1[n_seg],+0.5*phi_motion_c2[n_seg],+0.5*phi_motion_c3[n_seg])#(phase_blue_ofst,0,0,0)#

        qp.play(ToneDataPulse(int(round_fact*duration_dt_vec[n_seg]), frequency_hz = freq_red, amplitude = amp_red,
            phase_rad=phase_spline_red,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(round_fact*duration_dt_vec[n_seg]), frequency_hz = freq_blue, amplitude = amp_blue,
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
    Amp_norm_ind =0.5,Amp_norm_global_cm =0.3,amp_imbal_ratio_b2r = 1,lightshift_freq = 0,phi_spin = 0,
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

    freq_red = f_carrier_glob[0]+lightshift_freq
    freq_blue = f_carrier_glob[1]+lightshift_freq

    phase_ind = 0
    phase_red_ofst = phi_spin-phi_motion_offset
    phase_blue_ofst = phi_spin+phi_motion_offset


    ### Channel assignments
    chan_glob_tone_red = chan_glob_vec[0]
    chan_glob_tone_blue = chan_glob_vec[1]

    ### Sequence
    for n_seg in range(0,N_segments):
        ### GLOBAL BEAM - constant
        qp.play(ToneDataPulse(duration_dt_vec[n_seg], frequency_hz = freq_red, amplitude = amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(duration_dt_vec[n_seg], frequency_hz = freq_blue, amplitude = amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            freq_ind = f_carrier_ind_vec[counter]
            amp_spline_ind = spl.CubicSpline(Amp_norm_ind*amp_ind_c0[n_seg],Amp_norm_ind*amp_ind_c1[n_seg],Amp_norm_ind*amp_ind_c2[n_seg],Amp_norm_ind*amp_ind_c3[n_seg])

            qp.play(ToneDataPulse(duration_dt_vec[n_seg], frequency_hz = freq_ind, amplitude = amp_spline_ind,
                phase_rad=phase_ind,output_enable=False,sync=False), chan_ind_tone_0)

        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,duration_dt_vec[n_seg])

    return [duration_tot]

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

def Rabi_flop(chan_global,freq_glob,chan_ind,freq_ind,chan_ind_pad_vec,freq_ind_pad,Amp_norm_ind,Amp_norm_global,
                duration_microsec, detuning_Hz=0,phi_spin = 0,sideband_order=1,zero_pad_flag = True):
    ### This function flips at a given time the single qubit associated with chan_ind

    qp.play(ToneDataPulse(qp.seconds_to_samples(duration_microsec*1e-6), frequency_hz = freq_glob-detuning_Hz, amplitude = Amp_norm_global,
        phase_rad=phi_spin,output_enable=False,sync=False, _name="rabi_global"), chan_global)

    qp.play(ToneDataPulse(qp.seconds_to_samples(duration_microsec*1e-6), frequency_hz = freq_ind, amplitude = Amp_norm_ind,
        phase_rad=0,output_enable=False,sync=False, _name="rabi_individual"), chan_ind)

    if(zero_pad_flag):
        zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,qp.seconds_to_samples(duration_microsec*1e-6))

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

    # if(theta>0.0):
    #     ## correction pulse 0
    #     correction_amplitude = 2 * (Tpi/1e6) / qp.samples_to_seconds(sum(duration_vec))
    #     correction_coeffs_scaled = spline_coeffs * correction_amplitude * rect_to_tanh_scale_correction
    #     correction_splines_ind = [spl.CubicSpline(*coeffs) for coeffs in correction_coeffs_scaled]
    #     qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
    #         phase_rad=phi + pulse_phases[1],output_enable=False,sync=False, _name="sk1_correct1_global"), chan_global)
    #     for spline_ind, duration in zip(correction_splines_ind, duration_vec):
    #         qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
    #             phase_rad=0,output_enable=False,sync=False, _name="sk1_correct1_individual"), chan_ind)

    #     # # # correction pulse 1
    #     qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob-detuning_Hz, amplitude =Amp_norm_global,
    #         phase_rad=phi + pulse_phases[2],output_enable=False,sync=False, _name="sk1_correct2_global"), chan_global)

    #     for spline_ind, duration in zip(correction_splines_ind, duration_vec):
    #         qp.play(ToneDataPulse(duration, frequency_hz = freq_ind, amplitude =spline_ind,
    #             phase_rad=0,output_enable=False,sync=False, _name="sk1_correct2_individual"), chan_ind)

    #     ###padding all other channels
    #     if(zero_pad_flag):
    #         zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(3*duration_total))

    # else:
    #     if(zero_pad_flag):
    #         zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(1*duration_total))
    zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(1*duration_total))


### prepare channels maps and data
is_submit = False
# channel_map = RFSoCChannelMapping.from_pyon_file(pathlib.Path(os.getcwd(), 'zcu_hardware.pyon'))
# qp_backend = MinimalQiskitIonBackend(7, channel_map, endcap_ions=(0,0))

master_ip: str = "192.168.78.152"
schedules = []
qp_backend = qbe.get_default_qiskit_backend(master_ip, 3, with_2q_gate_solutions=False)
channel_map = qbe.get_default_rfsoc_map()
three_ion_config = QuickConfig(3,channel_map,{-1:1,0:3,1:5})
qp_backend._config = three_ion_config

Ions_displacement_ind = np.array([0,2],int)
Ions_squeezing_ind = np.array([1],int)
# Amp_norm_ind_squz = 0.5
Amp_norm_global_cm_disp = 0.45
Amp_norm_global_cm_squz = 0.45
amp_imbal_ratio_b2r_squz = 1

if((Amp_norm_global_cm_disp>=0.5) or (Amp_norm_global_cm_squz>=0.5)):
    print("error!! amplitude of two tones signal is saturated above 0.5")
lightshift_freq_disp = 0
lightshift_freq_squz = 0

##### calibrated at amp global 0.45 and am ind=0.06. then we get nominal values of T_pi_vec_microsec = np.array([4.556,4.062,4.624])
T_pi_vec_microsec = np.array([3.581,3.141,3.605])#this should be
tanh_scale_factor = [0.0689,0.068,0.0681]# these values were calibrated for  Tpi=np.array([4.556,4.062,4.624]) at 0.06 amp individual and 0.45 amp global

N_stg =0
N_stg_vec =[1,1,1,1]
ion_idx_disp_vec  = [Ions_displacement_ind[1],Ions_displacement_ind[1],Ions_displacement_ind[1],Ions_displacement_ind[1]] #[Ions_displacement_ind[0],Ions_displacement_ind[1],Ions_displacement_ind[0],Ions_displacement_ind[1]] ### the ion that is displaced as a function of the stage number
# spin_phase_squz = [0,0,0,0]
# motion_phase_offset_squz = [0,np.pi,0,np.pi] ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction

### Main Schedule
schedules = []
amp_imbal_ratio_b2r_disp = np.array([1.015])#
Amp_norm_ind_disp =  np.array([0.962])#np.linspace(0.952,0.973,9)
rel_phase_vec = np.array([np.pi/2]) # np.linspace(0,np.pi,21)#np.array([0])#
w_k_vec =np.array([2.839e6])#np.linspace(2.838e6,2.8392e6,11)##np.linspace(2.836e6,2.843e6,11)##np.linspace(2.80e6,2.86e6,21)#np.linspace(2.825e6,2.83e6,21)#
tau_wait = int(1e1)
light_shift_vec = [0] #np.linspace(-1e2,1e2,17)
dtheta =0.0
theta_scan_vec = np.linspace(0.1,2*np.pi,25)#[np.pi/2]#np.linspace(0.1,np.pi,25) # #
T_delay_microseconds_vec = [2000]#np.linspace(1,5000,17)
# theta_scan_vec = [2*np.pi]
# tanh_scale_factor_vec = np.linspace(0.0842-0.05, 0.0842+0.05, num=25)
Nsk1_vec = [1]#np.arange(1,32,10)
for T_delay_microseconds in T_delay_microseconds_vec:
    for light_shift in light_shift_vec:
        for theta_scan in theta_scan_vec:
            for Nsk1 in Nsk1_vec:
                for w_k in w_k_vec:
                    with qp.build(backend=qp_backend) as out_sched:

                        ## frequency & phases assignment
                        phi_scan=0
                        spin_phase_disp = [0,0,0,0]
                        motion_phase_offset_disp = [0*np.pi,0.5*(np.pi),(np.pi),1.5*(np.pi)] ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction
                        freq_carrier_glob = qp_backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
                        freq_global_vec_SB0 = np.array([freq_carrier_glob+light_shift,freq_carrier_glob+light_shift]) # red and blue tones' frequencies in Hz for first sideband
                        freq_global_vec_SB1 = np.array([freq_carrier_glob-w_k+light_shift,freq_carrier_glob+w_k+light_shift]) # red and blue tones' frequencies in Hz for first sideband
                        freq_global_vec_SB2 = np.array([freq_carrier_glob-2*w_k,freq_carrier_glob+2*w_k]) # red and blue tones' frequencies in Hz for second sideband
                        freq_ind_vec = np.array([qp_backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value]*3)
                        phase_correction = 0

                        ## assign channels
                        chan_glob_tone_red = qp.control_channels()[0]
                        chan_glob_tone_blue = qp.control_channels()[1]
                        chan_glob_vec = [chan_glob_tone_red, chan_glob_tone_blue]
                        chan_ind_vec = []
                        for ion_idx in range(-int((-1+freq_ind_vec.shape[0])/2),1+int((-1+freq_ind_vec.shape[0])/2)):
                            chan_ind_vec = chan_ind_vec + [qp_backend.configuration().individual_channel(ion_idx, 0)]

                        ## Initial sync of phases
                        initial_sync(ch_global_vec = chan_glob_vec, ch_ind_vec = chan_ind_vec,freq_global = freq_global_vec_SB0,freq_ind_list=freq_ind_vec)

                        def sequential_Rabi(T_rotation_rel  = np.array([0.0,0.0,0.0]),spin_phase_Rabi = np.zeros(3),Amp_norm_ind=0.5):
                            for n_ion in range(3):
                                indices_pad_Rabi= indices_pad_array(len(freq_ind_vec),[n_ion])
                                duration_microsec = T_rotation_rel[n_ion]*T_pi_vec_microsec[n_ion]
                                if(duration_microsec>0.02):
                                    channels_pad_Rabi = chan_vec_pad(chan_ind_vec,indices_pad_Rabi)+[chan_glob_tone_blue]
                                    freq_pad_Rabi = np.hstack((freq_ind_vec[indices_pad_Rabi],freq_global_vec_SB1[1]))
                                    Rabi_flop(chan_global=chan_glob_tone_red,freq_glob = freq_carrier_glob,chan_ind = chan_ind_vec[n_ion],freq_ind=freq_ind_vec[n_ion],
                                                chan_ind_pad_vec = channels_pad_Rabi,freq_ind_pad = freq_pad_Rabi,Amp_norm_ind=Amp_norm_ind,Amp_norm_global=0.45,duration_microsec=duration_microsec,
                                                detuning_Hz=0, phi_spin = spin_phase_Rabi[n_ion],zero_pad_flag = True)
                                    print('flipping ion' +str(n_ion-1))

                        def sequential_sk1(applied_gates, pi_times, thetas, phis):
                            for i, do_gate in enumerate(applied_gates):
                                if do_gate:
                                    indices_pad_Rabi= indices_pad_array(len(freq_ind_vec),[i])
                                    channels_pad_Rabi = chan_vec_pad(chan_ind_vec,indices_pad_Rabi)+[chan_glob_tone_blue]
                                    freq_pad_Rabi = np.hstack((freq_ind_vec[indices_pad_Rabi],freq_global_vec_SB1[1]))
                                    Rabi_flop_SK1(
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

                        def delay_all(T_delay_microseconds):
                            channels_pad_delay = chan_ind_vec+chan_glob_vec
                            freq_pad_delay = np.hstack((freq_ind_vec,freq_global_vec_SB0))
                            zero_padding_ind(channels_pad_delay,freq_pad_delay,qp.seconds_to_samples(T_delay_microseconds*1e-6))

                        for i in range(Nsk1):
                            # sequential_Rabi(T_rotation_rel  = np.array([0.0,0.0,0.0]),spin_phase_Rabi = np.zeros(3)+phi_scan)
                            sequential_sk1(applied_gates=[True, True, True], pi_times=T_pi_vec_microsec, thetas=[theta_scan, theta_scan, theta_scan], phis=[0, 0, 0])
                            # sequential_sk1(applied_gates=[False, True, False], pi_times=T_pi_vec_microsec, thetas=[np.pi/2, theta_scan, np.pi/2], phis=[0, 0, 0])
                            # sequential_sk1(applied_gates=[False, True, False], pi_times=T_pi_vec_microsec, thetas=[np.pi/2, theta_scan, np.pi/2], phis=[0, 0, 0])
                            # sequential_sk1(applied_gates=[False, True, False], pi_times=T_pi_vec_microsec, thetas=[np.pi/2, theta_scan, np.pi/2], phis=[0, 0, 0])
                            # # delay_all(T_delay_microseconds)
                            # sequential_sk1(applied_gates=[False, True, False], pi_times=T_pi_vec_microsec, thetas=[np.pi/2, np.pi/2, np.pi/2], phis=[0, np.pi, 0])

                        ## Alternating Displacement & Squeezing operations
                        for nn in range(N_stg):
                            ## Displacement stage
                            ion_idx_stg = ion_idx_disp_vec[nn]
                            ion_idx_stg_array = [ion_idx_stg]
                            indices_pad_disp = indices_pad_array(len(freq_ind_vec),ion_idx_stg_array)
                            [duration_disp] = edge(chan_glob_vec=chan_glob_vec,f_carrier_glob = freq_global_vec_SB1, chan_ind=chan_ind_vec[ion_idx_stg],
                                                        f_carrier_ind = freq_ind_vec[ion_idx_stg],chan_ind_pad_vec = chan_vec_pad(chan_ind_vec,indices_pad_disp),
                                                        freq_ind_pad = freq_ind_vec[indices_pad_disp],Amp_norm_ind =Amp_norm_ind_disp,Amp_norm_global_cm =Amp_norm_global_cm_disp,
                                                        amp_imbal_ratio_b2r = amp_imbal_ratio_b2r_disp,lightshift_freq = lightshift_freq_disp, phi_spin = spin_phase_disp[nn],
                                                        phi_motion_offset = motion_phase_offset_disp[nn],n_edge = N_stg_vec[nn]) ### displacement of a single ion

                            phase_correction = phase_correction+ 2*np.pi*qp.samples_to_seconds(duration_disp)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0])
                            ## delay
                            zero_padding_ind(chan_ind_vec+chan_glob_vec,np.hstack((freq_ind_vec,freq_global_vec_SB0)),tau_wait)

                        # sequential_Rabi(T_rotation_rel  = np.array([0.0,0.0,0.0]),spin_phase_Rabi =  np.array([np.pi,np.pi,np.pi]) + phase_correction+phi_scan)

                    schedules.append(out_sched)


default_freqs = {chan: 0 for chan in out_sched.channels}
if(is_submit):
    rfsoc_submit.submit_schedule(
        schedules,
        master_ip,
        qp_backend,
        experiment_kwargs={
            "xlabel": "red and blue detuning",
            "x_values": pyon.encode(w_k_vec),
            # "xlabel": "red blue spin phase",
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

else:
    print(schedules[0].filter(channels=[qp.ControlChannel(0)]).instructions[:10])
    matplotlib.use("agg")
    out_sched.draw()
    # plt.savefig('tmp/Nbody_gate/schedule_displace.png')
    print(default_freqs)
    # compiled_schedule = OpenPulseToOctetConverter.schedule_to_octet(out_sched, default_lo_freq_hz=default_freqs)
    # compiled_schedule_global_only = {k: v for k, v in compiled_schedule.items() if isinstance(k, qp.ControlChannel)}
    # print(record.text_channel_sequence(compiled_schedule_global_only))
    # record.save_text_channel_sequence(compiled_schedule, "./tmp/Nbody_gate/NOT_working_phase_func_commands.tones")

# x_values = w_k_vec
# x_label = "red and blue detuning"
# x_values = amp_imbal_ratio_b2r_disp_vec
# x_label = "global B2R ratio"
x_values = np.array(theta_scan_vec)
x_label = "theta (radians)"
# x_values = np.array(T_delay_microseconds_vec/1e3)
# x_label = "delay [ms]]"
# x_values = np.array(light_shift_vec/1e3)
# x_label = "light shift freq [kHz]]"
# x_values = Amp_norm_ind_disp_vec
# x_label = "ind beam amp"
# x_values = tanh_scale_factor_vec/0.0842
# x_label = "amplitude_scale"
# x_values = Nsk1_vec
# x_label = "Nsk1"

import pathlib
import os
import json
import pickle

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
    phi_motion_offset = 0,zero_pad_flag = True):
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

        qp.play(ToneDataPulse(int(round_fact*duration_act), frequency_hz = freq_blue, amplitude = 1*amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            freq_ind = f_carrier_ind_vec[counter]+LS_vec[counter]

            qp.play(ToneDataPulse(int(round_fact*duration_act), frequency_hz = freq_ind, amplitude = Amp_norm_ind*amp_scaling_pi_time[counter],
                phase_rad=phase_ind,output_enable=False,sync=False), chan_ind_tone_0)

            print( Amp_norm_ind*amp_scaling_pi_time[counter])
        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(round_fact*duration_act))
        duration_tot_actual = duration_tot_actual+int(round_fact*duration_act)

    return [int(duration_tot_actual)]


def Hamiltonian_evolution(chan_glob_vec,f_carrier_glob,chan_ind_vec,f_carrier_ind_vec,chan_ind_pad_vec,freq_ind_pad,
    LS_vec,amp_scaling_pi_time,Amp_norm_ind =0.5,Amp_norm_global_cm =0.45,amp_imbal_ratio_b2r = 1,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,N_cut_segment_from_end = 0,is_quench = False,Tquench=100):
    ### This function generates a squeezing pulse, addressing multiple ions simultaneously by modulating the amplitude of the individual beams only.
    ### Global's amplitude & phase are assumed to be constant
    ### chan_ind_vec is the list of channels of the ions that are modulated

    ### load splines
    this_dir = pathlib.Path(__file__).parent.resolve()
    ######    CSB  Phase ########
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_med_B_sign-1_N13.json'
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_largeB_sign-1_N13.json'
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_LL_B_sign-1_N13.json'
#
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz242_sign-1_N13.json'


    ######    XY  Phase ########
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_largeB_sign+1_N13.json'
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_medB_sign+1_N13.json'
    json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_try_sign+1_N13.json'
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz242_sign+1_N13.json'


    round_fact = 3.5
    kHz = 0.3e3
    off_ion_ind_list = []
    # chan_ind_quench = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    chan_ind_quench = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    # phase_quench = [0]*9 + [np.pi/2] + [np.pi]*9
    if(is_quench==False):
        phase_quench = [0]*23
    else:
        # phase_quench = [0,np.pi/2]*11 +[0]
        # phase_quench = [0]*18 + [np.pi]
        phase_quench = [0]*9 + [np.pi/2] + [np.pi]*9

    # phase_quench = [0,np.pi/2]*11 +[0]
    # phase_quench = [0,0]*11 +[0]

    data_splines = json.loads(json_file.read_text())

    duration_tot = data_splines["ramp_time_mu"] #### This is given in MU for entire pulse
    # print(duration_tot)
    duration_dt = data_splines["t_seg_mu"] #### This is given in MU for one segments
    N_segments_vec = data_splines["N_segments"]
    N_segments= int(np.sum(N_segments_vec)+N_cut_segment_from_end)

    #### Values for the Splines for individual beams amplitudes (without relative scaling of each channel)
    amp_ind_c0 = data_splines["splines_glob_amp"]["c0"]
    amp_ind_c1 = data_splines["splines_glob_amp"]["c1"]
    amp_ind_c2 = data_splines["splines_glob_amp"]["c2"]
    amp_ind_c3 = data_splines["splines_glob_amp"]["c3"]

    amp_scaling_ind_vec = data_splines["amp_ind_relative_scale"]  ### This is a relative scaling vector w. respect to the middle ion that is with amplitude 1

    #### Values for the Splines for individual beams frequencies
    freq_ind_odd_c0 = data_splines["freqlist_ind"][0]["splines"]["c0"] #### Splines for odd signed ind beam in KHz
    freq_ind_odd_c1 = data_splines["freqlist_ind"][0]["splines"]["c1"] #### Splines for odd signed ind beam in KHz
    freq_ind_odd_c2 = data_splines["freqlist_ind"][0]["splines"]["c2"] #### Splines for odd signed ind beam in KHz
    freq_ind_odd_c3 = data_splines["freqlist_ind"][0]["splines"]["c3"] #### Splines for odd signed ind beam in KHz

    freq_ind_even_c0 = data_splines["freqlist_ind"][1]["splines"]["c0"] #### Splines for odd signed ind beam in KHz
    freq_ind_even_c1 = data_splines["freqlist_ind"][1]["splines"]["c1"] #### Splines for odd signed ind beam in KHz
    freq_ind_even_c2 = data_splines["freqlist_ind"][1]["splines"]["c2"] #### Splines for odd signed ind beam in KHz
    freq_ind_even_c3 = data_splines["freqlist_ind"][1]["splines"]["c3"] #### Splines for odd signed ind beam in KHz

    LS_ind_c0 = data_splines["splines_ind_LS"]["c0"] #### Splines for odd signed ind beam in KHz
    LS_ind_c1 = data_splines["splines_ind_LS"]["c1"] #### Splines for odd signed ind beam in KHz
    LS_ind_c2 = data_splines["splines_ind_LS"]["c2"] #### Splines for odd signed ind beam in KHz
    LS_ind_c3 = data_splines["splines_ind_LS"]["c3"] #### Splines for odd signed ind beam in KHz


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
    ### Sequence
    duration_tot_actual = int(0)
    ### GLOBAL BEAM - constant
    for n_seg in range(0,N_segments):
        # print(n_seg)

        ### GLOBAL BEAM - constant
        qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_red, amplitude =1*amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_blue, amplitude = 1*amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM (Jij control) and FM (Stark shift control)
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            LS_amp = LS_vec[counter]
            freq_ind_carrier = f_carrier_ind_vec[counter]
            if counter in off_ion_ind_list:
                    freq_spline_ind = freq_ind_carrier
                    amp_scaling_ind = 0
            else:
                if(np.mod(counter,2)==1):
                    #### odd beams
                    freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_odd_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),
                                                    kHz*freq_ind_odd_c1[n_seg]+(LS_amp*LS_ind_c1[n_seg]),kHz*freq_ind_odd_c2[n_seg]+(LS_amp*LS_ind_c2[n_seg]),
                                                    kHz*freq_ind_odd_c3[n_seg]+(LS_amp*LS_ind_c3[n_seg]))
                else:
                    #### even beams
                    freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_even_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),
                                                    kHz*freq_ind_even_c1[n_seg]+(LS_amp*LS_ind_c1[n_seg]),kHz*freq_ind_even_c2[n_seg]+(LS_amp*LS_ind_c2[n_seg]),
                                                    kHz*freq_ind_even_c3[n_seg]+(LS_amp*LS_ind_c3[n_seg]))
                # if((counter==5) or (counter==6)):
                #     amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter]*amp_scaling_ind_vec[counter] #### Amp_norm_ind is input by the user and is ideally 1
                # else:
                #     amp_scaling_ind = 0
                # amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter]*amp_scaling_ind_vec[counter] #### Amp_norm_ind is input by the user and is ideally 1
                amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter] #### Amp_norm_ind is input by the user and is ideally 1
                amp_spline_ind = spl.CubicSpline(amp_scaling_ind*amp_ind_c0[n_seg],amp_scaling_ind*amp_ind_c1[n_seg],amp_scaling_ind*amp_ind_c2[n_seg],amp_scaling_ind*amp_ind_c3[n_seg])

            qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_spline_ind, amplitude = amp_spline_ind,
                phase_rad=phase_ind,output_enable=False,sync=False), chan_ind_tone_0)

        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(round_fact*duration_dt))
        duration_tot_actual = duration_tot_actual+int(round_fact*duration_dt)


    if(is_quench):
        qp.play(ToneDataPulse(int(Tquench), frequency_hz = freq_red, amplitude =1*amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(Tquench), frequency_hz = freq_blue, amplitude = 1*amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM (Jij control) and FM (Stark shift control)
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            LS_amp = LS_vec[counter]
            freq_ind_carrier = f_carrier_ind_vec[counter]
            n_seg = N_segments-1
            if(np.mod(counter,2)==1):
                #### odd beams
                # freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_odd_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),
                #                                 kHz*freq_ind_odd_c1[n_seg]+(LS_amp*LS_ind_c1[n_seg]),kHz*freq_ind_odd_c2[n_seg]+(LS_amp*LS_ind_c2[n_seg]),
                #                                 kHz*freq_ind_odd_c3[n_seg]+(LS_amp*LS_ind_c3[n_seg]))
                freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_odd_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),0,0,0)
            else:
                #### even beams
                # freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_even_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),
                #                                 kHz*freq_ind_even_c1[n_seg]+(LS_amp*LS_ind_c1[n_seg]),kHz*freq_ind_even_c2[n_seg]+(LS_amp*LS_ind_c2[n_seg]),
                #                                 kHz*freq_ind_even_c3[n_seg]+(LS_amp*LS_ind_c3[n_seg]))
                freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_odd_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),0,0,0)
            if counter in chan_ind_quench:
                print(counter)
                amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter] #### Amp_norm_ind is input by the user and is ideally 1
            else:
                amp_scaling_ind = 0
            amp_spline_ind = spl.CubicSpline(amp_scaling_ind*amp_ind_c0[n_seg],0,0,0)

            qp.play(ToneDataPulse(int(Tquench), frequency_hz = freq_spline_ind, amplitude = amp_spline_ind,
                phase_rad=phase_ind+phase_quench[counter],output_enable=False,sync=False), chan_ind_tone_0)

        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(Tquench))
        duration_tot_actual = duration_tot_actual+int(Tquench)


    return [int(duration_tot_actual),phase_quench]

def Hamiltonian_quench(chan_glob_vec,f_carrier_glob,chan_ind_vec,f_carrier_ind_vec,chan_ind_pad_vec,freq_ind_pad,
    LS_vec,amp_scaling_pi_time,Amp_norm_ind =0.5,Amp_norm_global_cm =0.45,amp_imbal_ratio_b2r = 1,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,N_cut_segment_from_end = 0,is_quench = False,Tquench=100):
    ### This function generates a squeezing pulse, addressing multiple ions simultaneously by modulating the amplitude of the individual beams only.
    ### Global's amplitude & phase are assumed to be constant
    ### chan_ind_vec is the list of channels of the ions that are modulated

    ### load splines
    this_dir = pathlib.Path(__file__).parent.resolve()
    ######    CSB  Phase ########
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_med_B_sign-1_N13.json'
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_largeB_sign-1_N13.json'
    json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_LL_B_sign-1_N13.json'

    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz242_sign-1_N13.json'


    ######    XY  Phase ########
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_largeB_sign+1_N13.json'
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_medB_sign+1_N13.json'
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz032_try_sign+1_N13.json'
    # json_file = this_dir /'CSB_ramp_v2_delta_COM_kHz242_sign+1_N13.json'


    round_fact = 1#1.25
    off_ion_ind_list = []
    chan_ind_quench = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    # phase_quench = [0]*11 + [np.pi/2] + [np.pi]*11
    if(is_quench==False):
        phase_quench = [0]*23
    else:
        phase_quench = [0,np.pi/2]*11 +[0]

    data_splines = json.loads(json_file.read_text())

    duration_tot = data_splines["ramp_time_mu"] #### This is given in MU for entire pulse
    # print(duration_tot)
    duration_dt = data_splines["t_seg_mu"] #### This is given in MU for one segments
    N_segments_vec = data_splines["N_segments"]
    N_segments= int(np.sum(N_segments_vec)+N_cut_segment_from_end)

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

    LS_ind_c0 = data_splines["splines_ind_LS"]["c0"] #### Splines for odd signed ind beam in KHz
    LS_ind_c1 = data_splines["splines_ind_LS"]["c1"] #### Splines for odd signed ind beam in KHz
    LS_ind_c2 = data_splines["splines_ind_LS"]["c2"] #### Splines for odd signed ind beam in KHz
    LS_ind_c3 = data_splines["splines_ind_LS"]["c3"] #### Splines for odd signed ind beam in KHz


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
    ### Sequence
    duration_tot_actual = int(0)
    ### GLOBAL BEAM - constant
    for n_seg in range(0,N_segments):
        # print(n_seg)

        ### GLOBAL BEAM - constant
        qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_red, amplitude =1*amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_blue, amplitude = 1*amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM (Jij control) and FM (Stark shift control)
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            LS_amp = LS_vec[counter]
            freq_ind_carrier = f_carrier_ind_vec[counter]
            if counter in off_ion_ind_list:
                    freq_spline_ind = freq_ind_carrier
                    amp_scaling_ind = 0
            else:
                if(np.mod(counter,2)==1):
                    #### odd beams
                    freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_odd_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),
                                                    kHz*freq_ind_odd_c1[n_seg]+(LS_amp*LS_ind_c1[n_seg]),kHz*freq_ind_odd_c2[n_seg]+(LS_amp*LS_ind_c2[n_seg]),
                                                    kHz*freq_ind_odd_c3[n_seg]+(LS_amp*LS_ind_c3[n_seg]))
                else:
                    #### even beams
                    freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_even_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),
                                                    kHz*freq_ind_even_c1[n_seg]+(LS_amp*LS_ind_c1[n_seg]),kHz*freq_ind_even_c2[n_seg]+(LS_amp*LS_ind_c2[n_seg]),
                                                    kHz*freq_ind_even_c3[n_seg]+(LS_amp*LS_ind_c3[n_seg]))
                # if((counter==5) or (counter==6)):
                #     amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter]*amp_scaling_ind_vec[counter] #### Amp_norm_ind is input by the user and is ideally 1
                # else:
                #     amp_scaling_ind = 0
                # amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter]*amp_scaling_ind_vec[counter] #### Amp_norm_ind is input by the user and is ideally 1
                amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter] #### Amp_norm_ind is input by the user and is ideally 1
                amp_spline_ind = spl.CubicSpline(amp_scaling_ind*amp_ind_c0[n_seg],amp_scaling_ind*amp_ind_c1[n_seg],amp_scaling_ind*amp_ind_c2[n_seg],amp_scaling_ind*amp_ind_c3[n_seg])

            qp.play(ToneDataPulse(int(round_fact*duration_dt), frequency_hz = freq_spline_ind, amplitude = amp_spline_ind,
                phase_rad=phase_ind,output_enable=False,sync=False), chan_ind_tone_0)

        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(round_fact*duration_dt))
        duration_tot_actual = duration_tot_actual+int(round_fact*duration_dt)


    if(is_quench):
        qp.play(ToneDataPulse(int(Tquench), frequency_hz = freq_red, amplitude =1*amp_red,
            phase_rad=phase_red_ofst,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(int(Tquench), frequency_hz = freq_blue, amplitude = 1*amp_blue,
            phase_rad=phase_blue_ofst,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beams - AM (Jij control) and FM (Stark shift control)
        for counter in range(len(chan_ind_vec)):
            chan_ind_tone_0 = chan_ind_vec[counter]
            LS_amp = LS_vec[counter]
            freq_ind_carrier = f_carrier_ind_vec[counter]
            n_seg = N_segments-1
            if(np.mod(counter,2)==1):
                #### odd beams
                freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_odd_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),
                                                kHz*freq_ind_odd_c1[n_seg]+(LS_amp*LS_ind_c1[n_seg]),kHz*freq_ind_odd_c2[n_seg]+(LS_amp*LS_ind_c2[n_seg]),
                                                kHz*freq_ind_odd_c3[n_seg]+(LS_amp*LS_ind_c3[n_seg]))
            else:
                #### even beams
                freq_spline_ind = spl.CubicSpline(freq_ind_carrier+(kHz*freq_ind_even_c0[n_seg])+(LS_amp*LS_ind_c0[n_seg]),
                                                kHz*freq_ind_even_c1[n_seg]+(LS_amp*LS_ind_c1[n_seg]),kHz*freq_ind_even_c2[n_seg]+(LS_amp*LS_ind_c2[n_seg]),
                                                kHz*freq_ind_even_c3[n_seg]+(LS_amp*LS_ind_c3[n_seg]))
            if counter in chan_ind_quench:
                amp_scaling_ind = Amp_norm_ind*amp_scaling_pi_time[counter] #### Amp_norm_ind is input by the user and is ideally 1
            else:
                amp_scaling_ind = 0
            amp_spline_ind = spl.CubicSpline(amp_scaling_ind*amp_ind_c0[n_seg],amp_scaling_ind*amp_ind_c1[n_seg],amp_scaling_ind*amp_ind_c2[n_seg],amp_scaling_ind*amp_ind_c3[n_seg])

            qp.play(ToneDataPulse(int(Tquench), frequency_hz = freq_spline_ind, amplitude = amp_spline_ind,
                phase_rad=phase_ind+phase_quench[counter],output_enable=False,sync=False), chan_ind_tone_0)

        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(Tquench))
        duration_tot_actual = duration_tot_actual+int(Tquench)


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
    scale_time =4
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
num_ions=23
scan_type = 'quench'   #### 'calibrate_J'  |  'motional_freq'   |'imbl'| 'N_cut_seg'  | 'calibrate_J_Hamil'  | 'phi_SK1' |  'LS'    | 'quench'
is_submit_via_python = False
is_debug_sequence = False ### only for debugging
is_get_dataset_vals =True

is_LS_flag = False

is_Y = 0 #### 0 for x and 1 for Y
if((scan_type == 'calibrate_J') or (scan_type == 'motional_freq')):
    is_SK1_initial = [False] *num_ions
    is_SK1_final = [False] *num_ions
    theta_SK1_vec_initial = [np.pi]*num_ions
    theta_SK1_vec_final = [np.pi]*num_ions

elif((scan_type == 'imbl')or (scan_type=='phi_SK1')):
    is_SK1_initial = [False] *num_ions
    is_SK1_final = [False] *num_ions
    theta_SK1_vec_initial = [np.pi/2]*num_ions
    theta_SK1_vec_final = [np.pi/2]*num_ions

elif((scan_type == 'N_cut_seg')or(scan_type=='quench')):
    # is_SK1_initial = ([False,True]*11)+[False]  #### XY phase
    is_SK1_initial = [True,False]*11+[True] #### CSB phase
    is_SK1_final = [True] *num_ions # False for Z, True for X and Y
    theta_SK1_vec_initial = [np.pi]*num_ions
    theta_SK1_vec_final = [np.pi/2]*num_ions


    # is_SK1_initial[11] = True
    # theta_SK1_vec_initial[11] = np.pi/299
elif((scan_type == 'calibrate_J_Hamil')):
    ion_row_bare =-8
    N_trunc = 5
    ion_row_offset = np.int(0.5*(num_ions-1)+ion_row_bare)
    if(is_LS_flag):
        is_SK1_initial = [False] *num_ions
        is_SK1_final = [False] *num_ions
        is_SK1_initial[ion_row_offset] = True
        is_SK1_final[ion_row_offset] = True
        theta_SK1_vec_initial = [np.pi/2]*num_ions
        theta_SK1_vec_final = [np.pi/2]*num_ions
    else:
        is_SK1_initial = [False] *num_ions
        is_SK1_final = [False] *num_ions
        is_SK1_initial[ion_row_offset] = True
        is_SK1_final[ion_row_offset] = False
        theta_SK1_vec_initial = [np.pi]*num_ions
        theta_SK1_vec_final = [np.pi]*num_ions




### channels and parameters kept fixed for the 3 ions experiment
master_ip: str = "192.168.78.152"
rfsoc_map = qbe.get_default_rfsoc_map()#rfsoc_mapping.RFSoCChannelMapping(self.rfsoc_board_description)
config = QuickConfig(num_ions, rfsoc_map, {-11: 14, -10: 13, -9: 12, -8: 11, -7: 0, -6: 10, -5: 7, -4: 9,
                                        -3: 8, -2: 2, -1: 1, 0: 3, 1: 5, 2: 4, 3: 16, 4: 17, 5: 15,
                                        6: 18, 7: 6, 8: 19, 9: 20, 10: 21, 11: 22})
qp_backend = qbe.get_default_qiskit_backend(master_ip, num_ions, with_2q_gate_solutions=False)
channel_map = qbe.get_default_rfsoc_map()
qp_backend._config = config


Amp_norm_global = 0.45
Amp_norm_global_SK1 = 0.45
SK1_nominal_duration = 5217*2
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

T_pi_vec_microsec_ref_J =  0.228#np.array([0.92891288, 0.26185544, 0.26274342, 0.25277198, 0.24154937, 0.36214298, 0.35827388, 0.33143372, 0.38263681, 0.35127014, 0.25426646, 0.28588474, 0.26714632, 0.28037115, 1.25087551])
amp_scaling_pi_time_J = (T_pi_vec_microsec/T_pi_vec_microsec_ref_J)

# if(scan_type == 'calibrate_J_Hamil'):
amp_ref_J= np.ones(num_ions)
amp_ref_J[0] = 0.8
amp_ref_J[0] = 0.7
amp_ref_J[1] = 0.8
amp_ref_J[2] = 0.85
amp_ref_J[3] = 0.9
amp_ref_J[4] = 0.9


amp_scaling_pi_time_J = amp_ref_J*amp_scaling_pi_time_J

### Main Schedule
schedules = []
freq_com =3.31080e6 + 5.5e3#3.04e6#
detuning_vec = [1.75e4]

phi_offset_2ndSK1 = -0.06 ### measured in RID 293253
phi_offset_XX_int = 0.0 ### measured in RID 293253
phi_SK1_vec = [0]
time_J_mu_vec = [3e5]
if(scan_type=='motional_freq'):
    amp_ind_scale = 0.01#0.2 ### we measured J of 3.8765 for amplitude of 0.2 (including the Tpi relative calibration)
else:
    amp_ind_scale = 0.06#0.2 ### we measured J of 3.8765 for amplitude of 0.2 (including the Tpi relative calibration)

if(scan_type=='quench'):
    is_quench = True
else:
    is_quench = False

amp_imbal_ratio_b2r_vec = [1.017]
LS_scan_vec = [0]
N_cut_seg_vec =[0] #### how many segments to cut from the end of the main pulse
Tquench_vec = [100]
LS_vec_calib = np.zeros(num_ions)



if(scan_type == 'calibrate_J'):
    T_final =int(5e6)
    Npts = 13
    time_J_mu_vec =np.array([*range(100, T_final, np.int(T_final/Npts))])
    calibration_pair_per_exp =[-1,0] ##[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]#
    print(time_J_mu_vec)

elif(scan_type == 'motional_freq'):
    Npts = 61
    detuning_vec = 0*detuning_vec[0]+np.linspace(-2e4,2e4,Npts)
    calibration_pair_per_exp =[-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11]#
    print(detuning_vec)

elif(scan_type == 'phi_SK1'):
    Npts = 17
    # phi_SK1_vec = np.linspace(0,2*np.pi,Npts)
    phi_SK1_vec = np.linspace(0,2*np.pi,Npts)
    calibration_pair_per_exp = [-11,-10] ##[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]#
    print(phi_SK1_vec)

elif(scan_type == 'imbl'):
    Npts = 31
    amp_imbal_ratio_b2r_vec = 1.018+np.linspace(-0.015,0.015,Npts)
    calibration_pair_per_exp =[0]#
    print(phi_SK1_vec)

elif(scan_type == 'LS'):
    Npts = 13
    LS_scan_vec = np.linspace(-4e2,4e2,Npts)
    calibration_pair_per_exp =[-1,0]#
    print(LS_scan_vec)

elif(scan_type == 'N_cut_seg'):
    # N_cut_seg_vec =np.array([i for i in range(-50,1,10)])
    N_cut_seg_vec =np.array([-2,-1,0])
    # N_cut_seg_vec =np.array([i for i in range(-50,1,5)])
    calibration_pair_per_exp =[-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11]#
    # calibration_pair_per_exp =[-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]#
    # calibration_pair_per_exp =[-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]#
    # calibration_pair_per_exp =[-3,-2,-1,0,1,2,3]#
    # calibration_pair_per_exp =[-5,-4,-3,-2,-1,0,1,2,3,4,5]#

    # calibration_pair_per_exp =[-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11]#
    Npts = N_cut_seg_vec.shape[0]
    print('cutting segments!!')
    print(N_cut_seg_vec)

elif(scan_type == 'quench'):
    # calibration_pair_per_exp =[-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11]#
    calibration_pair_per_exp =[-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]#
    T_final =int(1.8e6)
    Npts = 13
    Tquench_vec =np.array([*range(100+0, 1*T_final, np.int(T_final/Npts))])
elif((scan_type == 'calibrate_J_Hamil')):
    # Tlist = [[0,0.84722,1.3273,1.9643,2.8424,4.2633,6.482,10.0603,17.5988,20,20,20,20],[0,0,0.84212,1.3156,1.9608,2.9851,4.5646,6.9722,11.5383,20,20,20,20],[0,0,0,0.84853,1.3344,2.0901,3.2242,4.8998,7.8915,12.5021,20,20,20],[0,0,0,0,0.83964,1.3872,2.1939,3.3508,5.276,8.1252,13.2547,19.3553,20],[0,0,0,0,0,0.83989,1.4019,2.1811,3.4511,5.1245,8.1386,10.7024,17.9351],[0,0,0,0,0,0,0.85307,1.4021,2.2481,3.3586,5.2067,6.6803,10.5851],[0,0,0,0,0,0,0,0.83964,1.423,2.1631,3.3774,4.3046,6.7245],[0,0,0,0,0,0,0,0,0.83965,1.3486,2.1552,2.7762,4.3581],[0,0,0,0,0,0,0,0,0,0.843,1.4229,1.8829,3.0029],[0,0,0,0,0,0,0,0,0,0,0.8816,1.2322,2.0224],[0,0,0,0,0,0,0,0,0,0,0,0.83964,1.4551],[0,0,0,0,0,0,0,0,0,0,0,0,0.83965],[0,0,0,0,0,0,0,0,0,0,0,0,0]]
    Tlist = [[0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20,20,20,20,20,20,20,20,20],[0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20,20,20,20,20,20,20,20],[0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20,20,20,20,20,20,20],[0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20,20,20,20,20,20],[0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20,20,20,20,20],[0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20,20,20,20],[0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20,20,20],[0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20,20],[0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20,20],[0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20,20],[0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649,20],[0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976,19.7649],[0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85,17.7976],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239,15.85],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218,13.9239],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468,12.0218],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029,10.1468],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957,8.3029],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336,6.4957],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304,4.7336],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137,3.0304],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.4137],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    # print(Tlist)
    Npts_time = 9
    time_J_mu_vec = []
    calib_ind_vec = []
    for j in range(ion_row_offset+1,ion_row_offset+N_trunc+1):
        if(is_LS_flag):
            time_factor = 2.5
        else:
            time_factor = 3.5
        T_final = int((4.09e5)*np.copy(np.min([16,time_factor*np.array(Tlist)[ion_row_offset,j]])))
        print(T_final)
        time_J_mu_vec_short =[*range(100, T_final, np.int(T_final/Npts_time))]
        time_J_mu_vec = time_J_mu_vec + time_J_mu_vec_short
        calib_ind_vec_short = [j]*Npts_time
        calib_ind_vec = calib_ind_vec + calib_ind_vec_short

    # print(time_J_mu_vec)
    # print(calib_ind_vec)

counter_scan = 0
for Tquench in Tquench_vec:
    for N_cut_seg in N_cut_seg_vec:
        for time_J_mu in time_J_mu_vec:
            for detuning in detuning_vec:
                for phi_SK1 in phi_SK1_vec:
                    for amp_imbal_ratio_b2r in amp_imbal_ratio_b2r_vec:
                        for LS_scan in  LS_scan_vec:
                            with qp.build(backend=qp_backend) as out_sched:

                                if(scan_type == 'calibrate_J_Hamil'):
                                    calibration_pair_offset = [ion_row_offset,calib_ind_vec[counter_scan]]
                                    print(calibration_pair_offset)
                                else:
                                    calib_ind_vec = [calibration_pair_per_exp for i in range(Npts)]
                                    calibration_pair =calib_ind_vec[counter_scan]
                                    print(calibration_pair)
                                    calibration_pair_offset =  np.array([int((num_ions-1)*0.5)],int)+np.array(calibration_pair,int)

                                ### carrier
                                freq_carrier_glob = qp_backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
                                freq_global_vec_SB0 = np.array([freq_carrier_glob,freq_carrier_glob]) # red and blue tones' frequencies in Hz for carrier
                                freq_ind_vec = np.array([qp_backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value]*num_ions)
                                phase_correction0 = 0 ## correction of carrier phase w.respect to the current frequency of sideband 1

                                ### 1st sidebands
                                w_k = freq_com + detuning
                                freq_global_vec_SB1 = np.array([freq_carrier_glob-w_k,freq_carrier_glob+w_k]) # red and blue tones' frequencies in Hz for first sideband
                                phase_correction1 = 0 ## correction of motion_phase_offset_squzcarrier phase w.respect to the current frequency of sideband 1

                                LS_vec_calib_exp = LS_vec_calib[calibration_pair_offset]

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
                                    tanh_scale_factor = np.array([0.9854540668066544, 1.0161854854964625, 1.013523380742869, 1.007163657596779, 0.9929102305176052, 0.9959195798349382, 0.9971507829092647, 0.9908805061520014, 0.9975045535618642, 1.0027782098978542, 0.9942014803467186, 0.9856723536333182, 0.990783620600121, 0.9968945119570908, 0.9954115820003856, 1.0227979776191682, 1.0201266883538544, 1.0183452874422447, 1.014520651892784, 0.9870736907114992, 0.9771226792396365, 0.9825383099838269, 1.0654811611269686])*np.array([0.9916346232878552, 0.9460941975170369, 0.9566872082469615, 0.9555097840556399, 0.9445424509411736, 0.9510158159579819, 0.9592498852684893, 1.0036276976212744, 0.9686738143877522, 1.0107639643599653, 0.9873569721702156, 0.9606337249094362, 0.9604469888939018, 0.9611021921629044, 0.9150858924707299, 0.9213586083218156, 0.9481894644891496, 0.9428299501838813, 0.987470320168479, 0.9757381070718808, 0.9857229270503546, 0.998455000443369, 0.9757351748223905])*np.array([1.0538664034319545, 1.0718021543095189, 1.0577819265170791, 1.0557182477012215, 1.0760577531080364, 1.0775119826588795, 1.0726292207761365, 1.049724035001543, 1.038890181245028, 1.01887925960003, 1.015040644005963, 1.009741029857695, 1.04240091121784, 1.0431176237543798, 1.0405936389305275, 1.0762822467033997, 1.0287061227925767, 1.0093963707607536, 0.9619294512978723, 0.9435176150048181, 0.8982312743115383, 0.9324613664546328, 1.0090335483258712])*np.array([0.9292679534420171, 1.0096132905591992, 1.0385994588390117, 1.0199993052329557, 0.9896657883462842, 0.9378479969656163, 0.9298647621365772, 0.924108489252452, 0.9349000713979482, 0.9209305697152602, 0.9669390870525592, 1.0201008562991312, 1.021936964567396, 1.0285625814769623, 1.0704025288309307, 1.0335753905259388, 1.0478354972282657, 1.0222831233421128, 1.0321893882655244, 1.0237047942495925, 1.040193984575578, 1.0329888316635785, 1.0225644790082717])*np.array([1.0682844699302123, 0.9038937186597191, 0.9391188615075962, 0.9478466874461293, 0.9721660246689774, 0.9683194373095857, 0.9363597124815254, 0.9043307530664857, 0.8720513172522002, 0.9392908018916095, 0.9418096004918323, 0.9657116475331062, 0.957812462783673, 0.9514653427306475, 0.9341509515907037, 0.9447869548547483, 0.94564124973909, 0.969624329275144, 0.9801924135084769, 0.9482458678402715, 0.9187547117573144, 0.9119592057164415, 1.0831651375170623])*np.array([1.0606298899106223, 0.9581731702941048, 0.9168777832938718, 0.9404584627333517, 0.9678982964220223, 1.0347223882073575, 1.0993241046215094, 1.1485770018364287, 1.1554216761930003, 1.114437919432126, 1.0666388488075604, 0.9901535749470624, 0.9394389737393825, 0.8782321541620414, 0.82271690382986, 0.8474636455954148, 0.8715154466226649, 0.9392876222504984, 1.0144646806440185, 1.1145112254279235, 1.1719055657174717, 1.1125960954195444, 0.8741452189210155])*np.array([0.9230609992333972, 1.0242499526051638, 1.061255986418668, 1.0329938947072923, 1.0001793720169518, 0.9494295239856235, 0.91160021100471, 0.897464654656663, 0.8957166311996176, 0.9238844830078102, 0.9536176317167988, 0.9966670697956068, 1.0352296224860082, 1.0773856033548106, 1.1424871682198252, 1.1090442200382389, 1.092195847617751, 1.026583581805756, 0.9771445491439448, 0.9129482713603025, 0.8813554727092742, 0.9140780528425153, 1.089111050793778])*np.array([1.08317943687904, 0.9947089372472793, 0.9708953128514867, 1.0058678457892847, 1.0490066416783765, 1.0732355990714697, 1.0903262554453805, 1.0742253915503688, 1.0759573808673348, 1.0751562806079868, 1.0581002782938485, 1.0630546228299027, 1.0535470945765821, 1.0554771606742426, 1.0378434898458766, 1.0433567968505055, 1.0509699387251856, 1.082815296569976, 1.0717221795522258, 1.080781371826523, 1.0740639343314191, 1.0852340361619623, 1.1228240129111382])*np.array([1.0066859205585754, 1.1084701971963171, 1.148896991638129, 1.0896387462312958, 1.0304230311348814, 0.995555264751019, 0.9682512245979944, 0.9553508036878557, 0.9629854462216898, 0.9700494675505318, 0.9779711449237706, 0.9981197490894621, 1.0057294633044183, 1.034721533983873, 1.037800601107395, 1.050600978415855, 1.0266866320348083, 1.0030800584800468, 0.9965232968870245, 0.9939125904272633, 0.9915792681377336, 1.0059752994024922, 1.0112821452341845])*np.array([0.8340890808591064, 0.9776883963114388, 1.0052028236266908, 0.9915514909281493, 0.9869001986198694, 0.9790026410439283, 1.0149186668951522, 1.0236515396917512, 1.029192410762835, 1.0053285413001698, 0.9948701842779053, 0.983163330592007, 0.9820851862368223, 1.0160765405491765, 1.0258660905192067, 1.0234081924517737, 1.013884636987897, 0.9735977103949804, 0.9618937748974996, 0.9726696812920402, 0.9954028466817593, 0.9686160158563609, 0.8524278631081691])*np.array([1.530091316236628, 1.0988311617092028, 0.8854135904340471, 0.9294296580467925, 0.9607585589770087, 1.1250708289195157, 1.194550016487324, 1.2613139662427875, 1.3259249756743723, 1.2591498652524125, 1.1886706148723807, 1.1038853030487772, 1.0537036357617753, 0.9629889325879264, 0.9345650231415438, 0.948053656782288, 1.0123198286560047, 1.1446236394609837, 1.2420223923582332, 1.3904160082689296, 1.4679777312956939, 1.2916569950172514, 0.8340434138439796])*np.array([1.40372, 1.047  , 1.05722, 1.06391, 1.05211, 1.09141, 1.09983,1.08811, 1.07479, 1.10901, 1.06394, 1.05624, 1.04849, 1.05671, 1.02725,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05])*np.array([0.5678988986101344, 0.9845408345880869, 1.0897854520628518, 1.0421092183149139, 1.0548354756482519, 0.862329750451962, 0.8081825239393917, 0.7989695928542555, 0.7958668312888997, 0.793534059957885, 0.8844910059134918, 0.9152543461724242, 0.980424347148316, 1.0292957728312067, 1.0262109081861062, 1.016124249151123, 0.9722431847071622, 0.8615244189390483, 0.8482978684032405, 0.7391443683457082, 0.7196367739570299, 0.8134684192701348, 1.1544211576109344])*np.array([0.9271020106331694, 0.9837614127899345, 1.0105556591541902, 1.0156475997145267, 0.9996532571781608, 1.0091399120562534, 1.003753622529971, 0.9964666583057011, 0.982802966118653, 1.0046944829948337, 1.0044876474234075, 1.00483552051891, 1.0068424336398765, 1.0066871750776165, 1.0408118107421958, 1.0016783629906265, 1.0012319908816305, 1.004987514051287, 0.9788992626351499, 0.9818934317176278, 0.9641731326672401, 0.9828631221445137, 1.0175681683329403])
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

                                if((scan_type == 'calibrate_J')):
                                    print(calibration_pair_offset)
                                    is_SK1_initial[calibration_pair_offset[0]] = True
                                    # is_SK1_final[calibration_pair_offset[0]] = True
                                    # is_SK1_initial[calibration_pair_offset[1]] = True
                                    # is_SK1_final[calibration_pair_offset[1]] = True
                                elif((scan_type == 'phi_SK1')):
                                    is_SK1_initial[calibration_pair_offset[0]] = True
                                    is_SK1_final[calibration_pair_offset[0]] = True
                                    is_SK1_initial[calibration_pair_offset[1]] = True
                                    is_SK1_final[calibration_pair_offset[1]] = True

                                #########   Experimental sequence   ######
                                ## Initial sync of phases (b.c. sync is off)
                                print('')
                                print('Exp no.' +str(counter_scan+1))
                                initial_sync(ch_global_vec = chan_glob_vec, ch_ind_vec = chan_ind_vec,freq_global = freq_global_vec_SB0,freq_ind_list=freq_ind_vec)

                                ## initial SK1 pulse ##
                                phi0_SK1_vec = [0*np.pi]*num_ions
                                T_SK1_tot = sequential_sk1(applied_gates=is_SK1_initial, pi_times=T_pi_vec_microsec,
                                                thetas=np.array(theta_SK1_vec_initial), phis=np.array(phi0_SK1_vec)+phi_SK1)
                                phase_correction1 = phase_correction1 +2*np.pi*qp.samples_to_seconds(T_SK1_tot)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0]) ### we put this form because it is a correction on the motional frequency
                                print('duration of SK1 in mu' +str(T_SK1_tot))


                                # delay_all(time_J_mu)
                                indices_pad_calib = indices_pad_array(len(freq_ind_vec),calibration_pair_offset)

                                ## J_matrix calibration ##
                                if((scan_type=='calibrate_J') or (scan_type == 'motional_freq') or(scan_type=='phi_SK1') or (scan_type == 'imbl') or (scan_type == 'calibrate_J_Hamil')):
                                    [duration_calibrate] = calibrate_J_pair(chan_glob_vec,f_carrier_glob = freq_global_vec_SB1,chan_ind_vec = chan_vec_pad(chan_ind_vec,calibration_pair_offset),
                                                                            f_carrier_ind_vec = freq_ind_vec[calibration_pair_offset],chan_ind_pad_vec = chan_vec_pad(chan_ind_vec,indices_pad_calib),
                                                                            freq_ind_pad = freq_ind_vec[indices_pad_calib],duration = time_J_mu,LS_vec= LS_vec_calib_exp,
                                                                            amp_scaling_pi_time = amp_scaling_pi_time_J[calibration_pair_offset],Amp_norm_ind =amp_ind_scale,
                                                                            Amp_norm_global_cm =Amp_norm_global,amp_imbal_ratio_b2r = amp_imbal_ratio_b2r,phi_spin = phi_offset_XX_int,
                                                                            phi_motion_offset = phase_correction1,zero_pad_flag = True)
                                    phase_correction0 = phase_correction0 + 2*np.pi*qp.samples_to_seconds(duration_calibrate)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0])
                                    print('duration of J calibration in mu' +str(duration_calibrate))

                                elif((scan_type == 'N_cut_seg') or (scan_type=='quench')):
                                    [duration_Hamiltonian,phase_quench] = Hamiltonian_evolution(chan_glob_vec,f_carrier_glob = freq_global_vec_SB1,chan_ind_vec = chan_vec_pad(chan_ind_vec,calibration_pair_offset),
                                                                            f_carrier_ind_vec = freq_ind_vec[calibration_pair_offset],chan_ind_pad_vec = chan_vec_pad(chan_ind_vec,indices_pad_calib),
                                                                            freq_ind_pad = freq_ind_vec[indices_pad_calib],LS_vec = LS_vec_calib_exp,
                                                                            amp_scaling_pi_time= amp_scaling_pi_time_J[calibration_pair_offset],Amp_norm_ind =amp_ind_scale,
                                                                            Amp_norm_global_cm =Amp_norm_global,amp_imbal_ratio_b2r = amp_imbal_ratio_b2r,phi_spin = phi_offset_XX_int,
                                                                            phi_motion_offset = phase_correction1,zero_pad_flag = True,N_cut_segment_from_end = N_cut_seg,is_quench=is_quench,Tquench=Tquench)
                                    phase_correction0 = phase_correction0 + 2*np.pi*qp.samples_to_seconds(duration_Hamiltonian)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0])
                                    print('duration of Hamiltonian in mu' +str(duration_Hamiltonian))
                                if(scan_type=='quench'):
                                    phase_quench = np.array([0,0,np.pi] + [0]*20)#np.array([0]*11 + [np.pi/2] + [np.pi]*11)#1*np.array(phase_quench)#[0]*11 + [np.pi/2] + [np.pi]*11#
                                    phase_quench = np.array( [0]*20+ [np.pi,0,0] )#np.array([0]*11 + [np.pi/2] + [np.pi]*11)#1*np.array(phase_quench)#[0]*11 + [np.pi/2] + [np.pi]*11#
                                    phase_quench = np.array( [0]*2+[0]*9 + [np.pi/2] + [np.pi]*9+[0]*2)
                                    # phase_quench = np.zeros(num_ions)
                                else:
                                    phase_quench = np.zeros(num_ions)

                                ##Final SK1 pulse ##
                                if(is_LS_flag):
                                    phi_offset_LS = np.pi/2
                                else:
                                    phi_offset_LS = 0

                                T_SK1_tot2 = sequential_sk1(applied_gates=is_SK1_final, pi_times=T_pi_vec_microsec,
                                                thetas=np.array(theta_SK1_vec_final), phis=is_Y*0.5*np.pi+phase_quench+np.array(phi0_SK1_vec)+np.pi+phase_correction0+phi_offset_LS+phi_offset_2ndSK1+phi_SK1)
                                phase_correction1 = phase_correction1 +2*np.pi*qp.samples_to_seconds(T_SK1_tot2)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0]) ### we put this form because it is a correction on the motional frequency
                                print('duration of SK1 in mu' +str(T_SK1_tot2))

                                if((scan_type == 'calibrate_J')):
                                    is_SK1_initial[calibration_pair_offset[0]] = False
                                    is_SK1_final[calibration_pair_offset[0]] = False

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
    x_label = "light shift [kHz]"
elif(scan_type == 'N_cut_seg'):
    x_values = N_cut_seg_vec
    x_label = "number of cut segments from end"
elif(scan_type =='quench'):
    x_values = (1e-6)*(1000/409)*np.array(Tquench_vec)
    x_label = "T_quench"
elif(scan_type == 'calibrate_J_Hamil'):
    x_values = (1e-6)*(1000/409)*np.array(time_J_mu_vec)
    x_label = "time_J [ms]"

print("CSB_evolution_v1 experiment was submitted")

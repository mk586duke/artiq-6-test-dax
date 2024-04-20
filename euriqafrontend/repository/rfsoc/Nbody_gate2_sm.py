"""Temporary file for testing arbitrary Qiskit Pulse schedules."""
from email.policy import default
import importlib
import pathlib
import json
import logging

import artiq.language.environment as artiq_env
import oitg.fitting as fit
from artiq.experiment import TerminationRequested
from artiq.language.core import delay, delay_mu, host_only, kernel
from qiskit.assembler import disassemble
from artiq.language.units import us
import numpy as np
import qiskit.pulse as qp

import euriqafrontend.fitting as umd_fit
from euriqafrontend.repository.basic_environment import BasicEnvironment
import euriqafrontend.modules.population_analysis as pop_an
import euriqafrontend.modules.rfsoc as rfsoc
import artiq.language.scan as artiq_scan
from pulsecompiler.qiskit.pulses import CubicSplinePulse, ToneDataPulse
import pulsecompiler.rfsoc.structures.splines as spl
from euriqabackend import _EURIQA_LIB_DIR
import euriqabackend.waveforms.single_qubit as single_qubit


_LOGGER = logging.getLogger(__name__)


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
    phi_motion_offset = 0,zero_pad_flag = True,n_edge = 0,LS = 0):
    ### This function generates a single displacement pulse for one ion by modulating the amplitude of the individual beam and modulating the relative phase between the red and blue tones of the global beam
    ### Assumes that chan_ind is a single channel

    ### load splines
    this_dir = _EURIQA_LIB_DIR / "tmp/Nbody_gate/" # pathlib.Path(__file__).parent.resolve()
    # json_file = this_dir / "new_splines_displace_d_xxNN_mode3_pulsedur026_isAMPM0.json"
    json_file = this_dir / "new_july_splines_displace_xxNN_mode3_pulsedur026_isAMPM0.json"

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
    freq_ind = f_carrier_ind+LS

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
    Amp_norm_ind =0.5,Amp_norm_global_cm =0.3,amp_imbal_ratio_b2r = 1,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,n_corner = 0,LS = 0):
    ### This function generates a squeezing pulse, addressing multiple ions simultaneously by modulating the amplitude of the individual beams only.
    ### Global's amplitude & phase are assumed to be constant
    ### chan_ind_vec is the list of channels of the ions that are modulated

    ### load splines
    this_dir = _EURIQA_LIB_DIR / "tmp/Nbody_gate/"  # pathlib.Path(__file__).parent.resolve()
    json_file = this_dir / "splines_squeeze_compact__mode3_pulsedur029.json"
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
            freq_ind = f_carrier_ind_vec[counter]+LS
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
    qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob, amplitude =Amp_norm_global,
        phase_rad=phi + pulse_phases[0],output_enable=False,sync=False, _name="sk1_rotate_global"), chan_global)
    for spline_ind, duration in zip(rotation_splines_ind, duration_vec):
        qp.play(ToneDataPulse(duration, frequency_hz = freq_ind+detuning_Hz, amplitude =spline_ind,
            phase_rad=0,output_enable=False,sync=False, _name="sk1_rotate_individual"), chan_ind)

    if(theta>0.0):
        ## correction pulse 0
        correction_amplitude = 2 * (Tpi/1e6) / qp.samples_to_seconds(sum(duration_vec))
        correction_coeffs_scaled = spline_coeffs * correction_amplitude * rect_to_tanh_scale_correction
        correction_splines_ind = [spl.CubicSpline(*coeffs) for coeffs in correction_coeffs_scaled]
        qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob, amplitude =Amp_norm_global,
            phase_rad=phi + pulse_phases[1],output_enable=False,sync=False, _name="sk1_correct1_global"), chan_global)
        for spline_ind, duration in zip(correction_splines_ind, duration_vec):
            qp.play(ToneDataPulse(duration, frequency_hz = freq_ind+detuning_Hz, amplitude =spline_ind,
                phase_rad=0,output_enable=False,sync=False, _name="sk1_correct1_individual"), chan_ind)

        # # # correction pulse 1
        qp.play(ToneDataPulse(duration_total, frequency_hz = freq_glob, amplitude =Amp_norm_global,
            phase_rad=phi + pulse_phases[2],output_enable=False,sync=False, _name="sk1_correct2_global"), chan_global)

        for spline_ind, duration in zip(correction_splines_ind, duration_vec):
            qp.play(ToneDataPulse(duration, frequency_hz = freq_ind+detuning_Hz, amplitude =spline_ind,
                phase_rad=0,output_enable=False,sync=False, _name="sk1_correct2_individual"), chan_ind)

        ###padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(3*duration_total))

    else:
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,int(1*duration_total))

    return 3*duration_total


### channels and parameters kept fixed for the 3 ions experiment
Ions_displacement_ind = np.array([0,2],int)
Ions_squeezing_ind = np.array([1],int)
Amp_norm_global_cm_disp = 0.45
Amp_norm_global_cm_squz = 0.45
tanh_scale_factor = [0.0689,0.068,0.0681]# SK1 calibration parameters these values were calibrated for Tpi=np.array([4.556,4.062,4.624]) at 0.06 amp individual and 0.45 amp global and will be used in the SK1 as is
SK1_nominal_duration = 5217
phi_per_ramsey = 0*(0.5372+0.026) #### phase accumulated for displacement or squeezing per SK1 pulse

if((Amp_norm_global_cm_disp>=0.5) or (Amp_norm_global_cm_squz>=0.5)):
    print("error!! amplitude of two tones signal is saturated above 0.5")

##### calibrated at amp global 0.45 and am ind=0.06. then we get nominal values of T_pi_vec_microsec = np.array([4.556,4.062,4.624])
T_pi_vec_microsec_ref_lightshift =  np.array([3.643,3.17,3.662])##np.array([4.411,3.823,4.429])#this should be
T_pi_vec_microsec_ref_disp =  np.array([3.859,3.307,3.856])##np.array([4.411,3.823,4.429])#this should be
T_pi_vec_microsec =  np.array([3.765,3.292,3.826])##np.array([4.411,3.823,4.429])#this should be

### Main Schedule
amp_imbal_ratio_b2r_disp = np.array([1.009])#np.array([1.015])#

amp_imbal_ratio_b2r_squz = 1.018# at 0.7 ind amp it is 1.018

phiSK1_vec_bare = np.array([0,0.18,0])
Phi12 = 1.7
phi_Parity_vec = [-0]
LS_carr = 1.3e3
LS_sqz = 0.0
LS_disp = 0.0
phi_disp_per_ramsey = 0
phi_squeez_per_ramsey =0


default_amp_disp = 0.98
default_amp_sqz = 0.57
default_w_k = 2.808e6


def Nbody_schedule_calibrate(qp_backend,w_k,Amp_norm_ind_disp,Amp_norm_ind_squz,Flag_calibration,spin_state_up_X=True):
    ### Run flags
    if(Flag_calibration=="Freq"):
        seq_type1 = 'Sqz_Clos' #### 'Displacement'  |  'Squeezing'  |  'Both' |  'Sqz_Clos'  |
        seq_type2 = 'Gate' #### 'Closure'  |  'Gate'  |
        scan_type = 'Freq' #### 'Freq'  |  'ImbGlob'   |  'IndAmp'  | 'Tdelay'  |  'phiSK1'  | 'Phi12'
        is_Ramsey = False
        is_Echo = False
        is_initial_pi_pulse = False
        N_stg = 4
        spin_flip =  [True,False,True]
        spin_ramsey = [False,False , False]
        spin_echo = [True,True , True]
        phiSK1 = 0
        N_gate_rep = 1

    elif(Flag_calibration=="AmpDisp"):
        seq_type1 = 'Both' #### 'Displacement'  |  'Squeezing'  |  'Both'
        seq_type2 = 'Gate' #### 'Closure'  |  'Gate'  |
        scan_type = 'IndAmp' #### 'Freq'  |  'ImbGlob'   |  'IndAmp'  | 'Tdelay'  |  'phiSK1'  | 'Phi12'
        is_Ramsey = False
        is_Echo = False
        is_initial_pi_pulse = True
        N_stg = 4
        spin_flip =  [True,False,True]
        spin_ramsey = [False,True , False]
        spin_echo = [True,True , True]
        N_gate_rep = 2
        if(spin_state_up_X):
            phiSK1 = np.pi
        else:
            phiSK1 = 0

    if(seq_type1=='Displacement'):
        tau_wait= int(10000)# [int(14000)]given in machine units
    elif(seq_type1=='Squeezing'):
        tau_wait= int(10600)# given in machine units
    elif((seq_type1=='Both') or (seq_type1=='Sqz_Clos')):
        tau_wait= int(0)# given in machine units

    counter_scan = 0
    with qp.build(backend=qp_backend) as out_sched:

        w_k_squz = w_k
        #########   frequency & phases assignment   ######
        ### carrier
        freq_carrier_glob = qp_backend.properties().frequencies.global_carrier_frequency.value
        freq_global_vec_SB0 = np.array([freq_carrier_glob,freq_carrier_glob]) # red and blue tones' frequencies in Hz for carrier
        freq_ind_vec = np.array([qp_backend.properties().frequencies.individual_carrier_frequency.value]*3)
        T_echo = 0

        ### 1st sidebands
        spin_phase_disp = [0,0,0,0]
        phase_correction1 = 0 ## correction of carrier phase w.respect to the current frequency of sideband 1
        phase_correction_wait1 = 0 ## correction of motion_phase_offset_squzcarrier phase w.respect to the current frequency of sideband 1
        phase_correction1_corner = 0
        phase_correction_spin1 = 0
        if((seq_type2=='Closure') or (seq_type1== 'Sqz_Clos')):
            if((scan_type=='phiSK1') or (scan_type=='ImbGlob')):
                motion_phase_offset_disp = 0*np.array([0*np.pi,1*(np.pi),0.5*(np.pi),1.5*(np.pi)]) ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction
            else:
                motion_phase_offset_disp = np.array([0*np.pi,0.5*(np.pi),1.5*(np.pi),1*(np.pi)]) ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction
            N_stg_vec =[0,1,1,0]
        elif(seq_type2=='Gate'):
            motion_phase_offset_disp = np.array([0*np.pi,0.5*(np.pi),1*(np.pi),1.5*(np.pi)]) ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction
            N_stg_vec =[0,1,0,1]
        ion_idx_disp_vec  = [Ions_displacement_ind[N_stg_vec[0]],Ions_displacement_ind[N_stg_vec[1]],Ions_displacement_ind[N_stg_vec[2]],Ions_displacement_ind[N_stg_vec[3]]] #[Ions_displacement_ind[0],Ions_displacement_ind[1],Ions_displacement_ind[0],Ions_displacement_ind[1]] ### the ion that is displaced as a function of the stage number
        freq_global_vec_SB1 = np.array([freq_carrier_glob-w_k,freq_carrier_glob+w_k]) # red and blue tones' frequencies in Hz for first sideband


        ### 2nd sidebands
        spin_phase_squz = 0.03+ np.array([0,0,0,0])+np.pi/2
        phase_correction2 = 0  ## correction of carrier phase w.respect to the current frequency of sideband 2
        phase_correction_wait2 =0
        phase_correction2_edge = 0
        phase_correction_spin2 =0
        if(((scan_type=='ImbGlob')or(scan_type=='phiSK1')) and (seq_type2=='Closure')):
            motion_phase_offset_squz = Phi12+0*np.array([0,1*np.pi,0,1*np.pi]) ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction
        else:
            motion_phase_offset_squz = Phi12+np.array([0,1*np.pi,0,1*np.pi]) ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction
        freq_global_vec_SB2 = np.array([freq_carrier_glob-2*w_k_squz,freq_carrier_glob+2*w_k_squz]) # red and blue tones' frequencies in Hz for second sideband

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
                        detuning_Hz=LS_carr,
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
        initial_sync(ch_global_vec = chan_glob_vec, ch_ind_vec = chan_ind_vec,freq_global = freq_global_vec_SB0,freq_ind_list=freq_ind_vec)

        if(is_Ramsey):
            T_ramsey1 = sequential_sk1(applied_gates=spin_ramsey, pi_times=T_pi_vec_microsec,
                            thetas=np.array([np.pi/2,np.pi/2,np.pi/2]), phis=np.array([0.05,0.05, 0.05])+phiSK1_vec_bare+phiSK1)
            phase_correction_spin1 = phase_correction_spin1 +(T_ramsey1/SK1_nominal_duration)*phi_disp_per_ramsey ### we put this form because it is a correction on the motional frequency
            phase_correction_spin2 = phase_correction_spin2 +(T_ramsey1/SK1_nominal_duration)*phi_squeez_per_ramsey ### we put this form because it is a correction on the motional frequency
            phase_correction_wait1 = phase_correction_wait1 +2*np.pi*qp.samples_to_seconds(T_ramsey1)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0]) ### we put this form because it is a correction on the motional frequency
            phase_correction_wait2 = phase_correction_wait2 +2*np.pi*qp.samples_to_seconds(T_ramsey1)*(freq_global_vec_SB0[0]-freq_global_vec_SB2[0]) ### we put this form because it is a correction on the motional frequency
            print('duration of Ramsey1 in mu' +str(T_ramsey1))


        def displacement_stage(nn,_motion_phase_offset_disp,_phase_correction1,_phase_correction2_edge,_phase_correction_spin1,_phase_correction_wait1,_phase_correction1_corner):
            ion_idx_stg = ion_idx_disp_vec[nn]
            ion_idx_stg_array = [ion_idx_stg]
            phase_motion_disp = _motion_phase_offset_disp[nn]+_phase_correction_wait1+_phase_correction1_corner
            indices_pad_disp = indices_pad_array(len(freq_ind_vec),ion_idx_stg_array)
            [duration_disp] = edge(chan_glob_vec=chan_glob_vec,f_carrier_glob = freq_global_vec_SB1, chan_ind=chan_ind_vec[ion_idx_stg],
                                        f_carrier_ind = freq_ind_vec[ion_idx_stg],chan_ind_pad_vec = chan_vec_pad(chan_ind_vec,indices_pad_disp),
                                        freq_ind_pad = freq_ind_vec[indices_pad_disp],Amp_norm_ind =Amp_norm_ind_disp,Amp_norm_global_cm =Amp_norm_global_cm_disp,
                                        amp_imbal_ratio_b2r = amp_imbal_ratio_b2r_disp, phi_spin = spin_phase_disp[nn]+_phase_correction_spin1,
                                        phi_motion_offset = phase_motion_disp,n_edge = N_stg_vec[nn],LS = LS_disp) ### displacement of a single ion

            _phase_correction1 = _phase_correction1+ 2*np.pi*qp.samples_to_seconds(duration_disp)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0])
            _phase_correction2_edge = _phase_correction2_edge + 2*np.pi*qp.samples_to_seconds(duration_disp)*(freq_global_vec_SB1[0]-freq_global_vec_SB2[0])
            print('duration of displacement in mu' +str(duration_disp))
            return _phase_correction1,_phase_correction2_edge

        def squeezing_stage(nn,_motion_phase_offset_squz,_phase_correction2,_phase_correction1_corner,_phase_correction_spin2,_phase_correction_wait2,_phase_correction2_edge):
            phase_motion_squz = _motion_phase_offset_squz[nn]+_phase_correction_wait2+_phase_correction2_edge
            indices_pad_squz = indices_pad_array(len(freq_ind_vec),Ions_squeezing_ind)
            [duration_squz] = corner(chan_glob_vec=chan_glob_vec,f_carrier_glob = freq_global_vec_SB2,chan_ind_vec=chan_vec_pad(chan_ind_vec,Ions_squeezing_ind),
                                            f_carrier_ind_vec = freq_ind_vec[Ions_squeezing_ind],chan_ind_pad_vec = chan_vec_pad(chan_ind_vec,indices_pad_squz),
                                            freq_ind_pad = freq_ind_vec[indices_pad_squz],Amp_norm_ind =Amp_norm_ind_squz,Amp_norm_global_cm =Amp_norm_global_cm_squz,
                                            amp_imbal_ratio_b2r = amp_imbal_ratio_b2r_squz,phi_spin = spin_phase_squz[nn]+_phase_correction_spin2,
                                            phi_motion_offset = phase_motion_squz,LS = LS_sqz)### squeezing of multiple ions
            _phase_correction2 = _phase_correction2+ 2*np.pi*qp.samples_to_seconds(duration_squz)*(freq_global_vec_SB0[0]-freq_global_vec_SB2[0])
            _phase_correction1_corner = _phase_correction1_corner + 2*np.pi*qp.samples_to_seconds(duration_squz)*(freq_global_vec_SB2[0]-freq_global_vec_SB1[0])
            print('duration of squeezing in mu' +str(duration_squz))
            return _phase_correction2,_phase_correction1_corner

        for n_rep in range(N_gate_rep):

            if(is_initial_pi_pulse):
                T_pi_pulse = sequential_sk1(applied_gates=spin_flip, pi_times=T_pi_vec_microsec,
                                thetas=np.array([np.pi,np.pi,np.pi]), phis=np.array([0,0, 0])+phiSK1_vec_bare+phiSK1)
                phase_correction_spin1 = phase_correction_spin1 +(T_pi_pulse/SK1_nominal_duration)*phi_disp_per_ramsey ### we put this form because it is a correction on the motional frequency
                phase_correction_spin2 = phase_correction_spin2 +(T_pi_pulse/SK1_nominal_duration)*phi_squeez_per_ramsey ### we put this form because it is a correction on the motional
                phase_correction_wait1 = phase_correction_wait1 +2*np.pi*qp.samples_to_seconds(T_pi_pulse)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0]) ### we put this form because it is a correction on the motional frequency
                phase_correction_wait2 = phase_correction_wait2 +2*np.pi*qp.samples_to_seconds(T_pi_pulse)*(freq_global_vec_SB0[0]-freq_global_vec_SB2[0]) ### we put this form because it is a correction on the motional frequency


        ## Alternating Displacement & Squeezing operations
            for nn in range(N_stg):

                #### displacement sequence
                if(seq_type1=='Displacement'):
                    phase_correction1,phase_correction2_edge = displacement_stage(nn,motion_phase_offset_disp,phase_correction1,phase_correction2_edge,phase_correction_spin1,phase_correction_wait1,phase_correction1_corner)

                #### Squeezing sequence
                elif(seq_type1=='Squeezing'):
                    phase_correction2,phase_correction1_corner = squeezing_stage(nn,motion_phase_offset_squz,phase_correction2,phase_correction1_corner,phase_correction_spin2,phase_correction_wait2,phase_correction2_edge)

                #### Gate sequence
                elif(seq_type1=='Both'):
                    # if(np.mod(n_rep,2)==0):
                    phase_correction1,phase_correction2_edge = displacement_stage(nn,motion_phase_offset_disp,phase_correction1,phase_correction2_edge,phase_correction_spin1,phase_correction_wait1,phase_correction1_corner)
                    phase_correction2,phase_correction1_corner = squeezing_stage(nn,motion_phase_offset_squz,phase_correction2,phase_correction1_corner,phase_correction_spin2,phase_correction_wait2,phase_correction2_edge)
                    # else:
                    #     phase_correction1,phase_correction2_edge = displacement_stage(nn,motion_phase_offset_disp,phase_correction1,phase_correction2_edge,phase_correction_spin1,phase_correction_wait1,phase_correction1_corner)
                    #     phase_correction2,phase_correction1_corner = squeezing_stage(nn,motion_phase_offset_squz,phase_correction2,phase_correction1_corner,phase_correction_spin2,phase_correction_wait2,phase_correction2_edge)

                #### Combined Closure sequence
                elif (seq_type1=='Sqz_Clos'):
                    if(nn<2):
                        phase_correction1,phase_correction2_edge = displacement_stage(nn,motion_phase_offset_disp,phase_correction1,phase_correction2_edge,phase_correction_spin1,phase_correction_wait1,phase_correction1_corner)
                        phase_correction2,phase_correction1_corner = squeezing_stage(nn,motion_phase_offset_squz,phase_correction2,phase_correction1_corner,phase_correction_spin2,phase_correction_wait2,phase_correction2_edge)
                    else:
                        phase_correction2,phase_correction1_corner = squeezing_stage(nn,motion_phase_offset_squz,phase_correction2,phase_correction1_corner,phase_correction_spin2,phase_correction_wait2,phase_correction2_edge)
                        phase_correction1,phase_correction2_edge = displacement_stage(nn,motion_phase_offset_disp,phase_correction1,phase_correction2_edge,phase_correction_spin1,phase_correction_wait1,phase_correction1_corner)


                if(tau_wait>0):
                    delay_all(T_delay=tau_wait)
                    phase_correction_wait1 = phase_correction_wait1 +2*np.pi*qp.samples_to_seconds(tau_wait)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0]) ### we put this form because it is a correction on the motional frequency
                    phase_correction_wait2 = phase_correction_wait2 +2*np.pi*qp.samples_to_seconds(tau_wait)*(freq_global_vec_SB0[0]-freq_global_vec_SB2[0]) ### we put this form because it is a correction on the motional frequency
                    print('duration of waiting in mu' +str(tau_wait))

                if(is_Echo):
                    if(np.mod(nn,2)==1):
                        if(seq_type1=='Displacement'):
                            phase_correction_echo = phase_correction1
                        elif(seq_type1=='Squeezing'):
                            phase_correction_echo = phase_correction2
                        elif((seq_type1=='Both') or (seq_type1=='Sqz_Clos')):
                            phase_correction_echo = phase_correction1+phase_correction2
                        phase_echo = np.mod(int((nn-1)/2),2)
                        T_corr1 = 0
                        T_echo = sequential_sk1(applied_gates=spin_echo, pi_times=T_pi_vec_microsec,
                                            thetas=np.array([np.pi,np.pi,np.pi]), phis=np.array([np.pi/2+0.03, -0.04+np.pi/2,np.pi/2+0.03])+(np.pi*phase_echo)+phase_correction_echo+phiSK1_vec_bare)
                        phase_correction_wait1 = phase_correction_wait1 +2*np.pi*qp.samples_to_seconds(T_corr1+T_echo)*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0]) ### we put this form because it is a correction on the motional frequency
                        phase_correction_wait2 = phase_correction_wait2 +2*np.pi*qp.samples_to_seconds(T_corr1+T_echo)*(freq_global_vec_SB0[0]-freq_global_vec_SB2[0]) ### we put this form because it is a correction on the motional frequency
                        print('duration of echo in mu' +str(T_echo+T_corr1))


        if(seq_type1=='Displacement'):
            phase_correction_tot = phase_correction1
        elif(seq_type1=='Squeezing'):
            phase_correction_tot = phase_correction2
        elif(seq_type1=='Both'):
            phase_correction_tot = phase_correction1+phase_correction2
        if(is_Ramsey):
            T_ramsey2 = sequential_sk1(applied_gates=spin_ramsey, pi_times=T_pi_vec_microsec,
                                thetas=np.array([np.pi/2,np.pi/2,np.pi/2]), phis=np.array([0.05,0.05+-0.08, 0.05])+np.pi+phase_correction_tot+phiSK1_vec_bare+phiSK1)
            print('duration of Ramsey2 in mu' +str(T_ramsey2))
    return out_sched


class CalibrateNbodyAmpDisp2sm(BasicEnvironment, artiq_env.Experiment):

    data_folder = "Nbody"
    applet_name = "Nbody_calib"
    applet_group = "RFSOC"
    fit_type = fit.line
    units = 1
    ylabel = "Population Transfer"
    xlabel = "Ind disp amp"

    def build(self):
        """Initialize experiment & variables."""
        # Add RFSoC arguments
        super().build()

        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(True))
        self.setattr_argument("use_AWG", artiq_env.BooleanValue(False))
        self.setattr_argument("Nrep", artiq_env.NumberValue(default=1,scale=1,step=1,ndecimals=0))
        # self.population_analysis = pop_an.PopulationAnalysis(self)

    def prepare(self):

        # Magic BasicEnvironment attributes
        self.spin_state_up = [False, False, True,True]*self.Nrep
        self.num_steps = 4*self.Nrep
        self.scan_values = list(map(int, self.spin_state_up))

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        super().prepare()

        center_ion_zero_index = 8
        center_slot = 17
        zero_pmt_slot = center_slot - center_ion_zero_index
        # if self.use_population_analysis:
        # self.population_analysis.module_init(
        #     data_folder=self.data_folder,
        #     active_pmts=self.pmt_array.active_pmts,
        #     num_steps=self.num_steps,
        #     detect_thresh=self.detect_thresh,
        #     XX_slots=(zero_pmt_slot + 7, zero_pmt_slot + 8, zero_pmt_slot + 9),
        # )
        # self.population_analysis.target_states

    @host_only
    def custom_pulse_schedule_list(self):
        self.AmpDisp = self.get_dataset("data.Nbody.amp_disp",default=default_amp_disp)
        self.AmpSqueez = self.get_dataset("data.Nbody.amp_squeez",default=default_amp_sqz)
        w_k = self.get_dataset("data.Nbody.zigzag_frequency")

        schedules = []
        for spin_state in self.spin_state_up:
            schedules.append(Nbody_schedule_calibrate(self.rfsoc_sbc.qiskit_backend,w_k,self.AmpDisp,self.AmpSqueez,Flag_calibration="AmpDisp",spin_state_up_X = spin_state))
        return schedules

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_experiment_initialize()

            self.custom_init()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")
            self.experiment_idle()
            self.custom_idle()

    @host_only
    def custom_experiment_initialize(self):
        pass

    @kernel
    def custom_init(self):
        self.raman.collective_dds.enabled = False

    @kernel
    def custom_idle(self):
        pass

    @kernel
    def prepare_step(self, istep):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        self.raman.set_global_aom_source(dds=True)
        delay(5 * us)

    def analyze(self):

        # breakpoint()
        super().analyze()
        for fit_label, values in self.p_all.items():
            print(f"Fit[{fit_label}]: {values}")

        a_line = (self.p_all['a'][0]+self.p_all['a'][2])/2
        b_line = (self.p_all['b'][0]+self.p_all['b'][2])/2
        squeez_err = (2*a_line+b_line-1)
        print("a line is ",a_line)
        print("b line is ",b_line)
        print("sqz err",squeez_err)
        disp_err = (b_line)
        prop_disp = -1/(4*np.pi)
        prop_squeez = -0.64
        self.set_dataset("data.Nbody.amp_disp",self.AmpDisp+prop_disp*disp_err,broadcast=True,persist=True)
        print("updated AmpDisp is ",self.AmpDisp+prop_disp*disp_err)
        if(abs(squeez_err)<0.5):
            self.set_dataset("data.Nbody.amp_squeez",self.AmpSqueez*(1+prop_squeez*squeez_err),broadcast=True,persist=True)
            print("updated AmpSquz is ",self.AmpSqueez*(1+prop_squeez*squeez_err))
        # if(squeez_err<0.5):

        # if(disp_err<0.5):

        # w_k2 = self.p_all['position'][2]
        # frequency_rel_thresh = 150
        # frequency_abs_thresh = 550
        # if((np.abs(w_k1-w_k2)<frequency_rel_thresh) and (np.abs(0.5*(w_k1+w_k2)-self.base_frequency)<frequency_abs_thresh)):
        #     self.set_dataset("data.Nbody.zigzag_frequency",0.5*(w_k1+w_k2))
        # else:
        #     _LOGGER.error("calibrated frequency is out of range")


class CalibrateNbodyFrequency2sm(BasicEnvironment, artiq_env.Experiment):

    data_folder = "Nbody"
    applet_name = "Nbody_calib"
    applet_group = "RFSOC"
    fit_type = fit.shifted_parabola
    units = 1
    ylabel = "Population Transfer"
    xlabel = "Motional detuning displacement"

    def build(self):
        """Initialize experiment & variables."""
        # Add RFSoC arguments
        super().build()

        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(True))
        self.setattr_argument("use_AWG", artiq_env.BooleanValue(False))
        self.setattr_argument("frequency_range", rfsoc.FrequencyScan(default=artiq_scan.RangeScan(start=-600,stop=600,npoints=9)))
        # self.population_analysis = pop_an.PopulationAnalysis(self)

    def prepare(self):

        # Magic BasicEnvironment attributes
        self.base_frequency = self.get_dataset("data.Nbody.zigzag_frequency",default=default_w_k)
        self.num_steps = len(self.frequency_range)
        self.scan_values = [f + self.base_frequency for f in self.frequency_range]

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        super().prepare()

        # center_ion_zero_index = 8
        # center_slot = 17
        # zero_pmt_slot = center_slot - center_ion_zero_index
        # if self.use_population_analysis:
        # self.population_analysis.module_init(
        #     data_folder=self.data_folder,
        #     active_pmts=self.pmt_array.active_pmts,
        #     num_steps=self.num_steps,
        #     detect_thresh=self.detect_thresh,
        #     XX_slots=(zero_pmt_slot + 7, zero_pmt_slot + 8, zero_pmt_slot + 9),
        # )
        # self.population_analysis.target_states

    @host_only
    def custom_pulse_schedule_list(self):
        amp_ind_disp = self.get_dataset("data.Nbody.amp_disp",default=default_amp_disp)
        amp_ind_squeez = self.get_dataset("data.Nbody.amp_squeez",default=default_amp_sqz)

        schedules = []
        for w_k in self.scan_values:
            schedules.append(Nbody_schedule_calibrate(self.rfsoc_sbc.qiskit_backend,w_k,amp_ind_disp,amp_ind_squeez,Flag_calibration="Freq"))
        return schedules

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_experiment_initialize()

            self.custom_init()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")
            self.experiment_idle()
            self.custom_idle()

    @host_only
    def custom_experiment_initialize(self):
        pass

    @kernel
    def custom_init(self):
        self.raman.collective_dds.enabled = False

    @kernel
    def custom_idle(self):
        pass

    @kernel
    def prepare_step(self, istep):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        self.raman.set_global_aom_source(dds=True)
        delay(5 * us)

    def analyze(self):
        super().analyze()
        for fit_label, values in self.p_all.items():
            print(f"Fit[{fit_label}]: {values}")

        w_k1 = self.p_all['position'][0]
        w_k2 = self.p_all['position'][2]
        frequency_rel_thresh = 200
        frequency_abs_thresh = 750
        if((np.abs(w_k1-w_k2)<frequency_rel_thresh) and (np.abs(0.5*(w_k1+w_k2)-self.base_frequency)<frequency_abs_thresh)):
            self.set_dataset("data.Nbody.zigzag_frequency",0.5*(w_k1+w_k2),broadcast=True,persist=True)
            print("zigzag frequency was updated to " , (0.5*(w_k1+w_k2)))
        else:
            _LOGGER.error("calibrated frequency is out of range")

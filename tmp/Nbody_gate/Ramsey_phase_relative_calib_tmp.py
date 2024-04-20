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
    Amp_norm_ind,Amp_norm_global_cm,amp_imbal_ratio_b2r,lightshift_freq,phi_spin = 0,
    phi_motion_offset = 0,zero_pad_flag = True,n_edge = 0):
    ### This function generates a single displacement pulse for one ion by modulating the amplitude of the individual beam and modulating the relative phase between the red and blue tones of the global beam
    ### Assumes that chan_ind is a single channel

    ### load splines
    this_dir = pathlib.Path(__file__).parent.resolve()
    json_file = this_dir / "splines_displace_xpxp_mode3_pulsedur026_waitdur024.json"
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

    ### Sequence
    for n_seg in range(0,N_segments):
        ### GLOBAL BEAM - PM
        phase_spline_red = (phase_red_ofst-0.5*phi_motion_c0[n_seg],-0.5*phi_motion_c1[n_seg],-0.5*phi_motion_c2[n_seg],-0.5*phi_motion_c3[n_seg])
        phase_spline_blue = (phase_blue_ofst+0.5*phi_motion_c0[n_seg],+0.5*phi_motion_c1[n_seg],+0.5*phi_motion_c2[n_seg],+0.5*phi_motion_c3[n_seg])

        qp.play(ToneDataPulse(duration_dt_vec[n_seg], frequency_hz = freq_red, amplitude = amp_red,
            phase_rad=phase_spline_red,output_enable=False,sync=False), chan_glob_tone_red)

        qp.play(ToneDataPulse(duration_dt_vec[n_seg], frequency_hz = freq_blue, amplitude = amp_blue,
            phase_rad=phase_spline_blue,output_enable=False,sync=False), chan_glob_tone_blue)

        ## Individual beam - AM
        amp_spline_ind = spl.CubicSpline(Amp_norm_ind*amp_ind_c0[n_seg],Amp_norm_ind*amp_ind_c1[n_seg],Amp_norm_ind*amp_ind_c2[n_seg],Amp_norm_ind*amp_ind_c3[n_seg])
        qp.play(ToneDataPulse(duration_dt_vec[n_seg], frequency_hz = freq_ind, amplitude = amp_spline_ind,
            phase_rad=phase_ind,output_enable=False,sync=False), chan_ind1_tone_0)

        ## Padding all other channels
        if(zero_pad_flag):
            zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,duration_dt_vec[n_seg])

    return [duration_tot]

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

    # qp.call(single_qubit.square_rabi(ion_index=ion_index,duration=(duration_microsec*1e-6),phase=phi_spin,detuning=detuning_Hz,
    #     sideband_order=sideband_order,individual_amp=Amp_norm_ind,global_amp=Amp_norm_global))
    # breakpoint()
    qp.play(ToneDataPulse(qp.seconds_to_samples(duration_microsec*1e-6), frequency_hz = freq_glob-detuning_Hz, amplitude = Amp_norm_global,
        phase_rad=phi_spin,output_enable=False,sync=False, _name="rabi_global"), chan_global)

    qp.play(ToneDataPulse(qp.seconds_to_samples(duration_microsec*1e-6), frequency_hz = freq_ind, amplitude = Amp_norm_ind,
        phase_rad=0,output_enable=False,sync=False, _name="rabi_individual"), chan_ind)

    if(zero_pad_flag):
        zero_padding_ind(chan_ind_pad_vec,freq_ind_pad,qp.seconds_to_samples(duration_microsec*1e-6))



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
Amp_norm_ind_disp = 0.7
Amp_norm_ind_squz = 0.5
Amp_norm_global_cm_disp = 0.45
Amp_norm_global_cm_squz = 0.45
amp_imbal_ratio_b2r_disp = 1
amp_imbal_ratio_b2r_squz = 1

if((Amp_norm_global_cm_disp>=0.5) or (Amp_norm_global_cm_squz>=0.5)):
    print("error!! amplitude of two tones signal is saturated above 0.5")
lightshift_freq_disp = 0
lightshift_freq_squz = 0

T_pi_vec_microsec = np.array([1.65,1.41,1.62])

N_stg = 1
N_stg_vec =[0,0]
ion_idx_disp_vec  = [Ions_displacement_ind[0],Ions_displacement_ind[0]] #[Ions_displacement_ind[0],Ions_displacement_ind[1],Ions_displacement_ind[0],Ions_displacement_ind[1]] ### the ion that is displaced as a function of the stage number
phase_correction = 0
# spin_phase_squz = [0,0,0,0]
# motion_phase_offset_squz = [0,np.pi,0,np.pi] ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction

### Main Schedule
schedules = []
rel_phase_vec =np.array([0])# np.linspace(0,np.pi,9) #np.array([0])#
w_k_vec = np.linspace(2.795e6,2.845e6,21) #np.array([2.8215e6])
for phi_scan in rel_phase_vec:
    for w_k in w_k_vec:
        with qp.build(backend=qp_backend) as out_sched:

            ## frequency & phases assignment
            spin_phase_disp = [phi_scan,phi_scan]
            motion_phase_offset_disp = [0,np.pi] ### each stage the phase is increased by pi/2 through the optimizer solution. This should be a small offset / correction
            freq_carrier_glob = qp_backend.properties().rf_calibration.frequencies.global_carrier_frequency.value
            freq_global_vec_SB0 = np.array([freq_carrier_glob,freq_carrier_glob+w_k]) # red and blue tones' frequencies in Hz for first sideband
            freq_global_vec_SB1 = np.array([freq_carrier_glob-w_k,freq_carrier_glob+w_k]) # red and blue tones' frequencies in Hz for first sideband
            freq_global_vec_SB2 = np.array([freq_carrier_glob-2*w_k,freq_carrier_glob+2*w_k]) # red and blue tones' frequencies in Hz for second sideband
            freq_ind_vec = np.array([qp_backend.properties().rf_calibration.frequencies.individual_carrier_frequency.value]*3)

            ## assign channels
            chan_glob_tone_red = qp.control_channels()[0]
            chan_glob_tone_blue = qp.control_channels()[1]
            chan_glob_vec = [chan_glob_tone_red, chan_glob_tone_blue]
            chan_ind_vec = []
            for ion_idx in range(-int((-1+freq_ind_vec.shape[0])/2),1+int((-1+freq_ind_vec.shape[0])/2)):
                chan_ind_vec = chan_ind_vec + [qp_backend.configuration().individual_channel(ion_idx, 0)]

            ## Initial sync of phases
            qp.play(ToneDataPulse(6, frequency_hz = freq_global_vec_SB0[0], amplitude = 0,
                phase_rad=0,output_enable=False,sync=True,_name="sync_global"), chan_glob_tone_red)

            qp.play(ToneDataPulse(6, frequency_hz = freq_global_vec_SB0[1], amplitude = 0,
                phase_rad=0,output_enable=False,sync=True,_name="sync_global"), chan_glob_tone_blue)

            qp.play(ToneDataPulse(6, frequency_hz = freq_ind_vec[1], amplitude = 0,
                phase_rad=0,output_enable=False,sync=True,_name="sync_ind"), chan_ind_vec[1])

            ### 1st RAbi
            dT_Rabi = qp.seconds_to_samples(0.5*0.933*1e-6)
            qp.play(ToneDataPulse(dT_Rabi, frequency_hz = freq_global_vec_SB0[0], amplitude = 0.5,
                phase_rad=0,output_enable=False,sync=False, _name="rabi_global"), chan_glob_tone_red)

            qp.play(ToneDataPulse(dT_Rabi, frequency_hz = freq_global_vec_SB0[1], amplitude = 0.0,
                phase_rad=0,output_enable=False,sync=False, _name="rabi_global_zero"), chan_glob_tone_blue)

            qp.play(ToneDataPulse(dT_Rabi, frequency_hz = freq_ind_vec[1], amplitude = 0.5,
                phase_rad=0,output_enable=False,sync=False, _name="rabi_individual"), chan_ind_vec[1])


            ### delay
            dT_delay = qp.seconds_to_samples(20*1e-6)
            qp.play(ToneDataPulse(dT_delay, frequency_hz = freq_global_vec_SB1[0], amplitude = 0.0,
                phase_rad=0,output_enable=False,sync=False, _name="rabi_global"), chan_glob_tone_red)

            qp.play(ToneDataPulse(dT_delay, frequency_hz = freq_global_vec_SB0[1], amplitude = 0.0,
                phase_rad=0,output_enable=False,sync=False, _name="rabi_global_zero"), chan_glob_tone_blue)

            qp.play(ToneDataPulse(dT_delay, frequency_hz = freq_ind_vec[1], amplitude = 0.0,
                phase_rad=0,output_enable=False,sync=False, _name="rabi_individual"), chan_ind_vec[1])


            ### 2nd RAbi
            phase_correction = 2*np.pi*(freq_global_vec_SB0[0]-freq_global_vec_SB1[0])*qp.samples_to_seconds(dT_delay)
            qp.play(ToneDataPulse(dT_Rabi, frequency_hz = freq_global_vec_SB0[0], amplitude = 0.5,
                phase_rad=phase_correction+np.pi,output_enable=False,sync=False, _name="rabi_global"), chan_glob_tone_red)

            qp.play(ToneDataPulse(dT_Rabi, frequency_hz = freq_global_vec_SB0[1], amplitude = 0.0,
                phase_rad=0,output_enable=False,sync=False, _name="rabi_global_zero"), chan_glob_tone_blue)

            qp.play(ToneDataPulse(dT_Rabi, frequency_hz = freq_ind_vec[1], amplitude = 0.5,
                phase_rad=0,output_enable=False,sync=False, _name="rabi_individual"), chan_ind_vec[1])

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

else:
    matplotlib.use("agg")
    out_sched.draw()
    plt.savefig('tmp/Nbody_gate/schedule_displace.png')
    print(default_freqs)
    compiled_schedule = OpenPulseToOctetConverter.schedule_to_octet(out_sched, default_lo_freq_hz=default_freqs)
    # compiled_schedule_global_only = {k: v for k, v in compiled_schedule.items() if isinstance(k, qp.ControlChannel)}
    # print(record.text_channel_sequence(compiled_schedule_global_only))
    record.save_text_channel_sequence(compiled_schedule, "./tmp/Nbody_gate/working_phase_tonedata_commands.tones")

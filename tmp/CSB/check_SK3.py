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

from euriqabackend import _EURIQA_LIB_DIR
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
    scale_time =4
    duration_vec = [int(scale_time*1*251),int(scale_time*1*1237),int(scale_time*1*251)]
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
num_ions=23
scan_type = 'Angle'   #### 'Angle'  |  'Ngates'  |' SK1_amp'
is_submit_via_python = False
is_debug_sequence = False ### only for debugging
is_SK1 = [True] *num_ions
# is_SK1[0] = [True]
# is_SK1[11] = [True]
# is_SK1[22] = [True]

is_get_dataset_vals =True


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
                tanh_scale_factor = 1*amp_scale_factor*np.array([0.9854540668066544, 1.0161854854964625, 1.013523380742869, 1.007163657596779, 0.9929102305176052, 0.9959195798349382, 0.9971507829092647, 0.9908805061520014, 0.9975045535618642, 1.0027782098978542, 0.9942014803467186, 0.9856723536333182, 0.990783620600121, 0.9968945119570908, 0.9954115820003856, 1.0227979776191682, 1.0201266883538544, 1.0183452874422447, 1.014520651892784, 0.9870736907114992, 0.9771226792396365, 0.9825383099838269, 1.0654811611269686])*np.array([0.9916346232878552, 0.9460941975170369, 0.9566872082469615, 0.9555097840556399, 0.9445424509411736, 0.9510158159579819, 0.9592498852684893, 1.0036276976212744, 0.9686738143877522, 1.0107639643599653, 0.9873569721702156, 0.9606337249094362, 0.9604469888939018, 0.9611021921629044, 0.9150858924707299, 0.9213586083218156, 0.9481894644891496, 0.9428299501838813, 0.987470320168479, 0.9757381070718808, 0.9857229270503546, 0.998455000443369, 0.9757351748223905])*np.array([1.0538664034319545, 1.0718021543095189, 1.0577819265170791, 1.0557182477012215, 1.0760577531080364, 1.0775119826588795, 1.0726292207761365, 1.049724035001543, 1.038890181245028, 1.01887925960003, 1.015040644005963, 1.009741029857695, 1.04240091121784, 1.0431176237543798, 1.0405936389305275, 1.0762822467033997, 1.0287061227925767, 1.0093963707607536, 0.9619294512978723, 0.9435176150048181, 0.8982312743115383, 0.9324613664546328, 1.0090335483258712])*np.array([0.9292679534420171, 1.0096132905591992, 1.0385994588390117, 1.0199993052329557, 0.9896657883462842, 0.9378479969656163, 0.9298647621365772, 0.924108489252452, 0.9349000713979482, 0.9209305697152602, 0.9669390870525592, 1.0201008562991312, 1.021936964567396, 1.0285625814769623, 1.0704025288309307, 1.0335753905259388, 1.0478354972282657, 1.0222831233421128, 1.0321893882655244, 1.0237047942495925, 1.040193984575578, 1.0329888316635785, 1.0225644790082717])*np.array([1.0682844699302123, 0.9038937186597191, 0.9391188615075962, 0.9478466874461293, 0.9721660246689774, 0.9683194373095857, 0.9363597124815254, 0.9043307530664857, 0.8720513172522002, 0.9392908018916095, 0.9418096004918323, 0.9657116475331062, 0.957812462783673, 0.9514653427306475, 0.9341509515907037, 0.9447869548547483, 0.94564124973909, 0.969624329275144, 0.9801924135084769, 0.9482458678402715, 0.9187547117573144, 0.9119592057164415, 1.0831651375170623])*np.array([1.0606298899106223, 0.9581731702941048, 0.9168777832938718, 0.9404584627333517, 0.9678982964220223, 1.0347223882073575, 1.0993241046215094, 1.1485770018364287, 1.1554216761930003, 1.114437919432126, 1.0666388488075604, 0.9901535749470624, 0.9394389737393825, 0.8782321541620414, 0.82271690382986, 0.8474636455954148, 0.8715154466226649, 0.9392876222504984, 1.0144646806440185, 1.1145112254279235, 1.1719055657174717, 1.1125960954195444, 0.8741452189210155])*np.array([0.9230609992333972, 1.0242499526051638, 1.061255986418668, 1.0329938947072923, 1.0001793720169518, 0.9494295239856235, 0.91160021100471, 0.897464654656663, 0.8957166311996176, 0.9238844830078102, 0.9536176317167988, 0.9966670697956068, 1.0352296224860082, 1.0773856033548106, 1.1424871682198252, 1.1090442200382389, 1.092195847617751, 1.026583581805756, 0.9771445491439448, 0.9129482713603025, 0.8813554727092742, 0.9140780528425153, 1.089111050793778])*np.array([1.08317943687904, 0.9947089372472793, 0.9708953128514867, 1.0058678457892847, 1.0490066416783765, 1.0732355990714697, 1.0903262554453805, 1.0742253915503688, 1.0759573808673348, 1.0751562806079868, 1.0581002782938485, 1.0630546228299027, 1.0535470945765821, 1.0554771606742426, 1.0378434898458766, 1.0433567968505055, 1.0509699387251856, 1.082815296569976, 1.0717221795522258, 1.080781371826523, 1.0740639343314191, 1.0852340361619623, 1.1228240129111382])*np.array([1.0066859205585754, 1.1084701971963171, 1.148896991638129, 1.0896387462312958, 1.0304230311348814, 0.995555264751019, 0.9682512245979944, 0.9553508036878557, 0.9629854462216898, 0.9700494675505318, 0.9779711449237706, 0.9981197490894621, 1.0057294633044183, 1.034721533983873, 1.037800601107395, 1.050600978415855, 1.0266866320348083, 1.0030800584800468, 0.9965232968870245, 0.9939125904272633, 0.9915792681377336, 1.0059752994024922, 1.0112821452341845])*np.array([0.8340890808591064, 0.9776883963114388, 1.0052028236266908, 0.9915514909281493, 0.9869001986198694, 0.9790026410439283, 1.0149186668951522, 1.0236515396917512, 1.029192410762835, 1.0053285413001698, 0.9948701842779053, 0.983163330592007, 0.9820851862368223, 1.0160765405491765, 1.0258660905192067, 1.0234081924517737, 1.013884636987897, 0.9735977103949804, 0.9618937748974996, 0.9726696812920402, 0.9954028466817593, 0.9686160158563609, 0.8524278631081691])*np.array([1.530091316236628, 1.0988311617092028, 0.8854135904340471, 0.9294296580467925, 0.9607585589770087, 1.1250708289195157, 1.194550016487324, 1.2613139662427875, 1.3259249756743723, 1.2591498652524125, 1.1886706148723807, 1.1038853030487772, 1.0537036357617753, 0.9629889325879264, 0.9345650231415438, 0.948053656782288, 1.0123198286560047, 1.1446236394609837, 1.2420223923582332, 1.3904160082689296, 1.4679777312956939, 1.2916569950172514, 0.8340434138439796])*np.array([1.40372, 1.047  , 1.05722, 1.06391, 1.05211, 1.09141, 1.09983,1.08811, 1.07479, 1.10901, 1.06394, 1.05624, 1.04849, 1.05671, 1.02725,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05])*np.array([0.5678988986101344, 0.9845408345880869, 1.0897854520628518, 1.0421092183149139, 1.0548354756482519, 0.862329750451962, 0.8081825239393917, 0.7989695928542555, 0.7958668312888997, 0.793534059957885, 0.8844910059134918, 0.9152543461724242, 0.980424347148316, 1.0292957728312067, 1.0262109081861062, 1.016124249151123, 0.9722431847071622, 0.8615244189390483, 0.8482978684032405, 0.7391443683457082, 0.7196367739570299, 0.8134684192701348, 1.1544211576109344])*np.array([0.9271020106331694, 0.9837614127899345, 1.0105556591541902, 1.0156475997145267, 0.9996532571781608, 1.0091399120562534, 1.003753622529971, 0.9964666583057011, 0.982802966118653, 1.0046944829948337, 1.0044876474234075, 1.00483552051891, 1.0068424336398765, 1.0066871750776165, 1.0408118107421958, 1.0016783629906265, 1.0012319908816305, 1.004987514051287, 0.9788992626351499, 0.9818934317176278, 0.9641731326672401, 0.9828631221445137, 1.0175681683329403])
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

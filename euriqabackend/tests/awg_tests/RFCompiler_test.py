import numpy as np

import euriqabackend.devices.keysight_awg.RFCompiler as rfc
import euriqabackend.tests.awg_tests.cirq_tester as ct

if __name__ == "__main__":
    ##############################################
    #        DEFINE PHYSICAL PARAMETERS          #
    ##############################################

    f_ind = 57
    f_carrier = 161.748398
    t_delay = 3.0
    Rabi_max = 0.5
    PI_center_freq_1Q = f_ind
    PI_center_freq_2Q = f_ind
    monitor_ind = True
    monitor_detuning = 1
    amp_monitor = 1

    amp_ind_Rabi = 1000
    amp_global_Rabi = 1000
    Stark_shift_Rabi = 0

    amp_ind_SK1 = 1000
    amp_global_SK1 = 1000
    Tpi_multiplier_SK1 = 1
    Stark_shift_SK1 = 0

    amp_ind_SK1_AM = 1000
    amp_global_SK1_AM = 1000
    theta_SK1_AM = np.pi / 2
    envelope_type_SK1_AM = rfc.RFCompiler.InterpFunctionType.full_Gaussian
    envelope_scale_SK1_AM = 3
    rotation_pulse_length_SK1_AM = 2
    correction_pulse_1_length_SK1_AM = 4
    correction_pulse_2_length_SK1_AM = 8
    Stark_shift_SK1_AM = 0

    N_ions = 15
    XX_sols_dir = r"C:\Temp\XX Sols"
    # XX_sol_name = "XX_GateSolutions"
    # XX_sols_dir = r"Q:\CompactTrappedIonModule\Data\2019\05_2019\17_05_2019\"
    #   "gate_attempt"
    # XX_sols_dir = r"Q:\CompactTrappedIonModule\Data\2019\06_2019\21_06_2019"
    XX_sol_name = "interpolated_H_correctmodes"
    duration_adjust_XX = 0
    amp_ind_multiplier_XX = 1
    amp_global_XX = 1000
    sb_amplitude_imbalance_XX = 0
    Stark_shift_XX = 0.0007
    Stark_shift_diff_XX = 0.0
    motional_freq_adjust_XX = 0

    calibrated_Tpi = [0.95] * 32
    AOM_saturation_params = [100.0] * 32

    ##############################################
    #      CREATE COMPILER AND SET PARAMS        #
    ##############################################

    RFCompiler = rfc.RFCompiler()

    RFCompiler.physical_params.set_params(
        f_carrier=f_carrier,
        f_ind=f_ind,
        N_ions=N_ions,
        t_delay=t_delay,
        Rabi_max=Rabi_max,
        PI_center_freq_1Q=PI_center_freq_1Q,
        PI_center_freq_2Q=PI_center_freq_2Q,
    )

    RFCompiler.physical_params.monitor.set_params(
        monitor_ind=monitor_ind, detuning=monitor_detuning, amp=amp_monitor
    )

    RFCompiler.physical_params.Rabi.set_params(
        amp_ind=amp_ind_Rabi, amp_global=amp_global_Rabi, Stark_shift=Stark_shift_Rabi
    )

    RFCompiler.physical_params.SK1.set_params(
        amp_ind=amp_ind_SK1,
        amp_global=amp_global_SK1,
        Tpi_multiplier=Tpi_multiplier_SK1,
        Stark_shift=Stark_shift_SK1,
    )

    RFCompiler.physical_params.SK1_AM.set_params(
        amp_ind=amp_ind_SK1_AM,
        amp_global=amp_global_SK1_AM,
        theta=theta_SK1_AM,
        envelope_type=envelope_type_SK1_AM,
        envelope_scale=envelope_scale_SK1_AM,
        rotation_pulse_length=rotation_pulse_length_SK1_AM,
        correction_pulse_1_length=correction_pulse_1_length_SK1_AM,
        correction_pulse_2_length=correction_pulse_2_length_SK1_AM,
        Stark_shift=Stark_shift_SK1_AM,
    )

    # TODO: CHANGE
    RFCompiler.set_XX_solution_from_folder(sols_dir="/".join(XX_sols_dir,XX_sol_name))
    RFCompiler.set_XX_tweak(
        "all",
        XX_duration_us=duration_adjust_XX,
        individual_amplitude_multiplier=amp_ind_multiplier_XX,
        global_amplitude=amp_global_XX,
        sideband_amplitude_imbalance=sb_amplitude_imbalance_XX,
        stark_shift=Stark_shift_XX,
        stark_shift_differential=Stark_shift_diff_XX,
        motional_frequency_adjust=motional_freq_adjust_XX,
    )

    A_set = RFCompiler.set_AOM_levels(calibrated_Tpi)

    RFCompiler.set_AOM_saturation_params(AOM_saturation_params)

    ##############################################
    #           GENERATE GATE ARRAYS             #
    ##############################################

    exp_to_run = RFCompiler.ExperimentsAvailable.XX

    if exp_to_run == RFCompiler.ExperimentsAvailable.Circuit:
        test_cirq_file = r"C:\RF System\Test.cirq"
        ct.write_cirq_file(test_cirq_file, print_circuit=True)
        # test_cirq_file = r"Q:\CompactTrappedIonModule\Data\Circuits\"
        #   r"randomized_benchmarking_15ions\single_qubit_RB\
        # qubit_00_RB\circuit_004_04"
        circuit_string = RFCompiler.circuit(
            test_cirq_file,
            exp_name="Test circuit",
            use_SK1_AM=False,
            print_circuit=True,
            print_gate_list=True,
        )
        # print("\n\nReturned string:")
        # print(circuit_string)

    if exp_to_run == RFCompiler.ExperimentsAvailable.Rabi:
        RFCompiler.rabi_exp(
            slots=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            detuning=0,
            sideband_order=0,
            scan_parameter_int=int(RFCompiler.RabiScanParameter.duration),
            min_value=0,
            max_value=10,
            N_points=21,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.Rabi_AM:
        RFCompiler.rabi_am_exp(
            slots=[16, 17, 18],
            detuning=0,
            sideband_order=0,
            detuning_off=5,
            envelope_type_int=int(RFCompiler.InterpFunctionType.full_Gaussian),
            envelope_duration=15,
            envelope_scale=3,
            global_delay=0,
            global_duration=-1,
            scan_parameter_int=int(RFCompiler.RabiAMScanParameter.envelope_duration),
            min_value=0,
            max_value=10,
            N_points=21,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.Rabi_PI:
        RFCompiler.rabi_pi_exp(
            slots=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            detuning=0,
            sideband_order=0,
            scan_parameter_int=int(RFCompiler.RabiScanParameter.duration),
            min_value=0,
            max_value=10,
            N_points=21,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.SK1:
        RFCompiler.SK1_exp(
            slots=[15, 19],
            theta=np.pi / 2,
            phi=0,
            scan_parameter_int=int(RFCompiler.SK1ScanParameter.phi),
            min_value=0.0,
            max_value=np.pi * 2,
            N_points=11,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.SK1_AM:
        RFCompiler.SK1_am_exp(
            slots=[15, 19],
            phi=0,
            scan_parameter_int=int(RFCompiler.SK1AMScanParameter.phi),
            min_value=0.0,
            max_value=np.pi * 2,
            N_points=11,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.XX:
        RFCompiler.XX_exp(
            slots=[15, 19],
            N_gates=3,
            scan_parameter_int=int(RFCompiler.XXScanParameter.N_gates),
            min_value=1,
            max_value=5,
            N_points=5,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.Linescan:
        RFCompiler.linescan(
            slots=[17, 18, 19],
            detuning=3.05,
            sideband_order=+1,
            duration=100,
            scan_range=0.05,
            N_points=21,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.XX_parity_scan:
        RFCompiler.XX_parity_scan(
            slots=[15, 19],
            N_points=11,
            gate_sign=+1,
            sideband_imbalance=0,
            sweep_relative_phase=0,
            common_phase=0,
            relative_phase=0.1,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.XX_with_analysis:

        ind_imbalance = 0.1

        RFCompiler.XX_with_analysis(
            slots=[15, 19],
            scan_parameter_int=int(RFCompiler.XXScanParameter.N_gates),
            min_value=1,
            max_value=7,
            N_points=7,
            individual_amplitude_imbalance=ind_imbalance,
        )

    if exp_to_run == RFCompiler.ExperimentsAvailable.Stabilizer_readout:
        circuit_string = RFCompiler.stabilizer_readout(
            prep_state=6,
            post_prepare_phases=0.1,
            post_XX_phase=0.2,
            exp_name="Stabilizer readout",
            use_SK1_AM=True,
            print_circuit=False,
            print_gate_list=True,
            scan_parameter_int=int(RFCompiler.StabReadoutScanParameter.post_XX_phase),
            min_value=2,
            max_value=7,
            N_points=7,
        )
        # print("\n\nReturned string:")
        # print(circuit_string)

    ##############################################
    #             GENERATE WAVEFORMS             #
    ##############################################

    RFCompiler.generate_waveforms()

    ##############################################
    #               PROGRAM AWG                  #
    ##############################################

    # RFCompiler.select_exp_to_program()
    # RFCompiler.program_AWG()
    # RFCompiler.set_DDS([16, 18, 20, 23])

import logging

import artiq.language.environment as artiq_env
from artiq.language.environment import HasEnvironment
import numpy as np
from artiq.language.types import TBool
from artiq.language.core import kernel, delay, delay_mu, host_only, rpc, parallel, delay_mu ,now_mu
from artiq.language.units import MHz, ms, us, V
from artiq.experiment import TerminationRequested
from artiq.experiment import BooleanValue, NumberValue, StringValue

from euriqafrontend.repository.basic_environment_test import BasicEnvironment
from euriqafrontend.modules.artiq_dac import RampControl as rampcontrol_auto

# Sorting prototype code
# 1. Received a sorting solution from sorting_sol_generator in basic environment in the form of [k1, k2, ..., km], where each index
# indicates the ion to swap, and the sorting should be executed by performing swap from left to right in sol array.
# For index k1, I should swap (k1, k1+1) pair
# 2. Currently, we test the following scheme: for each swap, start with high RF, ramp RF down, swap, ramp RF up, merge.
# 3. Currently, we test without a working isotope checker.

_LOGGER = logging.getLogger(__name__)

class Sorting_Environment(BasicEnvironment, artiq_env.Experiment):
    data_folder = "Sort"
    applet_name = "SortingEnv"
    applet_group = "Sort"

    def build(self):
        super().build()
        self.rf_ramp = rampcontrol_auto(self)
        # Use AWG trigger for camera
        self.setattr_device("awg_trigger")
        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

    def prepare(self):
        super().prepare()
        self.rf_ramp.prepare()

        # to get it to run with basic environtment
        self.num_steps=1
        self.scan_values = [1]

        # 1: not necessary if keep the chain
        # self.dac_starting_node="Start"
        # # For some reason if this isn't here we lose the chain
        # self.sandia_box.dac_pc.apply_line_async(self.dac_starting_node, line_gain=.8)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            print("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            print("Termination Request Received: Ending Experiment")
        finally:
            print("Done with Experiment. Setting machine to idle state")

    @kernel
    def trigger_camera(self):
        self.core.break_realtime()
        self.awg_trigger.pulse(1 * ms)

    # experiment loop override to turn off what we don't need
    @kernel
    def experiment_loop(self):
        """Run the experiment on the core device."""

        # Create input buffers
        calib_counts = [0] * self.num_pmts
        doppler_counts = [0] * self.num_pmts
        ss_counts = [0] * self.num_pmts
        ss_coolant_counts = [0] * self.num_pmts
        ss_all_counts = [0] * self.num_pmts
        ss_edge_counts = [0] * self.num_pmts
        ss_edge_coolant_counts = [0] * self.num_pmts
        ss_edge_all_counts = [0] * self.num_pmts
        detect_counts = [0] * self.num_pmts
        config = [0] * self.num_pmts
        # here we assume that the number of pmts is equal to the chain length, the size of sorting solution buffer
        # should be no smaller than chain length
        sorting_sol = [-1] * self.num_pmts

        self.core.break_realtime()
        # Set switch network to sequence mode
        # if not self.use_RFSOC:
        #     self.AWG.setRunningSN(True)
        delay(10 * us)

        istep = 0  # Counter for steps
        iequil = 0  # Counter for equilibration rounds

        # Loop over main experimental scan parameters
        while istep < len(self.scan_values):
            # check for pause/abort from the host
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()

            # do any initialization that needs to be done for each step
            # such as variable hardware-shuttling, variable piezo setting, etc.
            self.prepare_step(istep)

            # initialization of local var
            reorder_flag = 0
            prev_config = config

            # Loop over to gather statistics
            for ishot in range(self.num_shots):

                self.core.break_realtime()
                if self.use_line_trigger:
                    trigger = self.wait_for_line_trigger()

                # Reset Switch Network
                # if not self.use_RFSOC:
                #     self.AWG.resetSN()
                #     delay(10 * us)
                #     # Advance Switch Network to its first line (calib)
                #     self.AWG.advanceSN()

                self.eom_935_3ghz.off()

                # if self.do_calib:
                #     stopcounter_calib_mu = self.measure_pointing()
                #     # Readout out counts and save the data
                #     calib_counts = self.pmt_array.count(
                #         up_to_time_mu=stopcounter_calib_mu, buffer=calib_counts
                #     )
                #     # Can break realtime because we're still not in experiment core.
                #     # might cause some nondeterministic timing relative to line trigger
                #     # Change originally made by @daiwei
                #     # Provides slack after reading PMTs
                #     # self.core.break_realtime()
                #     # try simple delay for now, to see if it works
                #     delay(350 * us)

                # Turn on Global AOM during cooling downtime
                if self.keep_global_on:
                    self.raman.global_dds.update_amp(np.int32(self._KEEP_GLOBAL_ON_DDS))
                    self.raman.global_dds.on()
                else:
                    self.raman.global_dds.off()

                delay(25 * us)
                # Doppler Cooling
                # self.doppler_cool(monitor=False)
                stopcounter_doppler_mu = self.doppler_cool(monitor=True)
                delay(50*us)
                doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)
                delay(10 * us)  # To give some slack back after reading PMTs

                # Second Stage Cooling
                if self.do_SS_cool:
                    # Middle ion counts
                    stopcounter_ss_mu = self.second_stage_cool(monitor=True)
                    delay(50 * us)
                    ss_counts = self.pmt_array.count(
                        up_to_time_mu=stopcounter_ss_mu, buffer=ss_counts
                    )
                    delay(200 * us)

                    stopcounter_ss_coolant_mu = self.second_stage_cool_coolant(monitor=True)
                    # stopcounter_ss_coolant_mu = self.second_stage_cool(monitor=True)
                    delay(50 * us)
                    ss_coolant_counts = self.pmt_array.count(
                        up_to_time_mu=stopcounter_ss_coolant_mu, buffer=ss_coolant_counts
                    )
                    delay(200 * us)

                    # stopcounter_ss_all_mu = self.second_stage_cool_all(monitor=True)
                    # # stopcounter_ss_all_mu = self.second_stage_cool(monitor=True)
                    # delay(50*us)
                    # ss_all_counts = self.pmt_array.count(
                    #     up_to_time_mu=stopcounter_ss_all_mu, buffer=ss_all_counts
                    # )
                    # delay(200*us)  # To give some slack back after reading PMTs

                    # Edge ion counts
                    self.sandia_box.default_to_edgeimage()
                    delay(200 * us)
                    # stopcounter_ss_edge_mu = self.second_stage_cool(monitor=True)
                    # delay(50*us)
                    # ss_edge_counts = self.pmt_array.count(
                    #     up_to_time_mu=stopcounter_ss_edge_mu, buffer=ss_edge_counts
                    # )
                    # delay(200*us)
                    stopcounter_ss_edge_coolant_mu = self.second_stage_cool_coolant(monitor=True)
                    delay(50 * us)
                    ss_edge_coolant_counts = self.pmt_array.count(
                        up_to_time_mu=stopcounter_ss_edge_coolant_mu, buffer=ss_edge_coolant_counts
                    )
                    delay(200 * us)
                    # stopcounter_ss_edge_all_mu = self.second_stage_cool_all(monitor=True)
                    # delay(50*us)
                    # ss_edge_all_counts = self.pmt_array.count(
                    #     up_to_time_mu=stopcounter_ss_edge_all_mu, buffer=ss_edge_all_counts
                    # )
                    # delay(200*us)
                    self.sandia_box.edgeimage_to_default()
                    delay(200 * us)
                else:
                    stopcounter_ss_mu = now_mu()
                    delay(10 * us)
                    stopcounter_ss_coolant_mu = now_mu()
                    delay(10 * us)
                    stopcounter_ss_all_mu = now_mu()
                    delay(20 * us)

                # Todo: Add my shuttling function here

                if self.do_pump or self.rfsoc_sbc.do_sbc:
                    # prepare pumping
                    self.pump_detect.prepare_pump()

                # Sideband Cooling
                if self.use_RFSOC and self.rfsoc_sbc.do_sbc:
                    self.SDDDS.write_to_dds()
                    self.raman.set_global_aom_source(dds=False)
                    self.rfsoc_sbc.kn_do_rfsoc_sbc()
                    self.raman.set_global_aom_source(dds=True)

                if self.do_pump and self.pump_detect.do_slow_pump:
                    self.pump_detect.update_amp(amp_int=self.pump_detect.slow_pump_amp)
                    delay(150 * us)  # wait for the amplitude to actually reach the set value

                self.raman.global_dds.off()
                self.raman.set_global_aom_source(dds=False)

                delay(5 * us)
                # Slow pump to the 0 state
                if self.do_pump:
                    self.pump_detect.pump()

                # prepare the AOMs for detection
                self.pump_detect.prepare_detect()

                # Perform the main experiment
                self.main_experiment(istep, ishot)

                # Detect Ions
                stopcounter_detect_mu = self.detect()

                self.eom_935_3ghz.off()

                if self.keep_global_on:
                    self.raman.global_dds.update_amp(np.int32(self._KEEP_GLOBAL_ON_DDS))
                    self.raman.global_dds.on()
                else:
                    self.raman.global_dds.off()

                # if self.do_calib:
                #     self.update_pointing_lock(calib_counts=calib_counts)

                # Readout out counts and save the data
                # calib_counts = self.pmt_array.count(up_to_time_mu=stopcounter_calib_mu, buffer=calib_counts)
                #doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)

                detect_counts = self.pmt_array.count(
                    up_to_time_mu=stopcounter_detect_mu, buffer=detect_counts
                )

                self.save_counts(
                    ishot=ishot,
                    istep=istep,
                    calib_counts=calib_counts,
                    doppler_counts=doppler_counts,
                    ss_counts=ss_counts,
                    ss_coolant_counts=ss_coolant_counts,
                    ss_all_counts=ss_all_counts,
                    ss_edge_counts=ss_edge_counts,
                    ss_edge_coolant_counts=ss_edge_coolant_counts,
                    ss_edge_all_counts=ss_edge_all_counts,
                    detect_counts=detect_counts,
                    tlock_feedback=self.calib_tlock_int,
                    xlock_feedback=self.calib_xlock_int,
                )

                self.custom_proc_counts(ishot=ishot, istep=istep)

                # detect current chain config, give reorder flag based on config in last shot and current config
                # if detected 171 and 172 at the same pmt, it's determined as an invalid count, and we terminate
                # the experiment
                prev_config = config

                # use the ss (qubit) counts and ss (coolant) counts to tell the config
                np_pmt_count_array_qubit = np.array(ss_counts)
                np_pmt_count_array_coolant = np.array(ss_coolant_counts)
                np_pmt_count_array_qubit_edge = np.array(ss_edge_counts)
                np_pmt_count_array_coolant_edge = np.array(ss_edge_coolant_counts)
                np_presence_qubit = [1 if np_pmt_count_array_qubit[k] > self.presence_thresh else 0 for k in
                                     range(self.num_pmts)]
                np_presence_coolant = [2 if np_pmt_count_array_coolant[k] > self.presence_thresh_coolant else 0 for k in
                                       range(self.num_pmts)]
                # Modify edge ions in presence with edge counts
                # Hardcoded edge length to be 2
                # Only execute when loaded more than 4 ions
                if self.num_pmts > 4:
                    for k in range(1, 3):
                        # left end
                        # if np_pmt_count_array_qubit_edge[k] > self.presence_thresh:
                        #     np_presence_qubit[k] = 1
                        # else:
                        #     np_presence_qubit[k] = 0
                        if np_pmt_count_array_coolant_edge[k - 1] > self.presence_thresh_coolant:
                            np_presence_coolant[k - 1] = 2
                        else:
                            np_presence_coolant[k - 1] = 0
                        # right end
                        # if np_pmt_count_array_qubit_edge[-k] > self.presence_thresh:
                        #     np_presence_qubit[-k] = 1
                        # else:
                        #     np_presence_qubit[-k] = 0
                        if np_pmt_count_array_coolant_edge[-k] > self.presence_thresh_coolant:
                            np_presence_coolant[-k] = 2
                        else:
                            np_presence_coolant[-k] = 0

                # config = [np_presence_qubit[k] + np_presence_coolant[k] for k in range(self.num_pmts)]

                # for k in range(self.num_pmts):
                #     assert config[k] < 3, "Invalid count occurs, experiment terminates"

                # use only 172 ss counts to get configuration of the chain (without edge correction)
                config = np_presence_coolant

                self.save_config(config, ishot, istep)
                reorder_flag = self.reorder_detect(prev_config, config, ishot, istep)
                self.save_reorder(reorder_flag, ishot, istep)

                if reorder_flag:
                    # break
                    pass
            # sorting: once reordered, sort the chain from config to prev_config
            if reorder_flag:
                sorting_sol = self.sorting_sol_generator(prev_config, config, sorting_sol)
                #sorting_sol = self.sorting_sol_generator([2, 2, 2, 1, 1], [1, 1, 2, 2, 2], sorting_sol)
                for i in range(len(sorting_sol)):
                    # actual sorting sol doesn't take the whole buffer, for those not in use are -1
                    if i >= 0:
                        self.swap(i, i + 1)

            # threshold raw counts, save data, and push it to the plot applet
            self.threshold_data(istep)

            # run any other code that is executed on the host
            # at the end of each x-point (after num_shots exps are taken)
            # this function should be RPC on the host
            self.custom_proc_data(istep)

            # Increment equilibration loops until specified then increment the step counter

            self.module_update(istep)

            if self.equilibrated:
                istep += 1
            else:
                iequil += 1
                if iequil >= self.equil_loops:
                    self.equilibrated = True

# class Swap_Check(Sorting_Environment):

#     data_folder = "Sort"
#     applet_name = "Swap Check"
#     applet_group = "Sort"

#     def build(self):
#         super(Swap_Check, self).build()

#         self.ramp_during_swap = self.get_argument(
#             "Ramp during swap",
#             artiq_env.BooleanValue(default=True)
#         )

#         self.ion_ind = self.get_argument(
#             "Ion to swap (zero indexed)",
#             NumberValue(default=11, unit='', ndecimals=0, scale=1, step=1)
#         )

#     def prepare(self):

#         super(Swap_Check, self).prepare()

#         self.split_shuttle = self.sandia_box.get_shuttle_data_stepnum(self.ion_ind, 0, self.ion_ind, 3)
#         self.swap_shuttle_one = self.sandia_box.get_shuttle_data_stepnum(self.ion_ind, 3, self.ion_ind, 4)
#         self.swap_shuttle_two = self.sandia_box.get_shuttle_data_stepnum(self.ion_ind, 4, self.ion_ind, 5)
#         self.merge_shuttle = self.sandia_box.get_shuttle_data_stepnum(self.ion_ind, 3, self.ion_ind, 0)

#     @kernel
#     def experiment_loop(self):
#         # Check chain config before swap
#         # self.trigger_camera()
#         self.core.break_realtime()
#         self.split()
#         self.swap_one()
#         self.trigger_camera()
#         delay(1000 * ms)
#         self.swap_two()
#         self.merge()

#         # Check chain after each swap
#         # self.trigger_camera()

#     @kernel
#     def split(self):
#         self.sandia_box.shuttle_path(self.split_shuttle)

#     @kernel
#     def swap_one(self):
#         self.sandia_box.shuttle_path(self.swap_shuttle_one)

#     @kernel
#     def swap_two(self):
#         self.sandia_box.shuttle_path(self.swap_shuttle_two)

#     @kernel
#     def merge(self):
#         self.sandia_box.shuttle_path(self.merge_shuttle)

#     @kernel
#     def ramp_down(self):
#         if self.ramp_rf:
#             self.core.break_realtime()
#             self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
#             self.core.break_realtime()
#             self.rf_lock_control(True)
#             self.rf_ramp.run_ramp_down_kernel()
#         self.kernel_status_check()

#     @kernel
#     def ramp_up(self):
#         if self.ramp_rf:
#             self.core.break_realtime()
#             self.rf_ramp.run_ramp_up_kernel()
#             self.rf_ramp.deactivate_ext_mod()
#             # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
#             self.core.break_realtime()
#             self.rf_lock_control(False)
#         self.kernel_status_check()

#     @kernel
#     def custom_init(self):
#         if self.ramp_rf and (not self.ramp_during_swap):
#             self.core.break_realtime()
#             self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
#             self.core.break_realtime()
#             self.rf_lock_control(True)
#             self.rf_ramp.run_ramp_down_kernel()
#         self.kernel_status_check()

#     @kernel
#     def custom_idle(self):
#         if self.ramp_rf and (not self.ramp_during_swap):
#             self.core.break_realtime()
#             self.rf_ramp.run_ramp_up_kernel()
#             self.rf_ramp.deactivate_ext_mod()
#             # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
#             self.core.break_realtime()
#             self.rf_lock_control(False)
#         self.kernel_status_check()

#     @kernel
#     def rf_lock_control(self, state: TBool):
#         # state: True, hold, not locking
#         # self.core.wait_until_mu(now_mu())
#         if not state:
#             self.rf_lock_switch.off()
#         else:
#             self.rf_lock_switch.on()

#     @rpc(flags={"async"})
#     def kernel_status_check(self):
#         # Check the status of the class: RampControl in artiq_dac.py
#         print("RampControl Seed:", self.rf_ramp.seed)
#         # Check the status of the class: DAC8568 in dac8568.py
#         print("DAC8568 Seed:", self.rf_ramp.SandiaSerialDAC.seed)
#         # Check the status of the class: RFSignalGenerator in smc100a.py
#         print("R&S Seed:", self.rf_ramp.rf_source.seed)

# class Sort(Sorting_Environment):
#
#     data_folder = "Sort"
#     applet_name = "Sort"
#     applet_group = "Sort"
#
#     # Need to check if there is a better way
#     split_path = []
#     swap_path = []
#     merge_path = []
#
#     def build(self):
#
#         super(Sort, self).build()
#
#         self.ramp_during_swap = self.get_argument(
#             "Ramp during swap",
#             artiq_env.BooleanValue(default=True)
#         )
#
#     def prepare(self):
#
#         super(Sort, self).prepare()
#
#         for ionind in range(self.num_pmts - 1):
#             self.split_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 0, ionind, 3))
#             self.swap_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 4))
#             self.merge_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 0))
#
#     @kernel
#     def experiment_loop(self):
#         # Create input buffer
#         calib_counts = [0] * self.num_pmts
#         doppler_counts = [0] * self.num_pmts
#         ss_counts = [0] * self.num_pmts
#         ss_coolant_counts = [0] * self.num_pmts
#         ss_all_counts = [0] * self.num_pmts
#         ss_edge_counts = [0] * self.num_pmts
#         ss_edge_coolant_counts = [0] * self.num_pmts
#         ss_edge_all_counts = [0] * self.num_pmts
#         detect_counts = [0] * self.num_pmts
#         config = [0] * self.num_pmts
#
#         # maximal number of swaps required for reorder: Coolantnum * self.num_pmts - Coolantnum ^ 2
#         # (23, 5): 90
#         sorting_sol = [-1] * 90
#         # Desired config
#         DC = [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2]
#         # CC
#         ss_counts, ss_coolant_counts, ss_edge_coolant_counts, CC = self.check_config(ss_counts, ss_coolant_counts, ss_edge_coolant_counts, config)
#         self.save_counts(
#                     ishot=0,
#                     istep=0,
#                     calib_counts=calib_counts,
#                     doppler_counts=doppler_counts,
#                     ss_counts=ss_counts,
#                     ss_coolant_counts=ss_coolant_counts,
#                     ss_all_counts=ss_coolant_counts,
#                     ss_edge_counts=ss_coolant_counts,
#                     ss_edge_coolant_counts=ss_edge_coolant_counts,
#                     ss_edge_all_counts=ss_coolant_counts,
#                     detect_counts=detect_counts,
#                     tlock_feedback=self.calib_tlock_int,
#                     xlock_feedback=self.calib_xlock_int,
#         )
#         self.save_config(config, 0, 0)
#         # Hard-coded CC (only involving good pairs)
#         # CC = [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1]
#         sorting_sol = self.sorting_sol_generator(DC, CC, sorting_sol)
#         print("CC", CC)
#         print("Sorting solution: ", sorting_sol)
#
#
#
#         # Execute sorting
#         for i in range(len(sorting_sol)):
#             # actual sorting sol doesn't take the whole buffer, for those not in use are -1
#             if sorting_sol[i] >= 0:
#                 self.split(sorting_sol[i])
#                 # Check config
#                 ss_counts, ss_coolant_counts, ss_edge_coolant_counts, config = self.check_config(ss_counts, ss_coolant_counts, ss_edge_coolant_counts, config)
#                 self.save_counts(
#                     ishot=1+3*i,
#                     istep=0,
#                     calib_counts=calib_counts,
#                     doppler_counts=doppler_counts,
#                     ss_counts=ss_counts,
#                     ss_coolant_counts=ss_coolant_counts,
#                     ss_all_counts=ss_coolant_counts,
#                     ss_edge_counts=ss_coolant_counts,
#                     ss_edge_coolant_counts=ss_edge_coolant_counts,
#                     ss_edge_all_counts=ss_coolant_counts,
#                     detect_counts=detect_counts,
#                     tlock_feedback=self.calib_tlock_int,
#                     xlock_feedback=self.calib_xlock_int,
#                 )
#                 self.save_config(config, 1+3*i, 0)
#                 self.swap(sorting_sol[i])
#                 self.save_counts(
#                     ishot=2+3*i,
#                     istep=0,
#                     calib_counts=calib_counts,
#                     doppler_counts=doppler_counts,
#                     ss_counts=ss_counts,
#                     ss_coolant_counts=ss_coolant_counts,
#                     ss_all_counts=ss_coolant_counts,
#                     ss_edge_counts=ss_coolant_counts,
#                     ss_edge_coolant_counts=ss_edge_coolant_counts,
#                     ss_edge_all_counts=ss_coolant_counts,
#                     detect_counts=detect_counts,
#                     tlock_feedback=self.calib_tlock_int,
#                     xlock_feedback=self.calib_xlock_int,
#                 )
#                 self.save_config(config, 2+3*i, 0)
#                 self.merge(sorting_sol[i])
#                 self.save_counts(
#                     ishot=3+3*i,
#                     istep=0,
#                     calib_counts=calib_counts,
#                     doppler_counts=doppler_counts,
#                     ss_counts=ss_counts,
#                     ss_coolant_counts=ss_coolant_counts,
#                     ss_all_counts=ss_coolant_counts,
#                     ss_edge_counts=ss_coolant_counts,
#                     ss_edge_coolant_counts=ss_edge_coolant_counts,
#                     ss_edge_all_counts=ss_coolant_counts,
#                     detect_counts=detect_counts,
#                     tlock_feedback=self.calib_tlock_int,
#                     xlock_feedback=self.calib_xlock_int,
#                 )
#                 self.save_config(config, 3+3*i, 0)
#
#             else:
#                 pass
#
#
#     @kernel
#     def check_config(self, ss_counts, ss_coolant_counts, ss_edge_coolant_counts, config):
#         # Middle ion counts
#         stopcounter_ss_mu = self.second_stage_cool(monitor=True)
#         delay(50 * us)
#         ss_counts = self.pmt_array.count(
#             up_to_time_mu=stopcounter_ss_mu, buffer=ss_counts
#         )
#         delay(200 * us)
#
#         stopcounter_ss_coolant_mu = self.second_stage_cool_coolant(monitor=True)
#         # stopcounter_ss_coolant_mu = self.second_stage_cool(monitor=True)
#         delay(50 * us)
#         ss_coolant_counts = self.pmt_array.count(
#             up_to_time_mu=stopcounter_ss_coolant_mu, buffer=ss_coolant_counts
#         )
#         delay(200 * us)
#
#         # Edge ion counts
#         self.sandia_box.default_to_edgeimage()
#         delay(200 * us)
#         # stopcounter_ss_edge_mu = self.second_stage_cool(monitor=True)
#         # delay(50*us)
#         # ss_edge_counts = self.pmt_array.count(
#         #     up_to_time_mu=stopcounter_ss_edge_mu, buffer=ss_edge_counts
#         # )
#         # delay(200*us)
#         stopcounter_ss_edge_coolant_mu = self.second_stage_cool_coolant(monitor=True)
#         delay(50 * us)
#         ss_edge_coolant_counts = self.pmt_array.count(
#             up_to_time_mu=stopcounter_ss_edge_coolant_mu, buffer=ss_edge_coolant_counts
#         )
#         delay(200 * us)
#         # stopcounter_ss_edge_all_mu = self.second_stage_cool_all(monitor=True)
#         # delay(50*us)
#         # ss_edge_all_counts = self.pmt_array.count(
#         #     up_to_time_mu=stopcounter_ss_edge_all_mu, buffer=ss_edge_all_counts
#         # )
#         # delay(200*us)
#         self.sandia_box.edgeimage_to_default()
#         delay(200 * us)
#
#         # Check config
#         np_pmt_count_array_qubit = np.array(ss_counts)
#         np_pmt_count_array_coolant = np.array(ss_coolant_counts)
#         np_pmt_count_array_coolant_edge = np.array(ss_edge_coolant_counts)
#
#         np_presence_coolant = [2 if np_pmt_count_array_coolant[k] > self.presence_thresh_coolant else 1 for k in
#                                        range(self.num_pmts)]
#
#         # Modify edge ions in presence with edge counts
#         # Hardcoded edge length to be 2
#         # Only execute when loaded more than 4 ions
#         if self.num_pmts > 4:
#             for k in range(1, 3):
#                 # left end
#                 # if np_pmt_count_array_qubit_edge[k] > self.presence_thresh:
#                 #     np_presence_qubit[k] = 1
#                 # else:
#                 #     np_presence_qubit[k] = 0
#                 if np_pmt_count_array_coolant_edge[k - 1] > self.presence_thresh_coolant:
#                     np_presence_coolant[k - 1] = 2
#                 else:
#                     np_presence_coolant[k - 1] = 1
#                 # right end
#                 # if np_pmt_count_array_qubit_edge[-k] > self.presence_thresh:
#                 #     np_presence_qubit[-k] = 1
#                 # else:
#                 #     np_presence_qubit[-k] = 0
#                 if np_pmt_count_array_coolant_edge[-k] > self.presence_thresh_coolant:
#                     np_presence_coolant[-k] = 2
#                 else:
#                     np_presence_coolant[-k] = 1
#
#         config = np_presence_coolant
#
#         return ss_counts, ss_coolant_counts, ss_edge_coolant_counts, config
#
#     @kernel
#     def split(self, ind):
#         self.sandia_box.shuttle_path(self.split_path[ind])
#
#     @kernel
#     def swap(self, ind):
#         self.sandia_box.shuttle_path(self.swap_path[ind])
#
#     @kernel
#     def merge(self, ind):
#         self.sandia_box.shuttle_path(self.merge_path[ind])
#
#     @kernel
#     def ramp_down(self):
#         if self.ramp_rf:
#             self.core.break_realtime()
#             self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
#             self.core.break_realtime()
#             self.rf_lock_control(True)
#             self.rf_ramp.run_ramp_down_kernel()
#         #self.kernel_status_check()
#
#     @kernel
#     def ramp_up(self):
#         if self.ramp_rf:
#             self.core.break_realtime()
#             self.rf_ramp.run_ramp_up_kernel()
#             self.rf_ramp.deactivate_ext_mod()
#             # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
#             self.core.break_realtime()
#             self.rf_lock_control(False)
#         #self.kernel_status_check()
#
#     @kernel
#     def custom_init(self):
#         if self.ramp_rf and (not self.ramp_during_swap):
#             self.core.break_realtime()
#             self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
#             self.core.break_realtime()
#             self.rf_lock_control(True)
#             self.rf_ramp.run_ramp_down_kernel()
#         #self.kernel_status_check()
#
#     @kernel
#     def custom_idle(self):
#         if self.ramp_rf and (not self.ramp_during_swap):
#             self.core.break_realtime()
#             self.rf_ramp.run_ramp_up_kernel()
#             self.rf_ramp.deactivate_ext_mod()
#             # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
#             self.core.break_realtime()
#             self.rf_lock_control(False)
#         #self.kernel_status_check()
#
#     @kernel
#     def rf_lock_control(self, state: TBool):
#         # state: True, hold, not locking
#         # self.core.wait_until_mu(now_mu())
#         if not state:
#             self.rf_lock_switch.off()
#         else:
#             self.rf_lock_switch.on()
#
#     @rpc(flags={"async"})
#     def kernel_status_check(self):
#         # Check the status of the class: RampControl in artiq_dac.py
#         print("RampControl Seed:", self.rf_ramp.seed)
#         # Check the status of the class: DAC8568 in dac8568.py
#         print("DAC8568 Seed:", self.rf_ramp.SandiaSerialDAC.seed)
#         # Check the status of the class: RFSignalGenerator in smc100a.py
#         print("R&S Seed:", self.rf_ramp.rf_source.seed)

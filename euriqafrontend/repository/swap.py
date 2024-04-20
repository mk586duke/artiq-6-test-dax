import logging
import time
import numpy as np
import math

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import oitg.fitting as fit
import euriqafrontend.fitting as umdfit
from artiq.experiment import TerminationRequested
from artiq.experiment import BooleanValue
from artiq.experiment import NumberValue
from artiq.experiment import StringValue
from artiq.language.core import kernel, delay, host_only, rpc, parallel, delay_mu, now_mu
from artiq.language.units import ms, MHz, us, kHz
from artiq.language import TInt32, TBool
from euriqabackend.coredevice.ad9912 import phase_to_mu, freq_to_mu

from euriqafrontend.repository.basic_environment import BasicEnvironment
from euriqafrontend.modules.artiq_dac import RampControl as rampcontrol_auto

_LOGGER = logging.getLogger(__name__)

class Split_Environment(BasicEnvironment, artiq_env.Experiment):
    """Split.Env
    Inherit from BasicEnvironment which handles most of the infrastructure
    Used to test Split operations and calibrate Dx offsets
    """

    data_folder = "Split"
    applet_name = "Split_Test_Async"
    applet_group = "Split"

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
        self.dac_starting_node="1-1"
        self.sandia_box.dac_pc.apply_line_async(self.dac_starting_node, line_gain=1)
    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_init()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            self.custom_idle()
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            self.custom_idle()
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def custom_init(self):
        # self.sandia_box.dac_pc.apply_line_async(self.dac_starting_node, line_gain=0, global_gain=0)
        pass

    @kernel
    def trigger_camera(self):
        # Camera external trigger tied to AWG
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
        detect_counts = [0] * self.num_pmts

        self.core.break_realtime()
        # Set switch network to sequence mode
        # if not self.use_RFSOC:
        #     self.AWG.setRunningSN(True)
        delay(10 * us)

        istep = 0  # Counter for steps
        iequil = 0  # Counter for equilibration rounds

        ion_swap_index = -1

        # Todo: Loop until fully sorted or timeout
        # Loop over main experimental scan parameters
        while istep < len(self.scan_values):
            # check for pause/abort from the host
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()

            # do any initialization that needs to be done for each step
            # such as variable hardware-shuttling, variable piezo setting, etc.
            self.prepare_step(istep)

            # Single shot for swapping
            self.num_shots = 1
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
                # if self.keep_global_on:
                #     self.raman.global_dds.update_amp(np.int32(self._KEEP_GLOBAL_ON_DDS))
                #     self.raman.global_dds.on()
                # else:
                #     self.raman.global_dds.off()

                delay(25 * us)
                # Doppler Cooling
                # self.doppler_cool(monitor=False)
                stopcounter_doppler_mu = self.doppler_cool(monitor=True)
                # doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)
                # delay(10 * us)  # To give some slack back after reading PMTs

                # Second Stage Cooling
                if self.do_SS_cool:
                    stopcounter_ss_mu = self.second_stage_cool(monitor=True)
                    # ss_counts = self.pmt_array.count(up_to_time_mu=stopcounter_ss_mu, buffer=ss_counts)
                    delay(20 * us)  # To give some slack back after reading PMTs
                else:
                    stopcounter_ss_mu = now_mu()

                # SWAP ions
                self.sandia_box.swap(ion_swap_index)

                if self.do_pump or self.rfsoc_sbc.do_sbc:
                    # prepare pumping
                    self.pump_detect.prepare_pump()

                # Sideband Cooling
                # if self.use_RFSOC and self.rfsoc_sbc.do_sbc:
                #     self.raman.set_global_aom_source(dds=False)
                #     self.rfsoc_sbc.kn_do_rfsoc_sbc()

                # self.raman.global_dds.off()
                # delay(5*us)
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

                # if self.keep_global_on:
                #     self.raman.global_dds.update_amp(np.int32(self._KEEP_GLOBAL_ON_DDS))
                #     self.raman.global_dds.on()
                # else:
                #     self.raman.global_dds.off()

                # if self.do_calib:
                #     self.update_pointing_lock(calib_counts=calib_counts)

                # Readout out counts and save the data
                # calib_counts = self.pmt_array.count(up_to_time_mu=stopcounter_calib_mu, buffer=calib_counts)
                doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)
                ss_counts = self.pmt_array.count(
                    up_to_time_mu=stopcounter_ss_mu, buffer=ss_counts
                )
                detect_counts = self.pmt_array.count(
                    up_to_time_mu=stopcounter_detect_mu, buffer=detect_counts
                )

                self.save_counts(
                    ishot=ishot,
                    istep=istep,
                    calib_counts=calib_counts,
                    doppler_counts=doppler_counts,
                    ss_counts=ss_counts,
                    detect_counts=detect_counts,
                    tlock_feedback=self.calib_tlock_int,
                    xlock_feedback=self.calib_xlock_int,
                )

                self.custom_proc_counts(ishot=ishot, istep=istep)

            # Todo: update ion swap index based on PMT
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

        # Clean up after the experiment
        # if self.lock_x_piezo:
        #     final_piezo = self.raman.ind_final_piezos.get_value1()
        #     self.set_dataset(
        #         "global.Raman.Piezos.Ind_FinalX", final_piezo, persist=True
        #     )
        # if self.lock_calib_t:
        #     # TODO: clean up this machine unit to second conversion by adding appropriate vars and helper functions
        #     final_t_mu = (
        #         self.calib_wait_time_mu
        #         - self.calib_tlock_step_mu * self.calib_tlock_int // 2
        #     )
        #     final_t = final_t_mu * self.core.ref_period
        #     self.set_dataset("monitor.Stark_Twait", final_t, persist=True)

class Swap_Test(Split_Environment):
    """Split.Swap_Test
    Inherit from BasicEnvironment which handles most of the infrastructure
    Used to test Split operations and calibrate Dx offsets
    """

    data_folder = "Split"
    applet_name = "Swap_test"
    applet_group = "Split"

    def build(self):
        super(Swap_Test, self).build()

        self.ion_swap_index = self.get_argument(
            "ion to swap",
            NumberValue(default=0, unit="", ndecimals=0, scale=1, step=1),
        )

        self.hold_time_home_ms = self.get_argument(
            "Hold Time Home (ms)",
            NumberValue(default=500 * ms, unit="ms", ndecimals=3),
        )

        self.N_repeats = self.get_argument(
            "Number of Repeats",
            NumberValue(default=10, unit="", ndecimals=0, scale=1, step=1),
        )

    def prepare(self):
        super(Swap_Test, self).prepare()


    @kernel
    def experiment_loop(self):
        """Start the experiment on the host."""
        _LOGGER.info("Start Run")
        self.core.break_realtime()
        # self.sandia_box.jump_to_start(self.ion_swap_index)
        for i in range(self.N_repeats):
            self.core.break_realtime()
            self.sandia_box.swap(self.ion_swap_index)
            self.trigger_camera()
            delay(self.hold_time_home_ms)

            _LOGGER.info("Shuttled %d", i)
            self.core.wait_until_mu(now_mu())

    @kernel
    def custom_init(self):
        self.core.break_realtime()
        if self.ramp_rf:
            # Ramp down
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_idle(self):
        self.core.break_realtime()
        if self.ramp_rf:
            # Ramp up
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()


# class Split_Test_Sync(Split_Environment):
#     """Split.Test_Sync
#     Inherit from BasicEnvironment which handles most of the infrastructure
#     Used to test Split operations and calibrate Dx offsets
#     Replaced by Voltage.Check.Sync, currently it is obsolete.
#     """

#     data_folder = "Split"
#     applet_name = "Split_Test_Sync"
#     applet_group = "Split"

#     kernel_invariants = {"shuttle_to", "shuttle_from"}

#     def build(self):
#         super(Split_Test_Sync, self).build()

#         self.ion_swap_index = self.get_argument(
#             "ion to swap",
#             NumberValue(default=0, unit="", ndecimals=0, scale=1, step=1),
#         )

#         self.final_node = self.get_argument(
#             "Final Node",
#             NumberValue(default=3, unit="", ndecimals=0, scale=1, step=1),
#         )

#         self.hold_time_split_ms = self.get_argument(
#             "Hold Time Split (ms)",
#             NumberValue(default=50*ms, unit="ms", ndecimals=3),
#         )

#         self.hold_time_home_ms = self.get_argument(
#             "Hold Time Home (ms)",
#             NumberValue(default=500 * ms, unit="ms", ndecimals=3),
#         )

#         self.N_repeats = self.get_argument(
#             "Number of Repeats",
#             NumberValue(default=10, unit="", ndecimals=0, scale=1, step=1),
#         )

#         self.trigger_at_swap = self.get_argument(
#             "Trigger at Swap",
#             artiq_env.BooleanValue(default=True),
#         )

#         self.trigger_at_home = self.get_argument(
#             "Trigger at Home",
#             artiq_env.BooleanValue(default=True),
#         )

#     def prepare(self):
#         super(Split_Test_Sync, self).prepare()

#         # Special case: When final_node is 5, attempt to simulate a swap
#         if(self.final_node == 5):
#             self.shuttle_to = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 0, self.ion_swap_index, 3)
#             self.shuttle_swap = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 3, self.ion_swap_index, 4)
#             self.shuttle_from = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 3, self.ion_swap_index, 0)
#         else:
#             self.shuttle_to = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 0, self.ion_swap_index, self.final_node)
#             self.shuttle_swap = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, self.final_node, self.ion_swap_index, self.final_node)
#             self.shuttle_from = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, self.final_node, self.ion_swap_index, 0)

#         _LOGGER.warning(f"Shuttle Time (ms): {1000 * self.core.mu_to_seconds(np.sum(self.shuttle_to.timings))}")


#     @kernel
#     def experiment_loop(self):
#         """Start the experiment on the host."""
#         _LOGGER.info("Start Run")
#         self.core.break_realtime()
#         # self.sandia_box.jump_to_start(self.ion_swap_index)
#         for i in range(self.N_repeats):
#             self.core.break_realtime()
#             self.sandia_box.shuttle_path(self.shuttle_to)

#             # For checking swap waveform
#             if self.trigger_at_swap and self.final_node == 5:
#                 self.trigger_camera()
#                 delay(self.hold_time_split_ms)

#             self.sandia_box.shuttle_path(self.shuttle_swap)

#             if self.trigger_at_swap:
#                 self.trigger_camera()

#             delay(self.hold_time_split_ms)

#             self.sandia_box.shuttle_path(self.shuttle_from)

#             if self.trigger_at_home:
#                 self.trigger_camera()

#             delay(self.hold_time_home_ms)

#             _LOGGER.info("Shuttled %d", i)
#             self.core.wait_until_mu(now_mu())

#     @kernel
#     def ramp_down(self):
#         if self.ramp_rf:
#             self.core.break_realtime()
#             self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
#             self.core.break_realtime()
#             self.rf_lock_control(True)
#             self.rf_ramp.run_ramp_down_kernel()


#     @kernel
#     def ramp_up(self):
#         if self.ramp_rf:
#             self.core.break_realtime()
#             self.rf_ramp.run_ramp_up_kernel()
#             self.rf_ramp.deactivate_ext_mod()
#             # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
#             self.core.break_realtime()
#             self.rf_lock_control(False)


#     @kernel
#     def rf_lock_control(self, state: TBool):
#         # state: True, hold, not locking
#         # self.core.wait_until_mu(now_mu())
#         if not state:
#             self.rf_lock_switch.off()
#         else:
#             self.rf_lock_switch.on()

#     @kernel
#     def custom_init(self):
#         if self.ramp_rf:
#             self.core.break_realtime()
#             self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
#             self.core.break_realtime()
#             self.rf_lock_control(True)
#             self.rf_ramp.run_ramp_down_kernel()


#     @kernel
#     def custom_idle(self):
#         if self.ramp_rf:
#             self.core.break_realtime()
#             self.rf_ramp.run_ramp_up_kernel()
#             self.rf_ramp.deactivate_ext_mod()
#             # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
#             self.core.break_realtime()
#             self.rf_lock_control(False)

class Split_Test_Async(Split_Environment):
    """Split.Test_Async
    Inherit from BasicEnvironment which handles most of the infrastructure
    Used to test Split operations and calibrate Dx offsets
    """

    data_folder = "Split"
    applet_name = "Split_Test_Async"
    applet_group = "Split"

    def build(self):
        super(Split_Test_Async, self).build()

        self.start_linenum = self.get_argument(
            "Starting Line Number",
            NumberValue(default=182, unit="", ndecimals=0, scale=1, step=1),
        )

        self.stop_linenum = self.get_argument(
            "Ending Line Number",
            NumberValue(default=222, unit="", ndecimals=0, scale=1, step=1),
        )# 182, 222, 364, 498

        self.delay_time_ms = self.get_argument(
            "Delay Time (ms)",
            NumberValue(default=10*ms, unit="ms", ndecimals=3),
        )

        self.N_repeats = self.get_argument(
            "Number of Repeats",
            NumberValue(default=10, unit="", ndecimals=0, scale=1, step=1),
        )

    def experiment_loop(self):
        """Start the experiment on the host."""
        _LOGGER.info("Start Run")
        for r in range(self.N_repeats):
            _LOGGER.info(r)
            for i in range(self.start_linenum, self.stop_linenum+1):
                time.sleep(self.delay_time_ms)
                self.sandia_box.dac_pc.apply_line_async(i, line_gain=1, global_gain=1)
                _LOGGER.info(f"{r}-{i}")
            time.sleep(.05)
            self.trigger_camera()
            time.sleep(.5)
            for i in reversed(range(self.start_linenum, self.stop_linenum+1)):
                time.sleep(self.delay_time_ms)
                self.sandia_box.dac_pc.apply_line_async(i, line_gain=1, global_gain=1)
                _LOGGER.info(f"{r}-{i}")

            self.trigger_camera()
            time.sleep(.5)
        time.sleep(1)

    @kernel
    def custom_init(self):
        self.core.break_realtime()
        if self.ramp_rf:
            # Ramp down
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_idle(self):
        self.core.break_realtime()
        if self.ramp_rf:
            # Ramp up
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

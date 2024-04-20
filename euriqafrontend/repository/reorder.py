import logging
import time

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import artiq.language.units as artiq_units
import numpy as np
import oitg.fitting as fit
from artiq.experiment import NumberValue
from artiq.experiment import TerminationRequested
from artiq.language.core import delay
from artiq.language.core import host_only, rpc
from artiq.language.core import kernel
from artiq.language.core import now_mu
from artiq.language.core import parallel
from artiq.language.types import TInt32, TInt64, TFloat, TBool, TList
from artiq.language.units import MHz
from artiq.language.units import us

import euriqafrontend.fitting as umd_fit
from euriqabackend.coredevice.ad9912 import freq_to_mu
# from euriqafrontend.repository.basic_environment_sort import BasicEnvironment
from euriqafrontend.repository.basic_env import BasicEnvironment
from euriqafrontend.modules.spec_analyzer import Spec_Analyzer
from artiq.language.units import ms, MHz, us, kHz
from euriqabackend.coredevice.ad9912 import phase_to_mu, freq_to_mu
from artiq.language.environment import HasEnvironment

from euriqafrontend.modules.artiq_dac import RampControl as rampcontrol_auto

_LOGGER = logging.getLogger(__name__)

class Reordering_Rate(BasicEnvironment, artiq_env.Experiment):
    """Reordering.Rate"""

    data_folder = "Reordering.Rate"
    applet_name = "Reordering.Rate"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

    def build(self):
        """Initialize experiment & variables."""

        super(Reordering_Rate, self).build()

        self.ion_swap_index = self.get_argument(
            "ion to swap",
            NumberValue(default=0, unit="", ndecimals=0, scale=1, step=1),
        )

        self.shuttle_flag = self.get_argument(
            "shuttle",
            artiq_env.BooleanValue(default=True),
        )

        self.shuttle_node = self.get_argument(
            "shuttle node",
            NumberValue(default=1, unit="", ndecimals=0, scale=1, step=1),
        )

        self.swap_flag = self.get_argument(
            "swap flag",
            artiq_env.BooleanValue(default=False)
        )

        self.doppler_switch = self.get_argument(
            "Doppler cooling switch",
            artiq_env.BooleanValue(default=False)
        )

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

        # scan arguments
        self.num_steps = self.get_argument(
            "Stepnum",
            NumberValue(default=20, unit="", ndecimals=0, scale=1, step=1)
        )

    def prepare(self):

        super(Reordering_Rate, self).prepare()

        self.shuttle_from_default = self.sandia_box.get_shuttle_data('Start', f'{self.ion_swap_index}-0')
        if self.swap_flag:
            self.shuttle_to = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 0, self.ion_swap_index, 4)
            self.shuttle_from = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 3, self.ion_swap_index, 0)
        else:
            self.shuttle_to = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 0, self.ion_swap_index, self.shuttle_node)
            self.shuttle_from = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, self.shuttle_node, self.ion_swap_index, 0)
        self.shuttle_to_default = self.sandia_box.get_shuttle_data(f'{self.ion_swap_index}-0', 'Start')

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        self.scan_values = [i for i in range(self.num_steps)]

        self.rf_ramp.prepare()

    @kernel
    def custom_kn_init(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_kn_idle(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

        self.core.break_realtime()
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.core.break_realtime()
        self.doppler_cooling_coolant.cool(True)

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        # Shuttle
        # Need to make sure the check is at tweaked default line
        if self.shuttle_flag:
            if self.doppler_switch:
                self.doppler_cooling.off()
                self.doppler_cooling_coolant.idle()
                delay(10 * us)
            self.sandia_box.shuttle_path(self.shuttle_from_default)
            delay(50 * us)
            self.core.break_realtime()
            self.sandia_box.shuttle_path(self.shuttle_to)
            delay(50 * us)
            self.core.break_realtime()
            self.sandia_box.shuttle_path(self.shuttle_from)
            delay(50 * us)
            self.core.break_realtime()
            self.sandia_box.shuttle_path(self.shuttle_to_default)
            delay(50 * us)
            self.core.break_realtime()
            if self.doppler_switch:
                self.doppler_cooling.on()
                self.doppler_cooling_coolant.init()
                delay(10 * us)
        else:
            if self.doppler_switch:
                self.doppler_cooling.off()
                self.doppler_cooling_coolant.idle()
                delay(350 * ms)
                self.doppler_cooling.on()
                self.doppler_cooling_coolant.init()
                delay(10 * us)
            else:
                delay(200 * us)
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

    def analyze(self):
        #super().analyze()
        pass

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

class Sort_Test(BasicEnvironment, artiq_env.Experiment):
    """Sort.Test"""
    # This experiment is for executing the complete sorting solution without PMT checking in the middle
    data_folder = "Sort.Test"
    applet_name = "Sort.Test"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

    split_path = []
    swap_path = []
    merge_path = []

    def build(self):
        """Initialize experiment & variables."""

        super(Sort_Test, self).build()

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

        # Maximum trial times to get a sorted chain: num_shots * num_steps
        self.num_shots = 10
        self.num_steps = 10

        # Sorting
        # Desired configuration of the chain
        self.DC = np.array([2,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,2,1,1,1,2])
        # self.DC = [1,1,1,1,2,1,1,1,2,1,2,1,1,1,1,1,1,1,2,1,1,1,2]
        # Target configuration by executing one step of sorting solution
        self.TC = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        # Sorting attempts number
        self.n_sort = 0

    def prepare(self):

        super(Sort_Test, self).prepare()

        for ionind in range(self.num_pmts - 1):
            self.split_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 0, ionind, 3))
            self.swap_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 4))
            self.merge_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 0))
        self.scan_values = [i for i in range(self.num_steps)]
        self.rf_ramp.prepare()

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
    def main_experiment(self, istep, ishot):
        # Counts and config are generated (not saved) in previous parts of experiment loop
        sort_flag = True
        for i in range(len(self.config)):
            if self.config[i] != self.DC[i]:
                sort_flag = False
                break
        print("sort flag", sort_flag)

        # Feedback on latest sorting result
        if (self.TC == self.config):
            print("Successful sort!")
        else:
            print("Failed sort!")
        self.TC = self.config

        if (not sort_flag):
            # Sort
            if (self.coolant_countnum == self.n_coolants) and (self.qubit_countnum == self.n_qubits):
                if self.np_presence_coolant == self.np_presence_qubit:
                    # Generate sorting solution
                    sorting_sol = [-1] * 90
                    sorting_sol = self.sorting_sol_generator(self.DC, self.config, sorting_sol)
                    # Execute sorting (without checking config in between)
                    for i in range(len(sorting_sol)):
                        if sorting_sol[i] >= 0:
                            # Check RTIO
                            self.split(sorting_sol[i])
                            self.swap(sorting_sol[i])
                            self.merge(sorting_sol[i])
                            self.n_sort += 1
                            self.TC[i], self.TC[i+1] = self.TC[i+1], self.TC[i]
                    print("Current Config: ", self.config)
                else:
                    print("Detected config based on coolants and qubits don't agree. ")
            else:
                print("Detected incorrect number:")
                print("Detected coolant number:", self.coolant_countnum)
                print("Detected qubit number: ", self.qubit_countnum)
        else:
            print("Sorted!")
        print("Current Config: ", self.config)

        self.save_sort_sol(self.TC, ishot, istep)

    @kernel
    def custom_kn_init(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_kn_idle(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

        # Doppler cooling is default on to continue to cool the ions after loading
        self.core.break_realtime()
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling_coolant.cool(True)

        # Sorting stat check
        print("Total number of sorting attempts: ", self.n_sort)

    @kernel
    def split(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.split_path[ind])

    @kernel
    def swap(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.swap_path[ind])

    @kernel
    def merge(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.merge_path[ind])

    def analyze(self):
        #super().analyze()
        pass

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

class Sort_Inspect(BasicEnvironment, artiq_env.Experiment):
    """Sort.Inspect"""
    # For check sorting by inspecting the chain config right after each sort attempt

    data_folder = "Sort.Inspect"
    applet_name = "Sort.Inspect"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

    split_path = []
    swap_path = []
    merge_path = []

    def build(self):
        """Initialize experiment & variables."""

        super(Sort_Inspect, self).build()

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

    def prepare(self):

        super(Sort_Inspect, self).prepare()

        for ionind in range(self.num_pmts - 1):
            self.split_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 0, ionind, 3))
            self.swap_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 4))
            self.merge_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 0))



        self.rf_ramp.prepare()

        # Maximum trial times to get a sorted chain: num_shots * num_steps
        self.num_shots = 10
        self.num_steps = 10
        self.scan_values = [i for i in range(self.num_steps)]

        # Sorting
        # Desired configuration of the chain
        self.DC = np.array([2,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,2,1,1,1,2])
        np.random.shuffle(self.DC)
        # self.DC = np.array([1,1,1,1,2,1,1,1,2,1,2,1,1,1,1,1,1,1,2,1,1,1,2])
        # Target configuration by executing one step of sorting solution
        self.TC = np.array([np.int64(1) for i in range(self.num_pmts)])

        # Sorting total attempts number
        self.n_sort = 0
        # Sorting executed attempts number
        self.n_sort_exe = 0
        # Invalid detection number
        self.n_inv_det = 0
        # Nominal sort number
        self.n_sort_nominal = 0
        # Flag variable to indicate whether this sort solution is the 1st one
        self.nominal_flag = 1
        # Sorting config error array
        self.EC = np.array([np.int64(0) for i in range(self.num_pmts)])

        # Timing
        self.start_time = time.time()
        self.end_time = 0
        self.duration_time = 0

    @host_only
    def custom_experiment_initialize(self):
        self.set_experiment_data(
            "EC",
            np.full((self.num_pmts, self.num_steps, self.num_shots), np.nan),
            broadcast=True
        )
        self.set_experiment_data(
            "TC",
            np.full((self.num_pmts, self.num_steps, self.num_shots), np.nan),
            broadcast=True
        )

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
    def main_experiment(self, istep, ishot):
        sort_flag = True
        det_valid = True
        self.EC = np.array([np.int64(0) for i in range(self.num_pmts)])

        for i in range(len(self.config)):
            if self.config[i] != self.DC[i]:
                sort_flag = False
                break

        # Feedback on previous step of sorting
        # if (self.TC == self.config):
        #     print("Successful sort!")
        # else:
        #     print("Failed sort!")

        # Check validity of detection
        if (self.coolant_countnum != self.n_coolants) or (self.qubit_countnum != self.n_qubits):
            det_valid = False

        # Update the EC
        if (sort_flag):
            # Sorted: err array = 0
            pass
        elif (not det_valid):
            # Invalid detection, set error array to be a constant array of -2
            self.EC = np.array([np.int64(-2) for i in range(self.num_pmts)])
        else:
            if self.n_sort > 0:
                # Valid detection, not sorted
                # Don't count the 1st step
                self.EC = np.array([np.int64(self.config[i] - self.TC[i]) for i in range(self.num_pmts)])

        # Update TC to be current config, to be updated into the intended sorted config at this step
        for i in range(len(self.TC)):
            self.TC[i] = self.config[i]

        if not sort_flag:
            if det_valid:
                self.reorder_flag = 0
                # Generate sorting solution
                sorting_sol = [-1] * 90
                sorting_sol = self.sorting_sol_generator(self.DC, self.config, sorting_sol)
                if self.nominal_flag == 1:
                    # Save the nominal number of sorts needed
                    for i in range(len(sorting_sol)):
                        if sorting_sol[i] != -1:
                            self.n_sort_nominal += 1
                        else:
                            self.nominal_flag = 0
                            break
                # Execute the 1st step of sorting (without checking config in between)
                for i in range(len(sorting_sol)):
                    if sorting_sol[i] >= 0:
                        # print("Start: ", now_mu())
                        # Check RTIO
                        self.split(sorting_sol[i])
                        self.swap(sorting_sol[i])
                        self.merge(sorting_sol[i])
                        # print("Sort")
                        # print("End: ", now_mu())
                        # Update TC into the intended sorted config
                        # Tested
                        self.TC[sorting_sol[i]], self.TC[sorting_sol[i]+1] = self.TC[sorting_sol[i]+1], self.TC[sorting_sol[i]]
                        self.n_sort_exe += 1
                        self.n_sort += 1
                        break
            else:
                # Invalid detect
                self.n_inv_det += 1
                self.n_sort += 1
        else:

            print("Sorted!")

        self.custom_save_data(ishot, istep, self.TC, self.EC)

    @rpc(flags={"async"})
    def custom_save_data(
        self,
        ishot: TInt32,
        istep: TInt32,
        TC: TList(TInt32),
        EC: TList(TInt32)
        ):
        np_TC = np.array(TC, ndmin=3).T
        np_EC = np.array(EC, ndmin=3).T

        self.mutate_experiment_data(
            "TC",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps), (ishot, None, self.num_shots)),
            np_TC,
        )

        self.mutate_experiment_data(
            "EC",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps), (ishot, None, self.num_shots)),
            np_EC,
        )


    @kernel
    def custom_kn_init(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_kn_idle(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

        # Doppler cooling is default on to continue to cool the ions after loading
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling_coolant.cool(True)

    @kernel
    def split(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.split_path[ind])

    @kernel
    def swap(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.swap_path[ind])

    @kernel
    def merge(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.merge_path[ind])

    def analyze(self):
        #super().analyze()
        # Timing update
        self.end_time = time.time()
        self.duration_time = self.end_time - self.start_time
        # Update dataset
        self.set_experiment_data("total number of sort attempts", self.n_sort, broadcast=True)
        self.set_experiment_data("total number of executed sort", self.n_sort_exe, broadcast=True)
        self.set_experiment_data("total number of invalid detection", self.n_inv_det, broadcast=True)
        self.set_experiment_data("total number of sort attempts by nominal solution", self.n_sort_nominal, broadcast=True)
        self.set_experiment_data("DC", self.DC, broadcast=True)
        self.set_experiment_data(
            "time log",
            np.array([self.start_time, self.end_time, self.duration_time]),
            broadcast=True
        )
        print(self.DC)

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

class Sort_Inspect_Stat(BasicEnvironment, artiq_env.Experiment):
    """Sort.Inspect.Stat"""
    # For check sorting by inspecting the chain config right after each sort attempt and generating sorting solution then sort
    # Statistics check: sort the ion in between 2 specified configurations
    data_folder = "Sort.Inspect.Stat"
    applet_name = "Sort.Inspect.Stat"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

    split_path = []
    swap_path = []
    merge_path = []

    def build(self):
        """Initialize experiment & variables."""

        super(Sort_Inspect_Stat, self).build()

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

    def prepare(self):

        super(Sort_Inspect_Stat, self).prepare()

        for ionind in range(self.num_pmts - 1):
            self.split_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 0, ionind, 3))
            self.swap_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 4))
            self.merge_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 0))

        self.rf_ramp.prepare()

        # Sorting
        # Half the number of total test number
        self.test_num = np.int64(5)
        # Test ind
        self.test_ind = np.int64(0)

        self.num_shots = 1
        # 2 more attempts for nominal attempts
        self.num_steps = self.test_num * 30
        self.scan_values = [i for i in range(self.num_steps)]

        # Shuffled desired configuration list
        self.config_unit = np.array([2,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,2,1,1,1,2])
        self.DC_list = np.zeros((self.test_num, self.num_pmts), np.int64)
        for i in range(self.test_num):
            np.random.shuffle(self.config_unit)
            for j in range(len(self.config_unit)):
                self.DC_list[i][j] = self.config_unit[j]
        # Desired configuration of the chain
        self.DC = self.DC_list[self.test_ind]
        # Target configuration by executing one step of sorting solution
        self.TC = np.array([np.int64(1) for i in range(self.num_pmts)])
        # Sorting config error array
        self.EC = np.array([np.int64(0) for i in range(self.num_pmts)])

        # Sorting total attempts number (accumulate across different testind)
        self.n_sort = np.int64(0)
        self.n_sort_list = np.array([np.int64(0) for i in range(self.test_num)])
        # Sorting executed attempts number (accumulate across different testind)
        self.n_sort_exe = np.int64(0)
        self.n_sort_exe_list = np.array([np.int64(0) for i in range(self.test_num)])
        # Invalid detection number (accumulate across different testind)
        self.n_inv_det = np.int64(0)
        self.n_inv_det_list = np.array([np.int64(0) for i in range(self.test_num)])
        # Nominal sort number
        self.n_sort_nominal = np.int64(0)
        self.n_sort_nominal_list = np.array([np.int64(0) for i in range(self.test_num)])
        # Flag variable to indicate whether this sort solution is the 1st one
        self.nominal_flag = np.int64(1)
        # Nominal sorting solution
        self.nominal_sort_sol = np.reshape(np.array([-1 for i in range(self.test_num * 90)]), (self.test_num, 90))
        # Config at deciding nominal sorting solution
        self.config_nominal_list = np.reshape(np.array([-1 for i in range(self.test_num * self.num_pmts)]), (self.test_num, self.num_pmts))

        # Timing
        self.start_time = 0.1
        self.duration = 0.1
        self.duration_list = np.array([0.1 for i in range(self.test_num)])

    @host_only
    def custom_experiment_initialize(self):
        self.set_experiment_data(
            "EC",
            np.full((self.num_pmts, self.num_steps, self.num_shots), np.nan),
            broadcast=True
        )
        self.set_experiment_data(
            "TC",
            np.full((self.num_pmts, self.num_steps, self.num_shots), np.nan),
            broadcast=True
        )
        self.set_experiment_data(
            "DC",
            np.full((self.num_pmts, self.num_steps, self.num_shots), np.nan),
        )

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
    def main_experiment(self, istep, ishot):
        if self.test_ind < self.test_num:
            # Sort
            # Initialize flag variable
            sort_flag = True
            det_valid = True
            self.EC = np.array([np.int64(0) for i in range(self.num_pmts)])
            for i in range(len(self.config)):
                if self.config[i] != self.DC[i]:
                    sort_flag = False
                    break

            # Check validity of detection
            coolant_countnum = 0
            qubit_countnum = 0
            for i in range(len(self.config)):
                if self.config[i] == 2:
                    coolant_countnum += 1
                else:
                    qubit_countnum += 1
            if (coolant_countnum != self.n_coolants) or (qubit_countnum != self.n_qubits):
                det_valid = False

            # Update the EC
            if (sort_flag):
                # Sorted: err array = 0
                pass
            elif (not det_valid):
                # Invalid detection, set error array to be a constant array of -2
                self.EC = np.array([np.int64(-2) for i in range(self.num_pmts)])
            else:
                if self.n_sort > 0:
                    # Valid detection, not sorted
                    # Don't count the 1st step
                    self.EC = np.array([np.int64(self.config[i] - self.TC[i]) for i in range(self.num_pmts)])

            # Update TC to be current config, to be updated into the intended sorted config at this step
            for i in range(len(self.TC)):
                self.TC[i] = self.config[i]
            # Not sorted and valid detect: execute sort
            # Not sorted and invalid detect: no sort, cost 1 sort attempt, add 1 invalid detect event
            # Sorted:
            # (1) Increase testind
            # (2) Update DC
            # (3) Save testind-indexed data
            # (4) Zero testind-local variables
            if not sort_flag:
                if det_valid:
                    # Generate sorting solution
                    sorting_sol = [-1] * 90
                    sorting_sol = self.sorting_sol_generator(self.DC, self.config, sorting_sol)
                    if self.nominal_flag == 1:
                        # Save the nominal number of sorts needed
                        for i in range(len(sorting_sol)):
                            if sorting_sol[i] != -1:
                                self.n_sort_nominal += 1
                                # First step of sort, start counting time
                                self.start_time = float(self.core.mu_to_seconds(now_mu()))
                                self.nominal_sort_sol[self.test_ind][i] = np.int64(sorting_sol[i])
                            else:
                                self.nominal_flag = np.int64(0)
                                break
                        # Update nominal config list
                        for i in range(len(self.config)):
                            self.config_nominal_list[self.test_ind][i] = self.config[i]
                    # Execute the 1st step of sorting (without checking config in between)
                    for i in range(len(sorting_sol)):
                        if sorting_sol[i] >= 0:
                            # print("Start: ", now_mu())
                            # Check RTIO
                            # self.split(sorting_sol[i])
                            self.swap(sorting_sol[i])
                            # self.merge(sorting_sol[i])
                            # print("Sort")
                            # print("End: ", now_mu())
                            # Update TC into the intended sorted config
                            self.TC[i], self.TC[i+1] = self.TC[i+1], self.TC[i]
                            self.n_sort_exe += 1
                            self.n_sort += 1
                            break
                else:
                    # Invalid detect
                    self.n_inv_det += 1
                    self.n_sort += 1
            else:
                # End the time
                self.duration = float(self.core.mu_to_seconds(now_mu())) - self.start_time

                # Save data
                self.n_sort_list[self.test_ind] = self.n_sort
                self.n_sort_exe_list[self.test_ind] = self.n_sort_exe
                self.n_inv_det_list[self.test_ind] = self.n_inv_det
                self.n_sort_nominal_list[self.test_ind] = self.n_sort_nominal
                self.duration_list[self.test_ind] = self.duration

                # Update(zero) local variable
                self.start_time = 0.1
                self.duration = 0.1
                self.n_sort = np.int64(0)
                self.n_sort_exe = np.int64(0)
                self.n_inv_det = np.int64(0)
                self.n_sort_nominal = np.int64(0)
                self.nominal_flag = np.int64(1)
                self.TC = np.array([np.int64(1) for i in range(self.num_pmts)])

                # Update testind
                self.test_ind += 1

                # Update DC
                if self.test_ind < self.test_num:
                    self.DC = self.DC_list[self.test_ind]

            # if self.n_inv_det >= 20:
            #     # Probably due to ion loss, request termination
            #     self.scheduler.request_termination(self._RID)

            self.custom_save_data(ishot, istep, self.TC, self.EC)

        else:
            self.custom_save_data(ishot, istep, [-2]*self.num_pmts, [-2]*self.num_pmts)
            print("Done")

    @rpc(flags={"async"})
    def custom_save_data(
        self,
        ishot,
        istep,
        TC,
        EC
        ):
        np_TC = np.array(TC, ndmin=3).T
        np_EC = np.array(EC, ndmin=3).T

        self.mutate_experiment_data(
            "TC",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps), (ishot, None, self.num_shots)),
            np_TC,
        )

        self.mutate_experiment_data(
            "EC",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps), (ishot, None, self.num_shots)),
            np_EC,
        )

    @kernel
    def custom_kn_init(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_kn_idle(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

        # Doppler cooling is default on to continue to cool the ions after loading
        self.core.break_realtime()
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling_coolant.cool(True)

    @kernel
    def split(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.split_path[ind])

    @kernel
    def swap(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.sandia_box._to_swap_data[ind])
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.sandia_box._from_swap_data[ind])

    @kernel
    def merge(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.merge_path[ind])


    def analyze(self):
        super().analyze()
        # Sorting stat check
        self.set_experiment_data("total number of sort attempts", self.n_sort_list, broadcast=True)
        self.set_experiment_data("total number of executed sort", self.n_sort_exe_list, broadcast=True)
        self.set_experiment_data("total number of invalid detection", self.n_inv_det_list, broadcast=True)
        self.set_experiment_data("total number of sort attempts by nominal solution", self.n_sort_nominal_list, broadcast=True)
        self.set_experiment_data("duration", self.duration_list, broadcast=True)
        self.set_experiment_data("DC", self.DC_list, broadcast=True)
        self.set_experiment_data("Nominal sort sol", self.nominal_sort_sol, broadcast=True)
        self.set_experiment_data("Config at nominal", self.config_nominal_list, broadcast=True)

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

class Sort_Fix_Sol(BasicEnvironment, artiq_env.Experiment):
    """Sort.Fixsol"""
    # This is an experiment to test sorting with fixed sorting solution, the final configuration is compared with expected configuration

    data_folder = "Sort.Fixsol"
    applet_name = "Sort.Fixsol"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

    split_path = []
    swap_path = []
    merge_path = []

    def build(self):
        """Initialize experiment & variables."""

        super(Sort_Fix_Sol, self).build()

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

        # Maximum trial times to get a sorted chain: num_shots * num_steps
        self.num_shots = 1

        # Sorting
        # self.sort_sol = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.sort_sol = np.array(np.linspace(0, 21, 22), int)
        np.random.shuffle(self.sort_sol)
        self.num_steps = len(self.sort_sol) + 2
        # Sorting attempts number
        self.n_sort = 0

    def prepare(self):

        super(Sort_Fix_Sol, self).prepare()

        self.scan_values = [i for i in range(self.num_steps)]

        for ionind in range(self.num_pmts - 1):
            self.split_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 0, ionind, 3))
            self.swap_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 4))
            self.merge_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 0))

        self.rf_ramp.prepare()

        self.TC = np.array([np.int64(-1) for i in range(self.num_pmts)])
        self.CC = np.array([np.int64(-1) for i in range(self.num_pmts)])
        self.FC = np.array([np.int64(-1) for i in range(self.num_pmts)])

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
    def main_experiment(self, istep, ishot):
        if (istep == 0):
            # Intial check and fix TC (target configuration)
            _LOGGER.info(self.sort_sol)
            for i in range(len(self.config)):
                # If directly using self.CC = self.config, self.CC would be updated into the final self.config
                self.CC[i] = self.config[i]
                self.TC[i] = self.config[i]
            for i in range(len(self.sort_sol)):
                temp1, temp2 = self.TC[self.sort_sol[i]], self.TC[self.sort_sol[i] + 1]
                self.TC[self.sort_sol[i] + 1] = temp1
                self.TC[self.sort_sol[i]] = temp2
                # self.TC[self.sort_sol[i]], self.TC[self.sort_sol[i] + 1] = self.TC[self.sort_sol[i] + 1], self.TC[self.sort_sol[i]]

        elif istep > 0 and istep <= self.num_steps - 2:
            # print("Start: ", self.core.mu_to_seconds(now_mu()))
            # Check RTIO
            self.split(self.sort_sol[istep - 1])
            self.swap(self.sort_sol[istep - 1])
            self.merge(self.sort_sol[istep - 1])
            # print("Sort")
            # print("End: ", self.core.mu_to_seconds(now_mu()))
            self.n_sort += 1
        else:
            self.FC = self.config
        self.core.break_realtime()

    @kernel
    def custom_kn_init(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_kn_idle(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)


        # Doppler cooling is default on to continue to cool the ions after loading
        self.core.break_realtime()
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling_coolant.cool(True)


    @kernel
    def split(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.split_path[ind])

    @kernel
    def swap(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.swap_path[ind])

    @kernel
    def merge(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.merge_path[ind])

    def analyze(self):
        # Sorting stat check

        filename = "/media/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/2024-01-10/config_" + str(self._RID) + ".txt"
        with open(filename, 'w') as outfile:
            np.savetxt(outfile, (self.CC, self.TC, self.FC), fmt='%d')

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

class Sort_Fix_Sol_Stat(BasicEnvironment, artiq_env.Experiment):
    """Sort.Fixsol.Stat"""
    # This is an experiment to test sorting with fixed sorting solution many times and collecting the statistics.

    data_folder = "Sort.Fixsol.Stat"
    applet_name = "Sort.Fixsol.Stat"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

    split_path = []
    swap_path = []
    merge_path = []

    def build(self):
        """Initialize experiment & variables."""

        super(Sort_Fix_Sol_Stat, self).build()

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

        # Maximum trial times to get a sorted chain: num_shots * num_steps
        self.num_shots = 1

        # Sorting
        self.test_num = 50
        self.sort_sol_unit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.sort_sol_test = np.zeros(self.test_num * len(self.sort_sol_unit), np.int64).reshape((self.test_num, len(self.sort_sol_unit)))
        self.sort_sol = np.zeros(len(self.sort_sol_unit), np.int64)
        self.num_steps = (len(self.sort_sol) + 2) * self.test_num
        # Sorting attempts number


    def prepare(self):

        super(Sort_Fix_Sol_Stat, self).prepare()

        self.scan_values = [i for i in range(self.num_steps)]

        for ionind in range(self.num_pmts - 1):
            self.split_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 0, ionind, 3))
            self.swap_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 4))
            self.merge_path.append(self.sandia_box.get_shuttle_data_stepnum(ionind, 3, ionind, 0))

        self.rf_ramp.prepare()

        # Single-sort configuration
        self.TC = np.array([np.int64(-1) for i in range(self.num_pmts)])
        self.CC = np.array([np.int64(-1) for i in range(self.num_pmts)])
        self.FC = np.array([np.int64(-1) for i in range(self.num_pmts)])

        # Collecting sorting statistics
        for i in range(self.test_num):
            np.random.shuffle(self.sort_sol_unit)
            for j in range(len(self.sort_sol_unit)):
                self.sort_sol_test[i][j] = self.sort_sol_unit[j]
        self.sort_ind = np.int64(0) # 0-indexed
        self.success_num = 0
        self.failed_detect_num = 0
        self.TC_list = np.array([np.int64(-1) for i in range(self.num_pmts * self.test_num)]).reshape((self.test_num, self.num_pmts))
        self.CC_list = np.array([np.int64(-1) for i in range(self.num_pmts * self.test_num)]).reshape((self.test_num, self.num_pmts))
        self.FC_list = np.array([np.int64(-1) for i in range(self.num_pmts * self.test_num)]).reshape((self.test_num, self.num_pmts))

        print("Start", time.time())
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
    def main_experiment(self, istep, ishot):
        if ((istep + 1) % (len(self.sort_sol) + 2) == 1):
            print("Sort", self.sort_ind)
            for i in range(len(self.sort_sol_unit)):
                self.sort_sol[i] = self.sort_sol_test[self.sort_ind][i]
            # Intial check and fix TC (target configuration)
            for i in range(len(self.config)):
                # If directly using self.CC = self.config, self.CC would be updated into the final self.config
                self.CC[i] = self.config[i]
                self.TC[i] = self.config[i]
            for i in range(len(self.sort_sol)):
                temp1, temp2 = self.TC[self.sort_sol[i]], self.TC[self.sort_sol[i] + 1]
                self.TC[self.sort_sol[i] + 1] = temp1
                self.TC[self.sort_sol[i]] = temp2
            for i in range(len(self.config)):
                self.CC_list[self.sort_ind][i] = self.CC[i]
                self.TC_list[self.sort_ind][i] = self.TC[i]

        elif ((istep + 1) % (len(self.sort_sol) + 2)  == 0):
            for i in range(len(self.config)):
                self.FC[i] = self.config[i]
            for i in range(len(self.config)):
                self.FC_list[self.sort_ind][i] = self.FC[i]
            self.sort_ind += 1

        else:
            # print("Sort: ", self.sort_sol[istep % (len(self.sort_sol) + 2) - 1])
            self.split(self.sort_sol[istep % (len(self.sort_sol) + 2) - 1])
            self.swap(self.sort_sol[istep % (len(self.sort_sol) + 2) - 1])
            self.merge(self.sort_sol[istep % (len(self.sort_sol) + 2) - 1])

        self.core.break_realtime()

    @kernel
    def custom_kn_init(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_kn_idle(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

        # Doppler cooling is default on to continue to cool the ions after loading
        self.core.break_realtime()
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling_coolant.cool(True)


    @kernel
    def split(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.split_path[ind])

    @kernel
    def swap(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.swap_path[ind])

    @kernel
    def merge(self, ind):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.merge_path[ind])

    def analyze(self):
        # Sorting stat check
        print("End", time.time())
        CCfilename = "/media/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/2024-02-01/config_" + str(self._RID) + "_CC.txt"
        with open(CCfilename, 'w') as outfile:
            np.savetxt(outfile, self.CC_list, fmt='%d')
        TCfilename = "/media/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/2024-02-01/config_" + str(self._RID) + "_TC.txt"
        with open(TCfilename, 'w') as outfile:
            np.savetxt(outfile, self.TC_list, fmt='%d')
        FCfilename = "/media/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/2024-02-01/config_" + str(self._RID) + "_FC.txt"
        with open(FCfilename, 'w') as outfile:
            np.savetxt(outfile, self.FC_list, fmt='%d')

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

class Voltage_Check_Sync(BasicEnvironment, artiq_env.Experiment):
    """Voltage.Check.Sync"""

    data_folder = "Voltage.Check.Sync"
    applet_name = "Voltage.Check.Sync"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

    def build(self):
        super(Voltage_Check_Sync, self).build()

        self.setattr_device("awg_trigger")

        self.ion_swap_index = self.get_argument(
            "ion to swap",
            NumberValue(default=0, unit="", ndecimals=0, scale=1, step=1),
        )

        self.final_node = self.get_argument(
            "Final Node",
            NumberValue(default=3, unit="", ndecimals=0, scale=1, step=1),
        )

        self.hold_time_split_ms = self.get_argument(
            "Hold Time Split (ms)",
            NumberValue(default=50*ms, unit="ms", ndecimals=3),
        )

        self.hold_time_home_ms = self.get_argument(
            "Hold Time Home (ms)",
            NumberValue(default=500 * ms, unit="ms", ndecimals=3),
        )

        self.N_repeats = self.get_argument(
            "Number of Repeats",
            NumberValue(default=10, unit="", ndecimals=0, scale=1, step=1),
        )

        self.trigger_at_swap = self.get_argument(
            "Trigger at Swap",
            artiq_env.BooleanValue(default=True),
        )

        self.trigger_at_home = self.get_argument(
            "Trigger at Home",
            artiq_env.BooleanValue(default=True),
        )
        self.rf_ramp = rampcontrol_auto(self)
        # Use AWG trigger for camera
        self.setattr_device("awg_trigger")
        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

    def prepare(self):
        super(Voltage_Check_Sync, self).prepare()
        self.rf_ramp.prepare()

        self.num_shots = 1
        self.num_steps = self.N_repeats
        self.scan_values = [i for i in range(self.num_steps)]

        if(self.final_node == 5):
            self.shuttle_to = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 0, self.ion_swap_index, 3)
            self.shuttle_swap = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 3, self.ion_swap_index, 4)
            self.shuttle_from = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 3, self.ion_swap_index, 0)
        else:
            self.shuttle_to = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, 0, self.ion_swap_index, self.final_node)
            self.shuttle_swap = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, self.final_node, self.ion_swap_index, self.final_node)
            self.shuttle_from = self.sandia_box.get_shuttle_data_stepnum(self.ion_swap_index, self.final_node, self.ion_swap_index, 0)

        _LOGGER.warning(f"Shuttle Time (ms): {1000 * self.core.mu_to_seconds(np.sum(self.shuttle_to.timings))}")

    @kernel
    def main_experiment(self, istep, ishot):
        self.core.break_realtime()
        self.sandia_box.shuttle_path(self.shuttle_to)
        self.core.break_realtime()
        self.doppler_cooling.on()
        delay(100 * us)
        # For checking swap waveform
        if self.trigger_at_swap and self.final_node == 5:
            self.trigger_camera()
            delay(self.hold_time_split_ms)

        self.sandia_box.shuttle_path(self.shuttle_swap)

        if self.trigger_at_swap:
            self.trigger_camera()

        delay(self.hold_time_split_ms)

        self.sandia_box.shuttle_path(self.shuttle_from)

        if self.trigger_at_home:
            self.trigger_camera()

        delay(self.hold_time_home_ms)

        _LOGGER.info("Shuttled %d", istep)
        self.core.wait_until_mu(now_mu())

    @kernel
    def custom_kn_init(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_kn_idle(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

        # Doppler cooling is default on to continue to cool the ions after loading
        self.core.break_realtime()
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling_coolant.cool(True)

    @kernel
    def trigger_camera(self):
        # Camera external trigger tied to AWG
        self.core.break_realtime()
        self.awg_trigger.pulse(1 * ms)

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

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

    def analyze(self):
        #super().analyze()
        pass

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

class Voltage_Check_Async(BasicEnvironment, artiq_env.Experiment):
    """Voltage.Check.Async"""

    data_folder = "Voltage.Check.Async"
    applet_name = "Voltage.Check.Async"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Population Transfer"
    xlabel = "Rabi Pulse Length (us)"

    def build(self):
        super(Voltage_Check_Async, self).build()

        self.setattr_device("awg_trigger")

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

        self.rf_ramp = rampcontrol_auto(self)
        # Use AWG trigger for camera
        self.setattr_device("awg_trigger")
        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

    def prepare(self):
        super(Voltage_Check_Async, self).prepare()
        self.rf_ramp.prepare()

        self.num_shots = 1
        self.num_steps = self.N_repeats
        self.scan_values = [i for i in range(self.num_steps)]

    @rpc
    def prepare_step(self, istep):
        for i in range(self.start_linenum, self.stop_linenum+1):
            time.sleep(self.delay_time_ms)
            self.sandia_box.dac_pc.apply_line_async(i, line_gain=1, global_gain=1)
            _LOGGER.info((istep, i))
        time.sleep(.05)
        self.trigger_camera()
        time.sleep(.5)
        for i in range(self.start_linenum, self.stop_linenum+1, -1):
            time.sleep(self.delay_time_ms)
            self.sandia_box.dac_pc.apply_line_async(i, line_gain=1, global_gain=1)
            _LOGGER.info((istep, i))

        self.trigger_camera()
        time.sleep(2)

    @kernel
    def custom_kn_init(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

    @kernel
    def custom_kn_idle(self):
        if self.ramp_rf:
            self.core.break_realtime()
            self.rf_ramp.run_ramp_up_kernel()
            self.rf_ramp.deactivate_ext_mod()
            # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
            self.core.break_realtime()
            self.rf_lock_control(False)

        # Doppler cooling is default on to continue to cool the ions after loading
        self.core.break_realtime()
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling_coolant.cool(True)

    @kernel
    def trigger_camera(self):
        # Camera external trigger tied to AWG
        self.core.break_realtime()
        self.awg_trigger.pulse(1 * ms)

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

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

    def analyze(self):
        #super().analyze()
        pass

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

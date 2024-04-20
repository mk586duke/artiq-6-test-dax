import numpy as np
import qiskit.pulse as qp
import logging
import datetime
import time
import typing
from collections import defaultdict
import copy

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
from artiq.experiment import NumberValue, StringValue, BooleanValue
from artiq.language.core import (
    delay,
    delay_mu,
    host_only,
    kernel,
    rpc,
    TerminationRequested,
)
from artiq.language.units import MHz, kHz, ms, us
from artiq.language.types import TInt32, TBool
import oitg.fitting as fit
from oitg.fitting.FitBase import FitError

import euriqafrontend.settings as settings
from euriqafrontend.modules.calibration import Submitter
from euriqafrontend.scheduler import ExperimentPriorities as priorities
from euriqafrontend.repository.basic_environment import BasicEnvironment
from euriqafrontend.repository.basic_env import BasicEnvironment as BasicEnvironment_Sort
# from euriqafrontend.repository.basic_environment_sort import BasicEnvironment as BasicEnvironment_Sort
import euriqafrontend.fitting as umd_fit
import euriqafrontend.modules.rfsoc as rfsoc
import euriqabackend.waveforms.single_qubit as single_qubit
from euriqafrontend.modules.calibration import log_calibration
from euriqafrontend.modules.artiq_dac import RampControl as rampcontrol_auto

_LOGGER = logging.getLogger(__name__)
_AUTO_CALIBRATION_SETTINGS = settings.auto_calibration
CHAIN_X2_DATASET = "monitor.chain_X2"
CHAIN_X4_DATASET = "monitor.chain_X4"
QUBIT_NUM_DATASET = "global.AWG.N_qubits"
COOLANT_NUM_DATASET = "global.AWG.N_coolants"
CENTER_CHECK_THRESH_DATASET = "global.Voltages.Center_Check_Thresh"

class Sort_Calibration(artiq_env.EnvExperiment):
    """Auto.Calibration.Sort"""

    def build(self):
        """Initialize experiment & variables."""

        self.do_load_mixed_23 = self.get_argument("Load 23 mixed-species chain", BooleanValue(default=True))

        self.check_center_mixed_23 = self.get_argument(
            "Check_center_mixed_23_ions", BooleanValue(default=True)
        )

        self.check_center_mixed_23_lowRF = self.get_argument(
            "Check_center_mixed_23_ions_lowRF", BooleanValue(default=True)
        )

        self.edge_X4_check = self.get_argument(
            "edge_X4_check", BooleanValue(default=True)
        )

        self.swap_check = self.get_argument(
            "swap_check", BooleanValue(default=True)
        )

        self.setattr_device("scheduler")
        self.submitter = Submitter(self.scheduler,__file__)

    @host_only
    def run(self):
        """Run each enabled calibration step."""

        if self.do_load_mixed_23:
            chain_X2 = self.get_dataset(CHAIN_X2_DATASET)
            self.set_dataset("global.Voltages.X2", chain_X2, persist=True)
            self.submitter.submit(
                "AutoloadMixedArbSeq",
                priority=priorities.CALIBRATION_CRITICAL,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["Autoload_mixed_23"].to_dict())
            )

        if self.check_center_mixed_23:
            self.submitter.submit(
                "Calibrate_CheckCenter_mixed23",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["CheckCentermixed23"].to_dict())
            )

        if self.check_center_mixed_23_lowRF:
            self.submitter.submit(
                "Calibrate_CheckCenter_mixed23_lowRF",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["CheckCentermixed23_lowRF"].to_dict())
            )

        if self.swap_check:
            self.submitter.submit(
                "Swap_Check",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["Swap_Check"].to_dict())
            )

        # if self.edge_X4_check:
        #     self.submitter.submit(
        #         "Edge_X4_Calibration",
        #         repetition=0,
        #         hold=True,
        #         **(_AUTO_CALIBRATION_SETTINGS["Edge_X4_calibration"].to_dict())
        #     )

class Calibrate_CheckCenter_mixed23(BasicEnvironment, artiq_env.EnvExperiment):
    """Calibrate.CheckCenter.mixed23.
    """

    # kernel_invariants = {"detect_time", "pmt_array", "num_pmts"}
    # applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi" + " "   # White space is required

    data_folder = "check.Center"
    applet_name = "Center Calibration"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian
    xlabel = "Potential Center (um)"
    ylabel = "Population Transfer"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=-2, stop=2, npoints=21, randomize=False, seed=int(time.time())
                ),
                unit="",
                global_min=-30.0,
                global_max=30.0,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=0.5 * us, unit="us", step=1e-7, ndecimals=7),
        )
        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=1.0))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=1.0))

        self.setattr_argument("auto_reload", artiq_env.BooleanValue(default=False))
        self.setattr_argument(
            "repetition", artiq_env.NumberValue(default=0, step=1, ndecimals=0)
        )

        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=8))

        self.scan_values = [np.float(0)]

        super().build()

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

        self.setattr_device("scheduler")

        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=0.5))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=0.5))

        self.submitter = Submitter(self.scheduler,__file__)
        # Manually override the check for lost ions to avoid false alarms during calibration routine
        self.do_calib = False
        self.use_RFSOC = True

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )

        chain_X2 = self.get_dataset(CHAIN_X2_DATASET)
        self.qubit_num = self.get_dataset(QUBIT_NUM_DATASET)
        self.set_dataset("global.Voltages.X2", chain_X2, persist=True)

        super().prepare()
        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]

        self.rf_ramp.prepare()

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def custom_experiment_initialize(self):
        pass

    @host_only
    def run(self):
        """Start the experiment on the host."""

        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.original_piezo_setting = self.get_dataset(
                "global.Raman.Piezos.Ind_FinalX"
            )
            # self.set_dataset("global.Raman.Piezos.Ind_FinalX", 2.5, persist=True)

            self.custom_experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")
            self.internal_analyze()

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for _center_position in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                all_ions = list(
                    qp.active_backend().configuration().all_qubit_indices_iter
                )
                for ion in all_ions:
                    qp.call(
                        single_qubit.square_rabi_by_amplitude(
                            ion_index=ion,
                            duration=self.rabi_time,
                            individual_amp=self.rabi_ind_amp,
                            global_amp=self.rabi_global_amp,
                            detuning=0.0
                        )
                )
                # qp.call(
                #     single_qubit.square_rabi_by_amplitude(
                #         ion_index=0,
                #         duration=self.rabi_time,
                #         individual_amp=self.rabi_ind_amp,
                #         global_amp=self.rabi_global_amp,
                #         detuning=0.0
                #     )
                # )

            schedule_list.append(out_sched)

        return schedule_list

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
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)
        delay(5 * us)


    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(1 * ms)

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

    @rpc
    def update_DAC(self, cent):
        self.sandia_box.center = cent
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start", line_gain=1)

    def analyze(self):
        pass

    def internal_analyze(self):
        """Analyze and Fit data"""

        center_check_thresh = self.get_dataset(CENTER_CHECK_THRESH_DATASET)

        if self._FIT_TYPE is not None:
            # self.set_experiment_data("fit_type", self._FIT_TYPE, broadcast=False)
            x = np.array(self.get_experiment_data("x_values"))
            y = self.get_experiment_data("avg_thresh")
            fit_plot_points = len(x) * 10
            # print("total num of points", fit_plot_points)

            self.p_all = defaultdict(list)
            self.p_error_all = defaultdict(list)
            # Can't change np.nan to 0, or else the fit is a constant line at 0
            y_fit_all = np.full((y.shape[0], fit_plot_points), np.nan)
            p = self._FIT_TYPE.parameter_names

            for iy in range(y.shape[0]):
                try:
                    p, p_error, x_fit, y_fit = self._FIT_TYPE.fit(
                        x,
                        y[iy, :],
                        evaluate_function=True,
                        evaluate_n=fit_plot_points
                    )

                    for ip in p:
                        self.p_all[ip].append(p[ip])
                        self.p_error_all[ip].append(p_error[ip])

                    y_fit_all[iy, :] = y_fit

                except FitError:
                    _LOGGER.info("Fit failed for y-data # %i", iy)
                    for ip in p:
                        self.p_all[ip].append(np.float(0))
                        self.p_error_all[ip].append(np.float(0))

                    y_fit_all[iy, :] = 0.0
                    continue



            for ip in p:
                self.set_experiment_data(
                    "fitparam_" + ip, self.p_all[ip], broadcast=True
                )
            self.set_experiment_data("fit_x", x_fit, broadcast=True)
            self.set_experiment_data("fit_y", y_fit_all, broadcast=True)

            # Only choose those fits with maximal PT >= center_check_thresh
            valid_ind = [i for i in range(y.shape[0]) if np.max(y[i, :]) >= center_check_thresh]

        # Set the center of coolant PMT to be 0
        for ind in range(y.shape[0]):
            if ind not in valid_ind:
                self.p_all["x0"][ind] = np.float(0)

        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        num_active_pmts = len(self.pmt_array.active_pmts)
        buf = "{"
        for ifit in range(num_active_pmts):
            ret = self.p_all["x0"][ifit]
            buf = buf + "%f," % ret
            # print(
            #     "Fit {}:\n\tx0 = ({} +- {})\n\tsigma = ({} +- {})".format(
            #         ifit,
            #         self.p_all["x0"][ifit],
            #         self.p_error_all["x0"][ifit],
            #         self.p_all["sigma"][ifit],
            #         self.p_error_all["sigma"][ifit],
            #     )
            # )
        buf = buf + "}"
        print(buf)
        print(
            "pmt {}: aligned Center = {}".format(
                center_pmt_idx + 1, self.p_all["x0"][center_pmt_idx]
            )
        )

        centers = np.array([])
        for ind in range(2, 21):
            if ind in valid_ind:
                centers = np.append(centers, self.p_all["x0"][ind])

        spread = np.std(centers)

        if (spread > 0.2) or (np.isnan(spread)) or (np.isinf(spread)) or (len(valid_ind) < self.qubit_num):
            self.set_dataset(
                "global.Raman.Piezos.Ind_FinalX",
                self.original_piezo_setting,
                persist=True,
            )
            if self.auto_reload:

                log_calibration(
                    type="Check_center_mixed_23",
                    operation="Warning",
                    value="wrong number of ions",
                )

                # see if reloading capacity is reached
                reload_capacity = self.get_dataset("monitor.reload_capacity")

                # if reload_capacity > 0:
                #     self.set_dataset(
                #         "monitor.reload_capacity",
                #         reload_capacity - 1,
                #         broadcast=True,
                #         persist=True,
                #     )
                #     if self.repetition <= 5:
                #         # check for one ion ~ 2 mean SS cooling counts per all PMTs
                #         if np.sum(np.sum(np.array(self.get_dataset("data.monitor.sscooling")))) > 2 * self.num_steps:
                #             log_calibration(
                #                 type="Check_center_23",
                #                 operation="engaging DX check",
                #                 repetition=self.repetition,
                #             )
                #             self.submitter.submit(
                #                 "Calibrate_DX", priority=priorities.CALIBRATION_PRIORITY1,
                #                 repetition=0,
                #                 **(_AUTO_CALIBRATION_SETTINGS["DX"].to_dict()),
                #                 hold=True,
                #             )
                #             self.submitter.submit(
                #                 "CalibrateX2Offset", priority=priorities.CALIBRATION_PRIORITY1,
                #                 **(_AUTO_CALIBRATION_SETTINGS["X2_Offset"].to_dict()),
                #                 hold=True
                #             )
                #         # if no ions are left, just reload 23
                #         self.submitter.submit(
                #                         "Autoload_23", priority=priorities.CALIBRATION_CRITICAL,
                #                         **(_AUTO_CALIBRATION_SETTINGS["Autoload_23"].to_dict()),
                #                         hold=True,
                #                     )
                #         self.submitter.submit(
                #             "Calibrate_CheckCenter23",
                #             repetition=0,
                #             hold=True,
                #             **(_AUTO_CALIBRATION_SETTINGS["CheckCenter23"].to_dict()),
                #             priority=priorities.CALIBRATION_PRIORITY2,
                #         )
                #         # self.submitter.submit(
                #         #     "PrecisionModeScan",
                #         #     repetition=0,
                #         #     hold=True,
                #         #     **(_AUTO_CALIBRATION_SETTINGS["PrecisionModeScan"].to_dict()),
                #         #     priority=priorities.CALIBRATION_PRIORITY2
                #         # )
                #     else:
                #         _LOGGER.error("auto-reload error: too many failed attempts")
                #         log_calibration(
                #             type="Check_center_23",
                #             operation="Error:too many reload attempts",
                #             repetition=self.repetition,
                #         )
                # else:
                #     _LOGGER.error("auto-reload error: reload capacity reached")
                #     log_calibration(
                #         type="Check_center_23",
                #         operation="Error:reload capacity reached",
                #     )

            else:

                _LOGGER.error("Doesn't have a mixed 23-ion chain")

        else:

            print("Successfully loaded mixed 23-ion chain")

            self.set_dataset(
                "global.Voltages.center",
                self.p_all["x0"][center_pmt_idx],
                broadcast=True,
                persist=True,
            )

            log_calibration(
                type="Center",
                operation="set",
                value=float(self.p_all["x0"][center_pmt_idx]),
            )

class Swap_Check(BasicEnvironment_Sort, artiq_env.Experiment):
    """Swap.Check"""

    data_folder = "Swap.Check"
    applet_name = "Swap.Check"
    applet_group = "Sort"
    fit_type = None
    units = us
    ylabel = "SS Counts"
    xlabel = "Step"

    def build(self):
        """Initialize experiment & variables."""

        super(Swap_Check, self).build()

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

        # Each step checking one pair
        self.num_steps = 22
        # Each pair check 15 times (some shots might be used for getting the pair ready for checking)
        self.num_shots = 15

    def prepare(self):

        super(Swap_Check, self).prepare()

        self.shuttle_to_list = []
        self.shuttle_from_list = []
        self.shuttle_from_default_list = []
        self.shuttle_to_default_list = []

        for i in range(22):
            # self.shuttle_to_list.append(self.sandia_box.get_shuttle_data_stepnum(21, 0, 21, 4))
            # self.shuttle_from_list.append(self.sandia_box.get_shuttle_data_stepnum(21, 3, 21, 0))
            # self.shuttle_from_default_list.append(self.sandia_box.get_shuttle_data('Start', f'{21}-0'))
            # self.shuttle_to_default_list.append(self.sandia_box.get_shuttle_data(f'{21}-0', 'Start'))

            # # For debug
            # if i == 2 or i == 18:
            #     self.shuttle_to_list.append(self.sandia_box.get_shuttle_data_stepnum(i, 0, i, 3))
            #     self.shuttle_from_list.append(self.sandia_box.get_shuttle_data_stepnum(i, 3, i, 0))
            #     self.shuttle_from_default_list.append(self.sandia_box.get_shuttle_data('Start', f'{i}-0'))
            #     self.shuttle_to_default_list.append(self.sandia_box.get_shuttle_data(f'{i}-0', 'Start'))
            # else:
            #     self.shuttle_to_list.append(self.sandia_box.get_shuttle_data_stepnum(i, 0, i, 4))
            #     self.shuttle_from_list.append(self.sandia_box.get_shuttle_data_stepnum(i, 3, i, 0))
            #     self.shuttle_from_default_list.append(self.sandia_box.get_shuttle_data('Start', f'{i}-0'))
            #     self.shuttle_to_default_list.append(self.sandia_box.get_shuttle_data(f'{i}-0', 'Start'))
            self.shuttle_to_list.append(self.sandia_box.get_shuttle_data_stepnum(i, 0, i, 4))
            self.shuttle_from_list.append(self.sandia_box.get_shuttle_data_stepnum(i, 3, i, 0))
            self.shuttle_from_default_list.append(self.sandia_box.get_shuttle_data('Start', f'{i}-0'))
            self.shuttle_to_default_list.append(self.sandia_box.get_shuttle_data(f'{i}-0', 'Start'))

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

        self.check_swap_flag = False

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

        # istep: used as the pair ind

        self.check_swap_flag = False

        # Check if the pair is prepared for checking
        if self.config[istep] != self.config[istep + 1]:
            self.check_swap_flag = True

        if not self.check_swap_flag:
            # The pair is not ready for checking
            # Distance between the closest the other isotope to the right ion in the pair
            rdist = 0
            # Distance between the closest the other isotope to the left ion in the pair
            ldist = 0
            for ind in range(istep + 2, 23):
                rdist += 1
                if self.config[ind] != self.config[istep + 1]:
                    break
            for ind in range(istep):
                ldist += 1
                if self.config[istep - 1 - ind] != self.config[istep]:
                    break

            if ldist == 0:
                ldist = 23
            if rdist == 0:
                rdist == 23

            if ldist <= rdist:
                # Move the left
                for ind in range(istep - ldist, istep):
                    # Swap the ion to the pair
                    self.sandia_box.shuttle_path(self.shuttle_from_default_list[ind])

                    self.core.break_realtime()
                    self.sandia_box.shuttle_path(self.shuttle_to_list[ind])

                    self.core.break_realtime()
                    self.sandia_box.shuttle_path(self.shuttle_from_list[ind])

                    self.core.break_realtime()
                    self.sandia_box.shuttle_path(self.shuttle_to_default_list[ind])

                    self.core.break_realtime()
            else:
                # Move the right
                for ind in range(rdist):
                    # Swap the ion to the pair
                    self.sandia_box.shuttle_path(self.shuttle_from_default_list[istep + rdist - ind])

                    self.core.break_realtime()
                    self.sandia_box.shuttle_path(self.shuttle_to_list[istep + rdist - ind])

                    self.core.break_realtime()
                    self.sandia_box.shuttle_path(self.shuttle_from_list[istep + rdist - ind])

                    self.core.break_realtime()
                    self.sandia_box.shuttle_path(self.shuttle_to_default_list[istep + rdist - ind])

                    self.core.break_realtime()

        else:
            # The pair is ready for checking
            # Shuttle
            # Need to make sure the check is at tweaked default line
            self.sandia_box.shuttle_path(self.shuttle_from_default_list[istep])

            self.core.break_realtime()
            # delay(10 * ms)
            self.sandia_box.shuttle_path(self.shuttle_to_list[istep])

            self.core.break_realtime()
            # delay(10 * ms)
            self.sandia_box.shuttle_path(self.shuttle_from_list[istep])

            self.core.break_realtime()
            self.sandia_box.shuttle_path(self.shuttle_to_default_list[istep])

            self.core.break_realtime()

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

class Edge_X4_Calibration(BasicEnvironment_Sort, artiq_env.Experiment):
    """EdgeX4.Calibration
    Must run this experiment with 23-coolant chain. 
    """

    data_folder = "EdgeX4.Calibration"
    applet_name = "EdgeX4.Calibration"
    applet_group = "Sort"
    fit_type = fit.rabi_flop
    units = us
    ylabel = "Counts"
    xlabel = "X4"

    def build(self):
        """Initialize experiment & variables."""

        super(Edge_X4_Calibration, self).build()

        self.rf_ramp = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=-0.0002,
                    stop=0.0002,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                ndecimals=6,
                unit="",
                global_min=-30.0,
                global_max=30.0,
            ),
        )

    def prepare(self):

        super(Edge_X4_Calibration, self).prepare()

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        # Get nominal X4
        X4 = self.get_dataset(CHAIN_X4_DATASET)
        self.scan_values = [x + X4 for x in self.scan_range]
        self.num_steps = len(self.scan_values)
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
    def main_experiment(self, istep, ishot):
        pass

    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(200 * us)

    @rpc
    def update_DAC(self, X4):
        self.sandia_box.X4 = X4 
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start", line_gain=1)

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
        y = self.get_dataset("data.monitor.sscooling_coolant")
        opt_ind = 0
        dark_thresh = 1
        for i in range(y.shape[1]):
            if i == 0:
                min_counts = min([ele for ele in y[:, i] if ele > dark_thresh])
            else:
                min_counts_tmp = min([ele for ele in y[:, i] if ele > dark_thresh])
                if min_counts_tmp > min_counts:
                    opt_ind = copy.copy(i)
                    min_counts = copy.copy(min_counts_tmp)
        x4_opt = self.scan_values[opt_ind]
        self.set_dataset(
            "global.Voltages.X4_EdgeImage",
            x4_opt,
            broadcast=True,
            persist=True,
        )        
        self.set_dataset(
            "global.SS_Cool.Presence_Thresh_Counts_Coolant",
            min(min_counts/2, 10),
            broadcast=True,
            persist=True,
        )
        _LOGGER.info("Find optimal X4 = {} for imaging, update coolant presence threshold counts to {}".format(x4_opt, min(min_counts/2, 10)))
    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

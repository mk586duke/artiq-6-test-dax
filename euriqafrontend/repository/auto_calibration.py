"""Experiments to Auto-Calibrate the EURIQA system."""
import functools
import logging

import typing
import time

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan

import numpy as np
import qiskit.pulse as qp
import oitg.fitting as fit

from artiq.experiment import NumberValue
from artiq.experiment import StringValue
from artiq.experiment import BooleanValue
from artiq.language.core import (
    delay,
    delay_mu,
    host_only,
    kernel,
    rpc,
    TerminationRequested,
)
from artiq.language.types import TInt32
from artiq.language.units import MHz, kHz, ms, us
from scipy import stats

import euriqafrontend.fitting as umd_fit
import euriqafrontend.settings as settings
import euriqafrontend.modules.rfsoc as rfsoc
import euriqabackend.waveforms.single_qubit as single_qubit
import euriqafrontend.repository.rfsoc.rabi_spectroscopy as rfsoc_spec
from euriqafrontend.repository.basic_environment import BasicEnvironment
from euriqafrontend.modules.spec_analyzer import Spec_Analyzer
from euriqafrontend.scheduler import ExperimentPriorities as priorities
from collections import defaultdict

from euriqafrontend.modules.calibration import CalibrationModule
from euriqafrontend.modules.calibration import log_calibration
from euriqafrontend.modules.calibration import Submitter

IntegerValue = functools.partial(NumberValue, scale=1, step=1, ndecimals=0)

_LOGGER = logging.getLogger(__name__)
_AUTO_CALIBRATION_SETTINGS = settings.auto_calibration
CHAIN_X2_DATASET = "monitor.chain_X2"
CHAIN_X4_DATASET = "monitor.chain_X4"

class Calibrate_QXZ_Prep(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.QXZ.Prep

    This experiment does a spectroscopy scan near an odd mode to fit for the peak frequency.
    When it finishes, a Calibrate.QXZ will be submitted
    (with the located peak frequency passed in as an argument)
    """

    data_folder = "raman_rabi_spec"
    applet_name = "Raman Rabi Spectroscopy"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian
    fit_type = None
    xlabel = "detuning (MHz)"
    ylabel = "population transfer"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            rfsoc.FrequencyScan(
                default=artiq_scan.RangeScan(
                    start=2.885 * MHz,
                    stop=2.965 * MHz,
                    npoints=30,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="MHz",
                global_min=-5 * MHz,
            ),
        )

        self.setattr_argument(
            "sideband_order",
            NumberValue(1, scale=1, min=-3, max=3, step=1, ndecimals=0),
        )

        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=400 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=0.2))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=0.2))

        super().build()

        self.spec_analyzer = Spec_Analyzer(self)
        self.setattr_device("scheduler")
        self.submitter = Submitter(self.scheduler,__file__)
        self.use_RFSOC = True

    def prepare(self):

        self.set_variables(
            self.data_folder,
            self.applet_name,
            self.applet_group,
            self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )

        self.scan_values = [f * self.sideband_order for f in self.scan_range]
        self.num_steps = len(self.scan_values)
        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()

            self.spec_analyzer.module_init(
                data_folder=self.data_folder, num_steps=self.num_steps
            )

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for detuning in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion_index=[-1,1],
                        duration=self.rabi_time,
                        individual_amp=self.rabi_ind_amp,
                        global_amp=self.rabi_global_amp,
                        detuning=detuning,
                        sideband_order=self.sideband_order,
                    )
                )

            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    def analyze(self):
        """Threshold values and analyze."""
        self.set_dataset("monitor.calibration_error", 0, persist=True)
        temp = self.spec_analyzer.fit_multiple_peaks()
        temp = np.divide(temp, 1e6)
        self.set_dataset("data.spec_analysis.peaks", temp, broadcast=True)

        temp = np.ndarray.tolist(temp)
        print("peaks are:", temp)

        self.submitter.submit(
            "Calibrate_QXZ",
            repetition=0,
            hold=False,
            **(_AUTO_CALIBRATION_SETTINGS["QXZ"].to_dict()),
            detuning=temp[0] * MHz,
        )


class Calibrate_QXZ(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.QXZ

    This experiment scan QXZ while driving detuned (to resonate with an odd radial mode) rabi flop
    The middle ion should have zero coupling to odd mode. So we find the dip in the population transfer
    and set the dip as the proper QXZ value.
    """

    data_folder = "raman_rabi_spec"
    applet_name = "Raman Rabi Spectroscopy"
    applet_group = "Raman Calib"
    fit_type = umd_fit.negative_gaussian
    xlabel = "QXZ"
    ylabel = "population transfer"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=-0.015,
                    stop=0.015,
                    npoints=30,
                    randomize=False,
                    seed=int(time.time()),
                ),
                ndecimals=4,
                global_min=-5e6,
            ),
        )

        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=8))

        self.setattr_argument(
            "detuning",
            NumberValue(
                0,
                unit="MHz",
                min=-10 * MHz,
                max=40 * MHz,
                step=0.0025 * MHz,
                ndecimals=6,
            ),
        )

        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=500 * us, unit="us", step=1e-7, ndecimals=7),
        )
        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=0.1))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=0.1))

        super().build()

        self.spec_analyzer = Spec_Analyzer(self)
        self.use_RFSOC = True

    def prepare(self):

        self.set_variables(
            self.data_folder,
            self.applet_name,
            self.applet_group,
            self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )
        QXZ = self.get_dataset("global.Voltages.QXZ")
        self.scan_values = [QXZ + x for x in self.scan_range]
        self.num_steps = len(self.scan_values)
        super().prepare()

        print(self.detuning)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()

            self.spec_analyzer.module_init(
                data_folder=self.data_folder, num_steps=self.num_steps
            )

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")


    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(100 * ms)

    @rpc
    def update_DAC(self, QXZ):
        self.sandia_box.QXZ = QXZ
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start", line_gain=1)

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for _qxz in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion_index=0,
                        duration=self.rabi_time,
                        individual_amp=self.rabi_ind_amp,
                        global_amp=self.rabi_global_amp,
                        detuning=self.detuning,
                        sideband_order=1
                    )
                )

            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    def analyze(self):
        """Threshold values and analyze."""
        super().analyze()
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)

        new_QXZ = self.p_all["x0"][center_pmt_idx]

        print("new QXZ value is set to: {}".format(new_QXZ))

        self.set_dataset("global.Voltages.QXZ", new_QXZ, persist=True)

        log_calibration(type="QXZ", operation="set", value=float(new_QXZ))


class Calibrate_CheckCenter23(BasicEnvironment, artiq_env.EnvExperiment):
    """Calibrate.CheckCenter23.
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
        self.set_dataset("global.Voltages.X2", chain_X2, persist=True)

        super().prepare()
        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]

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

            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(1 * ms)

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
        super().analyze()
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

        centers = np.array(self.p_all["x0"])[2:20]

        spread = np.std(centers)

        if (spread > 0.2) or (np.isnan(spread)) or (np.isinf(spread)):
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

                if reload_capacity > 0:
                    self.set_dataset(
                        "monitor.reload_capacity",
                        reload_capacity - 1,
                        broadcast=True,
                        persist=True,
                    )
                    if self.repetition <= 5:
                        # check for one ion ~ 2 mean SS cooling counts per all PMTs
                        if np.sum(np.sum(np.array(self.get_dataset("data.monitor.sscooling")))) > 2 * self.num_steps:
                            log_calibration(
                                type="Check_center_23",
                                operation="engaging DX check",
                                repetition=self.repetition,
                            )
                            self.submitter.submit(
                                "Calibrate_DX", priority=priorities.CALIBRATION_PRIORITY1,
                                repetition=0,
                                **(_AUTO_CALIBRATION_SETTINGS["DX"].to_dict()),
                                hold=True,
                            )
                            self.submitter.submit(
                                "CalibrateX2Offset", priority=priorities.CALIBRATION_PRIORITY1,
                                **(_AUTO_CALIBRATION_SETTINGS["X2_Offset"].to_dict()),
                                hold=True
                            )
                        # if no ions are left, just reload 23
                        self.submitter.submit(
                                        "Autoload_23", priority=priorities.CALIBRATION_CRITICAL,
                                        **(_AUTO_CALIBRATION_SETTINGS["Autoload_23"].to_dict()),
                                        hold=True,
                                    )
                        self.submitter.submit(
                            "Calibrate_CheckCenter23",
                            repetition=0,
                            hold=True,
                            **(_AUTO_CALIBRATION_SETTINGS["CheckCenter23"].to_dict()),
                            priority=priorities.CALIBRATION_PRIORITY2,
                        )
                        # self.submitter.submit(
                        #     "PrecisionModeScan",
                        #     repetition=0,
                        #     hold=True,
                        #     **(_AUTO_CALIBRATION_SETTINGS["PrecisionModeScan"].to_dict()),
                        #     priority=priorities.CALIBRATION_PRIORITY2
                        # )
                    else:
                        _LOGGER.error("auto-reload error: too many failed attempts")
                        log_calibration(
                            type="Check_center_23",
                            operation="Error:too many reload attempts",
                            repetition=self.repetition,
                        )
                else:
                    _LOGGER.error("auto-reload error: reload capacity reached")
                    log_calibration(
                        type="Check_center_23",
                        operation="Error:reload capacity reached",
                    )

            else:

                _LOGGER.error("Doesn't have a chain of 23")

        else:

            print("Successfully loaded 23")

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


class Calibrate_X3(BasicEnvironment, artiq_env.EnvExperiment):
    """Calibrate.X3.

    """

    # kernel_invariants = {"detect_time", "pmt_array", "num_pmts"}
    # applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi" + " "   # White space is required

    data_folder = "Auto_Calibration"
    applet_name = "Center Calibration"
    applet_group = "Raman Calib"
    xlabel = "Potential Center (um)"
    ylabel = "Population Transfer"
    fit_type = umd_fit.positive_gaussian

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
                global_min=-10.0,
                global_max=10.0,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=0.6 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument(
            "max_step", artiq_env.NumberValue(default=10, unit="", step=1, ndecimals=0)
        )

        self.setattr_argument(
            "repetition", artiq_env.NumberValue(default=0, step=1, ndecimals=0)
        )

        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=8))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=0.5))
        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=0.5))

        super().build()

        # Manually override the check for lost ions to avoid false alarms during calibration routine

        self.setattr_device("scheduler")
        self.submitter = Submitter(self.scheduler,__file__)

        self.use_RFSOC = True

    def prepare(self):

        chain_X2 = self.get_dataset(CHAIN_X2_DATASET)
        self.set_dataset("global.Voltages.X2", chain_X2, persist=True)

        self.feedback_coef = -0.0015/5  # was 0.0007 (tune the feedback here)
        self.precision_target = 0.05/2

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )

        super().prepare()

        # Having calibrate active can cause loss of ions because it will send HW triggers which can interrupt the
        # asyncronous data upload used to sweep Center. Here we manually override.
        self.do_calib = False
        self.num_steps = len(self.scan_range)
        self.scan_values = list(self.scan_range)

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            # self.set_dataset("global.Raman.Piezos.Ind_FinalX", 2.5, persist=True)

            self.experiment_initialize()

            _LOGGER.debug("Done Initializing Experiment. Starting Main Loop:")

            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.debug("Done with Experiment. Setting machine to idle state")

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for _x3 in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                all_ions = list(
                    qp.active_backend().configuration().all_qubit_indices_iter
                )
                for ion in all_ions:
                    qp.call(
                        single_qubit.square_rabi_by_amplitude(
                            ion,
                            duration=self.rabi_time,
                            individual_amp=self.rabi_ind_amp,
                            global_amp=self.rabi_global_amp,
                        )
                )

            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(1 * ms)

    @rpc
    def update_DAC(self, cent):
        self.sandia_box.center = cent
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start", line_gain=1)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        self.set_dataset("monitor.calibration_error", 0, persist=True)
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        num_active_pmts = len(self.pmt_array.active_pmts)

        foundCenter = self.p_all["x0"][center_pmt_idx]
        if np.isnan(foundCenter):
            _LOGGER.error("fit center failed")

        single_ion_center = self.get_dataset("monitor.calibration.single_ion_center")

        _LOGGER.info(
            "23 ion center is now measured to be: %0.3f, compared to single ion center %0.3f", foundCenter, single_ion_center
            )

        centers = np.array(self.p_all["x0"])[2:14]
        spread = np.std(centers)

        if (spread > 0.2) or (np.isnan(spread)) or (np.isinf(spread)):
            self.cancel_and_reload()
        else:

            diff = single_ion_center - foundCenter

            new_X3 = self.get_dataset("global.Voltages.X3")
            if abs(diff) < self.precision_target:
                self.set_dataset(
                    "global.Voltages.center",
                    foundCenter,
                    broadcast=True,
                    persist=True,
                )
                log_calibration(type="X3", operation="set", value=float(new_X3))
            else:
                new_X3 -= diff * self.feedback_coef
                if abs(diff * self.feedback_coef) < 0.001:  # 0.0002: #set the cutoff
                    self.set_dataset("global.Voltages.X3", new_X3, persist=True)
                else:
                    log_calibration(
                        type="X3",
                        operation="feedback step too large: rejected the new X3 value",
                        value=float(new_X3),
                    )
                    _LOGGER.info(
                        "feedback step of %0.3f too large, need manual adjustment:",
                        diff * feedback_coef
                    )

                if self.repetition <= self.max_step:
                    self.submitter.submit(
                        "Calibrate_X3",
                        repetition=self.repetition + 1,
                        **(_AUTO_CALIBRATION_SETTINGS["X3"].to_dict())
                    )
                else:
                    log_calibration(
                        type="X3",
                        operation="Error: fail to converge",
                        value=float(new_DX),
                    )
                    _LOGGER.error("X3 Calibration failed to converge")
                    self.set_dataset("monitor.calibration_error", 1, persist=True)


class Calibrate_X4(rfsoc_spec.RamanRabiSpec):
    """Calibrate.X4"""

    # data_folder = "raman_rabi_spec"
    # applet_name = "Raman Rabi Spectroscopy"
    applet_group = "Raman"
    fit_type = umd_fit.positive_gaussian

    # xlabel = "detuning (MHz)"
    # ylabel = "population transfer"

    # output in Hz
    def get_freq_standard(self):
        return MHz * 1.7635
        #return MHz * 1.755

    def build(self):
        super().build()

        self.setattr_device("scheduler")
        self.calibration = CalibrationModule(self,scheduler = self.scheduler, calibration_type = self.__class__.__name__, file=__file__)

        self.sideband_order = self.get_argument("sideband_order", NumberValue(1.0))

        self.setattr_argument(
            "base_scan_scale",
            artiq_env.NumberValue(1.0, min=1e-9),
            tooltip="Base scaling factor for this scan. The pulse duration will be "
                    "increased by this, and other factors will be adjusted accordingly. "
                    "Increasing values imply later iterations",
        )
        self.setattr_argument(
            "repeat_count",
            IntegerValue(0),
            tooltip="Count of how many times this experiment has been run",
        )
        self.setattr_argument(
            "max_repeats",
            IntegerValue(10),
            tooltip="Number of repeats before this experiment stops/fails",
        )
        self.dataset_x4 = "global.Voltages.X4"

        self.dataset_error = "monitor.calibration.error"
        self.spec_analyzer = Spec_Analyzer(self)

    def prepare(self):
        self.frequency_nominal = self.get_freq_standard()
        chain_X2 = self.get_dataset(CHAIN_X2_DATASET)
        self.set_dataset("global.Voltages.X2", chain_X2, persist=True)

        # scale values by base_scan_scale
        # increase duration, decrease frequency span & amplitude
        scan_scale = self.base_scan_scale
        _LOGGER.info("Current scan scale (iteration): %.2f", scan_scale)
        self.feedback_coef = 0.008*0.15

        self.rabi_individual_amplitude /= scan_scale
        self.scan_values = (
            np.array(list(self.scan_range)) / scan_scale + self.frequency_nominal
        )
        self.rabi_time *= scan_scale
        self.ions_to_address = "-7:7"

        super().prepare()

    @host_only
    def run(self):
        self.sandia_box.center += 0.25  # shift center, so more towards side of beam
        self.sandia_box.update_compensations()
        self.sandia_box.dac_pc.apply_line_async("QuantumRelaxed_ZeroLoad", line_gain=0)

        self.spec_analyzer.module_init(
            data_folder=self.data_folder, num_steps=self.num_steps
        )
        """Start the experiment on the host."""
        super().run()

    def analyze(self):
        # Constants
        ALLOWED_CORRECTION = 0.00015  # maximum correction step
        MIN_SCALE = 32 # minimal scale to terminate
        ALLOWED_ERROR = (
            0.5 * kHz
        )  # max allowed error. Will stop iteration if less than this
        dataset_name = self.dataset_x4

        self.set_dataset(self.dataset_error, 0, persist=True)
        try:
            peak_frequency = self.spec_analyzer.fit_single_peak()
            _LOGGER.info(
                "Found peak at %.3f MHz (nominal freq = %.3f MHz)",
                peak_frequency / MHz,
                self.frequency_nominal / MHz,
            )
            expected_peak_span_fwhm = 0.88 / self.rabi_time
            frequency_error = peak_frequency - self.frequency_nominal

        except RuntimeError as exc:
            _LOGGER.error("Peak fitting failed", exc_info=True)
            frequency_error = 100e6
        submit_next = (self.repeat_count < self.max_repeats) and (
            (np.abs(frequency_error) > ALLOWED_ERROR) or (self.base_scan_scale < MIN_SCALE)
        )
        new_scale = self.base_scan_scale

        if (
            abs(frequency_error)
            >= (np.max(self.scan_values) - np.min(self.scan_values)) * 0.35
        ):
            # frequency out of range, revert the step, increment counter
            _LOGGER.info(
                "Frequency out of range. Reverting step."
            )
            new_scale /= 2.0
        elif abs(frequency_error) >= expected_peak_span_fwhm*0.5:
            _LOGGER.info(
                "Peak %.3f out of range [%.3f, %.3f]. Correcting X4.",
                frequency_error / MHz,
                (self.frequency_nominal - expected_peak_span_fwhm*0.5) / MHz,
                (self.frequency_nominal + expected_peak_span_fwhm*0.5) / MHz,
            )

            # get current value
            last_value = self.get_dataset(dataset_name)
            # calculate correction
            correction = -(frequency_error / MHz) * self.feedback_coef
            _LOGGER.info(
                "X4 correction = %0.3E",
                correction
            )
            # check if correction step in allowed limits
            self.calibration.update_dataset_if_valid(
                dataset_name,
                last_value,
                float(correction),
                ALLOWED_CORRECTION,
                self.dataset_error,
            )
        else:
            _LOGGER.info(
                "Peak within expected FWHM of target. Zooming in."
            )
            new_scale *= 2.0

        if submit_next:
            updated_expid = self.scheduler.expid.copy()
            updated_expid["arguments"]["base_scan_scale"] = new_scale
            updated_expid["arguments"]["repeat_count"] = self.repeat_count + 1
            # submits with same parameters (due date, pipeline, priority), except updated values above
            self.scheduler.submit(expid=updated_expid)
        else:
            _LOGGER.info(
                "Finished iterating X4 Offset Calibration. Final error: %E Hz from (%E Hz) nominal freq",
                frequency_error,
                self.frequency_nominal,
            )
    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for detuning in self.scan_values:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:

                qp.call(
                    single_qubit.bichromatic_rabi_by_amplitude(
                            ion_index=self.qubits_to_address,
                            duration=self.rabi_time,
                            motional_detuning=detuning,
                            sideband_order=self.sideband_order,
                            individual_amp=self.rabi_individual_amplitude,
                            global_amp=self.rabi_global_amplitude,
                            phase_insensitive=self.phase_insensitive,
                            stark_shift=[0 for x in self.qubits_to_address]
                    )
                )
            schedule_list.append(out_sched)

        return schedule_list

# class Calibrate_EdgeX4(BasicEnvironment, artiq_env.EnvExperiment):
#     """Calibrate.EdgeX4
#     """
#     data_folder = "Auto_Calibration"
#     applet_name = "Edge X4 Calibration"
#     applet_group = "Raman Calib"
#     xlabel = "Potential Center (um)"
#     ylabel = "Population Transfer"
#     fit_type = umd_fit.positive_gaussian


class Modes_Scan_23ions(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.All_Radial_Modes"""

    data_folder = "raman_rabi_spec"
    applet_name = "Raman Rabi Spectroscopy"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian
    fit_type = None
    xlabel = "detuning (MHz)"
    ylabel = "population transfer"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=True))
        self.setattr_argument(
            "sideband_order",
            NumberValue(1, scale=1, min=-3, max=3, step=1, ndecimals=0),
        )

        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=500 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=0.01))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=0.01))

        super().build()

        self.spec_analyzer = Spec_Analyzer(self)

    def prepare(self):

        self.set_variables(
            self.data_folder,
            self.applet_name,
            self.applet_group,
            self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )
        self.rfsoc_sbc.initialize()
        # these scan values are hard coded to only scan around the desired gate modes
        self.scan_values = np.array([])
        self.mode_freq_offset0 = self.get_dataset("monitor.calibration.radial_mode_offset_frequency")
        self.gate_modes_23ions = [x + self.mode_freq_offset0 for x in self.rfsoc_sbc._rf_calib["chain_radial_modes.23.nominal.value"]]

        for val in self.gate_modes_23ions:
            self.scan_values = np.append(
                self.scan_values,
                [
                    val + offset
                    for offset in np.linspace(
                        -1.5 * 0.0015 * MHz, 1.5 * 0.0015 * MHz, num=16
                    )
                ],
            )
        self.scan_values = np.sort(self.scan_values, axis=None)
        self.num_steps = len(self.scan_values)
        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()

            self.spec_analyzer.module_init(
                data_folder=self.data_folder, num_steps=self.num_steps
            )

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for detuning in self.scan_values:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                all_ions = list(
                    qp.active_backend().configuration().all_qubit_indices_iter
                )
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        all_ions,
                        duration=self.rabi_time,
                        individual_amp=self.rabi_ind_amp,
                        global_amp=self.rabi_global_amp,
                        detuning=detuning,
                        sideband_order=self.sideband_order,
                        stark_shift=[0 for x in all_ions]
                    )
                )

            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    def analyze(self):

        """Threshold values and analyze."""
        self.set_dataset("monitor.calibration_error", 0, persist=True)
        temp = self.spec_analyzer.fit_multiple_peaks()
        self.set_dataset("data.spec_analysis.peaks", temp, broadcast=True)

        temp = np.ndarray.tolist(temp)
        print("peaks are:", temp)
        log_calibration(type="radial mode scan", operation="print peaks", value=temp)

        if len(temp) == 23 and self.set_globals:
            # send average difference to global gate tweak for motional freq adjust
            meas_modes = np.array(temp)
            middle_indices = range(4, 11, 1)

            diff = meas_modes - np.flip(self.gate_modes_23ions)
            avg_diff = np.average(diff[middle_indices])
            # this average difference is in MHz, and so is the global.
            self.set_dataset(
                "global.gate_tweak.motional_frequency_adjust",
                avg_diff,
                broadcast=True,
                persist=True,
            )

            # slope of the middle modes should be less than 100 Hz so let's do the fitting in H
            result = stats.linregress(middle_indices, diff[middle_indices])
            log_calibration(
                type="radial mode scan",
                operation="check slope",
                value=float(result.slope),
            )
            if np.abs(result.slope) < 100:
                print(
                    "The slope of the modes is {:.2f} Hz, which is less than 100 Hz.".format(
                        result.slope
                    )
                )
            else:
                print(
                    "The slope of the modes ({:.2f} Hz) is larger than 100 Hz and the modes may need to be recalibrated.".format(
                        result.slope
                    )
                )
        else:
            print("Not all the peaks were found, or set globals is False.")

import oitg.fitting as fit
from  oitg.fitting.FitBase import FitError
from euriqafrontend.fitting.PrecisionScanFit import sin_ffti_global, period_determination_FFT_global_min

class PrecisionModeScan(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.PrecisionModeScan

    Scan the delay of a Ramsey pulse sequence.
    """

    data_folder = "rfsoc_ramsey"
    applet_name = "Raman RFSOC Ramsey"
    applet_group = "RFSOC"
    fit_type = sin_ffti_global
    units = 1
    ylabel = "Population Transfer"
    unit = 1e-6
    xlabel = "delay"

    def build(self):
        """Initialize experiment & variables."""

        self.setattr_argument(
            "pulse_duration",
            artiq_env.NumberValue(default=250 * us, unit="us"),
            tooltip="Should be the pi/2 duration for a strongly-coupled ion",
        )
        self.setattr_argument("global_amplitude", rfsoc.AmplitudeArg(default=0.63))
        self.setattr_argument("individual_amplitude", rfsoc.AmplitudeArg(default=0.07))

        self.use_RFSOC = True

        # scan arguments
        self.setattr_argument(
            "delay_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(start=1 * us, stop=2500 * us, npoints=36),
                unit="us",
                global_min=0,
                ndecimals=5,
            ),
            tooltip="Scan of the delay time between the two pi/2 pulses in a Ramsey "
            "sequence",
        )
        self.setattr_argument("wait_detuning", artiq_env.NumberValue(default=2500, unit="Hz", step=10, ndecimals=0))
        self.setattr_argument(
            "global_phase_shift",
            rfsoc.PhaseArg(),
            tooltip="Phase to shift the global beam by during the Ramsey wait time",
        )
        self.setattr_argument(
            "repeat_count",
            IntegerValue(0),
            tooltip="Count of how many times this experiment has been run",
        )
        self.setattr_argument(
            "max_repeats",
            IntegerValue(10),
            tooltip="Number of repeats before this experiment stops/fails",
        )
        self.setattr_argument(
            "set_globals",
            BooleanValue(False)
        )
        self.dataset_x2 = "global.Voltages.X2"
        self.dataset_x4 = "global.Voltages.X4"
        self.dataset_error = "monitor.calibration.error"

        super().build()
        self.setattr_device("scheduler")
        self.calibration = CalibrationModule(self,scheduler = self.scheduler, calibration_type = self.__class__.__name__,file=__file__)

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )
        self.rfsoc_sbc.initialize()

        self.scan_values = list(self.delay_range)
        self.num_steps = len(self.scan_values)
        self.ions_to_address = [x-12+2 for x in [11, 4, 9, 13, 15, 16, 14, 5, 7, 6, 1, 19, 12, 2, 18, 17, 3, 8, 10]]
        self.predicted_couplings = [0.250699, 0.270786*0.7, 0.27288, 0.272107, 1.5*0.265556, 1.5*0.251231, 0.266791, \
                                    0.271939, 0.277955, 0.277942, 0.348612*0.7, 0.341712, 0.297248, 0.33855, \
                                    0.331253*1.3, 0.322167, 0.329293*1.12, 0.340971, 0.393797]
        self.addressed_modes = [13, 16, 19, 4, 10, 12, 11, 20, 5, 21, 3, 15, 6, 9, 7, 8, 18, 17, 14]

        self.mode_freq_offset0 = self.get_dataset("monitor.calibration.radial_mode_offset_frequency")
        self.gate_modes_23ions = [x + self.mode_freq_offset0 for x in self.rfsoc_sbc._rf_calib["chain_radial_modes.23.nominal.value"]]
        self.detuning_list = self.gate_modes_23ions[2:21]
        assert(len(self.detuning_list) == len(self.ions_to_address))
        super().prepare()

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []
        detunings = np.array(self.detuning_list)
        mean_detuning = np.mean(detunings)

        for wait_time in self.delay_range:
        #for ddet in dets:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                #for ion, detuning in zip(self.ions_to_address,self.detuning_list):
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion_index=self.ions_to_address,
                        duration=self.pulse_duration,
                        detuning=mean_detuning,# + ddet,
                        sideband_order=1,
                        individual_amp=list((self.individual_amplitude*0.3) / np.array(self.predicted_couplings)),
                        global_amp=self.global_amplitude,
                        stark_shift = detunings - mean_detuning - 200,
                    )
                )
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion_index=self.ions_to_address,
                        duration=wait_time,
                        detuning=mean_detuning,
                        sideband_order=1,
                        individual_amp=[0 for x in self.ions_to_address],
                        global_amp=self.global_amplitude,
                        stark_shift = (detunings - mean_detuning + self.wait_detuning),
                    )
                )

#                for ion, detuning in zip(self.ions_to_address,self.detuning_list):
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion_index=self.ions_to_address,
                        duration=self.pulse_duration,
                        detuning=mean_detuning,
                        phase=self.global_phase_shift,
                        sideband_order=1,
                        individual_amp=list((self.individual_amplitude*0.3) / np.array(self.predicted_couplings)),
                        global_amp=self.global_amplitude,
                        stark_shift=detunings - mean_detuning - 200,
                    )
                )

            schedule_list.append(out_sched)

        return schedule_list
    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_experiment_initialize()

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @host_only
    def custom_experiment_initialize(self):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    @staticmethod
    def nandot(a,b):
            ret = 0
            for i in range(len(b)):
                if (not np.isnan(b[i])) and (not np.isnan(a[i])):
                    ret+=a[i]*b[i]
            return ret

    @staticmethod
    def find_correction(detunings):

        VX2 = np.array([  0.6348041783221477, 0.7036671269755796, 1.0694758971553249, \
                1.2827384938977122, 1.5709941814419306, 1.8625936109369399, \
                2.3338533755324518, 2.6957827474651914, 2.9830920157486016, \
                3.5079236240733094, 4.012045260475369, 4.452571360155047, \
                5.031464381145012, 5.700726342614918, 6.184452608679615, \
                6.842555958099683, 7.407257438027321, 8.078173321803227, \
                9.364990152365579]) * 1e6
        VX4 =    np.array([6.52394, 10.025, 20.575, 28.65, 39.3366, 49.1341, 61.2916, 73.1413,
                81.2344, 93.7893, 107.165, 116.484, 132.055, 145.801, 153.978, \
                166.603, 173.437, 183.194, 193.389]) * 1e6
        VC  =  np.array([1 for x in VX4])

        M = np.array([
            [ np.dot(VX2,VX2), np.dot(VX2,VX4), np.dot(VC,VX2)],
            [ np.dot(VX4,VX2), np.dot(VX4,VX4), np.dot(VC,VX4)],
            [ np.dot(VX2,VC), np.dot(VX4,VC), np.dot(VC,VC)]
        ])
        return np.linalg.solve(M,np.array([PrecisionModeScan.nandot(VX2,detunings), PrecisionModeScan.nandot(VX4,detunings), PrecisionModeScan.nandot(VC, detunings)]))

    def analyze(self, constants={}, **kwargs):

        """Analyze and Fit data.
        """

        x = np.array(self.get_experiment_data("x_values"))
        y = self.get_experiment_data("avg_thresh")
        fit_plot_points = len(x) * 10

        self.p_all = defaultdict(list)
        self.p_error_all = defaultdict(list)
        y_fit_all = np.full((y.shape[0], fit_plot_points), np.nan)
        p = sin_ffti_global.parameter_names

        for iy in range(y.shape[0]):
            try:
                period_init, phase_init, amplitude_init, displacment_init = period_determination_FFT_global_min(x,y[iy, :])

                #for debugging
                #print('initialized parameters:',period_init, phase_init, amplitude_init, displacment_init)

                my_sin_ffti_global = sin_ffti_global

                my_sin_ffti_global.parameter_bounds = {
                'x0': (-np.pi,np.pi),
                'y0': (displacment_init*0.7,displacment_init*1.3),
                'a': (amplitude_init*0.7,amplitude_init*1.3),
                'period': (period_init*0.7,period_init*1.3),
                }

                p, p_error, x_fit, y_fit = my_sin_ffti_global.fit(
                    x,
                    y[iy, :],
                    evaluate_function=True,
                    evaluate_n=fit_plot_points,
                    constants=constants,
                )

                for ip in p:
                    self.p_all[ip].append(p[ip])
                    self.p_error_all[ip].append(p_error[ip])

                y_fit_all[iy, :] = y_fit

            except FitError:
                _LOGGER.info("Fit failed for y-data # %i", iy)
                for ip in p:
                    self.p_all[ip].append(np.float(np.NaN))
                    self.p_error_all[ip].append(np.float(np.NaN))

                y_fit_all[iy, :] = 0.0
                continue

        for ip in p:
            self.set_experiment_data(
                "fitparam_" + ip, self.p_all[ip], broadcast=True
            )
        self.set_experiment_data("fit_x", x_fit, broadcast=True)
        self.set_experiment_data("fit_y", y_fit_all, broadcast=True)

        fs = 1/np.array(self.p_all["period"])
        amps = np.array(self.p_all["a"])
        x0s = np.array(self.p_all["x0"])

        mode_frequencies = [0] * len(self.ions_to_address)
        mode_contrasts = [0] * len(self.ions_to_address)
        mode_phases = [0] * len(self.ions_to_address)

        for m in range(len(self.ions_to_address)):
            mode_frequencies[m] = np.abs(fs[ 9 + self.ions_to_address[m] ])
            mode_contrasts[m] = 2*np.abs(amps[ 9 + self.ions_to_address[m]])
            mode_phases[m] = x0s[ 9 + self.ions_to_address[m]]
            if amps[ 9 + self.ions_to_address[m]] < 0:
                mode_phases[m] += np.pi
            mode_phases[m] = np.mod(mode_phases[m], 2*np.pi)

        # higher X2 => lower mode frequency
        # higher X2 => higher fitted mode frequency
        # => the Ramsey idling frequency is above the mode frequencies
        # => subtract wait detuning to get negative mode offset frequencies
        mode_frequency_offsets = [self.wait_detuning - x for x in mode_frequencies]

        _LOGGER.info('mode frequency offsets : %s', ",".join(map(str,mode_frequency_offsets)))
        _LOGGER.info('mode contrasts : %s', ",".join(map(str,mode_contrasts)))
        _LOGGER.info('mode phases : %s', ",".join(map(str,mode_phases)))

        correction = self.find_correction(np.array(mode_frequency_offsets))
        mode_offset_freq = correction[2]
        # see prepare() for self.mode_freq_offset0

        if not self.set_globals:
            return
        self.set_dataset("monitor.calibration.radial_mode_offset_frequency", self.mode_freq_offset0 + mode_offset_freq, persist=True)
        _LOGGER.info("dX2 = %e, dX4 = %e, offset = %e Hz", correction[0], correction[1], mode_offset_freq)

        self.set_dataset("global.gate_tweak.motional_frequency_adjust", self.mode_freq_offset0 + mode_offset_freq, persist=True)

        # Constants
        FEEDBACK_GAIN = 0.8
        ALLOWED_X2_CORRECTION = 0.0009  # maximum X2 correction step
        ALLOWED_X4_CORRECTION = 0.0001
        ALLOWED_ERROR = ( 0.16 * kHz )  # max allowed error. Will stop iteration if less than this

        mean_error = (lambda v: np.sqrt(PrecisionModeScan.nandot(v,v) / (v.size-1)))(mode_frequency_offsets - np.mean(mode_frequency_offsets))
        _LOGGER.info("RMS mode frequency error = %e", mean_error)

        dataset_name_x4 = self.dataset_x4
        dataset_name_x2 = self.dataset_x2

        # get current X2 and X4 values
        last_x2_value = self.get_dataset(dataset_name_x2)
        last_x4_value = self.get_dataset(dataset_name_x4)

        submit_next = (self.repeat_count < self.max_repeats) and (
            (mean_error > ALLOWED_ERROR)
        )

        # check if correction step in allowed limits
        if np.abs(correction[0]) < ALLOWED_X2_CORRECTION and np.abs(correction[1]) < ALLOWED_X4_CORRECTION:
            self.calibration.update_dataset_if_valid(
                dataset_name_x2,
                last_x2_value,
                FEEDBACK_GAIN*float(correction[0]),
                ALLOWED_X2_CORRECTION,
                self.dataset_error,
            )
            self.calibration.update_dataset_if_valid(
                dataset_name_x4,
                last_x4_value,
                FEEDBACK_GAIN*float(correction[1]),
                ALLOWED_X4_CORRECTION,
                self.dataset_error,
            )
            self.calibration.update_dataset_if_valid(
                dataset_name_x2,
                last_x2_value,
                FEEDBACK_GAIN*float(correction[0]),
                ALLOWED_X2_CORRECTION,
                self.dataset_error,
            )
            self.calibration.update_dataset_if_valid(
                dataset_name_x4,
                last_x4_value,
                FEEDBACK_GAIN*float(correction[1]),
                ALLOWED_X4_CORRECTION,
                self.dataset_error,
            )
            self.set_dataset(self.dataset_error, 0, persist=True)
        else:
            submit_next = False
            self.set_dataset(self.dataset_error, 1, persist=True)
            _LOGGER.info(
                "Computed correction exceeds maximal value. Aborting."
            )

        if submit_next:
            updated_expid = self.scheduler.expid.copy()
            updated_expid["arguments"]["repeat_count"] = self.repeat_count + 1
            # submits with same parameters (due date, pipeline, priority), except updated values above
            self.scheduler.submit(expid=updated_expid)
        else:
            _LOGGER.info(
                "Finished iterating mode calibration. Final error: %E Hz from (%E Hz) nominal freq",
                mean_error,
                correction[2]
            )
            chain_X2 = self.get_dataset(dataset_name_x2)
            self.set_dataset(CHAIN_X2_DATASET,chain_X2, persist=True)
            chain_X4 = self.get_dataset(dataset_name_x4)
            self.set_dataset(CHAIN_X4_DATASET,chain_X4, persist=True)

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

class PrecisionTemperatureScan(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.PrecisionTemperatureScan

    Scan the delay of a Ramsey pulse sequence.
    """

    data_folder = "rfsoc_rabi"
    applet_name = "Raman RFSOC Rabi"
    applet_group = "RFSOC"
    fit_type = umd_fit.rabi_flop
    units = 1
    ylabel = "Population Transfer"
    unit = 1e-6
    xlabel = "delay"

    def build(self):
        """Initialize experiment & variables."""

        self.setattr_argument("global_amplitude", rfsoc.AmplitudeArg(default=0.5))
        self.setattr_argument("individual_amplitude", rfsoc.AmplitudeArg())

        self.use_RFSOC = True

        # scan arguments
        self.setattr_argument(
            "delay_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(start=1 * us, stop=1000 * us, npoints=20),
                unit="us",
                global_min=0,
                ndecimals=5,
            ),
            tooltip="Scan of the delay time between the two pi/2 pulses in a Ramsey "
            "sequence",
        )
        self.setattr_argument(
            "detuning",
            NumberValue(
                0,
                unit="kHz",
                min=-200 * kHz,
                max=200 * kHz,
                step=0.5 * kHz,
                ndecimals=6,
            ),
            tooltip="",
        )
        self.setattr_argument(
            "repeat_count",
            IntegerValue(0),
            tooltip="Count of how many times this experiment has been run",
        )
        self.setattr_argument(
            "max_repeats",
            IntegerValue(10),
            tooltip="Number of repeats before this experiment stops/fails",
        )

        super().build()

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        self.rfsoc_sbc.initialize()

        self.scan_values = list(self.delay_range)
        self.num_steps = len(self.scan_values)

        self.ions_to_address = [x-12+2 for x in [11, 4, 9, 13, 15, 16, 14, 5, 7, 6, 1, 19, 12, 2, 18, 17, 3, 8, 10]]
        self.predicted_couplings = [0.250699, 0.270786*0.7, 0.7*0.27288, 0.272107, 0.265556, 0.251231, 0.266791, \
                                    0.271939, 0.277955, 0.277942, 0.348612*0.7, 0.341712, 0.297248, 0.33855, \
                                    0.331253, 0.322167, 0.329293*1.12*1.7, 0.340971, 0.393797]
        self.addressed_modes = [13, 16, 19, 4, 10, 12, 11, 20, 5, 21, 3, 15, 6, 9, 7, 8, 18, 17, 14]

        self.mode_freq_offset0 = self.get_dataset("monitor.calibration.radial_mode_offset_frequency")

        self.gate_modes_23ions = [x + self.mode_freq_offset0 for x in self.rfsoc_sbc._rf_calib["chain_radial_modes.23.nominal.value"]]
        self.detuning_list = self.gate_modes_23ions[2:21]
        assert(len(self.detuning_list) == len(self.ions_to_address))

        super().prepare()

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []
        detunings = np.array(self.detuning_list)
        mean_detuning = np.mean(detunings)

        for pulse_duration in self.scan_values:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                #for ion, detuning in zip(self.ions_to_address,self.detuning_list):
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion_index=self.ions_to_address,
                        duration=pulse_duration,
                        detuning=self.detuning-mean_detuning,# + ddet,
                        sideband_order=1,
                        individual_amp=list((self.individual_amplitude*0.3) / np.array(self.predicted_couplings)),
                        global_amp=self.global_amplitude,
                        stark_shift = -(detunings - mean_detuning - 200),
                    )
                )

            schedule_list.append(out_sched)

        return schedule_list

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_experiment_initialize()

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @host_only
    def custom_experiment_initialize(self):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    def analyze(self, **kwargs):
        super().analyze(**kwargs)
        #num_active_pmts = len(self.p_all["period"])
        # fs = 1/np.array(self.p_all["period"])
        # res = [0] * len(self.ions_to_address)
        # for m in range(len(self.ions_to_address)):
        #     res[m] = np.abs(fs[ 9 + self.ions_to_address[m] ])

    @kernel
    def prepare_step(self, istep: TInt32):
        pass


class Voltage_Calibration_Chain(artiq_env.EnvExperiment):
    """Auto.Calibration.Voltages.Chain"""

    def build(self):
        """Initialize experiment & variables."""

        self.do_load_23 = self.get_argument("Load 23 ions", BooleanValue(default=True))

        self.check_center_23 = self.get_argument(
            "Check_center_23_ions", BooleanValue(default=True)
        )

        self.do_calib_X3 = self.get_argument(
            "Calibrate_X3 (make sure single ion center is up to date)",
            BooleanValue(default=True),
        )

        self.do_calib_X4 = self.get_argument(
            "Calibrate X4", BooleanValue(default=False)
        )

        self.do_calib_QXZ = self.get_argument(
            "Calibrate QXZ", BooleanValue(default=False)
        )

        self.do_spec = self.get_argument("Do Spectroscopy", BooleanValue(default=True))
        self.do_prec_mode_scan = self.get_argument("Do Precision Mode Scan", BooleanValue(default=True))

        self.do_RFSOC_calib = self.get_argument(
            "Do RFSOC calibration (use 0-ion indices)", BooleanValue(default=True)
        )

        self.ion_calib_list = self.get_argument(
            "Ion calibration sequences, separate by ;",
            StringValue(default="-9,-7,-4,-1;2,5,-6,-8;-3,0,3;8,6,-5,-2;1,4,7,9"),
        )
#"1,4,7,10,13;2,5,8,11,14;3,6,9,12,15"
        self.setattr_device("scheduler")
        self.submitter = Submitter(self.scheduler,__file__)

    @host_only
    def run(self):
        """Run each enabled calibration step."""

        if self.do_load_23:
            chain_X2 = self.get_dataset(CHAIN_X2_DATASET)
            self.set_dataset("global.Voltages.X2", chain_X2, persist=True)
            self.submitter.submit(
                "Autoload_23",
                priority=priorities.CALIBRATION_CRITICAL,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["Autoload_23"].to_dict())
            )

        if self.check_center_23:
            self.submitter.submit(
                "Calibrate_CheckCenter23",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["CheckCenter23"].to_dict())
            )
        if self.do_calib_X3:
            self.submitter.submit(
                "Calibrate_X3",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["X3"].to_dict())
            )

        if self.do_calib_X4:
            self.submitter.submit(
                "Calibrate_X4",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["X4"].to_dict())
            )

        if self.do_calib_QXZ:
            self.submitter.submit(
                "Calibrate_QXZ_Prep",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["QXZ_Prep"].to_dict())
            )

        if self.do_spec:
            self.submitter.submit(
                "Modes_Scan_23ions",
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["Modes_Scan_23ions"].to_dict())
            )

        if self.do_prec_mode_scan:
            self.submitter.submit(
                "PrecisionModeScan",
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["PrecisionModeScan"].to_dict())
            )

        if self.do_RFSOC_calib:
            calib_list = self.ion_calib_list.split(";")
            print(calib_list)
            for sequence in calib_list:
                self.submitter.submit(
                    "CalibrateRFSoCRabiAmplitude",
                    hold=True,
                    **{"ions_to_address": sequence},
                    **(_AUTO_CALIBRATION_SETTINGS["RFSOC_amp_calib"].to_dict())
                )

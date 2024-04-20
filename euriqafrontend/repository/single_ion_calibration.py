"""Single-Ion Calibrations for the EURIQA system."""
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
from artiq.experiment import BooleanValue
from artiq.experiment import StringValue

from artiq.language.core import (
    delay,
    delay_mu,
    host_only,
    kernel,
    rpc,
    now_mu,
    TerminationRequested,
)
from artiq.language.types import TInt32
from artiq.language.units import MHz, kHz, ms, us

import euriqabackend.coredevice.dac8568 as dac8568
import euriqafrontend.fitting as umd_fit
import euriqafrontend.settings as settings
import euriqafrontend.modules.rfsoc as rfsoc
import euriqabackend.waveforms.single_qubit as single_qubit
import euriqafrontend.repository.rfsoc.rabi_spectroscopy as rfsoc_spec
from euriqafrontend.repository.basic_environment_sort import BasicEnvironment
# from euriqafrontend.repository.basic_environment import BasicEnvironment
from euriqafrontend.modules.spec_analyzer import Spec_Analyzer
from euriqafrontend.scheduler import ExperimentPriorities as priorities
from euriqafrontend.modules.artiq_dac import RampControl as rampcontrol_auto

from euriqafrontend.modules.calibration import CalibrationModule
from euriqafrontend.modules.calibration import log_calibration
from euriqafrontend.modules.calibration import Submitter

IntegerValue = functools.partial(NumberValue, scale=1, step=1, ndecimals=0)

_LOGGER = logging.getLogger(__name__)
_AUTO_CALIBRATION_SETTINGS = settings.auto_calibration

class Calibrate_IndPiezoY(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.IndPiezoY."""

    data_folder = "piezo_scan"
    applet_name = "Rabi Piezo Scan"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian

    def build(self):
        """Initialize experiment & variables."""
        super().build()

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.CenterScan(
                    center=2.5,
                    span=4.99,
                    step=0.5,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="",
                global_min=0,
                global_max=5.0,
            ),
        )
        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=0.5))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=1.0))

        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=3 * us, unit="us", step=1e-7, ndecimals=7),
        )
        self.scan_values = [np.int32(0)]
        self.raman_time_mu = np.int32(0)
        self.use_RFSOC = True

    def prepare(self):

        # self.set_dataset(
        #     "global.Raman.Piezos.Ind_FinalX", 2.5, persist=True
        # )  # this is a hack, see if this fix the jitter issue of center scan .

        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.scan_values_mu = [
            dac8568.vout_to_mu(
                val, self.raman.ind_final_piezos.SandiaSerialDAC.V_OUT_MAX
            )
            for val in reversed(list(self.scan_range))
        ]
        self.scan_values = list(reversed(list(self.scan_range)))
        self.num_steps = len(self.scan_range)

        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        self.raman.ind_final_piezos.value2_mu = self.scan_values_mu[istep]
        self.raman.ind_final_piezos.update_value()

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for _piezo_value in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion_index=0,
                        duration=self.rabi_time,
                        individual_amp=self.rabi_ind_amp,
                        global_amp=self.rabi_global_amp,
                        detuning=0.0,
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
        """Analyze and Fit data.

        Threshold values and analyze."""
        super().analyze()
        meanTpi = 0.0
        # num_active_pmts = len(self.p_all["x0"])
        mean_x0 = 0.0
        buf = "{"
        for ifit in range(self.num_pmts):
            ret = self.p_all["x0"][ifit]
            buf = buf + "%f," % ret
            mean_x0 += self.p_all["x0"][ifit]
            print(
                "Fit {}: x0 = ({} +- {})".format(
                    ifit, self.p_all["x0"][ifit], self.p_error_all["x0"][ifit],
                )
            )
        buf = buf + "}"
        print(buf)
        mean_x0 /= self.num_pmts

        if self.set_globals:
            self.set_dataset("global.Raman.Piezos.Ind_FinalY", mean_x0, persist=True)

        log_calibration(type="Ind_FinalY", operation="set", value=float(mean_x0))


class Calibrate_DZ(BasicEnvironment, artiq_env.EnvExperiment):
    """Calibrate.DZ."""

    data_folder = "check.DZ"
    applet_name = "DZ Calibration"
    applet_group = "Raman Calib"
    fit_type = umd_fit.negative_gaussian

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=150,
                    stop=700,
                    npoints=16,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="",
                global_min=150,
                global_max=700,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=50 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=8))
        self.setattr_argument("offset", artiq_env.NumberValue(default=20))
        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=1.0))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=1.0))

        self.scan_values = [np.float(0)]
        self.raman_time_mu = np.int32(0)
        self.detuning_mu = np.int64(0)

        super().build()

        # Manually override the check for lost ions to avoid false alarms during calibration routine
        self.setattr_argument(
            "lost_ion_monitor", artiq_env.EnumerationValue(["False"], default="False")
        )
        self.use_RFSOC = True  # override any argument. not pretty b/c still leaves the argument in GUI

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.lost_ion_monitor = False

        # TODO: handle this. move to RF Calib struct?
        self.detuning_freq = self.get_dataset("global.Ion_Freqs.RF_Freq")
        super().prepare()

        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for _dz in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion_index=[0],
                        duration=self.rabi_time,
                        detuning=self.detuning_freq,
                        sideband_order=1,
                        individual_amp=self.rabi_ind_amp,
                        global_amp=self.rabi_global_amp
                    )
                )
            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(dz=self.scan_values[istep])
        delay(1 * ms)

    @rpc
    def update_DAC(self, dz):
        self.sandia_box.DZ = dz
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start", line_gain=1)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        # num_active_pmts = len(self.pmt_array.active_pmts)

        for ifit in range(self.num_pmts):
            print(
                "Fit {}:\n\tx0 = ({} +- {})".format(
                    ifit, self.p_all["x0"][ifit], self.p_error_all["x0"][ifit]
                )
            )

        print(
            "pmt {}: aligned DZ = {}".format(
                center_pmt_idx + 1, self.p_all["x0"][center_pmt_idx]
            )
        )

        log_calibration(
            type="DZ",
            operation="set value",
            value=float(self.p_all["x0"][center_pmt_idx] + self.offset),
        )

        if self.set_globals:
            self.set_dataset(
                "global.Voltages.Offsets.DZ",
                self.p_all["x0"][center_pmt_idx] + self.offset,
                persist=True,
            )

class Calibrate_DX(BasicEnvironment, artiq_env.EnvExperiment):
    """Calibrate.DX.

    Relax the potential to the relaxed axial line and center DX to maximize Rabi
    on the center PMT channel.
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
        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=1000))

        self.setattr_argument("auto_reload", artiq_env.BooleanValue(default=False))

        self.setattr_argument(
            "high_X2",
            artiq_env.NumberValue(default=0.4, unit="", step=2.5e-5, ndecimals=5),
        )

        self.setattr_argument(
            "low_X2",
            artiq_env.NumberValue(default=0.01, unit="", step=2.5e-5, ndecimals=5),
        )

        self.setattr_argument(
            "max_step", artiq_env.NumberValue(default=10, unit="", step=1, ndecimals=0)
        )

        self.setattr_argument(
            "repetition", artiq_env.NumberValue(default=0, scale=1, step=1, ndecimals=0)
        )

        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=8))

        self.scan_values = [np.float(0)]  # ??
        self.raman_time_mu = np.int32(0)  # ??

        super().build()

        # Manually override the check for lost ions to avoid false alarms during calibration routine
        # self.setattr_argument(
        #     "lost_ion_monitor", artiq_env.EnumerationValue(["False"], default="False")
        # )
        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=1.0))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=1.0))

        self.setattr_device("scheduler")

        self.submitter = Submitter(self.scheduler,__file__)
        self.use_RFSOC = True

    def prepare(self):
        self.lost_ion_monitor = False

        # self.set_dataset(
        #     "global.Raman.Piezos.Ind_FinalX", 2.5, persist=True
        # )  # this is a hack, see if this fix the jitter issue of center scan .

        # predicted el. field
        # = d(center)*1e-6 m * 171 amu * X2 MHz^2 * (2 pi)^2 / e
        # = 70 * d(center) X2 V/m
        self.feedback_coef = (
            self.low_X2 * 65
        )

        self.precision_target = 0.15

        if self.repetition == 0:
            self.set_dataset("global.Voltages.X2", self.high_X2, persist=True)
        else:
            self.set_dataset("global.Voltages.X2", self.low_X2, persist=True)

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
        self.scan_values = [val for val in self.scan_range]

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
            self.internal_analyze()

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for _dx_value in self.scan_range:
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

        # for _center_position in self.scan_range:
        #     with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
        #         all_ions = list(
        #             qp.active_backend().configuration().all_qubit_indices_iter
        #         )
        #         for ion in all_ions:
        #             qp.call(
        #                 single_qubit.square_rabi_by_amplitude(
        #                     ion_index=ion,
        #                     duration=self.rabi_time,
        #                     individual_amp=self.rabi_ind_amp,
        #                     global_amp=self.rabi_global_amp,
        #                     detuning=0.0
        #                 )
        #         )

        return schedule_list

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
        delay(1 * ms)  # allow DAC to settle
       
    @rpc
    def update_DAC(self, cent):
        self.sandia_box.center = cent
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start", line_gain=1)

    def terminate_all_expts(self):
        running_expts = self.submitter.scheduler.get_status()
        for rid in running_expts.keys():
            if rid != self.submitter.scheduler.rid:
                self.submitter.scheduler.request_termination(rid)

    def analyze(self):
        pass

    def internal_analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        self.set_dataset("monitor.calibration_error", 0, persist=True)
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        num_active_pmts = len(self.pmt_array.active_pmts)

        foundCenter = self.p_all["x0"][center_pmt_idx]
        foundSigma = self.p_all["sigma"][center_pmt_idx]

        more_than_one = False
        ss_cool_counts = self.get_dataset("data.monitor.sscooling")
        print("SS counts sum = ", np.sum(ss_cool_counts))

        # SS monitor is set to 0.004 to try isotope checker
        if np.sum(ss_cool_counts) > 3000:
            _LOGGER.error("Calibrate.DX: more than one ion present.")
            more_than_one = True

        no_ions = False
        if np.sum(ss_cool_counts) < 5:
            _LOGGER.error("Calibrate.DX: no ions present.")
            no_ions = True

        bad_center_fit = False
        if np.isnan(foundCenter) or np.abs(foundCenter) > 4 or np.abs(foundSigma) < 0.1 or np.abs(foundSigma) > 1:
            bad_center_fit = True
            _LOGGER.error("Calibrate.DX: center fit failed.")

        if more_than_one or bad_center_fit or no_ions:
            if not self.auto_reload:
                _LOGGER.error("Calibrate.DX: auto-reload off. Terminating the pipeline.")
                self.terminate_all_expts()
                return
            reload_capacity = self.get_dataset("monitor.reload_capacity")
            if reload_capacity <=0:
                _LOGGER.error("Calibrate.DX: reload capacity exceeded. Terminating the pipeline.")
                self.terminate_all_expts()
                return

            print("Calibrate.DX: reloading 1 ion.")
            self.set_dataset(
                "monitor.reload_capacity",
                reload_capacity - 1,
                broadcast=True,
                persist=True,
            )
            log_calibration(
                type="Calibrate_DX",
                operation="engaging autoloading"
            )
            self.submitter.submit(
                "Autoload_1", priority=priorities.CALIBRATION_CRITICAL,
                hold=True
            )
            self.submitter.submit(
                "Calibrate_DX",
                priority=self.submitter.scheduler.priority,
                repetition=0,
                **(_AUTO_CALIBRATION_SETTINGS["DX"].to_dict())
            )
            return

        # First repetition, done at high X2
        if self.repetition == 0:
            # set the center to the fitted value at high X2
            self.set_dataset(
                "global.Voltages.center", foundCenter, persist=True
            )

            _LOGGER.info("Calibrate.DX: Finished tight-X2 scan; setting single_ion_center to {center}".format(center=foundCenter))
            # set the single_ion_center for the subsequent X3 calibration
            self.set_dataset(
                "monitor.calibration.single_ion_center",
                foundCenter,
                persist=True,
            )

            log_calibration(
                type="Center",
                operation="set",
                value=float(self.p_all["x0"][center_pmt_idx]),
            )

            self.submitter.submit(
                "Calibrate_DX",
                priority=self.submitter.scheduler.priority,
                repetition=self.repetition + 1,
                **(_AUTO_CALIBRATION_SETTINGS["DX"].to_dict())
            )

        else:
            center_position = self.get_dataset("global.Voltages.center")
            diff = center_position - foundCenter
            _LOGGER.info("Iteration {iter}: center difference = {difference}".format(iter=self.repetition,difference=diff))
            if abs(diff) < self.precision_target:
                new_DX = self.get_dataset("global.Voltages.Offsets.DX")
                self.set_dataset("global.Voltages.X2", self.high_X2, persist=True)
                log_calibration(type="DX", operation="set", value=float(new_DX))
                _LOGGER.info("Found DX = {val}".format(val=new_DX))
            else:
                new_DX = self.get_dataset("global.Voltages.Offsets.DX")
                new_DX -= diff * self.feedback_coef

                if (
                    abs(diff * self.feedback_coef) < 2
                ):
                    _LOGGER.info("Updating DX to {val}".format(val=new_DX))
                    self.set_dataset("global.Voltages.Offsets.DX", new_DX, persist=True)
                else:
                    log_calibration(
                        type="DX",
                        operation="feedback step too large: rejected the new DX value",
                        value=float(new_DX),
                    )
                    _LOGGER.error("Calibrate.DX: feedback step too large: rejected the new DX value")

                if self.repetition <= self.max_step:
                    self.submitter.submit(
                        "Calibrate_DX",
                        priority=self.submitter.scheduler.priority,
                        repetition=self.repetition + 1,
                        **(_AUTO_CALIBRATION_SETTINGS["DX"].to_dict())
                    )
                else:
                    log_calibration(
                        type="DX",
                        operation="Error: fail to converge",
                        value=float(new_DX),
                    )
                    self.set_dataset("global.Voltages.X2", self.high_X2, persist=True)
                    _LOGGER.error("X2 Calibration failed to converge")
                    self.set_dataset("monitor.calibration_error", 1, persist=True)

class CalibrateX2Offset(rfsoc_spec.RamanRabiSpec):
    """Calibrate.X2

    Relax the potential to nominal 100 kHz axial and adjust X2 offset to match
    """

    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian

    def build(self):
        super().build()

        self.setattr_device("scheduler")
        self.calibration = CalibrationModule(self,scheduler = self.scheduler, calibration_type = self.__class__.__name__, file=__file__)

        self.sideband_order = self.get_argument("sideband_order", NumberValue(1.0))

        self.setattr_argument(
            "frequency_nominal",
            rfsoc.FrequencyArg(default=100 * kHz),
            tooltip="Nominal (center) frequency of the scan. "
            "Actual detunings will be frequency_nominal + scan_range, "
            "scaled by base_scan_scale",
        )

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
        self.dataset_x2 = "global.Voltages.X2"
        self.dataset_x2_offset = "global.Voltages.Offsets.X2"
        self.dataset_error = "monitor.calibration.error"
        self.spec_analyzer = Spec_Analyzer(self)

    def prepare(self):
        self.last_x2 = self.get_dataset(self.dataset_x2)
        self.set_dataset(self.dataset_x2, 0.01)

        # predicted freq. shift = d(sqrt(X2)) = d(X2) / sqrt(X2) / 2
        # so that, at our nominal X2 = 0.01
        # df (MHz) = d(X2) / sqrt(0.01) / 2 = d(X2) / 0.1 / 2 = 5 * d(X2)
        # and dX2 = df (MHz) / 5 = 0.2 * df (MHz)
        self.feedback_coef = (
            0.198
        )

        # scale values by base_scan_scale
        # increase duration, decrease frequency span & amplitude
        scan_scale = self.base_scan_scale
        _LOGGER.info("Current scan scale (iteration): {val}".format(val=scan_scale))

        self.rabi_individual_amplitude /= scan_scale
        self.scan_values = (
            np.array(list(self.scan_range)) / scan_scale + self.frequency_nominal
        )
        self.rabi_time *= scan_scale

        super().prepare()

    def run(self):
        self.sandia_box.center += 0.4  # shift center to move to the side of beam
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start", line_gain=1.0)

        self.spec_analyzer.module_init(
            data_folder=self.data_folder, num_steps=self.num_steps
        )
        super().run()

    def analyze(self):
        # Constants
        ALLOWED_CORRECTION = 0.025  # maximum correction step
        MIN_SCALE = 16 # minimal scale to terminate
        ALLOWED_ERROR = (
            0.5 * kHz
        )  # max allowed error. Will stop iteration if less than this
        dataset_name = self.dataset_x2_offset

        self.set_dataset(self.dataset_error, 0, persist=True)
        try:
            peak_frequency = self.spec_analyzer.fit_single_peak()
            _LOGGER.info(
                "Found peak at %.3f kHz (nominal freq = %.3f kHz)",
                peak_frequency / kHz,
                self.frequency_nominal / kHz,
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
                "Frequency outside of middle 0.70 of scan: stepping scale up"
            )
            new_scale /= 2.0
        elif abs(frequency_error) >= expected_peak_span_fwhm*0.5:
            _LOGGER.info(
                "Peak %.3f out of range [%.3f, %.3f]; Correcting X2",
                frequency_error * kHz,
                (self.frequency_nominal - expected_peak_span_fwhm*0.5) * kHz,
                (self.frequency_nominal + expected_peak_span_fwhm*0.5) * kHz,
            )

            # get current value
            last_value = self.get_dataset(dataset_name)
            # calculate correction

            correction = -(frequency_error / MHz) * self.feedback_coef

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
                "Peak within expected FWHM of target.f Zooming in."
            )
            new_scale *= 2.0

        # revert X2 value to original value. Always do this, so even failures restore value.
        # also allows not passing the original X2 to successive iterations
        self.set_dataset(self.dataset_x2, self.last_x2, persist=True)
        if submit_next:
            updated_expid = self.scheduler.expid.copy()
            updated_expid["arguments"]["base_scan_scale"] = new_scale
            updated_expid["arguments"]["repeat_count"] = self.repeat_count + 1
            # submits with same parameters (due date, pipeline, priority), except updated values above
            self.scheduler.submit(expid=updated_expid)
        else:
            _LOGGER.info(
                "Finished iterating X2 Offset Calibration. Final error: %E Hz from (%E Hz) nominal freq",
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

class Calibrate_QZY(rfsoc_spec.RamanRabiSpec):
    """Calibrate.QZY"""

    #data_folder = "raman_rabi_spec"
    #applet_name = "Raman Rabi Spectroscopy"
    applet_group = "Raman"
    fit_type = umd_fit.positive_gaussian
    #xlabel = "detuning (MHz)"
    #ylabel = "population transfer"

    # output in Hz
    def get_freq_standard(self):
        frf_sec = self.get_dataset("global.Ion_Freqs.frf_sec")
        qzy = 0.5
        qzz = self.get_dataset("global.Voltages.QZZ")
        x2 = 0.4
        return MHz*np.sqrt((frf_sec / 1e6) ** 2 - x2 / 2 - np.sqrt(qzy * qzy + qzz * qzz))

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
        self.dataset_x2 = "global.Voltages.X2"
        self.dataset_qzy = "global.Voltages.QZY"

        self.dataset_error = "monitor.calibration.error"
        self.spec_analyzer = Spec_Analyzer(self)

    def prepare(self):
        self.frequency_nominal = self.get_freq_standard()
        self.set_dataset(self.dataset_x2, 0.4, persist=True)
        #self.rfsoc_sbc.schedule_transform_aom_nonlinearity = False
        # scale values by base_scan_scale
        # increase duration, decrease frequency span & amplitude
        scan_scale = self.base_scan_scale
        _LOGGER.info("Current scan scale (iteration): %.2f", scan_scale)

        self.feedback_coef = (
            -5
        )
        self.rabi_individual_amplitude /= scan_scale
        self.scan_values = (
            np.array(list(self.scan_range)) / scan_scale + self.frequency_nominal
        )
        self.rabi_time *= scan_scale

        super().prepare()

    @host_only
    def run(self):
        self.spec_analyzer.module_init(
            data_folder=self.data_folder, num_steps=self.num_steps
        )
        """Start the experiment on the host."""
        super().run()

    def analyze(self):
        # Constants
        ALLOWED_CORRECTION = 0.07  # maximum correction step
        MIN_SCALE = 32 # minimal scale to terminate at
        ALLOWED_ERROR = (
            0.2 * kHz
        )  # max allowed error. Will stop iteration if less than this
        dataset_name = self.dataset_qzy

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
            np.abs(frequency_error) > ALLOWED_ERROR or (self.base_scan_scale < MIN_SCALE)
        )
        new_scale = self.base_scan_scale
        if (
            abs(frequency_error)
            >= (np.max(self.scan_values) - np.min(self.scan_values)) * 0.35
        ):
            # frequency out of range, revert the step, increment counter
            _LOGGER.info(
                "Frequency outside of middle 0.70 of scan: stepping scale up"
            )
            new_scale /= 2.0
        elif abs(frequency_error) >= expected_peak_span_fwhm*0.5:
            _LOGGER.info(
                "Correcting QZY, peak %.3f out of range [%.3f, %.3f]",
                frequency_error / MHz,
                (self.frequency_nominal - expected_peak_span_fwhm*0.5) / MHz,
                (self.frequency_nominal + expected_peak_span_fwhm*0.5) / MHz,
            )

            # get current value
            last_value = self.get_dataset(dataset_name)
            # calculate correction
            correction = -(frequency_error / MHz) * self.feedback_coef
            _LOGGER.info(
                "QZY correction = %.3f",
                correction
            )
            # check if correction step in allowed limits
            self.calibration.update_dataset_if_valid(
                dataset_name,
                last_value,
                correction,
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
                "Finished iterating QZY Offset Calibration. Final error: %E Hz from (%E Hz) nominal freq",
                frequency_error,
                self.frequency_nominal,
            )


class Voltage_Calibration_Single_Ion(artiq_env.EnvExperiment):
    """Auto.Calibration.Voltages.Single_Ion"""

    def build(self):
        """Initialize experiment & variables."""

        self.do_load_1 = self.get_argument("Load 1 ion", BooleanValue(default=False))

        self.do_calib_DX = self.get_argument("Calibrate DX", BooleanValue(default=False))

        self.do_calib_IndPiezoY = self.get_argument(
            "Calibrate IndPiezoY", BooleanValue(default=False)
        )

        self.do_calib_DZ = self.get_argument("Calibrate DZ", BooleanValue(default=False))

        self.do_calib_X2_offset = self.get_argument(
            "Calibrate X2 offset", BooleanValue(default=False)
        )

        self.do_calib_QZY = self.get_argument(
            "Calibrate QZY", BooleanValue(default=False)
        )

        self.do_mw_Ramsey = self.get_argument(
            "Qubit Frequency", BooleanValue(default=False)
        )

        self.do_RFSOC_calib = self.get_argument(
            "Do RFSOC calibration", BooleanValue(default=False)
        )

        self.setattr_device("scheduler")
        self.submitter = Submitter(self.scheduler,__file__)

    @host_only
    def run(self):
        """Run each enabled calibration step."""
        if self.do_load_1:
            self.set_dataset("global.Voltages.X2", 0.4, persist=True)
            self.submitter.submit(
                "Autoload_1", priority=priorities.CALIBRATION_CRITICAL, hold=True
            )

        if self.do_calib_DX:
            self.submitter.submit(
                "Calibrate_DX",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["DX"].to_dict())
            )

        if self.do_calib_IndPiezoY:
            self.submitter.submit(
                "Calibrate_IndPiezoY",
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["IndPiezoY"].to_dict())
            )

        if self.do_calib_DZ:
            self.submitter.submit(
                "Calibrate_DZ",
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["DZ"].to_dict()),
            )

        if self.do_calib_X2_offset:
            self.submitter.submit(
                "CalibrateX2Offset",
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["X2_Offset"].to_dict()),
            )

        if self.do_calib_QZY:
            self.submitter.submit(
                "Calibrate_QZY",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["QZY"].to_dict())
            )

        if self.do_mw_Ramsey:
            self.submitter.submit(
                "MW_Ramsey",
                repetition=0,
                hold=True,
                **(_AUTO_CALIBRATION_SETTINGS["MW_Ramsey"].to_dict())
            )

        if self.do_RFSOC_calib:
            self.submitter.submit(
                "CalibrateRFSoCRabiAmplitude",
                hold=True,
                **{"ions_to_address": "0"},
                **(_AUTO_CALIBRATION_SETTINGS["RFSOC_amp_calib_1"].to_dict())
            )

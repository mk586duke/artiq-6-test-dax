"""XX Gate Calibrations for the EURIQA system."""
import functools
import logging
import copy

import typing
import time
import euriqabackend.waveforms.decorators as wf_dec

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan

import numpy as np
import qiskit.pulse as qp

from artiq.experiment import NumberValue
from artiq.experiment import StringValue
from artiq.experiment import BooleanValue
from artiq.language.units import MHz, kHz, ms, us
from artiq.language.core import (
    delay,
    delay_mu,
    host_only,
    kernel,
    rpc,
    TerminationRequested,
)
from artiq.language.types import TInt32

import oitg.fitting as fit
import euriqafrontend.fitting as umd_fit
from oitg.fitting.FitBase import FitError

import euriqafrontend.settings as settings
import euriqafrontend.modules.rfsoc as rfsoc
import euriqabackend.waveforms.multi_qubit as multi_qubit
import euriqabackend.waveforms.single_qubit as single_qubit
from euriqafrontend.repository.basic_environment import BasicEnvironment
from euriqafrontend.modules.population_analysis import PopulationAnalysis
from euriqafrontend.scheduler import ExperimentPriorities as priorities
import euriqabackend.devices.keysight_awg.common_types as common_types

from euriqafrontend.modules.calibration import CalibrationModule
from euriqafrontend.modules.calibration import log_calibration
from euriqafrontend.modules.calibration import Submitter
from euriqafrontend.modules.utilities import parse_ion_input

IntegerValue = functools.partial(NumberValue, scale=1, step=1, ndecimals=0)

import more_itertools
from itertools import combinations
from collections import defaultdict
AmpSegmentList = typing.Sequence[
    typing.Tuple[float, typing.Union[typing.Tuple[float, float], float]]
]
@wf_dec.get_gate_solution(
{
    common_types.XXModulationType.AM_interp,
    common_types.XXModulationType.AM_segmented,
},
{"nominal_rabi_segments": "segments"},
convert_ions_to_slots=True,
convert_solution_units=True,
)
def gate_duration(ions: typing.Sequence[int], nominal_rabi_segments: AmpSegmentList):
    durations, rabi_freqs = more_itertools.unzip(nominal_rabi_segments)
    durations = np.array(list(durations))
    return sum(durations)

_LOGGER = logging.getLogger(__name__)
_AUTO_CALIBRATION_SETTINGS = settings.auto_calibration


def repeated_ions(pairs):
    addressed_gate_pairs = list(combinations(pairs,2))
    found_overlap = False
    for p in addressed_gate_pairs:
        if p[0][0]==p[1][0] or p[0][1]==p[1][1] or p[0][0]==p[1][1] or p[0][1]==p[1][0]:
            found_overlap = True
    return found_overlap

class XX_Amp_Calibration(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.XX.Amplitude."""

    data_folder = "raman_awg"
    applet_name = "Raman AWG"
    applet_group = "Raman"
    ylabel = "Population Transfer"
    xlabel = "XX Gate Amplitude"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            rfsoc.AmplitudeScan(
                default=artiq_scan.RangeScan(start=0, stop=2.0, npoints=20,),
                global_min=0.0,
            ),
        )

        self.addressed_ion_indices_str = self.get_argument(
            "Gate pair to be calibrated", artiq_env.StringValue(default="-1,1")
        )

        self.gate_angle = self.get_argument(
            "Designed gate angle",
            artiq_env.NumberValue(
                default=np.pi / 4, unit="pi", scale=np.pi, step=2.5e-5, ndecimals=5
            ),
        )

        self.setattr_argument(
            "N_gates", artiq_env.NumberValue(default=3, scale=1, step=1, ndecimals=0)
        )

        super().build()
        #self.population_analyzer = PopulationAnalysis(self)
        self.use_AWG = False
        self.use_RFSOC = True

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )
        self.addressed_ion_indices_str = self.addressed_ion_indices_str.split(";")        
        self.addressed_ion_indices = [parse_ion_input(str) for str in self.addressed_ion_indices_str]

        if repeated_ions(self.addressed_ion_indices):
            print("Cannot perform parallel calibration of gates involving same ions.")
            assert(False)

        self.scan_values = list(self.scan_range)
        self.num_steps = len(self.scan_values)
        self.num_ions = self.get_dataset("global.AWG.N_ions")

        super().prepare()
        self.resubmit_on_ion_loss = True
        #self.XX_slots=[common_types.ion_to_slot(x,self.rfsoc_sbc.number_of_ions(),one_indexed=False) for x in self.addressed_ion_indices]

    @rpc(flags={"async"})
    def custom_proc_data(self, istep):
        pass

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            # calls basic_environment.experiment_initialize(),
            # which will call AWG.experiment_initialize()
            self.experiment_initialize()
            # initialize the experiment-specific gate parameters
            # and program the AWG
            self.custom_experiment_initialize()
            # run the initialization on the core
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @host_only
    def custom_experiment_initialize(self):
        pass
        #self.population_analyzer.module_init(data_folder=self.data_folder, active_pmts=self.pmt_array.active_pmts, num_steps=self.num_steps, detect_thresh=self.detect_thresh, XX_slots = self.XX_slots)

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for xx_amp in self.scan_values:            
            with qp.build(self.rfsoc_sbc.qiskit_backend) as out_sched:
                for ion_pair in self.addressed_ion_indices:
                    for _i in range(self.N_gates):
                        qp.call(
                            multi_qubit.xx_am_gate(
                                ion_pair,
                                theta=self.gate_angle,
                                #individual_amplitude_multiplier=xx_amp,
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
        
        counts = np.array(self.get_dataset("data." + self.data_folder + ".raw_counts"))
        threshold_counts = np.where(counts > self.detect_thresh, 1, 0)
        for pair in self.addressed_ion_indices:
            self.analyze_pair(threshold_counts, pair)

    def analyze_pair(self, threshold_counts, ion_pair):
        
        XX_slots=[common_types.ion_to_slot(i,self.num_ions,one_indexed=False) for i in ion_pair]
        XX_slots_pmt_index = [self.pmt_array.active_pmts.index(x - 9) for x in XX_slots]
        
        # will be 1 for 11, -1 for 00 and 0 for 10 or 01
        tmp = (threshold_counts[XX_slots_pmt_index[0],:,:]-0.5) + (threshold_counts[XX_slots_pmt_index[1],:,:] - 0.5)
        tmp = tmp.flatten()
        population_osc = np.divide( np.sum(tmp), np.sum(np.abs(tmp)) )

        if population_osc < -0.9:
            print(ion_pair[0], ion_pair[1], "XX_Amp_Calibration : transfer too low.")
            return
        
        parity = np.prod(
            (2*threshold_counts[XX_slots_pmt_index[0],:,:]-1) + (2*threshold_counts[XX_slots_pmt_index[1],:,:] - 1),
            axis = 0
        )
        parity = np.mean(parity)

        rf_calib_struct = self.rfsoc_sbc.qiskit_backend.properties().rf_calibration
        gate_tweaks = rf_calib_struct.gate_tweaks.struct
        current_setting = gate_tweaks.loc[tuple(XX_slots),"individual_amplitude_multiplier"]
        
        if self.N_gates * self.gate_angle < np.pi / 2:
            new_amp_setting = current_setting * np.sqrt(
                self.N_gates * self.gate_angle / (0.5*np.arccos(-population_osc))
            )
        else:
            new_amp_setting = current_setting * np.sqrt(
                self.N_gates * self.gate_angle / (np.pi - 0.5*np.arccos(-population_osc))
            )

        print("new setting for ", XX_slots, "is : ", new_amp_setting)
        if new_amp_setting < 1.7:
            log_calibration(
                type="XX_amp",
                operation="set",
                value=round(float(new_amp_setting), 4),
                gate=str(self.addressed_ion_indices_str),
            )
            log_calibration(
                gate=str(self.addressed_ion_indices_str),
                type="P00,00+P11,11 (population)",
                operation="record",
                value=round(float((parity + 1) / 2.0), 4),
            )

            gate_tweaks.loc[tuple(XX_slots),"individual_amplitude_multiplier"] = float(new_amp_setting)
            gate_tweaks.to_h5(rf_calib_struct.gate_tweak_path)

        else:
            log_calibration(
                type="XX_amp",
                operation="Error: Overange",
                value=float(new_amp_setting),
                gate=str(self.addressed_ion_indices_str),
            )
            print("Calibration failed.")
            self.set_dataset("monitor.calibration_error", 1, persist=True)


class XX_StarkShift_Calibration(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.XX.Starkshift."""

    data_folder = "rfsoc_ramsey"
    applet_name = "Raman RFSOC Ramsey"
    applet_group = "RFSOC"
    ylabel = "Population Transfer"    
    xlabel = "Phase of Analysis Pulse"
    fit_type = fit.cos

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0, stop=2*np.pi, npoints=11, randomize=False, seed=int(time.time())
                ),
                unit="",
                ndecimals=4,
            ),
        )

        self.setattr_argument(
            "N_echo_blocks",
            artiq_env.NumberValue(default=1, scale=1, step=1, ndecimals=0),
        )

        self.addressed_ion_indices_str = self.get_argument(
            "Gate pair(s) to be calibrated", artiq_env.StringValue(default="-1,1")
        )

        self.setattr_argument(
            "clear_shift", artiq_env.BooleanValue(default=False)
        )
        self.do_parity_sk1 = True
        self.xx_offset_phase = 0.0*np.pi
        self.parity_sk1_phase = -np.pi/2
        self.do_xstart = True

        super().build()

        self.use_AWG = False
        self.use_RFSOC = True

    def prepare(self):

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )
        
        self.scan_values = list(self.scan_range)
        self.num_steps = len(self.scan_values)
        self.addressed_ion_indices_str = self.addressed_ion_indices_str.split(";")
        print(self.addressed_ion_indices_str)
        self.addressed_ion_indices = [parse_ion_input(str) for str in self.addressed_ion_indices_str]

        if repeated_ions(self.addressed_ion_indices):
            print("Cannot perform parallel calibration of gates involving same ions.")
            assert(False)

        self.num_ions = self.get_dataset("global.AWG.N_ions")
        
        super().prepare()
        self.resubmit_on_ion_loss = True

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            # calls basic_environment.experiment_initialize(),
            # which will call AWG.experiment_initialize()
            self.experiment_initialize()
            # initialize the experiment-specific gate parameters
            # and program the AWG
            self.custom_experiment_initialize()
            # run the initialization on the core
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
    def prepare_step(self, istep: TInt32):
        pass

    def pulse_schedule(self, ion_pair, scan_val) -> qp.Schedule:
        single_qubit_gate = lambda ions, theta, phi : single_qubit.sk1_gaussian(ions,theta,phi,backend=self.rfsoc_sbc.qiskit_backend)
        self.clear_shift = True
        with qp.build(self.rfsoc_sbc.qiskit_backend) as out_sched:
            self.gate_time = gate_duration(ion_pair)
            # overview:
            # * do pi/2 gate on both ions
            # * do echoes XX gates (alternating +XX/-XX)
            # * do pi/2 gate on both ions (scanning phase depending on scan param)

            def _do_echos(**kwargs):
                for i in range(self.N_echo_blocks * 2):
                    is_positive = ((i % 2) == 0)  # invert (echo) every other gate
                    if self.clear_shift:
                        qp.call(
                            multi_qubit.xx_am_gate(
                                ion_pair,
                                stark_shift = 0,
                                stark_shift_differential=0,
                                positive_gate=is_positive,                            
                                **kwargs,
                            )
                        )
                    else:
                        qp.call(
                            multi_qubit.xx_am_gate(
                                ion_pair,
                                positive_gate=is_positive,                            
                                **kwargs,
                            )
                        )


            if self.do_xstart:
                for ion in ion_pair:
                    qp.call(single_qubit_gate(ion, theta=np.pi / 2, phi=self.xx_offset_phase))
                        
            _do_echos()
            for ion in ion_pair:
                qp.call(single_qubit_gate(ion, theta=np.pi / 2, phi=scan_val + self.xx_offset_phase))

        return out_sched

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []
        
        for scan_val in self.scan_values:
            with qp.build(self.rfsoc_sbc.qiskit_backend) as out_sched:
                for pair in self.addressed_ion_indices:
                    qp.call(self.pulse_schedule(pair,scan_val))

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
        constants = {"period": 2 * np.pi}
        
        super().analyze(constants=constants)

        for ion_pair in self.addressed_ion_indices:
            self.analyze_pair(ion_pair)

    def analyze_pair(self, ion_pair):
        """Analyze and Fit data."""

        # self.set_dataset("monitor.calibration_error", 0, persist=True)

        XX_slots=[common_types.ion_to_slot(i,self.num_ions,one_indexed=False) for i in ion_pair]
        XX_slots_pmt_index = [self.pmt_array.active_pmts.index(x - 9) for x in XX_slots]

        rf_calib_struct = self.rfsoc_sbc.qiskit_backend.properties().rf_calibration
        gate_tweaks = rf_calib_struct.gate_tweaks.struct

        current_com_ss = gate_tweaks.loc[tuple(XX_slots),"stark_shift"] if not self.clear_shift else 0
        current_diff_ss = gate_tweaks.loc[tuple(XX_slots),"stark_shift_differential"] if not self.clear_shift else 0

        amplitude = np.array([self.p_all["a"][i] for i in XX_slots_pmt_index])
        phase_shift = np.array(
            [self.p_all["x0"][i] for i in XX_slots_pmt_index]
        )
        for i in np.arange(len(amplitude)):
            if np.sign(amplitude[i]) == -1:
                phase_shift[i] += np.pi
        num_gates = self.N_echo_blocks * 2
        # TODO this is hardcoded and will break for pi/8 gates for example
        diff_ss = (
            2
            * (phase_shift[0] - phase_shift[1])
            / (2 * np.pi * num_gates * self.gate_time)
        )
        common_ss = 2 * np.mean(phase_shift) / (2 * np.pi * num_gates * self.gate_time)

        new_diff_ss = current_diff_ss - (diff_ss)
        print(ion_pair[0], ion_pair[1],
            ": changing diff stark shift from {:4.4f} to {:4.4f} Hz".format(
                current_diff_ss, new_diff_ss
            )
        )
        log_calibration(
            type="XX_starkshift_differential (Hz)",
            operation="set",
            value=round(float(new_diff_ss)),
            gate=self.addressed_ion_indices_str
        )
        gate_tweaks.loc[tuple(XX_slots),"stark_shift_differential"] = new_diff_ss

        new_com_ss = current_com_ss - common_ss
        print(ion_pair[0], ion_pair[1], 
             ": changing comm stark shift from {:4.4f} to {:4.4f} Hz".format(
                current_com_ss, new_com_ss
            )
        )
        log_calibration(
            type="stark_shift (Hz)",
            operation="set",
            value=round(float(new_com_ss)),
            gate=self.addressed_ion_indices_str
        )
        gate_tweaks.loc[tuple(XX_slots),"stark_shift"] = new_com_ss

        gate_tweaks.to_h5(rf_calib_struct.gate_tweak_path)

class XX_Imbalance_Calibration(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.XX.Imbalance."""

    data_folder = "raman_awg"
    applet_name = "Raman AWG"
    applet_group = "Raman"
    ylabel = "Population Transfer"
    xlabel = "Global sideband imbalance"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=1, stop=5, npoints=5, randomize=False, seed=int(time.time())
                ),
                unit="",
                ndecimals=4,
            ),
        )

        self.setattr_argument(
            "N_echo_blocks",
            artiq_env.NumberValue(default=1, scale=1, step=1, ndecimals=0),
        )
        
        self.addressed_ion_indices_str = self.get_argument(
            "Gate pair(s) to be calibrated", artiq_env.StringValue(default="-1,1")
        )

        self.do_parity_sk1 = True
        self.xx_offset_phase = 0.0*np.pi
        self.parity_sk1_phase = -np.pi/2
        self.do_xstart = True

        super().build()
        self.resubmit_on_ion_loss = True
        self.use_AWG = False
        self.use_RFSOC = True

    def prepare(self):
        
        self.xlabel = "Sideband Imbalance"
        self.fit_type = fit.sin_fft
    
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        self.scan_values = list(self.scan_range)
        self.num_steps = len(self.scan_values)
        self.addressed_ion_indices_str = self.addressed_ion_indices_str.split(";")        
        self.addressed_ion_indices = [parse_ion_input(str) for str in self.addressed_ion_indices_str]

        if repeated_ions(self.addressed_ion_indices):
            print("Cannot perform parallel calibration of gates involving same ions.")
            assert(False)

        self.scan_offset = 0.003
        self.scan_values = [x + self.scan_offset for x in self.scan_values]
        self.num_ions = self.get_dataset("global.AWG.N_ions")
        
        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            # calls basic_environment.experiment_initialize(),
            # which will call AWG.experiment_initialize()
            self.experiment_initialize()
            # initialize the experiment-specific gate parameters
            # and program the AWG
            self.custom_experiment_initialize()
            # run the initialization on the core
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
    def prepare_step(self, istep: TInt32):
        pass

    def pulse_schedule(self, ion_pair, scan_val) -> qp.Schedule:
        single_qubit_gate = lambda ions, theta, phi : single_qubit.sk1_gaussian(ions,theta,phi,backend=self.rfsoc_sbc.qiskit_backend)

        with qp.build(self.rfsoc_sbc.qiskit_backend) as out_sched:
            self.gate_time = gate_duration(ion_pair)
            # overview:
            # * do pi/2 gate on both ions
            # * do echoes XX gates (alternating +XX/-XX)
            # * do pi/2 gate on both ions (scanning phase depending on scan param)

            def _do_echos(**kwargs):
                for i in range(self.N_echo_blocks * 2):
                    is_positive = ((i % 2) == 0)  # invert (echo) every other gate
                    qp.call(
                        multi_qubit.xx_am_gate(
                            ion_pair,
                            positive_gate=is_positive,
                            **kwargs,
                        )
                    )

            if self.do_xstart:
                for ion in ion_pair:
                    qp.call(single_qubit_gate(ion, theta=np.pi / 2, phi=self.xx_offset_phase))

            _do_echos(
                sideband_amplitude_imbalance=scan_val,
                stark_shift=0.0,
                stark_shift_differential=0.0,
            )
            for ion in ion_pair:
                qp.call(
                    single_qubit_gate(
                        ion, theta=np.pi / 2, phi=self.parity_sk1_phase + self.xx_offset_phase
                    )
            )
        
        return out_sched

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []
        
        for scan_val in self.scan_values:
            with qp.build(self.rfsoc_sbc.qiskit_backend) as out_sched:
                for pair in self.addressed_ion_indices:
                    qp.call(self.pulse_schedule(pair,scan_val))

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


    def analyze(self, constants={}):
        """Analyze and Fit data.

        Threshold values and analyze.
        """
    
        x = np.array(self.get_experiment_data("x_values"))
        y = self.get_experiment_data("avg_thresh")
        fit_plot_points = len(x) * 10

        self.p_all = defaultdict(list)
        self.p_error_all = defaultdict(list)
        y_fit_all = np.full((y.shape[0], fit_plot_points), np.nan)
        p = self._FIT_TYPE.parameter_names

#        for iy in range(y.shape[0]):
        for ion_pair in self.addressed_ion_indices:
            XX_slots=[common_types.ion_to_slot(i,self.num_ions,one_indexed=False) for i in ion_pair]
            iys = [self.pmt_array.active_pmts.index(x - 9) for x in XX_slots]
            fit_types = [self.fit_fnc(ion) for ion in ion_pair]
            p_local = defaultdict(list)
            fit_error = False
            for iy, fit_type in zip(iys, fit_types):
                try:
                    p, p_error, x_fit, y_fit = fit_type.fit(
                        x,
                        y[iy, :],
                        evaluate_function=True,
                        evaluate_n=fit_plot_points,
                        constants=constants,
                    )

                    for ip in p:
                        self.p_all[ip].append(p[ip])
                        p_local[ip].append(p[ip])
                        self.p_error_all[ip].append(p_error[ip])

                    y_fit_all[iy, :] = y_fit

                except FitError:
                    _LOGGER.info("Fit failed for y-data # %i", iy)
                    for ip in p:
                        self.p_all[ip].append(np.float(np.NaN))
                        p_local[ip].append(np.float(np.NaN))
                        self.p_error_all[ip].append(np.float(np.NaN))

                    y_fit_all[iy, :] = 0.0
                    fit_error = True
                    continue
            if not fit_error:
                self.analyze_pair(ion_pair, p_local)

        for ip in p:
            self.set_experiment_data(
                "fitparam_" + ip, self.p_all[ip], broadcast=True
            )
        self.set_experiment_data("fit_x", x_fit, broadcast=True)
        self.set_experiment_data("fit_y", y_fit_all, broadcast=True)

    def fit_fnc(self,ion):
        ret = copy.copy(self.fit_type)

        def parameter_initialiser(x, y, p):
            p['y0'] = np.mean(y)
            p['x0'] = self.scan_offset
            p['a'] = (np.max(y) - np.min(y))/2
            p['period'] = (np.max(x) - np.min(x))*0.75
        
        constants = {}
        parameter_bounds = {"a": (0,1), "period": (0.005,0.04)}
        ret.parameter_bounds=parameter_bounds
        ret.parameter_initialiser = parameter_initialiser
        return ret
 
    def analyze_pair(self, ion_pair, p_fits):
        """Analyze and Fit data."""

#        self.set_dataset("monitor.calibration_error", 0, persist=True)

        XX_slots=[common_types.ion_to_slot(i,self.num_ions,one_indexed=False) for i in ion_pair]        

        rf_calib_struct = self.rfsoc_sbc.qiskit_backend.properties().rf_calibration
        gate_tweaks = rf_calib_struct.gate_tweaks.struct

        current_sb_imb = gate_tweaks.loc[tuple(XX_slots),"sideband_amplitude_imbalance"]
                
        phases = [p_fits["x0"][i] for i in [0,1]]
        amplitudes = [p_fits["a"][i] for i in [0,1]]
        periods = [p_fits["period"][i] for i in [0,1]]
        print("phases are {}".format(phases))
        print("amplitudes are {}".format(amplitudes))
        for x in range(len(amplitudes)):
            if amplitudes[x] < 0:
                phases[x] = phases[x] + periods[x]
        print("phases are {}".format(phases))
        new_setting = np.mean(phases)
        if np.isnan(new_setting) or np.abs(new_setting) > 0.1:
            print(
                "calibration failed. returned values is {:4.4f}".format(new_setting)
            )
        else:
            print(
                "sideband_imbalance is originally set to {:4.4f}".format(
                    current_sb_imb
                )
            )
            print("now set the sideband_imbalance to {:4.4f}".format(new_setting))
            log_calibration(
                type="sideband_imbalance",
                operation="set",
                value=float(new_setting),
                gate=str(ion_pair)
            )
            gate_tweaks.loc[tuple(XX_slots),"sideband_amplitude_imbalance"] = new_setting
            gate_tweaks.loc[tuple(XX_slots),"stark_shift"] = 0

        gate_tweaks.to_h5(rf_calib_struct.gate_tweak_path)


class XX_Fidelity_Check(BasicEnvironment, artiq_env.Experiment):
    """Calibrate.XX.Fidelity.Check."""

    # TODO
    data_folder = "raman_awg"
    applet_name = "Raman AWG"
    applet_group = "Raman"
    ylabel = "Population Transfer"
    xlabel = "Analysis Pulse Phase (pi radians)"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                artiq_scan.RangeScan(
                    start=0, stop=1, npoints=11, randomize=False
                )
            )
        )

        self.addressed_ion_indices_str = self.get_argument(
            "Gate pair to be calibrated", artiq_env.StringValue(default="-1,1")
        )

        self.N_gates = self.get_argument(
            "Number of gates to run", artiq_env.NumberValue(default=1)
        )

        super().build()
        self.population_analyzer = PopulationAnalysis(self)
        self.use_AWG = False
        self.use_RFSOC = True

    def prepare(self):

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )
        self.scan_values = list(self.scan_range)
        self.num_steps = len(self.scan_values)
   
        self.addressed_ion_indices = [parse_ion_input(str) for str in self.addressed_ion_indices_str.split(";")]

        if repeated_ions(self.addressed_ion_indices):
            print("Cannot perform parallel calibration of gates involving same ions.")
            assert(False)

        # plot the live parity fringe only of the first pair
        self.num_ions = self.get_dataset("global.AWG.N_ions")
        self.XX_slots=[common_types.ion_to_slot(x,self.num_ions,one_indexed=False) for x in (self.addressed_ion_indices)[0]]

        super().prepare()

    @rpc(flags={"async"})
    def custom_proc_data(self, istep):
        pass

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            # calls basic_environment.experiment_initialize(),
            # which will call AWG.experiment_initialize()
            self.experiment_initialize()
            # initialize the experiment-specific gate parameters
            # and program the AWG
            self.custom_experiment_initialize()
            # run the initialization on the core
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @host_only
    def custom_experiment_initialize(self):
        self.population_analyzer.module_init(
            data_folder=self.data_folder,
            active_pmts=self.pmt_array.active_pmts,
            num_steps=self.num_steps,
            detect_thresh=self.detect_thresh,
            XX_slots=self.XX_slots
        )

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    def pulse_schedule(self, ion_pair, scan_val) -> typing.List[qp.Schedule]:
        with qp.build(self.rfsoc_sbc.qiskit_backend) as out_sched:
            for _i in range(int(self.N_gates)):
                qp.call(
                    multi_qubit.xx_am_gate(
                        ion_pair, theta=np.pi/4
                    )
                )
            for ion in ion_pair:
                qp.call(
                    single_qubit.sk1_gaussian(
                        ion, np.pi / 2, scan_val*np.pi,
                        backend=self.rfsoc_sbc.qiskit_backend
                    )
                )

        return out_sched

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for scan_val in self.scan_values:
            with qp.build(self.rfsoc_sbc.qiskit_backend) as out_sched:
                for pair in self.addressed_ion_indices:
                    qp.call(self.pulse_schedule(pair,scan_val))

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

        self.set_dataset("monitor.calibration_error", 0, persist=True)
        _LOGGER.debug("In XX gate fidelity check analyze phase.")
        (
            parity_amplitude,
            parity_phase,
            parity_y_offset,
        ) = self.population_analyzer.calculate_parity_contrast(
            num_entangled_ions=2,
            num_shots_per_phase=self.num_shots,
            phase_values=np.array(list(self.scan_range)) * np.pi,
            parity_values=self.get_dataset("data.population.parity"),
        )

        # Calculate & plot parity
        x_fit = np.linspace(min(self.scan_range), max(self.scan_range), 100) * np.pi
        y_fit = (
            parity_amplitude.n * np.sin(x_fit * 2 + parity_phase.n) + parity_y_offset.n
        )
        # NOTE: overwrites the previous values, won't be updated till end/analyze
        # OK b/c just set in deprecated population_analysis.py::function_fit
        self.set_dataset("data.population.x_fit", x_fit / np.pi, broadcast=True)
        self.set_dataset(
            "data.population.parity_y_fit", y_fit.reshape((1, -1)), broadcast=True
        )
        self.set_dataset(
            "data.population.parity_std",
            np.full(self.num_steps, parity_amplitude.s),
            broadcast=True,
        )

        counts = np.array(self.get_dataset("data." + self.data_folder + ".raw_counts"))
        threshold_counts = np.where(counts > self.detect_thresh, 1, 0)
        for pair in self.addressed_ion_indices:
            self.analyze_pair(threshold_counts, pair)

    def analyze_pair(self, threshold_counts, ion_pair):
        XX_slots=[common_types.ion_to_slot(i,self.num_ions,one_indexed=False) for i in ion_pair]
        XX_slots_pmt_index = [self.pmt_array.active_pmts.index(x - 9) for x in XX_slots]

        # will be 1 for 11, -1 for 00 and 0 for 10 or 01
        parity_osc = (2*threshold_counts[XX_slots_pmt_index[0],:,:]-1) * (2*threshold_counts[XX_slots_pmt_index[1],:,:] - 1)
        parity_osc = np.transpose(parity_osc)

        (
            parity_amplitude,
            parity_phase,
            parity_y_offset,
        ) = self.population_analyzer.calculate_parity_contrast(
            num_entangled_ions=2,
            num_shots_per_phase=self.num_shots,
            phase_values=np.array(list(self.scan_range)) * np.pi,
            parity_values=parity_osc
        )

        print(
            "parity amplitude: {0}\n"
            "phase_offset: {1}\n"
            "y_offset: {2}".format(parity_amplitude, parity_phase, parity_y_offset)
        )

        log_calibration(
            gate=str(self.addressed_ion_indices_str),
            type="P00,11+P11,00 (coherence) ",
            operation="record",
            value=float(abs(np.round(parity_amplitude.n, 4))),
        )

        log_calibration(
            gate=str(self.addressed_ion_indices_str),
            type="Parity Phase Offset ",
            operation="record",
            value=float(np.round(parity_phase.n, 3)),
        )
        

class Gate_Calibration(artiq_env.EnvExperiment):
    """Auto.Calibration.Gates"""

    def build(self):
        """Initialize experiment & variables."""

        self.gate_list = self.get_argument(
            "XX gates to be calibrated", StringValue(default="2,3;3,4")
        )

        self.do_amp_calib_single = self.get_argument(
            "do single-gate amplitude calibration", BooleanValue(default=True)
        )

        self.do_amp_calib_multi = self.get_argument(
            "do multi-gate amplitude calibration", BooleanValue(default=False),
        )

        self.do_imbalance_calib = self.get_argument(
            "do sideband imbalance calibration", BooleanValue(default=True),
        )

        self.do_stark_calib = self.get_argument(
            "do Stark shift calibration", BooleanValue(default=True),
        )

        self.do_fidelity_check = self.get_argument(
            "do fidelity check", BooleanValue(default=True)
        )

        self.calibration_per_hour = self.get_argument(
            "repeat how many times per hour", NumberValue(default=True)
        )

        self.setattr_device("scheduler")
        self.submitter = Submitter(self.scheduler,__file__)

    @host_only
    @staticmethod
    def split_into_disjunct(lst):
        def test_fnc(e1,e2):
            return e1[0]==e2[0] or e1[1]==e2[1] or e1[0]==e2[1] or e1[1]==e2[0]

        remaining = lst
        indep_sets = []

        while len(remaining) > 0:
            indep_set = []
            new_remaining = []
            for i in range(len(remaining)):
                e = remaining[i]
                found = False
                for j in indep_set:
                    if test_fnc(e,j):
                        found = True
                if not found:
                    indep_set.append(e)
                else:
                    new_remaining.append(e)
            remaining = new_remaining
            indep_sets.append(indep_set)

        return indep_sets

    @host_only
    def resubmit_until_success(
        self,
        experiment_prototype_name: str,
        priority: int = priorities.CALIBRATION,
        hold: bool = False,
        resubmit_on_fail: bool=False,
        **kwargs,
    ):
        
        success = False
        while not success:
            self.submitter.submit(
                    experiment_prototype_name,
                    priority,
                    hold,
                    **kwargs
                )
            success = (not resubmit_on_fail) or (self.get_dataset("monitor.calibration_error") == 0 and (not self.get_dataset("monitor.Lost_Ions")))        
    
    @host_only
    def run(self):
        """Run each enabled calibration step."""

        gate_list = self.gate_list.split(";")
        gate_list = [parse_ion_input(e) for e in gate_list]
        indep_gate_list = self.split_into_disjunct(gate_list)

        for gates in indep_gate_list:
            N_ss = 1
            print("now working on gates: ", gates)
            strGates = ';'.join([str(e[0])+','+str(e[1]) for e in gates])
            if self.do_amp_calib_single:
                self.resubmit_until_success(
                    "XX_Amp_Calibration",
                    hold=True,
                    resubmit_on_fail=True,
                    **{"Gate pair to be calibrated": strGates},
                    **(_AUTO_CALIBRATION_SETTINGS["XX_Amp"].to_dict())
                )

            if self.do_amp_calib_multi:
                self.resubmit_until_success(
                    "XX_Amp_Calibration_Multi",
                    hold=True,
                    resubmit_on_fail=True,
                    **{"Gate pair to be calibrated": strGates},
                    **(_AUTO_CALIBRATION_SETTINGS["XX_Amp"].to_dict())
                )

            if self.do_imbalance_calib:
                N_ss = 1
                self.resubmit_until_success(
                    "XX_Imbalance_Calibration",
                    hold=True,
                    resubmit_on_fail=True,
                    **{"Gate pair(s) to be calibrated": strGates},
                    **(_AUTO_CALIBRATION_SETTINGS["XX_Imbalance"].to_dict())
                )

            if self.do_stark_calib:
                for _ in range(N_ss):
                    self.resubmit_until_success(
                        "XX_StarkShift_Calibration",
                        hold=True,
                        resubmit_on_fail=True,
                        **{"Gate pair(s) to be calibrated": strGates},
                        **(_AUTO_CALIBRATION_SETTINGS["XX_StarkShift"].to_dict())
                    )

            if self.do_fidelity_check:
                self.submitter.submit(
                    "XX_Fidelity_Check",
                    hold=True,
                    **{"Gate pair to be calibrated": strGates},
                    **(_AUTO_CALIBRATION_SETTINGS["XX_Fidelity"].to_dict())
                )

        if self.calibration_per_hour < 0.1:
            return
        else:

            reschedule = self.scheduler.get_status()[self.scheduler.rid]["expid"]

            self.scheduler.submit(
                pipeline_name="main",
                expid=reschedule,
                priority=int(
                    priorities.CALIBRATION_BACKGROUND
                ),  # converts enum to int value if necessary
                due_date=time.time() + 3600 / self.calibration_per_hour,
                flush=False,
            )

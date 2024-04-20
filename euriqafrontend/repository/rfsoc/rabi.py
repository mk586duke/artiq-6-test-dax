import logging
import time
from turtle import back
import typing

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import numpy as np
import oitg.fitting as fit
import qiskit.pulse as qp
import pulsecompiler.qiskit.pulses as pc_pulses
from artiq.experiment import NumberValue, StringValue
from artiq.experiment import TerminationRequested
from artiq.language.core import delay, delay_mu, host_only, kernel
from artiq.language.types import TInt32
from artiq.language.units import MHz, kHz, ns, us
import euriqabackend.devices.keysight_awg.common_types as common_types

import euriqafrontend.fitting as umd_fit
import euriqabackend.waveforms.single_qubit as single_qubit
import euriqabackend.waveforms.multi_qubit as multi_qubit
import euriqabackend.waveforms.conversions as wf_convert
import euriqafrontend.modules.rfsoc as rfsoc
from euriqafrontend.modules.utilities import parse_ion_input
from euriqafrontend.repository.basic_environment import BasicEnvironment

from euriqafrontend.modules.population_analysis import PopulationAnalysis
_LOGGER = logging.getLogger(__name__)


import euriqabackend.waveforms.delay as delay_gate
from oitg.fitting.FitBase import FitError
import  oitg.fitting.FitBase as FitBase
import oitg.fitting as fit
from scipy.optimize import minimize_scalar
from scipy.signal import lombscargle


#####

def axial_rabi_parameter_initialiser(x, y, p):
    t_min = np.amin(x)
    t_range = np.amax(x) - t_min
    if t_range == 0.0:
        t_range = 1.0

    # Estimate frequency. Starting with a Lomb-Scargle periodogram (which
    # supports irregularly-spaced samples), we pick the strongest frequency
    # component which leads to a pi time larger than t_min.
    #
    # TODO: Could use better heuristics for frequency range based on minimum
    # distance between points -> aliasing.
    freq = np.pi / t_range
    freqs = np.linspace(0.1 * freq, 20 * freq, 2 * len(x))
    pgram = lombscargle(x, y, freqs, precenter=True)
    freq_order = np.argsort(-pgram)
    for f in freqs[freq_order]:
        t = 2 * np.pi / f
        if t / 2 > t_min:
            p["t_period"] = t
            break

    p["y_upper"] = np.clip(2 * np.mean(y), 0, 1)

    # TODO: Estimate decay time constant using RMS amplitude from global mean
    # in first and last chunk.
    p["theta"] = 0.05

def axial_rabi_fitting_function(x, p):
    y_lower = 0
    ph = 2 * np.pi / p["t_period"] * x

    y = p["y_upper"]/2 - (p["y_upper"] - y_lower) / 2 * (
        np.cos(ph) + p["theta"]*ph*np.sin(ph)
        ) / (1 +p["theta"]*p["theta"]*ph*ph)

    return y

def axial_rabi_derived_parameter_function(p, p_err):
    non_decaying_pi_time  = p["t_period"] / 2

    # Compute the point of maximum population transfer (minimum in y) which
    # will be slightly shifted towards zero in the face of non-zero tau_decay.
    fit = minimize_scalar(lambda t: axial_rabi_fitting_function(t, p), method="brent",
        bracket=[0.9*non_decaying_pi_time, non_decaying_pi_time])
    if fit.success:
        p["t_pi"] = fit.x
    else:
        p["t_pi"] = non_decaying_pi_time

    # This is just a Gaussian error propagation guess.
    p_err["t_pi"] = np.sqrt((p_err["t_period"] / 2)**2)
    return p, p_err

axial_rabi_flop = FitBase.FitBase(
    ["t_period", "y_upper", "theta"],
    axial_rabi_fitting_function, parameter_initialiser=axial_rabi_parameter_initialiser,
    derived_parameter_function=axial_rabi_derived_parameter_function,
    parameter_bounds={"t_period": (0, np.inf), "y_upper": (0, 1), "theta": (0, np.inf)})



class RFSoCRabi(BasicEnvironment, artiq_env.Experiment):
    """rfsoc.Rabi (by Amplitude)

    Plays Rabi pulse(s) on **multiple** ions in a system. Scans the duration
    of the Rabi pulse as given as an argument, with a given detuning and frequency.

    Plots & analyzes the result.
    """

    data_folder = "rfsoc_rabi"
    applet_name = "Raman Rabi RFSOC"
    applet_group = "RFSOC"
    fit_type = umd_fit.rabi_flop
    units = 1
    ylabel = "Population Transfer"
    xlabel = "Pulse Duration"

    def build(self):
        """Initialize experiment & variables."""
        # Add RFSoC arguments
        super().build()
        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=10 * ns, stop=20 * us, npoints=20, randomize=False,
                ),
                unit="us",
                global_min=10 * ns,
            ),
        )
        self.setattr_argument(
            "ions_to_address",
            artiq_env.StringValue("0"),
            tooltip="center-index! use python range notation e.g. -3,4:6 == [-3, 4, 5]",
        )

        self.setattr_argument(
            "detuning",
            NumberValue(
                0,
                unit="MHz",
                min=-10 * MHz,
                max=10 * MHz,
                step=0.0025 * MHz,
                ndecimals=6,
            ),
            tooltip="Must set sideband_order != 0 to apply the detuning.",
        )

        self.setattr_argument(
            "sideband_order",
            NumberValue(0, scale=1, min=-3, max=3, step=1, ndecimals=0),
            tooltip="Must be != 1 if using detuning.",
        )

        self.setattr_argument("rabi_global_amplitude", rfsoc.AmplitudeArg(default=0.3))
        self.setattr_argument("rabi_individual_amplitude", rfsoc.AmplitudeArg())

        self.setattr_argument(
            "number_of_gates",
            artiq_env.NumberValue(default=1, step=1, ndecimals=0),
            tooltip="Ignored if using 'calibrate' gate type",
        )

        self.setattr_argument(
            "pulse_shape",
            artiq_env.EnumerationValue(
                [
                    "square_rabi",
                    # Following types not yet supported
                    "gaussian_shaped_rabi",
                    # "phase-insensitive_Rabi",
                    # "square_SK1",
                    # "gaussian_shaped_SK1",
                    # "cross_SK1",
                ],
                default="square_rabi",
            ),
        )
        self.setattr_argument("phase_insensitive", artiq_env.BooleanValue(default=False))

        # Following arguments unused for now b/c their pulse shape is not supported
        # self.setattr_argument(
        #     "SK1_scan_type",
        #     artiq_env.EnumerationValue(
        #         ["static", "theta", "ind_amp"], default="static"
        #     ),
        #     group="SK1",
        # )

        # self.setattr_argument(
        #     "shaped_Rabi_scan_type",
        #     artiq_env.EnumerationValue(
        #         ["static", "global", "envelope", "ind_amp"], default="static"
        #     ),
        #     group="Shaped Rabi",
        # )

        # self.setattr_argument(
        #     "shaped_Rabi_envelope_duration",
        #     NumberValue(0, scale=1, min=0, max=1000, step=1, ndecimals=3, unit="us"),
        #     group="Shaped Rabi",
        # )

        # self.setattr_argument(
        #     "shaped_Rabi_global_delay",
        #     NumberValue(0, scale=1, min=0, max=1000, step=1, ndecimals=3, unit="us"),
        #     group="Shaped Rabi",
        # )

        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(True))

    def prepare(self):
        # Use & interpret arguments
        self.ions_to_address = parse_ion_input(self.ions_to_address)

        # Magic BasicEnvironment attributes
        self.num_steps = len(self.scan_range)
        self.scan_values = list(self.scan_range)

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_experiment_initialize()
            time.sleep(0.1)

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @host_only
    def custom_experiment_initialize(self):
        pass

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        """Generate a multi-ion Rabi schedule."""
        schedule_list = []
        for duration in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                if self.pulse_shape == "square_rabi":
                    for ion in self.ions_to_address:
                        for _ in range(self.number_of_gates):
                            qp.call(
                                single_qubit.square_rabi_by_amplitude(
                                    ion_index=ion,
                                    duration=duration,
                                    phase=0.0,
                                    detuning=self.detuning,
                                    phase_insensitive=self.phase_insensitive,
                                    sideband_order=self.sideband_order,
                                    individual_amp=self.rabi_individual_amplitude,
                                    global_amp=self.rabi_global_amplitude,
                                )
                            )
                elif self.pulse_shape == "gaussian_shaped_rabi":
                    # uses maximum amplitude here for the Gaussian.
                    # Should be slower than Square
                    assert duration >= 200e-9, (
                        "Gaussian pulses are subdivided into shorter pulses, "
                        "and have a minimum duration of ~100 clk cycles"
                    )
                    for ion in self.ions_to_address:
                        for _ in range(self.number_of_gates):
                            global_channel = qp.control_channels()[0]
                            individual_channel = qp.drive_channel(ion)
                            duration_dt = qp.seconds_to_samples(duration)
                            qp.play(
                                pc_pulses.LinearGaussian(
                                    duration_dt, amp=self.rabi_individual_amplitude
                                ),
                                individual_channel,
                            )
                            with qp.frequency_offset(
                                self.detuning * self.sideband_order, global_channel
                            ):
                                qp.play(
                                    qp.Constant(
                                        duration_dt, amp=self.rabi_global_amplitude
                                    ),
                                    global_channel,
                                )

                # elif self.pulse_shape == "shaped_Rabi":
                #     if self.shaped_Rabi_scan_type == "envelope":
                #         scan_val = gate.Rabi_AM.ScanParameter.envelope_duration
                #     elif self.shaped_Rabi_scan_type == "global":
                #         scan_val = gate.Rabi_AM.ScanParameter.global_duration
                #     elif self.shaped_Rabi_scan_type == "ind_amp":
                #         scan_val = gate.Rabi_AM.ScanParameter.ind_amplitude
                #     else:
                #         scan_val = gate.Rabi_AM.ScanParameter.static

                #     self.AWG.rf_compiler.rabi_am_exp(
                #         slots=self.slots_to_address,
                #         detuning=self.detuning_MHz,
                #         detuning_off=5,
                #         global_amp_off=-1,
                #         sideband_order=self.sideband_order,
                #         envelope_type_int=int(InterpFunction.full_Gaussian),
                #         envelope_duration=self.shaped_Rabi_envelope_duration,
                #         envelope_scale=1,
                #         global_delay=self.shaped_Rabi_global_delay,
                #         global_duration=self.shaped_Rabi_envelope_duration,
                #         scan_parameter_int=int(scan_val),
                #         min_value=self.scan_min_t,
                #         max_value=self.scan_max_t,
                #         N_points=self.num_steps_to_use,
                #     )

                # elif self.pulse_shape == "phase-insensitive_Rabi":
                #     self.AWG.rf_compiler.rabi_pi_exp(
                #         slots=self.slots_to_address,
                #         detuning=self.detuning_MHz,
                #         sideband_order=self.sideband_order,
                #         scan_parameter_int=int(gate.Rabi_PI.ScanParameter.duration),
                #         min_value=self.scan_min_t,
                #         max_value=self.scan_max_t,
                #         N_points=self.num_steps_to_use,
                #     )

                # elif self.pulse_shape == "square_SK1":

                #     if self.SK1_scan_type == "theta":
                #         scan_val = gate.SK1.ScanParameter.theta
                #     elif self.SK1_scan_type == "ind_amp":
                #         scan_val = gate.SK1.ScanParameter.ind_amplitude
                #     else:
                #         scan_val = gate.SK1.ScanParameter.static

                #     self.AWG.rf_compiler.SK1_exp(
                #         slots=self.slots_to_address,
                #         theta=np.pi/2,
                #         phi=0,
                #         scan_parameter_int=int(scan_val),
                #         min_value=self.scan_min_t,
                #         max_value=self.scan_max_t,
                #         N_points=self.num_steps_to_use,
                #     )

                # elif self.pulse_shape == "shaped_SK1":
                #     if self.SK1_scan_type == "ind_amp":
                #         scan_val = gate.SK1_AM.ScanParameter.ind_amplitude
                #     elif self.SK1_scan_type == "theta":
                #         scan_val = gate.SK1_AM.ScanParameter.theta
                #     else:
                #         scan_val = gate.SK1_AM.ScanParameter.static
                #     self.AWG.rf_compiler.SK1_am_exp(
                #         slots=self.slots_to_address,
                #         phi=i*phi_opt,
                #         theta=np.pi/2,
                #         use_global_segment_durations=False,
                #         scan_parameter_int=int(scan_val),
                #         min_value=self.scan_min_t,
                #         max_value=self.scan_max_t,
                #         N_points=self.num_steps_to_use,
                #     )

                # elif self.pulse_shape == "cross_SK1":

                #     if self.SK1_scan_type == "theta":
                #         scan_val = rf_common.CrossSK1ScanParameter.theta
                #     elif self.SK1_scan_type == "ind_amp":
                #         scan_val = rf_common.CrossSK1ScanParameter.ind_amplitude
                #     else:
                #         scan_val = rf_common.CrossSK1ScanParameter.static

                #     self.AWG.rf_compiler.cross_SK1(
                #         slots=self.slots_to_address,
                #         phi=0,
                #         theta=np.pi/2,
                #         use_AM=True,
                #         use_global_segment_durations=False,
                #         scan_parameter_int=int(scan_val),
                #         min_value=self.scan_min_t,
                #         max_value=self.scan_max_t,
                #         N_points=self.num_steps_to_use,
                #     )
                else:
                    raise NotImplementedError(f"Pulse type {self.pulse_shape}")

            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        # print("Starting shot")
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    def analyze(self):
        """Analyze and Fit data."""
        super().analyze()
        full_2pi_periods = np.array(self.p_all["t_period"])
        thetas = np.array(self.p_all["theta"])
        num_fits = len(full_2pi_periods)
        for i, (t_pi, tau_decay) in enumerate(
            zip(full_2pi_periods / 2, self.p_all["tau_decay"])
        ):
            _LOGGER.info(
                "Fit %i/%i: Tpi= %.3f us, tau= %.3f us",
                i + 1,
                num_fits,
                t_pi * 1e6,
                tau_decay,
            )

        #print("Pi Times: %s", list(full_2pi_periods / 2))
        print(list(1e6*full_2pi_periods / 2))
        print(list(thetas))


class CalibrateRFSoCRabiAmplitude(RFSoCRabi):
    """rfsoc.CalibratePiTime

    Calibrates the Rabi pi time for a given ion(s). Uses a square pulse at a
    pre-determined amplitude to get roughly 50 us pi time, which is then up-scaled
    to get the maximum pi time, assuming nonlinearity correction is applied.
    """

    data_folder = "rfsoc_rabi"
    applet_name = "Raman Rabi RFSOC"
    applet_group = "RFSOC"
    fit_type = umd_fit.rabi_flop
    ylabel = "Population Transfer"
    xlabel = "Pulse Duration"

    def build(self):
        """Initialize experiment & variables."""
        # Add RFSoC arguments
        super().build()
        self.delete_arguments("number_of_gates", "rabi_individual_amplitude")
        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=10 * ns, stop=100 * us, npoints=21, randomize=False,
                ),
                unit="us",
                global_min=10 * ns,
            ),
        )
        self.setattr_argument(
            "ions_to_address",
            artiq_env.StringValue("0"),
            tooltip="center-index! use python range notation e.g. -3,4:6 == [-3, 4, 5]",
        )
        self.setattr_argument(
            "clear_old_pi_times", artiq_env.BooleanValue(default=False)
        )
        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=True))
        # override nonlinearity -> off
        self.setattr_argument(
            "schedule_transform_aom_nonlinearity",
            artiq_env.BooleanValue(default=False),
            group="RFSoC",
        )

        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(True))
        self.use_AWG = False
        self.setattr_argument("rabi_global_amplitude", rfsoc.AmplitudeArg(default=0.63))
        self.max_nominal_amplitude = 0.5
        self.amplitude_derating = 0.01
        self.calibration_amplitude = (
            self.max_nominal_amplitude * self.amplitude_derating
        )
        self.rabi_individual_amplitude = self.calibration_amplitude
        self.setattr_argument(
            "pulse_shape",
            artiq_env.EnumerationValue(["square_rabi"], default="square_rabi"),
        )

    def prepare(self):
        super().prepare()
        # 1,4,7,10,13;2,5,8,11,14;3,6,9,12,15
        print(self.ions_to_address)
        rf_calib = self.rfsoc_sbc._rf_calib
        self.pi_time_dataset = rf_calib["rabi.pi_time_individual.key"]
        num_individual_aom_channels = rf_calib[
            "other.individual_aom_channel_count.value"
        ]
        if self.clear_old_pi_times:
            self.set_dataset(
                self.pi_time_dataset,
                np.full(
                    (num_individual_aom_channels,), np.nan
                ),  # [np.nan] * num_individual_aom_channels,
                persist=True,  # self.set_globals,
                archive=True,
            )
        # refresh RF Calibration w/ updated datasets
        self.rfsoc_sbc._rf_calib._init_dataset_args(self._HasEnvironment__dataset_mgr)
        self.calibrated_Tpi = self.get_dataset(self.pi_time_dataset)

    def analyze(self):
        super().analyze()
        rf_calib = self.rfsoc_sbc._rf_calib
        # haven't bothered to figure out the code for even # of ions.
        # Could probably use the qiskit backend.configuration()'s
        # center_index_to_zero_index method
        assert (
            rf_calib["other.number_of_ions.value"] % 2 == 1
        ), "Even number of ions not yet supported"
        rf_calib = self.rfsoc_sbc._rf_calib
        center_aom_index = rf_calib["other.center_aom_index.value"]
        center_pmt_channel = rf_calib["other.center_pmt_channel.value"]
        ion_aom_channel_map = {
            ion_center_idx: center_aom_index + ion_center_idx
            for ion_center_idx in self.ions_to_address
        }
        ion_pmt_map = {
            ion_center_idx: center_pmt_channel + ion_center_idx
            for ion_center_idx in self.ions_to_address
        }
        ion_fit_index = {
            ion_center_idx: self.pmt_array.active_pmts.index(
                ion_pmt_map[ion_center_idx]
            )
            for ion_center_idx in self.ions_to_address
        }

        for ion in self.ions_to_address:
            # divide by 2 to convert from 2-pi cycle -> pi time
            pi_time_measured = self.p_all["t_period"][ion_fit_index[ion]] / 2
            # assumes linearity, calculate the min pi time at max individual amplitude
            expected_min_pi_time = pi_time_measured * self.calibration_amplitude
            _LOGGER.info(
                "Measured pi time %.3E s for ion %d (center-index).",
                pi_time_measured,
                ion,
            )
            _LOGGER.info("Calculated min pi time: %.3E", expected_min_pi_time)
            self.calibrated_Tpi[ion_aom_channel_map[ion]] = expected_min_pi_time

        if self.set_globals:
            _LOGGER.info(
                "Updating dataset %s with pi times", self.pi_time_dataset,
            )
            self.set_dataset(
                self.pi_time_dataset, self.calibrated_Tpi, persist=True, archive=True
            )


class RFSoCRabiCalibrated(RFSoCRabi):
    """rfsoc.Rabi (by Rabi Frequency)

    Calibrated version of a RFSoC Rabi, controlling amplitude by specifying
    rabi frequency.
    Requires that CalibrateRFSoCRabiAmplitude has been run first.
    """

    def build(self):
        super().build()
        # amplitude arguments aren't relevant for this class
        self.delete_arguments("rabi_global_amplitude", "rabi_individual_amplitude")
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=10 * ns, stop=20 * us, npoints=20, randomize=False,
                ),
                unit="us",
                global_min=10 * ns,
            ),
        )
        self.setattr_argument(
            "rabi_frequency", artiq_env.NumberValue(default=50 * kHz, unit="kHz")
        )

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedules = []
        backend = self.rfsoc_sbc.qiskit_backend

        for duration in self.scan_range:
            with qp.build(backend) as sched:
                # sequential rabi pulses on given ions (NOT simultaneous)
                for ion in self.ions_to_address:
                    for _ in range(self.number_of_gates):
                        if self.pulse_shape == "square_rabi":
                            qp.call(
                                single_qubit.square_rabi_by_rabi_frequency(
                                    ion,
                                    duration,
                                    rabi_frequency=self.rabi_frequency,
                                    backend=backend,
                                )
                            )
                        elif self.pulse_shape == "gaussian_shaped_rabi":
                            # uses maximum amplitude here for the Gaussian.
                            # Should be slower than Square
                            assert duration >= 200e-9, (
                                "Gaussian pulses are subdivided into shorter pulses, "
                                "and have a minimum duration of ~100 clk cycles"
                            )
                            global_channel = qp.control_channels()[0]
                            individual_channel = qp.drive_channel(ion)
                            duration_dt = qp.seconds_to_samples(duration)
                            ind_amp = wf_convert.rabi_frequency_to_amplitude(
                                self.rabi_frequency, individual_channel, backend
                            )
                            rf_calib = backend.properties().rf_calibration
                            rabi_global_amp = (
                                rf_calib.rabi.global_amplitude_single_tone.value
                            )
                            qp.play(
                                pc_pulses.LinearGaussian(duration_dt, amp=ind_amp),
                                individual_channel,
                            )
                            with qp.frequency_offset(
                                self.detuning * self.sideband_order, global_channel
                            ):
                                qp.play(
                                    qp.Constant(duration_dt, amp=rabi_global_amp),
                                    global_channel,
                                )
                        else:
                            raise NotImplementedError(
                                f"Other pulse shapes ({self.pulse_shape}) not yet supported"
                            )

            schedules.append(sched)

        return schedules


class RFSoCSK1(RFSoCRabi):
    """rfsoc.SK1 (by Rabi Frequency)"""

    def build(self):
        super().build()
        # amplitude arguments aren't relevant for this class
        self.delete_arguments(
            "rabi_global_amplitude", "rabi_individual_amplitude",
        )
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=1 * kHz, stop=400 * kHz, npoints=20, randomize=False,
                ),
                unit="",
            ),
        )
        self.setattr_argument(
            "scan_type",
            artiq_env.EnumerationValue(
                choices=["theta", "phi", "rabi_frequency"], default="rabi_frequency"
            ),
        )
        self.setattr_argument(
            "pulse_shape",
            artiq_env.EnumerationValue(
                choices=["square_sk1", "gaussian_shaped_sk1"], default="square_sk1"
            ),
        )
        self.setattr_argument(
            "rabi_frequency", artiq_env.NumberValue(default=200 * kHz, unit="kHz")
        )
        self.setattr_argument(
            "theta",
            rfsoc.PhaseArg(default=0.0),
            tooltip="Overridden by theta scan type if selected",
        )
        self.setattr_argument(
            "phi",
            rfsoc.PhaseArg(default=0.0),
            tooltip="SK1 Azimuthal angle. Overriden by phi scan type if selected.",
        )

    def prepare(self):
        self.xlabel = self.scan_type
        super().prepare()

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedules = []
        backend = self.rfsoc_sbc.qiskit_backend

        for scan_val in self.scan_range:
            with qp.build(backend) as sched:
                # sequential rabi pulses on given ions (NOT simultaneous)
                theta = self.theta
                phi = self.phi
                individual_rabi_freq = self.rabi_frequency
                if self.scan_type == "theta":
                    theta = scan_val
                elif self.scan_type == "phi":
                    phi = scan_val
                else:
                    individual_rabi_freq = scan_val

                for ion in self.ions_to_address:
                    for _ in range(self.number_of_gates):
                        if self.pulse_shape == "square_sk1":
                            qp.call(
                                single_qubit.sk1_square_by_rabi_frequency(
                                    ion,
                                    theta,
                                    phi,
                                    individual_rabi_frequency=individual_rabi_freq,
                                    backend=backend,
                                )
                            )
                        elif self.pulse_shape == "gaussian_shaped_sk1":
                            qp.call(
                                single_qubit.sk1_gaussian(
                                    ion,
                                    theta,
                                    phi,
                                    #individual_rabi_frequency=individual_rabi_freq,
                                    ind_amp_multiplier = scan_val,
                                    backend=backend,
                                )
                            )
                        else:
                            raise NotImplementedError(
                                f"Other pulse shapes ({self.pulse_shape}) not yet supported"
                            )

            schedules.append(sched)

        return schedules

class RFSOCXXTuneup(BasicEnvironment, artiq_env.Experiment):
    """rfsoc.XXTuneup."""

    data_folder = "rfsoc_rabi"
    applet_name = "Raman Rabi RFSOC"
    applet_group = "RFSOC"
    fit_type = umd_fit.rabi_flop
    units = 1
    ylabel = "Population Transfer"
    xlabel = "Pulse Duration"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0, stop=5, npoints=20, randomize=False, seed=int(time.time())
                ),
                unit="",
                ndecimals=4
                #global_min=-1,
            ),
        )

        self.XX_ions_input = self.get_argument(
            "Ions to be entangled in 0-notation",
            StringValue(default="-6,-5"),
        )

        self.setattr_argument("N_gates", artiq_env.NumberValue(default=1))

        self.setattr_argument(
            "scan_parameter",
            artiq_env.EnumerationValue(
                [
                    "None",
                    "N gates",
                    "Analysis pulse phase (pi rad)",
                    "Duration adjust (additive)",
                    "Stark shift (additive)",
                    "Differential Stark shift (additive)",
                    "Motional freq adjust (additive)",
                    "Ind amplitude multiplier (multiplicative)",
#                    "Ind amplitude imbalance (additive)",
                    "Global amplitude (0-1000)",
                    "Sideband imbalance (additive)",
#                    "Pre-experiment delay"
                ],
                default="None",
            ),
        )

        self.setattr_argument(
            "manual_tweak",
            artiq_env.EnumerationValue(
                [
                    "None",
#                    "XX_duration_us",
                    "individual_amplitude_multiplier",
                    "global_amplitude",
                    "sideband_amplitude_imbalance",
                    "stark_shift",
                    "stark_shift_differential",
                    "motional_frequency_adjust",
                ],
                default="None",
            ),
        )

        self.setattr_argument("manual_tweak_value", artiq_env.NumberValue(default=1,ndecimals=6),)

        self.setattr_argument("analysis_pulse",
        artiq_env.EnumerationValue(["none","square_SK1","AM_SK1"], default="none")
        )

        self.setattr_argument("analysis_pulse_phase", artiq_env.NumberValue(default=0))

        super().build()
        self.population_analyzer = PopulationAnalysis(self)
        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(True))

    @staticmethod
    def parse_pmt_input(s: str) -> list :
        """
        Args:
            s (str): A Comma seperated string of inters. Python notation i:j can be used to indicate ranges

        Returns:
            A list of integers generated from the input string

            Example:
                >>parse_pmt_input("-9,-5,0:4")
                [-9, -5, 0, 1, 2, 3, 4]
        """
        nums = []
        for x in map(str.strip, s.split(",")):
            try:
                i = int(x)
                nums.append(i)
            except ValueError:
                if ":" in x:
                    xr = list(map(str.strip, x.split(":")))
                    nums.extend(range(int(xr[0]), int(xr[1]) + 1))
                else:
                    _LOGGER.warning("Unknown string format for PMT input: {0}".format(x))
        return nums

    def prepare(self):
        #self.scanning_delay = False
#        self.scan_values_mu = [self.core.seconds_to_mu(t_s * 1e-6) for t_s in self.scan_range]

        if self.scan_parameter == "N gates":
            self.xlabel = "Number of gates"
        elif self.scan_parameter == "Analysis pulse phase (pi rad)":
            self.xlabel = "Analysis pulse phase (pi rad)"
        elif self.scan_parameter == "Duration adjust (additive)":
            self.xlabel = "Duration adjust (us)"
        elif self.scan_parameter == "Stark shift (additive)":
            self.xlabel = "Stark shift (Hz)"
        elif self.scan_parameter == "Differential Stark shift (additive)":
            self.xlabel = "Differential Stark shift (MHz)"
        elif self.scan_parameter == "Motional freq adjust (additive)":
            self.xlabel = "Motional freq adjust (MHz)"
        elif self.scan_parameter == "Ind amplitude multiplier (multiplicative)":
            self.xlabel = "Ind. amplitude multiplier"
        elif self.scan_parameter == "Ind amplitude imbalance (additive)":
            self.xlabel = "Ind. amplitude imbalance"
        elif self.scan_parameter == "Global amplitude (0-1000)":
            self.xlabel = "Global amplitude"
        elif self.scan_parameter == "Sideband imbalance (additive)":
            self.xlabel = "Sideband imbalance"
        elif self.scan_parameter == "Pre-experiment delay":
            self.xlabel = "Experiment delay (us)"
            self.scanning_delay = True

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )
        # self.scan_values = [np.int32(int(self.awg_scan_length)-t-1) for t in range(int(self.awg_scan_length))]
        self.scan_values = [t_s for t_s in self.scan_range]
        self.num_steps = len(self.scan_values)
        self.scan_min_val = self.scan_values[0]
        self.scan_max_val = self.scan_values[-1]

        self.XX_indices = parse_ion_input(self.XX_ions_input)
        num_ions = self.get_dataset("global.AWG.N_ions")
        self.XX_slots=[common_types.ion_to_slot(x,num_ions,one_indexed=False) for x in self.XX_indices]
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

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:

        schedules = []
        backend = self.rfsoc_sbc.qiskit_backend

        individual_rabi_freq = 200e3
        gate_tweak_path = backend.properties().rf_calibration.gate_tweak_path

        if self.manual_tweak!="None":
            backend.properties().rf_calibration.gate_tweaks.struct.loc[tuple(self.XX_slots),self.manual_tweak] = self.manual_tweak_value
            backend.properties().rf_calibration.gate_tweaks.struct.to_h5(gate_tweak_path)

        equilibrated_scan_values = self.scan_values
        for scan_val in equilibrated_scan_values:
            with qp.build(backend) as sched:
                if self.scan_parameter == "Analysis pulse phase (pi rad)":
                    analysis_phase = scan_val
                else:
                    analysis_phase = self.analysis_pulse_phase

                if self.scan_parameter == "N gates":
                    Ngates = scan_val
                else:
                    Ngates = self.N_gates

                #self.analysis_pulse == "square_SK1"
                # qp.call(
                #     single_qubit.sk1_square_by_rabi_frequency(
                #         self.XX_indices[0],
                #         np.pi/2,
                #         0,
                #         individual_rabi_frequency=individual_rabi_freq,
                #         backend=backend,
                #     )
                # )
                # qp.call(
                #     single_qubit.sk1_square_by_rabi_frequency(
                #         self.XX_indices[1],
                #         np.pi/2,
                #         0,
                #         individual_rabi_frequency=individual_rabi_freq,
                #         backend=backend
                #     )
                # )
                for i in range(int(Ngates)):
                    if self.scan_parameter == "Stark shift (additive)":
                        qp.call(
                            multi_qubit.xx_am_gate(
                                self.XX_indices,
                                positive_gate=True,
                                theta=np.pi/4,
                                stark_shift=scan_val,
                            )
                        )
                    elif self.scan_parameter == "Differential Stark shift (additive)":
                        qp.call(
                            multi_qubit.xx_am_gate(
                                self.XX_indices,
                                positive_gate=True,
                                theta=np.pi/4,
                                stark_shift_differential=scan_val
                            )
                        )
                    elif self.scan_parameter == "Motional freq adjust (additive)":
                        qp.call(
                            multi_qubit.xx_gate(
                                self.XX_indices,
                                theta=np.pi/4,
                                positive_gate=True,
                                motional_frequency_adjustment=scan_val
                            )
                        )
                    elif self.scan_parameter == "Ind amplitude multiplier (multiplicative)":
                        qp.call(
                            multi_qubit.xx_am_gate(
                                self.XX_indices,
                                positive_gate=True,
                                theta=np.pi/4,
                                individual_amplitude_multiplier=scan_val,
                            )
                        )
                    #elif self.scan_parameter == "Ind amplitude imbalance (additive)":
                    elif self.scan_parameter == "Global amplitude (0-1000)":
                        qp.call(
                            multi_qubit.xx_am_gate(
                                self.XX_indices,
                                positive_gate=True,
                                theta=np.pi/4,
                                global_amplitude=scan_val
                            )
                        )
                    elif self.scan_parameter == "Sideband imbalance (additive)":
                        qp.call(
                            multi_qubit.xx_am_gate(
                                self.XX_indices,
                                positive_gate=True,
                                theta=np.pi/4,
                                sideband_amplitude_imbalance=scan_val
                            )
                        )
                    else:
                        qp.call(
                            multi_qubit.xx_am_gate(
                                self.XX_indices,
                                positive_gate=True,
                                theta=np.pi/4,
                            )
                        )
                if self.analysis_pulse!="none":
                    if self.analysis_pulse != "square_SK1":
                        qp.call(
                            single_qubit.sk1_gaussian(
                                self.XX_indices[0],np.pi/2,
                                analysis_phase,
                                backend=backend,
                            )
                        )
                        qp.call(
                            single_qubit.sk1_gaussian(
                                self.XX_indices[1],np.pi/2,
                                analysis_phase,
                                backend=backend,
                            )
                        )
                    else:
                        qp.call(
                            single_qubit.sk1_square_by_rabi_frequency(
                                self.XX_indices[0],
                                np.pi/2,
                                analysis_phase,
                                individual_rabi_frequency=individual_rabi_freq,
                                backend=backend,
                            )
                        )
                        qp.call(
                            single_qubit.sk1_square_by_rabi_frequency(
                                self.XX_indices[1],
                                np.pi/2,
                                analysis_phase,
                                individual_rabi_frequency=individual_rabi_freq,
                                backend=backend,
                            )
                        )
            schedules.append(sched)

        return schedules

    @host_only
    def custom_experiment_initialize(self):

        self.population_analyzer.module_init(data_folder=self.data_folder, active_pmts=self.pmt_array.active_pmts,
            num_steps=self.num_steps,
            detect_thresh=self.detect_thresh, XX_slots=self.XX_slots)

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        # print("Starting shot")
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    def analyze(self):
        """Analyze and Fit data."""
        if self.scan_parameter == "Analysis pulse phase (pi rad)":
            #elf.population_analyzer.sine_fit("parity")
            self.population_analyzer.XXparity_fit("parity")

        # Assume if we are doing SK1 pulses and sweeping the sideband imbalance
        # then the phase of the SK1 is 0 and we want a linear fit to find the null
        elif (self.scan_parameter == "Sideband imbalance (additive)") and ("analysis_pulse" != "none"):
            fit_param = self.population_analyzer.linear_fit("parity")
            a = fit_param[0]
            b = fit_param[1]
            optimal_sb_imb = -b/a

            current_sb_imb = self.AWG.rf_compiler.get_XX_tweak(
                self.XX_slots, "sideband_amplitude_imbalance"
            )["sideband_amplitude_imbalance"]
            print("Current sideband imbalance  = {0}".format(current_sb_imb))
            print("Updating sideband imbalance = {0}".format(optimal_sb_imb))
            self.AWG.rf_compiler.set_XX_tweak(self.XX_slots, sideband_amplitude_imbalance=optimal_sb_imb)

        # super().analyze()
        # for ifit in range(len(self.p_all["x0"])):
        #     _LOGGER.info(
        #     "Fit %i:\n\tTpi = (%f +- %f) us\n\ttau = (%f +- %f) us",
        #     ifit,
        #     self.p_all["t_period"][ifit] * 0.5e6,
        #     self.p_error_all["t_period"][ifit] * 0.5e6,
        #     self.p_all["tau_decay"][ifit] * 1e6,
        #     self.p_error_all["tau_decay"][ifit] * 1e6,
        # )

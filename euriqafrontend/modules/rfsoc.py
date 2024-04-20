"""ARTIQ module for simplifying access & control of the RF chain using RFSoC.

Compiles for the Sandia Octet RFSoC gateware. Meant to replace the old AWG & switch
network combination pre-2021.
"""
import copy
import itertools
import functools
import logging
import pathlib
import typing
import cProfile, pstats

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import more_itertools
import numpy as np
import jaqalpaw.utilities.parameters as OctetParams
import pulsecompiler.qiskit.backend as qbe
import pulsecompiler.qiskit.pulses as pc_pulse
import pulsecompiler.qiskit.schedule_converter as octetconverter
import pulsecompiler.rfsoc.structures.channel_map as rfsoc_mapping
import pulsecompiler.rfsoc.structures.splines as spl
import pulsecompiler.rfsoc.tones.upload as uploader
import pulsecompiler.rfsoc.tones.tonedata as tones
import pulsecompiler.rfsoc.tones.record as record
from pulsecompiler.qiskit.configuration import BackendConfigCenterIndex, QuickConfig
import pulsecompiler.qiskit.transforms.nonlinearity as pc_nonlinearity
import qiskit.pulse as qp
import qiskit.pulse.transforms as qp_transforms
import qiskit.qobj as qobj
import sipyco.pyon as pyon
from artiq.language.core import host_only, kernel
from pulsecompiler.qiskit.configuration import QuickConfig
from qiskit.assembler import disassemble

import euriqabackend.devices.keysight_awg.gate_parameters as gate_params
import euriqabackend.waveforms.conversions as wf_convert
import euriqafrontend.settings.calibration_box as calibrations
from euriqafrontend.modules.generic import EURIQAModule
from euriqabackend import _EURIQA_LIB_DIR
from euriqafrontend import EURIQA_NAS_DIR
from euriqafrontend.settings import RF_CALIBRATION_PATH


_LOGGER = logging.getLogger(__name__)


ScheduleList = typing.List[qp.Schedule]
ScheduleOrList = typing.Union[qp.Schedule, ScheduleList]
CompiledScheduleList = typing.List[tones.ChannelSequence]
QiskitChannel = typing.Union[qp.DriveChannel, qp.ControlChannel]

# Utility functions to default NumberValues to useful Amplitudes/Frequencies
AmplitudeArg = functools.partial(
    artiq_env.NumberValue, default=1.0, min=-1.0, max=1.0, ndecimals=4
)
AmplitudeScan = functools.partial(
    artiq_scan.Scannable, scale=1.0, global_min=-1.0, global_max=1.0, ndecimals=4
)
FrequencyArg = functools.partial(
    artiq_env.NumberValue,
    default=0.0,
    unit="MHz",
    min=-OctetParams.CLKFREQ,
    max=OctetParams.CLKFREQ,
    ndecimals=9,
)
FrequencyScan = functools.partial(
    artiq_scan.Scannable,
    unit="MHz",
    global_min=-OctetParams.CLKFREQ,
    global_max=OctetParams.CLKFREQ,
    ndecimals=9,
)
PhaseArg = functools.partial(
    artiq_env.NumberValue,
    default=0.0,
    min=-np.pi,
    max=np.pi,
    unit="radians (1/pi)",
    scale=np.pi,
    ndecimals=4,
)
PhaseScan = functools.partial(
    artiq_scan.Scannable,
    unit="radians (1/pi)",
    scale=np.pi,
    global_min=-np.pi,
    global_max=np.pi,
    ndecimals=4,
)


class RFSOC(EURIQAModule):
    """ARTIQ module to interface with the RFSoC (Sandia Octet).

    NOTE: this is different from other EURIQA modules.
    This module should be subclassed (i.e. ``class Exp(RFSOC)``),
    not used like: ``self.rfsoc = RFSOC(self)``
    """

    _SCHEDULE_TRANSFORM_TYPES = {
        "global_aom_delay": {
            "tooltip": "Compensate for delay in the global AOM relative to the "
            "individual AOMs by shifting the individual channels forward "
            "by a calibrated value",
            "default": True,
            "function_name": "compensate_global_aom_delay",
        },
        "fill_unused_channels": {
            "tooltip": "Fill unused channels in the schedules to prevent sync issues",
            "default": True,
            "function_name": "fill_unused_channels",
        },
        "pad_schedule": {
            "tooltip": "Fill blank timeslots in the schedule with delays",
            "default": True,
            "function_name": "pad_schedule",
        },
        "keep_global_beam_on": {
            "tooltip": "Minimize global AOM RF duty cycle variations by keeping the "
            "global beam on (but detuned) during schedule deadtime.",
            "default": True,
            "function_name": "keep_global_beam_on",
        },
        "aom_nonlinearity": {
            "tooltip": "Compensate for nonlinearity in the AOM(s) diffraction "
            "response by re-scaling the amplitude",
            "default": True,
            "function_name": "compensate_aom_nonlinearity",
        },
    }
    _SCHEDULE_TRANSFORM_ARG_PREFIX = "schedule_transform_"
    _DEFAULT_RFSOC_DESCRIPTION_PATH = (
        _EURIQA_LIB_DIR
        / "euriqabackend"
        / "databases"
        / "rfsoc_system_description.pyon"
    )

    def build(self):
        """Add arguments & devices used by the RFSoC module."""
        super().build()
        self.initialized = False

        self.setattr_device("core")
        # allows using @kernel decorators if not super-class
        self.setattr_argument(
            "num_shots", artiq_env.NumberValue(default=100, step=1, ndecimals=0)
        )
        # TODO: figure out how to use calibrated_tpis in RFSoC. currently unused
        self.setattr_argument(
            "upload_in_streaming_mode",
            artiq_env.BooleanValue(default=False),
            tooltip="Whether the sequence(s) should be uploaded to the RFSoC in "
            "streaming mode or LUT (look-up table) mode. "
            "LUT mode seems to be more reliable, but but can run out of space "
            "depending on sequence duration (# of pulses).",
            group="RFSoC",
        )
        self.setattr_argument(
            "record_upload_sequence",
            artiq_env.BooleanValue(default=True),
            tooltip="Whether to record the uploaded RFSoC sequence as text to a dataset. "
            "Enabling this option can incur significant slowdowns "
            "(~5 seconds for an average basic sequence).",
            group="RFSoC",
        )
        self.setattr_argument(
            "default_sync",
            artiq_env.BooleanValue(default=True),
            tooltip="Whether the scheduled pulses should by default have the RFSoC "
            "sync flag applied to their ToneData.",
            group="RFSoC",
        )
        self.setattr_device("rfsoc_trigger")
        self.setattr_argument(
            "openpulse_schedule_qobj", artiq_env.PYONValue(default=None), group="RFSoC"
        )
        self.setattr_argument(
            "rfsoc_board_description",
            artiq_env.PYONValue(
                default=pyon.load_file(self._DEFAULT_RFSOC_DESCRIPTION_PATH)
            ),
            group="RFSoC",
        )
        self._rf_calib_file_path = pathlib.Path(
            self.get_argument(
                "rf_calibration_file",
                artiq_env.StringValue(default=str(RF_CALIBRATION_PATH)),
                group="RFSoC",
                tooltip="Path to RF Calibration file (JSON/PYON)",
            )
            or ""  # at repo scan time, set path to "". pathlib.Path(None) errors
        )
        self.setattr_argument(
            "keep_global_on_global_beam_detuning",
            FrequencyArg(default=5e6),
            group="RFSoC",
            tooltip="Difference in frequency between the carrier global frequency &"
            "the detuned global beam frequency. See ``keep_global_beam_on``",
        )
        # Absolute amplitude of the global beam during the ``keep_global_beam_on``
        # transform. Applied to ALL RFSoC global channels, might need half amplitude
        self.keep_global_on_dual_tone_amplitude = self.get_dataset(
            "global.RFSOC.amplitude.global_dual_tone_amplitude"
        )
        for transform_type, options in self._SCHEDULE_TRANSFORM_TYPES.items():
            self.setattr_argument(
                self._SCHEDULE_TRANSFORM_ARG_PREFIX + transform_type,
                artiq_env.BooleanValue(default=options.get("default", True)),
                group="RFSoC",
                tooltip=options.get("tooltip", ""),
            )
        self._qiskit_backend = None

    @property
    def qiskit_backend(self) -> qbe.MinimalQiskitIonBackend:
        if self._qiskit_backend is None:
            raise RuntimeError("Trying to access qiskit backend before initialization")
        return self._qiskit_backend

    @qiskit_backend.setter
    def qiskit_backend(self, backend: qbe.MinimalQiskitIonBackend):
        self._qiskit_backend = backend

    @staticmethod
    def _load_gate_solutions(
        path: typing.Union[str, pathlib.Path],
        num_ions: int,
        load_calibrations: bool = True,
        tweak_file_name: str = "gate_solution_tweaks2.h5",
    ) -> calibrations.CalibrationBox:
        """Load Gate Solutions & tweaks from file path specified.

        If ``load_calibrations`` and the file does not exist, then it will
        create a new, empty :class:`GateCalibration` object.
        """
        result = calibrations.CalibrationBox({"gate_solutions": {}, "gate_tweaks": {}})
        solutions = pathlib.Path(path)
        if not solutions.is_file():
            raise FileNotFoundError(f"Solution file '{solutions}' does not exist")
        result["gate_solutions.struct"] = gate_params.GateSolution.from_h5(solutions)
        result["gate_solutions.path"] = solutions
        if load_calibrations:
            potential_tweaks_file = solutions.with_name(tweak_file_name)
            result["gate_tweaks.path"] = potential_tweaks_file
            if potential_tweaks_file.is_file():
                result["gate_tweaks.struct"] = gate_params.GateCalibrations.from_h5(
                    potential_tweaks_file
                )
            else:
                _LOGGER.warning(
                    "No pre-existing gate calibrations found. "
                    "Initializing from gate solutions"
                )
                result[
                    "gate_tweaks.struct"
                ] = gate_params.GateCalibrations.from_gate_solution(
                    result["gate_solutions.struct"]
                )
            assert (
                result["gate_solutions.struct"].solutions_hash
                == result["gate_tweaks.struct"].solutions_hash
            )
            assert (
                result["gate_solutions.struct"].num_ions
                == result["gate_tweaks.struct"].num_ions
            )
        assert num_ions == result["gate_solutions.struct"].num_ions
        return result

    @staticmethod
    def plot_schedule(
        schedule: qp.Schedule, save_path: pathlib.Path = None, **kwargs,
    ) -> "Figure":  # noqa: F821
        """Plot a Qiskit Schedule.

        Simple wrapper, but takes care of the proper imports, and allows saving
        automatically if needed.
        If ``save_path`` is not specified, this method will block until the figure is
        closed, then execution will continue. Else, it will attempt to save the figure
        to a specified file.
        Any ``kwargs`` are passed directly to the schedule drawing method.
        """
        fig = schedule.draw(**kwargs)
        if save_path is None:
            # plot immediately
            import matplotlib.pyplot as plt

            plt.show()
        else:
            fig.savefig(str(save_path))

        return fig

    @property
    def _has_schedule_argument(self):
        """Check if a schedule is being passed as an argument."""
        return self.openpulse_schedule_qobj is not None

    def number_of_ions(self):
        return int(self._rf_calib.other.number_of_ions.value)

    def initialize(
        self,
        load_gate_solutions: bool = True,
    ):
        """Load the parameters and calibration values that will be used.

        DO NOT SEND THEM TO RF COMPILER. Doing so will override whatever state
        the RF compiler is in, and it could be preparing/running some other experiment.
        """
        # Load RF calibrations
        # Based on settings/rf_calibration.json, adds datasets & calculated values
        self._rf_calib = calibrations.CalibrationBox.from_json(
            filename=self._rf_calib_file_path,
            dataset_dict=self._HasEnvironment__dataset_mgr,
        )

        num_ions = int(self._rf_calib.other.number_of_ions.value)
        rfsoc_map = rfsoc_mapping.RFSoCChannelMapping(self.rfsoc_board_description)

        # use the total number of ions in the chain as the # of qubits,
        # NOT the addressable qubits
        # global: bd 0, ch0
        # ion0: bd 0, ch4

        if(num_ions==1):
            config = QuickConfig(num_ions, rfsoc_map, {0: 3})

        if(num_ions==15):
            config = QuickConfig(num_ions, rfsoc_map, {
                7: 6, # bd0_01 # PMT1
                6: 18,# bd1_03 # PMT2
                5: 15, # bd1_00 # PMT3
                4: 17, # bd1_02 # PMT4
                3: 16, # bd1_01 # PMT5
                2: 4, # bd0_03 # PMT6
                1: 5, # bd0_02 # PMT7
                0: 3, # bd0_04 # PMT8
                -1: 1, # bd0_06 # PMT9
                -2: 2, # bd0_05 # PMT10
                -3: 8, # bd2_01 # PMT11
                -4: 9, # bd2_02 # PMT12
                -5: 7, # bd2_00 # PMT13
                -6: 10, # bd2_03 # PMT14
                -7: 0 # bd0_07 PMT15
                }
                )
        elif(num_ions == 23):
            config = QuickConfig(num_ions, rfsoc_map, {
                -11: 14,
                -10: 13,
                -9: 12,
                -8: 11,
                -7: 0,
                -6: 10,
                -5: 7,
                -4: 9,
                -3: 8,
                -2: 2,
                -1: 1,
                0: 3,
                1: 5,
                2: 4,
                3: 16,
                4: 17,
                5: 15,
                6: 18,
                7: 6,
                8: 19,
                9: 20,
                10: 21,
                11: 22})
        else:
            config = QuickConfig(num_ions, rfsoc_map, {
                -15:26,
                -14:25,
                -13:24,
                -12:23,
                -11: 14,
                -10: 13,
                -9: 12,
                -8: 11,
                -7: 0,
                -6: 10,
                -5: 7,
                -4: 9,
                -3: 8,
                -2: 2,
                -1: 1,
                0: 3,
                1: 5,
                2: 4,
                3: 16,
                4: 17,
                5: 15,
                6: 18,
                7: 6,
                8: 19,
                9: 20,
                10: 21,
                11: 22,
                12:27,
                13:28,
                14:29,
                15:30})

        self._qiskit_backend = qbe.MinimalQiskitIonBackend(
            num_ions,
            rfsoc_map,
            config=config,
            properties=self._rf_calib.to_backend_properties(num_ions),
        )
        # don't load gate solutions if only have one ion, the gate solutions are
        # only for >= 2 qubit gates

        # load_gate_solutions = False
        if load_gate_solutions and num_ions > 1:
            try:
                # first try the raw path. If that doesn't exist, search in the EURIQA
                # NAS, and try checking subfolder for # of ions
                path_test = pathlib.Path(
                    self._rf_calib.gate_solutions.solutions_top_path
                )
                if not path_test.exists():
                    path_test = EURIQA_NAS_DIR / path_test
                if (path_test / self._rf_calib[f"gate_solutions.{num_ions}"]).exists():
                    path_test = path_test / self._rf_calib[f"gate_solutions.{num_ions}"]
                    self._rf_calib["gate_tweak_path"] = path_test.with_name("gate_solution_tweaks2.h5")
                print(path_test)
                self._rf_calib.merge_update(
                    self._load_gate_solutions(path_test, num_ions,)
                )
            except KeyError:
                _LOGGER.warning(
                    "Could not find gate solutions for %s ions", num_ions, exc_info=True
                )
        self.initialized = True

    def prepare(
        self,
        schedules: ScheduleList = None,
        load_gate_solutions: bool = True,
        compile_schedules: bool = True,
    ):
        try:
            super().prepare()
        except AttributeError:
            pass
        self.initialize(load_gate_solutions)
        pr = cProfile.Profile()
        pr.enable()
        if compile_schedules:
            self.compiled_sequence_list = self.compile_pulse_schedule_to_octet()
        pr.disable()

        with open('/home/euriqa/git/euriqa-artiq/profile.txt', 'w') as io:
            stats = pstats.Stats(pr, stream=io)
            stats.sort_stats('tottime')
            stats.print_stats()

    @host_only
    def _find_pulse_schedule(
        self,
        schedule: typing.Optional[typing.Union[ScheduleOrList, qobj.PulseQobj]] = None,
    ) -> typing.Union[ScheduleOrList, qobj.PulseQobj]:
        # Find the schedule to be compiled. Search function args, then ARTIQ args,
        # then experiment's custom method
        schedule_methods = self.get_parent_attr_recursive("custom_pulse_schedule_list")
        if schedule is not None:
            _LOGGER.debug(
                "Using given function argument schedule for RFSoC compilation"
            )
            input_schedule = schedule
        elif self._has_schedule_argument:
            _LOGGER.debug("Using RFSoC PulseQObj from ARTIQ argument")
            # Compile schedule for RFSoC/Octet
            # Need to copy first b/c from_dict() modifies the dict
            input_schedule = qobj.PulseQobj.from_dict(
                copy.deepcopy(self.openpulse_schedule_qobj)
            )
        elif any(map(callable, schedule_methods)):
            # Check if self has a custom_pulse_schedule_list() function
            sched_meths = list(
                (m for m in schedule_methods if (m is not None and callable(m)))
            )
            assert (
                len(sched_meths) == 1
            ), f"Incorrect number of schedule methods found: {sched_meths}"
            _LOGGER.debug(
                "Using custom schedule generation function: %r", sched_meths[0]
            )
            input_schedule = sched_meths[0]()
        else:
            raise RuntimeError(
                "No schedule found. RFSoC module requires a Qiskit schedule to run. "
                "Can be provided as an ARTIQ argument, as a function argument, "
                "or as a custom method (``custom_pulse_schedule_list()``)."
            )

        return input_schedule

    @host_only
    def compile_pulse_schedule_to_octet(
        self,
        input_schedules: typing.Optional[
            typing.Union[ScheduleOrList, qobj.PulseQobj]
        ] = None,
    ) -> CompiledScheduleList:
        """
        Compile a given pulse schedule into the RFSoC output waveforms for runtime.

        The schedule is optional, because this will try to auto-find it.
        Search path (will stop at first found):
            1. Function input
            2. ARTIQ experiment arguments. ``openpulse_schedule_qobj``
            3. Custom function in ARTIQ experiment.
                Expected signature: ``def custom_pulse_schedule_list(self) -> qp.Schedule:``

        The schedule will have any enabled calibrations applied to it. See
        :func:`_apply_schedule_calibrations`.

        Args:
            input_schedules (Union[ScheduleOrList, qobj.PulseQobj], optional): OpenPulse
                schedule (or list of schedules) to run. Defaults to None.

        Raises:
            RuntimeError: If no schedule to run has been found by going through
                search path above.

        Returns:
            List[ChannelSequence]: An Octet-native gate representation that can be
            uploaded to the RFSoC.
        """
        input_schedules = self._find_pulse_schedule(input_schedules)

        # TODO: convert to separate method. too long of function
        # Convert QObj to schedule to unify execution flow path
        if isinstance(input_schedules, qobj.PulseQobj):
            (input_schedules_from_qobj, _run_config, _user_qobj_header,) = disassemble(
                input_schedules
            )
            input_schedules = input_schedules_from_qobj
        else:
            input_schedules = list(more_itertools.always_iterable(input_schedules))
            if not isinstance(more_itertools.first(input_schedules, None), qp.Schedule):
                raise ValueError(
                    "Invalid schedule passed: {} (type={}) is not an OpenPulse "
                    "Schedule".format(input_schedules, type(input_schedules))
                )

        octet_circlist = []
        # TODO: LO freqs from input PulseQObj are not currently respected.
        config = self._qiskit_backend.configuration()
        idle_shifts = self._rf_calib["frequencies.23.idle_stark_shifts"].value
        idle_shifts = [7 + x for x in self._rf_calib["frequencies.23.idle_stark_shifts"].value]

        for ideal_schedule in input_schedules:
            hardware_schedule = self._apply_schedule_calibrations(ideal_schedule)
            # Convert Schedule -> Octet
            global_carrier_freq = (
                self._rf_calib.frequencies.global_carrier_frequency.value
            )
            individual_carrier_freq = (
                self._rf_calib.frequencies.individual_carrier_frequency.value
            )

            center_ion_zero_index = (self.number_of_ions() - 1)//2
            default_individual_frequencies = {}
            for idx in range(self.number_of_ions()):
                chs = config._center_index_ion_to_channel_mapping[idx-center_ion_zero_index]
                for ch in chs:
                    if isinstance(ch,qp.DriveChannel):
                        default_individual_frequencies[ch] = individual_carrier_freq - idle_shifts[idx]

            default_frequencies = {
                c: global_carrier_freq
                if isinstance(c, qp.ControlChannel)
                else individual_carrier_freq
                for c in (hardware_schedule.channels - default_individual_frequencies.keys())
            }

            default_frequencies.update(default_individual_frequencies)
            octet_circlist.append(
                octetconverter.OpenPulseToOctetConverter.schedule_to_octet(
                    hardware_schedule,
                    default_lo_freq_hz=default_frequencies,
                    default_sync_enabled=self.default_sync,
                    default_frequency_feedback_enabled=False,
                )
            )

        return octet_circlist

    def _apply_schedule_calibrations(self, schedule: qp.Schedule) -> qp.Schedule:
        """Applies schedule adjustments based on low-level hardware adjustments.

        Examples: global AOM delay, individual channel scaling, etc.

        A little bit of magic is done here, basically it calls the function that has
        the same name as the name of the schedule adjustment
        (see :prop:`_SCHEDULE_TRANSFORM_TYPES`).
        """
        _LOGGER.debug("Original schedule pre-hardware calibrations: %s", schedule)
        for transform_type, options in self._SCHEDULE_TRANSFORM_TYPES.items():
            # Check if transform enabled in options
            if getattr(self, self._SCHEDULE_TRANSFORM_ARG_PREFIX + transform_type):
                _LOGGER.debug("Applying compensation '%s' to schedule", transform_type)
                schedule = getattr(self, options.get("function_name"))(schedule)

        _LOGGER.debug("Schedule after hardware calibrations: %s", schedule)
        return schedule

    # *** Schedule Transforms ***

    def compensate_global_aom_delay(self, schedule: qp.Schedule) -> qp.Schedule:
        """Apply Global AOM time shifting to the output schedule.

        Shifts non-global(control) channels forward in time by delay.
        """
        # Calculate delay
        global_aom_delay_time = self._rf_calib.delays.global_aom_to_individual_aom.value
        global_aom_delay_dt = int(global_aom_delay_time / self.dt)

        global_channels = set(
            filter(lambda c: isinstance(c, qp.ControlChannel), schedule.channels)
        )

        # Shift other channels back
        global_channel_sched = schedule.filter(channels=global_channels)
        other_channel_sched = schedule.exclude(channels=global_channels).shift(
            global_aom_delay_dt
        )
        return global_channel_sched.insert(0, other_channel_sched)

    def compensate_aom_nonlinearity(self, schedule: qp.Schedule) -> qp.Schedule:
        """Compensate for AOM nonlinearity (compression) by adjusting signal amplitudes."""
        schedule_instructions = schedule.instructions

        def _correct_instruction_amplitude(
            t_instr: typing.Tuple[int, qp.Instruction]
        ) -> typing.Iterable[typing.Tuple[int, qp.Instruction]]:
            t, instr = t_instr
            """Corrects the amplitude of a Qiskit Play instruction for AOM nonlinearity.

            Supported Play pulse instruction types are:
                * Constant
                * Gaussian
                * GaussianSquare
                * CubicSplinePulse
                * ToneDataPulse
            """

            if not isinstance(instr, qp.Play):
                # only process Play instructions
                return ((t, instr),)

            def _correct_basic_pulse(
                pulse: qp.Constant, nonlinearity: float
            ) -> qp.Constant:
                new_amp = pc_nonlinearity.calculate_new_coefficients_cached(
                    spl.CubicSpline(np.real(pulse.amp)), nonlinearity
                )
                return [qp.Constant(pulse.duration, new_amp[0])]

            def _correct_spline_pulse(
                pulse: typing.Union[pc_pulse.ToneDataPulse, pc_pulse.CubicSplinePulse],
                nonlinearity: float,
            ) -> typing.Union[pc_pulse.ToneDataPulse, pc_pulse.CubicSplinePulse]:
                if isinstance(pulse, pc_pulse.ToneDataPulse):
                    amp_spl = pulse.tonedata.amplitude
                elif isinstance(pulse, pc_pulse.CubicSplinePulse):
                    amp_spl = spl.CubicSpline(
                        pulse._spline0, pulse._spline1, pulse._spline2, pulse._spline3
                    )

                new_amp_spl = pc_nonlinearity.calculate_new_coefficients_cached(
                    amp_spl, nonlinearity
                )

                params = pulse.parameters
                if isinstance(pulse, pc_pulse.ToneDataPulse):
                    params["amplitude"] = new_amp_spl
                    return [pc_pulse.ToneDataPulse(**params)]
                else:
                    for i, coeff in enumerate(new_amp_spl):
                        params[f"spline{i}"] = coeff
                    return [pc_pulse.CubicSplinePulse(**params)]

            def _correct_gaussian_pulse(
                pulse: qp.Gaussian, nonlinearity: float
            ) -> typing.List[pc_pulse.CubicSplinePulse]:
                # sample gaussian at desired interval
                NUM_GAUSSIAN_SEGMENTS = 12
                (
                    gaussian_splines,
                    durations,
                ) = octetconverter._gaussian_to_octet_coefficients(
                    num_segments=NUM_GAUSSIAN_SEGMENTS,
                    duration=pulse.duration,
                    amp=pulse.amp,
                    sigma=pulse.sigma,
                )

                nonlinear_coefficients = (
                    pc_nonlinearity.calculate_new_coefficients_cached(spl, nonlinearity)
                    for spl in gaussian_splines
                )
                return [
                    pc_pulse.CubicSplinePulse(
                        dur, *coeffs, name=f"Gaussian[{i}/{NUM_GAUSSIAN_SEGMENTS}]"
                    )
                    for i, (dur, coeffs) in enumerate(
                        zip(durations, nonlinear_coefficients)
                    )
                ]

            def _correct_gaussian_square_pulse(
                pulse: qp.GaussianSquare, nonlinearity: float
            ) -> typing.List[pc_pulse.CubicSplinePulse]:
                # calculate gaussian w/o square segment
                equiv_gauss_pulse = qp.Gaussian(
                    pulse.duration - pulse.width, amp=pulse.amp, sigma=pulse.sigma
                )
                corrected_gaussian = _correct_gaussian_pulse(
                    equiv_gauss_pulse, nonlinearity
                )
                corrected_square = _correct_basic_pulse(
                    qp.Constant(pulse.width, pulse.amp), nonlinearity
                )[0]

                # insert square pulse
                corrected_gaussian.insert(
                    len(corrected_gaussian) // 2, corrected_square
                )
                return corrected_gaussian

            nonlinearity = wf_convert.channel_aom_saturation(instr.channel, self.qiskit_backend)
            pulse_correction_func = {
                pc_pulse.CubicSplinePulse: _correct_spline_pulse,
                pc_pulse.ToneDataPulse: _correct_spline_pulse,
                qp.Constant: _correct_basic_pulse,
                qp.Gaussian: _correct_gaussian_pulse,
                qp.GaussianSquare: _correct_gaussian_square_pulse,
            }

            try:
                corrected_pulse_list = pulse_correction_func[type(instr.pulse)](
                    instr.pulse, nonlinearity
                )
            except KeyError as err:
                raise ValueError(f"Unsupported pulse type {type(instr.pulse)}") from err
            start_time = t
            return_instructions = []
            for corr_pulse in corrected_pulse_list:
                # add each instruction to return, accumulating pulse duration
                return_instructions.append(
                    (start_time, qp.Play(corr_pulse, instr.channel))
                )
                start_time += corr_pulse.duration

            return tuple(return_instructions)

        new_instructions = map(_correct_instruction_amplitude, schedule_instructions)
        return qp.Schedule(*itertools.chain.from_iterable(new_instructions))

    def fill_unused_channels(self, schedule: qp.Schedule) -> qp.Schedule:
        """Add a blank ("delay") to any unused channels.

        This is to prevent channels from coming out-of-sync if they are used in one
        Schedule, and then not used in the next.
        PulseCompiler will not automatically insert delays on unused channels
        b/c PulseCompiler is agnostic at compilation time of the number of channels
        in your system.
        """
        available_channels = set(self._qiskit_backend.configuration().all_channels)
        used_channels = set(schedule.channels)
        unused_channels = available_channels - used_channels
        return qp_transforms.pad(schedule, channels=unused_channels, inplace=False)

    pad_schedule = staticmethod(qp_transforms.pad)
    """Wrapper around the Qiskit schedule transform `pad`.

    By default, adds delays during any unspecified timeslots, and extends
    all channels (w/ delays) to the end of the schedule's duration.
    """

    def prepend_schedules(
        self, new_schedules: ScheduleList, compiled_sequence_list: CompiledScheduleList,
    ) -> typing.Tuple[CompiledScheduleList, typing.List[int]]:
        """Prepend Qiskit schedule(s) to the pre-compiled experiment sequence.

        Returns the modified experiment sequence, as well as the duration of each
        prepended schedule.
        """
        new_compiled_sequence, durations = self.compile_and_merge_schedules(
            new_schedules
        )
        new_schedules = []
        for compiled in compiled_sequence_list:
            new_schedules.append(
                tones.merge_channel_sequences(
                    new_compiled_sequence, compiled, inplace=False
                )
            )

        return new_schedules, durations

    def compile_and_merge_schedules(
        self, schedules: ScheduleList
    ) -> typing.Tuple[tones.ChannelSequence, typing.List[int]]:
        """Compile each of a list of schedules, and merge into one ChannelSequence.

        The reduced (merged) schedule, as well as the duration of each schedule
        (in RFSoC clock cycles, dt) pre-merge is returned.
        Note that these durations are calculated purely within a schedule, so
        trigger delays are not accounted for.
        """
        compiled_sequences = self.compile_pulse_schedule_to_octet(schedules)
        sequence_durations = list(
            map(tones.sequence_duration_cycles, compiled_sequences)
        )
        # merge sequences together sequentially
        return (
            functools.reduce(
                functools.partial(tones.merge_channel_sequences, inplace=False),
                compiled_sequences,
            ),
            sequence_durations,
        )

    def keep_global_beam_on(self, schedule: qp.Schedule) -> qp.Schedule:
        """Schedule transform to keep the global beam at constant power.

        Prevents issues where turning the global beam on & off (or changing
        duty cycle) would cause thermal effects in the global AOM.

        Transforms the input schedule to return a modified schedule, where
        the global beam is turned on during all delays/dead times in the schedule.
        Times when it is specified (i.e. has some pulse played on it) are left alone.
        """
        # TODO: figure out actual desired global amplitude/frequency
        # TODO: make sure that off-resonant global channel doesn't play during single-qubit gate
        global_keep_on_freq = (
            self.keep_global_on_global_beam_detuning
            + self._rf_calib.frequencies.global_carrier_frequency.value
        )
        GLOBAL_ON_PULSE = functools.partial(
            pc_pulse.ToneDataPulse,
            frequency_hz=global_keep_on_freq,
            amplitude=self.keep_global_on_dual_tone_amplitude,
            phase_rad=0.0,
            # probably not necessary, but don't want to accidentally modify anything intentional
            sync=False,
        )

        sched_global_channels = set(
            filter(lambda c: isinstance(c, qp.ControlChannel), schedule.channels)
        )

        all_global_channels = self._qiskit_backend.configuration().global_channels()
        schedule = schedule.flatten()

        for glob_chan in all_global_channels:
            if glob_chan not in sched_global_channels:
                schedule.insert(
                    0,
                    qp.Play(GLOBAL_ON_PULSE(schedule.duration), glob_chan),
                    inplace=True,
                )
            else:
                for _start_time, global_delay_instr in schedule.filter(
                    channels={glob_chan}, instruction_types={qp.Delay}
                ).instructions:
                    schedule.replace(
                        global_delay_instr,
                        qp.Play(
                            GLOBAL_ON_PULSE(global_delay_instr.duration), glob_chan
                        ),
                        inplace=True,
                    )
                # If still not at end of schedule, add global_on_pulse to end of
                # channel instructions
                end_slack = schedule.duration - schedule.ch_duration(glob_chan)
                if end_slack > 0:
                    schedule.insert(
                        schedule.ch_stop_time(glob_chan),
                        qp.Play(GLOBAL_ON_PULSE(end_slack), glob_chan),
                        inplace=True,
                    )

        return schedule

    @property
    def dt(self) -> float:
        """Length of a schedule "dt" (base time unit) in seconds."""
        return self.qiskit_backend.configuration().dt

    @property
    def rfsoc_hardware_description(self) -> rfsoc_mapping.RFSoCChannelMapping:
        return self.qiskit_backend.configuration().rfsoc_channel_map

    def schedule_duration(
        self, schedule: typing.Union[qp.Schedule, tones.ChannelSequence]
    ) -> float:
        """Return the schedule duration in seconds.

        Accepts either a Pulse Schedule or the RFSoC ``ChannelSequence``.
        If you want the duration in RFSoC clock cycles, use ``schedule.duration``.
        """
        if isinstance(schedule, qp.Schedule):
            return schedule.duration * self.dt
        elif isinstance(schedule, dict):
            # i.e. isinstance(schedule, ChannelSequence)
            return tones.sequence_duration_cycles(schedule) * self.dt
        else:
            raise ValueError(f"Schedule type not recognized: {type(schedule)}")

    @host_only
    def experiment_initialize(self):
        """Upload the converted waveforms to the RFSoC at runtime."""
        self.upload_data(
            self.num_shots,
            self.compiled_sequence_list,
            debug_channel_mapping={},
            streaming=self.upload_in_streaming_mode,
        )

    @kernel
    def init(self):
        """Set the RFSoC trigger low."""
        self.rfsoc_trigger.off()

    @kernel
    def trigger(self):
        """Trigger the RFSoC with a TTL pulse."""
        # NOTE: the Octet Trigger is likely level-sensitive, so this could fail
        # (i.e. output multiple schedules) for very short schedule durations (< 100 ns)
        self.rfsoc_trigger.pulse_mu(100)

    @host_only
    def upload_data(
        self,
        num_shots: int,
        channel_sequences: CompiledScheduleList = None,
        interleaved_sweep: bool = False,
        debug_channel_mapping: typing.Dict[QiskitChannel, QiskitChannel] = None,
        **kwargs,
    ) -> typing.List[int]:
        """Upload the RF waveforms to the RFSoC.

        Should be called at start of experiment (``run()``).
        Uploads pre-compiled waveforms to the RFSoC via Octet.

        Saves the uploaded RFSoC waveforms to dataset ``rfsoc_output``.

        Args:
            num_shots (int): number of shots of each schedule to acquire
                (i.e. how many times to repeat each schedule).
            channel_schedules (List[ChannelSequence], optional): mapping between Qiskit
                channels and the list of RFSoC Octet tones that will be played on
                the channel.
            interleaved_sweep (bool, optional): if shots of each sweep point are
                interleaved between each other
                (i.e. 1 shot of schedule 0, 1 shot of schedule 1, ...,
                1 shot of schedule 0, 1 shot of schedule 1, x num_shots).
                Defaults to False.
            debug_channel_mapping (Dict[QiskitChannel, QiskitChannel], optional):
                Mapping between a channel to be monitored (source, key) and the
                desired output channel (target, value).
                When this is set to an appropriate dictionary, it will
                copy the output tones from the source channel to the target channel.
                Note that this is not guaranteed to be bug-free: there could be hidden
                timing or other issues introduced by adding in more output waveforms.
                This is useful for e.g. monitoring an output channel on an oscilloscope
                to see what it is doing.
                Example:
                ```{qp.ControlChannel(0): qp.DriveChannel(0),
                    qp.ControlChannel(1): qp.DriveChannel(1)}```. Defaults to None.

        Kwargs:
            Any remaining argument(s) will be passed to
            :func:`pulsecompiler.rfsoc.tones.upload.upload_multiple_channel_sequences`.

        Returns:
            List[int]: the duration of each schedule uploaded.
        """
        if interleaved_sweep:
            raise NotImplementedError("interleaved sweeps are not yet supported")
        if channel_sequences is None:
            channel_sequences = self.compiled_sequence_list

        if debug_channel_mapping is not None:
            for s in channel_sequences:
                for source_chan, target_chan in debug_channel_mapping.items():
                    if target_chan in s:
                        raise RuntimeError(
                            f"Duplicate output channel {target_chan} detected "
                            f"in output monitoring: {debug_channel_mapping}."
                            "Cannot monitor on a channel that is already being "
                            f"used: {s.keys()}"
                        )
                    s[target_chan] = s.get(source_chan, list())

        _LOGGER.debug(
            "Uploading %i sequences to RFSoC(s): %i shots each, kwargs=%s",
            len(channel_sequences),
            num_shots,
            kwargs,
        )
        durations, uploaded_sequences = uploader.upload_multiple_channel_sequences(
            channel_sequences,
            num_repeats_per_schedule=[num_shots] * len(channel_sequences),
            rfsoc_channel_map=self.rfsoc_hardware_description,
            **kwargs,
        )
        # Archive the channel-mapped RFSoC output for future reference,
        # so theoretically the exact same sequence could be played again.
        # This IS slow with large Sequences (e.g. long scans, etc),
        # so we might want to disable this in the future/give an arg to turn off.
        if self.record_upload_sequence:
            self.set_dataset(
                "rfsoc_output",
                record.text_channel_sequence(uploaded_sequences),
                broadcast=False,
                persist=False,
                archive=True,
            )
        return durations

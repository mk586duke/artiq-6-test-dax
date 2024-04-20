"""Top-level interface to the RF compiler & RF subsystem of the EURIQA experiment.

Currently provides an interface to run gates/circuits on an AWG.
Needs rewritten in the future for flexibility and to hard-code things less.
It's quite a monster...

Originally written by Mike Goldman, some edits/upgrades from the rest of the EURIQA
team (Drew Risinger, Daiwei Zhu, Laird Egan).
"""
import datetime
import enum
import logging
import pathlib
import typing
import warnings

import numpy as np

import euriqabackend.devices.keysight_awg.AOM_calibration as cal
import euriqabackend.devices.keysight_awg.common_types as rf_common
import euriqabackend.devices.keysight_awg.gate as gate
import euriqabackend.devices.keysight_awg.gate_parameters as gate_params
import euriqabackend.devices.keysight_awg.interpolation_functions as intfn
import euriqabackend.devices.keysight_awg.labview_programmer as LVprog
import euriqabackend.devices.keysight_awg.physical_parameters as pp
import euriqabackend.utilities.decorators as eur_dec

_LOGGER = logging.getLogger(__name__)


class RFCompiler:

    waveform_dir = r"C:\RF System\Generated Waveforms\\"

    RabiScanParameter = gate.Rabi.ScanParameter
    RabiAMScanParameter = gate.Rabi_AM.ScanParameter
    RabiPIScanParameter = gate.Rabi_PI.ScanParameter
    WindUnwindScanParameter = gate.WindUnwind.ScanParameter
    CrosstalkCalibScanParameter = rf_common.CrosstalkCalibScanParameter
    FastEchoScanParameter = gate.FastEcho.ScanParameter
    FastEchoActiveBeams = gate.FastEcho.ActiveBeams
    BichromaticScanParameter = gate.Bichromatic.ScanParameter
    BichromaticActiveBeams = gate.Bichromatic.ActiveBeams
    SK1ScanParameter = gate.SK1.ScanParameter
    SK1AMScanParameter = gate.SK1_AM.ScanParameter
    XXScanParameter = gate.XX.ScanParameter
    InterpFunctionType = intfn.InterpFunction.FunctionType
    ChargeResponseProbeScanParameter = rf_common.ChargeResponseProbeScanParameter

    _GATE_SOLUTION_FILE = "all_gate_solutions.h5"
    _GATE_TWEAKS_FILE = "gate_solution_tweaks.h5"

    def __init__(self):
        import euriqabackend.devices.keysight_awg.gate_compiler as gc
        import euriqabackend.devices.keysight_awg.circuit_interpreter as ci

        self.physical_params = pp.PhysicalParams()
        self.name = "Not yet set"
        self.total_duration = 0
        self.wait_after_time = 0
        self.timestep_times = []

        # Instantiate the gate compiler that will construct the arrays of gates to
        # output.
        self.gate_compiler = gc.GateCompiler(self)

        # Instantiate the circuit interpreter that will construct circuits in the
        # circuit() experimental function.
        self.circuit_interpreter = ci.CircuitInterpreter(self, self.physical_params)

        # Instantiate the calibration object, which rescales the waveform amplitudes
        # before writing to the AWG
        self.calibration = cal.Calibration()

        # create a variable for the gate solutions & gate tweaks, though can't
        # yet use b/c don't know how many ions yet.
        self.gate_solution = None
        self.gate_tweaks = None

    # _PASSTHROUGH_FUNCTIONS = {
    #     # FORMAT: "passthrough_name": ("top_level_attr", "second_level_attr", "..."),
    #     # also allow "RFCompiler.top_level_attr..." or "self.top_level_attr..."
    #     "set_params": "physical_params.set_params",
    #     "set_monitor_params": "physical_params.monitor.set_params",
    #     "set_Rabi_params": "self.physical_params.Rabi.set_params",
    #     "set_SK1_params": "self.physical_params.SK1.set_params",
    #     "set_SK1_AM_params": "self.physical_params.SK1_AM.set_params",
    #     "set_XX_params": "self.physical_params.XX.set_params",
    #     "set_AOM_levels": "self.calibration.write_levels",
    #     "set_AOM_saturation_params": "self.calibration.write_saturation_params",
    #     "disable_calibration": "self.calibration.disable_calibration",
    #     "clear_gates": "self.gate_compiler.clear_gates",
    #     "generate_waveforms": "self.gate_compiler.generate_waveforms",
    # }

    # def __getattr__(self, attr: str):
    #     """Provide passthrough functionality so sub-functions can be called via ARTIQ.

    #     Eliminates old way of doing it which just has TONS of basically empty
    #     pass-through function calls.
    #     This method is only called an attribute doesn't exist.
    #     """
    #     # only called when attr doesn't exist, so don't have to worry about it
    #     # screwing anything up, I think
    #     if attr in self._PASSTHROUGH_FUNCTIONS.keys():
    #         attr_path = self._PASSTHROUGH_FUNCTIONS[attr]
    #         if isinstance(attr_path, str):
    #             attr_path = attr_path.split(".")
    #         if attr_path[0].lower().startswith(("rfcompiler", "self")):
    #             # strip "rfcompiler" or "self" attr from path
    #             attr_path = attr_path[1:]
    #         _LOGGER.debug(
    #             "Mapping RFCompiler.%s to RFCompiler.%s", ".".join(attr_path), attr
    #         )
    #         final_attr = self
    #         try:
    #             for level_name in attr_path:
    #                 final_attr = getattr(final_attr, level_name)
    #             return final_attr
    #         except AttributeError:
    #             raise AttributeError(
    #                 "Could not find attr {}.{} in path self.{}".format(
    #                     final_attr, level_name, ".".join(attr_path)
    #                 )
    #             )
    #     else:
    #         raise AttributeError(
    #             "Attribute {} not found, and is NOT a passthrough".format(attr)
    #         )

    ###################################################
    #              SET PHYSICAL PARAMS                #
    ###################################################

    def setattr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_params(
        self,
        f_carrier: float,
        f_ind: float,
        N_ions: float,
        t_delay: float,
        Rabi_max: float,
        PI_center_freq_1Q: float,
        PI_center_freq_2Q: float,
    ):

        self.physical_params.set_params(
            f_carrier=f_carrier,
            f_ind=f_ind,
            N_ions=N_ions,
            t_delay=t_delay,
            Rabi_max=Rabi_max,
            PI_center_freq_1Q=PI_center_freq_1Q,
            PI_center_freq_2Q=PI_center_freq_2Q,
        )

    def set_monitor_params(self, monitor_ind: float, detuning: float, amp: float):

        self.physical_params.monitor.set_params(
            monitor_ind=monitor_ind, detuning=detuning, amp=amp
        )

    def set_Rabi_params(self, amp_ind: float, amp_global: float, Stark_shift: float):

        self.physical_params.Rabi.set_params(
            amp_ind=amp_ind, amp_global=amp_global, Stark_shift=Stark_shift
        )

    def set_SK1_params(
        self,
        amp_ind: float,
        amp_global: float,
        Tpi_multiplier: float,
        Stark_shift: float,
    ):

        self.physical_params.SK1.set_params(
            amp_ind=amp_ind,
            amp_global=amp_global,
            Tpi_multiplier=Tpi_multiplier,
            Stark_shift=Stark_shift,
        )

    def set_SK1_AM_params(
        self,
        amp_ind: float,
        amp_global: float,
        theta: float,
        envelope_type: int,
        envelope_scale: float,
        rotation_pulse_length: float,
        correction_pulse_1_length: float,
        correction_pulse_2_length: float,
        Stark_shift: float,
        phase_correction: float,
    ):

        self.physical_params.SK1_AM.set_params(
            amp_ind=amp_ind,
            amp_global=amp_global,
            theta=theta,
            envelope_type=self.InterpFunctionType(envelope_type),
            envelope_scale=envelope_scale,
            rotation_pulse_length=rotation_pulse_length,
            correction_pulse_1_length=correction_pulse_1_length,
            correction_pulse_2_length=correction_pulse_2_length,
            Stark_shift=Stark_shift,
            phase_correction=phase_correction
        )

    def set_XX_params(
        self,
        phase_offset: float,
    ):
        self.physical_params.XX.set_params(phase_offset=phase_offset)

    def set_XX_solution_from_file(
        self,
        solutions_file: str,
        autosave_cals: bool = True,
        autoload_cals: bool = True,
    ):
        """Load an XX gate solution from a specified folder.

        Saves the previous XX calibrations/tweaks in the directory where they
        came from, if ``autosave_cals``. It will save this to
        ``gate_tweaks.load_path/self._GATE_TWEAKS_FILE``

        It will also load previous XX calibrations/tweaks if there is already a
        file with the default name (see :attr:`_GATE_TWEAKS_FILE`),
        if ``autoload_cals``.
        """
        solutions_folder = pathlib.Path(solutions_file).parent

        # Save existing gate solutions & calibrations in their directory.
        if autosave_cals:
            if self.gate_tweaks is not None:
                calib_save_file = self.gate_tweaks.load_path / self._GATE_TWEAKS_FILE
                _LOGGER.info("Saving gate calibrations to file: %s", calib_save_file)
                self.gate_tweaks.to_h5(calib_save_file)

        # Attempt to load solution/calibration files from solutions_path if
        # they exist & are enabled
        # Otherwise, load the data structures from scratch.
        soln_file_path = pathlib.Path(solutions_file)
        assert soln_file_path.is_file(), "Given solution file does not exist"
        self.gate_solution = gate_params.GateSolution.from_h5(soln_file_path)
        assert self.gate_solution.num_ions == self.physical_params.N_ions
        # TODO: decide if keep
        #  assert self.gate_solution.load_path == str(solutions_folder)

        potential_cal_file = solutions_folder / self._GATE_TWEAKS_FILE
        if autoload_cals and potential_cal_file.is_file():
            _LOGGER.info("Reloading existing tweaks file: %s", potential_cal_file)
            try:
                self.gate_tweaks = gate_params.GateCalibrations.from_h5(potential_cal_file)
            except AssertionError as err:
                _LOGGER.error(
                    "Could not validate autoloaded calibration file. "
                    "It might be of a different (old) format. "
                    "Try disabling autoload or removing/renaming calibrations"
                )
                raise err
        else:
            self.gate_tweaks = gate_params.GateCalibrations.from_gate_solution(
                self.gate_solution
            )
            self.gate_tweaks.load_path = solutions_folder
        assert self.gate_tweaks.solutions_hash == self.gate_solution.solutions_hash

    def set_XX_solution_from_folder(
        self, sols_dir: str, autosave: bool = True, autoload: bool = True
    ):
        """Load XX solutions from a specified directory.

        Saves the previous XX solutions & calibration in the directory that
        the solution/calibration was based on (if ``autosave``)

        NOTE: saving this file will change the hash of the directory, meaning
        that if you try to reload solutions from the raw files and compare that
        hash against a saved calibration's hash, then it will fail.

        Defaults to loading the calibration & gate solution data from existing data
        structures if they exist, can be disabled by setting ``load_from_file=False``.
        This is a reasonable default, because we generally assume that you want to
        reload existing calibrations, but if you want to start from scratch you need
        to be aware this is what it's doing.

        Args:
            sols_dir (str): The FULL path of the directory where the XX gate solutions
                are stored. The solutions are e.g. from Laird's Gate Solver.
            autosave (bool, optional): Whether this function should autosave
                the calibration & solution data structures back to disk in the
                directory where they were loaded from. Defaults to True.
            autoload (bool, optional): Whether this function should attempt to
                load an existing calibration & solution from an HDF5 file in the given
                directory (``sols_dir``), if it exists. Note that this will cause
                (possibly incorrect) calibrations to be loaded. Defaults to True.
        """
        warnings.warn(
            "This function is meant for the old way of saving/reading XX Gate solutions. "
            "Solutions prior to Nov. 2019 should use this function, but ONLY THOSE",
            DeprecationWarning,
        )
        solutions_path = pathlib.Path(sols_dir)

        # Save existing gate solutions & calibrations in their directory.
        if autosave:
            if self.gate_solution is not None:
                solution_save_file = (
                    self.gate_solution.load_path / self._GATE_SOLUTION_FILE
                )
                _LOGGER.info("Saving gate solutions to file: %s", solution_save_file)
                self.gate_solution.to_h5(solution_save_file)
            if self.gate_tweaks is not None:
                calib_save_file = self.gate_tweaks.load_path / self._GATE_TWEAKS_FILE
                _LOGGER.info("Saving gate calibrations to file: %s", calib_save_file)
                self.gate_tweaks.to_h5(calib_save_file)

        # Attempt to load solution/calibration files from solutions_path if
        # they exist & are enabled
        # Otherwise, load the data structures from scratch.
        potential_soln_file = solutions_path / self._GATE_SOLUTION_FILE
        if autoload and potential_soln_file.is_file():
            _LOGGER.info(
                "Reloading existing gate solutions file: %s", potential_soln_file
            )
            self.gate_solution = gate_params.GateSolution.from_h5(potential_soln_file)
        else:
            self.gate_solution = gate_params.GateSolution(
                self.physical_params.N_ions, path=solutions_path
            )

        potential_cal_file = solutions_path / self._GATE_TWEAKS_FILE
        if autoload and potential_cal_file.is_file():
            _LOGGER.info("Reloading existing tweaks file: %s", potential_cal_file)
            self.gate_tweaks = gate_params.GateCalibrations.from_h5(potential_cal_file)
        else:
            self.gate_tweaks = gate_params.GateCalibrations.from_gate_solution(
                self.gate_solution
            )
        assert self.gate_tweaks.solutions_hash == self.gate_solution.solutions_hash

    def backup_XX_tweaks(self, backup_file: str = None, rid: int = None) -> None:
        """Backup the current XX tweaks to a given file.

        Args:
            backup_file (str): If provided, saves the gate calibrations to the provided
                file path. Otherwise, generates its own. Defaults to ``None`` (i.e.
                save to ``gate_tweaks.load_path/../tweak_backups/CURRENT_TIME.h5``)
            rid (int): If provided, and ``backup_file`` is not provided, prepends
                ``pre-rid-{rid}`` to the generated backup filename. This is intended
                to be called with the current ARTIQ RID, so we know what the
                gate calibrations were before some experiment potentially ruined them.
                Defaults to ``None``.
        """
        if backup_file is None:
            # generate file path
            # Defaults to gate_tweaks.load_path/../tweak_backups/CURRENT_TIME.h5
            current_time_str = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
            loaded_dir = self.gate_tweaks.load_path
            if rid:
                filename = "pre-rid-{rid}({curr_time}).h5".format(
                    rid=rid, curr_time=current_time_str
                )
            else:
                filename = "{curr_time}.h5".format(curr_time=current_time_str)
            backup_file = loaded_dir / ".." / "tweak_backups" / filename
        else:
            backup_file = pathlib.Path(backup_file)
        backup_file.mkdir(parents=True, exist_ok=True)
        self.gate_tweaks.to_h5(backup_file)

    def set_XX_tweak(self, slot_pair: rf_common.SlotPair, **kwargs) -> None:
        """Set a tweaked/compensated/calibrated value for an ion gate.

        Args:
            slot_pair (rf_common.SlotPair): The slot to tweak. If you set this to one of
                ``None, ":", "all", "ALL"``, then all possible gates will be modified.
                If you set it to "global", then the global slot will be modified
                (NOTE: global slot is applied as an offset to every gate, whereas ``:``
                overwrites the values for every gate.)

        Kwargs:
            You can set any number of tweaks at once (for a single ``slot_pair``)
            by passing valid tweak names as items in a dictionary
            (like ``RFCompiler.set_XX_tweak{name: value}``)

        Raises:
            RuntimeError: If you haven't called :meth:`set_XX_solution_from_file` first.
            You must choose solutions before trying to apply tweaks.

        """
        if self.gate_tweaks is None:
            raise RuntimeError(
                "Tried to set tweaks without choosing a solution. "
                "Run `set_XX_solution_from_file` first"
            )
        _ARG_ALL_SLOTS = {None, ":", "all", "ALL"}
        if isinstance(slot_pair, str) and slot_pair.lower() == "global":
            slot_pair = self.gate_tweaks.GLOBAL_CALIBRATION_SLOT
        if len(kwargs) == 0:
            _LOGGER.warning("Tried to set tweaks, but didn't provide any tweaks")

        if isinstance(slot_pair, (tuple, list)):
            slot_pair = tuple(slot_pair)
            assert len(slot_pair) == 2
        elif isinstance(slot_pair, str) and slot_pair in _ARG_ALL_SLOTS:
            pass
        else:
            raise TypeError("Invalid type for slot pair, {}".format(type(slot_pair)))

        for tweak_name, tweak_value in kwargs.items():
            assert tweak_name in self.gate_tweaks.df_columns, "Invalid tweak name"
            if slot_pair in _ARG_ALL_SLOTS:
                self.gate_tweaks.loc[:, tweak_name] = tweak_value
            else:
                self.gate_tweaks.loc[slot_pair, tweak_name] = tweak_value

    def get_XX_tweak(
        self, slot_pair: rf_common.SlotPair, tweak_names: typing.Sequence[str] = None
    ) -> typing.Dict[str, typing.Any]:
        """Get the tweak applied to a XX Gate.

        Args:
            slot_pair (rf_common.SlotPair): The slot index in the ``gate_tweaks``
                file to look up.
            tweak_name (typing.Sequence[str], optional): A specific tweak (
                or a list of tweaks) to look up for the given slot pair.
                Defaults to None. If not provided, will provide a dictionary of all
                tweaks.

        Returns:
            typing.Dict[str, typing.Any]: Mapping from calibration names to
                calibration values.

        """
        if isinstance(slot_pair, str) and slot_pair.lower() == "global":
            slot_pair = self.gate_tweaks.GLOBAL_CALIBRATION_SLOT
        if isinstance(slot_pair, list):
            slot_pair = tuple(slot_pair)
        if tweak_names is None:
            return self.gate_tweaks.loc[slot_pair, :].to_dict()
        if not isinstance(tweak_names, list):
            tweak_names = [tweak_names]  # prevent iterating over chars in string
        # Check all tweaks are valid
        assert all(tweak in self.gate_tweaks.df_columns for tweak in tweak_names)
        # Return dict of {tweak_name: value}
        return {tweak: self.gate_tweaks.loc[slot_pair, tweak] for tweak in tweak_names}

    ###################################################
    #             SET CALIBRATION LEVELS              #
    ###################################################

    def set_AOM_levels(
        self, calibrated_Tpi: typing.List[float], used_shaped: bool = False
    ):
        """Set the AOM levels that calibrate signals to the individual AOM channels.

        It also enables calibration.

        Args:
            calibrated_Tpi: The measured pi times of all channels
            used_shaped: Specifies whether shaped or square Rabi pulses were
                used to perform the calibration whose results are being passed in
        """
        return self.calibration.write_levels(
            nominal_Tpi=1 / (2 * self.physical_params.Rabi_max),
            calibrated_Tpi_array=calibrated_Tpi,
            used_shaped=used_shaped,
            envelope_type=self.physical_params.SK1_AM.envelope_type,
            envelope_scale=self.physical_params.SK1_AM.envelope_scale,
        )

    def set_AOM_saturation_params(self, saturation_params: typing.List[float]):
        """Set AOM saturation parameters that calibrate individual AOM channel signals.

        Args:
            saturation_params:
        """
        self.calibration.write_saturation_params(saturation_params)

    def disable_calibration(
        self, level_active: bool = False, linearity_active: bool = False
    ):
        """This function disables calibration.  Calibration is re-enabled when a new
        calibration object is created, which happens when a new RFCompiler object is
        created, or when new calibration values are written to file with set_AOM_levels.

        Args:
            level_active: Determines whether the level calibration is active
            linearity_active: Determines whether the linearity calibration is active
        """
        self.calibration.disable_calibration(
            level_active=level_active, linearity_active=linearity_active
        )

    ###################################################
    #               GENERATE WAVEFORMS                #
    ###################################################

    def clear_gates(self):

        self.gate_compiler.clear_gate_array()

    def get_circuit_scan_size(self,circuit_filepath: str = ""):
        return self.circuit_interpreter.get_circuit_scan_size(circuit_filepath)

    #@eur_dec.profiler
    def generate_waveforms(self):

        (
            self.total_duration,
            self.timestep_times,
        ) = self.gate_compiler.generate_waveforms()
        return self.total_duration

    def get_timestep_times(self):

        return self.timestep_times

    ###################################################
    #                PROGRAM LABVIEW                  #
    ###################################################

    # @eur_dec.profiler
    def select_exp_to_program(self):

        LVprog.set_experiment(self.name)

    # @eur_dec.profiler
    def program_AWG(self):

        LVprog.program_AWG()

    def set_DDS(self, slots: typing.List[int]):

        LVprog.set_DDS(slots)

    ###################################################
    #    ADD INSTANCES OF GATE PROTOTYPES TO ARRAY    #
    ###################################################

    def add_rabi(
        self,
        slots: typing.List[int],
        detuning: float,
        sideband_order: int,
        duration: float = 0,
        phase: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        wait_after: int = 0,
        gate_name: str = "Rabi",
        scan_parameter_int: int = int(RabiScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This sequence adds one series of Rabi gates to the gate array.  It exposes
        all parameters, both required and optional, of the Rabi gate prototype.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            duration: The duration of the Rabi drive
            phase: The phase of the Rabi drive
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_to_use = ind_amp if ind_amp >= 0 else self.physical_params.Rabi.amp_ind
        global_amp_to_use = (
            global_amp if global_amp >= 0 else self.physical_params.Rabi.amp_global
        )

        scan_parameter = self.RabiScanParameter(scan_parameter_int)

        max_duration = (
            max_value if scan_parameter == self.RabiScanParameter.duration else duration
        )

        for s in slots:
            self.gate_compiler.add(
                gate.Rabi(
                    self.physical_params,
                    slot=s,
                    detuning=sideband_order * detuning,
                    duration=max_duration,
                    phase=phase,
                    ind_amp=ind_amp_to_use,
                    global_amp=global_amp_to_use,
                    wait_after=wait_after,
                    name=gate_name,
                    scan_parameter=scan_parameter,
                    scan_values=scan_array,
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )

    def add_rabi_am(
        self,
        slots: typing.List[int],
        detuning: float,
        sideband_order: int,
        detuning_off: float,
        envelope_duration: float,
        envelope_type_int: int = -1,
        envelope_scale: float = -1,
        phase: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        global_amp_off: float = -1,
        global_delay: float = 0,
        global_duration: float = -1,
        wait_after: int = 0,
        # gate_name: str = "Rabi (AM)",
        gate_name: str = "Rabi (AM)",
        scan_parameter_int: int = int(RabiAMScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This sequence adds one series of amplitude-modulated Rabi gates to the gate
        array.  It exposes all parameters, both required and optional, of the Rabi_AM
        gate prototype.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            detuning_off: The detuning of the global beam during its nominal off state
            envelope_type_int: The function that defines the amplitude envelope
                of the individual waveform
            envelope_duration: The total duration of the individual waveform envelope
            envelope_scale: An optional parameter that sets the width of the
                individual waveform envelope
            phase: The phase of the Rabi drive
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel during its on state
            global_amp_off: The amplitude of the global channel during its off state
            global_delay: The delay between the start of the individual pulse
                and the start of the global pulse
            global_duration: The duration of the global pulse
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to
                which this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_to_use = ind_amp if ind_amp >= 0 else self.physical_params.Rabi.amp_ind
        global_amp_to_use = (
            global_amp if global_amp >= 0 else self.physical_params.Rabi.amp_global
        )
        global_amp_off_to_use = (
            global_amp_off if global_amp_off >= 0 else global_amp_to_use
        )

        scan_parameter = self.RabiAMScanParameter(scan_parameter_int)
        # -1 is not a valid value for envelope_type_int, so we use it as a signal
        # to default to the SK1_AM value
        envelope_type = (
            self.InterpFunctionType(envelope_type_int)
            if envelope_type_int >= 0
            else self.physical_params.SK1_AM.envelope_type
        )
        # Same for envelope_scale
        envelope_scale = (
            envelope_scale
            if envelope_scale >= 0
            else self.physical_params.SK1_AM.envelope_scale
        )

        max_duration = (
            max_value
            if scan_parameter == self.RabiAMScanParameter.envelope_duration
            else envelope_duration
        )
        global_duration_to_use = (
            global_duration if global_duration >= 0 else max_duration - global_delay
        )

        for s in slots:
            self.gate_compiler.add(
                gate.Rabi_AM(
                    self.physical_params,
                    slot=s,
                    detuning_on=sideband_order * detuning,
                    detuning_off=detuning_off,
                    envelope_type=envelope_type,
                    envelope_scale=envelope_scale,
                    envelope_duration=max_duration,
                    phase=phase,
                    ind_amp=ind_amp_to_use,
                    global_amp_on=global_amp_to_use,
                    global_amp_off=global_amp_off_to_use,
                    global_delay=global_delay,
                    global_duration=global_duration_to_use,
                    wait_after=wait_after,
                    name=gate_name,
                    scan_parameter=scan_parameter,
                    scan_values=scan_array,
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )

    def add_rabi_pi(
        self,
        slots: typing.List[int],
        detuning: float,
        sideband_order: int,
        duration: float = 0,
        phase: float = 0,
        ind_amp: float = -1,
        wait_after: int = 0,
        gate_name: str = "Rabi (PI)",
        scan_parameter_int: int = int(RabiPIScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This sequence adds one series of phase-insensitive (or co-propagating) Rabi
        gates to the gate array.  It exposes all parameters, both required and optional,
        of the Rabi_PI gate prototype.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            duration: The duration of the Rabi drive
            phase: The phase of the Rabi drive
            ind_amp: The amplitude of the individual channel
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_to_use = ind_amp if ind_amp >= 0 else self.physical_params.Rabi.amp_ind

        scan_parameter = self.RabiPIScanParameter(scan_parameter_int)

        max_duration = (
            max_value
            if scan_parameter == self.RabiPIScanParameter.duration
            else duration
        )

        for s in slots:
            self.gate_compiler.add(
                gate.Rabi_PI(
                    self.physical_params,
                    slot=s,
                    detuning=sideband_order * detuning,
                    duration=max_duration,
                    phase=phase,
                    ind_amp=ind_amp_to_use,
                    wait_after=wait_after,
                    name=gate_name,
                    scan_parameter=scan_parameter,
                    scan_values=scan_array,
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )

    def add_windunwind(
        self,
        slots: typing.List[int],
        detuning: float,
        sideband_order: int,
        duration_1: float = 0,
        duration_2: float = 0,
        ind_amp_1: float = -1,
        ind_amp_2: float = -1,
        global_amp_1: float = -1,
        global_amp_2: float = -1,
        gate_name: str = "WindUnwind",
        scan_parameter_int: int = int(WindUnwindScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This sequence adds one series of wind-unwind gates to the gate array.  It
        exposes all parameters, both required and optional, of the WindUnwind gate
        prototype, which consists of two subsequent Rabi pulses with the same frequency,
        settable amplitudes, and phases 0 and pi.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            duration_1: The duration of the first Rabi pulse
            duration_2: The duration of the second Rabi pulse
            ind_amp_1: The amplitude of the individual channel
                during the first Rabi pulse
            ind_amp_2: The amplitude of the individual channel
                during the second Rabi pulse
            global_amp_1: The amplitude of the global channel
                during the first Rabi pulse
            global_amp_2: The amplitude of the global channel
                during the second Rabi pulse
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_1_to_use = (
            ind_amp_1 if ind_amp_1 >= 0 else self.physical_params.Rabi.amp_ind
        )
        ind_amp_2_to_use = (
            ind_amp_2 if ind_amp_2 >= 0 else self.physical_params.Rabi.amp_ind
        )
        global_amp_1_to_use = (
            global_amp_1 if global_amp_1 >= 0 else self.physical_params.Rabi.amp_global
        )
        global_amp_2_to_use = (
            global_amp_2 if global_amp_2 >= 0 else self.physical_params.Rabi.amp_global
        )

        scan_parameter = self.WindUnwindScanParameter(scan_parameter_int)

        max_duration_1 = (
            max_value
            if scan_parameter == self.WindUnwindScanParameter.duration_1
            else duration_1
        )
        max_duration_2 = (
            max_value
            if scan_parameter == self.WindUnwindScanParameter.duration_2
            else duration_2
        )

        for s in slots:
            self.gate_compiler.add(
                gate.WindUnwind(
                    self.physical_params,
                    slot=s,
                    detuning=sideband_order * detuning,
                    duration_1=max_duration_1,
                    duration_2=max_duration_2,
                    ind_amp_1=ind_amp_1_to_use,
                    ind_amp_2=ind_amp_2_to_use,
                    global_amp_1=global_amp_1_to_use,
                    global_amp_2=global_amp_2_to_use,
                    name=gate_name,
                    scan_parameter=scan_parameter,
                    scan_values=scan_array,
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )

    def add_crosstalk_calib(
        self,
        slots_strong: typing.List[int],
        slots_weak: typing.List[int],
        detuning: float,
        sideband_order: int,
        duration: float = 0,
        ind_amp_strong: float = -1,
        ind_amp_weak: float = -1,
        global_amp: float = -1,
        phase_strong: float = 0.0,
        phase_weak: float = 0.0,
        gate_name: str = "WindUnwind",
        scan_parameter_int: int = int(CrosstalkCalibScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This sequence adds one series of crosstalk calibration gates to the gate array.
        This gate applies a strong Rabi drive on one slot and a weak Rabi drive on another
        slot.  By sweeping the phase and amplitude of the weak Rabi drive so as to minimize
        the population transfer, we can map out the phase and amplitude of crosstalk
        between the slots.  It exposes all parameters, both required and optional, of the
        CrosstalkCalib gate prototype, which consists of two subsequent Rabi pulses with
        the same frequency, settable amplitudes, and phases 0 and pi.

        Args:
            slots_strong: The slots to which the strong Rabi pulses will be applied,
                in series
            slots_weak: The slots to which the weak Rabi pulses will be applied,
                in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            duration: The duration of the Rabi pulse
            ind_amp_strong: The amplitude of the individual channel applying the strong
                Rabi pulse
            ind_amp_weak: The amplitude of the individual channel applying the weak
                Rabi pulse
            global_amp: The amplitude of the global channel
            phase_strong: The phase of the strong Rabi drive
            phase_weak: The phase of the weak Rabi drive
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which this
                gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be suppressed for
                the gate we are adding (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_strong_to_use = (
            ind_amp_strong if ind_amp_strong >= 0 else self.physical_params.Rabi.amp_ind
        )
        global_amp_to_use = (
            global_amp if global_amp >= 0 else self.physical_params.Rabi.amp_global
        )

        scan_parameter = self.CrosstalkCalibScanParameter(scan_parameter_int)

        max_duration = (
            max_value
            if scan_parameter == self.CrosstalkCalibScanParameter.duration
            else duration
        )

        for (ss, sw) in zip(slots_strong, slots_weak):
            self.gate_compiler.add(
                gate.CrosstalkCalib(
                    self.physical_params,
                    slot_strong=ss,
                    slot_weak=sw,
                    detuning=sideband_order * detuning,
                    duration=max_duration,
                    ind_amp_strong=ind_amp_strong_to_use,
                    ind_amp_weak=ind_amp_weak,
                    global_amp=global_amp_to_use,
                    phase_strong=phase_strong,
                    phase_weak=phase_weak,
                    name=gate_name,
                    scan_parameter=scan_parameter,
                    scan_values=scan_array,
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )

    def add_fastecho(
        self,
        slot: int,
        detuning: float,
        duration: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        wait_after: int = 0,
        active_beams: FastEchoActiveBeams = FastEchoActiveBeams.both,
        sideband_imbalance: float = 0,
        echo_duration: float = 0.1,
        gate_name: str = "FastEcho",
        scan_parameter_int: int = int(FastEchoScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """
        This sequence adds one fast echo gate to the gate array.

        It exposes all parameters, both required and optional, of the FastEcho
        gate prototype, which consists of a static pulse on the individual and
        a fast echo pulse, which quickly switches between the two sidebands with
        a relative phase offset of pi, on the global.

        Args:
            slot: The slot to which the Fast Echo gate will be applied
            detuning: The detuning of the two sidebands from the carrier
            duration: The total duration of the gate
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            wait_after: Determines whether we insert a long wait after this gate
            active_beams: Determines which of the beams are applied during this gate
            sideband_imbalance: The amplitude imbalance (additive, 0-biased)
                between the blue and red sideband pulses
            echo_duration: The total time of one red or blue sideband pulse
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_to_use = ind_amp if ind_amp >= 0 else self.physical_params.Rabi.amp_ind
        global_amp_to_use = (
            global_amp if global_amp >= 0 else self.physical_params.Rabi.amp_global
        )

        scan_parameter = self.FastEchoScanParameter(scan_parameter_int)

        max_duration = (
            max_value
            if scan_parameter == self.FastEchoScanParameter.duration
            else duration
        )

        self.gate_compiler.add(
            gate.FastEcho(
                self.physical_params,
                slot=slot,
                detuning=detuning,
                duration=max_duration,
                ind_amp=ind_amp_to_use,
                global_amp=global_amp_to_use,
                wait_after=wait_after,
                active_beams=active_beams,
                sideband_imbalance=sideband_imbalance,
                echo_duration=echo_duration,
                name=gate_name,
                scan_parameter=scan_parameter,
                scan_values=scan_array,
            ),
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
        )

    def add_bichromatic(
        self,
        slots: typing.List[int],
        diff_detuning: float,
        common_detuning: float=0,
        duration: float = 0,
        qubit_phase: float = np.pi/2,
        motional_phase: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        sideband_imbalance: float = 0,
        wait_after: int = 0,
        active_beams: BichromaticActiveBeams = BichromaticActiveBeams.both,
        gate_name: str = "Bichromatic",
        scan_parameter_int: int = int(BichromaticScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """
        This sequence adds one bichromatic tone of red/blue sidebandz to the gate array.

        It exposes all parameters, both required and optional, of the FastEcho
        gate prototype, which consists of a static pulse on the individual and
        a fast echo pulse, which quickly switches between the two sidebands with
        a relative phase offset of pi, on the global.

        Args:
            slot: The slot to which the Bichromatic gate will be applied
            diff_detuning: The differential detuning of each sideband from the carrier
            common_detuning: The common detuning of each sideband from the carrier
            duration: The total duration of the gate
            phase: Common phase added to both red and blue tones at turn on
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            sideband_imbalance: The amplitude imbalance (additive, 0-biased)
                between the blue and red sideband pulses
            wait_after: Determines whether we insert a long wait after this gate
            active_beams: Choose which beams the two tones are on. If both then global.
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_to_use = ind_amp if ind_amp >= 0 else self.physical_params.Rabi.amp_ind
        global_amp_to_use = (
            global_amp if global_amp >= 0 else self.physical_params.Rabi.amp_global
        )

        scan_parameter = self.BichromaticScanParameter(scan_parameter_int)

        max_duration = (
            max_value
            if scan_parameter == self.BichromaticScanParameter.duration
            else duration
        )
        for s in slots:
            self.gate_compiler.add(
                gate.Bichromatic(
                    physical_params=self.physical_params,
                    slot=s,
                    diff_detuning=diff_detuning,
                    common_detuning=common_detuning,
                    duration=max_duration,
                    qubit_phase=qubit_phase,
                    motional_phase=motional_phase,
                    ind_amp=ind_amp_to_use,
                    global_amp=global_amp_to_use,
                    sideband_imbalance=sideband_imbalance,
                    wait_after=wait_after,
                    active_beams=active_beams,
                    name=gate_name,
                    scan_parameter=scan_parameter,
                    scan_values=scan_array
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan)


    def add_SK1(
        self,
        slots: typing.List[int],
        theta: float,
        phi: float,
        ind_amp: float = -1,
        global_amp: float = -1,
        wait_after: int = 0,
        gate_name: str = "SK1",
        scan_parameter_int: int = int(SK1ScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """
        This sequence adds one series of SK1 gates to the gate array.

        Exposes all parameters, both required and optional, of the SK1 gate prototype.

        Args:
            slots: The slots to which SK1 gates will be applied, in series
            theta: The rotation angle of the SK1 pulse
            phi: The rotation axis angle of the SK1 pulse
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_to_use = ind_amp if ind_amp >= 0 else self.physical_params.SK1.amp_ind
        global_amp_to_use = (
            global_amp if global_amp >= 0 else self.physical_params.SK1.amp_global
        )

        scan_parameter = self.SK1ScanParameter(scan_parameter_int)

        for s in slots:
            self.gate_compiler.add(
                gate.SK1(
                    self.physical_params,
                    slot=s,
                    theta=theta,
                    phi=phi,
                    ind_amp=ind_amp_to_use,
                    global_amp=global_amp_to_use,
                    wait_after=wait_after,
                    name=gate_name,
                    scan_parameter=scan_parameter,
                    scan_values=scan_array,
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )

    def add_SK1_am(
        self,
        slots: typing.List[int],
        theta: float = np.pi / 2,
        phi: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        use_global_segment_durations: bool = False,
        wait_after: int = 0,
        gate_name: str = "SK1 (AM)",
        scan_parameter_int: int = int(SK1AMScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """Add one series of amplitude-modulated SK1 gates to the gate array.

        It exposes all parameters, both required and optional, of the SK1_AM
        gate prototype.

        Args:
            slots: The slots to which SK1 gates will be applied, in series
            theta: The rotation angle of the SK1 pulse
            phi: The rotation axis angle of the SK1 pulse
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            use_global_segment_durations: Determines whether the durations of
                the three SK1 segments are set from global values in physical_params
                or calculated from the known envelope shape
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )

        ind_amp_to_use = (
            ind_amp if ind_amp >= 0 else self.physical_params.SK1_AM.amp_ind
        )
        global_amp_to_use = (
            global_amp if global_amp >= 0 else self.physical_params.SK1_AM.amp_global
        )

        scan_parameter = self.SK1AMScanParameter(scan_parameter_int)

        # Fixed phase correction will be applied symmetrically around the SK1AM pulse.
        phase_correction = self.physical_params.SK1_AM.phase_correction

        for s in slots:
            self.gate_compiler.add(
                gate.Phase(
                    self.physical_params,
                    slot=s,
                    phase=phase_correction/2,
                    name="SK1AM Phase Correction"
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=True,
            )
            self.gate_compiler.add(
                gate.SK1_AM(
                    self.physical_params,
                    slot=s,
                    theta=theta,
                    phi=phi,
                    ind_amp=ind_amp_to_use,
                    global_amp=global_amp_to_use,
                    use_global_segment_durations=use_global_segment_durations,
                    wait_after=wait_after,
                    name=gate_name,
                    scan_parameter=scan_parameter,
                    scan_values=scan_array,
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )
            self.gate_compiler.add(
                gate.Phase(
                    self.physical_params,
                    slot=s,
                    phase=phase_correction/2,
                    name="SK1AM Phase Correction"
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=True,
            )

    def add_XX(
        self,
        slots: rf_common.SlotPair,
        gate_sign: float = +1,
        phi_ind1: float = 0,
        phi_ind2: float = 0,
        phi_global: float = 0,
        phi_motion: float = 0,
        wait_after: int = 0,
        gate_name: str = "XX",
        scan_parameter_int: int = int(XXScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
        **gate_param_modifications
    ):
        """This sequence adds one XX gate to the gate array.

        Exposes all parameters, both required and optional, of the XX gate prototype.

        Args:
            slots: The two slots between which the gate is applied
            gate_sign: The sign (either +1 or -1) of the geometric phase required
            phi_ind1: The sign of the individual tone applied to slot 1
            phi_ind2: The sign of the individual tone applied to slot 2
            phi_global: The common phase of the global tone
            phi_motion: The initial phase difference between the blue and red
                sidebands on the global tone
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)

        Kwargs:
            This function accepts kwargs that are valid tweaks/columns in
            :class:`.gate_parameters.GateCalibrations`. Evaluated using
            :attr:`GateCalibrations.df_columns`
        """
        # check all kwargs are valid keys in the gate_tweaks struct
        try:
            assert all(
                kw in self.gate_tweaks.df_columns
                for kw in gate_param_modifications.keys()
            )
        except AssertionError:
            raise KeyError(
                "Keys not in gate_tweaks: {}".format(
                    set(gate_param_modifications.keys()).difference(
                        self.gate_tweaks.df_columns
                    )
                )
            )
        scan_array = (
            np.linspace(min_value, max_value, N_points) if N_points != 0 else None
        )
        phase_offset = self.physical_params.XX.phase_offset
        scan_parameter = self.XXScanParameter(scan_parameter_int)

        self.gate_compiler.add(
            gate.XX(
                self.physical_params,
                gate_solution=self.gate_solution,
                gate_param_tweaks=self.gate_tweaks,
                slots=slots,
                gate_sign=gate_sign,
                phi_ind1=phi_ind1,
                phi_ind2=phi_ind2,
                phi_global=phi_global+phase_offset,
                phi_motion=phi_motion,
                wait_after=wait_after,
                name=gate_name,
                scan_parameter=scan_parameter,
                scan_values=scan_array,
                **gate_param_modifications
            ),
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
        )

    def add_phase(
        self,
        slots: typing.List[int],
        phase: float,
        wait_after: int = 0,
        gate_name: str = "Phase",
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This function adds one phase gate, (implemented entirely in software).

        Args:
            slots: The slots to which phase gates will be applied
            phase: The amount of phase acquired in the gate
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        for s in slots:
            self.gate_compiler.add(
                gate.Phase(self.physical_params, slot=s, phase=phase, wait_after=wait_after, name=gate_name),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )

    def add_oneQ_blank(
        self,
        slots: typing.List[int],
        wait_after: int = 0,
        gate_name: str = "Blank",
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This function inserts a blank 1Q gate into the array.

        Args:
            slots: The slots to which phase gates will be applied
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        for s in slots:
            self.gate_compiler.add(
                gate.Blank(
                    self.physical_params,
                    slots=[s],
                    twoQ_gate=0,
                    wait_after=wait_after,
                    name=gate_name,
                ),
                circuit_index=circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )

    def add_twoQ_blank(
        self,
        slots: typing.List[int],
        wait_after: int = 0,
        gate_name: str = "Blank",
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This function inserts a blank 2Q gate into the array.

        Args:
            slots: The slots to which phase gates will be applied
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            gate_name: The name of this specific gate within the gate sequence
            circuit_index: In circuit scan mode, the specific circuit to which
                this gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be
                suppressed for the gate we are adding
                (i.e., only one waveform file will be written)
        """

        self.gate_compiler.add(
            gate.Blank(
                self.physical_params,
                slots=slots,
                twoQ_gate=1,
                wait_after=wait_after,
                name=gate_name,
            ),
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
        )

    ###################################################
    #            DEFINE COMMON EXPERIMENTS            #
    ###################################################

    class ExperimentsAvailable(enum.IntEnum):
        Circuit = 0
        Rabi = 1
        Rabi_AM = 2
        Rabi_PI = 3
        Calibrate = 4
        SK1 = 5
        SK1_AM = 6
        XX = 7
        Linescan = 8
        XX_parity_scan = 9
        XX_with_analysis = 10
        Stabilizer_readout = 11

    def circuit(
        self,
        circuit_filepath: str = "",
        suppress_circuit_scan: typing.Union[np.ndarray, bool] = False,
        circuit_index: int = 0,
        use_SK1_AM: bool = True,
        exp_name: str = "Circuit",
        print_circuit: bool = False,
        print_gate_list: bool = False,
    ):

        self.name = exp_name
        return self.circuit_interpreter.load_circuit_from_file(
            cirq_file=circuit_filepath,
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
            use_SK1_AM=use_SK1_AM,
            print_circuit=print_circuit,
            print_gate_list=print_gate_list,
        )

    def rabi_exp(
        self,
        slots: typing.List[int],
        detuning: float,
        sideband_order: int,
        duration: float = 0,
        phase: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "Rabi",
        scan_parameter_int: int = int(RabiScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """This experiment consists of one fully scannable series of Rabi pulses.

        Exposes all parameters, both required and optional, of the Rabi gate prototype.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            duration: The duration of the Rabi drive
            phase: The phase of the Rabi drive
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name
        gate_name = "Rabi"

        self.add_rabi(
            slots=slots,
            detuning=detuning,
            sideband_order=sideband_order,
            duration=duration,
            phase=phase,
            ind_amp=ind_amp,
            global_amp=global_amp,
            wait_after=wait_after,
            gate_name=gate_name,
            scan_parameter_int=scan_parameter_int,
            min_value=min_value,
            max_value=max_value,
            N_points=N_points,
        )

    def rabi_am_exp(
        self,
        slots: typing.List[int],
        detuning: float,
        sideband_order: int,
        detuning_off: float,
        envelope_duration: float,
        envelope_type_int: int = -1,
        envelope_scale: float = -1,
        phase: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        global_amp_off: float = -1,
        global_delay: float = 0,
        global_duration: float = -1,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "Rabi (AM)",
        scan_parameter_int: int = int(RabiAMScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """This experiment consists of one fully scannable series of Rabi pulses.

        Exposes all parameters, both required and optional, of the Rabi gate prototype.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            detuning: The detuning (absolute value) of the Rabi drive from the
                carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            detuning_off: The detuning of the global beam during its nominal
                off state
            envelope_type_int: The function that defines the amplitude envelope
                of the individual waveform
            envelope_duration: The total duration of the individual waveform envelope
            envelope_scale: An optional parameter that sets the width of the
                individual waveform envelope
            phase: The phase of the Rabi drive
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            global_amp_off: The amplitude of the global channel during its off state
            global_delay: The delay between the start of the individual pulse
                and the start of the global pulse
            global_duration: The duration of the global pulse
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name
        gate_name = "Rabi (AM)"

        self.add_rabi_am(
            slots=slots,
            detuning=detuning,
            sideband_order=sideband_order,
            detuning_off=detuning_off,
            envelope_duration=envelope_duration,
            envelope_type_int=envelope_type_int,
            envelope_scale=envelope_scale,
            phase=phase,
            ind_amp=ind_amp,
            global_amp=global_amp,
            global_amp_off=global_amp_off,
            global_delay=global_delay,
            global_duration=global_duration,
            wait_after=wait_after,
            gate_name=gate_name,
            scan_parameter_int=scan_parameter_int,
            min_value=min_value,
            max_value=max_value,
            N_points=N_points,
        )

    def rabi_pi_exp(
        self,
        slots: typing.List[int],
        detuning: float,
        sideband_order: int,
        duration: float = 0,
        phase: float = 0,
        ind_amp: float = -1,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "Rabi (PI)",
        scan_parameter_int: int = int(RabiPIScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """This experiment consists of one fully scannable series of Rabi pulses.

        Exposes all parameters, both required and optional, of the Rabi gate prototype.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            duration: The duration of the Rabi drive
            phase: The phase of the Rabi drive
            ind_amp: The amplitude of the individual channel
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name
        gate_name = "Rabi (PI)"
        # HACK HACK HACK
        # self.add_rabi_pi(
        #     slots=slots,
        #     detuning=detuning,
        #     sideband_order=sideband_order,
        #     duration=1000,
        #     phase=-0.0*np.pi/2,
        #     ind_amp=ind_amp,
        #     wait_after=wait_after
        # )
        self.add_rabi_pi(
            slots=slots,
            detuning=detuning,
            sideband_order=sideband_order,
            duration=duration,
            phase=phase,
            ind_amp=ind_amp,
            wait_after=wait_after,
            gate_name=gate_name,
            scan_parameter_int=scan_parameter_int,
            min_value=min_value,
            max_value=max_value,
            N_points=N_points,
        )


    def calibrate(
        self,
        slots: typing.List[int],
        use_shaped: bool = False,
        ind_amp: float = 1000,
        global_amp: float = 1000,
        envelope_type_int: int = -1,
        envelope_scale: float = -1,
        exp_name: str = "Calibrate",
        min_duration: float = 0,
        max_duration: float = 0,
        N_points: int = 0,
    ):
        """This experiment enables us to calibrate the amplitudes of all individual
        channels.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            use_shaped: Determines whether we use a square or shaped Rabi pulse
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            envelope_type_int: The envelope function to use, if using shaped pulses
            envelope_scale: The envelope scale factor to use, if using shaped pulses
            exp_name: The name of this experiment
            min_duration: The minimum Rabi pulse duration
            max_duration: The maximum Rabi pulse duration
            N_points: The number of points of the parameter being swept
        """

        self.name = exp_name
        gate_name = "Rabi calibrate"

        self.disable_calibration(level_active=False, linearity_active=True)

        if use_shaped:
            self.add_rabi_am(
                slots=slots,
                detuning=0,
                sideband_order=0,
                detuning_off=0,
                envelope_duration=-1,
                envelope_type_int=envelope_type_int,
                envelope_scale=envelope_scale,
                phase=0,
                ind_amp=ind_amp,
                global_amp=global_amp,
                global_amp_off=global_amp,
                global_delay=0,
                global_duration=-1,
                wait_after=0,
                gate_name=gate_name,
                scan_parameter_int=int(self.RabiAMScanParameter.envelope_duration),
                min_value=min_duration,
                max_value=max_duration,
                N_points=N_points,
            )

        else:
            self.add_rabi(
                slots=slots,
                detuning=0,
                sideband_order=0,
                duration=0,
                phase=0,
                ind_amp=ind_amp,
                global_amp=global_amp,
                wait_after=0,
                gate_name=gate_name,
                scan_parameter_int=int(self.RabiScanParameter.duration),
                min_value=min_duration,
                max_value=max_duration,
                N_points=N_points,
            )

    def windunwind(
        self,
        slots: typing.List[int],
        detuning: float = 0,
        sideband_order: int = 0,
        duration_1: float = 0,
        duration_2: float = 0,
        ind_amp_1: float = -1,
        ind_amp_2: float = -1,
        global_amp_1: float = -1,
        global_amp_2: float = -1,
        exp_name: str = "WindUnwind",
    ):
        """This function enqueues a wind-unwind experiment, which consists of two
        subsequent Rabi pulses with the same frequency, settable amplitudes, and phases
        0 and pi.

        Args:
            slots: The slots to which Rabi gates will be applied, in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            duration_1: The duration of the first Rabi pulse
            duration_2: The duration of the second Rabi pulse
            ind_amp_1: The amplitude of the individual channel
                during the first Rabi pulse
            ind_amp_2: The amplitude of the individual channel
                during the second Rabi pulse
            global_amp_1: The amplitude of the global channel
                during the first Rabi pulse
            global_amp_2: The amplitude of the global channel
                during the second Rabi pulse
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.name = exp_name

        self.add_windunwind(
            slots=slots,
            detuning=detuning,
            sideband_order=sideband_order,
            duration_1=duration_1,
            duration_2=duration_2,
            ind_amp_1=ind_amp_1,
            ind_amp_2=ind_amp_2,
            global_amp_1=global_amp_1,
            global_amp_2=global_amp_2,
        )

    def charge_response_probe(
        self,
        push_slots: typing.List[int],
        probe_slot: int,
        push_detuning: float = 0,
        probe_detuning: float = 0,
        probe_sideband_order: int = 0,
        push_duration: float = 0,
        probe_duration: float = 0,
        push_ind_amp: float = -1,
        probe_ind_amp: float = -1,
        probe_global_amp: float = -1,
        use_PI_probe: bool = False,
        interpulse_delay: float = 0,
        exp_name: str = "ChargeResponseProbe",
        scan_parameter_int: int = int(
            rf_common.ChargeResponseProbeScanParameter.static
        ),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """The Charge Response Probe experiment consists of two Rabi pulses spaced by a delay.
        The first pulse is the "push" pulse, which is meant to either heat a motional mode or
        excite axial motion.  The second is the "probe" pulse, which measures the ions' motion
        by driving Rabi.  The push pulse can be either square or shaped, and will involve
        applying two tones with a variable detuning between them.  The probe pulse can be either
        phase-sensitive or -insensitive.

        Args:
            push_slots: The slots to which the push pulses will be applied, in series
            probe_slot: The slot to which the probe pulse will be applied
            push_detuning: The frequency difference of the two individual drives in the
                push pulse
            probe_detuning: The detuning (absolute value) of the Rabi drive from the
                carrier for the probe pulse
            probe_sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB) with the probe pulse
            push_duration: The duration of the probe pulse
            probe_duration: The duration of the second Rabi pulse
            push_ind_amp: The amplitude of the individual channel during the push pulse
            probe_ind_amp: The amplitude of the individual channel during the
                probe pulse
            probe_global_amp: The amplitude of the global channel during the
                probe pulse
            use_PI_probe: Determines whether the probe pulse is
                phase-sensitive or -insensitive
            interpulse_delay: The delay inserted after every push pulse
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.wait_after_time = interpulse_delay
        push_wait_after = 1 if interpulse_delay > 0 else 0

        self.name = exp_name

        # Set the scan parameters of the two pulses appropriately,
        # with static as the default
        push_scan_parameter_int = int(self.RabiPIScanParameter.static)
        probe_scan_parameter_int = (
            int(self.RabiPIScanParameter.static)
            if use_PI_probe
            else int(self.RabiScanParameter.static)
        )
        if scan_parameter_int == int(
            self.ChargeResponseProbeScanParameter.push_duration
        ):
            push_scan_parameter_int = int(self.RabiPIScanParameter.duration)
        elif scan_parameter_int == int(
            self.ChargeResponseProbeScanParameter.probe_duration
        ):
            probe_scan_parameter_int = (
                int(self.RabiPIScanParameter.duration)
                if use_PI_probe
                else int(self.RabiScanParameter.duration)
            )
        elif scan_parameter_int == int(
            self.ChargeResponseProbeScanParameter.push_detuning
        ):
            push_scan_parameter_int = int(self.RabiPIScanParameter.detuning)
        elif scan_parameter_int == int(
            self.ChargeResponseProbeScanParameter.probe_detuning
        ):
            probe_scan_parameter_int = (
                int(self.RabiPIScanParameter.detuning)
                if use_PI_probe
                else int(self.RabiScanParameter.detuning)
            )
        elif scan_parameter_int == int(
            self.ChargeResponseProbeScanParameter.push_ind_amp
        ):
            push_scan_parameter_int = int(self.RabiPIScanParameter.ind_amplitude)

        # The PI Rabi gates are meant to drive the carrier, which means that
        # they set the frequency difference between the two tones applied on
        # the individual tones to account for the frequency difference between
        # the driving Raman comb tooth and the carrier transition, plus the
        # desired detuning (from the carrier).  Here, we just want their
        # frequency difference to be equal to the detuning, so we subtract out
        # the other frequencies that the Rabi_PI gate will add back in.
        push_detuning_to_use = (self.physical_params.f_ind + 140) - (
            self.physical_params.f_carrier + push_detuning
        )

        # We also must do this math for the scan range if we are scanning the
        # push detuning
        if scan_parameter_int == int(
            self.ChargeResponseProbeScanParameter.push_detuning
        ):
            min_value_to_use = (self.physical_params.f_ind + 140) - (
                self.physical_params.f_carrier + min_value
            )
            max_value_to_use = (self.physical_params.f_ind + 140) - (
                self.physical_params.f_carrier + max_value
            )
        else:
            min_value_to_use = min_value
            max_value_to_use = max_value

        self.add_rabi_pi(
            slots=push_slots,
            detuning=push_detuning_to_use,
            sideband_order=1,
            duration=push_duration,
            ind_amp=push_ind_amp,
            wait_after=push_wait_after,
            gate_name="Push Pulse",
            scan_parameter_int=push_scan_parameter_int,
            min_value=min_value_to_use,
            max_value=max_value_to_use,
            N_points=N_points,
        )

        if use_PI_probe:
            self.add_rabi_pi(
                slots=[probe_slot],
                detuning=probe_detuning,
                sideband_order=probe_sideband_order,
                duration=probe_duration,
                ind_amp=probe_ind_amp,
                wait_after=0,
                gate_name="Probe Pulse",
                scan_parameter_int=probe_scan_parameter_int,
                min_value=min_value_to_use,
                max_value=max_value_to_use,
                N_points=N_points,
            )
        else:
            self.add_rabi(
                slots=[probe_slot],
                detuning=probe_detuning,
                sideband_order=probe_sideband_order,
                duration=probe_duration,
                ind_amp=probe_ind_amp,
                global_amp=probe_global_amp,
                wait_after=0,
                gate_name="Probe Pulse",
                scan_parameter_int=probe_scan_parameter_int,
                min_value=min_value_to_use,
                max_value=max_value_to_use,
                N_points=N_points,
            )

    def crosstalk_calib(
        self,
        slots_strong: typing.List[int],
        slots_weak: typing.List[int],
        detuning: float,
        sideband_order: int,
        duration: float = 0,
        ind_amp_strong: float = -1,
        ind_amp_weak: float = -1,
        global_amp: float = -1,
        phase_strong: float = 0.0,
        phase_weak: float = 0.0,
        exp_name: str = "Crosstalk Calib",
        scan_parameter_int: int = int(CrosstalkCalibScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        circuit_index: int = 0,
        suppress_circuit_scan: bool = False,
    ):
        """This sequence adds one series of crosstalk calibration gates to the gate array.

        This gate applies a strong Rabi drive on one slot and a weak Rabi drive on another
        slot. By sweeping the phase and amplitude of the weak Rabi drive so as to minimize
        the population transfer, we can map out the phase and amplitude of crosstalk
        between the slots.

        It exposes all parameters, both required and optional, of the
        CrosstalkCalib gate prototype, which consists of two subsequent Rabi pulses with
        the same frequency, settable amplitudes, and phases 0 and pi.

        Args:
            slots_strong: The slots to which the strong Rabi pulses will be applied,
                in series
            slots_weak: The slots to which the weak Rabi pulses will be applied,
                in series
            detuning: The detuning (absolute value) of the Rabi drive from the carrier
            sideband_order: The order of the sideband to probe
                (i.e., +1 for BSB, -1 for RSB)
            duration: The duration of the Rabi pulse
            ind_amp_strong: The amplitude of the individual channel applying the
                strong Rabi pulse
            ind_amp_weak: The amplitude of the individual channel applying the weak
                Rabi pulse
            global_amp: The amplitude of the global channel
            phase_strong: The phase of the strong Rabi drive
            phase_weak: The phase of the weak Rabi drive
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
            circuit_index: In circuit scan mode, the specific circuit to which this
                gate is being added
            suppress_circuit_scan: Flags whether circuit scan will be suppressed for
                the gate we are adding (i.e., only one waveform file will be written)
        """

        self.name = exp_name
        gate_name = "Crosstalk Calib"

        self.add_crosstalk_calib(
            slots_strong=slots_strong,
            slots_weak=slots_weak,
            detuning=detuning,
            sideband_order=sideband_order,
            duration=duration,
            ind_amp_strong=ind_amp_strong,
            ind_amp_weak=ind_amp_weak,
            global_amp=global_amp,
            phase_strong=phase_strong,
            phase_weak=phase_weak,
            gate_name=gate_name,
            scan_parameter_int=scan_parameter_int,
            min_value=min_value,
            max_value=max_value,
            N_points=N_points,
            circuit_index=circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
        )

    def starkshift(
        self,
        slot: int,
        active_beams_1: FastEchoActiveBeams,
        active_beams_2: FastEchoActiveBeams,
        duration: float,
        interpulse_delay: float,
        detuning: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        sideband_imbalance: float = 0,
        echo_duration: float = 0.1,
        exp_name: str = "StarkShift",
        scan_parameter_int: int = int(FastEchoScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """
        This function enqueues a Stark shift measurement, which consists of two
        fast echo pulses separated by a long delay.

        The fast echo pulse quickly switches between the two sidebands with a
        relative phase offset of pi, on the global. The set of beams that is
        applied during each pulse can be set independently.

        Args:
            slot: The slot to which the Fast Echo gate will be applied
            active_beams_1: Determines which of the beams are applied during
                the first fast echo gate
            active_beams_2: Determines which of the beams are applied during
                the second fast echo gate
            duration: The total duration of the gate
            interpulse_delay: The delay between the two FastEcho gates
            detuning: The detuning of the two sidebands from the carrier
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            sideband_imbalance: The amplitude imbalance (additive, 0-biased)
                between the blue and red sideband pulses
            echo_duration: The total time of one red or blue sideband pulse
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.name = exp_name
        gate_name = "FastEcho"

        self.wait_after_time = interpulse_delay

        self.add_fastecho(
            slot=slot,
            detuning=detuning,
            duration=duration,
            ind_amp=ind_amp,
            global_amp=global_amp,
            wait_after=1,
            active_beams=active_beams_1,
            sideband_imbalance=sideband_imbalance,
            echo_duration=echo_duration,
            gate_name=gate_name,
            scan_parameter_int=scan_parameter_int,
            min_value=min_value,
            max_value=max_value,
            N_points=N_points,
        )
        self.add_fastecho(
            slot=slot,
            detuning=detuning,
            duration=duration,
            ind_amp=ind_amp,
            global_amp=global_amp,
            wait_after=0,
            active_beams=active_beams_2,
            sideband_imbalance=sideband_imbalance,
            echo_duration=echo_duration,
            gate_name=gate_name,
            scan_parameter_int=scan_parameter_int,
            min_value=min_value,
            max_value=max_value,
            N_points=N_points,
        )

    def SK1_exp(
        self,
        slots: typing.List[int],
        theta: float,
        phi: float,
        ind_amp: float = -1,
        global_amp: float = -1,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "SK1",
        scan_parameter_int: int = int(SK1AMScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """
        This experiment consists of one fully scannable series of SK1 pulses.

        It exposes all parameters, both required
        and optional, of the SK1 gate prototype.

        Args:
            slots: The slots to which SK1 gates will be applied, in series
            theta: The rotation angle of the SK1 pulse
            phi: The rotation axis angle of the SK1 pulse
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name
        gate_name = "SK1"

        if scan_parameter_int == int(self.SK1ScanParameter.theta):

            theta_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )
            max_theta = max(min_value, max_value)

            for i, th in enumerate(theta_values):
                self.add_SK1(
                    slots=slots,
                    theta=th,
                    phi=phi,
                    ind_amp=ind_amp,
                    global_amp=global_amp,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=int(self.SK1ScanParameter.static),
                    circuit_index=i,
                )
                # HACK - we add this second dummy SK1 pulse to coerce the 1Q gate
                # lengths to be the same across all i
                self.add_SK1(
                    slots=slots,
                    theta=max_theta,
                    phi=phi,
                    ind_amp=0,
                    global_amp=0,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=int(self.SK1ScanParameter.static),
                    circuit_index=i,
                )
                self.gate_compiler.compile_circuit(i)

        elif scan_parameter_int == int(self.SK1ScanParameter.Stark_shift):

            Stark_shift_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            for i, ss in enumerate(Stark_shift_values):
                self.physical_params.SK1.Stark_shift = ss
                self.add_SK1(
                    slots=slots,
                    theta=theta,
                    phi=phi,
                    ind_amp=ind_amp,
                    global_amp=global_amp,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=int(self.SK1ScanParameter.static),
                    circuit_index=i,
                )
                self.gate_compiler.compile_circuit(i)

        else:

            self.add_SK1(
                slots=slots,
                theta=theta,
                phi=phi,
                ind_amp=ind_amp,
                global_amp=global_amp,
                wait_after=wait_after,
                gate_name=gate_name,
                scan_parameter_int=scan_parameter_int,
                min_value=min_value,
                max_value=max_value,
                N_points=N_points,
            )

    def SK1_am_exp(
        self,
        slots: typing.List[int],
        theta: float = np.pi / 2,
        phi: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        use_global_segment_durations: bool = False,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "SK1",
        scan_parameter_int: int = int(SK1AMScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """This experiment consists of one fully scannable series of amplitude-modulated
        SK1 pulses.

        It exposes all parameters, both required and optional, of the
        SK1_AM gate prototype.

        Args:
            slots: The slots to which SK1 gates will be applied, in series
            theta: The rotation angle of the SK1 pulse
            phi: The rotation axis angle of the SK1 pulse
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            use_global_segment_durations: Determines whether the durations of
                the three SK1 segments are set from global values in physical_params
                or calculated from the known envelope shape
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name
        gate_name = "SK1 (AM)"

        if scan_parameter_int == int(self.SK1AMScanParameter.theta):
            theta_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )
            max_theta = max(min_value, max_value)

            for i, th in enumerate(theta_values):
                self.add_SK1_am(
                    slots=slots,
                    theta=th,
                    phi=phi,
                    ind_amp=ind_amp,
                    global_amp=global_amp,
                    use_global_segment_durations=use_global_segment_durations,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=int(self.SK1AMScanParameter.static),
                    circuit_index=i,
                )
                # HACK - we add this second dummy SK1 pulse to coerce the
                # 1Q gate lengths to be the same across all i
                self.add_SK1_am(
                    slots=slots,
                    theta=max_theta,
                    phi=phi,
                    ind_amp=0,
                    global_amp=0,
                    use_global_segment_durations=use_global_segment_durations,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=int(self.SK1AMScanParameter.static),
                    circuit_index=i,
                )
                self.gate_compiler.compile_circuit(i)

        elif scan_parameter_int == int(self.SK1AMScanParameter.Stark_shift):

            Stark_shift_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            for i, ss in enumerate(Stark_shift_values):
                self.physical_params.SK1_AM.Stark_shift = ss
                self.add_SK1_am(
                    slots=slots,
                    theta=theta,
                    phi=phi,
                    ind_amp=ind_amp,
                    global_amp=global_amp,
                    use_global_segment_durations=use_global_segment_durations,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=int(self.SK1AMScanParameter.static),
                    circuit_index=i,
                )
                self.gate_compiler.compile_circuit(i)

        else:

            self.add_SK1_am(
                slots=slots,
                theta=theta,
                phi=phi,
                ind_amp=ind_amp,
                global_amp=global_amp,
                use_global_segment_durations=use_global_segment_durations,
                wait_after=wait_after,
                gate_name=gate_name,
                scan_parameter_int=scan_parameter_int,
                min_value=min_value,
                max_value=max_value,
                N_points=N_points,
            )

    def cross_SK1(
        self,
        slots: typing.List[int],
        theta: float = np.pi / 2,
        phi: float = 0,
        ind_amp: float = -1,
        global_amp: float = -1,
        use_AM: bool = True,
        use_global_segment_durations: bool = False,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "SK1",
        scan_parameter_int: int = int(rf_common.CrossSK1ScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """This experiment creates one small-angle SK1 gates out of two pi/2
        SK1 gates and a phase gate.  The theory is checked in
        Q:\\CompactTrappedIonModule\\Writeups\\Gate compilation\\Small-Angle SK1 Gate.nb.
        We can scan the rotation angle and axis of the gate.

        Args:
            slots: The slots to which SK1 gates will be applied, in series
            theta: The rotation angle of the SK1 pulse
            phi: The rotation axis angle of the SK1 pulse
            ind_amp: The amplitude of the individual channel
            global_amp: The amplitude of the global channel
            use_AM: Determines whether we use square or AM SK1 pulses
            use_global_segment_durations: Determines whether the durations of the three
                SK1 segments are set from global values in physical_params or
                calculated from the known envelope shape
            wait_after: The flag (0 or 1) that determines whether a long wait is
                inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name

        phi_1 = phi - np.pi / 2
        phi_2 = phi + np.pi / 2

        if scan_parameter_int == int(self.CrossSK1ScanParameter.theta):

            theta_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            for i, th in enumerate(theta_values):
                if use_AM:

                    for s in slots:
                        self.add_SK1_am(
                            slots=[s],
                            theta=np.pi / 2,
                            phi=phi_1,
                            ind_amp=ind_amp,
                            global_amp=global_amp,
                            use_global_segment_durations=use_global_segment_durations,
                            wait_after=wait_after,
                            circuit_index=i,
                        )
                        self.add_phase(slots=[s], phase=th, circuit_index=i)
                        self.add_SK1_am(
                            slots=[s],
                            theta=np.pi / 2,
                            phi=phi_2,
                            ind_amp=ind_amp,
                            global_amp=global_amp,
                            use_global_segment_durations=use_global_segment_durations,
                            wait_after=wait_after,
                            circuit_index=i,
                        )

                else:

                    for s in slots:
                        self.add_SK1(
                            slots=[s],
                            theta=np.pi / 2,
                            phi=phi_1,
                            ind_amp=ind_amp,
                            global_amp=global_amp,
                            wait_after=wait_after,
                            circuit_index=i,
                        )
                        self.add_phase(slots=[s], phase=th, circuit_index=i)
                        self.add_SK1(
                            slots=[s],
                            theta=np.pi / 2,
                            phi=phi_2,
                            ind_amp=ind_amp,
                            global_amp=global_amp,
                            wait_after=wait_after,
                            circuit_index=i,
                        )

                self.gate_compiler.compile_circuit(i)

        else:

            scan_phi_1_min = min_value - np.pi / 2
            scan_phi_1_max = max_value - np.pi / 2
            scan_phi_2_min = min_value + np.pi / 2
            scan_phi_2_max = max_value + np.pi / 2

            if use_AM:
                if scan_parameter_int == int(self.CrossSK1ScanParameter.phi):
                    scan_parameter_int_to_use = int(self.SK1AMScanParameter.phi)
                elif scan_parameter_int == int(
                    self.CrossSK1ScanParameter.ind_amplitude
                ):
                    scan_parameter_int_to_use = int(
                        self.SK1AMScanParameter.ind_amplitude
                    )
                else:
                    scan_parameter_int_to_use = int(self.SK1AMScanParameter.static)

                for s in slots:
                    self.add_SK1_am(
                        slots=[s],
                        theta=np.pi / 2,
                        phi=phi_1,
                        ind_amp=ind_amp,
                        global_amp=global_amp,
                        use_global_segment_durations=use_global_segment_durations,
                        wait_after=wait_after,
                        scan_parameter_int=scan_parameter_int_to_use,
                        min_value=scan_phi_1_min,
                        max_value=scan_phi_1_max,
                        N_points=N_points,
                    )
                    self.add_phase(slots=[s], phase=theta)
                    self.add_SK1_am(
                        slots=[s],
                        theta=np.pi / 2,
                        phi=phi_2,
                        ind_amp=ind_amp,
                        global_amp=global_amp,
                        use_global_segment_durations=use_global_segment_durations,
                        wait_after=wait_after,
                        scan_parameter_int=scan_parameter_int_to_use,
                        min_value=scan_phi_2_min,
                        max_value=scan_phi_2_max,
                        N_points=N_points,
                    )

            else:
                if scan_parameter_int == int(self.CrossSK1ScanParameter.phi):
                    scan_parameter_int_to_use = int(self.SK1ScanParameter.phi)
                elif scan_parameter_int == int(
                    self.CrossSK1ScanParameter.ind_amplitude
                ):
                    scan_parameter_int_to_use = int(self.SK1ScanParameter.ind_amplitude)
                else:
                    scan_parameter_int_to_use = int(self.SK1ScanParameter.static)

                for s in slots:
                    self.add_SK1(
                        slots=[s],
                        theta=np.pi / 2,
                        phi=phi_1,
                        ind_amp=ind_amp,
                        global_amp=global_amp,
                        wait_after=wait_after,
                        scan_parameter_int=scan_parameter_int_to_use,
                        min_value=scan_phi_1_min,
                        max_value=scan_phi_1_max,
                        N_points=N_points,
                    )
                    self.add_phase(slots=[s], phase=theta)
                    self.add_SK1(
                        slots=[s],
                        theta=np.pi / 2,
                        phi=phi_2,
                        ind_amp=ind_amp,
                        global_amp=global_amp,
                        wait_after=wait_after,
                        scan_parameter_int=scan_parameter_int_to_use,
                        min_value=scan_phi_2_min,
                        max_value=scan_phi_2_max,
                        N_points=N_points,
                    )

    def XX_crosstalk(
        self,
        XX_slots: typing.List[int],
        echo_slots: typing.List[int],
        N_gates: int = 1,
        do_echo_sk1: bool = False,
        echo_phases: typing.List[float] = None,
        echo_slot_to_scan: int = -1,
        do_parity_sk1: bool = False,
        parity_sk1_phase: float = 0,
        swap_xx_phase: bool = False,
        gate_sign: float = +1,
        phi_ind1: float = 0,
        phi_ind2: float = 0,
        phi_global: float = 0,
        phi_motion: float = 0,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "XX",
        scan_parameter_int: int = int(rf_common.XXCrosstalkScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        **gate_param_modifications
    ):
        """
        This experiment consists of one fully-entangling XX gate.

        It exposes all parameters, both required and optional, of the XX gate prototype.

        Args:
            XX_slots: The two slots between which the gate is applied
            echo_slots: The slots to which the echoing SK1 pulses are applied
            N_gates: The number of consecutive XX gates to apply
            do_echo_sk1: Determines whether the echoing SK1 pulses are applied to the
                neighboring slots
            echo_phases: The list of phases around which we apply the echo SK1 pulses
            echo_slot_to_scan: The specific slot for which we will scan the phase of
                the echo SK1 pulse
            do_parity_sk1: Determines whether we apply analysis pulses to read out a
                parity fringe
            parity_sk1_phase: The phase of the analysis pulses, if their phase is not
                scanned
            swap_xx_phase: Determines whether we alternate XX phase signs on subsequent
                gates
            gate_sign: The sign (either +1 or -1) of the geometric phase required
            phi_ind1: The sign of the individual tone applied to slot 1
            phi_ind2: The sign of the individual tone applied to slot 2
            phi_global: The common phase of the global tone
            phi_motion: The initial phase difference between the blue and red sidebands
                on the global tone
            wait_after: The flag (0 or 1) that determines whether a long wait is
                inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name
        gate_name = "XX"

        if swap_xx_phase:
            flip_XX_phase = -1
        else:
            flip_XX_phase = 1

        assert len(echo_phases) == len(echo_slots)

        if scan_parameter_int == int(rf_common.XXCrosstalkScanParameter.Stark_shift):

            Stark_shift_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            for i, ss in enumerate(Stark_shift_values):

                self.physical_params.XX.Stark_shift = ss

                for _ in range(N_gates):
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        circuit_index=i,
                        **gate_param_modifications
                    )

                    if not do_echo_sk1:
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=flip_XX_phase * gate_sign,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            circuit_index=i,
                            **gate_param_modifications
                        )

                if do_echo_sk1:
                    for slot, phase in zip(echo_slots, echo_phases):
                        self.add_SK1_am(
                            slots=[slot], theta=np.pi, phi=phase, circuit_index=i
                        )

                    for _ in range(N_gates):
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=flip_XX_phase * gate_sign,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            circuit_index=i,
                            **gate_param_modifications
                        )

                if do_parity_sk1:
                    self.add_SK1_am(
                        slots=XX_slots,
                        phi=parity_sk1_phase,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )

                self.gate_compiler.compile_circuit(i)

        elif scan_parameter_int == int(rf_common.XXCrosstalkScanParameter.N_gates):

            N_gates_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            if min(N_gates_values) < 1:
                _LOGGER.error(
                    "The number of XX gates must be greater than or equal to 1."
                )
                raise Exception(
                    "The number of XX gates must be greater than or equal to 1."
                )

            for i, ng in enumerate(N_gates_values):

                for _ in range(int(max_value) - int(ng)):
                    self.add_twoQ_blank(
                        slots=XX_slots,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        circuit_index=i,
                    )

                # self.add_XX(
                #     slots=XX_slots,
                #     gate_sign=-1 * gate_sign,
                #     phi_ind1=phi_ind1,
                #     phi_ind2=phi_ind2,
                #     phi_global=phi_global,
                #     phi_motion=phi_motion,
                #     wait_after=wait_after,
                #     gate_name=gate_name,
                #     scan_parameter_int=int(self.XXScanParameter.static),
                #     circuit_index=i,
                #     **gate_param_modifications
                # )

                for j in range(int(ng)):
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        circuit_index=i,
                        **gate_param_modifications
                    )

                    if not do_echo_sk1:
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=flip_XX_phase * gate_sign,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            circuit_index=i,
                        )

                if do_echo_sk1:
                    for slot, phase in zip(echo_slots, echo_phases):
                        self.add_SK1_am(
                            slots=[slot], theta=np.pi, phi=phase, circuit_index=i
                        )

                    for j in range(int(ng)):
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=flip_XX_phase * gate_sign,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            circuit_index=i,
                            **gate_param_modifications
                        )

                for j in range(int(max_value) - int(ng)):
                    self.add_twoQ_blank(
                        slots=XX_slots,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        circuit_index=i,
                    )

                if do_parity_sk1:
                    self.add_SK1_am(
                        slots=XX_slots,
                        phi=parity_sk1_phase,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )

                self.gate_compiler.compile_circuit(i)

        # Here, we scan the phases of the various SK1 pulses
        elif (
            scan_parameter_int
            == int(rf_common.XXCrosstalkScanParameter.echo_phase_single_absolute)
            or scan_parameter_int
            == int(rf_common.XXCrosstalkScanParameter.echo_phase_all_relative)
            or scan_parameter_int
            == int(rf_common.XXCrosstalkScanParameter.analysis_phase)
        ):

            for i in range(N_gates):
                self.add_XX(
                    slots=XX_slots,
                    gate_sign=gate_sign,
                    phi_ind1=phi_ind1,
                    phi_ind2=phi_ind2,
                    phi_global=phi_global,
                    phi_motion=phi_motion,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    **gate_param_modifications
                )

                if not do_echo_sk1:
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=flip_XX_phase * gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        **gate_param_modifications
                    )

            if do_echo_sk1:
                if scan_parameter_int == int(
                    self.XXCrosstalkScanParameter.echo_phase_single_absolute
                ):
                    # In this case, scan the phase of only the SK1 pulse specified by
                    # echo_slot_to_scan
                    for slot, phase in zip(echo_slots, echo_phases):
                        if slot == echo_slot_to_scan:
                            self.add_SK1_am(
                                slots=[slot],
                                theta=np.pi,
                                phi=0,
                                scan_parameter_int=int(self.SK1AMScanParameter.phi),
                                min_value=min_value,
                                max_value=max_value,
                                N_points=N_points,
                            )
                        else:
                            self.add_SK1_am(slots=[slot], theta=np.pi, phi=phase)
                elif scan_parameter_int == int(
                    self.XXCrosstalkScanParameter.echo_phase_all_relative
                ):
                    # In this case, scan the phases of all SK1 echo pulses,
                    # each offset by their specific echo phase
                    for slot, phase in zip(echo_slots, echo_phases):
                        self.add_SK1_am(
                            slots=[slot],
                            theta=np.pi,
                            phi=0,
                            scan_parameter_int=int(self.SK1AMScanParameter.phi),
                            min_value=min_value + phase,
                            max_value=max_value + phase,
                            N_points=N_points,
                        )
                else:
                    # In this case, we are scanning the phase of the analysis pulses,
                    # so enter static echo pulses
                    for slot, phase in zip(echo_slots, echo_phases):
                        self.add_SK1_am(slots=[slot], theta=np.pi, phi=phase)

                for i in range(N_gates):
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=flip_XX_phase * gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        **gate_param_modifications
                    )

            if do_parity_sk1:
                if scan_parameter_int == int(
                    rf_common.XXCrosstalkScanParameter.analysis_phase
                ):
                    self.add_SK1_am(
                        slots=XX_slots,
                        phi=0,
                        gate_name="Analysis SK1",
                        scan_parameter_int=int(self.SK1ScanParameter.phi),
                        min_value=min_value,
                        max_value=max_value,
                        N_points=N_points,
                    )
                else:
                    self.add_SK1_am(
                        slots=XX_slots, phi=parity_sk1_phase, gate_name="Analysis SK1"
                    )

            self.gate_compiler.compile_circuit(0)

        # If we're not scanning any of the parameters that need to be scanned explicitly
        # at this level, we pass the scan_parameter_int in to the XX gates.
        # This works because the members of the XXScanParameter enum, which is
        # what is the gates use, are a subset of the members of the
        # XXCrosstalkScanParameter members. This is dangerous, but it works.
        else:

            for i in range(N_gates):
                self.add_XX(
                    slots=XX_slots,
                    gate_sign=gate_sign,
                    phi_ind1=phi_ind1,
                    phi_ind2=phi_ind2,
                    phi_global=phi_global,
                    phi_motion=phi_motion,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=scan_parameter_int,
                    min_value=min_value,
                    max_value=max_value,
                    N_points=N_points,
                    **gate_param_modifications
                )

                if not do_echo_sk1:
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=flip_XX_phase * gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=scan_parameter_int,
                        min_value=min_value,
                        max_value=max_value,
                        N_points=N_points,
                        **gate_param_modifications
                    )

            if do_echo_sk1:
                for slot, phase in zip(echo_slots, echo_phases):
                    self.add_SK1_am(slots=[slot], theta=np.pi, phi=phase)

                for i in range(N_gates):
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=flip_XX_phase * gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=scan_parameter_int,
                        min_value=min_value,
                        max_value=max_value,
                        N_points=N_points,
                        **gate_param_modifications
                    )

            if do_parity_sk1:
                self.add_SK1_am(
                    slots=XX_slots, phi=parity_sk1_phase, gate_name="Analysis SK1"
                )

            self.gate_compiler.compile_circuit(0)

    def XX_Echo(
            self,
            XX_slots: typing.List[int],
            analyze_slots: typing.List[int],
            xstart_slots: typing.List[int],
            N_echo_blocks: int=1,
            N_gates: int = 1,
            do_parity_sk1: bool = False,
            parity_sk1_phase: float = 0,
            do_xstart:bool = False,
            phi_ind1: float = 0,
            phi_ind2: float = 0,
            phi_global: float = 0,
            phi_motion: float = 0,
            wait_after: int = 0,
            wait_after_time: float = 0,
            exp_name: str = "XX",
            scan_parameter_int: int = int(rf_common.XXCrosstalkScanParameter.static),
            min_value: float = 0,
            max_value: float = 0,
            N_points: int = 0,
            **gate_param_modifications
        ):
            """
            This experiment consists of one fully-entangling XX gate.

            It exposes all parameters, both required and optional, of the XX gate prototype.

            Args:
                XX_slots: The two slots between which the gate is applied
                echo_slots: The slots to which the echoing SK1 pulses are applied
                N_gates: The number of consecutive XX gates to apply
                do_echo_sk1: Determines whether the echoing SK1 pulses are applied to the
                    neighboring slots
                echo_phases: The list of phases around which we apply the echo SK1 pulses
                echo_slot_to_scan: The specific slot for which we will scan the phase of
                    the echo SK1 pulse
                do_parity_sk1: Determines whether we apply analysis pulses to read out a
                    parity fringe
                parity_sk1_phase: The phase of the analysis pulses, if their phase is not
                    scanned
                swap_xx_phase: Determines whether we alternate XX phase signs on subsequent
                    gates
                gate_sign: The sign (either +1 or -1) of the geometric phase required
                phi_ind1: The sign of the individual tone applied to slot 1
                phi_ind2: The sign of the individual tone applied to slot 2
                phi_global: The common phase of the global tone
                phi_motion: The initial phase difference between the blue and red sidebands
                    on the global tone
                wait_after: The flag (0 or 1) that determines whether a long wait is
                    inserted
                wait_after_time: The long delay to be inserted after select pulses
                exp_name: The name of this experiment
                scan_parameter_int: The specific parameter being swept
                min_value: The minimum value of the parameter being swept
                max_value: The maximum value of the parameter being swept
                N_points: The number of points of the parameter being swept
            """

            self.wait_after_time = wait_after_time
            self.name = exp_name
            gate_name = "XX"

            # if do_xstart:
            #     self.add_SK1_am(
            #                         slots=xstart_slots,
            #                         phi=parity_sk1_phase,
            #                         gate_name="Analysis SK1",
            #                         circuit_index=i,
            #                     )

            if scan_parameter_int == int(rf_common.XXCrosstalkScanParameter.Stark_shift):

                Stark_shift_values = (
                    np.linspace(min_value, max_value, N_points) if N_points != 0 else None
                )

                for i, ss in enumerate(Stark_shift_values):

                    # self.physical_params.XX.Stark_shift = ss
                    if do_xstart:
                        self.add_SK1_am(
                            slots=xstart_slots,
                            phi=0,
                            gate_name="Analysis SK1",
                            circuit_index=i,
                        )
                    for _ in range(N_echo_blocks):
                        for _ in range(N_gates):
                            self.add_XX(
                                slots=XX_slots,
                                gate_sign=+1,
                                phi_ind1=phi_ind1,
                                phi_ind2=phi_ind2,
                                phi_global=phi_global,
                                phi_motion=phi_motion,
                                wait_after=wait_after,
                                gate_name=gate_name,
                                scan_parameter_int=int(self.XXScanParameter.static),
                                stark_shift=ss,
                                circuit_index=i,
                                **gate_param_modifications
                            )

                        for _ in range(N_gates):
                            self.add_XX(
                                slots=XX_slots,
                                gate_sign=-1,
                                phi_ind1=phi_ind1,
                                phi_ind2=phi_ind2,
                                phi_global=phi_global,
                                phi_motion=phi_motion,
                                wait_after=wait_after,
                                gate_name=gate_name,
                                scan_parameter_int=int(self.XXScanParameter.static),
                                stark_shift=ss,
                                circuit_index=i,
                                **gate_param_modifications
                            )

                    if do_parity_sk1:
                        self.add_SK1_am(
                            slots=analyze_slots,
                            phi=parity_sk1_phase,
                            gate_name="Analysis SK1",
                            circuit_index=i,
                        )

                    self.gate_compiler.compile_circuit(i)
            elif scan_parameter_int == int(rf_common.XXCrosstalkScanParameter.Stark_shift_diff):

                Stark_shift_values = (
                    np.linspace(min_value, max_value, N_points) if N_points != 0 else None
                )

                for i, ss in enumerate(Stark_shift_values):

                    # self.physical_params.XX.Stark_shift = ss
                    if do_xstart:
                        self.add_SK1_am(
                            slots=xstart_slots,
                            phi=0,
                            gate_name="Analysis SK1",
                            circuit_index=i,
                        )
                    for _ in range(N_echo_blocks):
                        for _ in range(N_gates):
                            self.add_XX(
                                slots=XX_slots,
                                gate_sign=+1,
                                phi_ind1=phi_ind1,
                                phi_ind2=phi_ind2,
                                phi_global=phi_global,
                                phi_motion=phi_motion,
                                wait_after=wait_after,
                                gate_name=gate_name,
                                scan_parameter_int=int(self.XXScanParameter.static),
                                stark_shift_differential=ss,
                                circuit_index=i,
                                **gate_param_modifications
                            )

                        for _ in range(N_gates):
                            self.add_XX(
                                slots=XX_slots,
                                gate_sign=-1,
                                phi_ind1=phi_ind1,
                                phi_ind2=phi_ind2,
                                phi_global=phi_global,
                                phi_motion=phi_motion,
                                wait_after=wait_after,
                                gate_name=gate_name,
                                scan_parameter_int=int(self.XXScanParameter.static),
                                stark_shift_differential=ss,
                                circuit_index=i,
                                **gate_param_modifications
                            )

                    if do_parity_sk1:
                        self.add_SK1_am(
                            slots=analyze_slots,
                            phi=parity_sk1_phase,
                            gate_name="Analysis SK1",
                            circuit_index=i,
                        )

                    self.gate_compiler.compile_circuit(i)

            elif scan_parameter_int == int(rf_common.XXCrosstalkScanParameter.N_gates):

                N_gates_values = (
                    np.linspace(min_value, max_value, N_points) if N_points != 0 else None
                )

                if min(N_gates_values) < 1:
                    _LOGGER.error(
                        "The number of XX gates must be greater than or equal to 1."
                    )
                    raise Exception(
                        "The number of XX gates must be greater than or equal to 1."
                    )


                for i, ng in enumerate(N_gates_values):
                    ng = int(ng)
                    if do_xstart:
                        self.add_SK1_am(
                            slots=xstart_slots,
                            phi=0,
                            gate_name="Analysis SK1",
                            circuit_index=i,
                        )
                    for _ in range(N_echo_blocks):
                            for _ in range(int(max_value) - int(ng)):
                                self.add_twoQ_blank(
                                    slots=XX_slots,
                                    wait_after=wait_after,
                                    gate_name=gate_name,
                                    circuit_index=i,
                                )

                            # self.add_XX(
                            #     slots=XX_slots,
                            #     gate_sign=-1 * gate_sign,
                            #     phi_ind1=phi_ind1,
                            #     phi_ind2=phi_ind2,
                            #     phi_global=phi_global,
                            #     phi_motion=phi_motion,
                            #     wait_after=wait_after,
                            #     gate_name=gate_name,
                            #     scan_parameter_int=int(self.XXScanParameter.static),
                            #     circuit_index=i,
                            #     **gate_param_modifications
                            # )

                            for _ in range(ng):
                                self.add_XX(
                                    slots=XX_slots,
                                    gate_sign=+1,
                                    phi_ind1=phi_ind1,
                                    phi_ind2=phi_ind2,
                                    phi_global=phi_global,
                                    phi_motion=phi_motion,
                                    wait_after=wait_after,
                                    gate_name=gate_name,
                                    scan_parameter_int=int(self.XXScanParameter.static),
                                    circuit_index=i,
                                    **gate_param_modifications
                                )

                            for _ in range(ng):
                                self.add_XX(
                                    slots=XX_slots,
                                    gate_sign=-1,
                                    phi_ind1=phi_ind1,
                                    phi_ind2=phi_ind2,
                                    phi_global=phi_global,
                                    phi_motion=phi_motion,
                                    wait_after=wait_after,
                                    gate_name=gate_name,
                                    scan_parameter_int=int(self.XXScanParameter.static),
                                    circuit_index=i,
                                    **gate_param_modifications
                                )


                            for j in range(int(max_value) - int(ng)):
                                self.add_twoQ_blank(
                                    slots=XX_slots,
                                    wait_after=wait_after,
                                    gate_name=gate_name,
                                    circuit_index=i,
                                )

                    if do_parity_sk1:
                                self.add_SK1_am(
                                    slots=analyze_slots,
                                    phi=parity_sk1_phase,
                                    gate_name="Analysis SK1",
                                    circuit_index=i,
                                )

                    self.gate_compiler.compile_circuit(i)

            # Here, we scan the phases of the SK1 analysis pulse
            elif (
                scan_parameter_int
                == int(rf_common.XXCrosstalkScanParameter.analysis_phase)
            ):
                if do_xstart:
                    self.add_SK1_am(
                        slots=xstart_slots,
                        phi=0,
                        gate_name="Starting SK1",
                    )
                for _ in range(N_echo_blocks):
                    for _ in range(N_gates):
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=+1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            **gate_param_modifications
                        )

                    for _ in range(N_gates):
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=-1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            **gate_param_modifications
                        )

                if do_parity_sk1:
                    self.add_SK1_am(
                        slots=analyze_slots,
                        phi=0,
                        gate_name="Analysis SK1",
                        scan_parameter_int=int(self.SK1ScanParameter.phi),
                        min_value=min_value,
                        max_value=max_value,
                        N_points=N_points,
                    )

                self.gate_compiler.compile_circuit(0)

            # If we're not scanning any of the parameters that need to be scanned explicitly
            # at this level, we pass the scan_parameter_int in to the XX gates.
            # This works because the members of the XXScanParameter enum, which is
            # what is the gates use, are a subset of the members of the
            # XXCrosstalkScanParameter members. This is dangerous, but it works.
            else:
                if do_xstart:
                    self.add_SK1_am(
                        slots=xstart_slots,
                        phi=0,
                        gate_name="Analysis SK1",
                    )
                for _ in range(N_echo_blocks):
                    for _ in range(N_gates):
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=+1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            min_value=min_value,
                            max_value=max_value,
                            N_points=N_points,
                            scan_parameter_int=scan_parameter_int,
                            **gate_param_modifications
                        )

                    for _ in range(N_gates):
                        # TODO: Scanning here doesn't work b/c parameters not set right.
                        # TODO: needs to have min_value, max_value, and N_points set, remove static, and remove gate_compiler call
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=-1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            min_value=min_value,
                            max_value=max_value,
                            N_points=N_points,
                            scan_parameter_int=scan_parameter_int,
                            **gate_param_modifications
                        )

                if do_parity_sk1:
                        self.add_SK1_am(
                            slots=analyze_slots,
                            phi=parity_sk1_phase,
                            gate_name="Analysis SK1",
                        )

    def XX_Echo_Fidelity(
            self,
            XX_slots: typing.List[int],
            N_gates: int = 1,
            do_parity_sk1: bool = False,
            parity_sk1_phase: float = 0,
            phi_ind1: float = 0,
            phi_ind2: float = 0,
            phi_global: float = 0,
            phi_motion: float = 0,
            wait_after: int = 0,
            wait_after_time: float = 0,
            exp_name: str = "XX",
            scan_parameter_int: int = int(XXScanParameter.static),
            min_value: float = 0,
            max_value: float = 0,
            N_points: int = 0,
            **gate_param_modifications
        ):
            """
            This experiment consists of one fully-entangling XX gate.

            It exposes all parameters, both required and optional, of the XX gate prototype.

            Args:
                XX_slots: The two slots between which the gate is applied
                echo_slots: The slots to which the echoing SK1 pulses are applied
                N_gates: The number of consecutive XX gates to apply
                do_echo_sk1: Determines whether the echoing SK1 pulses are applied to the
                    neighboring slots
                echo_phases: The list of phases around which we apply the echo SK1 pulses
                echo_slot_to_scan: The specific slot for which we will scan the phase of
                    the echo SK1 pulse
                do_parity_sk1: Determines whether we apply analysis pulses to read out a
                    parity fringe
                parity_sk1_phase: The phase of the analysis pulses, if their phase is not
                    scanned
                swap_xx_phase: Determines whether we alternate XX phase signs on subsequent
                    gates
                gate_sign: The sign (either +1 or -1) of the geometric phase required
                phi_ind1: The sign of the individual tone applied to slot 1
                phi_ind2: The sign of the individual tone applied to slot 2
                phi_global: The common phase of the global tone
                phi_motion: The initial phase difference between the blue and red sidebands
                    on the global tone
                wait_after: The flag (0 or 1) that determines whether a long wait is
                    inserted
                wait_after_time: The long delay to be inserted after select pulses
                exp_name: The name of this experiment
                scan_parameter_int: The specific parameter being swept
                min_value: The minimum value of the parameter being swept
                max_value: The maximum value of the parameter being swept
                N_points: The number of points of the parameter being swept
            """

            self.wait_after_time = wait_after_time
            self.name = exp_name
            gate_name = "XX"

            # if do_xstart:
            #     self.add_SK1_am(
            #                         slots=xstart_slots,
            #                         phi=parity_sk1_phase,
            #                         gate_name="Analysis SK1",
            #                         circuit_index=i,
            #                     )

            if scan_parameter_int == int(rf_common.XXCrosstalkScanParameter.Stark_shift):

                Stark_shift_values = (
                    np.linspace(min_value, max_value, N_points) if N_points != 0 else None
                )

                for i, ss in enumerate(Stark_shift_values):

                    for _ in range(int(N_gates/2)):
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=+1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            stark_shift=ss,
                            circuit_index=i,
                            **gate_param_modifications
                        )
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=-1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            stark_shift=ss,
                            circuit_index=i,
                            **gate_param_modifications
                        )
                    if int(N_gates/2)*2 < N_gates:
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=+1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            stark_shift=ss,
                            circuit_index=i,
                            **gate_param_modifications
                                )

                    if do_parity_sk1:
                        self.add_SK1_am(
                            slots=analyze_slots,
                            phi=parity_sk1_phase,
                            gate_name="Analysis SK1",
                            circuit_index=i,
                        )

                    self.gate_compiler.compile_circuit(i)


            elif scan_parameter_int == int(rf_common.XXCrosstalkScanParameter.N_gates):

                N_gates_values = (
                    np.linspace(min_value, max_value, N_points) if N_points != 0 else None
                )

                if min(N_gates_values) < 1:
                    _LOGGER.error(
                        "The number of XX gates must be greater than or equal to 1."
                    )
                    raise Exception(
                        "The number of XX gates must be greater than or equal to 1."
                    )


                for i, ng in enumerate(N_gates_values):
                    ng = int(ng)

                    for _ in range(int(ng/2)):
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=+1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            circuit_index=i,
                            **gate_param_modifications
                        )

                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=-1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            circuit_index=i,
                            **gate_param_modifications
                        )
                    # is # of gates odd?
                    if int(ng/2)*2 < ng:
                        self.add_XX(
                            slots=XX_slots,
                            gate_sign=+1,
                            phi_ind1=phi_ind1,
                            phi_ind2=phi_ind2,
                            phi_global=phi_global,
                            phi_motion=phi_motion,
                            wait_after=wait_after,
                            gate_name=gate_name,
                            scan_parameter_int=int(self.XXScanParameter.static),
                            circuit_index=i,
                            **gate_param_modifications
                                )
                    for j in range(int(max_value) - int(ng)):
                                self.add_twoQ_blank(
                                    slots=XX_slots,
                                    wait_after=wait_after,
                                    gate_name=gate_name,
                                    circuit_index=i,
                                )
                    if do_parity_sk1:
                                self.add_SK1_am(
                                    slots=XX_slots,
                                    phi=parity_sk1_phase,
                                    gate_name="Analysis SK1",
                                    circuit_index=i,
                                )

                    self.gate_compiler.compile_circuit(i)

            # Here, we scan the phases of the SK1 analysis pulse
            elif (
                scan_parameter_int
                == int(rf_common.XXCrosstalkScanParameter.analysis_phase)
            ):
                for _ in range(int(N_gates/2)):
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=+1,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        **gate_param_modifications
                    )
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=-1,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        **gate_param_modifications
                    )
                if int(N_gates/2)*2 < N_gates:
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=+1,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        **gate_param_modifications
                    )

                if do_parity_sk1:
                    self.add_SK1_am(
                        slots=XX_slots,
                        phi=0,
                        gate_name="Analysis SK1",
                        scan_parameter_int=int(self.SK1ScanParameter.phi),
                        min_value=min_value,
                        max_value=max_value,
                        N_points=N_points,
                    )

                self.gate_compiler.compile_circuit(0)

            # If we're not scanning any of the parameters that need to be scanned explicitly
            # at this level, we pass the scan_parameter_int in to the XX gates.
            # This works because the members of the XXScanParameter enum, which is
            # what is the gates use, are a subset of the members of the
            # XXCrosstalkScanParameter members. This is dangerous, but it works.
            else:
                for _ in range(int(N_gates/2)):

                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=+1,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        **gate_param_modifications
                    )
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=-1,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        **gate_param_modifications
                    )
                # is number of gates odd?
                if int(N_gates/2)*2 < N_gates:
                    self.add_XX(
                        slots=XX_slots,
                        gate_sign=+1,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=scan_parameter_int,
                        min_value=min_value,
                        max_value=max_value,
                        N_points=N_points,
                        **gate_param_modifications
                    )

                if do_parity_sk1:
                        self.add_SK1_am(
                            slots=XX_slots,
                            phi=parity_sk1_phase,
                            gate_name="Analysis SK1",
                        )

    @eur_dec.profiler
    def XX_exp(
        self,
        slots: rf_common.SlotPair,
        N_gates: int = 1,
        gate_sign: float = +1,
        phi_ind1: float = 0,
        phi_ind2: float = 0,
        phi_global: float = 0,
        phi_motion: float = 0,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "XX",
        scan_parameter_int: int = int(XXScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        **gate_param_modifications
    ):
        """This experiment consists of one fully XX gate.

        It exposes all parameters, both required and optional, of the XX gate prototype.

        Args:
            slots: The two slots between which the gate is applied
            N_gates: The number of consecutive XX gates to apply
            gate_sign: The sign (either +1 or -1) of the geometric phase required
            phi_ind1: The sign of the individual tone applied to slot 1
            phi_ind2: The sign of the individual tone applied to slot 2
            phi_global: The common phase of the global tone
            phi_motion: The initial phase difference between the blue and red
                sidebands on the global tone
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept

        Kwargs:
            modifications to the gate parameters. See :meth:`add_XX` for more info.
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name
        gate_name = "XX"

        if scan_parameter_int == int(self.XXScanParameter.Stark_shift):

            Stark_shift_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            for i, ss in enumerate(Stark_shift_values):
                for _ in range(N_gates):
                    self.add_XX(
                        slots=slots,
                        gate_sign=gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        circuit_index=i,
                        stark_shift=ss,
                        **gate_param_modifications
                    )

                self.gate_compiler.compile_circuit(i)

        elif scan_parameter_int == int(self.XXScanParameter.N_gates):

            N_gates_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            for i, ng in enumerate(N_gates_values):

                for _ in range(int(ng)):
                    self.add_XX(
                        slots=slots,
                        gate_sign=gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        circuit_index=i,
                        **gate_param_modifications
                    )

                for _ in range(int(max_value - ng)):
                    self.add_twoQ_blank(
                        slots=slots,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        circuit_index=i,
                    )

                self.gate_compiler.compile_circuit(i)

        else:
            for i in range(N_gates):
                self.add_XX(
                    slots=slots,
                    gate_sign=gate_sign,
                    phi_ind1=phi_ind1,
                    phi_ind2=phi_ind2,
                    phi_global=phi_global,
                    phi_motion=phi_motion,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=scan_parameter_int,
                    min_value=min_value,
                    max_value=max_value,
                    N_points=N_points,
                    **gate_param_modifications
                )

    def linescan(
        self,
        slots: typing.List[int],
        detuning: float,
        sideband_order: int,
        duration: float,
        scan_range: float,
        N_points: int,
        ind_amp: float = -1,
        global_amp: float = -1,
        exp_name: str = r"Linescan",
    ):

        self.name = exp_name
        self.wait_after_time = 0

        min_value = sideband_order * (detuning - scan_range / 2)
        max_value = sideband_order * (detuning + scan_range / 2)

        self.add_rabi(
            slots=slots,
            detuning=detuning,
            sideband_order=sideband_order,
            duration=duration,
            ind_amp=ind_amp,
            global_amp=global_amp,
            scan_parameter_int=int(self.RabiScanParameter.frequency),
            min_value=min_value,
            max_value=max_value,
            N_points=N_points,
        )

    @eur_dec.profiler
    def XX_parity_scan(
        self,
        slots: typing.List[int],
        N_gates: int = 1,
        gate_sign: float = +1,
        use_SK1_am: bool = False,
        sweep_relative_phase: int = 0,
        common_phase: float = 0,
        relative_phase: float = 0,
        exp_name: str = "XX parity scan",
        phi_min: float = 0,
        phi_max: float = 2 * np.pi,
        N_points: int = 21,
        **gate_param_modifications
    ):
        """This experiment consists of one fully XX gate.  It exposes all parameters,
        both required and optional, of the XX gate prototype.

        Args:
            slots: The two slots between which the gate is applied

        Args:
            N_gates: The number of consecutive XX gates to apply
            gate_sign: The sign (either +1 or -1) of the geometric phase required
            use_SK1_am: Determines whether amplitude-modulated or square
                SK1 analysis pulses are used
            sweep_relative_phase: The flag (0 or 1) that determines whether this
                experiment sweeps the common phase of the SK1 analysis pulses
                relative to the XX pulse or the relative phase phase between
                the two SK1 pulses
            common_phase: The common phase of the SK1 analysis pulses,
                used when sweeping the relative phase
            relative_phase: The relative phase phase of the SK1 analysis pulses,
                used when sweeping the common phase
            exp_name: The name of this experiment
            phi_min: The initial point of the phase sweep
            phi_max: The end point of the phase sweep
            N_points: The number of data points to acquire
        """

        self.wait_after_time = 0
        self.name = exp_name

        if sweep_relative_phase == 0:
            min_phase_1 = -relative_phase / 2 + phi_min
            max_phase_1 = -relative_phase / 2 + phi_max
            min_phase_2 = relative_phase / 2 + phi_min
            max_phase_2 = relative_phase / 2 + phi_max
        else:
            min_phase_1 = common_phase - phi_min / 2
            max_phase_1 = common_phase - phi_max / 2
            min_phase_2 = common_phase + phi_min / 2
            max_phase_2 = common_phase + phi_max / 2

        for _ in range(N_gates):
            self.add_XX(
                slots=slots,
                gate_sign=gate_sign,
                gate_name="XX",
                **gate_param_modifications
            )

        if use_SK1_am:
            self.add_SK1_am(
                slots=[slots[0]],
                phi=0,
                gate_name="Analysis SK1",
                scan_parameter_int=int(self.SK1ScanParameter.phi),
                min_value=min_phase_1,
                max_value=max_phase_1,
                N_points=N_points,
            )
            self.add_SK1_am(
                slots=[slots[1]],
                phi=0,
                gate_name="Analysis SK1",
                scan_parameter_int=int(self.SK1ScanParameter.phi),
                min_value=min_phase_2,
                max_value=max_phase_2,
                N_points=N_points,
            )
        else:
            self.add_SK1(
                slots=[slots[0]],
                phi=0,
                theta=np.pi/2,
                gate_name="Analysis SK1",
                scan_parameter_int=int(self.SK1ScanParameter.phi),
                min_value=min_phase_1,
                max_value=max_phase_1,
                N_points=N_points,
            )
            self.add_SK1(
                slots=[slots[1]],
                phi=0,
                theta=np.pi/2,
                gate_name="Analysis SK1",
                scan_parameter_int=int(self.SK1ScanParameter.phi),
                min_value=min_phase_2,
                max_value=max_phase_2,
                N_points=N_points,
            )

    @eur_dec.profiler
    def XX_with_analysis(
        self,
        slots: rf_common.SlotPair,
        SK1_phase: float = 0,
        SK1_diff_phase: float = 0,
        use_SK1_am: bool = False,
        N_gates: int = 1,
        gate_sign: float = +1,
        phi_ind1: float = 0,
        phi_ind2: float = 0,
        phi_global: float = 0,
        phi_motion: float = 0,
        wait_after: int = 0,
        wait_after_time: float = 0,
        exp_name: str = "XX with analysis",
        scan_parameter_int: int = int(XXScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
        **gate_param_modifications
    ):
        """This experiment consists of one fully XX gate.

        It exposes all parameters, both required and optional, of the XX gate prototype.

        Args:
            SK1_phase: The phase of the two SK1 analysis pulses
            SK1_diff_phase: The differential phase of the two SK1
                analysis pulses
            use_SK1_am: Determines whether amplitude-modulated or square
                SK1 analysis pulses are used
            slots: The two slots between which the gate is applied
            N_gates: The number of consecutive XX gates to apply
            gate_sign: The sign (either +1 or -1) of the geometric phase required
            phi_ind1: The sign of the individual tone applied to slot 1
            phi_ind2: The sign of the individual tone applied to slot 2
            phi_global: The common phase of the global tone
            phi_motion: The initial phase difference between the blue and
                red sidebands on the global tone
            wait_after: The flag (0 or 1) that determines whether a long wait
                is inserted
            wait_after_time: The long delay to be inserted after select pulses
            exp_name: The name of this experiment
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept

        Kwargs:
            modifications to the gate parameters. See :meth:`add_XX` for more info.
        """

        self.wait_after_time = wait_after_time
        self.name = exp_name
        gate_name = "XX with analysis"

        if scan_parameter_int == int(self.XXScanParameter.Stark_shift):

            Stark_shift_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            for i, ss in enumerate(Stark_shift_values):
                for _ in range(N_gates):
                    self.add_XX(
                        slots=slots,
                        gate_sign=gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        circuit_index=i,
                        stark_shift=ss,
                        **gate_param_modifications
                    )

                if use_SK1_am:
                    self.add_SK1_am(
                        slots=[slots[0]],
                        phi=SK1_phase + SK1_diff_phase / 2,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )
                    self.add_SK1_am(
                        slots=[slots[1]],
                        phi=SK1_phase - SK1_diff_phase / 2,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )
                else:
                    self.add_SK1(
                        slots=[slots[0]],
                        theta=np.pi / 2,
                        phi=SK1_phase + SK1_diff_phase / 2,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )
                    self.add_SK1(
                        slots=[slots[1]],
                        theta=np.pi / 2,
                        phi=SK1_phase - SK1_diff_phase / 2,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )

                self.gate_compiler.compile_circuit(i)

        elif scan_parameter_int == int(self.XXScanParameter.N_gates):

            N_gates_values = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

            for i, ng in enumerate(N_gates_values):

                for _ in range(int(ng)):
                    self.add_XX(
                        slots=slots,
                        gate_sign=gate_sign,
                        phi_ind1=phi_ind1,
                        phi_ind2=phi_ind2,
                        phi_global=phi_global,
                        phi_motion=phi_motion,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        scan_parameter_int=int(self.XXScanParameter.static),
                        circuit_index=i,
                        **gate_param_modifications
                    )

                for _ in range(int(max_value - ng)):
                    self.add_twoQ_blank(
                        slots=slots,
                        wait_after=wait_after,
                        gate_name=gate_name,
                        circuit_index=i,
                    )

                if use_SK1_am:
                    self.add_SK1_am(
                        slots=[slots[0]],
                        phi=SK1_phase + SK1_diff_phase / 2,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )
                    self.add_SK1_am(
                        slots=[slots[1]],
                        phi=SK1_phase - SK1_diff_phase / 2,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )
                else:
                    self.add_SK1(
                        slots=[slots[0]],
                        theta=np.pi / 2,
                        phi=SK1_phase + SK1_diff_phase / 2,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )
                    self.add_SK1(
                        slots=[slots[1]],
                        theta=np.pi / 2,
                        phi=SK1_phase - SK1_diff_phase / 2,
                        gate_name="Analysis SK1",
                        circuit_index=i,
                    )

                self.gate_compiler.compile_circuit(i)

        else:

            for i in range(N_gates):
                self.add_XX(
                    slots=slots,
                    gate_sign=gate_sign,
                    phi_ind1=phi_ind1,
                    phi_ind2=phi_ind2,
                    phi_global=phi_global,
                    phi_motion=phi_motion,
                    wait_after=wait_after,
                    gate_name=gate_name,
                    scan_parameter_int=scan_parameter_int,
                    min_value=min_value,
                    max_value=max_value,
                    N_points=N_points,
                    **gate_param_modifications
                )

            if use_SK1_am:
                self.add_SK1_am(
                    slots=[slots[0]],
                    phi=SK1_phase + SK1_diff_phase / 2,
                    gate_name="Analysis SK1",
                )
                self.add_SK1_am(
                    slots=[slots[1]],
                    phi=SK1_phase - SK1_diff_phase / 2,
                    gate_name="Analysis SK1",
                )
            else:
                self.add_SK1(
                    slots=[slots[0]],
                    theta=np.pi / 2,
                    phi=SK1_phase + SK1_diff_phase / 2,
                    gate_name="Analysis SK1",
                )
                self.add_SK1(
                    slots=[slots[1]],
                    theta=np.pi / 2,
                    phi=SK1_phase - SK1_diff_phase / 2,
                    gate_name="Analysis SK1",
                )

    @eur_dec.profiler
    def stabilizer_readout(
        self,
        prep_state: int,
        post_prepare_phases: typing.List[float] = 0,
        post_XX_phase: float = 0,
        correction_length: float = 0,
        correction_phase: float = 0,
        gate_to_sweep: int = 0,
        exp_name: str = "Stablizer readout",
        use_SK1_AM: bool = True,
        print_circuit: bool = False,
        print_gate_list: bool = False,
        scan_parameter_int: int = int(rf_common.StabReadoutScanParameter.static),
        min_value: float = 0,
        max_value: float = 0,
        N_points: int = 0,
    ):
        """This experiment implements one stabilizer readout circuit, which involves
        preparing the selected qubits in X, mapping the stabilizer result to an ancilla
        qubit, and reading out the qubits in X.

        Args:
            prep_state: The default value of the prep_state int, which determines
                the qubits' initial state in X
            post_prepare_phases: The global phase shift to apply after preparing
                the qubits in the X basis
            post_XX_phase: The global phase shift to apply after performing the XX gates
            exp_name: The name of this experiment
            use_SK1_AM: Determines whether square or amplitude-modulated
                SK1 pulses are used
            print_circuit: Determines whether the circuit compilation code
                prints the circuit
            print_gate_list: Determines whether the circuit compilation code
                prints the list of generated gates
            scan_parameter_int: The specific parameter being swept
            min_value: The minimum value of the parameter being swept
            max_value: The maximum value of the parameter being swept
            N_points: The number of points of the parameter being swept
        """

        self.name = exp_name
        str_to_return = ""

        N_qubits_present = self.physical_params.N_ions
        N_qubits_used = 7
        qubit_offset = 4
        qubits_to_exclude = [3]

        if scan_parameter_int == int(self.StabReadoutScanParameter.initial_state):
            scan_array = list(range(int(min_value), int(max_value) + 1))
        else:
            scan_array = (
                np.linspace(min_value, max_value, N_points) if N_points != 0 else None
            )

        correction_scan_parameter_int = self.RabiScanParameter.static
        if scan_parameter_int == int(self.StabReadoutScanParameter.correction_duration):
            correction_scan_parameter_int = int(self.RabiScanParameter.duration)
            scan_array_to_scan = [0]
        elif scan_parameter_int == int(self.StabReadoutScanParameter.correction_phase):
            correction_scan_parameter_int = int(self.RabiScanParameter.phase)
            scan_array_to_scan = [0]
        else:
            scan_array_to_scan = scan_array

        from euriqabackend.devices.keysight_awg import circuit_prototypes as cp

        circuit_prototypes = cp.CircuitPrototypes(
            RFCompiler=self,
            physical_params=self.physical_params,
            N_qubits_present=N_qubits_present,
            N_qubits_used=N_qubits_used,
            qubit_offset=qubit_offset,
            use_SK1_AM=use_SK1_AM,
            print_circuit=print_circuit,
            print_gate_list=print_gate_list,
        )

        for i, sv in enumerate(scan_array_to_scan):

            if scan_parameter_int == int(self.StabReadoutScanParameter.initial_state):
                str_to_return += circuit_prototypes.prep_in_X_basis(
                    prep_state=sv,
                    qubits_to_exclude=qubits_to_exclude,
                    circuit_index=i,
                    suppress_circuit_scan=False,
                    suppress_print=(i != 0),
                )
            else:
                str_to_return += circuit_prototypes.prep_in_X_basis(
                    prep_state=prep_state,
                    qubits_to_exclude=qubits_to_exclude,
                    circuit_index=i,
                    suppress_circuit_scan=True,
                    suppress_print=(i != 0),
                )

            if scan_parameter_int == int(
                self.StabReadoutScanParameter.post_prepare_phase
            ):
                post_prepare_phases_temp = post_prepare_phases
                post_prepare_phases_temp[gate_to_sweep] = sv
                str_to_return += circuit_prototypes.phase_shifts(
                    phase_shifts=post_prepare_phases_temp,
                    qubits_to_exclude=qubits_to_exclude,
                    circuit_index=i,
                    suppress_circuit_scan=False,
                    suppress_print=(i != 0),
                )
            else:
                str_to_return += circuit_prototypes.phase_shifts(
                    phase_shifts=post_prepare_phases,
                    qubits_to_exclude=qubits_to_exclude,
                    circuit_index=i,
                    suppress_circuit_scan=True,
                    suppress_print=(i != 0),
                )

            str_to_return += circuit_prototypes.single_stabilizer_readout(
                circuit_index=i, suppress_circuit_scan=True, suppress_print=(i != 0)
            )

            if scan_parameter_int == int(
                self.StabReadoutScanParameter.post_prepare_phase
            ):
                str_to_return += circuit_prototypes.phase_shifts(
                    phase_shifts=[
                        post_XX_phase - ppp for ppp in post_prepare_phases_temp
                    ],
                    qubits_to_exclude=qubits_to_exclude,
                    circuit_index=i,
                    suppress_circuit_scan=False,
                    suppress_print=(i != 0),
                )
            elif scan_parameter_int == int(self.StabReadoutScanParameter.post_XX_phase):
                str_to_return += circuit_prototypes.global_phase_shift(
                    phase_shift=sv,
                    qubits_to_exclude=qubits_to_exclude,
                    circuit_index=i,
                    suppress_circuit_scan=False,
                    suppress_print=(i != 0),
                )
            else:
                str_to_return += circuit_prototypes.global_phase_shift(
                    phase_shift=post_XX_phase,
                    qubits_to_exclude=qubits_to_exclude,
                    circuit_index=i,
                    suppress_circuit_scan=True,
                    suppress_print=(i != 0),
                )

            str_to_return += circuit_prototypes.readout_X_basis(
                qubits_to_exclude=qubits_to_exclude,
                circuit_index=i,
                suppress_circuit_scan=True,
                suppress_print=(i != 0),
            )

            self.add_rabi(
                slots=[17],
                detuning=0,
                sideband_order=0,
                duration=correction_length,
                phase=correction_phase,
                gate_name="Correction SK1",
                scan_parameter_int=correction_scan_parameter_int,
                min_value=min_value,
                max_value=max_value,
                N_points=N_points,
                circuit_index=i,
                suppress_circuit_scan=True,
            )

        return str_to_return

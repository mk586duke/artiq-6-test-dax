import logging
import math
import typing
from enum import IntEnum

import numpy as np

import euriqabackend.devices.keysight_awg.AOM_calibration as cal
import euriqabackend.devices.keysight_awg.interpolation_functions as intfn
import euriqabackend.devices.keysight_awg.physical_parameters as pp
import euriqabackend.devices.keysight_awg.waveform as wf
import euriqabackend.devices.keysight_awg.waveform_prototypes as wp
from . import common_types

# We put this import down here because physical_parameters imports GateType

_LOGGER = logging.getLogger(__name__)


class Gate:
    """The Gate class defines a single time slice of the experimental sequence.

    It contains the waveforms and slot assignments for all AWG slots.
    """

    def __init__(
        self,
        gate_type: common_types.GateType,
        physical_params: pp.PhysicalParams,
        slot_array: typing.List[int],
        ind_amp: float,
        global_amp: float,
        twoQ_gate: int,
        name: str,
        scan_parameter: int = -1,
        scan_values: typing.List[float] = None,
        wait_after: int = 0,
    ):
        """This initializes a gate, which specifies a group of waveforms to be output by
        the AWG to realize a single gate.

        Args:
            gate_type: the specific type (e.g., Rabi, SK1) of this gate, which determines the specific subclass that
                is implemented
            physical_params: a PhysicalParams object, which contains various physical parameters that have been
                passed in from the top-level controller
            slot_array: the slots (in the Switch network output & fiber) to which the RF pulses will be applied
            ind_amp: the overall amplitude of the individual amp channels, scaled to +-1.0
            global_amp: the overall amplitude of the global amp channels, scaled to +-1.0
            twoQ_gate: 0 or 1, determines whether this is a single-qubit or two-qubit gate   # todo: remap to bool
            name: the name of this particular gate
            scan_parameter: the gate-specific parameter (e.g., sideband imbalance for XX) that is being scanned,
                which is encoded here as a generic int
            scan_values: the values of the parameter that is being scanned
            wait_after: 0 or 1, determines whether a wait is inserted after each timestep   # todo: remap to bool
        """

        self.gate_type = gate_type
        self.physical_params = physical_params
        self.slot_array = slot_array
        self.ind_amp = ind_amp
        self.global_amp = global_amp
        self.twoQ_gate = twoQ_gate
        self.name = name
        self.scan_parameter = scan_parameter
        self.scan_values = scan_values
        self.wait_after = wait_after

        if self.scan_values is None:
            self.scan_values = list()
        self.N_scan_values = len(self.scan_values)

        self.wav_array = list()

    def assign_phase_gates(self, phase_gate_array: typing.List[float]):
        """This function assigns a prev_phase_gates value to all of its waveforms based
        on the phase_gate_array and the slots to which the wavefoms have been assigned.

        Args:
            phase_gate_array:
        """

        for i in range(len(self.wav_array)):
            self.wav_array[i].prev_phase_gates = phase_gate_array[self.slot_array[i]]
            #Remove the phase gate from this slot once it has been applied, otherwise it is double counted later
            phase_gate_array[self.slot_array[i]] = 0.0
        return phase_gate_array


class Phase(Gate):
    """This class implements a reference pulse on Ch D of the AWG."""

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        phase: float,
        wait_after: int = 0,
        name: str = "Phase",

    ):

        gate_type = common_types.GateType.Phase
        twoQ_gate = 0
        slot_array = [slot]

        ind_amp = 0
        global_amp = 0
        scan_parameter = -1
        scan_values = list()

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.phase = phase

    def set_scan(self):
        """There is nothing to scan in a phase gate, so this is a dummy function."""
        pass

    def compile_waveform(self):
        """The phase gate does implement an RF pulse, so this is also a dummy function."""
        pass


class ReferencePulse(Gate):
    """This class implements a reference pulse on Ch D of the AWG."""

    def __init__(self, physical_params: pp.PhysicalParams):

        gate_type = common_types.GateType.ReferencePulse
        twoQ_gate = 0
        wait_after = 0
        slot_array = [-1, -1, -1, -1]

        ind_amp = 0
        global_amp = 0
        scan_parameter = -1
        scan_values = list()

        name = (
            "Reference (ind)"
            if physical_params.monitor.monitor_ind
            else "Reference (global)"
        )

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.phase = 0

    def set_scan(self):
        """There is nothing to scan in the reference pulse, so this is a dummy function."""
        pass

    def compile_waveform(self):
        """We construct the reference pulse with duration TBD."""

        monitor_freq = (
            self.physical_params.f_ind
            if self.physical_params.monitor.monitor_ind
            else self.physical_params.f_carrier
        ) + self.physical_params.monitor.detuning
        monitor_duration = (
            0  # We set the actual pulse duration after compiling the complete sequence
        )

        # Insert the reference pulse into Ch D and blanks into the other three channels
        self.wav_array = [
            wp.blank(),
            wp.blank(),
            wp.blank(),
            wp.sine(
                self.physical_params.monitor.amp,
                monitor_freq,
                0,
                monitor_duration,
                name=self.name,
            ),
        ]


class Blank(Gate):
    """This class implements a blank gate, which can either be 1Q or 2Q."""

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slots: typing.List[int],
        twoQ_gate: int = 0,
        wait_after: int = 0,
        name: str = "Blank",
    ):

        gate_type = common_types.GateType.Blank
        wait_after = wait_after
        slot_array = [slots[0], slots[1], 0] if twoQ_gate == 1 else [slots[0], -1, 0]

        ind_amp = 0
        global_amp = 0
        scan_parameter = -1
        scan_values = list()

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.phase = 0

    def set_scan(self):
        """There is nothing to scan in the reference pulse, so this is a dummy function."""
        pass

    def compile_waveform(self):
        """We construct the blank gate with a series of blank waveforms."""

        self.wav_array = [wp.blank(), wp.blank(), wp.blank()]


class Rabi(Gate):
    """This class implements a Rabi pulse, with the ability to sweep several
    parameters."""

    class ScanParameter(IntEnum):
        static = -1
        duration = 0
        detuning = 1
        phase = 2
        ind_amplitude = 3
        global_amplitude = 4
        t_delay = 5

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        detuning: float,
        duration: float,
        phase: float,
        ind_amp: float,
        global_amp: float,
        wait_after: int = 0,
        name: str = "Rabi",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):

        gate_type = common_types.GateType.Rabi
        twoQ_gate = 0
        slot_array = [slot, -1, 0]

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.detuning = detuning
        self.duration = duration
        self.phase = phase

        # Initialize the waveform scan parameters and values.  If some parameter is being scanned, initialize to the
        # default dummy scan for both individual and global, then overwrite one or both in set_scan().
        if scan_parameter == self.ScanParameter.static:
            self.ind_scan_parameter = wf.ScanParameter.static
            self.global_scan_parameter = wf.ScanParameter.static
            self.ind_scan_values = list()
            self.global_scan_values = list()
        else:
            self.ind_scan_parameter = wf.ScanParameter.dummy_scan
            self.global_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_scan_values = [0] * self.N_scan_values
            self.global_scan_values = [0] * self.N_scan_values

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""

        if self.scan_parameter == int(self.ScanParameter.duration):
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            self.ind_scan_values = self.scan_values
            self.global_scan_values = self.scan_values
        elif self.scan_parameter == int(self.ScanParameter.detuning):
            self.global_scan_parameter = wf.ScanParameter.frequency
            self.global_scan_values = [
                self.physical_params.f_carrier + d for d in self.scan_values
            ]
        elif self.scan_parameter == int(self.ScanParameter.phase):
            self.global_scan_parameter = wf.ScanParameter.phase
            self.global_scan_values = self.scan_values
        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude):
            self.ind_scan_parameter = wf.ScanParameter.amplitude
            self.ind_scan_values = self.scan_values
        elif self.scan_parameter == int(self.ScanParameter.global_amplitude):
            self.global_scan_parameter = wf.ScanParameter.amplitude
            self.global_scan_values = self.scan_values
        elif self.scan_parameter == int(self.ScanParameter.t_delay):
            self.ind_scan_parameter = wf.ScanParameter.t_delay
            self.ind_scan_values = self.scan_values

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform."""

        self.wav_array = [
            wp.sine(
                self.ind_amp,
                self.physical_params.f_ind - self.physical_params.Rabi.Stark_shift,
                0,
                self.duration,
                PA_freq=self.physical_params.f_ind - self.physical_params.Rabi.Stark_shift,
                t_delay=self.physical_params.t_delay,
                delay_PA_freq=self.physical_params.f_ind,
                name=self.name + " ({0})".format(self.slot_array[0]),
                scan_parameter=self.ind_scan_parameter,
                scan_values=self.ind_scan_values,
            ),
            wp.blank(),
            wp.sine(
                self.global_amp,
                self.physical_params.f_carrier + self.detuning,
                self.phase,
                self.duration,
                name=self.name + " (global)",
                scan_parameter=self.global_scan_parameter,
                scan_values=self.global_scan_values,
            ),
        ]


class Rabi_AM(Gate):
    """This class implements a Rabi pulse, with the ability to sweep several
    parameters."""

    class ScanParameter(IntEnum):
        static = -1
        envelope_duration = 0
        global_delay = 1
        global_duration = 2
        detuning = 3
        phase = 4
        ind_amplitude = 5
        global_amplitude = 6
        t_delay = 7

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        detuning_on: float,
        detuning_off: float,
        envelope_type: intfn.InterpFunction.FunctionType,
        envelope_scale: float,
        envelope_duration: float,
        phase: float,
        ind_amp: float,
        global_amp_on: float,
        global_amp_off: float,
        global_delay: float,
        global_duration: float,
        wait_after: int = 0,
        name: str = "Rabi",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):

        gate_type = common_types.GateType.Rabi_AM
        twoQ_gate = 0
        slot_array = [slot, -1, 0]

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp_on,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.detuning_on = detuning_on
        self.detuning_off = detuning_off
        self.envelope_type = envelope_type
        self.envelope_scale = envelope_scale
        self.envelope_duration = envelope_duration
        self.phase = phase
        self.global_amp_off = global_amp_off
        self.global_delay = global_delay
        self.global_duration = global_duration

        # Initialize the waveform scan parameters and values.  If some parameter is being scanned, initialize to the
        # default dummy scan for both individual and global, then overwrite one or both in set_scan().
        if scan_parameter == self.ScanParameter.static:
            self.ind_scan_parameter = wf.ScanParameter.static
            self.global_scan_parameter = wf.ScanParameter.static
            self.ind_scan_values = list()
            self.global_scan_values = list()
        else:
            self.ind_scan_parameter = wf.ScanParameter.dummy_scan
            self.global_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_scan_values = [[0] * self.N_scan_values]
            self.global_scan_values = [[0] * self.N_scan_values] * 3

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""

        if self.scan_parameter == int(self.ScanParameter.envelope_duration):
            # Loop over different scan points (inner), then over segments (outer)
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            self.ind_scan_values = [self.scan_values]
            self.global_scan_values = [
                [self.global_delay] * self.N_scan_values,
                self.scan_values,
                [self.global_delay] * self.N_scan_values,
            ]
            # [max(0., sv - self.global_delay - self.global_duration) for
            #  sv in self.scan_values]]
        elif self.scan_parameter == int(self.ScanParameter.global_delay):
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_values = [
                self.scan_values,
                [self.global_duration] * self.N_scan_values,
                self.scan_values,
            ]
            # [max(0., self.envelope_duration - sv - self.global_duration) for
            #  sv in self.scan_values]]
        elif self.scan_parameter == int(self.ScanParameter.global_duration):
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_values = [
                [self.global_delay] * self.N_scan_values,
                self.scan_values,
                [self.global_delay] * self.N_scan_values,
            ]
            #  [max(0., self.envelope_duration - self.global_delay - sv) for
            #  sv in self.scan_values]]
        elif self.scan_parameter == int(self.ScanParameter.detuning):
            self.global_scan_parameter = wf.ScanParameter.frequency
            self.global_scan_values = [
                [self.physical_params.f_carrier + self.detuning_off]
                * self.N_scan_values,
                [self.physical_params.f_carrier + d for d in self.scan_values],
                [self.physical_params.f_carrier + self.detuning_off]
                * self.N_scan_values,
            ]
        elif self.scan_parameter == int(self.ScanParameter.phase):
            self.global_scan_parameter = wf.ScanParameter.phase
            self.global_scan_values = [self.scan_values] * 3
        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude):
            self.ind_scan_parameter = wf.ScanParameter.amplitude
            self.ind_scan_values = [self.scan_values]
        elif self.scan_parameter == int(self.ScanParameter.global_amplitude):
            self.global_scan_parameter = wf.ScanParameter.amplitude
            self.global_scan_values = [
                [self.global_amp_off] * self.N_scan_values,
                [self.scan_values],
                [self.global_amp_off] * self.N_scan_values,
            ]
        elif self.scan_parameter == int(self.ScanParameter.t_delay):
            self.ind_scan_parameter = wf.ScanParameter.t_delay
            self.ind_scan_values = [self.scan_values]

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform."""
        pulse_envelope = intfn.InterpFunction(
            function_type=self.envelope_type,
            start_value=0,
            stop_value=1,
            scale_params=[self.envelope_scale],
        )

        global_amps = [self.global_amp, self.global_amp, self.global_amp]
        global_freqs = [
            self.physical_params.f_carrier + d
            for d in [self.detuning_on, self.detuning_on, self.detuning_on]
        ]
        # global_freqs = [
        #     self.physical_params.f_carrier + d
        #     for d in [self.detuning_on, self.detuning_on, self.detuning_on]
        # ]
        global_durations = [self.global_delay, self.global_duration, self.global_delay]
        # max(0., self.envelope_duration - self.global_delay - self.global_duration)]
        # global_shape_up = intfn.InterpFunction(
        #     function_type=intfn.InterpFunction.FunctionType.half_Gaussian,
        #     start_value=0,
        #     stop_value=1,
        #     scale_params=[self.envelope_scale],
        # )

        global_shape_constant = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.constant,
            start_value=1,
            stop_value=1,
            scale_params=[1],
        )

        global_shape_ramp_up = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.ramp,
            start_value=0,
            stop_value=0,
            scale_params=[1],
        )
        global_shape_ramp_down = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.ramp,
            start_value=0,
            stop_value=0,
            scale_params=[1],
        )
        # global_ramp_duration = self.physical_params.t_delay
        print("\nphysical delay = ",self.physical_params.t_delay)
        print("\nGlobal durations = ",global_durations)
        print("\nInd duration = ", self.envelope_duration)
        self.wav_array = [
            wp.multisegment_AM(
                amplitude_fns=[pulse_envelope],
                amplitudes=[self.ind_amp],
                freqs=[
                    self.physical_params.f_ind - self.physical_params.Rabi.Stark_shift
                ],
                phases=[0],
                durations=[self.envelope_duration],
                PA_freqs=[
                    self.physical_params.f_ind - self.physical_params.Rabi.Stark_shift
                ],
                t_delay=self.physical_params.t_delay+0*self.global_delay,
                delay_PA_freq=self.physical_params.f_ind,
                name=self.name + " ({0})".format(self.slot_array[0]),
                scan_parameter=self.ind_scan_parameter,
                scan_values=self.ind_scan_values,
            ),
            wp.blank(),
            # wp.blank()
            wp.multisegment_AM(
                amplitude_fns=[global_shape_ramp_up,global_shape_constant,global_shape_ramp_down],
                amplitudes=global_amps,
                freqs=global_freqs,
                phases=[self.phase] * 3,
                durations=global_durations,
                # PA_freqs = global_freqs,
                # t_delay=0.0,
                # delay_PA_freq=global_freqs[1],
                name=self.name + " (global)",
                scan_parameter=self.global_scan_parameter,
                scan_values=self.global_scan_values,
            ),
            # wp.multisegment_AM(
            #     amplitude_fns=[global_shape_constant],
            #     amplitudes=[self.global_amp],
            #     freqs=[global_freqs[0]],
            #     phases=[self.phase] ,
            #     durations=[self.global_duration],
            #     # PA_freqs = global_freqs,
            #     # t_delay=0.0,
            #     # delay_PA_freq=global_freqs[1],
            #     name=self.name + " (global)",
            #     scan_parameter=self.global_scan_parameter,
            #     scan_values=self.global_scan_values,
            # ),

        ]


class Rabi_PI(Gate):
    """This class implements a Rabi pulse, with the ability to sweep several
    parameters."""

    class ScanParameter(IntEnum):
        static = -1
        duration = 0
        detuning = 1
        phase = 2
        ind_amplitude = 3

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        detuning: float,
        duration: float,
        phase: float,
        ind_amp: float,
        wait_after: int = 0,
        name: str = "Rabi",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):
        # HACK HACK HACK
        self.put_on_global = False

        gate_type = common_types.GateType.Rabi_PI
        twoQ_gate = 0
        slot_array = [-1, -1, 0] if self.put_on_global else [slot]
        global_amp = 0

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.detuning = detuning
        self.duration = duration
        self.phase = phase

        self.center_freq = (
            self.physical_params.f_carrier
            if self.put_on_global
            else self.physical_params.PI_center_freq_1Q
        )

        # Initialize the waveform scan parameters and values.  If some parameter is being scanned, initialize to the
        # default dummy scan for both individual and global, then overwrite one or both in set_scan().
        if scan_parameter == self.ScanParameter.static:
            self.ind_scan_parameter = wf.ScanParameter.static
            self.ind_scan_values = list()
        else:
            self.ind_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_scan_values = [0, 0] * self.N_scan_values

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""

        if self.scan_parameter == int(self.ScanParameter.duration):
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.ind_scan_values = self.scan_values
        elif self.scan_parameter == int(self.ScanParameter.detuning):
            # Loop over tones (inner), then over different scan points (outer)
            self.ind_scan_parameter = wf.ScanParameter.frequency
            freq_splitting = [
                (self.physical_params.f_ind + 140)
                - (self.physical_params.f_carrier + d)
                for d in self.scan_values
            ]
            self.ind_scan_values = [
                [self.center_freq - fs / 2, self.center_freq + fs / 2]
                for fs in freq_splitting
            ]
        elif self.scan_parameter == int(self.ScanParameter.phase):
            self.ind_scan_parameter = wf.ScanParameter.phase
            self.ind_scan_values = [[-sv / 2, sv / 2] for sv in self.scan_values]
        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude):
            self.ind_scan_parameter = wf.ScanParameter.amplitude
            self.ind_scan_values = [[sv / 2] * 2 for sv in self.scan_values]

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform."""

        freq_splitting = (
            self.physical_params.f_ind + 140 - self.physical_params.Rabi.Stark_shift
        ) - (self.physical_params.f_carrier + self.detuning)
        freqs = [
            self.center_freq - freq_splitting / 2,
            self.center_freq + freq_splitting / 2,
        ]
        phases = [-self.phase / 2, self.phase / 2]

        if self.put_on_global:
            self.wav_array = [
                wp.blank(),
                wp.blank(),
                wp.multitone(
                    amplitudes=[self.ind_amp / 2] * 2,
                    freqs=freqs,
                    phases=phases,
                    duration=self.duration,
                    PA_freq=self.physical_params.f_ind
                    - self.physical_params.Rabi.Stark_shift,
                    delay_PA_freq=self.physical_params.f_ind,
                    name=self.name + " ({0})".format(self.slot_array[0]),
                    scan_parameter=self.ind_scan_parameter,
                    scan_values=self.ind_scan_values,
                ),
            ]
        else:
            self.wav_array = [
                wp.multitone(
                    amplitudes=[self.ind_amp / 2] * 2,
                    freqs=freqs,
                    phases=phases,
                    duration=self.duration,
                    PA_freq=self.physical_params.f_ind
                    - self.physical_params.Rabi.Stark_shift,
                    delay_PA_freq=self.physical_params.f_ind,
                    name=self.name + " ({0})".format(self.slot_array[0]),
                    scan_parameter=self.ind_scan_parameter,
                    scan_values=self.ind_scan_values,
                )
            ]


class Bichromatic(Gate):
    """This class implements a bichromatic Rabi pulse, with the ability to sweep several
    parameters."""

    class ScanParameter(IntEnum):
        static = -1
        duration = 0
        diff_detuning = 1
        common_detuning = 2
        qubit_phase = 3
        motional_phase = 4
        ind_amplitude = 5
        global_amplitude = 6
        sideband_imbalance = 7

    class ActiveBeams(IntEnum):
        neither = 0  # All off
        ind_only = 1  # Two tones on ind, no global. Could be used for phase insensitive
        global_only = (
            2  # Two tones on global, no ind. Could be used for phase insensitive
        )
        both = 3  # This will put the two tones on the global

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        diff_detuning: float,
        common_detuning: float,
        duration: float,
        motional_phase: float,
        qubit_phase: float,
        ind_amp: float,
        global_amp: float,
        sideband_imbalance: float = 0,
        wait_after: int = 0,
        active_beams: ActiveBeams = ActiveBeams.both,
        name: str = "Bichromatic",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):

        gate_type = common_types.GateType.Bichromatic
        twoQ_gate = 1
        slot_array = [slot, -1, 0]

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.diff_detuning = diff_detuning
        self.common_detuning = common_detuning
        self.duration = duration
        self.qubit_phase = qubit_phase
        self.motional_phase = motional_phase
        self.red_phase = qubit_phase-motional_phase-np.pi/2
        self.blue_phase = qubit_phase+motional_phase-np.pi/2
        self.sideband_imbalance = sideband_imbalance
        self.active_beams = active_beams

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""
        if self.scan_parameter == int(self.ScanParameter.static):
            self.ind_scan_parameter = wf.ScanParameter.static
            self.ind_scan_values = list()
            self.global_scan_parameter = wf.ScanParameter.static
            self.global_scan_values = list()
        else:
            self.ind_scan_parameter = wf.ScanParameter.dummy_scan
            self.global_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_scan_values = [[0] for _ in self.scan_values]
            self.global_scan_values = [[0, 0] for _ in self.scan_values]

        if self.scan_parameter == int(self.ScanParameter.duration):
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.ind_scan_values = self.scan_values
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_values = self.scan_values

        elif self.scan_parameter == int(self.ScanParameter.diff_detuning):
            self.global_scan_parameter = wf.ScanParameter.frequency

            if self.active_beams == self.ActiveBeams.ind_only:
                self.global_scan_values = [
                    [
                        self.physical_params.PI_center_freq_1Q
                        + sv
                        + self.common_detuning,
                        self.physical_params.PI_center_freq_1Q
                        - sv
                        + self.common_detuning,
                    ]
                    for sv in self.scan_values
                ]
            else:
                self.global_scan_values = [
                    [
                        self.physical_params.f_carrier + sv + self.common_detuning,
                        self.physical_params.f_carrier - sv + self.common_detuning,
                    ]
                    for sv in self.scan_values
                ]

        elif self.scan_parameter == int(self.ScanParameter.common_detuning):
            self.global_scan_parameter = wf.ScanParameter.frequency

            if self.active_beams == self.ActiveBeams.ind_only:
                self.global_scan_values = [
                    [
                        self.physical_params.PI_center_freq_1Q
                        + self.diff_detuning
                        + sv,
                        self.physical_params.PI_center_freq_1Q
                        - self.diff_detuning
                        + sv,
                    ]
                    for sv in self.scan_values
                ]
            else:
                self.global_scan_values = [
                    [
                        self.physical_params.f_carrier + self.diff_detuning + sv,
                        self.physical_params.f_carrier - self.diff_detuning + sv,
                    ]
                    for sv in self.scan_values
                ]

        elif self.scan_parameter == int(self.ScanParameter.qubit_phase):
            self.global_scan_parameter = wf.ScanParameter.phase
            self.global_scan_values = [[sv+self.motional_phase-np.pi/2, sv-self.motional_phase-np.pi/2] for sv in self.scan_values]

        elif self.scan_parameter == int(self.ScanParameter.motional_phase):
            self.global_scan_parameter = wf.ScanParameter.phase
            self.global_scan_values = [[self.qubit_phase+sv-np.pi/2, self.qubit_phase-sv-np.pi/2] for sv in self.scan_values]

        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude):
            if self.ActiveBeams.ind_only == self.active_beams:
                self.global_scan_parameter = wf.ScanParameter.amplitude
                self.global_scan_values = [
                    [
                        (1 + self.sideband_imbalance) / 2 * sv,
                        (1 - self.sideband_imbalance) / 2 * sv,
                    ]
                    for sv in self.scan_values
                ]
            else:
                self.ind_scan_parameter = wf.ScanParameter.amplitude
                self.ind_scan_values = self.scan_values

        elif self.scan_parameter == int(self.ScanParameter.global_amplitude):
            if self.ActiveBeams.ind_only == self.active_beams:
                pass
            else:
                self.global_scan_parameter = wf.ScanParameter.amplitude
                self.global_scan_values = [
                    [
                        (1 + self.sideband_imbalance) / 2 * sv,
                        (1 - self.sideband_imbalance) / 2 * sv,
                    ]
                    for sv in self.scan_values
                ]

        elif self.scan_parameter == int(self.ScanParameter.sideband_imbalance):
            self.global_scan_parameter = wf.ScanParameter.amplitude
            if self.ActiveBeams.ind_only == self.active_beams:
                self.global_scan_values = [
                    [(1 + sv) / 2 * self.ind_amp, (1 - sv) / 2 * self.ind_amp]
                    for sv in self.scan_values
                ]
            else:
                self.global_scan_values = [
                    [(1 + sv) / 2 * self.global_amp, (1 - sv) / 2 * self.global_amp]
                    for sv in self.scan_values
                ]

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform."""

        if self.active_beams == self.ActiveBeams.ind_only:
            freqs = [
                self.physical_params.PI_center_freq_1Q
                + self.diff_detuning
                + self.common_detuning,
                self.physical_params.PI_center_freq_1Q
                - self.diff_detuning
                + self.common_detuning,
            ]
            amps = [
                (1 + self.sideband_imbalance) / 2 * self.ind_amp,
                (1 - self.sideband_imbalance) / 2 * self.ind_amp,
            ]
        else:
            freqs = [
                self.physical_params.f_carrier
                + self.diff_detuning
                + self.common_detuning,
                self.physical_params.f_carrier
                - self.diff_detuning
                + self.common_detuning,
            ]
            amps = [
                (1 + self.sideband_imbalance) / 2 * self.global_amp,
                (1 - self.sideband_imbalance) / 2 * self.global_amp,
            ]

        if self.active_beams == self.ActiveBeams.neither:
            self.wav_array = [wp.blank(), wp.blank(), wp.blank()]

        elif self.active_beams == self.ActiveBeams.both:
            self.wav_array = [
                wp.sine(
                    amplitude=self.ind_amp,
                    freq=self.physical_params.f_ind,
                    phase=0.0,
                    duration=self.duration,
                    t_delay=self.physical_params.t_delay,
                    delay_PA_freq=self.physical_params.f_ind,
                    name=self.name + " ({0})".format(self.slot_array[1]),
                    scan_parameter=self.ind_scan_parameter,
                    scan_values=self.ind_scan_values,
                ),
                wp.blank(),
                wp.multitone(
                    amplitudes=amps,
                    freqs=freqs,
                    phases=[self.blue_phase,self.red_phase],
                    duration=self.duration,
                    name=self.name + " (global)",
                    scan_parameter=self.global_scan_parameter,
                    scan_values=self.global_scan_values,
                ),
            ]

        elif self.active_beams == self.ActiveBeams.ind_only:
            self.wav_array = [
                wp.multitone(
                    amplitudes=amps,
                    freqs=freqs,
                    phases=[self.blue_phase,self.red_phase],
                    duration=self.duration,
                    name=self.name + " ({0})".format(self.slot_array[1]),
                    scan_parameter=self.global_scan_parameter,
                    scan_values=self.global_scan_values,
                ),
                wp.blank(),
                wp.blank(),
            ]

        elif self.active_beams == self.ActiveBeams.global_only:
            self.wav_array = [
                wp.blank(),
                wp.blank(),
                wp.multitone(
                    amplitudes=amps,
                    freqs=freqs,
                    phases=[self.blue_phase,self.red_phase],
                    duration=self.duration,
                    name=self.name + " (global)",
                    scan_parameter=self.global_scan_parameter,
                    scan_values=self.global_scan_values,
                ),
            ]


class WindUnwind(Gate):
    """This class implements a wind-unwind pulse, which applies a square Rabi pulse of a
    given amplitude for a given duration and then applies a pi phase shift and undoes
    the oscillation with a second pulse with independent amplitude and duration."""

    class ScanParameter(IntEnum):
        static = -1
        duration_1 = 0
        duration_2 = 1

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        detuning: float,
        duration_1: float,
        duration_2: float,
        ind_amp_1: float,
        ind_amp_2: float,
        global_amp_1: float,
        global_amp_2: float,
        name: str = "Wind-Unwind",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):

        gate_type = common_types.GateType.WindUnwind
        twoQ_gate = 0
        slot_array = [slot, -1, 0]
        ind_amp = 0
        global_amp = 0
        wait_after = 0

        super().__init__(
            gate_type=gate_type,
            physical_params=physical_params,
            slot_array=slot_array,
            ind_amp=ind_amp,
            global_amp=global_amp,
            twoQ_gate=twoQ_gate,
            name=name,
            scan_parameter=int(scan_parameter),
            scan_values=scan_values,
            wait_after=wait_after,
        )

        self.detuning = detuning
        self.duration_1 = duration_1
        self.duration_2 = duration_2
        self.ind_amp_1 = ind_amp_1
        self.ind_amp_2 = ind_amp_2
        self.global_amp_1 = global_amp_1
        self.global_amp_2 = global_amp_2

        # Initialize the waveform scan parameters and values.  Both scan options are durations, so we may set both scan
        # parameters to seg_duration by default.
        if scan_parameter == self.ScanParameter.static:
            self.ind_scan_parameter = wf.ScanParameter.static
            self.global_scan_parameter = wf.ScanParameter.static
            self.ind_scan_values = list()
            self.global_scan_values = list()
        else:
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            self.ind_scan_values = [[0] * self.N_scan_values] * 2
            self.global_scan_values = [[0] * self.N_scan_values] * 2

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""

        if self.scan_parameter == int(self.ScanParameter.duration_1):
            self.ind_scan_values = [
                self.scan_values,
                [self.duration_2] * self.N_scan_values,
            ]
            self.global_scan_values = self.ind_scan_values
        elif self.scan_parameter == int(self.ScanParameter.duration_2):
            self.ind_scan_values = [
                [self.duration_1] * self.N_scan_values,
                self.scan_values,
            ]
            self.global_scan_values = self.ind_scan_values

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform."""

        self.wav_array = [
            wp.multisegment(
                amplitudes=[self.ind_amp_1, self.ind_amp_2],
                freqs=[
                    self.physical_params.f_ind - self.physical_params.Rabi.Stark_shift
                ]
                * 2,
                phases=[0, 0],
                durations=[self.duration_1, self.duration_2],
                PA_freqs=[
                    self.physical_params.f_ind - self.physical_params.Rabi.Stark_shift
                ]
                * 2,
                t_delay=self.physical_params.t_delay,
                delay_PA_freq=self.physical_params.f_ind,
                name=self.name + " ({0})".format(self.slot_array[0]),
                scan_parameter=self.ind_scan_parameter,
                scan_values=self.ind_scan_values,
            ),
            wp.blank(),
            wp.multisegment(
                amplitudes=[self.global_amp_1, self.global_amp_2],
                freqs=[self.physical_params.f_carrier + self.detuning] * 2,
                phases=[0, np.pi],
                durations=[self.duration_1, self.duration_2],
                name=self.name + " (global)",
                scan_parameter=self.global_scan_parameter,
                scan_values=self.global_scan_values,
            ),
        ]


class CrosstalkCalib(Gate):
    """This class implements a wind-unwind pulse, which applies a square Rabi pulse of a
    given amplitude for a given duration and then applies a pi phase shift and undoes
    the oscillation with a second pulse with independent amplitude and duration."""

    class ScanParameter(IntEnum):
        static = -1
        duration = 0
        phase_weak = 1
        ind_amp_weak = 2

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot_strong: int,
        slot_weak: int,
        detuning: float,
        duration: float,
        ind_amp_strong: float,
        ind_amp_weak: float,
        global_amp: float,
        phase_strong: float = 0,
        phase_weak: float = 0,
        name: str = "CrosstalkCalib",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):

        gate_type = common_types.GateType.CrosstalkCalib
        twoQ_gate = 1
        slot_array = [slot_weak, slot_strong, 0]
        ind_amp = 0
        wait_after = 0

        super().__init__(
            gate_type=gate_type,
            physical_params=physical_params,
            slot_array=slot_array,
            ind_amp=ind_amp,
            global_amp=global_amp,
            twoQ_gate=twoQ_gate,
            name=name,
            scan_parameter=int(scan_parameter),
            scan_values=scan_values,
            wait_after=wait_after,
        )

        self.detuning = detuning
        self.duration = duration
        self.ind_amp_strong = ind_amp_strong
        self.ind_amp_weak = ind_amp_weak
        self.phase_strong = phase_strong
        self.phase_weak = phase_weak

        # If some parameter is being scanned, initialize to the
        # default dummy scan for both individual and global, then overwrite one or both in set_scan().
        if scan_parameter == self.ScanParameter.static:
            self.ind_strong_scan_parameter = wf.ScanParameter.static
            self.ind_weak_scan_parameter = wf.ScanParameter.static
            self.global_scan_parameter = wf.ScanParameter.static
            self.ind_strong_scan_values = list()
            self.ind_weak_scan_values = list()
            self.global_scan_values = list()
        else:
            self.ind_strong_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_weak_scan_parameter = wf.ScanParameter.dummy_scan
            self.global_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_strong_scan_values = [0] * self.N_scan_values
            self.ind_weak_scan_values = [0] * self.N_scan_values
            self.global_scan_values = [0] * self.N_scan_values

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms.

        :return:
        """

        if self.scan_parameter == int(self.ScanParameter.duration):
            self.ind_strong_scan_parameter = wf.ScanParameter.seg_duration
            self.ind_weak_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            self.ind_strong_scan_values = self.scan_values
            self.ind_weak_scan_values = self.scan_values
            self.global_scan_values = self.scan_values
        elif self.scan_parameter == int(self.ScanParameter.phase_weak):
            self.ind_weak_scan_parameter = wf.ScanParameter.phase
            self.ind_weak_scan_values = self.scan_values
        elif self.scan_parameter == int(self.ScanParameter.ind_amp_weak):
            self.ind_weak_scan_parameter = wf.ScanParameter.amplitude
            self.ind_weak_scan_values = self.scan_values

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform.

        :return:
        """

        self.wav_array = [
            wp.sine(
                self.ind_amp_weak,
                self.physical_params.f_ind - self.physical_params.Rabi.Stark_shift,
                self.phase_weak,
                self.duration,
                PA_freq=self.physical_params.f_ind
                - self.physical_params.Rabi.Stark_shift,
                t_delay=self.physical_params.t_delay,
                delay_PA_freq=self.physical_params.f_ind,
                name=self.name + " ({0})".format(self.slot_array[0]),
                scan_parameter=self.ind_weak_scan_parameter,
                scan_values=self.ind_weak_scan_values,
            ),
            wp.sine(
                self.ind_amp_strong,
                self.physical_params.f_ind - self.physical_params.Rabi.Stark_shift,
                self.phase_strong,
                self.duration,
                PA_freq=self.physical_params.f_ind
                - self.physical_params.Rabi.Stark_shift,
                t_delay=self.physical_params.t_delay,
                delay_PA_freq=self.physical_params.f_ind,
                name=self.name + " ({0})".format(self.slot_array[0]),
                scan_parameter=self.ind_strong_scan_parameter,
                scan_values=self.ind_strong_scan_values,
            ),
            wp.sine(
                self.global_amp,
                self.physical_params.f_carrier + self.detuning,
                0,
                self.duration,
                name=self.name + " (global)",
                scan_parameter=self.global_scan_parameter,
                scan_values=self.global_scan_values,
            ),
        ]


class FastEcho(Gate):
    """
    This class implements a fast echo pulse, which drives either on the carrier or symmetrically around the carrier and
    flips the drive phase periodically.  This is useful for measuring the Stark shift.  See our notes from 03
    January 2019.

    We can apply any combination of individual and global beams.  Since we're currently running in the phase-sensitive
    mode, we apply the two sidebands to the global.  We therefore echo by switching the phase of the two global
    sidebands, but only in the case where we apply both beams.  If we want to sweep the duration while applying both
    beams, then we only sweep the duration of the individual beam and leave the phase-flipping global beam alone.
    """

    class ScanParameter(IntEnum):
        static = -1
        duration = 0
        detuning = 1
        ind_amplitude = 3
        global_amplitude = 2
        sb_amplitude_imbalance = 4

    class ActiveBeams(IntEnum):
        neither = 0
        ind_only = 1
        global_only = 2
        both = 3

    def coerce_echo_duration(self, exact_time: float):
        """Round to the nearest 10^-15 to avoid machine precision error, then round up to
        nearest multiple of twice the echo duration (each phase lasts one echo duration).

        :param exact_time: the exact time to be rounded up
        :return:
        """
        return (2 * self.echo_duration) * math.ceil(
            round(exact_time, 15) / (2 * self.echo_duration)
        )

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        detuning: float,
        duration: float,
        ind_amp: float,
        global_amp: float,
        wait_after: int = 0,
        active_beams: ActiveBeams = ActiveBeams.both,
        sideband_imbalance: float = 0,
        echo_duration: float = 0.1,  # Use a 100 ns echo period by default
        name: str = "Fast echo",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):

        gate_type = common_types.GateType.FastEcho
        twoQ_gate = 0
        slot_array = [slot, -1, 0]

        super().__init__(
            gate_type=gate_type,
            physical_params=physical_params,
            slot_array=slot_array,
            ind_amp=ind_amp,
            global_amp=global_amp,
            twoQ_gate=twoQ_gate,
            name=name,
            scan_parameter=int(scan_parameter),
            scan_values=scan_values,
            wait_after=wait_after,
        )

        self.detuning = detuning
        self.duration = duration
        self.active_beams = active_beams
        self.sideband_imbalance = sideband_imbalance
        self.echo_duration = echo_duration

        self.N_echo_cycles = int(self.duration / (2 * self.echo_duration))

        # if abs(self.duration - 2*self.N_echo_cycles*self.echo_duration) > 10**-10:
        #     _LOGGER.error(
        #           "The total duration must be an integer multiple of "
        #           "the echo period (%f us)",
        #           self.echo_duration
        #     )
        #     raise Exception(
        #       "The total duration must be an integer multiple of the echo period ({0} us)".
        #                     format(self.echo_duration))
        if abs(wf.coerce_delta_t(self.echo_duration) - self.echo_duration) > 10 ** -10:
            _LOGGER.error("The echo period must be an even integer multiple of 10 ns")
            raise Exception("The echo period must be an even integer multiple of 10 ns")

        # Initialize the waveform scan parameters and values.  If some parameter is being scanned, initialize to the
        # default dummy scan for both individual and global, then overwrite one or both in set_scan().
        if scan_parameter == self.ScanParameter.static:
            self.ind_scan_parameter = wf.ScanParameter.static
            self.global_scan_parameter = wf.ScanParameter.static
            self.ind_scan_values = list()
            self.global_scan_values_blue = list()
            self.global_scan_values_red = list()
        else:
            self.ind_scan_parameter = wf.ScanParameter.dummy_scan
            self.global_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_scan_values = [0] * self.N_scan_values
            self.global_scan_values_blue = [0] * self.N_scan_values
            self.global_scan_values_red = [0] * self.N_scan_values

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""

        if self.scan_parameter == int(self.ScanParameter.duration):
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            # Here, the ind_scan_values refer to the duration in terms of us, but the global_scan_values refer to the
            # duration in terms of echo cycles.  This is a result of the global waveform consisting of short, phase-
            # alternating segments, which necessarily means that its duration must be quantized.
            self.ind_scan_values = self.scan_values
            self.global_scan_values_blue = [
                int(d / (2 * self.echo_duration)) for d in self.scan_values
            ]
            self.global_scan_values_red = self.global_scan_values_blue
        elif self.scan_parameter == int(self.ScanParameter.detuning):
            self.global_scan_parameter = wf.ScanParameter.frequency
            self.global_scan_values_blue = [
                self.physical_params.f_carrier + d for d in self.scan_values
            ]
            self.global_scan_values_red = [
                self.physical_params.f_carrier - d for d in self.scan_values
            ]
        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude):
            self.ind_scan_parameter = wf.ScanParameter.amplitude
            self.ind_scan_values = self.scan_values
        elif self.scan_parameter == int(self.ScanParameter.global_amplitude):
            self.global_scan_parameter = wf.ScanParameter.amplitude
            self.global_scan_values_blue = [
                (1 + self.sideband_imbalance) / 2 * ga for ga in self.scan_values
            ]
            self.global_scan_values_red = [
                (1 - self.sideband_imbalance) / 2 * ga for ga in self.scan_values
            ]
        elif self.scan_parameter == int(self.ScanParameter.sb_amplitude_imbalance):
            self.global_scan_parameter = wf.ScanParameter.amplitude
            self.global_scan_values_blue = [
                (1 + si) / 2 * self.global_amp for si in self.scan_values
            ]
            self.global_scan_values_red = [
                (1 - si) / 2 * self.global_amp for si in self.scan_values
            ]

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform."""

        if self.active_beams == self.ActiveBeams.both:
            self.wav_array = [
                wp.sine(
                    amplitude=self.ind_amp,
                    freq=self.physical_params.f_ind,
                    phase=0,
                    duration=self.duration,
                    PA_freq=self.physical_params.f_ind
                    - self.physical_params.Rabi.Stark_shift,
                    t_delay=self.physical_params.t_delay,
                    delay_PA_freq=self.physical_params.f_ind,
                    name=self.name + " ({0})".format(self.slot_array[0]),
                    scan_parameter=self.ind_scan_parameter,
                    scan_values=self.ind_scan_values,
                ),
                wp.blank(),
                wp.fastecho(
                    mean_amplitude=self.global_amp,
                    amplitude_imbalance=self.sideband_imbalance,
                    center_freq=self.physical_params.f_carrier,
                    detuning=self.detuning,
                    echo_duration=self.echo_duration,
                    N_echo_cycles=int(self.duration / (2 * self.echo_duration)),
                    name=self.name + " (global)",
                    scan_parameter=self.global_scan_parameter,
                    scan_values_blue=self.global_scan_values_blue,
                    scan_values_red=self.global_scan_values_red,
                ),
            ]

        elif self.active_beams == self.ActiveBeams.global_only:
            self.wav_array = [
                wp.blank(),
                wp.blank(),
                wp.fastecho(
                    mean_amplitude=self.global_amp,
                    amplitude_imbalance=self.sideband_imbalance,
                    center_freq=self.physical_params.f_carrier,
                    detuning=self.detuning,
                    echo_duration=self.echo_duration,
                    N_echo_cycles=int(self.duration / (2 * self.echo_duration)),
                    name=self.name + " (global)",
                    scan_parameter=self.global_scan_parameter,
                    scan_values_blue=self.global_scan_values_blue,
                    scan_values_red=self.global_scan_values_red,
                ),
            ]

        elif self.active_beams == self.ActiveBeams.ind_only:
            self.wav_array = [
                wp.sine(
                    amplitude=self.ind_amp,
                    freq=self.physical_params.f_ind,
                    phase=0,
                    duration=self.duration,
                    PA_freq=self.physical_params.f_ind
                    - self.physical_params.Rabi.Stark_shift,
                    t_delay=self.physical_params.t_delay,
                    delay_PA_freq=self.physical_params.f_ind,
                    name=self.name + " ({0})".format(self.slot_array[0]),
                    scan_parameter=self.ind_scan_parameter,
                    scan_values=self.ind_scan_values,
                ),
                wp.blank(),
                wp.blank(),
            ]

        elif self.active_beams == self.ActiveBeams.neither:
            self.wav_array = [wp.blank(), wp.blank(), wp.blank()]


class SK1(Gate):
    """This class implements an SK1 pulse, with the ability to sweep either the Tpi
    value of the rotation phase phi."""

    class ScanParameter(IntEnum):
        static = -1
        theta = 0
        phi = 1
        Tpi = 2
        Stark_shift = 3
        ind_amplitude = 4
        global_amplitude = 5
        t_delay = 6

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        theta: float,
        phi: float,
        ind_amp: float,
        global_amp: float,
        wait_after: int = 0,
        name: str = "SK1",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):

        gate_type = common_types.GateType.SK1
        twoQ_gate = 0
        slot_array = [slot, -1, 0]

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.theta = theta
        self.phi = phi
        if theta >= 4 * np.pi:
            _LOGGER.error("SK1 rotation angles greater than 4pi are not allowed")
            raise ValueError(
                "ERROR: SK1 rotation angles greater than 4pi are not allowed"
            )

        # Initialize the waveform scan parameters and values.  If some parameter is being scanned, initialize to the
        # default dummy scan for both individual and global, then overwrite one or both in set_scan().
        if scan_parameter == self.ScanParameter.static:
            self.ind_scan_parameter = wf.ScanParameter.static
            self.global_scan_parameter = wf.ScanParameter.static
            self.ind_scan_values = [list()] * 3
            self.global_scan_values = [list()] * 3
        else:
            self.ind_scan_parameter = wf.ScanParameter.dummy_scan
            self.global_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_scan_values = [[0] * self.N_scan_values] * 3
            self.global_scan_values = [[0] * self.N_scan_values] * 3

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""

        if self.scan_parameter == int(self.ScanParameter.phi):
            self.global_scan_parameter = wf.ScanParameter.phase
            sv_global_rotation = self.scan_values
            sv_global_correction1 = [
                np.remainder(p - np.arccos(self.theta / (-4 * np.pi)), 2 * np.pi)
                for p in self.scan_values
            ]
            sv_global_correction2 = [
                np.remainder(p + np.arccos(self.theta / (-4 * np.pi)), 2 * np.pi)
                for p in self.scan_values
            ]
            self.global_scan_values = [
                sv_global_rotation,
                sv_global_correction1,
                sv_global_correction2,
            ]
        elif self.scan_parameter == int(self.ScanParameter.theta):
            _LOGGER.error(
                "Error: Theta should not be scanned by the SK1 gate because it is scanned at the circuit level"
            )
            raise Exception(
                "Error: Theta should not be scanned by the SK1 gate because it is scanned at the circuit level"
            )
        elif self.scan_parameter == int(self.ScanParameter.Tpi):
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            sv_ind_rotation = [Tpi * self.theta / np.pi for Tpi in self.scan_values]
            sv_ind_correction = [
                (0 if self.theta == 0 else 2 * Tpi) for Tpi in self.scan_values
            ]
            self.ind_scan_values = [
                sv_ind_rotation,
                sv_ind_correction,
                sv_ind_correction,
            ]
            self.global_scan_values = self.ind_scan_values
        elif self.scan_parameter == int(self.ScanParameter.Stark_shift):
            _LOGGER.error(
                "Stark shift should not be scanned by the SK1 gate because it should be scanned at the circuit level"
            )
            raise Exception(
                "Stark shift should not be scanned by the SK1 gate because it should be scanned at the circuit level"
            )
        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude):
            self.ind_scan_parameter = wf.ScanParameter.amplitude
            self.ind_scan_values = [self.scan_values] * 3
        elif self.scan_parameter == int(self.ScanParameter.global_amplitude):
            self.global_scan_parameter = wf.ScanParameter.amplitude
            self.global_scan_values = [self.scan_values] * 3
        elif self.scan_parameter == int(self.ScanParameter.t_delay):
            self.ind_scan_parameter = wf.ScanParameter.t_delay
            self.ind_scan_values = [list(self.scan_values)] * 3

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform."""

        Tpi = (
            1
            / (2 * self.physical_params.Rabi_max)
            * self.physical_params.SK1.Tpi_multiplier
        )
        T_rotation = Tpi * self.theta / np.pi
        T_correction = 0 if self.theta == 0 else 2 * Tpi

        phi_correction1 = np.remainder(
            self.phi - np.arccos(self.theta / (-4 * np.pi)), 2 * np.pi
        )
        phi_correction2 = np.remainder(
            self.phi + np.arccos(self.theta / (-4 * np.pi)), 2 * np.pi
        )

        self.wav_array = [
            wp.multisegment(
                [self.ind_amp] * 3,
                [self.physical_params.f_ind - self.physical_params.SK1.Stark_shift] * 3,
                [0] * 3,
                [T_rotation, T_correction, T_correction],
                PA_freqs=[
                    self.physical_params.f_ind - self.physical_params.SK1.Stark_shift
                ]
                * 3,
                t_delay=self.physical_params.t_delay,
                delay_PA_freq=self.physical_params.f_ind,
                name=self.name + " ({0})".format(self.slot_array[0]),
                scan_parameter=self.ind_scan_parameter,
                scan_values=self.ind_scan_values,
            ),
            wp.blank(),
            wp.multisegment(
                [self.global_amp] * 3,
                [self.physical_params.f_carrier] * 3,
                [self.phi, phi_correction1, phi_correction2],
                [T_rotation, T_correction, T_correction],
                name=self.name + " (global)",
                scan_parameter=self.global_scan_parameter,
                scan_values=self.global_scan_values,
            ),
        ]


class SK1_AM(Gate):
    """This class implements an SK1 pulse, with the ability to sweep either the Tpi
    value of the rotation phase phi.

    We shape the amplitude of the individual channel in order to avoid exciting the
    axial modes.
    """

    class ScanParameter(IntEnum):
        static = -1
        theta = 0
        phi = 1
        rotation_pulse_length = 2
        correction_pulse_1_length = 3
        correction_pulse_2_length = 4
        Stark_shift = 5
        ind_amplitude = 6
        global_amplitude = 7
        t_delay = 8

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        slot: int,
        theta: float,
        phi: float,
        ind_amp: float,
        global_amp: float,
        use_global_segment_durations: bool = False,
        global_ramp: float = 1.0,
        shaped_global: bool = False,
        wait_after: int = 0,
        name: str = "SK1_AM",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
    ):

        gate_type = common_types.GateType.SK1_AM
        twoQ_gate = 0
        slot_array = [slot, -1, 0]

        super().__init__(
            gate_type,
            physical_params,
            slot_array,
            ind_amp,
            global_amp,
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )
        self.shaped_global = shaped_global
        self.theta = (
            physical_params.SK1_AM.theta if use_global_segment_durations else theta
        )
        self.phi = phi
        if self.theta >= 4 * np.pi:
            _LOGGER.error("ERROR: SK1 rotation angles greater than 4pi are not allowed")
            raise Exception(
                "ERROR: SK1 rotation angles greater than 4pi are not allowed"
            )

        # Construct the pulse envelope and extract the segment durations
        self.pulse_envelope = intfn.InterpFunction(
            function_type=self.physical_params.SK1_AM.envelope_type,
            start_value=0,
            stop_value=1,
            scale_params=[self.physical_params.SK1_AM.envelope_scale],
        )

        temp_amp_global = self.physical_params.SK1_AM.amp_global/1000
        # nonlinearity for old aom prior to 11/2020
        # scale_global_param = np.sqrt(-.71*temp_amp_global**2+1.77*temp_amp_global-.05)

        # nonlinearity for new aom 11/2020
        scale_global_param = -.60 * temp_amp_global ** 2 + 1.68 * temp_amp_global - .06

        Tpi_to_use = (1/scale_global_param)*(1000/self.physical_params.SK1_AM.amp_ind)*(1/(2*self.physical_params.Rabi_max))
        # unscaled global amp
        # Tpi_to_use = (
        #     1.0
        #     / (2.0 * self.physical_params.Rabi_max)
        #     * 1000 ** 2
        #     / (
        #         self.physical_params.SK1_AM.amp_global
        #         * self.physical_params.SK1_AM.amp_ind
        #     )
        # )
        self.global_ramp = global_ramp

        if use_global_segment_durations:
            self.T_rotation = self.physical_params.SK1_AM.rotation_pulse_length
            self.T_correction1 = self.physical_params.SK1_AM.correction_pulse_1_length
            self.T_correction2 = self.physical_params.SK1_AM.correction_pulse_2_length
        else:
            [
                self.T_rotation,
                self.T_correction1,
                self.T_correction2,
            ] = self.pulse_envelope.calculate_SK1_segment_lengths(
                theta=theta, Tpi=Tpi_to_use
            )
            #self.T_rotation*=(0.9965)
            self.T_rotation*=(0.992)

        _LOGGER.debug(
            "SK1 time parameters: (rot, correct1, correct2) = %s, %s, %s",
            self.T_rotation,
            self.T_correction1,
            self.T_correction2,
        )

        # Initialize the waveform scan parameters and values.  If some parameter is being scanned, initialize to the
        # default dummy scan for both individual and global, then overwrite one or both in set_scan().
        if scan_parameter == self.ScanParameter.static:
            self.ind_scan_parameter = wf.ScanParameter.static
            self.global_scan_parameter = wf.ScanParameter.static
            self.ind_scan_values = [list()] * 3
            if self.shaped_global:
                self.global_scan_values = [list()] * 5
            else:
                self.global_scan_values = [list()] * 3
        else:
            self.ind_scan_parameter = wf.ScanParameter.dummy_scan
            self.global_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind_scan_values = [[0] * self.N_scan_values] * 3
            if self.shaped_global:
                self.global_scan_values = [[0] * self.N_scan_values] * 5
            else:
                self.global_scan_values = [[0] * self.N_scan_values] * 3

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""

        if self.scan_parameter == int(self.ScanParameter.phi):
            self.global_scan_parameter = wf.ScanParameter.phase
            sv_global_rotation = self.scan_values
            sv_global_correction1 = [
                np.remainder(p - np.arccos(self.theta / (-4 * np.pi)), 2 * np.pi)
                for p in self.scan_values
            ]
            sv_global_correction2 = [
                np.remainder(p + np.arccos(self.theta / (-4 * np.pi)), 2 * np.pi)
                for p in self.scan_values
            ]
            if self.shaped_global:
                self.global_scan_values = [
                    sv_global_rotation,
                    sv_global_rotation,
                    sv_global_correction1,
                    sv_global_correction2,
                    sv_global_correction2
                ]
            else:
                self.global_scan_values = [
                    sv_global_rotation,
                    sv_global_correction1,
                    sv_global_correction2,
                ]
        elif self.scan_parameter == int(self.ScanParameter.theta):
            _LOGGER.error(
                "Error: Theta should not be scanned by the SK1 gate because it is scanned at the circuit level"
            )
            raise Exception(
                "Error: Theta should not be scanned by the SK1 gate because it is scanned at the circuit level"
            )
        elif self.scan_parameter == int(self.ScanParameter.rotation_pulse_length):
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            sv_rotation = self.scan_values
            sv_correction1 = [self.T_correction1] * self.N_scan_values
            sv_correction2 = [self.T_correction2] * self.N_scan_values
            self.ind_scan_values = [
                sum(x) for x in zip(sv_rotation, sv_correction1, sv_correction2)
            ]
            self.global_scan_values = [sv_rotation, sv_correction1, sv_correction2]
        elif self.scan_parameter == int(self.ScanParameter.correction_pulse_1_length):
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            sv_rotation = [self.T_rotation] * self.N_scan_values
            sv_correction1 = self.scan_values
            sv_correction2 = [self.T_correction2] * self.N_scan_values
            self.ind_scan_values = [
                sum(x) for x in zip(sv_rotation, sv_correction1, sv_correction2)
            ]
            self.global_scan_values = [sv_rotation, sv_correction1, sv_correction2]
        elif self.scan_parameter == int(self.ScanParameter.correction_pulse_2_length):
            self.ind_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            sv_rotation = [self.T_rotation] * self.N_scan_values
            sv_correction1 = [self.T_correction2] * self.N_scan_values
            sv_correction2 = self.scan_values
            self.ind_scan_values = [
                sum(x) for x in zip(sv_rotation, sv_correction1, sv_correction2)
            ]
            self.global_scan_values = [sv_rotation, sv_correction1, sv_correction2]
        elif self.scan_parameter == int(self.ScanParameter.Stark_shift):
            _LOGGER.error(
                "Stark shift should not be scanned by the SK1_AM gate because it should be scanned at the circuit level"
            )
            raise Exception(
                "Stark shift should not be scanned by the SK1_AM gate because it should be scanned at the circuit level"
            )
        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude):
            self.ind_scan_parameter = wf.ScanParameter.amplitude
            self.ind_scan_values = [self.scan_values]
        elif self.scan_parameter == int(self.ScanParameter.global_amplitude):
            self.global_scan_parameter = wf.ScanParameter.amplitude
            self.global_scan_values = [self.scan_values] * 3
        elif self.scan_parameter == int(self.ScanParameter.t_delay):
            self.ind_scan_parameter = wf.ScanParameter.t_delay
            self.ind_scan_values = [list(self.scan_values)]

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.  To account
        for the 3 us delay in the global AOM, we prepend an additional 3 us to the
        beginning of the individual waveform."""
        global_shape_constant = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.constant,
            start_value=1,
            stop_value=1,
            scale_params=[1.0],
        )

        global_shape_ramp_up = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.ramp,
            start_value=0,
            stop_value=1,
            scale_params=[1.0],
        )
        global_shape_ramp_down = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.ramp,
            start_value=1,
            stop_value=0,
            scale_params=[1.0],
        )

        phi_correction1 = np.remainder(
            self.phi - np.arccos(self.theta / (-4 * np.pi)), 2 * np.pi
        )
        phi_correction2 = np.remainder(
            self.phi + np.arccos(self.theta / (-4 * np.pi)), 2 * np.pi
        )
        print("Shaped global = ",self.shaped_global,"\n")
        print([self.T_rotation, self.T_correction1, self.T_correction2])
        if self.shaped_global:
            global_durations = [self.global_ramp,self.T_rotation, self.T_correction1, self.T_correction2,self.global_ramp]
            global_phases = [self.phi,self.phi, phi_correction1, phi_correction2,phi_correction2]
            # Here, we produce one shaped envelope, which is a multisegment waveform with
            # only one segment specified
            self.wav_array = [
                wp.multisegment_AM(
                    amplitude_fns=[self.pulse_envelope],
                    amplitudes=[self.ind_amp],
                    freqs=[
                        self.physical_params.f_ind - self.physical_params.SK1_AM.Stark_shift
                    ],
                    phases=[0],
                    durations=[self.T_rotation + self.T_correction1 + self.T_correction2],
                    PA_freqs=[
                        self.physical_params.f_ind - self.physical_params.SK1_AM.Stark_shift
                    ],
                    t_delay=self.physical_params.t_delay+self.global_ramp,
                    delay_PA_freq=self.physical_params.f_ind,
                    name=self.name + " ({0})".format(self.slot_array[0]),
                    scan_parameter=self.ind_scan_parameter,
                    scan_values=self.ind_scan_values,
                ),
                wp.blank(),
                wp.multisegment_AM(
                    amplitude_fns=[global_shape_ramp_up, global_shape_constant, global_shape_constant, global_shape_constant, global_shape_ramp_down],
                    amplitudes=[self.global_amp] * 5,
                    freqs=[self.physical_params.f_carrier] * 5,
                    phases=global_phases,
                    durations=global_durations,
                    # PA_freqs = global_freqs,
                    # t_delay=0.0,
                    # delay_PA_freq=global_freqs[1],
                    name=self.name + " (global)",
                    scan_parameter=self.global_scan_parameter,
                    scan_values=self.global_scan_values,
                ),
                #old multisegment global piece:
                # wp.multisegment(
                #     amplitudes=[self.global_amp] * 3,
                #     freqs=[self.physical_params.f_carrier] * 3,
                #     phases=[self.phi, phi_correction1, phi_correction2],
                #     durations=[self.T_rotation, self.T_correction1, self.T_correction2],
                #     PA_freqs=[self.physical_params.f_carrier] * 3,
                #     name=self.name + " (global)",
                #     scan_parameter=self.global_scan_parameter,
                #     scan_values=self.global_scan_values,
                # ),
            ]
        else:
            # global_durations = [self.T_rotation, self.T_correction1, self.T_correction2]
            # global_phases = [self.phi, phi_correction1, phi_correction2]
            # Here, we produce one shaped envelope, which is a multisegment waveform with
            # only one segment specified
            self.wav_array = [
                wp.multisegment_AM(
                    amplitude_fns=[self.pulse_envelope],
                    amplitudes=[self.ind_amp],
                    freqs=[
                        self.physical_params.f_ind - self.physical_params.SK1_AM.Stark_shift
                    ],
                    phases=[0],
                    durations=[self.T_rotation + self.T_correction1 + self.T_correction2],
                    PA_freqs=[
                        self.physical_params.f_ind - self.physical_params.SK1_AM.Stark_shift
                    ],
                    t_delay=self.physical_params.t_delay,
                    delay_PA_freq=self.physical_params.f_ind,
                    name=self.name + " ({0})".format(self.slot_array[0]),
                    scan_parameter=self.ind_scan_parameter,
                    scan_values=self.ind_scan_values,
                ),
                wp.blank(),
                # wp.multisegment_AM(
                #     amplitude_fns=[global_shape_ramp_up, global_shape_constant, global_shape_constant, global_shape_constant, global_shape_ramp_down],
                #     amplitudes=[self.global_amp] * 5,
                #     freqs=[self.physical_params.f_carrier] * 5,
                #     phases=global_phases,
                #     durations=global_durations,
                #     # PA_freqs = global_freqs,
                #     # t_delay=0.0,
                #     # delay_PA_freq=global_freqs[1],
                #     name=self.name + " (global)",
                #     scan_parameter=self.global_scan_parameter,
                #     scan_values=self.global_scan_values,
                # ),
                #old multisegment global piece:
                wp.multisegment(
                    amplitudes=[self.global_amp] * 3,
                    freqs=[self.physical_params.f_carrier] * 3,
                    phases=[self.phi, phi_correction1, phi_correction2],
                    durations=[self.T_rotation, self.T_correction1, self.T_correction2],
                    PA_freqs=[self.physical_params.f_carrier] * 3,
                    name=self.name + " (global)",
                    scan_parameter=self.global_scan_parameter,
                    scan_values=self.global_scan_values,
                ),
            ]


class XX(Gate):
    """This class implements an XX gate pulse, with the ability to sweep several
    parameters."""

    class ScanParameter(IntEnum):
        static = -1
        duration_adjust = 0
        Stark_shift = 1
        Stark_shift_diff = 2
        motional_freq_adjust = 3
        phi_ind1 = 4
        phi_ind2 = 5
        phi_ind_com = 6
        phi_ind_diff = 7
        phi_global = 8
        phi_motional = 9
        phi_b = 10
        phi_r = 11
        ind_amplitude_multiplier = 12
        ind_amplitude_imbalance = 13
        global_amplitude = 14
        sb_amplitude_imbalance = 15
        t_delay = 16
        N_gates = 17

    def get_tweak_value(self, value_name: str) -> typing.Any:
        return self.gate_param_tweaks[value_name] + self.global_param_tweaks[value_name]

    def __init__(
        self,
        physical_params: pp.PhysicalParams,
        gate_solution: "gate_parameters.GateSolution",
        gate_param_tweaks: "gate_parameters.GateCalibrations",
        slots: common_types.SlotPair,
        gate_sign: float = +1,
        phi_ind1: float = 0,
        phi_ind2: float = 0,
        phi_global: float = 0,
        phi_motion: float = 0,
        wait_after: int = 0,
        global_ramp: float = 10.0,
        shaped_global: bool = False,
        name: str = "XX",
        scan_parameter: ScanParameter = ScanParameter.static,
        scan_values: typing.List[float] = None,
        **gate_param_modifications
    ):
        """Create an XX gate from given parameters.

        Args:
            physical_params (pp.PhysicalParams): physical parameters that describe
                the experiment and different aspects of the experiment.
            gate_solution (gate_parameters.GateSolution): A datastructure describing
                the calculated gate solution.
            gate_param_tweaks (gate_parameters.GateCalibrations): Datastructure that
                describes tweaks to be applied to the calculated gate solution
            slots (common_types.SlotPair): the pair of slots that the gate
                should be applied to. Used for calculating waveform AND loading
                slot-specific gate tweaks
            gate_sign (float, optional): The sign (either +1 or -1) of the
                geometric phase required. Defaults to +1.
            phi_ind1 (float, optional): The sign of the individual tone applied
                to slot 1. Defaults to 0.
            phi_ind2 (float, optional): The sign of the individual tone applied
                to slot 2. Defaults to 0.
            phi_global (float, optional): The common phase of the global tone.
                Defaults to 0.
            phi_motion (float, optional): The initial phase difference between
                the blue and red sidebands on the global tone. Defaults to 0.
            wait_after (int, optional): Time to wait after the gate (us??).
                Defaults to 0.
            name (str, optional): Name of the XX gate that you're applying.
                Defaults to "XX".
            scan_parameter (ScanParameter, optional): The value that is being scanned.
                Defaults to ScanParameter.static.
            scan_values (typing.List[float], optional): The values that are
                being changed during a scan. Defaults to None.

        Kwargs:
            This accepts kwargs that are columns in :class:`.GateCalibrations`.
            This means that any individual parameter can be tweaked for an individual
            gate without changing the larger data structure. Note that this OVERRIDES
            the corresponding "tweak" from the ``gate_param_tweaks`` argument.

        Raises:
            TypeError: If gate_solution[slot]["XX_gate_type"] has an
                unrecognized gate type (i.e. not in
                :class:`common_types.XXModulationType`)
        """
        assert gate_solution.solutions_hash == gate_param_tweaks.solutions_hash
        # HACK: Freeze gate parameters used at instantiation time.
        # Useful for sweeping values, where you change it in the top data structure,
        # then instantiate different copies of the same item
        if isinstance(slots, list):
            slots = tuple(slots)
        _LOGGER.debug("Adding XX Gate on slots: %s", slots)
        self.gate_param_tweaks = gate_param_tweaks.loc[slots, :].copy()
        self.global_param_tweaks = gate_param_tweaks.loc[
            gate_param_tweaks.GLOBAL_CALIBRATION_SLOT, :
        ].copy()
        for kw, value in gate_param_modifications.items():
            if kw in set(self.gate_param_tweaks.index):
                self.gate_param_tweaks[kw] = value
            else:
                raise ValueError("Modification `{}` not recognized".format(kw))

        twoQ_gate = 1
        slot_array = [slots[0], slots[1], 0]
        ind_amp = 0

        super().__init__(
            common_types.GateType.XX,
            physical_params,
            slot_array,
            ind_amp,
            self.get_tweak_value("global_amplitude"),
            twoQ_gate,
            name,
            int(scan_parameter),
            scan_values,
            wait_after,
        )

        self.phi_ind1 = phi_ind1
        self.phi_ind2 = phi_ind2
        self.phi_global = phi_global
        self.phi_motion = phi_motion

        self.shaped_global = shaped_global
        self.global_ramp = global_ramp

        # Read the solution parameters
        slot_gate_soln_raw = gate_solution.loc[slots, :]
        # Fast lookup to see if pandas Series for gate solution is empty
        assert (
            len(slot_gate_soln_raw.index) != 0
        ), "Gate Solution does not exist for given slot"
        self.slot_gate_solution = common_types.XXGateParams(*slot_gate_soln_raw)
        gate_rabi_amplitudes = self.slot_gate_solution.XX_amplitudes

        # The gate solutions define the gate amplitude of a segment as the rabi frequency of EACH sideband frequency
        # when brought into resonance. Whereas the RFCompiler defines the gate amplitude of a segment as the rabi
        # frequency when BOTH sidebands are brought into resonance. We multiply the gate amplitudes by 2 at this
        # interface to resolve the discrepancy.
        gate_rabi_amplitudes = (np.array(gate_rabi_amplitudes) * 2).tolist()

        # print(self.slot_gate_solution.XX_gate_type)
        # print("Gate params: " + str([self.slot_gate_solution.XX_detuning, sign_sol, self.slot_gate_solution.XX_duration_us]))
        # print(Rabi_array)
        self.N_segments = len(self.slot_gate_solution.XX_amplitudes)
        # print("XX Rabi frequencies from file: "+str(Rabi_array))

        gate_mod_type = self.slot_gate_solution.XX_gate_type
        if gate_mod_type == common_types.XXModulationType.AM_segmented:
            # for segmented, Rabi_array is array of amplitudes
            self.amplitudes = cal.Rabis_to_amplitudes(
                gate_rabi_amplitudes, self.physical_params.Rabi_max
            )
        elif gate_mod_type == common_types.XXModulationType.AM_interp:
            # for interpolated, Rabi_array is an array of start/stop points
            self.amplitudes = [
                cal.Rabis_to_amplitudes(ra, self.physical_params.Rabi_max)
                for ra in gate_rabi_amplitudes
            ]
            # copies of each other, used independently later on.
            self.amplitude_fns_1 = [
                intfn.InterpFunction(
                    function_type=intfn.InterpFunction.FunctionType.linear,
                    start_value=a[0],
                    stop_value=a[1],
                )
                for a in self.amplitudes
            ]
            self.amplitude_fns_2 = [
                intfn.InterpFunction(
                    function_type=intfn.InterpFunction.FunctionType.linear,
                    start_value=a[0],
                    stop_value=a[1],
                )
                for a in self.amplitudes
            ]
        else:
            raise TypeError(
                "XX gate has not been coded to accept the modulation type {}".format(
                    self.slot_gate_solution.XX_gate_type.name
                )
            )
        # print(self.amplitudes)
        # print("XX amplitudes from file: " + str(self.amplitudes))

        # The sign of the geometric phase is the product of the parity of the two ions'
        # motional mode participation, as given by the gate solution, and the relative
        # phase between the two individual RF drives
        self.invert_ind2 = gate_sign * self.slot_gate_solution.XX_sign

        # Initialize the waveform scan parameters and values.  If some parameter is being scanned, initialize to the
        # default dummy scan for both individual and global, then overwrite one or both in set_scan().
        if scan_parameter == self.ScanParameter.static:
            self.ind1_scan_parameter = wf.ScanParameter.static
            self.ind2_scan_parameter = wf.ScanParameter.static
            self.global_scan_parameter = wf.ScanParameter.static
            self.ind1_scan_values = [list()] * self.N_segments
            self.ind2_scan_values = [list()] * self.N_segments
            if self.shaped_global:
                self.global_scan_values = [list()]*3
            else:
                self.global_scan_values = list()
        else:
            self.ind1_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind2_scan_parameter = wf.ScanParameter.dummy_scan
            self.global_scan_parameter = wf.ScanParameter.dummy_scan
            self.ind1_scan_values = [[0] * self.N_scan_values] * self.N_segments
            self.ind2_scan_values = [[0] * self.N_scan_values] * self.N_segments
            if self.shaped_global:
                self.global_scan_values = [[0] * self.N_scan_values]*3
            else:
                self.global_scan_values = [0] * self.N_scan_values

    def set_scan(self):
        """This function translates the gate-specific scan parameter (e.g., sideband
        imbalance for XX) and values into the more generic scan parameter (e.g.,
        amplitude) and values that are passed to the waveforms."""

        gate_mod_type = self.slot_gate_solution.XX_gate_type

        if self.scan_parameter == int(self.ScanParameter.duration_adjust):
            self.ind1_scan_parameter = wf.ScanParameter.seg_duration
            self.ind2_scan_parameter = wf.ScanParameter.seg_duration
            self.global_scan_parameter = wf.ScanParameter.seg_duration
            self.ind1_scan_values = [
                [
                    (self.slot_gate_solution.XX_duration_us + da) / self.N_segments
                    for da in self.scan_values
                ]
            ] * self.N_segments
            self.ind2_scan_values = self.ind1_scan_values
            if self.shaped_global:
                self.global_scan_values = [
                    [self.global_ramp]*len(self.scan_values),
                    [self.slot_gate_solution.XX_duration_us + da for da in self.scan_values],
                    [self.global_ramp] * len(self.scan_values),
                ]
            else:
                self.global_scan_values = [
                    self.slot_gate_solution.XX_duration_us + da for da in self.scan_values
                ]
        elif self.scan_parameter == int(self.ScanParameter.Stark_shift):
            raise ValueError(
                "Stark shift should be scanned at the circuit level, not in XX gate."
            )
        elif self.scan_parameter == int(self.ScanParameter.Stark_shift_diff):
            stark_shift = self.get_tweak_value("stark_shift")
            self.ind1_scan_parameter = wf.ScanParameter.frequency
            self.ind2_scan_parameter = wf.ScanParameter.frequency
            self.ind1_scan_values = [
                [
                    self.physical_params.f_ind - (stark_shift + SSD)
                    for SSD in self.scan_values
                ]
            ] * self.N_segments
            self.ind2_scan_values = [
                [
                    self.physical_params.f_ind - (stark_shift - SSD)
                    for SSD in self.scan_values
                ]
            ] * self.N_segments
        elif self.scan_parameter == int(self.ScanParameter.motional_freq_adjust):
            detuning = self.slot_gate_solution.XX_detuning
            self.global_scan_parameter = wf.ScanParameter.frequency
            if self.shaped_global:
                self.global_scan_values = [
                    [self.physical_params.f_carrier]*len(self.scan_values),
                    [[self.physical_params.f_carrier + (detuning + d),self.physical_params.f_carrier - (detuning + d)] for d in self.scan_values],
                    [self.physical_params.f_carrier]*len(self.scan_values)
                ]
            else:
                self.global_scan_values = [
                    [self.physical_params.f_carrier + (detuning + d), self.physical_params.f_carrier - (detuning + d)]
                     for d in self.scan_values
                ]
        elif self.scan_parameter == int(self.ScanParameter.phi_ind1):
            self.ind1_scan_parameter = wf.ScanParameter.phase
            self.ind1_scan_values = [self.scan_values] * self.N_segments
        elif self.scan_parameter == int(self.ScanParameter.phi_ind2):
            self.ind2_scan_parameter = wf.ScanParameter.phase
            self.ind2_scan_values = [self.scan_values] * self.N_segments
        elif self.scan_parameter == int(self.ScanParameter.phi_ind_com):
            self.ind1_scan_parameter = wf.ScanParameter.phase
            self.ind2_scan_parameter = self.ind1_scan_parameter
            self.ind1_scan_values = [self.scan_values] * self.N_segments
            self.ind2_scan_values = self.ind1_scan_values
        elif self.scan_parameter == int(self.ScanParameter.phi_ind_com):
            self.ind1_scan_parameter = wf.ScanParameter.phase
            self.ind2_scan_parameter = self.ind1_scan_parameter
            self.ind1_scan_values = [
                [(self.phi_ind1 + self.phi_ind2 + sv) / 2 for sv in self.scan_values]
            ] * self.N_segments
            self.ind1_scan_values = [
                [(self.phi_ind1 + self.phi_ind2 - sv) / 2 for sv in self.scan_values]
            ] * self.N_segments
        elif self.scan_parameter == int(self.ScanParameter.phi_global):
            self.global_scan_parameter = wf.ScanParameter.phase
            if self.shaped_global:
                self.global_scan_values = [
                    [self.phi_global]*len(self.scan_values),
                    [[phg + self.phi_motion / 2, phg - self.phi_motion / 2]
                    for phg in self.scan_values],
                    [self.phi_global] * len(self.scan_values)
                ]
            else:
                self.global_scan_values = [
                    [phg + self.phi_motion / 2, phg - self.phi_motion / 2]
                     for phg in self.scan_values
                ]
        elif self.scan_parameter == int(self.ScanParameter.phi_motional):
            self.global_scan_parameter = wf.ScanParameter.phase
            self.global_scan_values = [
                [self.phi_global + phm / 2, self.phi_global - phm / 2]
                for phm in self.scan_values
            ]
        elif self.scan_parameter == int(self.ScanParameter.phi_b):
            self.global_scan_parameter = wf.ScanParameter.phase
            self.global_scan_values = [
                [phb, self.phi_global - self.phi_motion / 2] for phb in self.scan_values
            ]
        elif self.scan_parameter == int(self.ScanParameter.phi_r):
            self.global_scan_parameter = wf.ScanParameter.phase
            self.global_scan_values = [
                [self.phi_global + self.phi_motion / 2, phr] for phr in self.scan_values
            ]
        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude_multiplier):
            self.ind1_scan_parameter = wf.ScanParameter.amplitude
            self.ind2_scan_parameter = self.ind1_scan_parameter
            if gate_mod_type == common_types.XXModulationType.AM_segmented:
                ind_imbalance = self.get_tweak_value("individual_amplitude_imbalance")
                self.ind1_scan_values = [
                    [sv * (1 + ind_imbalance) * a for sv in self.scan_values]
                    for a in self.amplitudes
                ]
                self.ind2_scan_values = [
                    [
                        sv * (1 - ind_imbalance) * self.invert_ind2 * a
                        for sv in self.scan_values
                    ]
                    for a in self.amplitudes
                ]
            elif gate_mod_type == common_types.XXModulationType.AM_interp:
                ind_imbalance = self.get_tweak_value("individual_amplitude_imbalance")
                self.ind1_scan_values = [
                    [sv * (1 + ind_imbalance) for sv in self.scan_values]
                ] * self.N_segments
                self.ind2_scan_values = [
                    [
                        sv * (1 - ind_imbalance) * self.invert_ind2
                        for sv in self.scan_values
                    ]
                ] * self.N_segments
        elif self.scan_parameter == int(self.ScanParameter.ind_amplitude_imbalance):
            self.ind1_scan_parameter = wf.ScanParameter.amplitude
            self.ind2_scan_parameter = self.ind1_scan_parameter
            if gate_mod_type == common_types.XXModulationType.AM_segmented:
                ind_amp_multiplier = self.get_tweak_value(
                    "individual_amplitude_multiplier"
                )
                self.ind1_scan_values = [
                    [ind_amp_multiplier * (1 + sv) * a for sv in self.scan_values]
                    for a in self.amplitudes
                ]
                self.ind2_scan_values = [
                    [
                        ind_amp_multiplier * (1 - sv) * self.invert_ind2 * a
                        for sv in self.scan_values
                    ]
                    for a in self.amplitudes
                ]
            elif gate_mod_type == common_types.XXModulationType.AM_interp:
                ind_amp_multiplier = self.get_tweak_value(
                    "individual_amplitude_multiplier"
                )
                self.ind1_scan_values = [
                    [ind_amp_multiplier * (1 + sv) for sv in self.scan_values]
                ] * self.N_segments
                self.ind2_scan_values = [
                    [
                        ind_amp_multiplier * (1 - sv) * self.invert_ind2
                        for sv in self.scan_values
                    ]
                ] * self.N_segments
            print(self.ind2_scan_values)
        elif self.scan_parameter == int(self.ScanParameter.global_amplitude):
            sideband_imbalance = self.get_tweak_value("sideband_amplitude_imbalance")
            self.global_scan_parameter = wf.ScanParameter.amplitude
            if self.shaped_global:
                self.global_scan_values = [
                        [self.global_amp]*len(self.scan_values),
                        [[(1 + sideband_imbalance) / 2 * ga, (1 - sideband_imbalance) / 2 * ga]
                            for ga in self.scan_values],
                        [self.global_amp] * len(self.scan_values)
                ]
            else:
                self.global_scan_values = [
                    [(1 + sideband_imbalance) / 2 * ga, (1 - sideband_imbalance) / 2 * ga]
                        for ga in self.scan_values
                ]
        elif self.scan_parameter == int(self.ScanParameter.sb_amplitude_imbalance):
            self.global_scan_parameter = wf.ScanParameter.amplitude
            if self.shaped_global:
                self.global_scan_values = [
                        [self.global_amp]*len(self.scan_values),
                        [[(1 + si) / 2 * self.global_amp, (1 - si) / 2 * self.global_amp]
                            for si in self.scan_values],
                        [self.global_amp] * len(self.scan_values)
                ]
            self.global_scan_values = [
                [(1 + si) / 2 * self.global_amp, (1 - si) / 2 * self.global_amp]
                for si in self.scan_values
            ]
        elif self.scan_parameter == int(self.ScanParameter.t_delay):
            self.ind1_scan_parameter = wf.ScanParameter.t_delay
            self.ind2_scan_parameter = self.ind1_scan_parameter
            self.ind1_scan_values = [self.scan_values] * self.N_segments
            self.ind2_scan_values = self.ind1_scan_values
        elif self.scan_parameter == int(self.ScanParameter.N_gates):
            raise RuntimeError(
                "N_gates should be scanned at circuit level, not the XX gate level"
            )

    def compile_waveform(self):
        """We construct waveforms for the individual and global channels.

        To account for the 3 us delay in the global AOM, we prepend an
        additional 3 us to the beginning of the individual waveform.
        """
        gate_mod_type = self.slot_gate_solution.XX_gate_type
        ind1_amps = []
        ind2_amps = []
        if gate_mod_type == common_types.XXModulationType.AM_segmented:
            ind_amp_multiplier = self.get_tweak_value("individual_amplitude_multiplier")
            ind_imbalance = self.get_tweak_value("individual_amplitude_imbalance")
            ind1_amps = [
                ind_amp_multiplier * (1 + ind_imbalance) * a for a in self.amplitudes
            ]
            ind2_amps = [
                ind_amp_multiplier * (1 - ind_imbalance) * self.invert_ind2 * a
                for a in self.amplitudes
            ]
        elif gate_mod_type == common_types.XXModulationType.AM_interp:
            ind_amp_multiplier = self.get_tweak_value("individual_amplitude_multiplier")
            ind_imbalance = self.get_tweak_value("individual_amplitude_imbalance")
            ind1_amps = [ind_amp_multiplier * (1 + ind_imbalance)] * self.N_segments
            ind2_amps = [
                ind_amp_multiplier * (1 - ind_imbalance) * self.invert_ind2
            ] * self.N_segments

        if not (cal.check_amplitudes(ind1_amps) and cal.check_amplitudes(ind2_amps)):
            raise RuntimeError(
                "Individual channel amplitudes > max amplitude detected during XX gate"
            )

        stark_shift = self.get_tweak_value("stark_shift")
        stark_shift_diff = self.get_tweak_value("stark_shift_differential")
        ind1_freq = self.physical_params.f_ind - (stark_shift + stark_shift_diff/2)
        ind2_freq = self.physical_params.f_ind - (stark_shift - stark_shift_diff/2)

        seg_duration = (
            self.slot_gate_solution.XX_duration_us
            + self.get_tweak_value("XX_duration_us")
        ) / self.N_segments

        sideband_imbalance = self.get_tweak_value("sideband_amplitude_imbalance")
        mot_freq_adj = self.get_tweak_value("motional_frequency_adjust")
        detuning = self.slot_gate_solution.XX_detuning
        global_b_amp = (1 + sideband_imbalance) / 2 * self.global_amp
        global_r_amp = (1 - sideband_imbalance) / 2 * self.global_amp
        global_b_freq = self.physical_params.f_carrier + (detuning + mot_freq_adj)
        global_r_freq = self.physical_params.f_carrier - (detuning + mot_freq_adj)
        global_b_phi = self.phi_global + self.phi_motion / 2
        global_r_phi = self.phi_global - self.phi_motion / 2

        global_shape_constant = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.constant,
            start_value=1,
            stop_value=1,
            scale_params=[1.0],
        )

        global_shape_ramp_up = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.ramp,
            start_value=0,
            stop_value=1,
            scale_params=[1.0],
        )
        global_shape_ramp_down = intfn.InterpFunction(
            function_type=intfn.InterpFunction.FunctionType.ramp,
            start_value=1,
            stop_value=0,
            scale_params=[1.0],
        )
        global_durations = [self.global_ramp, self.slot_gate_solution.XX_duration_us + self.get_tweak_value("XX_duration_us"), self.global_ramp]
        global_phases = [global_r_phi,[global_b_phi, global_r_phi],global_r_phi]
        global_amplitudes = [self.global_amp,[global_b_amp, global_r_amp],self.global_amp]
        global_freqs = [self.physical_params.f_carrier,[global_b_freq, global_r_freq],self.physical_params.f_carrier]
        global_PA_freqs = [self.physical_params.f_carrier]*3

        if gate_mod_type == common_types.XXModulationType.AM_segmented:
            if self.shaped_global:
                self.wav_array = [
                    wp.multisegment(
                        amplitudes=ind1_amps,
                        freqs=[ind1_freq] * self.N_segments,
                        phases=[self.phi_ind1] * self.N_segments,
                        durations=[seg_duration] * self.N_segments,
                        PA_freqs=[ind1_freq] * self.N_segments,
                        t_delay=self.physical_params.t_delay+self.global_ramp,
                        delay_PA_freq=self.physical_params.f_ind,
                        name=self.name + " ({0})".format(self.slot_array[0]),
                        scan_parameter=self.ind1_scan_parameter,
                        scan_values=self.ind1_scan_values,
                    ),
                    wp.multisegment(
                        amplitudes=ind2_amps,
                        freqs=[ind2_freq] * self.N_segments,
                        phases=[self.phi_ind2] * self.N_segments,
                        durations=[seg_duration] * self.N_segments,
                        PA_freqs=[ind2_freq] * self.N_segments,
                        t_delay=self.physical_params.t_delay+self.global_ramp,
                        delay_PA_freq=self.physical_params.f_ind,
                        name=self.name + " ({0})".format(self.slot_array[1]),
                        scan_parameter=self.ind2_scan_parameter,
                        scan_values=self.ind2_scan_values,
                    ),
                    # On Global: multitone, single segment
                    # wp.multitone(
                    #     amplitudes=[global_b_amp, global_r_amp],
                    #     freqs=[global_b_freq, global_r_freq],
                    #     phases=[global_b_phi, global_r_phi],
                    #     duration=self.slot_gate_solution.XX_duration_us
                    #              + self.get_tweak_value("XX_duration_us"),
                    #     name=self.name + " (global)",
                    #     scan_parameter=self.global_scan_parameter,
                    #     scan_values=self.global_scan_values,
                    # ),
                    wp.multisegment_AM(
                        amplitude_fns=[global_shape_ramp_up, global_shape_constant, global_shape_ramp_down],
                        amplitudes=global_amplitudes,
                        freqs=global_freqs,
                        phases=global_phases,
                        durations=global_durations,
                        PA_freqs = global_PA_freqs,
                        # t_delay=0.0,
                        # delay_PA_freq=global_freqs[1],
                        name=self.name + " (global)",
                        scan_parameter=self.global_scan_parameter,
                        scan_values=self.global_scan_values,
                    ),
                ]
            else:
                self.wav_array = [
                    wp.multisegment(
                        amplitudes=ind1_amps,
                        freqs=[ind1_freq] * self.N_segments,
                        phases=[self.phi_ind1] * self.N_segments,
                        durations=[seg_duration] * self.N_segments,
                        PA_freqs=[ind1_freq] * self.N_segments,
                        t_delay=self.physical_params.t_delay,
                        delay_PA_freq=self.physical_params.f_ind,
                        name=self.name + " ({0})".format(self.slot_array[0]),
                        scan_parameter=self.ind1_scan_parameter,
                        scan_values=self.ind1_scan_values,
                    ),
                    wp.multisegment(
                        amplitudes=ind2_amps,
                        freqs=[ind2_freq] * self.N_segments,
                        phases=[self.phi_ind2] * self.N_segments,
                        durations=[seg_duration] * self.N_segments,
                        PA_freqs=[ind2_freq] * self.N_segments,
                        t_delay=self.physical_params.t_delay,
                        delay_PA_freq=self.physical_params.f_ind,
                        name=self.name + " ({0})".format(self.slot_array[1]),
                        scan_parameter=self.ind2_scan_parameter,
                        scan_values=self.ind2_scan_values,
                    ),
                    # On Global: multitone, single segment
                    wp.multitone(
                        amplitudes=[global_b_amp, global_r_amp],
                        freqs=[global_b_freq, global_r_freq],
                        phases=[global_b_phi, global_r_phi],
                        duration=self.slot_gate_solution.XX_duration_us
                        + self.get_tweak_value("XX_duration_us"),
                        name=self.name + " (global)",
                        scan_parameter=self.global_scan_parameter,
                        scan_values=self.global_scan_values,
                    ),
                ]
        elif gate_mod_type == common_types.XXModulationType.AM_interp:
            if self.shaped_global:
                self.wav_array = [
                    wp.multisegment_AM(
                        amplitude_fns=self.amplitude_fns_1,
                        amplitudes=ind1_amps,
                        freqs=[ind1_freq] * self.N_segments,
                        phases=[self.phi_ind1] * self.N_segments,
                        durations=[seg_duration] * self.N_segments,
                        PA_freqs=[ind1_freq] * self.N_segments,
                        t_delay=self.physical_params.t_delay+self.global_ramp,
                        delay_PA_freq=self.physical_params.f_ind,
                        name=self.name + " ({0})".format(self.slot_array[0]),
                        scan_parameter=self.ind1_scan_parameter,
                        scan_values=self.ind1_scan_values,
                    ),
                    wp.multisegment_AM(
                        amplitude_fns=self.amplitude_fns_2,
                        amplitudes=ind2_amps,
                        freqs=[ind2_freq] * self.N_segments,
                        phases=[self.phi_ind2] * self.N_segments,
                        durations=[seg_duration] * self.N_segments,
                        PA_freqs=[ind2_freq] * self.N_segments,
                        t_delay=self.physical_params.t_delay+self.global_ramp,
                        delay_PA_freq=self.physical_params.f_ind,
                        name=self.name + " ({0})".format(self.slot_array[1]),
                        scan_parameter=self.ind2_scan_parameter,
                        scan_values=self.ind2_scan_values,
                    ),

                    # wp.multitone(
                    #     amplitudes=[global_b_amp, global_r_amp],
                    #     freqs=[global_b_freq, global_r_freq],
                    #     phases=[global_b_phi, global_r_phi],
                    #     duration=self.slot_gate_solution.XX_duration_us
                    #              + self.get_tweak_value("XX_duration_us"),
                    #     name=self.name + " (global)",
                    #     scan_parameter=self.global_scan_parameter,
                    #     scan_values=self.global_scan_values,
                    # ),
                    wp.multisegment_AM(
                        amplitude_fns=[global_shape_ramp_up, global_shape_constant, global_shape_ramp_down],
                        amplitudes=global_amplitudes,
                        freqs=global_freqs,
                        phases=global_phases,
                        durations=global_durations,
                        PA_freqs = global_PA_freqs,
                        # t_delay=0.0,
                        # delay_PA_freq=global_freqs[1],
                        name=self.name + " (global)",
                        scan_parameter=self.global_scan_parameter,
                        scan_values=self.global_scan_values,
                    ),
                ]
            else:
                self.wav_array = [
                    wp.multisegment_AM(
                        amplitude_fns=self.amplitude_fns_1,
                        amplitudes=ind1_amps,
                        freqs=[ind1_freq] * self.N_segments,
                        phases=[self.phi_ind1] * self.N_segments,
                        durations=[seg_duration] * self.N_segments,
                        PA_freqs=[ind1_freq] * self.N_segments,
                        t_delay=self.physical_params.t_delay,
                        delay_PA_freq=self.physical_params.f_ind,
                        name=self.name + " ({0})".format(self.slot_array[0]),
                        scan_parameter=self.ind1_scan_parameter,
                        scan_values=self.ind1_scan_values,
                    ),
                    wp.multisegment_AM(
                        amplitude_fns=self.amplitude_fns_2,
                        amplitudes=ind2_amps,
                        freqs=[ind2_freq] * self.N_segments,
                        phases=[self.phi_ind2] * self.N_segments,
                        durations=[seg_duration] * self.N_segments,
                        PA_freqs=[ind2_freq] * self.N_segments,
                        t_delay=self.physical_params.t_delay,
                        delay_PA_freq=self.physical_params.f_ind,
                        name=self.name + " ({0})".format(self.slot_array[1]),
                        scan_parameter=self.ind2_scan_parameter,
                        scan_values=self.ind2_scan_values,
                    ),
                    wp.multitone(
                        amplitudes=[global_b_amp, global_r_amp],
                        freqs=[global_b_freq, global_r_freq],
                        phases=[global_b_phi, global_r_phi],
                        duration=self.slot_gate_solution.XX_duration_us
                        + self.get_tweak_value("XX_duration_us"),
                        name=self.name + " (global)",
                        scan_parameter=self.global_scan_parameter,
                        scan_values=self.global_scan_values,
                    ),
                ]

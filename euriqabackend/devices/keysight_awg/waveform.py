import logging
import math
import os
import struct
import typing
from enum import Enum

import numpy as np

from euriqabackend.devices.keysight_awg import AOM_calibration as cal
from euriqabackend.devices.keysight_awg import interpolation_functions as intfn

_LOGGER = logging.getLogger(__name__)

_WRITE_TO_BINARY = True
_DELTA_T = 10 ** -3


def coerce_10ns(exact_time: float):
    """Round to the nearest 10^-15 to avoid machine precision error, then round up to
    nearest 10 ns.

    Args:
        exact_time: the exact time to be rounded up
    """
    return 0.01 * math.ceil(round(exact_time, 15) / 0.01)


def coerce_delta_t(exact_time: float):
    """Round to the nearest 10^-15 to avoid machine precision error, then round up to
    nearest AWG generation period.

    Args:
        exact_time: the exact time to be rounded up
    """
    return _DELTA_T * math.ceil(round(exact_time, 15) / _DELTA_T)


class ScanParameter(Enum):
    static = -1
    seg_duration = 0
    amplitude = 1
    frequency = 2
    phase = 3
    PA_prefactor = 4
    t_delay = (
        5
    )  # This is supported only at the waveform_prototype level.  No waveform should be set to sweep t_delay.
    dummy_scan = 6


def filewrite(filename: str, data, waveform_dir: str):

    if _WRITE_TO_BINARY:
        with open(waveform_dir + "\\" + filename + ".awg", "wb") as bin_file:
            bin_file.write(struct.pack(">" + "f" * len(data), *data))
    else:
        header = (
            "waveformName,"
            + filename
            + "\nwaveformPoints,"
            + str(len(data))
            + "\nwaveformType,WAVE_ANALOG_16"
        )
        np.savetxt(
            waveform_dir + "\\" + filename + ".csv",
            data,
            fmt="%f",
            header=header,
            comments="",
        )

    _LOGGER.debug("Wrote %i points to %s.csv", len(data), filename)


class Segment:
    def __init__(
        self,
        duration: float,
        amplitudes: typing.List[float],
        freqs: typing.List[float],
        phases: typing.List[float],
        PA_prefactors: typing.List[float],
        PA_freq: float,
        scan_values: typing.Union[
            typing.List[float], typing.List[typing.List[float]]
        ] = None,
    ):
        """A segment encodes the function A1 Sin( 2π f1 t + φ1 + s1 φPA ) + Ai,2 Sin( …
        ) + …, which can have many components with independent amplitudes, frequencies,
        and phases, for a given evaluated for some duration ∆ti.

        Args:
            duration: the duration of this segment in us
            amplitudes: the amplitudes of the different frequency components that
                comprise this segment
            freqs: the frequencies in MHz of the different frequency components that
                comprise this segment
            phases: the phases of the different frequency components that
                comprise this segment
            PA_prefactors: the prefactor with which the overall phase-advance
                phase is added into each component
            PA_freq: phase-advance frequency, at which the phase of the ion
                advances during this segment
            scan_values: the values through which the specified parameter
                (phase, frequency, etc.) are scanned. The first dimension
                corresponds to different scan points, and the next dimension
                corresponds to different tones applied during the segment.
        """
        if scan_values is None:
            scan_values = list()
        self.duration = coerce_delta_t(duration)
        self.amplitudes = amplitudes
        self.freqs = freqs
        self.phases = phases
        self.PA_prefactors = PA_prefactors
        self.PA_freq = PA_freq
        self.scan_values = scan_values


        if not cal.check_amplitudes([sum([abs(a) for a in self.amplitudes])]):
            _LOGGER.error(
                "Sum of amplitudes during one segment is greater than max amplitude"
            )
            raise ValueError(
                "Sum of amplitudes during one segment is greater than max amplitude"
            )

    def evaluate(
        self, t_init: float, PA_phase: float, calibration: cal.Calibration, slot: int
    ):
        """This function evaluates this segment and returns the array of analog values.

        Args:
            t_init: The time at the beginning of this segment
            PA_phase: The PA phase at the beginning of this segment
            calibration: The Calibration object that will be used to calibrate
                the waveform output
            slot: The physical slot to which this segment will be routed
        """
        t_points = t_init + np.arange(0, self.duration, _DELTA_T)

        result = np.zeros(len(t_points))

        max_amp = 0
        for i in range(len(self.freqs)):
            max_amp += self.amplitudes[i]

        if not cal.check_amplitudes([max_amp]):
            _LOGGER.error("Channel amplitudes greater than max amplitude detected")
            raise Exception(
                "ERROR: Channel amplitudes greater than max amplitude detected"
            )

        for i in range(0, len(self.freqs)):
            result += calibration.calibrate(self.amplitudes[i], slot) * np.sin(
                2 * np.pi * self.freqs[i] * t_points
                + self.phases[i]
                + self.PA_prefactors[i] * PA_phase
            )

        return result

    def evaluate_scan(
        self,
        t_init: float,
        PA_phase: float,
        calibration: cal.Calibration,
        slot: int,
        scan_parameter: ScanParameter,
        index: int,
    ):
        """This function evaluates this segment, which has a parameter to scan, and
        returns the array of analog values.

        Args:
            t_init: The time at the beginning of this segment
            PA_phase: The PA phase at the beginning of this segment
            calibration: The Calibration object that will be used to calibrate
                the waveform output
            slot: The physical slot to which this segment will be routed
            scan_parameter: The parameter that is being scanned in this waveform
            index: The specific iteration of the scan parameter that is
                currently being evaluated
        """

        # First, set whichever parameter is being scanned
        if (
            scan_parameter == ScanParameter.seg_duration
            or scan_parameter == ScanParameter.t_delay
        ):
            self.duration = coerce_delta_t(self.scan_values[index])
        elif scan_parameter == ScanParameter.amplitude:
            self.amplitudes = self.scan_values[index]
        elif scan_parameter == ScanParameter.frequency:
            self.freqs = self.scan_values[index]
        elif scan_parameter == ScanParameter.phase:
            self.phases = self.scan_values[index]
        elif scan_parameter == ScanParameter.PA_prefactor:
            self.PA_prefactors = self.scan_values[index]

        return self.evaluate(t_init, PA_phase, calibration, slot)


class SegmentAM(Segment):
    def __init__(
        self,
        duration: float,
        amplitude_fns: typing.List[intfn.InterpFunction],
        amplitudes: typing.List[float],
        freqs: typing.List[float],
        phases: typing.List[float],
        PA_prefactors: typing.List[float],
        PA_freq: float,
        scan_values: typing.Union[
            typing.List[float], typing.List[typing.List[float]]
        ] = None,
    ):
        """A segment encodes the function A1 Sin( 2π f1 t + φ1 + s1 φPA ) + Ai,2 Sin( …
        ) + …, which can have many components with independent amplitudes, frequencies,
        and phases, for a given evaluated for some duration ∆ti.

        Args:
            duration: the duration of this segment in us
            amplitude_fns: the amplitude functions of the different frequency
                components that comprise this segment
            amplitudes: the amplitudes of the different frequency components
                that comprise this segment
            freqs: the frequencies in MHz of the different frequency components
                that comprise this segment
            phases: the phases of the different frequency components that
                comprise this segment
            PA_prefactors: the prefactor with which the overall phase-advance
                phase is added into each component
            PA_freq: phase-advance frequency, at which the phase of the ion
                advances during this segment
            scan_values: the values through which the specified parameter
                (phase, frequency, etc.) are scanned.  The first dimension
                corresponds to different scan points, and the next dimension
                corresponds to different tonesapplied during the segment.
        """

        self.amplitude_fns = amplitude_fns

        super().__init__(
            duration=duration,
            amplitudes=amplitudes,
            freqs=freqs,
            phases=phases,
            PA_prefactors=PA_prefactors,
            PA_freq=PA_freq,
            scan_values=scan_values,
        )

    def evaluate(
        self, t_init: float, PA_phase: float, calibration: cal.Calibration, slot: int
    ):
        """This function evaluates this segment and returns the array of analog values.

        Args:
            t_init: The time at the beginning of this segment
            PA_phase: The PA phase at the beginning of this segment
            calibration: The Calibration object that will be used to calibrate
                the waveform output
            slot: The physical slot to which this segment will be routed
        """
        t_points = t_init + np.arange(0, self.duration, _DELTA_T)

        result = np.zeros(len(t_points))

        max_amp = 0
        for i in range(len(self.freqs)):
            self.amplitude_fns[i].rescale(self.amplitudes[i])
            max_amp += self.amplitude_fns[i].abs_max()

        if not cal.check_amplitudes([max_amp]):
            _LOGGER.error(
                "ERROR: Channel amplitudes greater than max amplitude detected"
            )
            raise Exception(
                "ERROR: Channel amplitudes greater than max amplitude detected"
            )

        self.freqs = np.array(self.freqs).flatten()
        for i in range(len(self.freqs)):
            result += calibration.calibrate_array(
                self.amplitude_fns[i].evaluate(len(t_points)), slot
            ) * np.sin(
                2 * np.pi * self.freqs[i] * t_points
                + self.phases[i]
                + self.PA_prefactors[i] * PA_phase
            )

        # print('values for waveform '+str(slot))
        # print(result)
        return result

    def evaluate_scan(
        self,
        t_init: float,
        PA_phase: float,
        calibration: cal.Calibration,
        slot: int,
        scan_parameter: ScanParameter,
        index: int,
    ):
        """This function evaluates this segment, which has a parameter to scan, and
        returns the array of analog values.

        Args:
            t_init: The time at the beginning of this segment
            PA_phase: The PA phase at the beginning of this segment
            calibration: The Calibration object that will be used to calibrate
                the waveform output
            slot: The physical slot to which this segment will be routed
            scan_parameter: The parameter that is being scanned in this waveform
            index: The specific iteration of the scan parameter that is
                currently being evaluated
        """

        # First, set whichever parameter is being scanned
        if (
            scan_parameter == ScanParameter.seg_duration
            or scan_parameter == ScanParameter.t_delay
        ):
            self.duration = coerce_delta_t(self.scan_values[index])
        elif scan_parameter == ScanParameter.amplitude:
            self.amplitudes = self.scan_values[index]
        elif scan_parameter == ScanParameter.frequency:
            self.freqs = self.scan_values[index]
        elif scan_parameter == ScanParameter.phase:
            self.phases = self.scan_values[index]
        elif scan_parameter == ScanParameter.PA_prefactor:
            self.PA_prefactors = self.scan_values[index]

        return self.evaluate(t_init, PA_phase, calibration, slot)


class Waveform:
    def __init__(self, name: str, scan_parameter: ScanParameter = ScanParameter.static):
        """Define a waveform: a series of segments that define a periodic function
        to be output on an AWG channel.

        Args:
            name: The name of this waveform
            scan_parameter: The parameter that is being scanned in this waveform
        """
        self.name = name
        self.slot_string = ""
        self.scan_parameter = scan_parameter
        self.timeslot = 0
        self.channel = 0
        self.slot = 0
        self.segments = []
        self.length = 0
        self.N_scan_values = 0
        self.PA_phase_init = 0
        self.PA_phase_accumulated = 0
        self.calibration = None  # This will be populated in the assign function.
        self.prev_phase_gates = 0

    def assign(
        self, calibration: cal.Calibration, timeslot: int, channel: int, slot: int
    ):
        """This function assigns the waveform to a particular time slot, channel, and
        slot, populating those fields and also prepending the time slot and channel to
        the waveform name.

        Args:
            calibration: The Calibration object that will be used to calibrate the
                waveform output
            timeslot: The time slot into which this waveform is being placed
            channel: The AWG channel (0-indexed) that will produce this waveform
            slot: The physical slot (1-indexed, corresponds to AOM channel and
                fiber) that will receive this waveform
        """

        self.calibration = calibration
        self.timeslot = timeslot
        self.channel = channel
        self.slot = slot

        ch_strings = ["A", "B", "C", "D"]
        self.slot_string = "{0:02d}{1}{2:02d}".format(
            self.timeslot, ch_strings[self.channel], self.slot
        )
        self.name = self.slot_string + " " + self.name

    def add_segment(self, segment: Segment):
        """This function appends the given segment to this waveform.

        Args:
            segment: The segment to be added
        """
        # Check that the arrays of component amplitudes, frequencies, phases,
        # and PA_prefactors have equal lengths
        if (
            len(segment.amplitudes) != len(segment.freqs)
            or len(segment.freqs) != len(segment.phases)
            or len(segment.phases) != len(segment.PA_prefactors)
        ):
            _LOGGER.error(
                "Segment added to waveform '%s' with frequency/amplitude/phase/PA "
                "prefactor arrays of unequal length",
                self.name,
            )
            raise ValueError(
                'Error: Segment added to waveform "{}" with frequency/amplitude'
                "/phase/PA prefactor arrays of unequal length".format(self.name)
            )

        # Check that if this segment has a parameter that's being scanned, the number of parameter values is proper
        if len(segment.scan_values) > 0:
            if self.N_scan_values == 0:
                # If we haven't yet set the N_scan_values parameter for this waveform, set it based on this segment
                self.N_scan_values = len(segment.scan_values)
            else:
                if len(segment.scan_values) != self.N_scan_values:
                    # If the number of parameter values for this segment doesn't match the N_scan_values that's already
                    # been set, throw an error
                    _LOGGER.error(
                        "Segments added to waveform '%s' with different numbers "
                        "of values for the scanned parameter",
                        self.name,
                    )
                    raise ValueError(
                        'Error: Segments added to waveform "{}" with different'
                        " numbers of values for the scanned parameter".format(self.name)
                    )

        # If a segment of this waveform has a nonzero number of scan parameter values,
        # check that this waveform thinks that it should be scanning something
        if self.N_scan_values != 0 and self.scan_parameter == ScanParameter.static:
            _LOGGER.error(
                "Segments added to waveform '%s' with a non-null array of values "
                "for the scanned parameter, but no scan parameter set for the waveform",
                self.name,
            )
            raise ValueError(
                'Error: Segments added to waveform "{}" with a non-null array of '
                "values for the scanned parameter, but no scan parameter was set "
                "for the waveform".format(self.name)
            )

        self.segments.append(segment)
        effective_duration = (
            max(segment.scan_values)
            if self.scan_parameter == ScanParameter.seg_duration
            else segment.duration
        )
        self.length += effective_duration
        self.PA_phase_accumulated += 2 * np.pi * segment.PA_freq * effective_duration

    def generate_no_scan(self, waveform_dir: str):
        """This function generates the analog values corresponding to this waveform,
        which does not have a parameter that is scanned, and saves them to file.

        Args:
            waveform_dir: The directory where this waveform file will be written

        Returns:
            It returns a list of the number of files and total data length written
        """
        data_to_write = []
        t_init = 0
        for seg in self.segments:
            data_to_write.extend(
                seg.evaluate(t_init, self.PA_phase_init, self.calibration, self.slot)
            )
            t_init += seg.duration

        # Pad data_to_write to coerce length up to the nearest 10 ns
        N_points = len(data_to_write)
        data_to_write = np.concatenate(
            [
                data_to_write,
                np.zeros(int(10 * math.ceil(round(N_points / 10.0, 15)) - N_points)),
            ]
        )
        filewrite(self.name, data_to_write, waveform_dir)
        return [1, len(data_to_write)]

    def generate_scan(self, waveform_dir: str):
        """This function generates the analog values corresponding to this waveform,
        which does have a parameter that is scanned, and saves them to file.

        Args:
            waveform_dir: The directory where this waveform file will be written

        Returns:
            It returns a list of the number of files and total data length written
        """
        # First, make a directory into which we will put the array of waveform files
        scan_subdir = waveform_dir + r"\\" + self.name
        os.makedirs(scan_subdir)

        data_length = 0
        for i in range(self.N_scan_values):
            data_to_write = []
            t_init = 0
            for seg in self.segments:
                data_to_write.extend(
                    seg.evaluate_scan(
                        t_init,
                        self.PA_phase_init,
                        self.calibration,
                        self.slot,
                        self.scan_parameter,
                        i,
                    )
                )
                t_init += seg.duration

            # Pad data_to_write to coerce length up to the nearest 10 ns
            N_points = len(data_to_write)
            data_to_write = np.concatenate(
                [
                    data_to_write,
                    np.zeros(
                        int(10 * math.ceil(round(N_points / 10.0, 15)) - N_points)
                    ),
                ]
            )
            filewrite("{0:03d}".format(i), data_to_write, scan_subdir)
            data_length += len(data_to_write)

        return [self.N_scan_values, data_length]

    def generate_circuit_scan(self, waveform_dir: str, circuit_index: int):
        """This function generates the analog values corresponding to this waveform,
        which does have a parameter that is scanned, and saves them to file.

        Args:
            waveform_dir: The directory where this waveform file will be written
            circuit_index: In circuit scan mode, the index of the circuit
                represented by this sequence

        Returns:
            It returns a list of the number of files and total data length written
        """
        # Before writing the first file, make a directory into which we will put
        # the array of waveform files
        scan_subdir = (
            waveform_dir + r"\\" + self.slot_string + " Gate ({0})".format(self.slot)
        )
        if circuit_index == 0:
            os.makedirs(scan_subdir)

        data_to_write = []
        t_init = 0
        for seg in self.segments:
            data_to_write.extend(
                seg.evaluate(t_init, self.PA_phase_init, self.calibration, self.slot)
            )
            t_init += seg.duration

        # Pad data_to_write to coerce length up to the nearest 10 ns
        N_points = len(data_to_write)
        data_to_write = np.concatenate(
            [
                data_to_write,
                np.zeros(int(10 * math.ceil(round(N_points / 10.0, 15)) - N_points)),
            ]
        )
        filewrite("{0:03d}".format(circuit_index), data_to_write, scan_subdir)
        return [1, len(data_to_write)]

    def generate(self, waveform_dir: str, circuit_index: int):
        """This function generates the analog values corresponding to this waveform and
        saves them to file.

        Args:
            waveform_dir: The directory where this waveform file will be written
            circuit_index: In circuit scan mode, the index of the circuit
                represented by this sequence

        Returns:
            A list of the number of files and total data length written
        """

        if circuit_index >= 0:
            result = self.generate_circuit_scan(waveform_dir, circuit_index)
        elif self.scan_parameter == ScanParameter.static:
            result = self.generate_no_scan(waveform_dir)
        else:
            result = self.generate_scan(waveform_dir)

        return result

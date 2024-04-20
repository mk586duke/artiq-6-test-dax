import logging
import os
import shutil
import typing

import numpy as np

from euriqabackend.devices.keysight_awg import AOM_calibration as cal
from euriqabackend.devices.keysight_awg import waveform as wf


_LOGGER = logging.getLogger(__name__)


class Sequence:
    """The Sequence class defines the entire experimental sequence to be output by the
    AWG.

    It contains the waveform, slot assignments, and timing information.
    """

    def __init__(
        self,
        calibration: cal.Calibration,
        wav_array: typing.List[typing.List[wf.Waveform]],
        slot_array: typing.List[typing.List[int]],
        twoQ_gate_array: typing.List[int],
        waveform_dir: str,
        PA_dark_freqs: typing.List[float],
        wait_after_array: typing.List[int] = None,
        wait_after_time: float = 0,
    ):
        """This initializes a sequence, which specifies a sequence of waveforms to be
        output by the AWG.

        Args:
            calibration: The Calibration object that will be used to calibrate
                the waveform output
            wav_array: The 2D array of waveforms to be output
            slot_array: The slots two which the RF pulses will be applied
            twoQ_gate_array: An array of 0 or 1, determines whether each time
                step in the sequence corresponds to a single-qubit or two-qubit gate
            PA_dark_freqs: The LO frequencies for each slot, which determines
                how the waveform phase is advanced between each pulse.
                Note that the array is 0-indexed and has a length of 33,
                which enables it to track the phase of the global beam,
                which we assign to slot 0.
            wait_after_array: An array of 0 or 1, determines whether a wait
                is inserted after each timestep
            wait_after_time: The wait in us inserted after every timestep
                indicated in the wait_after_array
            waveform_dir: The directory where thie sequence's waveforms will be written
        """
        self.calibration = calibration
        self.wav_array = wav_array
        self.slot_array = slot_array
        self.twoQ_gate_array = twoQ_gate_array
        if wait_after_array is None:
            wait_after_array = [0] * len(self.twoQ_gate_array)
        self.wait_after_array = wait_after_array
        self.wait_after_time = wf.coerce_10ns(wait_after_time)
        self.waveform_dir = waveform_dir
        self.oneQ_length = 0
        self.twoQ_length = 0
        self.N_scan_values = 0
        self.total_duration = 0

        self.PA_dark_freqs = PA_dark_freqs
        self.PA_tprev_array = np.zeros(33)
        self.PA_phase_array = np.zeros(33)

        if (
            len(self.wav_array) != len(self.slot_array)
            or len(self.twoQ_gate_array) != len(self.wait_after_array)
            or len(self.wav_array) != len(self.twoQ_gate_array)
        ):
            _LOGGER.error(
                "The arrays of waveforms, slots, 2Q/1Q gate lengths, and wait times "
                "passed to the sequence have different lengths"
            )
            raise ValueError(
                "The arrays of waveforms, slots, 2Q/1Q gate lengths, and wait times "
                "passed to the sequence have different lengths"
            )

        if any(
            [
                len(self.wav_array[i]) != len(self.slot_array[i])
                for i in range(len(self.wav_array))
            ]
        ):
            _LOGGER.error(
                "The arrays of waveforms and slots do not have the same shape"
            )
            raise Exception(
                "Error: The arrays of waveforms and slots do not have the same shape"
            )

    def compile(self):
        """This compiles the sequence, which involves setting the max lengths of all
        one- and two-qubit gates and assigning each waveform to its proper timestep and
        channel."""

        # Query the number of scan values for all waveforms
        all_N_scan_values = [
            [x.N_scan_values for x in timestep] for timestep in self.wav_array
        ]
        self.N_scan_values = max([max(x) for x in all_N_scan_values])

        # Check that the number of scan values for all waveforms are all either the same or zero
        check_NSV = lambda N: N == 0 or N == self.N_scan_values
        All_NSVs_equal = all(
            [all([check_NSV(x) for x in timestep]) for timestep in all_N_scan_values]
        )
        if not All_NSVs_equal:
            _LOGGER.error(
                "Not all waveforms that are scanned have the same number of values "
                "for the scan parameter"
            )
            raise Exception(
                "Not all waveforms that are scanned have the same number of values "
                "for the scan parameter"
            )

        # Assign each waveform to a timeslot and channel
        for timeslot, timestep in enumerate(self.wav_array):
            for channel, x in enumerate(timestep):
                x.assign(
                    self.calibration,
                    timeslot,
                    channel,
                    self.slot_array[timeslot][channel],
                )

        # Query the lengths of all waveforms, divide them into oneQ and twoQ
        # categories, then find the max of each
        # We ignore the pulses on the ref_channel when calculating the lengths
        # because they are long by construction
        ref_channel = 3
        all_lengths = [
            [(x.length if x.channel != ref_channel else 0) for x in timestep]
            for timestep in self.wav_array
        ]
        all_oneQ_lengths = [
            all_lengths[i]
            for i in range(len(self.twoQ_gate_array))
            if self.twoQ_gate_array[i] == 0
        ]
        all_twoQ_lengths = [
            all_lengths[i]
            for i in range(len(self.twoQ_gate_array))
            if self.twoQ_gate_array[i] != 0
        ]
        # print(all_lengths)
        # print(all_oneQ_lengths)
        # print(all_twoQ_lengths)
        if len(all_oneQ_lengths) > 0:
            self.oneQ_length = wf.coerce_10ns(max([max(x) for x in all_oneQ_lengths]))
        if len(all_twoQ_lengths) > 0:
            self.twoQ_length = wf.coerce_10ns(max([max(x) for x in all_twoQ_lengths]))
        # print(self.oneQ_length)
        # print(self.twoQ_length)

    def phase_forward(self):
        """This function sets the PA_phase input for all of the waveforms depending on
        their position in the sequence and the history of each slot."""
        # HVI timing constants in us, starting with the delay after the Enqueue
        # D Waveforms subHVI
        enqueueD_delay = 1.000
        HVI_delay = 3.37
        switching_delay = 2.00
        twoQ_correction = 0.01
        oneQ_correction = 0.0

        # Calculate the times at which each timestep begins, which are determined
        # by the static delays listed above as well as whether specific timesteps
        # have the 2Q length or wait after flags activated
        timestep_times = np.zeros(len(self.wav_array) + 1)
        prev_timestep_time = 0
        for i in range(1, len(timestep_times)):
            timestep_times[i] = (
                prev_timestep_time
                + enqueueD_delay
                + HVI_delay
                + switching_delay
                + (
                    self.twoQ_length + twoQ_correction
                    if self.twoQ_gate_array[i - 1] == 1
                    else self.oneQ_length + oneQ_correction
                )
                + (self.wait_after_time if self.wait_after_array[i - 1] == 1 else 0)
            )
            prev_timestep_time = timestep_times[i]
        self.total_duration = timestep_times[-1]

        for _timeslot, timestep in enumerate(self.wav_array):
            for _channel, x in enumerate(timestep):
                # First, assign the initial PA phase of this waveform by taking the
                # previous PA phase corresponding to this slot and adding phase
                # given by propagation at the PA dark frequency from the last
                # time this channel was addressed until the beginning of this timestep
                x.PA_phase_init = (
                    self.PA_phase_array[x.slot]
                    + x.prev_phase_gates
                    + 2
                    * np.pi
                    * self.PA_dark_freqs[x.slot]
                    * (timestep_times[x.timeslot] - self.PA_tprev_array[x.slot])
                )
                # Set the new PA phase for this slot to this waveform's initial
                # phase plus its accumulated phase
                self.PA_phase_array[x.slot] = x.PA_phase_init + x.PA_phase_accumulated
                # Set the new tprev for this channel to the absolute end time of
                # this waveform
                self.PA_tprev_array[x.slot] = timestep_times[x.timeslot] + x.length

        return self.total_duration,timestep_times

    def write_metadata(self):
        """This function writes the sequence metadata to file so that it can be read by
        LabVIEW."""
        data_to_write = (
            [len(self.wav_array)]
            + [self.oneQ_length]
            + [self.twoQ_length]
            + [self.wait_after_time]
            + self.twoQ_gate_array
            + self.wait_after_array
        )

        np.savetxt(
            self.waveform_dir + "\\Sequence Metadata.csv", data_to_write, fmt="%f"
        )

    def write(self, circuit_indices: typing.List[int]):
        """This function generates the analog waveform files that comprise this
        sequence.

        Args:
            circuit_indices: In circuit scan mode, the index of the circuit
                represented by this sequence
        """
        # First, erase and recreate the waveform directory if we're not in circuit
        # scan mode (all circuit_indices = -1)
        # or if we are creating the first files in circuit scan mode
        # (circuit_indices <= 0).  Second, write metadata
        if max(circuit_indices) <= 0:
            shutil.rmtree(self.waveform_dir, ignore_errors=True)
            os.makedirs(self.waveform_dir, exist_ok=True)
            self.write_metadata()

        N_files_written = 0
        total_data_written = 0
        for timeslot, timestep in enumerate(self.wav_array):
            for _channel, x in enumerate(timestep):
                write_result = x.generate(self.waveform_dir, circuit_indices[timeslot])
                N_files_written += write_result[0]
                total_data_written += write_result[1]

        return [N_files_written, total_data_written]

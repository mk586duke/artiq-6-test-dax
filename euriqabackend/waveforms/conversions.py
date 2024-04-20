"""Implement unit conversions and other utility functions for EURIQA waveforms."""
import collections.abc
import functools
import logging
import typing

import numpy as np
import pulsecompiler.qiskit.backend as pc_be
import pulsecompiler.qiskit.configuration as pc_config
import qiskit.pulse as qp


_LOGGER = logging.getLogger(__name__)

import struct

def hack_angle(angle,trot):
    ret = bytearray(struct.pack("d", angle))
    ret[0] = 1 + 4*trot     
    ret2 = struct.unpack("d", ret)    
    return ret2[0]

def return_hidden_digit(my_angle):    
    ret = bytearray(struct.pack("d", my_angle))    
    return int(ret[0]/4)

def get_aom_array_index(
    config: typing.Union[
        pc_config.BackendConfigCenterIndex,
        pc_config.BackendConfigZeroIndex,
        pc_config.QuickConfig,
    ],
    qubit_index: int,
    array_center_index: int,
) -> int:
    """Get the equivalent AOM array index for a given qubit.

    This is roughly equivalent to the slot for the AWG, but it accepts either
    center-index or zero-index qubit indices, depending on the configuration.
    """
    if type(config) in {pc_config.BackendConfigCenterIndex, pc_config.QuickConfig}:
        return qubit_index + array_center_index
    elif type(config) == pc_config.BackendConfigZeroIndex:
        return (
            config.zero_index_to_center_index(qubit_index, config.num_qubits)
            + array_center_index
        )
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")


@functools.lru_cache(maxsize=256)
def channel_rabi_maximum_frequency(
    channel: typing.Union[qp.DriveChannel, qp.ControlChannel],
    backend: pc_be.MinimalQiskitIonBackend,
) -> float:
    """Maximum Rabi frequency for the channel, when output is played at full power.

    Currently this is just a global (one for individual, one for global) value
    from the RF Calibration data structure. This might change to per-channel
    for better power optimization in the future.
    """
    rf_calib: "CalibrationBox" = backend.properties().rf_calibration  # noqa: F821
    #print("channel_rabi_maximum_frequency : ", rf_calib.rabi.pi_time_individual_value)

    if isinstance(channel, qp.DriveChannel):
        max_rabi_individual = 1 / (rf_calib.rabi.pi_time_individual.value * 2)
        # TODO: convert this to either method. Meant for transition from
        # single max_rabi_frequency -> one per-channel
        if isinstance(max_rabi_individual, (collections.abc.Sequence, np.ndarray)):
            # index into arr/list using individual channel index via RFSoC mapping
            _LOGGER.debug(
                "Retrieving max rabi frequency for %s from %s (type: %s)",
                channel,
                max_rabi_individual,
                type(max_rabi_individual),
            )
            center_aom_index = rf_calib["other.center_aom_index.value"]
            individual_channel_aom_index = get_aom_array_index(
                backend.configuration(),
                backend.configuration().get_channel_qubits(channel)[0],
                center_aom_index,
            )
            return max_rabi_individual[individual_channel_aom_index]
        else:
            _LOGGER.debug(
                "Retrieved channel '%s' max rabi: %s", channel, max_rabi_individual
            )
            return max_rabi_individual
    elif isinstance(channel, qp.ControlChannel):
        raise NotImplementedError(
            "Rabi frequency control for global beam not implemented."
        )
    else:
        raise ValueError(f"Unrecognized channel provided: {channel}")


@functools.lru_cache(maxsize=256)
def channel_aom_saturation(
    channel: typing.Union[qp.DriveChannel, qp.ControlChannel],
    backend: pc_be.MinimalQiskitIonBackend,
):
    """Returns the AOM saturation parameter for the given Qiskit channel."""
    backend_calib = backend.properties().rf_calibration
    config = backend.configuration()
    if isinstance(channel, qp.DriveChannel):
        ind_aom_channel = get_aom_array_index(
            config,
            config.get_channel_qubits(channel)[0],
            backend_calib["other.center_aom_index.value"],
        )
        calib_array = backend_calib.aom_saturation.individual.value
        assert 0 <= ind_aom_channel < len(calib_array)
        return calib_array[ind_aom_channel]
    else:
        return backend_calib.aom_saturation.global_aom.value


def apply_nonlinearity(amp: typing.Union[float, np.ndarray], nonlinearity: float):
    """Apply nonlinearity correction to a given amplitude/array of amplitudes."""
    return nonlinearity * np.arcsin(amp / nonlinearity)


def get_amplitude_from_rabi_frequency(
    rabi_frequency: float,
    max_rabi_frequency: float,
    nonlinearity_factor: float = None,
    use_nonlinearity: bool = False,
) -> float:
    """Return the expected amplitude of the output signal given the nominal Rabi frequency.

    Applies correction based on both the overall output power (measured by Rabi time),
    as well as (optionally) compression due to the AOM saturation.
    Typically AOM saturation should be handled by the schedule calibration function
    for calibrating nonlinearity.

    Arguments:
        rabi_frequency (float): Rabi frequency that you would like to play on the
            channel, in Hz.
            The returned amplitude value will play this Rabi frequency,
            assuming an instantaneous pulse. i.e. a square pulse will produce this Rabi
            rate for its entire duration, but a Gaussian pulse will only produce this
            Rabi frequency at the maximum output amplitude (the peak).
        max_rabi_frequency (float): the maximum Rabi frequency in Hz that a
            given channel can play.
        nonlinearity_factor (float): Characterizes the AOM saturation parameter of a given
            AOM channel. This is produced by fitting Rabi frequencies for different
            amplitudes to a ``Omega = rabi_freq * sin(nonlinearity * output_amplitude)``
            model.
        use_nonlinearity (bool, optional): Whether the nonlinearity calibration
            should be applied.
    """
    #print("In corrector, use_nonlinearity = {nonlin}, rabi_freq = {freqReq}, max_rabi_freq = {maxFreq}".format(nonlin=use_nonlinearity, freqReq=rabi_frequency,maxFreq=max_rabi_frequency))
    if use_nonlinearity:
        assert nonlinearity_factor is not None, "Must provide nonlinearity factor"
    uncorrected_amplitude = rabi_frequency / max_rabi_frequency

    if use_nonlinearity:
        amplitude = apply_nonlinearity(uncorrected_amplitude, nonlinearity_factor)
    else:
        amplitude = uncorrected_amplitude
    assert (
        -1.0 <= amplitude <= 1.0
    ), f"Amplitude {amplitude} is out of range [-1.0, 1.0]"
    return amplitude


get_amplitude_from_rabi_frequency_vectorized = np.vectorize(
    get_amplitude_from_rabi_frequency
)


def rabi_frequency_to_amplitude(
    rabi_frequency: float,
    channel: typing.Union[qp.DriveChannel, qp.ControlChannel],
    backend: pc_be.MinimalQiskitIonBackend,
):
    """Return the amplitude for a given Rabi frequency on a given channel.

    Convenience function/wrapper around local sub-functions.
    """
    return get_amplitude_from_rabi_frequency(
        rabi_frequency, channel_rabi_maximum_frequency(channel, backend),
    )

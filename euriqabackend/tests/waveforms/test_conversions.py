"""Test :mod:`euriqabackend.waveforms.conversions`."""
import numpy as np
import pytest
import qiskit.pulse as qp

import euriqabackend.waveforms.conversions as wf_convert


def test_max_rabi_frequency_lookup_all_qubits(qiskit_backend_with_real_cals):
    config = qiskit_backend_with_real_cals.configuration()
    for c in map(config.individual_channel, config.all_qubit_indices_iter):
        max_freq = wf_convert.channel_rabi_maximum_frequency(
            c, qiskit_backend_with_real_cals
        )
        assert isinstance(max_freq, float)
        assert 1e3 < max_freq < 10e6


def test_max_rabi_frequency_lookup_freq_in_array(qiskit_backend_with_real_cals):
    config = qiskit_backend_with_real_cals.configuration()
    rf_calib = qiskit_backend_with_real_cals.properties().rf_calibration
    num_individual_channels = rf_calib.other.individual_aom_channel_count.value
    pi_times = np.linspace(200e-9, 500e-9, num=num_individual_channels)
    rabi_freqs = 1 / (pi_times * 2)

    rf_calib.rabi.pi_time_individual.value = pi_times
    for qubit_idx in config.all_qubit_indices_iter:
        channel = config.individual_channel(qubit_idx)
        max_freq = wf_convert.channel_rabi_maximum_frequency(
            channel, qiskit_backend_with_real_cals
        )
        assert max_freq == rabi_freqs[rf_calib.other.center_aom_index.value + qubit_idx]


def test_get_amplitude_from_rabi_frequency():
    assert (
        wf_convert.get_amplitude_from_rabi_frequency(
            1e6, 1e6, 1.2, use_nonlinearity=False
        )
        == 1.0
    )
    assert (
        wf_convert.get_amplitude_from_rabi_frequency_vectorized(
            np.array([1e6]), 1e6, 1.2, use_nonlinearity=False
        )
        == 1.0
    )
    # out-of-bounds amplitude
    with pytest.raises(AssertionError):
        wf_convert.get_amplitude_from_rabi_frequency(
            1e6, 1e6, 1.2, use_nonlinearity=True
        )

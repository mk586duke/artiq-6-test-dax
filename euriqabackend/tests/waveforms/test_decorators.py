"""Test :mod:`euriqabackend.waveforms.decorators`."""
import pytest
import qiskit.pulse as qp
import qiskit.pulse.exceptions as qp_except

import euriqabackend.devices.keysight_awg.common_types as common_types
import euriqabackend.waveforms.decorators as wf_dec


def test_get_calibration_value_from_backend(qiskit_backend_with_fake_cals):
    # retrieve dataset-mapped value, and check that it's properly input
    @wf_dec.default_args_from_calibration(
        {"rabi_max": "rabi.maximum_frequency_individual"}
    )
    def test_function(arg, rabi_max: float = None):
        return arg, rabi_max

    with qp.build(backend=qiskit_backend_with_fake_cals):
        assert test_function("test") == ("test", 1e6)
        assert test_function("test", rabi_max="fake_value") == ("test", "fake_value")


def test_get_calibration_no_backend():
    @wf_dec.default_args_from_calibration({})
    def test_function(arg):
        return arg

    with pytest.raises(qp_except.NoActiveBuilder):
        test_function()  # pylint: disable=no-value-for-parameter


def test_get_gate_solution(qiskit_backend_with_gate_solutions):
    """Test :func:`euriqabackend.waveforms.decorators.get_gate_solution`."""

    @wf_dec.get_gate_solution(
        {
            common_types.XXModulationType.AM_interp,
            common_types.XXModulationType.AM_segmented,
        },
        {"detuning": "detuning", "rabi_segments": "segments", "sign": "sign"},
        convert_ions_to_slots=True,
    )
    @wf_dec.ignore_arguments({}, remove_kwargs={"backend"})
    def fake_gate_function(ions, detuning, rabi_segments, sign, other_params=0.0):
        return (ions, detuning, rabi_segments, sign, other_params)

    with pytest.raises(ValueError, match="Ion indices not specified"):
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        fake_gate_function(backend=qiskit_backend_with_gate_solutions)

    with pytest.raises(qp_except.NoActiveBuilder):
        # pylint: disable=no-value-for-parameter
        fake_gate_function((0, 4))

    with qp.build(backend=qiskit_backend_with_gate_solutions):
        # pylint: disable=no-value-for-parameter
        # ignore first parameter b/c that's just the ions, not sorted by decorator
        assert fake_gate_function((0, 4))[1:] == fake_gate_function((4, 0))[1:]
        # for l, r in zip(fake_gate_function((0, 4))[1:], fake_gate_function((4, 0))[1:]):
        #     if isinstance(l, np.ndarray) or len(l) > 0:
        #         np.testing.assert_equal(l, r)
        #     else:
        #         assert l == r
        # Check that manually setting will override
        assert fake_gate_function(
            (0, 2), -1e6, [(1e-6, (0.5, 0.5))], other_params=0.0
        ) == ((0, 2), -1e6, [(1e-6, (0.5, 0.5))], 1, 0.0)


def test_get_gate_solution_fails_wrong_solution_type(
    qiskit_backend_with_gate_solutions,
):
    @wf_dec.get_gate_solution(
        {common_types.XXModulationType.FM_interp},
        {"detuning": "detuning", "rabi_segments": "segments", "sign": "sign"},
        convert_ions_to_slots=True,
    )
    def invalid_gate_function(ions, detuning, rabi_segments, sign, other_params=0.0):
        return (ions, detuning, rabi_segments, sign, other_params)

    with pytest.raises(AssertionError):
        with qp.build(backend=qiskit_backend_with_gate_solutions):
            # pylint: disable=no-value-for-parameter
            invalid_gate_function((0, 4))


def test_get_calibrated_gate_parameter_function(qiskit_backend_with_gate_solutions):
    """Test :func:`euriqabackend.waveforms.decorators._get_calibrated_gate_parameter`."""
    with pytest.raises(AssertionError):
        # pylint: disable=protected-access
        wf_dec._get_calibrated_gate_parameter(
            (0, 1), "nonexistent_calib_name", qiskit_backend_with_gate_solutions
        )

    gate_slot = (11, 12)
    global_slot = (-1, -1)
    # Test it applies the global & per-gate calibrations correctly
    qiskit_backend_with_gate_solutions.properties().rf_calibration.gate_tweaks.struct.loc[
        (-1, -1), "XX_duration_us"
    ] = 2
    qiskit_backend_with_gate_solutions.properties().rf_calibration.gate_tweaks.struct.loc[
        gate_slot, "XX_duration_us"
    ] = 5
    # pylint: disable=protected-access
    assert (
        wf_dec._get_calibrated_gate_parameter(
            gate_slot, "XX_duration_us", qiskit_backend_with_gate_solutions
        )
        == 230
    )
    assert (
        wf_dec._get_calibrated_gate_parameter(
            gate_slot,
            "XX_duration_us",
            qiskit_backend_with_gate_solutions,
            global_adjustment_key=global_slot,
        )
        == 232
    )


def test_get_gate_calibration_decorator(qiskit_backend_with_gate_solutions):
    """Test the decorator as it's designed to work."""

    @wf_dec.get_gate_parameters(
        {
            "glob_amp": "global_amplitude",
            "ind_amp_mult": "individual_amplitude_multiplier",
        },
        convert_ions_to_slots=True,
        rescale_amplitude=True,
    )
    def my_gate_function(ions, glob_amp, ind_amp_mult):
        return (ions, glob_amp, ind_amp_mult)

    with pytest.raises(qp_except.NoActiveBuilder):
        # pylint: disable=no-value-for-parameter
        my_gate_function((0, 2))

    with pytest.raises(ValueError):
        # pylint: disable=no-value-for-parameter
        my_gate_function()

    with qp.build(qiskit_backend_with_gate_solutions):
        # pylint: disable=no-value-for-parameter
        # Check that auto-fills values.
        assert my_gate_function((0, 2)) == ((0, 2), 1.0, 0.7)


def test_get_calibration_decorator_fail(qiskit_backend_with_gate_solutions):
    # pylint: disable=no-value-for-parameter
    @wf_dec.get_gate_parameters({"glob_amp": "invalid_param"})
    def invalid_param_function(ions, glob_amp):
        return ions, glob_amp

    @wf_dec.get_gate_parameters
    def invalid_decorator_function():
        pass

    @wf_dec.get_gate_parameters(
        {"invalid_name": "global_amplitude"}, convert_ions_to_slots=True
    )
    def missing_name_func(ions, glob_amp):
        return ions, glob_amp

    with pytest.raises(TypeError):
        invalid_decorator_function()

    with qp.build(qiskit_backend_with_gate_solutions):
        with pytest.raises(AssertionError, match="invalid_param"):
            invalid_param_function((0, 2))

        with pytest.raises(TypeError, match="invalid_name"):
            missing_name_func((0, 2))


def test_get_calibration_decorator_unneeded(rfsoc_qiskit_backend):
    """Test the decorator when no calibration data structure is supplied.

    If all parameters are supplied, it shouldn't need to access the
    :class:`GateCalibration`.
    """

    @wf_dec.get_gate_parameters(
        {"glob_amp": "global_amplitude"}, convert_ions_to_slots=True
    )
    def gate_function(ions, glob_amp):
        return ions, glob_amp

    with qp.build(rfsoc_qiskit_backend):
        assert gate_function((0, 2), 0.5) == ((0, 2), 0.5)


def test_get_gate_solution_decorator_unneeded(rfsoc_qiskit_backend):
    """Test the decorator when no gate solution data structure is supplied.

    If all parameters are supplied, it shouldn't need to access the
    :class:`GateSolution`.
    """

    @wf_dec.get_gate_solution(
        {
            common_types.XXModulationType.AM_interp,
            common_types.XXModulationType.AM_segmented,
        },
        {"detuning": "detuning"},
        convert_ions_to_slots=True,
    )
    def gate_function(ions, detuning):
        return ions, detuning

    with qp.build(rfsoc_qiskit_backend):
        assert gate_function((0, 2), 2.5e6) == ((0, 2), 2.5e6)


def test_manual_backend_specified_get_calibration(qiskit_backend_with_fake_cals):
    """Test that calling the "gate function" outside of a pulse builder context
    works if the ``backend`` is manually specified.
    """

    @wf_dec.default_args_from_calibration(
        {"rabi_max": "rabi.maximum_frequency_individual"}
    )
    @wf_dec.ignore_arguments(remove_kwargs={"backend"})
    def sample_gate_function(ion, rabi_max):
        return ion, rabi_max

    # note the lack of qp.build() here
    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    assert sample_gate_function(5, backend=qiskit_backend_with_fake_cals) == (
        5,
        qiskit_backend_with_fake_cals.properties().rf_calibration.rabi.maximum_frequency_individual.value,  # noqa: E501 line too long
    )


def test_manual_backend_specified_get_gate_solution(qiskit_backend_with_gate_solutions):
    @wf_dec.get_gate_solution(
        {
            common_types.XXModulationType.AM_interp,
            common_types.XXModulationType.AM_segmented,
        },
        {"detuning": "detuning"},
        convert_ions_to_slots=True,
    )
    def gate_function(ions, detuning, backend=None):
        return ions, detuning

    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    assert gate_function((0, 2), backend=qiskit_backend_with_gate_solutions) == (
        (0, 2),
        3.0164994285714286,
    )


@pytest.mark.parametrize(
    "second_channel_duration", range(99, 102),
)
def test_same_duration_passes(second_channel_duration):
    @wf_dec.check_all_channels_same_duration
    def gate_function():
        with qp.build() as out_sched:
            qp.play(qp.Constant(100, 1.0), qp.DriveChannel(0))
            qp.play(qp.Constant(second_channel_duration, 1.0), qp.DriveChannel(1))

        return out_sched

    if second_channel_duration == 100:
        gate_function()
    else:
        with pytest.raises(AssertionError):
            gate_function()


def test_ignore_argument():
    @wf_dec.ignore_arguments({"ignored"})
    def gate_function(a, ignored, b):
        return a, ignored, b

    assert gate_function(0, 2) == (0, None, 2)
    assert gate_function(0, "ignore", 2) == (0, "ignore", 2)


def test_ignore_kwarg():
    @wf_dec.ignore_arguments({}, remove_kwargs={"backend"})
    def gate_function(a, b):
        return a, b

    assert gate_function(1, 2, backend="test_backend") == (1, 2)
    assert gate_function(1, 2) == (1, 2)  # still works w/o backend arg specified


def test_ignore_no_overflow_args():
    @wf_dec.ignore_arguments({"ignore"})
    def gate_function(a, ignore, b, *, kw=5):
        return a, ignore, b, kw

    assert gate_function(1, 2, 3) == (1, 2, 3, 5)

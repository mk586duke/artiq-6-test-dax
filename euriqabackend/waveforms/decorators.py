"""Decorators to help when creating Qiskit schedules."""
import functools
import itertools
import inspect
import logging
import typing

import numpy as np
import qiskit.providers as q_prov
import qiskit.pulse as qp
import qiskit.pulse.exceptions as qp_except

# TODO: change this to a file in euriqabackend/waveforms
import euriqabackend.devices.keysight_awg.common_types as common_types
import euriqabackend.devices.keysight_awg.gate_parameters as gate_params

_LOGGER = logging.getLogger(__name__)


def check_all_channels_same_duration(function):
    """Check that all channels are the same duration in a gate function.

    Useful for catching off-by-one errors in the gate duration,
    due to e.g. math/rounding errors.
    """

    @functools.wraps(function)
    def _check_return_schedule(*args, **kwargs):
        return_schedule = function(*args, **kwargs)
        if len(return_schedule.channels) == 0:
            # empty schedule should always pass.
            return return_schedule
        channel_durations = tuple(
            map(return_schedule.ch_duration, return_schedule.channels)
        )
        assert max(channel_durations) == min(channel_durations)
        return return_schedule

    return _check_return_schedule


def default_args_from_calibration(argument_mapping: typing.Dict[str, str]):
    """Decorator to modify calls to the given function.

    Uses the given dictionary ``argument_mapping`` to lookup values in
    the backend's calibration values, and then automatically pass
    that value to the function.
    The keys of ``argument_mapping`` are the keyword arguments of the
    decorated function that should be output, and the values of ``argument_mapping``
    are the paths to the calibration value in the data structure.

    This requires that the overridden arguments are specified as keyword arguments.

    To use this, you should declare all "sweep-able" parameters in the decorator,
    and then users can override them with whatever desired value that they want.

    Example:
    ```python
    import qiskit.pulse as qp

    @default_args_from_calibration({"rabi_max": "rabi.maximum_frequency_individual"})
    def my_pulse_function(ion: int, rabi_max: float = None):
        with qp.build(qp.active_builder(), name="my_custom_pulse) as sched:
            do_something()

        return sched

    with qp.build(backend) as schedule:
        # Override the default calibration value
        qp.call(my_pulse_function(0, rabi_max=0.5))

        # Use the calibrated value
        qp.call(my_pulse_function(0))
    ```

    If you want to use the decorated function outside of a pulse builder context,
    you can explicitly pass the ``backend`` kwarg.
    The following is the same as calling it within the ``qp.build()`` context above:
    ```python
    my_pulse_function(ion, backend=qiskit_backend)
    ```
    """

    def _decorator_get_values(function):
        @functools.wraps(function)
        def _get_calibration_values(*args, **kwargs):
            """Retrieve the calibration value, and pass it to the decorated function."""
            has_backend_kwarg = "backend" in kwargs.keys()
            if has_backend_kwarg:
                rf_cal = kwargs["backend"].properties().rf_calibration
            else:
                rf_cal = qp.active_backend().properties().rf_calibration

            func_name = inspect.unwrap(function).__name__
            for argument, rf_cal_path in argument_mapping.items():
                if argument not in kwargs.keys():
                    value = rf_cal[rf_cal_path].value
                    _LOGGER.debug(
                        "Replacing argument %s with calibration value %s "
                        "(path: %s, function: %s(args=%s, kwargs=%s))",
                        argument,
                        value,
                        rf_cal_path,
                        func_name,
                        args,
                        kwargs,
                    )
                    kwargs.setdefault(argument, value)

            return function(*args, **kwargs)

        return _get_calibration_values

    return _decorator_get_values


def _get_calibrated_gate_parameter(
    ions: typing.Tuple[int, ...],
    key: str,
    backend: q_prov.BaseBackend,
    global_adjustment_key: typing.Any = None,
) -> float:
    """
    Get a calibrated gate parameter.

    Searches for the value in the gate solution, per-gate gate calibration, and
    global gate calibrations. The returned value is the sum of any values found
    in each source.

    Args:
        ions (typing.Tuple[int, ...]): Which ions that the calibrated value is for.
            Used for indexing into the gate solutions/calibrations.
        value (str): Name of the value being searched for.
        backend (BaseBackend): A Qiskit Backend with the :class:`CalibrationBox`
            accessible at ``backend.properties().rf_calibration``.
        global_adjustment_key (typing.Any, optional): If specified (i.e. not ``None``),
            what the key for the global adjustment is in the gate_tweaks data structure.
            E.g. :attr:`GateCalibrations.GLOBAL_CALIBRATION_SLOT`. Defaults to None.

    Returns:
        float: Calibrated value. Calculated as a simple sum of the gate solution,
        per-gate calibration, and global gate calibration.
    """
    # TODO: convert properties().gate_solutions.struct -> variable,
    # maybe in common_types-ish file?
    gate_solutions = backend.properties().rf_calibration.gate_solutions.struct
    gate_tweaks = backend.properties().rf_calibration.gate_tweaks.struct
    global_tweaks = {
        "motional_frequency_adjust" : backend.properties().rf_calibration["gate_tweaks.motional_frequency_offset"].value
    }

    assert (key in gate_solutions.columns) or (
        key in gate_tweaks.columns
    ), f"Unrecognized gate parameter {key} passed"

    result = 0.0
    value_found = False
    for series in (
        gate_solutions.loc[ions, :],
        gate_tweaks.loc[ions, :],
#        gate_tweaks.loc[global_adjustment_key, :]
        global_tweaks
        if global_adjustment_key is not None
        else {},
    ):
        if key in series:
            value_found = True
            result += series[key]

    if not value_found:
        raise RuntimeError(f"Calibration value not found for parameter {key}")

    _LOGGER.debug("Got calibrated value for gate: %s[ions %s] = %f", key, ions, result)
    return result


def _normalize_ion_indices(
    function_args: typing.Sequence[typing.Any],
    ion_arg_index: int,
    convert_ions_to_slots: bool = False,
    backend: q_prov.BaseBackend = None,
) -> typing.Tuple[int, ...]:
    """Extracts ion indices in center index notation from function arguments.

    Must pass the where the ions are located in the args, and if you want
    to convert the ions from center-index-notation to "slot" (AWG/RFCompiler-specific)
    notation.
    """
    try:
        ion_indices = tuple(sorted(function_args[ion_arg_index]))
    except IndexError:
        raise ValueError(f"Ion indices not specified in arguments: {function_args}")
    if convert_ions_to_slots:
        # Convert center-indexed ions to RFCompiler slot notation
        try:
            num_qubits = qp.num_qubits()
        except qp_except.NoActiveBuilder as exc:
            if backend is not None:
                num_qubits = backend.configuration().n_qubits
            else:
                raise exc
        ion_indices = tuple(
            sorted(
                map(
                    lambda ion: common_types.ion_to_slot(
                        ion, N_ions=num_qubits, one_indexed=False
                    ),
                    ion_indices,
                )
            )
        )

    return ion_indices


def get_gate_solution(
    solution_types: typing.Set[common_types.XXModulationType],
    solution_arguments: typing.Dict[str, str],
    convert_ions_to_slots: bool = False,
    ion_index_argument: int = 0,
    convert_solution_units: bool = False,
):
    """
    Retrieve gate solutions from the backend, and pass them to the decorated function.

    Arguments that are explicitly specified will be used instead of the default ones
    that this function provides.

    This does a little bit of behind-the-scenes magic, and it makes some assumptions:
    * The decorated function is either called within a qiskit pulse builder context,
        or has the backend explicitly passed as a kwarg:
        ``func(..., backend=qiskit_backend)``
    * gate solutions are specified as a Pandas dataframe accessed at '
        ``backend.properties().rf_calibration.gate_solutions.struct``.
        See :class:`GateSolutions`. The index is a tuple of the sorted ion indices.
    * gate calibrations are specified as a Pandas dataframe, accessed as above at
        ``backend.properties().rf_calibration.gate_tweaks.struct``.
        See :class:`GateCalibrations`.
    * The ion indices to use can be any length, but must be accessible in the
        gate_solutions accordingly. They are by default the first argument to the
        decorated function, can be changed with ``ion_index_argument``.
    * TODO: change. This assumes that the gate solutions have duration specified in
        us (microseconds) and rabi frequencies specified in MHz.

    Args:
        solution_types (typing.Set[common_types.XXModulationType]): The valid types
            of Gate Solutions that this function can retrieve.
            In other words, checks that the retrieved gate solution matches the
            type that the decorated function is designed to handle.
            (e.g. AM gate solution shouldn't be handled by an FM function)
        solution_arguments (typing.Dict[str, str]): Dictionary from decorated
            function argument names to the solution field. Valid values are:
            ``{"detuning", "sign", "segments", "angle"}``.
            E.g.: ``{"gate_detuning": "detuning"}``.
        convert_ions_to_slots (bool, optional): Whether the ion indices should
            be converted to AWG slot notation before looking up in the Gate Solutions
            data structure. Provided for backwards-compatibility with RFCompiler
            gate solutions. Defaults to False.
        ion_index_argument (int, optional): Index in the arguments for the ion
            indices to use for looking up the gate solution. I recommend it
            always being the first argument. Defaults to 0.
        convert_solution_units (bool, optional): Whether to convert from old
            RFCompiler-era units to standard SI units, or not. If True, assumes
            that times are given in microseconds, and frequencies in MHz.
            Defaults to False.

    Example:

    ```python
    @get_gate_solution({XXModulationType.AM_segmented}, {"gate_detuning": "detuning"})
    def my_gate_function(ions, gate_detuning) -> Schedule:
        with qp.build(qp.active_backend()) as return_schedule:
            # Do something
            pass

        return return_schedule

    with qp.build(backend_with_solutions) as schedule:
        qp.call(my_gate_function((0, 1)))
    ```
    """

    def _decorator_get_values(function):
        @functools.wraps(function)
        def _get_gate_solution(*args, **kwargs):
            """Retrieve the gate solution values, and pass to the decorated function."""
            # Get the current backend. Look in kwargs first
            backend = kwargs.get("backend")
            if backend is None:
                backend = qp.active_backend()

            # do nothing if all parameters to overwrite are already specified
            try:
                default_function_binding = inspect.signature(function).bind_partial(
                    *args, **kwargs
                )
            except TypeError as err:
                if "backend" in str(err):
                    kwargs_no_backend = kwargs.copy()
                    kwargs_no_backend.pop("backend")
                    default_function_binding = inspect.signature(function).bind_partial(
                        *args, **kwargs_no_backend
                    )
                else:
                    raise err

            if set(default_function_binding.arguments.keys()) >= set(
                solution_arguments.values()
            ):
                return function(*args, **kwargs)

            # *** Check ion indices ***
            ion_indices = _normalize_ion_indices(
                args, ion_index_argument, convert_ions_to_slots, backend=backend,
            )

            # *** Retrieve gate solutions & change units ***
            gate_solutions = backend.properties().rf_calibration.gate_solutions.struct
            new_kwargs = {}
            solution = common_types.XXGateParams(**gate_solutions.loc[ion_indices, :])
            assert solution.XX_gate_type in solution_types
            # TODO: change solution generator to durations & frequencies in seconds/Hz
            global_adjustment_key = (
                gate_params.GateCalibrations.GLOBAL_CALIBRATION_SLOT
                if convert_ions_to_slots
                else "global:TODO"
            )
            duration = _get_calibrated_gate_parameter(
                ion_indices,
                "XX_duration_us",
                backend,
                global_adjustment_key=global_adjustment_key,
            )
            amplitudes = np.array(solution.XX_amplitudes)
            if convert_solution_units:
                # Convert values from MHz/us to Hz/s
                duration *= 1e-6
                amplitudes *= 1e6
            output_segments = list(
                zip(
                    itertools.repeat(duration / len(amplitudes), len(amplitudes)),
                    amplitudes.tolist(),
                )
            )
            gate_solution_dict = {
                "detuning": solution.XX_detuning,
                "sign": solution.XX_sign,
                "segments": output_segments,
                "angle": solution.XX_angle,
                "type" : solution.XX_gate_type
            }

            # *** Set Gate solutions to override unspecified arguments ***
            # inspecting here just makes sure that we're not overriding any
            # arguments explicitly specified in the function call.
            func_name = inspect.unwrap(function).__name__
            for argument, solution_key in solution_arguments.items():
                # don't override any explicitly set kwargs
                if (
                    argument not in kwargs.keys()
                    and argument not in default_function_binding.arguments
                ):
                    value = gate_solution_dict[solution_key]
                    _LOGGER.debug(
                        "Replacing argument %s with gate solution value %s "
                        "(solution key: %s, function: %s(args=%s, kwargs=%s))",
                        argument,
                        value,
                        solution_key,
                        func_name,
                        args,
                        kwargs,
                    )
                    new_kwargs.setdefault(argument, value)

            bound_func = inspect.signature(function).bind_partial(
                *args, **new_kwargs, **kwargs
            )
            return function(*bound_func.args, **bound_func.kwargs)

        return _get_gate_solution

    return _decorator_get_values


def get_gate_parameters(
    gate_parameters: typing.Dict[str, str],
    convert_ions_to_slots: bool = False,
    ion_index_argument: int = 0,
    rescale_amplitude: bool = False,
):
    """
    Retrieve gate parameters from the backend, and pass them to the decorated function.

    Args:
        gate_parameters (typing.Dict[str, str]): Dictionary with gate parameters to
            retrieve. Keys are the argument of the decorated function to fill in,
            and the values are the keys (columns) of the GateCalibration to retrieve.
            See :class:`GateCalibrations` for details.
        convert_ions_to_slots (bool, optional): Whether ion indices should be
            converted from center-index notation to RFCompiler/AWG "slot" index
            for lookup in the :class:`GateCalibration`. Defaults to False.
        ion_index_argument (int, optional): Which index in the arguments that
            the ion indices reside in. Defaults to 0 (i.e. first).
        rescale_amplitude (bool, optional): Temporary argument. Whether amplitude
            values looked up should be rescaled from [0, 1000] to [0, 1.0].
            Defaults to False.

    Example:

    ```python
    @get_gate_parameters({"glob_amp": "global_amplitude"})
    def my_gate_function(ions, glob_amp: float) -> Schedule:
        with qp.build(qp.active_backend()) as return_schedule:
            qp.play(qp.Constant(1000, amp=glob_amp), qp.control_channels()[0])

        return return_schedule

    with qp.build(backend_with_solutions) as schedule:
        qp.call(my_gate_function((0, 1)))
    ```
    """

    def _decorator_get_values(function):
        @functools.wraps(function)
        def _get_gate_parameter(*args, **kwargs):
            """Retrieve the parameter values, and pass it to the decorated function."""
            # *** Check ion indices ***
            # TODO: enable short-circuiting if args already specified
            backend = kwargs.get("backend")
            ion_indices = _normalize_ion_indices(
                args, ion_index_argument, convert_ions_to_slots, backend
            )

            # TODO: change solution generator to durations & frequencies in seconds/Hz
            global_adjustment_key = (
                gate_params.GateCalibrations.GLOBAL_CALIBRATION_SLOT
                if convert_ions_to_slots
                else "global:TODO"
            )

            if "backend" in kwargs.keys():
                backend = kwargs.get("backend")
            else:
                backend = qp.active_backend()

            # *** Set Gate solutions to override unspecified arguments ***
            # inspecting here just makes sure that we're not overriding any
            # arguments explicitly specified in the function call.
            new_kwargs = {}
            default_function_binding = inspect.signature(function).bind_partial(
                *args, **kwargs
            )
            func_name = inspect.unwrap(function).__name__
            for argument, parameter_name in gate_parameters.items():
                # don't override any explicitly set kwargs
                if (
                    argument not in kwargs.keys()
                    and argument not in default_function_binding.arguments
                ):
                    value = _get_calibrated_gate_parameter(
                        ion_indices,
                        parameter_name,
                        backend,
                        global_adjustment_key=global_adjustment_key,
                    )
                    if rescale_amplitude and parameter_name in {"global_amplitude"}:
                        value /= 1000.0
                    _LOGGER.debug(
                        "Replacing argument %s with gate parameter value %s "
                        "(parameter name: %s, function: %s(args=%s, kwargs=%s))",
                        argument,
                        value,
                        parameter_name,
                        func_name,
                        args,
                        kwargs,
                    )
                    new_kwargs.setdefault(argument, value)

            bound_func = inspect.signature(function).bind_partial(
                *args, **new_kwargs, **kwargs
            )
            return function(*bound_func.args, **bound_func.kwargs)

        return _get_gate_parameter

    return _decorator_get_values


def ignore_arguments(
    empty_args: typing.Set[str] = None,
    remove_kwargs: typing.Set[str] = None,
    default_arg_value=None,
):
    """Modify functions to fill in unused arguments, and to remove undesired kwargs.

    Any parameter in ``empty_args`` will be set to ``default_arg_value`` (e.g. ``None``),
    if not otherwise specified.
    Any keyword argument listed in ``remove_kwargs`` will be removed when calling
    the decorated function.

    **IMPORTANT NOTE**: For this decorator to work properly with ``empty_args``,
    you might need to explicitly separate your positional args from your kwargs using
    ``*``.
    Example:
    ```python
    @ignore_arguments({"ignore"})
    def my_gate_function(a, ignore, b, *, kw=None):
        pass
    ```

    This function is useful when e.g. Qiskit expects to parametrize a gate with an
    angle, but you don't accept that angle/argument. Or it can be useful with a function
    that doesn't need a backend kwarg, but some level of the stack fills the backend
    information in automatically.
    """
    if empty_args is None:
        empty_args = set()
    if remove_kwargs is None:
        remove_kwargs = set()

    # TODO: extend arguments to function w/ remove_kwargs so that other decorators pass
    def _decorator_ignore_arguments(function):
        @functools.wraps(function)
        def _ignore_argument(*args, **kwargs):
            for a in remove_kwargs:
                kwargs.pop(a, None)
            sig = inspect.signature(function)
            sig_arg_keys = list(sig.parameters.keys())

            # for any arg in empty_arg, fill its position with default_arg_value
            num_positional_args = len(
                [
                    v
                    for v in sig.parameters.values()
                    if v.kind
                    in {
                        # pylint: disable=protected-access
                        inspect._ParameterKind.POSITIONAL_ONLY,
                        inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
                    }
                ]
            )
            for a in empty_args:
                if a in sig_arg_keys and len(args) < num_positional_args:
                    args_as_list = list(args)
                    args_as_list.insert(sig_arg_keys.index(a), default_arg_value)
                    args = tuple(args_as_list)
            bound_func = sig.bind_partial(*args, **kwargs)
            return function(*bound_func.args, **bound_func.kwargs)

        return _ignore_argument

    return _decorator_ignore_arguments

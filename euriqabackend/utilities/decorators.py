"""Decorators for backend functions."""
import cProfile
import datetime
import functools
import logging
import time
import typing

from .profiling import print_profiler_stats

_LOGGER = logging.getLogger(__name__)


def _generate_method_signature(*args, **kwargs):
    args_repr = [repr(a) for a in args]
    kwargs_repr = ["{}={}".format(repr(k), repr(v)) for k, v in kwargs.items()]
    signature = ", ".join(args_repr + kwargs_repr)
    return signature


# todo: add ability to change logging level


def debug(function):  # noqa: D202
    """Log the function call and return value."""

    @functools.wraps(function)
    def wrapper_debug(*args, **kwargs):
        signature = _generate_method_signature(args, kwargs)
        _LOGGER.debug("Calling %s(%s)", function.__name__, signature)
        value = function(*args, **kwargs)
        _LOGGER.debug("%s returned %s", function.__name__, repr(value))
        return value

    return wrapper_debug


def timed_method(wait_time_seconds):  # noqa: D202
    """Decorate a method to take AT LEAST specified amount of time.

    Uses `time.sleep()` internally, so it does not multi-thread and does not
    guarantee maximum amount of execution time.

    Saves start of last function in `self.method_called_times[method.__name__]`
    """

    def _timer_decorator(method):
        @functools.wraps(method)
        def _wait_till_done(self, *args, **kwargs):
            try:
                self.method_called_times[method.__name__] = datetime.datetime.now()
            except AttributeError:
                self.method_called_times = {method.__name__: datetime.datetime.now()}
            retval = method(self, *args, **kwargs)

            # wait until time elapsed
            sleep_time = (
                self.method_called_times[method.__name__]
                + datetime.timedelta(seconds=wait_time_seconds)
                - datetime.datetime.now()
            )
            _LOGGER.debug(
                "Sleeping in %s for %f seconds",
                method.__name__,
                sleep_time.total_seconds(),
            )
            time.sleep(sleep_time.total_seconds())
            # TODO: Make sure time.sleep() works with ARTIQ. Or make it only run on host
            _LOGGER.info(
                "Timed function took %f/%f seconds to execute",
                (
                    datetime.datetime.now() - self.method_called_times[method.__name__]
                ).total_seconds(),
                wait_time_seconds,
            )
            return retval

        return _wait_till_done

    return _timer_decorator


def timer(function):
    """Decorate a function to time function runtime and log it at debug level."""

    @functools.wraps(function)
    def _timer(*args, **kwargs):
        start = datetime.datetime.now()

        try:
            return function(*args, **kwargs)
        finally:
            end = datetime.datetime.now()
            _LOGGER.debug("%s: took %s", function.__name__, end - start)

    return _timer


def profiler(function):
    """Decorate a function to determine how long function calls take."""

    @functools.wraps(function)
    def _profiler(*args, **kwargs):
        func_profiler = cProfile.Profile()

        try:
            func_profiler.enable()
            return function(*args, **kwargs)
        finally:
            func_profiler.disable()
            print_profiler_stats(func_profiler)

    return _profiler


def default_from_object(
    obj: typing.Any, kwarg_to_obj_map: typing.Dict[str, str], separator: str = "."
):
    """Pull the default values for parameters from ``obj``.

    ``obj`` should hold all parameters that you are looking for. Right now,
    we assume that ``obj`` is an object where all parameters can be accessed
    with :func:``getattr`` (i.e. ``obj.val``), but we would like to add support for
    pandas or other objects.

    ``kwarg_to_obj_map`` should be a dictionary from kwarg names (strings) to the
    corresponding value in ``obj``. E.g. ``{"kwarg1": "val"} is similar to
    ```python
    def func_name(kwarg1= obj.val):
        pass
    ```

    Args:
        obj (typing.Any): object to pull values from
        kwarg_to_obj_map (typing.Dict[str, str]): Mapping from kwarg name to
            attribute in ``obj`` to populate the default kwarg value with.
        separator (str, optional): Separator in ``kwarg_to_obj_map`` values.
            Used to determine where to split to determine hierarchy.
            Defaults to ".".
            e.g. ``{"kwarg1": "lvl1.lvl2"}`` maps to ``obj1.lvl1.lvl2``

    Returns:
        function: wrapped function with default arguments populated from ``obj``.

    """
    # Handle arguments to decorator here, and pass them down to actual wrapper
    def _decorator_default_from_obj(function):
        @functools.wraps(function)
        def _wrap_pull_defaults(*args, **kwargs):
            """Pull default kwargs from ``obj``."""
            for kw in kwarg_to_obj_map.keys():
                # if kwarg is not defined or invalid, then overwrite from obj
                old_value = kwargs.get(kw, None)
                if old_value is None or old_value == -1 or old_value == "":
                    mapped_key = kwarg_to_obj_map[kw].split(separator)
                    new_value = obj
                    for level in mapped_key:
                        # TODO: handle pandas objects
                        new_value = getattr(new_value, level)
                    _LOGGER.debug("Set default value for '%s' to: %s", kw, new_value)
                    kwargs[kw] = new_value

            return function(*args, **kwargs)

        return _wrap_pull_defaults

    return _decorator_default_from_obj

"""Test decorators written for `euriqabackend`."""
import datetime

import pytest

from euriqabackend.utilities.decorators import _generate_method_signature
from euriqabackend.utilities.decorators import debug
from euriqabackend.utilities.decorators import default_from_object
from euriqabackend.utilities.decorators import profiler
from euriqabackend.utilities.decorators import timed_method
from euriqabackend.utilities.decorators import timer
from euriqabackend.utilities.profiling import print_profiler_stats


def test_decorator_debug(caplog):
    """Test the `@debug` decorator logs messages correctly."""
    # test params
    return_val = 5
    args = ("fake_arg1", "fake_arg2", "fake_arg3")
    kwargs = {"kwarg1": 1, "kwarg2": 2}

    signature = None

    @debug
    def _sig_gen_func(*args, **kwargs):
        nonlocal signature
        signature = _generate_method_signature(args, kwargs)
        return return_val

    _sig_gen_func(*args, **kwargs)

    print(caplog.record_tuples)
    print(caplog.records)

    for record in caplog.records:
        assert record.module == "decorators"
        assert record.levelname == "DEBUG"
        assert "_sig_gen_func" in record.message

    # check that the signature is in the first message, and the return value in last.
    assert signature in caplog.records[0].message
    assert repr(args) in caplog.records[0].message  # check that args are in signature
    assert repr(kwargs) in caplog.records[0].message
    assert str(return_val) in caplog.records[-1].message


def test_decorator_timed_method():
    """Test the `@timed_method` takes at least desired time and saves elapsed time."""
    sleep_time_seconds = 0.5

    class _FakeClass:
        @timed_method(sleep_time_seconds)
        def noop_method(self):
            pass

    obj = _FakeClass()
    with pytest.raises(AttributeError):
        # Should fail because variable method_called_times not declared.
        print("Time points available on init: {}".format(obj.method_called_times))

    start_time = datetime.datetime.now()
    obj.noop_method()
    end_time = datetime.datetime.now()

    print(obj.method_called_times)
    # check that it waited at least as long as it should have.
    assert end_time - start_time >= datetime.timedelta(seconds=sleep_time_seconds)
    # print(obj.method_called_times["noop_method"] - start_time))
    # Check that start times align.
    assert obj.method_called_times["noop_method"] - start_time <= datetime.timedelta(
        milliseconds=1
    )


def test_decorator_profiler(capsys):
    """Test the `@profiler` correctly profiles a function."""
    import cProfile
    import difflib

    def _function():
        v = []
        for i in range(1000):
            v.append(i)

    @profiler
    def _profiled_function():
        _function()

    def _unprofiled_function():
        _function()

    # run manual profile
    manual_profile = cProfile.Profile()
    manual_profile.enable()
    _unprofiled_function()
    manual_profile.disable()
    print_profiler_stats(manual_profile)
    manual_profile_output = capsys.readouterr().out

    # run decorated function
    _profiled_function()
    decorator_profile_output = capsys.readouterr().out

    def _similarity(a: str, b: str) -> float:
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        print("Profiler similarity ratio: {:.2f}".format(ratio))
        return ratio

    # check outputs are practically identical
    assert _similarity(manual_profile_output, decorator_profile_output) > 0.9


def test_decorator_timer(caplog):
    """Test the `@timer` gives accurate time and prints to the correct logger."""

    @timer
    def _timed_function():
        x = []
        return [x.insert(0, i) for i in range(10000)]

    start = datetime.datetime.now()
    _timed_function()
    end = datetime.datetime.now()

    manual_time = end - start

    print(caplog.records[0].args)
    assert "_timed_function" in caplog.records[0].args

    decorator_time_elapsed = caplog.records[0].args[1]
    assert decorator_time_elapsed > datetime.timedelta(milliseconds=10)
    assert (manual_time - decorator_time_elapsed) <= datetime.timedelta(milliseconds=1)


def test_decorator_default_args():
    """Test the `@default_from_object` decorator accurately pulls values."""

    class _Empty:
        pass

    default_object = _Empty()
    default_object.a = _Empty()
    default_object.b = 2
    default_object.a.third = "default_obj_string"

    with pytest.raises(TypeError):
        # Should fail if no arguments passed
        @default_from_object
        def _do_nothing():
            pass

        _do_nothing()

    kwarg_map = {"val1": "b", "val2": "a.third"}

    @default_from_object(default_object, kwarg_map)
    def _return_values(*args, **kwargs):
        # import logging
        # _LOGGER = logging.getLogger(__name__)
        # _LOGGER.debug("Args: %s, Kwargs: %s", args, kwargs)
        return args, kwargs

    default_arg_value = 1
    a, k = _return_values(default_arg_value, val1="overwrite_default", val2=None)
    assert a == (default_arg_value,)
    assert k == {"val1": "overwrite_default", "val2": "default_obj_string"}

    @default_from_object(default_object, kwarg_map)
    def _default_function(val1=None, val2="default_str", val3="some_string"):
        return val1, val2, val3

    v1, v2, v3 = _default_function(val1=5)
    assert v1 == 5
    assert v2 == "default_obj_string"
    assert v3 == "some_string"

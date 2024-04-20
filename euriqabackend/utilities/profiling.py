"""Utilities for profiling functions and writing output."""
import cProfile
import io
import profile
import pstats
from typing import Sequence
from typing import Union


def print_profiler_stats(
    profiler: Union[cProfile.Profile, profile.Profile],
    sort: str = "cumulative",
    restrictions: Sequence[Union[int, float, str]] = None,
) -> None:
    """
    Print profiler results to stdout.

    Args:
        profiler (Union[cProfile.Profile, profile.Profile]):
            profiler that has completed its run (been disabled)
        sort (str, optional):
            Defaults to "cumulative". How to sort the output.
            See :meth:`pstats.Stats.sort_stats`.
        restrictions (Sequence[Union[int, float, str]], optional):
            Restrictions on printing stats. Defaults to top 10%.
            See :meth:`pstats.Stats.print_stats`.

    Returns:
        None

    """
    if restrictions is None:
        restrictions = [0.1]
    string_stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=string_stream)
    ps.strip_dirs()
    ps.sort_stats(sort)
    ps.print_stats(*restrictions)
    print(string_stream.getvalue())

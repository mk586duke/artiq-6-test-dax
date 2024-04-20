"""
Store and fit experiment data to functions.

Inspired by NIST Stylus's :class:`Model`.


Notes:
    * WORK IN PROGRESS (9/17/2018).
        Just writing down some notes on what I would like to have.

"""
import abc
import typing


class DataModel(abc.ABC):
    """Data class for storing, retrieving, and fitting experiment data."""

    # todo: fill in docstring

    def fit(self, fit_function: typing.Callable = None):
        """Fit the internal dataset to a function (either provided or pre-set)."""
        # todo: figure out what fit_function should return? just f(x), i.e. sin(x)?
        # todo: figure out type of fit_function
        # todo: fill in body
        pass

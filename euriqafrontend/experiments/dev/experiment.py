"""
Base class for all EURIQA experiments on ARTIQ.

Key features:
    * Restoring experiment state after switching context to a different
        experiment or task.

Notes:
    * WORK IN PROGRESS (9/17/2018).
        Just writing down some notes on what I would like to have.

"""
import abc
import logging
import typing

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan

import euriqabackend

_LOGGER = logging.getLogger(__name__)
ArgumentRecord = typing.NamedTuple(
    "Argument", [("name", str), ("group", str), ("value", typing.Any)]
)


class EuriqaExperiment(artiq_env.EnvExperiment, abc.ABC):
    """
    The default experiment for all EURIQA experiments.

    Inspired by NIST STYLUS experiment. Experiments run on top of the ARTIQ backend.
    """

    # ***** Utility Functions
    __devices_used = set()

    def get_device(self, key: str):
        """Override to record devices used."""
        self.__devices_used.add(key)
        return super().get_device(key)

    __arguments_used = set()

    def get_argument(self, key: str, processor, group: str = None, tooltip: str = None):
        """Override to record arguments used."""
        arg = super().get_argument(key, processor, group=group, tooltip=tooltip)
        if isinstance(arg, artiq_scan.ScanObject):
            descriptor = tuple(arg.describe().items())
        else:
            descriptor = arg
        self.__arguments_used.add(ArgumentRecord(key, group, descriptor))
        return arg

    # NOTE: datasets already logged
    def _log_config(self):
        """Log all config settings to HDF record."""
        self.set_dataset("_devices_used", str(self.__devices_used))
        self.set_dataset("_arguments_used", str(self.__arguments_used))
        local_variables = list(
            filter(
                lambda i: not i.startswith("__") and not callable(getattr(self, i)),
                dir(self),
            )
        )
        local_values = list(str(repr(getattr(self, name))) for name in local_variables)
        self.set_dataset(
            "_class_variables", str(list(zip(local_variables, local_values)))
        )
        self.set_dataset("_euriqa_version", euriqabackend.__version__)

    def prepare(self):
        """MUST CALL THIS instead of overriding.

        NOTE: Should be called at the end of your method with `super().prepare()`.
        """
        return_val = super().prepare()
        self._log_config()
        return return_val

    # ***** EXPERIMENT TYPE VARIABLES *****
    _FEATURES = set()  # add features to this in Mixin.__init__

    def has_feature(self, feature_name: str) -> typing.Union[str, None]:
        """
        Check if experiment has the requested feature.

        Args:
            feature_name (str): Name of feature to search for. Can be partial name.

        Raises:
            ValueError: feature_name must be at least 2 characters to perform lookup,
                and be unique among features

        Returns:
            str: name of feature, or None if no feature found.

        """
        _min_feature_length = 2

        # check for exact match
        if feature_name in self._FEATURES:
            return feature_name

        # reject too vague feature names
        if len(feature_name) <= _min_feature_length:
            raise ValueError(
                "Requested feature '{}' is too short to search ({} < {})".format(
                    feature_name, len(feature_name), _min_feature_length
                )
            )

        feature_name = feature_name.lower()  # normalize

        # check if feature_name is a substring of any feature
        feature_matches = list(
            (feature for feature in self._FEATURES if feature_name in feature.lower())
        )
        if len(feature_matches) > 1:
            raise ValueError(
                "Feature '{}' was not specific enough."
                "Matched too many features: {}".format(feature_name, feature_matches)
            )
        if len(feature_matches) == 1:
            return feature_matches[0]

        return None  # default

    # @abc.abstractmethod()
    # def run(self):
    #     return

    # ***** EXPERIMENT SEQUENCING METHODS *****
    # these methods enforce the proper order of an experiment

    # ***** EXPERIMENT STATE METHODS *****
    # methods to save/restore experiment state

    # @abc.abstractmethod
    # def restore_experiment_state(self):
    #     """Return the experiment & any devices to where they can be used again."""
    #     pass

    # @abc.abstractmethod
    # todo: some sort of pause method
    # todo: some sort of safe-experiment (put experiment in a safe state)??
    #   Or implement as "default experiment"?

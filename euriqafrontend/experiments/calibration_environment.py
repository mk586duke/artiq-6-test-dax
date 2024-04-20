"""ARTIQ Environment specialization meant to encapsulate Calibrations.

Saves calibration data to specific dataset prefixes, and implements
checks on overwriting calibration values.

In the same style as :class:`artiq.language.environment.HasEnvironment`
"""
import abc
import logging
import time
import typing

import artiq.language.environment as artiq_env

import euriqafrontend.repository.basic_environment as basic_environment

_LOGGER = logging.getLogger(__name__)


class CalibrationEnvironment(abc.ABC, basic_environment.BasicEnvironment):
    """A base Environment type that Calibrations should descend from.

    An example of a Calibration experiment should be:
    ```python
    class NewCalibration(CalibrationEnvironment, artiq.language.environment.Experiment):
        pass
    ```
    """

    # pylint: disable=abstract-method

    # *** VARIABLES ***
    _CAL_DATA_KEY = "calibration.{type}.{value_name}"
    _CAL_ARG_GROUP = "Calibration Settings"

    @property
    def cal_type(self) -> str:
        """Name of the calibration type, used for dataset naming."""
        try:
            assert isinstance(self._CAL_NAME, str)
            return self._CAL_NAME
        except AttributeError:
            _LOGGER.debug("No _CAL_NAME var. Using class name instead")
            return self.__class__.__name__
        except AssertionError:
            _LOGGER.error(
                "Invalid type of _CAL_NAME: %s, should be string", type(self._CAL_NAME)
            )
            return "InvalidCalType"

    # *** METHODS ***

    def build(self):
        """Add Calibration-specific arguments & devices.

        SHOULD/MUST be called with ``super().build()``.
        """
        super().build()
        if not hasattr(self, "scheduler"):
            self.setattr_device("scheduler")

        # add Calibration settings
        self.do_save_calibration_to_dataset = bool(
            self.get_argument(
                "Save Calibration Data to Global Dataset?",
                artiq_env.BooleanValue(default=False),
                group=self._CAL_ARG_GROUP,
                tooltip="Don't use if you are running a test calibration or unsure if "
                "this calibration will work",
            )
        )
        self.do_reschedule_calibration = bool(
            self.get_argument(
                "Auto-Reschedule this calibration when finished?",
                artiq_env.BooleanValue(default=False),
                group=self._CAL_ARG_GROUP,
            )
        )
        self.cal_reschedule_period_seconds = int(
            self.get_argument(
                "Period between calibration experiments (ONLY if auto-reschedule)",
                artiq_env.NumberValue(default=5, unit="s", step=1, min=5, ndecimals=0),
                group=self._CAL_ARG_GROUP,
                tooltip="Ignored if auto-reschedule is disabled",
            )
        )

    def set_calibration(
        self,
        data_key: str,
        data: typing.Any,
        check_validity: bool = True,
        validity_tolerance: float = 0.05,
    ):
        """Change a calibration data value.

        Args:
            data_key (str): name of the dataset to change (will be modified slightly)
            data (typing.Any): Data to write to data_key
            check_validity (bool, optional): Whether the value of the calibration that
                you are writing should be checked for validity. Defaults to True.
        """
        calibration_data_name = self._CAL_DATA_KEY.format(
            type=self.cal_type, value_name=data_key
        )
        try:
            old_cal_data = self.get_dataset(calibration_data_name)
        except KeyError:
            _LOGGER.warning(
                "Old Calibration data for %s does not exist. Cannot check new value "
                "for validity",
                calibration_data_name,
            )
            check_validity = False
        if check_validity:
            if isinstance(old_cal_data, (int, float)):
                assert abs(data - old_cal_data) <= abs(
                    validity_tolerance * old_cal_data
                )
            elif isinstance(old_cal_data, str):
                raise NotImplementedError
            else:
                raise NotImplementedError
            # TODO: add more types as needed.

        _LOGGER.debug(
            "%s save calibration value '%s'='%s' to master",
            "Did" if self.do_save_calibration_to_dataset else "Didn't",
            calibration_data_name,
            data,
        )
        self.set_dataset(
            calibration_data_name,
            data,
            persist=self.do_save_calibration_to_dataset,
            archive=True,
        )

    def reschedule_self(
        self, advance_time_seconds: float = None, due_date: float = None
    ):
        """Reschedule this experiment in the future.

        I recommend calling this from the analyze/end of run stage, so that you don't
        run into an infinite loop of trying to schedule this experiment.
        That is, if you reschedule at beginning of run for 30 seconds in future,
        and your experiment takes 60 seconds to run, then you will only keep
        performing this experiment without yielding to other experiments.

        NOTE: this works properly when called with no arguments, and will
        automatically reschedule the same experiment for NOW.

        Args:
            advance_time_seconds (float): Time in advance of now to schedule,
                in seconds. Mutually exclusive with :arg:`due_date`.
            due_date (float): time at which to schedule the next experiment.
                Mutually exclusive with :arg:`advance_time_seconds`.
                Should be a time in the format like :func:`time.time`
        """
        # NOTE: giving no args is allowed, will just reschedule self immediately
        if advance_time_seconds is not None and due_date is not None:
            return ValueError(
                "Set both inputs, which is a conflict. Choose one or the other"
            )
        if advance_time_seconds is not None:
            due_date = time.time() + advance_time_seconds
        self.scheduler.submit(due_date=due_date)

    def analyze(self):
        """Reschedule this calibration experiment when finished, if set."""
        # NOTE: happens first so that slow analysis functions don't delay reschedule.
        if self.do_reschedule_calibration:
            self.reschedule_self(
                advance_time_seconds=self.cal_reschedule_period_seconds
            )
        super().analyze()

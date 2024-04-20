"""Possible experiment priority levels for standard experiments/calibrations."""
import enum


class ExperimentPriorities(enum.IntEnum):
    """Priority mapping for EURIQA experiments.

    Higher integer is higher priority. Left some space for expansion.
    """

    CALIBRATION_CRITICAL = 10
    CALIBRATION_PRIORITY1 = 9
    CALIBRATION_PRIORITY2 = 8
    CALIBRATION = 7
    CALIBRATION_BACKGROUND = 6
    EXPERIMENT_RESCHEDULE = 5
    HIGH_PRIORITY_EXPERIMENT = 3
    EXPERIMENT = 1
    BACKGROUND_EXPERIMENT = 0

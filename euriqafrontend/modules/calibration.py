import logging
import artiq.language.environment as artiq_env
import sipyco.pyon as pyon
import datetime
import copy
import h5py
import typing
import pathlib
import sys
import os
import inspect
from euriqabackend import _EURIQA_LIB_DIR
from euriqafrontend import EURIQA_NAS_DIR
from artiq.language.core import (
    host_only,
    kernel,
    rpc,
    TerminationRequested,
)
from euriqafrontend.scheduler import ExperimentPriorities as priorities

_LOGGER = logging.getLogger(__name__)
_NAS_PATH = EURIQA_NAS_DIR / "CompactTrappedIonModule"
_LOG_PATH = _NAS_PATH / "ARTIQ" / "auto_calibration_log"
_MODULE_PATH = _NAS_PATH / "ARTIQ" / "auto_calibration_modules_swap"
import yaml

def log_calibration(**entries):
    record = dict()
    title = str(datetime.datetime.now().time())
    record[title] = entries
    _LOG_PATH.mkdir(parents=True, exist_ok=True)
    file_name = str(datetime.date.today()) + ".yaml"
    with _LOG_PATH.joinpath(file_name).open("a") as log_file:
        yaml.dump(record, log_file)

class CalibrationModule(artiq_env.HasEnvironment):
    def build(self, **kwargs):
        self.scheduler = kwargs["scheduler"]
        self.calibration_type = kwargs["calibration_type"]
        file=kwargs["file"]
        self.submitter = Submitter(self.scheduler,file)

    def update_dataset(
        self, dataset_name: str, value: typing.Any, message: str = None, **kwargs
    ):
        """Wrapper around ``set_dataset()``, logs the value when updating."""
        log_calibration(type=self.calibration_type, message=message, value=float(value))
        _LOGGER.info("Updating %s to %E", dataset_name, value)
        self.set_dataset(dataset_name, value, **kwargs)

    def update_dataset_if_valid(
        self,
        dataset_name: str,
        last_value: float,
        correction: float,
        max_correction: float,
        error_dataset: str,
    ):
        """Check if an update is within allowed range. Update if allowed, warn if not."""
        if abs(correction) > max_correction:
            _LOGGER.warning(
                "Out-of-range correction calculated. Would have set '%s' (%E) -> (%E)",
                dataset_name,
                last_value,
                last_value + correction,
            )
            log_calibration(
                type=self.calibration_type,
                operation="Error: correction exceeded limit",
                value=float(correction),
            )
            self.update_dataset(
                error_dataset,
                1,
                message=f"Correction too large, manually verify the {dataset_name} setting",
            )
        else:
            # allowed value, update dataset.
            self.update_dataset(dataset_name, last_value + correction, persist=True)


class Submitter:
    """This submitter handles experiment submission. submit takes prototype name as argument.
    The priority is for the scheduled experiment. When hold is true, experiment artiq will wait for the submitted experiment to finish"""

    def __init__(self, artiq_scheduler, file):
        self.scheduler = artiq_scheduler
        file_list = os.listdir(str(_MODULE_PATH))
        self.modules = dict()
        # add experiments in this module. Do those first to allow prototypes to override
        this_file_relative = str(
            pathlib.Path(file).relative_to(
                _EURIQA_LIB_DIR / "euriqafrontend" / "repository"
            )
        )
        for name, _obj in inspect.getmembers(
            sys.modules[__name__],
            lambda obj: inspect.isclass(obj) and issubclass(obj, artiq_env.Experiment),
        ):
            self.modules[name] = {
                "class_name": name,
                "file": this_file_relative,
                "log_level": logging.INFO,
                "repo_rev": "N/A",
                "arguments": {},
            }
        # pull from experiment "prototypes"
        for file in file_list:
            temp = h5py.File(str(_MODULE_PATH / file), "r")
            name = file.split(".", 1)[0]
            self.modules[name] = pyon.decode(temp["expid"][()])

    @host_only
    def submit(
        self,
        experiment_prototype_name: str,
        priority: int = priorities.CALIBRATION,
        hold: bool = False,
        **kwargs,
    ):
        """Run a single step/calibration in the calibration sequence.

        Submits the calibration experiment with "calibration" priority, then
        waits for it to complete.

        Args:
            experiment_prototype_name (str): Name of the pre-run "experiment prototype"
                that you would like to run. Experiment prototypes are HDF5 files with
                an expid (experiment ID) that includes default arguments for an
                experiment, effectively letting you re-run a previous experiment
                without manually setting every argument.
            priority (int): priority level of the calibration to be run.
                Default priorities can be seen in :mod:`euriqafrontend.scheduler`.
            hold (boolean): indicate whether everything else should wait until this exp
                finishes.

        Kwargs:
            keyword arguments are passed as experiment arguments, for example
            ``num_shots=200``.

        WARNING: hold should only be set to TRUE at the scheduler level, do not set it to true at the experiment level.
        """
        exp_submit = copy.deepcopy(self.modules[experiment_prototype_name])

        for key in kwargs:
            exp_submit["arguments"][key] = kwargs[key]

        finished = False

        while not finished:

            rid = self.scheduler.submit(
                pipeline_name="main",
                expid=exp_submit,
                priority=int(priority),  # converts enum to int value if necessary
                due_date=None,
                flush=False,
            )

            # TODO: This might make calibrations run faster if can pre-prepare
            # wait until this experiment finishes
            finished = True

            if hold:
                keep_waiting = True
                while keep_waiting:
                    keep_waiting = False
                    task_list = self.scheduler.get_status()
                    for task in task_list:
                        if task_list[task]["priority"] > self.scheduler.priority:
                            if task != self.scheduler.rid:
                                keep_waiting = True

                    self.scheduler.pause()

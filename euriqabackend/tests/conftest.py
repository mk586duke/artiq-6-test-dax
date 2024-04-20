"""Pytest fixtures useful for all EURIQA or ARTIQ code."""
import itertools
import logging
import os
import pathlib
import sys

import pytest
from qiskit import __qiskit_version__
import qiskit.pulse as qp
from artiq.frontend.artiq_run import DummyScheduler
from artiq.language.environment import EnvExperiment
from artiq.master.databases import DatasetDB
from artiq.master.databases import DeviceDB
from artiq.master.worker_db import DatasetManager
from artiq.master.worker_db import DeviceError
from artiq.master.worker_db import DeviceManager

# from artiq.coredevice.core import CompileError

_LOGGER = logging.getLogger(__name__)


def _get_device_manager(root_path: pathlib.Path) -> DeviceManager:
    """Return an ARTIQ device manager based on a device database on the given path.

    Looks for device database at `root_path/device_db.py` and
    `root_path/device_db.pyon`. Fails test if `device_db.py[on]` not found.
    """
    device_db_path = root_path / "device_db.py"
    if not device_db_path.exists() or not device_db_path.is_file():
        device_db_path = root_path / "device_db.pyon"
        if not device_db_path.exists() or not device_db_path.is_file():
            pytest.fail(
                "device_db.py[on] does not exist in ARTIQ_ROOT = {}".format(root_path)
            )

    _LOGGER.debug("Starting ARTIQ Device DB at: %s", device_db_path)
    device_db = DeviceDB(device_db_path)
    device_manager = DeviceManager(
        device_db, virtual_devices={"scheduler": DummyScheduler()}
    )
    return device_manager


def _get_dataset_manager(root_path: pathlib.Path) -> DatasetManager:
    """Return an ARTIQ dataset manager based on a dataset database on the given path.

    Looks for dataset database at `root_path/dataset_db.py`.
    Fails test if `dataset_db.py` not found.
    """
    dataset_db_path = root_path / "dataset_db.pyon"
    if not dataset_db_path.exists() or not dataset_db_path.is_file():
        pytest.fail(
            "dataset_db.pyon does not exist in ARTIQ_ROOT = {}".format(root_path)
        )
    _LOGGER.debug("Starting ARTIQ Dataset DB at: %s", dataset_db_path)
    dataset_db = DatasetDB(dataset_db_path)
    dataset_manager = DatasetManager(dataset_db)
    return dataset_manager


@pytest.fixture(scope="session")
def artiq_experiment_run():
    """Yield factory to create and run an ARTIQ experiment."""
    artiq_root = os.getenv("ARTIQ_ROOT")
    if artiq_root is None:
        pytest.fail(
            "ARTIQ_ROOT must be defined on the command-line (e.g. set ARTIQ_ROOT=...)"
        )
    else:
        artiq_root = pathlib.Path(artiq_root).resolve()

    # Start device & dataset managers
    device_manager = _get_device_manager(artiq_root)
    dataset_manager = _get_dataset_manager(artiq_root)

    def _create_experiment(
        artiq_experiment_class: type(EnvExperiment), *args, **kwargs
    ) -> EnvExperiment:
        """Create the given ARTIQ experiment with given args, and prepare it."""
        try:
            experiment = artiq_experiment_class(
                (device_manager, dataset_manager, None), *args, **kwargs
            )
        except DeviceError as err:
            # fail if device database missing required device
            pytest.fail(
                "Missing test device in {}: `{}`".format(
                    device_manager.ddb.backing_file, *err.args
                )
            )

        experiment.prepare()
        return experiment

    def _create_and_run_experiment(
        artiq_experiment_class: type(EnvExperiment), *args, **kwargs
    ):
        """Create an ARTIQ experiment and run it."""
        experiment_id = {
            "file": sys.modules[artiq_experiment_class.__module__].__file__,
            "class_name": artiq_experiment_class.__name__,
            "arguments": dict(),
        }
        device_manager.virtual_devices["scheduler"].expid = experiment_id
        try:
            experiment = _create_experiment(artiq_experiment_class, *args, **kwargs)
            experiment.run()
            experiment.analyze()
        # except CompileError as error:
        #     # Reduce amount of text on terminal???
        #     raise error from None
        except Exception as exception:
            if hasattr(exception, "artiq_core_exception"):
                exception.args = "{}\n{}".format(
                    exception.args[0], exception.artiq_core_exception
                )
            raise exception
        return experiment

    yield _create_and_run_experiment

    device_manager.close_devices()


def pytest_assertrepr_compare(op: str, left: qp.Schedule, right: qp.Schedule):
    # not the best version checking, but it works
    if __qiskit_version__["qiskit-terra"] >= "0.17.0":
        schedule_types = (qp.Schedule, qp.ScheduleBlock)
    else:
        schedule_types = (qp.Schedule,)
    if (
        op == "=="
        and isinstance(left, schedule_types)
        and isinstance(right, schedule_types)
    ):
        different_instructions = []
        for i, (left_instr, right_instr) in enumerate(
            itertools.zip_longest(
                left.instructions, right.instructions, fillvalue="No instr"
            )
        ):
            if left_instr != right_instr:
                different_instructions.append(
                    f"Instr #{i}: {left_instr} != {right_instr}"
                )
        return [
            f"Qiskit schedules differ: {left} != {right}",
            f"Different channels: {set(left.channels) ^ set(right.channels)}",
        ] + different_instructions

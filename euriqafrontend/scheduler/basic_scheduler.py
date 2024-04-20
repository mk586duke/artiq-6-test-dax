"""Script to autoload & calibrate ion chain for further experiments.

TODO:
    * convert from experiment prototypes to some other system (YAML/PYON-like)
    * examine ARTIQ scheduler, figure out how to interface with that better
    * figure out how to schedule multiple experiments from different people
"""
import abc
import asyncio
import logging
import pathlib
import platform
import sys
import threading
import time
import typing

import h5py
import numpy as np
import pandas
import sipyco.pc_rpc as artiq_rpc
import sipyco.pyon as pyon

from euriqafrontend import EURIQA_NAS_DIR
from euriqabackend.databases.device_db_main_box import CONTROL_PC_IP_ADDRESS
from . import ExperimentPriorities as priority

_LOGGER = logging.getLogger(__name__)


def load_experiments(prototype_path: typing.Union[typing.AnyStr, pathlib.Path] = None):
    """Load 'prototype' experiment's arguments from a file folder.

    Catches all HDF5 files in the given folder, and copies their 'expid'
    (i.e. RID, path, arguments, etc).
    """
    assert type(prototype_path) in (
        type(None),
        str,
        pathlib.PosixPath,
        pathlib.WindowsPath,
    )
    if prototype_path is None:
        prototype_path = (
            EURIQA_NAS_DIR
            / "CompactTrappedIonModule"
            / "ARTIQ"
            / "experiment_prototypes"
        )

    if sys.version_info.major == 3 and sys.version_info.minor < 6:
        prototype_path = pathlib.Path(prototype_path).resolve()
    else:
        prototype_path = pathlib.Path(prototype_path).resolve(strict=True)
    experiments = dict()
    for exp in prototype_path.glob("*.h5"):
        file = h5py.File(str(exp), "r")
        name = exp.stem
        experiments[name] = {"expid": pyon.decode(file["expid"].value)}
    return experiments


def pull_global(
    arg_key: str, global_key: str, datasets: typing.Dict, experiments: typing.Dict
):
    """Update all local experiment arguments from a global dataset value."""
    global_val = datasets.get(global_key)
    for iexp in experiments:
        if arg_key in experiments[iexp]["expid"]["arguments"].keys():
            experiments[iexp]["expid"]["arguments"][arg_key] = global_val

    return experiments


def check_alive(schedule, experiments, datasets):

    center = datasets.get("global.Voltages.center")
    experiments["Check_center_15ions"]["expid"]["arguments"]["scan_range"]["start"] = (
        center - 2
    )
    experiments["Check_center_15ions"]["expid"]["arguments"]["scan_range"]["stop"] = (
        center + 2
    )
    experiments["Check_center_15ions"]["expid"]["arguments"]["set_globals"] = True

    rid = schedule.submit(
        pipeline_name="main",
        expid=experiments["Check_center_15ions"]["expid"],
        priority=0,
        due_date=None,
        flush=False,
    )

    while rid in schedule.get_status():
        time.sleep(0.1)
    ions = datasets.get("data.check.Center.fitparam_x0")
    if np.std(ions) < 0.75:
        alive = True
        datasets.set("global.Voltages.center", np.mean(ions))
    else:
        alive = False
        datasets.set("monitor.Lost_Ions", True)

    return alive


def run_calib(schedule, experiments, datasets):

    experiments["Check_radial_15ions"]["expid"]["arguments"]["lost_ion_monitor"] = False
    experiments["CheckDZ_15ions"]["expid"]["arguments"]["lost_ion_monitor"] = False
    experiments["Check_axial_15ions"]["expid"]["arguments"]["lost_ion_monitor"] = False

    axial = datasets.get("monitor.Modes.HighAxial")
    experiments["Check_axial_15ions"]["expid"]["arguments"]["do_SBC"] = True
    experiments["Check_axial_15ions"]["expid"]["arguments"]["scan_range"]["start"] = (
        axial - 2500
    )
    experiments["Check_axial_15ions"]["expid"]["arguments"]["scan_range"]["stop"] = (
        axial + 2500
    )

    rid = schedule.submit(
        pipeline_name="main",
        expid=experiments["Check_axial_15ions"]["expid"],
        priority=0,
        due_date=None,
        flush=True,
    )

    while rid in schedule.get_status():
        time.sleep(0.1)
    peak = datasets.get("data.raman_rabi_spec.fitparam_x0")
    datasets.set("monitor.Modes.HighAxial", peak[0])

    low_radial = datasets.get("monitor.Modes.LowRadial")
    experiments["Check_radial_15ions"]["expid"]["arguments"]["scan_range"]["start"] = (
        low_radial - 1500
    )
    experiments["Check_radial_15ions"]["expid"]["arguments"]["scan_range"]["stop"] = (
        low_radial + 1500
    )
    # experiments["Check_radial_15ions"]["expid"]["arguments"]["scan_range"]["start"] = 2914275 - 2000
    # experiments["Check_radial_15ions"]["expid"]["arguments"]["scan_range"]["stop"] = 2914275 + 2000
    experiments["Check_radial_15ions"]["expid"]["arguments"]["do_SBC"] = True

    rid = schedule.submit(
        pipeline_name="main",
        expid=experiments["Check_radial_15ions"]["expid"],
        priority=0,
        due_date=None,
        flush=True,
    )

    while rid in schedule.get_status():
        time.sleep(0.1)
    peak = datasets.get("data.awg_rabispec.fitparam_x0")
    datasets.set("monitor.Modes.LowRadial", peak[0])

    high_radial = datasets.get("monitor.Modes.HighRadial")
    experiments["Check_high_radial_15ions"]["expid"]["arguments"]["scan_range"][
        "start"
    ] = (high_radial - 1500)
    experiments["Check_high_radial_15ions"]["expid"]["arguments"]["scan_range"][
        "stop"
    ] = (high_radial + 1500)
    experiments["Check_high_radial_15ions"]["expid"]["arguments"]["scan_range"][
        "stop"
    ] = (high_radial + 1500)
    experiments["Check_high_radial_15ions"]["expid"]["arguments"]["do_SBC"] = True

    rid = schedule.submit(
        pipeline_name="main",
        expid=experiments["Check_high_radial_15ions"]["expid"],
        priority=0,
        due_date=None,
        flush=False,
    )

    while rid in schedule.get_status():
        time.sleep(0.1)
    peak = datasets.get("data.check.AWGSpec.fitparam_x0")
    datasets.set("monitor.Modes.HighRadial", peak[0])

    dz = datasets.get("monitor.DZ")
    experiments["CheckDZ_15ions"]["expid"]["arguments"]["scan_range"]["start"] = dz - 5
    experiments["CheckDZ_15ions"]["expid"]["arguments"]["scan_range"]["stop"] = dz + 5

    rid = schedule.submit(
        pipeline_name="main",
        expid=experiments["CheckDZ_15ions"]["expid"],
        priority=0,
        due_date=None,
        flush=True,
    )

    while rid in schedule.get_status():
        time.sleep(0.1)
    nulls = datasets.get("data.check.DZ.fitparam_x0")
    datasets.set("monitor.DZ", np.mean(nulls))


if __name__ == "__main__":

    # Load experiment prototypes from Q://
    base_dir = EURIQA_NAS_DIR / "CompactTrappedIonModule"
    prototype_path = base_dir / "ARTIQ/experiment_prototypes"

    experiments = load_experiments(prototype_path)
    print("Available experiments are:")
    for i in list(experiments.keys()):
        print(i)

    # Connect to the master
    master_ip = CONTROL_PC_IP_ADDRESS
    schedule, exps, datasets = [
        Client(master_ip, 3251, "master_" + i)
        for i in "schedule experiment_db dataset_db".split()
    ]

    # Run the main loop

    while True:
        repetition_period = 10 * 60  # 10 mins
        loop_start_time = time.monotonic()
        alive = check_alive(schedule, experiments, datasets)

        experiments = pull_global(
            "center (um)", "global.Voltages.center", datasets, experiments
        )
        if alive is False:
            while alive is False:
                print(
                    "{0}: Ions are dead".format(
                        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
                    )
                )
                rid = schedule.submit(
                    pipeline_name="main",
                    expid=experiments["Autoload_15ions"]["expid"],
                    priority=10,
                    due_date=None,
                    flush=False,
                )
                alive = check_alive(schedule, experiments, datasets)

        run_calib(schedule, experiments, datasets)
        sleep_time = repetition_period + loop_start_time - time.monotonic()
        time.sleep(sleep_time)

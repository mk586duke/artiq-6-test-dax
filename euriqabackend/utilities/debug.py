"""Utilities for debugging ARTIQ code or developing new devices."""
from artiq.master import databases as db
from artiq.master import worker_db


def setup_device_manager(device_db_file: str) -> worker_db.DeviceManager:
    """
    Create a DeviceManager for testing purposes.

    NOT compatible with commanding coredevices.

    Args:
        device_db_file (str): Path to a Device Database (e.g.
            :mod:`artiq.examples.master.device_db`)

    Returns:
        DeviceManager: A tool to allow you to access a named device
            without explicitly starting the device.
            Manages access & state of the devices.

    """
    return worker_db.DeviceManager(db.DeviceDB(device_db_file))


def setup_database_manager(dataset_db_file: str) -> worker_db.DatasetManager:
    """
    Create a DatasetManager for testing purposes.

    NOT compatible with commanding coredevices.

    Args:
        dataset_db_file (str): Path to a Dataset Database (e.g.
            :mod:`artiq.examples.master.dataset_db.pyon`)

    Returns:
        DatasetManager: A tool to allow you to access a named datasets

    """
    return worker_db.DatasetManager(db.DatasetDB(dataset_db_file))

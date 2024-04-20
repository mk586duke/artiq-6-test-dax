"""Helpers for retrieving the default ARTIQ RPC servers."""
import sipyco.pc_rpc as rpc
from artiq.master.databases import DatasetDB, DeviceDB
from artiq.master.worker_impl import Scheduler

_ARTIQ_MASTER_PORT = 3251


def get_master_client(
    client_name: str, master_ip: str, master_port: int = _ARTIQ_MASTER_PORT
) -> rpc.Client:
    try:
        return rpc.Client(master_ip, master_port, target_name=client_name)
    except TimeoutError as err:
        raise RuntimeError(
            f"Could not connect to ARTIQ master client '{client_name}' at {master_ip}:{master_port}"
        ) from err


def get_artiq_config(
    master_ip: str, master_port: int = _ARTIQ_MASTER_PORT
) -> rpc.Client:
    return get_master_client("master_config", master_ip, master_port)


def get_artiq_experiment_db(
    master_ip: str, master_port: int = _ARTIQ_MASTER_PORT
) -> rpc.Client:
    return get_master_client("master_experiment_db", master_ip, master_port)


def get_artiq_dataset_db(
    master_ip: str, master_port: int = _ARTIQ_MASTER_PORT
) -> DatasetDB:
    return get_master_client("master_dataset_db", master_ip, master_port)


def get_artiq_device_db(
    master_ip: str, master_port: int = _ARTIQ_MASTER_PORT
) -> DeviceDB:
    return get_master_client("master_device_db", master_ip, master_port)


def get_artiq_scheduler(
    master_ip: str, master_port: int = _ARTIQ_MASTER_PORT
) -> Scheduler:
    return get_master_client("master_schedule", master_ip, master_port)

"""Code for retrieving a Qiskit backend for building schedules for RFSoC."""
import pathlib
import typing

import pulsecompiler.rfsoc.structures.channel_map as rfsoc_mapping
import pulsecompiler.qiskit.backend as pulse_backend

import euriqafrontend.modules.rfsoc as rfsoc_module
import euriqafrontend.settings.calibration_box as calibration_box
import euriqafrontend.interactive.artiq_clients as artiq_clients
from euriqafrontend.settings import RF_CALIBRATION_PATH
from euriqafrontend import EURIQA_NAS_DIR


def get_calibration_box(
    filename: typing.Union[str, pathlib.Path], datasets: typing.Dict
) -> calibration_box.CalibrationBox:
    return calibration_box.CalibrationBox.from_json(
        filename=pathlib.Path(filename), dataset_dict=datasets
    )


def default_rf_calibration_path() -> pathlib.Path:
    return RF_CALIBRATION_PATH


def get_default_rfsoc_map() -> rfsoc_mapping.RFSoCChannelMapping:
    default_path = rfsoc_module.RFSOC._DEFAULT_RFSOC_DESCRIPTION_PATH
    return rfsoc_mapping.RFSoCChannelMapping.from_pyon_file(default_path)


def get_qiskit_backend(
    num_ions: int,
    rfsoc_map: rfsoc_mapping.RFSoCChannelMapping,
    calibrations: calibration_box.CalibrationBox,
    **kwargs,
) -> pulse_backend.MinimalQiskitIonBackend:
    return pulse_backend.MinimalQiskitIonBackend(
        num_ions,
        rfsoc_map,
        endcap_ions=(2, 2),
        properties=calibrations.to_backend_properties(num_ions),
        **kwargs,
    )


def get_default_qiskit_backend(
    master_ip: str, num_ions: int, with_2q_gate_solutions: bool = True
) -> pulse_backend.MinimalQiskitIonBackend:
    """Auto-populate several of the fields."""
    calibrations = get_calibration_box(
        default_rf_calibration_path(), artiq_clients.get_artiq_dataset_db(master_ip)
    )

    if with_2q_gate_solutions:
        # in order to use any two qubit gates in your circuit, this must be set to True
        # gate solutions are loaded here from the RF_CALIBRATION_PATH file
        try:
            # first try the raw path. If that doesn't exist, search in the EURIQA
            # NAS, and try checking subfolder for # of ions
            path_test = pathlib.Path(calibrations.gate_solutions.solutions_top_path)
            if not path_test.exists():
                path_test = EURIQA_NAS_DIR / path_test
            if (path_test / calibrations[f"gate_solutions.{num_ions}"]).exists():
                path_test = path_test / calibrations[f"gate_solutions.{num_ions}"]
            calibrations.merge_update(
                rfsoc_module.RFSOC._load_gate_solutions(path_test, num_ions,)
            )
        except KeyError as err:
            raise RuntimeError(
                f"Could not find gate solutions for {num_ions} ions in {path_test}"
            ) from err

    return get_qiskit_backend(
        num_ions, get_default_rfsoc_map(), calibrations, use_zero_index=True
    )

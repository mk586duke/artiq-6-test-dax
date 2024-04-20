import pathlib

import pulsecompiler.qiskit.backend as be
import pulsecompiler.rfsoc.structures.channel_map as rfsoc_mapping
import pytest
import sipyco.pyon as pyon

import euriqafrontend.settings.calibration_box as calibrations
import euriqafrontend.modules.rfsoc as rfsoc
import euriqabackend.devices.keysight_awg.gate_parameters as gate_params
from euriqabackend import _EURIQA_LIB_DIR


_live_rfsoc_backend_path = (
    pathlib.Path(__file__).parents[2] / "databases" / "rfsoc_system_description.pyon"
)
_rf_calibration_path = (
    _EURIQA_LIB_DIR / "euriqafrontend" / "settings" / "rf_calibration.json"
)
_dataset_db_path = (
    pathlib.Path(__file__).parents[1] / "resources" / "dataset_db-rfsoc-tests.pyon"
)
_gate_solutions_h5_path = _dataset_db_path.with_name(
    "gate_solutions_15ions_2019-12-10.h5"
)
_example_rfsoc_backend_path = (
    pathlib.Path(rfsoc_mapping.__file__).parent / "examples" / "example_hardware.pyon"
)


# pylint: disable=redefined-outer-name
@pytest.fixture(
    params=[
        (1, _live_rfsoc_backend_path),  # minimum number of qubits
        (7, _live_rfsoc_backend_path),  # max w/ one RFSoC
        (15, _example_rfsoc_backend_path),  # current breadboard capabilities
        (31, _example_rfsoc_backend_path),  # future breadboard
    ],
    ids=[
        "1_ion_backend_live_rfsoc_mapping",
        "7_ion_backend_live_rfsoc_mapping",
        "15_ion_backend_example_rfsoc_backend",
        "31_ion_backend_example_rfsoc_backend",
    ],
)
def rfsoc_qiskit_backend(request) -> be.MinimalQiskitIonBackend:
    """Generates a Qiskit backend as a fixture.

    The backend qubit indexing can be modified per-test by using
    ``pytest.mark.backend_zero_index(bool)`` (default=False (center-index)).
    """
    num_ions, rfsoc_board_path = request.param
    index_marker = request.node.get_closest_marker("backend_zero_index")
    if index_marker is None:
        use_zero_index = False
    else:
        use_zero_index = index_marker.args[0]
    rfsoc_channels = rfsoc_mapping.RFSoCChannelMapping.from_pyon_file(rfsoc_board_path)

    return be.MinimalQiskitIonBackend(
        num_qubits=num_ions,
        rfsoc_hardware_map=rfsoc_channels,
        endcap_ions=(0, 0),
        use_zero_index=use_zero_index,
    )


@pytest.fixture
def qiskit_backend_with_fake_cals(rfsoc_qiskit_backend):
    class FakeDatasetDB:
        # pylint: disable=unused-argument
        @staticmethod
        def get(*args, **kwargs):
            return 1.0

    cals = calibrations.CalibrationBox.from_json(
        filename=_rf_calibration_path, dataset_dict=FakeDatasetDB(),
    )
    rfsoc_qiskit_backend.set_properties(
        cals.to_backend_properties(rfsoc_qiskit_backend.configuration().n_qubits)
    )
    return rfsoc_qiskit_backend


@pytest.fixture
def qiskit_backend_with_real_cals(rfsoc_qiskit_backend):
    datasets = pyon.load_file(_dataset_db_path)
    cals = calibrations.CalibrationBox.from_json(
        filename=_rf_calibration_path, dataset_dict=datasets
    )
    rfsoc_qiskit_backend.set_properties(
        cals.to_backend_properties(rfsoc_qiskit_backend.configuration().n_qubits)
    )
    return rfsoc_qiskit_backend


@pytest.fixture
def qiskit_backend_with_gate_solutions(qiskit_backend_with_real_cals):
    # pylint: disable=protected-access
    loaded_gate_solutions_box = rfsoc.RFSOC._load_gate_solutions(
        _gate_solutions_h5_path, num_ions=15, load_calibrations=True
    )

    global_overrides = (
        ("global_amplitude", 0.0),
        ("individual_amplitude_multiplier", 0.0),
    )
    for param_name, value in global_overrides:
        loaded_gate_solutions_box["gate_tweaks.struct"].loc[
            gate_params.GateCalibrations.GLOBAL_CALIBRATION_SLOT, param_name
        ] = value

    qiskit_backend_with_real_cals.properties().rf_calibration.merge_update(
        loaded_gate_solutions_box
    )
    return qiskit_backend_with_real_cals

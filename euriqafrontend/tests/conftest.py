import pytest
import pathlib
import sipyco.pyon as pyon

import euriqafrontend.settings.calibration_box as cal_box
from euriqafrontend.settings import RF_CALIBRATION_PATH


_dataset_db_path = (
    pathlib.Path(__file__).parents[2]
    / "euriqabackend"
    / "tests"
    / "resources"
    / "dataset_db-2020-12-03.pyon"
)
# _live_rfsoc_backend_path = (
#     pathlib.Path(__file__).parents[2] / "databases" / "rfsoc_system_description.pyon"
# )


@pytest.fixture
def rf_calibration_box() -> cal_box.CalibrationBox:
    return cal_box.CalibrationBox.from_json(
        filename=RF_CALIBRATION_PATH, dataset_dict=pyon.load_file(_dataset_db_path)
    )

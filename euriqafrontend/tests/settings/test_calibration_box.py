"""Test :mod:`euriqafrontend.settings.calibration_box`."""
import datetime
import unittest.mock as mock

import numpy as np
import qiskit.providers.models.backendproperties as q_be_props

import euriqafrontend.settings.calibration_box as cal_box
from euriqafrontend.settings import RF_CALIBRATION_PATH


def test_calib_box_dataset_init():
    dataset_db_mock = mock.Mock()
    dataset_db_mock.get.return_value = 1.0
    cbox = cal_box.CalibrationBox.from_json(
        filename=RF_CALIBRATION_PATH, dataset_dict=dataset_db_mock,
    )

    dataset_keys = map(
        lambda k: k[: -len("type")] + "value",
        filter(
            lambda k: k.endswith("type") and cbox[k] == "dataset",
            cbox.keys(dotted=True),
        ),
    )
    for k in dataset_keys:
        assert k in cbox.keys(dotted=True) and cbox[k] > 0.0


def test_calib_box_calculated_value_init():
    dataset_db_mock = mock.Mock()
    dataset_db_mock.get.return_value = 0.0
    cbox = cal_box.CalibrationBox.from_json(
        filename=RF_CALIBRATION_PATH, dataset_dict=dataset_db_mock
    )

    dataset_keys = map(
        lambda k: k[: -len("type")] + "value",
        filter(
            lambda k: k.endswith("type") and cbox[k] == "calculated",
            cbox.keys(dotted=True),
        ),
    )

    for k in dataset_keys:
        assert k in cbox.keys(dotted=True) and cbox[k] > 0.0


def test_calib_box_has_dates(rf_calibration_box):
    value_parent_keys = cal_box._filter_keys(
        rf_calibration_box.keys(dotted=True), ".value"
    )
    for pk in value_parent_keys:
        date = rf_calibration_box[pk + ".date"]
        assert isinstance(date, (datetime.datetime, str))
        assert date < datetime.datetime.now().astimezone()
        assert date > datetime.datetime(2019, 1, 1).astimezone()


def test_calib_box_to_qiskit_properties(rf_calibration_box):
    props = rf_calibration_box.to_backend_properties(7)
    assert isinstance(props, q_be_props.BackendProperties)
    assert len(props.faulty_gates()) == 0
    assert props.is_qubit_operational(0)
    # assert props.is_gate_operational("rx", 1)


def test_calib_box_no_scale_list():
    dataset_db_mock = mock.Mock()
    mock_value = [1.0, 2.0]
    dataset_db_mock.get.return_value = mock_value
    cbox = cal_box.CalibrationBox(
        {
            "value": {
                "type": "dataset",
                "key": "fake",
                "scale": False,
                "list_to_array": False,
            }
        },
        dataset_dict=dataset_db_mock,
    )
    assert cbox["value.value"] == mock_value


def test_calib_box_convert_dataset_list_to_array():
    dataset_db_mock = mock.Mock()
    mock_value = [1.0, 2.0]
    dataset_db_mock.get.return_value = mock_value
    cbox = cal_box.CalibrationBox(
        {
            "value": {
                "type": "dataset",
                "key": "fake",
                "scale": 2.0,
                "list_to_array": True,
            }
        },
        dataset_dict=dataset_db_mock,
    )
    np.testing.assert_array_equal(cbox["value.value"], np.array(mock_value) * 2.0)
    assert isinstance(cbox["value.value"], np.ndarray)


def test_calib_box_convert_value_list_to_array():
    cbox = cal_box.CalibrationBox(
        {
            "value": {
                "value": [1.0, 2.0],
                "list_to_array": True,
                # "scale": 2.0  # NOT VALID. Currently not supported
            }
        },
    )
    assert isinstance(cbox["value.value"], np.ndarray)
    np.testing.assert_array_equal(cbox["value.value"], np.array([1.0, 2.0]))

"""Test :mod:`euriqabackend.devices.keysight_awg.gate_parameters`."""
import calendar
import logging
import pathlib
import pickle
import random
import time
import typing

import numpy as np
import pytest

import euriqabackend.devices.keysight_awg.common_types as rf_common
import euriqabackend.devices.keysight_awg.gate_parameters as gate_params

_LOGGER = logging.getLogger(__name__)

# *** Fixtures ***


@pytest.fixture(scope="function", params=[3, 15, 32])
def gate_solutions_dir(request, tmpdir) -> typing.Tuple[int, pathlib.Path]:
    """Generate a pytest fixture containing dummy gate solutions."""
    num_ions = request.param
    base_dir = pathlib.Path(str(tmpdir))
    for ion_pair in rf_common.gen_all_IonPairs(num_ions):
        _save_gate_solution(base_dir, ion_pair)

    yield num_ions, base_dir


def _save_gate_solution(
    base_dir: pathlib.Path,
    ion_pair: rf_common.IonPair,
    gate_data: rf_common.XXGateParams = None,
):
    """Save a dummy gate solution to a file."""
    if gate_data is None:
        gate_data = rf_common.XXGateParams(
            random.randint(
                rf_common.XXModulationType.AM_segmented,
                rf_common.XXModulationType.AMFM_interp,
            ),
            random.random(),  # float, [0, 1.0)
            random.randrange(-1, +2, 2),  # [-1, 1]
            random.random() * 300 + 1,  # [1, 300) us
            np.random.rand(
                random.randint(2, 20)
            ).tolist(),  # gate amplitudes, rand length
        )
    gate_file = base_dir / "{:02}-{:02}.sol".format(ion_pair[0], ion_pair[1])
    with gate_file.open(mode="wb") as f:
        pickle.dump(list(gate_data), file=f)


@pytest.fixture(scope="function")
def gate_sol_from_path(
    gate_solutions_dir: typing.Tuple[int, pathlib.Path]
) -> gate_params.GateSolution:
    """Generate a gate solution pytest fixture."""
    ions, soln_dir = gate_solutions_dir

    yield gate_params.GateSolution(num_ions=ions, path=soln_dir)


@pytest.fixture
def gate_cal_from_path(
    gate_solutions_dir: typing.Tuple[int, pathlib.Path]
) -> gate_params.GateCalibrations:
    """Generate a gate calibration pytest fixture."""
    ions, soln_dir = gate_solutions_dir

    yield gate_params.GateCalibrations(num_ions=ions, path=soln_dir)


# *** Begin Tests ***


def test_gatesolution_init_from_path(
    gate_solutions_dir: typing.Tuple[int, pathlib.Path]
):
    """Test that a :class:`GateSolution` can init from a path."""
    # pylint: disable=protected-access
    ions, path = gate_solutions_dir
    start_time = time.gmtime()
    soln = gate_params.GateSolution(ions, path=path)
    assert soln.num_ions == ions
    assert len(soln.gate_parameters_df.index) == len(
        tuple(rf_common.gen_all_IonPairs(ions))
    )
    assert soln._validate_df(soln.gate_parameters_df)
    assert start_time <= soln.last_modification_time <= time.gmtime()


def test_gatesolution_passthrough(gate_sol_from_path: gate_params.GateSolution):
    """Test that a :class:`GateSolution` can passthrough calls to internal pandas df."""
    soln = gate_sol_from_path
    for idx in soln.possible_indices:
        assert soln.loc[idx] is not None and soln.loc[idx] is not np.NaN


def _test_solutions_equal(
    soln1: gate_params.GateSolution, soln2: gate_params.GateSolution
):
    assert soln1.num_ions == soln2.num_ions
    assert soln1.gate_parameters_df.equals(soln2.gate_parameters_df)
    assert calendar.timegm(soln1.last_modification_time) == calendar.timegm(
        soln2.last_modification_time
    )
    assert soln1.solutions_hash == soln2.solutions_hash
    assert soln1.load_path == soln2.load_path


def test_gatesolution_load_save(gate_sol_from_path: gate_params.GateSolution, tmpdir):
    """Test loading/saving from h5 file.

    Indirectly tests loading/saving from dataframe, so don't need to test that.
    """
    soln = gate_sol_from_path
    tmpdir = pathlib.Path(str(tmpdir))
    h5file = tmpdir / "gatesol_test.hdf5"
    soln.to_h5(h5file)
    loaded_soln = gate_params.GateSolution.from_h5(h5file)
    _test_solutions_equal(soln, loaded_soln)


def test_gatecal_load_save(gate_cal_from_path: gate_params.GateCalibrations, tmpdir):
    """Test loading/saving Gate Calibration from h5 file.

    Indirectly tests loading/saving from dataframe, so don't need to test that.
    """
    cal = gate_cal_from_path
    h5file = pathlib.Path(str(tmpdir)) / "gatecal_test.h5"
    cal.to_h5(h5file)
    loaded_cal = gate_params.GateCalibrations.from_h5(h5file)
    _test_solutions_equal(cal, loaded_cal)


def test_gatecal_from_solution(gate_sol_from_path: gate_params.GateSolution):
    """Test that we can create a set of tweaks from an existing solution."""
    soln = gate_sol_from_path
    cal = gate_params.GateCalibrations.from_gate_solution(soln)
    assert cal.num_ions == soln.num_ions
    assert cal.load_path == soln.load_path
    assert cal.solutions_hash == soln.solutions_hash
    assert cal.loc[:, "XX_gate_type"].equals(soln.loc[:, "XX_gate_type"])


def test_gatecal_update_parameters(gate_cal_from_path: gate_params.GateCalibrations):
    """Test that we can update calibration parameters."""
    cal = gate_cal_from_path
    random_index = cal.possible_indices[random.randrange(0, len(cal.possible_indices))]

    # Test no change
    old_ion_pair_params = cal.loc[random_index, :]
    cal.update_gate(random_index)
    assert cal.loc[random_index, :].equals(old_ion_pair_params)

    # Test bad update value
    with pytest.raises(ValueError):
        cal.update_gate(random_index, invalid_value=True)

    # Test valid update value
    for col in cal.df_columns:
        # NOTE: need to use int for XXModulationType casting, otherwise would use float
        # Could alternatively have special case for the modulationtype column.
        kwarg_dict = {col: random.randint(0, 5)}
        cal.update_gate(random_index, **kwarg_dict)
        assert cal.loc[random_index, col] == kwarg_dict[col]

def test_gatesol_empty():
    solution = gate_params.GateSolution(num_ions=10)
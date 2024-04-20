import pathlib
import typing

import artiq.language.environment as artiq_env
import dax.sim.signal as dax_signal
import more_itertools
import numpy as np
import pulsecompiler.rfsoc.tones.tonedata as tones
import pytest
import qiskit.pulse as qp
from artiq.language.core import kernel

import euriqafrontend.modules.rfsoc_sbc as rfsoc_sbc
import euriqafrontend.modules.rfsoc as rfsoc
from .conftest import _TEST_RESOURCE_PATH


class RFSoCSBCExperiment(artiq_env.EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.rfsoc_sbc = rfsoc_sbc.RFSoCSidebandCooling(self)

    def custom_pulse_schedule_list(self) -> rfsoc.ScheduleList:
        with qp.build() as simple_schedule:
            qp.play(qp.Constant(100, 1.0), qp.DriveChannel(0))

        return [simple_schedule]

    def prepare(self):
        self.call_child_method("prepare")

    def run(self):
        self.run_kernel()

    @kernel
    def run_kernel(self):
        self.rfsoc_sbc.kn_init_rfsoc_sbc()
        # num_shots = self.rfsoc_sbc.num_shots
        num_shots = 5  # for testing
        for _shot in range(num_shots):
            self.rfsoc_sbc.kn_do_rfsoc_sbc()
            # play nominal circuit/pulse output (non-SBC)
            self.rfsoc_sbc.trigger()


@pytest.fixture
def rf_calibration_file_with_gate_solutions(tmp_path) -> pathlib.Path:
    """Modify the RF calibration file to use test resource gate solutions"""
    from euriqafrontend.settings import rf_calibration

    rf_calibration.gate_solutions.solutions_top_path = str(
        _TEST_RESOURCE_PATH / "gate_solutions_15ions_2019-12-10.h5"
    )
    modified_file_path = tmp_path / "modified_rf_calib.json"
    rf_calibration.to_json(filename=modified_file_path)
    return modified_file_path


@pytest.fixture
def experiment_args(
    request,
    rf_calibration_file_with_gate_solutions: pathlib.Path,
) -> typing.Dict:
    single_sbc_marker = request.node.get_closest_marker("single_sbc_schedule")
    if single_sbc_marker is None:
        single_sbc_schedule = False
    else:
        single_sbc_schedule = True
    return {
        "rfsoc_board_description": (
            _TEST_RESOURCE_PATH / "example_rfsoc_hardware.pyon"
        ).read_text(),
        "rf_calibration_file": str(rf_calibration_file_with_gate_solutions),
        "use_RFSOC": True,
        "use_single_schedule_sbc": single_sbc_schedule,
        "schedule_transform_aom_nonlinearity": False,
    }


@pytest.fixture
def sbc_experiment(artiq_experiment_managers):
    return RFSoCSBCExperiment(artiq_experiment_managers)


# TODO: need to change the hard-coded AWG-like SBC sequence to support >1 ion.
@pytest.mark.todo
def test_rfsoc_sbc_timing(sbc_experiment: RFSoCSBCExperiment):
    sbc_experiment.prepare()
    sbc_experiment.run()
    signal_manager = dax_signal.get_signal_manager()
    rfsoc_trigger = signal_manager.signal(
        sbc_experiment.rfsoc_sbc.rfsoc_trigger, "state"
    )
    rfsoc_trigger_toggle_intervals = np.diff(rfsoc_trigger._timestamps)
    # trigger on for 100 mu each time.
    assert np.allclose(rfsoc_trigger_toggle_intervals[::2], 100, atol=1.0)
    # increase duration by 215 mu each sequence
    assert np.allclose(np.diff(rfsoc_trigger_toggle_intervals[1::2]), 215, atol=2)

    dds_funcs = signal_manager.signal(
        sbc_experiment.rfsoc_sbc.pump_detect.pump_det_dds1, "function"
    )
    assert np.allclose(np.diff(np.diff(dds_funcs._timestamps[1:])), 215, atol=2.0)
    dds_func_calls = sorted(set(dds_funcs._events.values()))
    assert dds_func_calls[0].startswith("pulse")
    assert dds_func_calls[1].startswith("set_mu")
    assert len(dds_func_calls) == 2  # 1 for init, 1 for each pulse


# TODO: need to change the hard-coded AWG-like SBC sequence to support >1 ion.
@pytest.mark.todo
def test_rfsoc_sbc_sequence_duration(sbc_experiment: RFSoCSBCExperiment):
    sbc_experiment.prepare()

    first_sequence = sbc_experiment.rfsoc_sbc.compiled_sequence_list[0]
    # check number of SBC triggered segments is correct
    assert (
        sum(
            (
                1 if td.wait_trigger else 0
                for td in first_sequence[more_itertools.first(first_sequence.keys())]
            )
        )
        == 15 + 1
    )
    # check that first duration is ~initial SBC duration + AOM delay +
    # prepare/measure pulses + delay b/w set/shift freq (4 mu)
    assert sbc_experiment.rfsoc_sbc.cooling_schedule_durations[0] == pytest.approx(
        1.07e-06 + 0.9e-6 + (122 * 2 + 4 + 4) * tones.CLKPERIOD, abs=tones.CLKPERIOD
    )
    # check last pulse is roughly AOM delay + init SBC + incremental SBC +
    # prep/measure pulses + delay b/w set/shift freq (4 mu)
    assert sbc_experiment.rfsoc_sbc.cooling_schedule_durations[-1] == pytest.approx(
        0.9e-6 + 1.07e-6 + (215e-9 * 15) + (122 * 2 + 4 + 4) * tones.CLKPERIOD,
        abs=tones.CLKPERIOD,
    )


# TODO: test as part of e.g. Ramsey sequence, that the SBC schedule is properly used.

@pytest.mark.single_sbc_schedule
def test_rfsoc_sbc_single_schedule(sbc_experiment: RFSoCSBCExperiment):
    sbc_experiment.prepare()
    sbc_experiment.run()

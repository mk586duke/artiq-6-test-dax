"""Test :mod:`euriqafrontend.modules.rfsoc`."""
import logging
import typing
import unittest.mock as mock

import artiq.language.environment as artiq_env
import numpy as np
import more_itertools
import pulsecompiler.qiskit.backend as be
import pulsecompiler.qiskit.pulses as pc_pulse
import pulsecompiler.qiskit.configuration as pc_config
import pulsecompiler.qiskit.transforms.nonlinearity as spl_nonlinearity
import pulsecompiler.rfsoc.structures.splines as spl
import pytest
import qiskit.pulse as qp
import qiskit.qobj as qobj
from artiq.master.worker_db import DeviceManager
from artiq.language.environment import ProcessArgumentManager

import euriqafrontend.modules.rfsoc as rfsoc
import euriqafrontend.settings.calibration_box as calibrations
import euriqabackend.waveforms.conversions as wf_convert

_LOGGER = logging.getLogger(__name__)


class RFSoCExperimentExample(artiq_env.EnvExperiment):
    """Example experiment using RFSoC module for testing purposes."""

    def build(self, qiskit_backend: be.MinimalQiskitIonBackend):
        self.rfsoc = rfsoc.RFSOC(self)
        self._qiskit_backend = qiskit_backend

    def prepare(self):
        self.call_child_method("prepare")
        # override the qiskit backend that is auto-set in prepare()
        self.rfsoc.qiskit_backend = self._qiskit_backend


@pytest.fixture
def rfsoc_experiment(
    request,
    monkeypatch,
    qiskit_backend_with_real_cals: be.MinimalQiskitIonBackend,
    artiq_experiment_managers,
) -> RFSoCExperimentExample:
    """Generates a sample experiment using the RFSoC module as a fixture.

    The calibration enables can be modified per-test by using
    ``@pytest.mark.disable_calibrations(bool)`` (default=False (calibrations enabled)).
    """
    # Override a few datasets to expected values (or create datasets that don't exist)
    dataset_db = artiq_experiment_managers[1]
    dataset_db.set("global.AWG.N_ions", 7)  # lock to 7 ions for testing purposes
    exp = RFSoCExperimentExample(
        artiq_experiment_managers, qiskit_backend=qiskit_backend_with_real_cals,
    )

    disable_calibs_marker = request.node.get_closest_marker("disable_calibrations")
    if disable_calibs_marker:
        _LOGGER.debug("Disabling all schedule calibrations")
        for t, _ in exp.rfsoc._SCHEDULE_TRANSFORM_TYPES.items():
            setattr(
                exp.rfsoc, exp.rfsoc._SCHEDULE_TRANSFORM_ARG_PREFIX + t, False,
            )

    # Prepare module
    with monkeypatch.context() as m:
        # Patch out default auto-compilation for setup
        m.setattr(exp.rfsoc, "compile_pulse_schedule_to_octet", mock.MagicMock())
        exp.prepare()

    assert exp.rfsoc.qiskit_backend == qiskit_backend_with_real_cals
    return exp


@pytest.fixture
def basic_schedule(
    qiskit_backend_with_real_cals: be.MinimalQiskitIonBackend,
) -> qp.Schedule:
    with qp.build(qiskit_backend_with_real_cals) as sched:
        for chan in qp.qubit_channels(0):
            qp.play(qp.Constant(5000, 0.75), chan)
            qp.play(qp.Constant(5000, 0.5), chan)
            qp.play(qp.Constant(5000, 0.0), chan)

    return sched


def test_find_schedule(
    rfsoc_experiment: RFSoCExperimentExample, basic_schedule: qp.Schedule, monkeypatch,
):
    sched_qobj = qobj.PulseQobj(
        "fake_qobj_id",
        qobj.PulseQobjConfig(
            meas_level=2,
            meas_return=2,
            pulse_library=[],
            qubit_lo_freq=[],
            meas_lo_freq=[],
        ),
        experiments=[],
    )
    schedule_list = [basic_schedule]

    # test fail on no schedule
    with pytest.raises(RuntimeError, match="No schedule found"):
        rfsoc_experiment.rfsoc._find_pulse_schedule()

    # test pass on ARTIQ argument
    with monkeypatch.context() as m:
        m.setattr(
            rfsoc_experiment.rfsoc,
            "openpulse_schedule_qobj",
            sched_qobj.to_dict(validate=False),
        )
        assert rfsoc_experiment.rfsoc._find_pulse_schedule() == sched_qobj

    # test retrieving function argument
    assert rfsoc_experiment.rfsoc._find_pulse_schedule(schedule_list) == schedule_list

    # test custom pulse schedule generator function
    rfsoc_experiment.custom_pulse_schedule_list = mock.Mock(return_value=schedule_list)
    assert rfsoc_experiment.rfsoc._find_pulse_schedule() == schedule_list


def test_global_aom_delay(
    rfsoc_experiment: RFSoCExperimentExample,
    basic_schedule: qp.Schedule,
    rf_calibration_box: calibrations.CalibrationBox,
    qiskit_backend_with_real_cals: be.MinimalQiskitIonBackend,
):
    assert basic_schedule.start_time == 0
    assert basic_schedule.duration == 15000
    global_channel = more_itertools.first_true(
        basic_schedule.channels, pred=lambda c: isinstance(c, qp.ControlChannel)
    )
    individual_channel = more_itertools.first_true(
        basic_schedule.channels, pred=lambda c: isinstance(c, qp.DriveChannel)
    )
    assert basic_schedule.ch_duration(individual_channel) == basic_schedule.ch_duration(
        global_channel
    )

    # apply transform
    delayed_sched = rfsoc_experiment.rfsoc.compensate_global_aom_delay(basic_schedule)

    # Verify
    # Gross, that it's shifted
    assert delayed_sched.stop_time != basic_schedule.stop_time
    assert delayed_sched.duration != basic_schedule.duration
    assert delayed_sched.ch_duration(individual_channel) != delayed_sched.ch_duration(
        global_channel
    )

    # Fine, that it's shifted right amount
    aom_delay_seconds = rf_calibration_box.delays.global_aom_to_individual_aom.value
    aom_delay_dt = int(aom_delay_seconds / rfsoc_experiment.rfsoc.dt)
    aom_delay_backend_dt = int(
        aom_delay_seconds / qiskit_backend_with_real_cals.configuration().dt
    )
    assert aom_delay_dt == aom_delay_backend_dt
    assert (
        delayed_sched.ch_start_time(individual_channel)
        == delayed_sched.ch_start_time(global_channel) + aom_delay_dt
    )


def test_schedule_hardware_corrections(
    rfsoc_experiment: RFSoCExperimentExample,
    basic_schedule: qp.Schedule,
    qiskit_backend_with_real_cals: be.MinimalQiskitIonBackend,
):
    """Test that calibrations are properly applied to the schedule.

    The reference schedule is hardcoded, so this will need updated as more
    corrections are added.
    """
    # for transform_arg in filter(
    #         lambda attr: attr.startswith(rfsoc_experiment._SCHEDULE_TRANSFORM_ARG_PREFIX),
    #         vars(rfsoc_experiment).keys(),
    #     ):
    #         setattr(rfsoc_experiment, transform_arg, False)

    #     rfsoc_experiment.schedule_transform_global_aom_delay = True

    calibrated_schedule = rfsoc_experiment.rfsoc._apply_schedule_calibrations(
        basic_schedule
    )

    with qp.build(qiskit_backend_with_real_cals) as manual_corrected_schedule:
        be_config: typing.Union[
            pc_config.BackendConfigCenterIndex, pc_config.BackendConfigZeroIndex
        ] = qp.active_backend().configuration()
        center_chan_0 = be_config.drive(0, 0)
        center_chan_1 = be_config.drive(0, 1)
        unused_channels = set(be_config.all_channels) - {
            center_chan_0,
            center_chan_1,
            qp.ControlChannel(0),
            qp.ControlChannel(1),
        }
        for c in unused_channels:
            qp.delay(15417, c)
        qp.play(qp.Constant(5000, 0.7914948023318472), qp.ControlChannel(0))
        qp.play(qp.Constant(5000, 0.7914948023318472), qp.ControlChannel(1))
        qp.play(qp.Constant(5000, 0.5112901087461438), qp.ControlChannel(0))
        qp.play(qp.Constant(5000, 0.5112901087461438), qp.ControlChannel(1))
        qp.play(qp.Constant(5000, 0.0), qp.ControlChannel(0))
        qp.play(qp.Constant(5000, 0.0), qp.ControlChannel(1))
        qp.play(
            pc_pulse.ToneDataPulse(
                duration_cycles=417,
                frequency_hz=187449746.45000002,
                amplitude=0.3,
                phase_rad=0.0,
                frame_rotation_rad=0.0,
                wait_trigger=False,
                sync=False,
                output_enable=True,
                feedback_enable=False,
                frame_rotate_at_end=False,
                reset_frame=False,
                use_frame_a=None,
                use_frame_b=None,
                invert_frame_a=False,
                invert_frame_b=False,
                bypass_lookup_tables=True,
            ),
            qp.ControlChannel(0),
        )
        qp.play(
            pc_pulse.ToneDataPulse(
                duration_cycles=417,
                frequency_hz=187449746.45000002,
                amplitude=0.3,
                phase_rad=0.0,
                frame_rotation_rad=0.0,
                wait_trigger=False,
                sync=False,
                output_enable=True,
                feedback_enable=False,
                frame_rotate_at_end=False,
                reset_frame=False,
                use_frame_a=None,
                use_frame_b=None,
                invert_frame_a=False,
                invert_frame_b=False,
                bypass_lookup_tables=True,
            ),
            qp.ControlChannel(1),
        )
        qp.delay(417, center_chan_0)
        qp.delay(417, center_chan_1)
        if center_chan_0 == qp.DriveChannel(6):
            qp.play(qp.Constant(5000, 0.7914948023318472), center_chan_0)
            qp.play(qp.Constant(5000, 0.7914948023318472), center_chan_1)
            qp.play(qp.Constant(5000, 0.5112901087461438), center_chan_0)
            qp.play(qp.Constant(5000, 0.5112901087461438), center_chan_1)
        else:
            qp.play(qp.Constant(5000, 0.7968196389688408), center_chan_0)
            qp.play(qp.Constant(5000, 0.7968196389688408), center_chan_1)
            qp.play(qp.Constant(5000, 0.5125963843772001), center_chan_0)
            qp.play(qp.Constant(5000, 0.5125963843772001), center_chan_1)
        qp.play(qp.Constant(5000, 0.0), center_chan_0)
        qp.play(qp.Constant(5000, 0.0), center_chan_1)

    manual_corrected_schedule = qp.transforms.pad(manual_corrected_schedule)

    assert calibrated_schedule == manual_corrected_schedule


@pytest.mark.disable_calibrations(True)
def test_schedule_hardware_no_corrections(
    rfsoc_experiment: RFSoCExperimentExample, basic_schedule: qp.Schedule,
):
    """Test that when no corrections are enabled, no correction is applied."""

    # Disable adjustments
    for t, _ in rfsoc_experiment.rfsoc._SCHEDULE_TRANSFORM_TYPES.items():
        setattr(
            rfsoc_experiment.rfsoc,
            rfsoc_experiment.rfsoc._SCHEDULE_TRANSFORM_ARG_PREFIX + t,
            False,
        )

    assert basic_schedule == rfsoc_experiment.rfsoc._apply_schedule_calibrations(
        basic_schedule
    )


def test_rfsoc_find_experiment_custom_pulse_method(
    rfsoc_experiment: RFSoCExperimentExample,
):
    """Demonstrate & test the proper way of subclassing RFSOC module."""

    class FakeExperiment(artiq_env.EnvExperiment):
        def build(self):
            self.rfsoc = rfsoc.RFSOC(self)

        def custom_pulse_schedule_list(self):
            return [qp.Schedule()]

    device_manager = mock.MagicMock(spec=DeviceManager)
    dataset_manager = {
        "global.num_ions": 7,
        "global.AWG.calibrated_Tpi": np.array([1.0] * 32),
    }
    argument_manager = ProcessArgumentManager({})
    assert FakeExperiment(
        (device_manager, dataset_manager, argument_manager, {})
    ).rfsoc._find_pulse_schedule() == [qp.Schedule()]


def test_rfsoc_experiment_schedule_duration(
    rfsoc_experiment: RFSoCExperimentExample,
    basic_schedule: qp.Schedule,
    qiskit_backend_with_real_cals: be.MinimalQiskitIonBackend,
):
    assert basic_schedule.duration == 15000
    assert (
        rfsoc_experiment.rfsoc.schedule_duration(basic_schedule)
        == 15000 * qiskit_backend_with_real_cals.configuration().dt
    )


def test_aom_nonlinearity(
    rfsoc_experiment: RFSoCExperimentExample,
    basic_schedule: qp.Schedule,
    rf_calibration_box: calibrations.CalibrationBox,
):
    # apply transform
    calibrated_sched = rfsoc_experiment.rfsoc.compensate_aom_nonlinearity(
        basic_schedule
    )

    # Verify
    # Gross, that it's different
    assert calibrated_sched != basic_schedule

    # Fine, that the amplitudes changed by the right amount
    aom_ch0_saturation = rf_calibration_box.aom_saturation.individual.value[0]

    corrected = wf_convert.apply_nonlinearity(
        np.array([0.75, 0.5, 0.0]), aom_ch0_saturation
    )

    ch0_instructions = calibrated_sched.filter(
        channels=[qp.ControlChannel(0)]
    ).instructions
    assert np.array(
        [t_inst[1].pulse.amp for t_inst in ch0_instructions]
    ) == pytest.approx(corrected, abs=1e-4)


def test_aom_nonlinearity_measured_value(
    rfsoc_experiment: RFSoCExperimentExample, qiskit_backend_with_real_cals
):
    # nonlinearity increases the output amplitude, so the actual maximum input is ~0.9
    amp_vals = np.linspace(0, 0.9, num=15)
    with qp.build(qiskit_backend_with_real_cals) as test_schedule:
        for amp in amp_vals:
            qp.play(qp.Constant(100, amp=amp), qp.drive_channel(0))
        nonlinearity = rfsoc_experiment.rfsoc._aom_saturation(qp.drive_channel(0))

    calibrated_sched = rfsoc_experiment.rfsoc.compensate_aom_nonlinearity(test_schedule)
    assert calibrated_sched != test_schedule
    for i, (t, instr) in enumerate(calibrated_sched.instructions):
        calibrated_amp = instr.pulse.amp
        if amp_vals[i] > 0.0:
            assert calibrated_amp != amp_vals[i]
        assert calibrated_amp == pytest.approx(
            wf_convert.apply_nonlinearity(amp_vals[i], nonlinearity), rel=1e-2
        )


@pytest.fixture
def tonedata_schedule(
    qiskit_backend_with_real_cals: be.MinimalQiskitIonBackend,
) -> qp.Schedule:
    with qp.build(qiskit_backend_with_real_cals) as sched:
        channels = sorted(qp.qubit_channels(0), key=lambda c: c.name)
        for i, chan in enumerate(channels):
            qp.delay(i, chan)
            qp.play(pc_pulse.ToneDataPulse(5000, 200e6, 0.5, 0.0, sync=False), chan)

    return sched


@pytest.mark.disable_calibrations(True)
def test_aom_nonlinearity_tonedata_pulse(
    rfsoc_experiment: RFSoCExperimentExample,
    tonedata_schedule: qp.Schedule,
    qiskit_backend_with_real_cals: be.MinimalQiskitIonBackend,
):
    # turn off all calibrations except nonlinearity to make the resulting schedule simpler
    rfsoc_experiment.rfsoc.schedule_transform_aom_nonlinearity = True
    test_schedule = rfsoc_experiment.rfsoc._apply_schedule_calibrations(
        tonedata_schedule
    )

    with qp.build(qiskit_backend_with_real_cals) as correct_schedule:
        channels = sorted(qp.qubit_channels(0), key=lambda c: c.name)
        for i, channel in enumerate(channels):
            aom_sat = rfsoc_experiment.rfsoc._aom_saturation(channel)
            correct_amp = spl_nonlinearity.calculate_new_coefficients(
                spl.CubicSpline(0.5), aom_sat
            )
            qp.delay(i, channel)
            qp.play(
                pc_pulse.ToneDataPulse(5000, 200e6, correct_amp, 0.0, sync=False),
                channel,
            )

    assert test_schedule == correct_schedule


def test_aom_saturation_param(
    rfsoc_experiment: RFSoCExperimentExample,
    qiskit_backend_with_real_cals: be.MinimalQiskitIonBackend,
):
    """Test that all channels return valid index/value in the AOM saturation array."""
    for c in qiskit_backend_with_real_cals.configuration().all_channels:
        rfsoc_experiment.rfsoc._aom_saturation(c)


def test_rfsoc_experiment_channel_sequence_duration(
    rfsoc_experiment: RFSoCExperimentExample, basic_schedule: qp.Schedule,
):
    assert basic_schedule.duration == 15000
    rfsoc_experiment.schedule_transform_global_aom_delay = False
    rfsoc_sequence = rfsoc_experiment.rfsoc.compile_pulse_schedule_to_octet(
        basic_schedule
    )
    # 15000 == basic_schedule.duration
    # 122 == "measurement" pulse auto-appended duration (x2)
    # 4 == prepare pulse duration
    # 417 == individual/global AOM delay
    assert (
        rfsoc_experiment.rfsoc.schedule_duration(rfsoc_sequence[0])
        == (15000 + (2 * 122 + 4) + 417) * rfsoc_experiment.rfsoc.dt
    )


def test_rfsoc_experiment_transform_keep_global_beam_on(
    rfsoc_experiment: RFSoCExperimentExample,
):
    global_keep_on_frequency = (
        rfsoc_experiment.rfsoc._rf_calib.frequencies.global_carrier_frequency.value
        + rfsoc_experiment.rfsoc.keep_global_on_global_beam_detuning
    )
    with qp.build(rfsoc_experiment._qiskit_backend) as raw_schedule:
        qp.play(qp.Constant(5000, amp=1.0), qp.DriveChannel(0))
        qp.delay(2500, qp.DriveChannel(0))
        qp.play(qp.Constant(2000, amp=0.8), qp.DriveChannel(0))
        qp.play(qp.Constant(2000, amp=0.2), qp.ControlChannel(1))

    with qp.build(rfsoc_experiment._qiskit_backend) as correct_schedule:
        qp.play(qp.Constant(5000, amp=1.0), qp.DriveChannel(0))
        qp.delay(2500, qp.DriveChannel(0))
        qp.play(qp.Constant(2000, amp=0.8), qp.DriveChannel(0))
        qp.play(qp.Constant(2000, amp=0.2), qp.ControlChannel(1))
        qp.play(
            pc_pulse.ToneDataPulse(
                7500,
                frequency_hz=global_keep_on_frequency,
                amplitude=0.3,
                phase_rad=0.0,
                sync=False,
            ),
            qp.ControlChannel(1),
        )
        qp.play(
            pc_pulse.ToneDataPulse(
                9500,
                frequency_hz=global_keep_on_frequency,
                amplitude=0.3,
                phase_rad=0.0,
                sync=False,
            ),
            qp.ControlChannel(0),
        )

    transformed_schedule = rfsoc_experiment.rfsoc.keep_global_beam_on(raw_schedule)
    assert transformed_schedule == correct_schedule


def test_rfsoc_experiment_transform_aom_nonlinearity(rfsoc_experiment: rfsoc.RFSOC):
    with qp.build(rfsoc_experiment._qiskit_backend) as raw_schedule:
        out_chan = qp.DriveChannel(0)
        qp.play(qp.Constant(5000, amp=0.5), out_chan)
        qp.play(pc_pulse.CubicSplinePulse(1000, 0.0, 0.5), out_chan)
        qp.play(pc_pulse.ToneDataPulse(1500, 150e6, 0.4, phase_rad=0.0), out_chan)
        qp.play(qp.Gaussian(2000, amp=0.6, sigma=500), out_chan)
        qp.play(qp.GaussianSquare(5000, amp=0.6, sigma=500, width=3000), out_chan)

    transformed_schedule = rfsoc_experiment.rfsoc.compensate_aom_nonlinearity(
        raw_schedule
    )

    corrected_pulses = [
        qp.Constant(5000, amp=0.5112901087461438),
        pc_pulse.CubicSplinePulse(
            1000,
            -2.191228114090562e-05,
            0.5004097757241721,
            -0.001626998606189498,
            0.012498314286632829,
        ),
        pc_pulse.ToneDataPulse(
            1500,
            150e6,
            spl.CubicSpline(
                order0=0.4056523819781711,
                order1=1.442675235336488e-16,
                order2=-2.228800669942784e-16,
                order3=1.3927996999557308e-16,
            ),
            phase_rad=0.0,
        ),
        # Gaussian
        pc_pulse.CubicSplinePulse(
            167,
            0.00037446842612112464,
            0.0711906854631187,
            -2.6973887765775156e-05,
            0.007562368070003385,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.07910048999166408,
            0.09384624724890975,
            0.02266160469976802,
            -0.0036413544645202437,
        ),
        pc_pulse.CubicSplinePulse(
            166,
            0.19196756776485474,
            0.1282542793256671,
            0.011778948679529087,
            -0.002675027309743821,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.32932878568395857,
            0.14371981378923404,
            0.0040039859356813395,
            -0.006942117886521738,
        ),
        pc_pulse.CubicSplinePulse(
            166,
            0.4701093349299702,
            0.13078043915996704,
            -0.016918021941254763,
            -0.005729745216563754,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.578237353988577,
            0.0797613133699975,
            -0.03447220714745729,
            -0.003455573405761855,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.6200708296443214,
            0.0005767836156580212,
            -0.04484353664056131,
            0.00339002390476975,
        ),
        pc_pulse.CubicSplinePulse(
            166,
            0.5791987710470969,
            -0.07893178205382834,
            -0.034306990246398675,
            0.005717104823690124,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.47167829660444904,
            -0.1305151265767022,
            -0.01705527945663916,
            0.006951281720453152,
        ),
        pc_pulse.CubicSplinePulse(
            166,
            0.33105616237364505,
            -0.14384029831105324,
            0.003549133659858268,
            0.0027388295130066466,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.19350321638118412,
            -0.12851713135152715,
            0.011721550704889078,
            0.003548343951550987,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.08025603427196058,
            -0.09440648824824895,
            0.022364741998257952,
            -0.007463852809149437,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.00037446842612112464,
            0.0711906854631187,
            -2.6973887765775156e-05,
            0.007562368070003385,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.07910048999166408,
            0.09384624724890975,
            0.02266160469976802,
            -0.0036413544645202437,
        ),
        pc_pulse.CubicSplinePulse(
            166,
            0.19196756776485474,
            0.1282542793256671,
            0.011778948679529087,
            -0.002675027309743821,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.32932878568395857,
            0.14371981378923404,
            0.0040039859356813395,
            -0.006942117886521738,
        ),
        pc_pulse.CubicSplinePulse(
            166,
            0.4701093349299702,
            0.13078043915996704,
            -0.016918021941254763,
            -0.005729745216563754,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.578237353988577,
            0.0797613133699975,
            -0.03447220714745729,
            -0.003455573405761855,
        ),
        qp.Constant(duration=3000, amp=(0.6200754421352063 + 0j)),
        pc_pulse.CubicSplinePulse(
            167,
            0.6200708296443214,
            0.0005767836156580212,
            -0.04484353664056131,
            0.00339002390476975,
        ),
        pc_pulse.CubicSplinePulse(
            166,
            0.5791987710470969,
            -0.07893178205382834,
            -0.034306990246398675,
            0.005717104823690124,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.47167829660444904,
            -0.1305151265767022,
            -0.01705527945663916,
            0.006951281720453152,
        ),
        pc_pulse.CubicSplinePulse(
            166,
            0.33105616237364505,
            -0.14384029831105324,
            0.003549133659858268,
            0.0027388295130066466,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.19350321638118412,
            -0.12851713135152715,
            0.011721550704889078,
            0.003548343951550987,
        ),
        pc_pulse.CubicSplinePulse(
            167,
            0.08025603427196058,
            -0.09440648824824895,
            0.022364741998257952,
            -0.007463852809149437,
        ),
    ]

    assert [
        instr.pulse for _t, instr in transformed_schedule.instructions
    ] == corrected_pulses

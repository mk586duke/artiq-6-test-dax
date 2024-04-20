"""Module holding Ramsey experiments.

The experiments are defined in terms of Qiskit schedules that are designed
to be used with PulseCompiler & an Octet RFSoC.
"""
import logging
import typing

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import numpy as np
import oitg.fitting as fit
import qiskit.pulse as qp
from artiq.experiment import NumberValue
from artiq.experiment import TerminationRequested
from artiq.language.core import delay, delay_mu, host_only, kernel
from artiq.language.types import TInt32
from artiq.language.units import us

import euriqabackend.waveforms.single_qubit as single_qubit
import euriqabackend.waveforms.delay as delay_gate
import euriqafrontend.modules.rfsoc as rfsoc
from euriqafrontend.modules.utilities import parse_ion_input
from euriqafrontend.repository.basic_environment import BasicEnvironment

_LOGGER = logging.getLogger(__name__)


class _Ramsey(BasicEnvironment, artiq_env.Experiment):
    """Ramsey experiment prototype.

    This holds the common functionality of a Ramsey experiment, to allow maximum
    code reuse between experiment types.
    """

    data_folder = "rfsoc_ramsey"
    applet_name = "Raman RFSOC Ramsey"
    applet_group = "RFSOC"
    fit_type = fit.cos_fft
    units = 1
    ylabel = "Population Transfer"

    def build(self):
        """Initialize arguments."""
        super().build()
        self.setattr_argument(
            "pulse_1_duration",
            artiq_env.NumberValue(default=1 * us, unit="us"),
            tooltip="Should be the pi/2 duration for the given ion",
        )
        self.setattr_argument(
            "use_different_second_pulse", artiq_env.BooleanValue(default=False)
        )
        self.setattr_argument(
            "pulse_2_duration",
            artiq_env.NumberValue(default=1 * us, unit="us"),
            tooltip="Should be the pi/2 pulse duration at the end of the sequence, "
            "if it differs from the first. "
            "Only used if ``use_different_second_pulse = True``",
        )
        self.setattr_argument("global_amplitude", rfsoc.AmplitudeArg(default=0.5))
        self.setattr_argument("individual_amplitude", rfsoc.AmplitudeArg())
        self.setattr_argument("detuning", rfsoc.FrequencyArg())
        self.setattr_argument(
            "sideband_order",
            NumberValue(1, scale=1, min=-3, max=3, step=1, ndecimals=0),
        )
        self.setattr_argument(
            "ions_to_address",
            artiq_env.StringValue("0"),
            tooltip="center-index! use python range notation e.g. -3,4:6 == [-3, 4, 5]",
        )

        self.setattr_argument(
            "phase_insensitive", artiq_env.BooleanValue(default=False)
        )
        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(default=True))

    def prepare(self, scan_range: typing.Iterable[float]):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        # pylint: disable=attribute-defined-outside-init
        # These values are unfortunately "magic" expected values for BasicEnvironment.
        self.scan_values = list(scan_range)
        self.num_steps = len(self.scan_values)
        self.ions_to_address = parse_ion_input(self.ions_to_address)
        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_experiment_initialize()

            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @host_only
    def custom_experiment_initialize(self):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    def analyze(self, **kwargs):
        super().analyze(**kwargs)
        num_active_pmts = len(self.p_all["period"])
        for ifit in range(num_active_pmts):
            _LOGGER.info(
                "Fit %d : Amplitude = (%.3f +- %.3f) ",
                ifit,
                self.p_all["a"][ifit],
                self.p_error_all["a"][ifit],
            )
            _LOGGER.info(
                "Fit %d : Phase = (%.3f +- %.3f) rad",
                ifit,
                self.p_all["x0"][ifit] / 360 * 2 * np.pi,
                self.p_error_all["x0"][ifit] / 360 * 2 * np.pi,
            )
            _LOGGER.info(
                "Fit %d : Period = (%.3f +- %.3f) ms",
                ifit,
                self.p_all["period"][ifit] * 1e3,
                self.p_error_all["period"][ifit] * 1e3,
            )

    @kernel
    def prepare_step(self, istep: TInt32):
        pass


class RFSOCRamseyDelay(_Ramsey):
    """rfsoc.RamseyDelay

    Scan the delay of a Ramsey pulse sequence.
    """

    unit = 1e-3
    xlabel = "delay"

    def build(self):
        """Initialize experiment & variables."""
        super().build()
        # scan arguments
        self.setattr_argument(
            "delay_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(start=1 * us, stop=1000 * us, npoints=20),
                unit="us",
                global_min=0,
                ndecimals=5,
            ),
            tooltip="Scan of the delay time between the two pi/2 pulses in a Ramsey "
            "sequence",
        )
        self.setattr_argument(
            "global_phase_shift",
            rfsoc.PhaseArg(),
            tooltip="Phase to shift the global beam by during the Ramsey wait time",
        )

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        if self.use_different_second_pulse:
            second_pulse_duration = self.pulse_2_duration
        else:
            second_pulse_duration = self.pulse_1_duration

        for wait_time in self.delay_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                for ion in self.ions_to_address:
                    qp.call(
                        single_qubit.square_rabi_by_amplitude(
                            ion_index=ion,
                            duration=self.pulse_1_duration,
                            phase=0.0,
                            detuning=self.detuning,
                            sideband_order=self.sideband_order,
                            individual_amp=self.individual_amplitude,
                            global_amp=self.global_amplitude,
                            stark_shift = -2000,
                        )
                    )
                    qp.call(
                        single_qubit.square_rabi_by_amplitude(
                            ion_index=ion,
                            duration=1e-6,
                            detuning=self.detuning,
                            sideband_order=self.sideband_order,
                            individual_amp=0,
                            global_amp=self.global_amplitude,
                            stark_shift = -2000,
                        )
                    )

                    delay_gate.wait_gate_ions(wait_time)

                    qp.call(
                        single_qubit.square_rabi_by_amplitude(
                            ion_index=ion,
                            duration=second_pulse_duration,
                            detuning=self.detuning,
                            phase=self.global_phase_shift,
                            sideband_order=self.sideband_order,
                            individual_amp=self.individual_amplitude,
                            global_amp=self.global_amplitude,
                            stark_shift = -2000,
                        )
                    )

            schedule_list.append(out_sched)

        return schedule_list

    def prepare(self):
        super().prepare(self.delay_range)


class RFSOCRamseyPhase(_Ramsey):
    """rfsoc.RamseyPhase

    Scan the phase (of the global beam) during a Ramsey pulse sequence.
    """

    xlabel = "phase (radians)"

    def build(self):
        """Initialize experiment & variables."""
        super().build()
        self.setattr_argument(
            "global_phase_range",
            rfsoc.PhaseScan(
                default=artiq_scan.RangeScan(start=-np.pi, stop=np.pi, npoints=20)
            ),
            tooltip="Phase that the global beam is shifted by during the Ramsey "
            "sequence",
        )
        self.setattr_argument(
            "delay_time", artiq_env.NumberValue(default=100 * us, unit="us", min=0)
        )

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        if self.use_different_second_pulse:
            second_pulse_duration = self.pulse_2_duration
        else:
            second_pulse_duration = self.pulse_1_duration

        for phase in self.global_phase_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                for ion in self.ions_to_address:
                    qp.call(
                        single_qubit.square_rabi_by_amplitude(
                            ion_index=ion,
                            duration=self.pulse_1_duration,
                            detuning=self.detuning,
                            sideband_order=self.sideband_order,
                            individual_amp=self.individual_amplitude,
                            global_amp=self.global_amplitude,
                            backend=self.rfsoc_sbc.qiskit_backend
                        )
                    )

                    delay_gate.wait_gate_ions(self.delay_time)
                    qp.call(
                        single_qubit.square_rabi_by_amplitude(
                            ion_index=ion,
                            duration=second_pulse_duration,
                            detuning=self.detuning,
                            phase=phase,
                            sideband_order=self.sideband_order,
                            individual_amp=self.individual_amplitude,
                            global_amp=self.global_amplitude,
                        )
                    )

            schedule_list.append(out_sched)

        return schedule_list

    def prepare(self):
        super().prepare(self.global_phase_range)

    def analyze(self):
        super().analyze(constants={"period": 2 * np.pi})

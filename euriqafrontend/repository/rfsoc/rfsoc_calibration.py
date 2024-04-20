"""Temporary file for testing arbitrary Qiskit Pulse schedules."""
import logging
import typing

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import numpy as np
import qiskit.pulse as qp
from artiq.experiment import TerminationRequested
from artiq.language.core import delay, delay_mu
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.units import us, ns

import euriqabackend.waveforms.single_qubit as single_qubit
import euriqafrontend.modules.rfsoc as rfsoc
import euriqafrontend.fitting as umd_fit
from euriqafrontend.modules.utilities import parse_ion_input
from euriqafrontend.repository.basic_environment import BasicEnvironment


_LOGGER = logging.getLogger(__name__)


class LinearityCalibration(BasicEnvironment, artiq_env.Experiment):
    """RFSoC AOM Linearity Calibration

    Also called wind-unwind.
    """

    data_folder = "rfsoc_pulse_sequence"
    applet_name = "RFSoC Pulse Sequence"
    applet_group = "RFSOC"
    fit_type = umd_fit.negative_gaussian
    units = us
    xlabel = "Rabi Pulse Length"
    ylabel = "Population Transfer"

    def build(self):
        """Initialize experiment & variables."""
        # Add RFSoC arguments
        super().build()

        self.setattr_argument(
            "ions_to_address",
            artiq_env.StringValue("0"),
            tooltip="center-index! use python range notation e.g. -3,4:6 == [-3, 4, 5]",
        )
        self.setattr_argument(
            "unwind_durations",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(start=10 * ns, stop=200 * us, npoints=20,),
                unit="us",
                global_min=10 * ns,
            ),
        )
        self.setattr_argument("ind_amplitude1", rfsoc.AmplitudeArg())
        self.setattr_argument("global_amplitude1", rfsoc.AmplitudeArg())

        self.setattr_argument("ind_amplitude2", rfsoc.AmplitudeArg())
        self.setattr_argument("global_amplitude2", rfsoc.AmplitudeArg())

        self.setattr_argument(
            "rabi_time1",
            artiq_env.NumberValue(default=250 * us, unit="us", ndecimals=3),
        )

    def prepare(self):
        self.ions_to_address = parse_ion_input(self.ions_to_address)

        # Magic BasicEnvironment attributes
        self.num_steps = len(self.unwind_durations)
        self.scan_values = list(self.unwind_durations)

        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        super().prepare()

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = list()

        ion = self.ions_to_address[0]

        backend = backend = self.rfsoc_sbc.qiskit_backend

        for unwind_duration in self.unwind_durations:
            with qp.build(backend) as out_sched:
                global_channel = qp.control_channels()[0]
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion,
                        duration=self.rabi_time1,
                        individual_amp=self.ind_amplitude1,
                        global_amp=self.global_amplitude1,
                        backend=backend
                    )
                )
                qp.shift_phase(np.pi, global_channel)
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                        ion,
                        duration=unwind_duration,
                        individual_amp=self.ind_amplitude2,
                        global_amp=self.global_amplitude2,
                        backend=backend
                    )
                )

            schedule_list.append(out_sched)

        return schedule_list

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
    def prepare_step(self, istep):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        self.raman.set_global_aom_source(dds=True)
        delay(5 * us)

    def analyze(self):
        """Analyze and Fit data."""
        super(BasicEnvironment, self).analyze()

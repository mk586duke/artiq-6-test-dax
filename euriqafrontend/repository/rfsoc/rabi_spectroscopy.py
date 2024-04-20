import logging
import typing

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import qiskit.pulse as qp
from artiq.experiment import NumberValue
from artiq.experiment import TerminationRequested
from artiq.language.core import delay, delay_mu
from artiq.language.core import host_only, rpc
from artiq.language.core import kernel
from artiq.language.types import TInt32, TBool
from artiq.language.units import MHz
from artiq.language.units import us, ms

import euriqabackend.waveforms.single_qubit as single_qubit
import euriqafrontend.modules.rfsoc as rfsoc
import euriqafrontend.fitting as umd_fit
from euriqafrontend.modules.utilities import parse_ion_input
from euriqafrontend.repository.basic_environment import BasicEnvironment

from euriqafrontend.modules.artiq_dac import RampControl as rampcontrol_auto

_LOGGER = logging.getLogger(__name__)


class RamanRabiSpec(BasicEnvironment, artiq_env.Experiment):
    """rfsoc.RabiSpectroscopy"""

    data_folder = "rfsoc_rabispectroscopy"
    applet_name = "Raman RFSOC Spectroscopy"
    applet_group = "RFSOC"
    fit_type = umd_fit.positive_gaussian
    units = MHz
    ylabel = "Population Transfer"
    xlabel = "Detuning (MHz)"

    def build(self):
        """Initialize experiment & variables."""
        super().build()
        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=2.85 * MHz, stop=3.0 * MHz, npoints=20, randomize=False,
                ),
                ndecimals=4,
                unit="MHz",
                global_min=-5 * MHz,
                global_max=5 * MHz,
            ),
        )
        self.setattr_argument(
            "ions_to_address",
            artiq_env.StringValue("0"),
            tooltip="use python range notation e.g. -3,4:6 == [-3, 4, 5]",
        )

        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=250 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument(
            "sideband_order",
            NumberValue(1, scale=1, min=-3, max=3, step=1, ndecimals=0),
        )

        self.setattr_argument("rabi_global_amplitude", rfsoc.AmplitudeArg())
        self.setattr_argument("rabi_individual_amplitude", rfsoc.AmplitudeArg())

        self.setattr_argument(
            "phase_insensitive", artiq_env.BooleanValue(default=False)
        )
        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(default=True))
        self.use_AWG = False

        # for testing RF ramping
        self.rf_ramp_auto = rampcontrol_auto(self)

        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            tooltip="True - ramp the rf down and back up before raman pulse",
            group="RF Ramping",
        )

        # Hard coded modulation depth for ramping
        self.rf_ramp_modulation_depth = 76

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for detuning in self.scan_values:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
# for ion in self.qubits_to_address:
                qp.call(
                    single_qubit.square_rabi_by_amplitude(
                            ion_index=self.qubits_to_address,
                            duration=self.rabi_time,
                            detuning=detuning,
                            sideband_order=self.sideband_order,
                            individual_amp=self.rabi_individual_amplitude,
                            global_amp=self.rabi_global_amplitude,
                            phase_insensitive=self.phase_insensitive,
                            stark_shift=[0 for x in self.qubits_to_address]
                    )
                )

            schedule_list.append(out_sched)

        return schedule_list

    def prepare(self):
        # this is an rfsoc-based experiment
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )

        if not hasattr(self, "scan_values"):
            # prevent overwriting scan values if set at in an inheriting experiment.
            self.scan_values = list(self.scan_range)
        self.num_steps = len(self.scan_values)

        # TODO: change this conversion to match the new numbering scheme.
        self.qubits_to_address = parse_ion_input(self.ions_to_address)

        # for testing RF ramping
        self.rf_ramp_auto.prepare()

        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.custom_experiment_initialize()

            _LOGGER.debug("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.debug("Done with Experiment. Setting machine to idle state")

    @host_only
    def custom_experiment_initialize(self):
        pass

    @kernel
    def custom_kn_init(self):
        # if self.ramp_rf:
        #     self.core.break_realtime()
        #     self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
        #     self.core.break_realtime()
        #     self.rf_lock_control(True)
        #     self.rf_ramp.run_ramp_down_kernel()
        pass
    @kernel
    def custom_kn_idle(self):
        # if self.ramp_rf:
        #     self.core.break_realtime()
        #     self.rf_ramp.run_ramp_up_kernel()
        #     self.rf_ramp.deactivate_ext_mod()
        #     # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
        #     self.core.break_realtime()
        #     self.rf_lock_control(False)
        pass
    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        self.raman.set_global_aom_source(dds=True)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()

        num_fits = len(self.p_all["x0"])
        for i_fit, (x0, x0_error) in enumerate(
            zip(self.p_all["x0"], self.p_error_all["x0"])
        ):
            _LOGGER.info(
                "Fit %i/%i: x0 = (%f +- %f) MHz", i_fit, num_fits, x0, x0_error,
            )

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

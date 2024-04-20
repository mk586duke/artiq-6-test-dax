"""Temporary file for testing arbitrary Qiskit Pulse schedules."""
import importlib
import logging

import artiq.language.environment as artiq_env
import oitg.fitting as fit
from artiq.experiment import TerminationRequested
from artiq.language.core import delay, delay_mu, host_only, kernel
from qiskit.assembler import disassemble
from artiq.language.units import us

import euriqafrontend.fitting as umd_fit
from euriqafrontend.repository.basic_environment import BasicEnvironment

_LOGGER = logging.getLogger(__name__)


class RFSoCSequence(BasicEnvironment, artiq_env.Experiment):
    """Pulse Sequence Output

    Plays an arbitrary set of pulses from a Qiskit Pulse QObj.
    Requires compiling the Pulse QObj separately, but it can just be pasted in once it
    is compiled.
    """

    data_folder = "rfsoc_pulse_sequence"
    applet_name = "RFSoC Pulse Sequence"
    applet_group = "RFSOC"
    units = 1
    ylabel = "Population Transfer"

    fit_dict = {
        "None": None,
        "rabi_flop": umd_fit.rabi_flop,
        "cos_fft": fit.cos_fft,
        "positive_gaussian": umd_fit.positive_gaussian,
        "negative_gaussian": umd_fit.negative_gaussian,
        "cos2pi": umd_fit.cos2pi,
        "sin": fit.sin,
        "sin^2": fit.sin_2,
    }

    def build(self):
        """Initialize experiment & variables."""
        # Add RFSoC arguments
        super().build()

        self.setattr_argument("schedule_module", artiq_env.StringValue(default=""))
        self.setattr_argument(
            "import_schedule_from_module", artiq_env.BooleanValue(False)
        )
        self.setattr_argument("xlabel", artiq_env.StringValue(default="Sequence Index"))
        self.setattr_argument("x_values", artiq_env.PYONValue(default=None))

        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(True))
        self.setattr_argument("use_AWG", artiq_env.BooleanValue(False))
        self.fit_type_str = self.get_argument(
            "fit type",
            artiq_env.EnumerationValue(sorted(self.fit_dict.keys()), default="None"),
        )

    def prepare(self):
        if self.fit_type_str != "":
            self.fit_type = self.fit_dict[self.fit_type_str]
        # Use & interpret arguments
        if not self.import_schedule_from_module:
            pulse_qobj = self.rfsoc_sbc._find_pulse_schedule()

            # Magic BasicEnvironment attributes
            self.num_steps = len(pulse_qobj.experiments)
            if self.x_values is not None:
                self.scan_values = self.x_values
            else:
                self.scan_values = list(range(self.num_steps))

            self.set_variables(
                data_folder=self.data_folder,
                applet_name=self.applet_name,
                applet_group=self.applet_group,
                fit_type=self.fit_type,
                units=self.units,
                ylabel=self.ylabel,
                xlabel=self.xlabel,
            )

            (input_schedules, _run_config, _user_qobj_header,) = disassemble(pulse_qobj)
        else:
            pulse_module = importlib.import_module(self.schedule_module)
            self.schedules = pulse_module.schedules
            self.x_values = getattr(
                pulse_module, "x_values", range(len(self.schedules))
            )
            self.xlabel = getattr(pulse_module, "x_label", "index")
            self.num_steps = len(self.schedules)
            self.scan_values = self.x_values

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

    @host_only
    def custom_pulse_schedule_list(self):
        return self.schedules

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
        if self.fit_type is not None:
            super().analyze()
            for fit_label, values in self.p_all.items():
                print(f"Fit[{fit_label}]: {values}")

# NOTE: this file is out of date, and has not been updated with several improvements
# If you want to use it, check the updated API for the RFCompiler.
import logging
import time
import warnings

import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import artiq.language.units as artiq_units
import numpy as np
import oitg.fitting as fit
from artiq.experiment import NumberValue
from artiq.experiment import TerminationRequested
from artiq.language.core import delay
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import parallel
from artiq.language.types import TInt32
from artiq.language.units import MHz
from artiq.language.units import us

import euriqabackend.devices.keysight_awg.gate as gate
import euriqafrontend.fitting as umd_fit
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend.repository.basic_environment import BasicEnvironment

_LOGGER = logging.getLogger(__name__)


class RamanAWGParity(BasicEnvironment, artiq_env.Experiment):
    """Raman.AWG.Parity
    """

    data_folder = "raman_awg"
    applet_name = "Raman Parity"
    applet_group = "Raman"
    fit_type = umd_fit.positive_gaussian
    units = artiq_units.us
    ylabel = "Parity"
    xlabel = "Rabi Pulse Length (us)"

    def build(self):
        """Initialize experiment & variables."""

        self.setattr_argument(
            "awg_pulse_length",
            artiq_env.NumberValue(default=500 * us, unit="us", min=1 * us),
        )
        self.setattr_argument(
            "awg_scan_length", artiq_env.NumberValue(default=21, ndecimals=0)
        )

        self.awg_bits = 7
        self.awg_register = list()
        for i in range(self.awg_bits):
            self.awg_register.append(self.get_device("awg_bit{:d}".format(i)))
        super().build()

        self.setattr_device("rf_compiler")

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            ylabel=self.ylabel,
            xlabel=self.xlabel,
        )
        # self.scan_values = [np.int32(int(self.awg_scan_length)-t-1) for t in range(int(self.awg_scan_length))]
        self.scan_values = [np.int32(int(t)) for t in range(int(self.awg_scan_length))]
        print(self.scan_values)
        self.num_steps = len(self.scan_values)
        super().prepare()

        self.f_ind = 57
        self.f_carrier = 161.748398
        self.t_delay = 3.0
        self.Tpi = 2
        self.monitor_ind = True
        self.monitor_detuning = 1
        self.amp_monitor = 1

        self.amp_ind_Rabi = 1
        self.amp_global_Rabi = 1
        self.Stark_shift_Rabi = 0

        self.amp_ind_SK1 = (0.3 + 0.85) / 2
        self.amp_global_SK1 = 1
        self.Stark_shift_SK1 = 0

        self.N_ions = 15
        self.XX_sol_name = "XX_GateSolutions"
        self.amp_ind_XX = 0.61
        self.amp_global_XX = 1
        self.Stark_shift_XX = 0.0009

        self.calibrated_Tpi = [1.99] * 32
        # self.calibrated_Tpi[2] = 0.989
        # self.calibrated_Tpi[4] = 0.936
        # self.calibrated_Tpi[6] = 0.949
        # calibrated_Tpi = [0.99] * 32
        self.AOM_saturation_params = [100.0] * 32

        self.rf_compiler.set_general_params(
            self.f_carrier,
            self.f_ind,
            self.N_ions,
            self.t_delay,
            self.Tpi,
            self.monitor_ind,
            self.monitor_detuning,
            self.amp_monitor,
        )

        self.rf_compiler.set_rabi_params(
            self.amp_ind_Rabi, self.amp_global_Rabi, self.Stark_shift_Rabi
        )

        self.rf_compiler.set_SK1_params(
            self.amp_ind_SK1, self.amp_global_SK1, self.Stark_shift_SK1
        )

        warnings.warn(
            "Raman AWG parity experiment uses outdated version of RFCompiler. "
            "It will not work as expected, probably will fail.",
            DeprecationWarning,
        )
        # TODO: invalid after Drew's RF Compiler upgrade, but it was out of date before.
        self.rf_compiler.set_XX_params(
            self.XX_sol_name, self.amp_ind_XX, self.amp_global_XX, self.Stark_shift_XX
        )

        self.rf_compiler.set_AOM_levels(self.calibrated_Tpi)

        self.rf_compiler.set_AOM_saturation_params(self.AOM_saturation_params)

        # self.rf_compiler.rabi_exp(slots=[15,19],
        #                 detuning=0,
        #                 sideband_order=0,
        #                 scan_parameter_int=int(RFCompiler.RabiScanParameter.duration),
        #                 min_value=0,
        #                 max_value=40,
        #                 N_points=41)
        # self.rf_compiler.SK1_exp(slots=[19],
        #                          theta=1.07*np.pi/2,
        #                          scan_parameter_int=int(RFCompiler.SK1ScanParameter.ind_amplitude),
        #                          min_value=0.3,
        #                          max_value=0.85,
        #                          N_points=21,
        #                          phi=0)
        #
        self.rf_compiler.wait_after_time = 0
        self.rf_compiler.name = "XX parity scan"

        common_phase = 0
        relative_phase = 0

        sweep_relative_phase = 0

        if sweep_relative_phase == 0:
            min_phase_1 = -relative_phase / 2
            max_phase_1 = -relative_phase / 2 + 2 * np.pi
            min_phase_2 = relative_phase / 2
            max_phase_2 = relative_phase / 2 + 2 * np.pi
        else:
            min_phase_1 = common_phase
            max_phase_1 = common_phase + np.pi
            min_phase_2 = common_phase
            max_phase_2 = common_phase - np.pi

        mytheta = np.pi / 2 * 1.07

        self.rf_compiler.add_XX(slots=[15, 19], gate_name="XX")
        # scan_parameter_int = int(RFCompiler.XXScanParameter.ind_amplitude),
        # min_value = 0.61,
        # max_value = 0.62,
        # N_points = 25)
        self.rf_compiler.add_SK1(
            slots=[15],
            theta=mytheta,
            phi=0,
            gate_name="Analysis SK1",
            scan_parameter_int=int(gate.SK1.ScanParameter.phi),
            min_value=min_phase_1,
            max_value=max_phase_1,
            N_points=21,
        )
        self.rf_compiler.add_SK1(
            slots=[19],
            theta=mytheta,
            phi=0,
            gate_name="Analysis SK1",
            scan_parameter_int=int(gate.SK1.ScanParameter.phi),
            min_value=min_phase_2,
            max_value=max_phase_2,
            N_points=21,
        )

        self.rf_compiler.generate_waveforms()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.rf_compiler.select_exp_to_program()
            self.rf_compiler.program_AWG()

            self.custom_init()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")
            self.experiment_idle()
            self.custom_idle()

    @kernel
    def custom_init(self):
        self.raman.collective_dds.enabled = False

    @kernel
    def custom_idle(self):
        pass

    @kernel
    def prepare_step(self, istep: TInt32):
        self.set_scan_state(self.scan_values[istep])

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        # delay(150*us)
        self.awg_trigger.pulse(1 * us)
        delay(self.awg_pulse_length)
        self.raman.set_global_aom_source(dds=True)

    @kernel
    def set_scan_state(self, index: TInt32):
        loc_index = index
        for i in range(self.awg_bits):
            on = loc_index & 1
            loc_index = loc_index >> 1
            if on:
                self.awg_register[i].on()
            else:
                self.awg_register[i].off()
        delay(100 * us)

    def analyze(self):
        """Analyze and Fit data"""
        """Threshold values and analyze."""
        super().analyze()
        for ifit in range(len(self.p_all["x0"])):
            print(
                "Fit",
                ifit,
                ": ",
                "x0 = (",
                self.p_all["x0"][ifit],
                "+-",
                self.p_error_all["x0"][ifit],
                ") ",
            )

import logging
import time

import typing
import qiskit.pulse as qp
import euriqafrontend.modules.rfsoc as rfsoc
import euriqabackend.waveforms.multi_qubit as multi_qubit
import euriqabackend.waveforms.single_qubit as single_qubit


import artiq.language.environment as artiq_env
import artiq.language.scan as artiq_scan
import artiq.language.units as artiq_units
import numpy as np
import oitg.fitting as fit
from artiq.experiment import NumberValue
from artiq.experiment import StringValue
from artiq.language.core import delay, delay_mu
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import rpc
from artiq.language.core import TerminationRequested
from artiq.language.core import parallel, sequential
from artiq.language.types import TInt32
from artiq.language.units import MHz
from artiq.language.units import ms
from artiq.language.units import us

import euriqabackend.coredevice.dac8568 as dac8568
import euriqabackend.devices.keysight_awg.gate as gate
import euriqafrontend.fitting as umd_fit
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend.repository.basic_environment import BasicEnvironment

_LOGGER = logging.getLogger(__name__)


class CheckDX(BasicEnvironment, artiq_env.EnvExperiment):
    """Check.DX.

    Relax the potential to the relaxed axial line and center DX to maximize Rabi on the center PMT channel
    """

    # kernel_invariants = {"detect_time", "pmt_array", "num_pmts"}
    # applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi" + " "   # White space is required

    data_folder = "check.DX"
    applet_name = "DX Calibration"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=2, stop=5, npoints=11, randomize=False, seed=int(time.time())
                ),
                unit="",
                global_min=-70.0,
                global_max=70.0,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=2 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument(
            "X2", artiq_env.NumberValue(default=0.0, unit="", step=2.5e-5, ndecimals=5)
        )

        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=5))

        self.scan_values = [np.float(0)]
        self.raman_time_mu = np.int32(0)

        super().build()

        # Manually override the check for lost ions to avoid false alarms during calibration routine
        self.setattr_argument(
            "lost_ion_monitor", artiq_env.EnumerationValue(["False"], default="False")
        )

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.lost_ion_monitor = False

        super().prepare()

        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]
        self.raman_time_mu = self.core.seconds_to_mu(self.rabi_time)

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def main_experiment(self, istep, ishot):
        # reset DDS settings after potential SBC
        self.raman.global_dds.update_amp(self.raman.global_dds.amp)
        self.raman.switchnet_dds.update_amp(self.raman.switchnet_dds.amp)
        self.raman.set_global_detuning_mu(np.int64(0))
        # wait for the DDS amplitudes to settle
        delay(200 * us)
        self.raman.pulse_mu(self.raman_time_mu)

    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(x2=self.X2, dx=self.scan_values[istep])
        delay(100 * ms)

    @rpc(flags={"async"})
    def update_DAC(self, x2, dx):
        self.sandia_box.X2 = x2
        print("update_DAC : {}".format(dx))
        self.sandia_box.DX = dx
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Quantum_AxialRelaxed")

        # apply Quantum_AxialRelaxed
        self.sandia_box.dac_pc.apply_line_async("Quantum_AxialRelaxed")

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        num_active_pmts = len(self.pmt_array.active_pmts)

        for ifit in range(num_active_pmts):
            print(
                "Fit {}:\n\tx0 = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                )
            )

        print(
            "pmt {}: aligned DX = {}".format(
                center_pmt_idx + 1,
                self.p_all["x0"][center_pmt_idx],
            )
        )
        if self.set_globals:
            self.set_dataset(
                "global.Voltages.Offsets.DX",
                self.p_all["x0"][center_pmt_idx],
                persist=True,
            )

class CheckCenterRFSOC(BasicEnvironment, artiq_env.EnvExperiment):
    """Check.Center.RFSOC
    """

    # kernel_invariants = {"detect_time", "pmt_array", "num_pmts"}
    # applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi" + " "   # White space is required

    data_folder = "check.Center"
    applet_name = "Center Calibration"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian
    xlabel = "Potential Center (um)"
    ylabel = "Population Transfer"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=-2, stop=2, npoints=21, randomize=False, seed=int(time.time())
                ),
                unit="",
                global_min=-30.0,
                global_max=30.0,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=0.5 * us, unit="us", step=1e-7, ndecimals=7),
        )
        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=1.0))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=1.0))
        self.setattr_argument("phase_insensitive", artiq_env.BooleanValue(default=False))
        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument("auto_reload", artiq_env.BooleanValue(default=False))
        self.setattr_argument(
            "repetition", artiq_env.NumberValue(default=0, step=1, ndecimals=0)
        )

        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=8))

        self.scan_values = [np.float(0)]

        super().build()

        self.setattr_device("scheduler")

        self.setattr_argument("rabi_global_amp", rfsoc.AmplitudeArg(default=0.5))
        self.setattr_argument("rabi_ind_amp", rfsoc.AmplitudeArg(default=0.5))

        # Manually override the check for lost ions to avoid false alarms during calibration routine
        self.do_calib = False
        self.use_RFSOC = True

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )
        super().prepare()

        # self.set_dataset("global.Voltages.X2", 0.0032, persist=True)

        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def custom_experiment_initialize(self):
        pass

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.original_piezo_setting = self.get_dataset(
                "global.Raman.Piezos.Ind_FinalX"
            )

            self.custom_experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for _center_position in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                all_ions = list(
                    qp.active_backend().configuration().all_qubit_indices_iter
                )
                for ion in all_ions:
                    qp.call(
                        single_qubit.square_rabi_by_amplitude(
                            ion_index=ion,
                            phase_insensitive=self.phase_insensitive,
                            duration=self.rabi_time,
                            individual_amp=self.rabi_ind_amp,
                            global_amp=self.rabi_global_amp,
                            detuning=0.0
                        )
                )

            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(1 * ms)

    @rpc
    def update_DAC(self, cent):
        self.sandia_box.center = cent
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start", line_gain=1)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        num_active_pmts = len(self.pmt_array.active_pmts)
        buf = "{"
        for ifit in range(num_active_pmts):
            ret = self.p_all["x0"][ifit]
            buf = buf + "%f," % ret
            print(
                "Fit {}:\n\tx0 = ({} +- {})\n\tsigma = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                    self.p_all["sigma"][ifit],
                    self.p_error_all["sigma"][ifit],
                )
            )
        buf = buf + "}"
        print(buf)
        print(
            "pmt {}: aligned Center = {}".format(
                center_pmt_idx + 1, self.p_all["x0"][center_pmt_idx]
            )
        )
        if self.set_globals:
            self.set_dataset(
                "global.Voltages.center",
                self.p_all["x0"][center_pmt_idx],
                broadcast=True,
                persist=True,
            )


class CheckCenterBichromatic(BasicEnvironment, artiq_env.EnvExperiment):
    """Check.Center.Bichromatic
    """

    # kernel_invariants = {"detect_time", "pmt_array", "num_pmts"}
    # applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi" + " "   # White space is required

    data_folder = "check.Center"
    applet_name = "Center Calibration"
    applet_group = "Raman Calib"
    fit_type = umd_fit.negative_gaussian
    xlabel = "Potential Center (um)"
    ylabel = "Population Transfer"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=-2, stop=2, npoints=21, randomize=False, seed=int(time.time())
                ),
                unit="",
                global_min=-30.0,
                global_max=30.0,
            ),
        )
        self.setattr_argument(
            "drive_time",
            artiq_env.NumberValue(default=0.5 * us, unit="us", step=1e-7, ndecimals=7),
        )
        self.setattr_argument("global_amp", rfsoc.AmplitudeArg(default=1.0))
        self.setattr_argument("ind_amp", rfsoc.AmplitudeArg(default=1.0))
        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument("auto_reload", artiq_env.BooleanValue(default=False))
        self.setattr_argument(
            "repetition", artiq_env.NumberValue(default=0, step=1, ndecimals=0)
        )

        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=8))

        self.scan_values = [np.float(0)]

        super().build()

        self.setattr_device("scheduler")

        # Manually override the check for lost ions to avoid false alarms during calibration routine
        self.do_calib = False
        self.use_RFSOC = True

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )
        super().prepare()

        # self.set_dataset("global.Voltages.X2", 0.0032, persist=True)

        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def custom_experiment_initialize(self):
        pass

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            self.original_piezo_setting = self.get_dataset(
                "global.Raman.Piezos.Ind_FinalX"
            )

            self.custom_experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedule_list = []

        for _center_position in self.scan_range:
            with qp.build(backend=self.rfsoc_sbc.qiskit_backend) as out_sched:
                all_ions = list(
                    qp.active_backend().configuration().all_qubit_indices_iter
                )
                for ion in all_ions:
                    qp.call(single_qubit.sk1_gaussian(ion,np.pi/2,0))
                    qp.call(
                        multi_qubit.bichromatic_drive(
                            [ion],
                            self.rabi_time,
                            [self.rabi_ind_amp],
                            [0],
                            [np.pi/2], #phis
                            0,
                            0,
                            0.02, #sb imbalance
                            0.8,
                            motional_frequency_adjustment=632e3 #global amp
                        )
                    )
                    qp.call(single_qubit.sk1_gaussian(ion,-np.pi/2,0))

            schedule_list.append(out_sched)

        return schedule_list

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_aom_source(dds=False)
        delay(5 * us)
        self.rfsoc_sbc.trigger()
        delay_mu(self.sequence_durations_mu[istep])
        delay(5 * us)
        self.raman.set_global_aom_source(dds=True)

    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(1 * ms)

    @rpc
    def update_DAC(self, cent):
        self.sandia_box.center = cent
        self.sandia_box.update_compensations()
        self.sandia_box.dac_pc.apply_line_async("QuantumRelaxed_ZeroLoad", line_gain=0)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        num_active_pmts = len(self.pmt_array.active_pmts)
        buf = "{"
        for ifit in range(num_active_pmts):
            ret = self.p_all["x0"][ifit]
            buf = buf + "%f," % ret
            print(
                "Fit {}:\n\tx0 = ({} +- {})\n\tsigma = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                    self.p_all["sigma"][ifit],
                    self.p_error_all["sigma"][ifit],
                )
            )
        buf = buf + "}"
        print(buf)
        print(
            "pmt {}: aligned Center = {}".format(
                center_pmt_idx + 1, self.p_all["x0"][center_pmt_idx]
            )
        )
        if self.set_globals:
            self.set_dataset(
                "global.Voltages.center",
                self.p_all["x0"][center_pmt_idx],
                broadcast=True,
                persist=True,
            )

class CheckCenter(BasicEnvironment, artiq_env.EnvExperiment):
    """Check.Center.

    Relax the potential to the relaxed axial line and center DX to maximize Rabi on the center PMT channel
    """

    # kernel_invariants = {"detect_time", "pmt_array", "num_pmts"}
    # applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi" + " "   # White space is required

    data_folder = "check.Center"
    applet_name = "Center Calibration"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian
    xlabel = "Potential Center (um)"
    ylabel = "Population Transfer"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=-5, stop=5, npoints=21, randomize=False, seed=int(time.time())
                ),
                unit="",
                global_min=-30.0,
                global_max=30.0,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=2 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=5))

        self.scan_values = [np.float(0)]
        self.raman_time_mu = np.int32(0)

        super().build()

        # Manually override the check for lost ions to avoid false alarms during calibration routine
        self.setattr_argument(
            "lost_ion_monitor", artiq_env.EnumerationValue(["False"], default="False")
        )

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel
        )
        self.lost_ion_monitor = False
        super().prepare()

        # Having calibrate active can cause loss of ions because it will send HW triggers which can interrupt the
        # asyncronous data upload used to sweep Center. Here we manually override.
        if self.do_calib is True:
            self.do_calib = False

        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]
        self.raman_time_mu = self.core.seconds_to_mu(self.rabi_time)

        self.global_amp_mu = np.int32(self.raman.global_dds.amp)
        self.ind_amp_mu = np.int32(self.raman.switchnet_dds.amp)

        self.coll_freq_mu = np.int64(freq_to_mu(np.float(197.0)*MHz))

        self.piezoX_initial = 2.5

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def custom_experiment_initialize(self):
        # self.piezoX_initial = self.get_dataset("global.Raman.Piezos.Ind_FinalX")
        # self.set_dataset(
        #     "global.Raman.Piezos.Ind_FinalX",
        #     2.5,
        #     broadcast=True,
        #     persist=True,
        # )
        pass

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

    @kernel
    def main_experiment(self, istep, ishot):

        #reset DDS settings after potential SBC
        self.raman.global_dds.update_amp(self.global_amp_mu)
        delay(5 * us)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_global_detuning_mu(np.int64(0))
        delay(5*us)
        # self.raman.set_ind_attenuation(c8=True, c16=True)
        # self.raman.global_dds.update_amp(0)
        # delay(5 * us)
        # self.raman.switchnet_dds.update_amp(0)
        # delay(5 * us)
        # self.raman.set_global_detuning_mu(np.int64(0.1*MHz))

        # wait for the DDS amplitudes to settle
        delay(80 * us)
        #self.raman.pulse_mu(self.raman_time_mu)

        with parallel:
            with sequential:
                delay_mu(-self.raman._GLOBAL_DELAY_MU)
                self.raman.global_dds.on()
                delay_mu(self.raman._GLOBAL_DELAY_MU)
            self.raman.switchnet_dds.on()
        delay_mu(self.raman_time_mu)

        with parallel:
            with sequential:
                self.raman.switchnet_dds.off()
                self.raman.global_dds.off()


    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(100 * ms)

    @rpc(flags={"async"})
    def update_DAC(self, cent):
        self.sandia_box.dac_pc.line_gain = 0
        self.sandia_box.center = cent
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start",line_gain=1)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        num_active_pmts = len(self.pmt_array.active_pmts)
        buf = "{"
        for ifit in range(num_active_pmts):
            ret = self.p_all["x0"][ifit]
            buf = buf + "%f," % ret
            print(
                "Fit {}:\n\tx0 = ({} +- {})\n\tsigma = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                    self.p_all["sigma"][ifit],
                    self.p_error_all["sigma"][ifit],
                )
            )
        buf = buf + "}"
        print(buf)
        print(
            "pmt {}: aligned Center = {}".format(
                center_pmt_idx + 1,
                self.p_all["x0"][center_pmt_idx],
            )
        )
        if self.set_globals:
            # self.set_dataset(
            #     "global.Raman.Piezos.Ind_FinalX",
            #     self.piezoX_initial,
            #     broadcast=True,
            #     persist=True,
            # )
            self.set_dataset(
                "global.Voltages.center",
                self.p_all["x0"][center_pmt_idx],
                persist=True,
            )


class CheckInspectCenter(BasicEnvironment, artiq_env.EnvExperiment):
    """Check.Inspect.Center.

    Relax the potential to the relaxed axial line and center DX to maximize Rabi on the center PMT channel
    """

    # kernel_invariants = {"detect_time", "pmt_array", "num_pmts"}
    # applet_stream_cmd = "$python -m euriqafrontend.applets.plot_multi" + " "   # White space is required

    data_folder = "check.Center"
    applet_name = "Center Calibration"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian
    xlabel = "Potential Center (um)"
    ylabel = "Population Transfer"

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=-50, stop=50, npoints=21, randomize=False, seed=int(time.time())
                ),
                unit="",
                global_min=-100.0,
                global_max=100.0,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=2 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument(
            "X2",
            artiq_env.NumberValue(default = 0.02)
        )

        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=5))

        self.scan_values = [np.float(0)]
        self.raman_time_mu = np.int32(0)

        super().build()

        # Manually override the check for lost ions to avoid false alarms during calibration routine
        self.setattr_argument(
            "lost_ion_monitor", artiq_env.EnumerationValue(["False"], default="False")
        )

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            xlabel=self.xlabel,
            ylabel=self.ylabel
        )
        self.lost_ion_monitor = False

        # Autoloader uses a different shuttle file from basic environment
        self.sandia_box.shuttle_flag = False
        self.dac_starting_node = "QuantumRelaxed_LoadOn"
        super().prepare()

        # Having calibrate active can cause loss of ions because it will send HW triggers which can interrupt the
        # asyncronous data upload used to sweep Center. Here we manually override.
        if self.do_calib is True:
            self.do_calib = False

        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]
        self.raman_time_mu = self.core.seconds_to_mu(self.rabi_time)

        self.global_amp_mu = np.int32(self.raman.global_dds.amp)
        self.ind_amp_mu = np.int32(self.raman.switchnet_dds.amp)

        self.coll_freq_mu = np.int64(freq_to_mu(np.float(197.0)*MHz))

        self.piezoX_initial = 2.5

        center_global = self.get_dataset("global.Voltages.center")
        x2_offset_global = self.get_dataset("global.Voltages.Offsets.X2")
        self.DX_global = self.get_dataset("global.Voltages.Offsets.DX")
        DZ_global = self.get_dataset("global.Voltages.Offsets.DZ")
        # x2_global = self.get_dataset("global.Voltages.X2")

        self.sandia_box.X2_offset = float(x2_offset_global)
        # self.sandia_box.X1 = 0 * float(self.X1)
        self.sandia_box.X2 = self.X2
        self.sandia_box.X3 = 0
        self.sandia_box.X4 = 0
        self.sandia_box.X4_offset = 0
        self.sandia_box.QXZ = 0
        self.sandia_box.QXZ_offset = 0
        self.sandia_box.QZZ = 0
        self.sandia_box.QZZ_offset = 0
        self.sandia_box.QZY = 0
        self.sandia_box.QZY_offset = 0
        self.sandia_box.center = 0
        self.sandia_box.DX = 0
        self.sandia_box.DZ = DZ_global
        # self.sandia_box.dac_pc.tweak_dictionary = {613: {"DX": -20.0 / 1e3}}
        self.sandia_box.calculate_compensations()



        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def custom_experiment_initialize(self):
        # self.piezoX_initial = self.get_dataset("global.Raman.Piezos.Ind_FinalX")
        # self.set_dataset(
        #     "global.Raman.Piezos.Ind_FinalX",
        #     2.5,
        #     broadcast=True,
        #     persist=True,
        # )
        pass

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

    @kernel
    def main_experiment(self, istep, ishot):
        #reset DDS settings after potential SBC
        self.raman.global_dds.update_amp(self.global_amp_mu)
        delay(5 * us)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_global_detuning_mu(np.int64(0))
        delay(5*us)
        # self.raman.set_ind_attenuation(c8=True, c16=True)
        # self.raman.global_dds.update_amp(0)
        # delay(5 * us)
        # self.raman.switchnet_dds.update_amp(0)
        # delay(5 * us)
        # self.raman.set_global_detuning_mu(np.int64(0.1*MHz))

        # wait for the DDS amplitudes to settle
        delay(130 * us)
        #self.raman.pulse_mu(self.raman_time_mu)

        with parallel:
            with sequential:
                delay_mu(-self.raman._GLOBAL_DELAY_MU)
                self.raman.global_dds.on()
                delay_mu(self.raman._GLOBAL_DELAY_MU)
            self.raman.switchnet_dds.on()
        delay_mu(self.raman_time_mu)

        with parallel:
            self.raman.switchnet_dds.off()
            self.raman.global_dds.off()


    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(self.scan_values[istep])
        delay(100 * ms)

    @rpc(flags={"async"})
    def update_DAC(self, dx):
        self.sandia_box.dac_pc.line_gain = 1
        self.sandia_box.DX = dx + self.DX_global
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Pos.m200.p0")
        self.sandia_box.dac_pc.apply_line_async("Pos.m200.p0",line_gain=1)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        print(self.center_pmt_number)
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        num_active_pmts = len(self.pmt_array.active_pmts)
        buf = "{"
        for ifit in range(num_active_pmts):
            ret = self.p_all["x0"][ifit]
            buf = buf + "%f," % ret
            print(
                "Fit {}:\n\tx0 = ({} +- {})\n\tsigma = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                    self.p_all["sigma"][ifit],
                    self.p_error_all["sigma"][ifit],
                )
            )
        buf = buf + "}"
        print(buf)
        print(
            "pmt {}: aligned Center = {}".format(
                center_pmt_idx + 1,
                self.p_all["x0"][center_pmt_idx],
            )
        )
        if self.set_globals:
            self.set_dataset(
                "monitor.inspect_DX_Offset",
                self.p_all["x0"][center_pmt_idx],
                persist=True,)





class CheckDZ(BasicEnvironment, artiq_env.EnvExperiment):
    """Check.DZ."""

    data_folder = "check.DZ"
    applet_name = "DZ Calibration"
    applet_group = "Raman Calib"
    fit_type = umd_fit.negative_gaussian

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=180,
                    stop=500,
                    npoints=11,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="",
                global_min=0,
                global_max=800,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=50 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=5))
        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=1000))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=1000))

        self.scan_values = [np.float(0)]
        self.raman_time_mu = np.int32(0)
        self.detuning_mu = np.int64(0)

        super().build()

        # Manually override the check for lost ions to avoid false alarms during calibration routine
        self.setattr_argument(
            "lost_ion_monitor", artiq_env.EnumerationValue(["False"], default="False")
        )

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.lost_ion_monitor = False

        super().prepare()
        self.detuning_mu = freq_to_mu(self.get_dataset("global.Ion_Freqs.RF_Freq"))
        self.global_amp_mu = np.int32(self.rabi_global_amp)
        self.ind_amp_mu = np.int32(self.rabi_ind_amp)

        self.num_steps = len(self.scan_range)
        self.scan_values = [val for val in self.scan_range]
        self.raman_time_mu = self.core.seconds_to_mu(self.rabi_time)

        _LOGGER.debug("Done Preparing Experiment")

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.global_dds.update_amp(self.global_amp_mu)
        delay(5 * us)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_global_detuning_mu(self.detuning_mu)
        delay(150 * us)
        self.raman.pulse_mu(self.raman_time_mu)

    @kernel
    def prepare_step(self, istep: TInt32):
        self.update_DAC(dz=self.scan_values[istep])
        delay(100 * ms)

    @rpc(flags={"async"})
    def update_DAC(self, dz):
        self.sandia_box.DZ = dz
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_line_all_compensations("Start")
        self.sandia_box.dac_pc.apply_line_async("Start",line_gain=1)
        # self.sandia_box.dac_pc.shuttle_async(to_line_or_name="QuantumRelaxed_ZeroLoad")

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
        # num_active_pmts = len(self.pmt_array.active_pmts)

        for ifit in range(self.num_pmts):
            print(
                "Fit {}: x0 = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                )
            )

        print(
            "pmt {}: aligned DZ = {}".format(
                center_pmt_idx + 1,
                self.p_all["x0"][center_pmt_idx],
            )
        )
        if self.set_globals:
            self.set_dataset(
                "global.Voltages.Offsets.DZ",
                self.p_all["x0"][center_pmt_idx],
                persist=True,
            )


class CheckCOM(BasicEnvironment, artiq_env.Experiment):
    """Check.COM."""

    data_folder = "check.COM"
    applet_name = "Raman Rabi Spectroscopy"
    applet_group = "Raman"

    fit_type = umd_fit.positive_gaussian

    def get_lower_com_freq(self):
        frf_sec = self.get_dataset("global.Ion_Freqs.frf_sec")
        qzy = self.get_dataset("global.Voltages.QZY")
        qzz = self.get_dataset("global.Voltages.QZZ")
        x2 = self.get_dataset("global.Voltages.X2")
        return 1e6 * np.sqrt(
            (frf_sec / 1e6) ** 2 - x2 / 2 - np.sqrt(qzy * qzy + qzz * qzz)
        )

    def get_upper_com_freq(self):
        frf_sec = self.get_dataset("global.Ion_Freqs.frf_sec")
        qzy = self.get_dataset("global.Voltages.QZY")
        qzz = self.get_dataset("global.Voltages.QZZ")
        x2 = self.get_dataset("global.Voltages.X2")
        return frf_sec - x2 / 2 + np.sqrt(qzy * qzy + qzz * qzz)

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=self.get_lower_com_freq() - 0.007 * MHz,
                    stop=self.get_lower_com_freq() + 0.007 * MHz,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                ndecimals=4,
                unit="MHz",
                global_min=0,
            ),
        )
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=250 * us, unit="us", step=1e-7, ndecimals=7),
        )

        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=0))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=0))

        # self.setattr_argument("set_global_Tpi", BooleanValue(default=False))
        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=5))
        super().build()

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )

        self.scan_values = [f for f in self.scan_range]
        self.scan_values_mu = [freq_to_mu(f) for f in self.scan_range]
        self.rabi_time_mu = self.core.seconds_to_mu(self.rabi_time)
        self.num_steps = len(self.scan_values)
        super().prepare()

        self.global_amp_mu = np.int32(self.rabi_global_amp)
        self.ind_amp_mu = np.int32(self.rabi_ind_amp)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.global_dds.update_amp(self.global_amp_mu)
        delay(5 * us)
        self.raman.switchnet_dds.update_amp(self.ind_amp_mu)
        delay(5 * us)
        self.raman.set_global_detuning_mu(self.scan_values_mu[istep])
        delay(150 * us)
        self.raman.pulse_mu(self.rabi_time_mu)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        for ifit in range(len(self.p_all["x0"])):
            print(
                "Fit {}: x0 = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                )
            )


class CheckCarrierRabi(BasicEnvironment, artiq_env.Experiment):
    """Check.Carrier_Rabi."""

    data_folder = "raman_rabi"
    applet_name = "Raman Rabi"
    applet_group = "Raman"
    fit_type = fit.rabi_flop
    units = artiq_units.us
    ylabel = "Qubit Population"
    xlabel = "Raman Rabi Pulse Length (us)"

    def build(self):
        """Initialize experiment & variables."""
        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0 * us,
                    stop=20 * us,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="us",
                global_min=0,
            ),
        )

        self.setattr_argument("set_global_Tpi", artiq_env.BooleanValue(default=False))
        self.setattr_argument("center_pmt_number", artiq_env.NumberValue(default=5))
        super().build()

    def prepare(self):
        self.set_variables(
            data_folder=self.data_folder,
            applet_name=self.applet_name,
            applet_group=self.applet_group,
            fit_type=self.fit_type,
            units=self.units,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
        )
        # self.scan_values = [self.core.seconds_to_mu(t_s) for t_s in self.scan_range]
        self.scan_values = [t_s for t_s in self.scan_range]
        self.num_steps = len(self.scan_values)
        super().prepare()

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.set_global_detuning_mu(np.int64(0))
        delay(150 * us)
        self.raman.pulse(self.scan_values[istep])

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        meanTpi = 0.0
        num_active_pmts = len(self.p_all["t_period"])
        for ifit in range(num_active_pmts):
            meanTpi += self.p_all["t_period"][ifit] * 0.5
            _LOGGER.info(
                "Fit %i:\n\tTpi = (%f +- %f) us\n\ttau = (%f +- %f) us",
                ifit,
                self.p_all["t_period"][ifit] * 0.5e6,
                self.p_error_all["t_period"][ifit] * 0.5e6,
                self.p_all["tau_decay"][ifit] * 1e6,
                self.p_error_all["tau_decay"][ifit] * 1e6,
            )
        meanTpi /= num_active_pmts

        if self.set_global_Tpi:
            Tpis = [0.0] * self.num_pmts
            for ifit in range(len(self.p_all["t_period"])):
                Tpis[self.pmt_array.active_pmts[ifit] - 1] = (
                    self.p_all["t_period"][ifit] * 0.5
                )

            center_pmt_idx = self.pmt_array.active_pmts.index(self.center_pmt_number)
            print(
                "center PMT: idx = {}, Tpi = {}".format(
                    self.center_pmt_number,
                    Tpis[center_pmt_idx],
                )
            )
            self.set_dataset("global.Raman.Ts_pi", Tpis, persist=True)
            self.set_dataset(
                "global.Raman.T_pi", Tpis[center_pmt_idx], persist=True
            )


class PiezoScanX(BasicEnvironment, artiq_env.Experiment):
    """Check.IndPiezoX."""

    data_folder = "piezo_scan"
    applet_name = "Rabi Piezo Scan"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0,
                    stop=3.0,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="",
                global_min=0,
                global_max=5.0,
            ),
        )
        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=500))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=1000))

        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=5.5 * us, unit="us", step=1e-7, ndecimals=7),
        )
        self.scan_values = [np.int32(0)]
        self.raman_time_mu = np.int32(0)
        super().build()

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.scan_values_mu = [
            dac8568.vout_to_mu(
                val,
                self.raman.ind_final_piezos.SandiaSerialDAC.V_OUT_MAX
            )
            for val in self.scan_range
        ]
        self.scan_values_mu.reverse()
        self.scan_values = [val for val in self.scan_range]
        self.scan_values.reverse()
        self.num_steps = len(self.scan_range)

        super().prepare()
        self.raman_time_mu = self.core.seconds_to_mu(self.rabi_time)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        self.raman.ind_final_piezos.value1_mu = self.scan_values_mu[istep]
        self.raman.ind_final_piezos.update_value()

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.global_dds.update_amp(self.rabi_global_amp)
        self.raman.switchnet_dds.update_amp(self.rabi_ind_amp)
        self.raman.set_global_detuning_mu(np.int64(0))
        # wait for the DDS amplitudes to settle
        delay(200 * us)
        self.raman.pulse_mu(self.raman_time_mu)

    def analyze(self):
        """Analyze and Fit data"""
        super().analyze()
        meanTpi = 0.0
        # num_active_pmts = len(self.p_all["x0"])
        mean_x0 = 0.0
        buf = "{"
        for ifit in range(self.num_pmts):
            ret = self.p_all["x0"][ifit]
            buf = buf + "%f," % ret
            mean_x0 += self.p_all["x0"][ifit]
            print(
                "Fit {}: x0 = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                )
            )
        buf = buf + "}"
        print(buf)
        mean_x0 /= self.num_pmts

        if self.set_globals:
            self.set_dataset(
                "global.Raman.Piezos.Ind_FinalX", mean_x0, persist=True
            )


class PiezoScanY(BasicEnvironment, artiq_env.Experiment):
    """Check.IndPiezoY."""

    data_folder = "piezo_scan"
    applet_name = "Rabi Piezo Scan"
    applet_group = "Raman Calib"
    fit_type = umd_fit.positive_gaussian

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument(
            "scan_range",
            artiq_scan.Scannable(
                default=artiq_scan.RangeScan(
                    start=0,
                    stop=3.0,
                    npoints=20,
                    randomize=False,
                    seed=int(time.time()),
                ),
                unit="",
                global_min=0,
                global_max=5.0,
            ),
        )
        self.setattr_argument("rabi_global_amp", artiq_env.NumberValue(default=500))
        self.setattr_argument("rabi_ind_amp", artiq_env.NumberValue(default=1000))

        self.setattr_argument("set_globals", artiq_env.BooleanValue(default=False))
        self.setattr_argument(
            "rabi_time",
            artiq_env.NumberValue(default=5.5 * us, unit="us", step=1e-7, ndecimals=7),
        )
        self.scan_values = [np.int32(0)]
        self.raman_time_mu = np.int32(0)
        super().build()

    def prepare(self):
        self.set_variables(
            self.data_folder, self.applet_name, self.applet_group, self.fit_type
        )
        self.scan_values_mu = [
            dac8568.vout_to_mu(
                val,
                self.raman.ind_final_piezos.SandiaSerialDAC.V_OUT_MAX
            )
            for val in self.scan_range
        ]
        self.scan_values_mu.reverse()
        self.scan_values = [val for val in self.scan_range]
        self.scan_values.reverse()
        self.num_steps = len(self.scan_range)

        super().prepare()
        self.raman_time_mu = self.core.seconds_to_mu(self.rabi_time)
        self.rabi_global_amp_mu = np.int32(self.rabi_global_amp)
        self.rabi_ind_amp_mu = np.int32(self.rabi_ind_amp)

    @host_only
    def run(self):
        """Start the experiment on the host."""
        # RUN EXPERIMENT
        try:
            self.experiment_initialize()
            _LOGGER.info("Done Initializing Experiment. Starting Main Loop:")
            self.experiment_loop()

        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")
        finally:
            _LOGGER.info("Done with Experiment. Setting machine to idle state")

    @kernel
    def prepare_step(self, istep: TInt32):
        self.raman.ind_final_piezos.value2_mu = self.scan_values_mu[istep]
        self.raman.ind_final_piezos.update_value()

    @kernel
    def main_experiment(self, istep, ishot):
        self.raman.global_dds.update_amp(self.rabi_global_amp_mu)#self.raman.global_dds.amp
        self.raman.switchnet_dds.update_amp(self.rabi_ind_amp_mu)#self.raman.switchnet_dds.amp
        self.raman.set_global_detuning_mu(np.int64(0))
        # wait for the DDS amplitudes to settle
        delay(200 * us)

        self.raman.pulse_mu(self.raman_time_mu)

    def analyze(self):
        """Analyze and Fit data.

        Threshold values and analyze."""
        super().analyze()
        meanTpi = 0.0
        # num_active_pmts = len(self.p_all["x0"])
        mean_x0 = 0.0
        buf = "{"
        for ifit in range(self.num_pmts):
            ret = self.p_all["x0"][ifit]
            buf = buf + "%f," % ret
            mean_x0 += self.p_all["x0"][ifit]
            print(
                "Fit {}: x0 = ({} +- {})".format(
                    ifit,
                    self.p_all["x0"][ifit],
                    self.p_error_all["x0"][ifit],
                )
            )
        buf = buf + "}"
        print(buf)
        mean_x0 /= self.num_pmts

        if self.set_globals:
            self.set_dataset("global.Raman.Piezos.Ind_FinalY", mean_x0, persist=True)

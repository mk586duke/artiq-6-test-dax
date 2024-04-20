"""Basic Experiment Template."""
import logging
import typing
from collections import defaultdict

import h5py
import numpy as np
import artiq.language.environment as artiq_env
import sipyco.pyon as pyon
from artiq.language.core import delay
from artiq.language.core import host_only
from artiq.language.core import kernel
from artiq.language.core import now_mu
from artiq.language.core import parallel
from artiq.language.core import rpc
from artiq.language.core import sequential
from artiq.language.types import TBool
from artiq.language.types import TInt32
from artiq.language.types import TInt64
from artiq.language.types import TList
from artiq.language.types import TNone
from artiq.language.units import ms
from artiq.language.units import us
from oitg.fitting.FitBase import FitError

import euriqabackend.devices.other.minicircuits_rf_switch as rf_switch
from euriqabackend import _EURIQA_LIB_DIR
from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqafrontend import EURIQA_NAS_DIR
from euriqafrontend.modules.awg import AWG
from euriqafrontend.modules.cw_lasers import DopplerCooling
from euriqafrontend.modules.cw_lasers import PumpDetect
from euriqafrontend.modules.cw_lasers import SSCooling_Settings
from euriqafrontend.modules.cw_lasers import DopplerCoolingCoolant
from euriqafrontend.modules.dac import SandiaDAC
from euriqafrontend.modules.mw import Microwave
from euriqafrontend.modules.pmt import PMTArray
from euriqafrontend.modules.population_analysis import PopulationAnalysis
from euriqafrontend.modules.raman import Raman
from euriqafrontend.modules.raman import SBC_DDS
from euriqafrontend.modules.cooling import Cooling
from euriqafrontend.modules.rfsoc_sbc import RFSoCSidebandCooling
from euriqafrontend.scheduler import ExperimentPriorities as priorities
from euriqafrontend.settings import auto_calibration as _AUTO_CALIBRATION_SETTINGS


_LOGGER = logging.getLogger(__name__)
module_path = (
    EURIQA_NAS_DIR / "CompactTrappedIonModule" / "ARTIQ" / "auto_calibration_modules_swap"
)


class BasicEnvironment(Cooling):
    """Basic Environment.

    Will loop over some parameter and do
    Calibration = (
        Doppler cool -> pump -> Stark shift Ramsey -> detect
    ) ->
    Doppler cool -> SS cooling -> SBC -> pump ->
    YOUR EXPERIMENT -> detect
    The calibration, SS cooling, SBC and pumping can be toggled on and off.

    Includes basic data logging, plotting, and fitting
    Saves the calibration, Doppler cooling, SS, and detect counts.
    """

    kernel_invariants = {
        "num_shots",
        "detect_thresh",
        "pmt_array",
        "num_pmts",
        "calib_stark_detuning_mu",
        "calib_tpihalf_mu",
        "calib_wait_time_mu",
        "sequence_durations_mu",
    }

    artiq_dir = _EURIQA_LIB_DIR.as_posix()

    applet_stream_cmd = (
        "$python -m euriqafrontend.applets.plot_multi" + " "
    )  # White space is required
    applet_hist_cmd = (
        "$python -m euriqafrontend.applets.plot_hist" + " "
    )  # White space is required

    _DATA_FOLDER = "DataFolder"
    _APPLET_NAME = "Applet"
    _APPLET_GROUP_NAME = "AppletGroup"
    _FIT_TYPE = None  # This should typically be a oitg.fitting class
    _UNITS = 1
    _XLABEL = ""
    _YLABEL = ""
    _KEEP_GLOBAL_ON_DDS = 1e3
    _experiment_dataset_name_format = "data.{exp_name}.{dataset_name}"

    dac_starting_node = 'Start'

    def set_variables(
        self,
        data_folder: str = _DATA_FOLDER,
        applet_name: str = _APPLET_NAME,
        applet_group: str = _APPLET_GROUP_NAME,
        fit_type=_FIT_TYPE,
        units=_UNITS,
        xlabel: str = _XLABEL,
        ylabel: str = _YLABEL,
    ):
        """Set data output destinations.

        Should be called from prepare() of the child class.
        """
        self._DATA_FOLDER = data_folder
        self._APPLET_NAME = applet_name
        self._APPLET_GROUP_NAME = applet_group
        self._FIT_TYPE = fit_type
        self._UNITS = units
        self._XLABEL = xlabel
        self._YLABEL = ylabel

    def build(self):
        # TODO: IF we add another inheritance we should loop through __mro__() and call all parents.build()
        super().build()
        self.sandia_box = SandiaDAC(self)
        # self.AWG = AWG(self)
        self.mw = Microwave(self)
        self.rfsoc_sbc = RFSoCSidebandCooling(self.pump_detect,self.SDDDS,self)

        """Initialize experiment & variables."""
        # basic ARTIQ devices
        self.setattr_device("core")
        self.setattr_device("ccb")
        self.setattr_device("oeb")
        self.setattr_device("scheduler")

        self.setattr_argument("comment", artiq_env.StringValue(default=""))
        self.setattr_argument(
            "num_shots", artiq_env.NumberValue(default=100, step=1, ndecimals=0)
        )
        self.setattr_argument(
            "detect_thresh",
            artiq_env.NumberValue(default=1, unit="counts", ndecimals=0, scale=1),
        )
        self.setattr_argument(
            "equil_loops",
            artiq_env.NumberValue(default=0, unit="loops", min=0, ndecimals=0, scale=1),
        )
        self.setattr_argument(
            "presence_thresh",
            artiq_env.NumberValue(default=10, unit="counts", ndecimals=0, scale=1)
        )
        self.setattr_argument(
            "presence_thresh_coolant",
            artiq_env.NumberValue(default=10, unit="counts", ndecimals=0, scale=1)
        )
        self.setattr_argument("lost_ion_monitor", artiq_env.BooleanValue(default=True))
        self.setattr_argument(
            "auto_reload_when_lost_ion", artiq_env.BooleanValue(default=False)
        )
        self.setattr_argument(
            "keep_global_on",
            artiq_env.BooleanValue(default=True),
            tooltip="Maximize the Global Beam duty cycle by keeping"
            " it on during doppler cooling, etc.",
        )
        # disable pointing lock
        # self.setattr_argument("do_calib", artiq_env.BooleanValue(default=True))
        # self.setattr_argument("lock_x_piezo", artiq_env.BooleanValue(default=False))
        # self.setattr_argument("lock_calib_t", artiq_env.BooleanValue(default=False))
        self.do_calib = False
        self.lock_x_piezo = False
        self.lock_calib_t = False

        # disable AWG
        self.use_AWG = False

        # disable DDS-based SBC
        self.do_SBC = False
        self.setattr_argument(
            "do_SS_cool",
            artiq_env.BooleanValue(default=True),
            tooltip="Turning this OFF will also turn OFF lost_ion_monitor",
        )
        # self.setattr_argument("do_SBC", artiq_env.BooleanValue(default=True))
        self.setattr_argument("do_pump", artiq_env.BooleanValue(default=True))
        self.setattr_argument("use_line_trigger", artiq_env.BooleanValue(default=False))
        # self.setattr_argument("use_AWG", artiq_env.BooleanValue(default=False))
        self.setattr_argument("use_RFSOC", artiq_env.BooleanValue(default=True))

        self.line_trigger = self.get_device("line_trigger")
        self.setattr_device("eom_935_3ghz")
        self.setattr_device("freq_shift_935")
        self.setattr_device("magfield_x")
        self.setattr_device("magfield_y")
        self.setattr_device("magfield_z")
        self.setattr_device("rf_lock_switch")

        # self.population_analyzer=PopulationAnalysis(self)
        self.sequence_durations_mu = []

    def prepare(self):
        """Pre-calculate any values before addressing hardware."""
        # Run prepare method of all the imported modules
        # self.sandia_box.dac_pc.line_gain = 0
        #self.call_child_method("prepare")
        self.pmt_array.prepare()
        self.sandia_box.prepare()
        self.doppler_cooling.prepare()
        self.doppler_cooling_coolant.prepare()
        self.pump_detect.prepare()
        self.ss_cooling.prepare()
        self.SDDDS.prepare()
        #self.sbc.prepare()
        self.raman.prepare()
        #self.AWG.prepare()
        self.mw.prepare()
        # has to be run after the pump_detect is prepared since it polls the pump time
        # to generate the RFSOC sequences

        if self.use_RFSOC:
            self.rfsoc_sbc.equil_loops = int(self.equil_loops)
            self.rfsoc_sbc.prepare()

            # record the duration of each actual (non-SBC) pulse sequence
            # as Int64 b/c that's what ``delay_mu()`` expects.
            self.sequence_durations_mu = np.array(
                self.rfsoc_sbc.compensated_sequence_durations, dtype=np.int64
            )

        # record the number of pmts that are used as a kernel invariant
        self.num_pmts = self.pmt_array.num_active
        self.active_pmts = self.pmt_array.active_pmts
        self.n_ions = self.get_dataset("global.AWG.N_ions", archive=False)

        # initialize monitor integrator counters
        self.doppler_monitor = np.full(self.num_pmts, 0)
        self.ss_monitor = np.full(self.num_pmts, 0)
        self.ss_coolant_monitor = np.full(self.num_pmts, 0)
        self.ss_all_monitor = np.full(self.num_pmts, 0)
        self.ss_edge_monitor = np.full(self.num_pmts, 0)
        self.ss_edge_coolant_monitor = np.full(self.num_pmts, 0)
        self.ss_edge_all_monitor = np.full(self.num_pmts, 0)
        self.lost_ion_monitor = self.lost_ion_monitor and self.do_SS_cool
        self.resubmit_on_ion_loss = False

        # initialize the machine-unit constants that are used in the calibration shot
        self.calib_stark_detuning = self.get_dataset("monitor.Stark_Detuning")
        self.calib_stark_detuning_mu = freq_to_mu(self.calib_stark_detuning)
        self.calib_tpihalf_mu = self.core.seconds_to_mu(
            self.get_dataset("monitor.Carrier_TpiHalf")
        )
        self.calib_wait_time_mu = self.core.seconds_to_mu(
            self.get_dataset("monitor.Stark_Twait")
        )
        self.calib_tlock_step_mu = np.int64(
            np.round(1.0 / self.calib_stark_detuning / self.core.ref_period)
        )

        # initialize the X-piezo lock
        self.calib_xlock_int = 0
        self.calib_tlock_int = 0
        self.ind_final_x_piezo_0_mu = self.raman.ind_final_piezos.value1_mu

        # Use this to store last monitor for looking for ion loss events
        self.last_check_ions = np.full(self.num_pmts, 0)

        # Check to make sure coherent control is specified properly
        assert not (self.use_RFSOC and self.use_AWG), "Cannot use both AWG and RFSoC"

        # Fail loudly if you try to use the pointing lock or SBC with the RFSoC for now
        assert not (
            self.use_RFSOC and self.do_calib
        ), "RFSoC operation does not currently support the pointing lock"

        if self.equil_loops == 0:
            self.equilibrated = True
        else:
            self.equilibrated = False
        # TODO: Check this
        # self.sandia_box.dac_pc.line_gain = 0
        # self.sandia_box.update_compensations()
        self._RID = self.scheduler.rid
        self._priority = self.scheduler.priority

    def delete_argument(self, argument_name: str):
        """Delete argument from experiment.

        Helpful when subclassing an experiment, so that not all arguments in the super
        experiment need to be used in the sub-experiment.
        This method must be called in ``build()``, AFTER calling ``super().build()``.
        """
        if not self._HasEnvironment__in_build:
            raise RuntimeError("Can only delete arguments while in build()")
        if not isinstance(self._HasEnvironment__argument_mgr, artiq_env.TraceArgumentManager):
            _LOGGER.debug("Will not delete arguments except when examining experiment")
            return

        del self._HasEnvironment__argument_mgr.requested_args[argument_name]

    def delete_arguments(self, *args: typing.Iterable[str]):
        """Convenience wrapper around :meth:`delete_argument`."""
        for argument in args:
            self.delete_argument(argument)

    @host_only
    def experiment_initialize(self):
        """Start the experiment on the host."""
        # Experimental Data Handling

        #  for child in self.children:
        #      if hasattr(child, "module_init"):
        #         child.module_init()
        self.setattr_device("yb_oven")
        self.set_experiment_data(
            "raw_counts",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "raw_ss_counts",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "raw_ss_coolant_counts",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "raw_ss_all_counts",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "raw_ss_edge_counts",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "raw_ss_edge_coolant_counts",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "raw_ss_edge_all_counts",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data("x_values", self.scan_values, broadcast=True)
        self.set_experiment_data(
            "avg_thresh",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "error_bars_bottom",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "error_bars_top",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "fit_x", np.full(self.num_pmts, np.nan), broadcast=True
        )
        self.set_experiment_data(
            "fit_y", np.full(self.num_pmts, np.nan), broadcast=True
        )
        self.set_experiment_data(
            "active_pmts", np.array(self.active_pmts), broadcast=True
        )
        self.set_experiment_data("rid", self._RID, broadcast=True, archive=False)
        self.set_experiment_data("detect_thresh", self.detect_thresh, broadcast=True)

        # Monitoring Data Handling
        self.set_dataset("monitor.interrupted", False, broadcast=True)
        self.set_dataset(
            "data.monitor.calib_raw_counts",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data.monitor.calib_avg_thresh",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data.monitor.calib_avg_xlock_feedback",
            np.full((1, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data.monitor.calib_avg_tlock_feedback",
            np.full((1, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data.monitor.calib_xlock_feedback",
            np.full((1, self.num_shots, self.num_steps), np.nan),
            broadcast=False,
        )
        self.set_dataset(
            "data.monitor.calib_tlock_feedback",
            np.full((1, self.num_steps, self.num_steps), np.nan),
            broadcast=False,
        )
        self.set_experiment_data(
            "chain_config",
            np.full((self.num_pmts, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )
        self.set_experiment_data(
            "reorder_log",
            np.full((1, self.num_shots, self.num_steps), np.nan),
            broadcast=True,
        )

        # Main Applet Handling
        self.ccb.issue(
            "create_applet",
            name=self._APPLET_NAME,
            command=self.applet_stream_cmd
            + " --x {0}".format(self._exp_dataset_path("x_values"))
            + " --y-names {0}".format(self._exp_dataset_path("avg_thresh"))
            + " --x-fit {0}".format(self._exp_dataset_path("fit_x"))
            + " --y-fits {0}".format(self._exp_dataset_path("fit_y"))
            + " --units {0}".format(str(self._UNITS))
            + " --y-label '{0}'".format(self._YLABEL)
            + " --x-label '{0}'".format(self._XLABEL)
            + " --active-pmts {0}".format(self._exp_dataset_path("active_pmts"))
            + " --rid {0}".format(self._exp_dataset_path("rid"))
            # + " --error-bars-bottom " + self._exp_dataset_path("error_bars_bottom")
            # + " --error-bars-top " + self._exp_dataset_path("error_bars_top")
            ,
            group=self._APPLET_GROUP_NAME,
        )

        # Monitor Applet Handling

        self.ccb.issue(
            "create_applet",
            name="pointing calib",
            command=self.applet_stream_cmd
            + " --x "
            + self._exp_dataset_path("x_values")
            + " --y-names "
            + "data.monitor.calib_avg_thresh"
            + " --units "
            + str(self._UNITS)
            + " --y-label "
            + "Counts"
            + " --x-label "
            + "'"
            + self._XLABEL
            + "'"
            + " --active-pmts "
            + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.ccb.issue(
            "create_applet",
            name="pointing xlock",
            command=self.applet_stream_cmd
            + " --x "
            + self._exp_dataset_path("x_values")
            + " --y-names "
            + "data.monitor.calib_avg_xlock_feedback"
            + " --units "
            + str(self._UNITS)
            + " --y-label "
            + "'Feedback Out (arb.)'"
            + " --x-label "
            + "'"
            + self._XLABEL
            + "'"
            + " --active-pmts "
            + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.ccb.issue(
            "create_applet",
            name="pointing tlock",
            command=self.applet_stream_cmd
            + " --x "
            + self._exp_dataset_path("x_values")
            + " --y-names "
            + "data.monitor.calib_avg_tlock_feedback"
            + " --units "
            + str(self._UNITS)
            + " --y-label "
            + "'Feedback Out (arb.)'"
            + " --x-label "
            + "'"
            + self._XLABEL
            + "'"
            + " --active-pmts "
            + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.set_dataset(
            "data.monitor.dopplercooling",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="Doppler Cooling Monitor",
            command=self.applet_stream_cmd
            + " --x "
            + self._exp_dataset_path("x_values")
            + " --y-names "
            + "data.monitor.dopplercooling"
            + " --units "
            + str(self._UNITS)
            + " --y-label "
            + "Counts"
            + " --x-label "
            + "'"
            + self._XLABEL
            + "'"
            + " --active-pmts "
            + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.set_dataset(
            "data.monitor.sscooling",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="SS Cooling Monitor",
            command=self.applet_stream_cmd
            + " --x "
            + self._exp_dataset_path("x_values")
            + " --y-names "
            + "data.monitor.sscooling"
            + " --units "
            + str(self._UNITS)
            + " --y-label "
            + "Counts"
            + " --x-label "
            + "'"
            + self._XLABEL
            + "'"
            + " --active-pmts "
            + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.set_dataset(
            "data.monitor.sscooling_coolant",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="SS Cooling Coolant Monitor",
            command=self.applet_stream_cmd
                    + " --x "
                    + self._exp_dataset_path("x_values")
                    + " --y-names "
                    + "data.monitor.sscooling_coolant"
                    + " --units "
                    + str(self._UNITS)
                    + " --y-label "
                    + "Counts"
                    + " --x-label "
                    + "'"
                    + self._XLABEL
                    + "'"
                    + " --active-pmts "
                    + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.set_dataset(
            "data.monitor.sscooling_all",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="SS Cooling All Monitor",
            command=self.applet_stream_cmd
                    + " --x "
                    + self._exp_dataset_path("x_values")
                    + " --y-names "
                    + "data.monitor.sscooling_all"
                    + " --units "
                    + str(self._UNITS)
                    + " --y-label "
                    + "Counts"
                    + " --x-label "
                    + "'"
                    + self._XLABEL
                    + "'"
                    + " --active-pmts "
                    + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.set_dataset(
            "data.monitor.sscooling_edge",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="SS Cooling Edge Monitor",
            command=self.applet_stream_cmd
                    + " --x "
                    + self._exp_dataset_path("x_values")
                    + " --y-names "
                    + "data.monitor.sscooling_edge"
                    + " --units "
                    + str(self._UNITS)
                    + " --y-label "
                    + "Counts"
                    + " --x-label "
                    + "'"
                    + self._XLABEL
                    + "'"
                    + " --active-pmts "
                    + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.set_dataset(
            "data.monitor.sscooling_edge_coolant",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="SS Cooling Edge Coolant Monitor",
            command=self.applet_stream_cmd
                    + " --x "
                    + self._exp_dataset_path("x_values")
                    + " --y-names "
                    + "data.monitor.sscooling_edge_coolant"
                    + " --units "
                    + str(self._UNITS)
                    + " --y-label "
                    + "Counts"
                    + " --x-label "
                    + "'"
                    + self._XLABEL
                    + "'"
                    + " --active-pmts "
                    + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.set_dataset(
            "data.monitor.sscooling_edge_all",
            np.full((self.num_pmts, self.num_steps), np.nan),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="SS Cooling Edge All Monitor",
            command=self.applet_stream_cmd
                    + " --x "
                    + self._exp_dataset_path("x_values")
                    + " --y-names "
                    + "data.monitor.sscooling_edge_all"
                    + " --units "
                    + str(self._UNITS)
                    + " --y-label "
                    + "Counts"
                    + " --x-label "
                    + "'"
                    + self._XLABEL
                    + "'"
                    + " --active-pmts "
                    + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )

        self.ccb.issue(
            "create_applet",
            name="PMT Count Monitor",
            command=self.applet_hist_cmd
            + "--pmt_counts "
            + self._exp_dataset_path("raw_counts")
            + " --pmt_numbers "
            + self._exp_dataset_path("active_pmts"),
            group="Monitor",
        )
        Ix = self.get_dataset("global.B_coils.Ix")
        Iy = self.get_dataset("global.B_coils.Iy")
        Iz = self.get_dataset("global.B_coils.Iz")
        Vx = self.get_dataset("global.B_coils.Vx")
        Vy = self.get_dataset("global.B_coils.Vy")
        Vz = self.get_dataset("global.B_coils.Vz")

        self.magfield_x.set_voltage(Vx)
        self.magfield_x.set_current(Ix)
        self.magfield_y.set_voltage(Vy)
        self.magfield_y.set_current(Iy)
        self.magfield_z.set_voltage(Vz)
        self.magfield_z.set_current(Iz)
        # self.magfield_x.get_current()
        # self.magfield_y.get_current()
        # self.magfield_z.get_current()
        # DAC box Initialization
        # **this should be moved to dac.py **
        #Todo: only load if table is different??
        # if self.sandia_box.update_swap_lines:
        #     # Use the full shuttling graph
        #     shuttling_file_to_load = self.sandia_box.dac_pc.shuttling_graph
        # else:
        #     # Only use the section of the shuttling graph located in this file
        #     shuttling_file_to_load = self.sandia_box.dac_pc.get_shuttling_graph_from_xml()
        # self.sandia_box.dac_pc.send_voltage_lines_to_fpga(shuttling_file_to_load)
        self.sandia_box.dac_pc.send_voltage_lines_to_fpga(force_full_update = self.sandia_box.force_full_update)
        self.sandia_box.dac_pc.send_shuttling_lookup_table()
        self.sandia_box.dac_pc.apply_line_async(self.dac_starting_node, line_gain=1)
        _LOGGER.info(f"Set DAC to {self.dac_starting_node}")

        # RFSoC is connected to port 2
        # TODO: remove once we don't need to switch back to AWG
        awg_rfsoc_switch = rf_switch.SPDTNetworkSwitch("192.168.80.15")
        awg_rfsoc_switch.set_if_needed(self.use_RFSOC)

        # set the switch network
        activeSlots = [i - 1 + 10 for i in self.pmt_array.active_pmts]
        # print(activeSlots)
        # self.AWG.rf_compiler.set_DDS(activeSlots)
        if self.use_AWG:
            self.AWG.experiment_initialize()

        if self.use_RFSOC:
            self.rfsoc_sbc.experiment_initialize()

    @kernel
    def kn_initialize(self):
        """Initialize the core and all devices."""
        self.core.reset()
        self.core.break_realtime()
        self.oeb.off()
        self.doppler_cooling.init()
        self.doppler_cooling_coolant.init()
        self.core.break_realtime()
        self.pump_detect.prepare_detect()
        self.pmt_array.clear_buffer()
        self.raman.init()
        # self.sbc.init()
        #self.AWG.init()
        self.freq_shift_935.off()
        self.rfsoc_sbc.init()
        if self.use_RFSOC:
            self.rfsoc_sbc.kn_init_rfsoc_sbc()
        delay(10 * us)

        self.custom_kn_init()

    @kernel
    def custom_kn_init(self):
        """Put your kernel init function here."""
        pass

    @kernel
    def experiment_loop(self):
        # run kernel-mode initialization
        self.kn_initialize()

        """Run the experiment on the core device."""
        # Create input buffers
        calib_counts = [0] * self.num_pmts
        doppler_counts = [0] * self.num_pmts
        ss_counts = [0] * self.num_pmts
        ss_coolant_counts = [0] * self.num_pmts
        ss_all_counts = [0] * self.num_pmts
        ss_edge_counts = [0] * self.num_pmts
        ss_edge_coolant_counts = [0] * self.num_pmts
        ss_edge_all_counts = [0] * self.num_pmts
        detect_counts = [0] * self.num_pmts
        config = [0] * self.num_pmts
        reorder_flag = 0
        lost_ion_flag = False
        # here we assume that the number of pmts is equal to the chain length, the size of sorting solution buffer
        # should be no smaller than chain length
        # maximal number of swaps required for reorder: Coolantnum * self.num_pmts - Coolantnum ^ 2
        # (23, 5): 90
        sorting_sol = [-1] * 90

        self.core.break_realtime()
        # Set switch network to sequence mode
        # if not self.use_RFSOC:
        #     self.AWG.setRunningSN(True)
        delay(10 * us)

        istep = 0  # Counter for steps
        iequil = 0  # Counter for equilibration rounds

        # Loop over main experimental scan parameters
        while istep < len(self.scan_values):
            # check for pause/abort from the host
            if self.scheduler.check_pause():
                break
            self.core.break_realtime()

            # do any initialization that needs to be done for each step
            # such as variable hardware-shuttling, variable piezo setting, etc.
            self.prepare_step(istep)

            # initialization of local var
            prev_config = config

            # Loop over to gather statistics
            for ishot in range(self.num_shots):

                self.core.break_realtime()
                if self.use_line_trigger:
                    trigger = self.wait_for_line_trigger()

                # Reset Switch Network
                # if not self.use_RFSOC:
                #     self.AWG.resetSN()
                #     delay(10 * us)
                #     # Advance Switch Network to its first line (calib)
                #     self.AWG.advanceSN()

                self.eom_935_3ghz.off()

                # if self.do_calib:
                #     stopcounter_calib_mu = self.measure_pointing()
                #     # Readout out counts and save the data
                #     calib_counts = self.pmt_array.count(
                #         up_to_time_mu=stopcounter_calib_mu, buffer=calib_counts
                #     )
                #     # Can break realtime because we're still not in experiment core.
                #     # might cause some nondeterministic timing relative to line trigger
                #     # Change originally made by @daiwei
                #     # Provides slack after reading PMTs
                #     # self.core.break_realtime()
                #     # try simple delay for now, to see if it works
                #     delay(350 * us)

                # Turn on Global AOM during cooling downtime
                if self.keep_global_on:
                    self.raman.global_dds.update_amp(np.int32(self._KEEP_GLOBAL_ON_DDS))
                    self.raman.global_dds.on()
                else:
                    self.raman.global_dds.off()

                delay(25 * us)
                cool_attempts = 0
                max_cool_attempts = 5
                passed = False
                doppler_duration_mu = self.doppler_cooling.doppler_cooling_duration_mu

                while (not passed) and (cool_attempts < max_cool_attempts):
                    self.core.break_realtime()
                    # Doppler Cooling
                    self.doppler_cooling.update_amp(self.doppler_cooling.cool_amp)
                    stopcounter_doppler_mu = self.doppler_cool(monitor=False)
                    doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)
                    delay(100 * us)  # To give some slack back after reading PMTs

                    # Second Stage Cooling
                    if self.do_SS_cool:
                        # Middle ion counts
                        stopcounter_ss_mu = self.second_stage_cool(monitor=True)
                        delay(100*us)
                        ss_counts = self.pmt_array.count(
                            up_to_time_mu=stopcounter_ss_mu, buffer=ss_counts
                        )
                        delay(100*us)
                        stopcounter_ss_coolant_mu = self.second_stage_cool_coolant(monitor=False)
                        delay(100*us)
                        ss_coolant_counts = self.pmt_array.count(
                            up_to_time_mu=stopcounter_ss_coolant_mu, buffer=ss_coolant_counts
                        )
                        delay(100*us)
                        # stopcounter_ss_all_mu = self.second_stage_cool_all(monitor=True)
                        # delay(100*us)
                        # ss_all_counts = self.pmt_array.count(
                        #     up_to_time_mu=stopcounter_ss_all_mu, buffer=ss_all_counts
                        # )
                        # delay(100 * us)  # To give some slack back after reading PMTs
                    else:
                        stopcounter_ss_mu = now_mu()

                        stopcounter_ss_coolant_mu = now_mu()

                        stopcounter_ss_all_mu = now_mu()

                    if self.do_SS_cool and self.lost_ion_monitor:
                        total_ss_counts = 0
                        for i in range(self.num_pmts):
                            total_ss_counts += ss_counts[i]

                        # TODO: Need to bookkeeping 171 and 172 num respectively
                        if total_ss_counts < (1 * self.n_ions):
                            self.doppler_cooling.doppler_cooling_duration_mu = 40*doppler_duration_mu
                            cool_attempts+=1
                            print("Recooling...")
                        else:
                            passed = True
                    else:
                        passed = True

                self.doppler_cooling.doppler_cooling_duration_mu = doppler_duration_mu

                if not passed:
                    _LOGGER.error("Failed to recool on step %f, shot %f", istep, ishot)
                    self.set_dataset("monitor.Lost_Ions", True, broadcast=True)
                    self.cancel_and_reload()
                    return

                # Todo: Add my shuttling function here

                if self.do_pump or self.rfsoc_sbc.do_sbc:
                    # prepare pumping
                    self.pump_detect.prepare_pump()

                # Sideband Cooling
                if self.use_RFSOC and self.rfsoc_sbc.do_sbc:
                    self.SDDDS.write_to_dds()
                    self.raman.set_global_aom_source(dds=False)
                    self.rfsoc_sbc.kn_do_rfsoc_sbc()
                    self.raman.set_global_aom_source(dds=True)

                if self.do_pump and self.pump_detect.do_slow_pump:
                    self.pump_detect.update_amp(amp_int=self.pump_detect.slow_pump_amp)
                    delay(150 * us)  # wait for the amplitude to actually reach the set value

                self.raman.global_dds.off()
                self.raman.set_global_aom_source(dds=False)

                delay(5*us)
                # Slow pump to the 0 state
                if self.do_pump:
                    self.pump_detect.pump()

                # prepare the AOMs for detection
                self.pump_detect.prepare_detect()

                # Perform the main experiment
                self.main_experiment(istep, ishot)

                # Detect Ions
                stopcounter_detect_mu = self.detect()

                self.eom_935_3ghz.off()

                if self.keep_global_on:
                    self.raman.global_dds.update_amp(np.int32(self._KEEP_GLOBAL_ON_DDS))
                    self.raman.global_dds.on()
                else:
                    self.raman.global_dds.off()

                # if self.do_calib:
                #     self.update_pointing_lock(calib_counts=calib_counts)

                # Readout out counts and save the data
                # calib_counts = self.pmt_array.count(up_to_time_mu=stopcounter_calib_mu, buffer=calib_counts)
                # doppler_counts = self.pmt_array.count(up_to_time_mu=stopcounter_doppler_mu, buffer=doppler_counts)

                detect_counts = self.pmt_array.count(
                    up_to_time_mu=stopcounter_detect_mu, buffer=detect_counts
                )

                self.save_counts(
                    ishot=ishot,
                    istep=istep,
                    calib_counts=calib_counts,
                    doppler_counts=doppler_counts,
                    ss_counts=ss_counts,
                    ss_coolant_counts=ss_coolant_counts,
                    ss_all_counts=ss_coolant_counts,
                    detect_counts=detect_counts,
                    tlock_feedback=self.calib_tlock_int,
                    xlock_feedback=self.calib_xlock_int,
                )
                self.core.break_realtime()
                self.core.wait_until_mu(now_mu())

                self.custom_proc_counts(ishot=ishot, istep=istep)

                # detect current chain config, give reorder flag based on config in last shot and current config
                # if detected 171 and 172 at the same pmt, it's determined as an invalid count, and we terminate
                # the experiment
                # due to connectionreset error, only save ss counts and ss coolant counts for now, use the latter ones
                # to tell the chain configuration
                if reorder_flag == 0:
                    prev_config = config

                # use the ss (qubit) counts and ss (coolant) counts to tell the config
                np_pmt_count_array_qubit = np.array(ss_counts)
                np_pmt_count_array_coolant = np.array(ss_coolant_counts)
                np_pmt_count_array_qubit_edge = np.array(ss_edge_counts)
                np_pmt_count_array_coolant_edge = np.array(ss_edge_coolant_counts)
                np_presence_qubit = [1 if np_pmt_count_array_qubit[k] > self.presence_thresh else 0 for k in
                                     range(self.num_pmts)]
                np_presence_coolant = [2 if np_pmt_count_array_coolant[k] > self.presence_thresh_coolant else 0 for k in
                                       range(self.num_pmts)]
                config = [np_presence_qubit[k] + np_presence_coolant[k] for k in range(self.num_pmts)]
                for k in range(self.num_pmts):
                    assert config[k] < 3, "Invalid count occurs, experiment terminates"

                self.save_config(config, ishot, istep)
                reorder_flag = self.reorder_detect(prev_config, config, ishot, istep)
                self.save_reorder(reorder_flag, ishot, istep)

            #     if reorder_flag:
            #         break
            #
            # # sorting: once reordered, sort the chain from config to prev_config
            # if reorder_flag:
            #     sorting_sol = self.sorting_sol_generator(prev_config, config)
            #     for i in range(len(sorting_sol)):
            #         self.swap(i, i + 1)


            # threshold raw counts, save data, and push it to the plot applet
            self.threshold_data(istep)

            # run any other code that is executed on the host
            # at the end of each x-point (after num_shots exps are taken)
            # this function should be RPC on the host
            self.custom_proc_data(istep)

            # Increment equilibration loops until specified then increment the step counter

            self.module_update(istep)

            if self.equilibrated:
                istep += 1
            else:
                iequil += 1
                if iequil >= self.equil_loops:
                    self.equilibrated = True

        # Clean up after the experiment
        # if self.lock_x_piezo:
        #     final_piezo = self.raman.ind_final_piezos.get_value1()
        #     self.set_dataset(
        #         "global.Raman.Piezos.Ind_FinalX", final_piezo, persist=True
        #     )
        # if self.lock_calib_t:
        #     # TODO: clean up this machine unit to second conversion by adding appropriate vars and helper functions
        #     final_t_mu = (
        #         self.calib_wait_time_mu
        #         - self.calib_tlock_step_mu * self.calib_tlock_int // 2
        #     )
        #     final_t = final_t_mu * self.core.ref_period
        #     self.set_dataset("monitor.Stark_Twait", final_t, persist=True)
        self.kn_idle()

    @kernel
    def prepare_step(self, istep: TInt32):
        """Prepare the experiment to be run, to be implemented by user."""
        pass

    @kernel
    def main_experiment(self, istep, ishot):
        """The main experimental code block to perform an experiment.

        To be implemented by user.
        """
        raise NotImplementedError

    @kernel
    def wait_for_line_trigger(self) -> TBool:
        trigger = False
        try:
            while self.line_trigger.watch_stay_off():
                delay(200 * us)
        finally:
            delay(200 * us)
            self.line_trigger.watch_done()
            trigger = True
        delay(100 * us)
        self.core.break_realtime()
        delay(100 * us)
        return trigger

    @kernel
    def calib_raman_pulse(self, time_mu):
        """Apply Calibration Raman Pulse"""
        self.raman.global_dds.update_freq_mu(self.raman._CARRIER_FREQ_MU)
        delay(10 * us)
        self.raman.pulse_mu(time_mu)
        delay(10 * us)

    @kernel
    def measure_pointing(self):
        """Measure pointing noise."""
        # prepare the pump/detect path for pumping
        self.raman.switchnet_dds.update_amp(
            np.int32(1000)
        )  # self.raman.switchnet_dds.amp)
        self.raman.global_dds.update_amp(np.int32(self._KEEP_GLOBAL_ON_DDS))
        delay(120 * us)
        with parallel:
            with sequential:
                self.pump_detect.set_param_mu(
                    freq_mu=self.pump_detect.pump_frequency_mu,
                    amp_int=self.pump_detect.pump_amp,
                )
                self.pump_detect.write_to_dds()
            # while, in parallel, preparing the pump EOM
            self.pump_detect.pump_eom(pump=True)
            # while, in parallel, compressing the chain
            # self.sandia_box.hw_squeeze_quantum(True)
            # while, in parallel, doppler-cooling
            self.doppler_cool(monitor=False)
        # make sure that the shuttle has completed

        # If global is kept on turn it off before the calibration pulses and pump stage
        if self.keep_global_on:
            self.raman.global_dds.off()

        # pump
        self.pump_detect.pulse_mu(2 * self.pump_detect.pump_duration_mu)
        # prepare the pump/detect path for detection
        self.pump_detect.prepare_detect()

        # apply the Raman pulse for intensity-detection
        if self.lock_calib_t:
            pulse_t_mu = (
                self.calib_wait_time_mu
                - self.calib_tlock_step_mu * self.calib_tlock_int // 2
            )
            self.calib_stark_shift(pulse_t_mu)
        else:
            # self.calib_raman_pulse(self.calib_wait_time_mu)
            self.calib_stark_shift(self.calib_wait_time_mu)

        # Need more detection on the edge ions when we dont squeeze the potential
        with parallel:
            self.pump_detect.on()
            stopcounter_calib_mu = self.pmt_array.gate_rising_mu(
                self.pump_detect.detect_duration_mu
                + self.core.seconds_to_mu(25e-6)  # noqa: ATQ903
            )
        self.pump_detect.off()

        # If global is kept on turn it back on after the calibration pulses/detection
        if self.keep_global_on:
            self.raman.global_dds.on()
        # unsqueeze the axial potential
        # self.sandia_box.hw_squeeze_quantum(False)
        # make sure that the shuttle has completed -- not needed since next step is Doppler cool
        # delay(0.5 * ms)
        delay(0.2 * ms)

        return stopcounter_calib_mu

    @kernel
    def update_pointing_lock(self, calib_counts):
        """Update integrator and piezo voltages for pointing lock stabilization."""
        # if calib_counts[0] > 1:
        #     self.calib_xlock_int += 1
        #     self.calib_tlock_int += 1
        # else:
        #     self.calib_xlock_int -= 1
        #     self.calib_tlock_int -= 1
        #
        # if calib_counts[-1] > 1:
        #     self.calib_xlock_int -= 1
        #     self.calib_tlock_int += 1
        # else:
        #     self.calib_xlock_int += 1
        #     self.calib_tlock_int -= 1

        if calib_counts[0] > 1:
            self.calib_xlock_int -= 1
            self.calib_tlock_int += 1
        else:
            self.calib_xlock_int += 1
            self.calib_tlock_int -= 1

        if calib_counts[-1] > 1:
            self.calib_xlock_int += 1
            self.calib_tlock_int += 1
        else:
            self.calib_xlock_int -= 1
            self.calib_tlock_int -= 1

        if self.lock_x_piezo:
            self.raman.ind_final_piezos.value1_mu = self.ind_final_x_piezo_0_mu - (
                self.calib_xlock_int << 5
            )
            delay(20 * us)
            self.raman.ind_final_piezos.update_value()
            delay(20 * us)

    @kernel
    def calib_stark_shift(self, twait_mu: TInt64):
        shifted_freq_mu = self.raman._CARRIER_FREQ_MU + self.calib_stark_detuning_mu

        # apply carrier pi/2
        self.raman.set_global_detuning_mu(np.int64(0))
        # self.raman.global_dds.update_freq_mu(self.raman._CARRIER_FREQ_MU)
        delay(10 * us)
        self.raman.set_ind_detuning_mu(np.int64(0))
        # self.raman.switchnet_dds.update_freq_mu(self.raman._IND_BASE_MU)
        delay(10 * us)
        self.raman.pulse_mu(self.calib_tpihalf_mu)

        # detune rel. to the carrier
        # self.raman.global_dds.update_freq_mu(shifted_freq_mu)
        self.raman.set_global_detuning_mu(self.calib_stark_detuning_mu)
        delay(10 * us)
        # delay_mu(twait_mu)

        # apply carrier pi/2
        # self.raman.global_dds.update_freq_mu(self.raman._CARRIER_FREQ_MU)
        self.raman.set_global_detuning_mu(np.int64(0))

        delay(10 * us)

    @kernel
    def detect(self) -> TInt64:
        # delay(50 * us) # MC: 5/12/19: what is this for?
        with parallel:
            self.pump_detect.on()
            stopcounter_detect_mu = self.pmt_array.gate_rising(
                self.pump_detect.detect_duration
            )
        self.pump_detect.off()

        return stopcounter_detect_mu

    @kernel
    def num_counter(self, config, isotope):
        # 171:1, 172: 2
        num = 0
        for k in range(self.num_pmts):
            num += config[k] / isotope
        return num

    @kernel
    def reorder_detect(self, prev_config, config, ishot, istep) -> TInt32:
        reorder_flag = 0
        if ishot == 0 and istep == 0:
            # 1st shot of 1st step
            reorder_flag = 0
        else:
            for k in range(self.num_pmts):
                if prev_config[k] - config[k] != 0:
                    reorder_flag = 1
                    break
                reorder_flag = 0

        return reorder_flag

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, loop filter off
        self.core.break_realtime()
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

    @rpc(flags={"async"})
    def save_counts(
        self,
        ishot: TInt32,
        istep: TInt32,
        calib_counts: TList(TInt32),
        doppler_counts: TList(TInt32),
        ss_counts: TList(TInt32),
        ss_coolant_counts: TList(TInt32),
        ss_all_counts: TList(TInt32),
        detect_counts: TList(TInt32),
        tlock_feedback: TInt32,
        xlock_feedback: TInt32,
    ) -> TNone:

        if ishot == 0:
            self.doppler_monitor = np.full(len(doppler_counts), 0)
            self.ss_monitor = np.full(len(ss_counts), 0)
            self.ss_coolant_monitor = np.full(len(ss_coolant_counts), 0)
            self.ss_all_monitor = np.full(len(ss_all_counts), 0)

        self.doppler_monitor += np.array(doppler_counts)
        self.ss_monitor += np.array(ss_counts)
        self.ss_coolant_monitor += np.array(ss_coolant_counts)
        self.ss_all_monitor += np.array(ss_all_counts)

        if ishot == self.num_shots - 1:
            self.doppler_monitor = np.array(
                self.doppler_monitor / self.num_shots, ndmin=2
            ).T
            self.mutate_dataset(
                "data.monitor.dopplercooling",
                ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
                self.doppler_monitor,
            )
            self.ss_monitor = np.array(self.ss_monitor / self.num_shots, ndmin=2).T
            self.mutate_dataset(
                "data.monitor.sscooling",
                ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
                self.ss_monitor,
            )
            self.ss_coolant_monitor = np.array(self.ss_coolant_monitor / self.num_shots, ndmin=2).T
            self.mutate_dataset(
                "data.monitor.sscooling_coolant",
                ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
                self.ss_coolant_monitor,
            )
            self.ss_all_monitor = np.array(self.ss_all_monitor / self.num_shots, ndmin=2).T
            self.mutate_dataset(
                "data.monitor.sscooling_all",
                ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
                self.ss_all_monitor,
            )

            if self.lost_ion_monitor is True:
                check_ions = self.ss_monitor
                lost_ions = False
                if istep == 0:
                    self.last_check_ions = check_ions  # No previous history so just duplicate the first shot
                    if np.sum(check_ions) < (2 * self.n_ions):
                        lost_ions = True

                # obtain the pmt readng from each channel as vectors
                v1 = np.ndarray.flatten(self.last_check_ions)
                v2 = np.ndarray.flatten(check_ions)

                norm_v1 = np.inner(v1, v1)
                norm_v2 = np.inner(v2, v2)

                # compare the difference in the norm
                diff_norm = abs(norm_v1 - norm_v2) / (1 + norm_v2)
                if diff_norm > 0.5:
                    lost_ions = True

                # compare the difference in the orientation
                # (v1.v2)*(v2.v1)/(||v1||*||v2||)
                dot_product = np.dot(v1, v2) ** 2 / (1 + norm_v1 * norm_v2)
                if dot_product < 0.7:
                    lost_ions = True

                if lost_ions is True:
                    _LOGGER.error("LOST IONS AFTER STEP %f", istep)
                    self.set_dataset("monitor.Lost_Ions", True, broadcast=True)

                    # if np.mean(check_ions) > np.mean(self.last_check_ions)/4:
                    #     self.set_dataset("monitor.Lost_Ions.Yb_pp", True, broadcast=True)
                    #     _LOGGER.error("LOST IONS: Cause -> Yb++")
                    # else:
                    #     self.set_dataset("monitor.Lost_Ions.Collision", True, broadcast=True)
                    #     _LOGGER.error("LOST IONS: Cause -> Collision")
                    self.cancel_and_reload()

                self.last_check_ions = check_ions

        # Need to cast a 1-D list of length N into a 1-D slice of a 2D/3D numpy array of size = (N,1,...,1)
        np_calib_counts = np.array(doppler_counts, ndmin=3).T
        np_detect_counts = np.array(detect_counts, ndmin=3).T
        np_ss_counts = np.array(ss_counts, ndmin=3).T
        np_ss_coolant_counts = np.array(ss_coolant_counts, ndmin=3).T
        np_ss_all_counts = np.array(ss_all_counts, ndmin=3).T
        self.mutate_experiment_data(
            "raw_counts",
            (
                (0, self.num_pmts, 1),
                (ishot, None, self.num_shots),
                (istep, None, self.num_steps),
            ),
            np_detect_counts,
        )

        self.mutate_experiment_data(
            "raw_ss_counts",
            (
                (0, self.num_pmts, 1),
                (ishot, None, self.num_shots),
                (istep, None, self.num_steps),
            ),
            np_ss_counts,
        )

        self.mutate_experiment_data(
            "raw_ss_coolant_counts",
            (
                (0, self.num_pmts, 1),
                (ishot, None, self.num_shots),
                (istep, None, self.num_steps),
            ),
            np_ss_coolant_counts,
        )

        self.mutate_experiment_data(
            "raw_ss_all_counts",
            (
                (0, self.num_pmts, 1),
                (ishot, None, self.num_shots),
                (istep, None, self.num_steps),
            ),
            np_ss_all_counts,
        )

        self.mutate_dataset(
            "data.monitor.calib_raw_counts",
            (
                (0, self.num_pmts, 1),
                (ishot, None, self.num_shots),
                (istep, None, self.num_steps),
            ),
            np_calib_counts,
        )

        self.mutate_dataset(
            "data.monitor.calib_xlock_feedback",
            ((0, 1, 1), (ishot, None, self.num_shots), (istep, None, self.num_steps)),
            xlock_feedback,
        )

        self.mutate_dataset(
            "data.monitor.calib_tlock_feedback",
            ((0, 1, 1), (ishot, None, self.num_shots), (istep, None, self.num_steps)),
            tlock_feedback,
        )

    @rpc(flags={"async"})
    def save_reorder(
        self,
        reorder_flag: TInt32,
        ishot: TInt32,
        istep: TInt32,
    ) -> TNone:

        self.mutate_experiment_data(
            "reorder_log",
            ((0, 1, 1),  (istep, None, self.num_steps), (ishot, None, self.num_shots)),
            reorder_flag,
        )


    @rpc(flags={"async"})
    def save_config(
        self,
        config: TList(TInt32),
        ishot: TInt32,
        istep: TInt32,
    ) -> TNone:

        np_config = np.array(config, ndmin=3).T

        self.mutate_experiment_data(
            "chain_config",
            (
                (0, self.num_pmts, 1),
                (ishot, None, self.num_shots),
                (istep, None, self.num_steps),
            ),
            np_config,
        )


    @rpc(flags={"async"})
    def threshold_data(self, istep: TInt32):
        calib_counts = np.array(self.get_dataset("data.monitor.calib_raw_counts"))
        calib_thresh = np.mean(
            calib_counts[:, :, istep] > self.detect_thresh, axis=1
        ).tolist()

        calib_xlock_feedback = np.array(
            self.get_dataset("data.monitor.calib_xlock_feedback")
        )
        calib_xlock_feedback_mean = np.mean(
            calib_xlock_feedback[:, :, istep], axis=1
        ).tolist()

        calib_tlock_feedback = np.array(
            self.get_dataset("data.monitor.calib_tlock_feedback")
        )
        calib_tlock_feedback_mean = np.mean(
            calib_tlock_feedback[:, :, istep], axis=1
        ).tolist()

        counts = np.array(self.get_experiment_data("raw_counts"))
        thresholded_counts = counts[:, :, istep] > self.detect_thresh
        thresh = np.mean(thresholded_counts, axis=1).tolist()

        np_thresh = np.array(thresh, ndmin=2).T
        np_calib_thresh = np.array(calib_thresh, ndmin=2).T

        sigmas = np.transpose(
            np.array(
                np.sqrt(
                    np.sum(
                        np.square(np.subtract(thresholded_counts, np_thresh))
                        / (thresholded_counts.shape[1] - 1),
                        axis=1,
                    )
                )
                / np.sqrt(thresholded_counts.shape[1]),
                ndmin=2,
            )
        )

        self.mutate_experiment_data(
            "avg_thresh",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
            np_thresh,
        )
        self.mutate_dataset(
            "data.monitor.calib_avg_thresh",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
            np_calib_thresh,
        )

        self.mutate_experiment_data(
            "error_bars_bottom",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
            sigmas,
        )

        self.mutate_experiment_data(
            "error_bars_top",
            ((0, self.num_pmts, 1), (istep, None, self.num_steps)),
            sigmas,
        )

        self.mutate_dataset(
            "data.monitor.calib_avg_xlock_feedback",
            ((0, 1, 1), (istep, None, self.num_steps)),
            calib_xlock_feedback_mean,
        )
        self.mutate_dataset(
            "data.monitor.calib_avg_tlock_feedback",
            ((0, 1, 1), (istep, None, self.num_steps)),
            calib_tlock_feedback_mean,
        )

    @rpc(flags={"async"})
    def module_update(self, istep):
        for child in self.children:
            if hasattr(child, "module_update"):
                child.module_update(istep)

    @rpc(flags={"async"})
    def custom_proc_data(self, istep):
        pass

    @rpc(flags={"async"})
    def custom_proc_counts(self, ishot: TInt32, istep: TInt32):
        pass

    @rpc
    def cancel_and_reload(self):
        """If auto-reload not engaged, all the current expids will be deleted. Otherwise, delete the current expid and reload ions"""
        if not self.auto_reload_when_lost_ion:
            for rid in self.scheduler.get_status():
                self.scheduler.request_termination(rid)
        else:

            # A non-elegant way of submitting a center scan
            _LOGGER.info("Warming up the oven.")
            self.yb_oven.turn_on(current=2.05, voltage=1.65, timeout=60*60)
            self.yb_oven.turn_on(current=2.05, voltage=1.65, timeout=60*60)

            temp = h5py.File(str(module_path / "Calibrate_CheckCenter15.h5"), "r")
            reconfirm_center = pyon.decode(temp["expid"].value)
            for key in _AUTO_CALIBRATION_SETTINGS["CheckCenter15"]:
                reconfirm_center["arguments"][key] = _AUTO_CALIBRATION_SETTINGS["CheckCenter15"][key]

            rid = self.scheduler.submit(
                pipeline_name="main",
                expid=reconfirm_center,
                priority=int(
                    priorities.CALIBRATION_PRIORITY2
                ),  # converts enum to int value if necessary
                due_date=None,
                flush=False,
            )

            self.set_dataset("monitor.interrupted", True, broadcast=True)
            if self.resubmit_on_ion_loss:
                updated_expid = self.scheduler.expid.copy()
                # submits with same parameters (due date, pipeline, priority)
                self.scheduler.submit(expid=updated_expid)

            self.scheduler.request_termination(self._RID)

    def _exp_dataset_path(self, dataset_name: str) -> str:
        """Return the string-path where an experiment dataset is stored."""
        return self._experiment_dataset_name_format.format(
            exp_name=self._DATA_FOLDER, dataset_name=dataset_name
        )

    def set_experiment_data(
        self, dataset_name: str, value: typing.Any, **kwargs
    ) -> None:
        """Simplify saving datasets to this experiment."""
        self.set_dataset(self._exp_dataset_path(dataset_name), value, **kwargs)

    def get_experiment_data(self, dataset_name: str) -> typing.Any:
        return self.get_dataset(self._exp_dataset_path(dataset_name))

    def mutate_experiment_data(self, dataset_name: str, *args, **kwargs) -> None:
        self.mutate_dataset(self._exp_dataset_path(dataset_name), *args, **kwargs)

    @kernel
    def kn_idle(self):
        """Reset all used devices to default state."""
        self.core.break_realtime()
        self.pump_detect.idle()
        delay(10 * us)
        self.doppler_cooling.idle()
        self.doppler_cooling_coolant.idle()
        delay(10 * us)
        self.raman.idle()
        self.custom_kn_idle()
        # save ourselves from collisions by setting Doppler cooling power very low

    @kernel
    def custom_kn_idle(self):
        pass

    @kernel
    def sorting_sol_generator(self, DC, CC, sorting_sol) -> TList(TInt32):
        # 172: 2; 171: 1
        # numpy has nonetype problems unique to kernel
        DC_coolant_index = []
        CC_coolant_index = []
        # swap solution: k, means swap (k, k+1) pair, local var to be stored in sorting_sol
        swapsol = []
        for i in range(len(DC)):
            # don't use append, list in kernel doesn't have this attribute
            if DC[i] == 2:
                DC_coolant_index = DC_coolant_index + [i]
            if CC[i] == 2:
                CC_coolant_index = CC_coolant_index + [i]

        # Left movement starts from left; right movement starts from right
        # Bookkeeping the direction of each 172 ion
        # > 0: moving left ; <0: moving right.
        # movement in two direction is completely decoupled
        Dir = [CC_coolant_index[i] - DC_coolant_index[i] for i in range(len(CC_coolant_index))]

        for i in range(len(Dir)):
            if Dir[i] > 0:
                CC[CC_coolant_index[i]], CC[DC_coolant_index[i]] = CC[DC_coolant_index[i]], CC[CC_coolant_index[i]]
                # swapnum += abs(CC_coolant_index[i] - DC_coolant_index[i])
                for k in range(CC_coolant_index[i] - 1, DC_coolant_index[i] - 1, -1):
                    swapsol = swapsol + [k]

        for i in range(len(Dir) - 1, -1, -1):
            if Dir[i] < 0:
                CC[CC_coolant_index[i]], CC[DC_coolant_index[i]] = CC[DC_coolant_index[i]], CC[CC_coolant_index[i]]
                # swapnum += abs(CC_coolant_index[i] - DC_coolant_index[i])
                for k in range(CC_coolant_index[i], DC_coolant_index[i]):
                    swapsol = swapsol + [k]


        for i in range(len(swapsol)):
            sorting_sol[i] = swapsol[i]

        return sorting_sol

    @kernel
    def swap(self, leftind, rightind):
        pass

    def analyze(self, constants={}):
        """Analyze and Fit data.

        Threshold values and analyze.
        """
        if self._FIT_TYPE is not None:
            # self.set_experiment_data("fit_type", self._FIT_TYPE, broadcast=False)
            x = np.array(self.get_experiment_data("x_values"))
            y = self.get_experiment_data("avg_thresh")
            fit_plot_points = len(x) * 10
            # print("total num of points", fit_plot_points)

            self.p_all = defaultdict(list)
            self.p_error_all = defaultdict(list)
            # Can't change np.nan to 0, or else the fit is a constant line at 0
            y_fit_all = np.full((y.shape[0], fit_plot_points), np.nan)
            p = self._FIT_TYPE.parameter_names

            for iy in range(y.shape[0]):
                try:
                    p, p_error, x_fit, y_fit = self._FIT_TYPE.fit(
                        x,
                        y[iy, :],
                        evaluate_function=True,
                        evaluate_n=fit_plot_points,
                        constants=constants,
                    )

                    for ip in p:
                        self.p_all[ip].append(p[ip])
                        self.p_error_all[ip].append(p_error[ip])

                    y_fit_all[iy, :] = y_fit

                except FitError:
                    _LOGGER.info("Fit failed for y-data # %i", iy)
                    for ip in p:
                        self.p_all[ip].append(np.float(0))
                        self.p_error_all[ip].append(np.float(0))

                    y_fit_all[iy, :] = 0.0
                    continue

            for ip in p:
                self.set_experiment_data(
                    "fitparam_" + ip, self.p_all[ip], broadcast=True
                )
            self.set_experiment_data("fit_x", x_fit, broadcast=True)
            self.set_experiment_data("fit_y", y_fit_all, broadcast=True)

import copy
import logging
from time import sleep

import artiq
import numpy as np
import copy
from scipy.stats import(kstest, poisson)
from artiq.experiment import *
from artiq.language import BooleanValue
from artiq.language import NumberValue
from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.core import parallel
from artiq.language.units import A
from artiq.language.units import ms
from artiq.language.units import s
from artiq.language.units import us

# from euriqafrontend.modules.cw_lasers import DopplerCooling
from euriqafrontend.modules.cw_lasers import DopplerCooling, DopplerCoolingCoolant
from euriqafrontend.modules.dac import SandiaDAC
from euriqafrontend.modules.pmt import PMTArray
from euriqafrontend.modules.raman import Raman
from euriqafrontend.modules.artiq_dac import IonizationDiodeControl

_LOGGER = logging.getLogger(__name__)

# Oven current in amps. DO NOT INCREASE WITHOUT CONSULTING THE TEAM!
_OVEN_CURRENT = 2.12


class Autoload(artiq.language.environment.EnvExperiment):
    """Loading.Autoload"""

    kernel_invariants = {
        "load_time",
        "eject_time",
        "cool_time",
        "det_win",
        "inspect_win",
        "pmt_array",
        "num_pmts",
        "max_attempts",
        "clear_trap",
    }

    applet_stream_cmd = (
        "$python -m euriqafrontend.applets.plot_multi" + " "
    )  # White space is required

    def build(self):

        # Get Loading arguments
        self.load_time = self.get_argument(
            "load wait time",
            NumberValue(
                default=300 * ms, unit="ms", step=10 * ms, min=1 * ms, max=10 * s
            ),
            group="Loading Parameters",
        )
        self.eject_time = self.get_argument(
            "eject wait time",
            NumberValue(
                default=100 * ms, unit="ms", step=10 * ms, min=1 * ms, max=10 * s
            ),
            group="Loading Parameters",
        )
        self.cool_time = self.get_argument(
            "cool wait time",
            NumberValue(
                default=500 * ms, unit="ms", step=10 * ms, min=1 * ms, max=10 * s
            ),
            group="Loading Parameters",
        )
        self.det_win = self.get_argument(
            "detection window",
            NumberValue(default=10 * ms, unit="ms", step=1 * ms, min=1 * us, max=1 * s),
            group="Loading Parameters",
        )
        self.inspect_win = self.get_argument(
            "inspection detection window",
            NumberValue(default=10 * ms, unit="ms", step=1 * ms, min=1 * us, max=1 * s),
            group="Loading Parameters",
        )

        self.goal_ions = self.get_argument(
            "number of ions to load",
            NumberValue(default=15, unit="ions", step=1, min=1, max=32, scale=1),
            group="Loading Parameters",
        )
        self.max_attempts = self.get_argument(
            "max # load attempts",
            NumberValue(
                default=500,
                unit="attempts",
                step=1,
                min=1,
                max=500,
                ndecimals=0,
                scale=1,
            ),
            group="Loading Parameters",
        )
        self.turn_on_time = self.get_argument(
            "Oven Preheat Time",
            NumberValue(
                default=100, unit="s", step=1, min=1, max=240, ndecimals=0, scale=1
            ),
            group="Loading Parameters",
        )
        self.global_on = self.get_argument(
            "Use Global Beam During Cooliing",
            BooleanValue(default=False),
            group="Loading Parameters",
        )
        self.oven_off = self.get_argument(
            "Turn oven OFF after experiment",
            BooleanValue(default=True),
            group="Loading Parameters",
        )
        self.clear_trap = self.get_argument(
            "Dump all ions from trap before autoloading.",
            BooleanValue(default=True),
            group="Loading Parameters",
        )
        self.inspect_on = self.get_argument(
            "Inspect before merge",
            BooleanValue(default=True),
            group="Loading Parameters",
        )
        self.setattr_argument(
            "ion_present_counts",
            NumberValue(
                100,
                unit="counts",
                scale=1,
                ndecimals=0,
                step=1,
                min=0
            ),
            group="Loading Parameters",
            tooltip="Number of PMT counts across the entire chain to confirm an ion " \
                    "is present during pre-merge inspection."
        )

        # Get PMT Settings
        self.pmt_array = PMTArray(self)

        # Load core devices
        self.setattr_device("core")
        self.setattr_device("scheduler")
        self.setattr_device("oeb")
        self.setattr_device("ccb")
        self.raman = Raman(self)

        # Load other devices
        self.sandia_box = SandiaDAC(self)
        self.setattr_device("yb_oven")
        self.setattr_device("aoms_id")

        self.setattr_device("magfield_x")
        self.setattr_device("magfield_y")
        self.setattr_device("magfield_z")

        # Load Laser Shutters
        self.doppler_cooling = DopplerCooling(self)
        # self.ionization_beam = self.get_device("shutter_370ion")
        self.ionization_beam = self.get_device("shutter_394")
        self.setattr_device("shutter_399")

        # Overwrite ability to pull in global values. Loading requires certain compensation values.

        self.setattr_argument("use_global_voltages", EnumerationValue(["False"], default="False"), group="Sandia DAC")

        # TODO: set qxz_offset & qxz to correct value??
        self.QXZ_offset = self.get_argument(
            "QXZ offset (MHz^2)",
            EnumerationValue(["0.00"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QZZ_offset = self.get_argument(
            "QZZ offset (MHz^2)",
            EnumerationValue(["0.05"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QZY_offset = self.get_argument(
            "QZY offset (MHz^2)",
            EnumerationValue(["-0.5"]),
            group="Sandia DAC",
            tooltip="You cannot change this value if you want to load",
        )
        self.X4_offset = self.get_argument(
            "X4 offset (MHz^2 / (2.74 um)^2)",
            EnumerationValue(["-0.0006"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QXZ = self.get_argument(
            "QXZ (MHz^2)",
            EnumerationValue(["0.0"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QZZ = self.get_argument(
            "QZZ (MHz^2)",
            EnumerationValue(["0.0"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QZY = self.get_argument(
            "QZY (MHz^2)",
            EnumerationValue(["0.5"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.X1 = self.get_argument(
            "X1 (2.74 um * MHz^2)",
            EnumerationValue(["0.0"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )

        self.X2 = self.get_argument(
            "X2 (MHz^2)",
            NumberValue(default=0.001, unit="", min=-0.5, max=+0.5, ndecimals=4),
            group="Sandia DAC",
        )

        self.X3 = self.get_argument(
            "X3 (MHz^2 / (2.74 um))",
            EnumerationValue(["0.0"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."

        )
        self.X4 = self.get_argument(
            "X4 (MHz^2 / (2.74 um)^2)",
            EnumerationValue(["0.0006"]),
            # EnumerationValue(["0.000355"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )

        _LOGGER.debug("Done Building Experiment")

    def prepare(self):

        # Dont let the DAC pull global values
        self.sandia_box.use_global_voltages = False

        # Autoloader uses a different shuttle file from basic environment (which
        # includes swap)
        self.sandia_box.shuttle_flag = False
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        self.num_pmts = self.pmt_array.num_active
        self.active_pmts = self.pmt_array.active_pmts

        # Overwrite DAC values

        # Pull center, x2 offset from globals
        center_global = self.get_dataset("global.Voltages.center")
        x2_offset_global = self.get_dataset("global.Voltages.Offsets.X2")
        DX_global = self.get_dataset("global.Voltages.Offsets.DX")
        DZ_global = self.get_dataset("global.Voltages.Offsets.DZ")
        DX_inspect_Offset = self.get_dataset("monitor.inspect_DX_Offset")

        self.sandia_box.X2_offset = float(x2_offset_global)
        self.sandia_box.X1 = 0 * float(self.X1)
        self.sandia_box.X2 = 0.01
        self.sandia_box.X3 = 0 * float(self.X3)
        self.sandia_box.X4 = 0 * float(self.X4)
        self.sandia_box.X4_offset = 0 * float(self.X4_offset)
        self.sandia_box.QXZ = 0 * float(self.QXZ)
        self.sandia_box.QXZ_offset = 0 * float(self.QXZ_offset)
        self.sandia_box.QZZ = 0 * float(self.QZZ)
        self.sandia_box.QZZ_offset = 0 * float(self.QZZ_offset)
        self.sandia_box.QZY = 0 * float(self.QZY)
        self.sandia_box.QZY_offset = 0 * float(self.QZY_offset)
        self.sandia_box.center = 0 * center_global
        self.sandia_box.X6 = 0
        global_dx_offset = 0
        self.sandia_box.DX = DX_global + global_dx_offset
        self.sandia_box.DZ = DZ_global
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_global_all_compensations()
        self.sandia_box.dac_pc.tweak_dictionary[613]["DX"] = (DX_inspect_Offset - global_dx_offset - DX_global) / 1e3 + \
                                                             self.sandia_box.dac_pc.tweak_dictionary[613]["DX"]
        self.doppler_cooling.set_detuning(-40 * MHz)

        _LOGGER.debug("Done Preparing Experiment")

    def run(self):

        self.set_dataset(
            "data.loading.active_pmts", np.array(self.active_pmts), broadcast=True
        )

        self.set_dataset(
            "data.loading.load_attempts",
            np.linspace(1, self.max_attempts, self.max_attempts),
            broadcast=True,
        )
        self.set_dataset(
            "data.loading.pmt_counts",
            np.full((self.num_pmts, self.max_attempts), np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data.loading.inspect_counts",
            np.full((1, self.max_attempts), np.nan),
            broadcast=True
        )
        self.set_dataset(
            "data.loading.success_flags",
            np.linspace(1, self.max_attempts, self.max_attempts),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="Ion Loading Counts",
            command=self.applet_stream_cmd
                    + " --x " + "data.loading.load_attempts"
                    + " --y-names " + "data.loading.pmt_counts"
                    + " --y-label '{0}'".format("PMT Counts")
                    + " --x-label '{0}'".format("Load Attempt")
                    + " --active-pmts " + "data.loading.active_pmts",
            group="Loading",
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
        self.magfield_z.set_current(Iz + 0.2)

        try:

            # Turn on the oven and wait
            self.preheat()

            if self.clear_trap:
                self.host_eject_quantum()

            # Upload data to the Sandia Box
            self.sandia_box.dac_pc.send_voltage_lines_to_fpga()
            self.sandia_box.dac_pc.send_shuttling_lookup_table()
            sleep(0.1)

            # Initialize Synchronous Devices
            self.kn_initialize()
            self.kn_autoload()
            self.set_dataset("monitor.Lost_Ions", False, broadcast=True)
            self.set_dataset("global.AWG.N_ions", self.goal_ions, persist=True)


        except TerminationRequested:
            _LOGGER.info("Termination Request Received: Ending Experiment")

        finally:
            _LOGGER.info("Done with Experiment")
            self.kn_idle()
            # self.set_dataset("monitor.Lost_Ions.Yb_pp", False, broadcast=True)
            # self.set_dataset("monitor.Lost_Ions.Collision", False, broadcast=True)

            if self.oven_off:
                self.yb_oven.turn_off()

    def preheat(self):
        current = self.yb_oven.get_current()

        if (np.abs(current - _OVEN_CURRENT) < 0.05):
            print("Oven already on.")
            return
        self.yb_oven.turn_on(current=_OVEN_CURRENT, voltage=1.65)
        t_wait = 1
        t_verbose = 10
        t = 0
        while t < self.turn_on_time:
            if t % t_verbose == 0:
                print("Heating up oven. {:d} seconds remaining".format(self.turn_on_time - t))
            sleep(t_wait)
            t = t + t_wait
            if self.scheduler.check_pause():
                raise TerminationRequested

    @kernel
    def kn_autoload(self):

        self.core.break_realtime()
        self.raman.set_global_aom_source(dds=True)
        self.raman.global_dds.update_amp(np.int32(1000))
        self.raman.global_dds.off()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.aoms_id.on()  # Turn on ID beam for added Doppler cooling in load

        num_attempts = 0  # Counts the number of attempts to load
        num_ions = 0

        # Chains longer than 15 need a higher threshold
        if self.goal_ions > 15:
            threshold = 0.94
        else:
            threshold = 0.84
            # threshold = 0.76

        # Collect signal for 0 ions, so there is a consistent starting point
        counts = [0] * self.num_pmts
        prev_counts = [0] * self.num_pmts

        prev_counts = self.kn_check(prev_counts)
        delay(200 * us)
        counts = self.kn_check(counts)

        self.save_data(counts, num_attempts)

        self.core.break_realtime()

        check_for_new_ion = True
        while num_attempts < self.max_attempts - 1:
            # Flag variable for checking new ion
            dot_flag = 0

            if self.scheduler.check_pause():
                break

            self.core.break_realtime()
            self.kn_clear_load()
            counts = [0] * self.num_pmts
            with parallel:
                self.kn_load()
                counts = self.kn_check(counts)
            self.kn_cool()

            self.save_data(counts, num_attempts)

            num_ions_prev = num_ions
            num_ions_res = self.count_ions(num_ions, counts, prev_counts, threshold)
            if check_for_new_ion:
                num_ions = num_ions_res
            prev_counts = counts

            if num_ions == 3 and num_ions_prev == 2:
                self.sandia_box.X2 = 0.003
                print('Updating x2 after loading 2 ions')
                self.sandia_box.calculate_compensations()
                self.sandia_box.tweak_line_all_compensations("QuantumRelaxed_LoadOn")
                self.sandia_box.dac_pc.apply_line_async("QuantumRelaxed_LoadOn", line_gain=1)
                delay(500 * ms)
                with parallel:
                    self.kn_load()
                    counts = self.kn_check(counts)
                delay(10 * ms)

            if num_ions == self.goal_ions:
                break

            self.kn_eject_hot_ions()

            counts_inspect = 0
            if not self.inspect_on:
                # if inspection is off, shuttle from load to
                # the centered relaxed potential
                self.sandia_box.hw_shuttle_load_to_m100p100()
                delay(50 * ms)
                self.sandia_box.hw_shuttle_m100p100_to_relaxed()
            else:
                # if inspection is on
                # check if ions are present at the "inspection" stage.
                self.sandia_box.hw_shuttle_load_to_m100p100()
                delay(50 * ms)
                self.sandia_box.hw_shuttle_m100p100_to_inspect()
                counts_inspect = self.kn_check_accumulated(self.inspect_win)
                delay(500 * us)
                # if not present, dump the inspection well
                if counts_inspect < self.ion_present_counts:
                    self.sandia_box.hw_dump_inspect()
                    delay(500 * us)
                    self.sandia_box.hw_shuttle_center_to_relaxed()
                    check_for_new_ion = False
                # if an ion is present in the inspection well
                # rewind to m100p100
                # and then proceed to merge
                else:
                    self.sandia_box.hw_shuttle_inspect_to_m100p100()
                    delay(5 * ms)
                    self.sandia_box.hw_shuttle_m100p100_to_relaxed()
                    check_for_new_ion = True

            delay(50 * ms)

            num_attempts += 1

            if check_for_new_ion and num_ions > num_ions_prev:
                new_ion_flag = 1
            else:
                new_ion_flag = 0

            self.mutate_dataset(
                "data.loading.success_flags", num_attempts, new_ion_flag
            )

            self.mutate_dataset(
                "data.loading.load_attempts", num_attempts, num_attempts
            )
            self.mutate_dataset(
                "data.loading.inspect_counts",
                ((0, 1, 1), (num_attempts, None, self.max_attempts)),
                counts_inspect,
            )

    @rpc(flags={"async"})
    def save_data(self, counts, i_attempt):
        counts = np.array(counts, ndmin=2).T
        self.mutate_dataset(
            "data.loading.pmt_counts",
            ((0, self.num_pmts, 1), (i_attempt, None, self.max_attempts)),
            counts,
        )

    @rpc
    def count_ions(self, curr_ion_count, cnts, prev_cnts, threshold) -> TInt32:
        array_cnts = (1.0) * np.array(cnts)
        array_prev_cnts = (1.0) * np.array(prev_cnts)
        # print(cnts)
        # print(prev_cnts)
        numerator = np.dot(array_cnts, array_prev_cnts) * np.dot(
            array_cnts, array_prev_cnts
        )
        denominator = (1.0) + np.dot(array_cnts, array_cnts) * np.dot(
            prev_cnts, array_prev_cnts
        )
        dot_product = numerator / denominator
        # if we had low counts and we still have low counts, there is no transition
        # Set threshold at 50 counts
        if (
            np.dot(array_cnts, array_cnts) < 15 ** 2
            and np.dot(array_prev_cnts, array_prev_cnts) < 15 ** 2
        ):
            print("still zero ions...")
            new_ion_count = curr_ion_count

        # if the normalized counts changed by more than 2.5%, there was a transition
        # for 15-ion chains
        elif dot_product < threshold:
            print("transition!")
            new_ion_count = curr_ion_count + 1
        else:
            new_ion_count = curr_ion_count
        print(
            "Estimated Ion Count: {:d}. Dot Product {:f}".format(
                new_ion_count, dot_product
            )
        )
        return np.int32(new_ion_count)

    @kernel
    def kn_clear_load(self):
        self.sandia_box.hw_dump_load(dump=True)
        delay(1 * ms)
        self.sandia_box.hw_dump_load(dump=False)
        delay(1 * ms)

    def host_eject_quantum(self):
        # First save the desired compensations
        current_compensations = copy.deepcopy(self.sandia_box.dac_pc.adjustment_dictionary)

        # Now clear the compensation lines to only apply EJECT
        for key in self.sandia_box.dac_pc.adjustment_dictionary:
            self.sandia_box.dac_pc.adjustment_dictionary[key]["adjustment_gain"] = 0

        self.sandia_box.dac_pc.adjustment_dictionary["Eject"]["adjustment_gain"] = 1.0
        self.sandia_box.dac_pc.line_gain = 0.0
        # Upload new voltage lines (with no compensations)
        self.sandia_box.dac_pc.send_voltage_lines_to_fpga()
        self.sandia_box.dac_pc.send_shuttling_lookup_table()
        sleep(0.1)

        # Eject
        self.sandia_box.dac_pc.apply_line_async("Eject")
        sleep(1)

        # Now put the desired compensations back,
        # NOTE: voltage lines need to be re-uploaded to take effect, which they are in the self.run()
        self.sandia_box.dac_pc.line_gain = 1.0
        self.sandia_box.dac_pc.adjustment_dictionary = copy.deepcopy(current_compensations)
        self.sandia_box.dac_pc.adjustment_dictionary["Eject"]["adjustment_gain"] = 0.0
        self.sandia_box.dac_pc.send_voltage_lines_to_fpga()
        self.sandia_box.dac_pc.send_shuttling_lookup_table()
        self.sandia_box.dac_pc.apply_line_async("QuantumRelaxed_LoadOn")

    @kernel
    def kn_load(self):
        with parallel:
            self.ionization_beam.pulse(self.load_time)
            self.shutter_399.pulse(self.load_time)

    @kernel
    def kn_cool(self):
        if self.global_on:
            self.raman.global_dds.on()
        delay(self.cool_time)
        if self.global_on:
            self.raman.global_dds.off()

    @kernel
    def kn_eject_hot_ions(self):
        self.sandia_box.hw_boil_load(boil=True)
        delay(self.eject_time)
        self.sandia_box.hw_boil_load(boil=False)

    @kernel
    def kn_shuttle_merge(self):
        self.sandia_box.hw_shuttle_merge()

    @kernel
    def kn_check(self, buffer):
        self.doppler_cooling.set_power(0b10)
        for ipmt in range(self.num_pmts):
            stopcounter_mu = self.pmt_array.counter[ipmt].gate_rising(self.det_win)
            delay(10 * us)
            buffer[ipmt] = self.pmt_array.counter[ipmt].count(stopcounter_mu)
        self.doppler_cooling.set_power(0b10)
        return buffer

    @kernel
    def kn_check_accumulated(self, window) -> TInt32:
        """Check the PMT counts across all PMTs in the array."""
        self.doppler_cooling.set_power(0b10)
        acc = 0
        for ipmt in range(self.num_pmts):
            stopcounter_mu = self.pmt_array.counter[ipmt].gate_rising(window)
            delay(10 * us)
            acc = acc + self.pmt_array.counter[ipmt].count(stopcounter_mu)
        self.doppler_cooling.set_power(0b10)
        return acc

    @kernel
    def kn_initialize(self):
        """Initialize"""
        self.core.reset()
        self.core.break_realtime()
        self.pmt_array.clear_buffer()
        self.oeb.off()
        self.doppler_cooling.init()
        self.shutter_399.off()
        self.ionization_beam.off()

        if self.global_on:
            self.raman.global_dds.on()

        if self.clear_trap:
            self.sandia_box.hw_dump_quantum(dump=True)
            delay(500 * ms)
            self.sandia_box.hw_dump_quantum(dump=False)

        self.sandia_box.hw_idle(load=True)

    @kernel
    def kn_idle(self):
        self.core.break_realtime()
        self.raman.global_dds.off()
        self.doppler_cooling.idle()
        self.aoms_id.off()
        self.shutter_399.off()
        self.ionization_beam.off()
        # self.kn_clear_load()
        # self.sandia_box.hw_idle(load=False)  # TODO: This should be load false? Maybe eject load to be sure first

class AutoloadMixedArbSeq(artiq.language.environment.EnvExperiment):
    """Loading.Autoload.Mixed"""

    # TODO:
    # Replace the input of initial ion config with an array input

    kernel_invariants = {
        "load_time",
        "eject_time",
        "cool_time",
        "det_win",
        "inspect_win",
        "pmt_array",
        "num_pmts",
        "max_attempts",
        "clear_trap",
    }

    applet_stream_cmd = (
        "$python -m euriqafrontend.applets.plot_multi" + " "
    )  # White space is required

    def build(self):
        # Get Loading arguments
        # 172: 1; 171: 0
        # isotope sequence input is the decimal form of the array
        self.isotope_sequence_decimal = self.get_argument(
            "isotope_sequence_decimal",
            NumberValue(default=1, unit="ions", step=1, min=1, max=1e10, scale=1),
            group="Loading Parameters",
        )

        self.load_time = self.get_argument(
            "load wait time",
            NumberValue(
                default=300 * ms, unit="ms", step=10 * ms, min=1 * ms, max=10 * s
            ),
            group="Loading Parameters",
        )
        self.load_qubit_time = self.get_argument(
            "load qubit wait time",
            NumberValue(
                default=6 * ms, unit="ms", step=1 * ms, min=0 * ms, max=10 * s
            ),
            group="Loading Parameters",
        )
        self.eject_time = self.get_argument(
            "eject wait time",
            NumberValue(
                default=100 * ms, unit="ms", step=10 * ms, min=1 * ms, max=10 * s
            ),
            group="Loading Parameters",
        )
        self.cool_time = self.get_argument(
            "cool wait time",
            NumberValue(
                default=500 * ms, unit="ms", step=10 * ms, min=1 * ms, max=10 * s
            ),
            group="Loading Parameters",
        )
        self.det_win = self.get_argument(
            "detection window",
            NumberValue(default=10 * ms, unit="ms", step=1 * ms, min=1 * us, max=1 * s),
            group="Loading Parameters",
        )
        self.inspect_win = self.get_argument(
            "inspection detection window",
            NumberValue(default=10 * ms, unit="ms", step=1 * ms, min=1 * us, max=1 * s),
            group="Loading Parameters",
        )
        self.max_attempts = self.get_argument(
            "max # load attempts",
            NumberValue(
                default=500,
                unit="attempts",
                step=1,
                min=1,
                max=500,
                ndecimals=0,
                scale=1,
            ),
            group="Loading Parameters",
        )
        self.turn_on_time = self.get_argument(
            "Oven Preheat Time",
            NumberValue(
                default=100, unit="s", step=1, min=1, max=240, ndecimals=0, scale=1
            ),
            group="Loading Parameters",
        )
        self.global_on = self.get_argument(
            "Use Global Beam During Cooling",
            BooleanValue(default=False),
            group="Loading Parameters",
        )
        self.oven_off = self.get_argument(
            "Turn oven OFF after experiment",
            BooleanValue(default=True),
            group="Loading Parameters",
        )
        self.clear_trap = self.get_argument(
            "Dump all ions from trap before autoloading",
            BooleanValue(default=True),
            group="Loading Parameters",
        )
        self.inspect_on = self.get_argument(
            "Inspect before merge",
            BooleanValue(default=True),
            group="Loading Parameters",
        )
        self.setattr_argument(
            "ion_present_counts",
            NumberValue(
                100,
                unit="counts",
                scale=1,
                ndecimals=0,
                step=1,
                min=0
            ),
            group="Loading Parameters",
            tooltip="Number of PMT counts across the entire chain to confirm an ion " \
                    "is present during pre-merge inspection."
        )

        # Get PMT Settings
        self.pmt_array = PMTArray(self)

        # 394 diode control
        self.ionization_diode_control = IonizationDiodeControl(self)

        # Load Core devices
        self.setattr_device("core")
        self.setattr_device("scheduler")
        self.setattr_device("oeb")
        self.setattr_device("ccb")
        self.raman = Raman(self)

        # Voltage, oven, id aom, magfield
        self.sandia_box = SandiaDAC(self)
        self.setattr_device("yb_oven")
        self.setattr_device("aoms_id")

        self.setattr_device("magfield_x")
        self.setattr_device("magfield_y")
        self.setattr_device("magfield_z")

        # cooling, pumping and ionization
        self.doppler_cooling = DopplerCooling(self)
        self.doppler_cooling_coolant = DopplerCoolingCoolant(self)
        self.setattr_device("shutter_399")
        self.setattr_device("shutter_394")
        self.setattr_device("aom_399")
        self.setattr_device("eom_935_3ghz")

        # Overwrite ability to pull in global values. Loading requires certain compensation values.
        self.setattr_argument("use_global_voltages", EnumerationValue(["False"], default="False"), group="Sandia DAC")

        # TODO: set qxz_offset & qxz to correct value??
        # QXZ offset is the same as dataset, not sure about QXZ
        # QZY and QZZ offset correct (dataset: 0, -0.03)
        self.QXZ_offset = self.get_argument(
            "QXZ offset (MHz^2)",
            EnumerationValue(["0.00"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QZZ_offset = self.get_argument(
            "QZZ offset (MHz^2)",
            EnumerationValue(["0.05"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QZY_offset = self.get_argument(
            "QZY offset (MHz^2)",
            EnumerationValue(["-0.5"]),
            group="Sandia DAC",
            tooltip="You cannot change this value if you want to load",
        )
        self.X4_offset = self.get_argument(
            "X4 offset (MHz^2 / (2.74 um)^2)",
            EnumerationValue(["-0.0006"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QXZ = self.get_argument(
            "QXZ (MHz^2)",
            EnumerationValue(["0.0"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QZZ = self.get_argument(
            "QZZ (MHz^2)",
            EnumerationValue(["0.0"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.QZY = self.get_argument(
            "QZY (MHz^2)",
            EnumerationValue(["0.5"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )
        self.X1 = self.get_argument(
            "X1 (2.74 um * MHz^2)",
            EnumerationValue(["0.0"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )

        self.X2 = self.get_argument(
            "X2 (MHz^2)",
            NumberValue(default=0.001, unit="", min=-0.5, max=+0.5, ndecimals=4),
            group="Sandia DAC",
        )

        self.X3 = self.get_argument(
            "X3 (MHz^2 / (2.74 um))",
            EnumerationValue(["0.0"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."

        )
        self.X4 = self.get_argument(
            "X4 (MHz^2 / (2.74 um)^2)",
            EnumerationValue(["0.0006"]),
            # EnumerationValue(["0.000355"]),
            group="Sandia DAC",
            tooltip="You cannot change this value when autoloading. It will break the loading."
        )

        _LOGGER.debug("Done Building Experiment")

    def prepare(self):
        # ion sequence
        # transform back to the integer array with ion index going from lower to higher
        # from left to right
        self.isotope_sequence = str(bin(np.int(self.isotope_sequence_decimal))[2:])
        self.isotope_sequence_list = []
        for index in range(len(self.isotope_sequence)):
            self.isotope_sequence_list.append(int(self.isotope_sequence[index]))
        self.coolant_num = np.sum(self.isotope_sequence_list)
        self.qubit_num = len(self.isotope_sequence_list) - self.coolant_num
        self.set_dataset("global.AWG.N_qubits", self.qubit_num, broadcast=True)
        self.set_dataset("global.AWG.N_coolants", self.coolant_num, broadcast=True)

        # Don't use global voltage
        self.sandia_box.use_global_voltages = False

        # Use the shutting files for autoloader which doesn't involve split, merge and swap paths
        self.sandia_box.shuttle_flag = False

        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        self.ionization_diode_control.prepare()

        self.num_pmts = self.pmt_array.num_active
        self.active_pmts = self.pmt_array.active_pmts

        # Overwrite DAC values

        # Pull center, x2 offset from globals
        center_global = self.get_dataset("global.Voltages.center")
        x2_offset_global = self.get_dataset("global.Voltages.Offsets.X2")
        DX_global = self.get_dataset("global.Voltages.Offsets.DX")
        DZ_global = self.get_dataset("global.Voltages.Offsets.DZ")
        DX_inspect_Offset = self.get_dataset("monitor.inspect_DX_Offset")

        self.sandia_box.X2_offset = float(x2_offset_global)
        self.sandia_box.X1 = 0 * float(self.X1)
        self.sandia_box.X2 = 0.01
        self.sandia_box.X3 = 0 * float(self.X3)
        self.sandia_box.X4 = 0 * float(self.X4)
        self.sandia_box.X4_offset = 0 * float(self.X4_offset)
        self.sandia_box.QXZ = 0 * float(self.QXZ)
        self.sandia_box.QXZ_offset = 0 * float(self.QXZ_offset)
        self.sandia_box.QZZ = 0 * float(self.QZZ)
        self.sandia_box.QZZ_offset = 0 * float(self.QZZ_offset)
        self.sandia_box.QZY = 0 * float(self.QZY)
        self.sandia_box.QZY_offset = 0 * float(self.QZY_offset)
        self.sandia_box.center = 0 * center_global
        self.sandia_box.X6 = 0
        global_dx_offset = 0
        self.sandia_box.DX = DX_global + global_dx_offset
        self.sandia_box.DZ = DZ_global
        self.sandia_box.calculate_compensations()
        self.sandia_box.tweak_global_all_compensations()
        self.sandia_box.dac_pc.tweak_dictionary[613]["DX"] = (DX_inspect_Offset - global_dx_offset - DX_global) / 1e3 + \
                                                             self.sandia_box.dac_pc.tweak_dictionary[613]["DX"]
        self.doppler_cooling.set_detuning(-40 * MHz)

        self.num_attempts = 0
        self.num_qubit = 0
        self.num_qubit_prev = 0
        self.num_qubit_wrong = 0
        self.num_coolant = 0
        self.num_coolant_prev = 0
        self.num_coolant_wrong = 0

        # Loading statistics 
        if self.qubit_num > 0:
            self.qubit_load_attempt_list = [0] * self.qubit_num
        else:
            self.qubit_load_attempt_list = [0]
        if self.coolant_num > 0:
            self.coolant_load_attempt_list = [0] * self.coolant_num
        else:
            self.coolant_load_attempt_list = [0]
        self.qubit_loading_rate = 0
        self.coolant_loading_rate = 0
        self.qubit_Poisson_flag = True
        self.coolant_Poisson_flag = True
        self.loading_CI = 0.9

        # global index of the ion to be load (starts from 1)
        self.ind = 1

        # loading flag
        self.check_for_new_ion = 0

        self.qubit_counts_inspect = 0
        self.coolant_counts_inspect = 0

        _LOGGER.debug("Done Preparing Experiment")

    def run(self):
        self.ds_initialize()
        _LOGGER.info(self.isotope_sequence_list)

        self.ccb.issue(
            "create_applet",
            name="Ion Loading Counts",
            command=self.applet_stream_cmd
                    + " --x " + "data.loading.load_attempts"
                    + " --y-names " + "data.loading.pmt_counts"
                    + " --y-label '{0}'".format("PMT Counts")
                    + " --x-label '{0}'".format("Load Attempt")
                    + " --active-pmts " + "data.loading.active_pmts",
            group="Loading",
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
        self.magfield_z.set_current(Iz + 0.2)

        try:

            # Turn on the oven and wait
            self.preheat()

            if self.clear_trap:
                self.host_eject_quantum()

            # Upload data to the Sandia Box
            self.sandia_box.dac_pc.send_voltage_lines_to_fpga()
            self.sandia_box.dac_pc.send_shuttling_lookup_table()
            sleep(0.1)

            # Initialize Synchronous Devices
            self.kn_initialize()

            # for index in self.isotope_sequence:
            for index in self.isotope_sequence_list:
                if index == 1 and self.num_attempts < self.max_attempts - 1 and self.coolant_num > 0:
                    _LOGGER.debug("Coolant %d", self.ind)
                    self.isotope_autoload(172)
                    self.ind += 1
                    
                elif index == 0 and self.num_attempts < self.max_attempts - 1 and self.qubit_num > 0:
                    _LOGGER.debug("Qubit %d", self.ind)
                    self.isotope_autoload(171)
                    self.ind += 1
                    
                elif self.num_attempts >= self.max_attempts:
                    _LOGGER.info("Loading failed.")
                    break
                
                # Check dot product here
            
            # self.set_dataset("monitor.Lost_Ions", False, broadcast=True)
            # self.set_dataset("global.AWG.N_ions", self.qubit_num + self.coolant_num, persist=True)

        except TerminationRequested:
            # for termination request
            _LOGGER.info("Termination Request Received: Ending Experiment")

        finally:
            _LOGGER.info("Done with Experiment")
            self.kn_idle()
            # self.set_dataset("monitor.Lost_Ions.Yb_pp", False, broadcast=True)
            # self.set_dataset("monitor.Lost_Ions.Collision", False, broadcast=True)

            if self.oven_off:
                self.yb_oven.turn_off()

    def preheat(self):
        current = self.yb_oven.get_current()

        if (np.abs(current - _OVEN_CURRENT) < 0.05):
            print("Oven already on.")
            return
        self.yb_oven.turn_on(current=_OVEN_CURRENT, voltage=1.65)
        t_wait = 1
        t_verbose = 10
        t = 0
        while t < self.turn_on_time:
            if t % t_verbose == 0:
                print("Heating up oven. {:d} seconds remaining".format(self.turn_on_time - t))
            sleep(t_wait)
            t = t + t_wait
            if self.scheduler.check_pause():
                raise TerminationRequested

    @kernel
    def kn_check(self, buffer_qubit, buffer_coolant):
        # Start from default cooling setting, 370(171) is at 0b10, 370(172) is on
        # 171 counts
        self.doppler_cooling_coolant.cool(False)

        for ipmt in range(self.num_pmts):
            stopcounter_mu = self.pmt_array.counter[ipmt].gate_rising(self.det_win)
            buffer_qubit[ipmt] = self.pmt_array.counter[ipmt].count(stopcounter_mu)
        self.doppler_cooling_coolant.cool(True)

        # 172 counts
        self.doppler_cooling.set_power(0b00)

        for ipmt in range(self.num_pmts):
            stopcounter_coolant_mu = self.pmt_array.counter[ipmt].gate_rising(self.det_win)
            buffer_coolant[ipmt] = self.pmt_array.counter[ipmt].count(stopcounter_coolant_mu)
        self.doppler_cooling.set_power(0b10)

        return buffer_qubit, buffer_coolant

    @kernel
    def isotope_autoload(self, isotope):

        # Chains longer than 15 need a higher threshold
        # Currently only use inspect counts to check presence, threshold is not in use
        threshold = 0.94

        # Collect signal for 0 ions, so there is a consistent starting point
        counts_qubit = [0] * self.num_pmts
        prev_counts_qubit = [0] * self.num_pmts
        counts_coolant = [0] * self.num_pmts
        prev_counts_coolant = [0] * self.num_pmts

        # prev_counts_qubit, prev_counts_coolant = self.kn_check(prev_counts_qubit, prev_counts_coolant)
        # delay(200 * us)
        # counts_qubit, counts_coolant = self.kn_check(counts_qubit, counts_coolant)

        if isotope == 171:
            _LOGGER.debug("171 loader.")
            self.check_for_new_ion = 0
            while (self.num_attempts < self.max_attempts - 1) and self.check_for_new_ion == 0:

                if self.scheduler.check_pause():
                    raise TerminationRequested

                self.core.break_realtime()
                self.kn_clear_load()
                self.kn_load(171)
                self.kn_cool()
                self.qubit_load_attempt_list[self.num_qubit] += 1
                self.num_coolant_prev = self.num_coolant
                self.num_qubit_prev = self.num_qubit

                if self.num_qubit + self.num_coolant == 3 and self.num_qubit_prev + self.num_coolant_prev == 2:
                    self.sandia_box.X2 = 0.003
                    _LOGGER.debug('Updating x2 after loading 2 ions')
                    self.sandia_box.calculate_compensations()
                    self.sandia_box.tweak_line_all_compensations("QuantumRelaxed_LoadOn")
                    self.sandia_box.dac_pc.apply_line_async("QuantumRelaxed_LoadOn", line_gain=1)
                    delay(500 * ms)

                self.kn_eject_hot_ions()

                # Determine the 171 ion number with inspection
                # check if ions are present at the "inspection" stage.
                self.sandia_box.hw_shuttle_load_to_m100p100()
                delay(50 * ms)
                self.sandia_box.hw_shuttle_m100p100_to_inspect()
                # longer hold time in inspect
                (self.qubit_counts_inspect, self.coolant_counts_inspect) = self.kn_check_accumulated(self.inspect_win)
                delay(50 * ms)
                _LOGGER.debug("Qubit counts %d", self.qubit_counts_inspect)
                _LOGGER.debug("Coolant counts %d", self.coolant_counts_inspect)

                # if inspection for 171 and 172 both shows presence, the inspection fails,
                # and we stop the experiment immediately
                # commented out if during debugging
                assert (
                    self.qubit_counts_inspect < self.ion_present_counts or self.coolant_counts_inspect < self.ion_present_counts), "Inspection fails"

                # if neither 171 nor 172 is present, dump the inspection well
                if self.qubit_counts_inspect < self.ion_present_counts and self.coolant_counts_inspect < self.ion_present_counts:
                    self.sandia_box.hw_dump_inspect()
                    delay(50 * ms)
                    self.sandia_box.hw_shuttle_center_to_relaxed()
                    delay(50 * ms)
                    self.check_for_new_ion = 0
                    _LOGGER.info("No load.")
                # if a 171 ion is present in the inspection well, rewind to m100p100, and then proceed to merge
                elif self.qubit_counts_inspect >= self.ion_present_counts and self.coolant_counts_inspect < self.ion_present_counts:
                    self.sandia_box.hw_shuttle_inspect_to_m100p100()
                    delay(50 * ms)
                    self.sandia_box.hw_shuttle_m100p100_to_relaxed()
                    delay(50 * ms)
                    self.check_for_new_ion = 1
                    self.num_qubit += 1
                    _LOGGER.info((self.num_attempts, self.num_qubit, self.num_coolant, self.qubit_counts_inspect, self.coolant_counts_inspect))
                # if a 172 ion is present in the inspection well, dump the inspection well, mark a wrong loading event
                elif self.coolant_counts_inspect >= self.ion_present_counts:  # and self.qubit_counts_inspect < self.ion_present_counts:
                    self.sandia_box.hw_dump_inspect()
                    delay(50 * ms)
                    self.sandia_box.hw_shuttle_center_to_relaxed()
                    delay(50 * ms)
                    self.check_for_new_ion = 0
                    self.num_qubit_wrong += 1
                    _LOGGER.info("Wrong load.")

                delay(50 * ms)
                self.num_attempts += 1

                # Qubit inspection counts
                self.mutate_dataset(
                    "data.loading.qubit_inspect_counts",
                    ((0, 1, 1), (self.num_attempts, None, self.max_attempts)),
                    self.qubit_counts_inspect,
                )
                # Coolant inspection counts
                self.mutate_dataset(
                    "data.loading.coolant_inspect_counts",
                    ((0, 1, 1), (self.num_attempts, None, self.max_attempts)),
                    self.coolant_counts_inspect,
                )
                # Failed qubit load (load coolant instead)
                self.mutate_dataset(
                    "data.loading.wrong_qubit_load",
                    ((0, 1, 1), (self.num_attempts, None, self.max_attempts)),
                    self.num_qubit_wrong,
                )
                # Failed coolant load (load qubit instead)
                self.mutate_dataset(
                    "data.loading.wrong_coolant_load",
                    ((0, 1, 1), (self.num_attempts, None, self.max_attempts)),
                    self.num_coolant_wrong,
                )


        elif isotope == 172:
            _LOGGER.debug("172 loader.")
            self.check_for_new_ion = 0
            while (self.num_attempts < self.max_attempts - 1) and self.check_for_new_ion == 0:

                if self.scheduler.check_pause():
                    raise TerminationRequested
                
                self.core.break_realtime()
                self.kn_clear_load()
                self.kn_load(172)
                self.kn_cool()
                self.coolant_load_attempt_list[self.num_coolant] += 1
                # self.save_data(self.check_for_new_ion, self.num_attempts)
                self.num_coolant_prev = self.num_coolant
                self.num_qubit_prev = self.num_qubit

                if self.num_qubit + self.num_coolant == 3 and self.num_coolant_prev + self.num_qubit_prev == 2:
                    self.sandia_box.X2 = 0.003
                    _LOGGER.debug('Updating x2 after loading 2 ions')
                    self.sandia_box.calculate_compensations()
                    self.sandia_box.tweak_line_all_compensations("QuantumRelaxed_LoadOn")
                    self.sandia_box.dac_pc.apply_line_async("QuantumRelaxed_LoadOn", line_gain=1)
                    delay(500 * ms)

                self.kn_eject_hot_ions()

                # Determine the 172 ion number with inspection
                # check if ions are present at the "inspection" stage.
                self.sandia_box.hw_shuttle_load_to_m100p100()
                delay(50 * ms)
                self.sandia_box.hw_shuttle_m100p100_to_inspect()
                (self.qubit_counts_inspect, self.coolant_counts_inspect) = self.kn_check_accumulated(self.inspect_win)
                delay(50 * ms)
                _LOGGER.debug("Qubit counts %d", self.qubit_counts_inspect)
                _LOGGER.debug("Coolant counts %d", self.coolant_counts_inspect)

                # if inspection for 171 and 172 both shows presence, the inspection fails,
                # and we stop the experiment immediately
                # commented out during debugging
                assert (
                        self.qubit_counts_inspect < self.ion_present_counts or self.coolant_counts_inspect < self.ion_present_counts), "Inspection fails"

                # if neither 171 nor 172 is present, dump the inspection well
                if self.qubit_counts_inspect < self.ion_present_counts and self.coolant_counts_inspect < self.ion_present_counts:
                    self.sandia_box.hw_dump_inspect()
                    delay(50 * ms)
                    self.sandia_box.hw_shuttle_center_to_relaxed()
                    delay(50 * ms)
                    self.check_for_new_ion = 0
                    _LOGGER.info("No load.")
                # if a 172 ion is present in the inspection well, rewind to m100p100, and then proceed to merge
                elif self.qubit_counts_inspect < self.ion_present_counts and self.coolant_counts_inspect >= self.ion_present_counts:
                    self.sandia_box.hw_shuttle_inspect_to_m100p100()
                    delay(50 * ms)
                    self.sandia_box.hw_shuttle_m100p100_to_relaxed()
                    delay(50 * ms)
                    self.check_for_new_ion = 1
                    self.num_coolant += 1
                    _LOGGER.info((self.num_attempts, self.num_qubit, self.num_coolant, self.qubit_counts_inspect, self.coolant_counts_inspect))
                # if a 171 ion is present in the inspection well, dump the inspection well, mark a wrong loading event
                elif self.qubit_counts_inspect >= self.ion_present_counts:  # and self.coolant_counts_inspect < self.ion_present_counts:
                    self.sandia_box.hw_dump_inspect()
                    delay(50 * ms)
                    self.sandia_box.hw_shuttle_center_to_relaxed()
                    delay(50 * ms)
                    self.check_for_new_ion = 0
                    self.num_coolant_wrong += 1
                    _LOGGER.info("Wrong load.")

                delay(50 * ms)
                self.num_attempts += 1

                # Qubit inspection counts
                self.mutate_dataset(
                    "data.loading.qubit_inspect_counts",
                    ((0, 1, 1), (self.num_attempts, None, self.max_attempts)),
                    self.qubit_counts_inspect,
                )
                # Coolant inspection counts
                self.mutate_dataset(
                    "data.loading.coolant_inspect_counts",
                    ((0, 1, 1), (self.num_attempts, None, self.max_attempts)),
                    self.coolant_counts_inspect,
                )
                # Failed qubit load (load coolant instead)
                self.mutate_dataset(
                    "data.loading.wrong_qubit_load",
                    ((0, 1, 1), (self.num_attempts, None, self.max_attempts)),
                    self.num_qubit_wrong,
                )
                # Failed coolant load (load qubit instead)
                self.mutate_dataset(
                    "data.loading.wrong_coolant_load",
                    ((0, 1, 1), (self.num_attempts, None, self.max_attempts)),
                    self.num_coolant_wrong,
                )

    @rpc(flags={"async"})
    def save_counts(self, counts, i_attempt):
        counts = np.array(counts, ndmin=2).T
        self.mutate_dataset(
            "data.loading.pmt_counts",
            ((0, self.num_pmts, 1), (i_attempt, None, self.max_attempts)),
            counts,
        )

    @rpc(flags={"async"})
    def count_ions(self, curr_ion_count, cnts, prev_cnts, threshold) -> TInt32:
        array_cnts = (1.0) * np.array(cnts)
        array_prev_cnts = (1.0) * np.array(prev_cnts)

        numerator = np.dot(array_cnts, array_prev_cnts) * np.dot(
            array_cnts, array_prev_cnts
        )
        denominator = (1.0) + np.dot(array_cnts, array_cnts) * np.dot(
            prev_cnts, array_prev_cnts
        )
        dot_product = numerator / denominator
        # if we had low counts and we still have low counts, there is no transition
        # Set threshold at 50 counts
        # Wrong?
        if (
            np.dot(array_cnts, array_cnts) < 15 ** 2
            and np.dot(array_prev_cnts, array_prev_cnts) < 15 ** 2
        ):
            print("Zero ion.")
            new_ion_count = curr_ion_count

        # if the normalized counts changed by more than 2.5%, there was a transition
        # for 15-ion chains
        elif dot_product < threshold:
            print("transition!")
            new_ion_count = curr_ion_count + 1
        else:
            new_ion_count = curr_ion_count
        # print(
        #     "Estimated Ion Count: {:d}. Dot Product {:f}".format(
        #         new_ion_count, dot_product
        #     )
        # )
        print("No load.")
        return np.int32(new_ion_count)

    @kernel
    def kn_clear_load(self):
        self.sandia_box.hw_dump_load(dump=True)
        delay(30 * ms)
        self.sandia_box.hw_dump_load(dump=False)
        delay(30 * ms)

    def host_eject_quantum(self):
        # First save the desired compensations
        current_compensations = copy.deepcopy(self.sandia_box.dac_pc.adjustment_dictionary)

        # Now clear the compensation lines to only apply EJECT
        for key in self.sandia_box.dac_pc.adjustment_dictionary:
            self.sandia_box.dac_pc.adjustment_dictionary[key]["adjustment_gain"] = 0

        self.sandia_box.dac_pc.adjustment_dictionary["Eject"]["adjustment_gain"] = 1.0
        self.sandia_box.dac_pc.line_gain = 0.0
        # Upload new voltage lines (with no compensations)
        self.sandia_box.dac_pc.send_voltage_lines_to_fpga()
        self.sandia_box.dac_pc.send_shuttling_lookup_table()
        sleep(0.1)

        # Eject
        self.sandia_box.dac_pc.apply_line_async("Eject")
        sleep(1)

        # Now put the desired compensations back,
        # NOTE: voltage lines need to be re-uploaded to take effect, which they are in the self.run()
        self.sandia_box.dac_pc.line_gain = 1.0
        self.sandia_box.dac_pc.adjustment_dictionary = copy.deepcopy(current_compensations)
        self.sandia_box.dac_pc.adjustment_dictionary["Eject"]["adjustment_gain"] = 0.0
        self.sandia_box.dac_pc.send_voltage_lines_to_fpga()
        self.sandia_box.dac_pc.send_shuttling_lookup_table()
        self.sandia_box.dac_pc.apply_line_async("QuantumRelaxed_LoadOn")

    @kernel
    def kn_load(self, isotope):
        if isotope == 171:
            self.doppler_cooling_coolant.cool(False)
            with parallel:
                self.shutter_394.on()
                self.shutter_399.on()
            delay(self.load_qubit_time)
            with parallel:
                self.shutter_394.off()
                self.shutter_399.off()
            self.doppler_cooling_coolant.cool(True)
        elif isotope == 172:
            self.ionization_diode_control.set_to_high()
            # tested that delay is not necessary
            # delay(1 * ms)
            self.doppler_cooling.set_power(0b00)
            self.aoms_id.off()
            with parallel:
                self.shutter_394.on()
                self.aom_399.on()
            delay(self.load_time)
            with parallel:
                self.shutter_394.off()
                self.aom_399.off()
            self.doppler_cooling.set_power(0b10)
            self.aoms_id.on()
            self.ionization_diode_control.return_to_zero()

    @kernel
    def kn_cool(self):
        if self.global_on:
            self.raman.global_dds.on()
        delay(self.cool_time)
        if self.global_on:
            self.raman.global_dds.off()

    @kernel
    def kn_eject_hot_ions(self):
        self.sandia_box.hw_boil_load(boil=True)
        delay(self.eject_time)
        self.sandia_box.hw_boil_load(boil=False)

    @kernel
    def kn_shuttle_merge(self):
        self.sandia_box.hw_shuttle_merge()

    @kernel
    def kn_check_accumulated(self, window):
        """Check the PMT counts across all PMTs in the array."""
        accqubit = 0
        acccoolant = 0

        # qubit counts
        self.doppler_cooling_coolant.cool(False)
        # turn on both 3ghz and sawtooth of 935 (though this function is in coolant), and turn off 172 cooling
        self.doppler_cooling_coolant.pump_sawtooth(False)
        for ipmt in range(self.num_pmts):
            stopcounter_mu = self.pmt_array.counter[ipmt].gate_rising(window)
            delay(10 * us)
            accqubit = accqubit + self.pmt_array.counter[ipmt].count(stopcounter_mu)
            delay(10 * us)

        self.doppler_cooling_coolant.cool(True)

        # coolant counts
        # turn off 171 Doppler cooling completely
        self.doppler_cooling.set_power(0b00)
        # (physically) turn off 935_eom_3ghz only
        self.eom_935_3ghz.on()
        delay(1 * ms)
        for ipmt in range(self.num_pmts):
            stopcounter_mu = self.pmt_array.counter[ipmt].gate_rising(window)
            delay(10 * us)
            acccoolant = acccoolant + self.pmt_array.counter[ipmt].count(stopcounter_mu)
            delay(10 * us)

        # back to default setting
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.doppler_cooling_coolant.cool(True)
        self.eom_935_3ghz.off()
        self.doppler_cooling_coolant.pump_sawtooth(True)

        return (accqubit, acccoolant)

    @kernel
    def kn_initialize(self):
        """Initialize"""
        self.core.reset()
        self.core.break_realtime()
        self.oeb.off()
        self.pmt_array.clear_buffer()
        self.raman.set_global_aom_source(dds=True)
        self.raman.global_dds.update_amp(np.int32(1000))
        self.raman.global_dds.off()
        self.ionization_diode_control.return_to_zero()

        # Default status: both cooling on, both 935 eom physically on, ID beam turned on for additional Doppler cooling (171),
        # both 399 and 394 shuttered, 399 aom off
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.doppler_cooling_coolant.init()
        self.doppler_cooling_coolant.cool(True)
        self.eom_935_3ghz.off()
        self.doppler_cooling_coolant.pump_sawtooth(True)
        self.aoms_id.on()  # Turn on ID beam for added Doppler cooling in load
        self.shutter_399.off()
        self.shutter_394.off()
        self.aom_399.off()

        if self.global_on:
            self.raman.global_dds.on()

        if self.clear_trap:
            self.sandia_box.hw_dump_quantum(dump=True)
            delay(500 * ms)
            self.sandia_box.hw_dump_quantum(dump=False)

        self.sandia_box.hw_idle(load=True)

    @host_only
    def ds_initialize(self):
        self.set_dataset(
            "data.loading.pmt_counts",
            np.full((self.num_pmts, self.max_attempts), 0),
            broadcast=True,
        )
        self.set_dataset(
            "data.loading.qubit_inspect_counts",
            np.full((1, self.max_attempts), 0),
            broadcast=True
        )
        self.set_dataset(
            "data.loading.coolant_inspect_counts",
            np.full((1, self.max_attempts), 0),
            broadcast=True
        )
        self.set_dataset(
            "data.loading.wrong_qubit_load",
            np.full((1, self.max_attempts), 0),
            broadcast=True
        )
        self.set_dataset(
            "data.loading.wrong_coolant_load",
            np.full((1, self.max_attempts), 0),
            broadcast=True
        )
        self.set_dataset("loading.isotope_sequence", self.isotope_sequence_list, broadcast=True)

    @kernel
    def kn_idle(self):
        self.core.break_realtime()
        self.raman.global_dds.off()
        self.aoms_id.off()
        self.shutter_399.off()
        self.shutter_394.off()
        self.aom_399.off()
        self.core.break_realtime()
        # Doppler cooling is default on to continue to cool the ions after loading
        self.doppler_cooling.on()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling_coolant.cool(True)
        # Repumping is left on
        self.core.break_realtime()
        self.eom_935_3ghz.off()
        self.doppler_cooling_coolant.pump_sawtooth(True)
        self.ionization_diode_control.return_to_zero()
        # self.kn_clear_load()
        # self.sandia_box.hw_idle(load=False)
        
        # Saving loading statistics 
        self.Poisson_check()
        self.save_loading_stat()

    @rpc(flags={"async"})
    def Poisson_check(self):
        mulist_qubit = np.linspace(0, 2 * np.mean(self.qubit_load_attempt_list), 100)
        mulist_coolant = np.linspace(0, 2 * np.mean(self.coolant_load_attempt_list), 100)
        opt_pvalue_qubit = 0
        opt_ksstat_qubit = 1
        opt_mu_qubit = 0
        opt_pvalue_coolant = 0
        opt_ksstat_coolant = 1
        opt_mu_coolant = 0
        for mu in mulist_qubit:
            ksstat, pv = kstest(self.qubit_load_attempt_list, 'poisson', args=(mu, ), N=len(self.qubit_load_attempt_list))
            if pv > opt_pvalue_qubit:
                opt_ksstat_qubit = copy.copy(ksstat)
                opt_pvalue_qubit = copy.copy(pv)
                opt_mu_qubit = copy.copy(mu)
        for mu in mulist_coolant:
            ksstat, pv = kstest(self.coolant_load_attempt_list, 'poisson', args=(mu, ), N=len(self.coolant_load_attempt_list))
            if pv > opt_pvalue_coolant:
                opt_ksstat_coolant = copy.copy(ksstat)
                opt_pvalue_coolant = copy.copy(pv)
                opt_mu_coolant = copy.copy(mu)
        if opt_pvalue_qubit < (1 - self.loading_CI):
            self.qubit_Poisson_flag = False
            _LOGGER.info("Qubit loading not Poissonian, consider changing loading parameters.")
        if opt_pvalue_coolant < (1 - self.loading_CI):
            self.coolant_Poisson_flag = False
            _LOGGER.info("Coolant loading not Poissonian, consider changing loading parameters.")
        self.set_dataset(
            "data.loading.optimal_Poisson_fitted_qubit_loading_attempts", 
            np.float(opt_mu_qubit),
            broadcast=True
        )
        self.set_dataset(
            "data.loading.optimal_Poisson_fitted_coolant_loading_attempts", 
            np.float(opt_mu_coolant),
            broadcast=True
        )
        self.set_dataset(
            "data.loading.Pvalue_qubit", 
            np.float(opt_pvalue_qubit),
            broadcast=True
        )
        self.set_dataset(
            "data.loading.Pvalue_coolant", 
            np.float(opt_pvalue_coolant),
            broadcast=True
        )
        print("Coolant", opt_mu_coolant)
        print("Qubit", opt_mu_qubit)

    @rpc(flags={"async"})
    def save_loading_stat(self):
        self.set_dataset(
            "data.loading.qubit_loading_attempts",
            np.array(self.qubit_load_attempt_list),
            broadcast=True
        )
        self.set_dataset(
            "data.loading.coolant_loading_attempts",
            np.array(self.coolant_load_attempt_list),
            broadcast=True
        )

class LoadingRate(artiq.language.environment.EnvExperiment):
    """Loading.LoadingRate

    Note: this experiment will only work when aligned to the load slot with a PMT
    """

    kernel_invariants = set(
        ["load_win", "obs_win", "isotope", "thresh", "pmt"]
    )

    def build(self):
        """Get any ARTIQ arguments."""
        self.isotope = self.get_argument("isotope", EnumerationValue(["171", "172"], default="171"))
        self.num_stats = self.get_argument(
            "number of statistics",
            NumberValue(
                default=50.0, unit="load events", scale=1, step=1, min=1, ndecimals=0
            ),
        )

        self.load_win = self.get_argument(
            "load window",
            NumberValue(
                default=300.0 * ms, unit="ms", step=1 * ms, min=1 * ms, ndecimals=3
            ),
        )

        self.cool_win = self.get_argument(
            "cool window",
            NumberValue(
                default=2000.0 * ms, unit="ms", step=1 * ms, min=1 * ms, ndecimals=3
            ),
        )

        self.obs_win = self.get_argument(
            "observation window",
            NumberValue(
                default=20.0 * ms, unit="ms", step=1 * ms, min=1 * ms, ndecimals=3
            ),
        )

        self.thresh = self.get_argument(
            "ion count threshold",
            NumberValue(default=20, unit="counts", scale=1, step=1, min=1, ndecimals=0),
        )

        self.turn_on_time = self.get_argument(
            "Oven Preheat Time",
            NumberValue(
                default=50, unit="s", step=1, min=1, max=100, ndecimals=0, scale=1
            )
        )
        self.oven_off = self.get_argument(
            "Turn oven OFF after experiment",
            BooleanValue(default=True),
        )

        self.just_load_one = self.get_argument(
            "Load one ion in the Load slot and then stop", BooleanValue(default=False)
        )

        # Get PMT Settings
        self.pmt_array = PMTArray(self)

        # Load core devices
        self.setattr_device("scheduler")
        self.setattr_device("core")
        self.setattr_device("oeb")
        self.setattr_device("ccb")

        # Load other devices
        # self.dac = self.get_device("dac_pc_interface")
        self.sandia_box = SandiaDAC(self)
        self.dac = self.sandia_box.dac_pc
        self.setattr_device("yb_oven")

        # Load Laser Shutters
        self.doppler_cooling = DopplerCooling(self)
        self.setattr_device("shutter_399")
        self.setattr_device("shutter_394")
        self.setattr_device("eom_935_172")
        self.setattr_device("aom_172_369")
        self.setattr_device("aom_399")

        # 394 diode control
        self.ionization_diode_control = IonizationDiodeControl(self)

        _LOGGER.debug("Done Building Experiment")

    def prepare(self):
        """Prepare Children"""
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        assert (
            self.pmt_array.num_active == 1
        ), "PMT Array is longer than 1. Please specify which PMT is the load PMT"
        self.pmt = self.pmt_array.counter[0]

        self.doppler_cooling.set_detuning(-60 * MHz)

        self.ionization_diode_control.prepare()

        self.isotope = int(self.isotope)
        self.set_dataset(
            "data.loading.load_stats", np.full(self.num_stats, 0), broadcast=True
        )
        _LOGGER.debug("Done Preparing Experiment")

        self.pre_load = 0 * ms
        self.load_overlap = self.load_win
        if self.load_win < 20 * ms:
            self.pre_load = 20 * ms - self.load_win
            self.load_overlap = 20 * ms - self.pre_load

    def run(self):
        self.ccb.issue(
            "create_applet",
            name="Ion Loading Statistics",
            command="${artiq_applet}plot_xy_hist --y data.loading.load_stats",
            group="Loading",
        )

        try:
            # Turn on the oven and wait
            self.preheat()

            # if self.clear_trap:
            #     self.host_eject_quantum()

            self.dac.send_voltage_lines_to_fpga()
            self.dac.send_shuttling_lookup_table()
            self.dac.apply_line_async("Loading")

            self.kn_initialize_shutters()

            ion = (
                True
            )  # Assume that there is an ion in the trap to start we have to dump.
            ions_loaded = 0  # Counts the number of ions loaded

            while ions_loaded < self.num_stats:

                # Try to lose the ion in the junction
                while True:
                    self.scheduler.pause()

                    ion = self.kn_try_dump(check=True)
                    if ion is False:
                        _LOGGER.info(
                            "Trap is clear, trying to load ion # %i", ions_loaded + 1
                        )
                        break

                load_attempts = 0  # Counts the number of load attempts for a single ion

                while True:
                    self.scheduler.pause()
                    _LOGGER.info("Ion %d, Load attempt %d", ions_loaded, load_attempts)

                    ion = self.kn_try_load()
                    load_attempts += 1
                    if ion is True:
                        self.mutate_dataset(
                            "data.loading.load_stats",
                            ions_loaded,
                            load_attempts * self.load_win,
                        )
                        _LOGGER.info(
                            "Loaded ion after %f seconds", load_attempts * self.load_win
                        )
                        ions_loaded += 1  # Loaded an ion
                        break

                if ions_loaded == 1 and self.just_load_one:
                    _LOGGER.info(
                        "Loaded one ion. Stopping experiment because box is checked in the arguments"
                    )
                    break

        except TerminationRequested:
            _LOGGER.info("Experiment termination requested")

        finally:
            self.kn_idle

            if self.oven_off:
                self.yb_oven.turn_off()

    @kernel
    def kn_initialize_shutters(self):
        """Initialize Shutters"""
        self.core.reset()
        self.core.break_realtime()
        self.oeb.off()
        self.doppler_cooling.init()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.shutter_399.off()
        self.shutter_394.off()

        if self.isotope == 171:
            self.eom_935_172.off()
            self.aom_172_369.off()
            print("Loading 171")
        elif self.isotope == 172:
            self.eom_935_172.on()
            self.aom_172_369.on()
            print("Loading 172")

    @kernel
    def kn_idle(self):
        self.core.break_realtime()
        self.doppler_cooling.idle()
        self.shutter_399.off()
        self.shutter_394.off()
        self.eom_935_172.off()
        self.aom_172_369.off()
        self.aom_399.off()

    @kernel
    def kn_check_ion(self) -> TBool:
        end_window = self.pmt.gate_rising(self.obs_win)
        counts = self.pmt.count(end_window)
        delay(1 * ms)
        if counts > self.thresh:
            ion = True
        else:
            ion = False
        return ion

    @kernel
    def kn_pulse_load_lasers(self):
        self.core.break_realtime()

        if self.isotope == 171:

            self.shutter_394.on()
            delay(self.pre_load)
            self.shutter_399.on()
            delay(self.load_win - self.pre_load)
            self.shutter_394.off()
            delay(self.pre_load)
            self.shutter_399.off()

        elif self.isotope == 172:
            self.ionization_diode_control.set_to_high()
            with parallel:
                self.shutter_394.pulse(self.load_win)
                self.aom_399.pulse(self.load_win)
            self.ionization_diode_control.return_to_zero()

    @kernel
    def kn_try_load(self) -> TBool:
        self.core.break_realtime()
        self.kn_try_dump(check=False)
        self.kn_try_dump(check=False)

        self.kn_pulse_load_lasers()

        delay(self.cool_win)  # Cool Ion

        self.kn_eject_hot_ions()
        delay(1 * ms)

        ion = self.kn_check_ion()
        return ion

    @kernel
    def kn_try_dump(self, check: TBool = True) -> TBool:
        self.core.break_realtime()
        self.kn_clear_load()
        if check is True:
            ion = self.kn_check_ion()
        else:
            ion = False  # assume we dumped the ion if we don't check
        return ion

    @kernel
    def kn_clear_load(self):
        self.sandia_box.hw_dump_load(dump=True)
        delay(25 * ms)
        self.sandia_box.hw_dump_load(dump=False)
        delay(1 * ms)

    @kernel
    def kn_eject_hot_ions(self):
        self.sandia_box.hw_boil_load(boil=True)
        delay(100 * ms)
        self.sandia_box.hw_boil_load(boil=False)

    def preheat(self):
        self.yb_oven.turn_on(current=_OVEN_CURRENT, voltage=1.65, timeout=60 * 30)
        t_wait = 10
        t = 0
        while t < self.turn_on_time:
            print(
                "Heating up oven. {:d} seconds remaining".format(self.turn_on_time - t)
            )
            sleep(t_wait)
            t = t + t_wait
            if self.scheduler.check_pause():
                break


class MixtureRatio(artiq.language.environment.EnvExperiment):
    """Loading.MixtureRatio
    Will try to load using the 399 WM setpoint and shutter, and then record whether it gets 171 or

    Note: this experiment will only work when aligned to the load slot with a PMT
    """

    kernel_invariants = set(
        ["load_win", "obs_win", "thresh", "pmt"]
    )

    def build(self):
        """Get any ARTIQ arguments."""
        self.num_stats = self.get_argument(
            "number of statistics",
            NumberValue(
                default=50.0, unit="load events", scale=1, step=1, min=1, ndecimals=0
            ),
        )

        self.load_win = self.get_argument(
            "load window",
            NumberValue(
                default=50.0 * ms, unit="ms", step=1 * ms, min=1 * ms, ndecimals=3
            ),
        )

        self.obs_win = self.get_argument(
            "observation window",
            NumberValue(
                default=20.0 * ms, unit="ms", step=1 * ms, min=1 * ms, ndecimals=3
            ),
        )

        self.thresh = self.get_argument(
            "ion count threshold",
            NumberValue(default=20, unit="counts", scale=1, step=1, min=1, ndecimals=0),
        )

        self.turn_on_time = self.get_argument(
            "Oven Preheat Time",
            NumberValue(
                default=50, unit="s", step=1, min=1, max=100, ndecimals=0, scale=1
            )
        )
        self.oven_off = self.get_argument(
            "Turn oven OFF after experiment",
            BooleanValue(default=True),
        )

        self.just_load_one = self.get_argument(
            "Load one ion in the Load slot and then stop", BooleanValue(default=False)
        )

        # Get PMT Settings
        self.pmt_array = PMTArray(self)

        # Load core devices
        self.setattr_device("scheduler")
        self.setattr_device("core")
        self.setattr_device("oeb")
        self.setattr_device("ccb")

        # Load other devices
        # self.dac = self.get_device("dac_pc_interface")
        self.sandia_box = SandiaDAC(self)
        self.dac = self.sandia_box.dac_pc
        self.setattr_device("yb_oven")

        # Load Laser Shutters
        self.doppler_cooling = DopplerCooling(self)
        self.setattr_device("shutter_399")
        self.setattr_device("shutter_394")
        self.setattr_device("freq_shift_935")
        self.setattr_device("aom_172_369")

        _LOGGER.debug("Done Building Experiment")

    def prepare(self):
        """Prepare Children"""
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        assert (
            self.pmt_array.num_active == 1
        ), "PMT Array is longer than 1. Please specify which PMT is the load PMT"
        self.pmt = self.pmt_array.counter[0]

        self.set_dataset(
            "data.loading.load_time", np.full(self.num_stats, np.nan), broadcast=True
        )
        self.set_dataset(
            "data.loading.load_isotope", np.full(self.num_stats, np.nan), broadcast=True
        )
        _LOGGER.debug("Done Preparing Experiment")

    def run(self):
        self.ccb.issue(
            "create_applet",
            name="Ion Loading Statistics",
            command="${artiq_applet}plot_xy_hist --y data.loading.load_time",
            group="Loading",
        )

        try:
            # Turn on the oven and wait
            self.preheat()

            self.dac.send_voltage_lines_to_fpga()
            self.dac.send_shuttling_lookup_table()
            self.dac.apply_line_async("Loading")

            self.kn_initialize_shutters()

            #  Assume that there is an ion in the trap to start we have to dump.
            self.trapped_ion = True
            self.isotope = 0

            ions_loaded = 0  # Counts the number of ions loaded

            while ions_loaded < self.num_stats:

                # Try to lose the ion in the junction
                while True:
                    self.scheduler.pause()

                    self.kn_try_dump(check=True)
                    if self.trapped_ion is False:
                        _LOGGER.info(
                            "Trap is clear, trying to load ion # %i", ions_loaded + 1
                        )
                        break

                load_attempts = 0  # Counts the number of load attempts for a single ion

                while True:
                    self.scheduler.pause()
                    _LOGGER.info("Ion %d, Load attempt %d", ions_loaded, load_attempts)

                    self.kn_try_load()
                    load_attempts += 1
                    if self.trapped_ion is True:
                        if self.isotope == 171:
                            self.mutate_dataset("data.loading.load_time", ions_loaded, load_attempts * self.load_win)
                            self.mutate_dataset("data.loading.load_isotope", ions_loaded, 171)
                            _LOGGER.info("Loaded 171 after %f seconds", load_attempts * self.load_win)
                            ions_loaded += 1  # Loaded an ion
                        elif self.isotope == 172:
                            self.mutate_dataset("data.loading.load_time", ions_loaded, load_attempts * self.load_win)
                            self.mutate_dataset("data.loading.load_isotope", ions_loaded, 172)
                            _LOGGER.info("Loaded 172 after %f seconds", load_attempts * self.load_win)
                            ions_loaded += 1  # Loaded an ion
                        else:
                            _LOGGER.info(
                                "Loaded ion after %f seconds, could not distinguish between 171/172",
                                load_attempts * self.load_win)
                            self.mutate_dataset("data.loading.load_time", ions_loaded, load_attempts * self.load_win)
                            self.mutate_dataset("data.loading.load_isotope", ions_loaded, 0)
                            ions_loaded += 1  # Loaded an ion

                        break

                if ions_loaded == 1 and self.just_load_one:
                    _LOGGER.info(
                        "Loaded one ion. Stopping experiment because box is checked in the arguments"
                    )
                    break

        except TerminationRequested:
            _LOGGER.info("Experiment termination requested")

        finally:
            self.kn_idle

            if self.oven_off:
                self.yb_oven.turn_off()

    @kernel
    def kn_initialize_shutters(self):
        """Initialize Shutters"""
        self.core.reset()
        self.core.break_realtime()
        self.oeb.off()
        self.doppler_cooling.init()
        self.doppler_cooling.set_power(0b10)
        self.doppler_cooling.on()
        self.freq_shift_935.on()
        self.aom_172_369.on()
        self.shutter_399.off()
        self.shutter_394.off()

    @kernel
    def kn_idle(self):
        self.core.break_realtime()
        self.doppler_cooling.idle()
        self.shutter_399.off()
        self.shutter_394.off()
        self.eom_935_172.off()
        self.aom_172_369.off()

    @kernel
    def kn_check_ion(self):
        # Check for 171
        self.aom_172_369.off()
        self.doppler_cooling.on()
        delay(1 * ms)
        end_window = self.pmt.gate_rising(self.obs_win)
        counts = self.pmt.count(end_window)
        delay(1 * ms)
        if counts > self.thresh:
            ion171 = True
        else:
            ion171 = False

        # Now check for 172
        delay(1 * ms)
        self.aom_172_369.on()
        self.doppler_cooling.off()
        delay(1 * ms)
        end_window = self.pmt.gate_rising(self.obs_win)
        counts = self.pmt.count(end_window)
        delay(1 * ms)
        if counts > self.thresh:
            ion172 = True
        else:
            ion172 = False

        delay(1 * ms)
        self.doppler_cooling.on()
        if ion171 or ion172:
            self.trapped_ion = True
            if ion171 and ion172:
                self.isotope = 0
            elif ion171:
                self.isotope = 171
            elif ion172:
                self.isotope = 172
        else:
            self.trapped_ion = False
            self.isotope = 0

    @kernel
    def kn_pulse_load_lasers(self):
        with parallel:
            self.shutter_394.pulse(self.load_win)
            self.shutter_399.pulse(self.load_win)

    @kernel
    def kn_try_load(self):
        self.kn_try_dump(check=False)
        self.kn_pulse_load_lasers()
        delay(200 * ms)  # Cool Ion
        self.kn_eject_hot_ions()
        self.kn_check_ion()

    @kernel
    def kn_try_dump(self, check: TBool = True):
        self.core.break_realtime()
        self.kn_clear_load()
        if check is True:
            self.kn_check_ion()

    @kernel
    def kn_clear_load(self):
        self.sandia_box.hw_dump_load(dump=True)
        delay(25 * ms)
        self.sandia_box.hw_dump_load(dump=False)
        delay(1 * ms)

    @kernel
    def kn_eject_hot_ions(self):
        self.sandia_box.hw_boil_load(boil=True)
        delay(25 * ms)
        self.sandia_box.hw_boil_load(boil=False)
        delay(1 * ms)

    def preheat(self):
        self.yb_oven.turn_on(current=_OVEN_CURRENT, voltage=1.5, timeout=60 * 60)
        t_wait = 10
        t = 0
        while t < self.turn_on_time:
            print(
                "Heating up oven. {:d} seconds remaining".format(self.turn_on_time - t)
            )
            sleep(t_wait)
            t = t + t_wait
            if self.scheduler.check_pause():
                break

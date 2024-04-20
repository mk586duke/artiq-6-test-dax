"""Test Ion Autoloader, both by itself and in ARTIQ environment."""
import logging
import time

import artiq.language.environment as artiq_env
import artiq.test.hardware_testbench as artiq_tb
from artiq.language.units import s

import euriqafrontend.experiments.autoloader.ion_autoloader_fsm as autoloader


# import os
# import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_autoloader_run():
    """Check that the autoloader will run properly."""
    loader = autoloader.IonLoader()
    while not loader.is_stopped:
        loader.next_state()
    assert loader.state == "finish"


class ArtiqIonAutoloader(artiq_env.EnvExperiment, autoloader.IonLoader):
    """This is meant to be run as an experiment by artiq_run."""

    def __init__(self):
        """Initialize the Ion Autoloader.

        No arguments.
        """
        super().__init__()
        self.can_advance_prep_to_load = False
        self.start_load_time = None
        self.joined_chain_photon_counts = 0

    def build(self):
        """Build the ion autoloader experiment and sets params."""
        # internal components
        super().__init__()  # init autoloader
        self.logger = logging.getLogger(__name__)

        # ARTIQ components
        self.setattr_argument(
            "num_loops",
            artiq_env.NumberValue(step=1, ndecimals=0, min=1, max=1000, default=5),
        )
        self.setattr_device("core")
        # TODO: Check device names & in device db
        self.setattr_device("cooling_370")  # ionization shutter
        # 370 cooling light
        self.setattr_device("cooling_dds1")  # wrapped dds includes shutter
        self.setattr_device("cooling_dds2")

        # 399 excitation
        self.setattr_device("optpump_dds1")
        self.setattr_device("optpump_dds2")
        # attenuator for optical pumping (399)
        self.setattr_device("optpump_atten")

        self.setattr_device("sandia_dac")  # dac for shuttling ions
        # 370 shutter??

        # todo: test wrapped DDS's (wrap shutter & DDS).
        self.setattr_device("chain_pmt")

    def prepare(self):
        """Ready the Autoloader."""
        try:
            start_state = self.get_state(self.state)
            first_state = self.state_definitions[0]
            if isinstance(first_state, dict):
                first_state = first_state["name"]
            if start_state != first_state:
                raise RuntimeError("FSM not in init state, in {}".format(start_state))
        except RuntimeError as e:
            raise RuntimeError("FSM not initialized properly") from e
        return

    def load(self):
        """Load an ion."""
        self.begin()
        while not self.is_stopped:
            artiq_env.delay(1 * s)
            self.next_state()
        self.logger.debug("\nSuccessfully trapped an ion!!!\n\n")

    def run(self):
        """Run the autoloader experiment."""
        # self.core.reset()
        for _ in range(self.num_loops):
            self.load()

    def _cb_prep_on_enter(self):
        # todo: record current state of all devices to restore later
        self.start_load_time = time.now()
        self.can_advance_prep_to_load = False
        self.shutter_370.open()
        self.shutter_399.open()
        # todo: check syntax
        self.sandia_dac.set_global_adjust(0)
        self.cooling_power.set(self.cooling_power.max_power)

    def _cb_prep_on_timeout(self):
        self.can_advance_prep_to_load = True
        self.next_state()
        logging.info("Timed out of prep. Advanced to %s", self.get_state(self.state))

    def _cb_prep_to_load_condition(self):
        return self.can_advance_prep_to_load

    def _cb_load_on_enter(self):
        self.sandia_dac.apply_voltage("load")

    def _cb_joinchain_on_enter(self):
        self.sandia_dac.shuttle_async("center")

    def _cb_joinchain_to_finish_prepare(self):
        # record counts
        self.chain_pmt.gate_rising(0.5 * s)
        self.joined_chain_photon_counts = self.chain_pmt.count()
        # todo: set chain_pmt device

    def _cb_joinchain_to_finish_condition(self):
        # check counts
        return self.joined_chain_photon_counts > 500
        # note: joinchain_to_load_unless is defined in super()

    def _cb_finish_on_enter(self):
        # todo: reset all devices to old state
        time_to_load = time.now() - self.start_load_time
        print("Took {} seconds to load".format(time_to_load))
        # todo: finish writing


class TestAutoloaderExperiment(artiq_tb.ExperimentCase):
    """Unit test to test the autoloader experiment works properly."""

    def test_autoload(self):
        """Runs the :class:`.ArtiqIonAutoloader` experiment."""
        self.execute(ArtiqIonAutoloader)

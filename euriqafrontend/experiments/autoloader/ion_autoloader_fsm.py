"""Autoloader Finite State Machine. Loads ions into the trap automatically."""
import logging

import euriqabackend.utilities.finite_state_machine as fsm

logger = logging.getLogger(__name__)


class IonLoader(fsm.AutoCallbackFSM):
    """Autoloader Finite State Machine class. Loads ions into the trap automatically."""

    # todo: check timeout functions properly on core device
    state_definitions = [
        {"name": "init", "timeout": 1, "tags": ["pausable"]},
        {"name": "prep", "tags": ["pausable"], "timeout": 2},
        {"name": "load", "timeout": 2, "tags": ["pausable"]},
        "joinchain",
        {"name": "finish", "tags": ["pausable"]},
    ]
    transitions = [
        ["reset", "*", "finish", ["can_pause"]],
        ["end", "*", "finish", ["can_pause"]],
        ["pause", "*", "finish", ["can_pause"]],
        ["begin", "finish", "prep"],
        ["begin", "init", "prep"],
    ]

    def __init__(self):
        """Initialize the state machine.

        All settings defined in class variables, so no arguments here
        """
        super(IonLoader, self).__init__()
        self.add_transitions(self.transitions)
        self.add_transition("next_state", "joinchain", "load")

        self.to_init()  # set starting state of machine
        self.joined_chain_check = 0
        self.can_joinchain = False

    def can_pause(self):
        """Check if you can pause the autoloader from the current state.

        "Pausable" means that you can give control to some other device and then
        seamlessly resume later.
        """
        return self.get_state(self.state).is_pausable

    @property
    def is_stopped(self):
        """Check if the autoloader has finished loading."""
        return self.state == "finish"

    def _cb_init_on_enter(self):
        logger.info("Initializing any devices")

    def _cb_prep_on_enter(self):
        logger.info("Preparing to load ions. Firing lasers, etc!!")

    def _cb_load_on_enter(self):
        logger.info(
            "Turning on ovens, lasers, and waiting for ions to appear! "
            "Setting DACs to load/steady config"
        )
        self.can_joinchain = False

    def _cb_load_on_exit(self):
        logger.info("Exiting load. Put any exit code here")

    def _cb_load_on_timeout(self):
        self.to_joinchain()
        # NOTE: following happens after joinchain_on_enter
        logger.info("Waited too long loading. Advancing to next state: %s", self.state)

    def _cb_load_to_joinchain_prepare(self):
        logger.info("Checking if ion is present and can join chain")
        import random

        self.can_joinchain = random.random() < 0.5

    def _cb_load_to_joinchain_condition(self):
        return self.can_joinchain

    def _cb_joinchain_on_enter(self):
        logger.info("Writing DAC pulses to merge loaded ion with chain")

    def _cb_joinchain_to_finish_prepare(self):
        """Get the data needed to tell if can joinchain.

        Note:
            should have "getter" code in prepare, and then just check the condition in
            condition/unless methods
        """
        logging.info("Checking if ion is present")
        self.joined_chain_check += 1

    def _cb_joinchain_to_finish_condition(self):
        return self.joined_chain_check % 2 == 0

    def _cb_joinchain_to_load_unless(self):
        return self._cb_joinchain_to_finish_condition()

    def _cb_finish_on_enter(self):
        logging.info("Done loading. Returning system to steady state")

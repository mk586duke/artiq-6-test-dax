import logging

import artiq.language.environment as artiq_env
from artiq.language.environment import HasEnvironment
import numpy as np
from artiq.language.types import TBool
from artiq.language.core import delay, delay_mu, host_only, kernel, rpc, now_mu
from artiq.language.units import MHz, ms, us, V
from euriqafrontend.modules.artiq_dac import RampControl_GUI as rampcontrol
from euriqafrontend.modules.artiq_dac import RampControl as rampcontrol_auto

_LOGGER = logging.getLogger(__name__)

class Ramping_Test(artiq_env.EnvExperiment):
    """RFramping.Test"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("rf_lock_switch")
        self.rf_ramp = rampcontrol(self)
        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )
        self.setattr_argument(
            "rf_ramp_modulation_depth",
            artiq_env.NumberValue(default=76, unit='', min=0, max=80),
            group="RF Ramping"
        )
        _LOGGER.debug("Done Building Experiment")

    def prepare(self):
        # Run prepare method of all my imported modules
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()

        _LOGGER.debug("Done Preparing Experiment")

    @kernel
    def run(self):
        # 1
        # prepare is on host, before start kernel session, must have one
        self.core.break_realtime()
        if self.ramp_rf:
            self.rf_ramp.activate_ext_mod(self.rf_ramp_modulation_depth)
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_kernel()
        self.idle()

    @kernel
    def idle(self):
        if self.ramp_rf:
            if self.rf_ramp.ramp_shape == 3 or self.rf_ramp.ramp_shape == 2:
                self.rf_ramp.deactivate_ext_mod()
                # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
                self.rf_lock_control(False)

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

class Ramping_Auto_Test(artiq_env.EnvExperiment):
    """RFramping(Auto).Test"""
    def build(self):
        self.setattr_device("core")
        self.setattr_device("rf_lock_switch")
        self.rf_ramp = rampcontrol_auto(self)
        self.setattr_argument(
            "ramp_rf",
            artiq_env.BooleanValue(default=False),
            group="RF Ramping"
        )
        self.setattr_argument(
            "hold_time",
            artiq_env.NumberValue(default=5000, unit='ms', min=0, max=10000),
        )

        _LOGGER.debug("Done Building Experiment")

    def prepare(self):
        # Run prepare method of all my imported modules
        for child in self.children:
            if hasattr(child, "prepare"):
                child.prepare()
        _LOGGER.debug("Done Preparing Experiment")

    @kernel
    def run(self):
        # 1
        self.core.break_realtime()
        if self.ramp_rf:
            # Ramp down
            self.rf_ramp.activate_ext_mod(self.rf_ramp.ramp_modulation_depth)
            self.core.break_realtime()
            self.rf_lock_control(True)
            self.rf_ramp.run_ramp_down_kernel()

            self.kernel_status_check()

            # Hold
            delay(self.hold_time)

            # Ramp up
            self.rf_ramp.run_ramp_up_kernel()
            self.core.break_realtime()
            self.ramp_idle()

            self.kernel_status_check()


    @kernel
    def ramp_idle(self):
        # Causing problem of unsuccessful ramping
        self.rf_ramp.deactivate_ext_mod()
        self.core.break_realtime()
        # always make sure the dac is at zero before engaging the lock, limited by the low pass filter (1.2 kHz corner freq)
        self.rf_lock_control(False)

    @kernel
    def rf_lock_control(self, state: TBool):
        # state: True, hold, not locking
        # self.core.wait_until_mu(now_mu())
        if not state:
            self.rf_lock_switch.off()
        else:
            self.rf_lock_switch.on()

    @rpc(flags={"async"})
    def kernel_status_check(self):
        # Check the status of the class: RampControl in artiq_dac.py
        print("RampControl Seed:", self.rf_ramp.seed)
        # Check the status of the class: DAC8568 in dac8568.py
        print("DAC8568 Seed:", self.rf_ramp.SandiaSerialDAC.seed)
        # Check the status of the class: RFSignalGenerator in smc100a.py
        print("R&S Seed:", self.rf_ramp.rf_source.seed)

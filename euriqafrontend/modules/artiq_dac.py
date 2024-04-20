import logging
import os
import pathlib

import numpy as np
import time
from artiq.experiment import BooleanValue
from artiq.experiment import NumberValue
from artiq.experiment import TBool
from artiq.experiment import TFloat
from artiq.experiment import TInt32
from artiq.experiment import TInt64
from artiq.experiment import StringValue
from artiq.language.core import kernel, rpc, now_mu, sequential
from artiq.language.environment import HasEnvironment
import artiq.language.environment as artiq_env
from artiq.language.units import us, ms
from artiq.language import delay, delay_mu
import euriqabackend.coredevice.dac8568 as dac8568
from euriqabackend.devices.rohde_schwarz import smc100a

_LOGGER = logging.getLogger(__name__)

class SinglePiezo(HasEnvironment):
    def build(self, **kwargs):
        self.gui_default_value = kwargs["gui_default_value"]
        self.gui_name = kwargs["gui_name"]
        self.device_name = kwargs["device_name"]
        self.channel = kwargs["device_channel"]
        gui_group = kwargs["gui_group"]

        self.value_input = self.get_argument(
            self.gui_name,
            NumberValue(
                default=self.gui_default_value, unit="", min=0.0, max=3.0, ndecimals=2
            ),
            group=gui_group,
        )
        
        # self.SandiaSerialDAC = self.get_device(self.device_name)
        self.SandiaSerialDAC = self.get_device("dac8568_1")

    def prepare(self):
        self.value_mu = dac8568.vout_to_mu(
            self.value_input, self.SandiaSerialDAC.V_OUT_MAX
        )

    def set_value(self, val):
        self.value_mu = dac8568.vout_to_mu(
            val, self.SandiaSerialDAC.V_OUT_MAX
        )

    def get_value(self):
        v_out = dac8568.mu_to_vout(
            self.value_mu, self.SandiaSerialDAC.V_OUT_MAX
        )
        return v_out

    @kernel
    def update_value(self):
        self.SandiaSerialDAC.set_voltage_mu(self.channel, self.value_mu)

    @kernel
    def init(self):
        self.SandiaSerialDAC.reset()
        self.SandiaSerialDAC.init()


class DualPiezo(HasEnvironment):
    _MAX_VAL = 4.99
    _MIN_VAL = 0

    def build(self, **kwargs):
        self.gui_default_value1 = kwargs["gui_default_value1"]
        self.gui_default_value2 = kwargs["gui_default_value2"]
        self.gui_name1 = kwargs["gui_name1"]
        self.gui_name2 = kwargs["gui_name2"]
        self.device_name = kwargs["device_name"]
        self.channel1 = kwargs["device_channel1"]
        self.channel2 = kwargs["device_channel2"]
        gui_group = kwargs["gui_group"]

        self.value_input1 = self.get_argument(
            self.gui_name1,
            NumberValue(
                default=self.gui_default_value1, unit="", min=0.0, max=3.0, ndecimals=2
            ),
            group=gui_group,
        )
        self.value_input2 = self.get_argument(
            self.gui_name2,
            NumberValue(
                default=self.gui_default_value2, unit="", min=0.0, max=3.0, ndecimals=2
            ),
            group=gui_group,
        )

        # self.SandiaSerialDAC = self.get_device(self.device_name)
        self.SandiaSerialDAC = self.get_device("dac8568_1")

        self.feedforward_1_to_2 = 0
        self.feedforward_2_to_1 = 0
        self.feedforward_1_to_2_mu = 0
        self.feedforward_2_to_1_mu = 0
        self.value1_mu = 0
        self.value2_mu = 0

    def prepare(self):
        self._min_val1_mu = dac8568.vout_to_mu(
            self._MIN_VAL, self.SandiaSerialDAC.V_OUT_MAX
        )
        self._zero_val1_mu = dac8568.vout_to_mu(
            self._MAX_VAL / 2, self.SandiaSerialDAC.V_OUT_MAX
        )
        self._max_val1_mu = dac8568.vout_to_mu(
            self._MAX_VAL, self.SandiaSerialDAC.V_OUT_MAX
        )

        self._min_val2_mu = dac8568.vout_to_mu(
            self._MIN_VAL, self.SandiaSerialDAC.V_OUT_MAX
        )
        self._zero_val2_mu = dac8568.vout_to_mu(
            self._MAX_VAL / 2, self.SandiaSerialDAC.V_OUT_MAX
        )
        self._max_val2_mu = dac8568.vout_to_mu(
            self._MAX_VAL, self.SandiaSerialDAC.V_OUT_MAX
        )

        self.feedforward_2_to_1_mu = np.int32(self.feedforward_2_to_1 * 256)
        self.feedforward_1_to_2_mu = np.int32(self.feedforward_1_to_2 * 256)

        self.value1_mu = dac8568.vout_to_mu(
            self.value_input1, self.SandiaSerialDAC.V_OUT_MAX
        )

        self.value2_mu = dac8568.vout_to_mu(
            self.value_input2, self.SandiaSerialDAC.V_OUT_MAX
        )

    # def _vals_to_channels(self, val1, val2):
    #     ret1 = val1 + self.feedforward_2_to_1 * val2
    #     if ret1 > self._MAX_VAL:
    #         ret1 = self._MAX_VAL
    #     if ret1 < self._MIN_VAL:
    #         ret1 = self._MIN_VAL
    #
    #     ret2 = val2 + self.feedforward_1_to_2 * val1
    #     if ret2 > self._MAX_VAL:
    #         ret2 = self._MAX_VAL
    #     if ret2 < self._MIN_VAL:
    #         ret2 = self._MIN_VAL
    #
    #     return ret1, ret2

    @kernel
    def _vals_to_channels_mu(self, val1_mu, val2_mu):
        ret1 = val1_mu + (
            (self.feedforward_2_to_1_mu * (val2_mu - self._zero_val2_mu)) >> 8
        )
        ret2 = val2_mu + (
            (self.feedforward_1_to_2_mu * (val1_mu - self._zero_val2_mu)) >> 8
        )

        if ret1 > self._max_val1_mu:
            ret1 = self._max_val1_mu
        if ret1 < self._min_val1_mu:
            ret1 = self._min_val1_mu

        if ret2 > self._max_val2_mu:
            ret2 = self._max_val2_mu
        if ret2 < self._min_val2_mu:
            ret2 = self._min_val2_mu

        return ret1, ret2

    def set_value(self, val1, val2):
        self.value1_mu = dac8568.vout_to_mu(
            val1, self.SandiaSerialDAC.V_OUT_MAX
        )
        self.value2_mu = dac8568.vout_to_mu(
            val2, self.SandiaSerialDAC.V_OUT_MAX
        )

    @kernel
    def get_value1(self) -> TFloat:
        v_out1 = dac8568.mu_to_vout(
            self.value1_mu, self.SandiaSerialDAC.V_OUT_MAX
        )
        return v_out1

    @kernel
    def get_value2(self) -> TFloat:
        v_out2 = dac8568.mu_to_vout(
            self.value2_mu, self.SandiaSerialDAC.V_OUT_MAX
        )
        return v_out2

    @kernel
    def set_value_mu(self, val1_mu, val2_mu):
        self.value1_mu = val1_mu
        self.value2_mu = val2_mu

    @kernel
    def update_value(self):
        ch1_val_mu, ch2_val_mu = self._vals_to_channels_mu(
            self.value1_mu, self.value2_mu
        )
        self.SandiaSerialDAC.set_voltage_mu(self.channel1, ch1_val_mu)
        delay(20 * us)
        self.SandiaSerialDAC.set_voltage_mu(self.channel2, ch2_val_mu)

    @kernel
    def init(self):
        self.SandiaSerialDAC.reset()
        self.SandiaSerialDAC.init()
        self.update_value()

class RampControl_GUI(HasEnvironment):
    """For parameter testing only, defined with GUI"""
    _MAX_VAL = 4.99
    _MIN_VAL = 0

    def _compute_ramp_volt(self,
                           shape: int = 3,
                           high_voltage: float = _MAX_VAL,
                           low_voltage: float = _MIN_VAL,
                           ramp_time_us: float = 300.0,
                           ramp_step_us: float = 5.0,
                           ramp_constant_time_us: float = 0.0,
                           ):
        dt = ramp_step_us
        delta_voltage = high_voltage - low_voltage
        num_ramp_steps = int(ramp_time_us / dt)
        x = np.linspace(-5, 5, num_ramp_steps)
        num_constant_steps = int(ramp_constant_time_us / dt)
        if shape == 3:
            # Ramp down -> Hold -> Ramp up
            ramp_down = low_voltage + delta_voltage * self._sigmoid(x, 1)
            ramp_constant = self.low_voltage + delta_voltage * self._sigmoid(x, 1)[-1] + np.zeros(int(num_constant_steps))
            ramp_down_temp = np.append(ramp_down,ramp_constant)
            ramp_up = low_voltage + delta_voltage * self._sigmoid(x, -1)
            full_ramp = np.append(ramp_down_temp, ramp_up)
            return full_ramp
        elif shape == 2:
            # Ramp up
            ramp_up = low_voltage + delta_voltage * self._sigmoid(x, -1)
            return ramp_up
        elif shape == 1:
            # Ramp down
            ramp_down = low_voltage + delta_voltage * self._sigmoid(x, 1)
            return ramp_down
        else:
            raise NotImplementedError

    def _volt_to_mu_list(self, list: np.ndarray):
        return np.array([dac8568.vout_to_mu(
            x, self.SandiaSerialDAC.V_OUT_MAX
        ) for x in list])

    def _sigmoid(self,
                 x: np.ndarray,
                 a: float = 1):
        """returns one leg of the ramp. a>0 ramps up and a<0 ramps down.
            x should be in the range of at least -2 to 2 for a smooth ramp"""
        return 1 / (1 + np.exp(-x * a))

    @kernel
    def update_value_mu(self, voltage_mu):
        self.SandiaSerialDAC.set_voltage_mu(self.channel, voltage_mu)

    @kernel
    def run_ramp_kernel(self):
        for voltage_mu in self.ramp_values_mu:
            self.SandiaSerialDAC.set_voltage_mu(self.channel, voltage_mu)
            delay_mu(self.delay_mu)
        # 4 (If w/, no error, but won't physically modulate it)
        # No longer needed after changing ramp shape to be integer
        # self.core.break_realtime()

    @kernel
    def return_to_zero(self):
        self.SandiaSerialDAC.set_voltage_mu(self.channel, self.default_voltage_mu)

    @kernel
    def activate_ext_mod(self, rf_ramp_modulation_depth):
        self.return_to_zero()
        # wait to make sure the dac is set to zero
        delay(2 * ms)
        self.source_activate_amp_mod_ext(rf_ramp_modulation_depth)
        # 2
        self.core.break_realtime()

    @rpc
    def source_activate_amp_mod_ext(self, rf_ramp_modulation_depth):
        self.rf_source.activate_amp_mod_ext(rf_ramp_modulation_depth)

    @kernel
    def deactivate_ext_mod(self):
        self.return_to_zero()
        # wait to make sure the dac is set to zero
        delay(2 * ms)
        self.source_deactivate_amp_mod_ext()
        # 3
        self.core.break_realtime()

    @rpc
    def source_deactivate_amp_mod_ext(self):
        self.rf_source.deactivate_amp_mod_ext()

    def build(self, **kwargs):
        # self.gui_default_value = kwargs["gui_default_value"]
        self.channel = eval("dac8568.AOut.Out5")

        self.default_voltage_mu = dac8568.vout_to_mu(0.0, self._MAX_VAL)

        # basic ARTIQ devices
        self.setattr_device("core")

        self.SandiaSerialDAC = self.get_device("dac8568_1")

        # Trap RF source
        # Actual Setup
        self.rf_source = smc100a.RFSignalGenerator("192.168.78.22")
        # # Test Setup
        # self.rf_source = smc100a.RFSignalGenerator('192.168.80.7')

        self.setattr_argument(
            "low_voltage",
            NumberValue(
                default=0.0, unit="", min=0.0, max=4.99, ndecimals=2
            ),
            group="Artiq DAC",
        )

        self.setattr_argument(
            "high_voltage",
            NumberValue(
                default=1, unit="", min=0.0, max=4.99, ndecimals=2
            ),
            group="Artiq DAC",
        )

        self.setattr_argument(
            "ramp_time_us",
            NumberValue(
                default=1000, unit="", min=150., max=10000., ndecimals=2
            ),
            group="Artiq DAC",
        )

        self.setattr_argument(
            "ramp_step_us",
            NumberValue(
                default=5.0, unit="", min=2.0, max=100., ndecimals=2
            ),
            group="Artiq DAC",
        )

        self.setattr_argument(
            "ramp_constant_time_us",
            NumberValue(
                default=0.0, unit="", min=0.0, max=10000., ndecimals=2
            ),
            group="Artiq DAC",
        )

        self.setattr_argument(
            "ramp_shape",
            NumberValue(
                default=3, unit="", min=1, max=3, step=1, ndecimals=0,
            ),
            group="Artiq DAC"
        )
        # 1: Ramp down; 2: Ramp up; 3: Ramp tri.

    def prepare(self):

        if self.ramp_time_us < 300:
            raise ValueError("Ramp time is smaller than 300 us.")


        ramp_values_volt = self._compute_ramp_volt(shape=self.ramp_shape,
                                                   high_voltage=self.high_voltage,
                                                   low_voltage=self.low_voltage,
                                                   ramp_time_us=self.ramp_time_us,
                                                   ramp_step_us=self.ramp_step_us,
                                                   ramp_constant_time_us=self.ramp_constant_time_us,
                                                   )

        self.total_ramp_time = len(ramp_values_volt)*self.ramp_step_us*us

        self.ramp_values_mu = self._volt_to_mu_list(ramp_values_volt)

        self.delay_mu = self.core.seconds_to_mu(self.ramp_step_us * us)

class RampControl(HasEnvironment):

    _MAX_VAL = 4.99
    _MIN_VAL = 0

    def _compute_ramp_volt(self,
                           shape: int = 2,
                           high_voltage: float = _MAX_VAL,
                           low_voltage: float = _MIN_VAL,
                           ramp_time_us: float = 5000.0,
                           ramp_step_us: float = 100.0):
        dt = self.ramp_step_us
        delta_voltage = self.high_voltage - self.low_voltage
        num_ramp_steps = int(self.ramp_time_us / dt)
        x = np.linspace(-5, 5, num_ramp_steps)
        # num_constant_steps = int(self.ramp_constant_time_us / dt)
        if shape == 2:
            # Ramp up
            ramp_up = low_voltage + delta_voltage * self._sigmoid(x, -1)
            return ramp_up
        elif shape == 1:
            # Ramp down
            ramp_down = low_voltage + delta_voltage * self._sigmoid(x, 1)
            return ramp_down
        else:
            raise NotImplementedError

    def _volt_to_mu_list(self, list: np.ndarray):
        return np.array([dac8568.vout_to_mu(
            x, self.SandiaSerialDAC.V_OUT_MAX
        ) for x in list])

    def _sigmoid(self,
                 x: np.ndarray,
                 a: float = 1):
        """returns one leg of the ramp. a>0 ramps up and a<0 ramps down.
            x should be in the range of at least -2 to 2 for a smooth ramp"""
        return 1 / (1 + np.exp(-x * a))

    @kernel
    def update_value_mu(self, voltage_mu):
        self.SandiaSerialDAC.set_voltage_mu(self.channel, voltage_mu)

    @kernel
    def run_ramp_down_kernel(self):
        # wait until ext mod is fully enabled
        delay(50 * ms)
        for voltage_mu in self.ramp_down_values_mu:
            self.SandiaSerialDAC.set_voltage_mu(self.channel, voltage_mu)
            delay_mu(self.delay_mu)
        # 4 (If w/, no error, but won't physically modulate it)
        # No longer needed after changing ramp shape to be integer
        # self.core.break_realtime()

    @kernel
    def run_ramp_up_kernel(self):
        for voltage_mu in self.ramp_up_values_mu:
            self.SandiaSerialDAC.set_voltage_mu(self.channel, voltage_mu)
            delay_mu(self.delay_mu)
        # 4 (If w/, no error, but won't physically modulate it)
        # No longer needed after changing ramp shape to be integer
        # self.core.break_realtime()

    @kernel
    def return_to_zero(self):
        self.SandiaSerialDAC.set_voltage_mu(self.channel, self.default_voltage_mu)

    @kernel
    def activate_ext_mod(self, rf_ramp_modulation_depth):
        self.return_to_zero()
        # wait to make sure the dac is set to zero
        delay(2 * ms)
        self.source_activate_amp_mod_ext(rf_ramp_modulation_depth)
        # 2
        # self.core.break_realtime()

    @rpc
    def source_activate_amp_mod_ext(self, rf_ramp_modulation_depth):
        self.rf_source.activate_amp_mod_ext(rf_ramp_modulation_depth)

    @kernel
    def deactivate_ext_mod(self):
        self.return_to_zero()
        # wait to make sure the dac is set to zero
        delay(2 * ms)
        # causing problem
        self.core.wait_until_mu(now_mu())
        self.source_deactivate_amp_mod_ext()
        # 3
        # self.core.break_realtime()

    @rpc
    def source_deactivate_amp_mod_ext(self):
        # time.sleep(hold_time)
        # time.sleep(0.02)
        self.rf_source.deactivate_amp_mod_ext()
        # with async flag, subsequent lock control would not be delayed
        # w/, lock would be delayed

    def build(self):
        self.channel = eval("dac8568.AOut.Out5")
        self.default_voltage_mu = dac8568.vout_to_mu(0.0, 4.99)

        # basic ARTIQ devices
        self.setattr_device("core")

        self.SandiaSerialDAC = self.get_device("dac8568_1")

        # Trap RF source
        # Actual Setup
        self.rf_source = smc100a.RFSignalGenerator("192.168.78.22")
        # # Test Setup
        # self.rf_source = smc100a.RFSignalGenerator('192.168.80.7')

        self.ramp_time_us = 5000.0
        self.ramp_step_us = 100.0
        self.high_voltage = 4.99
        self.low_voltage = 0
        self.ramp_modulation_depth = 76
        self.seed = np.random.rand()

    def prepare(self):

        if self.ramp_time_us < 300:
            raise ValueError("Ramp time is smaller than 300 us.")

        # Symmetric ramping up/down
        ramp_down_values_volt = self._compute_ramp_volt(shape=1,
                                                   high_voltage=self.high_voltage,
                                                   low_voltage=self.low_voltage,
                                                   ramp_time_us=self.ramp_time_us,
                                                   ramp_step_us=self.ramp_step_us,
                                                   #ramp_constant_time_us=0,
                                                   )

        ramp_up_values_volt = self._compute_ramp_volt(shape=2,
                                                   high_voltage=self.high_voltage,
                                                   low_voltage=self.low_voltage,
                                                   ramp_time_us=self.ramp_time_us,
                                                   ramp_step_us=self.ramp_step_us,
                                                   #ramp_constant_time_us=0,
                                                   )

        self.total_ramp_time = (len(ramp_down_values_volt)+len(ramp_up_values_volt))*self.ramp_step_us*us

        self.ramp_down_values_mu = self._volt_to_mu_list(ramp_down_values_volt)
        self.ramp_up_values_mu = self._volt_to_mu_list(ramp_up_values_volt)

        self.delay_mu = self.core.seconds_to_mu(self.ramp_step_us * us)

class IonizationDiodeControl(HasEnvironment):
    """Loading.DiodeControl"""

    _MAX_VAL = 4.99
    _MIN_VAL = 0

    @kernel
    def return_to_zero(self):
        self.core.break_realtime()
        self.SandiaSerialDAC.set_voltage_mu(self.channel_LD1, self.zero_mu)
        self.SandiaSerialDAC.set_voltage_mu(self.channel_LD2, self.zero_mu)

    @kernel
    def set_to_high(self):
        self.core.break_realtime()
        self.SandiaSerialDAC.set_voltage_mu(self.channel_LD1, self.LD1_mu)
        self.SandiaSerialDAC.set_voltage_mu(self.channel_LD2, self.LD2_mu)

    def build(self):
        # Converting factor: 20 mA/V; 40 mA (zero voltage)

        self.SandiaSerialDAC = self.get_device("dac8568_1")

        # basic ARTIQ devices
        self.setattr_device("core")

    def prepare(self):

        self.zero_mu = dac8568.vout_to_mu(
            v_out=0, v_out_max=self.SandiaSerialDAC.V_OUT_MAX
        )

        self.LD1_mu = dac8568.vout_to_mu(
            v_out=2.5, v_out_max=self.SandiaSerialDAC.V_OUT_MAX
        )

        self.LD2_mu = dac8568.vout_to_mu(
            v_out=2.5, v_out_max=self.SandiaSerialDAC.V_OUT_MAX
        )

        self.channel_LD1 = eval("dac8568.AOut.Out3")
        self.channel_LD2 = eval("dac8568.AOut.Out4")

import logging
import numpy as np

import artiq.language.environment as artiq_env
from euriqabackend.coredevice.ad9912 import freq_to_mu
import euriqabackend.coredevice.dac8568 as dac8568
from artiq.language.core import (
    delay,
    delay_mu,
    host_only,
    kernel,
    now_mu,
    parallel,
    rpc,
    sequential,
)
from artiq.language.units import MHz, ms, us, V

_LOGGER = logging.getLogger(__name__)


class AddGlobal(artiq_env.EnvExperiment):
    """Utilities.Add_Globals"""

    def build(self):
        """Initialize experiment & variables."""

        # scan arguments
        self.setattr_argument("global_group", artiq_env.StringValue())
        self.setattr_argument("global_name", artiq_env.StringValue())
        self.setattr_argument("value", artiq_env.NumberValue(ndecimals=9))
        self.setattr_argument("broadcast", artiq_env.BooleanValue(default=True))
        self.setattr_argument("persist", artiq_env.BooleanValue(default=True))
        self.setattr_argument("archive", artiq_env.BooleanValue(default=True))

    def prepare(self):
        self.set_dataset(
            self.global_group + "." + self.global_name,
            self.value,
            broadcast=self.broadcast,
            persist=self.persist,
            archive=self.archive,
        )

    def run(self):
        pass


class SetDDS(artiq_env.EnvExperiment):
    """Utilities.SetDDS"""

    def build(self):
        num_dds = 10
        dds_list = ["w_dds" + str(i) for i in range(num_dds)]
        self.setattr_argument("dds_name", artiq_env.EnumerationValue(dds_list))
        self.setattr_argument(
            "freq",
            artiq_env.NumberValue(
                default=200 * MHz, unit="MHz", min=0 * MHz, max=450 * MHz, ndecimals=7
            ),
        )
        self.setattr_argument(
            "amp",
            artiq_env.NumberValue(default=1000, min=0, scale=1, max=1000, ndecimals=0),
        )
        self.setattr_argument("on", artiq_env.BooleanValue(default=True))

        self.dds = self.get_device(self.dds_name)

        # basic ARTIQ devices
        self.setattr_device("core")
        self.setattr_device("oeb")

    def prepare(self):
        self.freq_mu = freq_to_mu(self.freq)
        self.amp = np.int32(self.amp)

    @kernel
    def run(self):
        self.core.reset()
        self.oeb.off()
        self.dds.init()

        if self.on:
            self.dds.on()
        else:
            self.dds.off()

        self.dds.set_mu(bus_group=1, frequency_mu=self.freq_mu, amplitude_mu=self.amp)


class SetDAC(artiq_env.EnvExperiment):
    """Utilities.SetDAC"""

    def build(self):

        dac_list = [i.name for i in dac8568.AOut]

        self.setattr_argument("dac_channel", artiq_env.EnumerationValue(dac_list))
        self.setattr_argument(
            "voltage",
            artiq_env.NumberValue(
                default=1 * V, unit="V", min=0 * V, max=10 * V, ndecimals=5
            ),
        )

        self.SandiaSerialDAC = self.get_device("dac8568_1")

        # basic ARTIQ devices
        self.setattr_device("core")

    def prepare(self):
        _LOGGER.info(
            "Setting voltage on {0} to {1} V".format(self.dac_channel, self.voltage)
        )

        self.voltage_mu = dac8568.vout_to_mu(
            v_out=self.voltage, v_out_max=self.SandiaSerialDAC.V_OUT_MAX
        )

        _LOGGER.info(
            "Setting voltage on {0} to {1} machine units".format(
                self.dac_channel, self.voltage_mu
            )
        )

        self.channel = eval("dac8568.AOut." + self.dac_channel)

        self.voltage_v_from_mu = dac8568.mu_to_vout(
            v_mu=self.voltage_mu, v_out_max=self.SandiaSerialDAC.V_OUT_MAX
        )

        _LOGGER.info(
            "Check conversion back to real units: {0} V".format(self.voltage_v_from_mu)
        )

    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()
        self.SandiaSerialDAC.reset()
        self.SandiaSerialDAC.init()
        self.SandiaSerialDAC.settings_internal_reference(
            static_reference=True, enable=False
        )
        self.SandiaSerialDAC.settings_internal_reference(
            static_reference=True, enable=True
        )
        self.SandiaSerialDAC.set_voltage_mu(self.channel, self.voltage_mu)


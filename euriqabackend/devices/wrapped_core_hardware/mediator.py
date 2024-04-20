"""Wrap basic hardware devices into a larger class, implementing Facade pattern.

TODO:
    * remove DDS "bus_group"
    * test steppedAttenuator
"""
import enum
import logging
import typing
import warnings

from artiq.language import delay
from artiq.language import delay_mu
from artiq.language import kernel
from artiq.language import now_mu
from artiq.language.types import TBool
from artiq.language.types import TInt32
from artiq.language.types import TInt64
from artiq.language.types import TList
from artiq.language.types import TNone
from artiq.language.units import us
from artiq.master.worker_db import DeviceManager

from euriqabackend.coredevice.ad9912 import freq_to_mu
from euriqabackend.coredevice.ad9912 import phase_to_mu

_LOGGER = logging.getLogger(__name__)


class WrappedDDSHardware:
    """DDS, output switch TTL, and reset/bus TTL wrapped."""

    dds_update_time_us = 10 * us  # DDS wait time after programming frequency

    kernel_invariants = {"dds_switch", "dds", "reset", "bus"}

    def __init__(
        self,
        device_manager: DeviceManager,
        dds_device: str,
        dds_switch_device: str,
        dds_reset_device: str,
        bus_group: int = 0,
        coredevice: str = "core",
    ):
        """Initialize the complete DDS board."""
        self.core = device_manager.get(coredevice)
        self.dds = device_manager.get(dds_device)
        self.dds_switch = device_manager.get(dds_switch_device)
        self.reset = device_manager.get(dds_reset_device)
        self.bus = bus_group

    # Convert DDS frequency/phase to values used in DDS hardware
    freq_to_bin = freq_to_mu
    phase_to_bin = phase_to_mu

    @kernel
    def init(self):
        """Initialize the hardware for the DDS."""
        self.reset.off()
        self.dds_switch.off()

    @kernel
    def setup_bus(self, bus_group: TInt32):
        """DEPRECATED. Not needed on ARTIQ 4.

        Set up DDS bus for the DDS's we are actively using.

        Since 0/1, 2/3, etc share a bus, setting up the other DDS will prevent
        changes to these.
        """
        # pylint: disable=unused-argument
        # TODO: delete method
        warnings.warn(
            "setup_bus is not needed for ARTIQ 4 DDS's. Change your code accordingly",
            DeprecationWarning,
        )

    @kernel
    def set_mu(
        self,
        bus_group: TInt32,
        frequency_mu: TInt64 = -1,
        phase_mu: TInt32 = -1,
        amplitude_mu: TInt32 = -1,
        preset: TBool = False,
    ) -> TBool:
        """Set frequency, phase or amplitude of the DDS.

        Requires machine units. These can be pre-calculated with
        :meth:`freq_to_bin` and :meth:`phase_to_mu`. This is done because calculating
        the phase/freq on chip is computationally-expensive, and can lead to
        timing issues. So this enforces best practices.

        `preset` is a flag to determine whether the DDS should be set in past & ready
        at present (`True`), or set at present and increase time cursor.
        Preset allows setting multiple

        Only allows changes if the current output bus group is the same as the
        one this is assigned to. Returns True if was used in correct bus group, else
        returns False.
        """
        if preset and self.bus == bus_group or not preset:
            self.dds.set_mu(
                frequency=frequency_mu,
                phase=phase_mu,
                amplitude=amplitude_mu,
                preset=preset,
            )
            return True
        else:
            # TODO: use assert to fail when used in not correct bus group?
            return False
        # if state is not None:
        #     if state:
        #         self.dds_switch.on()
        #     else:
        #         self.dds_switch.off()

    @kernel
    def load(self):
        self.dds.load()

    @kernel
    def on(self):
        """Turn the DDS output on."""
        self.dds_switch.on()

    @kernel
    def off(self):
        """Turn the DDS output off."""
        self.dds_switch.off()

    @kernel
    def pulse_mu(self, time_mu):
        """Pulse the DDS output for a time (in machine units)."""
        self.dds_switch.pulse_mu(time_mu)

    @kernel
    def pulse(self, time):
        """Pulse the DDS output for a time (in real time units)."""
        self.dds_switch.pulse(time)


class WrappedSteppedAttenuator:
    """Switch TTLs to implement a stepped attenuator."""

    kernel_invariants = {"switch", "look_up_table"}

    def __init__(
        self,
        device_manager: DeviceManager,
        switches: typing.Sequence[str],
        lut: typing.Sequence[int],
    ):
        """Initialize Variable (stepped, digital) attenuator.

        Args:
            device_manager (deviceManager): Device Manager
            switches (list of digital TTL switches): list of switches
            lut (List[int]): (look up table) List of binary values
                (i.e. a switch combination) representing increasing output power

        Raises:
            ValueError: When `lut` is the incorrect size

        """
        self.core = device_manager.get("core")

        self.switch = []
        for switch in switches:
            self.switch.append(device_manager.get(switch))
        if len(lut) != len(switches) ** 2:
            raise ValueError(
                "Input LUT is incorrect size: is {}, "
                "should be {}".format(len(lut), len(switches) ** 2)
            )
        self.look_up_table = lut
        self._state = None

    @kernel
    def set_min(self):
        """Set the output power to minimum."""
        self.set(state=0)

    @kernel
    def set_max(self):
        """Set the output power to maximum."""
        self.set(state=len(self.look_up_table) - 1)

    @kernel
    def set(self, state: TInt32):
        """Set the output to a given state in the lookup table."""
        if state != self._state:
            # NOTE: bin() is probably slow b/c does string conversion. could be faster
            lut_out_as_bin_str = bin(self.look_up_table[state])[
                2:
            ]  # get output as bin string
            for i, c in enumerate(reversed(lut_out_as_bin_str)):
                if c == "0":
                    self.switch[i].off()
                else:
                    self.switch[i].on()
            self._state = state


class WrappedInputArray:
    """Bundle input channels into one class."""

    # TODO: allow setting single counters from this interface
    class _Gating(enum.IntEnum):
        """Internal flags to set input edge sensitivity (what type of edge trigger)."""

        none = 0b00
        rising = 0b01
        falling = 0b10
        both = 0b11

    kernel_invariants = {"_Gating", "num_inputs", "counter"}

    def __init__(
        self,
        device_manager: DeviceManager,
        counters: typing.Sequence[str],
        coredevice: str = "core",
    ):
        """Start the InputArray with specified counter devices."""
        self.core = device_manager.get(coredevice)
        self.num_inputs = len(counters)
        self.counter = [device_manager.get(counter) for counter in counters]
        _LOGGER.debug("Counters: %s", self.counter)

    def __len__(self):
        """Return number of counters in array."""
        return self.num_inputs

    @kernel
    def _set_edge_sensitivity(self, edge_type) -> TNone:
        # if channel is None:
        for counter in self.counter:
            counter._set_sensitivity(edge_type)  # pylint: disable=protected-access
        # else:
        #     self.counter[channel]._set_sensitivity(edge_type)

    @kernel
    def gate_both(self, duration):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.gate_both`."""
        self._set_edge_sensitivity(self._Gating.both)
        delay(duration)
        self._set_edge_sensitivity(self._Gating.none)
        return now_mu()

    @kernel
    def gate_both_mu(self, duration):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.gate_both_mu`."""
        self._set_edge_sensitivity(self._Gating.both)
        delay_mu(duration)
        self._set_edge_sensitivity(self._Gating.none)
        return now_mu()

    @kernel
    def gate_falling(self, duration):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.gate_falling`."""
        self._set_edge_sensitivity(self._Gating.falling)
        delay(duration)
        self._set_edge_sensitivity(self._Gating.none)
        return now_mu()

    @kernel
    def gate_falling_mu(self, duration):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.gate_falling_mu`."""
        self._set_edge_sensitivity(self._Gating.falling)
        delay_mu(duration)
        self._set_edge_sensitivity(self._Gating.none)
        return now_mu()

    @kernel
    def gate_rising(self, duration):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.gate_rising`."""
        self._set_edge_sensitivity(self._Gating.rising)
        delay(duration)
        self._set_edge_sensitivity(self._Gating.none)
        return now_mu()

    @kernel
    def gate_rising_mu(self, duration):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.gate_rising_mu`."""
        self._set_edge_sensitivity(self._Gating.rising)
        delay_mu(duration)
        self._set_edge_sensitivity(self._Gating.none)
        return now_mu()

    @kernel
    def timestamp_mu(self, buffer: TList(TInt64)) -> TList(TInt64):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.timestamp_mu`.

        Buffer must be pre-allocated to the correct size
        """
        assert len(buffer) == self.num_inputs
        for i in range(self.num_inputs):
            buffer[i] = self.counter[i].timestamp_mu()
        return buffer

    @kernel
    def count(self, up_to_time_mu, buffer: TList(TInt32)) -> TList(TInt32):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.count`."""
        assert len(buffer) == self.num_inputs
        for i in range(self.num_inputs):
            buffer[i] = self.counter[i].count(up_to_time_mu)
        return buffer

    @kernel
    def single_timestamp_mu(self, input_ind: TInt32) -> TInt64:
        """See :meth:`artiq.coredevice.ttl.TTLInOut.timestamp_mu`.

        Timestamps a single input.
        """
        return self.counter[input_ind].timestamp_mu()

    @kernel
    def single_count(self, input_ind: TInt32, up_to_time_mu) -> TList(TInt64):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.count`.

        Counts a single input.
        """
        return self.counter[input_ind].count(up_to_time_mu)

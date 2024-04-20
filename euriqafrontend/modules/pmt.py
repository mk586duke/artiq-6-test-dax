import enum
import logging

import numpy as np
from artiq.coredevice import RTIOOverflow
from artiq.experiment import StringValue
from artiq.language import delay
from artiq.language import delay_mu
from artiq.language import kernel
from artiq.language import now_mu
from artiq.language.environment import HasEnvironment
from artiq.language.types import TBool
from artiq.language.types import TInt32
from artiq.language.types import TInt64
from artiq.language.types import TList
from artiq.language.types import TNone
from artiq.language.units import ns
from artiq.language.units import us
from artiq.master.worker_db import DeviceError

_LOGGER = logging.getLogger(__name__)


class PMTArray(HasEnvironment):
    """Bundle input channels into one class."""

    # TODO: allow setting single counters from this interface
    class _Gating(enum.IntEnum):
        """Internal flags to set input edge sensitivity (what type of edge trigger)."""

        none = 0b00
        rising = 0b01
        falling = 0b10
        both = 0b11

    kernel_invariants = {"_Gating", "counter", "num_active"}

    def build(self):
        self.pmt_input_s = self.get_argument(
            "PMT Input String",
            StringValue(default="-3:19"),
            tooltip = "Comma separated integers corresponding to pmtXX. "
                      "Range symbol (:) is inclusive on both ends - (e.g., '1, 4, 6:9' -> [1,4,6,7,8,9])",
            group="PMT",
        )
        self.setattr_device("core")
        _LOGGER.debug("Done building PMTs")

    def prepare(self):

        self.active_pmts = self.parse_pmt_input(self.pmt_input_s)
        self.counter_names = []
        self.counter = []
        # on 5/7/21, discoverd that the PMT array is flipped about its middle
        # relative to the AWG slot assignment and the x-coordinate of the system
        # fixed by remapping the requested PMTs
        new_active_pmts = [8-(i-8) for i in self.active_pmts]# PMT array was flipped, redo mapping
        for i_pmt in new_active_pmts:
            if i_pmt < 0:
                self.counter_names.append("pmt_{0}".format(abs(i_pmt)))
            else:
                self.counter_names.append("pmt{0}".format(i_pmt))

            try:
                self.counter.append(self.get_device(self.counter_names[-1]))
            except DeviceError as error:
                _LOGGER.error(error)

        self.num_active = len(self.counter)
        _LOGGER.debug("Done preparing PMTs")


    @staticmethod
    def parse_pmt_input(s: str) -> list :
        """
        Args:
            s (str): A Comma seperated string of inters. Python notation i:j can be used to indicate ranges

        Returns:
            A list of integers generated from the input string

            Example:
                >>parse_pmt_input("-9,-5,0:4")
                [-9, -5, 0, 1, 2, 3, 4]
        """
        nums = []
        for x in map(str.strip, s.split(",")):
            try:
                i = int(x)
                nums.append(i)
            except ValueError:
                if ":" in x:
                    xr = list(map(str.strip, x.split(":")))
                    nums.extend(range(int(xr[0]), int(xr[1]) + 1))
                else:
                    _LOGGER.warning("Unknown string format for PMT input: {0}".format(x))
        return nums

    @kernel
    def _set_edge_sensitivity(self, edge_type) -> TNone:
        # if channel is None:
        for counter in self.counter:
            counter._set_sensitivity(edge_type)  # pylint: disable=protected-access
            # Delay one coarse clock cycle so that we only fill one
            # ARTIQ RTIO Output lane
            delay(8 * ns)
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
        # assert len(buffer) == self.num_active
        for i in range(self.num_active):
            buffer[i] = self.counter[i].timestamp_mu()
        return buffer

    @kernel
    def count(self, up_to_time_mu, buffer: TList(TInt32)) -> TList(TInt32):
        """See :meth:`artiq.coredevice.ttl.TTLInOut.count`."""
        # assert len(buffer) == self.num_active
        for i in range(self.num_active):
            buffer[i] = self.counter[i].count(up_to_time_mu)
            delay(8 * ns)
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

    @kernel
    def clear_buffer(self) -> TBool:
        """
        Empties out the buffer to avoid overflows during the experiment
        """
        self.core.break_realtime()
        empty = False
        while empty is False:
            try:
                stopcounter_mu = now_mu()
                buffer = [0] * self.num_active
                delay(500 * us)
                for i in range(self.num_active):
                    buffer[i] = self.counter[i].count(stopcounter_mu)
                sum = 0
                for i in range(len(buffer)):
                    sum = sum + buffer[i]
                if sum == 0:
                    empty = True

            except RTIOOverflow:
                empty = False

        return empty

class PMTEdgeCounter(HasEnvironment):
    # For getting PMT counts without time stamp using edgecounter

    kernel_invariants = {"counter", "num_active"}

    def build(self):
        self.pmt_input_s = self.get_argument(
            "PMT Input String",
            StringValue(default="-3:19"),
            tooltip = "Comma separated integers corresponding to pmtXX. "
                      "Range symbol (:) is inclusive on both ends - (e.g., '1, 4, 6:9' -> [1,4,6,7,8,9])",
            group="PMT",
        )
        self.setattr_device("core")
        _LOGGER.debug("Done building PMTs")

    def prepare(self):

        self.active_pmts = self.parse_pmt_input(self.pmt_input_s)
        self.counter_names = []
        self.counter = []
        # on 5/7/21, discoverd that the PMT array is flipped about its middle
        # relative to the AWG slot assignment and the x-coordinate of the system
        # fixed by remapping the requested PMTs
        new_active_pmts = [8-(i-8) for i in self.active_pmts]# PMT array was flipped, redo mapping
        for i_pmt in new_active_pmts:
            if i_pmt < 0:
                self.counter_names.append("pmt_{0}EdgeCounter".format(abs(i_pmt)))
            else:
                self.counter_names.append("pmt{0}EdgeCounter".format(i_pmt))

            try:
                self.counter.append(self.get_device(self.counter_names[-1]))
            except DeviceError as error:
                _LOGGER.error(error)

        self.num_active = len(self.counter)
        _LOGGER.debug("Done preparing PMTs")

    @staticmethod
    def parse_pmt_input(s: str) -> list :
        """
        Args:
            s (str): A Comma seperated string of inters. Python notation i:j can be used to indicate ranges

        Returns:
            A list of integers generated from the input string

            Example:
                >>parse_pmt_input("-9,-5,0:4")
                [-9, -5, 0, 1, 2, 3, 4]
        """
        nums = []
        for x in map(str.strip, s.split(",")):
            try:
                i = int(x)
                nums.append(i)
            except ValueError:
                if ":" in x:
                    xr = list(map(str.strip, x.split(":")))
                    nums.extend(range(int(xr[0]), int(xr[1]) + 1))
                else:
                    _LOGGER.warning("Unknown string format for PMT input: {0}".format(x))
        return nums

    @kernel
    def gate_falling(self, duration):
        """See :meth:`artiq.coredevice.edge_counter.EdgeCounter.gate_falling`."""
        for counter in self.counter:
            counter.gate_falling_mu(self.core.seconds_to_mu(duration))
            # Delay one coarse clock cycle so that we only fill one ARTIQ RTIO Output lane
            delay(8 * ns)

        return now_mu()

    @kernel
    def gate_falling_mu(self, duration):
        """See :meth:`artiq.coredevice.edge_counter.EdgeCounter.gate_falling`."""
        for counter in self.counter:
            counter.gate_falling_mu(duration)
            # Delay one coarse clock cycle so that we only fill one ARTIQ RTIO Output lane
            delay(8 * ns)

        return now_mu()

    @kernel
    def gate_rising(self, duration):
        """See :meth:`artiq.coredevice.edge_counter.EdgeCounter.gate_rising`."""
        for counter in self.counter:
            counter.gate_rising_mu(self.core.seconds_to_mu(duration))
            # Delay one coarse clock cycle so that we only fill one ARTIQ RTIO Output lane
            delay(8 * ns)

        return now_mu()

    @kernel
    def gate_rising_mu(self, duration):
        """See :meth:`artiq.coredevice.edge_counter.EdgeCounter.gate_rising`."""
        for counter in self.counter:
            counter.gate_rising_mu(duration)
            # Delay one coarse clock cycle so that we only fill one ARTIQ RTIO Output lane
            delay(8 * ns)

        return now_mu()

    @kernel
    def count(self, up_to_time_mu, buffer: TList(TInt32)) -> TList(TInt32):
        """See :meth:`artiq.coredevice.edge_counter.EdgeCounter.set_config` and `artiq.coredevice.edge_counter.EdgeCounter.fetch_count`."""
        for i in range(self.num_active):
            buffer[i] = self.counter[i].fetch_count()
            delay(8 * ns)
        return buffer

    @kernel
    def single_count(self, input_ind: TInt32, up_to_time_mu) -> TList(TInt64):
        """See :meth:`artiq.coredevice.edge_counter.EdgeCounter.set_config` and `artiq.coredevice.edge_counter.EdgeCounter.fetch_count`."""
        return self.counter[input_ind].fetch_count()

    @kernel
    def clear_buffer(self) -> TBool:
        """
        Empties out the buffer to avoid overflows during the experiment
        """
        self.core.break_realtime()
        empty = False
        buffer = [0] * self.num_active
        # while empty is False:
        #     try:
        #         stopcounter_mu = now_mu()
        #         buffer = [0] * self.num_active
        #         delay(500 * us)
        #         buffer = self.count(stopcounter_mu, buffer)
        #         sum = 0
        #         for i in range(len(buffer)):
        #             sum = sum + buffer[i]
        #         if sum == 0:
        #             empty = True

        #     except RTIOOverflow:
        #         empty = False

        return empty

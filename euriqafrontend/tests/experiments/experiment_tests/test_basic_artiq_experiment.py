"""Basic experiment to test ARTIQ connectivity & experiments."""
import logging

import artiq.language.environment as artiq_env
import pytest
from artiq.language.units import s

_LOGGER = logging.getLogger(__name__)


class BasicInOutExperiment(artiq_env.EnvExperiment):
    """Basic Experiment for testing ARTIQ GPIO/TTL with Photomultiplier Tubes (PMTs).

    Should feed > 200 Hz clock into pmt_device (e.g. "in1_1")
    """

    count_rate_arr = list()
    pmt_device_name = "in1_1"
    output_device_names = [
        "out2_7",
        "out2_5",
        "out2_3",
        "out2_1",
        "out1_0",
        "out2_6",
        "out2_4",
        "out2_2",
        "out2_0",
    ]
    loops = 10

    def build(self):
        """Get devices for experiment."""
        self.setattr_device("core")
        self.setattr_device(self.pmt_device_name)
        for dev in self.output_device_names:
            self.setattr_device(dev)

    def prepare(self):
        """Prepare devices to run experiment."""
        getattr(self, self.pmt_device_name).input()
        # delay(0.1 * s)

    def run(self):
        """Loop over counting PMT counts."""
        # self.core.reset()
        # self.core.break_realtime()
        observe_time = 0.5 * s

        for i in range(self.loops):
            getattr(self, self.pmt_device_name).gate_rising(observe_time)
            self.count_rate_arr.append(
                getattr(self, self.pmt_device_name).count() / observe_time
            )
            _LOGGER.debug("Recorded count rate: %i", self.count_rate_arr[i])

            # for dev in self.output_devices:
            #     getattr(self, dev).pulse(0.5 * s)
            #     getattr(self, dev).sync()

            getattr(self, self.pmt_device_name).sync()

    def analyze(self):
        """Check results to see if test worked and can pass."""
        assert len(self.count_rate_arr) == self.loops
        for rate in self.count_rate_arr:
            assert rate is not None
            assert rate > 100


@pytest.mark.hardware
def test_basic_artiq_experiment(artiq_experiment_run):
    """Run a basic experiment to test ARTIQ input/output capability."""
    artiq_experiment_run(BasicInOutExperiment)

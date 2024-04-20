"""Test the input channels."""
import itertools

from artiq.language.core import at_mu
from artiq.language.core import kernel
from artiq.language.core import now_mu
from artiq.language.environment import EnvExperiment
from artiq.language.units import ms

# import more_itertools


class TestInputs(EnvExperiment):
    """Test input channels.

    Hook channels to a function generator or similar to generate inputs.
    """

    kernel_invariants = set(["inputs", "counting_time"])

    def build(self):
        """Use input devices."""
        self.setattr_device("core")
        self.setattr_device("oeb")

        input_names = [
            "{}_{}".format(bank, i)
            for bank, i in itertools.product(["in1", "in2", "in3"], range(8))
        ]
        self.inputs = [self.get_device(name) for name in input_names]

        self.counting_time = 1 * ms

    @kernel
    def run(self):
        """Record PMT/input counts."""
        self.core.reset()
        self.oeb.off()

        counts = [0] * len(self.inputs)
        stop_times = [0] * len(self.inputs)

        start_time = now_mu()
        # record counts (NOTE: parallel)
        for i in range(len(self.inputs)):
            at_mu(start_time)  # reset time, with parallel doesn't work
            stop_times[i] = self.inputs[i].gate_rising(self.counting_time)

        # check that all recordings run in parallel
        assert now_mu() == start_time + self.core.seconds_to_mu(self.counting_time)
        for t in stop_times:
            assert t == stop_times[0]

        # count number of edges received. not time-constrained.
        for i in range(len(self.inputs)):
            counts[i] = self.inputs[i].count(stop_times[i])

        # print edges
        for i in range(len(counts) >> 3):
            for j in range(8):
                c = counts[i * 8 + j]
                if c > 0:
                    print("IN", i + 1, "-", j, ":", c)


class TestWrappedInputs(EnvExperiment):
    """Test wrapped input channels (multiple PMT's grouped into single device).

    Hook channels to a function generator or similar to generate inputs.
    """

    kernel_invariants = set(["inputs", "detect_time"])

    def build(self):
        """Use input devices."""
        self.setattr_device("core")
        self.setattr_device("oeb")

        self.inputs = self.get_device("pmt_array")

        self.detect_time = 100 * ms

    @kernel
    def run(self):
        """Record PMT/input counts."""
        self.core.reset()
        self.oeb.off()

        num_inputs = self.inputs.num_inputs

        # pre-allocate buffers
        counts = [0] * num_inputs

        start_time = now_mu()
        # record counts (NOTE: parallel)
        stop_time = self.inputs.gate_rising(self.detect_time)

        # check that all recordings run in parallel
        assert now_mu() == start_time + self.core.seconds_to_mu(self.detect_time)

        # count number of edges received. not time-constrained.
        counts = self.inputs.count(stop_time, buffer=counts)

        # print number of edges/counts
        for i in range(len(counts) >> 3):
            for j in range(8):
                c = counts[i * 8 + j]
                if c > 0:
                    print("IN", i + 1, "-", j, ":", c)

        print("Done")


# class TestInputsLoopback(EnvExperiment):
#     """Test input channels by looping back from TTLOutputs.

#     Outputs pulses on the TTL Output lines, and uses those as inputs.
#     """

#     kernel_invariants = set(["inputs", "outputs", "counting_time"])

#     def build(self):
#         self.setattr_device("core")
#         self.setattr_device("oeb")

#         output_names = [
#             "{}_{}".format(bank, i)
#             for bank, i in itertools.product(
#                 ["out1", "out2", "out3", "out4"], range(8)
#             )
#         ]
#         self.outputs = [self.get_device(name) for name in output_names]

#         input_names = [
#             "{}_{}".format(bank, i)
#             for bank, i in itertools.product(["in1", "in2", "in3"], range(8))
#         ]
#         self.inputs = [self.get_device(name) for name in input_names]

#         self.output_chunks = more_itertools.chunked(self.outputs, 4)

#         self.counting_time = 1 * ms

#     @kernel
#     def run(self):
#         self.core.reset()
#         self.oeb.off()

#         counts = [0] * len(self.inputs)
#         # output pulses, and read in.
#         with parallel:
#             # output pulses on all channels, but in chunks b/c limited outputs
#             for chunk in self.output_chunks:
#                 start_time = now_mu()
#                 for output in chunk:
#                     at_mu(start_time)
#                     output.pulse(self.counting_time/ len(self.output_chunks))
#             start_count_time = now_mu()
#             for i in range(len(self.inputs)):
#                 at_mu(start_time)
#                 counts[i] = self.inputs[i].count(self.inputs[i].gate_rising(self.counting_time))

#         # print results
#         for i in range(len(counts)):
#             for j in range(8):
#                 print("IN{}-{}: {}".format(i, j, counts[i * 8 + j]))

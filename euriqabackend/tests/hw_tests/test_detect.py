import random
import time

import numpy as np
from artiq.experiment import *


def input_state() -> TBool:
    return input("Enter desired TTL state: ") == "1"


def oeb_state() -> TBool:
    return input("Output on?: ") == "1"


class TestDetect(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        # self.setattr_device("led0")
        self.setattr_device("oeb")
        self.setattr_device("out1_0")
        self.setattr_device("io_update0")
        self.setattr_device("reset01")

        self.qubit_count = 8  # qubits under test
        self.n_loops = 1  # number of times to loop detection sequence, for statistics
        self.t_mu = [np.int64(0)] * self.n_loops  # list of detection times in mu units
        self.loop_idx = 0  # PMT measurement index
        # long list of random "measurements" that handles n_loops loops. Max 3 * qubit_count measurements
        PMT_max = 100
        # self.PMT_count_idx = 0  # index through the list of measurements
        self.threshold = (
            PMT_max // 2
        )  # state detection threshold. DO NOT LEAVE AS A FLOAT

        self.PMT_A = [
            [random.randint(0, PMT_max) for _ in range(self.qubit_count)]
            for _ in range(self.n_loops)
        ]
        self.PMT_B = self.PMT_A.copy()
        self.PMT_B = [
            [random.randint(0, PMT_max) for _ in range(self.qubit_count)]
            for _ in range(self.n_loops)
        ]
        self.PMT_C = [
            [random.randint(0, PMT_max) for _ in range(self.qubit_count)]
            for _ in range(self.n_loops)
        ]
        self.C_idx = 0

        # self.PMT_counts = [random.randint(0,PMT_max) for _ in range(3*self.n_loops*self.qubit_count)]
        # self.PMT_counts = PMT_A + PMT_B + PMT_C

        # self.LUT_AB = []
        # self.LUT_C = []
        # for idx in range(1, 2**self.qubit_count + 1):
        #     for _ in range(idx):
        #         self.LUT_AB[idx].append()

    @kernel
    def mask(self, count, threshold):
        if count >= threshold:
            return 1
        return 0

    @kernel
    def run_step1(self):
        # initialize all 3 PMTs even if C not used -> can we append to lists?
        DET_A = [0] * self.qubit_count
        DET_B = [0] * self.qubit_count
        DET_C = [0] * self.qubit_count

        delay(50 * us)

        # start realtime timed loop
        t0 = self.core.get_rtio_counter_mu()
        for qb in range(self.qubit_count):
            DET_A[qb] = self.threshold()
        for qb in range(self.qubit_count):
            DET_B[qb] = self.threshold()

        if DET_A == DET_B:
            self.out1_0.pulse(1 * us)
            delay(1 * us)
        else:
            for qb in range(self.qubit_count):
                DET_C[qb] = self.threshold()
            self.out1_0.pulse(10 * us)
            delay(10 * us)
        t1 = self.core.get_rtio_counter_mu()
        # end realtime timed loop

        self.t_mu[self.loop_idx] = t1 - t0
        self.loop_idx += 1

    @kernel
    def detect(self):
        DET_A, DET_B, DET_C = 0, 0, 0

        delay(50 * us)

        # start realtime timed loop
        t0 = self.core.get_rtio_counter_mu()
        for idx in range(self.qubit_count):
            # roll function bit-shifts and keeps current binary value, tacks on detect value as LSB
            DET_A = (DET_A << 1) | self.mask(
                self.PMT_A[self.loop_idx][idx], self.threshold
            )
        for idx in range(self.qubit_count):
            DET_B = (DET_B << 1) | self.mask(
                self.PMT_B[self.loop_idx][idx], self.threshold
            )

        if DET_A == DET_B:
            # self.out1_0.pulse(1*us)
            # delay(1*us)
            pass
        else:
            for idx in range(self.qubit_count):
                DET_C = (DET_C << 1) | self.mask(
                    self.PMT_C[self.loop_idx][idx], self.threshold
                )
            self.C_idx += 1
            # self.out1_0.pulse(1*us)
            # delay(10*us)
            pass
        t1 = self.core.get_rtio_counter_mu()
        # end realtime timed loop

        self.t_mu[self.loop_idx] = t1 - t0
        self.loop_idx += 1

    @kernel
    def run_exp(self):
        self.core.break_realtime()
        for loop in range(self.n_loops):
            # self.core.break_realtime()
            self.core.reset()
            self.detect()

    def run(self):
        self.core.reset()
        self.run_exp()

        t_sec_list_us = [
            mu_to_seconds(self.t_mu[i], self.core) * 10 ** 6
            for i in range(len(self.t_mu))
        ]
        t_sec_us = np.sum(t_sec_list_us)
        t_sec_std_us = np.std(t_sec_list_us)

        print("")
        print("Done")
        print("Real time taken for {} runs: {:.3f} us".format(self.n_loops, t_sec_us))
        print("Time per run: {:.3f} us".format(t_sec_us / self.n_loops))
        print("Standard devation per run: {} us".format(t_sec_std_us))
        print(
            "Total number of C detect sequences: {} out of {}".format(
                self.C_idx, self.n_loops
            )
        )
        print(
            "Detection sequences per run: {}".format(
                self.qubit_count * (2 + self.C_idx / self.n_loops)
            )
        )

        tmp = [
            "{:.3f} us".format(t_sec_list_us[i])
            for i in range(len(t_sec_list_us) - 10, len(t_sec_list_us))
        ]
        print("Last 10 timed steps to check for buffer overflow:")
        print(tmp)
        print("")

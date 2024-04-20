"""Quick-and-dirty experiment for testing RFSoC hardware triggering of large numbers
of shots/scan points.

We were seeing that certain pulse sequences were ending earlier than they were supposed
to (too many triggers per schedule, roughly), and wanted to test it in the simplest
possible case.

This could probably be removed in the future, just for testing purposes.
"""
import itertools
import typing

import qiskit.pulse as qp
import functools
from artiq.experiment import (
    BooleanValue,
    delay,
    kernel,
    EnvExperiment,
    now_mu,
    NumberValue,
    rpc,
    ns,
    us,
)

import euriqafrontend.modules.rfsoc as rfsoc

IntegerValue = functools.partial(NumberValue, scale=1, step=1, ndecimals=0)


class TriggerTest(EnvExperiment):
    kernel_invariants = {
        "trigger_duration",
        "schedule_duration",
        "delay_between_schedules",
        "num_schedules" "num_shots",
    }

    def build(self):
        self.setattr_device("core")
        self.rfsoc = rfsoc.RFSOC(self)

        self.num_schedules = self.get_argument(
            "number_of_test_schedules", IntegerValue(default=1000, min=1)
        )
        self.num_iters = self.get_argument(
            "number of shots per schedule", IntegerValue(100)
        )
        self.schedule_duration = self.get_argument(
            "duration of each schedule",
            NumberValue(default=50 * us, unit="us", min=1 * us, max=100 * us),
        )
        self.trigger_duration = self.get_argument(
            "rfsoc trigger duration",
            NumberValue(default=100 * ns, min=10 * ns, unit="ns"),
        )
        self.pulse_amp = self.get_argument(
            "rfsoc pulse amplitude", rfsoc.AmplitudeArg()
        )
        self.channel_index = self.get_argument(
            "qiskit channel index", IntegerValue(6, min=0, max=13)
        )
        self.channel_is_drive = self.get_argument(
            "drive channel? (false = Control)", BooleanValue(True)
        )
        self.wait_for_input = self.get_argument(
            "wait for input between schedule?", BooleanValue(True)
        )
        self.use_streaming_upload = self.get_argument(
            "streaming upload? (false = LUT mode)", BooleanValue(False)
        )
        # self.delay_between_schedules = self.get_argument(
        #     "delay between schedule outputs",
        #     NumberValue(100 * us, unit="us", min=10 * us),
        # )

    def custom_pulse_schedule_list(self) -> typing.List[qp.Schedule]:
        schedules = []
        if self.channel_is_drive:
            output_channel = qp.DriveChannel(int(self.channel_index))
        else:
            output_channel = qp.ControlChannel(int(self.channel_index))
        with qp.build(self.rfsoc._qiskit_backend):
            pulse_types = (
                qp.Constant,
                # functools.partial(
                #     qp.Gaussian,
                #     sigma=qp.seconds_to_samples(self.schedule_duration) / (2 * 50)
                # ),
                # functools.partial(
                #     qp.GaussianSquare,
                #     width=qp.seconds_to_samples(self.schedule_duration) / 2,
                #     sigma=self.schedule_duration / (2 * 50),
                # ),
            )
        pulse_iter = itertools.cycle(pulse_types)
        for i in range(self.num_schedules):
            with qp.build(self.rfsoc._qiskit_backend) as new_schedule:
                qp.play(
                    next(pulse_iter)(
                        qp.seconds_to_samples(self.schedule_duration),
                        amp=self.pulse_amp,
                    ),
                    output_channel,
                )

            schedules.append(new_schedule)

        print(schedules[:2])
        return schedules

    def prepare(self):
        self.call_child_method("prepare")

    def run(self):
        self.rfsoc.upload_data(
            num_shots=self.num_iters, streaming=self.use_streaming_upload
        )
        run_duration = self.run_schedules()
        print(
            f"\nDone outputting schedules. Took {run_duration / 1e9} s ({run_duration} mu)"
        )

    @rpc
    def wait_input(self, i):
        print(f"About to start schedule {i + 1}/{self.num_schedules}")
        input("ENTER TO CONTINUE")

    @kernel
    def run_schedules(self):
        self.core.reset()
        self.core.break_realtime()
        self.rfsoc.rfsoc_trigger.off()
        self.core.break_realtime()

        start_time = now_mu()
        for i in range(self.num_schedules):
            print("%i/%i", i + 1, self.num_schedules)
            if self.wait_for_input:
                self.wait_input(i)
            for j in range(self.num_iters):
                self.core.break_realtime()
                self.rfsoc.rfsoc_trigger.pulse(self.trigger_duration)
                delay(self.schedule_duration + 1 * us)
                # delay(self.delay_between_schedules)

        stop_time = now_mu()

        return stop_time - start_time

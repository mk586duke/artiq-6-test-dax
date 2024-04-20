import artiq
from artiq.language import NumberValue
from artiq.language.core import delay
from artiq.language.core import kernel
from artiq.language.types import TInt32
from artiq.language.units import ms
from artiq.language.units import s


class RPC_test(artiq.language.environment.EnvExperiment):
    def build(self):
        self.loops = self.get_argument(
            "loops",
            NumberValue(
                default=10.0, unit="loops", scale=1, step=1, min=1, ndecimals=0
            ),
        )
        self.setattr_device("core")
        self.ttl_1 = self.get_device("shutter_370ion")

    def increment(self, i) -> TInt32:
        i += 1
        print(i)
        return int(i)

    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()
        x = int(0)
        for i in range(self.loops):
            self.ttl_1.pulse(500 * ms)
            x = self.increment(x)
            delay(1.0 * s)

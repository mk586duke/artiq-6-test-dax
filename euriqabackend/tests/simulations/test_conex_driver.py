import time

import numpy as np
from artiq.experiment import *


class _HarrisTester(HasEnvironment):
    def argumentToCommand(self, prestr, command, paramvec):
        str = "{}.{}(".format(prestr, command)
        paramvec = np.array(paramvec)[np.array(paramvec) != ""]
        if len(paramvec) > 0:
            for i in range(len(paramvec) - 1):
                str = "{}{},".format(str, paramvec[i])
            str = "{}{})".format(str, paramvec[len(paramvec) - 1])
        else:
            str = "{})".format(str)
        return str

    def dictToDataset(self, prestr, data):
        for key in data:
            if type(data[key]) != dict:
                self.set_dataset(
                    "{}.{}".format(prestr, key),
                    data[key],
                    persist=True,
                    archive=False,
                )
            else:
                self.dictToDataset("{}.{}".format(prestr, key), data[key])


class ConexTester(EnvExperiment, _HarrisTester):
    """Conex Driver Tester"""

    def build(self):
        self.setattr_device("conex_controller_sim0")
        self.setattr_device("conex_controller_sim1")
        self.setattr_device("conex_controller_sim2")
        self.setattr_device("conex_controller_sim3")
        self.setattr_argument(
            "command",
            EnumerationValue(
                ["connect", "close", "move_abs", "move_rel"], default="connect"
            ),
        )
        self.setattr_argument("param1", StringValue(default=""))
        self.setattr_argument("param2", StringValue(default=""))
        self.setattr_argument("device", StringValue(default="0"))

    def prepare(self):
        str = self.argumentToCommand(
            "self.conex_controller_sim" + self.device,
            self.command,
            [self.param1, self.param2],
        )
        self.command = str
        print(str)

    def run(self):
        data = eval(self.command)
        # print(self.command)
        if data is not None:
            self.dictToDataset("conex_controller_sim", data)

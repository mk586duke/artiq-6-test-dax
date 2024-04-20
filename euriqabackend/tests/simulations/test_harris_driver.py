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


class HarrisTesterMulti(EnvExperiment, _HarrisTester):
    """Harris Driver Tester for Multichannel AOM"""

    def build(self):
        self.setattr_device("harris_multichannel")
        self.setattr_argument(
            "command",
            EnumerationValue(
                [
                    "status_check",
                    "unit_info",
                    "status",
                    "measure",
                    "calpower",
                    "setoverpower",
                    "setovertemp",
                    "setmasterref",
                    "blankall",
                    "triggerall",
                    "setmastertriggersource",
                    "setperiod",
                    "setdutycycle",
                    "clearchannelfault",
                    "setchannelrfsource",
                    "setchannelmodulation",
                    "setchannelgain",
                    "setchannelfrequency",
                    "setchannelphase",
                    "setchannelamplitude",
                    "resetramcounters",
                    "loadchannelramtable",
                ],
                default="status_check",
            ),
        )
        self.setattr_argument("param1", StringValue(default=""))
        self.setattr_argument("param2", StringValue(default=""))

    def prepare(self):
        str = self.argumentToCommand(
            "self.harris_multichannel", self.command, [self.param1, self.param2]
        )
        self.command = str

    def run(self):
        data = eval(self.command)
        # print(self.command)
        if data is not None:
            self.dictToDataset("harris_multichannel", data)


class HarrisTesterGlobal(EnvExperiment, _HarrisTester):
    """Harris Driver Tester for Global AOM"""

    def build(self):
        self.setattr_device("harris_global")
        self.setattr_argument(
            "command",
            EnumerationValue(
                [
                    "status_check",
                    "unit_info",
                    "status",
                    "measure",
                    "setmaxpower",
                    "setmaxcelltemp",
                    "setmaxdrivertemp",
                    "setchannelgain",
                    "setrf",
                    "setlinearity",
                    "rfcalibrate",
                    "reset",
                ],
                default="status_check",
            ),
        )
        self.setattr_argument("param1", StringValue(default=""))
        self.setattr_argument("param2", StringValue(default=""))

    def prepare(self):
        self.command = self.argumentToCommand(
            "self.harris_global", self.command, [self.param1, self.param2]
        )

    def run(self):
        data = eval(self.command)
        # print(self.command)
        if data is not None:
            self.dictToDataset("harris_global", data)

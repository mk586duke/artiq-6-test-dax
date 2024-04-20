import time

from artiq.experiment import *

from euriqafrontend.examples.EURIQA_utilities import _EURIQA_util


def input_filename(prompt) -> TStr:
    return input("Enter path to file {}: ".format(prompt))


def input_adjust(adjust, value) -> TStr:
    return input("Update adjust value for {} (currently {}) to: ".format(adjust, value))


def printAdjustDict(adjustDict):
    print("Adjust Voltages:")
    for key in adjustDict:
        print(
            "\t{}: Line {} with gain {}".format(
                key, adjustDict[key]["line"], adjustDict[key]["adjustment_gain"]
            )
        )


def printOutputVoltage(electrodes, outputVoltage, x=22, y=5):
    for i in range(x):
        str = ""
        for j in range(y):
            if (i * y + j) < len(electrodes):
                str = "{}{} {}\t".format(
                    str, electrodes[i * y + j], round(outputVoltage[i * y + j], 4)
                )
        print(str)


def printShuttleEdge(shuttlingGraph):
    print("Shuttling Definition:")
    for edge in shuttlingGraph:
        print(
            "   Edge from {} (line {}) to {} (line {})".format(
                edge.startName, edge.startLine, edge.stopName, edge.stopLine
            )
        )


class VoltageTester(EnvExperiment, _EURIQA_util):
    """Sandia DAC tester"""

    kernel_invariants = {"electrodes"}

    def build(self):
        self.electrodes = self.get_device("dac_pc_interface")

        self.setattr_argument("line", NumberValue(0, step=1, min=0, ndecimals=0))
        self.setattr_argument("PAD", NumberValue(1, min=0, ndecimals=4))
        self.setattr_argument("DB", NumberValue(1, min=0, ndecimals=4))
        self.setattr_argument("SHIFT", NumberValue(1, min=0, ndecimals=4))
        self.setattr_argument("line_gain", NumberValue(1, min=0, ndecimals=4))
        self.setattr_argument("global_gain", NumberValue(1, min=0, ndecimals=4))
        self.setattr_argument(
            "path", StringValue("/home/monroelab/artiq/artiq-master/dac/")
        )

    def run(self):
        initTic = time.time()
        path = self.path
        voltageload = "EURIQA_Transport"
        voltageload = "EURIQA_socket_voltage"
        tic = time.time()
        self.electrodes.loadMapping(path + "EURIQA_socket_map.txt")
        self.electrodes.loadVoltage(path + voltageload + ".txt")
        self.electrodes.loadShuttleDef(path + voltageload + "_shuttling.xml")
        self.electrodes.loadGlobalAdjust(path + "EURIQA_socket_global_addconst.txt")
        self.electrodes.writeData(None)
        toc = time.time()
        print("time to load definitions: ", toc - tic)

        state = self.electrodes.saveState()

        state["adjustDict"]["DB"]["adjustment_gain"] = self.DB
        state["adjustDict"]["PAD"]["adjustment_gain"] = self.PAD
        state["adjustDict"]["SHIFT"]["adjustment_gain"] = self.SHIFT
        tic = time.time()
        self.electrodes.setAdjust(state["adjustDict"], 1)
        self.electrodes.writeData(None)
        toc = time.time()
        print("time to set global adjust: ", toc - tic)

        self.electrodes.applyLine(
            self.line, self.line_gain, self.global_gain, updateHardware=True
        )

        printShuttleEdge(state["shuttlingGraph"])
        print(self.electrodes.shuttlingDataHash())
        # print(lineno)
        # self.electrodes.shuttle_async([('Loading', 'JunctionL',shuttlingGraph[0], 0)], None)
        # self.electrodes.trigger()
        # self.electrodes.shuttle_async([('Trapping', 'Loading',shuttlingGraph[0],1)], None)
        # self.electrodes.trigger()
        # self.electrodes.trigger()
        #
        # data, header, adjustDict, adjustLines, mappingpath, lineGain, globalGain, lineno, shuttlingGraph = self.electrodes.debugData()
        # print(lineno)
        finalTic = time.time()
        print("total time: ", finalTic - initTic)
        state = self.electrodes.saveState()
        del state["shuttlingGraph"]
        self.dictToDataset("electrodes", state)
        # edge.startName = 'Loading'
        # edge.startLine = 0
        #         'stopName': 'Trapping', 'stopLine': 1}
        printOutputVoltage(state["electrodes"], state["outputVoltage"])
        # self.electrodes.shuttle_async([('Loading','Trapping',{},0)], None)

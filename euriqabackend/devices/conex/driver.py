import logging

import serial

logger = logging.getLogger(__name__)


class ConexBox:
    def __init__(self, com_x, com_y, simulation=False):
        self.simulation = simulation
        self.pos = None
        if self.simulation:
            self.x = ConexSim(com_x)
            self.y = ConexSim(com_y)
        else:
            self.x = Conex(com_x)
            self.y = Conex(com_y)

    def connect(self):
        self.x.connect()
        self.y.connect()
        self.pos = [self.x.pos, self.y.pos]

    def move_abs(self, dx, dy):
        self.x.move_abs(dx)
        self.y.move_abs(dy)
        self.pos = [self.x.pos, self.y.pos]

    def move_rel(self, dx, dy):
        self.x.move_rel(dx)
        self.y.move_rel(dy)
        self.pos = [self.x.pos, self.y.pos]

    def close(self):
        self.x.close()
        self.y.close()


class Conex:
    def __init__(self, com_port):
        self.ser = serial.Serial()
        self.ser.baudrate = 921600
        self.ser.port = com_port
        self.ser.timeout = 1
        self.pos = None
        self.state = None
        self.error = None
        self.dict = ConexDict()
        self.lims = []

    def connect(self):
        self.ser.open()
        self.get_lims()
        self.get_status()
        if "NOT REFERENCED" in self.state:
            self.ser.write(b"1OR\r\n")
            self.get_status()
            while self.state == "HOMING":
                self.get_status()
        self.get_pos()

    def get_lims(self):
        self.ser.write(b"1SR?\r\n")
        r_lim = self.ser.readline()
        self.ser.write(b"1SL?\r\n")
        l_lim = self.ser.readline()
        self.lims = [
            float(l_lim.decode("utf-8")[3:-2]),
            float(r_lim.decode("utf-8")[3:-2]),
        ]

    def close(self):
        self.ser.close()

    def get_status(self):
        self.ser.write(b"1TS\r\n")
        status = self.ser.readline()
        self.state = self.dict.state(status)
        self.error = self.dict.error(status)

    def get_pos(self):
        self.ser.write(b"1TP\r\n")
        pos = self.ser.readline()
        self.pos = float(pos.decode("utf-8")[3:-2])

    def move_rel(self, dx):
        lim_check = ((self.pos + dx) <= self.lims[1]) & (
            (self.pos + dx) >= self.lims[0]
        )
        if lim_check:
            message = "1PR" + str(dx) + "\r\n"
            self.ser.write(message.encode("utf-8"))
            self.get_status()
            while self.state == "MOVING":
                self.get_status()
            self.get_pos()
        else:
            logger.warning("Requested CONEX move outside hardware limits. Try Again")

    def move_abs(self, dx):
        lim_check = (dx <= self.lims[1]) & (dx >= self.lims[0])
        if lim_check:
            message = "1PA" + str(dx) + "\r\n"
            self.ser.write(message.encode("utf-8"))
            self.get_status()
            while self.state == "MOVING":
                self.get_status()
            self.get_pos()
        else:
            logger.warning("Requested CONEX move outside hardware limits. Try Again")


class ConexSim:
    def __init__(self, name):
        self.name = name
        self.pos = None
        self.state = None
        self.lims = []

    def connect(self):
        self.pos = 0
        self.lims = [0, 12]
        self.state = "SIM"
        print("Opening Connection Conex-{0}".format(self.name))

    def get_pos(self):
        print("Conex-{0} is currently at position {1}".format(self.name, self.pos))

    def move_rel(self, dx):
        lim_check = ((self.pos + dx) <= self.lims[1]) & (
            (self.pos + dx) >= self.lims[0]
        )
        if lim_check:
            self.pos = self.pos + dx
            print("Moving Conex-{0} by {1} units".format(self.name, self.pos))
        else:
            print("Requested CONEX move outside hardware limits. Try Again")

    def move_abs(self, dx):
        lim_check = (dx <= self.lims[1]) & (dx >= self.lims[0])
        if lim_check:
            self.pos = dx
            print("Moving Conex-{0} to position {1}".format(self.name, self.pos))
        else:
            print("Requested CONEX move outside hardware limits. Try Again")

    def close(self):
        print("Closing Connection Conex-{0}".format(self.name))


class ConexDict:
    def __init__(self):
        self.statedict = {
            "0A": "NOT REFERENCED from RESET",
            "0B": "NOT REFERENCED from HOMING",
            "0C": "NOT REFERENCED from CONFIGURATION",
            "0D": "NOT REFERENCED from DISABLE",
            "0E": "NOT REFERENCED from READY",
            "0F": "NOT REFERENCED from MOVING",
            "10": "NOT REFERENCED - NO PARAMETERS IN MEMORY",
            "14": "CONFIGURATION",
            "1E": "HOMING",
            "28": "MOVING",
            "32": "READY from HOMING",
            "33": "READY from MOVING",
            "34": "READY from DISABLE",
            "36": "READY T from READY",
            "37": "READY T from TRACKING",
            "38": "READY T from DISABLE T",
            "3C": "DISABLE from READY",
            "3D": "DISABLE from MOVING",
            "3E": "DISABLE from TRACKING",
            "3F": "DISABLE from READY T",
            "46": "TRACKING from READY T",
            "47": "TRACKING from TRACKING",
        }

    def state(self, status):
        status = status.decode("utf8")
        return self.statedict[status[7:9]]

    def error(self, status):
        status = status.decode("utf8")
        h_error = status[3:7]
        b_error = format(int(h_error, 16), "#018b")
        b_error = b_error[2:]

        if b_error == "0000000000000000":
            error_out = None
        else:
            error_out = []
            if b_error[6] == "1":
                error_out.append("80 W output power exceeded")
            if b_error[7] == "1":
                error_out.append("DC voltage too low")
            if b_error[8] == "1":
                error_out.append("Wrong ESP stage")
            if b_error[9] == "1":
                error_out.append("Homing time out")
            if b_error[10] == "1":
                error_out.append("Following error")
            if b_error[11] == "1":
                error_out.append("Short circuit detection")
            if b_error[12] == "1":
                error_out.append("RMS current limit")
            if b_error[13] == "1":
                error_out.append("Peak current limit")
            if b_error[14] == "1":
                error_out.append("Positive end of run")
            if b_error[15] == "1":
                error_out.append("Negative end of run")

        return error_out

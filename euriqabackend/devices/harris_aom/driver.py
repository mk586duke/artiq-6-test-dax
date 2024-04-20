import logging
import socket

logger = logging.getLogger(__name__)


class Message:
    def __init__(self, id, param1="", param2=""):
        self.id = id
        self.param1 = "" if (param1 == None) else param1
        self.param2 = "" if (param2 == None) else param2

    def __str__(self):
        msg = "{} {} {}".format(self.id, self.param1, self.param2).strip()
        return " ".join(msg.split())

    def send(self):
        temp = self.__str__() + "\r\n"
        return temp.encode()


class _HarrisAOM:
    def __init__(self, port=-1, ipaddr="", simulation=True):
        self.port = port
        self.ipaddr = ipaddr
        self.simulation = simulation
        if simulation is not True:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ipaddr, self.port))

    def send(self, message):
        self.socket.send(message.send())
        dataout = "".encode()
        while True:
            data = self.socket.recv(1064)
            dataout = dataout + data
            if data[len(data) - 1] == int("ff", 16):
                break
        if len(dataout) == 1:
            return
        else:
            return self.fmt(
                dataout[1:-3].decode()
            )  # trim start and end characters, decode and parse

    def close(self):
        self.socket.close()

    def send_request(self, id, param1=None, param2=None):
        message = Message(id, param1=param1, param2=param2)
        if self.simulation:
            print(message.__str__())
            return
        else:
            return self.send(message)

    def fmt(self, data):
        msg = [[j.strip() for j in i.split(",")] for i in data.split("\r\n")]
        return msg

    def status_check(self):
        raise NotImplementedError

    def ping(self):
        try:
            self.status()
        except:
            logger.warning("ping failed")
            return False
        return True


class HarrisMultichannelAOM(_HarrisAOM):
    def __init__(self, port=2101, ipaddr="192.168.1.160", simulation=True):
        _HarrisAOM.__init__(self, port=port, ipaddr=ipaddr, simulation=simulation)
        self.status_check()

    def status_check(self):
        aom = self.unit_info()
        aom.update(self.status())
        aom.update(self.measure())
        return aom

    def _setsource(self, request, source, param1=None):
        if source == "i" or source == "internal":
            self.send_request(request, param1=param1, param2="i")
        elif source == "e" or source == "external":
            self.send_request(request, param1=param1, param2="e")
        elif source == "0" or source == "off" or source == "o":
            self.send_request(request, param1=param1, param2="0")
        else:
            raise ValueError

    def unit_info(self):
        if self.simulation:
            ret = [
                ["?", "100432A", "000.000", "001"],
                ["00", "01"],
                ["01", "01"],
                ["02", "01"],
            ]
        else:
            ret = self.send_request("?")
        channel = {}
        for i in range(len(ret) - 2):
            channel.update(
                {"{}".format(ret[i + 1][0]): {"board_logic_rev": ret[i + 1][1]}}
            )
        out = {
            "unit_name": ret[0][1],
            "firmware_rev": ret[0][2],
            "logic_rev": ret[0][3],
            "boards_connected": len(ret) - 2,
            "board_logic_rev": ret[1][1],
        }
        out.update({"channel": channel})
        return out

    def status(self):
        if self.simulation:
            ret = [
                ["Status", "0", "i", "10", "5", "i", "0", "064", "0794"],
                ["00", "1", "0", "i", "0", "13", "200000000", "000", "02950"],
                ["01", "1", "0", "i", "0", "13", "200000000", "000", "02700"],
                ["02", "1", "0", "e", "0", "13", "200000000", "000", "02950"],
            ]
        else:
            ret = self.send_request("Status")
        out = {
            "controller_fault": ret[0][1] == 1,
            "trigger_source": "internal" if ret[0][2] == "i" else "external",
            "trigger_duty_cycle": int(ret[0][3]),
            "period_multiplier": int(ret[0][4]),
            "rf_blanking_on": ret[0][5] == 1,
            "over_temperature_limit": "{} C".format(ret[0][6]),
            "over_power_limit": "{} mW".format(ret[0][7]),
        }
        channel = {}
        for i in range(len(ret) - 2):
            channel.update(
                {
                    "{}".format(ret[i + 1][0]): {
                        "fault": ret[i + 1][1] == 1,
                        "rf_on": ret[i + 1][2] == 1,
                        "input_source": "internal"
                        if ret[i + 1][3] == "i"
                        else "external",
                        "mod_source": "off"
                        if ret[i + 1][4] == "0"
                        else (
                            "direct_trigger" if ret[i + 1][4] == "d" else "ram_table"
                        ),
                        "gain": int(ret[i + 1][5]),
                        "dds_frequency": "{} Hz".format(ret[i + 1][6]),
                        "dds_phase": "{} degrees".format(ret[i + 1][7]),
                        "dds_amplitude": int(ret[i + 1][8]),
                    }
                }
            )
        out.update({"channel": channel})
        return out

    def measure(self):
        if self.simulation:
            ret = [
                ["Meas", "1", "255", "255"],
                ["00", "0", "0000", "041"],
                ["01", "0", "0000", "040"],
                ["02", "0", "0000", "042"],
            ]
        else:
            ret = self.send_request("Meas")
        out = {
            "controller_fault": ret[0][1] == 1,
            "cell_temperature_a": int(ret[0][2]),
            "cell_temperature_b": int(ret[0][3]),
        }
        channel = {}
        for i in range(len(ret) - 2):
            channel.update(
                {
                    "{}".format(ret[i + 1][0]): {
                        "fault_meas": ret[i + 1][1] == 1,
                        "meas_rf_power": int(ret[i + 1][2]),
                        "meas_temperature": int(ret[i + 1][3]),
                    }
                }
            )
        out.update({"channel": channel})
        return out

    def calpower(self, power):
        """Calibrates the power meter on a channel assuming the output power level is set to 500mW.
         I think the input power level here should actually be the channel - KMB 3/16/2018"""
        self.send_request("CalPower", param1=power)

    def setoverpower(self, power: int):
        """Set over power for all channels. Power in mW. To disable, set to zero."""
        if power >= 0:
            self.send_request("SetOverPower", param1=power)
        else:
            raise ValueError

    def setovertemp(self, temp):
        """Set over temperature limit in degrees C."""
        self.send_request("SetOverTemp", param1=temp)

    def setmasterref(self, source):
        """Sets the master reference source to internal or external """
        self._setsource("SetRef", source)

    def blankall(self, blank: bool):
        """Sets or clears the global RF blanking signal"""
        self.send_request("Blank", param1="1" if blank else "0")

    def triggerall(self, trigger: bool):
        """Enables the global trigger (useful for synchronizing RAM-based modulation look up channels"""
        self.send_request("EnTrig", param1="1" if trigger else "0")

    def setmastertriggersource(self, source):
        """Sets the global trigger source"""
        self._setsource("SetTrig", source)

    def setperiod(self, period: int):
        """Set the internal trigger period. The period is 312.5 us*2^period, with a default value of 5 (10ms)"""
        if period in range(8):
            self.send_request("SetPeriod", param1=period)
        else:
            raise ValueError

    def setdutycycle(self, duty):
        """Set the internal duty cycle. Default is 10%. Accepted values are 10 and 50"""
        self.send_request("SetDuty", param1=duty)

    def clearchannelfault(self, channel):
        if (channel in range(32)) or (channel == "all"):
            self.send_request("ClearFault", param1=channel)
        else:
            raise ValueError

    def setchannelrfsource(self, channel: int, source):
        """Configures the RF channel"""
        if channel in range(32):
            self._setsource("SetRF", source, param1=channel)
        else:
            raise ValueError

    def setchannelmodulation(self, channel: int, modulationtype: str):
        """Turns on/off modulation"""
        if (channel in range(32)) and (modulationtype in ["0", "D", "R"]):
            self.send_request("SetMod", param1=channel, param2=modulationtype)
        else:
            raise ValueError

    def setchannelgain(self, channel: int, gain: int):
        """Sets the RF channel gain"""
        if (channel in range(32)) and (gain in range(24)):
            self.send_request("SetGain", param1=channel, param2=gain)
        else:
            raise ValueError

    def setchannelfrequency(self, channel: int, frequency: int):
        """Set the DDS channel frequency"""
        if channel in range(32):
            ftw = int(round(frequency * 2 ** 23 * 1e-9))
            self.send_request(
                "SetFreq", param1=channel, param2="{} {}".format(frequency, ftw)
            )
        else:
            raise ValueError

    def setchannelphase(self, channel: int, phase: int):
        """Set the DDS channel phase"""
        if channel in range(32):
            self.send_request("SetPhase", param1=channel, param2=phase)
        else:
            raise ValueError

    def setchannelamplitude(self, channel: int, amplitude: int):
        """Set the DDS channel amplitude"""
        if (amplitude in range(16384)) and (channel in range(32)):
            self.send_request("SetAmp", param1=channel, param2=int(amplitude))
        else:
            raise ValueError

    def resetramcounters(self):
        """Reset RAM table counters"""
        self.send_request("RAMCntRs")

    def loadchannelramtable(self, channel: int, tabledata):
        """Load the RAM based look up table into the selected channel"""
        if channel in range(32):
            self.send_request("LoadTable", param1=channel, param2="\n" + tabledata)
        else:
            raise ValueError


class HarrisGlobalAOM(_HarrisAOM):
    def __init__(self, port=2101, ipaddr="192.168.1.162", simulation=True):
        _HarrisAOM.__init__(self, port=port, ipaddr=ipaddr, simulation=simulation)
        self.status_check()

    def status_check(self):
        aom = self.unit_info()
        aom.update(self.status())
        aom.update(self.measure())
        return aom

    def unit_info(self):
        if self.simulation:
            ret = [["?", "100435A", "000.000"]]
        else:
            ret = self.send_request("?")
        return {"unit_name": ret[0][1], "firmware_rev": ret[0][2]}

    def status(self):
        if self.simulation:
            ret = [["Status", "040", "060", "060", "1", "050", "040", "041", "040"]]
        else:
            ret = self.send_request("Status")
        return {
            "over_power_limit": int(ret[0][1]),
            "cell_over_temp_limit": int(ret[0][2]),
            "driver_over_temp_limit": int(ret[0][3]),
            "rf_on": ret[0][4] == "1",
            "linearity_setting": int(ret[0][5]),
            "ChA": {"gain": int(ret[0][6])},
            "ChB": {"gain": int(ret[0][7])},
            "ChC": {"gain": int(ret[0][8])},
        }

    def measure(self):
        if self.simulation:
            ret = [["Meas", "0", "0553", "0519", "0462", "036", "037", "037"]]
        else:
            ret = self.send_request("Meas")
        return {
            "alarm": ret[0][1] == "1",
            "cell_temperature_a": int(ret[0][2]),
            "cell_temperature_b": int(ret[0][3]),
            "driver_temp": int(ret[0][4]),
            "ChA": {"rf_power": int(ret[0][5])},
            "ChB": {"rf_power": int(ret[0][6])},
            "ChC": {"rf_power": int(ret[0][7])},
        }

    def setmaxpower(self, power):
        self.send_request("SetMaxP", param1=power)

    def setmaxcelltemp(self, temp):
        self.send_request("SetMaxCellT", param1=temp)

    def setmaxdrivertemp(self, power):
        self.send_request("SetMaxDrvT", param1=power)

    def setchannelgain(self, channel, gain):
        self.send_request("SetGain", param1=channel, param2=gain)

    def setrf(self, state):
        self.send_request("SetRF", param1=1 if state == "on" else 0)

    def setlinearity(self, linearity):
        self.send_request("SetLin", param1=linearity)

    def rfcalibrate(self, channel):
        self.send_request("Calibrate", param1=channel)

    def reset(self):
        self.send_request("Reset")

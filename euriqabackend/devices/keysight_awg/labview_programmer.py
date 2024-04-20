"""This program implements a communication protocol with the Keysight Programmer LabVIEW
program.  It implements the following commands:

"Set exp: [experiment name]" - selects and loads the given experiment,
    does not program AWG
"Program" - programs the AWG
"Set DDS: [data TBD]" - programs the switch network to address some pattern of
    slots independent of the AWG

Note that the command names are all seven letters long and are separated from
the parameters by one space. These lengths are used by the LabVIEW decoder.
"""
import logging
import socket
import typing

_LOGGER = logging.getLogger(__name__)

TCP_IP = "127.0.0.1"
TCP_PORT = 5001
BUFFER_SIZE = 1024


def send_command(command: str):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    # s.send(command2)
    s.send((command + "\r\n").encode("utf-8"))
    _LOGGER.debug("Sent command to Keysight Programmer: `%s`", command)
    response = s.recv(BUFFER_SIZE).decode("utf-8")
    s.close()

    _LOGGER.debug("Response from Keysight Programmer: %s", response)

    if response != "OK":
        _LOGGER.error(response)
        raise Exception(response)


def set_experiment(experiment: str):
    send_command("Set exp: " + experiment)


def program_AWG():
    send_command("Program")


def set_DDS(slots: typing.List[int]):
    slot_string = ""
    for s in slots:
        slot_string += str(s) + ","
    slot_string = slot_string[:-1]

    send_command("Set DDS: " + slot_string)

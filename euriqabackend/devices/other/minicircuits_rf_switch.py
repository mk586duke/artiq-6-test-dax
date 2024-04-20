"""Temporary driver for Minicircuits 8-channel RF switch RC-8SPDT-A18."""

import logging
import urllib.request as request


_LOGGER = logging.getLogger(__name__)


class SPDTNetworkSwitch:
    """SCPI Network driver for Minicircuits RC-8SPDT-A18."""

    def __init__(self, ip_address: str, port: int = 80):
        self.ip_address = ip_address
        self.port = port
        model_num = self._get_response("MN?")
        serial_num = self._get_response("SN?")
        _LOGGER.debug(
            "Connection started to Minicircuits 8SPDT switch %s (%s)",
            model_num,
            serial_num,
        )

    def _get_response(self, scpi_command: str) -> bytes:
        return request.urlopen(
            "http://{ip_address}:{port}/{command}".format(
                ip_address=self.ip_address, port=self.port, command=scpi_command
            )
        ).read()

    def set_switch(self, set_on: bool) -> None:
        """Set all switches together to either ON or OFF.

        Note ON = COM connected to port 2, OFF = COM connected to port 1.
        """
        state = 0xFF if set_on else 0x00
        resp = self._get_response("SETP={}".format(state))
        _LOGGER.debug("Setting switch to state %i", state)
        assert int(resp) == 1

    def set_if_needed(self, set_on: bool) -> None:
        """Check the state of the switch, and only change if needed.

        See :meth:`set_switch` for details.
        """
        desired_switch_state = 0xFF if set_on else 0x00
        current_switch_state = int(self._get_response("SWPORT?"))
        if current_switch_state != desired_switch_state:
            self.set_switch(set_on)

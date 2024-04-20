"""Driver for controling R&S signal generator"""

import logging
import threading
import time

import visa
import numpy as np

_LOGGER = logging.getLogger(__name__)


class _RSSource:
    """General R&S signal generator"""

    def __init__(
        self, ip_address: str, port: int = 5025,
    ):
        self.ip_address = ip_address
        self.port = port
        self._resource_manager = visa.ResourceManager()
        self._socket_connection = self._resource_manager.open_resource(
            "TCPIP::{}::{}::SOCKET".format(ip_address, port),
            read_termination="\n",
            timeout=1000,  # milliseconds
        )

    def ping(self):
        """Get status of the power supply.

        Required by :mod:`artiq_ctlmgr`.
        """
        try:
            self._socket_connection.session
        except visa.InvalidSession:
            return False
        else:
            return True

    def _open_connection(self):
        if self.ip_address is None:
            raise IOError("Cannot connect to NULL address")
        self._ensure_connection()

    def _close_connection(self):
        self._socket_connection.close()

    def _reset_connection(self):
        self._close_connection()
        self._open_connection()

    def _ensure_connection(self):
        if self.ip_address is None:
            raise ValueError("Cannot ensure connection to null address.")
        try:
            self._socket_connection.session
        except visa.InvalidSession:
            _LOGGER.debug(
                "Restarting connection to smc100A: '%s:%i'", self.ip_address, self.port
            )
            self._socket_connection = self._resource_manager.open_resource(
                "TCPIP::{}::{}::SOCKET".format(self.ip_address, self.port),
                read_termination="\n",
                timeout=1000,  # milliseconds
            )

    def _command(self, command: str, expect_response: bool = False):
        """
        Send a command to the RF source.

        Args:
            command (str): VISA-style command to send
            expect_response (bool, optional): Defaults to True. If True, runs a query.
                Otherwise, just sends the data.
        """
        full_command = command
        guess_is_query = "?" in command
        if expect_response:
            if not guess_is_query:
                _LOGGER.warning(
                    "Your command doesn't include '?', but called the query function."
                )
                _LOGGER.warning("Command: %s", full_command)
            _LOGGER.debug("Querying device: %s", full_command)
            return self._socket_connection.query(full_command)
        else:
            if guess_is_query:
                _LOGGER.warning(
                    "Your command includes '?', but didn't call the query function."
                )
                _LOGGER.warning("Command: %s", full_command)
            _LOGGER.debug("Sending command: %s", full_command)
            return self._socket_connection.write(full_command)

    def _turn_off_display(self):
        """Turn off display update for faster switching.

        Useful for short dwell times.
        """
        # self._ensure_connection()
        return self._command("SYST:DISP:UPD OFF")

    def _turn_on_display(self):
        """Turn on display after fast switching is done."""
        # self._ensure_connection()
        return self._command("SYST:DISP:UPD ON")

    def _activate_level_sweep(self):
        """Settings should already be set for the sweep
        includes: start,stop,manual vs trigger
        """

        """visacmd: SOUR:POW:MODE SWEep"""

        return self._command("SOUR:POW:MODE SWEep")

    def _set_trigger_level_sweep(self, trigger_type: str = "SING"):
        """Set the trigger for a sweep.

        Commands:
            * TRIG:PSW:SOUR EXT
            * TRIG:PSW:SOUR SING
        """
        return self._command("TRIG:FSW:SOUR {}".format(trigger_type))

    def _set_level_sweep_shape(self, shape: str = "TRI"):
        """Choose a SAWTooth or TRIangle wave."""
        return self._command("SOUR:SWE:POW:SHAP {}".format(shape))

    def _set_dwell_time(self, dwell_time: float):
        """set the dwell time in seconds

        Args:
            dwell_time (float): Value in range of 10e-3 to 100 in increments of 100e-6.
        """
        if dwell_time < 10e-3 or dwell_time > 100:
            raise ValueError(
                "Dwell time {:.3f} is out of range [10e-3, 100].".format(dwell_time)
            )
        else:
            return self._command("SOUR:SWE:POW:DWEL {}".format(dwell_time))

    def _set_level_sweep_start_power(self, start_power: float):
        """visacmd: POW:STAR -20 dBm"""
        return self._command("POW:STAR {}".format(start_power))

    def _set_level_sweep_stop_power(self, stop_power: float):
        """visacmd: POW:STOP 3"""
        return self._command("POW:STOP {}".format(stop_power))

    def _execute_single_sweep(self):
        """must already be in single sweep mode
        SOUR:SWE:POW:EXEC"""
        return self._command("SOUR:SWE:POW:EXEC")

    def _set_step_size(self, step_size: float = 0.5):
        """sets the step size for a logarithmic ramp in dBm"""
        return self._command("SOUR:SWE:POW:STEP {}".format(step_size))

    def _activate_ext_mod(self):
        self._command("SOUR:AM:SOUR EXT")
        return self._command("SOUR:AM:STAT ON")

    def _deactivate_ext_mod(self):
        self._command("SOUR:AM:SOUR EXT")
        return self._command("SOUR:AM:STAT OFF")

    def _set_mod_depth(self, depth_percent: float = 0.0):
        if abs(depth_percent) > 99:
            raise ValueError("Modulation depth {}% is too large.".format(depth_percent))
        return self._command("SOUR:AM:DEPT "+str(depth_percent))

    def _set_mod_coupling(self, coupling: str="DC"):
        """DC takes the AC and DC signal. AC ignores the DC"""
        return self._command("SOUR:AM:EXT:COUP "+coupling)

    def _set_ext_mod_impedance_high(self):
        return self._command("INP:MOD:IMP HIGH")

    # PUBLIC Methods

    def get_freq_Hz(self):
        """SOUR:FREQ"""
        return float(self._command("SOUR:FREQ?", expect_response=True))

    def get_power_dBm(self):
        """SOUR:POW?"""
        return float(self._command("SOUR:POW?", expect_response=True))

    def return_to_manual(self):
        return self._command("&GTL")

    def get_mod_sensitivity(self):
        return self._command("SOUR:AM:SENS?",expect_response=True)


class RFSignalGenerator(_RSSource):
    """Rohde & Schwarz signal generator that supplies rf voltages to the EURIQA trap."""

    def __init__(
        self, ip_address: str,
    ):
        super().__init__(ip_address)

        self.ramp_start_freq = self.get_freq_Hz()
        self.ramp_start_power = self.get_power_dBm()
        _LOGGER.info("Initial ramp power: %f", self.ramp_start_power)

        self.sweep_activated = False
        self.mod_activated = False
        self.seed = np.random.rand()

    def get_ramp_settings(self):
        if not self.sweep_activated:
            raise RuntimeError("Sweep has not been activated.")
        elif self.ramp_type == "power":
            return {
                "start power dBm": self.ramp_start_power,
                "stop power dBm": self.ramp_stop_power,
                "dwell time s": self.dwell_time,
            }

    def play_single_ramp(self):
        if self.sweep_activated:
            self._execute_single_sweep()
        else:
            raise RuntimeError("Sweep has not been activated")

    def activate_sweep(
        self,
        # start_power: float,
        stop_power_dBm: float,
        dwell_time_s: float = 10e-3,
        step_size_dBm: float = 0.5,
        trig_mode: str = "SING",
        ramp_type: str = "TRI",
    ):
        self.ramp_type = ramp_type
        self.ramp_start_power = float(self.get_power_dBm())
        self.ramp_stop_power = stop_power_dBm
        self.dwell_time = dwell_time_s
        self.step_size = step_size_dBm

        if self.ramp_start_power < 0.01 or self.ramp_stop_power < 0.01:
            self.sweep_activated = False
            raise ValueError(
                "power is too low for fixed output mode sweep. "
                "start: {:.1f} stop: {:.1f}.".format(
                    self.ramp_start_power, self.ramp_stop_power
                )
            )
        self.total_ramp_time = (
            np.abs((self.ramp_start_power - self.ramp_stop_power))
            / self.step_size
            * self.dwell_time
        )
        if self.total_ramp_time > 10:
            self.sweep_activated = False
            raise ValueError(
                "Total ramp time {} s is longer than 10 s".format(self.total_ramp_time)
            )
        elif self.total_ramp_time < 1e-3:
            self.sweep_activated = False
            raise ValueError(
                "Total ramp time {} s is shorter than 1 ms".format(self.total_ramp_time)
            )

        if self.get_power_dBm() != self.ramp_start_power:
            self.sweep_activated = False
            raise RuntimeError(
                "End power {} does not equal current power {}".format(
                    self.ramp_stop_power, self.get_power_dBm()
                )
            )

        # turn off the display for fast ramping
        self._turn_off_display()
        self._set_level_sweep_shape(ramp_type)

        self._set_level_sweep_start_power(self.ramp_start_power)
        self._set_level_sweep_stop_power(self.ramp_stop_power)
        self._set_dwell_time(self.dwell_time)

        self._set_step_size(self.step_size)

        # start with auto mode for the power sweep
        self._command("SOUR:SWE:POW:MODE AUTO")
        # change to the trigger of your choice.
        self._set_trigger_level_sweep(trig_mode)
        self._activate_level_sweep()

        self.sweep_activated = True
        _LOGGER.debug("Ramp settings are %s", self.get_ramp_settings())
        _LOGGER.info("Sweep activated")

    def activate_amp_mod_ext(self, depth_percent: float):
        self._set_ext_mod_impedance_high()
        self._set_mod_depth(depth_percent)
        self._set_mod_coupling("DC")
        self._activate_ext_mod()

    def deactivate_amp_mod_ext(self):
        self._deactivate_ext_mod()

    def turn_off_display(self):
        self._turn_off_display()

    def turn_on_display(self):
        self._turn_on_display()

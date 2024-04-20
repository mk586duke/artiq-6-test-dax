"""Driver for Keysight N6700B MPS (Power Supply Unit).

Adapted by Marko Cetina from sample code to drive EURIQA ovens.
Implements keepalive timers and keepalive network communication to prevent
running ovens for too long.

NOTE: *you should not use the :meth:`N6700bPowerSupply.enable` or
:meth:`N6700bPowerSupply.disable` methods on the oven.
:meth:`OvenPowerSupply.turn_on` and :meth:`OvenPowerSupply.turn_off`
are much safer and add extra functionality (like shutting down oven after
a certain time) that are not available otherwise.*

TODO:
    * Integrate with InfluxDB to log channel voltages/currents.
    * move test_oven() to separate file in test folder
    * Add simulation to the Oven class
"""
import logging
import threading
import time

import visa

_LOGGER = logging.getLogger(__name__)


class N6700bPowerSupply:
    """Generic power supply using the Keysight N6700b instrument.

    Provides watchdog control & current/voltage reading/setting.
    """

    _current_voltage = 0
    _current_current = 0

    # default watchdog period
    _watchdog_timeout = 60 * 60  # (seconds), 60 mins
    # default watchdog TCP keepalive period
    _keepalive_period_seconds = 1.0  # (seconds)

    def __init__(self, ip_address: str, port: int, instrument_channel: int):
        """
        Proper constructor of an Oven object.

        Args:
            ip_address(str): IP address of the instrument
            instrument_channel (int): number of the power supply module
                within the frame (starting at 1)
        """
        if instrument_channel not in range(1, 5):
            raise ValueError("Invalid channel: must be 1-4")
        self.ip_address = ip_address
        self.instrument_channel = instrument_channel
        self.port = port
        self._resource_manager = visa.ResourceManager()
        self._socket_connection = self._resource_manager.open_resource(
            "TCPIP::{}::{}::SOCKET".format(ip_address, port),
            read_termination="\n",
            timeout=1000,  # milliseconds
        )
        self._keepalive_timer = None
        self._is_keepalive_running = False
        self._shutdown_timer = None

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
                "Restarting connection to n6700b: '%s:%i'", self.ip_address, self.port
            )
            self._socket_connection = self._resource_manager.open_resource(
                "TCPIP::{}::{}::SOCKET".format(self.ip_address, self.port),
                read_termination="\n",
                timeout=1000,  # milliseconds
            )

    def _global_command(self, command: str):
        """Send a command to the whole n6700b device."""
        self._socket_connection.write(command)

    def _channel_command(self, command: str, expect_response: bool = False):
        """
        Send a command to a specific channel/slot on the n6700b supply.

        Args:
            command (str): VISA-style command to send
            expect_response (bool, optional): Defaults to True. If True, runs a query.
                Otherwise, just sends the data.
        """
        full_command = "{}{}".format(command, self._instrument_string())
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

    def _instrument_string(self):
        return "(@{})".format(self.instrument_channel)

    def _keep_alive_function(self):
        """Run to keep the connection to the n6700b alive.

        Meant to be run from thread.
        """
        _LOGGER.debug("Keeping n6700b connection alive")
        while self._is_keepalive_running:
            time.sleep(self._keepalive_period_seconds)
            self._ensure_connection()

        _LOGGER.debug("Ended keepalive function")

    def _stop_keepalive_timer(self):
        if self._keepalive_timer is not None:
            _LOGGER.debug("Attempting to stop keepalive timer (for watchdog)")
            self._is_keepalive_running = False
            self._keepalive_timer.join()
            _LOGGER.debug("Stopped watchdog keepalive timer")
            self._keepalive_timer = None
        else:
            _LOGGER.warning("Called stop keepalive timer when it's not running.")

    def _start_keepalive_timer(self):
        if self._keepalive_timer is not None:
            self._stop_keepalive_timer()
        self._is_keepalive_running = True
        self._keepalive_timer = threading.Thread(target=self._keep_alive_function)
        self._keepalive_timer.start()

    @property
    def is_shutdown_timer_running(self):
        """State of the shutdown timer thread."""
        if self._shutdown_timer is not None:
            return self._shutdown_timer.is_alive()
        else:
            return False

    def _shutdown_timer_function(self):
        """Disable the oven.

        Meant to be run by :class:`threading.Timer`.
        """
        # timeout has elapsed
        _LOGGER.info("Shutdown timer triggered. Shut down oven")
        self._disable()

    # ****************************
    # PUBLIC
    # ****************************

    def set_voltage(self, voltage: float):
        """
        Set the output voltage of the oven output.

        Args:
            voltage (float) : voltage in Volts
        """
        self._ensure_connection()
        self._channel_command("SOUR:VOLT {},".format(voltage))
        readback = float(self._channel_command("MEAS:VOLT? ", expect_response=True))
        self._current_voltage = readback
        if abs(float(voltage) - readback) > 20e-3:
            # NOTE: this will error if the output is not enabled,
            # OR if it takes time to ramp voltage.
            _LOGGER.warning(
                ">20 mV error in read-back voltage: set %.2f != %.2f", voltage, readback
            )

    def get_current(self) -> float:
        """
        Get the output current of the oven output.

        Arguments:
            current_amps (float): setpoint for output current in amps
        """
        self._ensure_connection()
        readback = float(self._channel_command("MEAS:CURR? ", expect_response=True))
        self._current_current = readback
        return readback
        #print("Current = ", readback)

    def set_current(self, current_amps: float):
        """
        Set the output current of the oven output.

        Arguments:
            current_amps (float): setpoint for output current in amps
        """
        self._ensure_connection()
        self._channel_command("SOUR:CURR {},".format(current_amps))
        readback = float(self._channel_command("MEAS:CURR? ", expect_response=True))
        self._current_current = readback
        if abs(float(current_amps) - readback) > 20e-3:
            # NOTE: this will error if the output is disabled.
            _LOGGER.warning(
                ">20 mA error in read-back current: set %.2f != %.2f",
                current_amps,
                readback,
            )

    #    def initialize(self):
    #        self.__ensureConnection()
    #        # self.__socketConnection.send()

    def _enable(self):
        """Enable the oven output."""
        self._ensure_connection()
        self._channel_command("OUTP 1,")

    def _disable(self):
        """Disable the oven output."""
        self._ensure_connection()
        self._channel_command("OUTP 0,")

    def start_shutdown_timer(self, timeout: float):
        """
        Start the oven shutdown timer.

        Calling this will cause the oven to automatically shut down its output
        through a call to :meth:`disable` after the given time has elapsed.

        Args:
            timeout (float): timeout period in seconds
        """
        if self._shutdown_timer is not None:
            _LOGGER.info("Shutdown timer is already running. Canceling it.")
            self.stop_shutdown_timer()
            _LOGGER.debug("Shutdown timer ended.")

        self._shutdown_timer = threading.Timer(
            interval=timeout, function=self._shutdown_timer_function
        )
        self._shutdown_timer.start()

    def stop_shutdown_timer(self):
        """Stop the oven shutdown timer."""
        if self._shutdown_timer is not None:
            _LOGGER.debug("Waiting for shutdown timer to stop")
            self._shutdown_timer.cancel()
        else:
            _LOGGER.warning("No shutdown timer to stop")
        self._shutdown_timer = None  # clear to be used next time.

    def enable_watchdog(self):
        """
        Enable the N6700B watchdog functionality.

        ALL the channels of the instrument will shut off their outputs if
        no network traffic is received for :attr:`_watchdog_timeout` seconds.

        Starts a _keepalive_period-period keepalive timer to prevent
        the watchdog from tripping.

        NOTE: tripping watchdog will protect ALL channels, need to run
        :meth:`clear_watchdog_fault` with NO ARG to clear all faults.
        """
        self._ensure_connection()
        _LOGGER.info("Starting watchdog (%.1f seconds)", self._watchdog_timeout)
        self._global_command("OUTP:PROT:WDOG 1")
        self._global_command("OUTP:PROT:WDOG:DEL {}".format(self._watchdog_timeout))
        self._start_keepalive_timer()

    def disable_watchdog(self):
        """Disables the N6700B watchdog functionality."""
        self._ensure_connection()
        self._global_command("OUTP:PROT:WDOG 0")
        self._stop_keepalive_timer()

    def clear_watchdog_fault(self, channel: int = None) -> None:
        """Clear watchdog or other protection fault on a/all channel(s).

        Make sure you check that the watchdog is disabled first, otherwise
        you could just continually trip the watchdog. If your channel was set to output,
        then immediately after clearing fault it will output again.

        NOTE: no way to check protection state via network.
        """
        if channel is None:
            _LOGGER.debug("Clearing protection faults in all channels")
            for i in range(1, 5):
                self._global_command("OUTP:PROT:CLE (@{})".format(i))
        else:
            _LOGGER.debug("Clearing protection fault in ONLY channel %i", channel)
            self._global_command("OUTP:PROT:CLE (@{})".format(channel))

    def increment_voltage(self, increment: float = 0.1):
        """Increment the output voltage."""
        self._current_voltage += increment
        if self._current_voltage > 5:
            self._current_voltage = 0
        self.set_voltage(self._current_voltage)
        _LOGGER.info("Updated current voltage to: %f", self._current_voltage)


class OvenError(Exception):
    """Raised when the Oven Power Supply has an issue."""

    pass


class OvenPowerSupply(N6700bPowerSupply):
    """Power supply for the Barium/Ytterbium ovens used by the UMD Euriqa team.

    Use this, and not :class:`N6700bPowerSupply`, to only enable certain SAFE features.
    You should also mainly use the :meth:`turn_on` and :meth:`turn_off` functions
    to have auto-shutdown if you forget to turn off oven.
    """

    # NOT user-modifiable for safety
    MAX_VOLTAGE = 1.7
    MAX_CURRENT = 2.2  # limited by interlock, might want to change later for pulsing

    def turn_on(self, current: float, voltage: float, timeout: float = 15 * 60):
        """Start the oven with specified current & voltage.

        Timeout is time before oven shuts off. Defaults to 15 minutes.
        If you DEFINITELY want the oven on longer, you can either increase
        `timeout` or call :meth:`stop_shutdown_timer`.
        """
        self.set_voltage(voltage)
        self.set_current(current)
        self.start_shutdown_timer(timeout)
        self.enable_watchdog()
        self._enable()

    def turn_off(self):
        """Turn off the oven."""
        self._disable()  # disable, then disable safeguards
        self.stop_shutdown_timer()
        self.disable_watchdog()

    def increment_voltage(self, increment: float = 0.1):
        """Disabled to protect the oven."""
        raise NotImplementedError("This method not allowed on an oven")

    def set_current(self, current_amps: float):
        """Check to see if the current is in an acceptable bound before setting."""
        if abs(current_amps) > self.MAX_CURRENT:
            _LOGGER.error(
                "Current setpoint out of bounds: abs(%f) > %f",
                current_amps,
                self.MAX_CURRENT,
            )
            raise OvenError("Current setpoint ({}) out of range".format(current_amps))
        else:
            super().set_current(current_amps)

    def set_voltage(self, voltage: float):
        """Check to see if the voltage is in an acceptable bound before setting."""
        if abs(voltage) > self.MAX_VOLTAGE:
            _LOGGER.error(
                "Voltage setpoint out of bounds: abs(%f) > %f",
                voltage,
                self.MAX_VOLTAGE,
            )
            raise OvenError("Voltage setpoint ({}) out of range".format(voltage))
        else:
            super().set_voltage(voltage)

    def disconnect(self, turn_off: bool = True, disable_watchdog: bool = False):
        """Disconnect and close connections.

        Args:
            turn_off (bool, optional): Turn off the channel on disconnect.
                Defaults to True.
            disable_watchdog (bool, optional): Disable the hardware watchdog timer
                on the oven PSU. This is done automatically if
                ``turn_off == True``. The safest thing is to leave it on (default),
                but you risk turning all other channels off when the watchdog expires.
                Defaults to False.
        """
        if turn_off:
            _LOGGER.debug("Turning off oven on disconnect")
            self.turn_off()
        else:
            # shutdown timer MUST be disabled on disconnect/shutdown to end program
            self.stop_shutdown_timer()
        if disable_watchdog:
            _LOGGER.warning(
                "Oven watchdog disabled on disconnect. "
                "Make sure that the oven is in a safe state."
            )
            self.disable_watchdog()
        self._close_connection()

    def _ensure_connection(self):
        super()._ensure_connection()
        result = self._socket_connection.query("*IDN?")
        if result != "Agilent Technologies,N6700B,MY54005645,D.04.01":
            raise IOError("Cannot ensure connection to the oven.")


def test_oven(host: str = "192.168.78.247", port: int = 5025):
    """Test code to test the oven works properly."""
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect(('192.168.78.172',5025))
    # s.send(b'*IDN?\n')
    # data = s.recv(8)
    # print(data)
    # s.close()
    # oven = SCPIsocket(HOST)

    # print(oven.get_error('*IDN?'))
    # print(oven.send('*IDN?'))
    # print(oven.get_error('*IDN?'))
    # oven.start()
    oven = OvenPowerSupply(host, port, 4)
    # print(oven.query('*IDN1?'))
    oven.turn_on(current=1, voltage=1, timeout=10 * 60 * 1000)
    time.sleep(5)
    oven.turn_off()
    # oven.disable()
    # oven.disableWatchdog()
    # oven.start_shutdown_timer(10000)
    # oven.startTimer()

    # oven.openConnection()
    # session = SCPI_sock_connect(HOST)
    # SCPI_sock_send(session,"*IDN?")
    # dataReceived = getDataFromSocket(session)
    # print('Received: ',dataReceived)


if __name__ == "__main__":
    test_oven()

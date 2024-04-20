"""Applet class to control an ARTIQ device/controller."""
# import abc
import argparse
import asyncio
import logging
import os

import artiq.master.databases as db
import artiq.master.worker_db as mgr
import sipyco.sync_struct as artiq_sync
from artiq.applets.simple import AppletIPCClient
from quamash import QEventLoop
from quamash import QtCore
from quamash import QtWidgets

from euriqabackend import _EURIQA_LIB_DIR

_LOGGER = logging.getLogger(__name__)


class ControllerWidget(QtWidgets.QMainWindow):
    """Example/abstract class showing how to use the Controller applet."""

    def __init__(
        self,
        device_mgr: mgr.DeviceManager,
        cmd_args: argparse.Namespace,
        *args,
        **kwargs
    ):
        """Initialize the selected device."""
        super().__init__(*args, **kwargs)
        self.controlled_device = device_mgr.get(cmd_args.device_name)


class ControllerApplet:
    """Connect to and control an ARTIQ device/controller.

    Aware of device databases. Based on :class:`artiq.applets.simple.SimpleApplet`.
    """

    def __init__(self, main_widget_class, cmd_description: str = None):
        """Initialize an applet based on a passed widget.

        NOTE: must call :meth:`run` before it does anything.

        Args:
            main_widget_class (QtGui.QGraphicsView): An applet that controls an
                ARTIQ device.
            cmd_description (str, optional): Description of the applet to
                be shown on the command line. Defaults to None.
        """
        self.main_widget_class = main_widget_class

        self.argparser = argparse.ArgumentParser(
            description=cmd_description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        standalone_group = self.argparser.add_argument_group(
            "standalone mode (default)"
        )
        standalone_group.add_argument(
            "--server",
            default="::1",
            help="hostname or IP of the ARTIQ master to connect to "
            "for controller notifications & device database (ignored in embedded mode)",
        )
        standalone_group.add_argument(
            "--port", default=3250, type=int, help="ARTIQ Master TCP port to connect to"
        )
        # HACK: Used to hide main window when main applet launches sub-applets.
        standalone_group.add_argument(
            "--hide-main-window",
            "--hm",
            action="store_true",
            help="Do not display main window of applet (e.g. launching sub-applets)",
        )
        standalone_group.add_argument(
            "-v",
            "--verbosity",
            action="count",
            default=0,
            help="increase logging verbosity level (default=WARNING)",
        )
        # TODO: support getting devices from ARTIQ Master.
        # Look at :mod:`artiq.frontend.artiq_ctlmgr` for example
        # (which uses :class:`artiq.devices.ctlmgr.ControllerManager`)
        # issue right now is that state won't be shared between say a DACInterface in
        # this applet and a DAC interface on the master. Might need to make DACInterface
        # into a controller.

        device_args = self.argparser.add_argument_group("Device arguments")
        device_args.add_argument(
            "--device-db",
            "-db",
            help="Path to device database in which to find selected device",
        )
        device_args.add_argument(
            "device_name", type=str, help="Device to control (from device database)"
        )

        # internal state variables
        self.ipc = None
        self.subscriber = None
        self.args = None
        self.embed = None
        self.device_mgr = None
        self.loop = None
        self.main_widget = None

    def run(self):
        """Initialize & run the widget."""
        self._args_init()
        self._quamash_init()
        try:
            self._ipc_init()
            try:
                self._create_main_widget()
                if self.args.hide_main_window:
                    # HACK: Hides main window when main applet launches sub-applets.
                    # Mostly b/c I'm having trouble figuring out inheritance
                    self.main_widget.hide()
                # self._subscribe()
                try:
                    self.loop.run_forever()
                finally:
                    # self._unsubscribe()
                    pass
            finally:
                self._ipc_close()
        finally:
            self.loop.close()

    def _args_init(self):
        """Parse & process the arguments, starting up databases."""
        self.args = self.argparser.parse_args()
        self.embed = os.getenv("ARTIQ_APPLET_EMBED")
        logging.basicConfig(level=logging.WARNING - 10 * self.args.verbosity)
        _LOGGER.debug("Starting device manager/database: `%s`", self.args.device_db)
        self.device_mgr = mgr.DeviceManager(db.DeviceDB(self.args.device_db))

    def _quamash_init(self):
        """Initialize the GUI loops."""
        app = QtWidgets.QApplication([])
        self.loop = QEventLoop(app)
        asyncio.set_event_loop(self.loop)

    def _ipc_init(self):
        """Initialize interprocess communication to ARTIQ if embedded.

        IPC is faster than normal TCP/IP communication.
        """
        if self.embed is not None:
            self.ipc = AppletIPCClient(self.embed)
            self.loop.run_until_complete(self.ipc.connect())
            self.loop.run_until_complete(self.ipc.connect())

    def _ipc_close(self):
        """Close communication at end."""
        if self.embed is not None:
            self.ipc.close()

    def _create_main_widget(self):
        """Instantiate & show the main widget based on arguments passed."""
        self.main_widget = self.main_widget_class(self.device_mgr, self.args)
        if self.embed is not None:
            self.ipc.set_close_cb(self.main_widget.close)
            if os.name == "nt":
                # HACK: if the window has a frame, there will be garbage
                # (usually white) displayed at its right and bottom borders
                #  after it is embedded.
                self.main_widget.setWindowFlags(QtCore.Qt.FramelessWindowHint)
                self.main_widget.show()
                win_id = int(self.main_widget.winId())
                self.loop.run_until_complete(self.ipc.embed(win_id))
            else:
                # HACK:
                # Qt window embedding is ridiculously buggy, and empirical
                # testing has shown that the following procedure must be
                # followed exactly on Linux:
                # 1. applet creates widget
                # 2. applet creates native window without showing it, and
                #    gets its ID
                # 3. applet sends the ID to host, host embeds the widget
                # 4. applet shows the widget
                # 5. parent resizes the widget
                win_id = int(self.main_widget.winId())
                self.loop.run_until_complete(self.ipc.embed(win_id))
                self.main_widget.show()
                self.ipc.fix_initial_size()
        else:
            self.main_widget.show()

    def _subscribe(self):
        """Subscribe the the relevant devices."""
        if self.embed is None:
            self.subscriber = artiq_sync.Subscriber(
                "devices", self.sub_init, self.sub_mod
            )
            self.loop.run_until_complete(
                self.subscriber.connect(self.args.server, self.args.port)
            )
        else:
            self.ipc.subscribe(self.datasets, self.sub_init, self.sub_mod)

    def _unsubscribe(self):
        self.device_mgr.close_devices()
        if self.embed is None:
            self.loop.run_until_complete(self.subscriber.close())

"""GUI application to control the Sandia 100x DAC.

Similar to Sandia's VoltageControl applet in PyIonControl.

TODO:
    * I'm not sure if state is shared/preserved across instances of the DAC interface.
        i.e. can't share across computers?? But I think the same ctlmgr/device manager
        will share the same DAC_interface between local devices running from the same
        python kernel. So might work if spawned from artiq_dashboard, but not with
        my hack to add device_manager on.
        ***So need to link the device manager here to one from ARTIQ master/ctlmgr***
        Tried making dac_interface into a controller, but ARTIQ doesn't like
        making RPC controllers that share properties vs methods
        (i.e. PyQT signals, interface.hardware, interface.adjustment_dict, etc.)
        So a clever solution could be found, but doesn't work right now
    * would be nice to have some way to restore last-used file/file directory,
        or share state of dac_interface between applet & one used by the ARTIQ devices.
        not sure if this is possible if run through artiq_master.
    * not 100% tested, could still have some linking bugs due to differences
        in how we handle state/less use of PyQT signaling.
    * Code takes a while to launch the voltage Control GUI:
        VoltageAdjust.py:setupUi. setWidgets is big time hog, as well as show()
    * analog out channel numbers in voltage value table is broken.
        probably not linked right?

To run:
NOTE: you first need to launch a Sandia 100x DAC Controller:
```bash
aqctl_sandia_dac_100x -f "DAC box" -s
```
(for test) Might have to modify ``device_db.py`` to change
``sandia_dac`` device to IP: ``::1``.

Then, to launch applet (see below if running PC via Nix):
```bash
100x_dac_gui dac_pc_interface --device-db ./PATH/TO/DEVICE_DB.PY --hide-main-window
```

Using nix:
```bash
cd euriqa-artiq
nix-build
./result/bin/100x_dac_gui [...]
```
"""
import euriqafrontend.applets.common.controller_applet as ctl_applet
import euriqafrontend.applets.DAC100xControl.VoltageControl as voltage_applet


class ControlSandia100xDACWidget(ctl_applet.ControllerWidget):
    """Starts an interface for the Sandia 100x DAC."""

    def __init__(self, *args, **kwargs) -> None:
        """Start a Controller Widget that displays the Sandia 100x DAC adjustments."""
        super().__init__(*args, **kwargs)
        self.voltage_gui = voltage_applet.VoltageControl(
            config=dict(),
            dac_interface=self.controlled_device,
            parent=self,
            globalDict=dict(),
        )
        self.voltage_gui.setupUi(self.voltage_gui)
        self.voltage_gui.show()


def main() -> None:
    """Start the 100x DAC applet."""
    # Declare applet
    applet = ctl_applet.ControllerApplet(ControlSandia100xDACWidget)

    applet.run()


if __name__ == "__main__":
    main()

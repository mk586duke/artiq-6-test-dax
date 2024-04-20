"""Utility for speeding up debugging of Sandia DAC."""
import logging
import pathlib

import euriqabackend.utilities.debug as dbg
import euriqabackend.voltage_solutions as voltage_solutions
from euriqafrontend.modules.dac import SandiaDAC


class CommandLineDac(SandiaDAC):
    """Start a connection to the DAC.

    Shortcut so you don't have to type all of this every time.

    Returns:
        DAC interface (mediator). See
            :class:`euriqabackend.devices.sandia_dac.interface.SandiaDACInterface`.

    """

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        # start DAC controller.
        # not working. Must run below command from another command prompt
        # subprocess.Popen(["aqctl_sandia_dac_100x", "DAC Box", "-v"])
        # time.sleep(1)

        device_manager = dbg.setup_device_manager(
            "../../../euriqabackend/databases/device_db_main_box.py"
        )
        self._HasEnvironment__dataset_mgr = dbg.setup_database_manager(
            "../../../dataset_db.pyon"
        )
        self.path = voltage_solutions.__path__[0]  # For when __path__  returns a list
        print(self.path)
        self.module_path = pathlib.Path(voltage_solutions.__file__).resolve().parent

        self.dac_pc = device_manager.get("dac_pc_interface")
        self.dac_realtime = device_manager.get("realtime_sandia_dac")
        self.core = device_manager.get("core")

        self.pin_file = "EURIQA_socket_map.txt"
        self.voltage_file = "tilted_merge_relaxedLoading_PAD.txt"
        self.global_file = "EURIQA_MHz_units_plus_load_flat.txt"
        self.shuttle_file = "tilted_merge_relaxedLoading_PAD.xml"
        self.use_global_voltages = True
        self.prepare()
        self.dac_pc.send_voltage_lines_to_fpga()
        self.dac_pc.send_shuttling_lookup_table()

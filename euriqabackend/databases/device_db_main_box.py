"""Device database describing the main pulser/control box.

Designed to SYNCHRONOUSLY control TTLs (GPIO's) & DDS's as a master ARTIQ FPGA.
"""
# ******************************************************************************
#                            Core Address
# ******************************************************************************
CORE_IP_ADDRESS = "192.168.78.105"
CONTROL_PC_IP_ADDRESS =  "192.168.78.152"
OLD_WINDOWS_CONTROL_PC_IP_ADDRESS = "192.168.78.44"
HARRIS_PC_IP_ADDRESS = "192.168.78.143"


device_db = {
    # **************************************************************************
    #                            Functional Mapping
    # **************************************************************************
    # fmt: off

    "power_cool_1"       : "out1_0",
    "power_cool_2"       : "out1_1",
    "optical_pumping"    : "out1_2",    # on for detection, off for pumping
    "shutter_399"        : "out1_3",
    "shutter_mw"         : "out1_4",
    "awg_trigger"        : "out1_5",
    "global_aom_source"  : "out1_6",    # on for AWG, off for DDS
    "freq_shift_935"     : "out1_7",    # 935 frequency shifter switch. 1 = 172, 0 = 171
    "eom_935_172"        : "out1_7",    # Extra for compatibility w/ repo/loading.py
    "eom_935_3ghz"       : "out4_5",    # off = modulation, on = no modulation

    "shutter_370ion"     : "out2_1",
    "attenuator_c8"      : "out2_3",
    "attenuator_c16"     : "out2_4",
    "dac_trigger"        : "out2_6",
    "dac_serial"         : "sandia_dac_spi",    # physically connected to out2_7

    "shutter_394"       : "out3_2",
    # "out3_3"          : "out3_3",
    "sn_enable"         : "out3_4",  # Switch network - enable line
    "sn_running"        : "out3_5",  # Switch network - experiment running line
    "rf_lock_switch"        : "out3_6",  # Switch network - advance sequence line
    "sn_reset"          : "out3_7",  # Switch network - reset line

    "sma9"              : "out4_7",
    "awg_bit0"    :"out3_0",
    "awg_bit1"    :"out3_1",
    "awg_bit2"    :"out3_3",
    "awg_bit3"    :"out2_2",
    "awg_bit4"    :"out2_5",
    "awg_bit5"    :"out4_6",
    "awg_bit6"    :"out4_2", # was out4_7
    "aom_399"     :"out4_3",
    "aom_172_369" :"out4_4",
    "rfsoc_trigger" : "out2_0",
    "camera_flip" : "out4_1",
    "aoms_id"      : "out4_0",

    "dds0_switch"       : "sma0",
    "dds1_switch"       : "sma1",
    "dds2_switch"       : "sma2",
    "dds3_switch"       : "sma3",
    "dds4_switch"       : "sma4",
    "dds5_switch"       : "sma5",
    "dds6_switch"       : "sma6",
    "dds7_switch"       : "sma7",
    "dds8_switch"       : "sma8",
    "dds9_switch"       : "sma9",

    "microwave_dds"  : "w_dds0",
    "global_raman_dds"  : "w_dds1",

    "switchnet_dds"     : "w_dds2",
    "sd_435_dds1"            : "w_dds3",

    "pump_det_dds1"     : "w_dds4",
    "pump_det_dds2"     : "w_dds5",

    "cooling_dds1"      : "w_dds6",
    "cooling_dds2"      : "w_dds7",

    "cool_172_dds": "w_dds8",
    "sd_435_dds2": "w_dds9", # broken dds

    "pmt1"              : "in1_0",
    "pmt2"              : "in1_1",
    "pmt3"              : "in1_2",
    "pmt4"              : "in1_3",
    "pmt5"              : "in1_4",
    "pmt6"              : "in1_5",
    "pmt7"              : "in1_6",
    "pmt8"              : "in1_7",

    "pmt9"              : "in2_0",
    "pmt10"             : "in2_1",
    "pmt11"             : "in2_2",
    "pmt12"             : "in2_3",
    "pmt13"             : "in2_4",
    "pmt14"             : "in2_5",
    "pmt15"             : "in2_6",

    "pmt_3"             : "in3_0",
    "pmt_2"             : "in3_1",
    "pmt_1"             : "in3_2",
    "pmt0"              : "in2_7",

    "pmt16"             : "in3_6",
    "pmt17"             : "in3_3",
    "pmt18"             : "in3_4",
    "pmt19"             : "in3_5",

    "pmt1EdgeCounter"              : "in1_0_edgecounter",
    "pmt2EdgeCounter"              : "in1_1_edgecounter",
    "pmt3EdgeCounter"              : "in1_2_edgecounter",
    "pmt4EdgeCounter"              : "in1_3_edgecounter",
    "pmt5EdgeCounter"              : "in1_4_edgecounter",
    "pmt6EdgeCounter"              : "in1_5_edgecounter",
    "pmt7EdgeCounter"              : "in1_6_edgecounter",
    "pmt8EdgeCounter"              : "in1_7_edgecounter",

    "pmt9EdgeCounter"              : "in2_0_edgecounter",
    "pmt10EdgeCounter"             : "in2_1_edgecounter",
    "pmt11EdgeCounter"             : "in2_2_edgecounter",
    "pmt12EdgeCounter"             : "in2_3_edgecounter",
    "pmt13EdgeCounter"             : "in2_4_edgecounter",
    "pmt14EdgeCounter"             : "in2_5_edgecounter",
    "pmt15EdgeCounter"             : "in2_6_edgecounter",

    "pmt_3EdgeCounter"             : "in3_0_edgecounter",
    "pmt_2EdgeCounter"             : "in3_1_edgecounter",
    "pmt_1EdgeCounter"             : "in3_2_edgecounter",
    "pmt0EdgeCounter"              : "in2_7_edgecounter",

    "pmt16EdgeCounter"             : "in3_6_edgecounter",
    "pmt17EdgeCounter"             : "in3_3_edgecounter",
    "pmt18EdgeCounter"             : "in3_4_edgecounter",
    "pmt19EdgeCounter"             : "in3_5_edgecounter",

    "line_trigger"      : "in3_7",

    # fmt: on
    # **************************************************************************
    #                         Instrument Controllers
    #                       i.e. Oven, Conex, Amp, DAC
    #                             DO NOT CHANGE
    # **************************************************************************
    "harris_multichannel": {
        "type": "controller",
        "best_effort": True,
        "host": HARRIS_PC_IP_ADDRESS,
        "port": 3273,
        "command": "aqctl_multiaom_harris --bind {bind} --port {port}",
    },
    "harris_global": {
        "type": "controller",
        "best_effort": True,
        "host": HARRIS_PC_IP_ADDRESS,
        "port": 3272,
        "command": "aqctl_globalaom_harris --bind {bind} --port {port}",
    },
    # "conex_controller_sim0": {
    #     "type": "controller",
    #     "best_effort": True,
    #     "host": HARRIS_PC_IP_ADDRESS,
    #     "port": 3274,
    #     "command": "aqctl_conex -x=COM11 -y=COM13 -s --bind {bind} --port {port}",
    # },
    "sandia_dac": {
        "type": "controller",
        "best_effort": False,
        "host": OLD_WINDOWS_CONTROL_PC_IP_ADDRESS,
        "port": 3270,
        "command": 'aqctl_sandia_dac_100x --fpga "DAC box" '
        + "--bind {bind} --port {port}",
    },
    # Runs locally on ARTIQ Master/Control PC
    "dac_pc_interface": {
        "type": "local",
        "module": "euriqabackend.devices.sandia_dac.interface",
        "class": "SandiaDACInterface",
        "arguments": {"dac_device": "sandia_dac"},
    },
    "magfield_x": {
        "type": "controller",
        "best_effort": False,
        "host": CONTROL_PC_IP_ADDRESS,
        "port": 3280,
        "command": 'aqctl_n6700b --psu-ip="192.168.80.11" --channel=1 --bind {bind} --port {port}',
    },
    "magfield_y": {
        "type": "controller",
        "best_effort": False,
        "host": CONTROL_PC_IP_ADDRESS,
        "port": 3281,
        "command": 'aqctl_n6700b --psu-ip="192.168.80.11" --channel=2 --bind {bind} --port {port}',
    },
    "magfield_z": {
        "type": "controller",
        "best_effort": False,
        "host": CONTROL_PC_IP_ADDRESS,
        "port": 3282,
        "command": 'aqctl_n6700b --psu-ip="192.168.80.11" --channel=3 --bind {bind} --port {port}',
    },
    "yb_oven": {
        "type": "controller",
        "best_effort": False,
        "host": CONTROL_PC_IP_ADDRESS,
        "port": 3276,
        "command": 'aqctl_oven --oven-ip="192.168.80.11" --channel=4 --bind {bind} --port {port}',
    },
    "rf_compiler": {
        "type": "controller",
        "best_effort": False,
        "host": "192.168.80.9",
        "port": 3277,
        "timeout": 30,
        "command": "aqctl_rfcompiler --bind {bind} --port {port}",
    },
    # **************************************************************************
    #                       Hardware (FPGA) Devices
    #                             DO NOT CHANGE
    # **************************************************************************
    # fmt: off
    "core": {
        "type": "local",
        "module": "artiq.coredevice.core",
        "class": "Core",
        "arguments": {
            "ref_period": 1e-9,
            "host": CORE_IP_ADDRESS,
        }
    },
    # fmt: on
    # "influxdb": {
    #     "type": "controller",
    #     "host": '192.168.78.144',
    #     "port": 3248,
    #     "commmand": "artiq_influxdb --baseurl-db=\"http://192.168.78.111:8086\" --user-db=editor --password-db=LogiQYbIons " +
    #                 "--database=artiq --pattern-file=\"./influxdb_patterns.cfg\""
    # },
    "core_log": {
        "type": "controller",
        "host": "::1",
        "port": 1068,
        "command": "aqctl_corelog -p {port} --bind {bind} " + CORE_IP_ADDRESS,
    },
    "core_cache": {
        "type": "local",
        "module": "artiq.coredevice.cache",
        "class": "CoreCache",
    },
    # fmt: off
    "core_dma": {
        "type": "local",
        "module": "artiq.coredevice.dma",
        "class": "CoreDMA",
    },
    # fmt: on
    # *** FPGA RTIO Channels ***
    "out1_0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 0},
        "comment": "out_1_0",
    },
    "out1_1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 1},
    },
    "out1_2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 2},
    },
    "out1_3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 3},
    },
    "out1_4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 4},
    },
    "out1_5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 5},
    },
    "out1_6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 6},
    },
    "out1_7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 7},
    },
    "out2_0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 8},
        "comment": "This is a comment",
    },
    "out2_1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 9},
    },
    "out2_2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 10},
    },
    "out2_3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 11},
    },
    "out2_4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 12},
    },
    "out2_5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 13},
    },
    "out2_6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 14},
    },
    "out2_7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 15},
    },
    "out3_0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 16},
        "comment": "This is a comment",
    },
    "out3_1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 17},
    },
    "out3_2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 18},
    },
    "out3_3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 19},
    },
    "out3_4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 20},
    },
    "out3_5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 21},
    },
    "out3_6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 22},
    },
    "out3_7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 23},
    },
    "out4_0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 24},
        "comment": "This is a comment",
    },
    "out4_1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 25},
    },
    "out4_2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 26},
    },
    "out4_3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 27},
    },
    "out4_4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 28},
    },
    "out4_5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 29},
    },
    "out4_6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 30},
    },
    "out4_7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 31},
    },
    "in1_0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 32},
    },
    "in1_0_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 32},
    },
    "in1_1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 33},
    },
    "in1_1_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 33},
    },
    "in1_2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 34},
    },
    "in1_2_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 34},
    },
    "in1_3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 35},
    },
    "in1_3_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 35},
    },
    "in1_4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 36},
    },
    "in1_4_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 36},
    },
    "in1_5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 37},
    },
    "in1_5_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 37},
    },
    "in1_6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 38},
    },
    "in1_6_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 38},
    },
    "in1_7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 39},
    },
    "in1_7_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 39},
    },
    "in2_0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 40},
    },
    "in2_0_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 40},
    },
    "in2_1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 41},
    },
    "in2_1_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 41},
    },
    "in2_2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 42},
    },
    "in2_2_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 42},
    },
    "in2_3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 43},
    },
    "in2_3_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 43},
    },
    "in2_4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 44},
    },
    "in2_4_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 44},
    },
    "in2_5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 45},
    },
    "in2_5_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 45},
    },
    "in2_6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 46},
    },
    "in2_6_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 46},
    },
    "in2_7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 47},
    },
    "in2_7_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 47},
    },
    "in3_0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 48},
    },
    "in3_0_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 48},
    },
    "in3_1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 49},
    },
    "in3_1_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 49},
    },
    "in3_2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 50},
    },
    "in3_2_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 50},
    },
    "in3_3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 51},
    },
    "in3_3_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 51},
    },
    "in3_4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 52},
    },
    "in3_4_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 52},
    },
    "in3_5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 53},
    },
    "in3_5_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 53},
    },
    "in3_6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 54},
    },
    "in3_6_edgecounter": {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {"channel": 54},
    },
    "in3_7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {"channel": 55},
    },
    "led0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 56},
    },
    "led1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 57},
    },
    "oeb": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 58},
    },
    "sma0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 59},
    },
    "sma1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 60},
    },
    "sma2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 61},
    },
    "sma3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 62},
    },
    "sma4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 63},
    },
    "sma5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 64},
    },
    "sma6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 65},
    },
    "sma7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 66},
    },
    "sma8": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 67},
    },
    "io_update0": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 68},
    },
    "io_update1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 69},
    },
    "io_update2": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 70},
    },
    "io_update3": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 71},
    },
    "io_update4": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 72},
    },
    "io_update5": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 73},
    },
    "io_update6": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 74},
    },
    "io_update7": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 75},
    },
    "io_update8": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 76},
    },
    "io_update9": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 77},
    },
    "reset01": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 78},
    },
    "reset23": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 79},
    },
    "reset45": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 80},
    },
    "reset67": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 81},
    },
    "reset89": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 82},
    },
    "spi0": {
        "type": "local",
        "module": "artiq.coredevice.spi2",
        "class": "SPIMaster",
        "arguments": {"channel": 83},
    },
    "spi1": {
        "type": "local",
        "module": "artiq.coredevice.spi2",
        "class": "SPIMaster",
        "arguments": {"channel": 84},
    },
    "spi2": {
        "type": "local",
        "module": "artiq.coredevice.spi2",
        "class": "SPIMaster",
        "arguments": {"channel": 85},
    },
    "spi3": {
        "type": "local",
        "module": "artiq.coredevice.spi2",
        "class": "SPIMaster",
        "arguments": {"channel": 86},
    },
    "spi4": {
        "type": "local",
        "module": "artiq.coredevice.spi2",
        "class": "SPIMaster",
        "arguments": {"channel": 87},
    },
    # DAC 8568 8x DAC control lines
    "spi5": {
        "type": "local",
        "module": "artiq.coredevice.spi2",
        "class": "SPIMaster",
        "arguments": {"channel": 88},
    },
    "load_dac_1": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {"channel": 89},
    },
    # unused channel for rtio_log
    "sandia_dac_spi": {
        "type": "local",
        "module": "artiq.coredevice.spi2",
        "class": "SPIMaster",
        "arguments": {"channel": 90},
    },
    # *** FPGA Coredevice/Kernel Drivers ***
    "dds0": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi0",
            "io_update": "io_update0",
            "chip_select": 1,
        },
    },
    "dds1": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi0",
            "io_update": "io_update1",
            "chip_select": 2,
        },
    },
    "dds2": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi1",
            "io_update": "io_update2",
            "chip_select": 1,
        },
    },
    "dds3": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi1",
            "io_update": "io_update3",
            "chip_select": 2,
        },
    },
    "dds4": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi2",
            "io_update": "io_update4",
            "chip_select": 1,
        },
    },
    "dds5": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi2",
            "io_update": "io_update5",
            "chip_select": 2,
        },
    },
    "dds6": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi3",
            "io_update": "io_update6",
            "chip_select": 1,
        },
    },
    "dds7": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi3",
            "io_update": "io_update7",
            "chip_select": 2,
        },
    },
    "dds8": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi4",
            "io_update": "io_update8",
            "chip_select": 1,
        },
    },
    "dds9": {
        "type": "local",
        "module": "euriqabackend.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "spi_device": "spi4",
            "io_update": "io_update9",
            "chip_select": 2,
        },
    },
    "dac8568_1": {
        "type": "local",
        "module": "euriqabackend.coredevice.dac8568",
        "class": "DAC8568",
        "arguments": {
            "spi_device": "spi5",
            "chip_select": 1,
            "ldac_trigger": "load_dac_1",
            "v_out_max": 5.0,
        },
    },
    # *** UNUSED/UNTESTED (commented-out) ***
    # "testSPI": {
    #     "type": "local",
    #     "module": "artiq.coredevice.TestSPI",
    #     "class": "TestSPI",
    #     "arguments": {"spi_device": "spi0", "chip_select": 1},
    # },
    # "pmt_vectorgroup_test": {
    #     "type": "local",
    #     "module": "euriqabackend.utilities.grouped_devices",
    #     "class": "VectorGrouper",
    #     "arguments": {
    #         "wrapped_objs": [
    #             "in1_0",
    #             "in1_1",
    #             "in1_2",
    #             "in1_3",
    #             "in1_4",
    #             "in1_5",
    #             "in1_6",
    #             "in1_7",
    #         ]
    #     },
    # },
    # "pmt_array_in1": {
    #     "type": "local",
    #     "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
    #     "class": "WrappedInputArray",
    #     "arguments": {
    #         "counters": [
    #             "in1_0",
    #             "in1_1",
    #             "in1_2",
    #             "in1_3",
    #             "in1_4",
    #             "in1_5",
    #             "in1_6",
    #             "in1_7",
    #         ]
    #     },
    # },
    "cooling_power": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedSteppedAttenuator",
        "arguments": {
            "switches": ["out1_0", "out1_1"],
            "lut": [0b00, 0b01, 0b11, 0b10],
        },
    },
    "w_dds0": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds0",
            "dds_switch_device": "dds0_switch",
            "dds_reset_device": "reset01",
            "bus_group": 1,
        },
    },
    "w_dds1": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds1",
            "dds_switch_device": "dds1_switch",
            "dds_reset_device": "reset01",
            "bus_group": 2,
        },
    },
    "w_dds2": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds2",
            "dds_switch_device": "dds2_switch",
            "dds_reset_device": "reset23",
            "bus_group": 1,
        },
    },
    "w_dds3": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds3",
            "dds_switch_device": "dds3_switch",
            "dds_reset_device": "reset23",
            "bus_group": 2,
        },
    },
    "w_dds4": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds4",
            "dds_switch_device": "dds4_switch",
            "dds_reset_device": "reset45",
            "bus_group": 1,
        },
    },
    "w_dds5": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds5",
            "dds_switch_device": "dds5_switch",
            "dds_reset_device": "reset45",
            "bus_group": 2,
        },
    },
    "w_dds6": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds6",
            "dds_switch_device": "dds6_switch",
            "dds_reset_device": "reset67",
            "bus_group": 1,
        },
    },
    "w_dds7": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds7",
            "dds_switch_device": "dds7_switch",
            "dds_reset_device": "reset67",
            "bus_group": 2,
        },
    },
    "w_dds8": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds8",
            "dds_switch_device": "dds8_switch",
            "dds_reset_device": "reset89",
            "bus_group": 1,
        },
    },
    "w_dds9": {
        "type": "local",
        "module": "euriqabackend.devices.wrapped_core_hardware.mediator",
        "class": "WrappedDDSHardware",
        "arguments": {
            "dds_device": "dds9",
            "dds_switch_device": "dds9_switch",
            "dds_reset_device": "reset89",
            "bus_group": 2,
        },
    },
    "realtime_sandia_dac": {
        "type": "local",
        "module": "euriqabackend.coredevice.sandia_dac_core",
        "class": "DACSerialCoredevice",
        "arguments": {"serial_device": "dac_serial", "trigger_line": "dac_trigger"},
    },
}

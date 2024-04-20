"""Extension card for the EURIQA project.

Describes the "FMC" adapter that plugs into the KC705 and interfaces with the
Sandia/Duke Pulser hardware: DDS's, ADC, DAC, GPIO (TTL), etc.

Description of the pins & interfaces available on the FMC connector.
"""
from migen.build.generic_platform import IOStandard
from migen.build.generic_platform import Pins
from migen.build.generic_platform import Subsignal

fmc_adapter_io = [
    # Input Bank 1
    ("in1", 0, Pins("HPC:LA17_CC_P"), IOStandard("LVCMOS33")),  # JP2-20
    ("in1", 1, Pins("HPC:LA14_P"), IOStandard("LVCMOS33")),  # JP2-19
    ("in1", 2, Pins("HPC:LA16_N"), IOStandard("LVCMOS33")),  # JP2-18
    ("in1", 3, Pins("HPC:LA14_N"), IOStandard("LVCMOS33")),  # JP2-17
    ("in1", 4, Pins("HPC:LA16_P"), IOStandard("LVCMOS33")),  # JP2-16
    ("in1", 5, Pins("HPC:LA15_P"), IOStandard("LVCMOS33")),  # JP2-15
    ("in1", 6, Pins("HPC:LA05_P"), IOStandard("LVCMOS33")),  # JP2-59
    ("in1", 7, Pins("HPC:HA07_P"), IOStandard("LVCMOS33")),  # JP3-46
    # Input Bank 2
    ("in2", 0, Pins("HPC:LA19_P"), IOStandard("LVCMOS33")),  # JP2-28
    ("in2", 1, Pins("HPC:LA12_P"), IOStandard("LVCMOS33")),  # JP2-27
    ("in2", 2, Pins("HPC:LA18_CC_N"), IOStandard("LVCMOS33")),  # JP2-26
    ("in2", 3, Pins("HPC:LA12_N"), IOStandard("LVCMOS33")),  # JP2-25
    ("in2", 4, Pins("HPC:LA18_CC_P"), IOStandard("LVCMOS33")),  # JP2-24
    ("in2", 5, Pins("HPC:LA13_P"), IOStandard("LVCMOS33")),  # JP2-23
    ("in2", 6, Pins("HPC:LA17_CC_N"), IOStandard("LVCMOS33")),  # JP2-22
    ("in2", 7, Pins("HPC:LA13_N"), IOStandard("LVCMOS33")),  # JP2-21
    # Input Bank 3
    ("in3", 0, Pins("HPC:LA31_N"), IOStandard("LVCMOS33")),  # JP2-58
    ("in3", 1, Pins("HPC:LA29_N"), IOStandard("LVCMOS33")),  # JP2-54
    ("in3", 2, Pins("HPC:LA31_P"), IOStandard("LVCMOS33")),  # JP2-52
    ("in3", 3, Pins("HPC:LA29_P"), IOStandard("LVCMOS33")),  # JP2-50
    ("in3", 4, Pins("HPC:LA30_N"), IOStandard("LVCMOS33")),  # JP2-48
    ("in3", 5, Pins("HPC:LA28_N"), IOStandard("LVCMOS33")),  # JP2-46
    ("in3", 6, Pins("HPC:LA30_P"), IOStandard("LVCMOS33")),  # JP2-44
    ("in3", 7, Pins("HPC:LA28_P"), IOStandard("LVCMOS33")),  # JP2-42
    # Output Bank 1
    ("out1", 0, Pins("HPC:LA27_N"), IOStandard("LVCMOS33")),  # JP3-65
    ("out1", 1, Pins("LPC:LA21_N"), IOStandard("LVCMOS33")),  # JP3-67 # EXT4
    ("out1", 2, Pins("LPC:LA20_N"), IOStandard("LVCMOS33")),  # JP3-69 # EXT2
    ("out1", 3, Pins("HPC:HA18_N"), IOStandard("LVCMOS33")),  # JP3-71
    ("out1", 4, Pins("HPC:HA18_P"), IOStandard("LVCMOS33")),  # JP3-73
    ("out1", 5, Pins("HPC:HA17_CC_N"), IOStandard("LVCMOS33")),  # JP3-75
    ("out1", 6, Pins("HPC:HA17_CC_P"), IOStandard("LVCMOS33")),  # JP3-77
    ("out1", 7, Pins("HPC:HA16_N"), IOStandard("LVCMOS33")),  # JP3-79
    # Output Bank 2
    ("out2", 0, Pins("HPC:LA25_N"), IOStandard("LVCMOS33")),  # JP3-47
    ("out2", 1, Pins("HPC:LA22_P"), IOStandard("LVCMOS33")),  # JP3-49
    ("out2", 2, Pins("HPC:LA26_P"), IOStandard("LVCMOS33")),  # JP3-51
    ("out2", 3, Pins("HPC:LA22_N"), IOStandard("LVCMOS33")),  # JP3-53
    ("out2", 4, Pins("HPC:LA26_N"), IOStandard("LVCMOS33")),  # JP3-57
    ("out2", 5, Pins("HPC:LA23_P"), IOStandard("LVCMOS33")),  # JP3-59
    ("out2", 6, Pins("HPC:LA27_P"), IOStandard("LVCMOS33")),  # JP3-61
    ("out2", 7, Pins("HPC:LA23_N"), IOStandard("LVCMOS33")),  # JP3-63
    # This is out8 on the board, but out3 on the front panel
    ("out3", 0, Pins("HPC:HA23_P"), IOStandard("LVCMOS33")),  # JP2-70
    ("out3", 1, Pins("HPC:LA02_N"), IOStandard("LVCMOS33")),  # JP2-69
    ("out3", 2, Pins("HPC:HA21_P"), IOStandard("LVCMOS33")),  # JP2-68
    ("out3", 3, Pins("HPC:LA03_P"), IOStandard("LVCMOS33")),  # JP2-67
    ("out3", 4, Pins("HPC:HA22_N"), IOStandard("LVCMOS33")),  # JP2-66
    ("out3", 5, Pins("HPC:HA20_N"), IOStandard("LVCMOS33")),  # JP2-64
    ("out3", 6, Pins("HPC:HA22_P"), IOStandard("LVCMOS33")),  # JP2-62
    ("out3", 7, Pins("HPC:HA20_P"), IOStandard("LVCMOS33")),  # JP2-60
    # This is out9 on the board, but out4 on the front panel
    ("out4", 0, Pins("HPC:LA00_CC_P"), IOStandard("LVCMOS33")),  # JP2-79
    ("out4", 1, Pins("HPC:LA00_CC_N"), IOStandard("LVCMOS33")),  # JP2-77
    ("out4", 2, Pins("HPC:HA10_P"), IOStandard("LVCMOS33")),  # JP2-76
    ("out4", 3, Pins("HPC:LA01_CC_P"), IOStandard("LVCMOS33")),  # JP2-75
    ("out4", 4, Pins("HPC:HA23_N"), IOStandard("LVCMOS33")),  # JP2-74
    ("out4", 5, Pins("HPC:LA01_CC_N"), IOStandard("LVCMOS33")),  # JP2-73
    ("out4", 6, Pins("HPC:HA21_N"), IOStandard("LVCMOS33")),  # JP2-72
    ("out4", 7, Pins("HPC:LA02_P"), IOStandard("LVCMOS33")),  # JP2-71
    # SMA output/input??
    ("sma", 0, Pins("HPC:LA11_P"), IOStandard("LVCMOS33")),  # JP2-31
    ("sma", 1, Pins("HPC:LA33_P"), IOStandard("LVCMOS33")),  # JP2-34
    ("sma", 2, Pins("HPC:LA19_N"), IOStandard("LVCMOS33")),  # JP2-30
    ("sma", 3, Pins("HPC:LA10_N"), IOStandard("LVCMOS33")),  # JP2-33
    ("sma", 4, Pins("HPC:LA11_N"), IOStandard("LVCMOS33")),  # JP2-29
    ("sma", 5, Pins("HPC:LA33_N"), IOStandard("LVCMOS33")),  # JP2-32
    ("sma", 6, Pins("HPC:LA10_P"), IOStandard("LVCMOS33")),  # JP2-37
    ("sma", 7, Pins("HPC:LA32_N"), IOStandard("LVCMOS33")),  # JP2-38
    ("sma", 8, Pins("HPC:LA09_N"), IOStandard("LVCMOS33")),  # JP2-39
    # OEB = Output enable buffer
    ("oeb", 0, Pins("HPC:LA32_P"), IOStandard("LVCMOS33")),  # JP2-40
    # trigger for updating output of the DDS
    ("io_update", 0, Pins("HPC:LA25_P"), IOStandard("LVCMOS33")),  # JP3-43
    ("io_update", 1, Pins("HPC:HA09_P"), IOStandard("LVCMOS33")),  # JP3-38
    ("io_update", 2, Pins("HPC:LA24_P"), IOStandard("LVCMOS33")),  # JP3-33
    ("io_update", 3, Pins("HPC:HA11_N"), IOStandard("LVCMOS33")),  # JP3-28
    ("io_update", 4, Pins("HPC:HA04_P"), IOStandard("LVCMOS33")),  # JP3-25
    ("io_update", 5, Pins("HPC:HA13_N"), IOStandard("LVCMOS33")),  # JP3-20
    ("io_update", 6, Pins("HPC:HA02_P"), IOStandard("LVCMOS33")),  # JP3-17
    ("io_update", 7, Pins("HPC:LA04_N"), IOStandard("LVCMOS33")),  # JP2_61
    ("io_update", 8, Pins("HPC:LA06_N"), IOStandard("LVCMOS33")),  # JP2_51
    ("io_update", 9, Pins("HPC:LA09_P"), IOStandard("LVCMOS33")),  # JP2_41
    # Resets for the DDS control boards
    ("reset", 0, Pins("HPC:LA21_P"), IOStandard("LVCMOS33")),  # JP3-41
    ("reset", 1, Pins("HPC:LA20_P"), IOStandard("LVCMOS33")),  # JP3-31
    ("reset", 2, Pins("HPC:HA03_N"), IOStandard("LVCMOS33")),  # JP3-23
    ("reset", 3, Pins("HPC:HA01_CC_N"), IOStandard("LVCMOS33")),  # JP3-15
    ("reset", 4, Pins("HPC:LA07_N"), IOStandard("LVCMOS33")),  # JP2_47
    # data channel for odd DDS channels on the DDS board,
    # Because the data channels are separated
    # the even channels are in the mosi Subsignal of the SPI busses
    ("odd_channel_sdio", 0, Pins("HPC:LA24_N"), IOStandard("LVCMOS33")),  # JP3-39
    ("odd_channel_sdio", 1, Pins("HPC:HA19_N"), IOStandard("LVCMOS33")),  # JP3-29
    ("odd_channel_sdio", 2, Pins("HPC:HA03_P"), IOStandard("LVCMOS33")),  # JP3-21
    ("odd_channel_sdio", 3, Pins("HPC:LA04_P"), IOStandard("LVCMOS33")),  # JP2_63
    ("odd_channel_sdio", 4, Pins("HPC:LA08_N"), IOStandard("LVCMOS33")),  # JP2_43
    # SPI outputs
    (
        "spi",
        0,
        Subsignal("clk", Pins("HPC:HA08_P")),  # JP3-42
        Subsignal("cs_n", Pins("HPC:LA21_N HPC:HA08_N")),  # JP3-45, JP3-40
        Subsignal("mosi", Pins("HPC:HA07_N")),  # JP3-44
        IOStandard("LVCMOS33"),
    ),
    # SPI channels for communicating to DDS boards
    (
        "spi",
        1,
        Subsignal("clk", Pins("HPC:HA10_N")),  # JP3-32
        Subsignal("cs_n", Pins("HPC:LA20_N HPC:HA11_P")),  # JP3-37, JP3-30
        Subsignal("mosi", Pins("HPC:HA09_N")),  # JP3-34
        IOStandard("LVCMOS33"),
    ),
    (
        "spi",
        2,
        Subsignal("clk", Pins("HPC:HA12_N")),  # JP3-24
        Subsignal("cs_n", Pins("HPC:HA19_P HPC:HA13_P")),  # JP3-27, JP3-22
        Subsignal("mosi", Pins("HPC:HA12_P")),  # JP3-26
        IOStandard("LVCMOS33"),
    ),
    (
        "spi",
        3,
        Subsignal("clk", Pins("HPC:HA14_N")),  # JP3-16
        Subsignal("cs_n", Pins("HPC:HA02_N HPC:LA03_N")),  # JP3-19, JP2_65
        Subsignal("mosi", Pins("HPC:HA14_P")),  # JP3-18
        IOStandard("LVCMOS33"),
    ),
    (
        "spi",
        4,
        Subsignal("clk", Pins("HPC:LA07_P")),  # JP2_49
        Subsignal("cs_n", Pins("HPC:LA05_N HPC:LA08_P")),  # JP2_57, JP2_45
        Subsignal("mosi", Pins("HPC:LA06_P")),  # JP2_53
        IOStandard("LVCMOS33"),
    ),
    # DAC8568 Control pins: SPI & Load DAC TTL/GPIO trigger (LDAC)
    (
        "spi",
        5,
        Subsignal("clk", Pins("HPC:HA00_CC_P")),  # JP2-76
        Subsignal("cs_n", Pins("LPC:LA20_P")),  # JP2-70 (EXT-1, LPC board J20)
        Subsignal("mosi", Pins("HPC:HA00_CC_N")),  # JP2-74
        IOStandard("LVCMOS33"),
    ),
    ("ldac", 0, Pins("LPC:LA21_P"), IOStandard("LVCMOS33")),  # JP2-68 (EXT-3, LPC J20)
]

x100_dac_spi = [
    # SPI to hack UART-like communication to Sandia DAC. Takes over out2-7.
    # SHOULD NOT ALWAYS BE USED. Only if using real-time comm to 100x DAC.
    # Only relevant pin is "miso", overwrites the SD card pins.
    # COULD SCREW UP YOUR SD CARD. CAREFUL
    (
        "spi",
        6,
        Subsignal("clk", Pins("AB22")),  # Unassigned # AB22 = SD Card MOSI
        Subsignal("mosi", Pins("HPC:LA23_N")),  # JP3-63
        Subsignal("cs_n", Pins("AC21")),  # AC21 = SD Card CS_n
        IOStandard("LVCMOS33"),
    ),
    # unused acts as dummy pin, not routed to relevant location.
    ("unused", 0, Pins("AC20"), IOStandard("LVCMOS33")),  # AC20 = SD Card MISO
]

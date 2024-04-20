# ARTIQ notes

Report by _Jonathan Mizrahi_

## Compiling from source:

follow the instructions here: https://m-labs.hk/artiq/manual-master/installing_from_source.html

You also need to get cmake (not listed) if you don’t already have it:
$ apt-get cmake

You also need Vivado, with a license to compile for the Kintex7. Such a license comes with the KC705, but you only get a license for ONE computer, and you can only move it five times. Make sure you use Vivado and not ISE, I had bad experiences with ISE compilations failing. This means that when you build ARTIQ, you need the –toolchain vivado option.

I wrote a little python script for compilation that issues the command and also saves the output to a log and measures how long compilation takes. It is in /artiq-work/build-artiq.py. On my computer it’s about 7 minutes to compile.

To flash the board, make sure the SW13 switches are in state 00001, where towards the PCI connector and LCD display is 0. I also wrote a python script for gathering the binaries and flashing the board, it is artiq-work/flash_binaries.py. It takes about 1 minute to flash.

The serial connection is a nice way to test that ARTIQ loaded. If you push ‘t’ after boot up, you can run some simple test commands, like turning on and off the LEDs and looking at what’s in the memory.  Type ‘help’ to see commands.

After you plug into the serial port, you need to know what device number it is on the Linux filesystem to use flterm (i.e. /dev/ttyUSB0 or something else). To do this you can try:

option 1)
unplug/replug, then:
$ dmesg

option 2)

```bash
#!/bin/bash

for sysdevpath in $(find /sys/bus/usb/devices/usb*/ -name dev); do
    (
        syspath="${sysdevpath%/dev}"
        devname="$(udevadm info -q name -p $syspath)"
        [[ "$devname" == "bus/"* ]] && continue
        eval "$(udevadm info -q property --export -p $syspath)"
        [[ -z "$ID_SERIAL" ]] && continue
        echo "/dev/$devname - $ID_SERIAL"
    )
done
```

To communicate with the board over Ethernet, you need to flash the IP and MAC address to the flash memory.

## Modifying gateware

### Pin Mapping

The base pin mapping file is found in the migen repository, under /migen/build/platforms/kc705.py. This file is a bit analogous to a UCF file. It provides a name and voltage standard for the various resources on the kc705 board. In the artiq repository, the file /artiq/gateware/euriqa.py (or nist_clock.py, etc.) extends the migen kc705.py file, providing names for pins that are connected to the FMC connectors. This is different for different hardware configurations, hence there are different files. These files also can provide names for higher level objects with subsignals. For example, an SPI bus has 3 or 4 signals, with the names “clk,” “cs_n,” “mosi,” and “miso.” This is specified in the platform file as one spi object, with four subsignals connected to specific pins.

### Top Level Gateware

The top level module, which is actually built into the gateware, is in /artiq/gateware/targets/kc705.py. It has different classes for the different hardware configurations, and when the gateware is built, you specify a hardware target (via the -H option). This file is where all the resources specified in the platform files (/migen/build/platforms/kc705.py and /artiq/gateware/euriqa.py) are actually used, via the platform.request() command. Each requested object or pin is passed to a specific physical object type (or “phy”), specified in /artiq/gateware/rtio/phy, and added as a submodule. It is also added to the list of rtio_channels. The channel number in the device_db.pyon file is index in this list.
The device_db.pyon file allows you to specify names that correspond to various resources, including the RTIO channel numbers set up in /artiq/gateware/targets/kc705.py. You don’t have to list everything. You can also set up aliases, so that multiple names refer to the same resource (such as ttl0 and io_update mapping to the same physical resource). Devices specified in device_db.pyon typically have an associated class, specified in artiq/coredevice for things directly controlled by the FPGA (which ARTIQ calls the “core device”), or in artiq/devices for things controlled by the computer (what we’ve been calling “external parameters”).

## SPI bus (Used for DDS and other devices)

The control over SPI devices is very abstracted, which is nice. First, you have an SPI device in the platform file (/artiq/gateware/euriqa.py), which specify which SPI pins are connected to which physical pins. Then, in the top level module (/artiq/gateware/targets/kc705.py), you request the SPI device via self.platform.request(“spi,” i) for device I, and you pass that to the SPI object type constructor (spi.SPIMaster). That SPI device is then added to your device_db.pyon file, and you have access to an SPI bus.
Next, I wrote the driver for the AD9912, artiq/coredevice/ad9912.py. It has methods that do three things: Firt, setup the SPI bus, by telling it the SPI config parameters, clock speed, chip select line, and how many bits to send in one transaction. Then it has a load method to pulse the io_update line, and a set method to actually program the 9912. Because of the high level SPI interface, all you have to do is call self.bus.write(instruction), and it is appropriately clocked out over the SPI bus.
Finally, this device is added to the device_db.pyon file, with arguments specifying the SPI bus to use, and the io_update line to use.

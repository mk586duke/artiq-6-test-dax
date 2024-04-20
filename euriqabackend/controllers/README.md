# ARTIQ Front-End & Controllers

## Overview

These are controllers that provide interfaces from ARTIQ to a lab hardware device.
Most are meant to be run from the command line.

## Development

1. Create a file for your device.
2. Add the script to *REPO_TOP_DIRECTORY/setup.py*, in the *console_scripts* variable.
3. Install the ARTIQ package using pip.
    * Alternatively, run the script using Python from the command line.

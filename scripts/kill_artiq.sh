#! /bin/bash
# kill all ARTIQ & Python processes. Could have a few false positives, so be careful of using.
pkill python
pkill aqctl
pkill artiq

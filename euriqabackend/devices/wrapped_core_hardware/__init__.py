"""Package for "grouped"/"wrapped" hardware.

Wrapped/grouped hardware provides a "facade" to the more complicated hardware
underneath, or joins multiple devices together into one meta-device.
For example, the DDS boards offer resets and switches to turn on/off outputs.
To simplify controlling them, there is a class "WrappedDDS".
"""

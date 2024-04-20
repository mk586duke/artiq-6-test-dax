"""Test the full RF Compiler stack.

(theoretically... doesn't appear to have RFCompiler.full_stack()).
"""
from euriqabackend.devices.keysight_awg import RFCompiler as rfc

if __name__ == "__main__":
    r = rfc.RFCompiler()
    r.full_stack()

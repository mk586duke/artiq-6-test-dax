"""Waveforms to be used in EURIQA.

Divided into single-qubit and multi-qubit waveforms.

These functions are all supposed to run within the post-qiskit-terra v0.15.1
:func:`qiskit.pulse.build` pulse-builder contexts, and will cause unexpected
errors if not properly called.

TODO: each function does (does not??) actually return anything, but instead
modifies the schedule currently being built in the higher level.
"""

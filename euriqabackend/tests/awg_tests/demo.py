import numpy as np

import euriqabackend.devices.keysight_awg.AOM_calibration as aom_cal
import euriqabackend.devices.keysight_awg.sequence as seq
import euriqabackend.devices.keysight_awg.waveform_prototypes as pt

waveform_dir = r"C:\Users\logiq\Desktop\AWG Testing\Generated Waveforms"

# noinspection PyTypeChecker
if __name__ == "__main__":
    f_test = np.pi * 120
    seq_array = [
        [pt.sine(0.2, f_test, 0, 10), pt.phase_advance_sine(0.2, f_test, 0, 10)],
        [pt.sine(0.4, f_test, 0, 10), pt.phase_advance_sine(0.2, f_test, 0, 10)],
        [
            pt.sine(0.6, f_test, 0, 20),
            pt.phase_advance_sine(0.2, f_test, 0, 20),
            pt.sine(1, f_test, 0, 20),
            pt.sine(1, f_test, 0, 20),
        ],
        [pt.sine(0.8, f_test, 0, 10), pt.phase_advance_sine(0.2, f_test, 0, 10)],
        [pt.sine(1.0, f_test, 0, 10), pt.phase_advance_sine(0.2, f_test, 0, 10)],
    ]

    AOMch_array = [[1, 2], [1, 2], [1, 2, 3, 4], [1, 2], [1, 2]]

    twoQ_gate_array = [0, 0, 1, 0, 0]

    wait_after_array = [0, 1, 1, 0, 0]
    wait_after = 20

    seq_to_run = seq.Sequence(
        aom_cal.Calibration(),
        seq_array,
        AOMch_array,
        twoQ_gate_array,
        waveform_dir,
        f_test,
        wait_after_array=wait_after_array,
        wait_after_time=wait_after,
    )

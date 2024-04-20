import numpy as np

import euriqabackend.devices.keysight_awg.AOM_calibration as aom_cal
import euriqabackend.devices.keysight_awg.sequence as seq
import euriqabackend.devices.keysight_awg.waveform as wf
import euriqabackend.devices.keysight_awg.waveform_prototypes as pt

waveform_dir = r"C:\Users\logiq\Desktop\AWG Testing\Generated Waveforms"

# noinspection PyTypeChecker
if __name__ == "__main__":
    f_test = np.pi * 120
    seq_array = [
        [
            pt.sine(0.2, f_test, 0, 9.93),
            pt.phase_advance_sine(0.2, f_test, 0, 9.82),
            pt.sine(1, f_test, 0, 500),
            pt.sine(0.2, f_test, 0, 500),
        ],
        [
            pt.sine(
                0.4,
                f_test,
                0,
                10,
                scan_parameter=wf.ScanParameter.seg_duration,
                scan_values=np.linspace(0, 10, 11),
            ),
            pt.phase_advance_sine(
                0.2,
                f_test,
                0,
                10,
                scan_parameter=wf.ScanParameter.seg_duration,
                scan_values=np.linspace(0, 2, 11),
            ),
        ],
        # [pt.sine(0.4, f_test, 0, 10, scan_parameter=wf.ScanParameter.amplitude, scan_values=[0.4] * 16),
        # pt.phase_advance_sine(0.2, f_test, 0, 10,
        # scan_parameter=wf.ScanParameter.phase, scan_values=np.linspace(0, 2*np.pi, 16))],
        # [pt.sine(0.4, f_test, 0, 9.72), pt.phase_advance_sine(0.2, f_test, 0, 8.56)],
        [
            pt.sine(0.6, f_test, 0, 19.54545),
            pt.phase_advance_sine(0.2, f_test, 0, 17.15643),
        ],
        [pt.sine(0.8, f_test, 0, 6.72), pt.phase_advance_sine(0.2, f_test, 0, 9.99)],
        [
            pt.sine(1.0, f_test, 0, 8.354),
            pt.phase_advance_sine(0.2, f_test, 0, 9.26578),
        ],
    ]

    AOMch_array = [[1, 2, 3, 4], [1, 2], [1, 2], [1, 2], [1, 2]]

    AOMch_array2 = [[1, 2, 3, 4], [1, 5], [1, 6], [1, 7], [1, 8]]

    twoQ_gate_array = [0, 0, 1, 0, 0]

    wait_after_array = [0, 0, 0, 1, 0]
    wait_after = 5.001

    PA_dark_freq = f_test

    seq_to_run = seq.Sequence(
        aom_cal.Calibration(),
        seq_array,
        AOMch_array,
        twoQ_gate_array,
        waveform_dir,
        PA_dark_freq,
        wait_after_array=wait_after_array,
        wait_after_time=wait_after,
    )

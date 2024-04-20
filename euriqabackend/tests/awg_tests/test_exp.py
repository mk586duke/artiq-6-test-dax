import numpy as np

import euriqabackend.devices.keysight_awg.AOM_calibration as aom_cal
import euriqabackend.devices.keysight_awg.sequence as seq
import euriqabackend.devices.keysight_awg.waveform as wf
import euriqabackend.devices.keysight_awg.waveform_prototypes as pt

waveform_dir = r"C:\Users\logiq\Desktop\AWG Testing\Generated Waveforms"

# noinspection PyTypeChecker
if __name__ == "__main__":
    seq_array = [
        [pt.sine(1, 20, 0, 10), pt.sine(0.8, 20, 0, 10), pt.sine(0.6, 20, 0, 10)],
        [pt.multitone_sine([0.5, 0.5], [19.9, 20.1], [0, 0], 10)],
        [pt.phase_advance_sine(0.8, 20, 0, 20), pt.phase_advance_sine(0.6, 20, 0, 20)],
        [
            pt.sine(
                1,
                20,
                0,
                10,
                scan_parameter=wf.ScanParameter.amplitude,
                scan_values=np.linspace(0, 1, 11),
            ),
            pt.multitone_sine(
                [0.5, 0.5],
                [19.9, 20.1],
                [0, 0],
                10,
                scan_parameter=wf.ScanParameter.frequency,
                scan_values=[[20 + df, 20 - df] for df in np.linspace(0, 0.5, 11)],
            ),
        ],
        [
            pt.phase_advance_sine(
                0.8,
                20,
                0,
                10.001,
                scan_parameter=wf.ScanParameter.phase,
                scan_values=np.linspace(0, 2 * np.pi, 11),
            )
        ],
    ]

    AOMch_array = [[1, 16, 3], [4], [1, 16], [1, 16], [1]]

    twoQ_gate_array = [0, 0, 1, 0, 0]

    wait_after_array = [0, 0, 0, 0, 0]
    wait_after = 20

    PA_dark_freq = 20

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

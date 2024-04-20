import euriqabackend.devices.keysight_awg.AOM_calibration as aom_cal
import euriqabackend.devices.keysight_awg.sequence as seq
import euriqabackend.devices.keysight_awg.waveform_prototypes as pt

# from euriqabackend.devices.keysight_awg import waveform as wf

waveform_dir = r"C:\Users\logiq\Desktop\AWG Testing\Generated Waveforms"

# noinspection PyTypeChecker
if __name__ == "__main__":
    f_test = 57
    seq_array = [[pt.sine(1.0, f_test, 0, 500)]]

    slot_array = [[9]]

    twoQ_gate_array = [0]

    wait_after_array = [0]
    wait_after = 5

    PA_dark_freq = f_test

    seq_to_run = seq.Sequence(
        aom_cal.Calibration(),
        seq_array,
        slot_array,
        twoQ_gate_array,
        waveform_dir,
        PA_dark_freq,
        wait_after_array=wait_after_array,
        wait_after_time=wait_after,
    )

import logging

import sipyco.pyon as pyon
import numpy as np
import qiskit.pulse as qp
import euriqabackend.waveforms.single_qubit as oneq
import euriqabackend.waveforms.delay as delay_gate

import euriqafrontend.interactive.rfsoc.qiskit_backend as qbe
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit
import pulsecompiler.qiskit.schedule_converter as schedule_converter
import pulsecompiler.rfsoc.tones.record as record
import pulsecompiler.rfsoc.structures.channel_map as rfsoc_map
import pulsecompiler.qiskit.backend as qbe
import pulsecompiler.qiskit.schedule_converter as sched_conv
import pulsecompiler.rfsoc.tones.upload as rfsoc_upload


logging.basicConfig(level=logging.DEBUG)
board = rfsoc_map.RFSoCBoardDescriptor(index=0, ip_address="192.168.83.102", ip_port=50052, has_global_output=False)
board_map = rfsoc_map.RFSoCChannelMapping([board])
# backend = qbe.MinimalQiskitIonBackend(7, rfsoc_hardware_map=board_map, endcap_ions=(0, 0), use_zero_index=False)

nominal_freq = 100e6
duration_cycles = int(20 / 2.5e-9)
with qp.build() as schedule:
    out_chan = qp.DriveChannel(10)
    qp.set_frequency(nominal_freq, out_chan)
    qp.play(qp.Constant(duration_cycles, 0.5), out_chan)

tones = sched_conv.OpenPulseToOctetConverter.schedule_to_octet(schedule, default_lo_freq_hz={out_chan:nominal_freq})
rfsoc_upload.upload_channel_sequence(tones, num_repeats=5, rfsoc_channel_map=board_map, streaming=False, flush=True)

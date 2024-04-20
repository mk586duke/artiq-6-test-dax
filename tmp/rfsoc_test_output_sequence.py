"""Play a simple test tone on all RFSoC output channels of a given channel."""
import pulsecompiler.rfsoc.structures.channel_map as rfsoc_channels
import pulsecompiler.rfsoc.structures.common_tones as common
import pulsecompiler.rfsoc.tones.tonedata as td
import pulsecompiler.rfsoc.tones.upload as upload


if __name__ == "__main__":
    board_ips = ("192.168.83.103", "192.168.83.104")
    board_port = 50052
    seq = {}
    num_repeats = 20
    pulse_duration_cycles = 10000 # ~25 us pulse

    # if not using full board description file
    board_descriptions = [
        rfsoc_channels.RFSoCBoardDescriptor(i, ip, board_port, has_global_output=False)
        for i, ip in enumerate(board_ips)
    ]
    channel_map = rfsoc_channels.RFSoCChannelMapping(board_descriptions)
    # if using board description file
    # channel_map = rfsoc_channels.RFSoCChannelMapping.from_pyon_file("/path/to/board_description.pyon")
    for chan in channel_map.rfsoc_channels:
        # Different amplitude/freq on each tone (per-channel)
        if chan.channel_tone_index == 0:
            pulse_freq = 5e6
            pulse_amp = 0.5
        else:
            pulse_freq = 3e6
            pulse_amp = 0.3
        seq[chan] = [
            # prepare (wait for trigger)
            td.ToneData(chan.board_channel_index, chan.channel_tone_index, 4, 0, 0, 0, wait_trigger=True),
            # output constant-amplitude pulse
            td.ToneData(chan.board_channel_index, chan.channel_tone_index, pulse_duration_cycles, pulse_freq, pulse_amp, 0.0)
        ]

    common.measure_all_dict(seq)

    upload.upload_channel_sequence(seq, num_repeats=num_repeats, streaming=False)

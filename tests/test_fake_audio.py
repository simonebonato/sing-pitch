"""The fake audio hub: real-time pace, and what the file mic hears relative to the output."""
import time

import numpy as np

from sing_pitch.testmode import FakeAudioHub, FakeMic

SR, STEP = 48000, 256


def test_the_clock_runs_at_time_scale_times_the_wall_clock():
    hub = FakeAudioHub(time_scale=2.0)
    try:
        t0, a0 = time.perf_counter(), hub.now_s
        time.sleep(0.5)
        elapsed = hub.now_s - a0
        assert abs(elapsed - 2.0 * (time.perf_counter() - t0)) < 0.1
    finally:
        hub.close()


def test_the_file_mic_starts_with_the_output_and_hears_it_delay_late():
    """Sample k of the wav (counted from the offset) is heard at output frame k + delay."""
    hub = FakeAudioHub(time_scale=20.0)
    ramp = np.arange(SR, dtype=np.float32)          # sample value = its index in the file
    chunks, out_frames = [], []

    def callback(outdata, frames, time_info, status):
        out_frames.append(frames)
        outdata[:] = 0

    try:
        with hub.held():
            out = hub.open_output(callback=callback, channels=2, blocksize=1024, samplerate=SR, dtype="float32")
            out.start()
            mic = FakeMic(hub, ramp, offset_s=0.25, delay_s=0.1)
            mic.start(chunks.append)
        while len(chunks) < 200:
            time.sleep(0.01)
        mic.stop(), out.stop()
    finally:
        hub.close()
    heard = np.concatenate(chunks)
    assert all(len(c) == STEP for c in chunks)
    delay = round(0.1 * SR)
    assert np.all(heard[:delay] == 0)                                 # nothing yet: the voice is late
    assert heard[delay] == round(0.25 * SR)                           # then the file from its offset
    rest = SR - round(0.25 * SR)
    assert np.all(np.diff(heard[delay : delay + rest]) == 1)          # in real time, no gap or repeat
    assert np.all(heard[delay + rest :] == 0)                         # the file ended: silence
    assert set(out_frames) == {STEP}                                  # just in time, one step per call


def test_the_file_mic_is_late_by_exactly_the_latency_the_window_assumes():
    """The window maps input frame i to output frame i + (rendered at the first chunk - chunk) - latency (live.py,
    calibration.py). The file mic must sing the file on that same mapping, else a perfect singer scores as early."""
    from sing_pitch.playback import BackingPlayer

    hub = FakeAudioHub(time_scale=20.0)
    song = np.arange(2 * SR, dtype=np.float32)        # song sample value = its song frame
    player = BackingPlayer(song, SR, sample_rate=SR, stream_factory=hub.open_output)
    chunks, first = [], []

    def on_chunk(chunk):
        if not first:
            first.append(round(player.song_log.rendered_s * SR) - len(chunk))
        chunks.append(chunk)

    latency = round(0.1 * SR)
    try:
        with hub.held():
            player.start()
            mic = FakeMic(hub, song, delay_s=latency / SR)
            mic.start(on_chunk)
        while len(chunks) < 100:
            time.sleep(0.01)
        mic.stop(), player.stop()
    finally:
        hub.close()
    heard = np.concatenate(chunks)
    i = np.arange(latency + 1000, len(heard))
    assert np.array_equal(heard[i], (i + first[0] - latency).astype(np.float32))

"""The fake hardware's faults (#49) do what a real device does under load, and its timestamps tell the truth."""
import time

import numpy as np
import pytest

from sing_pitch.testmode import FakeAudioHub, FakeMic, Faults, hold_the_gil, parse_flags

SR, STEP = 48000, 256


def test_a_fault_spec_reads_into_faults_and_a_bad_one_is_refused():
    f = Faults.parse("underflow=2:50,overflow=3:20,drift=150,deadline=20:60,ui_stall=1.5:80:sleep")
    assert (f.underflow_every_s, f.underflow_ms, f.overflow_every_s, f.overflow_ms) == (2.0, 50.0, 3.0, 20.0)
    assert (f.drift_ppm, f.deadline_out_ms, f.deadline_in_ms) == (150.0, 20.0, 60.0)
    assert (f.ui_stall_every_s, f.ui_stall_ms, f.ui_stall_gil) == (1.5, 80.0, False)
    assert Faults.parse(None) == Faults() and Faults.parse("deadline=30").deadline_in_ms == 30.0
    with pytest.raises(ValueError):
        Faults.parse("underflow=2")
    assert parse_flags(["--fake-faults", "drift=100"])[0].fake_faults == "drift=100"
    with pytest.raises(SystemExit):
        parse_flags(["--fake-faults", "wobble=3"])


def run(faults, seconds=2.0, scale=20.0, mic=True, during=None):
    """Open one fake output (recording each callback's DAC time) and one fake mic; run `seconds` of app time."""
    hub = FakeAudioHub(time_scale=scale, faults=faults)
    dacs, chunks = [], []
    frames = [0]

    def callback(outdata, n, time_info, status):
        dacs.append((frames[0], time_info.outputBufferDacTime))
        frames[0] += n
        outdata[:] = 0.1

    try:
        with hub.held():
            out = hub.open_output(callback=callback, channels=2, blocksize=1024, samplerate=SR, dtype="float32")
            out.start()
            m = FakeMic(hub, np.zeros(SR * 10, np.float32))
            if mic:
                m.start(chunks.append)
        if during is not None:
            during(hub)
        while hub.now_s < seconds:
            time.sleep(0.005)
        out.stop(), m.stop()
    finally:
        hub.close()
    return hub, out, m, dacs, chunks


def dac_jumps(dacs):
    """DAC time steps beyond what the frames between them take: the silences the output played."""
    f = np.array([d[0] for d in dacs], float) / SR
    t = np.array([d[1] for d in dacs])
    jump = np.diff(t) - np.diff(f)
    return jump[jump > 1e-4]


def test_an_underflow_is_silence_the_dac_time_shows_and_the_frame_count_does_not():
    hub, out, mic, dacs, _ = run("underflow=0.5:40")
    jumps = dac_jumps(dacs)
    assert len(jumps) >= 2 and np.allclose(jumps, 8 * STEP / SR, atol=1e-6)      # 40 ms -> whole steps (42.7 ms)
    assert out.played.glitches == len(jumps)                                     # the truth agrees
    assert 8 * len(jumps) <= hub.lost_out_steps <= 8 * (len(jumps) + 1)          # a loss at the stop shows no jump
    assert mic.times.glitches == 0                                               # the mic heard on


def test_an_overflow_is_mic_audio_never_delivered_and_its_timestamps_show_the_gap():
    hub, out, mic, dacs, chunks = run("overflow=0.5:20")
    assert mic.times.glitches >= 2 and mic.times.lost_s == pytest.approx(4 * STEP / SR * mic.times.glitches)
    unstamped = hub.lost_in_steps * STEP / SR - mic.times.lost_s                  # a loss at the stop shows no gap
    assert -1e-9 <= unstamped <= 4 * STEP / SR + 1e-9
    assert all(len(c) == STEP for c in chunks)
    assert len(dac_jumps(dacs)) == 0


def test_drift_makes_the_output_play_more_frames_than_the_mic_hears():
    hub, out, mic, dacs, chunks = run("drift=2000", seconds=3.0)
    played = out._played
    heard = sum(len(c) for c in chunks)
    assert played / heard == pytest.approx(1.002, abs=2e-4)
    f = np.array([d[0] for d in dacs], float)
    t = np.array([d[1] for d in dacs])
    assert np.polyfit(t, f / SR, 1)[0] == pytest.approx(1.002, abs=1e-5)       # the DAC times say so too


def test_a_deadline_turns_a_held_gil_into_lost_blocks_on_both_sides():
    """The UI thread holds the GIL 150 ms (app time at scale 1): the hub cannot run, and once it can, the output steps
    more than 20 ms late and the mic steps more than 60 ms late are lost, as the device would have lost them."""
    def stall(hub):
        while hub.now_s < 0.5:
            time.sleep(0.002)
        hold_the_gil(150)

    hub, out, mic, dacs, chunks = run("deadline=20:60", seconds=1.0, scale=1.0, during=stall)
    lost_out = hub.lost_out_steps * STEP / SR
    lost_in = hub.lost_in_steps * STEP / SR
    late = lost_out - 0.130               # scheduling delay on a loaded machine, the same for both sides
    assert -0.015 < late < 0.08 and lost_in == pytest.approx(lost_out - 0.040, abs=0.011)   # 150 - 20 / 150 - 60 ms
    assert sum(dac_jumps(dacs)) == pytest.approx(lost_out, abs=1e-6)
    assert mic.times.lost_s == pytest.approx(lost_in, abs=1e-6)


def test_without_faults_a_held_gil_loses_nothing():
    def stall(hub):
        hold_the_gil(100)

    hub, out, mic, dacs, chunks = run(None, seconds=0.5, scale=1.0, during=stall)
    assert hub.lost_out_steps == hub.lost_in_steps == 0 and len(dac_jumps(dacs)) == 0


def test_the_hub_clock_runs_between_steps_and_matches_the_output_timestamps():
    hub = FakeAudioHub(time_scale=1.0)
    try:
        time.sleep(0.1)
        a = hub.now()
        time.sleep(0.003)
        b = hub.now()
        assert 0.0 < b - a < 0.02                       # moves within a 5.3 ms step, never ahead of what is played
        assert hub.now_s - STEP / SR <= b <= hub.now_s + 1e-9
    finally:
        hub.close()

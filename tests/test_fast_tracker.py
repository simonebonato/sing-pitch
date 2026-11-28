"""FastTracker (pitch/fast.py): the classic tracker behind Tracker: Fast. Measured 2026-11-26 against SwiftF0 on a real vocal
and on it one and two octaves down: dots ~32 ms after the voice, 6.6 % of voiced frames over 50 cents on the real vocal
(plain YIN 13.5 %); notes in ~/Builds/.notes/sing-pitch/latency/fast/report.md."""
import numpy as np
import pytest

from sing_pitch.config import ScoreConfig
from sing_pitch.take_engine.gate import VoicingGate
from sing_pitch.pitch.fast import FastTracker

SR = 48000


def tone(f, dur=1.0, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def run(tr, x, chunk=256):
    tr.reset()
    fr = []
    for i in range(0, len(x), chunk):
        fr += tr.push(x[i:i + chunk], SR)
    return fr + tr.flush()


def voiced(frames):
    g = VoicingGate(ScoreConfig())
    return np.array([g.step(f) for f in frames])


@pytest.mark.parametrize("f", [70, 110, 220, 440, 880])
def test_pure_tone_within_10_cents(f):
    fr = run(FastTracker(), tone(f))
    v = voiced(fr)
    t = np.array([x.t for x in fr]); hz = np.array([x.hz for x in fr])
    late = (t > 0.2) & (t < 0.97)                     # flush() frames at the very end use a zero-padded tail: not judged
    assert v[late].mean() > 0.95                      # the app's gate calls it voiced
    cents = 1200 * np.abs(np.log2(hz[late & v] / f))
    assert cents.max() < 10, f"{f} Hz: worst {cents.max():.1f} cents"


def test_release_age_at_most_40_ms():
    tr = FastTracker(); tr.reset()
    x = tone(220, 0.6)
    ages = []
    pushed = 0
    for i in range(0, len(x), 256):
        out = tr.push(x[i:i + 256], SR)
        pushed += 256
        ages += [(pushed / SR - f.t) * 1000 for f in out]
    assert ages
    assert max(ages) == pytest.approx(1280 / SR * 1000, abs=0.01)   # frames are released the moment their window is complete: window / 2
    # the experiments' convention: release age = window / 2 + one hop (the audio of the newest chunk is up to one hop old when it arrives)
    assert max(ages) + 256 / SR * 1000 <= 40.0


@pytest.mark.parametrize("kind", ["zeros", "very_quiet", "noise_below_gate"])
def test_silence_gives_no_voiced_frame(kind):
    n = SR
    rng = np.random.default_rng(1)
    x = {"zeros": np.zeros(n), "very_quiet": 1e-5 * tone(220, 1.0) / 0.3, "noise_below_gate": rng.normal(0, 10 ** (-60 / 20), n)}[kind].astype(np.float32)
    fr = run(FastTracker(), x)
    assert len(fr) > 100
    assert not voiced(fr).any()


def vocal_like():
    """A held note with vibrato, a gap, an octave jump, noise: enough variety to expose any state that depends on chunking."""
    rng = np.random.default_rng(3)
    parts = []
    for f, d in ((196, 0.5), (0, 0.15), (392, 0.5), (294, 0.4), (0, 0.1), (110, 0.6)):
        n = int(d * SR); t = np.arange(n) / SR
        if f == 0:
            parts.append(rng.normal(0, 0.003, n))
        else:
            ph = 2 * np.pi * np.cumsum(f * 2 ** (20 / 1200 * np.sin(2 * np.pi * 5.5 * t))) / SR
            parts.append(0.25 * sum(np.sin(k * ph) / k for k in (1, 2, 3, 4)) + rng.normal(0, 0.003, n))
    return np.concatenate(parts).astype(np.float32)


def test_chunked_stream_equals_one_big_chunk():
    x = vocal_like()
    a = run(FastTracker(), x, chunk=256)
    b = FastTracker(); b.reset(); big = b.push(x, SR) + b.flush()
    c = run(FastTracker(), x, chunk=777)              # not a multiple of the hop
    for other in (big, c):
        assert len(a) == len(other)
        for p, q in zip(a, other):
            assert p.t == pytest.approx(q.t, abs=1e-9)
            assert p.hz == pytest.approx(q.hz, rel=1e-9, abs=1e-9)
            assert p.confidence == pytest.approx(q.confidence, abs=1e-9)
            assert p.loudness_db == pytest.approx(q.loudness_db, abs=1e-9)


def test_reset_starts_a_fresh_stream():
    x = vocal_like()
    tr = FastTracker()
    a = run(tr, x)
    b = run(tr, x)
    assert [f.hz for f in a] == [f.hz for f in b] and a[0].t == b[0].t


def test_frame_semantics():
    fr = run(FastTracker(), tone(220, 0.5))
    assert fr[0].t == pytest.approx(1280 / SR)        # centre of the first 2560-sample window
    assert all(0.0 <= f.confidence <= 1.0 for f in fr)
    assert all(b.t - a.t == pytest.approx(256 / SR) for a, b in zip(fr, fr[1:]))
    assert -20 < fr[-5].loudness_db < -10             # 0.3 amplitude sine = -13.5 dBFS RMS in SwiftF0's convention (mean square)


def test_rejects_other_sample_rates():
    with pytest.raises(ValueError):
        FastTracker().push(np.zeros(256, np.float32), 44100)


# ---------------- octave consensus gate ----------------
@pytest.mark.parametrize("K", [2, 3, 4, 6])
def test_consensus_switch_after_exactly_K_frames(K):
    tr = FastTracker(octave_consensus=K)
    for i in range(30):
        assert tr._consensus(220.0, True) == pytest.approx(220.0)
    out = [tr._consensus(440.0, True) for _ in range(10)]
    folded = [o for o in out if o == pytest.approx(220.0)]
    assert len(folded) == K - 1                       # frames 1..K-1 are folded back at once (the dot still appears)
    assert out[K - 1] == pytest.approx(440.0)         # frame K is the first on the new octave
    assert all(o == pytest.approx(440.0) for o in out[K - 1:])


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("blip", [440.0, 110.0, 880.0])
def test_consensus_folds_a_short_octave_blip(n, blip):
    tr = FastTracker(octave_consensus=3)
    seq = [220.0] * 20 + [blip] * n + [220.0] * 20
    out = [tr._consensus(h, True) for h in seq]
    assert all(o == pytest.approx(220.0) for o in out)


def test_consensus_leaves_non_octave_jumps_alone():
    tr = FastTracker(octave_consensus=3)
    for _ in range(10):
        tr._consensus(220.0, True)
    assert tr._consensus(220.0 * 2 ** (7 / 12), True) == pytest.approx(220.0 * 2 ** (7 / 12))   # a fifth is a real note
    assert tr._consensus(220.0 * 2 ** (4 / 12), True) == pytest.approx(220.0 * 2 ** (4 / 12))


def test_consensus_forgets_after_a_long_gap():
    tr = FastTracker(octave_consensus=3, consensus_gap_s=0.4)
    for _ in range(10):
        tr._consensus(220.0, True)
    for _ in range(int(0.4 * SR / 256) + 2):
        tr._consensus(0.0, False)
    assert tr._consensus(440.0, True) == pytest.approx(440.0)
    tr2 = FastTracker(octave_consensus=3, consensus_gap_s=0.4)
    for _ in range(10):
        tr2._consensus(220.0, True)
    for _ in range(20):
        tr2._consensus(0.0, False)
    assert tr2._consensus(440.0, True) == pytest.approx(220.0)


def sung(f_a, f_b, t_jump, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    f = np.where(t < t_jump, f_a, f_b) * 2 ** (25 / 1200 * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 7))
    return (x / np.max(np.abs(x)) * 0.3).astype(np.float32)


def test_audio_octave_jump_is_followed_and_a_held_note_is_not_split():
    x = sung(220, 440, 1.0, 2.0)
    fr = run(FastTracker(), x)
    v = voiced(fr)
    t = np.array([f.t for f in fr]); hz = np.array([f.hz for f in fr])
    held = (t > 0.2) & (t < 0.95) & v
    assert (np.abs(1200 * np.log2(hz[held] / 220)) < 60).mean() > 0.98
    after = (t > 1.3) & v
    assert (np.abs(1200 * np.log2(hz[after] / 440)) < 60).mean() > 0.98      # the leap is accepted: no lock on the old octave
    first = t[(t > 1.0) & v & (np.abs(1200 * np.log2(np.maximum(hz, 1) / 440)) < 60)][0]
    assert first - 1.0 < 0.2                                                  # within 200 ms of the leap


def test_skip_lets_audio_go_by_and_the_frames_after_keep_their_times():
    """LiveRunner skips chunks for the display tracker when it is behind: no frames for the windows that touch the
    skipped audio, the next ones on the same 5.33 ms grid as without the skip, voiced again once a window is full."""
    x = tone(220.0, 1.5)
    whole = run(FastTracker(), x)
    tr = FastTracker()
    a, b = int(0.5 * SR), int(0.7 * SR)
    fr = []
    for i in range(0, a, 256):
        fr += tr.push(x[i:i + 256], SR)
    tr.skip(b - a)
    for i in range(b, len(x), 256):
        fr += tr.push(x[i:i + 256], SR)
    fr += tr.flush()
    window_s = 2560 / SR
    times = np.array([f.t for f in fr])
    assert not np.any((times > a / SR + 1e-9) & (times - window_s / 2 < b / SR - 1e-9))   # nothing touches the gap
    grid = {round(f.t, 6) for f in whole}
    assert {round(t, 6) for t in times} <= grid                                         # same times as unskipped
    after = [f for f in fr if f.t - window_s / 2 >= b / SR]
    assert after and voiced(after)[5:].all() and np.median([f.hz for f in after]) == pytest.approx(220.0, abs=1.0)

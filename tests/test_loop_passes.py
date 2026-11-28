"""Section loop end to end without a sound card: the real Transport core rendered block by block, a synthetic voice
that copies exactly what it heard one calibrated latency earlier, and the live runner splitting it into one Take per
Pass. The voice is built from the loop's own arithmetic (start - Lead-in, Section end, Speed), never from the log."""
import csv
import time

import numpy as np
from frames import HOP, hz_at
from melody_helpers import library_song, reference_line
from test_take_boundary_on_jump import Backing, LogView, Mic, VoiceTracker

from sing_pitch.live import LiveRunner
from sing_pitch.pitch import PitchFrame
from sing_pitch.playback.transport import Loop, Transport
from sing_pitch.reference import Melody
from sing_pitch.take_engine import LinearClock, LogClock, PassTag, TakeEngine

NOTES = [(-900, 0.6), (-700, 0.6), (-500, 0.6), (-400, 0.6), (-200, 0.6), (-500, 0.6), (-900, 1.0)] * 2   # 9.2 s
SR, MIC_SR, CHUNK, BLOCK = 48000, 16000, 160, 1024
LATENCY, OUT_AT_ZERO = 0.23, 0.05
SECTION = Loop(4.0, 8.0, "Chorus 1", lead_in_s=2.0)      # a Pass: song 2.0 .. 8.0, the first 2 s are the Lead-in


def note_at(song_t):
    start = 0.0
    for cents, seconds in NOTES:
        if start <= song_t < start + seconds:
            return cents
        start += seconds
    return None


def heard_in_loop(speed, section=SECTION):
    """Song time heard at output time out_s with the loop on from output 0: Passes of (end - pass start) / speed."""
    pass_s = (section.end_s - section.pass_start_s) / speed

    def heard(out_s):
        return section.pass_start_s + (out_s % pass_s) * speed if out_s >= 0 else None
    return heard, pass_s


def voice(heard, seconds):
    out = []
    for i in range(round(seconds / HOP)):
        t = i * HOP
        song_t = heard(t - LATENCY + OUT_AT_ZERO)
        cents = None if song_t is None else note_at(song_t)
        out.append(PitchFrame(t, 0.0, 0.0, -140.0) if cents is None else PitchFrame(t, hz_at(cents), 0.98, -20.0))
    return out


def looped_transport(seconds, speed=1.0, section=SECTION):
    tr = Transport(np.zeros((round(9.2 * SR), 2), np.float32), sample_rate=SR)
    if speed != 1.0:
        tr.set_speed(speed, np.zeros((round(9.2 * SR / speed), 2), np.float32))
    tr.set_loop(section)
    events, t = [], 0.0
    while t < seconds:
        events += tr.render(BLOCK).events
        t += BLOCK / SR
    return tr, events


def melody():
    return Melody(reference_line(NOTES), "do-re-mi")


def pass_takes(speed):
    """Three wraps (four Passes, the last one cut short) scored Pass by Pass through the engine."""
    heard, pass_s = heard_in_loop(speed)
    tr, events = looped_transport(3.5 * pass_s + 1, speed)
    starts = [e for e in events if e.kind == "pass_started"]
    assert [e.pass_index for e in starts[:4]] == [1, 2, 3, 4]
    assert np.allclose([e.output_s for e in starts[:4]], [k * pass_s for k in range(4)], atol=1 / SR)
    frames = voice(heard, 3.5 * pass_s)
    clock = LogClock(tr.log, LATENCY, OUT_AT_ZERO)
    cuts = [k * pass_s + LATENCY - OUT_AT_ZERO for k in range(1, 4)] + [1e9]
    takes, lo = [], -1e9
    for k, hi in enumerate(cuts, start=1):
        engine = TakeEngine(melody(), song_clock=clock,
                            pass_tag=PassTag("Chorus 1", SECTION.start_s, SECTION.end_s, k))
        engine.push([f for f in frames if lo <= f.t < hi])
        takes.append(engine.stop())
        lo = hi
    return takes, frames


def check_passes(takes, speed=1.0):
    for k, record in enumerate(takes[:3], start=1):
        s = record.summary
        assert (record.section_name, record.pass_index) == ("Chorus 1", k)
        assert s.pct_in_tune >= 98.0 and s.median_abs_cents < 1.0
        # only the Section is scored: 4 s of song (4 / speed s of singing), never the 2 s Lead-in
        assert abs(s.n_scored * HOP - 4.0 / speed) < 0.15
        lead_in = [st for f, st in zip(record.frames, record.song_times) if f.hz > 0 and st is None]
        assert abs(len(lead_in) * HOP - 2.0 / speed) < 0.15


def test_three_loop_wraps_give_passes_each_about_100_in_tune_with_the_lead_in_unscored():
    takes, frames = pass_takes(1.0)
    check_passes(takes)
    # the old constant offset loses the sync from the first wrap on
    old = TakeEngine(melody(), song_clock=LinearClock(LATENCY - OUT_AT_ZERO))
    old.push(frames)
    assert old.stop().summary.pct_in_tune < 70.0


def test_the_passes_stay_about_100_in_tune_at_70_percent_speed():
    takes, _ = pass_takes(0.7)
    check_passes(takes, 0.7)


def test_a_pass_take_rescores_to_the_same_numbers(tmp_path):
    from sing_pitch.take_store import rescore_take, save_take

    library_song(tmp_path, reference_line(NOTES))
    takes, _ = pass_takes(1.0)
    saved = save_take(tmp_path, takes[1])
    again = rescore_take(tmp_path, saved.id, write=True)
    assert again.summary == saved.summary and again.song_times == saved.song_times
    assert (again.section_name, again.pass_index) == ("Chorus 1", 2)
    (row,) = csv.DictReader(open(tmp_path / "takes.csv"))
    assert (row["section_name"], row["pass_index"]) == ("Chorus 1", "2")


# --- through the live runner ----------------------------------------------------------------------------------------

def sing_looped(tmp_path, passes, speed=1.0, section=SECTION, expect_shown=0):
    """One Start..Stop with the loop on from the start; the Transport's events are handed to the runner just before
    the voice that answers them, as the audio thread would."""
    heard, pass_s = heard_in_loop(speed, section)
    seconds = passes * pass_s
    tr, events = looped_transport(seconds + 1, speed, section)
    frames = voice(heard, seconds)
    backing = Backing(LogView(tr.log, OUT_AT_ZERO + CHUNK / MIC_SR))
    mic = Mic()
    runner = LiveRunner(mic, VoiceTracker(frames), melody(), tmp_path, backing=backing, latency_s=LATENCY)
    runner.start()
    pending = sorted(events, key=lambda e: e.output_s)
    fed = 0
    if pending and pending[0].output_s == 0.0:      # the loop was set before Start: its first Pass starts at once
        backing.on_event(pending.pop(0))
    while fed < round(seconds * MIC_SR):
        while pending and fed / MIC_SR >= pending[0].output_s + LATENCY - OUT_AT_ZERO - 0.1:
            backing.on_event(pending.pop(0))
        mic.on_chunk(np.zeros(CHUNK, np.float32))
        fed += CHUNK
    shown, deadline = [], time.perf_counter() + 5.0       # before Stop: what the UI timer would have picked up
    while time.perf_counter() < deadline and len(shown) < expect_shown:
        shown += runner.drain_saved()
        time.sleep(0.01)
    record = runner.stop()
    return record, shown, list(csv.DictReader(open(tmp_path / "takes.csv")))


def test_three_wraps_through_the_runner_save_three_tagged_pass_takes_while_the_loop_goes_on(tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    record, shown, rows = sing_looped(tmp_path, passes=3.5, expect_shown=3)
    # Passes 1-3 complete; Pass 4 has only ~1 s of its 4 s Section (under half) and is dropped at Stop
    assert [(r["section_name"], r["pass_index"]) for r in rows] == [("Chorus 1", "1"), ("Chorus 1", "2"),
                                                                    ("Chorus 1", "3")]
    assert all(float(r["pct_in_tune"]) >= 98.0 for r in rows)
    assert all(abs(int(r["n_scored_frames"]) * HOP - 4.0) < 0.15 for r in rows)
    # each Pass's result reached the UI while the loop went on, not only at Stop
    assert [t.pass_index for t in shown] == [1, 2, 3] and record.pass_index == 3


def test_passes_of_a_section_shorter_than_3_s_are_saved_and_a_pass_cut_short_is_discarded(tmp_path):
    """A Pass Take is kept when it scored at least half of its Section (3 s is for Takes cut by a jump): every full
    Pass of a 1.2 s Section is saved; the last Pass, stopped 0.25 s into its Section, is not."""
    library_song(tmp_path, reference_line(NOTES))
    short = Loop(4.0, 5.2, "Short", lead_in_s=2.0)           # 1.2 s of Section per Pass (a Pass = 3.2 s)
    _, _, rows = sing_looped(tmp_path, passes=3 + 2.25 / 3.2, section=short, expect_shown=3)
    assert [(r["section_name"], r["pass_index"]) for r in rows] == [("Short", "1"), ("Short", "2"), ("Short", "3")]
    assert all(abs(int(r["n_scored_frames"]) * HOP - 1.2) < 0.15 for r in rows)


def test_a_two_and_a_half_second_section_keeps_its_full_passes(tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    short = Loop(4.0, 6.5, "Short", lead_in_s=2.0)           # 2.5 s of Section per Pass
    _, _, rows = sing_looped(tmp_path, passes=2.5, section=short, expect_shown=2)
    assert [r["pass_index"] for r in rows] == ["1", "2"]     # Pass 3 has 0.25 s of Section: discarded at Stop


def test_replaying_a_pass_take_draws_the_reference_through_its_lead_in_too(qtbot, tmp_path):
    """The Lead-in is never scored, but the singer heard the song there: a replay draws it, as the live view did."""
    from audio import quiet
    from test_melody_window import window

    from sing_pitch.take_store import save_take

    library_song(tmp_path, reference_line(NOTES))
    takes, _ = pass_takes(1.0)
    saved = save_take(tmp_path, takes[1])                 # Pass 2: voice 6.18 .. 12.18 s, the first 2 s are Lead-in
    lead_in = [f.t for f, st in zip(saved.frames, saved.song_times) if st is None]
    lo, hi = min(lead_in), max(lead_in)
    assert hi - lo > 1.5
    win = window(qtbot, tmp_path, quiet(0.5))
    win.reference_box.setCurrentText("Melody")
    win.toggle_replay()
    assert win._replay is not None, win.summary_label.text()
    win._draw_melody(hi)
    x, y = win._ref_curve.getData()
    in_lead_in = (x >= lo + 0.2) & (x <= hi - 0.2)
    assert in_lead_in.any() and np.isfinite(y[in_lead_in]).mean() > 0.9
    win.toggle_replay()


def test_with_a_count_in_each_pass_scores_only_its_section_and_the_count_in_is_never_scored(tmp_path):
    """#42: a 2 s bar of clicks (120 bpm, 4/4) instead of the Lead-in. The voice even sings through the count-in
    (it follows the reference there): none of that is scored, each Pass scores its 4 s Section at about 100 %."""
    from sing_pitch.playback.transport import CountIn

    library_song(tmp_path, reference_line(NOTES))
    counted = Loop(4.0, 8.0, "Chorus 1", count_in=CountIn(120.0, 4))
    _, _, rows = sing_looped(tmp_path, passes=3.3, section=counted, expect_shown=3)
    assert [(r["section_name"], r["pass_index"]) for r in rows] == [("Chorus 1", "1"), ("Chorus 1", "2"),
                                                                    ("Chorus 1", "3")]
    assert all(float(r["pct_in_tune"]) >= 98.0 for r in rows)
    assert all(abs(int(r["n_scored_frames"]) * HOP - 4.0) < 0.15 for r in rows)


def test_at_70_percent_the_count_in_lasts_a_bar_at_the_slower_tempo_and_is_still_unscored(tmp_path):
    from sing_pitch.playback.transport import CountIn

    library_song(tmp_path, reference_line(NOTES))
    counted = Loop(4.0, 8.0, "Chorus 1", count_in=CountIn(120.0, 4))
    _, _, rows = sing_looped(tmp_path, passes=2.3, speed=0.7, section=counted, expect_shown=2)
    assert [r["pass_index"] for r in rows] == ["1", "2"]
    assert all(float(r["pct_in_tune"]) >= 98.0 for r in rows)
    assert all(abs(int(r["n_scored_frames"]) * HOP - 4.0 / 0.7) < 0.2 for r in rows)

"""Each Take records the practice settings it was sung with (ticket #15): Guide vocal (on / off / mixed), its gain,
and the Speed. Rows saved before these columns read as guide off, Speed 100."""
import csv
import dataclasses
import shutil
import time
from datetime import datetime

import numpy as np
from frames import HOP, hz_at
from melody_helpers import library_song, reference_line
from test_old_takes_rescore import FIXTURE, MELODY
from test_loop_passes import NOTES, SECTION, melody, note_at
from test_take_boundary_on_jump import Backing, LogView, Mic, VoiceTracker
from test_take_store import make_take

from sing_pitch.history import load_history
from sing_pitch.live import LiveRunner
from sing_pitch.pitch import PitchFrame
from sing_pitch.playback.transport import Transport
from sing_pitch.take_store import COLUMNS, rescore_take, save_take, summary_path

SR, MIC_SR, CHUNK, BLOCK = 48000, 16000, 160, 1024
LATENCY, OUT_AT_ZERO = 0.23, 0.05


# --- the table ------------------------------------------------------------------------------------------------------

def test_a_take_row_carries_guide_vocal_guide_gain_and_speed_pct(tmp_path):
    record = dataclasses.replace(make_take(datetime(2026, 11, 24, 9, 0, 0)), guide_vocal="mixed", guide_gain=0.6,
                                 speed_pct=70)
    save_take(tmp_path, record)
    (row,) = csv.DictReader(open(summary_path(tmp_path)))
    assert (row["guide_vocal"], row["guide_gain"], row["speed_pct"]) == ("mixed", "0.6", "70")
    assert COLUMNS[-10:-4] == ("guide_vocal", "guide_gain", "speed_pct", "octave_mode", "note_mode",
                            "min_note_s")      # new columns go at the end


def test_a_take_without_practice_settings_is_saved_as_guide_off_speed_100(tmp_path):
    save_take(tmp_path, make_take(datetime(2026, 11, 24, 9, 0, 0)))
    (row,) = csv.DictReader(open(summary_path(tmp_path)))
    assert (row["guide_vocal"], row["guide_gain"], row["speed_pct"]) == ("off", "0.0", "100")


def test_rescoring_a_take_keeps_its_settings(tmp_path):
    save_take(tmp_path, dataclasses.replace(make_take(datetime(2026, 11, 24, 9, 0, 0)), guide_vocal="on",
                                            guide_gain=0.4, speed_pct=85))
    again = rescore_take(tmp_path, "20261124-090000", write=True)
    assert (again.guide_vocal, again.guide_gain, again.speed_pct) == ("on", 0.4, 85)
    (row,) = csv.DictReader(open(summary_path(tmp_path)))
    assert (row["guide_vocal"], row["guide_gain"], row["speed_pct"]) == ("on", "0.4", "85")


def old_home(tmp_path):
    shutil.copy(FIXTURE / "takes.csv", tmp_path / "takes.csv")
    shutil.copytree(FIXTURE / "takes", tmp_path / "takes")
    return tmp_path


def test_old_rows_load_as_guide_off_speed_100_and_no_gain(tmp_path):
    home = old_home(tmp_path)
    library_song(home, reference_line(MELODY))
    old = rescore_take(home, "20261120-210000")      # Melody Take saved before any of the new columns
    assert (old.guide_vocal, old.guide_gain, old.speed_pct) == ("off", 0.0, 100)
    points = load_history(home)._points
    assert points and all((p.guide_vocal, p.speed_pct) == ("off", 100) for p in points)


def test_saving_next_to_old_rows_upgrades_the_header_once_and_leaves_old_rows_empty_in_the_new_columns(tmp_path):
    home = old_home(tmp_path)
    old_rows = list(csv.DictReader(open(summary_path(home))))
    save_take(home, dataclasses.replace(make_take(datetime(2026, 11, 24, 9, 0, 0)), guide_vocal="on", guide_gain=1.0,
                                        speed_pct=60))
    rows = list(csv.DictReader(open(summary_path(home))))
    assert len(rows) == len(old_rows) + 1
    assert all((r["guide_vocal"], r["guide_gain"], r["speed_pct"]) == ("", "", "") for r in rows[:-1])
    assert (rows[-1]["guide_vocal"], rows[-1]["speed_pct"]) == ("on", "60")
    assert [r["pct_in_tune"] for r in rows[:-1]] == [r["pct_in_tune"] for r in old_rows]
    assert open(summary_path(home)).read().count("guide_vocal") == 1


def test_a_header_upgrade_that_fails_halfway_leaves_the_old_table_whole(tmp_path, monkeypatch):
    home = old_home(tmp_path)
    before = open(summary_path(home)).read()

    def crash(self, rows):
        raise OSError("disk full")
    monkeypatch.setattr(csv.DictWriter, "writerows", crash)
    try:
        save_take(home, make_take(datetime(2026, 11, 24, 9, 0, 0)))
    except OSError:
        pass
    assert open(summary_path(home)).read() == before
    assert [p.name for p in home.iterdir() if p.name.startswith("takes.csv") and p.name != "takes.csv"] == []


# --- through the live runner ------------------------------------------------------------------------------------------

class SettingsBacking(Backing):
    """The fake Backing of the other runner tests, plus what the real player tells the runner at Start."""

    def __init__(self, log, gains=(1.0, 0.0), speed=1.0):
        super().__init__(log)
        self.current_gains, self.speed = gains, speed


def sing(tmp_path, seconds, acts, start_gains=(1.0, 0.0), loop=True):
    """One Start..Stop with the loop on from the start; `acts` = [(output_s, fn(transport))] are control calls made
    while it plays (a Speed change, a gain change). The voice copies what the song played one latency earlier."""
    tr = Transport(np.zeros((round(9.2 * SR), 2), np.float32), sample_rate=SR, gains=start_gains)
    if loop:
        tr.set_loop(SECTION)
    todo, events, t = sorted(acts, key=lambda a: a[0]), [], 0.0
    while t < seconds:
        while todo and t >= todo[0][0]:
            todo.pop(0)[1](tr)
        events += tr.render(BLOCK).events
        t += BLOCK / SR
    frames = []
    for i in range(round(seconds / HOP)):
        song_t = tr.log.song_time(i * HOP - LATENCY + OUT_AT_ZERO) if i * HOP - LATENCY + OUT_AT_ZERO >= 0 else None
        cents = None if song_t is None else note_at(song_t)
        frames.append(PitchFrame(i * HOP, 0.0, 0.0, -140.0) if cents is None
                      else PitchFrame(i * HOP, hz_at(cents), 0.98, -20.0))
    backing = SettingsBacking(LogView(tr.log, OUT_AT_ZERO + CHUNK / MIC_SR), gains=start_gains)
    mic = Mic()
    runner = LiveRunner(mic, VoiceTracker(frames), melody(), tmp_path, backing=backing, latency_s=LATENCY)
    runner.start()
    pending, fed = sorted(events, key=lambda e: e.output_s), 0
    while pending and pending[0].output_s == 0.0:
        backing.on_event(pending.pop(0))
    while fed < round(seconds * MIC_SR):
        while pending and fed / MIC_SR >= pending[0].output_s + LATENCY - OUT_AT_ZERO - 0.1:
            backing.on_event(pending.pop(0))
        mic.on_chunk(np.zeros(CHUNK, np.float32))
        fed += CHUNK
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline and len(runner._saved) < 1:
        time.sleep(0.01)
    runner.stop()
    return [(r["pass_index"], r["guide_vocal"], r["guide_gain"], r["speed_pct"])
            for r in csv.DictReader(open(summary_path(tmp_path)))]


def stretched(tr, speed):
    tr.set_speed(speed, np.zeros((round(9.2 * SR / speed), 2), np.float32))


def test_a_speed_change_mid_loop_is_recorded_from_the_next_pass_not_the_one_it_was_asked_in(tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    # Pass 1 = 6 s at Speed 1; Speed 0.7 asked at 3 s; Pass 2 = 6/0.7 = 8.6 s ... output 14.6 s; Pass 3 starts then
    rows = sing(tmp_path, 14.6 + 8.6 + 0.5, [(3.0, lambda tr: stretched(tr, 0.7))])
    assert [(p, spd) for p, _, _, spd in rows] == [("1", "100"), ("2", "70"), ("3", "70")]


def test_guide_vocal_is_off_on_or_mixed_when_it_was_switched_during_the_take(tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    # off in Pass 1; switched on (gain 0.8) at 8 s, inside Pass 2 (6 s .. 12 s) -> mixed; still on for Pass 3
    rows = sing(tmp_path, 6.0 * 3 + 0.5, [(8.0, lambda tr: tr.set_gains(1.0, 0.8))])
    assert [(p, g, gain) for p, g, gain, _ in rows] == [("1", "off", "0.0"), ("2", "mixed", "0.8"), ("3", "on", "0.8")]


def test_a_take_started_with_the_guide_on_records_its_gain(tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    rows = sing(tmp_path, 6.0 * 2 + 0.5, [], start_gains=(1.0, 0.5))
    assert [(g, gain) for _, g, gain, _ in rows] == [("on", "0.5"), ("on", "0.5")]


def test_a_speed_change_outside_a_loop_closes_the_take_so_no_take_is_half_at_one_speed(tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    rows = sing(tmp_path, 11.5, [(4.5, lambda tr: stretched(tr, 0.7))], loop=False)
    assert [(g, spd) for _, g, _, spd in rows] == [("off", "100"), ("off", "70")]


def test_a_take_opened_at_a_cut_starts_when_its_first_voice_was_sung_not_when_the_worker_got_there(tmp_path):
    """The run takes well under a second of real time, but its Passes start 6 s of voice apart: started_at (and so
    the id) comes from the run start plus the cut's voice time."""
    library_song(tmp_path, reference_line(NOTES))
    sing(tmp_path, 6.0 * 3 + 0.5, [])
    rows = list(csv.DictReader(open(summary_path(tmp_path))))
    starts = [datetime.fromisoformat(r["started_at"]) for r in rows]
    gaps = [(b - a).total_seconds() for a, b in zip(starts, starts[1:])]
    assert len(rows) == 3 and all(abs(g - 6.0) <= 1.0 for g in gaps), gaps
    assert len({r["id"] for r in rows}) == 3 and not any("-" in r["id"][9:] for r in rows)    # no collision suffix

"""Ticket #42 in the window: the Between Passes choice (Lead-in or Count-in) with its BPM and Beats-per-bar boxes, the
tempo detected lazily for a song imported before, his own BPM remembered per song, and a click inside a looping
Section that keeps it looping. End to end on the QA driver with the echo singer (it sings exactly what the Guide vocal
plays, so every Pass scores near 100 %)."""
import json

import pytest

from audio import quiet
from melody_helpers import library_song, reference_line
from qa_helpers import NOTES, configure_latency, singing_song
from test_melody_window import window
from test_navigate_and_loops import SCALE, click_timeline

from sing_pitch.library import Library
from sing_pitch.library.tempo import TEMPO, Tempo, TempoUnavailable, load_tempo, save_tempo
from sing_pitch.playback.transport import CountIn
from sing_pitch.qa import QA

LINE = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]


class FakeDetector:
    def __init__(self, result=Tempo(96.0, (0.5, 1.125))):
        self.result, self.calls = result, []

    def __call__(self, backing):
        self.calls.append(backing)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def melody_window(qtbot, home, detector):
    win = window(qtbot, home, quiet(0.5), tempo_detector=detector)
    win.reference_box.setCurrentText("Melody")
    return win


# --- the controls ------------------------------------------------------------------------------------------------

def test_count_in_shows_the_stored_tempo_and_builds_a_count_in_loop(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(LINE))
    save_tempo(song.dir, Tempo(128.0))
    detect = FakeDetector()
    win = melody_window(qtbot, tmp_path, detect)
    assert win.between_box.currentText() == "Lead-in"
    win.timeline.section_marked.emit(0.5, 1.5)
    win.loop_box.setChecked(True)
    assert win._loop().count_in is None                       # Lead-in: as before
    win.between_box.setCurrentText("Count-in")
    assert win.bpm_box.value() == 128.0 and win.beats_box.value() == 4 and detect.calls == []
    assert win._loop().count_in == CountIn(128.0, 4) and win.tempo_label.text() == ""
    win.beats_box.setValue(3)
    assert win._loop().count_in == CountIn(128.0, 3)


def test_a_song_imported_before_gets_its_tempo_detected_when_count_in_is_first_chosen(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(LINE))
    detect = FakeDetector()
    win = melody_window(qtbot, tmp_path, detect)
    assert detect.calls == []                                 # nothing is computed until a Count-in needs it
    win.between_box.setCurrentText("Count-in")
    qtbot.waitUntil(lambda: win.bpm_box.value() == 96.0, timeout=5000)
    assert load_tempo(song.dir) == Tempo(96.0, (0.5, 1.125)) and len(detect.calls) == 1
    assert win.tempo_label.text() == ""


def test_without_a_beat_tracker_the_tempo_is_unknown_and_count_in_waits_for_his_bpm(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(LINE))
    win = melody_window(qtbot, tmp_path, FakeDetector(TempoUnavailable("no librosa")))
    win.timeline.section_marked.emit(0.5, 1.5)
    win.loop_box.setChecked(True)
    win.between_box.setCurrentText("Count-in")
    qtbot.waitUntil(lambda: win.tempo_label.text().startswith("tempo unknown — set BPM"), timeout=5000)
    assert win._loop().count_in is None                       # the Lead-in until he types one
    assert not (song.dir / TEMPO).exists()
    win.bpm_box.setValue(84.0)
    assert win._loop().count_in == CountIn(84.0, 4) and win.tempo_label.text() == ""


def test_his_bpm_wins_over_the_detected_one_and_is_remembered_per_song(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(LINE))
    save_tempo(song.dir, Tempo(144.0))
    win = melody_window(qtbot, tmp_path, FakeDetector())
    win.between_box.setCurrentText("Count-in")
    win.bpm_box.setValue(72.0)                                # a prog song: the detector doubled it
    win.beats_box.setValue(7)
    saved = json.loads((song.dir / "player.json").read_text())
    assert (saved["between"], saved["bpm"], saved["beats_per_bar"]) == ("count-in", 72.0, 7)
    win.close()
    again = melody_window(qtbot, tmp_path, FakeDetector())
    assert (again.between_box.currentText(), again.bpm_box.value(), again.beats_box.value()) == ("Count-in", 72.0, 7)
    assert load_tempo(Library(tmp_path).get(song.id).dir).bpm == 144.0     # the detected one is kept as it was


def test_bpm_0_or_empty_means_the_detected_tempo_and_forgets_his_own(qtbot, tmp_path):
    """QA #48: typing 0 (or emptying the box) said 'tempo unknown' and played the Lead-in though tempo.json had one."""
    song = library_song(tmp_path, reference_line(LINE))
    save_tempo(song.dir, Tempo(143.55))
    win = melody_window(qtbot, tmp_path, FakeDetector())
    win.timeline.section_marked.emit(0.5, 1.5)
    win.loop_box.setChecked(True)
    win.between_box.setCurrentText("Count-in")
    win.bpm_box.setValue(72.0)
    win.bpm_box.setValue(0.0)
    assert win.bpm_box.value() == pytest.approx(143.55, abs=0.06)
    assert win._loop().count_in == CountIn(win.bpm_box.value(), 4)
    assert "detected tempo" in win.tempo_label.text() and "unknown" not in win.tempo_label.text()
    assert json.loads((song.dir / "player.json").read_text())["bpm"] is None


@pytest.mark.parametrize("typed, kept", [(10.0, 30.0), (29.9, 30.0)])
def test_a_bpm_below_30_is_raised_to_30_and_says_the_range(qtbot, tmp_path, typed, kept):
    song = library_song(tmp_path, reference_line(LINE))
    save_tempo(song.dir, Tempo(120.0))
    win = melody_window(qtbot, tmp_path, FakeDetector())
    win.between_box.setCurrentText("Count-in")
    win.bpm_box.setValue(typed)
    assert win.bpm_box.value() == kept
    assert "30" in win.tempo_label.text() and "300" in win.tempo_label.text()


def test_the_bpm_box_cannot_go_above_300(qtbot, tmp_path):
    library_song(tmp_path, reference_line(LINE))
    win = melody_window(qtbot, tmp_path, FakeDetector())
    win.bpm_box.setValue(500.0)
    assert win.bpm_box.value() == 300.0


# --- end to end --------------------------------------------------------------------------------------------------

def echo_singer(home, record_output=None):
    song = singing_song(home, notes=NOTES * 3, tail_s=3.0)
    save_tempo(song.dir, Tempo(120.0))
    configure_latency(home, 0.12)
    qa = QA(home, mic="echo", time_scale=SCALE, record_output=record_output)
    qa.set("reference_box", "Melody")
    qa.set("guide_box", True)
    qa.set("guide_slider", 100)
    qa.set("backing_slider", 0)
    return qa


def test_a_count_in_loop_end_to_end_scores_each_pass_on_its_section_only(tmp_path):
    with echo_singer(tmp_path) as qa:
        qa.mark_section(4.0, 8.0)
        qa.set("between_box", "Count-in")                     # 120 bpm, 4/4: a 2 s bar of clicks before each Pass
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait_until(lambda: qa.window.loop_label.text().startswith("Loop · Pass 3"), timeout_s=30)
        qa.stop_take()
        rows = [r for r in qa.takes() if r["pass_index"] in ("1", "2")]
    assert [r["pass_index"] for r in rows] == ["1", "2"]
    for r in rows:
        assert float(r["pct_in_tune"]) >= 90.0
        assert 3.0 < int(r["n_scored_frames"]) * 0.016 <= 4.1      # the Section only: the count-in is never scored


def test_a_click_inside_the_looping_section_jumps_there_and_keeps_looping_and_one_outside_ends_it(tmp_path):
    with echo_singer(tmp_path) as qa:
        w = qa.window
        qa.mark_section(4.0, 8.0)
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait_until(lambda: (qa.song_position() or 0) > 4.5, timeout_s=10)
        click_timeline(qa, 6.5)
        qa.wait(0.3)
        assert w.loop_box.isChecked() and w._backing.loop is not None and w.timeline.looping
        assert 6.5 <= qa.song_position() < 8.0                 # jumped, still inside (loose above: a loaded machine reads late, #69)
        assert w.loop_label.text().startswith("Loop · Pass 2")
        qa.wait_until(lambda: w.loop_label.text().startswith("Loop · Pass 3"), timeout_s=10)   # it still wraps
        assert qa.song_position() < 8.0
        click_timeline(qa, 10.0)                               # outside: the loop ends, the song plays on
        qa.wait(0.3)
        assert not w.loop_box.isChecked() and w._backing.loop is None and not w.timeline.looping
        assert qa.song_position() >= 10.0


def test_the_loop_box_and_the_loop_never_disagree_after_a_click_inside_the_widened_section(tmp_path):
    """Review 5 #4: a Section shorter than MIN_LOOP_S loops widened to it; a click in the widened part is inside the
    loop (the Transport goes on looping), so the Loop box stays ticked and the player keeps its loop."""
    with echo_singer(tmp_path) as qa:
        w = qa.window
        qa.mark_section(6.0, 6.05)                                 # loops 6.0 .. 6.2
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait_until(lambda: w.loop_label.text().startswith("Loop · Pass 2"), timeout_s=10)
        qa.seek(6.15)
        qa.wait(0.3)
        looping = w._backing.transport.loop is not None
        assert looping and w.loop_box.isChecked() and w._backing.loop is not None and w.timeline.looping


def test_editing_the_bpm_while_looping_keeps_the_pass_count(tmp_path):
    """Review 5 #5: a new BPM (or the tempo detected) while looping swaps the Count-in at the next wrap, and the
    Passes go on counting."""
    with echo_singer(tmp_path) as qa:
        w = qa.window
        qa.mark_section(4.0, 6.0)
        qa.set("between_box", "Count-in")
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait_until(lambda: w.loop_label.text().startswith("Loop · Pass 2"), timeout_s=20)
        qa.set("bpm_box", 90.0)
        qa.wait(0.2)
        assert w.loop_label.text().startswith("Loop · Pass 2")
        qa.wait_until(lambda: w.loop_label.text().startswith("Loop · Pass 3"), timeout_s=20)
        assert w._backing.transport.loop.count_in.bpm == 90.0
        qa.stop_take()
        passes = [r["pass_index"] for r in qa.takes() if r["pass_index"]]
    assert "1" in passes and passes.count("1") == 1                 # never back at Pass 1

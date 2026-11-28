"""Ticket #39 (QA fixes A): window size, stale status text, Loop and Section controls, locked devices during
calibration, replay playhead and numbers, plot axis, Guide lines, Speed progress, restoring the last choice, tooltips."""
import threading
import time

import numpy as np
import pytest
from audio import SR, quiet
from melody_helpers import library_song, reference_line
from PySide6 import QtWidgets
from app_speed import APP_SPEED
from qa_helpers import configure_latency, singing_song
from test_melody_window import window

from sing_pitch.library import Library
from sing_pitch.qa import QA
from sing_pitch.settings import Settings

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]
LONG_TITLE = "Ed Sheeran - Shape of You (Official Music Video) [Remastered 4K HD] feat. Someone Else"


def melody_window(qtbot, tmp_path, title="Do Re Mi", **kw):
    song = library_song(tmp_path, reference_line(NOTES), title=title, seconds=2.0)
    win = window(qtbot, tmp_path, quiet(3.0), **kw)
    win.reference_box.setCurrentText("Melody")
    return win, song


# --- #25 + #35: the window shrinks to about 1000 px and stays that size ---------------------------------------------

def test_the_window_can_be_1000_px_wide_with_a_youtube_length_song_title(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path, title=LONG_TITLE)
    assert win.song_box.currentText() == LONG_TITLE
    assert win.minimumSizeHint().width() <= 1000
    win.resize(900, 600)
    win.show()
    qtbot.waitExposed(win)
    assert win.width() <= 1000


def test_the_big_note_label_never_changes_the_window_width_while_singing(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path, title=LONG_TITLE)
    win.resize(1000, 650)
    win.show()
    qtbot.waitExposed(win)
    idle = (win.minimumSizeHint().width(), win.width(), win.note_label.width())
    for text in ("A4  -1 c", "G#4  +39 c", "C#4  out of key", "-"):
        win.note_label.setText(text)
        QtWidgets.QApplication.processEvents()
        assert (win.minimumSizeHint().width(), win.width(), win.note_label.width()) == idle, text
    assert idle[0] <= 1000


def row_widgets(win):
    """The visible widgets of each control row (the window's horizontal layouts), in order."""
    rows = []
    for i in range(win.layout().count()):
        row = win.layout().itemAt(i).layout()
        if row is not None:
            items = [row.itemAt(k).widget() for k in range(row.count())]
            rows.append([w for w in items if w is not None and w.isVisible()])
    return rows


def at_minimum_width(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path, title=LONG_TITLE)
    win.resize(900, 600)                       # asks for less than the minimum: gets the minimum
    win.show()
    qtbot.waitExposed(win)
    QtWidgets.QApplication.processEvents()
    return win


def test_at_the_minimum_width_every_slider_keeps_a_usable_width(qtbot, tmp_path):
    """QA #46: at 978 px the Guide vocal and Backing sliders were 20 px wide, like check boxes."""
    win = at_minimum_width(qtbot, tmp_path)
    for slider in (win.guide_slider, win.backing_slider, win.note_length_slider, win.tolerance_slider):
        assert slider.width() >= 80, slider.objectName()


def test_at_the_minimum_width_no_two_controls_overlap(qtbot, tmp_path):
    win = at_minimum_width(qtbot, tmp_path)
    for row in row_widgets(win):
        for a, b in zip(row, row[1:]):
            assert a.geometry().right() < b.geometry().left(), (a.objectName(), b.objectName())


def test_the_note_labels_thin_out_so_they_never_overlap_on_a_short_plot(qtbot, tmp_path):
    """QA #46: 24 semitone labels 10.9 px apart on a 261 px plot overlapped (G#4 on G4)."""
    win = at_minimum_width(qtbot, tmp_path)
    for height in (900, 600):
        win.resize(win.width(), height)
        QtWidgets.QApplication.processEvents()          # a resize alone re-spaces them
        labelled = [v for v, text in win._plot.getAxis("left")._tickLevels[0] if text]
        px_per_semitone = win._plot.getViewBox().height() / 24
        assert min(np.diff(labelled)) * px_per_semitone >= 14, height
        assert len(labelled) >= 6                              # still a note name every few semitones


def test_the_history_window_can_be_narrow_with_long_song_and_section_names(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    view = HistoryView(tmp_path)
    qtbot.addWidget(view)
    for box in (view.song_box, view.section_box):
        box.addItem(LONG_TITLE * 2, "x")
    assert view.minimumSizeHint().width() <= 1000


# --- #27: the status line never keeps stale text -------------------------------------------------------------------

def test_saving_a_section_replaces_the_earlier_error(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    win.timeline.section_marked.emit(0.5, 1.5)
    win.save_section_button.click()                       # no name yet
    assert "name" in win.summary_label.text()
    win.section_name_edit.setText("Chorus 1")
    win.save_section_button.click()
    assert win.summary_label.text() == 'Saved Section "Chorus 1".'


def test_renaming_a_section_replaces_the_earlier_error(qtbot, tmp_path):
    win, song = melody_window(qtbot, tmp_path)
    store = Library(tmp_path).sections(song.id)
    store.save("Chorus 1", 0.2, 0.9), store.save("Verse", 1.0, 1.8)
    win.reload_sections(select="Verse")
    win.section_name_edit.setText("Chorus 1")
    win.rename_section_button.click()                     # a name already taken
    assert win.summary_label.text().startswith("Cannot rename")
    win.section_name_edit.setText("Verse 2")
    win.rename_section_button.click()
    assert win.summary_label.text() == 'Renamed Section "Verse" to "Verse 2".'


def test_deleting_a_section_says_so(qtbot, tmp_path):
    win, song = melody_window(qtbot, tmp_path)
    Library(tmp_path).sections(song.id).save("Verse", 1.0, 1.8)
    win.reload_sections(select="Verse")
    win.summary_label.setText("something old")
    win.delete_section_button.click()
    assert win.summary_label.text() == 'Deleted Section "Verse".'


def test_save_rename_and_delete_are_greyed_when_there_is_nothing_for_them_to_act_on(qtbot, tmp_path):
    win, song = melody_window(qtbot, tmp_path)
    assert win.section_box.currentText() == "unsaved mark" and win.timeline.section is None
    assert not (win.save_section_button.isEnabled() or win.rename_section_button.isEnabled()
                or win.delete_section_button.isEnabled())
    win.timeline.section_marked.emit(0.5, 1.5)            # a mark: it can be saved, but not renamed or deleted
    assert win.save_section_button.isEnabled()
    assert not (win.rename_section_button.isEnabled() or win.delete_section_button.isEnabled())
    Library(tmp_path).sections(song.id).save("Verse", 1.0, 1.8)
    win.reload_sections(select="Verse")
    assert win.rename_section_button.isEnabled() and win.delete_section_button.isEnabled()


def test_stopping_a_replay_shows_the_takes_numbers_not_replaying(qtbot, tmp_path):
    from audio import harmonic
    from frames import hz_at
    from sing_pitch.audio_input import ArrayInput
    from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
    from sing_pitch.ui import MainWindow

    audio = np.concatenate([quiet(0.3), harmonic(hz_at(10), 1.0), quiet(0.3)])
    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(audio, SR, 256))
    qtbot.addWidget(win)
    win.toggle()
    qtbot.wait(100)
    win.toggle()
    win.toggle_replay()
    assert win.summary_label.text().startswith("Replaying")
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    win.toggle_replay()                                   # Stop replay
    assert "Replaying" not in win.summary_label.text() and "In tune" in win.summary_label.text()


def test_a_finished_replay_keeps_the_takes_numbers_and_says_it_finished(qtbot, tmp_path):
    from audio import harmonic
    from frames import hz_at
    from sing_pitch.audio_input import ArrayInput
    from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
    from sing_pitch.ui import MainWindow

    audio = np.concatenate([quiet(0.3), harmonic(hz_at(10), 1.0), quiet(0.3)])
    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(audio, SR, 256))
    qtbot.addWidget(win)
    win.toggle()
    qtbot.wait(100)
    win.toggle()
    numbers = win.summary_label.text()
    win.toggle_replay()
    qtbot.waitUntil(lambda: win._replay is None, timeout=6000)
    text = win.summary_label.text()
    assert "Replay finished" in text and "In tune" in text
    assert numbers.split("   |   ")[0] in text


# --- #28: Loop cannot stay ticked without a Section ----------------------------------------------------------------

def test_loop_unticks_and_greys_when_the_song_has_no_section_to_loop(qtbot, tmp_path):
    song_a = library_song(tmp_path, reference_line(NOTES), title="A song", seconds=2.0)
    library_song(tmp_path, reference_line(NOTES), title="B song", seconds=2.0)
    Library(tmp_path).sections(song_a.id).save("Hook", 0.2, 1.2)
    win = window(qtbot, tmp_path, quiet(3.0))
    win.reference_box.setCurrentText("Melody")
    win.song_box.setCurrentIndex(win.song_box.findData(song_a.id))
    win.reload_sections(select="Hook")
    assert win.loop_box.isEnabled()
    win.loop_box.setChecked(True)
    win.song_box.setCurrentIndex(1 - win.song_box.currentIndex())          # B: no Section, no mark
    assert win.section_box.currentText() == "unsaved mark" and win.timeline.section is None
    assert not win.loop_box.isChecked() and not win.loop_box.isEnabled()
    win.toggle()
    assert win._backing.__dict__.get("loop") is None
    win.toggle()


def test_loop_is_enabled_again_once_a_section_is_marked(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    assert not win.loop_box.isEnabled()
    win.timeline.section_marked.emit(0.5, 1.5)
    assert win.loop_box.isEnabled()
    win.loop_box.setChecked(True)
    win.clear_button.click()                              # the mark is forgotten: no Section left to loop
    assert not win.loop_box.isChecked() and not win.loop_box.isEnabled()


# --- #30: devices are locked while calibrating ---------------------------------------------------------------------

def calibrating_window(qtbot, tmp_path):
    from sing_pitch.calibration import click_track
    from test_calibration import performance
    from test_output_selector import window as device_window

    class Player:
        def __init__(self, audio, rate):
            pass

        def start(self): ...
        def position_s(self):
            return 256 / SR
        def stop(self): ...

    _, times = click_track(SR)
    return device_window(qtbot, tmp_path, audio=performance(times, 0.27, times[-1] + 1.5),
                         calibration_player_factory=Player)


def test_input_output_and_refresh_are_locked_during_calibration_and_free_after(qtbot, tmp_path):
    win = calibrating_window(qtbot, tmp_path)
    win.start_calibration()
    assert not (win.device_box.isEnabled() or win.output_box.isEnabled() or win.refresh_button.isEnabled())
    win.finish_calibration()
    assert win.device_box.isEnabled() and win.output_box.isEnabled() and win.refresh_button.isEnabled()


def test_the_latency_is_stored_for_the_output_that_was_measured(qtbot, tmp_path):
    from test_output_selector import choose

    win = calibrating_window(qtbot, tmp_path)
    choose(win, "MacBook Pro Speakers")
    win.start_calibration()
    choose(win, "WF-1000XM5")                             # however it got changed meanwhile
    win.finish_calibration()
    saved = Settings.load(tmp_path)
    assert saved.is_calibrated("MacBook Pro Speakers") and not saved.is_calibrated("WF-1000XM5")
    assert "MacBook Pro Speakers" in win.summary_label.text()


# --- #36: the replay drives the playhead ---------------------------------------------------------------------------

def test_the_timeline_playhead_follows_the_replayed_song_time(tmp_path):
    singing_song(tmp_path, tail_s=3.0)
    configure_latency(tmp_path, 0.12)
    with QA(tmp_path, mic="echo", time_scale=APP_SPEED) as qa:
        qa.set("reference_box", "Melody")
        qa.set("guide_box", True)
        qa.set("backing_slider", 0)
        qa.start_take(at_s=0.5)
        qa.wait(2.0)
        qa.stop_take()
        taken_to = qa.window.timeline.playhead_s
        assert taken_to > 2.0
        qa.click("replay_button")
        seen = []
        for wall_s in (0.3, 0.8):
            until = time.perf_counter() + wall_s
            qa.wait_until(lambda: time.perf_counter() >= until)
            seen.append(qa.window.timeline.playhead_s)
        assert 0.3 < seen[0] < seen[1] < taken_to - 0.5       # moving from where the Take began, not stuck at its end


# --- plot axis: song time ------------------------------------------------------------------------------------------

def test_the_plot_axis_shows_song_time_as_minutes_and_seconds(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    win._show_melody(win._reference(), 0.5)               # the song clock: song time = voice time - latency
    axis = win._plot.getAxis("bottom")
    assert axis.tickStrings([65.5, 66.5, 125.5], 1, 1) == ["1:05", "1:06", "2:05"]


def test_the_plot_axis_puts_its_ticks_on_whole_song_seconds(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    win._show_melody(win._reference(), 0.5)               # song time = voice time - 0.5
    axis = win._plot.getAxis("bottom")
    (_, values), = axis.tickValues(0.0, 8.0, 900)         # the view shows song time -0.5 .. 7.5
    assert values == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])      # song seconds 0 .. 7
    assert axis.tickStrings(values, 1, 1) == ["0:00", "0:01", "0:02", "0:03", "0:04", "0:05", "0:06", "0:07"]


def test_the_plot_axis_of_a_take_without_a_song_shows_time_since_start(qtbot, tmp_path):
    win = window(qtbot, tmp_path, quiet(1.0))
    assert win._plot.getAxis("bottom").tickStrings([-2.0, 0.0, 61.0], 1, 1) == ["", "0:00", "1:01"]


# --- Guide lines ---------------------------------------------------------------------------------------------------

def test_guide_lines_are_clearly_visible_but_fainter_and_no_thicker_than_the_active_line(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    active = win._ref_curve.opts["pen"]
    for curve in win._guide_curves:
        pen = curve.opts["pen"]
        assert 120 <= pen.color().alpha() < active.color().alpha()
        assert pen.width() >= 2 and pen.width() <= active.width()     # the active line is 2 px: its Tolerance band is its body (#51)


# --- Speed preparation progress ------------------------------------------------------------------------------------

class ProgressStretcher:
    """Reports 45 % done, then waits for the test."""

    reports_progress = True

    def __init__(self):
        self.gate = threading.Event()

    def __call__(self, audio, sample_rate, speed, progress=None):
        progress(0.45)
        assert self.gate.wait(10), "the test never opened the gate"
        n = round(len(audio) / speed)
        return np.ascontiguousarray(audio[np.minimum((np.arange(n) * speed).astype(int), len(audio) - 1)])


def test_speed_preparation_shows_how_far_it_is(qtbot, tmp_path):
    stretcher = ProgressStretcher()
    win, _ = melody_window(qtbot, tmp_path, stretcher=stretcher)
    win.speed_box.setValue(70)
    qtbot.waitUntil(lambda: win.speed_label.text() == "preparing 70 % … 45 %", timeout=3000)
    stretcher.gate.set()
    qtbot.waitUntil(lambda: win.speed_label.text() == "", timeout=5000)


# --- restoring the last Reference, song and Section ------------------------------------------------------------------

def test_the_last_reference_song_and_section_come_back_on_launch(qtbot, tmp_path):
    library_song(tmp_path, reference_line(NOTES), title="A song", seconds=2.0)
    song_b = library_song(tmp_path, reference_line(NOTES), title="B song", seconds=2.0)
    first = window(qtbot, tmp_path, quiet(3.0))
    assert first.reference_box.currentText() == "Nearest note"              # nothing remembered yet
    first.reference_box.setCurrentText("Melody")
    first.song_box.setCurrentIndex(first.song_box.findData(song_b.id))
    Library(tmp_path).sections(song_b.id).save("Hook", 0.2, 1.2)
    first.reload_sections(select="Hook")
    assert Settings.load(tmp_path).last_reference == "Melody" and Settings.load(tmp_path).last_song == song_b.id

    again = window(qtbot, tmp_path, quiet(3.0))
    assert again.reference_box.currentText() == "Melody" and again.song_box.currentData() == song_b.id
    assert again.section_box.currentData() == "Hook" and again.timeline.section == pytest.approx((0.2, 1.2))
    assert again.loop_box.isEnabled()                      # L then Start: two actions to a looped Section


def test_a_remembered_song_that_is_gone_falls_back_to_the_first(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(NOTES), title="A song", seconds=2.0)
    s = Settings.load(tmp_path)
    s.last_reference, s.last_song = "Melody", "no-such-song"
    s.save()
    win = window(qtbot, tmp_path, quiet(3.0))
    assert win.reference_box.currentText() == "Melody" and win.song_box.currentData() == song.id


# --- tooltips -------------------------------------------------------------------------------------------------------

CONTROLS = (QtWidgets.QAbstractButton, QtWidgets.QComboBox, QtWidgets.QAbstractSlider, QtWidgets.QAbstractSpinBox,
            QtWidgets.QLineEdit)


def controls_of(root):
    """Every control widget under root (scroll bars of the plots are not controls)."""
    found = [w for w in root.findChildren(QtWidgets.QWidget) if isinstance(w, CONTROLS)
             and not isinstance(w.parent(), QtWidgets.QAbstractItemView)]      # not a table's own corner button
    return [w for w in found if not isinstance(w, QtWidgets.QScrollBar)]


def named_controls(root):
    return [w for w in controls_of(root) if w.objectName() and not w.objectName().startswith("qt_")]


def test_every_control_of_the_main_window_has_a_tooltip(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    controls = named_controls(win)
    assert len(controls) > 20
    assert [w.objectName() for w in controls if not w.toolTip().strip()] == []


def test_every_control_of_the_history_window_has_a_tooltip(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    view = HistoryView(tmp_path)
    qtbot.addWidget(view)
    controls = controls_of(view)
    assert len(controls) == 9 and [w for w in controls if not w.toolTip().strip()] == []

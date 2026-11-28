"""Ticket #23, end to end on the QA driver (real window, fake mic and output): move around while paused, see that a loop
repeats, remove a loop or a mark. The song is the oracle song three times over (about 11 s); the echo singer sings
exactly what the Guide vocal plays, so every Pass of a Section scores near 100 %."""
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest
from app_speed import APP_SPEED
from qa_helpers import NOTES, configure_latency, singing_song

from sing_pitch.qa import QA

SCALE = APP_SPEED


def echo_singer(home):
    singing_song(home, notes=NOTES * 3, tail_s=3.0)
    configure_latency(home, 0.12)
    qa = QA(home, mic="echo", time_scale=SCALE)
    qa.set("reference_box", "Melody")
    qa.set("guide_box", True)
    qa.set("guide_slider", 100)
    qa.set("backing_slider", 0)
    return qa


def click_timeline(qa, song_s):
    """A real left click on the strip at song time song_s."""
    tl = qa.window.timeline
    vb = tl.plot.vb
    point = tl.mapFromScene(vb.mapViewToScene(QtCore.QPointF(song_s, sum(vb.viewRange()[1]) / 2)))
    QTest.mouseClick(tl.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
    qa._app.processEvents()


def reference_song_at_now_line(qa):
    """The song time the plot draws at its now-line (the reference line is sampled at this song time)."""
    w = qa.window
    import numpy as np

    return float(w._song_times(np.array([w._now_line.value()]))[0])


# --- 1. navigate while paused ---------------------------------------------------------------------------------------

def test_a_click_while_paused_moves_the_song_the_playhead_and_the_drawn_reference_and_play_goes_on_from_there(tmp_path):
    with echo_singer(tmp_path) as qa:
        qa.start_take()
        qa.wait(3.0)
        qa.pause()
        qa.wait(0.5)
        paused_at = qa.song_position()
        click_timeline(qa, 8.0)
        qa.wait(0.3)
        assert abs(qa.song_position() - 8.0) < 0.05 and qa.song_position() != paused_at   # still paused: no drift
        assert abs(qa.window.timeline.playhead_s - 8.0) < 0.05
        assert abs(reference_song_at_now_line(qa) - 8.0) < 0.1
        qa.wait(0.5)
        assert abs(qa.song_position() - 8.0) < 0.05
        qa.pause()                                                   # Play
        qa.wait(0.5)
        assert 8.4 < qa.song_position() < 9.5                        # went on from the new spot (loose above: a loaded machine reads late, #69)


def test_the_arrow_keys_move_the_song_while_paused(tmp_path):
    with echo_singer(tmp_path) as qa:
        qa.start_take()
        qa.wait(6.0)
        qa.pause()
        qa.wait(0.3)
        before = qa.song_position()
        qa.press("Left")
        qa.wait(0.3)
        assert abs(qa.song_position() - (before - 5.0)) < 0.05
        assert abs(qa.window.timeline.playhead_s - qa.song_position()) < 0.05
        qa.press("Right")
        qa.press("Right")
        qa.wait(0.3)
        assert abs(qa.song_position() - (before + 5.0)) < 0.05


def test_dragging_the_playhead_while_paused_moves_the_song(tmp_path):
    with echo_singer(tmp_path) as qa:
        qa.start_take()
        qa.wait(2.0)
        qa.pause()
        qa.wait(0.3)
        tl = qa.window.timeline
        assert tl._playhead.movable
        tl._playhead.setPos(7.0)                                     # what dragging does, then dropping it
        tl._playhead.sigPositionChangeFinished.emit(tl._playhead)
        qa._app.processEvents()
        qa.wait(0.3)
        assert abs(qa.song_position() - 7.0) < 0.05 and abs(tl.playhead_s - 7.0) < 0.05


def test_seeking_twice_while_paused_does_not_split_the_take_again(tmp_path):
    with echo_singer(tmp_path) as qa:
        qa.start_take()
        qa.wait(4.0)
        qa.pause()
        qa.wait(0.3)
        qa.seek(2.0)
        qa.wait(0.2)
        qa.seek(6.0)
        qa.wait(0.2)
        qa.pause()
        qa.wait(1.0)
        qa.stop_take()
        rows = qa.takes()
        assert len(rows) == 1 and float(rows[0]["pct_in_tune"]) > 90   # the 4 s before the first jump; the 1 s after is too short


# --- 2. a loop that visibly repeats ---------------------------------------------------------------------------------

def test_a_looping_section_is_highlighted_and_counts_its_passes(tmp_path):
    with echo_singer(tmp_path) as qa:
        w = qa.window
        qa.mark_section(4.0, 8.0)                                    # 4 s: each Pass has the 3 s of singing it needs to be kept
        assert w.loop_label.text() == "" and not w.timeline.looping
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait_until(lambda: w.loop_label.text().startswith("Loop · Pass 1"), timeout_s=10)
        assert w.timeline.looping
        qa.wait_until(lambda: w.loop_label.text().startswith("Loop · Pass 3"), timeout_s=20)     # increments on every wrap
        assert "%" in w.loop_label.text()                                                          # the last Pass's result
        qa.set("loop_box", False)
        assert w.loop_label.text() == "" and not w.timeline.looping


def test_unticking_loop_lets_the_song_play_on_past_the_section(tmp_path):
    with echo_singer(tmp_path) as qa:
        qa.mark_section(4.0, 5.5)
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait(5.0)
        assert qa.song_position() < 5.6                              # held inside the Lead-in .. Section end
        qa.set("loop_box", False)
        qa.wait(2.5)
        assert qa.song_position() > 6.0 and qa.window._backing.loop is None


def test_the_l_key_stops_the_loop_even_after_a_section_was_named(tmp_path):
    """Focus stayed in the Section name field after Save, so L typed an "l" instead of reaching the Loop toggle."""
    with echo_singer(tmp_path) as qa:
        w = qa.window
        w.show()
        w.activateWindow()
        qa.mark_section(4.0, 5.5)
        qa.set("section_name_edit", "Verse")
        w.section_name_edit.setFocus()
        qa.click("save_section_button")
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait(2.0)
        assert w.loop_box.isChecked()
        w.section_name_edit.setFocus()                               # the user is typing a name...
        QTest.keyClick(w.section_name_edit, QtCore.Qt.Key_Return)    # ... and confirms it
        QTest.keyClick(QtWidgets.QApplication.focusWidget() or w, QtCore.Qt.Key_L)
        qa._app.processEvents()
        assert not w.loop_box.isChecked() and w._backing.loop is None and w.section_name_edit.text() == "Verse"


# --- 3. clear ------------------------------------------------------------------------------------------------------

def test_clear_removes_the_mark_and_stops_the_loop_but_saved_sections_stay(tmp_path):
    with echo_singer(tmp_path) as qa:
        w = qa.window
        qa.mark_section(2.0, 3.0)
        qa.set("section_name_edit", "Keep")
        qa.click("save_section_button")
        qa.mark_section(4.0, 5.5)                                    # the unsaved mark
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait(1.0)
        qa.click("clear_button")
        assert w.timeline.section is None and w._backing.loop is None and not w.loop_box.isChecked()
        assert w.section_box.currentData() is None and w.section_box.findData("Keep") >= 0   # saved one is still there
        qa.wait(3.0)
        assert qa.song_position() > 5.6                              # the song went on past the old Section
        assert w._sections().mark is None and [s.name for s in w._sections().saved()] == ["Keep"]


def test_escape_clears_a_saved_section_from_view_without_deleting_it(tmp_path):
    with echo_singer(tmp_path) as qa:
        w = qa.window
        qa.mark_section(2.0, 3.0)
        qa.set("section_name_edit", "Keep")
        qa.click("save_section_button")
        assert w.timeline.section is not None and w.section_box.currentData() == "Keep"
        qa.set("loop_box", True)
        qa.press("Escape")
        assert w.timeline.section is None and not w.loop_box.isChecked()
        assert [s.name for s in w._sections().saved()] == ["Keep"]
        w.section_box.setCurrentIndex(w.section_box.findData("Keep"))      # and it can be picked again
        assert w.timeline.section == (2.0, 3.0)

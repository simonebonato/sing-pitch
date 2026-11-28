"""#64: keyboard focus is never left parked in a field where Space or the arrows would be swallowed, so "I pressed Space
and the song kept playing" cannot come from the keyboard. Real mouse and key events through the QA driver."""
import pytest
from app_speed import APP_SPEED
from pause_fuzz import build_song
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from sing_pitch.qa import QA

SCALE = APP_SPEED


@pytest.fixture
def qa(tmp_path):
    build_song(tmp_path)
    with QA(tmp_path, mic="echo", time_scale=SCALE) as qa:
        qa.set("reference_box", "Melody")
        qa.window.show()
        qa.window.activateWindow()
        qa.wait_until(lambda: QtWidgets.QApplication.activeWindow() is qa.window, timeout_s=5)
        qa.mark_section(10.0, 14.0)
        yield qa


def key(widget, name):
    QTest.keyClick(widget, getattr(QtCore.Qt, f"Key_{name}"))
    QtWidgets.QApplication.processEvents()


def click(qa, name, pos=None):
    """A real left click on the widget (its viewport for a scroll area), then close any popup it opened."""
    widget = qa.widget(name)
    target = widget.viewport() if isinstance(widget, QtWidgets.QAbstractScrollArea) else widget
    QTest.mouseClick(target, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, pos or QtCore.QPoint())
    popup = QtWidgets.QApplication.activePopupWidget()
    if popup is not None:
        popup.close()
    QtWidgets.QApplication.processEvents()


def typing_in_the_name_field(qa):
    click(qa, "section_name_edit")
    assert QtWidgets.QApplication.focusWidget() is qa.widget("section_name_edit")
    QTest.keyClicks(qa.widget("section_name_edit"), "ab")


def space_pauses(qa) -> bool:
    """Space to whatever has the focus now: did the song pause? (Resumes it afterwards.)"""
    win = qa.window
    assert not win._paused
    key(QtWidgets.QApplication.focusWidget() or win, "Space")
    paused = win._paused
    if paused:
        win.toggle_pause()
    return paused


def test_while_typing_a_name_space_is_a_space_and_does_not_pause(qa):
    qa.start_take(at_s=2)
    typing_in_the_name_field(qa)
    key(qa.widget("section_name_edit"), "Space")
    QTest.keyClicks(qa.widget("section_name_edit"), "c")
    assert qa.text("section_name_edit") == "ab c" and not qa.window._paused
    qa.stop_take()


def leave_by_enter(qa):
    key(qa.widget("section_name_edit"), "Return")


def leave_by_escape(qa):
    key(qa.widget("section_name_edit"), "Escape")


def leave_by_click_on_the_plot(qa):
    click(qa, "plot_widget")


def leave_by_click_on_the_timeline(qa):
    click(qa, "timeline")


def leave_by_click_on_empty_window_area(qa):
    QTest.mouseClick(qa.window, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(1, 1))
    QtWidgets.QApplication.processEvents()


def leave_by_save(qa):
    click(qa, "save_section_button")


@pytest.mark.parametrize("leave", [leave_by_enter, leave_by_escape, leave_by_click_on_the_plot,
                                   leave_by_click_on_the_timeline, leave_by_click_on_empty_window_area, leave_by_save])
def test_focus_leaves_the_name_field_and_space_pauses_again(qa, leave):
    qa.start_take(at_s=2)
    typing_in_the_name_field(qa)
    leave(qa)
    assert QtWidgets.QApplication.focusWidget() is not qa.widget("section_name_edit")
    assert space_pauses(qa), "Space was swallowed"
    qa.stop_take()


def test_escape_in_the_field_only_leaves_it_and_does_not_clear_the_section(qa):
    qa.start_take(at_s=2)
    typing_in_the_name_field(qa)
    leave_by_escape(qa)
    assert qa.window.timeline.section is not None
    qa.stop_take()


def test_a_take_starting_takes_the_focus_out_of_the_field(qa):
    click(qa, "section_name_edit")
    assert QtWidgets.QApplication.focusWidget() is qa.widget("section_name_edit")
    qa.window.toggle()                                   # Start from the keyboard or a shortcut, no click
    assert qa.running
    assert space_pauses(qa)
    qa.stop_take()


@pytest.mark.parametrize("name", ["bpm_box", "beats_box", "speed_box"])
def test_a_click_into_a_number_box_still_lets_space_and_the_arrows_act(qa, name):
    qa.start_take(at_s=20)
    qa.wait(0.3)
    click(qa, name)
    assert space_pauses(qa), f"Space was swallowed by {name}"
    before = qa.song_position()
    key(QtWidgets.QApplication.focusWidget() or qa.window, "Left")
    qa.wait(0.1)
    assert qa.song_position() < before - 2, f"Left was swallowed by {name}"
    qa.stop_take()


@pytest.mark.parametrize("name", ["reference_box", "song_box", "device_box", "output_box", "guide_box", "guide_slider",
                                  "backing_slider", "refresh_button", "clear_button", "timeline",
                                  "plot_widget", "pause_button"])
def test_every_other_focusable_control_lets_space_and_the_arrows_act(qa, name):
    qa.start_take(at_s=20)
    qa.wait(0.3)
    widget = qa.widget(name)
    if widget.isEnabled():
        click(qa, name)
    assert qa.running
    if name == "pause_button":
        qa.window.toggle_pause()                          # the click paused it: back to playing
    assert space_pauses(qa), f"Space was swallowed after a click on {name}"
    qa.seek(20)
    qa.wait(0.2)
    before = qa.song_position()
    key(QtWidgets.QApplication.focusWidget() or qa.window, "Left")
    qa.wait(0.1)
    assert qa.song_position() < before - 2, f"Left was swallowed after a click on {name}"
    qa.stop_take()


def test_a_field_that_holds_focus_during_a_take_says_what_a_space_will_do(qa):
    qa.start_take(at_s=2)
    before = qa.summary()
    typing_in_the_name_field(qa)
    assert "Space" in qa.summary() and "Enter" in qa.summary()
    leave_by_enter(qa)
    assert qa.summary() == before
    qa.stop_take()


def test_space_in_the_history_window_pauses_the_take_too(qa):
    qa.start_take(at_s=2)
    qa.window.open_history()
    history = qa.window.history_window
    history.activateWindow()
    qa.wait_until(lambda: QtWidgets.QApplication.activeWindow() is history, timeout_s=5)
    key(history, "Space")
    assert qa.window._paused
    qa.window.toggle_pause()
    qa.stop_take()

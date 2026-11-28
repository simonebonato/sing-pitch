"""Space, the Pause button and Stop reach the sound (#50): the song is paused in one place, whatever else is going on,
and a stop leaves no stream playing.

The fuzz test (test_pause_random.py) holds the invariants over random sequences of actions; these are the faults it
found or the review of the lifecycle did, one by one, each red before its fix."""
import time
from types import SimpleNamespace

import numpy as np
import pytest
from audio import SR, harmonic, quiet
from pytest import approx
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest
from test_live_runner_failures import Backing, QuietTracker, tracker_threads
from test_view_follows_the_song_heard import World, same_view
from test_window_timeline import running_window

from sing_pitch.audio_input import ArrayInput
from sing_pitch.live import LiveRunner
from sing_pitch.playback import BackingPlayer
from sing_pitch.reference import NearestNote
from sing_pitch.ui import main_window


class MicThatCannotStop(ArrayInput):
    def stop(self):
        raise OSError("the input device went away")


class Stream:
    """A fake output stream that fails to stop, as a device that switched profile or was unplugged does."""

    def __init__(self, **kwargs):
        self.callback, self.stops, self.aborted, self.closed = kwargs["callback"], 0, False, False

    def start(self):
        pass

    def stop(self):
        self.stops += 1
        raise OSError("PortAudio: invalid device")

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


def test_the_song_is_stopped_even_when_the_microphone_cannot_be_stopped(tmp_path):
    backing, before = Backing(), len(tracker_threads())
    runner = LiveRunner(MicThatCannotStop(quiet(0.2), SR, 256), QuietTracker(), NearestNote(), tmp_path,
                        backing=backing)
    runner.start()
    runner.stop()                                     # the Take is saved, the failure does not lose it
    assert backing.stopped and len(tracker_threads()) == before


def test_a_stream_that_will_not_stop_is_aborted_and_closed_and_forgotten():
    streams = []

    def factory(**kwargs):
        streams.append(Stream(**kwargs))
        return streams[-1]

    player = BackingPlayer(np.zeros((4800, 2), np.float32), 48000, stream_factory=factory, sample_rate=48000)
    player.start()
    player.stop()                                     # nothing raised: the sound has to stop, the error cannot help
    assert streams[0].aborted and streams[0].closed
    player.stop()
    assert streams[0].stops == 1                      # a second stop has nothing to do


def test_the_player_says_it_is_paused_from_the_transport_and_nowhere_else():
    player = BackingPlayer(np.zeros((4800, 2), np.float32), 48000, stream_factory=lambda **kw: Stream(**kw),
                           sample_rate=48000)
    assert not player.paused
    player.pause()
    assert player.paused and player.transport.paused
    player.play()
    assert not player.paused


def test_the_pause_button_follows_the_player_whoever_paused_it(qtbot, tmp_path):
    win, backings = running_window(qtbot, tmp_path)
    win.toggle()
    backings[0].pause()                               # not through the window
    win.tick()
    assert win.pause_button.text() == "Play" and win._paused
    win.toggle_pause()                                # Space now resumes what really is paused
    assert backings[0].calls[-1] == ("play",) and win.pause_button.text() == "Pause" and not win._paused
    win.toggle()


def test_space_is_not_a_repeating_key_so_holding_it_cannot_flip_pause_and_play(qtbot, tmp_path):
    win, _ = running_window(qtbot, tmp_path)
    space = [s for s in win.findChildren(QtGui.QShortcut) if s.key() == QtGui.QKeySequence(QtCore.Qt.Key_Space)]
    assert len(space) == 1 and not space[0].autoRepeat()


def test_space_pauses_from_the_tempo_box_too_where_it_would_have_been_typed(qtbot, tmp_path):
    win, backings = running_window(qtbot, tmp_path)
    win.show()
    win.activateWindow()
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is win)
    win.toggle()
    win.bpm_box.setFocus()
    focus = QtWidgets.QApplication.focusWidget()
    assert focus is win.bpm_box
    QTest.keyClick(focus, QtCore.Qt.Key_Space)
    assert backings[0].calls == [("pause",)] and win.pause_button.text() == "Play"
    win.toggle()


def test_a_start_that_fails_after_the_song_began_stops_it_and_says_so(qtbot, tmp_path, monkeypatch):
    win, backings = running_window(qtbot, tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("the plot could not be set up")

    monkeypatch.setattr(win, "_show_melody", boom)
    win.toggle()
    assert backings[0].started and backings[0].stopped
    assert win._runner is None and win._backing is None and win.button.text() == "Start"
    assert "Could not start" in win.summary_label.text()


def test_closing_the_window_mid_take_saves_the_take_stops_the_song_and_closes_history_too(qtbot, tmp_path):
    from sing_pitch.take_store import _rows

    win, backings = running_window(qtbot, tmp_path)
    win._input_factory = lambda _i: ArrayInput(harmonic(220.0, 1.0), SR, 256)    # something is sung: a Take with nothing scored is not saved
    win.show()
    win.open_history()
    before = len(tracker_threads())
    win.toggle()
    assert backings[0].started and not backings[0].stopped
    qtbot.wait(300)                                            # the worker scores what was sung
    assert len(tracker_threads()) == before + 1
    win.close()
    assert backings[0].stopped and win._runner is None
    assert len(_rows(tmp_path)) == 1                           # saved, as Stop saves it
    assert len(tracker_threads()) == before                    # the worker is gone
    assert win.history_window.isHidden()                       # nothing keeps the app alive
    assert not any(w.isVisible() for w in (win, win.history_window))


def test_closing_the_window_during_a_replay_ends_it_cleanly(qtbot, tmp_path):
    world = World(qtbot, tmp_path)
    world.advance(2.0)
    world.win.toggle()
    world.win.toggle_replay()
    assert world.win._replay is not None
    world.win.close()
    assert world.win._replay is None and not world.win._timer.isActive()


def test_space_pauses_a_replay_and_it_goes_on_from_where_it_stood(qtbot, tmp_path, monkeypatch):
    world = World(qtbot, tmp_path)
    world.advance(3.0)
    world.win.toggle()                                # a saved Take to replay
    win = world.win
    clock = [1000.0]
    monkeypatch.setattr(main_window, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    win.toggle_replay()
    win._timer.stop()                                 # the test ticks
    assert win.pause_button.isEnabled() and win.pause_button.text() == "Pause"
    clock[0] += 1.5
    win.tick()
    win.toggle_pause()
    assert win.pause_button.text() == "Play" and win._paused
    frozen, playhead = world.view(), win.timeline.playhead_s
    for _ in range(3):
        clock[0] += 2.0                               # time passes, the replay does not
        win.tick()
        assert same_view(world.view(), frozen) and win.timeline.playhead_s == playhead
    win.toggle_pause()
    assert win.pause_button.text() == "Pause" and not win._paused
    assert win._replay_elapsed() == approx(1.5)       # it goes on from where it stood
    clock[0] += 0.5
    assert win._replay_elapsed() == approx(2.0)
    win.toggle_replay()
    assert not win.pause_button.isEnabled() and win.pause_button.text() == "Pause"

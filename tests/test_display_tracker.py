"""Tracker choice (Simone, 2026-11-26): with Fast chosen, the live dots come from a second, faster tracker while the
saved Take (its frames, summary and History) is still scored from the neural tracker, so History stays comparable."""
import numpy as np
import pytest
from audio import SR, harmonic, quiet
from frames import hz_at

from sing_pitch.audio_input import ArrayInput
from sing_pitch.live import LiveRunner
from sing_pitch.pitch import PitchFrame
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.reference import Band, NearestNote


class InTuneTracker:
    """A stand-in display tracker that hears an A4 exactly in tune, one frame per 5.3 ms of audio, from 0.4 s on."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._n, self.resets = 0, getattr(self, "resets", 0) + 1

    def push(self, chunk, sample_rate):
        out = []
        for _ in range(len(chunk) // 256):
            t = self._n * 256 / sample_rate
            self._n += 1
            voiced = t >= 0.4
            out.append(PitchFrame(t, hz_at(0) if voiced else 0.0, 0.95 if voiced else 0.0, -20.0 if voiced else -90.0))
        return out

    def flush(self):
        return []


@pytest.fixture(scope="module")
def tracker():
    return SwiftF0Tracker()


def run(tracker, tmp_path, display):
    audio = np.concatenate([quiet(0.4), harmonic(hz_at(30), 2.0), quiet(0.4)])        # sung 30 cents sharp
    runner = LiveRunner(ArrayInput(audio, SR, chunk=256), tracker, NearestNote(), home=tmp_path,
                        display_tracker=display)
    runner.start()
    record = runner.stop()
    return runner, record


def test_with_a_display_tracker_the_dots_come_from_it_and_the_saved_take_from_the_neural_tracker(tracker, tmp_path):
    display = InTuneTracker()
    runner, record = run(tracker, tmp_path, display)
    dots = runner.drain_display()
    assert dots and {s.band for s in dots if s.band} == {Band.GREEN}           # what the fast tracker heard
    assert {s.band for s in runner.drain() if s.band} == {Band.YELLOW}       # the neural scores, not drawn
    assert record.summary.tendency == "sharp" and record.summary.median_abs_cents > 20   # saved from the neural one
    assert display.resets >= 2                                                # reset with the Take like the neural one


def test_without_a_display_tracker_nothing_changes(tracker, tmp_path):
    runner, record = run(tracker, tmp_path, None)
    assert runner.drain_display() is None
    assert {s.band for s in runner.drain() if s.band} == {Band.YELLOW}
    assert record.summary.tendency == "sharp"


def test_the_display_dots_are_on_the_same_time_axis_as_the_neural_ones(tracker, tmp_path):
    runner, _ = run(tracker, tmp_path, InTuneTracker())
    dots = [s for s in runner.drain_display() if s.band]
    scores = [s for s in runner.drain() if s.band]
    assert abs(dots[0].t - 0.4) < 0.03 and abs(scores[0].t - 0.4) < 0.08     # both start where the voice does


# --- the window: a Tracker box, Accurate (the neural network) or Fast (the display tracker) --------------------------
def window(qtbot, home, audio, made=None):
    from sing_pitch.ui import MainWindow

    def fast():
        t = InTuneTracker()
        if made is not None:
            made.append(t)
        return t

    win = MainWindow(SwiftF0Tracker(), home=home, input_factory=lambda _i: ArrayInput(audio, SR, 256),
                     fast_tracker_factory=fast)
    qtbot.addWidget(win)
    win.reference_box.setCurrentText("Nearest note")
    return win


SHARP = np.concatenate([quiet(0.4), harmonic(hz_at(30), 2.0), quiet(0.4)])


def sing_a_take(qtbot, win):
    win.toggle()
    qtbot.waitUntil(lambda: sum(b is not None for b in win._band) > 40, timeout=5000)
    win.toggle()


def test_the_tracker_box_offers_accurate_and_fast_and_starts_on_accurate(qtbot, tmp_path):
    win = window(qtbot, tmp_path, SHARP)
    assert [win.tracker_box.itemText(i) for i in range(win.tracker_box.count())] == ["Accurate", "Fast"]
    assert win.tracker_box.currentText() == "Accurate"


def test_the_choice_is_remembered_for_the_next_window(qtbot, tmp_path):
    from sing_pitch.settings import Settings

    win = window(qtbot, tmp_path, SHARP)
    win.tracker_box.setCurrentText("Fast")
    assert Settings.load(tmp_path).tracker == "fast"
    assert window(qtbot, tmp_path, SHARP).tracker_box.currentText() == "Fast"


def test_fast_draws_the_fast_trackers_dots_and_saves_the_neural_score(qtbot, tmp_path):
    made = []
    win = window(qtbot, tmp_path, SHARP, made)
    win.tracker_box.setCurrentText("Fast")
    win.toggle()
    assert not win.tracker_box.isEnabled()                        # fixed during a Take
    qtbot.waitUntil(lambda: sum(b is not None for b in win._band) > 40, timeout=5000)
    assert {b for b in win._band if b is not None} == {Band.GREEN}           # the fast tracker's dots
    win.toggle()
    assert win.tracker_box.isEnabled()
    assert made                                                   # made when the Take started
    import csv
    rows = list(csv.DictReader((tmp_path / "takes.csv").open()))
    assert rows[-1]["tendency"] == "sharp"                        # scored from the neural tracker


def test_accurate_draws_the_neural_dots(qtbot, tmp_path):
    made = []
    win = window(qtbot, tmp_path, SHARP, made)
    sing_a_take(qtbot, win)
    assert {b for b in win._band if b is not None} == {Band.YELLOW}
    assert not made


class SlowTracker(InTuneTracker):
    """A display tracker much slower than the audio (20 ms of work per 5.3 ms chunk): LiveRunner must let it skip."""

    def __init__(self):
        super().__init__()
        self.skipped = 0

    def push(self, chunk, sample_rate):
        import time
        time.sleep(0.02)
        return super().push(chunk, sample_rate)

    def skip(self, n):
        self._n += n // 256
        self.skipped += n


class BrokenTracker(InTuneTracker):
    def push(self, chunk, sample_rate):
        raise RuntimeError("display tracker bug")


def test_a_display_tracker_that_falls_behind_skips_and_the_saved_take_does_not_change(tracker, tmp_path):
    """Measured 2026-11-27 (Windows laptop, Fast): on the worker beside the neural tracker the dots fell 4.1 s behind
    in 30 s. The display tracker has its own thread and lets chunks go by while more than DISPLAY_BACKLOG_CHUNKS wait."""
    _, alone = run(tracker, tmp_path / "alone", None)
    slow = SlowTracker()
    runner, record = run(tracker, tmp_path / "slow", slow)
    assert slow.skipped > 0 and runner.display_skipped == slow.skipped
    assert runner.drain_display()                                         # still dots, from what it did track
    assert record.summary.n_scored == alone.summary.n_scored              # the neural Take, every frame of it
    assert record.summary.median_abs_cents == pytest.approx(alone.summary.median_abs_cents)


def test_a_failing_display_tracker_ends_the_dots_not_the_take(tracker, tmp_path):
    runner, record = run(tracker, tmp_path, BrokenTracker())
    assert isinstance(runner.display_error, RuntimeError) and runner.display_error.__traceback__ is None
    assert runner.drain_display() == []
    assert record.summary.tendency == "sharp"                             # saved from the neural tracker as always

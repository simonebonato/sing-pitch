"""LiveRunner seam when things go wrong: a crashing worker is reported, a failed Start leaves no thread behind."""
import threading
import time

import numpy as np
import pytest
from audio import SR, quiet
from melody_helpers import reference_line

from sing_pitch.audio_input import ArrayInput
from sing_pitch.live import LiveRunner
from sing_pitch.playback import SongTimeLog
from sing_pitch.reference import Melody, NearestNote


class CrashingTracker:
    def reset(self):
        pass

    def push(self, chunk, sample_rate):
        raise RuntimeError("tracker blew up")

    def flush(self):
        return []


class QuietTracker(CrashingTracker):
    def push(self, chunk, sample_rate):
        return []


class Backing:
    def __init__(self, fail_start=False):
        self.fail_start, self.started, self.stopped = fail_start, False, False
        self.song_log = SongTimeLog(SR)
        self.song_log.append(0, 0.0)

    def start(self):
        if self.fail_start:
            raise OSError("no output device")
        self.started = True

    def position_s(self):
        return 0.0

    def stop(self):
        self.stopped = True


class DeadInput(ArrayInput):
    def start(self, on_chunk):
        raise OSError("no input device")


def tracker_threads():
    return [t for t in threading.enumerate() if t.name == "sing-pitch-tracker" and t.is_alive()]


def test_a_crash_in_the_worker_surfaces_from_drain_and_from_stop(tmp_path):
    runner = LiveRunner(ArrayInput(quiet(0.5), SR, 256), CrashingTracker(), NearestNote(), tmp_path)
    runner.start()
    deadline = time.perf_counter() + 3
    with pytest.raises(RuntimeError, match="tracker blew up"):
        while time.perf_counter() < deadline:
            runner.drain()
            time.sleep(0.01)
    with pytest.raises(RuntimeError, match="tracker blew up"):
        runner.stop()                       # the Take is not saved as if nothing happened
    assert not (tmp_path / "takes.csv").exists()
    assert not tracker_threads()            # and stop() did not hang on a dead worker


@pytest.mark.parametrize("backing", [None, Backing()])
def test_a_failed_start_leaves_no_tracker_thread(tmp_path, backing):
    before = len(tracker_threads())
    runner = LiveRunner(DeadInput(quiet(0.1), SR), QuietTracker(), NearestNote(), tmp_path, backing=backing)
    with pytest.raises(OSError, match="no input device"):
        runner.start()
    assert len(tracker_threads()) == before
    if backing is not None:
        assert backing.stopped


def test_a_backing_that_fails_to_start_leaves_no_thread_and_never_opens_the_input(tmp_path):
    before = len(tracker_threads())
    source = ArrayInput(quiet(0.1), SR)
    started = []
    source.start = lambda on_chunk: started.append(1)
    runner = LiveRunner(source, QuietTracker(), Melody(reference_line([(0, 0.5)]), "x"), tmp_path,
                        backing=Backing(fail_start=True))
    with pytest.raises(OSError, match="no output device"):
        runner.start()
    assert len(tracker_threads()) == before and not started

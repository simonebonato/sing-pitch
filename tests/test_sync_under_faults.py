"""#49 end to end (real window, tracker, Take engine; fake mic and output): the now-line, the drawn Melody and the
scoring stay on the song heard when the audio hardware misbehaves the way it does under a slow, large-window redraw:
output blocks lost (underflow), mic chunks lost (overflow), the two devices' clocks drifting, the UI thread holding the
GIL past the audio deadlines, any window size, any Speed.

The singer is the echo mic: it sings exactly what the Guide vocal plays, as late as the configured latency, so it
follows every silence a lost block puts into the song. Display error = song time under the centre line minus song time
heard at that moment (QA.display_error_s), sampled every 0.2 s of app time. Counting frames (before #49) is 0.3-2.4 s
off under these faults and the oracle falls to 13-78 %; see ~/Builds/.notes/sing-pitch/handoff-sync.md."""
import numpy as np
import pytest
from app_speed import APP_SPEED
from qa_helpers import NOTES, configure_latency, singing_song

from sing_pitch.qa import QA

SCALE = APP_SPEED
MEAN_BOUND_S = 0.015      # display error: mean |error| over the Take
MAX_BOUND_S = 0.030       # ...and no sample beyond this
ORACLE_PCT = 95.0

FAULTS = {
    "underflow": "underflow=0.5:40",            # 43 ms of output lost every 0.5 s
    "overflow": "overflow=0.5:20",              # 21 ms of mic input lost every 0.5 s
    "drift": "drift=3000",                      # the output crystal 0.3 % fast (exaggerated, to add up in 15 s)
    "gil_stall": "deadline=20:60,ui_stall=1:150",   # the UI holds the GIL 150 ms each second: both sides lose blocks
    "ui_sleep": "ui_stall=1:150:sleep",         # the UI is blocked but the audio runs: fewer redraws, no loss
}


def echo_take(home, faults, speed=100, sizes=None, repeats=2):
    """One whole-song Melody Take by the echo singer (the oracle notes `repeats` times over, 3.5 s each); returns
    (display errors, pct in tune, glitches)."""
    song = singing_song(home, notes=NOTES * repeats, tail_s=1.0)
    configure_latency(home, 0.12)
    with QA(home, mic="echo", time_scale=SCALE, faults=faults) as qa:
        qa.set("reference_box", "Melody")
        qa.set("song_box", song.meta.title)
        qa.set("guide_box", True)
        qa.set("guide_slider", 100)
        qa.set("backing_slider", 0)
        if speed != 100:
            qa.set("speed_box", speed)
            qa.wait_until(lambda: qa.window._stems_for_start(song.id, speed) is not None, timeout_s=60)
        qa.start_take()
        errors, k, glitches = [], 0, None
        while qa.running:
            glitches = qa.glitches()          # while the Take runs: its streams are gone after
            if sizes:
                qa.window.resize(*sizes[k % len(sizes)])
                k += 1
            qa.wait(0.2)
            error = qa.display_error_s()
            if error is not None:
                errors.append(error)
        pct = float(qa.takes()[-1]["pct_in_tune"])
    return np.abs(np.array(errors)), pct, glitches


def assert_in_sync(errors, pct):
    assert len(errors) >= 10
    assert errors.mean() < MEAN_BOUND_S and errors.max() < MAX_BOUND_S, (errors.mean(), errors.max())
    assert pct >= ORACLE_PCT


@pytest.mark.parametrize("fault", list(FAULTS))
def test_the_now_line_and_the_score_stay_on_the_song_heard_under_each_fault(tmp_path, fault):
    errors, pct, glitches = echo_take(tmp_path, FAULTS[fault], repeats=4 if fault == "drift" else 2)
    assert_in_sync(errors, pct)


@pytest.mark.parametrize("speed", [50, 70, 120])
def test_lost_output_blocks_do_not_shift_the_song_at_any_speed(tmp_path, speed):
    errors, pct, _ = echo_take(tmp_path, FAULTS["underflow"], speed=speed)
    assert_in_sync(errors, pct)


def test_resizing_the_window_mid_take_from_small_to_huge_keeps_the_sync(tmp_path):
    sizes = [(640, 420), (3000, 1800), (1200, 700), (2000, 1200)]
    errors, pct, _ = echo_take(tmp_path, FAULTS["gil_stall"], sizes=sizes)
    assert_in_sync(errors, pct)


def test_every_lost_block_and_chunk_is_counted(tmp_path):
    _, _, glitches = echo_take(tmp_path, "underflow=1:40,overflow=1.5:20")
    out_n, out_s = glitches["out"]
    in_n, in_s = glitches["inp"]
    step = 256 / 48000                        # the last loss may not be stamped yet when the snapshot is taken
    eps = 1e-9                                # sums of step-sized floats: exactly 4 steps read 4 steps + 1e-16
    assert out_n >= 5 and abs(out_s - glitches["lost_out_steps"] * step) <= 8 * step + eps
    assert in_n >= 3 and abs(in_s - glitches["lost_in_steps"] * step) <= 4 * step + eps

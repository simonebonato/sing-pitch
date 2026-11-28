"""Round 2 of the device-timestamp sync through the real window (fake mic and output, QA driver): #63 (the now-line never
steps back), #62 (no random offset per Take or calibration, and a stale calibration is said), stamp jitter, the Timeline
playhead on the song heard, the audio health record and the dropout note.

`render=40` makes the fake output render whole 1024-frame blocks 40 ms ahead of what plays, like a real device, so the
render frontier at the first mic chunk lies anywhere within a block (#62's phase); without it the fake renders just in
time and the phase is always 0."""
import numpy as np
import pytest
from app_speed import APP_SPEED
from qa_helpers import NOTES, configure_latency, singing_song

from sing_pitch import audio_health
from sing_pitch.qa import QA
from sing_pitch.settings import Settings
from sing_pitch.testmode import FAKE_OUTPUT

SCALE = APP_SPEED
LATENCY = 0.12


def window(home, faults=None, repeats=2, stale_latency=None):
    """stale_latency: the window's latency stored the old way (before #62); the echo mic is still LATENCY late."""
    song = singing_song(home, notes=NOTES * repeats, tail_s=1.0)
    configure_latency(home, LATENCY)
    if stale_latency is not None:
        s = Settings.load(home)
        s.set_latency(FAKE_OUTPUT, stale_latency, on_clocks=False)
        s.save()
    qa = QA(home, mic="echo", latency_s=LATENCY, time_scale=SCALE, faults=faults)
    qa.set("reference_box", "Melody")
    qa.set("song_box", song.meta.title)
    qa.set("guide_box", True)
    qa.set("guide_slider", 100)
    qa.set("backing_slider", 0)
    return qa


def take_errors(qa, seconds=3.0, every=0.2):
    errors = []
    until = qa.app_time() + seconds
    while qa.running and qa.app_time() < until:
        qa.wait(every)
        e = qa.display_error_s()
        if e is not None:
            errors.append(e)
    return np.array(errors)


def test_the_now_line_never_steps_back_when_the_mic_loses_chunks(tmp_path):
    with window(tmp_path, "overflow=0.5:100") as qa:
        qa.start_take()
        heard, drawn = [], []
        until = qa.app_time() + 6.0
        while qa.running and qa.app_time() < until:
            qa._app.processEvents()
            heard.append(qa.window._runner.heard_t)
            drawn.append(qa.window._now_line.value())
        assert qa.hub.lost_in_steps > 50
        errors = take_errors(qa, 1.0)
    dh, dd = np.diff(heard), np.diff(drawn)
    assert np.all(dh >= 0) and np.all(dd >= 0), (np.flatnonzero(dh < 0)[:5], dh[dh < 0][:5], np.flatnonzero(dd < 0)[:5],
                                                   dd[dd < 0][:5], len(dh))
    assert np.abs(errors).max() < 0.002


def test_a_calibration_on_a_device_that_renders_ahead_measures_the_pure_loop_and_takes_need_no_luck(tmp_path):
    """#62 end to end: calibrate on the clocks, then several Takes started at different moments, so at different
    phases of the output's blocks: all in sync."""
    rng = np.random.default_rng(4)
    with window(tmp_path, "render=40,mic_phase=1", repeats=3) as qa:
        qa.click("calibrate_button")
        qa.wait_until(lambda: qa.window._calibration is None, timeout_s=60)
        settings = Settings.load(tmp_path)
        assert settings.latency_for(FAKE_OUTPUT) == pytest.approx(LATENCY, abs=0.003)
        assert settings.is_loop_latency(FAKE_OUTPUT) and not settings.needs_recalibration(FAKE_OUTPUT)
        qa.window._settings = settings
        means = []
        for _ in range(4):
            qa.wait(float(rng.uniform(0.05, 0.3)))     # a different phase of the output's 1024-frame blocks
            qa.start_take()
            errors = take_errors(qa, 2.0)
            qa.stop_take()
            means.append(np.mean(errors))
    assert np.abs(means).max() < 0.002, means


def test_a_latency_stored_before_the_fix_still_works_the_old_way_and_asks_for_a_recalibration(tmp_path):
    rng = np.random.default_rng(5)
    # the old meaning: the loop + the gap from the render frontier, here 40 ms ahead + up to a block - a chunk
    with window(tmp_path, "render=40,mic_phase=1", repeats=3, stale_latency=LATENCY + 0.050) as qa:
        means = []
        for _ in range(4):
            qa.wait(float(rng.uniform(0.05, 0.3)))
            qa.start_take()
            assert "Calibrate latency once more" in qa.summary()
            errors = take_errors(qa, 1.5)
            qa.stop_take()
            means.append(np.mean(errors))
    assert np.abs(means).max() < 0.12                    # roughly in place (this fake's render gap is ~100 ms)...
    assert max(means) - min(means) > 0.004               # ...and its random offset is still there


def test_a_first_mic_chunk_held_up_at_start_does_not_shift_the_take(tmp_path):
    with window(tmp_path, "render=40,mic_late=60") as qa:
        qa.start_take()
        errors = take_errors(qa, 3.0)
    assert np.abs(errors).mean() < 0.002


def test_jittery_stamps_keep_the_sync_and_the_oracle(tmp_path):
    with window(tmp_path, "jitter=400:400") as qa:
        qa.start_take()
        errors = take_errors(qa, 20.0)
        pct = float(qa.takes()[-1]["pct_in_tune"])
        rows = audio_health.read(tmp_path)
    assert np.abs(errors).mean() < 0.003 and np.abs(errors).max() < 0.008
    assert pct >= 95
    row = rows[-1]
    assert int(row["out_anchors"]) < 10 and int(row["in_anchors"]) < 10
    assert 150 < int(row["out_jitter_p50_us"]) < 600         # median |residual| of a 400 us SD


def test_the_timeline_playhead_shows_the_song_heard_and_stands_with_the_now_line_when_paused(tmp_path):
    with window(tmp_path, "render=40") as qa:
        qa.start_take()
        qa.wait(2.0)
        w = qa.window
        for _ in range(5):
            qa.wait(0.2)
            w.tick()
            drawn = float(w._song_times(np.array([w._now_line.value()]))[0])
            assert w.timeline.playhead_s == pytest.approx(drawn, abs=1e-9)
            assert qa.song_position() - w.timeline.playhead_s > LATENCY        # the rendered song runs ahead
            assert abs(qa.display_error_s()) < 0.002
        qa.pause()
        qa.wait(1.0)
        w.tick()
        still = w.timeline.playhead_s
        qa.wait(0.5)
        w.tick()
        assert w.timeline.playhead_s == still
        assert still == pytest.approx(float(w._song_times(np.array([w._now_line.value()]))[0]), abs=1e-9)


def test_a_run_writes_its_audio_health_and_says_how_many_dropouts_it_had(tmp_path):
    with window(tmp_path, "underflow=1:40,overflow=1.5:20") as qa:
        qa.start_take()
        qa.wait(3.0)
        qa.window.resize(900, 600)
        qa.wait(2.0)
        qa.stop_take()
        summary = qa.summary()
    row = audio_health.read(tmp_path)[-1]
    n = int(row["lost_out_blocks"]) + int(row["lost_in_chunks"])
    assert n >= 4 and f"{n} audio dropouts" in summary
    assert row["timed"] == "True" and row["loop_latency"] == "True" and row["output_device"] == FAKE_OUTPUT
    assert float(row["lost_out_s"]) > 0.1 and float(row["lost_in_s"]) > 0.03
    assert int(row["window_size_changes"]) >= 1 and int(row["ticks"]) > 10
    assert row["take_ids"] and row["input_block_max"] == "256"


def test_a_clean_run_has_no_dropout_note(tmp_path):
    with window(tmp_path) as qa:
        qa.start_take()
        qa.wait(2.0)
        qa.stop_take()
        assert "dropout" not in qa.summary()


def test_a_jump_followed_by_mic_losses_files_every_frame_under_the_right_take(tmp_path):
    """#63 end to end: the routing on host time files every frame under its Take with mic chunks lost around a jump.
    A guard, not a proof: in this fake a loss rarely falls inside the latency between the jump and its answer, so the
    old cut passed it too; test_stamps_round2 holds the exact case (the #61 review's) that the old cut fails."""
    from sing_pitch.take_store import saved_song_times

    with window(tmp_path, "overflow=0.15:40", repeats=5) as qa:
        qa.start_take()
        qa.wait(7.0)
        qa.seek(10.0)
        qa.wait(6.5)
        qa.stop_take()
        rows = qa.takes()
    assert len(rows) == 2
    old = [s for s in saved_song_times(tmp_path, rows[0]["id"]) if s is not None]
    new = [s for s in saved_song_times(tmp_path, rows[1]["id"]) if s is not None]
    assert max(old) < 7.5 and min(new) >= 10.0 - 1e-6

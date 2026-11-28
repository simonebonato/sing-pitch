"""The QA driver drives the real window on the fake audio: Start, pause, seek, loop, Stop, read the summary."""
import numpy as np
from app_speed import APP_SPEED
from qa_helpers import configure_latency, singing_song

from sing_pitch.qa import QA

SCALE = APP_SPEED


def echo_singer(home, **kw):
    """The window with the echo mic: with only the Guide vocal playing it sings exactly what plays."""
    qa = QA(home, mic="echo", time_scale=SCALE, **kw)
    qa.set("reference_box", "Melody")
    qa.set("guide_box", True)
    qa.set("guide_slider", 100)
    qa.set("backing_slider", 0)
    return qa


def test_driver_starts_pauses_seeks_loops_stops_and_reads_the_summary(tmp_path):
    singing_song(tmp_path, tail_s=3.0)
    configure_latency(tmp_path, 0.12)
    with echo_singer(tmp_path) as qa:
        assert "pause_button" in qa.names() and "loop_box" in qa.names()
        qa.start_take()
        assert qa.running
        qa.wait(1.0)
        assert abs(qa.song_position() - 1.0) < 0.5               # the song clock runs at app time (loose: a loaded machine reads late, #69)

        qa.pause()
        paused_at = qa.song_position()
        qa.wait(0.5)
        assert qa.song_position() == paused_at                  # paused: the song stands still
        qa.pause()
        qa.wait(0.3)
        assert qa.song_position() > paused_at + 0.2             # resumed

        before = qa.song_position()
        qa.seek(0.4)
        qa.wait(0.1)
        assert before > 1.2 and 0.45 < qa.song_position() < before   # jumped back (wait() lets at least 0.1 s pass; loose above: a loaded machine reads late, #69)

        qa.mark_section(1.2, 3.6)
        qa.set("loop_box", True)
        qa.wait_until(lambda: sum(r["section_name"] != "" for r in qa.takes()) >= 2, timeout_s=120)        # app time: a loaded machine saves a Pass late (#69)
        passes = [r for r in qa.takes() if r["section_name"]]
        assert [r["pass_index"] for r in passes[:2]] == ["1", "2"]
        assert all(float(r["pct_in_tune"]) >= 90 for r in passes[:2])     # the echo singer stays in sync

        qa.stop_take()
        assert not qa.running
        assert "In tune" in qa.summary()


def test_plot_data_and_screenshot(tmp_path):
    singing_song(tmp_path)
    configure_latency(tmp_path, 0.12)
    with echo_singer(tmp_path) as qa:
        qa.start_take()
        qa.wait(2.0)
        plot = qa.plot_data()
        voiced = np.isfinite(plot["voice_midi"])
        assert voiced.sum() > 20 and np.isfinite(plot["reference_midi"]).any()
        lo, hi = plot["x_range"]
        assert lo < plot["now"] < hi
        shot = qa.screenshot(tmp_path / "shot.png")
        assert shot.stat().st_size > 1000


def test_record_output_writes_what_was_played(tmp_path):
    """--record-output: the Guide vocal heard through the null output is the song's vocals.wav, in real time."""
    import soundfile as sf

    song = singing_song(tmp_path)
    configure_latency(tmp_path, 0.12)
    rec = tmp_path / "out.wav"
    with echo_singer(tmp_path, record_output=rec) as qa:
        qa.start_take()
        qa.wait_until(lambda: not qa.running, timeout_s=15)
    played, sr = sf.read(str(rec), dtype="float32")
    vocals, _ = sf.read(str(song.vocals_path), dtype="float32")
    start = np.flatnonzero(np.abs(played[:, 0]) > 1e-3)[0] - np.flatnonzero(np.abs(vocals[:, 0]) > 1e-3)[0]
    n = len(vocals)
    assert np.allclose(played[start : start + n, 0], vocals[:, 0], atol=1e-4)


def test_keys_reach_the_window_shortcuts(tmp_path):
    singing_song(tmp_path)
    with echo_singer(tmp_path) as qa:
        qa.press("+")
        assert qa.widget("speed_box").value() == 105
        qa.press("-")
        assert qa.widget("speed_box").value() == 100
        qa.mark_section(1.0, 2.0)
        qa.press("L")
        assert qa.widget("loop_box").isChecked()

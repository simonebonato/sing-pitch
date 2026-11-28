"""The perfect-singer oracle: the song's own vocals.wav as the fake mic, heard as late as the configured latency,
must score ~100 % in tune on a whole-song Melody Take through the real window, tracker and Take engine."""
from app_speed import APP_SPEED
from qa_helpers import configure_latency, singing_song

from sing_pitch.qa import QA

SCALE = APP_SPEED   # app time runs this much faster than the wall clock (app_speed.py; the tracker keeps up ~12x on a Mac)


def whole_song_take(home, song, mic_latency_s):
    with QA(home, mic=song.vocals_path, latency_s=mic_latency_s, time_scale=SCALE) as qa:
        qa.set("reference_box", "Melody")
        qa.set("song_box", song.meta.title)
        qa.start_take()
        qa.wait_until(lambda: not qa.running, timeout_s=15)     # the window stops itself after the song end
        return qa.takes()[-1], qa.summary()


def test_perfect_singer_scores_at_least_95_percent_in_tune(tmp_path):
    song = singing_song(tmp_path)
    configure_latency(tmp_path, 0.15)
    row, summary = whole_song_take(tmp_path, song, mic_latency_s=0.15)
    assert (row["reference"], row["song_id"]) == ("melody", song.id)
    assert float(row["n_scored_frames"]) * 0.016 > 3.0          # nearly all of the 3.5 s of notes was scored
    assert float(row["pct_in_tune"]) >= 95, summary


def test_a_singer_late_by_more_than_the_configured_latency_is_not_in_tune(tmp_path):
    """The oracle can fail: the same voice 0.45 s later than the configured latency lands on the wrong notes."""
    song = singing_song(tmp_path)
    configure_latency(tmp_path, 0.05)
    row, summary = whole_song_take(tmp_path, song, mic_latency_s=0.5)
    assert float(row["pct_in_tune"]) < 60, summary

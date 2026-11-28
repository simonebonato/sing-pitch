"""QA #45: in Notes mode a note's live verdict colour is final once given and is the verdict the summary uses. The
tracker's frames for the end of a note arrive after the song has passed it (look-ahead plus latency); they used to
re-open the note and close it again on those few tail frames only, overwriting its colour. Seams: NoteTally, and the
window end to end on the QA driver with the echo singer."""
import numpy as np
import soundfile as sf
from audio import SR, harmonic, quiet
from frames import hz_at
from app_speed import APP_SPEED
from qa_helpers import LEAD_S, REF_HOP, configure_latency

from sing_pitch.config import ScoreConfig
from sing_pitch.library import Library
from sing_pitch.library.library import BACKING, REFERENCE, VOCALS, save_reference
from sing_pitch.library.ports import ReferenceLine
from sing_pitch.qa import QA
from sing_pitch.reference import Band, Melody
from sing_pitch.take_engine import NoteTally, note_verdicts, replay_scores
from sing_pitch.take_store import (last_take_id, load_frames, saved_config, saved_octave, saved_reference,
                                   saved_song_times)
from sing_pitch.ui.main_window import slider_of_length

NOTES = [-900, -400, -800, -200, -700, -300]
NOTE_S, TAIL_S = 0.8, 0.2            # each note's last 0.2 s is sung 300 cents sharp: its median stays on the bar


def test_a_closed_note_is_never_closed_again_by_frames_that_arrive_late():
    from melody_helpers import reference_line, voice_following

    notes = Melody(reference_line([(-900, 0.5), (-400, 0.5)]), "two").notes(0.3)
    config = ScoreConfig()
    scores = replay_scores(voice_following([(-900, 0.5), (-400, 0.5)], 0.0), notes, config)
    tally = NoteTally(notes, config)
    tally.add([s for s in scores if s.t < 0.4])
    (first,) = tally.close_until(0.7)
    assert first.note == 0 and first.band is Band.GREEN
    late = [s for s in scores if 0.4 <= s.t < 0.5]
    tally.add([type(s)(**{**s.__dict__, "cents": 300.0}) for s in late])    # note 0's tail, late and off
    assert [v.note for v in tally.close_until(2.0)] == []                  # note 1 was never sung: nothing; note 0: once


def sharp_tail_song(home):
    vocals = [quiet(LEAD_S)]
    t, hz, start = [], [], LEAD_S
    for c in NOTES:
        vocals += [harmonic(hz_at(c), NOTE_S - TAIL_S), harmonic(hz_at(c + 300), TAIL_S)]
        n = round(NOTE_S / REF_HOP)
        t += [start + i * REF_HOP for i in range(n)]
        hz += [hz_at(c)] * n
        start += NOTE_S
    vocals = np.concatenate(vocals + [quiet(1.0)])
    lib = Library(home)
    staging = lib.new_staging()
    sf.write(str(staging / VOCALS), np.stack([vocals, vocals], axis=1), SR)
    sf.write(str(staging / BACKING), np.zeros((len(vocals), 2), np.float32), SR)
    save_reference(staging / REFERENCE, ReferenceLine(np.array(t, np.float32), np.array(hz, np.float32),
                                                      np.full(len(t), 0.9, np.float32)))
    lib.commit(staging, dict(source_kind="file", source_key="sha256:tail", imported_at="2026-11-24T10:00:00",
                             stems_from="separated", separator_model="fake", extractor_model="fake"), "Sharp Tails")


def test_the_live_bar_colours_are_the_summary_verdicts(tmp_path):
    sharp_tail_song(tmp_path)
    configure_latency(tmp_path, 0.12)
    with QA(tmp_path, mic="echo", time_scale=APP_SPEED) as qa:
        w = qa.window
        for name, value in (("reference_box", "Melody"), ("notes_box", True), ("guide_box", True),
                            ("guide_slider", 100), ("backing_slider", 0)):
            qa.set(name, value)
        w.note_length_slider.setValue(slider_of_length(0.3))
        qa.start_take()
        qa.wait_until(lambda: (qa.song_position() or 0) > LEAD_S + NOTE_S * len(NOTES) + 0.5, timeout_s=30)
        live = dict(w._verdicts)
        qa.stop_take()
    home, take = tmp_path, last_take_id(tmp_path)
    reference, config = saved_reference(home, take), saved_config(home, take)
    scores = replay_scores(load_frames(home, take), reference, config, saved_song_times(home, take),
                           saved_octave(home, take))
    final = {v.note: v.band for v in note_verdicts(scores, reference, config)}
    assert len(live) == len(NOTES) and set(final.values()) == {Band.GREEN}
    assert live == final

"""The plot's now-line is the song heard in the headphones now (#22), and Pause freezes the whole plot (#20).

The window runs on its real path (LiveRunner, worker thread, Take engine) with three fakes driven in step by the test:
a mic it pushes chunks into, a player stub whose song-time log it writes, and a tracker that emits each 16 ms frame
only once LOOKAHEAD seconds of audio after it have arrived (as SwiftF0 does, ~0.18 s). At real time R the mic has
delivered R seconds and the output has rendered R + OUT_LAT seconds, so the song heard at R is the one rendered at
output time R. The calibrated latency is all output latency here (the input reports none), so a voice frame at input
time t answers song time heard at R = t. Every expected song time is worked out here from that schedule."""
import time

import numpy as np
from pytest import approx
from audio import SR, harmonic
from melody_helpers import library_song, reference_line
from test_melody_window import FakeBacking

from sing_pitch.pitch import PitchFrame
from sing_pitch.playback import SongTimeLog
from sing_pitch.settings import Settings
from sing_pitch.take_store import _rows
from sing_pitch.ui import MainWindow

SONG_S = 24
NOTES = [(-1200 + 100 * k, 1.0) for k in range(SONG_S)]    # note k sounds from k to k + 1 s and is MIDI 57 + k
OUT_LAT = 0.2            # rendered -> heard; also the calibrated latency (the input part is 0)
LOOKAHEAD = 0.2          # the tracker emits a frame once this much audio after it has arrived
HOP = 0.016
CHUNK = 256
DRAW_STEP = 0.01


def note_midi(song_s):
    return 57.0 + np.floor(song_s)


class LookaheadTracker:
    """A 220 Hz frame every HOP, emitted LOOKAHEAD s late. A zero-length chunk is the test's marker: when the tracker
    sees it, every chunk before it has been scored and queued for the window."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._received, self._next, self.markers = 0.0, 0.0, 0

    def push(self, chunk, sample_rate):
        if len(chunk) == 0:
            self.markers += 1
            return []
        self._received += len(chunk) / sample_rate
        out = []
        while self._next + LOOKAHEAD <= self._received + 1e-9:
            out.append(PitchFrame(self._next, 220.0, 0.99, -20.0))
            self._next = round(self._next + HOP, 9)
        return out

    def flush(self):
        return []


class ManualMic:
    sample_rate, latency_s = SR, 0.0

    def __init__(self):
        self.on_chunk = None

    def start(self, on_chunk):
        self.on_chunk = on_chunk

    def stop(self):
        self.on_chunk = None


class Player(FakeBacking):
    duration_s = float(SONG_S)

    def __init__(self, path, vocals, log):
        super().__init__(path, vocals)
        self.song_log, self.loop, self.playing, self.song_frames, self.rate = log, None, True, 0.0, 1.0

    def position_s(self):
        return self.song_frames / SR

    @property
    def paused(self):
        return not self.playing

    def pause(self):
        self.playing = False

    def play(self):
        self.playing = True

    def seek(self, song_s):
        self.song_frames = song_s * SR


class World:
    """Real time, advanced by the test: each mic chunk is matched by the same number of rendered output frames."""

    def __init__(self, qtbot, tmp_path):
        library_song(tmp_path, reference_line(NOTES), seconds=float(SONG_S))
        settings = Settings.load(tmp_path)
        settings.set_latency("WF-1000XM5", OUT_LAT)
        settings.save()
        self.qtbot, self.home = qtbot, tmp_path
        self.log, self.mic, self.tracker, self.players = SongTimeLog(SR), ManualMic(), LookaheadTracker(), []
        self.voice = harmonic(220.0, SONG_S + 5.0)
        self.received = 0            # mic samples delivered

        players, log, mic = self.players, self.log, self.mic    # not self: World <-> window would be a cycle

        def make(path, vocals=None):
            players.append(Player(path, vocals, log))
            return players[-1]

        self.win = MainWindow(self.tracker, home=tmp_path, input_factory=lambda _i: mic, backing_factory=make,
                              output_info=lambda: ("WF-1000XM5", "headphones"))
        qtbot.addWidget(self.win)
        self.win.reference_box.setCurrentText("Melody")
        self.win.toggle()
        self.win._timer.stop()       # the test ticks
        self._render(round(OUT_LAT * SR))

    @property
    def player(self):
        return self.players[0]

    def _render(self, frames):
        p = self.player
        if p.playing:
            self.log.append(frames, float(p.song_frames), p.rate)
            p.song_frames += frames * p.rate
        else:
            self.log.append(frames, None)

    def advance(self, real_s):
        """Run real time on to real_s (in whole chunks), let the worker score everything, then tick the window."""
        while self.received + CHUNK <= round(real_s * SR):
            self._render(CHUNK)
            self.mic.on_chunk(self.voice[self.received:self.received + CHUNK])
            self.received += CHUNK
        markers = self.tracker.markers
        self.mic.on_chunk(np.zeros(0, np.float32))
        self.qtbot.waitUntil(lambda: self.tracker.markers > markers, timeout=5000)
        self.win.tick()

    @property
    def heard_song(self):
        """The song time in the headphones now: rendered OUT_LAT ago."""
        return self.log.song_time(self.log.rendered_s - OUT_LAT)

    @property
    def now(self):
        return self.win._now_line.value()

    def song_at_now_line(self):
        return self.win._song_times(np.array([self.now]))[0]

    def midi_at_now_line(self):
        x, midi = self.win._ref_curve.getData()
        return midi[int(np.argmin(np.abs(x - self.now)))]

    def view(self):
        """Everything on the plot that could move."""
        w = self.win
        return (tuple(w._plot.getViewBox().viewRange()[0]), self.now, *(np.copy(a) for a in w._curve.getData()),
                *(np.copy(a) for a in w._dots.getData()), *(np.copy(a) for a in w._ref_curve.getData()))


def same_view(a, b):
    return all(np.array_equal(p, q, equal_nan=True) for p, q in zip(a, b))


def test_the_now_line_shows_the_song_heard_now_and_the_voice_ends_left_of_it(qtbot, tmp_path):
    world = World(qtbot, tmp_path)
    world.advance(6.1)
    heard = world.heard_song
    assert abs(heard - world.received / SR) < 1e-9               # the schedule: heard song = real time here
    assert world.song_at_now_line() == approx(heard, abs=DRAW_STEP)
    assert world.midi_at_now_line() == approx(note_midi(heard))  # (6.1 is away from a note change)
    voice_x, _ = world.win._curve.getData()
    assert voice_x.max() <= world.now + 1e-9                      # drawn where it was sung...
    assert voice_x.max() == approx(world.now - LOOKAHEAD, abs=2 * HOP)    # ...so the tracker's delay shows
    world.win.toggle()


def test_pause_freezes_the_whole_plot_while_the_mic_goes_on(qtbot, tmp_path):
    world = World(qtbot, tmp_path)
    world.advance(3.0)
    world.win.toggle_pause()                                     # output paused from 3.2: heard paused from R = 3.2
    world.advance(3.6)
    frozen = world.view()
    assert world.now == approx(3.2, abs=DRAW_STEP)
    for real in (4.0, 4.6, 5.0):                                 # he goes on singing into the mic
        world.advance(real)
        assert same_view(world.view(), frozen), real
    voice_x, _ = world.win._curve.getData()
    assert voice_x.max() <= 3.2 + HOP                            # nothing sung while paused is drawn
    world.win.toggle()


def test_resume_goes_on_exactly_where_pause_stopped_and_the_take_is_not_split(qtbot, tmp_path):
    world = World(qtbot, tmp_path)
    world.advance(3.0)
    world.win.toggle_pause()                                     # heard paused from R = 3.2 (song 3.2)
    world.advance(5.0)
    world.win.toggle_pause()                                     # heard again from R = 5.2, song 3.2 on
    world.advance(6.2)
    heard = world.heard_song
    assert heard == approx(4.2, abs=CHUNK / SR)
    assert world.now == approx(3.2 + (6.2 - 5.2), abs=2 * CHUNK / SR)    # 1 s of playing after the frozen point
    assert world.song_at_now_line() == approx(heard, abs=DRAW_STEP)
    voice_x, _ = world.win._curve.getData()
    assert np.diff(voice_x).max() <= HOP + 1e-6                  # one unbroken voice line across the pause
    _, ref = world.win._ref_curve.getData()
    x = world.win._ref_curve.getData()[0]
    assert np.isfinite(ref[(x > world.now - 4.0 + DRAW_STEP) & (x < world.now + 4.0)]).all()   # no gap in the melody
    world.win.toggle()
    assert len(_rows(tmp_path)) == 1                             # one Take


def test_the_now_line_stays_on_the_song_heard_across_a_seek_and_a_speed_change(qtbot, tmp_path):
    world = World(qtbot, tmp_path)
    world.advance(2.0)
    world.win.seek(12.0)                                         # rendered from output 2.2 on: song 12.0
    world.advance(3.5)
    assert world.heard_song == approx(12.0 + 3.5 - 2.2, abs=CHUNK / SR)
    assert world.song_at_now_line() == approx(world.heard_song, abs=DRAW_STEP)
    world.player.rate = 0.5                                      # Speed 50 % from output 3.7 on
    world.advance(5.0)
    assert world.heard_song == approx(12.0 + 1.5 + 0.5 * (5.0 - 3.7), abs=CHUNK / SR)
    assert world.song_at_now_line() == approx(world.heard_song, abs=DRAW_STEP)
    world.win.toggle()


def test_a_replay_of_a_paused_take_stands_still_through_the_pause_and_goes_on(qtbot, tmp_path):
    world = World(qtbot, tmp_path)
    world.advance(3.0)
    world.win.toggle_pause()                                     # paused at voice time 3.2 ...
    world.advance(5.0)
    world.win.toggle_pause()                                     # ... to 5.2
    world.advance(7.0)
    world.win.toggle()
    win = world.win
    win.toggle_replay()
    win._timer.stop()

    def at(voice_t):                                             # the replay's clock reads voice_t (first frame at 0)
        win._replay_started = time.perf_counter() - voice_t
        win.tick()
        return world.now

    assert at(3.0) == approx(3.0, abs=DRAW_STEP)
    paused = [at(v) for v in (3.6, 4.4, 5.1)]
    assert paused == approx([3.2] * 3, abs=HOP)                  # stands still through the pause
    assert win._curve.getData()[0].max() <= 3.2 + HOP
    assert at(6.2) == approx(4.2, abs=HOP)                       # and goes on from where it stopped
    assert np.diff(win._curve.getData()[0]).max() <= HOP + 1e-6
    win.toggle_replay()

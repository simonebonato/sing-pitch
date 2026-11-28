"""Glue without a UI: audio chunks -> pitch tracker -> Take engine. Qt-free and device-free, so it is testable."""
from __future__ import annotations

import dataclasses
import math
import queue
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio_input import AudioInput
from .config import DEFAULT_SCORE_CONFIG, ScoreConfig
from .pitch import PitchFrame, PitchTracker
from .reference import Reference
from .take_engine import (AUTO, FrameScore, LogClock, OctaveSetting, PassTag, SongClock, TakeEngine, TakeRecord,
                          device_times)
from .take_engine.engine import score_frame
from .take_engine.gate import VoicingGate
from .take_store import save_take

FRAME_S = 0.016              # one tracker frame
MIN_TAKE_SCORED_S = 3.0      # a Take closed by a jump with less scored singing than this is discarded
MIN_PASS_FRACTION = 0.5      # a Pass Take is kept when it scored this share of its Section's singable time...
MIN_PASS_SCORED_S = 0.3      # ...but at least this much, and never more is asked than MIN_TAKE_SCORED_S
DISPLAY_BACKLOG_CHUNKS = 8   # mic chunks waiting (~43 ms at 256 frames, 48 kHz) past which the display tracker skips


def pass_tag(loop, pass_index: int) -> PassTag:
    """The tag of Pass `pass_index` of a playback.transport.Loop; an unnamed mark is named by its span."""
    name = loop.name or f"{loop.start_s:.1f}-{loop.end_s:.1f} s"
    return PassTag(name, loop.start_s, loop.end_s, pass_index)


def _guide_settings(history: list[tuple[float, float]], frames) -> tuple[str, float]:
    """(guide_vocal, guide_gain) of a Take: `history` = [(voice time, Guide vocal gain from then on)], oldest first,
    the first entry at -inf. The Guide vocal is "on" when its gain is above 0; "mixed" when it was on for part of the
    Take; the gain is the highest one heard (0 when off)."""
    if frames:
        first, last = frames[0].t, frames[-1].t
        gains = [next(g for t, g in reversed(history) if t <= first)] + [g for t, g in history if first < t <= last]
    else:
        gains = [history[-1][1]]
    on = [g > 0 for g in gains]
    if not any(on):
        return "off", 0.0
    return ("on" if all(on) else "mixed"), round(max(gains), 3)


def _singable_s(reference: Reference, tag: PassTag) -> float:
    """Song seconds of the Section the Reference can score (a Reference without a line: the whole Section)."""
    if not hasattr(reference, "sample"):
        return tag.end_s - tag.start_s
    midi, _ = reference.sample(np.arange(tag.start_s, tag.end_s, FRAME_S))
    return float(np.isfinite(midi).sum()) * FRAME_S


def _too_short(record: TakeRecord, reference: Reference, tag: PassTag | None) -> bool:
    """Too little scored singing to keep: under MIN_TAKE_SCORED_S, or for a Pass under MIN_PASS_FRACTION of its
    Section's singable time (in voice seconds at its Speed), so Sections shorter than 3 s still save their Passes."""
    need = MIN_TAKE_SCORED_S
    if tag is not None:
        voice_s = _singable_s(reference, tag) / (record.speed_pct / 100)
        need = min(need, max(MIN_PASS_SCORED_S, MIN_PASS_FRACTION * voice_s))
    return record.summary.n_scored * FRAME_S < need


class Playback(Protocol):
    """What the runner needs from the Backing player (playback.BackingPlayer)."""

    def start(self) -> None: ...

    def position_s(self) -> float: ...

    @property
    def song_log(self): ...         # playback.SongTimeLog: output time -> song time

    def stop(self) -> None: ...

    def play(self) -> None: ...

    def pause(self) -> None: ...

    @property
    def paused(self) -> bool: ...     # the one answer to "is the song paused" (the Transport's)

    on_event: object             # settable: called from the audio thread with each TransportEvent

    # Optional, read once at start() (a player without them counts as Guide vocal off, Speed 100):
    #   current_gains: tuple[float, float]   (Backing, Guide vocal) in effect
    #   speed: float                 Speed in effect, 1.0 = original

    def set_gains(self, backing: float, vocal: float) -> None: ...

    def set_speed(self, speed: float, backing=None, vocal=None) -> None: ...


class LiveTake:
    """One Take, fed chunk by chunk. Drive from ONE thread. song_clock: see TakeEngine."""

    def __init__(self, tracker: PitchTracker, reference: Reference, sample_rate: int,
                 song_clock: SongClock | None = None, reset_tracker: bool = True, pass_tag: PassTag | None = None,
                 started_at: datetime | None = None, octave: OctaveSetting = AUTO,
                 config: ScoreConfig = DEFAULT_SCORE_CONFIG):
        """reset_tracker False: a Take that goes on from the previous one's audio (after a jump or a loop wrap), so
        the tracker keeps its state and its frame times. pass_tag, started_at, octave, config: see TakeEngine."""
        self._tracker = tracker
        self._engine = TakeEngine(reference, config, song_clock=song_clock, pass_tag=pass_tag, started_at=started_at,
                                  octave=octave)
        self._sample_rate = sample_rate
        if reset_tracker:
            tracker.reset()

    @property
    def song_clock(self) -> SongClock:
        return self._engine.song_clock

    @song_clock.setter
    def song_clock(self, value: SongClock) -> None:
        self._engine.song_clock = value

    @property
    def pass_tag(self) -> PassTag | None:
        return self._engine.pass_tag

    @pass_tag.setter
    def pass_tag(self, value: PassTag | None) -> None:
        self._engine.pass_tag = value

    @property
    def live_shift(self) -> tuple[int, float]:
        return self._engine.live_shift

    def feed(self, chunk: np.ndarray) -> list[FrameScore]:
        return self._engine.push(self._tracker.push(chunk, self._sample_rate))

    def push_frames(self, frames: list[PitchFrame]) -> list[FrameScore]:
        """Frames the tracker already made (the runner splits them between Takes)."""
        return self._engine.push(frames)

    def close(self) -> TakeRecord:
        """End this Take here, leaving the tracker alone."""
        return self._engine.stop()

    def finish(self) -> TakeRecord:
        self._engine.push(self._tracker.flush())
        return self._engine.stop()


class LiveRunner:
    """Start/Stop for the UI. The audio callback only queues chunks; a worker thread runs tracker + engine;
    the UI calls drain() from its timer. Stop() ends the Take and saves it.

    Melody Takes pass the Backing player and the calibrated loop latency. The Backing starts first; at the first
    mic chunk the runner reads how much output the Backing had rendered when the input clock was at 0, and from then
    on a voice frame at input time t is scored against the song time the Backing's song-time log shows at output
    time t - latency_s (a LogClock), so pauses and seeks (later loops, Speed) keep the sync. When both the Backing
    (output_times) and the source (times) carry device timestamps on one clock, the LogClock meets voice and output
    on that host clock (#49): a lost output block or mic chunk, or drift between the devices, no longer shifts them.

    A jump (the Backing's "jumped" event) closes the current Take and opens the next one: voice frames before the
    jump's input time (its output time on the voice clock, LogClock.to_voice) belong to the old Take, the rest to the
    new one. A Take closed that way with less than MIN_TAKE_SCORED_S of scored singing is discarded (a Pass Take: less than
    MIN_PASS_FRACTION of its Section, see _too_short); the others are saved at once
    and listed by drain_saved(). On stop() the last Take is saved too, unless it is that short and an earlier Take of
    the run was saved (a run always leaves at least one Take), and never when nothing at all was scored: `dropped`
    then holds the Take that was not saved (None when the last Take was saved), and stop() returns the Take that
    stands for the run: the last saved one, or the unsaved empty record when the run saved nothing.

    A Section loop works the same way: each "pass_started" event closes the Take there and opens the next one, tagged
    with the Section and the pass index (PassTag), so every Pass is its own Take and its Lead-in is never scored.
    "loop_ended" closes the last Pass Take and opens an untagged one. A Pass that starts before the song clock exists
    (the loop was on at Start) tags the first Take.

    Each saved Take records the practice settings it was sung with. Speed: a "speed_changed" outside a loop is a
    boundary too (a Take is never half at one Speed); inside a loop the Transport applies it at a Pass boundary, which
    is a boundary already. Guide vocal: the runner keeps the history of the Guide vocal gain ("gains_changed" events)
    and a Take that saw it both on and off is "mixed"."""

    def __init__(self, source: AudioInput, tracker: PitchTracker, reference: Reference, home: Path,
                 backing: Playback | None = None, latency_s: float = 0.0, octave: OctaveSetting = AUTO,
                 config: ScoreConfig = DEFAULT_SCORE_CONFIG, loop_latency: bool = False,
                 display_tracker: PitchTracker | None = None):
        """loop_latency: latency_s was calibrated on the device timestamps (the DAC-to-ADC loop, see LogClock).
        display_tracker: a faster tracker whose frames are the live dots (drain_display), scored at the neural Take's
        clock, Section and live Octave and timing shift; the saved Takes are still made from `tracker` alone."""
        self._source = source
        self._display_tracker = display_tracker
        self._display: queue.SimpleQueue = queue.SimpleQueue()
        self._display_gate = VoicingGate(config)
        self._display_in: queue.SimpleQueue = queue.SimpleQueue()      # chunks for the display thread
        self._display_thread: threading.Thread | None = None
        self.display_skipped = 0                  # mic samples the display tracker let go by (it was behind)
        self.display_error: Exception | None = None   # what ended the display thread's dots, if anything
        self._loop_latency = loop_latency
        self._config = config                     # every Take of the run is scored with it (Tolerance, Timing window)
        self._tracker = tracker
        self._reference = reference
        self._home = home
        self._backing = backing
        self._latency_s = latency_s
        self._octave = octave                     # every Take of the run is scored with it
        self._chunks: queue.SimpleQueue = queue.SimpleQueue()
        self._scores: queue.SimpleQueue = queue.SimpleQueue()
        self._take: LiveTake | None = None
        self._cuts: list[tuple[float, PassTag | None, int | None]] = []   # (host time, tag of the next Take, its
                                                              # Speed percent or None = as before) of boundaries not
                                                              # yet reached by the frames (worker only)
        self._speed_pct = 100                     # Speed of the running Take (worker only)
        self._speed_next: int | None = None       # a Speed applied at a Pass boundary, for the cut that follows
        self._guide: list[tuple[float, float]] = [(-math.inf, 0.0)]    # (voice time, Guide vocal gain) (worker only)
        self._saved: list[TakeRecord] = []        # Takes saved so far in this run (worker, then stop())
        self.dropped: TakeRecord | None = None    # the last Take of the run when stop() did not save it (see stop())
        self._saved_news: queue.SimpleQueue = queue.SimpleQueue()
        self._cut_news: queue.SimpleQueue = queue.SimpleQueue()    # voice time of each Take boundary passed
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None      # what killed the worker, if anything
        self._run_started = datetime.now()        # wall clock at input time 0: a cut's Take starts at this + its time
        self._received = 0                        # mic samples delivered so far (audio thread writes, UI reads)
        self._heard = -math.inf                   # the largest heard_t given out

    def start(self) -> None:
        """Open the Backing and the input, then start the worker: a Start that fails leaves no thread behind."""
        self._error = None
        self._cuts, self._saved = [], []
        self.dropped = None
        self._speed_pct = round(100 * getattr(self._backing, "speed", 1.0))
        self._speed_next = None
        self._guide = [(-math.inf, getattr(self._backing, "current_gains", (1.0, 0.0))[1])]
        self._run_started = datetime.now()
        self._received = 0
        self._heard = -math.inf
        self._take = LiveTake(self._tracker, self._reference, self._source.sample_rate, started_at=self._run_started,
                              octave=self._octave, config=self._config)
        if self._display_tracker is not None:
            self._display_tracker.reset()         # its frame times count from the same first sample
            self._display_gate = VoicingGate(self._config)
            self.display_skipped, self.display_error = 0, None
            self._display_in = queue.SimpleQueue()
        if self._backing is None:
            def on_chunk(chunk: np.ndarray) -> None:      # PortAudio thread
                self._received += len(chunk)
                self._chunks.put(chunk)
        else:
            self._backing.on_event = self._on_event
            try:
                self._backing.start()
            except Exception:
                self._quietly(self._backing.stop)
                raise
            clock: LogClock | None = None

            def on_chunk(chunk: np.ndarray) -> None:      # PortAudio thread: a few reads and assignments
                nonlocal clock
                self._received += len(chunk)
                log, end = self._backing.song_log, self._received / self._source.sample_rate
                if clock is None:
                    devices = device_times(getattr(self._backing, "output_times", None),
                                           getattr(self._source, "times", None))
                    clock = LogClock(log, self._latency_s, log.rendered_s - end, devices, aligned_at_s=end,
                                     loop_latency=self._loop_latency)
                    self._chunks.put(("align", clock))
                elif not clock.aligned and log.rendered_s > 0:     # nothing was played at the first chunk
                    clock.realign(log.rendered_s - end, end)
                self._chunks.put(chunk)

        try:
            self._source.start(on_chunk)
        except Exception:
            self._quietly(self._source.stop)
            if self._backing is not None:
                self._quietly(self._backing.stop)
            raise
        self._thread = threading.Thread(target=self._work, name="sing-pitch-tracker", daemon=True)
        self._thread.start()          # chunks that arrived meanwhile wait in the queue
        if self._display_tracker is not None:
            self._display_thread = threading.Thread(target=self._work_display, name="sing-pitch-display", daemon=True)
            self._display_thread.start()

    def _on_event(self, event) -> None:       # audio thread: only hand over, in order with the chunks
        if event.kind in ("jumped", "loop_ended"):
            self._chunks.put(("jump", (event.output_s, None, None)))
        elif event.kind == "pass_started":
            self._chunks.put(("jump", (event.output_s, pass_tag(event.loop, event.pass_index), None)))
        elif event.kind == "speed_changed":
            pct = round(100 * event.speed)
            if event.loop is None:
                self._chunks.put(("jump", (event.output_s, None, pct)))
            else:                         # applied at a Pass boundary: the pass_started that follows is the cut
                self._chunks.put(("speed", pct))
        elif event.kind == "gains_changed":
            self._chunks.put(("gains", (event.output_s, event.gains[1])))

    @staticmethod
    def _quietly(cleanup) -> None:
        try:
            cleanup()
        except Exception:
            pass

    def _work(self) -> None:
        while (chunk := self._chunks.get()) is not None:
            if self._error is not None:       # crashed earlier: keep emptying the queue until stop() ends us
                continue
            try:
                if isinstance(chunk, tuple):
                    self._control(*chunk)
                else:
                    if self._display_tracker is not None:
                        self._display_in.put(chunk)       # tracked meanwhile, on the display thread
                    self._scores.put(self._route(self._tracker.push(chunk, self._source.sample_rate)))
            except Exception as exc:
                self._error = exc

    def _control(self, kind: str, value) -> None:
        if kind == "align":             # ("align", clock): set before the first frame is scored
            self._take.song_clock = value
        elif kind == "speed":           # ("speed", percent): applies to the next cut
            self._speed_next = value
        elif kind == "gains":           # ("gains", (output_s, Guide vocal gain)): from that voice time on
            output_s, gain = value
            if isinstance(self._take.song_clock, LogClock):
                self._guide.append((self._take.song_clock.to_voice(output_s), gain))
            else:                       # before the clock exists: it is how the first Take starts
                self._guide = [(-math.inf, gain)]
        elif kind == "jump":            # ("jump", (output_s, tag, speed)): a boundary before the clock exists has no
            output_s, tag, pct = value  # Take to close, so the first Take simply takes its tag and Speed
            pct, self._speed_next = pct if pct is not None else self._speed_next, None
            if isinstance(self._take.song_clock, LogClock):
                self._cuts.append((self._take.song_clock.answer_host(output_s), tag, pct))
            else:
                self._take.pass_tag = tag
                self._speed_pct = pct if pct is not None else self._speed_pct

    def _route(self, frames: list[PitchFrame]) -> list[FrameScore]:
        """Tracker frames to the current Take; at each pending jump the Take is closed and the next one opened."""
        scores: list[FrameScore] = []
        clock = self._take.song_clock
        while self._cuts and frames:                # compared on capture host time: a mic loss after the cut was
            cut = self._cuts[0][0]                  # computed cannot move it (#63); counting, host time = voice time
            first_after = next((i for i, f in enumerate(frames) if clock.voice_host(f.t) >= cut), None)
            if first_after is None:
                break
            scores += self._take.push_frames(frames[:first_after])
            frames = frames[first_after:]
            host, tag, pct = self._cuts.pop(0)
            self._next_take(tag, pct, clock.voice_at_host(host), clock.plot_of_host(host))
        return scores + self._take.push_frames(frames)

    def _work_display(self) -> None:
        """The display tracker's own thread, beside the worker: the neural tracker's inference runs without the GIL
        (onnxruntime), so the two overlap instead of adding up. Measured 2026-11-27 on a Windows laptop in test mode
        with Fast, both on the worker: neural push 38 % of real time, its scoring 21 %, Fast 30 % (8 % computing, 22 %
        waiting for the GIL), ~90 % in all; any hiccup and the queue grew, the dots 4.1 s late after 30 s.

        A chunk is skipped (the tracker lets it go by) while more than DISPLAY_BACKLOG_CHUNKS wait: dots that late are
        no use, and catching up must not cost the next ones. A failure ends the dots (display_error), never the Take:
        the saved Take comes from the neural tracker alone."""
        while (chunk := self._display_in.get()) is not None:
            if self.display_error is not None:
                continue
            try:
                if self._display_in.qsize() > DISPLAY_BACKLOG_CHUNKS and hasattr(self._display_tracker, "skip"):
                    self._display_tracker.skip(len(chunk))
                    self.display_skipped += len(chunk)
                else:
                    frames = self._display_tracker.push(chunk, self._source.sample_rate)
                    self._display.put(self._score_display(frames))
            except Exception as exc:
                self.display_error = exc.with_traceback(None)   # no frames of this runner kept: no cycle (_pop_error)

    def _score_display(self, frames: list[PitchFrame]) -> list[FrameScore]:
        """The display tracker's frames as dots: on the running Take's song clock and Section, at the Octave and
        timing shift its newest neural frame was scored at, with their own voicing gate. Never saved."""
        take, out = self._take, []
        octave, lag = take.live_shift
        for f in frames:
            song_t = take.song_clock(f.t)
            if take.pass_tag is not None and not take.pass_tag.scores(song_t):
                song_t = None
            out.append(score_frame(f, self._display_gate.step(f), self._reference, self._config, song_t, octave, lag))
        return out

    def _with_settings(self, record: TakeRecord) -> TakeRecord:
        guide, gain = _guide_settings(self._guide, record.frames)
        return dataclasses.replace(record, guide_vocal=guide, guide_gain=gain, speed_pct=self._speed_pct)

    def _next_take(self, tag: PassTag | None = None, speed_pct: int | None = None, at: float = 0.0,
                   plot_at: float | None = None) -> None:
        """Close the running Take and open the next one at voice time `at` (plot_at on the plot's axis, for
        drain_cuts): it started at the run start + `at` (not when the worker got here), so Takes cut a few seconds apart
        get their own started_at and id."""
        old, clock = self._take, self._take.song_clock
        record = self._with_settings(old.close())
        if speed_pct is not None:
            self._speed_pct = speed_pct
        if not _too_short(record, self._reference, old.pass_tag):
            saved = save_take(self._home, record)
            self._saved.append(saved)
            self._saved_news.put(saved)
        self._take = LiveTake(self._tracker, self._reference, self._source.sample_rate, song_clock=clock,
                              reset_tracker=False, pass_tag=tag, octave=self._octave, config=self._config,
                              started_at=self._run_started + timedelta(seconds=max(0.0, at)))
        self._cut_news.put(at if plot_at is None else plot_at)   # before any score of the new Take is handed over

    def drain_saved(self) -> list[TakeRecord]:
        """Takes saved at a jump or a loop wrap since the last call (the last Take of the run comes from stop())."""
        out: list[TakeRecord] = []
        while True:
            try:
                out.append(self._saved_news.get_nowait())
            except queue.Empty:
                return out

    def drain_cuts(self) -> list[float]:
        """Times on the plot's axis (see drain) of the Take boundaries (jumps, loop wraps, Speed changes) the worker has passed since the last
        call, oldest first. A boundary is listed before any score after it reaches drain(): call this after drain()
        and every score of the new Take drained so far has its boundary here."""
        out: list[float] = []
        while True:
            try:
                out.append(self._cut_news.get_nowait())
            except queue.Empty:
                return out

    @property
    def song_clock(self) -> SongClock | None:
        """Voice time -> song time of the running Take (the same clock across jumps)."""
        return None if self._take is None else self._take.song_clock

    @property
    def latency_s(self) -> float | None:
        """The song clock's offset at the Take start (voice time - song time while the song plays linearly), for
        drawing the reference line on the voice clock."""
        return None if self._take is None else self._take.song_clock.offset_s

    @property
    def heard_t(self) -> float:
        """The time of what is in the headphones now, on the plot's axis (see drain). With device timestamps (a timed
        LogClock): the host clock now, so it moves smoothly between mic chunks, does not depend on when a callback or
        the worker last ran (#49) and never goes back when the mic loses a chunk (#63). Without: the newest mic sample's time plus the
        input's own latency. The calibrated latency is the output part plus the input part, so in song terms this is
        the rendered output minus the output part of the latency. Every tracker frame is older than this by the
        tracker's look-ahead (SwiftF0: ~0.18 s) and the queueing, so the plot's now-line is drawn here, not at the
        newest frame."""
        clock = self.song_clock
        now = clock.plot_now() if isinstance(clock, LogClock) else None
        if now is None:
            now = self._received / self._source.sample_rate + getattr(self._source, "latency_s", 0.0)
        self._heard = max(self._heard, now)       # never back, also when the clock switches from counting to stamps
        return self._heard

    @property
    def draw_clock(self) -> SongClock | None:
        """The song clock on the plot's axis (LogClock.drawn): what the window draws the Reference with."""
        clock = self.song_clock
        return clock.drawn() if isinstance(clock, LogClock) else clock

    # --- for the audio health record (audio_health.py) ------------------------------------------------------------
    @property
    def saved(self) -> list[TakeRecord]:
        """Takes saved so far in this run."""
        return list(self._saved)

    @property
    def loop_latency(self) -> bool:
        return self._loop_latency

    @property
    def calibrated_latency_s(self) -> float:
        return self._latency_s

    @property
    def output_times(self):
        return getattr(self._backing, "output_times", None)

    @property
    def input_times(self):
        return getattr(self._source, "times", None)

    @property
    def output_latency_s(self) -> float | None:
        return getattr(self._backing, "reported_latency_s", None)

    @property
    def input_latency_s(self) -> float | None:
        value = getattr(self._source, "latency_s", None)
        return None if value is None else float(value)

    @property
    def glitches(self) -> tuple[int, int]:
        """(lost output blocks, lost mic chunks) the device timestamps showed so far in this run (0, 0 untimed)."""
        out = getattr(self._backing, "output_times", None)
        inp = getattr(self._source, "times", None)
        return (out.glitches if out is not None else 0, inp.glitches if inp is not None else 0)

    def drain_display(self) -> list[FrameScore] | None:
        """The display tracker's dots since the last call, on the plot's axis like drain(); None without one."""
        if self._display_tracker is None:
            return None
        out: list[FrameScore] = []
        while True:
            try:
                out += self._display.get_nowait()
            except queue.Empty:
                break
        return self._on_plot_axis(out)

    def drain(self) -> list[FrameScore]:
        """Scores made since the last call, their times on the plot's axis: capture host time minus a fixed origin
        (LogClock.plot_t), which is voice time while nothing is lost; after a lost mic chunk the voice that follows is
        placed where it was sung, leaving a gap, so the plot never steps back (#63). Saved Takes keep voice time."""
        out: list[FrameScore] = []
        while True:
            try:
                out += self._scores.get_nowait()
            except queue.Empty:
                break
        if not out and self._error is not None:       # the scores made before the crash are shown first
            raise self._error
        return self._on_plot_axis(out)

    def _on_plot_axis(self, out: list[FrameScore]) -> list[FrameScore]:
        clock = self.song_clock
        if out and isinstance(clock, LogClock) and clock.timed:
            plot = clock.plot_t(np.array([s.t for s in out]))
            out = [dataclasses.replace(s, t=float(u)) for s, u in zip(out, plot)]
        return out

    def _pop_error(self) -> Exception:
        """The worker's exception, no longer kept here and in no local of the raising frame: its traceback holds
        frames whose `self` is this runner, so keeping it is a cycle only gc would free, on any thread (ui/gui_gc.py)."""
        error, self._error = self._error, None
        return error

    def stop(self) -> TakeRecord:
        """End the Take. The Backing is stopped whatever the microphone does: an input that fails to stop must never
        leave the song playing into the headphones. Its failure is not worth the Take: the audio is already here."""
        try:
            self._source.stop()
        except Exception:
            pass
        finally:
            try:
                if self._backing is not None:
                    self._backing.stop()
            finally:
                self._chunks.put(None)
                self._thread.join()
                if self._display_thread is not None:      # after the worker: it hands over no more chunks
                    self._display_in.put(None)
                    self._display_thread.join()
                    self._display_thread = None
        if self._error is not None:
            raise self._pop_error()
        record = self._with_settings(self._take.finish())
        self.dropped = None
        if _too_short(record, self._reference, self._take.pass_tag):
            if self._saved:                       # an earlier Take of the run is the result; the last bit is dropped
                self.dropped = record
                return self._saved[-1]
            if record.summary.n_scored == 0:      # nothing scored and nothing before it: no Take at all (#54). Not
                self.dropped = record             # saved, so it can never be "Replay last Take"; returned unsaved
                return record
        return save_take(self._home, record)

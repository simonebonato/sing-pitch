"""Song clocks: which song time a voice frame is scored against. A voice frame at input time t answers what the
singer heard one calibrated latency earlier; the clock turns t into that song time (None = no song was playing)."""
from __future__ import annotations

import bisect
import threading
import time
from typing import Protocol, Sequence

import numpy as np

from ..pitch import PitchFrame


class SongLog(Protocol):
    """What LogClock reads (playback.transport.SongTimeLog)."""

    def song_time(self, output_s: float) -> float | None: ...

    def tail(self) -> tuple[float, float | None, float] | None: ...

    def paused_spans(self) -> list[tuple[float, float]]: ...


class SongClock(Protocol):
    offset_s: float      # t - song time at the Take start: drawn on the plot and kept in takes.csv (latency_offset_s)

    def __call__(self, t: float) -> float | None: ...


class LinearClock:
    """Song time = t - offset_s. Nearest note and Key (offset 0), and Melody Takes saved before the song-time log
    (their row's latency_offset_s), which were played linearly from start to end."""

    def __init__(self, offset_s: float = 0.0):
        self.offset_s = offset_s

    def __call__(self, t: float) -> float | None:
        return t - self.offset_s


class HostClock:
    """The clock PortAudio stamps every callback with (time_info) and reads in Pa_GetStreamTime. Input and output
    streams of one process share it, so a mic sample and a frame in the headphones can be compared on it.
    time.perf_counter() is that clock on both systems: on macOS it is mach_absolute_time, the same clock as
    time.monotonic() (measured 2026-11-25 on CoreAudio: Pa_GetStreamTime - time.monotonic() = -0.4 us); on Windows it
    is QueryPerformanceCounter (measured 2026-11-26 on WASAPI: Pa_GetStreamTime - perf_counter = -0.7 us), while
    time.monotonic() there is GetTickCount64, which ticks every 15.6 ms. tests/test_host_clock.py checks both."""

    @staticmethod
    def now() -> float:
        return time.perf_counter()


PORTAUDIO = HostClock()


class StreamTimeline:
    """When each frame of one audio stream met the hardware, on a host clock: the DAC time an output frame left the
    device, or the ADC time a mic sample entered it, as the device reports it (PortAudio time_info).

    A stream's frame count alone is a clock that slips (#49): a block the callback rendered too late is a silence the
    count never sees (output underflow: everything after is heard later than counted), a chunk lost before its callback
    ran is a gap the count closes (input overflow: everything after was sung later than counted), and two devices'
    crystals drift apart. The timeline keeps anchors (stream seconds by frame count, host seconds); from an anchor on,
    the stream runs one second per second. add() is called once per block with that block's own timestamp and decides
    against the model (the last anchor) what the stamp says. No anchors: host time = stream time (frame counting).

    How a stamp is read (#63 review: Bluetooth and the real mic were never measured, so jitter must not hurt):
    - The residual (stamp minus model) of every block feeds a jitter estimate: sigma = 1.4826 x the median of the
      block-to-block residual differences over the last JITTER_WINDOW blocks / sqrt 2 (differences, so drift and steps
      do not count as jitter). The tolerance is max(TOLERANCE_S, JITTER_K x sigma).
    - Within the tolerance: nothing changes (jitter, or drift too small to matter yet).
    - Later than the model by more: a tentative anchor at once (a lost block must show before the next one), which the
      next block confirms (it agrees with the new anchor within the tolerance and not with the old model) or retracts
      (the stamp was a spike, counted in `spikes`). A confirmed jump of more than max(GLITCH_S, 2 x tolerance)
      is lost audio: `glitches`, its length in `lost_s`; smaller ones are drift (a drift step is about one tolerance).
    - Earlier than the model by more: never lost audio. It must repeat in the next block (a lasting step: the reported
      latency changed, e.g. a Bluetooth renegotiation, or drift) before an anchor is made; one alone is a spike. A
      confirmed backward step beyond max(GLITCH_S, 2 x tolerance) is counted in `steps_back` / `steps_back_s`. The
      anchor's host time is kept after the previous anchor's (a step back larger than the time since it would otherwise fold
      host time back on itself): the frames reported as played twice are mapped to the newer ones.
    - An anchor stands on one stamp, so with jitter it may sit off by up to the tolerance: when a whole window of
      residuals against it has a median off by more than BIAS_S, it is re-anchored on that median (`corrections`;
      this also follows slow drift, about one anchor per window when the clocks drift 50 ppm or more).
    So jitter alone adds next to no anchors, a spike at most one tentative one that is taken back, and anchors grow only
    with real lasting steps and drift (at most one per window). Lookups search preallocated arrays (no copy per call), so a paint costs the same after an hour.

    One audio thread adds, any thread reads: an entry is written before the count that publishes it, and readers read
    the count first."""

    TOLERANCE_S = 0.0005       # the tolerance never goes below this
    JITTER_K = 5.0             # ...and is this many sigmas of the measured jitter above it
    JITTER_WINDOW = 256        # blocks the jitter is measured over
    GLITCH_S = 0.002           # a confirmed jump beyond max(this, 2 x tolerance) is lost audio (or a step back)
    BIAS_S = 0.00025           # a full window whose median residual is off by more than this (or 5 standard errors of
                               # that median, if more) re-anchors on the median
    HIST_STEP_S = 20e-6        # |residual| histogram for the health record: 20 us bins up to 100 ms
    HIST_BINS = 5000

    def __init__(self, clock: object = PORTAUDIO):
        self.clock = clock          # what the host times are measured on (has now()); timelines on different clocks
        self._s = np.empty(64)      # cannot be compared
        self._h = np.empty(64)
        self._n = 0
        self._tentative: float | None = None     # the jump of the last anchor while it waits for its confirmation
        self._pending: tuple[float, float, float] | None = None    # a backward step waiting for its repeat
        self._res = np.zeros(self.JITTER_WINDOW)
        self._res_n = 0
        self._since = 0             # blocks since the last anchor
        self.corrections = 0        # re-anchorings on a window's median (bias of a noisy anchor, slow drift)
        self._sigma = 0.0
        self.hist = np.zeros(self.HIST_BINS + 1, np.int64)
        self.glitches = 0
        self.lost_s = 0.0
        self.steps_back = 0
        self.steps_back_s = 0.0
        self.spikes = 0
        self.flagged = 0            # blocks the driver itself flagged (PortAudio status: underflow / overflow)
        self.blocks = 0
        self.frames_min: int | None = None
        self.frames_max: int | None = None

    def __len__(self) -> int:
        return self._n

    # --- the writer ------------------------------------------------------------------------------------------------
    @property
    def tolerance_s(self) -> float:
        return max(self.TOLERANCE_S, self.JITTER_K * self._sigma)

    @property
    def jitter_s(self) -> float:
        """The stamps' jitter (sigma) measured over the last JITTER_WINDOW blocks."""
        return self._sigma

    def _append(self, stream_s: float, host_s: float) -> None:
        n = self._n
        if n == len(self._s):
            s, h = np.empty(2 * n), np.empty(2 * n)
            s[:n], h[:n] = self._s, self._h
            self._h, self._s = h, s          # a reader holding the old arrays still has its n entries in them
        if n and host_s <= self._h[n - 1]:
            host_s = self._h[n - 1] + 1e-6
        self._s[n], self._h[n] = stream_s, host_s
        self._n = n + 1
        self._since = 0

    def _residual(self, r: float) -> None:
        self._res[self._res_n % self.JITTER_WINDOW] = r
        self._res_n += 1
        a = abs(r)
        self.hist[min(int(a / self.HIST_STEP_S), self.HIST_BINS)] += 1
        if self._res_n % 8 == 0 and self._res_n > 2:
            k = min(self._res_n, self.JITTER_WINDOW)          # in time order, the newest last
            r = np.roll(self._res, -(self._res_n % self.JITTER_WINDOW))[-k:]
            # block-to-block differences: drift (a slow ramp) and steps (one outlier) drop out, jitter stays
            self._sigma = 1.4826 * float(np.median(np.abs(np.diff(r)))) / np.sqrt(2)

    def add(self, stream_s: float, host_s: float, frames: int | None = None) -> None:
        """The block starting at stream time stream_s met the hardware at host time host_s (see the class doc)."""
        stream_s, host_s = float(stream_s), float(host_s)
        self.blocks += 1
        if frames is not None:
            self.frames_min = frames if self.frames_min is None else min(self.frames_min, frames)
            self.frames_max = frames if self.frames_max is None else max(self.frames_max, frames)
        n = self._n
        if n == 0:
            self._append(stream_s, host_s)
            return
        tol = self.tolerance_s
        jump = float(host_s - (self._h[n - 1] + stream_s - self._s[n - 1]))
        if self._tentative is not None:          # this block decides the tentative anchor before it
            step, self._tentative = self._tentative, None
            if abs(jump) <= tol < abs(jump + step):   # confirmed: it fits the new anchor and not the old model
                self._residual(jump)
                if step > max(self.GLITCH_S, 2 * tol):
                    self.glitches += 1
                    self.lost_s += step
                return
            self._n = n = n - 1                   # retracted: it was a spike; judge this block by the old model
            self.spikes += 1
            jump = float(host_s - (self._h[n - 1] + stream_s - self._s[n - 1]))
        self._residual(jump)
        if abs(jump) <= tol:
            if self._pending is not None:         # a step back that did not last
                self.spikes += 1
                self._pending = None
            self._since += 1
            if self._since >= self.JITTER_WINDOW:     # every residual of the window is against this anchor
                med = float(np.median(self._res))
                if abs(med) > max(self.BIAS_S, 5 * 1.2533 * self._sigma / np.sqrt(self.JITTER_WINDOW)):
                    self._append(stream_s, host_s - jump + med)
                    self.corrections += 1
            return
        if jump > 0:                              # later than the model: tentative at once
            self._pending = None
            self._append(stream_s, host_s)
            self._tentative = jump
            return
        pending, self._pending = self._pending, None
        if pending is not None and abs(jump - pending[2]) <= tol:    # a step back that lasts
            self._append(pending[0], pending[1])
            if -pending[2] > max(self.GLITCH_S, 2 * tol):
                self.steps_back += 1
                self.steps_back_s += -pending[2]
            return
        if pending is not None:
            self.spikes += 1
        self._pending = (stream_s, host_s, jump)

    # --- readers -----------------------------------------------------------------------------------------------------
    def _arrays(self):
        n = self._n                               # the count first: the arrays hold at least n entries
        return n, self._s, self._h

    def host(self, stream_s):
        """Host time of stream time stream_s (scalar or array): from the last anchor at or before it."""
        n, s, h = self._arrays()
        if n == 0:
            return stream_s
        if np.ndim(stream_s):
            x = np.asarray(stream_s, dtype=np.float64)
            i = np.maximum(np.searchsorted(s[:n], x, side="right") - 1, 0)
            return h[i] + (x - s[i])
        i = max(0, int(np.searchsorted(s[:n], stream_s, side="right")) - 1)
        return float(h[i] + (stream_s - s[i]))

    def stream(self, host_s):
        """Stream time at host time host_s (scalar or array). Between a lost block's end and the next anchor nothing
        met the hardware: the stream stands at the next anchor's frame (an output plays silence, an input hears
        nothing) until then."""
        n, s, h = self._arrays()
        if n == 0:
            return host_s
        if np.ndim(host_s):
            x = np.asarray(host_s, dtype=np.float64)
            i = np.maximum(np.searchsorted(h[:n], x, side="right") - 1, 0)
            nxt = np.append(s[1:n], np.inf)
            return np.minimum(s[i] + (x - h[i]), nxt[i])
        i = max(0, int(np.searchsorted(h[:n], host_s, side="right")) - 1)
        v = float(s[i] + (host_s - h[i]))
        return min(v, float(s[i + 1])) if i + 1 < n else v

    def residual_percentile_s(self, q: float) -> float | None:
        """The q-th percentile of |stamp - model| over every block so far (20 us resolution); None without blocks."""
        total = int(self.hist.sum())
        if total == 0:
            return None
        k = int(np.searchsorted(np.cumsum(self.hist), q / 100 * total))
        return (k + 0.5) * self.HIST_STEP_S

    def health(self) -> dict:
        """What the stamps showed, for the audio health record (audio_health.py)."""
        p50, p99 = self.residual_percentile_s(50), self.residual_percentile_s(99)
        return dict(blocks=self.blocks, anchors=self._n, glitches=self.glitches, lost_s=round(self.lost_s, 4),
                    steps_back=self.steps_back, steps_back_s=round(self.steps_back_s, 4), spikes=self.spikes,
                    flagged=self.flagged, jitter_p50_us=None if p50 is None else round(1e6 * p50),
                    jitter_p99_us=None if p99 is None else round(1e6 * p99), corrections=self.corrections,
                    frames_min=self.frames_min,
                    frames_max=self.frames_max)


_COUNTING = StreamTimeline(clock=None)      # no anchors: host time = frame count


def device_times(output: StreamTimeline | None, input: StreamTimeline | None):
    """(output, input) timelines when they can be used together (both exist, on one clock), else None: then the song
    clock counts frames as before #49. Either may still be waiting for its first timestamp (see LogClock)."""
    if output is None or input is None or output.clock is None or output.clock is not input.clock:
        return None
    return output, input


class LogClock:
    """The song time the Transport rendered at output time t - latency_s, where the output clock read
    output_at_input_zero_s when the input clock was at 0. Correct across pauses (later: seeks, loops, Speed).

    devices = (output, input) StreamTimelines (see device_times): then voice time and output time meet on the host
    clock instead of by counting frames, so a lost output block, a dropped mic chunk or the two devices drifting apart
    no longer shift the song against the voice for the rest of the Take (#49). loop_s is then the host seconds from a
    frame leaving the output (its DAC time) to the voice answering it entering the input (its ADC time).

    What latency_s means depends on how it was calibrated:
    - loop_latency=True (calibrated on the device timestamps, #62): latency_s IS that loop. Nothing depends on when the
      first mic callback ran, so no Take starts with a random offset.
    - False (a latency calibrated by counting frames, before #62, or no timestamps at all): latency_s also holds the gap
      between the output's render frontier and the input's newest sample at the alignment (the first mic chunk,
      aligned_at_s = its end in voice time). Timed, that gap is taken out once, as read at the alignment, so the old
      number keeps its old meaning (and its random 0-21 ms, which only a recalibration removes).
    A timeline without a timestamp yet leaves the clock counting until both have one (see _settle).

    The plot draws on its own axis (#63, plot_t, PlotClock): host time minus a constant fixed at the alignment, so it
    equals voice time while nothing is lost, and a lost mic chunk is a gap in the voice line instead of a step of the
    whole plot. Cuts between Takes are kept as host times (answer_host) and voice frames compared on their capture
    host time (voice_host), so a mic chunk lost after the cut was computed cannot move it."""

    def __init__(self, log: SongLog, latency_s: float, output_at_input_zero_s: float,
                 devices: tuple[StreamTimeline, StreamTimeline] | None = None, aligned_at_s: float = 0.0,
                 loop_latency: bool = False):
        self._log = log
        self.latency_s = latency_s
        self.loop_latency = loop_latency
        self._lock = threading.RLock()                        # alignment and settling may race (audio vs tracker)
        self._alignment = (output_at_input_zero_s, aligned_at_s)
        self._devices = devices
        self._map = self._mapping(_COUNTING, _COUNTING)      # (output, input, loop_s, host0): swapped whole
        self.offset_s = aligned_at_s - self.to_output(aligned_at_s)    # = latency_s - output_at_input_zero_s untimed
        self._settle()

    @property
    def output_at_input_zero_s(self) -> float:
        return self._alignment[0]

    @property
    def aligned(self) -> bool:
        """The output had handed something to the device at the alignment. When not (the Backing's first block came
        after the first mic chunk) there was no render frontier to align on, and the runner calls realign() at the
        first chunk after one exists."""
        output_at_zero, at = self._alignment
        return output_at_zero + at > 0

    def realign(self, output_at_input_zero_s: float, aligned_at_s: float) -> None:
        """Align again at a later mic chunk (see aligned); any thread."""
        with self._lock:
            self._alignment = (output_at_input_zero_s, aligned_at_s)
            devices = self._devices
            timed = devices is None and self._map[0] is not _COUNTING
            out, inp = (self._map[0], self._map[1]) if timed else (_COUNTING, _COUNTING)
            self._map = self._mapping(out, inp)
            self.offset_s = aligned_at_s - self.to_output(aligned_at_s)
        self._settle()

    def _mapping(self, out: StreamTimeline, inp: StreamTimeline):
        output_at_zero, at = self._alignment
        if self.loop_latency and out is not _COUNTING:
            loop = self.latency_s
        else:
            loop = self.latency_s - (out.host(output_at_zero + at) - inp.host(at))
        return out, inp, loop, inp.host(at) - at

    def _settle(self) -> None:
        """Switch to the device timestamps once both timelines have one (any thread; idempotent)."""
        devices = self._devices
        if devices is None or not (len(devices[0]) and len(devices[1])):
            return
        with self._lock:
            if self._devices is None:
                return
            self._map = self._mapping(*devices)
            self._devices = None
            self.offset_s = self._alignment[1] - self.to_output(self._alignment[1])

    def _current(self):
        if self._devices is not None:
            self._settle()
        return self._map

    @property
    def timed(self) -> bool:
        """The song is read through the device timestamps (not by counting frames)."""
        return self._current()[0] is not _COUNTING

    @property
    def loop_s(self) -> float:
        return self._current()[2]

    def _output_at_host(self, h):
        """Output time heard at host capture time h (scalar or array). Timed, never past what has been rendered: a
        block whose callback has not run yet cannot be playing, so while one is late (the GIL held elsewhere) the song
        waits at the last frame handed over, as the silence in the headphones does."""
        out, _, loop_s, _ = self._current()
        o = out.stream(h - loop_s)
        if out is _COUNTING:
            return o
        rendered = getattr(self._log, "rendered_s", None)
        if rendered is None:
            return o
        return np.minimum(o, rendered) if np.ndim(o) else float(min(o, rendered))

    def to_output(self, t):
        """Output time (the Transport's frame count) answered by voice time t (scalar or array)."""
        return self._output_at_host(self.voice_host(t))

    def voice_host(self, t):
        """Host time voice time t was captured at (voice time itself, counting)."""
        return self._current()[1].host(t)

    def answer_host(self, output_s: float) -> float:
        """Host capture time of the voice that answers output time output_s (a Transport event): a cut between Takes
        is kept as this and compared with voice_host(frame), so a mic chunk lost meanwhile cannot move it (#63)."""
        out, _, loop_s, _ = self._current()
        return float(out.host(output_s) + loop_s)

    def to_voice(self, output_s):
        """Voice time that answers output time output_s, as the input timeline reads now (a time in the future is
        extrapolated: prefer answer_host where a later mic loss matters)."""
        if output_s == np.inf:
            return np.inf
        return self._current()[1].stream(self.answer_host(output_s))

    def voice_at_host(self, host_s: float) -> float:
        return float(self._current()[1].stream(host_s))

    def voice_now(self) -> float | None:
        """The voice time being sung now (its sample enters the mic now): the input stream read on the host clock now.
        None untimed. Runs ahead during a mic loss (see plot_now for drawing)."""
        if not self.timed:
            return None
        inp = self._map[1]
        return float(inp.stream(inp.clock.now()))

    # --- the plot's axis (#63) ---------------------------------------------------------------------------------------
    def plot_t(self, t):
        """Voice time t on the plot's axis: its capture host time minus the origin (= t while nothing is lost)."""
        _, inp, _, host0 = self._current()
        return inp.host(t) - host0

    def plot_of_host(self, host_s: float) -> float:
        return host_s - self._current()[3]

    def plot_now(self) -> float | None:
        """Now on the plot's axis: the host clock now minus the origin. Moves with the clock, never backwards, whatever
        the mic loses. None untimed."""
        if not self.timed:
            return None
        _, inp, _, host0 = self._map
        return float(inp.clock.now()) - host0

    def drawn(self) -> PlotClock:
        """This clock read on the plot's axis (what the window draws with)."""
        return PlotClock(self)

    # --- voice time ------------------------------------------------------------------------------------------------
    def __call__(self, t: float) -> float | None:
        return self._log.song_time(self.to_output(t))

    def song_times(self, t: np.ndarray) -> np.ndarray:
        """__call__ over an array of voice times, NaN where paused (for drawing: one pass, not a call per point)."""
        return _song_times(self._log, self.to_output(np.asarray(t, dtype=np.float64)))

    def tail(self) -> tuple[float, float | None, float] | None:
        """(voice time, song time, rate) where the rendered song ends, for projecting the song that is still to be
        heard: the voice time is ahead of the live voice by about the latency. Song time None = paused there. Timed,
        never before the voice now: while a block is late the song cannot go on before it arrives (see to_output)."""
        tail = self._log.tail()
        if tail is None:
            return None
        voice = self.to_voice(tail[0])
        now = self.voice_now()
        return (voice if now is None else max(voice, now), tail[1], tail[2])

    def paused_spans(self) -> list[tuple[float, float]]:
        """(start, end) voice times the song was paused (end inf = still paused): the voice there answers nothing."""
        return [(self.to_voice(start), self.to_voice(end)) for start, end in self._log.paused_spans()]


def _song_times(log, output_s: np.ndarray) -> np.ndarray:
    many = getattr(log, "song_times", None)
    if many is not None:
        return many(output_s)
    return np.array([np.nan if (s := log.song_time(o)) is None else s for o in output_s], dtype=np.float64)


class PlotClock:
    """A LogClock on the plot's axis (LogClock.plot_t): the same song clock interface (call, song_times, tail,
    paused_spans, offset_s) with plot time in place of voice time. Untimed the two axes are the same."""

    def __init__(self, clock: LogClock):
        self._c = clock

    @property
    def offset_s(self) -> float:
        return self._c.offset_s

    def _host(self, u):
        return np.asarray(u, dtype=np.float64) + self._c._current()[3] if np.ndim(u) else u + self._c._current()[3]

    def __call__(self, u: float) -> float | None:
        return self._c._log.song_time(self._c._output_at_host(self._host(u)))

    def song_times(self, u: np.ndarray) -> np.ndarray:
        return _song_times(self._c._log, self._c._output_at_host(self._host(np.asarray(u, dtype=np.float64))))

    def tail(self) -> tuple[float, float | None, float] | None:
        tail = self._c._log.tail()
        if tail is None:
            return None
        u = self._c.plot_of_host(self._c.answer_host(tail[0]))
        now = self._c.plot_now()
        return (u if now is None else max(u, now), tail[1], tail[2])

    def paused_spans(self) -> list[tuple[float, float]]:
        c = self._c
        return [(c.plot_of_host(c.answer_host(a)), np.inf if b == np.inf else c.plot_of_host(c.answer_host(b)))
                for a, b in c._log.paused_spans()]


class SampledClock:
    """A saved Take's own clock: the song time of each of its frames (None = paused), read between and just beyond
    them. Draws the Reference on a replay of a Take that was jumped into or paused. Beyond the last frame the song goes
    on at the rate it had there (the median step of its last frames, so a Take at Speed 50 % goes on at 0.5, not at
    1.0), and ends at song_end_s when that is given."""

    TAIL_FRAMES = 20     # how many of the last frames set the rate beyond the end

    def __init__(self, t: Sequence[float], song_times: Sequence[float | None], song_end_s: float | None = None):
        self._t = np.asarray(t, dtype=np.float64)
        self._song = np.array([np.nan if s is None else s for s in song_times], dtype=np.float64)
        self._song_end_s = song_end_s
        valid = np.flatnonzero(np.isfinite(self._song))
        self.offset_s = float(self._t[valid[0]] - self._song[valid[0]]) if len(valid) else 0.0
        self._rate = self._tail_rate()

    def _tail_rate(self) -> float:
        t, song = self._t[-self.TAIL_FRAMES:], self._song[-self.TAIL_FRAMES:]
        with np.errstate(invalid="ignore", divide="ignore"):
            steps = np.diff(song) / np.diff(t)
        steps = steps[np.isfinite(steps)]
        rate = float(np.median(steps)) if len(steps) else 1.0
        return rate if rate > 0 else 1.0

    def __call__(self, t: float) -> float | None:
        if not len(self._t):
            return None
        if t <= self._t[0]:
            edge, at, rate = self._song[0], self._t[0], 1.0
        elif t >= self._t[-1]:
            edge, at, rate = self._song[-1], self._t[-1], self._rate
        else:
            value = float(np.interp(t, self._t, self._song))
            return None if np.isnan(value) else value
        if np.isnan(edge):
            return None
        value = float(edge + (t - at) * rate)
        return None if self._song_end_s is not None and value > self._song_end_s else value

    def paused_spans(self) -> list[tuple[float, float]]:
        """(start, end) voice times of each run of frames without a song time (paused): from its first frame to the
        next frame with one (inf when the Take ended paused)."""
        paused = np.isnan(self._song).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], paused, [0]])))
        return [(float(self._t[a]), float(self._t[b]) if b < len(self._t) else np.inf)
                for a, b in zip(edges[::2], edges[1::2])]


class PauseFold:
    """Voice time <-> view time, the plot's x axis: voice time with every pause of the song folded away. While the
    song is paused the view time stands still, and on resume it goes on from the same place, so the plot freezes and
    then continues seamlessly. spans: the clock's paused_spans() (voice times, sorted, apart; end inf = paused now).
    No spans (Nearest note, Key, a Take never paused): view time = voice time."""

    def __init__(self, spans: Sequence[tuple[float, float]] = ()):
        self._start = np.array([s for s, _ in spans], dtype=np.float64)
        self._end = np.array([e for _, e in spans], dtype=np.float64)
        length = self._end - self._start
        self._before = np.concatenate([[0.0], np.cumsum(length)[:-1]])     # folded away before each span
        self._at = self._start - self._before                              # the view time each span folds to

    def paused(self, t: float) -> bool:
        """The voice time t lies in a pause: the voice there is neither drawn nor scored."""
        return bool(np.any((self._start <= t) & (t < self._end)))

    def to_view(self, t):
        t = np.asarray(t, dtype=np.float64)
        view = t.copy()
        for start, end in zip(self._start, self._end):
            view = view - np.clip(t - start, 0.0, end - start)
        return float(view) if view.ndim == 0 else view

    def to_voice(self, x):
        """The voice time shown at view time x. At a fold the voice time just after the pause (where the song went
        on); past the fold of a pause still going on, its start (the song waits there)."""
        x = np.asarray(x, dtype=np.float64)
        voice = x.copy()
        for start, end, before, at in zip(self._start, self._end, self._before, self._at):
            after = x >= at
            voice = np.where(after, start if np.isinf(end) else x + before + (end - start), voice)
        return float(voice) if voice.ndim == 0 else voice


def project_song_time(dt: np.ndarray, song_s: float, rate: float, loop_end_s: float | None = None,
                      loop_pass_start_s: float = 0.0, song_end_s: float | None = None) -> np.ndarray:
    """The song time that will be heard `dt` output seconds after the song was at `song_s` playing at `rate` (song
    seconds per output second). Inside a loop the song runs to loop_end_s and wraps to loop_pass_start_s (the Lead-in),
    again and again; outside one it is NaN (no song) past song_end_s. Assumes what is playing now keeps playing: a
    seek, pause or Speed change that has not been rendered yet is not foreseen."""
    dt = np.asarray(dt, dtype=np.float64)
    if loop_end_s is not None and rate > 0 and song_s >= loop_end_s:
        song_s = loop_pass_start_s          # already past the Section end: the wrap is due, the song goes on from the Lead-in
    song = song_s + dt * rate
    if loop_end_s is not None and rate > 0:
        pass_s = max(loop_end_s - loop_pass_start_s, 1e-6)
        over = song - loop_end_s                        # song seconds past the Section end, as if it ran on
        wrapped = loop_pass_start_s + np.mod(over, pass_s)
        return np.where(over >= 0, wrapped, song)
    if song_end_s is not None:
        song = np.where(song > song_end_s, np.nan, song)
    return song


def linear_song_times(frames: Sequence[PitchFrame], offset_s: float = 0.0) -> list[float]:
    """Per-frame song times of a linearly played Take (t - offset_s)."""
    return [f.t - offset_s for f in frames]

"""Test mode: a fake mic and a null output so the real app runs end to end without a microphone or speakers.

One FakeAudioHub thread is the audio hardware. Every step (BLOCKSIZE frames of app time) it first lets each open
output stream play a step (calling its callback for that step), then gives each fake mic one chunk. Steps are paced by the wall clock times `time_scale`, so the Transport's song clock advances exactly
as it would with a device. Because one thread drives both sides, output and input stay in a fixed relation:

- FakeMic(wav) reads the wav from `offset_s`, starting at the moment the open output stream started (or the mic's
  own start when nothing plays) and heard `delay_s` later: a singer who sings the file in time with the song,
  as late as the output->input latency. Pauses and seeks after the start are not followed: it is a file.
- FakeMic(ECHO) hears the output mix itself, `delay_s` late: with the Guide vocal on and the Backing at 0 it sings
  exactly what plays, through pauses, seeks, loops and Speed; it also answers the calibration clicks.

Like a real device, every output block is stamped with the hub time it is played at (time_info.outputBufferDacTime)
and every mic chunk with the hub time it was heard at (FakeMic.times), on the hub's own clock (FakeAudioHub.now).
Faults (#49) make the fake hardware misbehave the way a real one does under load: lost output blocks, dropped mic
chunks, the output's crystal drifting against the mic's, and the UI thread blocked; see Faults.

Selected only by flags (`sing-pitch --fake-mic ...`, see parse_flags) or by the QA driver (sing_pitch.qa)."""
from __future__ import annotations

import argparse
import ctypes
import functools
import math
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import BLOCKSIZE, SAMPLE_RATE

FAKE_OUTPUT = "Fake output"        # the output's name: settings.json keys its latency by it
FAKE_MIC = "Fake mic"
ECHO = "echo"                      # --fake-mic echo: the mic hears the output mix
ECHO_KEEP_S = 5.0                  # the longest echo delay


@dataclass(frozen=True)
class Faults:
    """Fake hardware faults, all off by default (#49). Times in app seconds / milliseconds.

    - underflow_every_s, underflow_ms: every so often the output callback "misses its deadline": the output plays
      silence for that long, its callback is not called and the stream does not advance (what CoreAudio does when a
      Python callback waits too long for the GIL; it raises no status flag, measured 2026-11-25).
    - overflow_every_s, overflow_ms: every so often the mic loses that much input: the chunks are never delivered.
    - drift_ppm: the output's clock runs this many parts per million faster (negative: slower) than the mic's.
    - deadline_out_ms, deadline_in_ms: real-time deadlines. When the hub thread runs late (the GIL held elsewhere), a
      step more than this late is lost: silence out (an underflow) / nothing in (an overflow). None = no deadline (the
      hub catches up, as before #49).
    - ui_stall_every_s, ui_stall_ms, ui_stall_gil: the UI thread is blocked that long every so often (UiStall), holding
      the GIL (a slow C++ paint) or not (a sleep).
    Device shapes (#62, #63 review), also off by default:
    - render_ahead_ms: the output renders whole blocks of its `blocksize` (the Backing asks 1024) and keeps this much
      rendered ahead of what plays, like a real device, instead of one hub step just in time. The render frontier read
      at the first mic chunk then lies anywhere within a block (the phase), as on CoreAudio. Do not combine with
      underflow or deadline: a lost step does not move blocks already stamped.
    - mic_late_ms: the mic's first chunks reach the app this late, in one burst (a first callback held up at Start);
      their stamps stay true.
    - mic_phase: each mic start hears its first chunk 0-3 hub steps (0-16 ms) after it opens, at random: with
      render_ahead, a different phase of the output's blocks every Take, as two real devices have.
    - jitter_out_us, jitter_in_us: Gaussian noise (SD) on the reported DAC / ADC stamps only; the audio is on time.
    - latency_step_at_s, latency_step_ms: from that many seconds after the output started, its reported DAC stamps are
      this much later (negative: earlier), the audio unchanged: a reported latency that changes (e.g. Bluetooth).
    Spec string for --fake-faults and QA(faults=...): "underflow=EVERY_S:MS,overflow=EVERY_S:MS,drift=PPM,
    deadline=OUT_MS:IN_MS,ui_stall=EVERY_S:MS[:sleep],render=AHEAD_MS,mic_late=MS,mic_phase=1,jitter=OUT_US:IN_US,
    latency_step=AT_S:MS", any subset."""

    underflow_every_s: float | None = None
    underflow_ms: float = 0.0
    overflow_every_s: float | None = None
    overflow_ms: float = 0.0
    drift_ppm: float = 0.0
    deadline_out_ms: float | None = None
    deadline_in_ms: float | None = None
    ui_stall_every_s: float | None = None
    ui_stall_ms: float = 0.0
    ui_stall_gil: bool = True
    render_ahead_ms: float | None = None
    mic_late_ms: float = 0.0
    mic_phase: bool = False
    jitter_out_us: float = 0.0
    jitter_in_us: float = 0.0
    latency_step_at_s: float | None = None
    latency_step_ms: float = 0.0

    @classmethod
    def parse(cls, spec: str | Faults | None) -> Faults:
        if isinstance(spec, Faults):
            return spec
        kw: dict = {}
        for part in filter(None, (p.strip() for p in (spec or "").split(","))):
            name, _, value = part.partition("=")
            args = value.split(":")
            try:
                if name in ("underflow", "overflow"):
                    kw[f"{name}_every_s"], kw[f"{name}_ms"] = float(args[0]), float(args[1])
                elif name == "drift":
                    kw["drift_ppm"] = float(args[0])
                elif name == "deadline":
                    kw["deadline_out_ms"] = float(args[0])
                    kw["deadline_in_ms"] = float(args[1]) if len(args) > 1 else float(args[0])
                elif name == "render":
                    kw["render_ahead_ms"] = float(args[0])
                elif name == "mic_phase":
                    kw["mic_phase"] = value not in ("0", "false", "no")
                elif name == "mic_late":
                    kw["mic_late_ms"] = float(args[0])
                elif name == "jitter":
                    kw["jitter_out_us"] = float(args[0])
                    kw["jitter_in_us"] = float(args[1]) if len(args) > 1 else float(args[0])
                elif name == "latency_step":
                    kw["latency_step_at_s"], kw["latency_step_ms"] = float(args[0]), float(args[1])
                elif name == "ui_stall":
                    kw["ui_stall_every_s"], kw["ui_stall_ms"] = float(args[0]), float(args[1])
                    kw["ui_stall_gil"] = not (len(args) > 2 and args[2] == "sleep")
                else:
                    raise ValueError(name)
            except (ValueError, IndexError):
                raise ValueError(f"bad fault {part!r}; see testmode.Faults") from None
        return cls(**kw)


class _Periodic:
    """Every `every_s` app seconds from the first step seen, `ms` of steps are hit."""

    def __init__(self, every_s: float | None, ms: float, sample_rate: int, step: int):
        self._every = None if every_s is None else max(step, round(every_s * sample_rate))
        self._len = math.ceil(ms / 1000 * sample_rate / step) * step
        self._origin: int | None = None

    def hit(self, frame: int) -> bool:
        if self._every is None or self._len <= 0:
            return False
        if self._origin is None:
            self._origin = frame
        k = frame - self._origin
        return k >= self._every and k % self._every < self._len


class TimeInfo:
    """What a sounddevice callback gets as time_info: host times on the hub clock."""

    def __init__(self, current: float, dac: float = 0.0, adc: float = 0.0):
        self.currentTime, self.outputBufferDacTime, self.inputBufferAdcTime = current, dac, adc


class FakeOutputStream:
    """A null sounddevice.OutputStream: the hub consumes its audio at real-time pace. Same kwargs as the real one.
    The callback is asked for one hub step (256 frames) at a time, just in time, whatever `blocksize` says: then
    what is rendered when a mic chunk arrives is what is playing, which is how the window aligns the song clock
    (live.py), so a fake latency of L is exactly a calibrated latency of L. A real device renders ahead; calibration
    measures that in. Each callback gets time_info.outputBufferDacTime = the hub time its first frame is played at.
    `played` is the truth: which stream frame was played at which hub time (for measuring, never read by the app)."""

    def __init__(self, hub: FakeAudioHub, callback: Callable, channels: int = 2, blocksize: int = 1024,
                 samplerate: int = SAMPLE_RATE, dtype: str = "float32", device=None, **_):
        from .take_engine import StreamTimeline

        if samplerate != hub.sample_rate:
            raise ValueError(f"fake output runs at {hub.sample_rate} Hz, not {samplerate}")
        self._hub, self._callback = hub, callback
        self.channels, self.blocksize, self.device = channels, blocksize, device
        self.clock = hub                          # the clock time_info is on
        self.started_frame: int | None = None     # hub frame of the first frame played
        self._block = np.zeros((0, channels), np.float32)
        self._at = 0                              # frames of _block already played
        self._played = 0                          # stream frames played so far
        self._carry = 0.0                         # drift: the fraction of a stream frame owed to the next step
        self._buf = np.zeros((0, channels), np.float32)   # render_ahead: rendered, not yet played
        self.played = StreamTimeline(hub)
        self.played.TOLERANCE_S = 1e-7
        self.failed: Exception | None = None

    @property
    def time(self) -> float:
        """Like sounddevice's Stream.time: now on the clock of time_info."""
        return self._hub.now()

    def start(self) -> None:
        self._hub._add_output(self)

    def stop(self) -> None:
        self._hub._remove(self)

    def close(self) -> None:
        pass

    def _call(self, block: np.ndarray, dac: float) -> None:
        if self.failed is not None:
            return
        f, sr = self._hub.faults, self._hub.sample_rate
        if f.jitter_out_us:
            dac += self._hub._rng.normal(0.0, f.jitter_out_us * 1e-6)
        if f.latency_step_at_s is not None and self.started_frame is not None \
                and dac - self.started_frame / sr >= f.latency_step_at_s:
            dac += f.latency_step_ms / 1000
        try:
            self._callback(block, len(block), TimeInfo(self._hub.now_s, dac=dac), None)
        except Exception as exc:    # PortAudio would abort the stream: stay silent, say why once
            self.failed = exc
            print(f"fake output: callback failed: {exc!r}", file=sys.stderr)
            block[:] = 0

    def _pull(self, frames: int, hub_frame: int, rate: float) -> np.ndarray:
        """The next `frames` stream frames, calling back for new blocks just in time, or `blocksize` blocks kept
        render_ahead_ms ahead (hub thread)."""
        ahead = self._hub.faults.render_ahead_ms
        sr = self._hub.sample_rate
        if ahead is not None:
            need = frames + round(ahead / 1000 * sr)
            while len(self._buf) < need:
                block = np.zeros((self.blocksize, self.channels), np.float32)
                self._call(block, (hub_frame + len(self._buf) / rate) / sr)
                self._buf = np.concatenate([self._buf, block])
            out, self._buf = self._buf[:frames], self._buf[frames:]
            return out
        out = np.empty((frames, self.channels), np.float32)
        done = 0
        while done < frames:
            if self._at >= len(self._block):
                block = np.zeros((frames - done, self.channels), np.float32)
                self._call(block, (hub_frame + done / rate) / sr)
                self._block, self._at = block, 0
            n = min(frames - done, len(self._block) - self._at)
            out[done : done + n] = self._block[self._at : self._at + n]
            done, self._at = done + n, self._at + n
        return out

    def _play(self, frames: int, hub_frame: int = 0, lost: bool = False) -> np.ndarray:
        """What reaches the speaker over hub frames [hub_frame, hub_frame + frames) (hub thread). lost: the callback
        missed its deadline, the speaker plays silence and the stream stands still."""
        sr = self._hub.sample_rate
        if lost:
            return np.zeros((frames, self.channels), np.float32)
        rate = 1.0 + self._hub.faults.drift_ppm * 1e-6
        self.played.add(self._played / sr, hub_frame / sr)
        want = frames * rate + self._carry
        n = max(0, math.floor(want))
        self._carry = want - n
        audio = self._pull(n, hub_frame, rate)
        self._played += n
        if n == frames:
            return audio
        if n == 0:
            return np.zeros((frames, self.channels), np.float32)
        return audio[np.minimum((np.arange(frames) * rate).astype(int), n - 1)]    # the drifting crystal


class FakeMic:
    """An AudioInput fed by the hub: chunks of BLOCKSIZE at real-time pace. `source` is a mono float32 array at
    the hub rate, or ECHO. See the module docstring for what it hears."""

    def __init__(self, hub: FakeAudioHub, source, offset_s: float = 0.0, delay_s: float = 0.0, loop: bool = False):
        from .take_engine import StreamTimeline

        self._hub = hub
        self._source = source
        self.sample_rate = hub.sample_rate
        self.offset_s, self.delay_s, self.loop = float(offset_s), float(delay_s), loop
        self._on_chunk: Callable[[np.ndarray], None] | None = None
        self._origin = 0          # hub frame where the source's offset_s is heard, before the delay
        self._delivered = 0       # samples handed to on_chunk since start()
        self._late_until: int | None = None     # mic_late: hub frame the first chunks are held back until
        self._held: list = []
        self._opens_at = 0
        self.times = StreamTimeline(hub)    # hub time each delivered sample was heard at (like an ADC timestamp)

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        from .take_engine import StreamTimeline

        self._on_chunk = on_chunk
        self._delivered, self.times = 0, StreamTimeline(self._hub)
        self._late_until, self._held, self._opens_at = None, [], 0
        self._hub._add_input(self)

    def _deliver(self, frame: int, chunk: np.ndarray) -> None:
        """Hand the chunk heard from hub frame `frame` on to the app (hub thread); while mic_late holds the first
        chunks back, keep them and hand them all over at once when it ends."""
        if self._late_until is None:
            skip = int(self._hub._rng.integers(0, 4)) * self._hub.step if self._hub.faults.mic_phase else 0
            self._opens_at = frame + skip
            self._late_until = frame + skip + round(self._hub.faults.mic_late_ms / 1000 * self.sample_rate)
        if frame < self._opens_at:
            return
        if frame < self._late_until:
            self._held.append((frame, chunk))
            return
        held, self._held = self._held, []
        for f, c in held + [(frame, chunk)]:
            adc = f / self.sample_rate
            if self._hub.faults.jitter_in_us:
                adc += self._hub._rng.normal(0.0, self._hub.faults.jitter_in_us * 1e-6)
            self.times.add(self._delivered / self.sample_rate, adc, len(c))
            self._delivered += len(c)
            self._on_chunk(c)

    def stop(self) -> None:
        self._hub._remove(self)

    def _chunk(self, frame: int, n: int) -> np.ndarray:
        """Samples heard at hub frames [frame, frame + n) (hub thread)."""
        delay = round(self.delay_s * self.sample_rate)
        if isinstance(self._source, str):
            return self._hub._mix_at(frame - delay, n)
        rel = (frame - self._origin - delay) + np.arange(n)      # frames since the voice started singing
        idx = round(self.offset_s * self.sample_rate) + rel
        src = self._source
        if self.loop and len(src):
            idx = idx % len(src)
        ok = (rel >= 0) & (idx >= 0) & (idx < len(src))          # silent until then, and after the end
        out = np.zeros(n, np.float32)
        out[ok] = src[idx[ok]]
        return out


class FakeAudioHub:
    """The fake audio hardware: one thread steps every fake output and mic, paced at time_scale x real time."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, step: int = BLOCKSIZE, time_scale: float = 1.0,
                 record: bool = False, faults: Faults | str | None = None):
        self.sample_rate, self.step, self.time_scale = sample_rate, step, float(time_scale)
        self.faults = Faults.parse(faults)
        self._rng = np.random.default_rng(0)     # stamp jitter
        self._underflow = _Periodic(self.faults.underflow_every_s, self.faults.underflow_ms, sample_rate, step)
        self._overflow = _Periodic(self.faults.overflow_every_s, self.faults.overflow_ms, sample_rate, step)
        self.lost_out_steps = self.lost_in_steps = 0     # steps the faults cost (for tests and the QA report)
        self._cond = threading.Condition()   # guards everything below; held while a step runs
        self._frame = 0
        self._pace = (time.perf_counter(), 0)   # (wall clock, hub frame) the pace counts from
        self._late_s = 0.0                   # how late the step being run is (app seconds)
        self._outputs: list[FakeOutputStream] = []
        self._inputs: list[FakeMic] = []
        self._held = 0
        self._closed = False
        self._echo = np.zeros((round(ECHO_KEEP_S * sample_rate) // step + 1) * step, np.float32)  # ring, mono mix
        self._recording: list[np.ndarray] | None = [] if record else None
        self._thread = threading.Thread(target=self._run, name="sing-pitch-fake-audio", daemon=True)
        self._thread.start()

    # --- what the app opens ------------------------------------------------------------------------------------
    def open_output(self, **kwargs) -> FakeOutputStream:
        """A stream_factory for BackingPlayer, DronePlayer and the calibration player."""
        return FakeOutputStream(self, **kwargs)

    # --- clock ---------------------------------------------------------------------------------------------------
    @property
    def now_s(self) -> float:
        """App time: seconds of audio the hub has stepped since it was made."""
        return self._frame / self.sample_rate

    def now(self) -> float:
        """App time now, between steps too: the clock the fake time_info is on (like Pa_GetStreamTime). The frame
        being played by the wall clock, kept within the last step played."""
        frame = self._frame
        t0, f0 = self._pace
        due = f0 + (time.perf_counter() - t0) * self.time_scale * self.sample_rate
        return min(max(due, frame - self.step), frame) / self.sample_rate

    def output_frame_at(self, host_s: float, stream: FakeOutputStream | None = None) -> float | None:
        """The truth: the stream frame (default: the newest open output) played at hub time host_s; None without one."""
        stream = stream or (self._outputs[-1] if self._outputs else None)
        if stream is None or not len(stream.played):
            return None
        return min(stream.played.stream(host_s) * self.sample_rate, stream._played)   # silent since: stands still

    @contextmanager
    def held(self):
        """Stop the clock while the block runs (e.g. Start + seek land before the first block is played)."""
        with self._cond:
            self._held += 1
        try:
            yield
        finally:
            with self._cond:
                self._held -= 1
                self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join()

    def recording(self) -> np.ndarray:
        """Everything played, stereo, from the hub's first frame (only with record=True)."""
        if self._recording is None:
            raise RuntimeError("the hub was made without record=True")
        with self._cond:
            return np.concatenate(self._recording) if self._recording else np.zeros((0, 2), np.float32)

    def write_recording(self, path: Path) -> None:
        import soundfile as sf

        sf.write(str(path), self.recording(), self.sample_rate)

    # --- streams --------------------------------------------------------------------------------------------------
    def _add_output(self, stream: FakeOutputStream) -> None:
        with self._cond:
            stream.started_frame = self._frame
            self._outputs.append(stream)

    def _add_input(self, mic: FakeMic) -> None:
        with self._cond:
            mic._origin = self._outputs[-1].started_frame if self._outputs else self._frame
            self._inputs.append(mic)

    def _remove(self, stream) -> None:
        with self._cond:     # waits for a running step: no callback runs after stop() returns, as with PortAudio
            for streams in (self._outputs, self._inputs):
                if stream in streams:
                    streams.remove(stream)

    def _mix_at(self, frame: int, n: int) -> np.ndarray:
        """The mono output mix at hub frames [frame, frame + n); silence before the hub started or out of the ring."""
        out = np.zeros(n, np.float32)
        lo = max(frame, 0, self._frame + self.step - len(self._echo))
        hi = min(frame + n, self._frame + self.step)
        if hi > lo:
            idx = np.arange(lo, hi) % len(self._echo)
            out[lo - frame : hi - frame] = self._echo[idx]
        return out

    # --- the thread -----------------------------------------------------------------------------------------------
    def _step(self) -> None:
        """One step. self._late_s: app seconds this step runs after it was due (the deadline faults; set by _run, so
        the signature stays the one tests wrap)."""
        n, frame, faults, late_s = self.step, self._frame, self.faults, self._late_s
        lost_out = self._underflow.hit(frame) or (faults.deadline_out_ms is not None
                                                  and late_s * 1000 > faults.deadline_out_ms)
        lost_in = self._overflow.hit(frame) or (faults.deadline_in_ms is not None
                                                and late_s * 1000 > faults.deadline_in_ms)
        mix = np.zeros((n, 2), np.float32)
        for stream in list(self._outputs):
            played = stream._play(n, frame, lost_out)
            mix[:, : played.shape[1]] += played[:, :2]
        self.lost_out_steps += bool(lost_out and self._outputs)
        if self._recording is not None:
            self._recording.append(mix)
        self._echo[np.arange(frame, frame + n) % len(self._echo)] = mix.mean(axis=1)
        self.lost_in_steps += bool(lost_in and self._inputs)
        for mic in list(self._inputs):
            chunk = mic._chunk(frame, n)
            if lost_in:
                continue
            try:
                mic._deliver(frame, chunk)
            except Exception as exc:
                print(f"fake mic: on_chunk failed: {exc!r}", file=sys.stderr)
        self._frame = frame + n

    def _run(self) -> None:
        t0, f0 = time.perf_counter(), 0
        self._pace = (t0, f0)
        while True:
            with self._cond:
                if self._held and not self._closed:
                    while self._held and not self._closed:
                        self._cond.wait()
                    t0, f0 = time.perf_counter(), self._frame       # the clock restarts where it was stopped
                    self._pace = (t0, f0)
                if self._closed:
                    return
                due = t0 + (self._frame - f0) / self.sample_rate / self.time_scale
                self._late_s = max(0.0, time.perf_counter() - due) * self.time_scale
                self._step()
                due = t0 + (self._frame - f0) / self.sample_rate / self.time_scale
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)


# --- window wiring ----------------------------------------------------------------------------------------------
def load_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    from .playback.backing import load_audio

    return np.ascontiguousarray(load_audio(Path(path), sample_rate).mean(axis=1), dtype=np.float32)


class FakeMicFactory:
    """MainWindow's input_factory: every Start and calibration opens a new FakeMic with these settings. The
    attributes may be changed between Takes (the QA driver moves offset_s to where a Take starts)."""

    def __init__(self, hub: FakeAudioHub, source, offset_s: float = 0.0, delay_s: float = 0.0, loop: bool = False):
        self.hub, self.source, self.offset_s, self.delay_s, self.loop = hub, source, offset_s, delay_s, loop

    def __call__(self, _device_index=None) -> FakeMic:
        return FakeMic(self.hub, self.source, self.offset_s, self.delay_s, self.loop)


class _CalibrationPlayer:
    """CalibrationRun's player_factory: the click track through a BackingPlayer on the fake output."""

    def __init__(self, hub: FakeAudioHub):
        self._hub = hub

    def __call__(self, audio, rate):
        from .playback import BackingPlayer

        return BackingPlayer(audio, rate, sample_rate=rate, stream_factory=self._hub.open_output)


def _fake_output_info() -> tuple[str, str]:
    return FAKE_OUTPUT, "headphones"


def _no_devices() -> list[dict]:
    return []


def _fake_mic_devices() -> list[dict]:
    return [dict(index=0, name=FAKE_MIC, max_input_channels=1, max_output_channels=0)]


def _always_headphones(_name: str) -> str:
    return "headphones"


def window_kwargs(hub: FakeAudioHub, mic: FakeMicFactory | None, fake_output: bool) -> dict:
    """MainWindow keyword arguments that put the fakes in place (module-level callables only: nothing here holds
    the window, so no reference cycle through it, see handoff-segfault)."""
    from .playback import BackingPlayer, DronePlayer

    kwargs: dict = {}
    if mic is not None:
        kwargs.update(input_factory=mic, input_devices=_fake_mic_devices)
    if fake_output:
        kwargs.update(backing_factory=functools.partial(BackingPlayer.from_file, stream_factory=hub.open_output),
                      drone_factory=functools.partial(DronePlayer, stream_factory=hub.open_output),
                      calibration_player_factory=_CalibrationPlayer(hub), output_info=_fake_output_info,
                      output_devices=_no_devices, output_kind_of=_always_headphones)
    return kwargs


def hold_the_gil(ms: float) -> None:
    """Keep the GIL for `ms` of wall time, as a slow C++ paint does: no other Python thread runs meanwhile. With the C
    library's usleep where there is one (macOS, Linux: ctypes.PyDLL keeps the GIL through the call, as before);
    without one (Windows has no libc to load), a pure-Python wait with the switch interval raised past it, so CPython
    never hands the GIL over (measured on Windows: 150 ms held, 10 of 10)."""
    usleep = _libc_usleep()
    if usleep is not None:
        usleep(int(ms * 1000))
        return
    interval = sys.getswitchinterval()
    sys.setswitchinterval(max(interval, ms / 1000 + 1.0))
    try:
        end = time.perf_counter() + ms / 1000
        while time.perf_counter() < end:
            pass
    finally:
        sys.setswitchinterval(interval)


def _libc_usleep():
    try:
        return ctypes.PyDLL(None).usleep
    except (OSError, TypeError, AttributeError):    # no C library to load (Windows), or no usleep in it
        return None


class UiStall:
    """Blocks the UI thread for `ms` every `every_s` app seconds (Faults.ui_stall_*): holding the GIL, as a slow
    C++ paint does (hold_the_gil), or sleeping. Keep it alive as long as it should run;
    stop() ends it. Built on the UI thread after the QApplication."""

    def __init__(self, hub: FakeAudioHub, faults: Faults):
        from PySide6 import QtCore

        self._hub, self._ms, self._gil = hub, faults.ui_stall_ms, faults.ui_stall_gil
        self._every = faults.ui_stall_every_s
        self._next = hub.now_s + self._every
        self.stalls = 0
        self._timer = QtCore.QTimer()
        self._timer.setInterval(5)
        self._timer.timeout.connect(self._check)         # a bound method: nothing here holds a window
        self._timer.start()

    def _check(self) -> None:
        if self._hub.now_s < self._next:
            return
        self._next = self._hub.now_s + self._every
        self.stalls += 1
        if self._gil:
            hold_the_gil(self._ms)
        else:
            time.sleep(self._ms / 1000)

    def stop(self) -> None:
        self._timer.stop()


# --- flags --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Flags:
    fake_mic: str | None = None           # a wav path, or ECHO
    fake_mic_offset: float = 0.0
    fake_mic_loop: bool = False
    fake_latency: float | None = None     # None = the latency configured for the fake output
    fake_output: bool = False
    record_output: str | None = None
    fake_faults: str | Faults | None = None   # Faults or its spec string

    @property
    def any(self) -> bool:
        return self.fake_mic is not None or self.fake_output


def parse_flags(argv: list[str]) -> tuple[Flags, list[str]]:
    """The test-mode flags out of argv, and the rest. Without them the app is exactly the normal one."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--fake-mic")
    p.add_argument("--fake-mic-offset", type=float, default=0.0)
    p.add_argument("--fake-mic-loop", action="store_true")
    p.add_argument("--fake-latency", type=float)
    p.add_argument("--fake-output", action="store_true")
    p.add_argument("--record-output")
    p.add_argument("--fake-faults")
    ns, rest = p.parse_known_args(argv)
    if ns.record_output and not ns.fake_output:
        raise SystemExit("--record-output needs --fake-output")
    if ns.fake_faults is not None:
        try:
            Faults.parse(ns.fake_faults)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    return Flags(ns.fake_mic, ns.fake_mic_offset, ns.fake_mic_loop, ns.fake_latency, ns.fake_output,
                 ns.record_output, ns.fake_faults), rest


def build(flags: Flags, home: Path, time_scale: float = 1.0) -> tuple[FakeAudioHub, FakeMicFactory | None, dict]:
    """(hub, mic factory, MainWindow kwargs) for these flags. The mic's delay defaults to the latency the window
    will use for the fake output, so the file mic is a perfect singer out of the box."""
    from .settings import Settings

    hub = FakeAudioHub(time_scale=time_scale, record=flags.record_output is not None, faults=flags.fake_faults)
    mic = None
    if flags.fake_mic is not None:
        delay = flags.fake_latency
        if delay is None:
            delay = Settings.load(home).latency_for(FAKE_OUTPUT if flags.fake_output else None)
        source = ECHO if flags.fake_mic == ECHO else load_mono(Path(flags.fake_mic), hub.sample_rate)
        mic = FakeMicFactory(hub, source, flags.fake_mic_offset, delay, flags.fake_mic_loop)
    return hub, mic, window_kwargs(hub, mic, flags.fake_output)

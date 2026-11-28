"""FastTracker: a low-latency classical pitch tracker (YIN with extra octave-error rules), drop-in for the app's SwiftF0 tracker.

Interface (same as sing_pitch.pitch.swiftf0.SwiftF0Tracker): reset(), push(chunk, sample_rate) -> list[PitchFrame], flush() -> list[PitchFrame].
Only numpy is needed (numba was tried for the dip search and gave no gain: the FFTs dominate, see report.md).

What it does, per hop of 256 samples at 48 kHz (5.33 ms), on the newest 2560 samples (53 ms):
  1. YIN difference function (FFT), cumulative-mean normalised -> dn[tau].
  2. Pick a lag: first dip below `thr`, else the shortest dip close to the global minimum; then
     - sub-harmonic rule: a dip at tau/2, tau/3 or tau/4 that is nearly as low (within `sub_abs`) wins;
     - up rule: a dip at 2*tau that is clearly lower (by the factor `up_rel`) wins.
  3. Causal continuity: if the pick is >= `cont_far` cents from the last voiced pitch (at most `cont_age` frames old) and dn has a
     dip < `cont_thr` within `cont_win` cents of that pitch, take that dip. After `cont_release` frames in a row where the raw picks
     agree with each other against the continuity, the new pitch is accepted (a genuine leap).
  4. Octave consensus gate (octave_consensus = K): a voiced pitch 12 (or 24) semitones +-2.5 from the established pitch is folded back
     by that octave at once; only K voiced frames in a row on the new octave switch the established octave. The established octave
     is forgotten after `consensus_gap_s` of unvoiced frames.
Frame semantics: t = centre of the window in seconds since reset(); confidence = 0.6 at aperiodicity `enter`, 0.4 at `leave`
(linear, so the app's gate enter 0.6 / leave 0.4 is a hysteresis on aperiodicity); loudness_db = 10 log10 of the mean square of the
512 samples around the centre (what SwiftF0 reports: 32 ms around the frame), 0 dBFS = full-scale square wave.
Release age: a frame is emitted the moment its window is complete = window / 2 = 26.7 ms after its moment for the default window; the
experiments' convention (window / 2 + one hop, because the newest chunk is up to one hop old when it arrives) gives 32.0 ms.
"""
from __future__ import annotations

import math

import numpy as np

from .frame import PitchFrame

SR = 48000
HOP = 256
_TMIN = 43            # floor(48000 / 1100)


class FastTracker:
    def __init__(self, window: int = 2560, fmin_hz: float = 65.0, fmax_hz: float = 1100.0,
                 thr: float = 0.10, fb_margin: float = 0.05,
                 sub: bool = True, sub_abs: float = 0.15, up: bool = True, up_rel: float = 0.3, up_abs: float = 0.02,
                 cont: bool = True, cont_far: float = 350.0, cont_thr: float = 0.5, cont_win: float = 80.0,
                 cont_age: int = 9, cont_release: int = 15,
                 octave_consensus: int | None = 3, consensus_establish: int = 5, consensus_tol_st: float = 2.5,
                 consensus_offsets: tuple = (12.0, 24.0), consensus_fifths: bool = False, consensus_gap_s: float = 0.4,
                 enter: float = 0.20, leave: float = 0.30, gate_db: float = -50.0, parabolic: bool = True) -> None:
        self.N = int(window)
        self.tmax = int(SR // fmin_hz)
        if self.tmax > self.N // 2:
            raise ValueError("window too short for fmin_hz")
        self.W = self.N - self.tmax
        self.fmin, self.fmax = 60.0, float(fmax_hz)
        self.thr, self.fb_margin = thr, fb_margin
        self.sub, self.sub_abs, self.up, self.up_rel, self.up_abs = sub, sub_abs, up, up_rel, up_abs
        self.cont, self.cont_far, self.cont_thr, self.cont_win = cont, cont_far, cont_thr, cont_win
        self.cont_age, self.cont_release = cont_age, cont_release
        self.K = octave_consensus
        self.E, self.tol = consensus_establish, consensus_tol_st
        self.offsets, self.fifths = tuple(consensus_offsets), consensus_fifths
        self.gap_frames = int(round(consensus_gap_s * SR / HOP))
        self.enter, self.leave, self.gate_db, self.parabolic = enter, leave, gate_db, parabolic
        self._ar = np.arange(self.tmax + 1, dtype=np.float64)
        self.reset()

    # ------------------------------------------------------------------ public interface
    def reset(self) -> None:
        self._buf = np.zeros(0)
        self._start = 0          # global sample index of _buf[0]
        self._next = 0           # global start index of the next window
        self._total = 0          # samples received
        # continuity
        self._prev = None        # (tau, frames_ago) of the last voiced pitch
        self._raw_run = 0
        self._last_raw = None
        # internal gate (mirror of the app's hysteresis)
        self._voiced = False
        # octave consensus
        self._est = None
        self._agree = 0
        self._last_l = None
        self._gap = 0
        self._streak = 0
        self._streak_dir = 0
        self.n_folded = 0
        self.n_switch = 0

    def push(self, chunk: np.ndarray, sample_rate: int) -> list[PitchFrame]:
        if sample_rate != SR:
            raise ValueError(f"FastTracker needs {SR} Hz audio, got {sample_rate}")
        x = np.asarray(chunk, dtype=np.float64).reshape(-1)
        self._total += len(x)
        self._buf = np.concatenate([self._buf, x]) if len(self._buf) else x.copy()
        out = []
        N = self.N
        while self._next + N <= self._start + len(self._buf):
            o = self._next - self._start
            out.append(self._frame(self._buf[o:o + N]))
            self._next += HOP
        cut = self._next - self._start
        if cut > 0:
            self._buf = self._buf[cut:]
            self._start = self._next
        return out

    def skip(self, n: int) -> None:
        """Let n samples go by unheard (the app is behind, see LiveRunner): no frames for windows that touch them, the
        state moves on as through that many frames of silence, and the frames after keep their times."""
        self._total += int(n)
        frames = max(0, -(-(self._total - self._next) // HOP))      # windows starting before the new audio
        for _ in range(frames):
            if self._prev is not None:
                self._prev = (self._prev[0], self._prev[1] + 1)
            self._voiced = False
            if self.K:
                self._consensus(0.0, False)
        self._next += frames * HOP
        self._buf = np.zeros(0)
        self._start = self._total

    def flush(self) -> list[PitchFrame]:
        """Frames whose centre is still inside the audio, using the missing tail as silence. Ends the stream (state is reset)."""
        out = []
        N = self.N
        have = self._buf
        while self._next + N // 2 < self._total:
            o = self._next - self._start
            w = np.zeros(N)
            seg = have[o:o + N]
            w[:len(seg)] = seg
            out.append(self._frame(w))
            self._next += HOP
        folded, switched = self.n_folded, self.n_switch      # statistics survive the end of the stream
        self.reset()
        self.n_folded, self.n_switch = folded, switched
        return out

    # ------------------------------------------------------------------ one frame
    def _frame(self, x: np.ndarray) -> PitchFrame:
        N, tmax, W = self.N, self.tmax, self.W
        t = (self._next + N / 2) / SR
        if self._prev is not None:                               # the last voiced pitch gets one frame older
            self._prev = (self._prev[0], self._prev[1] + 1)
        c = N // 2
        ms = float(np.dot(x[c - 256:c + 256], x[c - 256:c + 256])) / 512
        loud = 10 * math.log10(max(ms, 1e-12))                   # floor -120 dBFS
        cs = np.empty(N + 1)
        cs[0] = 0.0
        np.cumsum(x * x, out=cs[1:])
        if cs[N] < 1e-9 * N:                                     # digital silence: nothing to track
            return self._finish(t, 0.0, 9.0, loud, silent=True)
        A = np.fft.rfft(x[:W], n=N)
        F = np.fft.rfft(x)
        r = np.fft.irfft(np.conj(A) * F, n=N)[:tmax + 1]
        d = (cs[W] + cs[W:W + tmax + 1] - cs[:tmax + 1]) - 2 * r
        d[0] = 0.0
        cm = np.cumsum(d[1:])
        dn = np.ones(tmax + 1)
        dn[1:] = d[1:] * self._ar[1:] / np.maximum(cm, 1e-12)
        tau, ap, released = self._pick(dn)
        if ap < 0.35:
            self._prev = (tau, 1)
        return self._finish(t, SR / tau, ap, loud, released=released)

    def _finish(self, t, hz, ap, loud, silent=False, released=False):
        in_range = self.fmin <= hz <= self.fmax
        loud_ok = loud >= self.gate_db
        th = self.leave if self._voiced else self.enter
        voiced = (not silent) and loud_ok and in_range and ap <= th
        self._voiced = voiced
        if self.K:
            hz = self._consensus(hz, voiced, released)
        if silent or not in_range:
            return PitchFrame(float(t), 0.0, 0.0, float(loud))
        conf = 0.6 - 0.2 * (ap - self.enter) / (self.leave - self.enter)
        conf = min(1.0, max(0.0, conf)) if loud_ok else 0.0
        return PitchFrame(float(t), float(hz), float(conf), float(loud))

    # ------------------------------------------------------------------ pick
    def _local_min(self, dn, lo, hi):
        lo = max(lo, _TMIN)
        hi = min(hi, self.tmax - 1)
        if hi < lo:
            return None, 9.0
        i = lo + int(np.argmin(dn[lo:hi + 1]))
        return i, float(dn[i])

    def _pick(self, dn):
        tmax = self.tmax
        seg = dn[_TMIN - 1:tmax + 1]
        cs = np.nonzero((seg[1:-1] <= seg[:-2]) & (seg[1:-1] < seg[2:]))[0] + _TMIN
        if len(cs) == 0:
            i = _TMIN + int(np.argmin(dn[_TMIN:tmax]))
            return float(i), float(dn[i]), False
        cv = dn[cs]
        below = np.nonzero(cv < self.thr)[0]
        if len(below):
            i = int(cs[below[0]])
        else:
            ok = np.nonzero(cv <= cv.min() + self.fb_margin)[0]
            i = int(cs[ok[0]])
        if self.sub:
            best = i
            for m in (2, 3, 4):
                t2 = i / m
                if t2 < _TMIN:
                    break
                w = max(2, int(round(0.03 * t2)))
                j, vj = self._local_min(dn, int(round(t2)) - w, int(round(t2)) + w)
                if j is not None and vj <= dn[i] + self.sub_abs and vj < 0.6 and j < best:
                    best = j
            i = best
        if self.up:
            t2 = 2 * i
            if t2 < tmax - 2:
                w = max(2, int(round(0.03 * t2)))
                j, vj = self._local_min(dn, t2 - w, t2 + w)
                if j is not None and vj < dn[i] * (1 - self.up_rel) - self.up_abs:
                    i = j
        raw_i = i
        ovr = released = False
        run = 0
        if self.cont and self._prev is not None and self._prev[1] <= self.cont_age:
            pt = self._prev[0]
            if abs(1200 * math.log2(i / pt)) >= self.cont_far:
                w = self.cont_win / 1200
                j, vj = self._local_min(dn, int(math.floor(pt * 2 ** -w)), int(math.ceil(pt * 2 ** w)))
                if j is not None and vj < self.cont_thr:
                    # run = consecutive frames the continuity has overridden raw picks that agree with each other (within 100 cents)
                    if self._raw_run and self._last_raw is not None and abs(1200 * math.log2(raw_i / self._last_raw)) < 100:
                        run = self._raw_run + 1
                    else:
                        run = 1
                    if self.cont_release and run >= self.cont_release:
                        released = True          # the raw picks have agreed long enough: a genuine leap, take it
                        run = 0
                    else:
                        i = j
                        ovr = True
        self._raw_run = run
        self._last_raw = raw_i
        ap = float(dn[i])
        ti = float(i)
        if self.parabolic and 0 < i < tmax:
            a, b, c = dn[i - 1], dn[i], dn[i + 1]
            den = a - 2 * b + c
            if den != 0:
                ti = i + 0.5 * (a - c) / den
        return max(ti, 1.0), ap, released

    # ------------------------------------------------------------------ octave consensus gate
    def _fold_dir(self, d):
        cands = [(o, o / 12.0) for o in self.offsets]
        if self.fifths:
            cands += [(7.0, math.log2(1.5)), (19.0, math.log2(3.0))]
        best = None
        for sgn in (1, -1):
            for o, lg in cands:
                err = abs(d - sgn * o)
                if err <= self.tol and (best is None or err < best[0]):
                    best = (err, sgn, lg)
        return None if best is None else (best[1], best[2])

    def _consensus(self, hz, voiced, released=False):
        """Causal. Returns the pitch to output for this frame. `released` = the continuity rule just accepted a genuine leap."""
        if not voiced or not hz > 0:
            self._gap += 1
            self._streak = 0
            self._streak_dir = 0
            if self._est is None:
                self._agree = 0
                self._last_l = None
            if self._gap > self.gap_frames:
                self._est = None
                self._agree = 0
                self._last_l = None
            return hz
        self._gap = 0
        l = math.log2(hz)
        if self._est is None:
            if self._last_l is not None and abs(12 * (l - self._last_l)) < 1.0:
                self._agree += 1
            else:
                self._agree = 1
            self._last_l = l
            if self._agree >= self.E:
                self._est = l
            return hz
        if released:                      # the continuity rule decided this is a real leap: switch now, no second delay
            if abs(12 * (l - self._est)) > 6:
                self.n_switch += 1
            self._est = l
            self._streak = 0
            self._streak_dir = 0
            return hz
        fd = self._fold_dir(12 * (l - self._est))
        if fd is None:
            self._streak = 0
            self._streak_dir = 0
            self._est = l
            return hz
        sgn, lg = fd
        if sgn == self._streak_dir:
            self._streak += 1
        else:
            self._streak_dir = sgn
            self._streak = 1
        if self._streak >= self.K:
            self._est = l
            self._streak = 0
            self._streak_dir = 0
            self.n_switch += 1
            return hz
        self.n_folded += 1
        out_l = l - sgn * lg
        self._est = out_l
        return 2 ** out_l

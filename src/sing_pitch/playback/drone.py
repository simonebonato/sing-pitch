"""The Drone: a held key note in the headphones. The tone is plain numpy (testable without audio hardware);
DronePlayer is the thin sounddevice output stream around it."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..config import SAMPLE_RATE
from ..music import A4_MIDI, midi_to_hz
from ..reference import Key

PEAK = 0.25                       # headroom: the Drone is a reference, not the loudest thing in the headphones
HARMONICS = (1.0, 0.5, 0.25)      # fundamental plus two soft overtones: audible on small earbuds, still one pitch
ATTACK_S = 0.03
RELEASE_S = 0.06                  # fades avoid clicks on Start and Stop


def drone_midi(pitch_class: int) -> int:
    """The key note in the octave C3-B3 (130-247 Hz): inside a low-to-mid voice's range."""
    return 48 + pitch_class % 12


class Drone:
    """An endless tone, rendered block by block with continuous phase."""

    def __init__(self, pitch_class: int, sample_rate: int = SAMPLE_RATE):
        self._hz = midi_to_hz(drone_midi(pitch_class))
        self._sr = sample_rate
        self._n = 0                 # samples rendered so far: the phase clock
        self._gain = 0.0
        self._target = 1.0
        norm = PEAK / sum(HARMONICS)
        self._weights = [norm * w for w in HARMONICS]

    @property
    def sounding(self) -> bool:
        return self._target > 0 or self._gain > 0

    def release(self) -> None:
        self._target = 0.0

    def render(self, frames: int) -> np.ndarray:
        idx = self._n + np.arange(frames, dtype=np.float64)
        phase = (self._hz * idx / self._sr) % 1.0
        x = sum(w * np.sin(2 * np.pi * (k + 1) * phase) for k, w in enumerate(self._weights))
        steps = np.arange(1, frames + 1)
        if self._target > self._gain:
            gain = np.minimum(self._gain + steps / (ATTACK_S * self._sr), 1.0)
        elif self._target < self._gain:
            gain = np.maximum(self._gain - steps / (RELEASE_S * self._sr), 0.0)
        else:
            gain = np.full(frames, self._gain)
        self._n += frames
        self._gain = float(gain[-1]) if frames else self._gain
        return (x * gain).astype(np.float32)


def _open_output(**kwargs):
    import sounddevice as sd

    from ..audio_backend import stream_options

    return sd.OutputStream(**stream_options(kwargs.get("device")), **kwargs)


class DronePlayer:
    """Plays the Drone of a Key to the chosen output device (None = the system default) until stop()."""

    def __init__(self, stream_factory: Callable[..., object] = _open_output, sample_rate: int = SAMPLE_RATE,
                 device: int | None = None):
        self._factory = stream_factory
        self._device = device
        self._sr = sample_rate
        self._stream = None
        self._drone: Drone | None = None

    @property
    def playing(self) -> bool:
        return self._stream is not None

    def start(self, key: Key) -> None:
        self.stop()
        drone = Drone(key.tonic, self._sr)

        def callback(outdata, frames, time_info, status):
            outdata[:] = drone.render(frames)[:, None]

        kwargs = dict(channels=2, samplerate=self._sr, blocksize=1024, dtype="float32", callback=callback)
        if self._device is not None:
            kwargs["device"] = self._device
        stream = self._factory(**kwargs)
        stream.start()
        self._drone, self._stream = drone, stream

    def stop(self) -> None:
        if self._stream is None:
            return
        stream, drone = self._stream, self._drone
        self._stream = self._drone = None
        drone.release()
        deadline = time.perf_counter() + 2 * RELEASE_S      # let the callback play the fade-out
        while drone.sounding and time.perf_counter() < deadline:
            time.sleep(0.005)
        try:
            stream.stop()
        except Exception:               # as BackingPlayer.stop: a stream that will not stop is aborted and closed
            try:
                stream.abort()
            except Exception:
                pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

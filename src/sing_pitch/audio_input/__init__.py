"""Audio input: device choice (Yeti preferred) and a chunk stream. sounddevice is imported lazily."""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from ..config import BLOCKSIZE, SAMPLE_RATE

PLAUSIBLE_STAMP_S = 1.0   # a chunk's ADC stamp further than this from the clock's now is no stamp at all
VIRTUAL_HINTS = ("blackhole", "background music", "aggregate", "multi-output", "zoom", "teams")


class AudioInput(Protocol):
    sample_rate: int

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None: ...

    def stop(self) -> None: ...


def pick_input(devices: Sequence[dict]) -> int | None:
    """Index of the input to use: the Blue Yeti if present, else the built-in mic, else the first
    non-virtual input. Pure function over sounddevice's device dicts, so it is tested without hardware."""
    inputs = [d for d in devices if d["max_input_channels"] > 0]
    for d in inputs:
        if "yeti" in d["name"].lower():
            return d["index"]
    for d in inputs:
        name = d["name"].lower()
        if "macbook" in name and "microphone" in name:
            return d["index"]
    for d in inputs:
        if not any(h in d["name"].lower() for h in VIRTUAL_HINTS):
            return d["index"]
    return None


def list_inputs() -> list[dict]:
    import sounddevice as sd

    from ..audio_backend import devices_of_host_api

    return [dict(d) for d in devices_of_host_api(sd.query_devices()) if d["max_input_channels"] > 0]


class SoundDeviceInput:
    """Real microphone. The PortAudio callback only copies the chunk out; all work happens elsewhere. `times` is when
    each delivered sample was captured on the host clock (the device's ADC timestamps, take_engine.StreamTimeline),
    keyed by the count of samples delivered since start(): a chunk the device lost shows there, not in the count."""

    def __init__(self, device_index: int, sample_rate: int = SAMPLE_RATE, blocksize: int = BLOCKSIZE):
        from ..take_engine import StreamTimeline

        self.device_index = device_index
        self.sample_rate = sample_rate
        self._blocksize = blocksize
        self._stream = None
        self._latency = 0.0
        self.times = StreamTimeline()

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd

        from ..audio_backend import stream_options
        from ..take_engine import StreamTimeline

        times = self.times = StreamTimeline()
        received = 0
        rate = self.sample_rate

        def callback(indata, frames, time_info, status):
            nonlocal received
            if status and status.input_overflow:
                times.flagged += 1
            adc = time_info.inputBufferAdcTime if time_info is not None else 0.0
            now = times.clock.now()
            if adc > 0 and abs(adc - now) > PLAUSIBLE_STAMP_S:
                # Not a time on the host clock (Windows MME: currentTime 0, adc a few ms into the buffer; measured
                # 2026-11-26): stamp the chunk as it arrives, its last sample captured just now.
                adc = now - frames / rate
            if adc > 0:                    # 0: the driver had no timestamp for this block
                times.add(received / rate, adc, frames)
            received += frames
            on_chunk(indata[:, 0].copy())  # buffer is reused by PortAudio

        self._stream = sd.InputStream(
            device=self.device_index, channels=1, samplerate=self.sample_rate, blocksize=self._blocksize,
            dtype="float32", latency="low", callback=callback, **stream_options(self.device_index),
        )
        self._stream.start()
        self._latency = float(getattr(self._stream, "latency", 0.0) or 0.0)

    @property
    def latency_s(self) -> float:
        """The input latency PortAudio reports (mic -> callback), 0 before start(); kept after stop()."""
        return self._latency

    def stop(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.stop()
            finally:
                stream.close()


class ArrayInput:
    """Test double: plays a numpy array as if it came from a mic, synchronously, on start()."""

    latency_s = 0.0

    def __init__(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE, chunk: int = BLOCKSIZE):
        self._samples = np.asarray(samples, dtype=np.float32)
        self.sample_rate = sample_rate
        self._chunk = chunk

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        for i in range(0, len(self._samples), self._chunk):
            on_chunk(self._samples[i : i + self._chunk])

    def stop(self) -> None:
        pass

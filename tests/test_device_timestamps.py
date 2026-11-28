"""The real device shells record the devices' own timestamps (#49), and calibration reads them like a Take does."""
import sys
import types

import numpy as np
import pytest
from app_speed import APP_SPEED
from qa_helpers import configure_latency

from sing_pitch.playback import BackingPlayer
from sing_pitch.qa import QA
from sing_pitch.settings import Settings
from sing_pitch.take_engine import PORTAUDIO
from sing_pitch.testmode import FAKE_OUTPUT, TimeInfo

SR = 48000


class Flags:
    def __init__(self, underflow=False, overflow=False):
        self.output_underflow, self.input_overflow = underflow, overflow

    def __bool__(self):
        return self.output_underflow or self.input_overflow


class Stream:
    """A sounddevice stream stand-in: the test calls the callback with the time_info a device would give."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def block(self, frames, dac=0.0, adc=0.0, status=None):
        cb = self.kwargs["callback"]
        if "channels" in self.kwargs and self.kwargs["channels"] == 2:
            out = np.zeros((frames, 2), np.float32)
            cb(out, frames, TimeInfo(dac - 0.03, dac=dac), status or Flags())
        else:
            cb(np.zeros((frames, 1), np.float32), frames, TimeInfo(adc + 0.01, adc=adc), status or Flags())


def test_the_backing_stamps_each_block_with_its_dac_time_and_a_late_block_shows():
    streams = []
    player = BackingPlayer(np.zeros((SR * 5, 2), np.float32), SR, stream_factory=lambda **kw: streams.append(
        Stream(**kw)) or streams[-1])
    player.start()
    s, times = streams[0], player.output_times
    assert times.clock is PORTAUDIO
    for k in range(10):
        s.block(1024, dac=100.0 + k * 1024 / SR)
    s.block(1024, dac=100.0 + 10 * 1024 / SR + 0.0213, status=Flags(underflow=True))    # one block late
    for k in range(11, 20):
        s.block(1024, dac=100.0 + k * 1024 / SR + 0.0213)
    assert len(times) == 2 and times.glitches == 1 and times.lost_s == pytest.approx(0.0213)
    assert times.flagged == 1
    assert times.host(15 * 1024 / SR) == pytest.approx(100.0 + 15 * 1024 / SR + 0.0213)


def test_a_block_without_a_timestamp_adds_nothing():
    streams = []
    player = BackingPlayer(np.zeros((SR, 2), np.float32), SR, stream_factory=lambda **kw: streams.append(
        Stream(**kw)) or streams[-1])
    player.start()
    streams[0].block(1024, dac=0.0)
    streams[0].kwargs["callback"](np.zeros((1024, 2), np.float32), 1024, None, None)    # test fakes pass None
    assert len(player.output_times) == 0


def test_the_mic_stamps_each_chunk_with_its_adc_time_keyed_by_samples_delivered(monkeypatch):
    streams = []
    fake_sd = types.SimpleNamespace(InputStream=lambda **kw: streams.append(Stream(**kw)) or streams[-1])
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    from sing_pitch.audio_input import SoundDeviceInput

    mic = SoundDeviceInput(0)
    chunks = []
    mic.start(chunks.append)
    s, chunk = streams[0], 256 / SR
    t0 = PORTAUDIO.now()                 # a real device stamps on the host clock, near its now
    for k in range(5):
        s.block(256, adc=t0 + k * chunk)
    s.block(256, adc=t0 + 7 * chunk, status=Flags(overflow=True))      # chunks 5 and 6 were lost
    s.block(256, adc=t0 + 8 * chunk)
    times = mic.times
    assert len(chunks) == 7 and times.clock is PORTAUDIO
    assert times.glitches == 1 and times.lost_s == pytest.approx(2 * chunk) and times.flagged == 1
    assert times.host(6 * chunk) == pytest.approx(t0 + 8 * chunk)      # the 7th chunk delivered was sung 2 later


def test_a_mic_stamp_that_is_no_host_time_is_replaced_by_the_arrival_time(monkeypatch):
    """Windows MME stamps input chunks a few ms into the buffer instead of on the host clock (measured 2026-11-26:
    currentTime 0, inputBufferAdcTime 0 / 5.8 / 11.6 ms). Such a stamp is far from the clock's now on any system, so
    the chunk is stamped as it arrives instead, its last sample captured just then."""
    streams = []
    fake_sd = types.SimpleNamespace(InputStream=lambda **kw: streams.append(Stream(**kw)) or streams[-1])
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    from sing_pitch.audio_input import SoundDeviceInput

    mic = SoundDeviceInput(0)
    mic.start(lambda chunk: None)
    before = PORTAUDIO.now()
    for k in range(3):
        streams[0].block(256, adc=(k % 3) * 256 / SR)                        # MME: 0 (no stamp), 5.3 ms, 10.7 ms
    after = PORTAUDIO.now()
    times = mic.times
    assert len(times) >= 1                                                 # the 0 adds nothing, the others do
    stamped = times.host(2 * 256 / SR)                                     # the last chunk's first sample
    assert before - 256 / SR <= stamped <= after


@pytest.mark.parametrize("faults", [None, "underflow=1:40"])
def test_calibration_finds_the_latency_even_when_blocks_are_lost_during_the_clicks(tmp_path, faults):
    """The echo mic answers every click 0.12 s after it is played. Lost output blocks delay the clicks that follow
    them; counting frames would add that delay to the latency (about 0.17 s more, here)."""
    configure_latency(tmp_path, 0.12)
    with QA(tmp_path, mic="echo", time_scale=APP_SPEED, faults=faults) as qa:
        qa.click("calibrate_button")
        qa.wait_until(lambda: qa.window._calibration is None, timeout_s=60)
        assert "Latency for" in qa.summary(), qa.summary()
        if faults:
            assert qa.hub.lost_out_steps > 0
    assert Settings.load(tmp_path).latency_for(FAKE_OUTPUT) == pytest.approx(0.12, abs=0.004)

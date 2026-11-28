"""The host clock the app compares device stamps on is time.perf_counter() on every OS (take_engine.clock.HostClock,
docs/cross-platform.md). These checks run in CI on macOS and Windows, so they hold the measurements to account."""
import sys
import time

import pytest

from sing_pitch.take_engine import PORTAUDIO


def test_the_host_clock_is_perf_counter():
    a = time.perf_counter()
    b = PORTAUDIO.now()
    c = time.perf_counter()
    assert a <= b <= c


def test_the_host_clock_resolves_well_below_a_mic_chunk():
    """A 256-frame chunk at 48 kHz is 5.3 ms; Python 3.12's time.monotonic() on Windows ticks every 15.6 ms."""
    assert time.get_clock_info("perf_counter").resolution < 1e-4
    steps = []
    last = PORTAUDIO.now()
    while len(steps) < 50:
        now = PORTAUDIO.now()
        if now != last:
            steps.append(now - last)
            last = now
    assert max(steps) < 1e-3


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS: perf_counter must stay the clock monotonic was")
def test_on_macos_perf_counter_is_the_clock_monotonic_was():
    """Before, the app read time.monotonic() on macOS, measured equal to CoreAudio's stamps (mach_absolute_time)."""
    perf, mono = time.get_clock_info("perf_counter"), time.get_clock_info("monotonic")
    assert perf.implementation == mono.implementation
    assert abs(time.perf_counter() - time.monotonic()) < 1e-3


@pytest.mark.skipif(sys.platform != "win32", reason="Windows: PortAudio stamps on QueryPerformanceCounter")
def test_on_windows_perf_counter_is_query_performance_counter():
    assert time.get_clock_info("perf_counter").implementation == "QueryPerformanceCounter()"

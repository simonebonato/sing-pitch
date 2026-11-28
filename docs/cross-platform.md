# One code path on macOS and Windows

sing-pitch was built and tested on an Apple Silicon Mac. This change makes it run, and its tests pass, on Windows
too, **without a second version of anything**: the code decides from what the machine offers and from whether the
data it gets make sense, not from which OS it is on. Where two systems really differ, the difference lives in one
small file. CI runs the same tests on both (`.github/workflows/tests.yml`), so a change that breaks either one shows
on the pull request.

Everything below was measured on a Windows 11 laptop (Realtek audio, built-in mic) unless it says otherwise. The
macOS numbers are the ones already in the code's docstrings.

## The principle

> Ask the machine, check the data, never branch on `sys.platform`.

| Instead of | The code does |
| --- | --- |
| "on Windows, use WASAPI" | use the best host API the machine has, from one preference list |
| "on Windows, the mic stamps are broken" | a stamp that is not a time on the host clock is replaced, from any driver |
| "on Windows, use `perf_counter`" | use `perf_counter` everywhere: on macOS it *is* the clock used before |
| "on Windows, the data folder is `%APPDATA%`" | ask `platformdirs`, the Python standard for per-OS folders |
| "on Windows, skip the GIL test" | hold the GIL in a way every OS has |

On macOS each rule lands exactly where the old code was, and `tests/test_host_clock.py` and
`tests/test_audio_backend.py` check that.

## The app

### 1. The audio host API: `audio_backend.py` (new)

PortAudio lists every device once per *host API* (the OS's audio system). A Mac has one, Core Audio. Windows has
four (MME, DirectSound, WASAPI, WDM-KS) and starts on **MME**, the oldest one. The Input and Output boxes showed 28
entries for one mic and one pair of headphones, and MME measured badly:

| Windows, same laptop | MME (PortAudio's default) | WASAPI |
| --- | --- | --- |
| output latency (song -> ears) | ~200 ms | **46 ms** |
| input latency | 16 ms reported | 22 ms |
| input timestamps | none: `currentTime` 0, `inputBufferAdcTime` 0 / 5.8 / 11.6 ms | real, on PortAudio's clock |
| entries in the device boxes | 28 (every API, duplicated) | 3 (mic, headphones, speakers) |

**Rule:** `PREFERRED = ("Core Audio", "Windows WASAPI", "ALSA")`. The first one the machine has is used: only its
devices are listed, and its default devices become sounddevice's defaults when it is not already PortAudio's default.
On a Mac that is Core Audio, already the only one and the default, so **nothing changes on macOS**.

The one real difference between the systems is here, and only here. WASAPI shared mode opens a stream only at the
device's own sample rate and channel count, so WASAPI streams get `WasapiSettings(auto_convert=True)`. Then the app's
48 kHz mono input and the song's own rate open on any device, as they do on Core Audio. With nothing else running,
WASAPI with auto_convert delivered 100.0 % of the mic's samples over 8 s (same as native mode and as MME).

Used by `audio_input.list_inputs`, `SoundDeviceInput.start`, `playback.outputs.list_outputs`,
`default_output_name` and `playback.drone._open_output` (the Backing, Drone and calibration streams).

### 2. The host clock: `take_engine/clock.py`

The app compares device stamps (when a block met the DAC, when a chunk left the ADC) with "now", so "now" must be
PortAudio's clock. It read `time.monotonic()`:

| | macOS | Windows |
| --- | --- | --- |
| PortAudio's clock | `mach_absolute_time` | `QueryPerformanceCounter` |
| Python 3.12 `time.monotonic()` | `mach_absolute_time` (Pa_GetStreamTime − monotonic = −0.4 µs) | **`GetTickCount64`, ticks every 15.6 ms** |
| Python 3.12 `time.perf_counter()` | `mach_absolute_time` (the same clock as monotonic) | `QueryPerformanceCounter` (Pa_GetStreamTime − perf_counter = −0.7 µs) |

`HostClock.now()` is now `time.perf_counter()` on every OS. On macOS that is the same clock as before:
`test_on_macos_perf_counter_is_the_clock_monotonic_was` checks it in CI. On Windows the mic stamp jitter went from
~16 ms to ~0.5 ms (median, real mic). The other `time.monotonic()` reads (replay timing, playhead hold, Drone
fade-out, test mode's pacing) moved to `perf_counter` too, so there is one clock in the app.

### 3. Mic stamps that are not times: `audio_input/__init__.py`

**Rule:** a chunk's `inputBufferAdcTime` more than `PLAUSIBLE_STAMP_S` (1 s) from the clock's now is not a time on
that clock. The chunk is then stamped as it arrives (its last sample captured just now). A stamp of 0 still means
"no stamp" and adds nothing, as before.

Before this, on MME the voice was placed ~17 hours from the song: no dots in Melody mode at all (Nearest note worked,
because it does not use stamps). Core Audio's and WASAPI's stamps are on the clock, so the rule never fires there. It
exists for any driver that does what MME does.

### 4. The data folder: `config.data_dir`

`platformdirs.user_data_dir("sing-pitch", appauthor=False, roaming=True)`:
`~/Library/Application Support/sing-pitch` on macOS (where it always was), `%APPDATA%\sing-pitch` on Windows.
`platformdirs` is added to the dependencies. It was already in `uv.lock` (librosa → pooch, for the import extra), so
the lock gains two lines. Relock with **uv 0.9.30**: newer uv versions rewrite the lock to a newer format (revision 5)
and change unrelated markers.

## The tests

The fast suite had 40 failures and 2 errors on Windows (836 passed). None was a bug in the app. They were four kinds:

### Too fast for a slower machine: `tests/app_speed.py` (new)

The window tests run the app 4x faster than the wall clock (`time_scale=4`): each wall second must process 4 s of
audio (Backing, tracker, scoring) plus the redraws. An Apple Silicon Mac keeps up. The Windows laptop did not: a
single `processEvents()` took up to 1.4 s, `qa.wait(0.2)` returned after 2 s, and the sync tests collected 3 samples
where they need 10. **At 1x the same tests pass**: 30+ samples, 0 ms display error, 99 % in tune under every fault.
So the sync logic holds on Windows. The tests only asked for more speed than the machine has.

`APP_SPEED = float(os.environ.get("SING_PITCH_TEST_SPEED", "4"))`, used everywhere a test said `4.0`. Locally
nothing changes (4x); CI sets 1 on both OSes, because GitHub's runners are slower than a desktop Mac.

### Holding the GIL with libc: `testmode.hold_the_gil`

`UiStall` and `test_fake_faults` held the GIL with `ctypes.PyDLL(None).usleep(...)`, the C library's `usleep`, which
Windows does not have (`ctypes.PyDLL(None)` fails). `hold_the_gil` now asks for it: where the C library has `usleep`
(macOS, Linux) it is used as before, so the Mac runs exactly the old test. Without one, a pure-Python wait with the
switch interval raised past it holds the GIL just as well (Windows, measured: 150 ms held, +3 ms late, 10 of 10). A
first version used the pure-Python wait everywhere; on GitHub's macOS runner the GIL test then failed 2 runs of 3.

### No fonts offscreen: `sing_pitch/offscreen.py` (new)

Qt's offscreen platform finds fonts through `QT_QPA_FONTDIR`, which defaults to Qt's own `lib/fonts`. PySide6's
wheels do not ship that folder. On Windows the font list was then **empty** (`QFontDatabase.families() == []`),
text was measured with a stand-in, and the window came out 1298 px wide instead of ~1000. The 6 window-size tests
failed on that. `use_offscreen()` points `QT_QPA_FONTDIR` at the system's font folder, which Qt itself reports
(`QStandardPaths.FontsLocation`). With that, all 37 layout tests pass on Windows. On macOS the offscreen platform
reads fonts through Core Text and ignores the variable. Used by `tests/conftest.py` and the QA driver.

### Fake stamps far from now

`test_the_mic_stamps_each_chunk_with_its_adc_time_keyed_by_samples_delivered` fed stamps around t = 50 s. Under
rule 3 those are "not a time", so the test now stamps near the clock's now. A new test covers MME's stamps.

### One timing bound too tight

`test_qa_driver`: after a seek back, `wait(0.1)` lasted ~0.6 s on the laptop, so the position was past the upper
bound. The check is now "jumped back" (below the position before the seek), like the other `#69` loosenings for a
loaded machine.

## How to check

```
uv sync --locked                       # the tests do not need the import extra
uv run pytest                          # locally at 4x
SING_PITCH_TEST_SPEED=1 uv run pytest  # as CI runs it
```

On real hardware (not in CI): the Input and Output boxes list one entry per device, and `audio_health.csv` shows
`output_latency_s` ~0.04 on Windows.

## Found on the way, not changed here

These need their own pull requests.

- **Fast tracker falls behind on a slower CPU.** With Tracker: Fast on the Windows laptop, the live dots drift
  seconds behind (0.03 s at 5 s, 4.1 s at 30 s; reproduced in test mode with the song's own vocal as the mic). The
  chunk queue grows to hundreds: the worker thread runs the neural tracker *and* Fast (3x more frames) under one GIL
  with the 60 Hz redraw, and the sum exceeds what the machine does in real time. Alone, Fast runs at 40x real time.
  Ideas: a Fast frame every 2 chunks, letting the display tracker skip a backlog instead of queueing it, lighter
  dot drawing. Accurate stays at 0.16 s behind (its look-ahead).
- **"Lost" mic chunks on WASAPI.** `audio_health.csv` reports 50–440 lost chunks per Take, while a direct count shows
  every sample arriving. WASAPI delivers chunks in bursts, and the stamp model reads the bursts as losses. The saved
  Takes show no drift, so scoring is unaffected. The count, and maybe the plot's gaps, may be off.
- **Short dips in the reference melody.** About one every 2–3 s in imported songs, 30–80 ms long, mostly 5–11
  semitones down, with RMVPE's confidence ~0.6 against ~0.9 elsewhere. 85–100 % sit at a phrase start or end.
  Smaller quick dips mid-phrase are visible too and are not measured yet.

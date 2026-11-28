# Tracker: Fast keeps up on a slower machine

With **Tracker: Fast** on a Windows laptop the live dots fell further and further behind the song: 0.7 s after 15 s,
4.1 s after 30 s, and still growing. Accurate on the same machine stayed at its usual ~0.16 s (its look-ahead). This
change keeps Fast's dots next to the voice on such a machine. Fast's own algorithm and parameters are untouched, so
the accuracy measured for it (6.6 % of voiced frames over 50 cents) still holds.

## How it was measured

Test mode, so the numbers do not depend on a singer: the song's own `vocals.wav` is the fake mic, a perfect singer in
time with the song (`QA(home, mic=song / "vocals.wav")`, Reference: Melody, Tracker: Fast). Every 5 s of app time
the script reads the newest dot on the plot and the now-line (`QA.plot_data()`), the worker's queue length
(`LiveRunner._chunks.qsize()`), and how long each part of the work took, as wall time and as CPU time
(`time.perf_counter()` and `time.thread_time()` around each call).

## What was wrong

Before, the worker thread did everything for one mic chunk (256 samples, 5.3 ms), one step after another. Share of
real time each part took with Fast on:

| part | wall time | of which computing | of which waiting for the GIL |
| --- | --- | --- | --- |
| neural tracker (`SwiftF0Tracker.push`) | 38 % | 29 % | 10 % |
| its scoring (`_route`) | 21 % | 8 % | 13 % |
| Fast (`FastTracker.push`) | 30 % | **8 %** | **22 %** |
| the display scoring (`_score_display`) | 1 % | 0 % | 1 % |
| **the worker in all** | **~90 %** | | |

The worker needed ~90 % of real time just to keep pace. Any hiccup (a slow redraw, the OS) pushed it past 100 %, and
from then on the queue only grew: hundreds of chunks, seconds of dots, because a late chunk is still processed in
order. With Accurate alone the worker needs ~57 %, and it keeps up.

Fast itself is cheap. Alone it runs at ~40x real time, and on the worker it computes for only 8 % of real time. Most
of its share was **waiting for the GIL**, held by the window's 60 Hz redraw and the scoring. The neural tracker's
inference (onnxruntime) runs *without* the GIL, so the time the worker spent in it could have been spent on Fast.

## The change

1. **The display tracker gets its own thread** (`LiveRunner._work_display`). The worker hands each chunk to it
   *before* the neural push, so Fast works while onnxruntime runs instead of after it. Fast's frames are scored
   against the Take the worker keeps (song clock, Section, Octave, timing shift), as before. They are only ever
   drawn, never saved.
2. **A backlog is skipped, not queued.** While more than `DISPLAY_BACKLOG_CHUNKS` (8 chunks, ~43 ms) wait for the
   display thread, each chunk is let go by: `FastTracker.skip(n)`. Dots that late are no use, and catching up on
   them would make the next ones late too. `skip` moves Fast's state on as through that much silence (the last
   voiced pitch ages, the voicing gate and octave consensus see unvoiced frames), emits no frame for any window that
   touches the skipped audio, and keeps the next frames on the same 5.33 ms grid, at the same times as without the
   skip. `LiveRunner.display_skipped` counts the samples skipped. A display tracker without `skip` is never skipped.
3. **A failure ends the dots, never the Take.** An exception in the display thread is kept in
   `LiveRunner.display_error` (without its traceback, so no reference cycle holds the runner, as in `_pop_error`),
   and the display thread stops tracking. The saved Take comes from the neural tracker alone, as it always did.

The neural tracker still processes every chunk, in order. The saved Take, its score and History do not change.

## Result

Same measurement, same machine:

| app time | before: dots behind | after: dots behind | after: audio skipped by Fast (cumulative) |
| --- | --- | --- | --- |
| 5 s | 0.03 s | 0.04 s | 0.0 s |
| 15 s | 0.73 s | 0.05 s | 0.3 s |
| 30 s | 4.1 s | 0.09 s | 1.8 s |
| 40 s | still growing | 0.15 s | 3.0 s (7.5 %) |

The saved Take scored 93.6 % in tune with the change, as without it.

Two minutes on the same laptop with nothing else running: the dots 0.03–0.07 s behind at every 15 s check up to
120 s, 0.3 s of audio skipped in all (0.25 %), the saved Take 93.3 %. The 40 s run above shared the machine with
other work. Sung for real on the laptop, two whole songs (~5 min each) with Fast felt in time.

## Considered, not done

- **A Fast frame every 2 hops instead of every hop.** It would halve Fast's work, but its continuity and octave rules
  count in frames (`cont_age`, `cont_release`, `consensus_gap_s` through `HOP`). They were tuned and measured at one
  frame per 256 samples, and a different hop would change the accuracy that was measured.
- **Only the backlog skip, both trackers still on the worker.** Tried first. The lag stayed bounded, but Fast skipped
  ~75 % of the audio (few dots), and the worker still fell behind at 35–40 s. The extra thread was what made the
  difference.
- **Fast in a separate process.** No GIL at all, but audio would have to be copied between processes, and the
  display dots would need the Take's clock from the other process. Not needed now.

## Tests

- `test_fast_tracker.py::test_skip_lets_audio_go_by_and_the_frames_after_keep_their_times`: no frame touches the
  skipped audio, the frames after are at the times an unskipped run gives, and the pitch is right again once a window
  is full.
- `test_display_tracker.py::test_a_display_tracker_that_falls_behind_skips_and_the_saved_take_does_not_change`: a
  display tracker 4x slower than the audio is skipped, still draws dots, and the saved Take scores exactly the same
  frames as without it.
- `test_display_tracker.py::test_a_failing_display_tracker_ends_the_dots_not_the_take`.

On a fast Mac nothing should change: the display thread keeps up, `display_skipped` stays 0, and the dots come
~32 ms after the voice, as before.

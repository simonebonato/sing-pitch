"""How much faster than the wall clock the window tests run the app (testmode.FakeAudioHub time_scale).

At 4x every wall second must process 4 s of audio (Backing, tracker, scoring) plus the redraws; an Apple Silicon Mac
keeps up, a slower machine (a Windows laptop, a CI runner) falls behind and the tests then see too few redraws.
SING_PITCH_TEST_SPEED sets it: CI runs at 1 on every OS (.github/workflows/tests.yml); locally the default stays 4."""
import os

APP_SPEED = float(os.environ.get("SING_PITCH_TEST_SPEED", "4"))

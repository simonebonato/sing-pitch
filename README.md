<p align="center"><img src="docs/logo/logo.svg" width="160" alt="sing-pitch logo: voice dots landing on melody notes"></p>

<h1 align="center">sing-pitch</h1>

<p align="center">Sing along with a song and see, live, how close your voice is to the original singer's melody.</p>

You pick a song, put on headphones and press Start. The song plays, and the app draws the singer's melody as a blue
line and your voice as dots on top of it: green when you are in tune, yellow when you are a little off, red when you
are further off. Every attempt (a **Take**) is scored and saved, so you can watch yourself improve over days and weeks.

![A Take in progress: the voice (dots) on the song's melody (blue bars)](docs/screenshots/take.png)

It runs on a Mac or a Windows PC, and everything stays on your computer: the songs you import, your recordings'
scores and your settings.

## What you need

- **A Mac or a Windows PC.** Tested on an Apple Silicon Mac with macOS 26 and on a Windows 11 laptop. Intel Macs and
  Linux are not tested. On Windows, see [Install on Windows](#install-on-windows).
- **Headphones.** The song must go to your ears only. Through speakers, the microphone hears the song as well as you.
  Wired headphones are best. Bluetooth earbuds work once you calibrate them (see [Calibrate your headphones](#calibrate-your-headphones)).
- **A microphone.** The Mac's built-in one is fine. A Blue Yeti is picked automatically when it is plugged in.
- **About 5 GB of free disk space:** about 1.5 GB for the program, 1.2 GB for the sound models, and about 0.4 GB per
  imported song.

## Install

You type a few commands into the **Terminal** app once. Open it with Spotlight (⌘ Space, type "Terminal", Enter),
then paste each block below and press Enter.

**1. Install Homebrew**, the usual way to install developer tools on a Mac. Skip this if `brew --version` already
prints a version. It asks for your Mac password, then prints two or three "Next steps" commands at the end: run those
too.

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Install the three tools the app needs:** `uv` runs Python programs, `ffmpeg` reads audio files, `git`
downloads this code.

```
brew install uv ffmpeg git
```

**3. Download the app** into a folder called `sing-pitch` in your home folder.

```
cd ~ && git clone https://github.com/simonebonato/sing-pitch.git
```

**4. Install its parts.** This downloads Python and about 1.5 GB of libraries (the sound separation runs on
PyTorch), so it takes a while the first time.

```
cd ~/sing-pitch && uv sync --extra import
```

**5. Start it.**

```
cd ~/sing-pitch && uv run sing-pitch
```

The first time, macOS asks whether Terminal may use the microphone: click **Allow**. If you clicked Don't Allow,
open System Settings → Privacy & Security → Microphone and switch Terminal on.

From now on, step 5 is the only one you need. To get the newest version later:

```
cd ~/sing-pitch && git pull && uv sync --extra import
```

### Install on Windows

Open **PowerShell** (Start menu, type "PowerShell", Enter) and paste each line below. `winget` comes with Windows 11.

```
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e
winget install --id Git.Git -e
```

Close PowerShell and open it again (so it finds the new tools), then:

```
cd ~; git clone https://github.com/simonebonato/sing-pitch.git
cd ~/sing-pitch; uv sync --extra import
cd ~/sing-pitch; uv run sing-pitch
```

The commands in the rest of this page work the same in PowerShell. If Windows asks whether Python may use the
microphone, allow it (Settings → Privacy & security → Microphone). The app uses Windows' WASAPI audio, so each
device appears once in the Input and Output boxes; see [`docs/cross-platform.md`](docs/cross-platform.md).

## Add a song

Songs are added from the Terminal, with the window closed or open. The app separates the singer's voice from the band,
then extracts the melody from the voice. Give it a YouTube link or an audio file on your Mac:

```
cd ~/sing-pitch && uv run sing-pitch import "https://www.youtube.com/watch?v=..."
cd ~/sing-pitch && uv run sing-pitch import "/Users/you/Music/song.mp3"
```

Paste your own link or file path between the quotes (you can drag a file from Finder into the Terminal window to type
its path). The first import also downloads about 1.3 GB of sound models; after that, importing works offline except
for YouTube. Each import takes a while, because the separation runs on the Mac's processor. Importing a song you
already have does nothing; add `--force` to redo it.

If a YouTube link fails, YouTube may have changed something on its side. Download the song's audio some other way and
import the file instead.

Import only songs you are allowed to use. The imported audio stays in the app's folder on your Mac and is never
uploaded anywhere.

## Sing

![Before Start: the melody of the chosen part, with the tolerance band around it](docs/screenshots/preview.png)

1. Choose **Reference: Melody** and your song in the song list. Before you press Start, the plot already shows the
   melody where Start will begin.
2. Press **Start** (Space also pauses and resumes) and sing along. Your voice appears as dots on the left half of the
   plot. The right half shows what is coming, so you can see the next notes before you sing them.
3. Press **Stop**. The line at the bottom says how the Take went: the percentage of the time you were in tune, how
   many cents off you were in the middle of your notes (a cent is a hundredth of a semitone), and whether you tend to
   sing flat or sharp.

![After a Take: the score at the bottom of the window](docs/screenshots/after-take.png)

What the plot shows:

- **Blue line or bars:** the original singer's melody. The faint lines above and below are the same melody an octave
  higher and lower.
- **Soft blue band** around the melody: how far off still counts as in tune. Set its width with **Tolerance** (±10 to
  ±100 cents, ±25 by default).
- **Dots:** your voice. Green inside the Tolerance, yellow up to twice the Tolerance, red beyond. Gray dots are singing
  where the melody has nothing to compare with.
- **Gray stripes:** pauses in the song's singing (an instrumental part or a breath). Nothing is scored there.
- **The strip at the bottom (Timeline):** the whole song. Click to jump there, drag the white line to move through the
  song.

Settings worth knowing:

- **Guide vocal / Backing:** the volume of the original singer and of the band. Turn the singer up to learn a line,
  down to sing it alone.
- **Speed:** 50 % to 120 %, the pitch stays the same. The first time you choose a new Speed for a song the app
  prepares it, which takes a moment; after that it is instant.
- **Octave: Auto** finds whether you sing an octave below or above the singer (a man singing a woman's song, for
  example) and scores you there. Choose a number to fix it.
- **Notes** turns the melody into flat bars on exact notes, which is easier to read. The slider next to it sets the
  shortest note it draws.
- **Tracker** (top row) chooses how your pitch is found for the live dots. **Accurate**, a neural network, draws a dot
  about 0.13 s after you sing. **Fast**, a classic pitch tracker, draws it about 0.03 s after, but more dots land in the
  wrong place (about 7 in 100 on a real voice, more on very low notes). Your saved score always comes from the neural
  network, so History compares like with like whichever you choose.

## Practise one part on repeat

![A Section repeating: Pass 3 running, the last Pass scored 77 %](docs/screenshots/notes-loop.png)

Drag across the Timeline to mark a **Section** (it turns yellow). Tick **Loop** (or press L) and press Start: the
Section repeats, and each repetition (a **Pass**) is scored and saved on its own. Before each Pass you get 2 seconds of
the song before it (**Lead-in**) or a few clicks on the beat (**Count-in**). Type a name and press **Save** to keep the
Section for next time.

## Your progress

![History: % in tune and cents off over time, with every Take below](docs/screenshots/history.png)

**History** shows every Take: the percentage in tune and the cents off over time, with filters for the song, the
Section, the Speed and the other settings, so you compare like with like. Select a Take and press **Replay** to watch
it again on the plot. **Replay last Take** in the main window does the same for the last one.

## Without a song

![Nearest note: a tuner for your voice](docs/screenshots/nearest-note.png)

- **Nearest note** works like a tuner: it shows which note you are singing and how far you are from it.
- **Key** does the same and flags notes outside the key you choose. Tick **Drone** to hear the key's note held in
  your headphones while you sing.

## Calibrate your headphones

Headphones, and Bluetooth earbuds most of all, play the sound slightly late. If the app does not know by how much, it
scores you as late. Choose your headphones under **Output**, press **Calibrate latency**, and clap or sing "ta" exactly
on each of the 8 clicks you hear. The app remembers the result for those headphones. Do it once per pair.

## Keyboard shortcuts

- **Space:** pause and resume
- **← / →:** jump 5 seconds back or forward
- **L:** loop the Section on or off
- **Esc:** remove the Section mark
- **+ / −:** Speed up or down by 5 %

Clicking a text field (the Section name) turns it brown: while it is brown, keys type into the field, and Space types
a space. Press Enter or Esc to give the keys back to the app.

## Where your data lives

Everything is in `~/Library/Application Support/sing-pitch/` (on Windows `%APPDATA%\sing-pitch\`): `songs/`
(imported songs), `takes.csv` and `takes/` (your Takes), `models/` (the sound models) and `settings.json`. To remove
the app completely, delete that folder and the `~/sing-pitch` folder.

## For developers

```
uv sync --extra import
uv run pytest                 # the fast tests: no microphone or speakers needed (they also run without --extra import)
uv run pytest -m slow         # opt-in: the real separator and RMVPE on a generated clip (~1 min)
QT_QPA_PLATFORM=offscreen uv run sing-pitch --smoke    # starts and closes the window without a screen
```

- How it works: [`docs/spec.md`](docs/spec.md) and [`docs/spec-practice-controls.md`](docs/spec-practice-controls.md).
- The words used in the code, issues and commits: [`CONTEXT.md`](CONTEXT.md).
- `SING_PITCH_HOME` sets another data folder, `SING_PITCH_MODELS` another models folder.
- `SING_PITCH_TEST_SPEED` sets how much faster than real time the window tests run the app (default 4; CI uses 1).
- macOS and Windows share one code path: [`docs/cross-platform.md`](docs/cross-platform.md). CI runs the fast tests on
  both (`.github/workflows/tests.yml`).
- `uv run sing-pitch import "/path/to/Logic stems folder"` imports a folder with a Vocals file and other stems
  without separating them; `--model` picks another separator model.
- Pinned: Python 3.12, pyqtgraph 0.14.0, pyside6-essentials 6.12.0.

Melody extraction uses [RMVPE](https://github.com/Dream-High/RMVPE) (through `rmvpe-onnx`), voice separation uses
[audio-separator](https://github.com/nomadkaraoke/python-audio-separator), and the live pitch of your voice comes from
[SwiftF0](https://github.com/lars76/swift-f0).

## License

MIT: see [`LICENSE`](LICENSE). You may use, copy and change the code, as long as the copyright notice stays in.

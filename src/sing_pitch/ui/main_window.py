"""Live view: pitch line on a note grid coloured by Band, current note + cents, Take summary on Stop."""
from __future__ import annotations

from collections import deque
import math
import time
import weakref
from time import perf_counter
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..audio_input import AudioInput, SoundDeviceInput, list_inputs, pick_input
from ..config import SAMPLE_RATE, TOLERANCE_RANGE, TOLERANCE_STEP, ScoreConfig, data_dir
from ..calibration import CalibrationRun
from ..library import Library, PlayerSettings, player_settings
from .. import audio_health
from ..audio_health import dropout_note
from ..live import LiveRunner, Playback
from ..music import NOTE_NAMES, note_name
from ..pitch import PitchTracker
from ..playback import BackingPlayer, DronePlayer
from ..library.tempo import TempoDetector, detect_tempo, load_tempo
from ..playback.transport import LEAD_IN_S, CountIn, Loop
from ..playback.stretch import SpeedCache, Stretcher, time_stretch
from ..playback.outputs import OutputKind, default_output_name, headphone_warning, list_outputs, output_kind
from ..reference import Band, Key, Melody, NearestNote, Reference
from ..reference.melody import NotesMelody
from ..reference.notes import PER_PHRASE, format_length
from ..replay import Replay
from ..settings import DEFAULT_OUTPUT, TRACKERS, Settings, allowed_tolerance
from ..take_engine import (AUTO, FrameScore, LinearClock, NoteTally, PauseFold, SampledClock, SongClock, TakeRecord,
                           pinned, project_song_time, replay_scores)
from ..take_engine.note_scores import scored_per_note
from ..take_store import (last_take_id, load_frames, rescore_take, row_notes, row_settings, saved_clock_times,
                          saved_config, saved_latency, saved_octave, saved_reference, saved_row, saved_song_times)
from .eliding import ElidedLabel, eliding_combo
from .gui_gc import collect_on_the_ui_thread_only
from .song_axis import SongTimeAxis
from .timeline import Timeline
from .speed_prep import SpeedPreparer
from .y_window import YWindow
from .tempo_find import TempoFinder

PAST_S = 4.0          # the plot is centred on now: this much of the voice and the Reference behind the now-line...
FUTURE_S = 4.0        # ...and this much of the Reference ahead of it
TRACKER_TEXT = ["Accurate", "Fast"]   # the Tracker box, in settings.TRACKERS order


def _fast_tracker() -> PitchTracker:
    from ..pitch.fast import FastTracker

    return FastTracker()


TICK_MS = 16          # ~60 Hz redraw asked, ~52 Hz delivered in the real macOS look (2026-11-26): a dot waits ~10 ms for its redraw, not ~17
PLAYHEAD_HOLD_S = 1.0   # after a click on the Timeline the playhead stays on it until the jump is heard, at most this
PLAYHEAD_NEAR_S = 0.05  # ...heard: the song under the now-line within this before the target (or up to 1 s after it)
SPEED_MIN, SPEED_MAX, SPEED_STEP = 50, 120, 5     # Speed in percent of the original
REFERENCE_TEXT = {"nearest": "Nearest note", "key": "Key", "melody": "Melody"}    # reference.kind -> the box's text
SPEED_PRESETS = (70, 85, 100)    # the buttons beside the Speed box
SPEED_DEBOUNCE_MS = 400   # +/- pressed several times: stretch only the Speed he stops on
END_GRACE_S = 1.0     # Melody: keep listening this long after the Backing ends
JUMP_S = 5.0          # the arrow keys jump this far
DRAW_STEP_S = 0.01    # the reference line is drawn at this resolution on the voice clock
HALF_RANGE = 12      # semitones shown above and below the centre (4 s of a song spans up to ~20)
NOTE_LABEL_MIN_PX = 14   # note names on the left axis at least this far apart: on a short plot only every 2nd (3rd...)
SLIDER_MIN_W = 80
SECTION_NAME_MIN_W = 90                          # the Section name field never shrinks to a stub (#53)
# First-launch size: on the macOS style the five control rows take about 270 px, so 620 left the plot about 150 px; the
# minimum size (set by the rows) is not raised for it.
DEFAULT_SIZE = (1000, 800)
ROW_SPACING = 8
JUMPED_MESSAGE_MS = 5000                         # how long "Jumped. Take saved: ..." stays: a new Take is running
PLOT_MIN_H = 200                                 # the plot is never squeezed below this, even at the minimum window size
SCREEN_MARGIN = (0, 60)                          # room the window frame and the menu bar / Dock take from the screen
Y_VOICE_S = 2.0       # the voice's median over this many seconds decides where the vertical range sits
Y_LOOK_S = 6.0        # a move of the vertical range also looks this many seconds beyond the right half
Y_EASE = 0.3          # share of the way to the new centre covered per redraw (about 0.3 s for a move)
Y_EASE_SNAP = 0.05    # semitones: closer than this the range lands on the centre
BAND_COLOUR = {Band.GREEN: "#2ecc71", Band.YELLOW: "#f1c40f", Band.RED: "#e74c3c"}
UNSCORED_COLOUR = "#8a8a8a"   # sung where the Reference has nothing reliable: shown, never scored
BRUSH = {b: pg.mkBrush(c) for b, c in BAND_COLOUR.items()}
BRUSH[None] = pg.mkBrush(UNSCORED_COLOUR)
UNSCORED_SPAN_BRUSH = pg.mkBrush(138, 138, 138, 70)   # Unscored stretch of the Melody, greyed on the plot
TOLERANCE_BAND_ALPHA = ((25, 150), (100, 80))        # fill opacity (0-255) at the Tolerance in cents: a wide band is fainter,
                                                    # so +-100 is a halo and not a ribbon, while +-25 still shows by its fill
REFERENCE_PEN = pg.mkPen("#5dade2", width=2)     # thin: the Tolerance band around it gives the line its body
VERDICT_PEN = {b: pg.mkPen(c, width=2) for b, c in BAND_COLOUR.items()}   # Notes mode: a bar whose note has its verdict
GUIDE_PEN = pg.mkPen(93, 173, 226, 150, width=2)    # the Reference an octave above and below: clear, but fainter and thinner than the blue line
NOTE_LABEL_COLOR = (235, 245, 255)                  # Notes mode: the note name over each bar
VERDICT_ERAS = 4                                    # Notes mode: the Takes (Passes) before whose verdicts are still drawn
NOTE_GAP_SAMPLES = 2                                # Notes mode: blank draw samples at the end of a bar, so bars stay apart
NOTE_LABEL_WIDTH = 270                              # room for "C#4  out of key" at 32 px bold
NOTE_SLIDER_PHRASE = 41                             # the Minimum note length slider: 1..40 tenths of a second, 41 per Phrase


def _default_output() -> tuple[str | None, OutputKind]:
    name = default_output_name()
    return name, output_kind(name) if name else "unknown"

pg.setConfigOptions(antialias=False)


def steadiness_text(s) -> str:
    if s.steadiness_cents is None:
        return "steadiness: no held note"
    return f"steadiness {s.steadiness_cents:.0f} cents"


def describe(record: TakeRecord) -> str:
    s = record.summary
    if s.n_scored == 0:
        return f"Take {record.id}: no voiced sound heard. Saved."
    key = f"{record.key}   |   {s.out_of_key_count} out of key   |   " if record.reference_kind == "key" else ""
    if record.reference_kind == "melody":
        mode = f" ({s.octave_mode})" if s.octave_mode else ""
        notes = f"notes {format_length(record.min_note_s)}   |   " if record.note_mode == "on" else ""
        octave = f"octave shift {s.octave_shift:+d}{mode}"
        if len(s.octave_phrases) > 1:        # Auto and the Phrases did not agree: one number would hide it
            octave = "octave per Phrase: " + ", ".join(f"{k:+d} ×{n}".replace("+0", "0") for k, n in s.octave_phrases)
        key = f"{record.song_id}   |   {notes}{octave}   |   "
    return (
        f"{key}In tune {s.pct_in_tune:.0f}% (±{record.tolerance_cents:.0f} c)   |   "
        f"median {s.median_abs_cents:.0f} cents off   |   "
        f"tends {s.tendency}   |   {steadiness_text(s)}   |   range {s.lowest_note}-{s.highest_note}   |   "
        f"{s.n_scored * 0.016:.1f} s sung   |   saved as {record.id}"
    )


def pass_text(record: TakeRecord) -> str:
    """One Pass's result, shown while the loop goes on."""
    s = record.summary
    if s.n_scored == 0:
        return f"Pass {record.pass_index} of {record.section_name}: nothing sung in the Section."
    return (f"Pass {record.pass_index} of {record.section_name}: In tune {s.pct_in_tune:.0f}% "
            f"(±{record.tolerance_cents:.0f} c)   |   "
            f"median {s.median_abs_cents:.0f} cents off   |   tends {s.tendency}   |   saved as {record.id}")


def length_of_slider(value: int) -> float:
    """The Minimum note length a slider position stands for: tenths of a second, the far end one note per Phrase."""
    return PER_PHRASE if value >= NOTE_SLIDER_PHRASE else value / 10


def slider_of_length(min_note_s: float) -> int:
    return NOTE_SLIDER_PHRASE if not np.isfinite(min_note_s) else min(40, max(1, round(10 * min_note_s)))


def _volume_slider(tip: str) -> QtWidgets.QSlider:
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(100)
    slider.setMaximumWidth(140)
    slider.setToolTip(tip)
    return slider


PAUSED_TEXT = "Paused. Press Space to go on."
_PLAYING_TEXTS = ("Singing...", "Replaying Take")
FIELD_HINT = "Typing in a text field: Space types a space here. Press Enter or Esc, then Space pauses the song."
FIELD_FOCUS_STYLE = "background-color: #5a3d12;"


def _text_field_of(widget) -> QtWidgets.QWidget | None:
    """The text field (a line edit, a number box) this widget is or sits in; None for anything else."""
    while widget is not None:
        if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)):
            parent = widget.parentWidget()
            return parent if isinstance(parent, QtWidgets.QAbstractSpinBox) else widget
        widget = widget.parentWidget()
    return None


class _FocusKeeper(QtCore.QObject):
    """Keeps keyboard focus from being parked in a field that would swallow Space or the arrows (#64). Installed on the
    application, it acts only for events inside its window:
    - a click anywhere else (plot, Timeline, empty area, another control) takes the focus out of a text field first;
    - Esc in a text field takes the focus out of it (and does not also Clear the Section);
    - in a number box Space and Left/Right are not text, so the window's shortcuts get them;
    - a text field gaining or losing focus tells the window (during a Take it says what Space will do).
    Holds the window weakly: it is a child of the window, which must never be kept alive by it."""

    def __init__(self, window: QtWidgets.QWidget):
        super().__init__(window)
        self._window = weakref.ref(window)

    def eventFilter(self, obj, event) -> bool:
        window = self._window()
        if window is None or not isinstance(obj, QtWidgets.QWidget) or obj.window() is not window:
            return False
        kind = event.type()
        if kind == QtCore.QEvent.MouseButtonPress:
            field = _text_field_of(QtWidgets.QApplication.focusWidget())
            if field is not None and _text_field_of(obj) is not field:
                field.clearFocus()
        elif kind in (QtCore.QEvent.ShortcutOverride, QtCore.QEvent.KeyPress):
            field, key = _text_field_of(obj), event.key()
            if field is None:
                return False
            if isinstance(field, QtWidgets.QAbstractSpinBox) and kind == QtCore.QEvent.ShortcutOverride and key in (
                    QtCore.Qt.Key_Space, QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                event.ignore()
                return True                      # the shortcut, not the box, takes it
            if key == QtCore.Qt.Key_Escape:
                if kind == QtCore.QEvent.ShortcutOverride:
                    event.accept()               # not the Clear shortcut: Esc here means "back to the window"
                else:
                    field.clearFocus()
                return True
        elif kind in (QtCore.QEvent.FocusIn, QtCore.QEvent.FocusOut) and obj is _text_field_of(obj):
            window.field_focus_changed(obj, kind == QtCore.QEvent.FocusIn)
        return False


def stop_text(saved: TakeRecord | None, dropped: TakeRecord | None) -> str:
    """What Stop says: the Take that stands for the run, and, when the last bit sung was not saved, what became of it."""
    if dropped is None:
        return describe(saved)
    s = dropped.summary
    what = f"Pass {dropped.pass_index} of {dropped.section_name}" if dropped.pass_index is not None else "The last Take"
    if saved is None:
        return "No voiced sound heard. The Take is not saved." if dropped.pass_index is None else \
            f"{what}: nothing sung, not saved."
    why = "nothing sung" if s.n_scored == 0 else f"only {s.n_scored * 0.016:.1f} s sung"
    last = pass_text(saved) if saved.pass_index is not None else describe(saved)
    return f"{what} not saved: {why}.   |   Last saved: {last}"


class MainWindow(QtWidgets.QWidget):
    def __init__(
        self,
        tracker: PitchTracker,
        home: Path | None = None,
        input_factory: Callable[[int], AudioInput] = SoundDeviceInput,
        drone_factory: Callable[[], DronePlayer] = DronePlayer,
        backing_factory: Callable[[Path, Path], Playback] = BackingPlayer.from_file,
        output_info: Callable[[], tuple[str | None, OutputKind]] = _default_output,
        calibration_player_factory: Callable | None = None,
        stretcher: Stretcher = time_stretch,
        input_devices: Callable[[], list[dict]] = list_inputs,
        output_devices: Callable[[], list[dict]] = list_outputs,
        output_kind_of: Callable[[str], OutputKind] = output_kind,
        tempo_detector: TempoDetector = detect_tempo,
        fast_tracker_factory: Callable[[], PitchTracker] | None = None,
    ):
        super().__init__()
        collect_on_the_ui_thread_only()       # the worker threads below must never finalise a Qt object (gui_gc.py)
        self.setWindowTitle("sing-pitch")
        self.resize(*self._first_size())
        self._tracker, self._home, self._input_factory = tracker, home or data_dir(), input_factory
        self._fast_tracker_factory = fast_tracker_factory or _fast_tracker
        self._fast_tracker: PitchTracker | None = None    # made at the first Fast Take, then kept
        self._drone_factory = drone_factory
        self._drone: DronePlayer | None = None
        self._backing_factory = backing_factory
        self._backing: Playback | None = None
        self._playhead_hold = (0.0, 0.0)               # (song time, perf_counter until): see _playhead_heard
        self._health_ui = audio_health.UiStats()     # tick times and size changes of the running Take
        self._output_info = output_info                  # the system default output: (name, kind)
        self._input_devices, self._output_devices, self._output_kind_of = input_devices, output_devices, output_kind_of
        self._calibration_player_factory = calibration_player_factory
        self._calibration: CalibrationRun | None = None
        self._calibration_output: str | None = None   # the Output the running calibration measures
        self._speed_ready: tuple[str, int, tuple] | None = None   # (song id, percent, stems) loaded for the chosen Speed
        self._speed_job: tuple[int, str, int] | None = None       # (job, song id, percent) being prepared
        self._start_waiting = False      # Start was pressed while the Speed was prepared: the Take starts when it is ready
        self._speed_playing = 100                                 # the percent the player is set to (or will start at)
        home = self._home     # not self in the lambda: window <-> preparer would be a cycle, freed by gc on any thread
        self._preparer = SpeedPreparer(lambda song_id: SpeedCache(Library(home).get(song_id).dir, SAMPLE_RATE,
                                                                  stretcher), self)
        self._preparer.done.connect(self._speed_prepared)
        self._preparer.progress.connect(self._speed_progress)
        self._speed_timer = QtCore.QTimer(self)
        self._speed_timer.setSingleShot(True)
        self._speed_timer.setInterval(SPEED_DEBOUNCE_MS)
        self._speed_timer.timeout.connect(self._prepare_speed)
        self._jumped_text = ""                       # "Jumped. Take saved: ..." as said, until it has been on show a while
        self._jumped_timer = QtCore.QTimer(self)
        self._jumped_timer.setSingleShot(True)
        self._jumped_timer.setInterval(JUMPED_MESSAGE_MS)
        self._jumped_timer.timeout.connect(self._jumped_expired)
        self._tempo_finder = TempoFinder(tempo_detector, self)
        self._tempo_finder.done.connect(self._tempo_found)
        self._detected_bpm: float | None = None      # the chosen song's tempo.json (None = not known yet)
        self._bpm_note = ""                          # what the BPM box did with a typed value, for the tempo label
        self._settings = Settings.load(self._home)
        self._melody: Melody | None = None           # drawn beside the live line during Melody Takes and replays
        self._melody_latency = 0.0
        self._replay_clock: SongClock | None = None  # the replayed Take's own song times, for drawing the reference
        self._unpaused_text = "Singing..."           # what the status line said before the pause text replaced it
        self._speed_hint = ""                        # the "70 % from the next Pass" text on show, if any
        self._hint_under = ""                        # ... and before the text-field hint did
        self._replay_paused_at: float | None = None  # perf_counter time a replay was paused at (None = running)
        self._runner: LiveRunner | None = None
        self._replay: Replay | None = None
        self._visited: dict | None = None            # the controls as they were before a replay put a Take's on them
        self._replay_locked: list[tuple[QtWidgets.QWidget, bool]] = []    # controls a replay greyed, and their state
        self._speed_deferred = False                 # a Speed preparation that came due during a replay
        self._replay_started = 0.0
        self._replay_take_id: str | None = None
        self._t: deque[float] = deque()           # voice time of each drawn frame (the plot shows view time, see _fold)
        self._midi: deque[float] = deque()
        self._band: deque[Band | None] = deque()
        self._out: deque[bool] = deque()          # True = sung note outside the Key
        self._oct: deque[int] = deque()           # the Octave shift each frame was scored at
        self._oct_known = False                   # a scored frame has set it (until then the line sits at 0)
        self._y = YWindow(60, HALF_RANGE)            # where the vertical range is heading; _y_drawn is what shows
        self._y_drawn = 60.0
        self._y_ticks_from: tuple[int, int] | None = None
        self._ahead = np.empty(0)                    # MIDI of the Reference coming in the right half, as drawn
        self._band_line: tuple[np.ndarray, np.ndarray] | None = None   # (x, MIDI) the Tolerance band is drawn around
        self._idle_melody: Melody | None = None      # the Melody the controls ask for (song, Notes): what the preview shows
        self._previewing = False                     # the plot shows the Melody preview (no Take, no replay)
        self._preview_clock: LinearClock | None = None   # the preview's: song time = view time + where Start would begin
        self._idle_octave = 0                        # the Octave shift the preview is drawn at (no frames to say)
        self._keep_plot = False                      # Stop and the end of a replay leave the plot as it is
        self._replay_tolerance: float | None = None  # a replay's: the Tolerance its Take was saved with (else the slider's)
        self._note_tally: NoteTally | None = None   # Notes mode per note: the running Take's (or replay's) notes
        self._verdicts: dict[int, Band] = {}        # ...and the verdict of each note that has ended
        self._verdicts_since = -math.inf            # voice time the running Take (this Pass) began: verdicts apply after
        self._tallied_t: float | None = None        # voice time of the newest frame the note tally has
        self._verdict_eras: list[tuple[float, dict[int, Band]]] = []   # (voice start, verdicts) of the Takes before
        self._pending_cuts: list[float] = []        # Take boundaries (voice time) whose later scores are not in yet
        self._tally_args: tuple[Reference, ScoreConfig] | None = None

        self.device_box = eliding_combo()
        self.output_box = eliding_combo()
        self.output_box.setToolTip("Where you hear the Backing, Guide vocal and Drone. Fixed during a Take.")
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.setToolTip("Look for newly connected inputs and outputs")
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.refresh_devices()
        self.output_box.currentIndexChanged.connect(self._output_changed)       # after the first fill: no save at startup
        self.reference_box = QtWidgets.QComboBox()
        self.reference_box.addItems(["Nearest note", "Key", "Melody"])
        self.key_box = QtWidgets.QComboBox()
        for k in Key.all():
            self.key_box.addItem(k.name, k)
        self.drone_box = QtWidgets.QCheckBox("Drone")
        self.song_box = eliding_combo()
        self.reload_songs()
        self.song_box.currentIndexChanged.connect(self._restore_player_settings)   # first: Speed and Section read them
        self.song_box.currentIndexChanged.connect(self._speed_changed)
        self.guide_box = QtWidgets.QCheckBox("Guide vocal")
        self.guide_slider = _volume_slider("Guide vocal volume")
        self.backing_slider = _volume_slider("Backing volume")
        for w in (self.guide_box, self.guide_slider, self.backing_slider):
            (w.toggled if w is self.guide_box else w.valueChanged).connect(self._apply_gains)
            (w.toggled if w is self.guide_box else w.valueChanged).connect(self._remember_player_settings)
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setToolTip("Space. Left/Right arrows jump 5 s; click the timeline to jump anywhere.")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.timeline = Timeline()
        self.timeline.seek_requested.connect(self.seek)
        for key, action in ((QtCore.Qt.Key_Space, self.toggle_pause), (QtCore.Qt.Key_Left, self.jump_back),
                            (QtCore.Qt.Key_Right, self.jump_forward)):    # bound methods: a lambda keeps self alive
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.WindowShortcut)
            shortcut.setAutoRepeat(key != QtCore.Qt.Key_Space)    # a held Space would flip pause and play again and again
            shortcut.activated.connect(action)
        self.timeline.section_marked.connect(self._section_marked)
        self.timeline.section_changed.connect(self._section_edited)
        self.section_box = eliding_combo(8)
        self.section_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.section_box.setToolTip("Drag across the timeline to mark a Section; drag its edges to adjust it")
        self.section_box.currentIndexChanged.connect(self._section_chosen)
        self.section_name_edit = QtWidgets.QLineEdit()
        self.section_name_edit.setPlaceholderText("Section name")
        self.section_name_edit.setMinimumWidth(SECTION_NAME_MIN_W)    # the others give way, not this (#53)
        self.section_name_edit.setMaximumWidth(140)
        self.section_name_edit.returnPressed.connect(self.section_name_edit.clearFocus)   # keys go back to the window
        self.save_section_button = QtWidgets.QPushButton("Save")
        self.save_section_button.clicked.connect(self.save_section)
        self._pending_overwrite: tuple[str, float, float] | None = None     # a Save waiting for Overwrite / Cancel
        self._section_message: str | None = None     # the last thing said about a Section, until the state it described is gone
        self.overwrite_button = QtWidgets.QPushButton("Overwrite")
        self.overwrite_button.clicked.connect(self.overwrite_section)
        self.cancel_overwrite_button = QtWidgets.QPushButton("Cancel")
        self.cancel_overwrite_button.clicked.connect(self._forget_overwrite)
        self.overwrite_button.setToolTip("Replace the saved Section with this name by the current mark.")
        self.cancel_overwrite_button.setToolTip("Keep the saved Section; change the name to save the mark separately.")
        self.section_name_edit.textChanged.connect(self._forget_overwrite)
        self.section_name_edit.textChanged.connect(self._forget_section_message)
        self.rename_section_button = QtWidgets.QPushButton("Rename")
        self.rename_section_button.clicked.connect(self.rename_section)
        # Overwrite and Cancel take the places of Save and Rename while a Save waits for the answer: the row never grows
        self._save_slot, self._rename_slot = (self._stacked(a, b) for a, b in (
            (self.save_section_button, self.overwrite_button), (self.rename_section_button, self.cancel_overwrite_button)))
        self.delete_section_button = QtWidgets.QPushButton("Delete")
        self.delete_section_button.clicked.connect(self.delete_section)
        self.loop_box = QtWidgets.QCheckBox("Loop (L)")
        self.loop_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.loop_box.setToolTip(f"Repeat the Section; each Pass starts with a {LEAD_IN_S:.0f} s Lead-in or a Count-in "
                                 "(never scored) and is saved as its own Take. A click inside the Section jumps there "
                                 "and keeps looping; a click outside ends the loop")
        self.loop_box.toggled.connect(self._loop_toggled)
        self.between_box = QtWidgets.QComboBox()
        self.between_box.addItems(["Lead-in", "Count-in"])
        self.between_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.between_box.setToolTip(f"Between Passes: Lead-in = {LEAD_IN_S:.0f} s of song before the Section; Count-in = "
                                    "one bar of clicks at the song's tempo (scaled by Speed), the song silent, then "
                                    "the song starts exactly at the Section start")
        self.bpm_box = QtWidgets.QDoubleSpinBox()
        self.bpm_box.setRange(0.0, player_settings.BPM_RANGE[1])
        self.bpm_box.setDecimals(1)
        self.bpm_box.setSpecialValueText("? BPM")          # 0 = tempo unknown
        self.bpm_box.setSuffix(" BPM")
        self.bpm_box.setKeyboardTracking(False)            # one change when he has typed it, not one per digit
        self.bpm_box.setToolTip("The song's tempo for the Count-in, detected at import; type your own if it is wrong "
                                "(30–300, remembered for this song); 0 or empty goes back to the detected tempo")
        self.bpm_box.editingFinished.connect(self.bpm_box.clearFocus)    # keys go back to the window
        self.beats_box = QtWidgets.QSpinBox()
        self.beats_box.setRange(*player_settings.BEATS_PER_BAR_RANGE)
        self.beats_box.setValue(4)
        self.beats_box.setSuffix(" / bar")
        self.beats_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.beats_box.setToolTip("Beats per bar: the Count-in plays this many clicks, the first one accented")
        self.tempo_label = ElidedLabel(takes_the_rest=True)
        self.tempo_label.setStyleSheet("color: #e67e22;")
        for w in (self.between_box, self.bpm_box, self.beats_box):
            (w.currentIndexChanged if w is self.between_box else w.valueChanged).connect(self._between_changed)
        self.loop_label = ElidedLabel()             # "Loop · Pass N · last 92 %" while the Section repeats
        self.loop_label.setStyleSheet("color: #f1c40f; font-weight: bold;")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.setToolTip("Esc. Remove the mark (or the shown Section) from the timeline and stop looping. "
                                     "Saved Sections stay saved: Delete removes them.")
        self.clear_button.clicked.connect(self.clear_section)
        clear_key = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        clear_key.setContext(QtCore.Qt.WindowShortcut)
        clear_key.activated.connect(self.clear_section)               # bound method: a lambda keeps self alive
        self._last_pass_pct: float | None = None
        loop_key = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_L), self)
        loop_key.setContext(QtCore.Qt.WindowShortcut)
        loop_key.activated.connect(self.toggle_loop)                  # bound method: a lambda keeps self alive
        self.speed_box = QtWidgets.QSpinBox()
        self.speed_box.setRange(SPEED_MIN, SPEED_MAX)
        self.speed_box.setSingleStep(SPEED_STEP)
        self.speed_box.setValue(100)
        self.speed_box.setSuffix(" %")
        self.speed_box.setFocusPolicy(QtCore.Qt.NoFocus)         # so + and - stay shortcuts of the window
        self.speed_box.setToolTip("Speed, pitch unchanged (+ and - change it by 5 %)")
        self.speed_label = ElidedLabel()
        self.speed_70_button, self.speed_85_button, self.speed_100_button = (
            self._preset_button(percent) for percent in SPEED_PRESETS)
        self.octave_box = QtWidgets.QComboBox()
        for k in (None, *player_settings.OCTAVE_CHOICES):      # Auto, then the pinned shifts
            self.octave_box.addItem("Auto" if k is None else f"{k:+d}" if k else "0", k)
        self.octave_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.octave_box.setToolTip("Octave shift: Auto picks the octave you sing in for each Phrase; a number pins it "
                                   "for the whole Take. Faint lines show the melody one octave above and below.")
        self.octave_box.currentIndexChanged.connect(self._remember_player_settings)
        self.octave_box.currentIndexChanged.connect(self._show_preview)
        self.notes_box = QtWidgets.QCheckBox("Notes")
        self.notes_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.notes_box.setToolTip("Notes mode: show and score the melody as flat notes on exact semitones")
        self.note_length_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.note_length_slider.setRange(1, NOTE_SLIDER_PHRASE)
        self.note_length_slider.setValue(slider_of_length(player_settings.DEFAULT_MIN_NOTE_S))
        self.note_length_slider.setFixedWidth(120)
        self.note_length_slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.note_length_slider.setToolTip("Minimum note length: melody movements shorter than this merge into the "
                                           "neighbouring note; far right = one note per Phrase")
        self.note_length_label = QtWidgets.QLabel(format_length(player_settings.DEFAULT_MIN_NOTE_S))
        self.note_length_label.setMinimumWidth(70)
        self.tolerance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.tolerance_slider.setRange(*TOLERANCE_RANGE)
        self.tolerance_slider.setSingleStep(TOLERANCE_STEP)
        self.tolerance_slider.setPageStep(TOLERANCE_STEP)
        self.tolerance_slider.setFixedWidth(110)
        self.tolerance_slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tolerance_slider.setToolTip("Tolerance: how far from the target still counts as in tune (green); yellow "
                                         "reaches twice as far. All References; fixed during a Take.")
        self.tolerance_label = QtWidgets.QLabel("")
        self.tolerance_label.setMinimumWidth(48)
        self.tolerance_label.setToolTip("The Tolerance now set, in cents (c)")
        self.tolerance_slider.setValue(self._settings.tolerance_cents)
        self.tolerance_label.setText(f"±{self.tolerance_slider.value()} c")
        self.tolerance_slider.valueChanged.connect(self._tolerance_moved)   # bound method: a lambda keeps self alive
        self.tracker_box = QtWidgets.QComboBox()
        self.tracker_box.addItems(TRACKER_TEXT)
        self.tracker_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tracker_box.setToolTip("How your voice's pitch is found for the live dots. Accurate: a neural network, the "
                                    "dots come ~0.13 s after you sing. Fast: a classic pitch tracker, ~0.03 s, but more "
                                    "dots in the wrong place, mostly on low notes. The saved score always comes from the "
                                    "neural network, so History compares like with like. Fixed during a Take.")
        self.tracker_box.setCurrentIndex(TRACKERS.index(self._settings.tracker))
        self.tracker_box.currentIndexChanged.connect(self._tracker_chosen)   # bound method: a lambda keeps self alive
        self.notes_box.toggled.connect(self._notes_changed)                # bound methods: a lambda keeps self alive
        self.note_length_slider.valueChanged.connect(self._note_length_moved)
        self.note_length_slider.sliderReleased.connect(self._notes_changed)
        self.speed_box.valueChanged.connect(self._speed_changed)
        self.speed_box.valueChanged.connect(self._remember_player_settings)
        self.speed_box.valueChanged.connect(self._sync_speed_presets)
        self.speed_70_button.clicked.connect(self._preset_70)           # bound methods: a lambda keeps self alive
        self.speed_85_button.clicked.connect(self._preset_85)
        self.speed_100_button.clicked.connect(self._preset_100)
        for key, slot in (("+", self.speed_up), ("=", self.speed_up), ("-", self.speed_down)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)    # bound method: a lambda keeps self alive
        self.calibrate_button = QtWidgets.QPushButton("Calibrate latency")
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.output_label = ElidedLabel(takes_the_rest=True, lines=2)
        self.output_label.setStyleSheet("color: #e67e22;")
        self.headphones_button = QtWidgets.QPushButton("These are headphones")
        self.headphones_button.clicked.connect(self.confirm_headphones)
        for w in (self.headphones_button, self.output_label):
            keep_room = w.sizePolicy()
            keep_room.setRetainSizeWhenHidden(True)       # the notice row keeps its height with or without the notice
            w.setSizePolicy(keep_room)
        self.reference_box.currentTextChanged.connect(self._reference_changed)
        self.song_box.currentIndexChanged.connect(self._preview_song)
        self._reference_changed()
        self.button = QtWidgets.QPushButton("Start")
        self.button.clicked.connect(self.toggle)
        self.replay_button = QtWidgets.QPushButton("Replay last Take")
        self.replay_button.clicked.connect(self.toggle_replay)
        self.history_button = QtWidgets.QPushButton("History")
        self.history_button.clicked.connect(self.open_history)
        self.history_window = None
        self.note_label = QtWidgets.QLabel("-")
        self.note_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.note_label.setFixedSize(NOTE_LABEL_WIDTH, 48)
        self.note_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.summary_label = ElidedLabel(self._library_notice or "Press Start and sing.", lines=2)   # any text: same size (#66)

        song_axis = SongTimeAxis()
        self.plot_widget = pg.PlotWidget(background="#1e1e1e", axisItems={"bottom": song_axis})
        song_axis.set_source(self._axis_seconds)
        plot = self.plot_widget.getPlotItem()
        plot.setMouseEnabled(False, False)
        plot.hideButtons()
        plot.showGrid(y=True, alpha=0.35)
        self._unscored = pg.BarGraphItem(x0=[], x1=[], y0=[], y1=[], brush=UNSCORED_SPAN_BRUSH, pen=None)
        self._unscored.setZValue(-5)
        plot.addItem(self._unscored)
        self._tolerance_band = QtWidgets.QGraphicsPathItem()    # one path for the whole band, under the lines and dots
        self._tolerance_band.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self._tolerance_band.setZValue(-4)
        plot.addItem(self._tolerance_band, ignoreBounds=True)
        self._guide_curves = [plot.plot(pen=GUIDE_PEN, connect="finite") for _ in (12, -12)]   # Guide lines
        self._ref_curve = plot.plot(pen=REFERENCE_PEN, connect="finite")
        self._verdict_curves = {b: plot.plot(pen=pen, connect="finite") for b, pen in VERDICT_PEN.items()}
        self._note_labels: list[pg.TextItem] = []      # Notes mode: a pool of note-name labels, reused every frame
        self._now_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(255, 255, 255, 110, width=1))
        self._now_line.setZValue(5)
        plot.addItem(self._now_line)
        self._curve = plot.plot(pen=pg.mkPen("#777777", width=1), connect="finite")
        self._dots = pg.ScatterPlotItem(size=6, pen=None)
        plot.addItem(self._dots)
        self._plot = plot
        plot.getViewBox().sigResized.connect(self._apply_y)     # a taller or shorter plot re-spaces the note names
        self._set_y(60)
        self._set_now(0.0)

        devices = QtWidgets.QHBoxLayout()               # two short rows, so the window can be narrow (#25)
        for w in (QtWidgets.QLabel("Input"), self.device_box, QtWidgets.QLabel("Tracker"), self.tracker_box,
                  QtWidgets.QLabel("Output"), self.output_box,
                  self.refresh_button, self.calibrate_button):
            devices.addWidget(w)
        devices.setStretchFactor(self.device_box, 1)
        devices.setStretchFactor(self.output_box, 1)
        top = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Reference"), self.reference_box, self.key_box, self.drone_box, self.song_box,
                  self.button, self.replay_button, self.history_button):
            top.addWidget(w)
        top.setStretchFactor(self.song_box, 1)
        for slider in (self.guide_slider, self.backing_slider, self.note_length_slider, self.tolerance_slider):
            slider.setMinimumWidth(SLIDER_MIN_W)        # usable with the mouse at the minimum window width (QA #46)
        mix = QtWidgets.QHBoxLayout()
        for w in (self.guide_box, self.guide_slider, QtWidgets.QLabel("Backing"), self.backing_slider,
                  self.pause_button, QtWidgets.QLabel("Speed"), self.speed_box, self.speed_70_button, self.speed_85_button,
                  self.speed_100_button, self.speed_label, self.loop_label):
            mix.addWidget(w)           # the three labels ask for no minimum width: a text never resizes the window (#52)
        mix.addWidget(self.tempo_label, 1)
        mix.setSpacing(10)
        sections = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Section"), self.section_box, self.section_name_edit, self._save_slot,
                  self._rename_slot, self.delete_section_button, self.clear_button, self.loop_box,
                  self.between_box, self.bpm_box, self.beats_box):
            sections.addWidget(w)
        sections.setSpacing(6)
        sections.setStretchFactor(self.section_box, 1)       # a wider window shows more of a Section's name
        sections.addStretch(1)
        warn = QtWidgets.QHBoxLayout()                  # how it is scored: Octave, Notes, Tolerance (QA #46: off the mix row)
        for w in (QtWidgets.QLabel("Octave"), self.octave_box, self.notes_box, self.note_length_slider,
                  self.note_length_label, QtWidgets.QLabel("Tolerance"), self.tolerance_slider, self.tolerance_label):
            warn.addWidget(w)
        warn.addStretch(1)
        warn.addWidget(self.note_label)                 # a fixed width: what is sung never resizes the window (#35)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(devices)
        layout.addLayout(top)
        layout.addLayout(mix)
        layout.addLayout(sections)
        layout.addLayout(warn)
        notice = QtWidgets.QHBoxLayout()                # why this Output may not suit a Melody Take, and its button
        notice.addWidget(self.output_label, 1)
        notice.addWidget(self.headphones_button)
        layout.addLayout(notice)
        layout.addWidget(self.plot_widget, 1)
        layout.addWidget(self.timeline)
        layout.addWidget(self.summary_label)

        self._focus_keeper = _FocusKeeper(self)
        QtWidgets.QApplication.instance().installEventFilter(self._focus_keeper)

        self.plot_widget.setMinimumHeight(PLOT_MIN_H)       # the plot keeps a usable height at the smallest window
        layout.setSpacing(ROW_SPACING)                      # a fixed gap (the style's differs per pair of widgets), so
        shown = not self.timeline.isHidden()               # the Timeline's room is exactly its height plus one gap
        self._plot_min_h, self._timeline_room = PLOT_MIN_H, max(self.timeline.minimumSize().height(), self.timeline.minimumSizeHint().height()) + ROW_SPACING
        self._show_timeline(shown)

        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.timeout.connect(self.tick)
        self._describe_controls()
        self._name_widgets()
        self._restore_last_choice()

    @staticmethod
    def _first_size() -> tuple[int, int]:
        """The first-launch size: DEFAULT_SIZE, clamped to the screen the window opens on (a 1280x720 laptop)."""
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return DEFAULT_SIZE
        room = screen.availableGeometry()
        return (min(DEFAULT_SIZE[0], room.width() - SCREEN_MARGIN[0]),
                min(DEFAULT_SIZE[1], room.height() - SCREEN_MARGIN[1]))

    def _say_jumped(self, text: str) -> None:
        """A jump saved the Take before it; a new Take is running. Said for a few seconds, then the line says what is
        true now: it must not outlive what it describes."""
        self.summary_label.setText(text)
        self._jumped_text = text
        self._jumped_timer.start()

    def _jumped_expired(self) -> None:
        if self._jumped_text and self.summary_label.text() == self._jumped_text:    # not if something else was said
            self.summary_label.setText("Singing..." if self._runner is not None else "")
        self._jumped_text = ""

    def _describe_controls(self) -> None:
        """A tooltip on every control (the ones with a longer story to tell were given theirs where they are built)."""
        tips = {
            self.device_box: "The microphone you sing into",
            self.reference_box: "What your voice is compared with: the nearest note (a tuner), a Key, or a song's Melody",
            self.key_box: "Key mode: notes outside this key are flagged",
            self.drone_box: "Key mode: hold the key's note in your headphones as a Drone",
            self.song_box: "Melody mode: the imported song to sing along with (long titles are shortened here)",
            self.guide_box: "Play the original singer's isolated vocal with the Backing, as a guide",
            self.section_name_edit: "The name to save, or rename, the Section under",
            self.save_section_button: "Save the Section on the timeline under the name typed beside it",
            self.rename_section_button: "Rename the chosen saved Section to the name typed beside it",
            self.delete_section_button: "Delete the chosen saved Section",
            self.calibrate_button: "Measure this Output's delay with clicks you answer, so Melody Takes are scored in sync",
            self.headphones_button: "Tell the app this Output is headphones: the Backing will not leak into the mic",
            self.button: "Start a Take: sing and watch your pitch. Press again to stop, score and save it",
            self.replay_button: "Play the last saved Take back through the plot (press again to stop)",
            self.history_button: "Open the history of your Takes: % in tune and cents off over time",
            self.note_label: "The note you are singing and how far off it is, in cents (c)",
            self.summary_label: "The result of the last Take, or what the app is doing now",
            self.loop_label: "The Section's Pass count and the last Pass's % in tune while it repeats",
            self.speed_label: "Shown while a new Speed is being prepared",
            self.tempo_label: "Why the Count-in cannot play yet (no tempo known for this song)",
            self.note_length_label: "The Minimum note length now set",
            self.output_label: "Why this Output may not suit a Melody Take",
            self.timeline: "The whole song: click to jump, drag to mark a Section, drag the white line to seek",
            self.plot_widget: "Your voice (dots) against the Reference line; the axis is song time",
        }
        for widget, tip in tips.items():
            if not widget.toolTip():
                widget.setToolTip(tip)
        self.device_box.setToolTip(self.device_box.toolTip() + ". Fixed during a Take.")

    def _restore_last_choice(self) -> None:
        """Open on the Reference and song of last time (the song brings back its own Section and Speed), then keep
        remembering them. A song that is gone falls back to the first."""
        if (at := self.song_box.findData(self._settings.last_song)) >= 0 and self._settings.last_song:
            self.song_box.setCurrentIndex(at)
        if (at := self.reference_box.findText(self._settings.last_reference or "")) >= 0:
            self.reference_box.setCurrentIndex(at)
        self.reference_box.currentTextChanged.connect(self._remember_choice)
        self.song_box.currentIndexChanged.connect(self._remember_choice)

    def _remember_choice(self, *_) -> None:
        if self._replay is not None:                 # a replay shows its Take's settings: nothing reacts
            return
        self._settings.last_reference = self.reference_box.currentText()
        if self.song_box.currentData() is not None:             # "no songs" is not a choice
            self._settings.last_song = self.song_box.currentData()
        self._settings.save()

    def _tracker_chosen(self, index: int) -> None:
        self._settings.tracker = TRACKERS[index]
        self._settings.save()

    def _display_tracker(self) -> PitchTracker | None:
        """The tracker the live dots come from when Fast is chosen (None = the neural tracker's own frames)."""
        if TRACKERS[self.tracker_box.currentIndex()] != "fast":
            return None
        if self._fast_tracker is None:
            self._fast_tracker = self._fast_tracker_factory()
        return self._fast_tracker

    def _tolerance_moved(self, value: int) -> None:
        """The Tolerance slider moved: keep it on a 5-cent step, show it, remember it for the next Take and window."""
        cents = allowed_tolerance(value)
        if cents != value:
            self.tolerance_slider.setValue(cents)          # comes back here with the allowed value
            return
        self.tolerance_label.setText(f"±{cents} c")
        self._settings.tolerance_cents = cents
        self._settings.save()
        self._draw_band()
        if not self._previewing:
            self._show_preview()                       # idle: the Tolerance has a Melody to show itself on

    def _score_config(self) -> ScoreConfig:
        """How the next Take is scored: the Tolerance set (and the default Timing window)."""
        return ScoreConfig.with_tolerance(self.tolerance_slider.value())

    def _name_widgets(self) -> None:
        """Stable objectName()s for scripts (sing_pitch.qa): each public widget is named after its attribute."""
        self.setObjectName("main_window")
        for name, value in vars(self).items():
            if isinstance(value, QtWidgets.QWidget) and not name.startswith("_"):
                value.setObjectName(name)

    def refresh_devices(self) -> None:
        """List the inputs and outputs again (a headset connected since the start). Each keeps its selection by name:
        device indices move when devices come and go. The remembered Output stays in settings while it is unplugged."""
        kept_input = self.device_box.currentText()
        self.device_box.blockSignals(True), self.output_box.blockSignals(True)
        try:
            self._fill_inputs(kept_input)
            self._fill_outputs()
        finally:
            self.device_box.blockSignals(False), self.output_box.blockSignals(False)
        if hasattr(self, "output_label"):               # not yet built when the first fill runs
            self.update_output_warning()

    def _fill_inputs(self, kept: str) -> None:
        self.device_box.clear()
        try:
            devices = self._input_devices()
        except Exception as exc:  # PortAudio missing or no devices: the window still opens
            self.device_box.addItem(f"no input ({exc})", None)
            return
        for d in devices:
            self.device_box.addItem(d["name"], d["index"])
        at = self.device_box.findText(kept) if kept else -1
        if at >= 0:
            self.device_box.setCurrentIndex(at)
        elif (best := pick_input(devices)) is not None:
            self.device_box.setCurrentIndex(self.device_box.findData(best))

    def _fill_outputs(self) -> None:
        self.output_box.clear()
        name, _ = self._output_info()
        self.output_box.addItem(f"System default ({name})" if name else "System default", None)
        try:
            devices = self._output_devices()
        except Exception:             # the box then offers the system default only
            return
        for d in devices:
            self.output_box.addItem(d["name"], d["index"])
        at = self.output_box.findText(self._settings.output_device) if self._settings.output_device else -1
        self.output_box.setCurrentIndex(max(at, 0))

    def _output_changed(self, *_) -> None:
        """Simone chose an Output: remember it by name (None = the system default) and judge it for headphones."""
        self._settings.output_device = None if self.output_box.currentData() is None else self.output_box.currentText()
        self._settings.save()
        self.update_output_warning()

    def _current_output(self) -> tuple[str | None, OutputKind]:
        """(name, kind) of the Output that Backing, Guide vocal, Drone and the calibration use."""
        if self.output_box.currentData() is None:
            return self._output_info()
        name = self.output_box.currentText()
        return name, self._output_kind_of(name)

    def _output_kwargs(self) -> dict:
        """device= for a player's stream when a specific Output is chosen; nothing for the system default."""
        index = self.output_box.currentData()
        return {} if index is None else {"device": index}

    def reload_songs(self) -> None:
        self.song_box.clear()
        songs, problems = Library(self._home).scan()
        self._library_notice = f"Skipped a song that cannot be read: {'; '.join(problems)}" if problems else ""
        for song in songs:
            self.song_box.addItem(song.meta.title, song.id)
        if not self.song_box.count():
            self.song_box.addItem("no songs: sing-pitch import <file or link>", None)

    def _reference_changed(self) -> None:
        if self._replay is not None:                 # a replay shows its Take's settings: nothing reacts
            return
        is_key = self.reference_box.currentText() == "Key"
        is_melody = self.reference_box.currentText() == "Melody"
        self.key_box.setEnabled(is_key)
        self.drone_box.setEnabled(is_key)
        self.song_box.setEnabled(is_melody)
        self._forget_section_message()
        self._show_tempo()
        self._restore_player_settings()
        self._preview_song()
        for w in (self.guide_box, self.guide_slider, self.backing_slider, self.speed_box, self.speed_70_button,
                  self.speed_85_button, self.speed_100_button, self.octave_box, self.notes_box, self.note_length_slider,
                  self.section_box, self.section_name_edit, self.clear_button, self.between_box, self.bpm_box,
                  self.beats_box):
            w.setEnabled(is_melody)
        self._sync_section_controls()
        self.update_output_warning()
        self._speed_changed()

    def _show_timeline(self, visible: bool) -> None:
        """Show or hide the Timeline without changing the window's minimum size: hidden, its room goes to the plot's
        minimum height (a window whose minimum changes with the Reference resizes itself, #52)."""
        if not hasattr(self, "_timeline_room"):              # before the layout is built
            self.timeline.setVisible(visible)
            return
        if visible:                    # the plot gives its extra room back first, so the minimum never stands higher
            self.plot_widget.setMinimumHeight(self._plot_min_h)
            self.timeline.setVisible(True)
        else:
            self.timeline.setVisible(False)
            self.plot_widget.setMinimumHeight(self._plot_min_h + self._timeline_room)

    def _preview_song(self, *_) -> None:
        """Idle with Melody chosen: the Timeline shows the chosen song's reference melody. Not while a Take runs."""
        if self._runner is not None or self._replay is not None:
            return
        melody = None
        if self.reference_box.currentText() == "Melody" and self.song_box.currentData() is not None:
            try:
                melody = self._in_notes_mode(Melody.from_song(Library(self._home).get(self.song_box.currentData())))
            except Exception:       # a song that cannot be read: Start says why
                melody = None
        self.timeline.set_melody(melody)
        self._show_timeline(melody is not None)
        self._idle_melody = melody
        self.reload_sections()
        self._show_preview()

    def _show_preview(self, *_) -> None:
        """Idle with Melody chosen: the plot shows the chosen song's reference (line, Guide lines, Unscored stretches,
        Notes bars) around where Start would begin, centred as in a Take, with the Tolerance band on it. No voice, no
        scoring, no timer: it is drawn when a control that shapes it is touched, with _draw_melody and _draw_band.
        Outside Melody a preview is taken down. Stop and the end of a replay leave the plot as it was (_keep_plot)."""
        if self._keep_plot or self._runner is not None or self._replay is not None or getattr(self, "_plot", None) is None:
            return
        melody = self._idle_melody if self.reference_box.currentText() == "Melody" else None
        if melody is None:
            if self._previewing:
                self._show_melody(NearestNote(), 0.0)
            return
        octave_k = self._octave_setting().k
        self._show_melody(melody, 0.0, octave_k)
        span = self.timeline.section
        start = span[0] if span is not None else 0.0
        if span is None:                                # a song that is silent for its first seconds: show where it sings
            t, midi = melody.midi_line()
            sung = np.flatnonzero(np.isfinite(midi))
            if len(sung) and not (np.isfinite(midi[(t >= -PAST_S) & (t <= FUTURE_S)])).any():
                start = max(0.0, float(t[sung[0]]) - 1.0)
        self._preview_clock = LinearClock(-start)
        self._previewing, self._idle_octave = True, octave_k
        self._verdicts, self._verdict_eras, self._note_tally, self._pending_cuts = {}, [], None, []
        self._t.clear(), self._midi.clear(), self._band.clear(), self._out.clear(), self._oct.clear()
        self._oct_known = False
        self._curve.setData([], [])
        self._dots.setData([], [])
        self._set_now(0.0)
        self._draw_melody(0.0)
        axis = self._plot.getAxis("bottom")
        axis.picture = None                             # the x range is the same: the song times on it are not
        axis.update()
        seen = self._band_line[1] if self._band_line is not None else np.empty(0)
        if np.isfinite(seen).any():
            self._set_y(float(np.nanmedian(seen)))      # where the previewed stretch is, shown as the Take would

    def seek(self, song_s: float) -> None:
        """Jump the running Melody Take's song to song_s (a click on the Timeline). A jump closes the Take. Inside a
        looping Section it starts the next Pass there and the loop goes on; anywhere else it ends the loop."""
        if self._backing is None:
            return
        inside = bool(self._backing.seek(song_s))    # the Transport's decision, on its fitted loop: inside, the next
        if not inside:                                 # Pass starts there
            self._set_loop_box(False)
        self.timeline.set_playhead(song_s)
        self._playhead_hold = (float(song_s), time.perf_counter() + PLAYHEAD_HOLD_S)
        self._show_loop()

    # --- Sections and the loop ---------------------------------------------------------------------------------
    def _sections(self):
        """The chosen song's Section store, or None (no Melody song chosen)."""
        song_id = self._speed_song()
        return None if song_id is None else Library(self._home).sections(song_id)

    def reload_sections(self, select: str | None = None) -> None:
        """Fill the picker from the library: the unnamed mark first, then the saved Sections; show the chosen one
        (else the one the song was last left on)."""
        store = self._sections()
        if select is None and (player := self._player_store()) is not None:
            select = player.load().section
        self.section_box.blockSignals(True)
        self.section_box.clear()
        self.section_box.addItem("unsaved mark", None)
        for section in (store.saved() if store else []):
            self.section_box.addItem(section.name, section.name)
        if select is not None:
            self.section_box.setCurrentIndex(max(0, self.section_box.findData(select)))
        self.section_box.blockSignals(False)
        self._section_chosen()

    def _chosen_section(self):
        store = self._sections()
        if store is None:
            return None
        name = self.section_box.currentData()
        try:
            return store.mark if name is None else store.get(name)
        except StopIteration:
            return None

    def _section_chosen(self, *_) -> None:
        section = self._chosen_section()
        self.timeline.set_section(*((None,) if section is None else (section.start_s, section.end_s)))
        self._forget_overwrite()
        # the field shows the chosen Section's name; with "unsaved mark" it is empty, so Save cannot land on the last one
        self.section_name_edit.setText(self.section_box.currentData() or "")
        self._sync_section_controls()
        self._remember_player_settings()
        self._apply_loop()
        self._show_preview()

    def _sync_section_controls(self) -> None:
        """Grey what has nothing to act on: Save needs a Section on the timeline, Rename and Delete a saved one, Loop a
        Section to repeat (a Loop ticked for a Section that is gone is unticked). Greyed buttons need no error text."""
        melody = self.reference_box.currentText() == "Melody"
        shown, saved = self.timeline.section is not None, self.section_box.currentData() is not None
        self.save_section_button.setEnabled(melody and shown)
        self.rename_section_button.setEnabled(melody and saved)
        self.delete_section_button.setEnabled(melody and saved)
        if not shown:
            self._set_loop_box(False)
        self.loop_box.setEnabled(melody and shown)

    def _section_marked(self, start_s: float, end_s: float) -> None:
        """A new drag on the strip: it becomes the unnamed mark (kept until replaced)."""
        store = self._sections()
        if store is None:
            return
        store.set_mark(start_s, end_s)
        self.timeline.set_section(start_s, end_s)
        self._forget_overwrite()
        self._forget_section_message()
        self.section_name_edit.setText("")                 # the name belonged to the previous mark
        self._sync_section_controls()
        if self.section_box.currentIndex() != 0:
            self.section_box.setCurrentIndex(0)            # shows the mark and re-applies the loop
        else:
            self._apply_loop()
            self._show_preview()

    def _section_edited(self, start_s: float, end_s: float) -> None:
        """Edges moved: a saved Section keeps its name and takes the new span, otherwise the mark does."""
        store, name = self._sections(), self.section_box.currentData()
        if store is None:
            return
        self._forget_overwrite()
        if name is None:
            store.set_mark(start_s, end_s)
        else:
            store.save(name, start_s, end_s)
        self._apply_loop()
        self._show_preview()

    def save_section(self) -> None:
        store, span, name = self._sections(), self.timeline.section, self.section_name_edit.text().strip()
        if store is None or span is None:
            self._say_about_section("Mark a Section on the timeline first (drag across it).")
            return
        if not name:
            self._say_about_section("Type a name for the Section to save it.")
            return
        taken = next((s for s in store.saved() if s.name == name), None)
        if taken is not None and self.section_box.currentData() != name:
            self._pending_overwrite = (name, *span)      # another Section has this name: ask, inline, before replacing it
            self._save_slot.layout().setCurrentWidget(self.overwrite_button)
            self._rename_slot.layout().setCurrentWidget(self.cancel_overwrite_button)
            self.summary_label.setText(f"A Section called {name} already exists ({taken.start_s:.1f}-{taken.end_s:.1f} s). "
                                       f"Overwrite it with the new mark ({span[0]:.1f}-{span[1]:.1f} s), or Cancel?")
            return
        self._save_section(store, name, span)

    def _save_section(self, store, name: str, span) -> None:
        store.save(name, *span)
        self._forget_overwrite()
        self.section_name_edit.clearFocus()          # else L, Space and the arrows keep typing into the field
        self.reload_sections(select=name)
        self._say_about_section(f'Saved Section "{name}".')

    def overwrite_section(self) -> None:
        """Overwrite, after Save asked: replace the Section that has this name with the mark that was shown."""
        pending, store = self._pending_overwrite, self._sections()
        self._forget_overwrite()
        if pending is not None and store is not None:
            self._save_section(store, pending[0], pending[1:])

    @staticmethod
    def _stacked(shown: QtWidgets.QWidget, other: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """A slot that holds two buttons in one place (the wider one sets its size): what is shown there can change
        without the row, or the window, changing size (#52)."""
        slot = QtWidgets.QWidget()
        stack = QtWidgets.QStackedLayout(slot)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.addWidget(shown)
        stack.addWidget(other)
        stack.setCurrentWidget(shown)
        return slot

    def _say_about_section(self, text: str) -> None:
        self.summary_label.setText(text)
        self._section_message = text

    def _forget_section_message(self, *_) -> None:
        """A Section message is true only until the mark, the name or the Reference changes: then it goes, unless the
        status line has said something else since."""
        if self._section_message is not None and self.summary_label.text() == self._section_message:
            self.summary_label.setText("")
        self._section_message = None

    def _forget_overwrite(self, *_) -> None:
        """Drop the Overwrite question (Cancel, or the mark, name or picker changed under it)."""
        if self._pending_overwrite is not None:
            self._pending_overwrite = None
            self.summary_label.setText("")
        self._save_slot.layout().setCurrentWidget(self.save_section_button)
        self._rename_slot.layout().setCurrentWidget(self.rename_section_button)

    def rename_section(self) -> None:
        store, old, new = self._sections(), self.section_box.currentData(), self.section_name_edit.text().strip()
        if store is None or old is None:
            self._say_about_section("Pick a saved Section to rename.")
            return
        try:
            store.rename(old, new)
        except ValueError as exc:
            self._say_about_section(f"Cannot rename: {exc}")
            return
        self.section_name_edit.clearFocus()
        self.reload_sections(select=new)
        self._say_about_section(f'Renamed Section "{old}" to "{new}".')

    def delete_section(self) -> None:
        store, name = self._sections(), self.section_box.currentData()
        if store is None or name is None:
            self._say_about_section("Pick a saved Section to delete.")
            return
        store.delete(name)
        self.reload_sections()
        self._say_about_section(f'Deleted Section "{name}".')

    def clear_section(self) -> None:
        """Clear / Esc: no mark and no loop. The unsaved mark is forgotten and nothing is shown on the timeline; a saved
        Section is only taken off the strip (still in the picker, Delete removes it). The song plays on."""
        if not self.clear_button.isEnabled():
            return
        self._set_loop_box(False)
        self._forget_section_message()
        store = self._sections()
        if store is not None:
            store.set_mark(None)
        if self.section_box.currentIndex() != 0:
            self.section_box.setCurrentIndex(0)      # "unsaved mark" = nothing now; shows that and ends the loop
        else:
            self._section_chosen()
        self._apply_loop()
        self._show_loop()

    def toggle_loop(self) -> None:
        if self.loop_box.isEnabled():
            self.loop_box.setChecked(not self.loop_box.isChecked())

    def _loop(self) -> Loop | None:
        """The loop the controls ask for: the shown Section, named when it is a saved one."""
        span = self.timeline.section
        if not self.loop_box.isChecked() or span is None:
            return None
        return Loop(span[0], span[1], self.section_box.currentData(), LEAD_IN_S, self._count_in())

    # --- Between Passes: Lead-in or Count-in --------------------------------------------------------------------------
    def _count_in(self) -> CountIn | None:
        """The Count-in the controls ask for; None = the Lead-in (chosen, or no tempo known yet)."""
        bpm = self.bpm_box.value()
        if self.between_box.currentText() != "Count-in" or bpm < player_settings.BPM_RANGE[0]:
            return None
        return CountIn(bpm, self.beats_box.value())

    def _between_changed(self, *_) -> None:
        if self._replay is not None:                 # a replay shows its Take's settings: nothing reacts
            return
        low, high = player_settings.BPM_RANGE
        bpm = self.bpm_box.value()
        if 0 < bpm < low:                            # a slip while typing: the slowest tempo, and say why
            self._bpm_note = f"BPM is {low:.0f}–{high:.0f}: {bpm:g} became {low:.0f}"
            self.bpm_box.setValue(low)               # comes back here with the allowed value
            return
        if bpm == 0 and self._detected_bpm is not None:      # 0 or empty: back to the detected tempo (QA #48)
            self._set_bpm_box(self._detected_bpm)
            self._bpm_note = f"0 or empty = the detected tempo ({self._detected_bpm:.1f} BPM)"
        self._find_tempo()
        self._remember_player_settings()
        self._show_tempo()
        self._apply_loop()

    def _find_tempo(self) -> None:
        """Count-in chosen for a song with no tempo.json (imported before #42): detect it now, off the UI thread."""
        song_id = self._speed_song()
        if song_id is None or self.between_box.currentText() != "Count-in" or self._detected_bpm is not None:
            return
        self._tempo_finder.find(song_id, Library(self._home).get(song_id).dir)

    def _tempo_found(self, song_id: str, tempo) -> None:
        if song_id != self._speed_song():
            return
        self._detected_bpm = None if tempo is None else tempo.bpm
        if self._detected_bpm is not None and self.bpm_box.value() == 0:
            self._set_bpm_box(self._detected_bpm)
            self._apply_loop()
        self._show_tempo()

    def _show_tempo(self) -> None:
        song_id = self._speed_song()
        note, self._bpm_note = self._bpm_note, ""    # what the BPM box just did with a typed value: said once
        if self.reference_box.currentText() != "Melody":      # the Count-in is Melody's: its note goes with it
            self.tempo_label.setText("")
        elif self.between_box.currentText() != "Count-in" or song_id is None or self.bpm_box.value() > 0:
            self.tempo_label.setText(note if self.between_box.currentText() == "Count-in" else "")
        elif self._tempo_finder.busy(song_id):
            self.tempo_label.setText("finding the tempo…")
        else:
            self.tempo_label.setText("tempo unknown — set BPM (Lead-in until then)")

    def _set_bpm_box(self, bpm: float | None) -> None:
        self.bpm_box.blockSignals(True)
        self.bpm_box.setValue(bpm or 0.0)
        self.bpm_box.blockSignals(False)

    def _own_bpm(self) -> float | None:
        """The BPM to remember for the song: what he typed, None when it is the detected one (or none)."""
        bpm = self.bpm_box.value()
        if bpm < player_settings.BPM_RANGE[0]:
            return None
        if self._detected_bpm is not None and abs(bpm - self._detected_bpm) < 0.05:
            return None
        return bpm

    def _loop_toggled(self, on: bool) -> None:
        if on and self.timeline.section is None:
            self._set_loop_box(False)
            self._say_about_section("Mark a Section on the timeline first (drag across it), then loop it.")
            return
        self._last_pass_pct = None
        self._remember_player_settings()
        self._apply_loop()
        self._show_loop()

    def _apply_loop(self) -> None:
        """The running Take's player loops what the controls show (a changed Section starts again at Pass 1)."""
        if self._backing is not None and hasattr(self._backing, "set_loop"):
            loop = self._loop()
            if loop != getattr(self._backing, "loop", None):
                self._backing.set_loop(loop)
        self._show_loop()

    def _show_loop(self) -> None:
        """Make the loop visible: the Section drawn brighter while it repeats, and "Loop · Pass N" (with the last Pass's
        result) beside the controls. N is the Pass the Transport is playing, so it counts up on every wrap."""
        backing = self._backing
        if backing is not None:
            looping = getattr(backing, "loop", None) is not None
        else:
            looping = self._loop() is not None
        self.timeline.set_looping(looping)
        if not looping:
            self._last_pass_pct = None
            self.loop_label.setText("")
            return
        if backing is None:
            self.loop_label.setText("Loop · on")
            return
        text = f"Loop · Pass {max(1, getattr(backing, 'pass_index', 1))}"
        if self._last_pass_pct is not None:
            text += f" · last {self._last_pass_pct:.0f} %"
        self.loop_label.setText(text)

    def _set_loop_box(self, on: bool) -> None:
        self.loop_box.blockSignals(True)
        self.loop_box.setChecked(on)
        self.loop_box.blockSignals(False)

    def jump_back(self) -> None:
        self.jump_by(-JUMP_S)

    def jump_forward(self) -> None:
        self.jump_by(JUMP_S)

    def jump_by(self, seconds: float) -> None:
        if self._backing is not None:
            pending = getattr(self._backing, "pending_seek_s", lambda: None)()   # a jump not yet applied: start from it
            base = self._backing.position_s() if pending is None else pending
            self.seek(max(0.0, base + seconds))

    @property
    def _paused(self) -> bool:
        """Is the song paused: the player's own answer (the Transport's), or a replay's. The window keeps no flag."""
        if self._backing is not None:
            return bool(getattr(self._backing, "paused", False))     # a player that cannot pause is never paused
        return self._replay_paused_at is not None

    def toggle_pause(self) -> None:
        """Space: pause or resume the song of the running Melody Take, or the replay on show. The Take goes on; the plot
        stands still while the song is paused (singing then is neither drawn nor scored) and goes on from the same
        place on resume. What is paused is read from the player, so Space always flips what is really happening."""
        if self._backing is not None:
            (self._backing.play if self._paused else self._backing.pause)()
        elif self._replay is not None:
            self._toggle_replay_pause()
        else:
            return
        self._show_pause()

    def _show_pause(self) -> None:
        """The Pause button and the status line say what the player is doing (also when something else changed it)."""
        paused = self._paused
        text = "Play" if paused else "Pause"
        if self.pause_button.text() != text:
            self.pause_button.setText(text)
        shown = self.summary_label.text()
        if paused and shown.startswith(_PLAYING_TEXTS):
            self._unpaused_text = shown
            self.summary_label.setText(PAUSED_TEXT)
        elif not paused and shown == PAUSED_TEXT:
            self.summary_label.setText(self._unpaused_text)

    def _set_start_waiting(self, waiting: bool) -> None:
        """A Start waiting for its Speed shows on the button as "Cancel start" (pressing it again cancels), so a slow
        preparation never leaves a Start button that looks idle and is not."""
        self._start_waiting = waiting
        if self._runner is None:
            self.button.setText("Cancel start" if waiting else "Start")

    def _show_speed_hint(self) -> None:
        """In a loop a Speed change applies from the next Pass: until then the box says the new Speed while the old one
        plays, so the label says "70 % from the next Pass"."""
        backing, hint = self._backing, ""
        playing = getattr(backing, "speed_playing", None)
        if playing is not None and getattr(backing, "loop", None) is not None:
            asked = round(100 * backing.speed)
            if asked != round(100 * playing):
                hint = f"{asked} % from the next Pass"
        shown = self.speed_label.text()
        if hint and shown in ("", self._speed_hint):
            self.speed_label.setText(hint)
        elif not hint and shown == self._speed_hint:
            self.speed_label.setText("")
        self._speed_hint = hint

    def _leave_text_fields(self) -> None:
        """Keyboard focus out of any text field: from here on Space and the arrows belong to the song."""
        field = _text_field_of(QtWidgets.QApplication.focusWidget())
        if field is not None and field.window() is self:
            field.clearFocus()

    def field_focus_changed(self, field: QtWidgets.QWidget, focused: bool) -> None:
        """A text field gained or lost keyboard focus. During a Take or replay, say what Space does there (it types)."""
        if focused and (self._runner is not None or self._replay is not None):
            field.setStyleSheet(FIELD_FOCUS_STYLE)
            shown = self.summary_label.text()
            if shown != FIELD_HINT:
                self._hint_under = shown
                self.summary_label.setText(FIELD_HINT)
        elif not focused:
            field.setStyleSheet("")
            if self.summary_label.text() == FIELD_HINT:
                self.summary_label.setText(self._hint_under)

    def _toggle_replay_pause(self) -> None:
        now = time.perf_counter()
        if self._replay_paused_at is None:
            self._replay_paused_at = now
        else:
            self._replay_started += now - self._replay_paused_at     # the replay goes on from where it stood
            self._replay_paused_at = None

    def _replay_elapsed(self) -> float:
        """Seconds of the replayed Take shown so far: standing still while it is paused."""
        return (time.perf_counter() if self._replay_paused_at is None else self._replay_paused_at) - self._replay_started

    def _apply_gains(self, *_) -> None:
        """Slider and checkbox changes reach the player at once, also mid-Take; it ramps them within a block."""
        if self._backing is not None:
            self._backing.set_gains(*self._gains())

    # --- per-song player settings -------------------------------------------------------------------------------
    def _player_store(self):
        """The chosen song's player settings (Melody with a song chosen), else None."""
        song_id = self._speed_song()
        return None if song_id is None else Library(self._home).player_settings(song_id)

    def _remember_player_settings(self, *_) -> None:
        """Keep the controls as the chosen song's settings; the song reopens like this."""
        if self._replay is not None:                 # a replay shows its Take's settings: nothing reacts
            return
        store = self._player_store()
        if store is not None:
            store.save(PlayerSettings(self.guide_box.isChecked(), self.guide_slider.value() / 100,
                                      self.backing_slider.value() / 100, self.speed_box.value(),
                                      self.section_box.currentData(), self.octave_box.currentData(),
                                      self.notes_box.isChecked(), self._note_length(),
                                      between="count-in" if self.between_box.currentText() == "Count-in" else "lead-in",
                                      bpm=self._own_bpm(), beats_per_bar=self.beats_box.value(),
                                      loop=self.loop_box.isChecked()))

    def _restore_player_settings(self, *_) -> None:
        """Put the chosen song's saved Guide vocal, volumes and Speed on the controls (the Section is picked by
        reload_sections). Not while a Take runs."""
        if self._replay is not None:                 # a replay shows its Take's settings: nothing reacts
            return
        store = self._player_store()
        if store is None or self._runner is not None:
            return
        saved = store.load()
        for widget, value in ((self.guide_box, saved.guide_on), (self.guide_slider, round(100 * saved.guide_gain)),
                              (self.backing_slider, round(100 * saved.backing_gain)),
                              (self.speed_box, saved.speed_pct)):
            widget.blockSignals(True)
            (widget.setChecked if widget is self.guide_box else widget.setValue)(value)
            widget.blockSignals(False)
        self.octave_box.blockSignals(True)
        self.octave_box.setCurrentIndex(max(0, self.octave_box.findData(saved.octave)) if saved.octave is not None
                                        else 0)
        self.octave_box.blockSignals(False)
        for widget in (self.notes_box, self.note_length_slider):
            widget.blockSignals(True)
        self.notes_box.setChecked(saved.notes)
        self.note_length_slider.setValue(slider_of_length(saved.min_note_s))
        for widget in (self.notes_box, self.note_length_slider):
            widget.blockSignals(False)
        self.note_length_label.setText(format_length(self._note_length()))
        self._sync_speed_presets()
        tempo = load_tempo(Library(self._home).get(self._speed_song()).dir)
        self._detected_bpm = None if tempo is None else tempo.bpm
        for widget in (self.between_box, self.beats_box):
            widget.blockSignals(True)
        self.between_box.setCurrentText("Count-in" if saved.between == "count-in" else "Lead-in")
        self.beats_box.setValue(saved.beats_per_bar)
        for widget in (self.between_box, self.beats_box):
            widget.blockSignals(False)
        self._set_bpm_box(saved.bpm if saved.bpm is not None else self._detected_bpm)
        self._set_loop_box(saved.loop)               # unticked again by reload_sections when its Section is gone
        self._find_tempo()
        self._show_tempo()

    # --- Notes mode -------------------------------------------------------------------------------------------------
    def _note_length(self) -> float:
        return length_of_slider(self.note_length_slider.value())

    def _in_notes_mode(self, melody: Melody) -> Melody:
        """The Melody as the controls want it: its Notes-mode bars when Notes is ticked."""
        return melody.notes(self._note_length()) if self.notes_box.isChecked() else melody

    def _note_length_moved(self, *_) -> None:
        """The slider moved: the label follows at once; while it is being dragged the song is reloaded only on release."""
        self.note_length_label.setText(format_length(self._note_length()))
        if not self.note_length_slider.isSliderDown():
            self._notes_changed()

    def _notes_changed(self, *_) -> None:
        self.note_length_label.setText(format_length(self._note_length()))
        self._remember_player_settings()
        self._preview_song()

    def _octave_setting(self):
        """Auto, or the box's pinned shift; only a Melody has an original singer to be an octave away from."""
        k = self.octave_box.currentData()
        return AUTO if k is None or self.reference_box.currentText() != "Melody" else pinned(k)

    # --- Speed -------------------------------------------------------------------------------------------------
    def _preset_button(self, percent: int) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(f"{percent} %")
        button.setCheckable(True)
        button.setChecked(percent == 100)
        button.setFocusPolicy(QtCore.Qt.NoFocus)                 # so + and - stay shortcuts of the window
        button.setToolTip(f"Set the Speed to {percent} %: the same as typing it in the box")
        button.setFixedWidth(52)                                  # "100 %" needs 45 on the macOS style (#68)
        return button

    def _preset_70(self) -> None:
        self._press_preset(70)

    def _preset_85(self) -> None:
        self._press_preset(85)

    def _preset_100(self) -> None:
        self._press_preset(100)

    def _press_preset(self, percent: int) -> None:
        self.speed_box.setValue(percent)
        self._sync_speed_presets()                                # a press on the Speed already set must stay pressed

    def _sync_speed_presets(self, *_) -> None:
        """The preset of the Speed now set is shown pressed; none when the box holds some other Speed."""
        for button, percent in zip((self.speed_70_button, self.speed_85_button, self.speed_100_button), SPEED_PRESETS):
            button.setChecked(self.speed_box.value() == percent)

    def speed_up(self) -> None:
        self._step_speed(1)

    def speed_down(self) -> None:
        self._step_speed(-1)

    def _step_speed(self, direction: int) -> None:
        if self.speed_box.isEnabled():
            self.speed_box.setValue(self.speed_box.value() + direction * SPEED_STEP)      # stays within 50-120

    def _speed_song(self) -> str | None:
        return self.song_box.currentData() if self.reference_box.currentText() == "Melody" else None

    def _speed_changed(self, *_) -> None:
        """The chosen Speed or song changed (or the Reference did). 100 % and Speeds whose stems are already loaded
        apply at once; a Speed cached on disk loads in the background; a new one is stretched in the background
        after a short pause (so a run of +/- presses prepares only the Speed he stops on) while the player keeps its
        old Speed. The message says what is going on."""
        if self._replay is not None:                 # a replay shows its Take's Speed: nothing is stretched
            return
        song_id, percent = self._speed_song(), self.speed_box.value()
        self._speed_timer.stop()
        self._speed_job = None
        if song_id is None:
            self.speed_label.setText("")
        elif percent == 100:
            self._speed_ready = None
            self._set_player_speed(100, None)
        elif self._speed_ready and self._speed_ready[:2] == (song_id, percent):
            self._set_player_speed(percent, self._speed_ready[2])
        elif self._preparer.is_cached(song_id, percent):
            self._prepare_speed()
        else:
            self.speed_label.setText(f"preparing {percent} %…")
            self._speed_timer.start()
        self._start_when_ready()

    def _start_when_ready(self, failed: str | None = None) -> None:
        """A Start waiting for its Speed: start now when the chosen Speed is ready (or 100 %), keep waiting (saying for
        which Speed) while it is prepared, give up when it failed or there is no song to start any more."""
        if not self._start_waiting:
            return
        song_id, percent = self._speed_song(), self.speed_box.value()
        if failed is not None or song_id is None or self._replay is not None or self._calibration is not None:
            self._set_start_waiting(False)
            self.summary_label.setText(f"Not started: {failed}" if failed else "Start cancelled.")
        elif percent == 100 or (self._speed_ready and self._speed_ready[:2] == (song_id, percent)):
            self._set_start_waiting(False)
            self._start_or_undo()
        else:
            self.summary_label.setText(f"Starting when {percent} % is ready… (Start again to cancel)")

    def _prepare_speed(self) -> None:
        if self._replay is not None:                 # due during a replay: the box shows the Take's Speed; after it
            self._speed_deferred = True
            return
        song_id, percent = self._speed_song(), self.speed_box.value()
        if song_id is not None and percent != 100:
            self._speed_job = (self._preparer.start(song_id, percent), song_id, percent)
            if not self._preparer.is_cached(song_id, percent):
                self.speed_label.setText(f"preparing {percent} % … 0 %")

    def _speed_progress(self, job: int, fraction: float) -> None:
        """The stretch got this far: "preparing 70 % … 45 %" (Speed, then how much of it is done)."""
        if self._speed_job is not None and self._speed_job[0] == job:
            self.speed_label.setText(f"preparing {self._speed_job[2]} % … {round(100 * fraction)} %")

    def _speed_prepared(self, job: int, stems, error) -> None:
        if self._speed_job is None or self._speed_job[0] != job:
            return                                        # superseded: its result is cached on disk for next time
        _, song_id, percent = self._speed_job
        self._speed_job = None
        if error is not None:
            self.speed_label.setText(f"Speed {percent} % failed: {error}")
            self.speed_box.blockSignals(True)
            self.speed_box.setValue(self._speed_playing)
            self.speed_box.blockSignals(False)
            self._sync_speed_presets()
            self._remember_player_settings()          # the failed Speed is not kept as the song's Speed
            self._start_when_ready(failed=f"Speed {percent} % failed: {error}")
            return
        self._speed_ready = (song_id, percent, stems)
        self._start_when_ready()
        self._set_player_speed(percent, stems)

    def _set_player_speed(self, percent: int, stems) -> None:
        """Hand the player this Speed. The Speed it already plays is not handed again: a speed_changed would split
        the running Take for nothing (during a Take, one song and percent always mean the same stems)."""
        self.speed_label.setText("")
        if self._backing is not None and percent == self._speed_playing:
            return
        self._speed_playing = percent
        if self._backing is not None:
            self._backing.set_speed(percent / 100, *(stems or ()))

    def _stems_for_start(self, song_id: str, percent: int):
        """The stretched stems a Take at `percent` starts with, or None while they are still being prepared."""
        if self._speed_ready and self._speed_ready[:2] == (song_id, percent):
            return self._speed_ready[2]
        if self._preparer.is_cached(song_id, percent):
            stems = self._preparer.load_now(song_id, percent)
            self._speed_ready = (song_id, percent, stems)
            return stems
        if self._speed_job is None and not self._speed_timer.isActive():
            self._speed_changed()
        return None

    def _gains(self) -> tuple[float, float]:
        """(Backing, Guide vocal) linear gains from the controls; the Guide vocal is 0 while its box is unticked."""
        return self.backing_slider.value() / 100, self.guide_slider.value() / 100 if self.guide_box.isChecked() else 0.0

    def update_output_warning(self) -> None:
        """Melody only: warn when the Backing would go to speakers (it leaks into the mic)."""
        text = None
        if self.reference_box.currentText() == "Melody":
            name, kind = self._current_output()
            if not self._settings.headphones_confirmed(name):
                text = headphone_warning(kind, name)
        self.output_label.setText(text or "")
        self.output_label.setVisible(bool(text))
        self.headphones_button.setVisible(bool(text) and "Cannot tell" in (text or ""))

    def confirm_headphones(self) -> None:
        name, _ = self._current_output()
        if name:
            self._settings.confirm_headphones(name)
            self._settings.save()
        self.update_output_warning()

    def _reference(self) -> Reference:
        if self.reference_box.currentText() == "Key":
            return self.key_box.currentData()
        if self.reference_box.currentText() == "Melody":
            song_id = self.song_box.currentData()
            if song_id is None:
                raise ValueError("no song in the library yet: import one with `sing-pitch import <file or link>`")
            return self._in_notes_mode(Melody.from_song(Library(self._home).get(song_id)))
        return NearestNote()

    def _set_now(self, now: float) -> None:
        """Centre the plot on voice time `now`: the now-line in the middle, PAST_S behind it and FUTURE_S ahead."""
        self._now_line.setValue(now)
        self._plot.setXRange(now - PAST_S, now + FUTURE_S, padding=0)

    def _axis_seconds(self, x: np.ndarray) -> np.ndarray:
        """What the plot's x axis says at view times x: the song time heard there (Melody), else the time since Start."""
        return np.asarray(x, dtype=float) if self._melody is None else self._song_times(np.asarray(x, dtype=float))

    @property
    def _centre(self) -> int:
        """Where the vertical range is heading (whole semitones); what is drawn catches up with it, see _ease_y."""
        return self._y.centre

    def _set_y(self, centre: float) -> None:
        """Put the vertical range on `centre` at once (a new Take or replay opens where the voice will be)."""
        self._y.set(centre)
        self._y_drawn = float(self._y.centre)
        self._apply_y()

    def _apply_y(self) -> None:
        lo, hi = self._y_drawn - HALF_RANGE, self._y_drawn + HALF_RANGE
        self._plot.setYRange(lo, hi, padding=0)
        first = math.floor(lo)
        px = self._plot.getViewBox().height() / (2 * HALF_RANGE)          # pixels per semitone
        step = max(1, math.ceil(NOTE_LABEL_MIN_PX / px)) if px > 0 else 1
        if (first, step) != self._y_ticks_from:     # the labels are the same until the view passes a semitone
            self._y_ticks_from = (first, step)
            names = range(first - 1, first + 2 * HALF_RANGE + 3)
            # every semitone keeps its grid line; on a short plot only every step-th has its name (C always among them)
            ticks = [(m, f"{NOTE_NAMES[m % 12]}{m // 12 - 1}" if m % step == 0 else "") for m in names]
            self._plot.getAxis("left").setTicks([ticks])

    def _follow_y(self) -> None:
        """Once per redraw: let the voice (its median over the last seconds) and the coming Reference decide whether the
        vertical range must move (see y_window); the picture then eases to it."""
        voice = np.empty(0)
        if self._t:
            t, midi = np.fromiter(self._t, float), np.fromiter(self._midi, float)
            voice = midi[t >= t[-1] - Y_VOICE_S]
        self._y.follow(voice, self._ahead, self._further_reference)
        self._ease_y()

    def _further_reference(self) -> np.ndarray:
        """MIDI of the Reference in the Y_LOOK_S seconds after the right half, at the Octave shift held (a move of the
        vertical range centres on this too)."""
        if self._melody is None or not self._t:
            return np.empty(0)
        start = self._now_line.value() + FUTURE_S
        x = np.arange(start, start + Y_LOOK_S, DRAW_STEP_S)
        fold = self._fold()
        _, midi, unscored = self._sample_reference(x, fold)
        return (midi + 12.0 * self._octaves_at(x, fold))[~unscored]

    def _ease_y(self) -> None:
        """Move the drawn range a fraction of the way to where it is heading (a pan of a few tenths of a second)."""
        gap = self._y.centre - self._y_drawn
        if gap == 0:
            return
        self._y_drawn = float(self._y.centre) if abs(gap) < Y_EASE_SNAP else self._y_drawn + Y_EASE * gap
        self._apply_y()

    def toggle_replay(self) -> None:
        """Play the most recent saved Take's pitch line back through the live view (press again to stop)."""
        if self._replay is not None:
            self._end_replay()
            return
        take_id = last_take_id(self._home)
        if take_id is None:
            self.summary_label.setText("No saved Take to replay yet.")
            return
        self.replay_take(take_id)

    def replay_take(self, take_id: str) -> None:
        """Play any saved Take back through the live view, with the settings it was sung with on the controls
        (Reference, song, Octave, Notes, Speed). A replay in progress is replaced. The controls go back to what they
        were when it ends, and nothing is remembered or stretched on the way."""
        if self._runner is not None or self._calibration is not None:
            self._say_replay_failed("Stop the Take before replaying another." if self._runner is not None
                                    else "Wait for the calibration to finish before replaying.")
            return
        try:
            reference = saved_reference(self._home, take_id)       # a Melody Take whose song was removed raises here
            latency = saved_latency(self._home, take_id)
            song_times = saved_song_times(self._home, take_id)
            frames = load_frames(self._home, take_id)
            octave = saved_octave(self._home, take_id)
            config = saved_config(self._home, take_id)          # coloured as it was scored, whatever the slider says
            scores = replay_scores(frames, reference, config, song_times, octave)
            replay = Replay(scores)
            octave_k = next((s.octave for s in scores if s.band is not None), 0)      # where the Take was scored
            replay_clock = SampledClock([f.t for f in frames], saved_clock_times(self._home, take_id),   # + Lead-in
                                        getattr(reference, "end_s", None))
            saved = self._saved_settings(take_id, reference, octave)
        except Exception as exc:
            self._say_replay_failed(f"Cannot replay Take {take_id}: {exc}")
            return
        if self._replay is not None:
            self._end_replay()
        self._visited = self._show_settings(saved)
        self._replay_locked = [(w, w.isEnabled()) for w in self._replay_controls()]
        for w, _ in self._replay_locked:          # what the replay shows cannot be changed meanwhile
            w.setEnabled(False)
        self._replay, self._replay_clock, self._replay_take_id = replay, replay_clock, take_id
        self._replay_tolerance = config.green_cents
        self._replay_started, self._replay_paused_at = time.perf_counter(), None
        self.pause_button.setEnabled(True)               # Space pauses a replay too
        self._show_pause()
        self._show_melody(reference, latency, octave_k)
        self._start_note_tally(reference, config)
        self._t.clear(), self._midi.clear(), self._band.clear(), self._out.clear(), self._oct.clear()
        self._oct_known = False
        self._set_now(0.0)
        self.replay_button.setText("Stop replay")
        self.button.setEnabled(False)
        for w in (self.notes_box, self.note_length_slider):       # a replay is drawn as it was scored: they change nothing
            w.setEnabled(False)
        self.summary_label.setText(f"Replaying Take {take_id}...")
        self._timer.start(TICK_MS)
        if self.history_window is not None:
            self.history_window.show_status("")
        self.raise_()

    def _say_replay_failed(self, text: str) -> None:
        self.summary_label.setText(text)
        if self.history_window is not None:
            self.history_window.show_status(text)

    def _saved_settings(self, take_id: str, reference: Reference, octave) -> dict:
        """What a saved Take's table row says the controls were: Reference, song, Key, Octave, Notes, Speed."""
        row = saved_row(self._home, take_id)
        notes, length = row_notes(row)
        return {"reference": REFERENCE_TEXT.get(reference.kind, "Nearest note"), "song": row["song_id"] or None,
                "key": reference if reference.kind == "key" else None,
                "octave": octave.k if octave.mode == "pinned" else None, "notes": notes == "on",
                "length": player_settings.DEFAULT_MIN_NOTE_S if length is None else length, "speed": row_settings(row)[2],
                "tolerance": saved_config(self._home, take_id).green_cents}

    def _replay_controls(self) -> tuple[QtWidgets.QWidget, ...]:
        """The controls a replay puts its Take's settings on, and those that would act on them (Speed presets, the
        Count-in's tempo): greyed while it runs."""
        return (self.reference_box, self.song_box, self.key_box, self.octave_box, self.speed_box,
                self.speed_70_button, self.speed_85_button, self.speed_100_button, self.between_box, self.bpm_box,
                self.beats_box, self.tolerance_slider)

    def _controls_now(self) -> dict:
        return {"reference": self.reference_box.currentText(), "song": self.song_box.currentData(),
                "key": self.key_box.currentData(), "octave": self.octave_box.currentData(),
                "notes": self.notes_box.isChecked(), "length": self._note_length(), "speed": self.speed_box.value(),
                "tolerance": self.tolerance_slider.value()}

    def _show_settings(self, settings: dict) -> dict:
        """Put these settings on the controls without anything reacting to them (nothing remembered, nothing
        stretched, no preview). Returns the settings that were there, to put back."""
        before = self._controls_now()
        boxes = (self.reference_box, self.song_box, self.key_box, self.octave_box, self.notes_box,
                 self.note_length_slider, self.speed_box, self.tolerance_slider)
        for box in boxes:
            box.blockSignals(True)
        self.reference_box.setCurrentText(settings["reference"])
        if settings["song"] is not None and self.song_box.findData(settings["song"]) >= 0:
            self.song_box.setCurrentIndex(self.song_box.findData(settings["song"]))
        if settings["key"] is not None and self.key_box.findText(settings["key"].name) >= 0:     # by name: a saved
            self.key_box.setCurrentIndex(self.key_box.findText(settings["key"].name))        # Key is another object
        self.octave_box.setCurrentIndex(max(0, self.octave_box.findData(settings["octave"])))
        self.notes_box.setChecked(settings["notes"])
        self.note_length_slider.setValue(slider_of_length(settings["length"]))
        self.speed_box.setValue(settings["speed"])
        self.tolerance_slider.setValue(settings["tolerance"])      # a replay shows its Take's Tolerance; nothing is saved
        for box in boxes:
            box.blockSignals(False)
        self.tolerance_label.setText(f"±{self.tolerance_slider.value()} c")
        self.note_length_label.setText(format_length(self._note_length()))
        self._sync_speed_presets()
        return before

    def _take_numbers(self, take_id: str | None) -> str | None:
        """The numbers of a saved Take as Stop showed them (None when they cannot be worked out)."""
        try:
            record = rescore_take(self._home, take_id)
        except Exception:
            return None
        return pass_text(record) if record.pass_index is not None else describe(record)

    def _end_replay(self, finished: bool = False) -> None:
        """The replay is over (it ran out, or Stop replay): the Take's numbers are on the line again, with "Replay
        finished." beside them when it ran out."""
        self._timer.stop()
        numbers = self._take_numbers(self._replay_take_id)
        self._replay = self._replay_clock = self._replay_take_id = self._replay_tolerance = None
        self._replay_paused_at = None
        self.pause_button.setEnabled(False)
        self._show_pause()
        if self._visited is not None:
            self._show_settings(self._visited)                    # the controls as they were before the replay
            self._visited = None
        self._draw_band()                                         # at his Tolerance again (the slider showed the Take's)
        for w, enabled in self._replay_locked:
            w.setEnabled(enabled)
        self._replay_locked = []
        if self._speed_deferred:                                  # the Speed he had chosen, prepared now
            self._speed_deferred = False
            self._speed_changed()
        self.replay_button.setText("Replay last Take")
        self.button.setEnabled(True)
        for w in (self.notes_box, self.note_length_slider):
            w.setEnabled(self.reference_box.currentText() == "Melody")
        self._keep_plot = True
        self._preview_song()                                      # the Timeline as the controls now want it
        self._keep_plot = False
        note = "Replay finished." if finished else "Replay stopped."
        self.summary_label.setText(f"{numbers}   |   {note}" if numbers else note)

    def open_history(self) -> None:
        from .history_view import HistoryView

        if self.history_window is None:
            self.history_window = HistoryView(self._home)
            self.history_window.replay_requested.connect(self.replay_take)       # a bound method: no lambda
            self.history_window.pause_requested.connect(self.toggle_pause)
        else:
            self.history_window.reload()
        self.history_window.show()
        self.history_window.raise_()

    def closeEvent(self, event) -> None:
        """Closing the window is Stop, and more: the Take is ended and saved as Stop does, a replay and a calibration
        end, a Speed being prepared is forgotten, History closes with it (it is a window of its own, so the app would
        otherwise live on with the song playing from a hidden window). The worker thread is stopped here, before any
        Qt object goes. Each step is tried even if one before it fails."""
        for step in (self._close_take, self._close_replay, self._close_calibration, self._close_speed_work,
                     self._close_history):
            try:
                step()
            except Exception:
                pass
        super().closeEvent(event)

    def _close_take(self) -> None:
        if self._runner is not None:
            self._stop()

    def _close_replay(self) -> None:
        if self._replay is not None:
            self._end_replay()

    def _close_calibration(self) -> None:
        if self._calibration is not None:
            run, self._calibration = self._calibration, None
            run.finish()

    def _close_speed_work(self) -> None:
        self._set_start_waiting(False)
        self._speed_timer.stop()
        self._speed_job = None                   # a stretch still running finishes into the cache; nobody listens

    def _close_history(self) -> None:
        if self.history_window is not None:
            self.history_window.close()

    def toggle(self) -> None:
        if self._replay is not None:
            return
        if self._start_waiting:                      # pressed again while waiting for the Speed: no Take after all
            self._set_start_waiting(False)
            self.summary_label.setText("Start cancelled.")
            return
        if self._runner is None:
            self._start_or_undo()
        else:
            self._stop()

    def _start_or_undo(self) -> None:
        """Start; if anything fails once the Backing is already playing, stop it again: a Take that is half set up is
        worse than none (the song would play on with nothing showing it and nothing to pause it)."""
        try:
            self._start()
        except Exception as exc:
            if self._runner is not None:
                self._stop()
            self.summary_label.setText(f"Could not start: {exc}")

    def _show_melody(self, reference: Reference, latency_s: float, octave_k: int = 0) -> None:
        self._previewing, self._preview_clock, self._idle_octave = False, None, 0     # a preview sets these after
        self._melody = reference if reference.kind == "melody" else None
        self._melody_latency = latency_s
        self._ahead = np.empty(0)
        for curve in (self._ref_curve, *self._guide_curves, *self._verdict_curves.values()):
            curve.setData([], [])
        self._show_labels([])
        self._unscored.setOpts(x0=[], x1=[], y0=[], y1=[])
        self._band_line = None
        self._draw_band()
        self.timeline.set_melody(self._melody, getattr(self._backing, "duration_s", None))
        self._show_timeline(self._melody is not None)
        if self._melody is not None:
            _, midi = self._melody.midi_line()
            if np.isfinite(midi).any():
                self._set_y(float(np.nanmedian(midi)) + 12 * octave_k)    # a pinned Octave: look where he sings

    def _start(self) -> None:
        index = self.device_box.currentData()
        try:
            reference = self._reference()
        except Exception as exc:
            self.summary_label.setText(f"Cannot start: {exc}")
            return
        backing, latency, loop_latency = None, 0.0, False
        try:
            if reference.kind == "melody":
                output, _ = self._current_output()
                latency = self._settings.latency_for(output)
                loop_latency = self._settings.is_loop_latency(output)
                percent = self.speed_box.value()
                stems = None if percent == 100 else self._stems_for_start(reference.song_id, percent)
                if percent != 100 and stems is None:          # starts by itself once ready (QA #32)
                    self._set_start_waiting(True)
                    self.summary_label.setText(f"Starting when {percent} % is ready… (Start again to cancel)")
                    return
                song = Library(self._home).get(reference.song_id)
                backing = self._backing_factory(song.backing_path, song.vocals_path, **self._output_kwargs())
                backing.set_gains(*self._gains())
                if self._loop() is not None:                 # the Take starts on the Section's Lead-in, as Pass 1
                    backing.set_loop(self._loop())
                if stems is not None:
                    backing.set_speed(percent / 100, *stems)
                self._speed_playing = percent
            config = self._score_config()
            self._runner = LiveRunner(self._input_factory(index), self._tracker, reference, self._home,
                                      backing=backing, latency_s=latency, octave=self._octave_setting(), config=config,
                                      loop_latency=loop_latency, display_tracker=self._display_tracker())
            self._runner.start()
        except Exception as exc:
            self._runner = None
            self.summary_label.setText(f"Could not open the input or the output: {exc}")
            return
        self._backing = backing
        self._leave_text_fields()
        self._last_pass_pct = None
        self._show_loop()
        self._show_melody(reference, latency, self._octave_setting().k)
        self._start_note_tally(reference, config)
        self.pause_button.setEnabled(backing is not None)
        self._show_pause()
        self._t.clear(), self._midi.clear(), self._band.clear(), self._out.clear(), self._oct.clear()
        self._oct_known = False
        self._set_now(0.0)
        self.button.setText("Stop")
        for w in (self.device_box, self.output_box, self.refresh_button, self.reference_box, self.key_box, self.drone_box,
                  self.song_box, self.octave_box, self.notes_box, self.note_length_slider, self.calibrate_button,
                  self.tolerance_slider, self.tracker_box):
            w.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.summary_label.setText("Singing...")
        if reference.kind == "melody" and not self._settings.is_calibrated(self._current_output()[0]):
            self.summary_label.setText(f"Singing... (latency not calibrated for this output: using {latency:.2f} s)")
        elif reference.kind == "melody" and self._settings.needs_recalibration(self._current_output()[0]):
            self.summary_label.setText("Singing... (this output's latency was calibrated before the sync fix: "
                                       "Calibrate latency once more for exact sync)")
        if reference.kind == "key" and self.drone_box.isChecked():
            try:
                self._drone = self._drone_factory(**self._output_kwargs())
                self._drone.start(reference)
            except Exception as exc:  # the Take goes on without the Drone
                self._drone = None
                self.summary_label.setText(f"Singing... (Drone unavailable: {exc})")
        self._health_ui = audio_health.UiStats()
        self._timer.start(TICK_MS)

    def _stop(self) -> None:
        self._timer.stop()
        if self._drone is not None:
            self._drone.stop()
            self._drone = None
        try:
            record, error = self._runner.stop(), None
            self._backing = None
            self.tick(drain_only=True)
        except Exception as exc:        # the worker thread crashed: the Take is lost, say why
            record, error = None, str(exc)    # the text, not exc: a local holding it is a cycle via its traceback
        self._backing = None
        dropped = getattr(self._runner, "dropped", None)
        dropouts = self._write_health(record, error)
        self._runner = None
        self._show_loop()
        self.pause_button.setEnabled(False)
        self._show_pause()     # reads "Pause" now: no Backing, no replay paused
        self.button.setText("Start")
        self.device_box.setEnabled(True)
        self.output_box.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.reference_box.setEnabled(True)
        self.calibrate_button.setEnabled(True)
        self.tolerance_slider.setEnabled(True)
        self.tracker_box.setEnabled(True)
        self._keep_plot = True                 # the Take just sung stays on the plot until a control is touched
        self._reference_changed()
        self._keep_plot = False
        self.replay_button.setEnabled(True)
        self.summary_label.setText((stop_text(None if record is dropped else record, dropped) if record
                                    else f"The Take failed: {error}") + dropout_note(dropouts))

    def _write_health(self, record, error) -> int:
        """Append the run's audio health record to <home>/audio_health.csv; returns its audio dropouts (lost output
        blocks + lost mic chunks). Never stops a Stop."""
        runner = self._runner
        if runner is None:
            return 0
        try:
            output, _ = self._current_output()
            row = audio_health.row(runner, self._health_ui, input_name=self.device_box.currentText(),
                                   output_name=output or DEFAULT_OUTPUT,
                                   take_ids=[r.id for r in runner.saved] + ([] if record is None or
                                                                           record is runner.dropped else [record.id]),
                                   error=error)
            audio_health.append(self._home, row)
            return int(row["lost_out_blocks"] or 0) + int(row["lost_in_chunks"] or 0)
        except Exception:
            return 0

    def ingest(self, scores: list[FrameScore], dots: list[FrameScore] | None = None) -> None:
        """scores: the Take's own (neural) scores, which the Notes verdicts count; dots: what to draw instead, when a
        display tracker made them (Fast)."""
        fold = self._fold()
        if self._note_tally is not None:
            self._tally(scores)
        if dots is not None:
            scores = dots
        scores = [s for s in scores if not fold.paused(s.t)]     # sung while the song was paused: not drawn
        for s in scores:
            self._t.append(s.t)
            self._midi.append(np.nan if s.midi is None else s.midi)
            self._band.append(s.band)
            self._out.append(s.in_key is False)
            if not self._oct_known and s.band is not None:      # the first scored frame: the Phrase's octave is known,
                self._oct_known = True                           # so the silence before it is drawn at it too
                self._oct = deque(s.octave for _ in self._oct)
            self._oct.append(s.octave)
        if scores:
            end = fold.to_view(self._t[-1])
            while end - fold.to_view(self._t[0]) > PAST_S:
                self._t.popleft(), self._midi.popleft(), self._band.popleft(), self._out.popleft(), self._oct.popleft()
            last = scores[-1]
            if last.band is None:
                self.note_label.setText("-")
            elif last.in_key is False:
                self.note_label.setText(f"{note_name(round(last.midi))}  out of key")
            else:
                self.note_label.setText(f"{last.note_name}  {last.cents:+.0f} c")

    def start_calibration(self) -> None:
        """Clicks in the headphones; clap or sing "ta" on each. The measured delay becomes the default latency
        for the current output device."""
        if self._runner is not None or self._replay is not None or self._calibration is not None:
            return
        output, _ = self._current_output()
        try:
            kwargs = {"player_factory": self._calibration_player_factory} if self._calibration_player_factory else \
                {"output_device": self.output_box.currentData()}
            self._calibration = CalibrationRun(self._input_factory(self.device_box.currentData()), **kwargs)
            self._calibration.start()
        except Exception as exc:
            self._calibration = None
            self.summary_label.setText(f"Calibration could not start: {exc}")
            return
        self._calibration_output = output                # what is measured is what gets the result, whatever happens
        for w in (self.button, self.replay_button, self.calibrate_button, self.device_box, self.output_box,
                  self.refresh_button):
            w.setEnabled(False)
        self.summary_label.setText("Calibrating: clap or sing 'ta' exactly on each click you hear (8 clicks)...")
        QtCore.QTimer.singleShot(int(1000 * self._calibration.duration_s) + 300, self.finish_calibration)

    def finish_calibration(self) -> None:
        if self._calibration is None:
            return
        run, self._calibration = self._calibration, None
        result = run.finish()
        for w in (self.button, self.replay_button, self.calibrate_button, self.device_box, self.output_box,
                  self.refresh_button):
            w.setEnabled(True)
        output = self._calibration_output
        if result.latency_s is None:
            self.summary_label.setText(f"Calibration failed: heard {result.matched} of 8 answers. Try again, louder.")
            return
        self._settings.set_latency(output, result.latency_s, on_clocks=result.on_clocks)
        self._settings.save()
        self.summary_label.setText(
            f"Latency for {output or DEFAULT_OUTPUT}: {1000 * result.latency_s:.0f} ms "
            f"({result.matched}/8 answers, spread {1000 * result.spread_s:.0f} ms). Saved as the default.")

    def _start_note_tally(self, reference: Reference, config: ScoreConfig) -> None:
        """Notes mode scored per note: tally the Take's (or replay's) frames by note, for each note's verdict."""
        self._verdicts, self._verdicts_since, self._verdict_eras, self._pending_cuts = {}, -math.inf, [], []
        self._tally_args = (reference, config)
        self._tallied_t: float | None = None        # voice time of the newest frame the tally has
        self._note_tally = NoteTally(reference, config) if scored_per_note(reference, config) else None

    def _tally(self, scores: list[FrameScore]) -> None:
        """Add scores to the note tally; at each Take boundary (a loop wrap, a jump) the Take before gets its last
        verdicts and the next one starts afresh, so a Pass is judged on its own singing only."""
        i = 0
        while self._pending_cuts:
            at = self._pending_cuts[0]
            j = next((k for k in range(i, len(scores)) if scores[k].t >= at), None)
            if j is None:
                break
            self._note_tally.add(scores[i:j])
            i = j
            self._pending_cuts.pop(0)
            self._new_note_era(at)
        self._note_tally.add(scores[i:])
        if scores:
            self._tallied_t = scores[-1].t

    def _new_note_era(self, at: float) -> None:
        """A Take boundary at voice time `at`: every note of the Take before is judged now (one that ends at the
        Section end too, whose Timing window the song never plays), and kept for drawing that Take's bars."""
        self._verdicts.update((v.note, v.band) for v in self._note_tally.close_until(math.inf))
        self._verdict_eras = [*self._verdict_eras, (self._verdicts_since, self._verdicts)][-VERDICT_ERAS:]
        self._verdicts, self._verdicts_since = {}, at
        self._note_tally = NoteTally(*self._tally_args)

    def _close_notes(self, fold: PauseFold) -> None:
        """Give each note the voice has passed (its end plus the Timing window, at the song time of the newest frame
        tallied, not of the song heard now: the tracker's frames lag that) its verdict colour, final (QA #45)."""
        if self._note_tally is None or self._tallied_t is None:
            return
        song_now = self._song_times(fold.to_view(np.array([self._tallied_t])), fold)[0]
        if np.isfinite(song_now):
            self._verdicts.update((v.note, v.band) for v in self._note_tally.close_until(float(song_now)))

    def _draw_clock(self) -> SongClock:
        """Voice time -> song time for drawing: the running Take's clock (it knows every jump and pause), the replayed
        Take's own song times, else song time + latency."""
        if self._runner is not None and self._runner.song_clock is not None:
            return self._runner.draw_clock               # on the plot's axis (LiveRunner.drain, #63)
        return self._replay_clock or self._preview_clock or LinearClock(self._melody_latency)

    def _fold(self, clock: SongClock | None = None) -> PauseFold:
        """Voice time <-> the plot's view time: the draw clock's pauses folded away, so the plot stands still while the
        song is paused and resumes where it stopped (no pauses: the two are the same)."""
        spans = getattr(clock or self._draw_clock(), "paused_spans", None)
        return PauseFold(spans() if spans is not None else ())

    def _now_voice(self) -> float:
        """The voice time of the song heard now, where the now-line goes: the running Take's heard_t (the tracker's
        frames lag it, so the voice ends a little left of the line), a replay's own clock, else the newest frame. Never
        before the newest frame, so no voice is drawn right of the line."""
        newest = self._t[-1]
        if self._runner is not None:
            return max(newest, self._runner.heard_t)
        if self._replay is not None:
            return max(newest, self._replay.origin_s + self._replay_elapsed())
        return newest

    def _octaves_at(self, x: np.ndarray, fold: PauseFold | None = None) -> np.ndarray:
        """The Octave shift in force at each view time x: that of the latest frame at or before it (the first frame's
        before them, the last frame's beyond them)."""
        if not self._t:
            return np.full(len(x), float(self._idle_octave))
        t = (fold or self._fold()).to_view(np.fromiter(self._t, float))
        octs = np.fromiter(self._oct, float)
        return octs[np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)]

    def _song_times(self, x: np.ndarray, fold: PauseFold | None = None) -> np.ndarray:
        """The song time heard at each view time x of the plot (NaN = no song there). Up to what the Transport has
        rendered that is the song clock's own answer at the voice time shown there, so seeks, Speed changes and loop
        wraps show as they happened and pauses are folded away (see _fold). Beyond it (the right half, up to FUTURE_S
        ahead of now) nothing is known yet: the song is taken to go on from where the rendering ended at its Speed,
        wrapping at the end of a looped Section to the next Pass's Lead-in, and ending at the song end. A seek, pause,
        Speed change or Section change made inside those seconds is not foreseen (it shows once rendered); while paused
        the song goes on, on resume, from where the player waits."""
        clock = self._draw_clock()
        fold = fold or self._fold(clock)
        voice = fold.to_voice(x)
        many = getattr(clock, "song_times", None)          # one pass over the array (a LogClock), not a call per point
        song = many(voice) if many is not None else np.array([np.nan if (t := clock(v)) is None else t for v in voice])
        tail = getattr(clock, "tail", None)
        tail = tail() if tail is not None else None
        if tail is None:                                 # a replay's clock knows its whole Take, a linear one goes on
            return song
        x_end, song_end, rate = tail
        backing = self._backing
        if song_end is None:                             # paused: the song waits where the player stands
            if backing is None:
                return song
            song_end, rate = backing.position_s(), getattr(backing, "speed", 1.0)
        x_end = fold.to_view(x_end)                      # paused: the fold the pause stands at
        ahead = x >= x_end - 1e-6                        # at a pause's fold both sides are the fold, give or take rounding
        duration = getattr(backing, "duration_s", None)
        loop = getattr(backing, "loop", None)
        loop_end = None if loop is None else min(loop.end_s, duration if duration is not None else loop.end_s)
        song[ahead] = project_song_time(x[ahead] - x_end, song_end, rate, loop_end,
                                        0.0 if loop is None else loop.pass_start_s, duration)
        return song

    def _sample_reference(self, x: np.ndarray,
                          fold: PauseFold | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(song time, MIDI, unscored) of the Melody at each view time x of the plot, past and future halves alike:
        the one place that turns the plot's x-range into the Reference, for the line, the Guide lines, the Unscored
        bands and whatever else is drawn in the right half."""
        song = self._song_times(x, fold)
        midi, unscored = self._melody.sample(song)
        return song, midi, unscored

    def _draw_melody(self, t_end: float) -> None:
        """The reference line over the plot's [t_end - PAST_S, t_end + FUTURE_S] (view time, see _fold)."""
        if self._melody is None:
            return
        x = np.arange(t_end - PAST_S, t_end + FUTURE_S + DRAW_STEP_S, DRAW_STEP_S)
        fold = self._fold()
        song, midi, unscored = self._sample_reference(x, fold)
        octaves = self._octaves_at(x, fold)
        shift = 12.0 * octaves                           # drawn where the Take is scored, not at the singer's octave
        verdict = {b: np.full(len(x), np.nan) for b in VERDICT_PEN}
        if isinstance(self._melody, NotesMelody):
            midi = self._note_bars(x, song, midi, octaves, shift)
            guide = midi.copy()                            # the Guide lines follow every bar, judged or not
            eras = [*self._verdict_eras, (self._verdicts_since, self._verdicts)]
            if any(v for _, v in eras):                    # a note that has its verdict: its bar in the verdict colour,
                index = self._melody.note_index_at(song)   # each Take's (Pass's) verdicts over that Take's bars only
                edges = [fold.to_view(start) for start, _ in eras] + [math.inf]
                for (_, verdicts), lo, hi in zip(eras, edges, edges[1:]):
                    span = (x >= lo) & (x < hi)
                    for k, band in verdicts.items():
                        on = span & (index == k)
                        verdict[band][on] = midi[on]
                        midi[on] = np.nan
        else:
            self._show_labels([])
            guide = midi
        for band, curve in self._verdict_curves.items():
            curve.setData(x, verdict[band] + shift)
        self._ref_curve.setData(x, midi + shift)
        coming = (x > t_end) & ~unscored
        self._ahead = (midi + shift)[coming]
        for curve, d in zip(self._guide_curves, (12, -12)):
            curve.setData(x, guide + shift + d)
        self._band_line = (x, guide + shift)             # the band covers every bar, judged or not, and no Guide line
        self._draw_band()
        edges = np.flatnonzero(np.diff(np.concatenate([[0], unscored.astype(np.int8), [0]])))
        starts, ends = edges[::2], edges[1::2]
        self._unscored.setOpts(x0=x[starts], x1=x[ends - 1] + DRAW_STEP_S, y0=[0.0] * len(starts),
                               y1=[127.0] * len(starts))

    def _draw_band(self) -> None:
        """The Tolerance as a soft band: exactly +- the Tolerance (cents, a hundredth of a semitone each) around the
        blue line as last drawn, as ONE path with a closed piece per unbroken stretch of the line (so it breaks where
        the line does: Unscored stretches, gaps between bars). A replay shows its Take's Tolerance, else the slider's."""
        path = QtGui.QPainterPath()
        if self._band_line is not None:
            x, y = self._band_line
            cents = self.tolerance_slider.value() if self._replay_tolerance is None else self._replay_tolerance
            half = cents / 100
            (c0, a0), (c1, a1) = TOLERANCE_BAND_ALPHA
            alpha = int(np.interp(cents, (c0, c1), (a0, a1)))
            self._tolerance_band.setBrush(pg.mkBrush(93, 173, 226, alpha))
            edges = np.flatnonzero(np.diff(np.concatenate([[0], np.isfinite(y).astype(np.int8), [0]])))
            xs, ys, gap = [], [], np.array([np.nan])
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a > 1:                            # one sample has no width
                    xs += [x[a:b], x[a:b][::-1], gap]    # the upper edge there, then the lower edge back
                    ys += [y[a:b] + half, y[a:b][::-1] - half, gap]
            if xs:
                path = pg.functions.arrayToQPath(np.concatenate(xs), np.concatenate(ys), connect="finite")
        self._tolerance_band.setPath(path)

    def _note_bars(self, x: np.ndarray, song: np.ndarray, midi: np.ndarray, octaves: np.ndarray,
                   shift: np.ndarray) -> np.ndarray:
        """Notes mode: the sampled notes as separate bars (the line blanked just before a bar that follows a different
        note, or a new Octave shift, so neighbouring bars never join into a staircase), each labelled with its note
        name at its start, the left edge of the plot for a bar already under way. An Unscored dip inside a note breaks
        its line but neither trims the bar before it nor labels the note again after it."""
        index = self._melody.note_index_at(song)
        index[~np.isfinite(midi)] = -1
        midi = midi.copy()
        on = np.flatnonzero(index >= 0)                  # the samples that belong to a note
        if not len(on):
            self._show_labels([])
            return midi
        different = (index[on][1:] != index[on][:-1]) | (octaves[on][1:] != octaves[on][:-1])    # vs the previous note sample
        for i, j in zip(on[1:][different], on[:-1][different]):
            if i == j + 1:                               # bars side by side: break the line (not across a dip, already NaN)
                midi[max(0, i - NOTE_GAP_SAMPLES):i] = np.nan
        starts = np.concatenate([[on[0]], on[1:][different]])
        notes = self._melody.note_list
        labels = [(x[i], notes[index[i]].midi + shift[i], note_name(int(round(notes[index[i]].midi + shift[i]))))
                  for i in starts]
        self._show_labels(labels)
        return midi

    def _show_labels(self, labels: list[tuple[float, float, str]]) -> None:
        """Put (x, MIDI, text) labels on the plot from the pool: items are created once and reused, the rest hidden."""
        while len(self._note_labels) < len(labels):
            item = pg.TextItem("", color=NOTE_LABEL_COLOR, anchor=(0, 1))
            item.setZValue(4)
            self._plot.addItem(item)
            self._note_labels.append(item)
        for item, (lx, ly, text) in zip(self._note_labels, labels):
            if item.toPlainText() != text:
                item.setText(text)
            item.setPos(lx, ly)
            if not item.isVisible():
                item.setVisible(True)
        for item in self._note_labels[len(labels):]:
            if item.isVisible():
                item.setVisible(False)

    def _playhead_heard(self, now: float, fold: PauseFold) -> None:
        """The Timeline playhead on the song heard, the one under the now-line (not the rendered position, ahead by the
        output latency). Paused, that is where the player stands, so both stand still on one song time. Right after a
        click on the Timeline (or an arrow key) it stays on the target until the song heard gets there (at most
        PLAYHEAD_HOLD_S), so it does not step back and forth."""
        target, until = self._playhead_hold
        song = float(self._song_times(np.array([now]), fold)[0])
        if until and time.perf_counter() < until and not target - PLAYHEAD_NEAR_S <= song <= target + 1.0:
            self.timeline.set_playhead(target)       # the jump is not heard yet
            return
        self._playhead_hold = (0.0, 0.0)
        if np.isfinite(song):
            self.timeline.set_playhead(song)

    def _health_tick(self, started: float) -> None:
        """Tick duration and window size changes for the audio health record (audio_health.py)."""
        if self._runner is None:
            return
        self._health_ui.tick(perf_counter() - started, (self.width(), self.height()))

    def tick(self, drain_only: bool = False) -> None:
        tick_started = perf_counter()
        if self._runner is not None:
            try:
                scores = self._runner.drain()
                dots = self._runner.drain_display()     # Fast: the dots come from the display tracker
                cuts = self._runner.drain_cuts()        # after drain(): every drained score's boundary is here
                if self._note_tally is not None:
                    self._pending_cuts += cuts
                self.ingest(scores, dots)
            except Exception:           # the worker thread died: stop() ends the Take and reports why
                self._stop()
                return
            for record in self._runner.drain_saved():
                if record.pass_index is not None:
                    self.summary_label.setText(pass_text(record))
                else:
                    self._say_jumped(f"Jumped. Take saved: {describe(record)}")
                if record.pass_index is not None and record.summary.n_scored:
                    self._last_pass_pct = record.summary.pct_in_tune
            b = self._backing
            if b is not None and not drain_only and getattr(b, "finished", False) \
                    and b.position_s() > b.duration_s + END_GRACE_S:
                self._stop()            # the song is over (plus time for the late voice to arrive)
                return
        elif self._replay is not None:
            elapsed = self._replay_elapsed()
            self.ingest(self._replay.advance(elapsed))
            song_s = self._replay_clock(self._replay.origin_s + elapsed)       # the playhead follows the replayed song
            if song_s is not None:
                self.timeline.set_playhead(song_s)
        if self._backing is not None and not drain_only:
            self.timeline.set_playhead(self._backing.position_s())     # until a song is heard (just below)
            self._show_loop()
            self._show_pause()
            self._show_speed_hint()
        self._health_tick(tick_started)
        if not self._t:
            return
        fold = self._fold()
        t, midi = fold.to_view(np.fromiter(self._t, float)), np.fromiter(self._midi, float)
        self._curve.setData(t, midi)
        voiced = ~np.isnan(midi)
        bands = [b for b, v in zip(self._band, voiced) if v]
        out = [o for o, v in zip(self._out, voiced) if v]
        self._dots.setData(t[voiced], midi[voiced], brush=[BRUSH[b] for b in bands],
                           symbol=["x" if o else "o" for o in out],   # out-of-key notes are drawn as big crosses
                           size=[11 if o else 6 for o in out])
        now = fold.to_view(self._now_voice())
        self._set_now(now)
        if self._backing is not None and not drain_only:
            self._playhead_heard(now, fold)
        self._close_notes(fold)
        self._draw_melody(now)
        self._follow_y()
        if self._replay is not None and self._replay.finished:
            self._end_replay(finished=True)

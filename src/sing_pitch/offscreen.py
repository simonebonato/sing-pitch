"""Headless Qt (tests, the QA driver) with the system's fonts on every OS.

Qt's offscreen platform finds fonts through QT_QPA_FONTDIR, defaulting to Qt's own lib/fonts folder, which PySide6's
wheels do not ship; on Windows it then has no font at all (QFontDatabase.families() == []), text is measured with a
stand-in and the window comes out ~300 px wider than on the screen (the window-size tests fail). On macOS the
offscreen platform reads the system fonts through Core Text and ignores QT_QPA_FONTDIR, so this changes nothing
there. Call before the first QApplication."""
from __future__ import annotations

import os


def use_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ["QT_QPA_PLATFORM"] != "offscreen" or "QT_QPA_FONTDIR" in os.environ:
        return
    from PySide6.QtCore import QStandardPaths

    for folder in QStandardPaths.standardLocations(QStandardPaths.FontsLocation):
        if os.path.isdir(folder):
            os.environ["QT_QPA_FONTDIR"] = folder
            return

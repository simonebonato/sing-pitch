import gc
import threading

import pytest

from sing_pitch.offscreen import use_offscreen

use_offscreen()  # headless Qt with the system's fonts; must run before any QApplication

from qt_leaks import collect_and_report_live_qt_garbage  # noqa: E402

_collections_off_the_ui_thread: list[str] = []


def _note_where_gc_runs(phase, info) -> None:
    if phase == "start" and threading.current_thread() is not threading.main_thread():
        _collections_off_the_ui_thread.append(threading.current_thread().name)


gc.callbacks.append(_note_where_gc_runs)


@pytest.fixture(autouse=True, scope="session")
def _garbage_is_collected_on_the_ui_thread_only(qapp):
    """The app's policy (sing_pitch.ui.gui_gc) for the whole run, so no test depends on which one builds a window."""
    from sing_pitch.ui.gui_gc import collect_on_the_ui_thread_only

    collect_on_the_ui_thread_only()


@pytest.fixture(autouse=True)
def _no_qt_object_is_left_to_the_cycle_collector(request):
    """Every test, after pytest-qt has closed its widgets: nothing Qt may be waiting for the cycle collector, and the
    collector must never have run on another thread. This guards every widget and every signal connection any test
    builds, not only the ones a dedicated test thinks of."""
    yield
    off_thread = list(_collections_off_the_ui_thread)
    _collections_off_the_ui_thread.clear()
    left = collect_and_report_live_qt_garbage()
    failed = getattr(request.node, "rep_call", None)
    if failed is not None and failed.failed:
        return                                    # a failing test's traceback keeps its window alive: not a leak
    assert not gc.isenabled(), "a test switched the automatic cycle collector back on (see sing_pitch/ui/gui_gc.py)"
    assert not off_thread, f"the cycle collector ran off the UI thread, on {sorted(set(off_thread))}"
    assert not left, f"live Qt objects in reference cycles (freed only by gc, on any thread): {dict(left)}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    setattr(item, f"rep_{report.when}", report)
    return report

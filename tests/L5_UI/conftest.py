"""Shared fixtures for L5_UI split-first test modules.

Qt + pyqtgraph setup: many L5_UI modules (extracted from
the GUI entry point) touch Qt widgets. Qt's C++ side strictly
requires a QApplication instance before any widget creation, even
under the offscreen platform plugin.

The fixture is session-scoped + autouse so individual test files
don't have to declare it. Tests still work if QT_QPA_PLATFORM is
already set to something else (xcb, eglfs) — we only force offscreen
if no setting is present.

Pattern reusable by future Dashboard/gpu_ui/qt_interface mixin
tests once those decompositions land.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure the CRISPI module path is importable before any test in this
# directory imports from dashboard_*.
REPO_ROOT = Path(__file__).resolve().parents[2]
CRISPI_PATH = REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI/CS"
if str(CRISPI_PATH) not in sys.path:
    sys.path.insert(0, str(CRISPI_PATH))

# qt_interface.py / camera.py / button_bar.py do unconditional
# `from ids_peak import ids_peak` at module load and reference module-
# level constants like `ids_peak_ipl.PixelFormatName_Mono8`. The IDS
# Peak SDK is proprietary and not redistributable on CI; the tests in
# this directory only exercise mixin inheritance + Qt widget
# construction, not actual camera I/O.
#
# MagicMock stubs satisfy both the import AND arbitrary attribute
# access — any `<stub>.SOME_CONSTANT` lookup returns another
# MagicMock, which is enough to let module load complete. If a test
# ever actually calls into the SDK it'll get a MagicMock call result
# (typically not a useful behavior, but these tests don't do that).
#
# Run BEFORE any test imports a module that pulls qt_interface.
for _name in (
    "ids_peak",
    "ids_peak.ids_peak",
    "ids_peak.ids_peak_ipl_extension",
    "ids_peak_ipl",
    "ids_peak_ipl.ids_peak_ipl",
    "ids_peak_afl",
    "ids_peak_afl.ids_peak_afl",
):
    sys.modules.setdefault(_name, MagicMock(name=_name))


# Force offscreen Qt BEFORE PyQt5 imports. Setdefault preserves the
# operator's choice if they've explicitly set QT_QPA_PLATFORM (e.g.
# xcb for a real display during interactive debugging).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


from PyQt5.QtWidgets import QApplication  # noqa: E402

# Created at import time (before any test collection) so test
# parametrize/collection that imports widgets doesn't crash.
_QAPP = QApplication.instance() or QApplication(["pytest-l3_5"])


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Return the session-scoped QApplication instance.

    Autouse so tests don't have to request the fixture explicitly —
    the QApp existence is enough to prevent Qt-widget crashes.
    """
    return _QAPP

"""StartupWindowMixin — extracted from qt_interface.py.

Bundles five startup / window-management methods:

* ``start_window()`` (~70 LOC) — connect camera signals to the GUI;
  wire image_update_signal → on_image_received.
* ``_ensure_projection()`` (~35 LOC) — lazy-init the second-monitor
  projection window.
* ``start_interface()`` (~7 LOC) — Qt event-loop entry.
* ``_open_tiff_viewer()`` (~21 LOC) — file picker + napari TIFF viewer
  launch.
* ``_open_tiff_external()`` (~42 LOC) — file picker + xdg-open
  fallback for system viewer.

Method bodies are byte-identical to the pre-extraction code at
``qt_interface.py:324-498`` (commit ``3fb0ab2``); only the
surrounding module-level frame changed.

Mixin contract:
  * ``self._camera`` — image source signal
  * ``self.image_update_signal`` — pyqtSignal wired to on_image_received
  * ``self.projection`` — second-monitor window
  * ``self._qt_instance`` — QApplication ref for exec_

See ``docs/specs/L5_UI/qt_interface.md``.
"""

import os
import sys
import time

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QGuiApplication, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

class StartupWindowMixin:
    """Cluster 19 — window startup + viewer launchers."""

    def start_window(self):
        connected = False
        candidate_names = ("frame_ready", "image_ready", "new_frame", "frame", "qsignal_frame", "qsignal_image")

        for name in candidate_names:
            sig = getattr(self._camera, name, None)
            if sig is None:
                continue
            try:
                try:
                    sig.disconnect(self.on_image_received)
                except (TypeError, RuntimeError):
                    pass
                sig.connect(self.on_image_received, QtCore.Qt.QueuedConnection)
                print(f"Connected camera signal: {name} → on_image_received (QueuedConnection)")
                connected = True
                break
            except Exception:
                pass

        if not connected:
            for setter in ("set_frame_callback", "set_image_callback"):
                cb = getattr(self._camera, setter, None)
                if callable(cb):
                    try:
                        cb(self.on_image_received)
                        print(f"Installed camera callback via {setter}()")
                        connected = True
                        break
                    except Exception:
                        pass

        if not connected:
            print("Could not connect any camera frame signal; preview will be blank.")
        else:
            print("Camera connected to UI.")

        # Wake the live preview when calibration finishes — replaces the
        # workaround where the user had to wiggle digital gain to refresh.
        if hasattr(self._camera, "calibrationFinished"):
            try:
                self._camera.calibrationFinished.connect(
                    self._on_calibration_finished_refresh,
                    QtCore.Qt.QueuedConnection)
            except Exception as e:
                print(f"Could not hook calibrationFinished signal: {e}")

        self._create_button_bar()
        self._create_statusbar()

        try:
            self.image_update_signal.connect(self.display.on_image_received, QtCore.Qt.QueuedConnection)
            print("Bound image_update_signal → Display.on_image_received")
        except Exception as e1:
            print(f"Primary connect failed ({e1}); falling back to setImage alias")
            try:
                self.image_update_signal.connect(self.display.setImage, QtCore.Qt.QueuedConnection)
                print("Bound image_update_signal → Display.setImage")
            except Exception as e2:
                print(f"Display signal hookup failed: {e2}")
        # Wire pixel probe signal from Display to statusbar
        try:
            self.display.pixel_probe_signal.connect(self._on_pixel_probe_result)
        except Exception as e:
            print(f"Pixel probe signal connect failed: {e}")

        # Delay creating the projector window until actually needed (calibration/projection)
        # This avoids early windowing/GL issues on some Jetson setups.
        self.projection = None

    def _ensure_projection(self):
        if self.projection is not None:
            try:
                # Verify the Qt C++ object is still alive (WA_DeleteOnClose
                # destroys it when the window is closed, leaving a stale ref)
                self.projection.isVisible()
                return True
            except RuntimeError:
                self.projection = None
        if self.projection is not None:
            return True
        try:
            from projection import ProjectDisplay
            screens = QGuiApplication.screens()
            if not screens:
                print("No screens available for projection")
                return False
            screen = screens[1] if len(screens) > 1 else screens[0]
            try:
                self.projection = ProjectDisplay(screen, parent=self)
            except TypeError:
                self.projection = ProjectDisplay(screen)
                self.projection.setParent(self)
            self.projection.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            return True
        except Exception as e:
            print(f"Failed to create projection window: {e}")
            self.projection = None
            return False


    # _update_recording_button_text + _on_recording_{started,stopped} +
    # _on_auto_start_recording extracted to qt_interface_hw_acq.py
    # (HardwareAcqMixin) per L5 §0.5 decomposition (iter-2).

    def start_interface(self):
        self._gain_slider.setMaximum(int(self._camera.max_gain * 100))
        
        QtCore.QCoreApplication.setApplicationName("STIMViewer")
        self.show()
        self._qt_instance.exec()

    def _open_tiff_viewer(self):
        """Open a file dialog to pick a recorded TIFF, then launch the viewer."""
        try:
            default_dir = os.environ.get("STIM_SAVE_DIR", "./Saved_Media")
            if not os.path.isabs(default_dir):
                default_dir = os.path.abspath(default_dir)
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select TIFF recording", default_dir, "TIFF files (*.tif *.tiff);;All files (*)")
            if not path:
                return
            try:
                import tifffile  # noqa: F401
            except ImportError:
                self.warning("tifffile not available — cannot open TIFF viewer")
                return
            # Lazy import: _TiffViewer lives in qt_interface.py. qt_interface
            # has fully loaded by the time this method runs (it's a button
            # click handler), so the import succeeds without circular issues.
            from qt_interface import _TiffViewer
            viewer = _TiffViewer(path, parent=self)
            viewer.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            viewer.show()
        except Exception as e:
            self.warning(f"View Recording error: {e}")

    def _open_tiff_external(self):
        """File-picker → launch the TIFF in the system's default app.

        Uses `xdg-open` (Linux freedesktop standard) so the operator's
        configured default for `.tiff` files opens (typically Fiji /
        ImageJ on lab Jetsons). Doesn't block the GUI — runs in
        background process.

        Replaces the prior in-app `_TiffPlayer` (cv2 mp4v transcode +
        QTimer-driven playback). Removed  because:
        (a) mp4v is lossy → not science-grade for scientific imagery,
        (b) Fiji already does everything the player was trying to do
            but with better contrast tools + ROI + 16-bit precision +
            the whole ImageJ plugin ecosystem,
        (c) `xdg-open` is one line + respects user tool choice.
        """
        try:
            default_dir = os.environ.get("STIM_SAVE_DIR", "./Saved_Media")
            if not os.path.isabs(default_dir):
                default_dir = os.path.abspath(default_dir)
            try:
                os.makedirs(default_dir, exist_ok=True)
            except Exception:
                pass
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select TIFF recording to open externally", default_dir,
                "TIFF files (*.tif *.tiff);;All files (*)")
            if not path:
                return
            import shutil as _sh, subprocess as _sp
            # Try the freedesktop handler first (respects the operator's default
            #.tiff app, e.g. Fiji/ImageJ), then fall back to any installed
            # viewer. The image ships eog; Fiji/ImageJ gives full stack tools.
            openers = ["xdg-open", "eog", "feh", "display"]
            opener = next((o for o in openers if _sh.which(o)), None)
            if opener is None:
                self.warning(
                    "No external image viewer found. Install one (eog, feh, "
                    "ImageMagick, or Fiji/ImageJ) to use this button. "
                    f"Path: {path}"
                )
                return
            try:
                # nosec B603: fixed opener binary + a path already validated by
                # Qt's file dialog. Invoking the OS viewer is the intent.
                _sp.Popen([opener, path])  # nosec B603
                print(f"[GUI] Opened {path} in external viewer ({opener})")
            except Exception as e:
                self.warning(f"{opener} failed: {e}. Path: {path}")
        except Exception as e:
            self.warning(f"Open in External Viewer error: {e}")


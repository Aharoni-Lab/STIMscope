"""CalibrationProjectorMixin — extracted from qt_interface.py.

Bundles three calibration / projection methods that don't fit the
existing SL or button-bar mixins:

* ``_send_hmatrix_to_projector()`` (~17 LOC) — push the camera→DMD
  homography matrix to the projector engine over ZMQ.
* ``_asift_calibrate()`` (~80 LOC) — A-SIFT-based homography
  calibration (alternative to SL calibration).
* ``_on_calibration_finished_refresh()`` (~29 LOC) — refresh live
  preview after Calibrate completes (resets camera state +
  display update).

Method bodies are byte-identical to the pre-extraction code at
``qt_interface.py:867-1173`` (commit ``f56890d``); only the
surrounding module-level frame changed.

Mixin contract (Interface attributes the method reads/writes):
  * ``self._camera`` — for trigger params + parameter map
  * ``self.projection`` — second-monitor window
  * ``self._ensure_projection`` — guards projection availability
  * ``self._proc_projector`` — QProcess ref to the engine
  * ``self.display`` — live-preview widget

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

from qt_interface_mixins._shared import ASSETS


class CalibrationProjectorMixin:
    """Cluster 17 — calibration + projector control + DMD diagnostic."""

    def _send_hmatrix_to_projector(self):
        try:
            import numpy as np
            # Prefer in-memory H from last calibration
            H = getattr(self._camera, 'translation_matrix', None)
            if H is None or not hasattr(H, 'shape'):
                # Fallback to npy on disk
                npy_path = (ASSETS / 'Generated' / 'homography_cam2proj.npy').resolve()
                if npy_path.exists():
                    H = np.load(str(npy_path))
            if H is None:
                print("No H-matrix available. Calibrate first.")
                return
            self._camera._send_h_to_projector(H)
        except Exception as e:
            print(f"REQ H-Matrix failed: {e}")

    def _asift_calibrate(self):
        """Compute 3x3 H via ASIFT (fallback SIFT), update camera H and projector.

        - Loads reference/capture paths from Assets/Generated
        - Uses ZMQ_sender_mask.asift_calibration backend
        - Writes homography_cam2proj.txt next to existing files
        """
        try:
            from pathlib import Path
            import cv2
            # Import backend (ensure repository path is on sys.path or installed)
            try:
                from ZMQ_sender_mask.asift_calibration import run_asift_calibration_and_send
            except Exception:
                # Attempt to add ZMQ_sender_mask directory to sys.path
                try:
                    import sys as _sys
                    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ZMQ_sender_mask"))
                    from asift_calibration import run_asift_calibration_and_send
                except Exception as e2:
                    print(f"ASIFT backend import failed: {e2}")
                    return

            assets = Path(__file__).resolve().parent.parent / "Assets" / "Generated"
            ref_path = (assets / "custom_registration_image.png").as_posix()
            cam_path = (assets / "calibration_capture_image.png").as_posix()
            save_txt = (assets / "homography_cam2proj.txt").as_posix()

            # Prerequisite check: ASIFT compares a projected reference
            # (custom_registration_image.png) against a captured frame
            # (calibration_capture_image.png). Both are produced by the
            # regular Calibrate flow — without a prior Calibrate the backend
            # fails silently inside imread. Surface the missing prerequisite
            # clearly so the operator knows the required sequence.
            missing = []
            if not Path(ref_path).exists():
                missing.append("custom_registration_image.png (projected reference)")
            if not Path(cam_path).exists():
                missing.append("calibration_capture_image.png (camera capture)")
            if missing:
                msg = ("ASIFT needs files from a prior Calibrate run: "
                       + "; ".join(missing) + ". Click Calibrate first, then "
                       "ASIFT Calibration.")
                print(f"[ASIFT] prerequisites missing: {msg}")
                try:
                    if hasattr(self, "warning"):
                        self.warning(msg)
                except Exception:
                    pass
                return

            ok, H = run_asift_calibration_and_send(ref_path, cam_path, endpoint="tcp://127.0.0.1:5560", save_txt=save_txt)
            if not ok or H is None:
                print("ASIFT calibration failed: no H")
                return

            # Update in-memory camera H so the rest of UI uses the new matrix
            try:
                if hasattr(self, "_camera") and (self._camera is not None):
                    self._camera.translation_matrix = H
            except Exception:
                pass

            # Send to projector immediately
            try:
                self._camera._send_h_to_projector(H)
            except Exception as esend:
                print(f"Could not send ASIFT H to projector: {esend}")
            print(f"ASIFT Calibration OK. Wrote: {save_txt}")

            # Immediately apply H to the registration image and project it for confirmation
            try:
                if not self._ensure_projection():
                    print("ASIFT confirm: projection window unavailable.")
                    return
                img_path = (Path(__file__).resolve().parent.parent / "Assets" / "Generated" / "custom_registration_image.png").as_posix()
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"ASIFT confirm: cannot read {img_path}")
                    return
                # Use current warp mode H; show image with H
                try:
                    Hn = H / H[2, 2] if abs(float(H[2, 2])) > 1e-12 else H
                except Exception:
                    Hn = H
                try:
                    self.projection.show_image_fullscreen_on_second_monitor(img, Hn)
                except Exception:
                    # Fallback to interface method
                    self.on_projection_received(img, Hn)
                print("ASIFT confirm: projected registration with new H")
            except Exception as econf:
                print(f"ASIFT confirm failed: {econf}")
        except Exception as e:
            print(f"ASIFT Calibration error: {e}")

    # _select_warp_h + _select_warp_lut + _on_warp_h_toggled +
    # _on_warp_lut_toggled extracted to qt_interface_camera_controls.py
    # (CameraControlsMixin) per L5 §0.5 decomposition (iter-8).

    # Overlay + pixel-probe methods extracted to qt_interface_overlay_probe.py
    # (OverlayProbeMixin) per L5 §0.5 decomposition (iter-1, 5 methods, 162 LOC).

    def _on_calibration_finished_refresh(self):
        """Triggered after a successful Calibrate. Wakes up the live preview
        so the user sees fresh frames immediately, without having to touch
        digital gain to kick the acquisition path."""
        try:
            # No-op gain re-set: pokes an IDS Peak GenICam node and flushes
            # stale buffers (mimics what the user was doing manually).
            cur_gain = None
            if hasattr(self._camera, 'get_gain'):
                try: cur_gain = float(self._camera.get_gain())
                except Exception: cur_gain = None
            if cur_gain is None:
                cur_gain = float(getattr(self._camera, 'target_gain', 1.0))
            if hasattr(self._camera, 'set_gain'):
                try:
                    self._camera.set_gain(cur_gain)
                except Exception:
                    pass
            # Belt + suspenders: invalidate the display widget directly.
            try:
                self.display.update()
            except Exception:
                pass
            print("[CALIB] Live preview refreshed after calibration")
        except Exception as e:
            print(f"_on_calibration_finished_refresh error: {e}")


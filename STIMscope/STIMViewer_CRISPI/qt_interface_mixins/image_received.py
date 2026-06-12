"""ImageReceivedMixin — extracted from qt_interface.py.

Bundles the two image-callback methods:

* ``on_image_received(image)`` — main camera frame callback,
  updates preview + ROI overlay + pixel-probe readout (~284 LOC).
* ``on_projection_received(image, homography_matrix=None)`` —
  push an image to the second-monitor projection window (~9 LOC).

Method bodies are byte-identical to the pre-extraction code at
``qt_interface.py:779-1072`` (commit ``7463a6e``); only the
surrounding module-level frame changed.

Mixin contract (Interface attributes the method reads/writes):
  * ``self.display`` — preview widget (frame paint + ROI overlay)
  * ``self.projection`` — second-monitor window
  * ``self._overlay_*`` — overlay state (set up in ``__init__``)
  * ``self._camera`` — for FPS / shape metadata

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

class ImageReceivedMixin:
    """Cluster 16 — camera-frame + projection-frame received callbacks."""

    def on_image_received(self, image):
        # DEBUG (off unless STIM_FRAME_DEBUG=1): count how many frames reach
        # the Interface from the camera. Throttled to ~1/sec at 30 fps.
        if os.environ.get("STIM_FRAME_DEBUG") == "1":
            self._iface_frame_count = getattr(self, "_iface_frame_count", 0) + 1
            if self._iface_frame_count % 30 == 1:  # log first frame + every 30th
                print(f"[FRAME-DEBUG iface] on_image_received #{self._iface_frame_count} "
                      f"(type={type(image).__name__})")
        try:
            import numpy as np
            import cv2


            def _get_attr(obj, names):
                for n in names:
                    v = getattr(obj, n, None)
                    if callable(v):
                        try:
                            return v()
                        except Exception:
                            continue
                    elif v is not None:
                        return v
                return None

            def _get_int(obj, names):
                v = _get_attr(obj, names)
                try:
                    return int(v)
                except Exception:
                    return None

            def _bayer_code(pf_str: str):
                s = (pf_str or "").upper()
                if "BAYERRG" in s: return cv2.COLOR_BayerRG2RGB
                if "BAYERBG" in s: return cv2.COLOR_BayerBG2RGB
                if "BAYERGB" in s: return cv2.COLOR_BayerGB2RGB
                if "BAYERGR" in s: return cv2.COLOR_BayerGR2RGB
                return None

            def _bit_depth_shift(pf_str: str):
                s = (pf_str or "").upper()

                if "12" in s: return 4
                if "10" in s: return 2
                if "16" in s: return 8
                return 0

            def _numpy_from_ids(img_obj):
                for n in ("get_numpy", "get_numpy_view", "get_numpy_array", "get_numpy_1D"):
                    f = getattr(img_obj, n, None)
                    if callable(f):
                        try:
                            arr = f()
                            if isinstance(arr, np.ndarray):
                                return arr
                        except Exception:
                            pass

                f = getattr(img_obj, "get_buffer", None)
                if callable(f):
                    try:
                        raw = f()
                        if raw is not None:
                            return np.frombuffer(raw, dtype=np.uint8)
                    except Exception:
                        pass
                return None


            pf_str = ""

            if isinstance(image, np.ndarray):
                arr = image
                h, w = arr.shape[:2]
                ch = 1 if arr.ndim == 2 else arr.shape[2]
            else:

                w = _get_int(image, ("Width", "width", "GetWidth", "ImageWidth"))
                h = _get_int(image, ("Height", "height", "GetHeight", "ImageHeight"))
                pf   = _get_attr(image, ("PixelFormat", "pixel_format", "GetPixelFormat", "PixelFormatName"))
                pf_str = str(pf) if pf is not None else ""

                arr = _numpy_from_ids(image)
                if arr is None:
                    print("on_image_received: no buffer -> dropping frame")
                    return

                if arr.ndim == 3:

                    h, w, ch = arr.shape
                elif arr.ndim == 2:

                    ch = 1
                else:

                    channels = 4 if ("BGRA" in pf_str or "RGBA" in pf_str) else 3 if ("BGR" in pf_str or "RGB" in pf_str) else 1
                    if not (w and h):
                        print("on_image_received: unknown WxH for 1D buffer")
                        return
                    expected = w * h * channels
                    if arr.size < expected:
                        print("on_image_received: buffer smaller than expected")
                        return
                    arr = arr[:expected].reshape(h, w, channels) if channels > 1 else arr[:w*h].reshape(h, w)
                    ch = channels



            if arr.dtype == np.uint16:

                shift = _bit_depth_shift(pf_str) if pf_str else 8
                arr8 = (arr >> shift).astype(np.uint8, copy=False)
            elif arr.dtype != np.uint8:
                arr8 = arr.astype(np.uint8, copy=False)
            else:
                arr8 = arr


            bayer = _bayer_code(pf_str)
            if (arr8.ndim == 2 or (arr8.ndim == 3 and arr8.shape[2] == 1)) and bayer is not None:
                try:
                    rgb = cv2.cvtColor(arr8 if arr8.ndim == 2 else arr8[:, :, 0], bayer)
                    qsrc = rgb
                    # Optional software contrast (fallback if camera lacks hardware contrast)
                    try:
                        cf = float(getattr(self, "_contrast_factor", 1.0))
                        apply_sw = bool(getattr(self, "_soft_contrast_active", False))
                        if apply_sw and abs(cf - 1.0) > 1e-3:
                            lut = getattr(self, "_contrast_lut", None)
                            lutf = getattr(self, "_contrast_lut_factor", None)
                            if lut is None or lutf is None or float(lutf) != float(cf):
                                lut = self._make_contrast_lut(cf)
                                self._contrast_lut = lut
                                self._contrast_lut_factor = cf
                            if lut is not None:
                                try:
                                    cv2.LUT(qsrc, lut, dst=qsrc)
                                except Exception:
                                    qsrc = cv2.LUT(qsrc, lut)
                    except Exception:
                        pass
                    fmt = QtGui.QImage.Format_RGB888
                    h, w = qsrc.shape[:2]
                    bpl = int(qsrc.strides[0])
                    qimg = QtGui.QImage(qsrc.data, w, h, bpl, fmt).copy()
                except Exception as e:
                    print(f"Demosaic failed ({pf_str}), falling back to grayscale: {e}")
                    qsrc = arr8 if arr8.ndim == 2 else arr8[:, :, 0]
                    # Optional software contrast for grayscale
                    try:
                        cf = float(getattr(self, "_contrast_factor", 1.0))
                        apply_sw = bool(getattr(self, "_soft_contrast_active", False))
                        if apply_sw and abs(cf - 1.0) > 1e-3:
                            lut = getattr(self, "_contrast_lut", None)
                            lutf = getattr(self, "_contrast_lut_factor", None)
                            if lut is None or lutf is None or float(lutf) != float(cf):
                                lut = self._make_contrast_lut(cf)
                                self._contrast_lut = lut
                                self._contrast_lut_factor = cf
                            if lut is not None:
                                try:
                                    cv2.LUT(qsrc, lut, dst=qsrc)
                                except Exception:
                                    qsrc = cv2.LUT(qsrc, lut)
                    except Exception:
                        pass
                    fmt = QtGui.QImage.Format_Grayscale8
                    h, w = qsrc.shape[:2]
                    bpl = int(qsrc.strides[0])
                    qimg = QtGui.QImage(qsrc.data, w, h, bpl, fmt).copy()
            else:

                if arr8.ndim == 2 or (arr8.ndim == 3 and arr8.shape[2] == 1):
                    qsrc = arr8 if arr8.ndim == 2 else arr8[:, :, 0]
                    h, w = qsrc.shape[:2]
                    fmt = QtGui.QImage.Format_Grayscale8
                    bpl = int(qsrc.strides[0])
                elif arr8.shape[2] == 3:


                    if "BGR" in (pf_str or "").upper():
                        qsrc = cv2.cvtColor(arr8, cv2.COLOR_BGR2RGB)
                    else:

                        qsrc = arr8
                    h, w = qsrc.shape[:2]
                    fmt = QtGui.QImage.Format_RGB888
                    bpl = int(qsrc.strides[0])
                else:


                    if "BGRA" in (pf_str or "").upper():
                        qsrc = cv2.cvtColor(arr8, cv2.COLOR_BGRA2RGBA)
                    else:
                        qsrc = arr8
                    h, w = qsrc.shape[:2]
                    fmt = QtGui.QImage.Format_RGBA8888
                    bpl = int(qsrc.strides[0])

                # Optional software contrast (handles gray, RGB, and preserves alpha)
                try:
                    cf = float(getattr(self, "_contrast_factor", 1.0))
                    apply_sw = bool(getattr(self, "_soft_contrast_active", False))
                    if apply_sw and abs(cf - 1.0) > 1e-3:
                        lut = getattr(self, "_contrast_lut", None)
                        lutf = getattr(self, "_contrast_lut_factor", None)
                        if lut is None or lutf is None or float(lutf) != float(cf):
                            lut = self._make_contrast_lut(cf)
                            self._contrast_lut = lut
                            self._contrast_lut_factor = cf
                        if lut is not None:
                            if qsrc.ndim == 2:
                                try:
                                    cv2.LUT(qsrc, lut, dst=qsrc)
                                except Exception:
                                    qsrc = cv2.LUT(qsrc, lut)
                            elif qsrc.ndim == 3 and qsrc.shape[2] == 3:
                                try:
                                    cv2.LUT(qsrc, lut, dst=qsrc)
                                except Exception:
                                    qsrc = cv2.LUT(qsrc, lut)
                            elif qsrc.ndim == 3 and qsrc.shape[2] == 4:
                                rgb = qsrc[:, :, :3]
                                try:
                                    cv2.LUT(rgb, lut, dst=rgb)  # in-place on first 3 channels
                                except Exception:
                                    rgb2 = cv2.LUT(rgb, lut)
                                    qsrc[:, :, :3] = rgb2
                except Exception:
                    pass
                # Apply camera orientation transforms (rotate/flip)
                try:
                    rot = getattr(self, '_cam_rotation', 0)
                    fh = getattr(self, '_cam_flip_h', False)
                    fv = getattr(self, '_cam_flip_v', False)
                    # Use cv2 for efficient transforms (single allocation)
                    if fh and fv:
                        qsrc = cv2.flip(qsrc, -1)  # both = flip code -1
                    elif fh:
                        qsrc = cv2.flip(qsrc, 1)
                    elif fv:
                        qsrc = cv2.flip(qsrc, 0)
                    if rot == 90:
                        qsrc = cv2.rotate(qsrc, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rot == 180:
                        qsrc = cv2.rotate(qsrc, cv2.ROTATE_180)
                    elif rot == 270:
                        qsrc = cv2.rotate(qsrc, cv2.ROTATE_90_CLOCKWISE)
                except Exception:
                    pass

                # NOTE: ROI segmentation contour overlay on the camera preview
                # is intentionally NOT drawn here even when the main GUI's
                # "Overlay On" button is checked. That button toggles the
                # projector engine's frame-counter/digit overlay (a projection-
                # side feature), not a camera-preview annotation. Drawing ROI
                # contours on the preview is a feature owned by the RTTE / CS
                # Pipeline dialogs (each provides its own overlay control).
                # We only draw ROI contours here if explicitly opted in via
                # _show_roi_overlay_on_preview (separate flag, not wired to the
                # main "Overlay On" button — reserved for future RTTE re-use).
                try:
                    if getattr(self, '_show_roi_overlay_on_preview', False) \
                       and getattr(self, '_overlay_contours', None):
                        qsrc = self._draw_overlay_on_frame(qsrc)
                        if qsrc.ndim == 3 and qsrc.shape[2] == 3:
                            fmt = QtGui.QImage.Format_RGB888
                except Exception:
                    pass

                # Recompute shape/stride after any adjustment
                h, w = qsrc.shape[:2]
                bpl = int(qsrc.strides[0])
                qimg = QtGui.QImage(qsrc.data, w, h, bpl, fmt).copy()


            # HW-1 fix: frame_arrival is now recorded on the camera thread
            # (camera.py:1264) — unconditionally per processed frame, not
            # dependent on Qt event-loop dispatch. Removing this duplicate
            # eliminated a 2× FPS doubling bug.

            self.image_update_signal.emit(qimg)
            # DEBUG (off unless STIM_FRAME_DEBUG=1): trace QImage hand-off
            # to display. Throttled to ~1/sec at 30 fps. If iface counts
            # but this never logs, an exception above (silently caught) is
            # dropping the frame before emit.
            if os.environ.get("STIM_FRAME_DEBUG") == "1":
                if self._iface_frame_count % 30 == 1:
                    try:
                        non_zero = "yes" if qimg.bits().asarray(qimg.byteCount())[:64].count(b"\x00") < 64 else "ALL-ZERO"
                    except Exception:
                        non_zero = "?"
                    print(f"[FRAME-DEBUG iface] emit image_update_signal "
                          f"#{self._iface_frame_count} {qimg.width()}x{qimg.height()} "
                          f"(first-64-bytes-nonzero={non_zero})")

        except Exception as e:
            print(f"on_image_received failed: {e}")




    def on_projection_received(self, image, homography_matrix = None):
        """
        Update Projection Image
        """


        try:
            self.projection.show_image_fullscreen_on_second_monitor(image, homography_matrix)
        except Exception as e:
            print(f"Error updating Projection, {e}")


"""OverlayProbeMixin — extracted from qt_interface.py per L5 §0.5 decomposition.

Cluster 8: contour overlay toggle, ROI contour load/draw, pixel-probe enable
and result display. 5 methods, ~162 LOC.

Mixin contract:
    Inherits implicit access to the following state, set up by Interface.__init__:
        self._button_toggle_overlay : QPushButton | None
        self._button_pixel_probe    : QPushButton
        self._overlay_on            : bool
        self._overlay_contours      : list | None
        self._overlay_shape         : tuple | None
        self._proc_projector        : QProcess | None
        self.display                : preview widget (has _pixel_probe_enabled, setCursor)
        self.acq_label              : QLabel  (statusbar pixel-probe readout)
        self.image_update_signal    : pyqtSignal

Pure hoist — no behavior change vs. monolith. See spec docs/specs/L5_UI/qt_interface.md.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PyQt5 import QtCore


class OverlayProbeMixin:
    """Cluster 8 — overlay + pixel-probe controls."""

    def _toggle_overlay(self, checked: bool):
        try:
            if not hasattr(self, '_button_toggle_overlay') or self._button_toggle_overlay is None:
                return
            self._button_toggle_overlay.setText("Overlay: On" if checked else "Overlay: Off")
            self._overlay_on = checked
            # Pre-load ROI contours (for any future RTTE/CS preview overlay path)
            if checked and not getattr(self, '_overlay_contours', None):
                self._load_overlay_contours()
            # Push a runtime visible_id update to the projector engine so the
            # toggle takes effect *immediately* — without this, visible_id is
            # only honored at engine launch from the CLI flag, and Overlay Off
            # would persist on-screen until projection is fully restarted.
            try:
                # _proc_projector is a QProcess (PyQt) — uses state(), not poll().
                _proc = getattr(self, '_proc_projector', None)
                _engine_up = False
                if _proc is not None:
                    if hasattr(_proc, 'state'):
                        _engine_up = (int(_proc.state()) != 0)
                    elif hasattr(_proc, 'poll'):
                        _engine_up = (_proc.poll() is None)
                if _engine_up:
                    import numpy as _np
                    from projector_client import ProjectorClient
                    cli = ProjectorClient()
                    proj_w = getattr(cli, 'width', 1920)
                    proj_h = getattr(cli, 'height', 1080)
                    # Black frame just to carry the meta — won't visibly disturb
                    # the current pattern much (one frame at projector cadence).
                    cli.send_gray(_np.zeros((proj_h, proj_w), dtype=_np.uint8),
                                  frame_id=8895, visible_overlay=bool(checked),
                                  immediate=True)
                    try: cli.close()
                    except Exception: pass
                    print(f"[PROJ] Overlay {'ON' if checked else 'OFF'} sent to engine via visible_overlay flag")
            except Exception as e:
                print(f"[PROJ] Overlay runtime toggle send failed: {e}")
            # Force an immediate preview redraw
            try:
                if hasattr(self, 'image_update_signal'):
                    self.update()
            except Exception:
                pass
        except Exception as e:
            print(f"_toggle_overlay error: {e}")

    def _load_overlay_contours(self):
        """Load ROI contours from rois.npz for camera-preview overlay."""
        try:
            import numpy as _np
            candidates = [
                Path(__file__).resolve().parent.parent / "CS" / "data" / "rois.npz",
                Path.cwd() / "data" / "rois.npz",
                Path.cwd() / "rois.npz",
                Path(__file__).resolve().parent.parent / "rois.npz",
            ]
            roi_path = None
            for p in candidates:
                if p.exists():
                    roi_path = str(p)
                    break
            if roi_path is None:
                print("[OVERLAY] No rois.npz found — overlay will be empty")
                self._overlay_contours = []
                return
            data = _np.load(roi_path, allow_pickle=False)
            labels = data.get('labels', None)
            if labels is None:
                print("[OVERLAY] rois.npz has no 'labels' key")
                self._overlay_contours = []
                return
            neuron_ids = data.get('neuron_ids', _np.unique(labels[labels > 0]))
            # Build contour list: [(contour_points, centroid, nid),...]
            import cv2 as _cv2
            contours_list = []
            for nid in neuron_ids:
                roi_mask = (labels == int(nid)).astype(_np.uint8)
                cnts, _ = _cv2.findContours(roi_mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    ys, xs = _np.where(roi_mask)
                    cx, cy = float(xs.mean()), float(ys.mean())
                    contours_list.append((cnts, (cx, cy), int(nid)))
            self._overlay_contours = contours_list
            self._overlay_shape = labels.shape  # (H, W) of the label map
            print(f"[OVERLAY] Loaded {len(contours_list)} ROI contours from {roi_path}")
        except Exception as e:
            print(f"[OVERLAY] Failed to load contours: {e}")
            self._overlay_contours = []

    def _draw_overlay_on_frame(self, frame):
        """Draw ROI contours and ID labels on a camera frame (in-place)."""
        contours = getattr(self, '_overlay_contours', None)
        if not contours:
            return frame
        # Scale contours if frame size differs from label map size
        ov_shape = getattr(self, '_overlay_shape', None)
        h, w = frame.shape[:2]
        sx = sy = 1.0
        if ov_shape is not None and (ov_shape[0] != h or ov_shape[1] != w):
            sy = h / ov_shape[0]
            sx = w / ov_shape[1]
        # Ensure frame is color (3-channel) for drawing
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        for cnts, (cx, cy), nid in contours:
            color = (0, 255, 0)  # green contours
            if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
                import numpy as _np
                scaled = []
                for c in cnts:
                    sc = c.astype(_np.float32)
                    sc[:, :, 0] *= sx
                    sc[:, :, 1] *= sy
                    scaled.append(sc.astype(_np.int32))
                cv2.drawContours(frame, scaled, -1, color, 1)
                tx, ty = int(cx * sx), int(cy * sy)
            else:
                cv2.drawContours(frame, cnts, -1, color, 1)
                tx, ty = int(cx), int(cy)
            cv2.putText(frame, str(nid), (tx - 6, ty + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return frame

    def _toggle_pixel_probe(self, checked: bool):
        """Toggle pixel probe mode on the camera preview."""
        try:
            self._button_pixel_probe.setText("Probe: On" if checked else "Pixel Probe")
            self.display._pixel_probe_enabled = checked
            if checked:
                self.display.setCursor(QtCore.Qt.CrossCursor)
            else:
                self.display.setCursor(QtCore.Qt.OpenHandCursor)
                # Clear the stale probe dot from the projector — otherwise the
                # last bilinear-weighted pixel persists on the DMD and shows up
                # the next time the user enables Overlay or any other action
                # that doesn't push its own frame.
                try:
                    import numpy as _np
                    from projector_client import ProjectorClient
                    cli = ProjectorClient()
                    proj_w = getattr(cli, 'width', 1920)
                    proj_h = getattr(cli, 'height', 1080)
                    blank = _np.zeros((proj_h, proj_w), dtype=_np.uint8)
                    cli.send_gray(blank, frame_id=8889, visible_id=0, immediate=True)
                    try: cli.close()
                    except Exception: pass
                    print("[PROBE] Cleared stale probe pattern from projector")
                except Exception as e:
                    print(f"[PROBE] Could not clear projector: {e}")
        except Exception as e:
            print(f"_toggle_pixel_probe error: {e}")

    def _on_pixel_probe_result(self, x, y, info):
        """Display pixel probe result in the statusbar."""
        try:
            self.acq_label.setText(f"Pixel Probe: ({x}, {y}) {info}")
        except Exception:
            pass

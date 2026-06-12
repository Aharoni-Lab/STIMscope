"""SLCalibrateMixin — extracted from qt_interface.py.

Bundles the two structured-light calibration methods:

* ``_sl_calibrate()`` — end-to-end SL calibration with Gray-code +
  Phase-shift patterns (~246 LOC).
* ``_sl_project_registration()`` — project the prewarped registration
  image after calibration (~86 LOC).

Method bodies are byte-identical to the pre-extraction code at
``qt_interface.py:755-1086`` (commit ``7463a6e``); only the
surrounding module-level frame changed.

Mixin contract (Interface attributes the method reads/writes):
  * ``self._camera`` — image source
  * ``self.projection`` — second-monitor projection window
  * ``self._ensure_projection`` — guards projection availability
  * ``self.sl_decode_done`` — pyqtSignal emitted on completion
  * ``self.message`` / ``self.warning`` — operator-facing surfaces

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
from pathlib import Path

class SLCalibrateMixin:
    """Cluster 15 — structured-light calibration + registration projection."""

    def _sl_calibrate(self):
        """Run Structured-Light calibration end-to-end (Gray + Phase subpixel)."""
        try:
            from calibration import (
                generate_gray_code_patterns,
                generate_phase_shift_patterns,
                save_structured_light_patterns,
            )
        except Exception as e:
            print(f"Structured-light not available: {e}")
            return

        if not self._ensure_projection():
            print("Projection window unavailable.")
            return

        # 1) Generate patterns at projector resolution (Gray + Phase)
        try:
            scr = self.projection.windowHandle().screen() if self.projection.windowHandle() else None
            geo = scr.geometry() if scr else None
            proj_w = geo.width() if geo else 1920
            proj_h = geo.height() if geo else 1080
            gray_patterns = generate_gray_code_patterns(proj_w, proj_h)
            use_phase = getattr(self, '_chk_phase_refine', None) is not None and self._chk_phase_refine.isChecked()
            if use_phase:
                # Enable phase-shift patterns for subpixel refinement
                phase_patterns = generate_phase_shift_patterns(
                    proj_w, proj_h, num_phases=3, cycles_x=1, cycles_y=1, gamma=1.0
                )
                patterns = gray_patterns + phase_patterns
            else:
                patterns = gray_patterns
            pattern_paths = save_structured_light_patterns(patterns)
            print(f"Generated {len(pattern_paths)} structured-light patterns (Gray+Phase)")
        except Exception as e:
            print(f"Failed to generate patterns: {e}")
            return

        # Disable LUT-warp button and show progress while running
        try:
            if hasattr(self, '_button_sl_project_reg') and self._button_sl_project_reg is not None:
                self._button_sl_project_reg.setEnabled(False)
            if getattr(self, '_sl_progress', None):
                self._sl_progress.setVisible(True)
                self._sl_status.setText("Capturing structured-light patterns…")
        except Exception:
            pass

        # 2) Project each pattern and capture a camera frame
        capture_paths = []
        last_pidx = None
        # If using engine, clear any homography so patterns are unwarped on output
        try:
            use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
            if use_engine:
                try:
                    import zmq as _zmq
                    _ctx = _zmq.Context.instance(); _s = _ctx.socket(_zmq.REQ)
                    _s.setsockopt(_zmq.LINGER, 0)
                    _s.connect("tcp://127.0.0.1:5560")
                    _s.send(b"IDENTITY")
                    _ = _s.recv()
                    _s.close()
                except Exception:
                    pass
        except Exception:
            pass
        for idx, (ppath, meta) in enumerate(zip(pattern_paths, patterns)):
            try:
                # Prefer in-memory pattern image to avoid disk I/O latency
                img = None
                try:
                    img = meta.get("image", None)
                except Exception:
                    img = None
                if img is None:
                    img = cv2.imread(ppath, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                # If projection engine is running and triggers are armed, stream via ZMQ to sync with projector
                use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
                if use_engine:
                    try:
                        from projector_client import ProjectorClient
                        # Projector engine expects 1920x1080 luminance frames; client resizes as needed
                        client = ProjectorClient()
                        # Pace strictly: wait for next trigger from last_pidx, then send one frame, then wait until that vis_id appears
                        if last_pidx is None:
                            client.wait_next_trigger(0, timeout_ms=500)
                        else:
                            client.wait_next_trigger(last_pidx, timeout_ms=500)
                        # Force engine overlay OFF for SL, and request immediate scheduling
                        client.send_gray(img, frame_id=idx+1, visible_id=0, immediate=True)
                        matched = client.wait_visible(idx+1, timeout_ms=500)
                        if matched is not None:
                            last_pidx = matched
                        # Allow camera to expose the just-shown pattern before snapshot
                        try:
                            QtCore.QThread.msleep(60)
                        except Exception:
                            pass
                        client.close()
                    except Exception as ez:
                        print(f"[SL] ZMQ send failed, falling back to local display: {ez}")
                        try:
                            self.projection.show_image_raw_no_warp_no_flip(img)
                        except Exception:
                            self.projection.show_image_fullscreen_on_second_monitor(img, None)
                else:
                    # Local path without engine
                    try:
                        self.projection.show_image_raw_no_warp_no_flip(img)
                    except Exception:
                        self.projection.show_image_fullscreen_on_second_monitor(img, None)
                # Allow minimal UI processing without delaying engine-paced path
                QtCore.QCoreApplication.processEvents()
                if not use_engine:
                    QtCore.QThread.msleep(40)
                # Capture a frame
                save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                os.makedirs(save_dir, exist_ok=True)
                cap_path = os.path.join(save_dir, f"sl_cap_{idx:03d}.png")
                if hasattr(self._camera, "snapshot"):
                    self._camera.snapshot(cap_path)
                    capture_paths.append(cap_path)
                else:
                    # As a fallback, mark missing
                    capture_paths.append("")
            except Exception as e:
                print(f"Pattern {idx} projection/capture failed: {e}")

        # 3) Decode LUTs (offload to background thread to keep GUI responsive)
        try:
            def _sl_decode_worker(paths, pats, pw, ph, asset_dir):
                try:
                    import numpy as _np
                    import cv2 as _cv2
                    from calibration import (
                        decode_gray_code_from_files as _decode_gray,
                        decode_phase_shift_from_files as _decode_phase,
                        invert_cam_to_proj_lut as _invert,
                    )
                    # Split captures: Gray-code vs Phase (optional)
                    pairs = [(p, m) for p, m in zip(paths, pats)]
                    gray_pairs  = [(p, m) for (p, m) in pairs if isinstance(m, dict) and ('bit' in m)]
                    phase_pairs = [(p, m) for (p, m) in pairs if isinstance(m, dict) and (m.get('type') == 'phase')]
                    paths_gray  = [p for (p, _) in gray_pairs]
                    meta_gray   = [m for (_, m) in gray_pairs]
                    paths_phase = [p for (p, _) in phase_pairs]
                    meta_phase  = [m for (_, m) in phase_pairs]

                    cam_h, cam_w = 1080, 1920
                    for _fp in reversed(paths_gray):  # Only check Gray patterns
                        if not _fp:
                            continue
                        _img = _cv2.imread(_fp, _cv2.IMREAD_GRAYSCALE)
                        if _img is not None:
                            cam_h, cam_w = _img.shape[:2]
                            break
                    print(f"[SL] Decoding Gray-code at {cam_w}x{cam_h} → proj {pw}x{ph}…")
                    proj_x_of_cam, proj_y_of_cam = _decode_gray(paths_gray, meta_gray, cam_h, cam_w, pw, ph)
                    
                    # Optionally apply phase-shift refinement only if present and valid
                    try:
                        if len(paths_phase) > 0 and len(meta_phase) > 0:
                            print("[SL] Decoding Phase-shift for subpixel refinement…")
                            px_phase, py_phase, ax, ay = _decode_phase(paths_phase, meta_phase, cam_h, cam_w, pw, ph, num_phases=3, amp_thresh=5.0)
                            # Adaptive amplitude gating: use stricter threshold if coverage is low
                            amp_thr = 5.0
                            # Estimate potential coverage
                            cov_x = float((_np.sum(ax > amp_thr)) / (ax.size if ax.size else 1))
                            cov_y = float((_np.sum(ay > amp_thr)) / (ay.size if ay.size else 1))
                            # If coverage < 20%, try lower threshold 3.0 to rescue weak areas
                            if cov_x < 0.2 or cov_y < 0.2:
                                amp_thr = 3.0
                            use_x = (px_phase >= 0.0) & (ax > amp_thr)
                            use_y = (py_phase >= 0.0) & (ay > amp_thr)
                            applied_x = int(_np.sum(use_x)); applied_y = int(_np.sum(use_y))
                            # Only apply if meaningful coverage (e.g., >10% of pixels)
                            min_cov = 0.10
                            if (applied_x / float(px_phase.size if px_phase.size else 1) > min_cov) or (applied_y / float(py_phase.size if py_phase.size else 1) > min_cov):
                                proj_x_of_cam = proj_x_of_cam.astype(_np.float32, copy=True)
                                proj_y_of_cam = proj_y_of_cam.astype(_np.float32, copy=True)
                                # Phase provides subpixel refinement WITHIN Gray code cells.
                                # Keep the Gray code integer part, replace only the fractional part
                                # from phase. Only apply where Gray code and phase agree within 1 pixel.
                                if applied_x > 0:
                                    gray_int_x = _np.floor(proj_x_of_cam[use_x])
                                    phase_frac_x = px_phase[use_x] - _np.floor(px_phase[use_x])
                                    refined_x = gray_int_x + phase_frac_x
                                    # Only apply where phase agrees with Gray code (within 1.5 pixels)
                                    agree_x = _np.abs(refined_x - proj_x_of_cam[use_x]) < 1.5
                                    temp = proj_x_of_cam[use_x].copy()
                                    temp[agree_x] = refined_x[agree_x]
                                    proj_x_of_cam[use_x] = temp
                                if applied_y > 0:
                                    gray_int_y = _np.floor(proj_y_of_cam[use_y])
                                    phase_frac_y = py_phase[use_y] - _np.floor(py_phase[use_y])
                                    refined_y = gray_int_y + phase_frac_y
                                    agree_y = _np.abs(refined_y - proj_y_of_cam[use_y]) < 1.5
                                    temp = proj_y_of_cam[use_y].copy()
                                    temp[agree_y] = refined_y[agree_y]
                                    proj_y_of_cam[use_y] = temp
                                n_refined_x = int(agree_x.sum()) if applied_x > 0 else 0
                                n_refined_y = int(agree_y.sum()) if applied_y > 0 else 0
                                print(f"[SL] Phase refinement applied: {n_refined_x}/{applied_x} X px, {n_refined_y}/{applied_y} Y px (thr={amp_thr})")
                            else:
                                print(f"[SL] Phase refinement skipped due to low coverage (X={applied_x}, Y={applied_y})")
                        else:
                            print("[SL] Phase patterns not included; using Gray-code only")
                    except Exception as _pe:
                        print(f"[SL] Phase refinement skipped: {_pe}")
                        print("[SL] Using Gray-code only (phase refinement failed)")
                    _np.save("/".join([asset_dir, "proj_from_cam_x.npy"]), proj_x_of_cam)
                    _np.save("/".join([asset_dir, "proj_from_cam_y.npy"]), proj_y_of_cam)
                    inv_x, inv_y = _invert(proj_x_of_cam, proj_y_of_cam, pw, ph)
                    _np.save("/".join([asset_dir, "cam_from_proj_x.npy"]), inv_x)
                    _np.save("/".join([asset_dir, "cam_from_proj_y.npy"]), inv_y)
                    
                    # Generate diagnostic visualization
                    try:
                        from calibration import visualize_lut_quality
                        diag_path = "/".join([asset_dir, "lut_diagnostic.png"])
                        visualize_lut_quality(inv_x, inv_y, diag_path)
                    except Exception as diag_e:
                        print(f"Could not generate diagnostic: {diag_e}")
                    
                    print("✅ Structured-light LUTs (subpixel) saved (background)")
                    try:
                        # Notify GUI thread
                        self.sl_decode_done.emit(True, "LUTs saved")
                    except Exception:
                        pass
                except Exception as _e:
                    print(f"Structured-light decoding failed: {_e}")
                    try:
                        self.sl_decode_done.emit(False, str(_e))
                    except Exception:
                        pass

            import threading as _th
            _th.Thread(target=_sl_decode_worker, args=(capture_paths, patterns, proj_w, proj_h, self._camera.asset_dir), daemon=True).start()
            print("[SL] Decoding LUTs in background… GUI remains responsive")
        except Exception as e:
            print(f"Structured-light decoding thread failed to start: {e}")
    
    def _sl_project_registration(self):
        """Prewarp and project the custom registration image using LUTs."""
        try:
            from calibration import prewarp_with_inverse_lut
        except Exception as e:
            print(f"Structured-light prewarp not available: {e}")
            return
        if not self._ensure_projection():
            print("Projection window unavailable.")
            return
        try:
            # Load LUTs
            asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent / "Assets" / "Generated").resolve()))
            inv_x = np.load("/".join([asset_dir, "cam_from_proj_x.npy"]))
            inv_y = np.load("/".join([asset_dir, "cam_from_proj_y.npy"]))
            proj_h, proj_w = inv_x.shape[:2]
            # Load registration image in camera space (same as camera preview size preferred). If sizes differ, we will scale.
            img_path = (Path(asset_dir).parent / "Generated" / "custom_registration_image.png").resolve()
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Registration image not readable: {img_path}")
                return
            # Resize registration to camera frame size if we can detect it from a snapshot
            cam_h, cam_w = img.shape[:2]
            try:
                # Try loading a recent snapshot to infer true camera dims
                save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                candidates = sorted([p for p in os.listdir(save_dir) if p.endswith('.png')])
                for name in reversed(candidates):
                    probe = cv2.imread(os.path.join(save_dir, name), cv2.IMREAD_GRAYSCALE)
                    if probe is not None:
                        cam_h, cam_w = probe.shape[:2]
                        break
                if (img.shape[1], img.shape[0]) != (cam_w, cam_h):
                    img = cv2.resize(img, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
            # Prewarp with error handling
            try:
                warped = prewarp_with_inverse_lut(img, inv_x, inv_y, proj_w, proj_h)
            except Exception as warp_e:
                print(f"Warping failed: {warp_e}")
                # Try simple resize as fallback
                warped = cv2.resize(img, (proj_w, proj_h), interpolation=cv2.INTER_LINEAR)
                print("Using simple resize as fallback")
            
            # Prefer projection engine via ZMQ if running; ensures sync with triggers
            use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
            if use_engine:
                try:
                    from projector_client import ProjectorClient
                    # Engine expects 1920x1080; client will resize
                    client = ProjectorClient()
                    # Clear engine homography so the prewarped image is not warped again
                    try:
                        import zmq as _zmq
                        _ctx = _zmq.Context.instance(); _s = _ctx.socket(_zmq.REQ)
                        _s.setsockopt(_zmq.LINGER, 0)
                        _s.connect("tcp://127.0.0.1:5560"); _s.send(b"IDENTITY"); _ = _s.recv(); _s.close()
                    except Exception:
                        pass
                    if getattr(self, '_button_hw_trig', None) and self._button_hw_trig.isChecked():
                        client.enable_gpio_trigger(22)
                    client.send_gray(
                        warped,
                        frame_id=9999,
                        visible_id=int(bool(self._button_toggle_overlay.isChecked()))
                    )
                    # Optionally wait for visibility, but pulsing is now handled by background subscriber when enabled
                    _ = client.wait_visible(9999, timeout_ms=250)
                    client.close()
                except Exception as ez:
                    print(f"[SL] ZMQ send failed, falling back to local display: {ez}")
                    try:
                        self.projection.show_image_raw_no_warp_no_flip(warped)
                    except Exception:
                        self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            else:
                # Project raw without flip/warp (LUT already maps correctly)
                try:
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            print("✅ Projected LUT-prewarped registration")
        except Exception as e:
            print(f"LUT projection failed: {e}")


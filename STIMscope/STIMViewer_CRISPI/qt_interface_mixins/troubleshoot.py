"""TroubleshootMixin — extracted from qt_interface.py.

Extracts the 1,412-LOC ``_open_troubleshoot_window`` method into a
dedicated mixin so the parent Interface class drops below the
§3.2 Hard band. Method body is byte-identical to the pre-extraction
code at ``qt_interface.py:1295-2706`` (commit ``39a188b``); only the
surrounding module-level frame changed.

The method itself is a single huge dialog factory with many nested
closures (engine monitor, FPS sampling, LUT diagnostics, pixel-probe
diagnostics, calibration-character, dot-array test, edge-strip test,
round-trip evaluation, etc.). Per §3.2 cohesion-over-arbitrary-split,
the closures stay together inside the method — they share dialog
widgets + locks by reference. Future-iteration recovery path: extract
each closure-group into its own helper function or small class so the
method body can be re-read in one pass.

Mixin contract (Interface attributes the method reads/writes through
``self.``):
  * ``self._test_hw_trigger_pulse`` — invoked from a QPushButton
  * ``self._camera`` — read for FPS, exposure, LUT diagnostic shapes
  * ``self.display`` — read to seed the LUT plot
  * ``self._proc_projector`` / ``self._proc_dlpc`` — QProcess refs
  * ``self._helper_python_path_for_i2c`` — invoked to spawn engine sub
  * ``self.is_gui`` — used by some sub-callbacks
  * Many nested closures bind local-frame state; nothing escapes.

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


class TroubleshootMixin:
    """Cluster 9 — the troubleshooting dialog with live engine monitor."""

    # ---------------- Troubleshooting Window ----------------
    def _open_troubleshoot_window(self):
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QMessageBox
            import psutil
            import os
            import cv2
            import numpy as _np
        except Exception as e:
            print(f"Troubleshooting UI error: {e}")
            return

        # Optional plotting
        try:
            import pyqtgraph as pg
            _HAS_PG = True
        except Exception:
            _HAS_PG = False

        dlg = QDialog(self)
        dlg.setWindowTitle("Troubleshooting")
        dlg.setMinimumSize(680, 420)
        lay = QVBoxLayout(dlg)

        # Row: quick actions & engine monitor toggle
        row = QHBoxLayout()
        btn_test = QtWidgets.QPushButton("Test HW Trigger Out Pulse")
        btn_test.clicked.connect(self._test_hw_trigger_pulse)
        btn_mon = QtWidgets.QPushButton("Start Engine Monitor")
        btn_mon.setCheckable(True)
        status_lbl = QLabel("Engine: idle")
        last_lbl = QLabel("Last: pidx=-- vis=-- rate=-- Hz")
        # Trigger indicator button (non-interactive)
        ind_btn = QtWidgets.QPushButton("Projector Trigger: OFF")
        ind_btn.setEnabled(False)
        ind_btn.setStyleSheet("QPushButton{background-color:#ff4d4f; color:white; border-radius:6px; padding:4px 8px;}")
        row.addWidget(btn_test)
        row.addSpacing(10)
        row.addWidget(btn_mon)
        row.addSpacing(10)
        row.addWidget(status_lbl)
        row.addSpacing(10)
        row.addWidget(ind_btn)
        row.addStretch()
        lay.addLayout(row)

        # Live graphs (CPU, GPU, Mem)
        grid = QGridLayout()
        if _HAS_PG:
            pg.setConfigOptions(antialias=True)
            def _small_plot(title, pen_color):
                w = pg.PlotWidget()
                w.setTitle(title)
                w.setMinimumSize(160, 100)
                w.setMaximumHeight(110)
                c = w.plot(pen=pg.mkPen(pen_color, width=2))
                w.getPlotItem().hideButtons()
                w.getPlotItem().setLabel('bottom', '')
                w.getPlotItem().setLabel('left', '')
                w.getPlotItem().getAxis('left').setStyle(showValues=False)
                w.getPlotItem().getAxis('bottom').setStyle(showValues=False)
                return w, c
            cpu_plot, cpu_curve = _small_plot("CPU %", '#2ecc71')
            mem_plot, mem_curve = _small_plot("Mem %", '#3498db')
            gpu_plot, gpu_curve = _small_plot("GPU %", '#9b59b6')
            grid.addWidget(cpu_plot, 0, 0)
            grid.addWidget(mem_plot, 0, 1)
            grid.addWidget(gpu_plot, 0, 2)
        else:
            lbl_cpu = QLabel("CPU: -- %")
            lbl_mem = QLabel("Mem: -- %")
            lbl_gpu = QLabel("GPU: -- %")
            grid.addWidget(lbl_cpu, 0, 0)
            grid.addWidget(lbl_mem, 0, 1)
            grid.addWidget(lbl_gpu, 0, 2)
        lay.addLayout(grid)

        # ---------------- Structured-Light Validation ----------------
        def _load_luts():
            asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent.parent / "Assets" / "Generated").resolve()))
            xfp = os.path.join(asset_dir, "cam_from_proj_x.npy")
            yfp = os.path.join(asset_dir, "cam_from_proj_y.npy")
            if not (os.path.isfile(xfp) and os.path.isfile(yfp)):
                QMessageBox.warning(dlg, "LUTs Missing", "cam_from_proj_{x,y}.npy not found. Run Structured-Light calibration first.")
                return None, None, asset_dir
            try:
                inv_x = _np.load(xfp).astype(_np.float32)
                inv_y = _np.load(yfp).astype(_np.float32)
                return inv_x, inv_y, asset_dir
            except Exception as e:
                QMessageBox.critical(dlg, "LUT Load Error", str(e))
                return None, None, asset_dir

        from PyQt5.QtWidgets import QGridLayout as _QGrid
        sl_row = _QGrid()
        sl_title = QLabel("Structured-Light Validation:")
        try: sl_title.setStyleSheet("font-weight:600;")
        except Exception: pass
        lay.addWidget(sl_title)

        btn_diag = QPushButton("LUT Diagnostics")
        btn_proj = QPushButton("Project Grid (LUT)")
        btn_eval = QPushButton("Capture + Evaluate")
        btn_rterr = QPushButton("Round-Trip Error (Maps)")
        btn_probe = QPushButton("Pixel Probe (1px)")
        btn_dots  = QPushButton("Dot Array Test")
        btn_rtphy = QPushButton("Round-Trip (Physical)")
        btn_edge  = QPushButton("Edge Strip Test")
        btn_calib_char = QPushButton("Calib Grid Characterization")
        # arrange buttons in two rows
        btns = [btn_diag, btn_proj, btn_eval, btn_rterr, btn_probe, btn_dots, btn_rtphy, btn_edge, btn_calib_char]
        for i, b in enumerate(btns):
            r = i // 4
            c = i % 4
            sl_row.addWidget(b, r, c)
        lay.addLayout(sl_row)

        # Zoomable preview (with mouse wheel zoom + double-click reset)
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
        class _ZoomGraphicsView(QGraphicsView):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                try:
                    self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
                    self.setDragMode(QGraphicsView.ScrollHandDrag)
                except Exception:
                    pass
            def wheelEvent(self, ev):
                try:
                    angle = ev.angleDelta().y() / 120.0
                    factor = 1.25 ** max(-3.0, min(3.0, angle))
                    self.scale(factor, factor)
                    ev.accept()
                except Exception:
                    super().wheelEvent(ev)
            def mouseDoubleClickEvent(self, ev):
                try:
                    self.setTransform(QtGui.QTransform())
                    # Fit current pixmap item if present
                    items = self.scene().items()
                    for it in items:
                        if isinstance(it, QGraphicsPixmapItem):
                            self.fitInView(it, Qt.KeepAspectRatio)
                            break
                    ev.accept()
                except Exception:
                    super().mouseDoubleClickEvent(ev)

        sl_scene = QGraphicsScene()
        sl_view = _ZoomGraphicsView(sl_scene)
        sl_view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, on=True)
        sl_view.setMinimumSize(360, 220)
        sl_view.setStyleSheet("border:1px solid #d1d1d6;")
        sl_pix = QGraphicsPixmapItem()
        sl_scene.addItem(sl_pix)
        lay.addWidget(sl_view)
        # Save current calibration preview (original resolution) as TIFF
        try:
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            btn_save_tiff = QPushButton("Save Current View (TIFF)")
            try:
                btn_save_tiff.setToolTip("Save the current calibration preview image at original resolution in.tiff format")
            except Exception:
                pass
            def _on_save_current_tiff():
                try:
                    pm = sl_pix.pixmap()
                    if pm is None or pm.isNull():
                        QMessageBox.warning(dlg, "Save Image", "No image available to save.")
                        return
                    try:
                        save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                    except Exception:
                        save_dir = './Saved_Media'
                    try:
                        os.makedirs(save_dir, exist_ok=True)
                    except Exception:
                        pass
                    default_name = time.strftime("calibration_%Y%m%d_%H%M%S.tiff")
                    fp, _ = QFileDialog.getSaveFileName(
                        dlg,
                        "Save Calibration Image (TIFF)",
                        os.path.join(save_dir, default_name),
                        "TIFF Image (*.tiff *.tif);;All Files (*)"
                    )
                    if not fp:
                        return
                    # Ensure.tiff extension
                    fpl = fp.lower()
                    if not (fpl.endswith(".tiff") or fpl.endswith(".tif")):
                        fp = fp + ".tiff"
                    ok = False
                    try:
                        ok = pm.save(fp, "TIFF")
                    except Exception:
                        ok = False
                    if not ok:
                        try:
                            qimg = pm.toImage()
                            ok = qimg.save(fp, "TIFF")
                        except Exception:
                            ok = False
                    if not ok:
                        QMessageBox.warning(dlg, "Save Failed", "Could not save image to TIFF.")
                        return
                    QMessageBox.information(dlg, "Saved", f"Saved image:\n{fp}")
                except Exception as _e:
                    try:
                        QMessageBox.warning(dlg, "Save Failed", str(_e))
                    except Exception:
                        print(f"[TSAVE] Save failed: {_e}")
            btn_save_tiff.clicked.connect(_on_save_current_tiff)
            lay.addWidget(btn_save_tiff)
        except Exception as _e:
            print(f"[TSAVE] Save button init failed: {_e}")

        # Metrics output (textbox - not on top of the image)
        metrics_lbl = QLabel("Metrics / Logs:")
        metrics_box = QtWidgets.QPlainTextEdit(dlg)
        try:
            metrics_box.setReadOnly(True)
            metrics_box.setMaximumHeight(120)
        except Exception:
            pass
        lay.addWidget(metrics_lbl)
        lay.addWidget(metrics_box)

        def _to_pix(img_bgr):
            try:
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = img_bgr
            h, w = rgb.shape[:2]
            from PyQt5.QtGui import QImage
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())

        def _on_lut_diag():
            try:
                from calibration import visualize_lut_quality as _viz
            except Exception:
                _viz = None
            inv_x, inv_y, asset_dir = _load_luts()
            if inv_x is None or _viz is None:
                return
            diag = _viz(inv_x, inv_y, os.path.join(asset_dir, "lut_diagnostics.png"))
            try:
                pm = _to_pix(diag)
                sl_pix.setPixmap(pm)
                sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
            except Exception:
                pass

        def _infer_cam_size():
            try:
                save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                names = sorted([p for p in os.listdir(save_dir) if p.endswith('.png')])
                for nm in reversed(names):
                    fp = os.path.join(save_dir, nm)
                    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        return img.shape[1], img.shape[0]
            except Exception:
                pass
            try:
                return int(self._camera.sensor_width), int(self._camera.sensor_height)
            except Exception:
                return 1920, 1080

        def _make_cam_grid(cam_w, cam_h, cell=32, pitch=None):
            """
            Build a binary checkerboard-like grid image in camera space.
            - cell: side length of each bright square in pixels
            - pitch: center-to-center spacing (>= cell). If None or <= cell, fall back to contiguous chessboard.
            """
            g = _np.zeros((cam_h, cam_w), _np.uint8)
            cell = int(max(1, cell))
            if pitch is None or int(pitch) <= cell:
                # Classic contiguous checkerboard
                for y in range(0, cam_h, cell):
                    for x in range(0, cam_w, cell):
                        if ((x//cell)+(y//cell)) & 1:
                            y1 = min(y+cell, cam_h)
                            x1 = min(x+cell, cam_w)
                            g[y:y1, x:x1] = 255
                return g
            # Spaced squares with given pitch (>= cell)
            pitch = int(max(cell, int(pitch)))
            for y in range(0, cam_h, pitch):
                for x in range(0, cam_w, pitch):
                    # Alternate parity across pitched grid cells
                    if ((x//pitch) + (y//pitch)) & 1:
                        y1 = min(y+cell, cam_h)
                        x1 = min(x+cell, cam_w)
                        g[y:y1, x:x1] = 255
            return g

        def _on_project_grid():
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                _prewarp = None
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None or _prewarp is None:
                return
            cam_w, cam_h = _infer_cam_size()
            try:
                _cell = max(1, int(sp_cell.value()))
            except Exception:
                _cell = 16
            try:
                _pitch = max(_cell, int(sp_pitch.value()))
            except Exception:
                _pitch = _cell
            grid = _make_cam_grid(cam_w, cam_h, cell=_cell, pitch=_pitch)
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR), inv_x, inv_y, proj_w, proj_h)
            # Prefer sending to the projection engine to avoid GL/X context conflicts
            use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
            if use_engine:
                try:
                    # Clear H so prewarped content is not warped again
                    import zmq as _zmq
                    _ctx = _zmq.Context.instance(); _s = _ctx.socket(_zmq.REQ)
                    _s.setsockopt(_zmq.LINGER, 0)
                    _s.connect("tcp://127.0.0.1:5560"); _s.send(b"IDENTITY"); _ = _s.recv(); _s.close()
                except Exception:
                    pass
                try:
                    from projector_client import ProjectorClient
                    client = ProjectorClient()
                    # Engine expects 1920x1080 luminance; client will resize.
                    client.send_gray(warped, frame_id=7777, visible_id=0, immediate=True)
                    client.close()
                    return
                except Exception:
                    pass
            # Fallback: draw via Qt projector window
            try:
                self.projection.show_image_raw_no_warp_no_flip(warped)
            except Exception:
                self.projection.show_image_fullscreen_on_second_monitor(warped, None)

        # ---------------- Homography (H) Validation (simple calibration) ----------------
        h_title = QLabel("Calibration (H) Validation:")
        try: h_title.setStyleSheet("font-weight:600;")
        except Exception: pass
        lay.addWidget(h_title)
        h_row = _QGrid()
        btn_h_proj = QPushButton("Project Grid (H)")
        btn_h_eval = QPushButton("Capture + Evaluate (H)")
        btn_h_dots = QPushButton("Dot Array Test (H)")
        h_row.addWidget(btn_h_proj, 0, 0)
        h_row.addWidget(btn_h_eval, 0, 1)
        h_row.addWidget(btn_h_dots, 0, 4)
        # Grid pitch control
        lbl_cell = QLabel("Cell (px):")
        sp_cell = QtWidgets.QSpinBox(dlg)
        try:
            sp_cell.setRange(1, 256)
            sp_cell.setSingleStep(1)
            sp_cell.setValue(16)
            sp_cell.setToolTip("Grid square size in camera pixels")
        except Exception:
            pass
        h_row.addWidget(lbl_cell, 0, 2)
        h_row.addWidget(sp_cell, 0, 3)
        # Pitch control (>= Cell)
        lbl_pitch = QLabel("Pitch (px):")
        sp_pitch = QtWidgets.QSpinBox(dlg)
        try:
            sp_pitch.setRange(1, 512)
            sp_pitch.setSingleStep(1)
            sp_pitch.setValue(int(sp_cell.value()))
            sp_pitch.setToolTip("Center-to-center spacing of squares; must be >= Cell")
        except Exception:
            pass
        def _sync_pitch_min():
            try:
                sp_pitch.setMinimum(int(sp_cell.value()))
                if int(sp_pitch.value()) < int(sp_cell.value()):
                    sp_pitch.setValue(int(sp_cell.value()))
            except Exception:
                pass
        try:
            sp_cell.valueChanged.connect(_sync_pitch_min)
        except Exception:
            pass
        h_row.addWidget(lbl_pitch, 0, 5)
        h_row.addWidget(sp_pitch, 0, 6)
        lay.addLayout(h_row)

        def _on_h_project_grid():
            try:
                import cv2
                import numpy as _np
            except Exception:
                QMessageBox.warning(dlg, "Dependencies", "OpenCV not available")
                return
            H = getattr(self._camera, 'translation_matrix', None)
            if not isinstance(H, _np.ndarray) or H.shape != (3, 3):
                QMessageBox.warning(dlg, "Calibration", "No homography available. Run Calibrate first.")
                return
            cam_w, cam_h = _infer_cam_size()
            try:
                _cell = max(1, int(sp_cell.value()))
            except Exception:
                _cell = 16
            try:
                _pitch = max(_cell, int(sp_pitch.value()))
            except Exception:
                _pitch = _cell
            grid = _make_cam_grid(cam_w, cam_h, cell=_cell, pitch=_pitch)
            img = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
            # Ensure local projector window exists and use H path (no LUT)
            if not self._ensure_projection():
                # Fallback: show warped preview inside troubleshooting
                try:
                    h, w = img.shape[:2]
                    prev = cv2.warpPerspective(img, H.astype(_np.float64), (w, h))
                    pm = _to_pix(prev); sl_pix.setPixmap(pm); sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                except Exception:
                    QMessageBox.warning(dlg, "Projection", "Projection window unavailable")
                return
            try:
                self.projection.show_image_fullscreen_on_second_monitor(img, H)
            except Exception as e:
                # Also show preview in troubleshooting for confirmation
                try:
                    h, w = img.shape[:2]
                    prev = cv2.warpPerspective(img, H.astype(_np.float64), (w, h))
                    pm = _to_pix(prev); sl_pix.setPixmap(pm); sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                except Exception:
                    pass
                QMessageBox.warning(dlg, "Projection", str(e))

        # Hold last H evaluation images for mode switching
        _h_last_grid = {'img': None}
        _h_last_cap = {'img': None}
        _h_last_overlap = {'img': None}
        # Track whether we've fitted the view once for this set; preserves zoom on toggles
        _h_view_fit = {'done': False}

        # Crosstalk metric: mean/p95 of neighbor(off)/on intensities across pitched grid
        def _compute_crosstalk(cap_gray, cell, pitch):
            try:
                import numpy as _np
            except Exception:
                return None
            if cap_gray is None or getattr(cap_gray, 'ndim', 0) != 2:
                return None
            h, w = cap_gray.shape
            cell = int(max(1, int(cell)))
            pitch = int(max(cell, int(pitch)))
            img = cap_gray.astype(_np.float32)
            ratios = []
            on_means = []
            off_means = []
            for y0 in range(0, h - cell + 1, pitch):
                for x0 in range(0, w - cell + 1, pitch):
                    if ((x0 // pitch) + (y0 // pitch)) & 1:
                        on_roi = img[y0:y0+cell, x0:x0+cell]
                        on_mean = float(on_roi.mean())
                        if on_mean <= 1e-6:
                            continue
                        for dx, dy in ((pitch,0),(-pitch,0),(0,pitch),(0,-pitch)):
                            xn = x0 + dx; yn = y0 + dy
                            if xn < 0 or yn < 0 or xn + cell > w or yn + cell > h:
                                continue
                            off_roi = img[yn:yn+cell, xn:xn+cell]
                            off_mean = float(off_roi.mean())
                            ratios.append(off_mean / on_mean)
                            on_means.append(on_mean)
                            off_means.append(off_mean)
            if not ratios:
                return None
            ratios = _np.array(ratios, dtype=_np.float32)
            return {
                'ratio_mean': float(_np.mean(ratios)),
                'ratio_p95': float(_np.percentile(ratios, 95)),
                'samples': int(ratios.size),
                'on_mean': float(_np.mean(on_means)) if on_means else 0.0,
                'off_mean': float(_np.mean(off_means)) if off_means else 0.0
            }

        def _update_h_preview(mode: str):
            src = None
            if mode == 'ref' and _h_last_grid['img'] is not None:
                src = _h_last_grid['img']
            elif mode == 'cap' and _h_last_cap['img'] is not None:
                src = _h_last_cap['img']
            elif mode == 'ov' and _h_last_overlap['img'] is not None:
                src = _h_last_overlap['img']
            if src is not None:
                try:
                    pm = _to_pix(src)
                    sl_pix.setPixmap(pm)
                    if not _h_view_fit['done']:
                        sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                        _h_view_fit['done'] = True
                except Exception:
                    pass

        def _on_h_capture_eval():
            try:
                import cv2
                import numpy as _np
                import time as _t
            except Exception:
                QMessageBox.warning(dlg, "Dependencies", "OpenCV not available")
                return
            H = getattr(self._camera, 'translation_matrix', None)
            if not isinstance(H, _np.ndarray) or H.shape != (3, 3):
                QMessageBox.warning(dlg, "Calibration", "No homography available. Run Calibrate first.")
                return
            cam_w, cam_h = _infer_cam_size()
            try:
                _cell = max(1, int(sp_cell.value()))
            except Exception:
                _cell = 16
            try:
                _pitch = max(_cell, int(sp_pitch.value()))
            except Exception:
                _pitch = _cell
            grid = _make_cam_grid(cam_w, cam_h, cell=_cell, pitch=_pitch)
            img = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
            if self._ensure_projection():
                try:
                    self.projection.show_image_fullscreen_on_second_monitor(img, H)
                    _t.sleep(0.15)
                except Exception:
                    pass
            cap = _capture_gray()
            if cap is None:
                QMessageBox.warning(dlg, "Capture Failed", "No camera snapshot available")
                return
            if cap.shape != grid.shape:
                try:
                    cap = cv2.resize(cap, (grid.shape[1], grid.shape[0]), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            # Crosstalk (report in textbox, not overlay)
            try:
                ctk = _compute_crosstalk(cap, _cell, _pitch)
                if ctk:
                    metrics_box.appendPlainText(
                        f"Crosstalk (H): cell={_cell}px, pitch={_pitch}px -> mean={ctk['ratio_mean']*100:.1f}%, "
                        f"p95={ctk['ratio_p95']*100:.1f}% (N={ctk['samples']})"
                    )
            except Exception as _e:
                try:
                    metrics_box.appendPlainText(f"Crosstalk (H) error: {_e}")
                except Exception:
                    pass
            # Threshold to binary masks
            try:
                _, cap_bin = cv2.threshold(cap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                cap_bin = (cap > 128).astype(_np.uint8) * 255
            grid_bin = (grid > 128).astype(_np.uint8) * 255
            diff = (cap_bin.astype(_np.int16) - grid_bin.astype(_np.int16)).astype(_np.float32)
            mse = float(_np.mean((diff/255.0)**2)) * (255.0*255.0)
            psnr = 99.0 if mse <= 1e-9 else float(10.0 * _np.log10((255.0*255.0)/mse))
            # Build color-coded overlap: green where both 1, red where mismatch, black elsewhere
            both = ((cap_bin == 255) & (grid_bin == 255))
            xor  = ((cap_bin == 255) ^ (grid_bin == 255))
            vis = _np.zeros((cam_h, cam_w, 3), _np.uint8)
            vis[both] = (0, 255, 0)      # green (BGR)
            vis[xor]  = (0, 0, 255)      # red (BGR)
            try:
                _h_last_grid['img'] = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
                _h_last_cap['img']  = cv2.cvtColor(_np.clip(cap, 0, 255).astype(_np.uint8), cv2.COLOR_GRAY2BGR)
                _h_last_overlap['img'] = vis
                # Reset fit for new images; subsequent toggles preserve zoom
                _h_view_fit['done'] = False
                _update_h_preview('ov')
            except Exception:
                pass

        def _on_h_dot_array_test():
            try:
                import cv2
                import numpy as _np
                import time as _t
            except Exception:
                QMessageBox.warning(dlg, "Dependencies", "OpenCV not available")
                return
            H = getattr(self._camera, 'translation_matrix', None)
            if not isinstance(H, _np.ndarray) or H.shape != (3, 3):
                QMessageBox.warning(dlg, "Calibration", "No homography available. Run Calibrate first.")
                return
            cam_w, cam_h = _infer_cam_size()
            try:
                pitch = max(1, int(sp_cell.value()))
            except Exception:
                pitch = 16
            # Build dot array in camera space
            ref = _np.zeros((cam_h, cam_w), _np.uint8)
            # Choose a conservative radius relative to pitch
            radius = max(2, int(round(pitch * 0.18)))
            try:
                for y in range(radius + 1, cam_h - radius - 1, pitch):
                    for x in range(radius + 1, cam_w - radius - 1, pitch):
                        cv2.circle(ref, (int(x), int(y)), radius, 255, thickness=-1)
            except Exception:
                # Fallback: sparse centers without cv2
                ref[::pitch, ::pitch] = 255
            img = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
            if self._ensure_projection():
                try:
                    self.projection.show_image_fullscreen_on_second_monitor(img, H)
                    _t.sleep(0.15)
                except Exception:
                    pass
            cap = _capture_gray()
            if cap is None:
                QMessageBox.warning(dlg, "Capture Failed", "No camera snapshot available")
                return
            if cap.shape != ref.shape:
                try:
                    cap = cv2.resize(cap, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            # Threshold both
            try:
                _, cap_bin = cv2.threshold(cap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                cap_bin = (cap > 128).astype(_np.uint8) * 255
            ref_bin = (ref > 128).astype(_np.uint8) * 255
            # Compute simple metrics
            diff = (cap_bin.astype(_np.int16) - ref_bin.astype(_np.int16)).astype(_np.float32)
            mse = float(_np.mean((diff/255.0)**2)) * (255.0*255.0)
            psnr = 99.0 if mse <= 1e-9 else float(10.0 * _np.log10((255.0*255.0)/mse))
            # Overlap viz
            both = ((cap_bin == 255) & (ref_bin == 255))
            xor  = ((cap_bin == 255) ^ (ref_bin == 255))
            vis = _np.zeros((cam_h, cam_w, 3), _np.uint8)
            vis[both] = (0, 255, 0)
            vis[xor]  = (0, 0, 255)
            try:
                _h_last_grid['img'] = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
                _h_last_cap['img']  = cv2.cvtColor(_np.clip(cap, 0, 255).astype(_np.uint8), cv2.COLOR_GRAY2BGR)
                _h_last_overlap['img'] = vis
                _h_view_fit['done'] = False
                _update_h_preview('ov')
            except Exception:
                pass

        btn_h_proj.clicked.connect(_on_h_project_grid)
        btn_h_eval.clicked.connect(_on_h_capture_eval)
        btn_h_dots.clicked.connect(_on_h_dot_array_test)

        # H view mode (Reference / Captured / Overlap)
        try:
            from PyQt5.QtWidgets import QHBoxLayout as _QHBox, QRadioButton as _QRB, QButtonGroup as _QBG
            mode_row = _QHBox()
            mode_row.addWidget(QLabel("View:"))
            rb_ref = _QRB("Reference")
            rb_cap = _QRB("Captured")
            rb_ov  = _QRB("Overlap")
            rb_ov.setChecked(True)
            bg = _QBG(dlg)
            bg.addButton(rb_ref); bg.addButton(rb_cap); bg.addButton(rb_ov)
            mode_row.addWidget(rb_ref); mode_row.addWidget(rb_cap); mode_row.addWidget(rb_ov)
            # Legend
            leg = QLabel("Legend: \nGreen=overlap, Red=mismatch")
            try: leg.setStyleSheet("color:#1c1c1e;")
            except Exception: pass
            mode_row.addSpacing(12); mode_row.addWidget(leg)
            lay.addLayout(mode_row)
            def _on_mode_change():
                if rb_ref.isChecked():
                    _update_h_preview('ref')
                elif rb_cap.isChecked():
                    _update_h_preview('cap')
                else:
                    _update_h_preview('ov')
            rb_ref.toggled.connect(_on_mode_change)
            rb_cap.toggled.connect(_on_mode_change)
            rb_ov.toggled.connect(_on_mode_change)
        except Exception:
            pass

        def _on_calib_char():
            try:
                import numpy as _np
                import cv2
                from scipy.spatial import cKDTree
            except Exception as e:
                QMessageBox.warning(dlg, "Dependencies", f"Missing scipy or cv2: {e}")
                return
            try:
                # Build camera grid points
                cam_w, cam_h = _infer_cam_size()
                cell = 64
                pts = []
                for y in range(cell//2, cam_h, cell):
                    for x in range(cell//2, cam_w, cell):
                        pts.append([x, y, 1.0])
                P = _np.array(pts, dtype=_np.float64).T  # 3xN
                # Load H (camera->projector)
                H = getattr(self._camera, 'translation_matrix', None)
                if not isinstance(H, _np.ndarray) or H.shape != (3,3):
                    try:
                        from pathlib import Path as _P
                        npy = (_P(__file__).resolve().parent / 'Assets' / 'Generated' / 'homography_cam2proj.npy').as_posix()
                        if os.path.isfile(npy):
                            H = _np.load(npy)
                    except Exception:
                        H = None
                if H is None:
                    QMessageBox.warning(dlg, "Calibration", "No homography available. Run Calibrate first.")
                    return
                # Map to projector space
                X = H @ P; X /= _np.clip(X[2:3, :], 1e-9, None)
                proj_xy = X[:2, :].T
                # Ideal projector grid
                try:
                    proj_w = int(getattr(self, '_proj_w', 1920))
                    proj_h = int(getattr(self, '_proj_h', 1080))
                except Exception:
                    proj_w, proj_h = 1920, 1080
                gx = _np.arange(cell//2, proj_w, cell)
                gy = _np.arange(cell//2, proj_h, cell)
                grid_xy = _np.stack(_np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
                # Nearest neighbor errors
                try:
                    tree = cKDTree(grid_xy)
                    dists, _ = tree.query(proj_xy, k=1)
                except Exception:
                    dists = _np.linalg.norm(proj_xy[:, None, :] - grid_xy[None, :, :], axis=2).min(axis=1)
                rmse = float(_np.sqrt(_np.mean(dists**2))) if dists.size else 0.0
                # Visualization
                vis = _np.zeros((proj_h, proj_w, 3), _np.uint8)
                for y in range(cell//2, proj_h, cell):
                    cv2.line(vis, (0, y), (proj_w-1, y), (64,64,64), 1)
                for x in range(cell//2, proj_w, cell):
                    cv2.line(vis, (x, 0), (x, proj_h-1), (64,64,64), 1)
                for (x, y) in proj_xy.astype(_np.int32):
                    if 0 <= x < proj_w and 0 <= y < proj_h:
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)
                pm = _to_pix(vis); sl_pix.setPixmap(pm); sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
            except Exception as e:
                QMessageBox.critical(dlg, "Calibration Characterization", str(e))

        def _on_capture_evaluate():
            # Structured-light LUT: project prewarped grid, capture, and overlap
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                _prewarp = None
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None or _prewarp is None:
                QMessageBox.warning(dlg, "LUT Missing", "cam_from_proj LUTs not available. Run SL calibration first.")
                return
            # Build grid with chosen cell
            cam_w, cam_h = _infer_cam_size()
            try:
                _cell = max(1, int(sp_cell.value()))
            except Exception:
                _cell = 16
            try:
                _pitch = max(_cell, int(sp_pitch.value()))
            except Exception:
                _pitch = _cell
            grid = _make_cam_grid(cam_w, cam_h, cell=_cell, pitch=_pitch)
            grid_rgb = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(grid_rgb, inv_x, inv_y, proj_w, proj_h)
            # Try to project via engine; fallback to local window
            sent = _send_to_engine_gray(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
            if not sent:
                try:
                    if not self._ensure_projection():
                        raise RuntimeError("Projection window unavailable")
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    pass
            # Short wait and capture
            try:
                import time as _t
                _t.sleep(0.15)
            except Exception:
                pass
            cap = _capture_gray()
            if cap is None:
                QMessageBox.warning(dlg, "Capture Failed", "Could not read snapshot.")
                return
            if cap.shape[:2] != (cam_h, cam_w):
                try:
                    cap = cv2.resize(cap, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            # Crosstalk (report to textbox)
            try:
                ctk = _compute_crosstalk(cap, _cell, _pitch)
                if ctk:
                    metrics_box.appendPlainText(
                        f"Crosstalk (LUT): cell={_cell}px, pitch={_pitch}px -> mean={ctk['ratio_mean']*100:.1f}%, "
                        f"p95={ctk['ratio_p95']*100:.1f}% (N={ctk['samples']})"
                    )
            except Exception as _e:
                try:
                    metrics_box.appendPlainText(f"Crosstalk (LUT) error: {_e}")
                except Exception:
                    pass
            # Build binary masks and overlap vis
            try:
                _, cap_bin = cv2.threshold(cap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                cap_bin = (cap > 128).astype(_np.uint8) * 255
            grid_bin = (grid > 128).astype(_np.uint8) * 255
            both = ((cap_bin == 255) & (grid_bin == 255))
            xor  = ((cap_bin == 255) ^ (grid_bin == 255))
            vis = _np.zeros((cam_h, cam_w, 3), _np.uint8)
            vis[both] = (0, 255, 0)
            vis[xor]  = (0, 0, 255)
            diff = (cap_bin.astype(_np.int16) - grid_bin.astype(_np.int16)).astype(_np.float32)
            mse = float(_np.mean((diff/255.0)**2)) * (255.0*255.0)
            psnr = 99.0 if mse <= 1e-9 else float(10.0 * _np.log10((255.0*255.0)/mse))
            # Update preview with overlap and store ref/cap for view toggles
            try:
                _h_last_grid['img'] = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
                _h_last_cap['img']  = cv2.cvtColor(_np.clip(cap, 0, 255).astype(_np.uint8), cv2.COLOR_GRAY2BGR)
                _h_last_overlap['img'] = vis
                # Preserve current zoom on toggles; fit only once for new set
                _h_view_fit = {'done': False}
                pm = _to_pix(vis)
                sl_pix.setPixmap(pm)
                sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                _h_view_fit['done'] = True
            except Exception:
                pass

        def _on_round_trip():
            try:
                asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent.parent / "Assets" / "Generated").resolve()))
                fpx = os.path.join(asset_dir, "proj_from_cam_x.npy")
                fpy = os.path.join(asset_dir, "proj_from_cam_y.npy")
                inv_x, inv_y, _ = _load_luts()
                if inv_x is None or (not (os.path.isfile(fpx) and os.path.isfile(fpy))):
                    QMessageBox.warning(dlg, "Missing Maps", "Need proj_from_cam and cam_from_proj maps.")
                    return
                fx = _np.load(fpx).astype(_np.float32); fy = _np.load(fpy).astype(_np.float32)
                cam_h, cam_w = fx.shape
                step = max(4, min(cam_w, cam_h)//200)
                ys = _np.arange(0, cam_h, step, dtype=_np.int32)
                xs = _np.arange(0, cam_w, step, dtype=_np.int32)
                yy, xx = _np.meshgrid(ys, xs, indexing='ij')
                px = fx[yy, xx]; py = fy[yy, xx]
                ph, pw = inv_x.shape
                x0 = _np.floor(px).astype(_np.int32); y0 = _np.floor(py).astype(_np.int32)
                dx = px - x0; dy = py - y0
                x1 = _np.clip(x0+1, 0, pw-1); y1 = _np.clip(y0+1, 0, ph-1)
                def _bil(inmap):
                    v00 = inmap[_np.clip(y0,0,ph-1), _np.clip(x0,0,pw-1)]
                    v10 = inmap[y0, x1]; v01 = inmap[y1, x0]; v11 = inmap[y1, x1]
                    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 + (1-dx)*dy*v01 + dx*dy*v11
                rx = _bil(inv_x); ry = _bil(inv_y)
                err = _np.sqrt((_np.maximum(0, rx) - xx.astype(_np.float32))**2 + (_np.maximum(0, ry) - yy.astype(_np.float32))**2)
                mean_err = float(_np.mean(err[_np.isfinite(err)]))
                p95_err = float(_np.percentile(err[_np.isfinite(err)], 95))
                QMessageBox.information(dlg, "Round-Trip Error", f"Mean error: {mean_err:.2f} px\n95th %: {p95_err:.2f} px")
            except Exception as e:
                QMessageBox.warning(dlg, "Round-Trip Error", str(e))

        btn_diag.clicked.connect(_on_lut_diag)
        btn_proj.clicked.connect(_on_project_grid)
        btn_eval.clicked.connect(_on_capture_evaluate)
        btn_rterr.clicked.connect(_on_round_trip)
        btn_calib_char.clicked.connect(_on_calib_char)

        def _send_to_engine_gray(img_gray):
            try:
                from projector_client import ProjectorClient
                client = ProjectorClient()
                client.send_gray(img_gray, frame_id=8888, visible_id=0, immediate=True)
                client.close()
                return True
            except Exception:
                return False

        def _capture_gray():
            # Prefer RAM-backed path to avoid heavy disk I/O during probes
            try:
                # nosec B108: /dev/shm is POSIX-standard tmpfs for fast
                # shared-memory IPC. We probe with isdir + W_OK before use
                # and fall back to./Saved_Media if unavailable. The path
                # is hardcoded (not user-controlled), and the file we write
                # ("sl_validation_cap.png") is a known constant. This is a
                # performance optimization for probe-frame I/O during
                # structured-light calibration, not a security boundary.
                tmp_dir = "/dev/shm"  # nosec B108
                if os.path.isdir(tmp_dir) and os.access(tmp_dir, os.W_OK):
                    cap_path = os.path.join(tmp_dir, "sl_validation_cap.png")
                else:
                    save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                    os.makedirs(save_dir, exist_ok=True)
                    cap_path = os.path.join(save_dir, "sl_validation_cap.png")
            except Exception:
                save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                os.makedirs(save_dir, exist_ok=True)
                cap_path = os.path.join(save_dir, "sl_validation_cap.png")
            self._camera.snapshot(cap_path)
            return cv2.imread(cap_path, cv2.IMREAD_GRAYSCALE)

        def _on_pixel_probe():
            # Memory-safe pixel probe: avoid full-frame prewarp per point and reuse client/buffers
            # Uses forward LUT to place a subpixel dot in projector space via bilinear weights
            try:
                asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent.parent / "Assets" / "Generated").resolve()))
                fpx = os.path.join(asset_dir, "proj_from_cam_x.npy")
                fpy = os.path.join(asset_dir, "proj_from_cam_y.npy")
                fx = _np.load(fpx).astype(_np.float32)
                fy = _np.load(fpy).astype(_np.float32)
            except Exception as e:
                QMessageBox.warning(dlg, "Missing Maps", f"Need proj_from_cam_{'{x,y}'} maps: {e}")
                return
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None:
                return
            proj_h, proj_w = inv_x.shape
            cam_w, cam_h = fx.shape[1], fx.shape[0]
            step = max(96, min(cam_w, cam_h)//12)
            points = [(x, y) for y in range(step//2, cam_h, step) for x in range(step//2, cam_w, step)]
            # Limit total samples aggressively to avoid overloading system
            try:
                max_samples = 40
                if len(points) > max_samples:
                    stride = int(_np.ceil(len(points) / float(max_samples)))
                    points = points[::max(1, stride)]
            except Exception:
                pass
            # Preallocate projector-space grayscale buffer
            proj_img = _np.zeros((proj_h, proj_w), _np.uint8)
            vis = _np.zeros((cam_h, cam_w, 3), _np.uint8)
            errors = []
            # Reuse ZMQ client if available
            client = None
            try:
                from projector_client import ProjectorClient
                client = ProjectorClient()
            except Exception:
                client = None
            # Optional progress dialog
            try:
                from PyQt5.QtWidgets import QProgressDialog
                prog = QProgressDialog("Probing pixels…", "Cancel", 0, len(points), dlg)
                prog.setWindowModality(Qt.WindowModal)
                prog.setAutoClose(False)
                prog.setAutoReset(False)
                prog.show()
            except Exception:
                prog = None
            import gc as _gc
            import time as _t
            from PyQt5.QtWidgets import QApplication as _QApp
            t_start = _t.time()
            consecutive_fail = 0
            for i, (x0, y0) in enumerate(points):
                # Hard overall time cap (e.g., ~8s)
                if (_t.time() - t_start) > 8.0:
                    break
                # Early cancel check to keep UI responsive
                if prog is not None:
                    try:
                        if prog.wasCanceled():
                            break
                    except Exception:
                        pass
                # Build sparse subpixel dot in projector space using forward LUT
                px = float(fx[y0, x0]); py = float(fy[y0, x0])
                if not _np.isfinite(px) or not _np.isfinite(py):
                    continue
                if px < 0 or py < 0 or px > (proj_w - 1.001) or py > (proj_h - 1.001):
                    continue
                xz = int(_np.floor(px)); yz = int(_np.floor(py))
                dx = px - xz; dy = py - yz
                xz1 = min(proj_w - 1, xz + 1); yz1 = min(proj_h - 1, yz + 1)
                # Clear buffer and write four bilinear weights scaled to 255
                proj_img.fill(0)
                w00 = (1.0 - dx) * (1.0 - dy)
                w10 = dx * (1.0 - dy)
                w01 = (1.0 - dx) * dy
                w11 = dx * dy
                proj_img[yz,  xz ] = int(255.0 * w00)
                proj_img[yz,  xz1] = int(255.0 * w10)
                proj_img[yz1, xz ] = int(255.0 * w01)
                proj_img[yz1, xz1] = int(255.0 * w11)
                # Send to engine (reuse client) or fallback to Qt projector
                sent = False
                if client is not None:
                    try:
                        client.send_gray(proj_img, frame_id=8888, visible_id=0, immediate=True)
                        sent = True
                    except Exception:
                        sent = False
                if not sent:
                    try:
                        self.projection.show_image_raw_no_warp_no_flip(cv2.cvtColor(proj_img, cv2.COLOR_GRAY2BGR))
                    except Exception:
                        try:
                            self.projection.show_image_fullscreen_on_second_monitor(cv2.cvtColor(proj_img, cv2.COLOR_GRAY2BGR), None)
                        except Exception:
                            pass
                # Allow a short time for the projector to present the dot
                try:
                    _t.sleep(0.02)
                except Exception:
                    pass
                # Capture and compute subpixel center near (x0,y0)
                cap = _capture_gray()
                if cap is None:
                    consecutive_fail += 1
                    if consecutive_fail >= 20:
                        break
                    continue
                x1 = max(0, x0 - 4); x2 = min(cam_w, x0 + 5)
                y1 = max(0, y0 - 4); y2 = min(cam_h, y0 + 5)
                roi = cap[y1:y2, x1:x2].astype(_np.float32)
                if roi.size == 0:
                    consecutive_fail += 1
                    if consecutive_fail >= 20:
                        break
                    continue
                yy, xx = _np.mgrid[y1:y2, x1:x2]
                w = _np.maximum(0.0, roi - roi.mean())
                # Require sufficient local signal; skip if no visible dot
                amp = float(roi.max() - roi.mean())
                if not _np.isfinite(amp) or amp < 25.0 or w.sum() <= 1e-3:
                    consecutive_fail += 1
                    if consecutive_fail >= 20:
                        break
                    continue
                s = w.sum()
                cx = float((w * xx).sum() / s); cy = float((w * yy).sum() / s)
                errors.append(_np.hypot(cx - x0, cy - y0))
                consecutive_fail = 0
                cv2.circle(vis, (int(cx), int(cy)), 2, (0,255,0), -1)
                cv2.arrowedLine(vis, (x0, y0), (int(cx), int(cy)), (0,255,255), 1, tipLength=0.3)
                # UI/progress and periodic GC to keep memory in check
                if prog is not None:
                    try:
                        prog.setValue(i + 1)
                        _QApp.processEvents()
                        if prog.wasCanceled():
                            break
                    except Exception:
                        pass
                if (i & 7) == 7:
                    try: _gc.collect()
                    except Exception: pass
                # Small throttle to reduce CPU pressure
                try: _t.sleep(0.002)
                except Exception: pass
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
            # Always close the progress dialog first — it was setAutoClose(False)
            # so without an explicit close it sticks at the last value behind the
            # summary messagebox (the "stuck at 37%" symptom). Close before the
            # summary so the operator sees a clean teardown.
            try:
                if prog is not None:
                    prog.close()
            except Exception:
                pass
            n_attempted = (i + 1) if 'i' in locals() else 0
            n_detected = len(errors)
            elapsed = _t.time() - t_start
            if n_detected:
                mean_err = float(_np.mean(errors)); p95 = float(_np.percentile(errors, 95))
                detect_pct = 100.0 * n_detected / max(1, n_attempted)
                msg = (f"Detected {n_detected} of {n_attempted} points "
                       f"({detect_pct:.0f}%) in {elapsed:.1f}s.\n"
                       f"Mean centroid error: {mean_err:.2f} px\n"
                       f"95th percentile:     {p95:.2f} px")
                if n_detected < n_attempted // 2 and n_attempted > 2:
                    msg += ("\n\nLow detection ratio. Common causes: SL LUT inaccurate, "
                            "projector too dim, camera exposure too low, or capture "
                            "happening before the projector commits the dot. Re-run "
                            "Structured-Light Calibrate or raise exposure.")
                QMessageBox.information(dlg, "Pixel Probe", msg)
                try:
                    pm = _to_pix(vis)
                    sl_pix.setPixmap(pm)
                    sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                except Exception:
                    pass
            else:
                QMessageBox.warning(dlg, "Pixel Probe",
                    f"No detections (attempted {n_attempted} in {elapsed:.1f}s).\n"
                    "Possible causes: SL LUT inaccurate, projector dim, exposure "
                    "too low, or capture-projector timing off. Try increasing "
                    "camera exposure or re-running Structured-Light Calibrate.")

        def _on_dot_array():
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                QMessageBox.warning(dlg, "Missing", "prewarp not available")
                return
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None:
                return
            cam_w, cam_h = _infer_cam_size()
            spacing = max(24, min(cam_w, cam_h)//24)
            dot_r = 3
            img = _np.zeros((cam_h, cam_w), _np.uint8)
            pts = []
            for y in range(spacing//2, cam_h, spacing):
                for x in range(spacing//2, cam_w, spacing):
                    cv2.circle(img, (x,y), dot_r, 255, -1); pts.append((x,y))
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), inv_x, inv_y, proj_w, proj_h)
            sent = _send_to_engine_gray(warped)
            if not sent:
                try:
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            cap = _capture_gray()
            if cap is None:
                return
            # Threshold and find blobs
            _, bw = cv2.threshold(cap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            num, labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
            centers = cent[1:, :] if num>1 else _np.zeros((0,2), _np.float32)
            used = _np.zeros(len(centers), dtype=bool)
            errors = []
            overlay = cv2.cvtColor(cap, cv2.COLOR_GRAY2BGR)
            for (x,y) in pts:
                # find nearest center
                if centers.shape[0]==0:
                    continue
                d2 = _np.sum((centers - _np.array([[x,y]], _np.float32))**2, axis=1)
                idx = int(_np.argmin(d2))
                c = centers[idx]
                if used[idx]:
                    continue
                used[idx] = True
                err = float(_np.hypot(c[0]-x, c[1]-y))
                errors.append(err)
                cv2.circle(overlay, (int(c[0]), int(c[1])), 3, (0,255,0), -1)
                cv2.arrowedLine(overlay, (x,y), (int(c[0]), int(c[1])), (0,255,255), 1, tipLength=0.3)
            if errors:
                mean_err = float(_np.mean(errors)); p95 = float(_np.percentile(errors, 95))
                QMessageBox.information(dlg, "Dot Array", f"Samples: {len(errors)}\nMean: {mean_err:.2f} px\n95th %: {p95:.2f} px")
                try:
                    pm = _to_pix(overlay)
                    sl_pix.setPixmap(pm)
                    sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                except Exception:
                    pass

        def _on_round_trip_physical():
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                QMessageBox.warning(dlg, "Missing", "prewarp not available")
                return
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None:
                return
            cam_w, cam_h = _infer_cam_size()
            grid = _make_cam_grid(cam_w, cam_h)
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR), inv_x, inv_y, proj_w, proj_h)
            sent = _send_to_engine_gray(warped)
            if not sent:
                try:
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            cap = _capture_gray()
            if cap is None:
                return
            # Map the captured camera image into projector space with inv LUT and compare to warped(gray)
            cap_bgr = cv2.cvtColor(cap, cv2.COLOR_GRAY2BGR)
            pred = cv2.remap(cap_bgr, inv_x, inv_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            diff = (warped_gray.astype(_np.float32) - pred_gray.astype(_np.float32))
            mse = float(_np.mean(diff*diff)); psnr = 99.0 if mse<=1e-9 else 10.0*_np.log10((255.0*255.0)/mse)
            QMessageBox.information(dlg, "Round-Trip (Physical)", f"MSE: {mse:.1f}\nPSNR: {psnr:.2f} dB")
            try:
                pm = _to_pix(cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR))
                sl_pix.setPixmap(pm)
                sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
            except Exception:
                pass

        def _on_edge_strip():
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                QMessageBox.warning(dlg, "Missing", "prewarp not available")
                return
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None:
                return
            cam_w, cam_h = _infer_cam_size()
            positions = [int(cam_w*0.25), int(cam_w*0.5), int(cam_w*0.75)]
            img = _np.zeros((cam_h, cam_w), _np.uint8)
            for x in positions:
                img[:, max(0, x-1):min(cam_w, x+1)] = 255
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), inv_x, inv_y, proj_w, proj_h)
            sent = _send_to_engine_gray(warped)
            if not sent:
                try:
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            cap = _capture_gray()
            if cap is None:
                return
            errs = []
            for x0 in positions:
                x1 = max(0, x0-20); x2 = min(cam_w, x0+21)
                roi = cap[:, x1:x2].astype(_np.float32)
                gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                prof = _np.mean(_np.abs(gx), axis=0)
                # subpixel via quadratic fit around peak
                i = int(_np.argmax(prof))
                i0 = max(1, min(len(prof)-2, i))
                y1 = prof[i0-1]; y2 = prof[i0]; y3 = prof[i0+1]
                denom = (y1 - 2*y2 + y3)
                delta = 0.0 if abs(denom) < 1e-6 else 0.5 * (y1 - y3) / denom
                xpos = x1 + i0 + delta
                errs.append(abs(xpos - x0))
            if errs:
                mean_err = float(_np.mean(errs)); p95 = float(_np.percentile(errs, 95))
                QMessageBox.information(dlg, "Edge Strip", f"Lines: {len(errs)}\nMean: {mean_err:.2f} px\n95th %: {p95:.2f} px")

        btn_probe.clicked.connect(_on_pixel_probe)
        btn_dots.clicked.connect(_on_dot_array)
        btn_rtphy.clicked.connect(_on_round_trip_physical)
        btn_edge.clicked.connect(_on_edge_strip)

        # State for monitors. Deques grow at the sample rate; perf timer now
        # ticks every 250ms (4Hz) so maxlen=120 => 30s of history.
        from collections import deque
        cpu_hist = deque(maxlen=120)
        mem_hist = deque(maxlen=120)
        gpu_hist = deque(maxlen=120)
        trig_times = deque(maxlen=200)
        last_pidx = [0]
        running = {"engine": False}

        # GPU utilization source.
        # Primary: pynvml (NVIDIA NVML library). DOES NOT WORK on Tegra/Jetson —
        # libnvidia-ml.so is not shipped in L4T. NVML init fails with
        # "NVML Shared Library Not Found" so we fall through.
        # Fallback: Jetson sysfs /sys/devices/gpu.0/load. The value is in
        # 0–1000 range (0.1% per unit), NOT 0–255.
        _HAS_NVML = False
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvdev = pynvml.nvmlDeviceGetHandleByIndex(0)
            _HAS_NVML = True
        except Exception:
            _HAS_NVML = False

        _JETSON_GPU_PATH = "/sys/devices/gpu.0/load"
        _JETSON_GPU_OK = os.path.exists(_JETSON_GPU_PATH)

        def _sample_perf():
            try:
                cpu_hist.append(psutil.cpu_percent(interval=None))
                mem_hist.append(psutil.virtual_memory().percent)
            except Exception:
                cpu_hist.append(0.0)
                mem_hist.append(0.0)
            if _HAS_NVML:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(_nvdev)
                    gpu_hist.append(float(util.gpu))
                except Exception:
                    gpu_hist.append(0.0)
            elif _JETSON_GPU_OK:
                # Tegra GPU load is 0–1000 (0.1% per unit). Divide by 10 for %.
                try:
                    with open(_JETSON_GPU_PATH, "r") as f:
                        val = f.read().strip()
                        v = float(val) if val else 0.0
                        gpu_hist.append(min(100.0, max(0.0, v / 10.0)))
                except Exception:
                    gpu_hist.append(0.0)
            else:
                gpu_hist.append(0.0)
            if _HAS_PG:
                # y-only setData: pyqtgraph auto-generates x. Same pattern as
                # the Trace Test plots — avoids list(range(...)) each tick.
                cpu_curve.setData(list(cpu_hist))
                mem_curve.setData(list(mem_hist))
                gpu_curve.setData(list(gpu_hist))
            else:
                try:
                    lbl_cpu.setText(f"CPU: {cpu_hist[-1]:.1f} %")
                    lbl_mem.setText(f"Mem: {mem_hist[-1]:.1f} %")
                    lbl_gpu.setText(f"GPU: {gpu_hist[-1]:.1f} %")
                except Exception:
                    pass

        # Engine subscriber thread
        last_event_ts = {"t": 0.0}
        engine_status = {"text": "idle"}

        def _set_indicator(on: bool):
            # This indicator reflects whether GPIO trigger events are arriving
            # from the C++ engine's ZMQ status socket (tcp://127.0.0.1:5562),
            # which happens when the DMD sequencer is actively firing triggers.
            # It is NOT synced to the Start/Stop Projector Trigger button — the
            # DMD can still be running from a prior session even if the button
            # hasn't been pressed this session.
            try:
                if on:
                    ind_btn.setText("GPIO Triggers Detected")
                    ind_btn.setStyleSheet("QPushButton{background-color:#52c41a; color:white; border-radius:6px; padding:4px 8px;}")
                else:
                    ind_btn.setText("No GPIO Triggers")
                    ind_btn.setStyleSheet("QPushButton{background-color:#ff4d4f; color:white; border-radius:6px; padding:4px 8px;}")
            except Exception:
                pass

        def _start_engine_sub():
            import threading as _th
            import zmq as _zmq
            import json
            running["engine"] = True
            engine_status["text"] = "connecting…"
            def _loop():
                try:
                    ctx = _zmq.Context.instance()
                    sub = ctx.socket(_zmq.SUB)
                    sub.setsockopt(_zmq.LINGER, 0)
                    # Bound the subscriber buffer. Default HWM is 1000 but the
                    # engine publishes at up to 60 Hz and we only need the
                    # latest event for the indicator — 16 messages is plenty
                    # and caps memory (previously grew unboundedly when
                    # consumer lagged). CONFLATE keeps only the newest message.
                    sub.setsockopt(_zmq.RCVHWM, 16)
                    sub.setsockopt(_zmq.CONFLATE, 1)
                    sub.setsockopt_string(_zmq.SUBSCRIBE, "")
                    sub.connect("tcp://127.0.0.1:5562")
                    # Use a poller with short timeout instead of a NOBLOCK
                    # spin loop — yields CPU and avoids a hot busy-wait.
                    poller = _zmq.Poller()
                    poller.register(sub, _zmq.POLLIN)
                except Exception as e:
                    engine_status["text"] = f"error {e}"
                    running["engine"] = False
                    return
                engine_status["text"] = "monitoring"
                while running["engine"]:
                    try:
                        events = dict(poller.poll(timeout=50))
                    except Exception:
                        events = {}
                    if sub in events:
                        try:
                            msg = sub.recv(flags=_zmq.NOBLOCK)
                            d = json.loads(msg.decode('utf-8', errors='ignore'))
                            p = int(d.get('pidx', 0))
                            if p > last_pidx[0]:
                                last_pidx[0] = p
                                from time import time as now
                                ts = now()
                                trig_times.append(ts)
                                last_event_ts["t"] = ts
                        except Exception:
                            pass
                try:
                    sub.close(0)
                except Exception:
                    pass
                engine_status["text"] = "stopped"
            th = _th.Thread(target=_loop, daemon=True)
            th.start()
            dlg._engine_thread = th

        def _stop_engine_sub():
            running["engine"] = False

        def _toggle_engine_monitor(checked: bool):
            if checked:
                btn_mon.setText("Stop Engine Monitor")
                _start_engine_sub()
            else:
                btn_mon.setText("Start Engine Monitor")
                _stop_engine_sub()

        btn_mon.toggled.connect(_toggle_engine_monitor)

        # Periodic perf updates and trigger indicator decay
        try:
            from PyQt5.QtCore import QTimer
            tm = QTimer(dlg)
            def _tick():
                _sample_perf()
                # turn indicator OFF if no triggers for 0.5s
                try:
                    import time as _t
                    if running["engine"]:
                        if (_t.time() - last_event_ts.get("t", 0.0)) > 0.5:
                            _set_indicator(False)
                        else:
                            _set_indicator(True)
                    # update engine status and last rate text
                    status_lbl.setText(f"Engine: {engine_status.get('text','')}" )
                    # compute rate over last second for display
                    if trig_times:
                        t1 = trig_times[-1]
                        n = len([t for t in trig_times if t1 - t <= 1.0])
                        last_lbl.setText(f"Last: pidx={last_pidx[0]} vis=? rate={n} Hz")
                except Exception:
                    pass
            tm.timeout.connect(_tick)
            # 4 Hz — responsive graphs (was 1 Hz, looked frozen). deque maxlen=120
            # gives ~30s history. psutil.cpu_percent(interval=None) is fine at
            # this rate (it reads accumulated counters since last call).
            tm.start(250)
        except Exception:
            pass

        def _on_close():
            try:
                _stop_engine_sub()
            except Exception:
                pass

        try:
            dlg.finished.connect(lambda *_: _on_close())
        except Exception:
            pass

        dlg.show()


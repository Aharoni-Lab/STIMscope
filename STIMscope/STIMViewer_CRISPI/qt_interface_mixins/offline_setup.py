"""OfflineSetupDialogMixin — extracted from qt_interface.py.

Extracts the 1,037-LOC ``_open_offline_setup_dialog`` method into a
dedicated mixin so the parent Interface class drops below the §3.2
Hard band. Method body is byte-identical to the pre-extraction code
at ``qt_interface.py:3467-end`` (commit ``75e0487``); only the
surrounding module-level frame changed.

The method opens the Offline Setup dialog — the pre-experiment
workflow for ROI segmentation, calibration loading, and engine
warmup. Many nested closures handle file dialogs, image loading,
Cellpose/manual ROI flows, calibration apply, and ROI export.

§3.2 BLOCK disclosure: this mixin lands in the Hard band (>1000 LOC,
~1075 actual). **Cohesion reason:** single dialog factory with
nested closures sharing dialog widgets by lexical scope. **Recovery
path:** internal sub-split into helper methods
(`_offline_build_roi_group`, `_offline_build_calib_group`,
`_offline_build_engine_group`, `_offline_wire_launch_button`) beforeclose-out.

Mixin contract (Interface attributes the method reads/writes):
  * ``self._offline_setup_dlg`` — duplicate-window guard
  * ``self._proc_projector``, ``self._proc_dlpc`` — process refs
  * ``self._camera`` — for live preview / hardware run
  * ``self.display`` — for ROI overlay

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

class OfflineSetupDialogMixin:
    """Cluster 11 — Offline Setup pre-experiment dialog."""

    # ------------------------------------------------------------------
    # Offline Setup Dialog
    # ------------------------------------------------------------------
    def _open_offline_setup_dialog(self):
        """Open the Offline Setup dialog for pre-experiment ROI segmentation workflow."""
        # Prevent duplicate windows
        if hasattr(self, '_offline_setup_dlg') and self._offline_setup_dlg is not None:
            try:
                if self._offline_setup_dlg.isVisible():
                    self._offline_setup_dlg.raise_()
                    self._offline_setup_dlg.activateWindow()
                    return
            except Exception:
                pass

        try:
            from PyQt5.QtWidgets import (
                QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                QPushButton, QLineEdit, QComboBox, QSpinBox,
                QDoubleSpinBox, QFileDialog, QGroupBox,
            )
            from PyQt5.QtCore import Qt
            from pathlib import Path
            import numpy as np
            import pyqtgraph as pg

            dlg = QDialog(self)
            dlg.setWindowTitle("Offline Setup - ROI Segmentation")
            dlg.setWindowFlags(
                Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint
            )
            dlg.setModal(False)
            dlg.setMinimumSize(800, 600)
            # Force every spinbox in this dialog to a readable minimum height.
            # Default Qt rendering let them get squashed to the point you couldn't
            # see the digit inside. Also keeps comboboxes consistent.
            dlg.setStyleSheet(
                "QSpinBox, QDoubleSpinBox { min-height: 18px; padding: 0px 3px; }"
                "QComboBox { min-height: 18px; }"
                "QLineEdit { min-height: 18px; }"
            )
            main_layout = QVBoxLayout(dlg)

            # Shared state dict for the dialog
            state = {
                'recording_path': '',
                'stack': None,
                'mean_img': None,
                'norm_img': None,
                'labels': None,
                'neuron_ids': None,
                'centroids': None,
            }

            # ── A. Recording Selection ──
            rec_group = QGroupBox("A. Recording Selection")
            rec_grid = QGridLayout(rec_group)

            rec_grid.addWidget(QLabel("File:"), 0, 0)
            file_label = QLineEdit()
            file_label.setReadOnly(True)
            file_label.setPlaceholderText("No recording loaded")
            rec_grid.addWidget(file_label, 0, 1)

            load_btn = QPushButton("Load Recording")
            load_btn.setStyleSheet(
                "background-color: #2d5aa0; color: white; font-weight: bold;"
            )
            rec_grid.addWidget(load_btn, 0, 2)

            # Row 1: Projection method + compute button
            rec_grid.addWidget(QLabel("Projection:"), 1, 0)
            proj_combo = QComboBox()
            proj_combo.addItems(["Mean", "Max", "Std Dev", "Mean + Std"])
            proj_combo.setToolTip(
                "Mean: average brightness (standard, finds most neurons)\n"
                "Max: brightest frame per pixel (finds rarely active neurons)\n"
                "Std Dev: activity variance (highlights active neurons)\n"
                "Mean + Std: combined (best overall detection)")
            rec_grid.addWidget(proj_combo, 1, 1)

            compute_mean_btn = QPushButton("Compute Projection")
            compute_mean_btn.setStyleSheet("background-color: #2d5aa0; color: white; font-weight: bold;")
            compute_mean_btn.setEnabled(False)
            rec_grid.addWidget(compute_mean_btn, 1, 2)

            save_tiff_btn = QPushButton("Save as TIFF")
            save_tiff_btn.setEnabled(False)
            save_tiff_btn.setToolTip("Convert loaded video to TIFF for faster reloading")
            rec_grid.addWidget(save_tiff_btn, 1, 3)
            def _on_save_tiff():
                if state['stack'] is None:
                    return
                tpath, _ = QFileDialog.getSaveFileName(dlg, "Save as TIFF", "flood_recording.tiff", "TIFF (*.tiff *.tif)")
                if tpath:
                    try:
                        import tifffile
                        rec_status.setText("Saving TIFF...")
                        tifffile.imwrite(tpath, state['stack'], compression='zstd')
                        rec_status.setText(f"Saved: {tpath} ({state['stack'].shape[0]} frames)")
                    except Exception as e:
                        rec_status.setText(f"Save failed: {e}")
            save_tiff_btn.clicked.connect(_on_save_tiff)

            rec_status = QLabel("")
            rec_grid.addWidget(rec_status, 2, 0, 1, 4)

            main_layout.addWidget(rec_group)

            # ── B. Segmentation ──
            seg_group = QGroupBox("B. Segmentation")
            seg_grid = QGridLayout(seg_group)

            seg_grid.addWidget(QLabel("Method:"), 0, 0)
            method_combo = QComboBox()
            method_combo.addItems(["Otsu", "Cellpose"])
            seg_grid.addWidget(method_combo, 0, 1)

            # Otsu parameters — equal column stretch keeps the spinboxes from
            # getting squished and label/value pairs lined up across rows.
            otsu_frame = QtWidgets.QFrame()
            otsu_lay = QGridLayout(otsu_frame)
            otsu_lay.setContentsMargins(0, 0, 0, 0)
            for _c in range(4):
                otsu_lay.setColumnStretch(_c, 1 if _c % 2 else 0)

            otsu_lay.addWidget(QLabel("Min Area Frac:"), 0, 0)
            min_area_spin = QDoubleSpinBox()
            min_area_spin.setRange(0.0001, 0.1); min_area_spin.setDecimals(4)
            min_area_spin.setSingleStep(0.0001); min_area_spin.setValue(0.0002)
            min_area_spin.setToolTip("Minimum ROI area as fraction of image (filter tiny noise)")
            otsu_lay.addWidget(min_area_spin, 0, 1)

            otsu_lay.addWidget(QLabel("Max Area Frac:"), 0, 2)
            max_area_spin = QDoubleSpinBox()
            max_area_spin.setRange(0.001, 0.5); max_area_spin.setDecimals(3)
            max_area_spin.setSingleStep(0.001); max_area_spin.setValue(0.05)
            max_area_spin.setToolTip("Maximum ROI area as fraction of image (filter large blobs)")
            otsu_lay.addWidget(max_area_spin, 0, 3)

            otsu_lay.addWidget(QLabel("Blur Kernel:"), 1, 0)
            blur_kernel_spin = QSpinBox()
            blur_kernel_spin.setRange(1, 15); blur_kernel_spin.setSingleStep(2); blur_kernel_spin.setValue(3)
            blur_kernel_spin.setToolTip("Gaussian blur kernel size (odd number, larger = more smoothing)")
            otsu_lay.addWidget(blur_kernel_spin, 1, 1)

            otsu_lay.addWidget(QLabel("Blur Sigma:"), 1, 2)
            blur_sigma_spin = QDoubleSpinBox()
            blur_sigma_spin.setRange(0.1, 10.0); blur_sigma_spin.setDecimals(1)
            blur_sigma_spin.setSingleStep(0.5); blur_sigma_spin.setValue(1.5)
            blur_sigma_spin.setToolTip("Gaussian blur sigma (larger = more smoothing)")
            otsu_lay.addWidget(blur_sigma_spin, 1, 3)

            otsu_lay.addWidget(QLabel("Hole Fill Area:"), 2, 0)
            hole_fill_spin = QDoubleSpinBox()
            hole_fill_spin.setRange(0.0001, 0.01); hole_fill_spin.setDecimals(4)
            hole_fill_spin.setSingleStep(0.0001); hole_fill_spin.setValue(0.001)
            hole_fill_spin.setToolTip("Fill holes smaller than this fraction of image area")
            otsu_lay.addWidget(hole_fill_spin, 2, 1)

            otsu_watershed_check = QtWidgets.QCheckBox("Watershed splitting")
            otsu_watershed_check.setToolTip("Split large merged ROIs using watershed algorithm")
            otsu_lay.addWidget(otsu_watershed_check, 2, 2, 1, 2)

            seg_grid.addWidget(otsu_frame, 1, 0, 1, 4)

            # Cellpose parameters
            cellpose_frame = QtWidgets.QFrame()
            cellpose_lay = QGridLayout(cellpose_frame)
            cellpose_lay.setContentsMargins(0, 0, 0, 0)
            for _c in range(4):
                cellpose_lay.setColumnStretch(_c, 1 if _c % 2 else 0)

            cellpose_lay.addWidget(QLabel("Diameter:"), 0, 0)
            diameter_spin = QSpinBox()
            diameter_spin.setRange(1, 100); diameter_spin.setValue(9)
            diameter_spin.setToolTip("Expected cell diameter in pixels (0 = auto-estimate)")
            cellpose_lay.addWidget(diameter_spin, 0, 1)

            cellpose_lay.addWidget(QLabel("Model:"), 0, 2)
            cp_model_combo = QComboBox()
            cp_model_combo.addItems(["cyto2", "cyto", "nuclei", "custom"])
            cp_model_combo.setToolTip("Cellpose model: cyto2 (default), cyto (older), nuclei, or custom.pt file")
            cellpose_lay.addWidget(cp_model_combo, 0, 3)

            cellpose_lay.addWidget(QLabel("Flow Threshold:"), 1, 0)
            flow_thresh_spin = QDoubleSpinBox()
            flow_thresh_spin.setRange(0.0, 3.0); flow_thresh_spin.setDecimals(2)
            flow_thresh_spin.setSingleStep(0.1); flow_thresh_spin.setValue(0.5)
            flow_thresh_spin.setToolTip("Flow error threshold — lower = stricter segmentation (default 0.5)")
            cellpose_lay.addWidget(flow_thresh_spin, 1, 1)

            cellpose_lay.addWidget(QLabel("Cell Prob:"), 1, 2)
            cellprob_spin = QDoubleSpinBox()
            cellprob_spin.setRange(-6.0, 6.0); cellprob_spin.setDecimals(1)
            cellprob_spin.setSingleStep(0.5); cellprob_spin.setValue(-1.0)
            cellprob_spin.setToolTip("Cell probability threshold — lower = more permissive (default -1.0)")
            cellpose_lay.addWidget(cellprob_spin, 1, 3)

            cellpose_lay.addWidget(QLabel("Custom Model:"), 2, 0)
            cp_model_path = QLineEdit()
            cp_model_path.setPlaceholderText("Path to.pt model file (only for 'custom')")
            cp_model_path.setEnabled(False)
            cellpose_lay.addWidget(cp_model_path, 2, 1, 1, 2)
            cp_browse_btn = QPushButton("Browse")
            cp_browse_btn.setEnabled(False)
            cp_browse_btn.clicked.connect(lambda: cp_model_path.setText(
                QFileDialog.getOpenFileName(dlg, "Select Cellpose model", "", "Model files (*.pt *.pth)")[0] or cp_model_path.text()))
            cellpose_lay.addWidget(cp_browse_btn, 2, 3)

            def _on_cp_model_changed(idx):
                is_custom = cp_model_combo.currentText() == "custom"
                cp_model_path.setEnabled(is_custom)
                cp_browse_btn.setEnabled(is_custom)
            cp_model_combo.currentIndexChanged.connect(_on_cp_model_changed)

            cellpose_frame.setVisible(False)
            seg_grid.addWidget(cellpose_frame, 2, 0, 1, 4)

            # Video processing options
            proc_frame = QtWidgets.QFrame()
            proc_lay = QGridLayout(proc_frame)
            proc_lay.setContentsMargins(0, 0, 0, 0)

            proc_lay.addWidget(QLabel("Frame Range:"), 0, 0)
            frame_start_spin = QSpinBox()
            frame_start_spin.setRange(0, 999999); frame_start_spin.setValue(0)
            frame_start_spin.setToolTip("First frame to include in mean projection (skip calibration frames)")
            proc_lay.addWidget(frame_start_spin, 0, 1)
            proc_lay.addWidget(QLabel("to"), 0, 2)
            frame_end_spin = QSpinBox()
            frame_end_spin.setRange(0, 999999); frame_end_spin.setValue(0)
            frame_end_spin.setToolTip("Last frame (0 = all frames)")
            proc_lay.addWidget(frame_end_spin, 0, 3)

            gpu_seg_check = QtWidgets.QCheckBox("GPU acceleration")
            gpu_seg_check.setChecked(True)
            gpu_seg_check.setToolTip("Use CuPy/CUDA for faster segmentation (falls back to CPU if unavailable)")
            proc_lay.addWidget(gpu_seg_check, 1, 0, 1, 2)

            proc_lay.addWidget(QLabel("Overlay Opacity:"), 1, 2)
            opacity_spin = QDoubleSpinBox()
            opacity_spin.setRange(0.1, 1.0); opacity_spin.setDecimals(1)
            opacity_spin.setSingleStep(0.1); opacity_spin.setValue(0.6)
            opacity_spin.setToolTip("ROI overlay opacity on mean projection (0.1 = faint, 1.0 = solid)")
            proc_lay.addWidget(opacity_spin, 1, 3)

            seg_grid.addWidget(proc_frame, 3, 0, 1, 4)

            def _on_method_changed(idx):
                otsu_frame.setVisible(idx == 0)
                cellpose_frame.setVisible(idx == 1)

            method_combo.currentIndexChanged.connect(_on_method_changed)

            run_seg_btn = QPushButton("Run Segmentation")
            run_seg_btn.setEnabled(False)
            run_seg_btn.setStyleSheet(
                "background-color: #2d8a4e; color: white; font-weight: bold; padding: 6px;"
            )
            seg_grid.addWidget(run_seg_btn, 4, 0, 1, 2)

            seg_status = QLabel("")
            seg_grid.addWidget(seg_status, 4, 2, 1, 2)

            main_layout.addWidget(seg_group)

            # ── C. ROI Visualization ──
            vis_group = QGroupBox("C. ROI Visualization")
            vis_layout = QVBoxLayout(vis_group)

            gw = pg.GraphicsLayoutWidget()
            gw.setMinimumHeight(300)
            plot = gw.addPlot()
            plot.setAspectLocked(True)
            plot.invertY(True)
            img_item = pg.ImageItem()
            plot.addItem(img_item)
            vis_layout.addWidget(gw)

            vis_stats = QLabel("No segmentation results yet.")
            vis_layout.addWidget(vis_stats)

            main_layout.addWidget(vis_group, stretch=1)

            # ── D. Export ──
            export_group = QGroupBox("D. Export")
            export_lay = QHBoxLayout(export_group)

            save_btn = QPushButton("Save ROIs")
            save_btn.setEnabled(False)
            save_btn.setStyleSheet(
                "background-color: #b45309; color: white; font-weight: bold;"
            )
            export_lay.addWidget(save_btn)

            export_status = QLabel("")
            export_lay.addWidget(export_status)
            export_lay.addStretch()

            main_layout.addWidget(export_group)

            # ==============================================================
            # Helper: load recording from path
            # ==============================================================
            def _load_recording_from_path(path):
                ext = Path(path).suffix.lower()
                if ext in ('.npy',):
                    arr = np.load(path)
                    if arr.ndim == 2:
                        arr = arr[np.newaxis,...]
                    return arr
                elif ext in ('.npz',):
                    d = np.load(path)
                    arr = d[list(d.keys())[0]]
                    if arr.ndim == 2:
                        arr = arr[np.newaxis,...]
                    return arr
                elif ext in ('.tif', '.tiff'):
                    import tifffile
                    return tifffile.imread(path)
                else:
                    import cv2 as _cv2
                    cap = _cv2.VideoCapture(str(path))
                    frames = []
                    while True:
                        ret, f = cap.read()
                        if not ret:
                            break
                        if f.ndim == 3:
                            f = _cv2.cvtColor(f, _cv2.COLOR_BGR2GRAY)
                        frames.append(f)
                    cap.release()
                    if not frames:
                        raise RuntimeError(f"Could not read any frames from {path}")
                    return np.array(frames)

            def _set_recording(path):
                state['recording_path'] = str(path)
                file_label.setText(str(path))
                compute_mean_btn.setEnabled(True)
                rec_status.setText("Recording loaded. Click 'Compute Mean Projection'.")

            # ==============================================================
            # A. Load Recording
            # ==============================================================
            def _on_load_recording():
                # Start in host Desktop if mounted, falling through to broader host roots
                # so the user can navigate anywhere on the host machine.
                # /host_home, /host_media, /host_mnt come from bind-mounts in docker-compose.yml.
                _start_dir = ""
                for _sd in [
                    "/host_home/Desktop",
                    "/host_home/Videos",
                    "/host_home/Downloads",
                    "/host_home",
                    "/host_media",
                    "/host_mnt",
                    str(Path(__file__).resolve().parent / "Saved_Media"),
                    ".",
                ]:
                    if os.path.isdir(_sd):
                        _start_dir = _sd
                        break
                fpath, _ = QFileDialog.getOpenFileName(
                    dlg,
                    "Select flood recording",
                    _start_dir,
                    "Recordings (*.tif *.tiff *.mp4 *.avi *.mov *.npy *.npz);;All (*)",
                )
                if fpath:
                    _set_recording(fpath)

            load_btn.clicked.connect(_on_load_recording)

            # ==============================================================
            # A. Compute Mean Projection
            # ==============================================================
            def _on_compute_mean():
                path = state['recording_path']
                if not path:
                    rec_status.setText("No recording loaded.")
                    return
                proj_method = proj_combo.currentText()
                rec_status.setText(f"Computing {proj_method} projection...")
                rec_status.setStyleSheet("color: orange;")
                compute_mean_btn.setEnabled(False)
                dlg.repaint()

                def _do_compute():
                    try:
                        import cv2 as _cv2
                        import time as _time
                        ext = Path(path).suffix.lower()
                        t0 = _time.time()

                        # Frame range filter — wired 
                        # frame_end=0 means "all frames"
                        _frame_start = int(frame_start_spin.value())
                        _frame_end = int(frame_end_spin.value())
                        if _frame_end > 0 and _frame_end <= _frame_start:
                            return False, (
                                f"Frame range invalid: end ({_frame_end}) "
                                f"must be > start ({_frame_start})"
                            )

                        # Try GPU-accelerated path
                        _use_gpu = gpu_seg_check.isChecked()
                        _cp = None
                        if _use_gpu:
                            try:
                                import cupy as _cp_mod
                                _cp = _cp_mod
                            except Exception:
                                _cp = None

                        if ext in ('.mp4', '.avi', '.mov', '.mkv'):
                            cap = _cv2.VideoCapture(str(path))
                            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 0
                            # Apply frame range to total before subsampling
                            _eff_end = _frame_end if _frame_end > 0 else total
                            _eff_total = max(0, _eff_end - _frame_start)
                            step = max(1, _eff_total // 500) if _eff_total > 500 else 1
                            if step > 1:
                                print(f"  [Proj] Subsampling: every {step}th frame ({_eff_total // step} of {_eff_total})", flush=True)
                            if _frame_start > 0 or _frame_end > 0:
                                print(f"  [Proj] Frame range: [{_frame_start}, {_eff_end})", flush=True)

                            # Streaming projection — supports Mean, Max, Std, Mean+Std
                            acc_sum = None  # for mean
                            acc_max = None  # for max
                            acc_sq = None   # for std (sum of squares)
                            n = 0
                            frame_idx = 0
                            while True:
                                ok, frame = cap.read()
                                if not ok:
                                    break
                                # Frame range gate
                                if frame_idx < _frame_start:
                                    frame_idx += 1
                                    continue
                                if _frame_end > 0 and frame_idx >= _frame_end:
                                    break
                                # Subsample relative to frames-after-start
                                _rel = frame_idx - _frame_start
                                if step > 1 and _rel % step != 0:
                                    frame_idx += 1
                                    continue
                                frame_idx += 1
                                if frame.ndim == 3:
                                    frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)

                                if _cp is not None:
                                    f = _cp.asarray(frame, dtype=_cp.float32)
                                else:
                                    f = frame.astype(np.float32)

                                if acc_sum is None:
                                    _xp = _cp if _cp is not None else np
                                    acc_sum = _xp.zeros_like(f)
                                    acc_max = f.copy()
                                    acc_sq = _xp.zeros_like(f)

                                acc_sum += f
                                if proj_method in ("Max", "Mean + Std"):
                                    _xp = _cp if _cp is not None else np
                                    acc_max = _xp.maximum(acc_max, f)
                                if proj_method in ("Std Dev", "Mean + Std"):
                                    acc_sq += f * f

                                n += 1
                                if n % 100 == 0:
                                    _backend = "GPU" if _cp else "CPU"
                                    print(f"  [Proj-{_backend}] {n} frames ({_time.time()-t0:.1f}s)...", flush=True)

                            cap.release()
                            if n == 0:
                                raise RuntimeError(f"No frames read from {path}")

                            _to_np = (lambda x: _cp.asnumpy(x)) if _cp is not None else (lambda x: x)

                            if proj_method == "Mean":
                                mean_img = _to_np(acc_sum / float(n)).astype(np.float64)
                            elif proj_method == "Max":
                                mean_img = _to_np(acc_max).astype(np.float64)
                            elif proj_method == "Std Dev":
                                variance = (acc_sq / float(n)) - (acc_sum / float(n)) ** 2
                                _xp = _cp if _cp is not None else np
                                mean_img = _to_np(_xp.sqrt(_xp.maximum(variance, 0))).astype(np.float64)
                            elif proj_method == "Mean + Std":
                                mean_part = acc_sum / float(n)
                                variance = (acc_sq / float(n)) - mean_part ** 2
                                _xp = _cp if _cp is not None else np
                                std_part = _xp.sqrt(_xp.maximum(variance, 0))
                                # Normalize each to [0,1] then combine
                                def _norm01(x):
                                    mn, mx = float(x.min()), float(x.max())
                                    return (x - mn) / max(mx - mn, 1e-8)
                                combined = _norm01(mean_part) * 0.5 + _norm01(std_part) * 0.5
                                mean_img = _to_np(combined).astype(np.float64)
                            else:
                                mean_img = _to_np(acc_sum / float(n)).astype(np.float64)

                            if _cp is not None:
                                del acc_sum, acc_max, acc_sq
                                _cp.get_default_memory_pool().free_all_blocks()

                            state['stack'] = None
                            state['_n_frames'] = n
                            print(f"  [Proj] Done: {proj_method}, {n} frames in {_time.time()-t0:.1f}s", flush=True)
                        else:
                            # TIFF/NPY/NPZ — load into array
                            stack = _load_recording_from_path(path)
                            # Apply frame range to stack — wired 
                            if stack.ndim == 3 and (_frame_start > 0 or _frame_end > 0):
                                _end = _frame_end if _frame_end > 0 else stack.shape[0]
                                stack = stack[_frame_start:_end]
                                print(f"  [Proj] Sliced stack to frames [{_frame_start}, {_end}): {stack.shape[0]} frames", flush=True)
                            if stack.ndim == 3:
                                _xp = _cp if _cp is not None else np
                                if _cp is not None:
                                    s = _cp.asarray(stack, dtype=_cp.float32)
                                else:
                                    s = stack.astype(np.float32)
                                # Dead-line removed  (vulture
                                # close-out finding): the original line was
                                #   `mean_img = float(...) if False else np.zeros(1)`
                                # with an `if False` branch that could never
                                # execute, plus an immediate overwrite below
                                # by the proj_method dispatch chain. Pure
                                # noise; removing.
                                _to_np2 = (lambda x: _cp.asnumpy(x)) if _cp is not None else (lambda x: x)
                                if proj_method == "Mean":
                                    mean_img = _to_np2(_xp.mean(s, axis=0)).astype(np.float64)
                                elif proj_method == "Max":
                                    mean_img = _to_np2(_xp.max(s, axis=0)).astype(np.float64)
                                elif proj_method == "Std Dev":
                                    mean_img = _to_np2(_xp.std(s, axis=0)).astype(np.float64)
                                elif proj_method == "Mean + Std":
                                    def _n01(x):
                                        mn, mx = float(x.min()), float(x.max())
                                        return (x - mn) / max(mx - mn, 1e-8)
                                    mean_img = _to_np2(_n01(_xp.mean(s, axis=0)) * 0.5 + _n01(_xp.std(s, axis=0)) * 0.5).astype(np.float64)
                                else:
                                    mean_img = _to_np2(_xp.mean(s, axis=0)).astype(np.float64)
                                if _cp is not None:
                                    del s; _cp.get_default_memory_pool().free_all_blocks()
                            else:
                                mean_img = stack.astype(np.float64)
                            state['stack'] = stack
                            state['_n_frames'] = stack.shape[0] if stack.ndim == 3 else 1

                        vmin, vmax = mean_img.min(), mean_img.max()
                        norm_img = (mean_img - vmin) / max(vmax - vmin, 1e-8)
                        state['mean_img'] = mean_img
                        state['norm_img'] = norm_img
                        return True, None
                    except Exception as ex:
                        import traceback; traceback.print_exc()
                        return False, str(ex)

                import threading

                # Use a signal for reliable cross-thread UI update
                class _MeanDoneSignaler(QtCore.QObject):
                    done = QtCore.pyqtSignal(bool, str)
                _sig = _MeanDoneSignaler()
                _sig.done.connect(lambda ok, err: _on_mean_done(ok, err), QtCore.Qt.QueuedConnection)

                def _bg():
                    ok, err = _do_compute()
                    _sig.done.emit(ok, err or "")

                threading.Thread(target=_bg, daemon=True).start()

            def _on_mean_done(ok, err):
                if not ok:
                    rec_status.setText(f"Error: {err}")
                    return
                norm = state['norm_img']
                gray = (norm * 200).astype(np.uint8)
                H, W = gray.shape
                rgba = np.zeros((H, W, 4), dtype=np.uint8)
                rgba[:, :, 0] = gray
                rgba[:, :, 1] = gray
                rgba[:, :, 2] = gray
                rgba[:, :, 3] = 255
                # pyqtgraph ImageItem expects (W, H, 4) — transpose
                img_item.setImage(rgba.transpose(1, 0, 2))
                run_seg_btn.setEnabled(True)
                run_seg_btn.setStyleSheet("background-color: #2d8a4e; color: white; font-weight: bold; padding: 6px;")
                compute_mean_btn.setEnabled(True)
                save_tiff_btn.setEnabled(state['stack'] is not None)
                _n_frames = state.get('_n_frames', 1)
                rec_status.setStyleSheet("color: green; font-weight: bold;")
                rec_status.setText(
                    f"READY — Mean projection: "
                    f"{state['mean_img'].shape[1]}x{state['mean_img'].shape[0]}, "
                    f"{_n_frames} frames. Ready to segment."
                )
                print(f"[Offline] Mean projection done: {state['mean_img'].shape}, {_n_frames} frames")

            compute_mean_btn.clicked.connect(_on_compute_mean)

            # ==============================================================
            # B. Run Segmentation
            # ==============================================================
            def _on_run_segmentation():
                norm = state.get('norm_img')
                if norm is None:
                    seg_status.setText("Compute mean projection first.")
                    return
                seg_status.setText("Running segmentation...")
                run_seg_btn.setEnabled(False)
                dlg.repaint()

                method = method_combo.currentText()

                def _do_seg():
                    try:
                        from scipy import ndimage
                        H, W = norm.shape

                        if method == "Otsu":
                            from skimage.filters import threshold_otsu
                            from skimage.morphology import (
                                remove_small_objects,
                                remove_small_holes,
                            )

                            min_af = min_area_spin.value()
                            max_af = max_area_spin.value()
                            hole_af = hole_fill_spin.value()

                            # Optional Gaussian blur preprocessing — wired 
                            blur_k = int(blur_kernel_spin.value())
                            blur_s = float(blur_sigma_spin.value())
                            if blur_k > 1 and blur_s > 0:
                                norm_in = ndimage.gaussian_filter(
                                    norm, sigma=blur_s
                                )
                            else:
                                norm_in = norm

                            thr = threshold_otsu(norm_in)
                            binary = norm_in > thr
                            n_pix = H * W
                            min_area = max(5, int(n_pix * min_af))
                            max_area = int(n_pix * max_af)
                            hole_area = max(1, int(n_pix * hole_af))

                            binary = remove_small_holes(
                                binary, area_threshold=hole_area
                            )
                            binary = remove_small_objects(
                                binary, min_size=min_area
                            )

                            raw_labels, n_found = ndimage.label(binary)

                            # Optional watershed splitting of merged ROIs
                            if otsu_watershed_check.isChecked():
                                try:
                                    from skimage.segmentation import watershed
                                    from skimage.feature import (
                                        peak_local_max,
                                    )
                                    distance = ndimage.distance_transform_edt(
                                        binary
                                    )
                                    # Local maxima as watershed markers; min
                                    # distance scales with expected cell size
                                    expected_radius = max(
                                        3,
                                        int(np.sqrt(min_area / np.pi)),
                                    )
                                    coords = peak_local_max(
                                        distance,
                                        min_distance=expected_radius,
                                        labels=binary,
                                    )
                                    markers = np.zeros(
                                        binary.shape, dtype=np.int32
                                    )
                                    for mi, (yy, xx) in enumerate(coords):
                                        markers[yy, xx] = mi + 1
                                    if markers.max() > 0:
                                        raw_labels = watershed(
                                            -distance,
                                            markers,
                                            mask=binary,
                                        )
                                        n_found = int(raw_labels.max())
                                except Exception as _wex:
                                    print(
                                        f'[Otsu watershed] failed: {_wex} — '
                                        f'falling back to connected components'
                                    )

                            labels = np.zeros((H, W), dtype=np.int32)
                            new_id = 1
                            for roi_id in range(1, n_found + 1):
                                area = int((raw_labels == roi_id).sum())
                                if min_area <= area <= max_area:
                                    labels[raw_labels == roi_id] = new_id
                                    new_id += 1

                        elif method == "Cellpose":
                            try:
                                from cellpose import models
                            except ImportError:
                                return (
                                    False,
                                    "Cellpose not installed. "
                                    "Run: pip install cellpose",
                                )
                            # Cellpose model + flow/cellprob — wired 
                            cp_model_name = cp_model_combo.currentText()
                            cp_path = cp_model_path.text().strip()
                            try:
                                if cp_model_name == "custom" and cp_path:
                                    model = models.CellposeModel(
                                        pretrained_model=cp_path
                                    )
                                else:
                                    model = models.Cellpose(
                                        model_type=cp_model_name
                                        if cp_model_name in (
                                            "cyto2", "cyto", "nuclei"
                                        )
                                        else "cyto2"
                                    )
                            except Exception as _mex:
                                print(
                                    f'[Cellpose] model init fallback: {_mex}'
                                )
                                model = models.Cellpose(model_type='cyto2')
                            diam = diameter_spin.value()
                            flow_thr = float(flow_thresh_spin.value())
                            cell_prob = float(cellprob_spin.value())
                            img_uint8 = (norm * 255).astype(np.uint8)
                            try:
                                masks, _, _, _ = model.eval(
                                    img_uint8,
                                    diameter=diam,
                                    channels=[0, 0],
                                    flow_threshold=flow_thr,
                                    cellprob_threshold=cell_prob,
                                )
                            except TypeError:
                                # Older cellpose APIs may not accept these
                                # kwargs — fall back gracefully
                                masks, _, _, _ = model.eval(
                                    img_uint8,
                                    diameter=diam,
                                    channels=[0, 0],
                                )
                            labels = masks.astype(np.int32)
                        else:
                            return False, f"Unknown method: {method}"

                        neuron_ids = np.unique(labels)
                        neuron_ids = neuron_ids[neuron_ids > 0].astype(
                            np.int32
                        )
                        n_neurons = len(neuron_ids)
                        if n_neurons == 0:
                            return (
                                False,
                                "Segmentation found 0 ROIs. "
                                "Adjust parameters and retry.",
                            )

                        centroids = ndimage.center_of_mass(
                            labels > 0, labels, neuron_ids.tolist()
                        )
                        centroids = np.array(centroids, dtype=np.float32)

                        state['labels'] = labels
                        state['neuron_ids'] = neuron_ids
                        state['centroids'] = centroids

                        return True, None
                    except Exception as ex:
                        import traceback
                        traceback.print_exc()
                        return False, str(ex)

                import threading

                class _SegDoneSignaler(QtCore.QObject):
                    done = QtCore.pyqtSignal(bool, str)
                _seg_sig = _SegDoneSignaler()
                _seg_sig.done.connect(lambda ok, err: _on_seg_done(ok, err), QtCore.Qt.QueuedConnection)

                def _bg_seg():
                    ok, err = _do_seg()
                    _seg_sig.done.emit(ok, err or "")

                threading.Thread(target=_bg_seg, daemon=True).start()

            def _on_seg_done(ok, err):
                run_seg_btn.setEnabled(True)
                if not ok:
                    seg_status.setText(f"Error: {err}")
                    return

                labels = state['labels']
                norm = state['norm_img']
                neuron_ids = state['neuron_ids']
                centroids = state['centroids']
                H, W = labels.shape
                n_neurons = len(neuron_ids)

                seg_status.setText(f"Done: {n_neurons} neurons found.")
                vis_stats.setText(f"Found {n_neurons} neurons")

                # Build RGBA overlay
                gray = (norm * 200).astype(np.uint8)
                rgba = np.zeros((H, W, 4), dtype=np.uint8)
                rgba[:, :, 0] = gray
                rgba[:, :, 1] = gray
                rgba[:, :, 2] = gray
                rgba[:, :, 3] = 255

                colors = [
                    (255, 100, 100),
                    (100, 255, 100),
                    (100, 100, 255),
                    (255, 255, 100),
                    (255, 100, 255),
                    (100, 255, 255),
                    (200, 150, 100),
                    (100, 200, 150),
                    (150, 100, 200),
                    (220, 180, 80),
                ]

                for i, nid in enumerate(neuron_ids):
                    c = colors[int(i) % len(colors)]
                    roi = labels == int(nid)
                    for ch in range(3):
                        vals = rgba[roi, ch].astype(np.float32)
                        # opacity_spin wired 
                        _ov = float(opacity_spin.value())
                        blended = (vals * (1.0 - _ov) + c[ch] * _ov).astype(np.uint8)
                        rgba[roi, ch] = blended

                # pyqtgraph expects (W, H, 4)
                img_item.setImage(rgba.transpose(1, 0, 2))

                # Add text labels at centroids
                for old_item in list(plot.items):
                    if isinstance(old_item, pg.TextItem):
                        plot.removeItem(old_item)
                for i, nid in enumerate(neuron_ids):
                    cy, cx = centroids[i]
                    txt = pg.TextItem(
                        str(int(nid)),
                        color=(255, 255, 255),
                        anchor=(0.5, 0.5),
                    )
                    txt.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
                    txt.setPos(float(cx), float(cy))
                    plot.addItem(txt)

                save_btn.setEnabled(True)

            run_seg_btn.clicked.connect(_on_run_segmentation)

            # ==============================================================
            # E. Save ROIs
            # ==============================================================
            def _on_save_rois():
                labels = state.get('labels')
                if labels is None:
                    export_status.setText("No segmentation to save.")
                    return
                # Default to the writable, host-mounted save dir (/data) so the
                # dialog opens somewhere the container can actually write. The
                # old default (CS/data, inside the source tree) is read-only in
                # some launchers and root-owned; navigating into /host_home/* —
                # which is mounted read-only — is what produced the
                # "[Errno 30] Read-only file system" save error.
                import os as _os
                default_dir = _os.environ.get("STIM_SAVE_DIR") or str(
                    Path(__file__).resolve().parent / "CS" / "data"
                )
                try:
                    _os.makedirs(default_dir, exist_ok=True)
                except Exception:
                    pass
                default_path = str(Path(default_dir) / "rois.npz")
                fpath, _ = QFileDialog.getSaveFileName(
                    dlg,
                    "Save ROIs",
                    default_path,
                    "NPZ files (*.npz)",
                )
                if not fpath:
                    return
                try:
                    save_dict = {
                        'labels': labels,
                    }
                    if state.get('mean_img') is not None:
                        save_dict['mean_img'] = state['mean_img']
                    if state.get('neuron_ids') is not None:
                        save_dict['neuron_ids'] = state['neuron_ids']
                    if state.get('centroids') is not None:
                        save_dict['centroids'] = state['centroids']
                    np.savez_compressed(fpath, **save_dict)
                    export_status.setText(f"Saved to {fpath}")
                except Exception as ex:
                    export_status.setText(f"Save error: {ex}")

            save_btn.clicked.connect(_on_save_rois)

            # Show the dialog
            dlg.show()
            self._offline_setup_dlg = dlg

        except Exception as e:
            import traceback
            print(f"Offline Setup dialog error: {e}")
            traceback.print_exc()

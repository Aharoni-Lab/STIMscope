"""TraceTestMixin — extracted from qt_interface.py per L5 §0.5 decomposition.

Cluster 9 subset (interactive trace-extraction test dialog).
1 method, ~275 LOC.

Method:
- ``_open_trace_test_dialog()``    — Build & show the Trace Extraction
  Test QDialog. User clicks the camera feed to set an ROI center; a
  QTimer polls the camera pipeline_queue and updates two pyqtgraph
  plots (raw mean intensity + ΔF/F) at ~30 fps. Used to verify that
  the trace-extraction pipeline responds spatially to SLM stimulation.

Mixin contract — subclass provides:
    self._camera                      : OptimizedCamera-like (with.start_pipeline_feed,.stop_pipeline_feed,.pipeline_queue)

Pure hoist — no behavior change vs. monolith.
"""

from __future__ import annotations


class TraceTestMixin:
    """Cluster 9 subset — interactive trace extraction test dialog."""

    def _open_trace_test_dialog(self):
        """Interactive trace extraction test.

        User clicks on the camera feed to define an ROI region.
        Real-time trace extraction runs continuously, showing mean intensity.
        User moves mouse on the SLM monitor to create a light spot and
        verifies that the trace responds only when the spot is inside the ROI.
        """
        try:
            import cv2
            import numpy as np
            import pyqtgraph as pg
            from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                         QLabel, QPushButton, QSpinBox, QGroupBox)
            from PyQt5.QtCore import QTimer, Qt
            from PyQt5.QtGui import QImage, QPixmap
        except ImportError as e:
            print(f"Trace test dependencies not available: {e}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Trace Extraction Test — Click camera feed to set ROI")
        dlg.setMinimumSize(1200, 700)
        layout = QHBoxLayout(dlg)

        # Left: camera feed with ROI overlay
        left_panel = QVBoxLayout()
        feed_label = QLabel("Click on the image to set ROI center")
        feed_label.setStyleSheet("color: white; font-weight: bold;")
        left_panel.addWidget(feed_label)

        cam_label = QLabel()
        cam_label.setMinimumSize(640, 480)
        cam_label.setStyleSheet("background: black;")
        cam_label.setAlignment(Qt.AlignCenter)
        cam_label.setFixedSize(640, 480)
        left_panel.addWidget(cam_label, stretch=0)

        # ROI size + orientation controls
        roi_ctrl = QHBoxLayout()
        roi_ctrl.addWidget(QLabel("ROI radius:"))
        radius_spin = QSpinBox()
        radius_spin.setRange(5, 200)
        radius_spin.setValue(40)
        roi_ctrl.addWidget(radius_spin)

        from PyQt5.QtWidgets import QCheckBox
        flip_h_check = QCheckBox("Flip H")
        flip_v_check = QCheckBox("Flip V")
        rotate_label = QLabel("Rot°:")
        rotate_spin = QSpinBox()
        rotate_spin.setRange(0, 359)
        rotate_spin.setValue(0)
        rotate_spin.setSingleStep(90)
        roi_ctrl.addWidget(flip_h_check)
        roi_ctrl.addWidget(flip_v_check)
        roi_ctrl.addWidget(rotate_label)
        roi_ctrl.addWidget(rotate_spin)
        left_panel.addLayout(roi_ctrl)

        layout.addLayout(left_panel, stretch=2)

        # Right: trace plot + status
        right_panel = QVBoxLayout()

        # Trace plot — auto-range to show actual signal changes
        trace_plot = pg.PlotWidget(title="Real-Time Trace (ROI Mean Intensity)")
        trace_plot.setLabel('bottom', 'Frame')
        trace_plot.setLabel('left', 'Mean Intensity')
        trace_plot.setBackground('#0d1117')
        trace_plot.setMinimumHeight(200)
        trace_plot.enableAutoRange()
        trace_curve = trace_plot.plot(pen=pg.mkPen('#58a6ff', width=2))
        right_panel.addWidget(trace_plot, stretch=1)

        # Delta-F/F plot
        dff_plot = pg.PlotWidget(title="ΔF/F (baseline from first 30 frames)")
        dff_plot.setLabel('bottom', 'Frame')
        dff_plot.setLabel('left', 'ΔF/F')
        dff_plot.setBackground('#0d1117')
        dff_plot.setMinimumHeight(200)
        dff_plot.enableAutoRange()
        dff_curve = dff_plot.plot(pen=pg.mkPen('#3fb950', width=2))
        right_panel.addWidget(dff_plot, stretch=1)

        # Status
        status_label = QLabel("Status: Click on camera feed to set ROI")
        status_label.setStyleSheet("color: #c9d1d9; font-size: 12px;")
        right_panel.addWidget(status_label)

        # Instructions
        instr = QLabel(
            "1. Click on the camera feed to place your observation ROI\n"
            "2. Slide your mouse to the second monitor (SLM)\n"
            "3. Move your cursor OUTSIDE the ROI area → trace should be flat\n"
            "4. Move your cursor INSIDE the ROI area → trace should spike\n"
            "5. This proves trace extraction is spatially accurate"
        )
        instr.setStyleSheet("color: #8b949e; font-size: 11px;")
        right_panel.addWidget(instr)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear ROI")
        clear_btn.clicked.connect(lambda: _clear_roi())
        btn_row.addWidget(clear_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(close_btn)
        right_panel.addLayout(btn_row)

        layout.addLayout(right_panel, stretch=1)

        # State
        _state = {
            'roi_center': None,  # (row, col) in camera pixel coords
            'roi_radius': 40,
            'trace': [],
            'dff_trace': [],
            'baseline_frames': [],
            'baseline': None,
            'frame_count': 0,
            'max_trace_len': 500,
        }

        def _clear_roi():
            _state['roi_center'] = None
            _state['trace'] = []
            _state['dff_trace'] = []
            _state['baseline_frames'] = []
            _state['baseline'] = None
            _state['frame_count'] = 0
            status_label.setText("Status: Click on camera feed to set ROI")
            trace_curve.setData([])
            dff_curve.setData([])

        # Store latest frame from camera signal (updated on every camera frame)
        _state['latest_frame'] = None
        _state['cam_h'] = 0
        _state['cam_w'] = 0

        # Use the same pipeline_queue mechanism as the hardware pipeline
        self._camera.start_pipeline_feed()

        # Mouse click on camera label to set ROI
        DISPLAY_W, DISPLAY_H = 640, 480

        def _on_cam_click(event):
            pos = event.pos()
            cam_h = _state['cam_h']
            cam_w = _state['cam_w']
            if cam_h == 0 or cam_w == 0:
                return
            # Map 640x480 display coords → camera pixel coords
            # Image is scaled with KeepAspectRatio inside DISPLAY_W x DISPLAY_H
            scale = min(DISPLAY_W / cam_w, DISPLAY_H / cam_h)
            disp_w = int(cam_w * scale)
            disp_h = int(cam_h * scale)
            off_x = (DISPLAY_W - disp_w) // 2
            off_y = (DISPLAY_H - disp_h) // 2
            img_x = int((pos.x() - off_x) / scale)
            img_y = int((pos.y() - off_y) / scale)
            if 0 <= img_x < cam_w and 0 <= img_y < cam_h:
                _state['roi_center'] = (img_y, img_x)  # (row, col)
                _state['trace'] = []
                _state['dff_trace'] = []
                _state['baseline_frames'] = []
                _state['baseline'] = None
                _state['frame_count'] = 0
                status_label.setText(f"ROI at ({img_x}, {img_y}) in {cam_w}x{cam_h} — extracting...")

        cam_label.mousePressEvent = _on_cam_click

        # Timer: grab frame from pipeline_queue (same as hardware pipeline), display + extract
        def _update():
            _state['roi_radius'] = radius_spin.value()

            # Grab latest frame from pipeline_queue (same path as hardware pipeline)
            frame = None
            try:
                # Drain queue, keep only latest frame
                while not self._camera.pipeline_queue.empty():
                    try:
                        ts, ipl_img = self._camera.pipeline_queue.get_nowait()
                        arr = ipl_img.get_numpy_3D() if hasattr(ipl_img, 'get_numpy_3D') else ipl_img.get_numpy_2D()
                        if arr.ndim == 3:
                            arr = arr[:, :, 0]
                        frame = arr.astype(np.float32)
                    except Exception:
                        break
            except Exception:
                pass

            if frame is None:
                return

            # Apply orientation transforms
            if flip_h_check.isChecked():
                frame = np.fliplr(frame)
            if flip_v_check.isChecked():
                frame = np.flipud(frame)
            rot = rotate_spin.value()
            if rot == 90:
                frame = np.rot90(frame, k=1)
            elif rot == 180:
                frame = np.rot90(frame, k=2)
            elif rot == 270:
                frame = np.rot90(frame, k=3)
            elif rot != 0:
                M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), rot, 1.0)
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            cam_h, cam_w = frame.shape[:2]
            _state['cam_h'] = cam_h
            _state['cam_w'] = cam_w

            # Draw camera feed with ROI overlay
            _max = frame.max()
            if _max > 0:
                disp = ((frame / _max) * 255).astype(np.uint8)
            else:
                disp = np.zeros((cam_h, cam_w), dtype=np.uint8)

            center = _state['roi_center']
            r = _state['roi_radius']
            if center is not None:
                cy, cx = center
                cv2.circle(disp, (cx, cy), r, 255, 2)
                cv2.circle(disp, (cx, cy), 2, 255, -1)

            qimg = QImage(disp.data.tobytes(), cam_w, cam_h, cam_w, QImage.Format_Grayscale8)
            pm = QPixmap.fromImage(qimg)
            cam_label.setPixmap(pm.scaled(DISPLAY_W, DISPLAY_H, Qt.KeepAspectRatio, Qt.FastTransformation))

            # Extract trace from ROI (same np.mean as hardware pipeline)
            if center is not None:
                cy, cx = center
                yy, xx = np.ogrid[:cam_h, :cam_w]
                mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
                roi_pixels = frame[mask]
                mean_val = float(roi_pixels.mean()) if len(roi_pixels) > 0 else 0.0

                _state['frame_count'] += 1
                _state['trace'].append(mean_val)
                if len(_state['trace']) > _state['max_trace_len']:
                    _state['trace'] = _state['trace'][-_state['max_trace_len']:]

                if _state['frame_count'] <= 30:
                    _state['baseline_frames'].append(mean_val)
                    _state['baseline'] = float(np.mean(_state['baseline_frames']))

                f0 = _state['baseline']
                dff = (mean_val - f0) / max(f0, 1e-6) if f0 is not None else 0.0
                _state['dff_trace'].append(dff)
                if len(_state['dff_trace']) > _state['max_trace_len']:
                    _state['dff_trace'] = _state['dff_trace'][-_state['max_trace_len']:]

                trace_curve.setData(_state['trace'])
                dff_curve.setData(_state['dff_trace'])

                _f0_str = f"{f0:.1f}" if f0 is not None else "---"
                status_label.setText(
                    f"ROI ({cx},{cy}) r={r} | Frame {_state['frame_count']} | "
                    f"Mean={mean_val:.1f} | F0={_f0_str} | "
                    f"ΔF/F={dff:.4f} | Pixels={len(roi_pixels)}")

        timer = QTimer(dlg)
        timer.timeout.connect(_update)
        timer.start(33)  # ~30 fps

        def _on_close():
            timer.stop()
            self._camera.stop_pipeline_feed()

        dlg.finished.connect(_on_close)
        dlg.setModal(False)
        dlg.show()

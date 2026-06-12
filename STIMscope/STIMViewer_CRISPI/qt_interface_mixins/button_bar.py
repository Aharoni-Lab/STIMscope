"""ButtonBarMixin — extracted from qt_interface.py.

Extracts the 894-LOC ``_create_button_bar`` method into a dedicated
mixin. Method body is byte-identical to the pre-extraction code at
``qt_interface.py:297-1190`` (commit ``c08662f``); only the
surrounding module-level frame changed.

The method builds the main button bar at the top of the Interface
window. It contains many nested closures wiring up button signals:
calibration buttons, projector controls, ROI tools, recording
toggles, mode selectors, FPS controls, persistence helpers,
mask-flip handlers, and orientation toggles.

§3.2 BLOCK disclosure: this mixin is in the Cohesion-over-budget
band (701-1000 LOC, ~930 actual including header). **Cohesion
reason:** single UI scaffolding method with all button widget
constructors + signal-wire closures sharing the local
``button_bar_layout``. **Recovery path before:** sub-split
into ``_build_calib_buttons``, ``_build_projector_buttons``,
``_build_roi_buttons``, ``_build_recording_buttons``,
``_build_mode_combo``, etc., each taking the layout as a parameter
and returning the row of widgets. Expected post-recovery: 8-10
sub-methods each ≤120 LOC.

Mixin contract (Interface attributes the method reads/writes):
  * ``self._layout`` — main window's QVBoxLayout receives the bar
  * ``self._sl_progress`` / ``self._sl_status`` — set to None for
    later population by ``_create_statusbar``
  * Many ``self._button_*`` attributes set during construction.

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

from qt_interface_mixins._shared import _GPU_AVAILABLE
from ids_peak import ids_peak
from pathlib import Path


class ButtonBarMixin:
    """Cluster 12 — main button-bar construction + signal wiring."""

    def _create_button_bar(self):
       
        # Helper to force a widget width to match its current text
        def _set_compact_width_to_text(widget, extra_px: int = 24):
            try:
                fm = widget.fontMetrics()
                text = widget.currentText() if hasattr(widget, 'currentText') else widget.text()
                width = fm.horizontalAdvance(text) + extra_px
                if width > 0:
                    widget.setFixedWidth(width)
            except Exception:
                pass


        button_bar = QtWidgets.QWidget(self.centralWidget())
        button_bar_layout = QtWidgets.QGridLayout()


        self._button_start_hardware_acquisition = QtWidgets.QPushButton("Start Hardware Acquisition")
        self._button_start_hardware_acquisition.clicked.connect(self._start_hardware_acquisition)
        try:
            self._button_start_hardware_acquisition.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._set_compact_width_to_text(self._button_start_hardware_acquisition)
        except Exception:
            pass


        self._button_start_recording = QtWidgets.QPushButton("Start Recording")
        self._button_start_recording.clicked.connect(self._start_recording)

        self._button_view_recording = QtWidgets.QPushButton("View Recording")
        self._button_view_recording.clicked.connect(self._open_tiff_viewer)
        self._button_view_recording.setToolTip("Open a saved TIFF recording in a viewer with frame slider and auto-contrast.")

        # : "Open in External Viewer" replaces a short-lived
        # in-app TIFF playback widget. Fiji (ImageJ) is the scientific
        # community's standard for multi-page TIFF analysis — better
        # contrast tools, ROI tools, 16-bit precision preservation
        # (in-app cv2-mp4v transcode was lossy), full plugin ecosystem.
        # `xdg-open` launches with the user's default app for.tiff
        # files, which is Fiji on most lab Jetsons.
        self._button_play_recording = QtWidgets.QPushButton("Open in External Viewer")
        self._button_play_recording.clicked.connect(self._open_tiff_external)
        self._button_play_recording.setToolTip(
            "Open the selected TIFF recording in the system's default app "
            "(typically Fiji / ImageJ on lab Jetsons). For pixel-precise "
            "scientific analysis use this rather than 'View Recording' (which "
            "is a quick in-app slider peek with auto-contrast)."
        )

        # New: External control buttons
        self._button_start_projector = QtWidgets.QPushButton("Start Projection Engine")
        self._button_start_projector.clicked.connect(self._toggle_start_projector)
        self._seq_type_label = QtWidgets.QLabel("Sequence Type")
        self._seq_type_dropdown = QtWidgets.QComboBox()
        # Default = 8-bit RGB (0x03) — this is the proven-working sequence-type
        # byte from the original boot sequence the lab used for months. Our
        # 4-command boot helper (dlpc_i2c.boot_external_pattern_streaming)
        # uses the proven timing values (11 ms illum / 2.2 ms pre / 5 ms post)
        # which the DLPC accepts for any of these four sequence types.
        self._seq_type_dropdown.addItems([
            "8-bit RGB (0x03)",
            "8-bit Mono (0x02)",
            "1-bit RGB (0x01)",
            "1-bit Mono (0x00)",
        ])
        try:
            self._seq_type_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._seq_type_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass
        try:
            self._seq_type_dropdown.currentTextChanged.connect(self._on_seq_type_changed)
        except Exception:
            pass

        # LED color setting — translated to the 0x96 byte 3 Illumination Select
        # bitmask and passed to i2c_test_send_commands.py `boot --illum <hex>` at
        # Start Projector Trigger. In Light Control – External Pattern Streaming
        # (mode 03h), per-pattern LED selection lives in 0x96 byte 3 — NOT in
        # 0x52, which (p. 42) "does not apply to Light Control modes".
        #
        # STIMscope calcium-imaging protocol (full detail in
        # docs/hardware/DMD_RED_BLUE_WORKFLOW.md §0):
        #   - Single color (Red / Blue / R+B / RGB): this dropdown + Start
        #     Projector Trigger → DMD illuminates with the chosen color.
        #   - Red-stim + blue-observe per camera frame (the real experimental
        #     mode): requires 8-bit RGB sub-frame sequencing — DMD in
        #     seq_type=0x03 with illum_select=0x05 (R+B only, G dead), HDMI
        #     carries stim mask in R channel + global mask in B channel, camera
        #     triggers on TRIG_OUT_2 with delay tuned to the blue sub-frame.
        #     This is implemented in Stream R (see docs/EXECUTION_PLAN_20260417.md).
        self._led_color_label = QtWidgets.QLabel("LED Color")
        self._led_color_dropdown = QtWidgets.QComboBox()
        # Green is intentionally omitted — the optical path has a dichroic
        # that blocks red toward the camera and passes blue/green to the
        # camera; green LED is not useful for stim or observation in our
        # optogenetics workflow. Supported: Red (stim), Blue (observe),
        # R+B (pink/magenta for alignment), RGB white (all three for
        # diagnostic).
        self._led_color_dropdown.addItems([
            "Red (0x01)",          # stim
            "Blue (0x04)",         # observe
            "R+B (0x05)",          # alignment / diagnostic
            "White / RGB (0x07)",  # full diagnostic
        ])
        self._led_color_dropdown.setToolTip(
            "Illumination Select for the initial pattern at Start Projector Trigger "
            "(0x96 byte 3). To switch colors: Stop Projector Trigger → pick color → "
            "Start Projector Trigger. Fast per-frame alternation (red-stim/blue-observe) "
            "is handled by the frame scheduler, not this dropdown.")
        try:
            self._led_color_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._led_color_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass

        self._overlay_on = False
        self._overlay_contours = None
        self._button_toggle_overlay = QtWidgets.QPushButton("Enable Overlay")
        self._button_toggle_overlay.setCheckable(True)
        self._button_toggle_overlay.setChecked(False)
        self._button_toggle_overlay.toggled.connect(self._toggle_overlay)
        # Initialize label to current state
        try:
            self._toggle_overlay(self._button_toggle_overlay.isChecked())
        except Exception:
            pass
        # Pixel Probe toggle — left-click on camera preview shows (x, y, intensity) in statusbar
        self._button_pixel_probe = QtWidgets.QPushButton("Pixel Probe")
        self._button_pixel_probe.setCheckable(True)
        self._button_pixel_probe.setChecked(False)
        self._button_pixel_probe.setToolTip(
            "Toggle pixel probe mode. When ON, click on the camera preview "
            "to see pixel coordinates and intensity values in the status bar.")
        self._button_pixel_probe.toggled.connect(self._toggle_pixel_probe)

        self._proj_warp_mode = "NONE"  # default: no warp until user selects
        self._button_req_hmatrix = QtWidgets.QPushButton("REQ H-Matrix")
        self._button_req_hmatrix.setCheckable(True)
        self._button_req_hmatrix.setChecked(False)
        self._button_req_hmatrix.toggled.connect(self._on_warp_h_toggled)
        self._button_use_lut = QtWidgets.QPushButton("REQ LUT")
        self._button_use_lut.setCheckable(True)
        self._button_use_lut.setChecked(False)
        self._button_use_lut.toggled.connect(self._on_warp_lut_toggled)
        # Mask pattern selection UI
        self._mask_pattern_label = QtWidgets.QLabel("Mask Pattern")
        self._mask_pattern_dropdown = QtWidgets.QComboBox()
        self._mask_pattern_dropdown.addItems([
            "Seg Mask", "Moving Bar", "Checkerboard", "Solid", "Circle", "Gradient", "Image", "Folder", "Custom"
        ])
        self._mask_pattern_dropdown.currentTextChanged.connect(self._on_mask_pattern_changed)
        self._mask_pattern_browse = QtWidgets.QPushButton("Browse…")
        self._mask_pattern_browse.clicked.connect(self._browse_mask_pattern_path)
        self._mask_pattern_browse.setEnabled(False)
        self._mask_pattern_path = ""
        self._button_send_triggers = QtWidgets.QPushButton("Start Projector Trigger")
        self._button_send_triggers.clicked.connect(self._toggle_send_triggers)
        self._stim_mode_label = QtWidgets.QLabel("Projection Mode")
        self._stim_mode_dropdown = QtWidgets.QComboBox()
        self._stim_mode_dropdown.addItems([
            "Simultaneous (Mode B)",
            "Temporal (Mode A)",
        ])
        self._stim_mode_dropdown.setToolTip(
            "How red (stim) and blue (observe) masks are presented on the DMD.\n"
            "Simultaneous: R+B sub-frame multiplexing (composite RGB HDMI)\n"
            "Temporal: 16ms RED then 16ms BLUE, alternating per frame")
        try:
            self._stim_mode_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._stim_mode_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass
        self._proc_scheduler = None
        self._button_send_masks = QtWidgets.QPushButton("Send Masks")
        self._button_send_masks.clicked.connect(self._toggle_send_masks)
        # Live LED switching: when the projector trigger is already running,
        # changing the LED dropdown should immediately update 0x96 byte 3 so
        # the next HDMI frame fires the chosen color. When the trigger is OFF,
        # the dropdown selection is just remembered for the next Start click.
        try:
            self._led_color_dropdown.currentTextChanged.connect(
                self._on_led_color_changed_live)
        except Exception:
            pass
        self._button_i2c_custom = QtWidgets.QPushButton("I²C Burst Sender")
        self._button_i2c_custom.clicked.connect(self._open_i2c_custom_dialog)
        self._button_i2c_custom.setToolTip(
            "Multi-line I²C burst editor. Type one write per line, click Send All "
            "to fire them as an atomic burst (in-process raw_write, no inter-write delay). "
            "Required for DLPC multi-step transitions — the firmware enters safety shutdown "
            "on malformed sequences. Includes templates: boot MONO+RED, switch to BLUE, etc.")
        # Keep trigger/mask buttons compact to text, slightly larger
        try:
            self._button_send_triggers.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_send_triggers, 28)
        except Exception:
            pass
        try:
            self._button_send_masks.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_send_masks, 28)
        except Exception:
            pass




        
        self._button_show_gpu_ui = QtWidgets.QPushButton("Real-Time Trace Extraction")
        self._button_show_gpu_ui.clicked.connect(self.show_gpu_ui)
        self._button_show_gpu_ui.setEnabled(_GPU_AVAILABLE)
        try:
            self._button_show_gpu_ui.setStyleSheet(
                """
                QPushButton {
                    color: #000000; /* keep text black */
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f5eeff, stop:1 #ece2ff);
                    border: 1px solid #cdbcf3;
                    border-radius: 6px;
                    padding: 4px 10px;
                }
                QPushButton:hover {
                    color: #000000;
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f2e9ff, stop:1 #e4d6ff);
                    border: 1px solid #b49cf0;
                }
                QPushButton:pressed {
                    color: #000000;
                    background-color: #dbcaff;
                }
                QPushButton:disabled {
                    color: #b8b6c9;
                    background-color: #fafafa;
                    border: 1px solid #eeeeee;
                }
                """
            )
        except Exception:
            pass
        



        self._dropdown_trigger_line = QtWidgets.QComboBox()
        self._label_trigger_line = QtWidgets.QLabel("Change Hardware Trigger Line:")



        self._dropdown_trigger_line.addItem("Line0")
        self._dropdown_trigger_line.addItem("Line1")   
        self._dropdown_trigger_line.addItem("Line2")
        self._dropdown_trigger_line.addItem("Line3")


        self._dropdown_trigger_line.currentIndexChanged.connect(self.change_hardware_trigger_line)
        # Make combo compact to fit content
        try:
            self._dropdown_trigger_line.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._dropdown_trigger_line.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._dropdown_trigger_line.currentTextChanged.connect(lambda *_: _set_compact_width_to_text(self._dropdown_trigger_line, 36))
            _set_compact_width_to_text(self._dropdown_trigger_line, 36)
        except Exception:
            pass


        self._dropdown_pixel_format = QtWidgets.QComboBox()
        try:
            formats = self._camera.node_map.FindNode("PixelFormat").Entries()
        except Exception:
            formats = []

        
        na = getattr(ids_peak, "NodeAccessStatus_NotAvailable", None)
        ni = getattr(ids_peak, "NodeAccessStatus_NotImplemented", None)

        for idx in formats:
            try:
                acc = idx.AccessStatus()
                if (na is not None and acc == na) or (ni is not None and acc == ni):
                    continue
                if self._camera.conversion_supported(idx.Value()):
                    self._dropdown_pixel_format.addItem(idx.SymbolicValue())
            except Exception:

                continue
        self._dropdown_pixel_format.currentIndexChanged.connect(self.change_pixel_format)
        # Make combo compact to fit content
        try:
            self._dropdown_pixel_format.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._dropdown_pixel_format.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._dropdown_pixel_format.currentTextChanged.connect(lambda *_: _set_compact_width_to_text(self._dropdown_pixel_format, 36))
            _set_compact_width_to_text(self._dropdown_pixel_format, 36)
        except Exception:
            pass


        self._dropdown_pixel_format.setEnabled(True)
        self._dropdown_trigger_line.setEnabled(True)



        

        self._button_software_trigger = QtWidgets.QPushButton("Snapshot")
        self._button_software_trigger.clicked.connect(self._trigger_sw_trigger)
        # Keep buttons compact
        try:
            self._button_software_trigger.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_software_trigger)
        except Exception:
            pass
        
        

        self._button_calibrate = QtWidgets.QPushButton("Calibrate")
        self._button_calibrate.clicked.connect(self._calibrate)
        try:
            self._button_calibrate.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            # a bit larger than text
            _set_compact_width_to_text(self._button_calibrate, 28)
        except Exception:
            pass

        # Structured-Light calibration & projection buttons
        self._button_sl_calibrate = QtWidgets.QPushButton("Structured-Light Calibrate")
        self._button_sl_calibrate.clicked.connect(self._sl_calibrate)
        try:
            self._button_sl_calibrate.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_sl_calibrate, 28)
        except Exception:
            pass
        self._button_sl_project_reg = QtWidgets.QPushButton("Project LUT-Warped")
        self._button_sl_project_reg.clicked.connect(self._sl_project_registration)
        try:
            self._button_sl_project_reg.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_sl_project_reg, 28)
        except Exception:
            pass

        # Project intensity controls
        self._project_intensity_label = QtWidgets.QLabel("Project Intensity")
        self._project_intensity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._project_intensity_slider.setRange(0, 255)
        self._project_intensity_slider.setValue(255)
        self._project_intensity_slider.setSingleStep(1)
        self._project_intensity_slider.setMaximumWidth(150)  # Make slider shorter
        self._project_intensity_slider.valueChanged.connect(self._update_project_intensity)
        
        self._project_intensity_value_label = QtWidgets.QLabel("255")
        self._project_intensity_value_label.setMinimumWidth(30)
        self._project_intensity_value_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self._button_project_on = QtWidgets.QPushButton("Project ON")
        self._button_project_on.clicked.connect(self._project_on)
        
        self._button_project_off = QtWidgets.QPushButton("Project OFF")
        self._button_project_off.clicked.connect(self._project_off)

        # Camera type selection
        self._camera_type_label = QtWidgets.QLabel("Camera Type")
        self.camera_type_dropdown = QtWidgets.QComboBox()
        self.camera_type_dropdown.addItems(["IDS_Peak", "MIPI", "Generic Camera"])
        self.camera_type_dropdown.setCurrentText(self.selected_camera_type)
        self.camera_type_dropdown.currentTextChanged.connect(self._on_camera_type_changed)
        # Make combo compact to fit content
        try:
            self.camera_type_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self.camera_type_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self.camera_type_dropdown.currentTextChanged.connect(lambda *_: _set_compact_width_to_text(self.camera_type_dropdown, 36))
            _set_compact_width_to_text(self.camera_type_dropdown, 36)
        except Exception:
            pass

        self._gain_label = QtWidgets.QLabel("AG")
        self._gain_label.setMaximumWidth(70)

        self._gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._gain_slider.setRange(100, 1000)
        self._gain_slider.setSingleStep(1)
        self._gain_slider.valueChanged.connect(self._update_gain)

        

        self._dgain_label = QtWidgets.QLabel("DG")
        self._dgain_label.setMaximumWidth(70)

        self._dgain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._dgain_slider.setRange(100, 1000)
        self._dgain_slider.setSingleStep(1)
        self._dgain_slider.valueChanged.connect(self._update_dgain)


        # Zoom slider removed - using mouse wheel zoom instead



        config_group = QtWidgets.QGroupBox("")
        config_layout = QtWidgets.QGridLayout()
        config_layout.setSpacing(3)  # Reduce spacing
        try:
            config_layout.setHorizontalSpacing(2)  # Tighter space between top-row buttons
        except Exception:
            pass
        config_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        config_group.setLayout(config_layout)
        config_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #d1d1d6;
                border-radius: 6px;
                margin-top: 2px;
                font-weight: 500;
                font-size: 11px;
                color: #1c1c1e;
                background-color: #ffffff;
                padding: 4px;
            }
        """)


        # Row 0: Main action buttons (tightly packed, left-aligned)
        row0_layout = QtWidgets.QHBoxLayout()
        row0_layout.setContentsMargins(0, 0, 0, 0)
        row0_layout.setSpacing(4)
        row0_layout.addWidget(self._button_start_hardware_acquisition)
        # Move Start Projection Engine next to Start Hardware Acquisition (right side)
        row0_layout.addWidget(self._button_start_projector)
        # The calibration-related buttons are moved to a dedicated top panel
        # (Calibrate, Structured-Light Calibrate, Subpixel, Project LUT-Warped)
        try:
            self._chk_phase_refine = QtWidgets.QCheckBox("Subpixel")
            self._chk_phase_refine.setChecked(False)
            self._chk_phase_refine.setToolTip("Enable sinusoidal phase refinement for subpixel LUT. If results degrade, uncheck.")
        except Exception:
            pass
        row0_widget = QtWidgets.QWidget()
        row0_widget.setLayout(row0_layout)
        config_layout.addWidget(row0_widget,                             0, 0, 1, 2, Qt.AlignLeft)
        # Row 1: Projection engine and trigger controls
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.addWidget(self._seq_type_label)
        row1_layout.addWidget(self._seq_type_dropdown)
        row1_layout.addWidget(self._led_color_label)
        row1_layout.addWidget(self._led_color_dropdown)
        row1_layout.addWidget(self._button_toggle_overlay)
        row1_layout.addWidget(self._button_pixel_probe)
        row1_layout.addWidget(self._button_req_hmatrix)
        row1_layout.addWidget(self._button_use_lut)
        row1_widget = QtWidgets.QWidget()
        row1_widget.setLayout(row1_layout)
        config_layout.addWidget(row1_widget,                             1, 0, 1, 2)
        
        # New Row 2: mask pattern selection and send controls
        row2_layout = QtWidgets.QHBoxLayout()
        try:
            row2_layout.setSpacing(2)  # tighter gap between label and dropdown
            row2_layout.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass
        # Hardware trigger out toggle (left side of Mask Pattern)
        self._button_hw_trig = QtWidgets.QPushButton("HW Trigger Out")
        self._button_hw_trig.setCheckable(True)
        self._button_hw_trig.setChecked(False)
        try:
            self._button_hw_trig.setToolTip("Toggle GPIO trigger out on every projector frame (BOARD pin 22)")
        except Exception:
            pass
        self._button_hw_trig.toggled.connect(self._toggle_hw_trigger_out)
        row2_layout.addWidget(self._button_hw_trig)
        try:
            self._mask_pattern_label.setContentsMargins(0, 0, 0, 0)
            self._mask_pattern_label.setStyleSheet("margin:0px; padding-right:2px;")
        except Exception:
            pass
        # Tight pair: label + dropdown with zero spacing
        try:
            mp_pair_widget = QtWidgets.QWidget()
            mp_pair_layout = QtWidgets.QHBoxLayout(mp_pair_widget)
            mp_pair_layout.setContentsMargins(0, 0, 0, 0)
            mp_pair_layout.setSpacing(0)
            try:
                self._mask_pattern_label.setContentsMargins(0, 0, 0, 0)
                self._mask_pattern_label.setStyleSheet("margin:0px; padding-right:1px;")
            except Exception:
                pass
            mp_pair_layout.addWidget(self._mask_pattern_label)
            mp_pair_layout.addWidget(self._mask_pattern_dropdown)
            row2_layout.addWidget(mp_pair_widget)
        except Exception:
            # Fallback: add directly
            row2_layout.addWidget(self._mask_pattern_label)
            row2_layout.addWidget(self._mask_pattern_dropdown)
        row2_layout.addWidget(self._mask_pattern_browse)
        # Shift buttons left: replace stretch with a small spacing
        row2_layout.addSpacing(8)
        # New: Set Trig Params button (kept on HW Trigger Out row)
        self._button_set_trig_params = QtWidgets.QPushButton("Set Trig Params")
        try:
            self._button_set_trig_params.setToolTip("Configure TriggerDelay (µs) and ExposureTime (µs)")
        except Exception:
            pass
        self._button_set_trig_params.clicked.connect(self._open_trig_params_dialog)
        row2_layout.addWidget(self._button_set_trig_params)
        row2_widget = QtWidgets.QWidget()
        row2_widget.setLayout(row2_layout)
        config_layout.addWidget(row2_widget,                             2, 0, 1, 2)
        
        # New Row (under HW Trigger Out row): start projector trigger and send masks
        row2b_layout = QtWidgets.QHBoxLayout()
        row2b_layout.setContentsMargins(0, 0, 0, 0)
        row2b_layout.setSpacing(6)
        row2b_layout.addWidget(self._button_send_triggers)
        row2b_layout.addWidget(self._stim_mode_label)
        row2b_layout.addWidget(self._stim_mode_dropdown)
        row2b_layout.addWidget(self._button_send_masks)
        row2b_layout.addStretch(1)
        row2b_widget = QtWidgets.QWidget()
        row2b_widget.setLayout(row2b_layout)
        config_layout.addWidget(row2b_widget,                            3, 0, 1, 2, Qt.AlignLeft)
        
        # Row 3: Project ON/OFF buttons
        project_buttons_layout = QtWidgets.QHBoxLayout()
        project_buttons_layout.addWidget(self._button_project_on)
        project_buttons_layout.addWidget(self._button_project_off)
        project_buttons_layout.addSpacing(12)
        project_buttons_layout.addWidget(self._project_intensity_label)
        project_buttons_layout.addWidget(self._project_intensity_slider)
        project_buttons_layout.addWidget(self._project_intensity_value_label)
        project_buttons_layout.addStretch()
        project_buttons_widget = QtWidgets.QWidget()
        project_buttons_widget.setLayout(project_buttons_layout)
        config_layout.addWidget(project_buttons_widget,                  4, 0, 1, 2)
        
        # Row 4: Combine Trigger Line, Camera Type, and Camera Format in one row
        self._camera_format_label = QtWidgets.QLabel("Camera Format")
        row_cam_all = QtWidgets.QHBoxLayout()
        row_cam_all.setContentsMargins(0, 0, 0, 0)
        row_cam_all.setSpacing(6)
        row_cam_all.addWidget(self._label_trigger_line)
        row_cam_all.addWidget(self._dropdown_trigger_line)
        row_cam_all.addSpacing(12)
        row_cam_all.addWidget(self._camera_type_label)
        row_cam_all.addWidget(self.camera_type_dropdown)
        row_cam_all.addSpacing(12)
        row_cam_all.addWidget(self._camera_format_label)
        row_cam_all.addWidget(self._dropdown_pixel_format)
        row_cam_all_widget = QtWidgets.QWidget()
        row_cam_all_widget.setLayout(row_cam_all)
        config_layout.addWidget(row_cam_all_widget,                      5, 0, 1, 2, Qt.AlignLeft)


        capture_group = QtWidgets.QGroupBox("")
        capture_layout = QtWidgets.QGridLayout()
        capture_layout.setSpacing(3)  # Reduce spacing
        capture_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        capture_group.setLayout(capture_layout)
        capture_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #d1d1d6;
                border-radius: 6px;
                margin-top: 2px;
                font-weight: 500;
                font-size: 11px;
                color: #1c1c1e;
                background-color: #ffffff;
                padding: 4px;
            }
        """)


        capture_layout.addWidget(self._button_start_recording, 0, 0)
        capture_layout.addWidget(self._button_software_trigger, 0, 1)
        # Row 1: View Recording (single-frame slider) + Play Recording (auto-advance)
        # side-by-side. Was: View Recording spanning both cols 0-1.
        capture_layout.addWidget(self._button_view_recording, 1, 0)
        capture_layout.addWidget(self._button_play_recording, 1, 1)
        # Keep Start Recording compact and responsive to text changes
        try:
            self._button_start_recording.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_start_recording)
        except Exception:
            pass
        # Pixel format moved under Camera Type below
        # Place Real-Time Trace on the same row
        capture_layout.addWidget(self._button_show_gpu_ui, 0, 2)


        control_group = QtWidgets.QGroupBox("")
        control_group_layout = QtWidgets.QGridLayout()
        control_group_layout.setSpacing(2)  # Reduce spacing for sliders
        control_group_layout.setContentsMargins(4, 4, 4, 4)  # Reduce margins
        control_group.setLayout(control_group_layout)
        control_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #d1d1d6;
                border-radius: 6px;
                margin-top: 2px;
                font-weight: 500;
                font-size: 11px;
                color: #1c1c1e;
                background-color: #ffffff;
                padding: 4px;
            }
        """)


        self._gain_label.setAlignment(Qt.AlignCenter)
        self._gain_slider.setFixedWidth(15)  # Make narrower
        # Removed from panel; accessible via Sensor Settings window
        self._gain_value_label = QtWidgets.QLabel("1.00")
        self._gain_value_label.setAlignment(Qt.AlignCenter)
        self._gain_value_label.setStyleSheet("font-size: 10px;")
        # not added to layout


        self._dgain_label.setAlignment(Qt.AlignCenter)
        self._dgain_slider.setFixedWidth(15)  # Make narrower
        # Removed from panel; accessible via Sensor Settings window
        self._dgain_value_label = QtWidgets.QLabel("1.00")
        self._dgain_value_label.setAlignment(Qt.AlignCenter)
        self._dgain_value_label.setStyleSheet("font-size: 10px;")
        # not added to layout

        # Exposure entry (µs)
        self._exp_label = QtWidgets.QLabel("EXP (µs)")
        self._exp_label.setAlignment(Qt.AlignCenter)
        # Removed from panel; accessible via Sensor Settings window
        self._exp_line = QtWidgets.QLineEdit("")
        self._exp_line.setAlignment(Qt.AlignCenter)
        self._exp_line.setValidator(QtGui.QDoubleValidator(1.0, 1e9, 3))
        self._exp_line.editingFinished.connect(self._apply_exposure_from_text)
        # not added to layout

        # Buttons row (horizontal)
        btn_row = QtWidgets.QHBoxLayout()
        self._button_sensor_settings = QtWidgets.QPushButton("Sensor Settings")
        self._button_sensor_settings.clicked.connect(self._open_sensor_settings)
        btn_row.addWidget(self._button_sensor_settings)
        self._button_troubleshoot = QtWidgets.QPushButton("Troubleshooting")
        try:
            self._button_troubleshoot.setToolTip("Open troubleshooting tools: GPIO test, engine/camera status, performance graphs")
        except Exception:
            pass
        self._button_troubleshoot.clicked.connect(self._open_troubleshoot_window)
        btn_row.addWidget(self._button_troubleshoot)
        # ASIFT Calibration button has been moved to the top calibration panel
        # (next to Project LUT-Warped). Placed the Send I2C Command button here
        # instead so hardware-control actions (I2C) sit alongside Sensor Settings
        # and Troubleshooting.
        self._button_asift = QtWidgets.QPushButton("ASIFT Calibration")
        try:
            self._button_asift.setToolTip("Compute 3x3 H using Affine-SIFT and apply to projector")
        except Exception:
            pass
        self._button_asift.clicked.connect(self._asift_calibrate)
        btn_row.addWidget(self._button_i2c_custom)
        control_group_layout.addLayout(btn_row, 5, 0, 1, 2)

        # Camera + projection-mask orientation controls (independent flips for
        # the camera preview and the outgoing DMD mask). Persisted to one JSON
        # so the user's choices survive restarts.
        orient_row = QtWidgets.QHBoxLayout()
        _orient_file = Path(__file__).resolve().parent.parent / 'Assets' / 'Generated' / 'camera_orientation.json'
        self._cam_orient_path = _orient_file
        self._cam_rotation = 0
        self._cam_flip_h = False
        self._cam_flip_v = False
        self._mask_flip_h = False
        self._mask_flip_v = False
        try:
            if _orient_file.exists():
                import json as _jco
                with open(_orient_file) as _fco:
                    _oc = _jco.load(_fco)
                    self._cam_rotation = int(_oc.get('rotation', 0))
                    self._cam_flip_h = bool(_oc.get('flip_h', False))
                    self._cam_flip_v = bool(_oc.get('flip_v', False))
                    self._mask_flip_h = bool(_oc.get('mask_flip_h', False))
                    self._mask_flip_v = bool(_oc.get('mask_flip_v', False))
        except Exception:
            pass

        def _persist_orient():
            try:
                import json as _jco2
                self._cam_orient_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cam_orient_path, 'w') as _fco2:
                    _jco2.dump({
                        'rotation': self._cam_rotation,
                        'flip_h': self._cam_flip_h,
                        'flip_v': self._cam_flip_v,
                        'mask_flip_h': self._mask_flip_h,
                        'mask_flip_v': self._mask_flip_v,
                    }, _fco2)
            except Exception:
                pass

        self._button_rotate = QtWidgets.QPushButton(f"Rotate 90\u00b0 ({self._cam_rotation}\u00b0)")
        def _on_rotate():
            self._cam_rotation = (self._cam_rotation + 90) % 360
            self._button_rotate.setText(f"Rotate 90\u00b0 ({self._cam_rotation}\u00b0)")
            _persist_orient()
        self._button_rotate.clicked.connect(_on_rotate)
        self._button_rotate.setToolTip("Rotate the camera preview by 90°. Does not rotate the projection mask.")
        orient_row.addWidget(self._button_rotate)

        self._check_flip_h = QtWidgets.QCheckBox("Cam Flip H")
        self._check_flip_h.setChecked(self._cam_flip_h)
        self._check_flip_h.setToolTip("Mirror the camera preview horizontally. Affects display + recording. Independent of projection mask.")
        self._check_flip_h.toggled.connect(lambda v: (setattr(self, '_cam_flip_h', v), _persist_orient()))
        orient_row.addWidget(self._check_flip_h)

        self._check_flip_v = QtWidgets.QCheckBox("Cam Flip V")
        self._check_flip_v.setChecked(self._cam_flip_v)
        self._check_flip_v.setToolTip("Mirror the camera preview vertically. Affects display + recording. Independent of projection mask.")
        self._check_flip_v.toggled.connect(lambda v: (setattr(self, '_cam_flip_v', v), _persist_orient()))
        orient_row.addWidget(self._check_flip_v)

        # Projection-mask flips — applied inside zmq_mask_sender.py via --flip-x/--flip-y.
        # Auto-restarts the mask sender if it's already running.
        def _on_mask_flip_changed(attr, v):
            setattr(self, attr, v)
            _persist_orient()
            # If mask sender is running, restart it so the new flip takes effect
            try:
                QProcess = self._ensure_qprocess()
                if (self._proc_masks is not None
                        and self._proc_masks.state() != QProcess.NotRunning):
                    print("[MASK] Flip changed — restarting mask sender")
                    self._proc_masks.kill()
                    self._proc_masks.waitForFinished(2000)
                    # Re-launch after a short delay to let cleanup finish
                    from PyQt5.QtCore import QTimer as _QT
                    _QT.singleShot(300, self._toggle_send_masks)
            except Exception as e:
                print(f"[MASK] Flip restart failed: {e}")

        self._check_mask_flip_h = QtWidgets.QCheckBox("Mask Flip H")
        self._check_mask_flip_h.setChecked(self._mask_flip_h)
        self._check_mask_flip_h.setToolTip("Flip the outgoing DMD projection mask horizontally. Auto-restarts mask sender.")
        self._check_mask_flip_h.toggled.connect(lambda v: _on_mask_flip_changed('_mask_flip_h', v))
        orient_row.addWidget(self._check_mask_flip_h)

        self._check_mask_flip_v = QtWidgets.QCheckBox("Mask Flip V")
        self._check_mask_flip_v.setChecked(self._mask_flip_v)
        self._check_mask_flip_v.setToolTip("Flip the outgoing DMD projection mask vertically. Auto-restarts mask sender.")
        self._check_mask_flip_v.toggled.connect(lambda v: _on_mask_flip_changed('_mask_flip_v', v))
        orient_row.addWidget(self._check_mask_flip_v)

        control_group_layout.addLayout(orient_row, 6, 0, 1, 2)

        # Offline Setup button
        self._button_offline_setup = QtWidgets.QPushButton("Offline Setup")
        self._button_offline_setup.setStyleSheet("background-color: #1f6feb; color: white; font-weight: bold;")
        self._button_offline_setup.clicked.connect(self._open_offline_setup_dialog)
        control_group_layout.addWidget(self._button_offline_setup, 7, 0)

        # Trace Extraction Test button
        self._button_trace_test = QtWidgets.QPushButton("Trace Test")
        self._button_trace_test.setStyleSheet("background-color: #d29922; color: black; font-weight: bold;")
        self._button_trace_test.clicked.connect(self._open_trace_test_dialog)
        control_group_layout.addWidget(self._button_trace_test, 8, 0, 1, 2)

        # Zoom controls removed - using mouse wheel zoom instead


        # Set control panel widths for larger buttons
        control_group.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed
        )
        
        for grp in (config_group, capture_group):
            grp.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Preferred
            )


        # Remove stretching for more compact layout
        button_bar_layout.setColumnStretch(4, 0)  # No stretching for compact panels
        button_bar_layout.setColumnStretch(5, 0)  # No stretching for sliders
        button_bar_layout.setColumnStretch(6, 0)
        button_bar_layout.setColumnStretch(7, 0)

        # New: Top calibration panel (above hardware trigger/config zone)
        try:
            calib_panel = QtWidgets.QWidget()
            calib_panel.setObjectName("calib_panel")
            calib_layout = QtWidgets.QHBoxLayout(calib_panel)
            calib_layout.setContentsMargins(6, 6, 6, 6)
            calib_layout.setSpacing(6)
            # Style similar to other panels but without a title area
            calib_panel.setStyleSheet(
                "border: 1px solid #d1d1d6;"
                "border-radius: 6px;"
                "margin-top: 2px;"
                "font-size: 11px;"
                "color: #1c1c1e;"
                "background-color: #ffffff;"
                "padding: 4px;"
                " QPushButton { font-weight: normal; color: #000000;"
                "   background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f5f5f7, stop:1 #eaeaef);"
                "   border: 1px solid #cfcfd6; border-radius: 6px; padding: 4px 10px; }"
                " QPushButton:hover {"
                "   background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f1f6);"
                "   border: 1px solid #bdbdca; }"
                " QPushButton:pressed { background-color: #e6e6ee; }"
                " QPushButton:disabled { color: #b8b6c9; background-color: #fafafa; border: 1px solid #eeeeee; }"
            )
            # Move calibration-related controls here
            calib_layout.addWidget(self._button_calibrate)
            calib_layout.addWidget(self._button_sl_calibrate)
            try:
                calib_layout.addWidget(self._chk_phase_refine)
            except Exception:
                pass
            calib_layout.addWidget(self._button_sl_project_reg)
            # ASIFT Calibration moved here (was in the mid row next to Troubleshooting)
            calib_layout.addWidget(self._button_asift)
            # Place the new panel at the very top-left
            button_bar_layout.addWidget(calib_panel, 0, 0, 1, 1)
        except Exception:
            pass

        # Shift everything to the left to align with video preview; push existing panels down
        button_bar_layout.addWidget(config_group, 1, 0, 4, 1)       # Column 0 (under calibration panel)
        button_bar_layout.addWidget(capture_group, 5, 0, 2, 1)      # Column 0, below config
        # Keep control panel as a separate panel below the left column panels
        button_bar_layout.addWidget(control_group,                  7, 0, 1, 1, Qt.AlignLeft)
        
        # Add spacer to push everything to the left
        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        button_bar_layout.addItem(spacer, 0, 2, 7, 1)  # Column 2, fill remaining space



        self._button_start_hardware_acquisition.setToolTip("Start/Stop acquiring images using hardware triggering rather than real time(RT) acquisition. Hardware Trigger FPS must stay <45 hz")
        self._button_start_recording.setToolTip("Start/Stop recording video of the live feed.")
        self._button_software_trigger.setToolTip("Save the next processed frame.")
        self._button_send_triggers.setToolTip("Start/Stop sending projector triggers over I2C.")
        self._button_send_masks.setToolTip("Start/Stop sending masks over ZMQ to the projector.")
        self._button_start_projector.setToolTip("Start/Stop the projection engine binary with configured options.")


        self._gain_label.setToolTip("Adjust the analog gain level (brightness).")
        self._dgain_label.setToolTip("Adjust the digital gain level.")
        try:
            self._exp_label.setToolTip("Exposure in microseconds. Default 33333.333 (≈30 FPS).")
            self._exp_line.setToolTip("Type exposure in µs and press Enter.")
        except Exception:
            pass
        # Zoom tooltip removed - using mouse wheel zoom instead


        button_bar.setLayout(button_bar_layout)
        self._layout.addWidget(button_bar)

        # SL progress widgets are created in _create_statusbar so they sit on the status bar row
        self._sl_progress = None
        self._sl_status = None


"""LiveTracesMixin — extracted from ``gpu_ui.py`` per L5 SPLIT-FIRST.

Cluster #3 of the 9-sub-module decomposition (see
``docs/specs/L5_UI/gpu_ui.md`` §0.5). Contains the methods that
manage live trace extraction lifecycle:

- ``_on_trace_mode_changed(mode)`` — combobox slot for plot mode
- ``start_live_traces()`` — Qt slot; spawns the LiveTraceExtractor
- ``_toggle_oasis(checked)`` — toggle online OASIS deconvolution
- ``stop_live_traces()`` — tear-down + cleanup

Pure mixin (does NOT inherit from QWidget). The host class is
expected to be a ``QtWidgets.QWidget`` subclass and to provide the
following:

Required state attributes (set by ``__init__``):
    - ``self.camera`` — IDS Peak camera handle (used for FPS / acquisition)
    - ``self.proj_display`` — ``ProjectDisplay`` instance or ``None``
    - ``self.rois_path: str`` — ROI NPZ path (default ``"rois.npz"``)
    - ``self.plot_widget`` — pyqtgraph PlotWidget or ``None``
    - ``self.live_extractor: Optional[LiveTraceExtractor]`` — the engine
    - ``self._trace_mode_combo`` — QComboBox for trace plot mode
    - ``self._button_oasis_online`` — QPushButton (checkable) or None
    - ``self.current_labels`` — optional in-memory labels override

Required Qt signals on the host:
    - ``start_live_traces`` is connected to ``requestStartLiveTraces``
      in the host's ``__init__``; the mixin assumes that wiring exists.

The mixin defines the ``@pyqtSlot()`` decorator on ``start_live_traces``
and ``@pyqtSlot()``-equivalent semantics on ``stop_live_traces`` to
preserve the existing signal wiring contract.
"""

from __future__ import annotations

import os
import time

from PyQt5.QtCore import pyqtSlot

from live_trace.extractor import LiveTraceExtractor


class LiveTracesMixin:
    """Live trace extraction lifecycle.

    See module docstring for the host-class contract.
    """

    def _on_trace_mode_changed(self, mode: str):
        if self.live_extractor is not None:
            try:
                self.live_extractor.set_plot_normalization(mode)
            except Exception:
                pass

    @pyqtSlot()
    def start_live_traces(self):
        # Shutdown guard: refuse to start when the host is closing.
        # Queued QTimer.singleShot(N, self.start_live_traces) callbacks
        # (scheduled by gpu_ui error-recovery / memory-pressure paths) can
        # fire during closeEvent's processEvents() drain — that's how a
        # new LiveTraceExtractor was being spawned AFTER the user closed
        # the window. The host sets `_shutting_down=True` at the very top
        # of closeEvent so this guard fires before construction begins.
        if getattr(self, "_shutting_down", False):
            print("⛔ Refusing to start live traces during shutdown")
            return

        print("🚀 Starting live traces with enhanced safety...")


        if self.live_extractor is not None:
            print("🔄 Live extractor already exists. Performing clean restart...")
            try:
                self.stop_live_traces()

                from PyQt5.QtCore import QCoreApplication
                QCoreApplication.processEvents()
                import time
                time.sleep(0.1)
            except Exception as stop_error:
                print(f"⚠️ Error during extractor stop: {stop_error}")


        if not getattr(self.camera, "acquisition_running", False):
            print("📷 Starting camera acquisition for live traces...")
            try:
                if not self.camera.start_realtime_acquisition():
                    print("❌ Failed to start camera acquisition; cannot start live traces.")
                    return
                print("✅ Camera acquisition started")
            except Exception as cam_error:
                print(f"❌ Camera acquisition error: {cam_error}")
                return

        roi_path = self.rois_path
        if not os.path.exists(roi_path):
            print("❌ No ROI file found. Run Discover/Manual Mask first.")
            return

        print(f"📊 Using ROI file: {roi_path}")

        try:

            use_pygame = (self.plot_widget is None)

            self.live_extractor = LiveTraceExtractor(
                camera=self.camera,
                label_path=self.rois_path,
                plot_widget=self.plot_widget,
                max_points=300,
                max_rois=50,
                use_pygame_plot=False,
                enable_sync=False,
            )

            try:
                enabled = getattr(self, '_button_oasis_online', None) is not None and self._button_oasis_online.isChecked()
                if enabled and hasattr(self.live_extractor, 'set_oasis_enabled'):
                    self.live_extractor.set_oasis_enabled(True)
            except Exception:
                pass

            try:
                mode = self._trace_mode_combo.currentText()
                self.live_extractor.set_plot_normalization(mode)
            except Exception:
                pass

            print("Live trace extractor started.")
        except Exception as e:
            print(f"Failed to start live traces: {e}")

    def _toggle_oasis(self, checked: bool):
        try:
            if self.live_extractor is not None and hasattr(self.live_extractor, 'set_oasis_enabled'):
                self.live_extractor.set_oasis_enabled(bool(checked))
                print(f"[UI] OASIS online deconvolution {'enabled' if checked else 'disabled'}")
        except Exception as e:
            print(f"[UI] Failed to toggle OASIS: {e}")


    def stop_live_traces(self):
        try:
            if self.live_extractor is not None:
                # LiveTraceExtractor.stop() internally disconnects the
                # camera signal it actually attached to (tracked in
                # _connected_camera_signal). The earlier code referenced
                # self.camera.image_update_signal which doesn't exist on
                # OptimizedCamera — that failed silently and left the real
                # `frame_ready` → on_frame connection in place, so restarts
                # accumulated duplicate connections.
                try:
                    self.live_extractor.stop()
                except Exception as e:
                    print(f"⚠️ live_extractor.stop() raised: {e}")
                self.live_extractor = None
                print("Live trace extractor stopped.")
        except Exception as e:
            print(f"Error stopping live trace extractor: {e}")

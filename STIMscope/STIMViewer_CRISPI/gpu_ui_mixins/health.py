"""HealthMonitoringMixin — error-handling + orderly shutdown for the GPU UI.

Bundles the methods that handle errors and process termination:

* ``_setup_long_term_stability()`` — init error counters; register atexit +
  SIGINT/SIGTERM signal handlers.
* ``_handle_error(error, context)`` / ``_safe_cleanup()`` /
  ``_emergency_cleanup()`` — error-handling ladder; escalate to emergency
  teardown after sustained error rate.
* ``_signal_handler(signum, frame)`` / ``shutdown()`` / ``closeEvent(event)``
  — orderly process termination, deliberate one-time cleanup at the
  teardown point (no periodic monitoring).

Periodic memory/CPU/GPU monitoring + threshold-gated gc.collect were
removed: they added overhead, fired spurious warnings on the 64 GB
unified-memory Jetson, and their cleanup paths were unreliable (monitor
threads outliving the window after close). Python's automatic gc and
CuPy's pool defaults are correct for this workload.
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
import atexit
import gc
import signal
from collections import deque
import psutil
from PyQt5.QtCore import QTimer
from gpu_ui_mixins._shared import CUDA_AVAILABLE, CUDA_USABLE, cp

class HealthMonitoringMixin:
    """Cluster 8 — long-term stability + health monitoring + shutdown."""

    def _setup_long_term_stability(self):
        # Error-counter state for _handle_error / _safe_cleanup / _emergency_cleanup.
        # Periodic memory/CPU/GPU monitoring and the threshold-gated gc.collect
        # were removed: they added overhead, fired spurious warnings on the
        # 64 GB unified-memory Jetson, and their cleanup paths were unreliable
        # (monitor threads outliving the window after close). Python's automatic
        # gc + CuPy's pool defaults are correct for this workload; explicit
        # cleanup still runs at deliberate teardown points (_safe_cleanup on
        # error, _emergency_cleanup on atexit / signal, closeEvent on UI close).
        self._error_count = 0
        self._last_error_time = 0.0
        self._max_errors_per_minute = 5

        atexit.register(self._emergency_cleanup)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    # ─── ROI discovery cluster extracted to gpu_ui_roi_discovery.ROIDiscoveryMixin
    #     (iter-1 of L5 SPLIT-FIRST per docs/specs/L5_UI/gpu_ui.md §0.5)
    # ─── Live traces cluster extracted to gpu_ui_traces.LiveTracesMixin
    #     (iter-2 of L5 SPLIT-FIRST per docs/specs/L5_UI/gpu_ui.md §0.5)
    # ─── Napari ROI editor launch extracted to gpu_ui_napari.NapariViewerMixin
    #     (iter-3 of L5 SPLIT-FIRST per docs/specs/L5_UI/gpu_ui.md §0.5)
    # ─── FAST export path extracted to gpu_ui_export_fast.FastExportMixin
    #     (iter-4 of L5 SPLIT-FIRST per docs/specs/L5_UI/gpu_ui.md §0.5)
    # ─── SLOW export path extracted to gpu_ui_export_slow.SlowExportMixin
    #     (iter-5 of L5 SPLIT-FIRST per docs/specs/L5_UI/gpu_ui.md §0.5)
    # ─── Export viewer dialog skeleton extracted to gpu_ui_export_viewer.ExportViewerMixin
    #     (iter-6 of L5 SPLIT-FIRST per docs/specs/L5_UI/gpu_ui.md §0.5)



    def _handle_error(self, error: Exception, context: str = ""):
        self._error_count += 1
        self._last_error_time = time.time()
        print(f"Error in {context}: {error}")
        self._safe_cleanup()
        if self._error_count > self._max_errors_per_minute:
            print("Too many errors; performing emergency cleanup")
            self._emergency_cleanup()

    def _safe_cleanup(self):
        try:
            gc.collect()
            if CUDA_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
        except Exception as e:
            print(f"Safe cleanup error: {e}")

    def _emergency_cleanup(self):
       
        try:
            print("🆘 Emergency cleanup initiated...")
            

            self.stop_live_traces()
            

            try:
                if hasattr(self.camera, 'stop_realtime_acquisition'):
                    self.camera.stop_realtime_acquisition()
                    print("📷 Camera acquisition stopped")
            except Exception as e:
                print(f"⚠️ Camera cleanup warning: {e}")
            

            try:
                gc.collect()
                print("🗑️ Memory garbage collected")
            except Exception:
                pass
                

            if CUDA_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    print("🎮 GPU memory cleaned")
                except Exception:
                    pass
            
            print("✅ Emergency cleanup completed successfully")
            
        except Exception as e:
            print(f"❌ Error during emergency cleanup: {e}")

    def _signal_handler(self, signum, frame):
        print(f"🛑 Received signal {signum}, performing graceful cleanup…")
        self._emergency_cleanup()
        
    def shutdown(self):
        self._shutting_down = True
        self.close()

    def closeEvent(self, event):
        # Always tear down fully on close (operator X *or* app shutdown). Keep
        # the shutdown guard SET so the every() monitor timers stop rescheduling
        # — a monitor resuming on a kept-alive window is exactly what caused the
        # post-close "high memory" warnings + gc.collect churn. Release buffers,
        # then accept() so the widget is destroyed (WA_DeleteOnClose) and memory
        # is freed. The parent clears its reference via the `closed` signal, so
        # reopening reconstructs a fresh instance (~0.04 s) — graceful reopen
        # AND freed memory, without the hide-and-leak tradeoff.
        self._shutting_down = True

        try:
            print("Real-time trace window closing — cleaning up...")


            try:
                self.stop_live_traces()
                print("Live traces stopped")
            except Exception as e:
                print(f"Error stopping live traces: {e}")


            try:
                if hasattr(self, 'proj_display') and self.proj_display:
                    self.proj_display.close()
                    self.proj_display = None
                    print("Projection display closed")
            except Exception as e:
                print(f"Error closing projection display: {e}")


            # Deliberate one-time teardown of the CuPy memory pool at UI close.
            # Buffers are released by stop_live_traces above; this returns the
            # cached pool memory to the OS (the pool otherwise stays allocated
            # for the process lifetime). Single explicit free at a deliberate
            # teardown point — not periodic, not threshold-gated.
            try:
                gc.collect()
                if CUDA_AVAILABLE:
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error in close-time cleanup: {e}")


            try:
                self.closed.emit()
            except Exception as e:
                print(f"Error emitting close signal: {e}")


            try:
                from PyQt5.QtCore import QCoreApplication
                QCoreApplication.processEvents()
            except Exception as e:
                print(f"Error processing events: {e}")

            event.accept()
            print("Real-time trace window closed")

        except Exception as e:
            print(f"Critical close event error: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")
            try:
                self.closed.emit()
            except Exception:
                pass
            event.accept()

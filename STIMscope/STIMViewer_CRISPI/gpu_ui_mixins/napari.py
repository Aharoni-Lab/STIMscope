"""NapariViewerMixin — Napari-based ROI editor launch path.

PLANNED REMOVAL (not yet executed): this mixin and the underlying
napari dependency are slated for removal because the napari ROI-refine
workflow is incomplete and not part of the publication scope. Holding
off on the actual deletion pending a decision on whether to (a) drop
the "Refine ROIs" feature entirely, or (b) replace it with a
non-napari editor. See docs/IMPLEMENTATION_NOTES.md ("Planned
removals") for the full deletion checklist.

Currently active. The mixin is still wired into gpu_ui.py and the
refineRequested signal is still connected to _launch_napari_viewer.



Cluster #4 of the 9-sub-module decomposition (see
``docs/specs/L5_UI/gpu_ui.md`` §0.5). Contains the single very large
method that launches the Napari ROI editor — pausing camera /
projector / live-trace extraction, dispatching the refine workflow,
and restoring all paused subsystems on close:

- ``_launch_napari_viewer(mean, masks)`` — Qt slot wired to
  ``refineRequested`` signal. Pauses live-traces + camera +
  projector, validates mask shape (3D-stack vs 2D-labels),
  launches ``roi_editor.refine_rois`` with a restore-on-close
  callback that re-projects the updated mask + restarts traces.

Pure mixin (does NOT inherit from QWidget). The host class is
expected to be a ``QtWidgets.QWidget`` subclass and to provide the
following host contract:

Required state attributes:
    - ``self.camera`` — IDS Peak camera handle. The mixin reads
      ``is_recording``, ``acquisition_running``, ``translation_matrix``
      and calls ``stop_realtime_acquisition()`` /
      ``start_realtime_acquisition()``.
    - ``self.proj_display`` — ``ProjectDisplay`` instance or ``None``;
      reassigned during the restore closure.
    - ``self.rois_path: str`` — ROI NPZ path; read + written
      (``np.savez_compressed``).
    - ``self.plot_widget`` — pyqtgraph PlotWidget or ``None``;
      may be re-created inside the restart closure.
    - ``self.live_extractor`` — set/cleared by ``start_live_traces`` /
      ``stop_live_traces``; the restart closure also performs an
      in-place ``cleanup()`` to drop the extractor before restart.
    - ``self.layout`` — QVBoxLayout (or similar) on the host widget;
      used by the restart-with-new-rois closure when the plot widget
      needs reattachment.
    - ``self.current_labels`` — written with the refined labels array
      returned from ``refine_rois``.

Required host methods (provided by either the residual ``GPU`` class
or sibling mixins):
    - ``self.stop_live_traces()`` — from ``LiveTracesMixin``.
    - ``self.start_live_traces()`` — from ``LiveTracesMixin``.
    - ``self._handle_error(error, context)`` — from residual GPU.

Required Qt signal wiring (set up by host ``__init__``):
    - ``self.refineRequested.connect(self._launch_napari_viewer)``
      — the host still owns the signal; the mixin only provides the
      slot.

The mixin preserves the ``@pyqtSlot(object, object)`` decorator on
``_launch_napari_viewer`` to keep the existing signal-wiring contract.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, pyqtSlot

# Mirror gpu_ui.py module-top constant (defined there at module load,
# always True; reproduced here so the mixin is self-contained and
# avoids a circular import on gpu_ui).
PLOT_WITH_PYQTGRAPH = True


class NapariViewerMixin:
    """Napari ROI editor launch + restore-on-close workflow.

    See module docstring for the host-class contract.
    """

    @pyqtSlot(object, object)
    def _launch_napari_viewer(self, mean, masks):

        try:

            was_recording = self.camera.is_recording if self.camera else False
            was_live_traces = hasattr(self, 'live_extractor') and self.live_extractor is not None



            if was_live_traces:
                self.stop_live_traces()
                print("📊 Live traces paused for Napari launch")


            was_camera_running = self.camera.acquisition_running if self.camera else False
            if was_camera_running:
                self.camera.stop_realtime_acquisition()
                print("📷 Camera acquisition paused for Napari launch")


            try:
                if self.proj_display:
                    self.proj_display.close()
            except Exception:
                pass


            time.sleep(0.2)

            def restore_after_napari(event=None):

                try:
                    print("🔄 Restoring operations after Napari close...")


                    time.sleep(0.1)


                    if was_camera_running and self.camera:
                        self.camera.start_realtime_acquisition()
                        print("📷 Camera acquisition restored")


                    try:
                        from projection import ProjectDisplay
                        from PyQt5.QtGui import QGuiApplication


                        if os.path.exists(self.rois_path):
                            try:
                                roi_data = np.load(self.rois_path)
                                if 'binary' in roi_data:
                                    # Prefer union binary mask
                                    binary = roi_data["binary"].astype(np.uint8)
                                    print("🔄 Re-projecting updated binary mask")
                                    labels = (binary > 0).astype(np.int32)
                                elif 'labels' in roi_data:
                                    labels = roi_data["labels"]
                                    print(f"🔄 Re-projecting updated ROIs: {len(np.unique(labels))-1} ROIs")
                                else:
                                    labels = np.load(self.rois_path)["labels"]
                                    print("🔄 Re-projecting original ROIs")
                            except Exception as e:
                                print(f"⚠️ Could not load updated ROIs: {e}")

                                labels = np.load(self.rois_path)["labels"]
                        else:
                            print("⚠️ No ROI file found for re-projection")
                            return

                        # Build grayscale from binary/labels
                        if labels.dtype != np.int32:
                            labels = labels.astype(np.int32)
                        img_gray = ((labels > 0).astype(np.uint8) * 255).astype(np.uint8)

                        screens = QGuiApplication.screens()
                        scr = screens[1] if len(screens) > 1 else screens[0]
                        size = scr.size()
                        tgt_w, tgt_h = size.width(), size.height()

                        # If mask image is smaller than projector screen, pad with black instead of resizing
                        h, w = img_gray.shape[:2]
                        if h <= tgt_h and w <= tgt_w:
                            pad_top = (tgt_h - h) // 2
                            pad_bottom = tgt_h - h - pad_top
                            pad_left = (tgt_w - w) // 2
                            pad_right = tgt_w - w - pad_left
                            try:
                                img_gray = cv2.copyMakeBorder(
                                    img_gray, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=0
                                )
                            except Exception:
                                # Fallback to numpy pad if OpenCV fails
                                img_gray = np.pad(
                                    img_gray,
                                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                                    mode='constant', constant_values=0
                                )
                        else:
                            # If larger or mismatched, keep existing nearest-neighbor resize
                            img_gray = cv2.resize(img_gray, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)

                        if self.proj_display:
                            try:
                                self.proj_display.close()
                            except Exception:
                                pass
                        self.proj_display = ProjectDisplay(scr)
                        H = getattr(self.camera, "translation_matrix", None)
                        self.proj_display.show_image_fullscreen_on_second_monitor(img_gray, H)
                        print("🖥️ Updated binary mask re-projected")


                        if was_live_traces:
                            def restart_with_new_rois():
                                try:
                                    print("🔄 Attempting to restart live traces with updated ROIs...")


                                    if hasattr(self, 'live_extractor') and self.live_extractor:
                                        print("🧹 Cleaning up existing extractor...")
                                        self.live_extractor.cleanup()
                                        self.live_extractor = None


                                    import gc
                                    gc.collect()


                                    from PyQt5.QtCore import QCoreApplication
                                    QCoreApplication.processEvents()
                                    import time
                                    time.sleep(0.1)


                                    if not self.plot_widget or not hasattr(self.plot_widget, 'plot'):
                                        print("📊 Reinitializing plot widget for live traces...")
                                        try:
                                            if PLOT_WITH_PYQTGRAPH:
                                                import pyqtgraph as pg
                                                self.plot_widget = pg.PlotWidget()
                                                self.plot_widget.setLabel('left', 'Intensity')
                                                self.plot_widget.setLabel('bottom', 'Time (frames)')
                                                self.plot_widget.showGrid(x=True, y=True)


                                                if self.plot_widget not in [self.layout.itemAt(i).widget() for i in range(self.layout.count()) if self.layout.itemAt(i) and self.layout.itemAt(i).widget()]:
                                                    self.layout.addWidget(self.plot_widget)
                                                print("✅ Plot widget reinitialized")
                                        except Exception as plot_error:
                                            print(f"⚠️ Plot widget reinit failed: {plot_error}")


                                    self.start_live_traces()


                                    if hasattr(self, 'live_extractor') and self.live_extractor:

                                        if hasattr(self.live_extractor, 'restart_after_napari'):
                                            restart_success = self.live_extractor.restart_after_napari(self.plot_widget)
                                            if restart_success:
                                                print("✅ LiveTraceExtractor restarted successfully after Napari")
                                            else:
                                                print("⚠️ LiveTraceExtractor restart had issues, using fallback")

                                                self.live_extractor.plot_widget = self.plot_widget
                                                if hasattr(self.live_extractor, '_setup_pagination_controls'):
                                                    self.live_extractor._setup_pagination_controls()
                                        else:

                                            self.live_extractor.plot_widget = self.plot_widget
                                            if hasattr(self.live_extractor, '_setup_pagination_controls'):
                                                self.live_extractor._setup_pagination_controls()

                                    print("✅ Live traces restarted successfully with updated ROIs")
                                except Exception as restart_error:
                                    print(f"❌ Failed to restart live traces: {restart_error}")
                                    import traceback
                                    print(f"   Stack trace: {traceback.format_exc()}")


                                    def fallback_restart():
                                        try:
                                            self.start_live_traces()
                                            print("✅ Fallback restart successful")
                                        except Exception as fallback_error:
                                            print(f"❌ Fallback restart also failed: {fallback_error}")

                                    QTimer.singleShot(2000, fallback_restart)

                            QTimer.singleShot(1000, restart_with_new_rois)  # Increased delay
                            print("📊 Live traces scheduled for restart with updated ROIs")

                    except Exception as e:
                        print(f"⚠️ Failed to re-project mask: {e}")

                        if was_live_traces:
                            QTimer.singleShot(500, self.start_live_traces)
                            print("📊 Live traces scheduled for restart (projection failed)")

                    print("✅ All operations restored successfully")

                except Exception as e:
                    print(f"❌ Error restoring operations: {e}")
                    self._handle_error(e, "restore_after_napari")


            try:


                try:
                    from roi_editor import refine_rois
                    roi_editor_available = True
                except ImportError as e:
                    print(f"❌ roi_editor import failed: {e}")
                    print("❌ Cannot proceed without roi_editor")
                    restore_after_napari()
                    return
                except Exception as e:
                    print(f"❌ roi_editor import failed with unexpected error: {e}")
                    print("❌ Cannot proceed without roi_editor")
                    restore_after_napari()
                    return
                from roi_editor import refine_rois


                if isinstance(masks, np.ndarray):

                    if masks.ndim == 3:

                        if masks.shape[0] > 0 and masks.shape[1:] == mean.shape:
                            print(f"🔄 Converting 3D mask array ({masks.shape}) to list of 2D masks")
                            mask_list = []
                            for i in range(masks.shape[0]):
                                mask = masks[i].astype(bool)
                                if mask.sum() > 0:  # Only add non-empty masks
                                    mask_list.append(mask)
                            masks = mask_list
                            print(f"✅ Converted to {len(masks)} individual masks")
                        else:
                            # Attempt to resize masks to match mean shape using nearest neighbor
                            try:
                                H, W = mean.shape
                                print(f"ℹ️ Resizing 3D masks from {masks.shape[1:]} to {(H, W)} with nearest-neighbor")
                                mask_list = []
                                for i in range(masks.shape[0]):
                                    m = masks[i]
                                    if m.shape != mean.shape:
                                        m_resized = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                                    else:
                                        m_resized = m.astype(np.uint8)
                                    mr = m_resized.astype(bool)
                                    if mr.sum() > 0:
                                        mask_list.append(mr)
                                if len(mask_list) == 0:
                                    print("❌ All resized masks were empty; aborting")
                                    restore_after_napari()
                                    return
                                masks = mask_list
                                print(f"✅ Resized and converted to {len(masks)} individual masks")
                            except Exception as rez_err:
                                print(f"❌ Failed to resize 3D masks: {rez_err}")
                                restore_after_napari()
                                return
                    elif masks.ndim == 2:

                        # If labels array doesn't match mean shape, resize labels with nearest neighbor
                        if masks.shape != mean.shape:
                            try:
                                H, W = mean.shape
                                print(f"ℹ️ Resizing 2D labels from {masks.shape} to {(H, W)} with nearest-neighbor")
                                masks = cv2.resize(masks.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
                            except Exception as rez2_err:
                                print(f"❌ Failed to resize labels: {rez2_err}")
                                restore_after_napari()
                                return

                        print(f"🔄 Converting 2D labels array ({masks.shape}) to list of 2D masks")
                        unique_ids = np.unique(masks)
                        mask_list = []
                        for rid in unique_ids[1:]:  # Skip background (0)
                            mask = masks == rid
                            if mask.sum() > 0:  # Only add non-empty masks
                                mask_list.append(mask)
                        masks = mask_list
                        print(f"✅ Converted to {len(masks)} individual masks")
                    else:
                        print(f"⚠️ Unexpected mask array shape: {masks.shape}")
                        restore_after_napari()
                        return


                if not isinstance(masks, list) or len(masks) == 0:
                    print("❌ No valid masks found")
                    restore_after_napari()
                    return


                for i, mask in enumerate(masks):
                    if not isinstance(mask, np.ndarray) or mask.shape != mean.shape:
                        print(f"⚠️ Mask {i} has invalid shape: {mask.shape if hasattr(mask, 'shape') else type(mask)}, expected {mean.shape}")
                        masks[i] = None


                masks = [mask for mask in masks if mask is not None]

                if len(masks) == 0:
                    print("❌ No valid masks after validation")
                    restore_after_napari()
                    return

                print(f"✅ Prepared {len(masks)} valid masks for ROI editor")


                if 'refine_rois' in locals() and roi_editor_available:

                                    try:
                                        labels_array = refine_rois(mean, masks, return_viewer=False, on_close_callback=restore_after_napari)


                                        self.current_labels = labels_array


                                        if labels_array is not None:

                                            try:

                                                existing_data = np.load(self.rois_path)


                                                updated_data = {
                                                    'labels': labels_array,
                                                    'masks': existing_data.get('masks', []),
                                                    'sizes': existing_data.get('sizes', [])
                                                }


                                                np.savez_compressed(self.rois_path, **updated_data)
                                                print(f"✅ Updated ROI file saved: {self.rois_path}")

                                            except Exception as save_error:
                                                print(f"⚠️ Could not save updated ROIs: {save_error}")

                                    except Exception as napari_error:
                                        print(f"❌ Napari ROI editing failed: {napari_error}")
                                        restore_after_napari()  # Still restore state
                                        return

                                    print("✅ Napari ROI editor launched successfully with OpenGL safety")

                else:
                    print("❌ refine_rois function not available")
                    restore_after_napari()
                    return

            except Exception as e:
                print(f"❌ Error launching Napari: {e}")
                self._handle_error(e, "launch_napari")
                restore_after_napari()

        except Exception as e:
            print(f"❌ Error in Napari launch process: {e}")
            self._handle_error(e, "napari_launch")

"""ROIDiscoveryMixin — extracted from ``gpu_ui.py`` per L5 SPLIT-FIRST.

Cluster #2 of the 9-sub-module decomposition (see
``docs/specs/L5_UI/gpu_ui.md`` §0.5). Contains the 8 methods that wrap
the user-driven ROI discovery surface:

- ``_select_video`` — file dialog for video source
- ``_run_make_memmap`` / ``_thread_make_memmap`` — memmap conversion
- ``_load_roi_file`` — pick an existing NPZ ROI file
- ``_run_discover_rois`` / ``_thread_discover_rois`` — OTSU + Cellpose
  segmentation entry points
- ``_run_refine_rois`` / ``_thread_refine_rois`` — napari refinement

Pure mixin (does NOT inherit from QWidget). The host class is expected
to be a ``QtWidgets.QWidget`` subclass and to provide the following:

Required state attributes (set by ``__init__``):
    - ``self.camera`` — IDS Peak camera handle (has ``translation_matrix``)
    - ``self.video_path: Optional[str]`` — currently-selected source file
    - ``self.memmap_path: str`` — target path for memmap (default
      ``"movie_mmap.npy"``)
    - ``self.rois_path: str`` — ROI NPZ path (default ``"rois.npz"``)
    - ``self._discover_method: str`` — segmentation backend
      ("OTSU"/"Cellpose"/"CNMF"/"Custom")
    - ``self.proj_display`` — ``ProjectDisplay`` instance or ``None``

Required Qt signals (defined as class attributes on the host):
    - ``refineRequested(object, object)`` — emit ``(mean, masks)``
    - ``requestStartLiveTraces()`` / ``requestStopLiveTraces()``

Required host methods:
    - ``self._handle_error(exc, where)`` — error sink
    - ``self.start_live_traces()`` — provided by the live-traces mixin
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading

import cv2
import numpy as np
from PyQt5 import QtWidgets


class ROIDiscoveryMixin:
    """Methods responsible for ROI discovery, refinement, and memmap I/O.

    See module docstring for the host-class contract.
    """

    def _select_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video files (*.avi *.mp4 *.h5 *.npy *.npz *.tif *.tiff *.ome.tif *.ome.tiff)"
        )
        if path:
            self.video_path = path
            print(f"Selected video: {path}")

    def _run_make_memmap(self):
        threading.Thread(target=self._thread_make_memmap, daemon=True).start()

    def _thread_make_memmap(self):
        print("Making memmap…")
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                print("No valid video file selected")
                return
            size_mb = os.path.getsize(self.video_path) / (1024 * 1024)
            if size_mb > 500:
                print(f"Large video file detected: {size_mb:.1f} MB")
            gc.collect()
            from make_mmap import make_memmap
            make_memmap(self.video_path, self.memmap_path)
            print(f"Memmap saved to {self.memmap_path}")
            gc.collect()
        except MemoryError as e:
            self._handle_error(e, "Memmap (MemoryError)")
            print("Try processing a smaller video file or restart the app")
        except Exception as e:
            self._handle_error(e, "Memmap")

    def _load_roi_file(self):
        """Open a file picker for an existing ROI NPZ (from Offline Setup or a
        prior discovery run). Sets self.rois_path and optionally starts live
        traces immediately. Does NOT run segmentation — the user already did
        that."""
        try:
            import shutil
            from pathlib import Path
            # Default search dir: Offline Setup writes rois.npz to data/ next
            # to STIMViewer_CRISPI by convention, but fall back to CWD.
            here = Path(__file__).resolve().parent
            candidates = [
                here.parent / "data",
                here / "data",
                Path.cwd(),
            ]
            default_dir = next((str(p) for p in candidates if p.exists()), str(here))
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load ROI file (NPZ)",
                default_dir,
                "ROI archives (*.npz);;All files (*)",
            )
            if not path:
                return
            # Sanity-check: must be an NPZ with a 'labels' key.
            try:
                with np.load(path, allow_pickle=True) as z:
                    keys = set(z.files)
                    if "labels" not in keys:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Load ROI file",
                            f"{path} has no 'labels' array.\nKeys present: {sorted(keys)}",
                        )
                        return
                    labels = z["labels"]
                    n_rois = int(labels.max()) if labels.size else 0
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Load ROI file", f"Could not read {path}:\n{e}")
                return

            # Copy into self.rois_path so the rest of the dialog (which reads
            # from a fixed filename) picks it up without code changes. The
            # previous code hardcoded rois_path="rois.npz" in CWD.
            try:
                if os.path.abspath(path) != os.path.abspath(self.rois_path):
                    shutil.copyfile(path, self.rois_path)
                print(f"✅ Loaded ROI file: {path} ({n_rois} ROIs) → {self.rois_path}")
            except Exception as e:
                print(f"⚠️ copyfile failed, will read directly: {e}")
                self.rois_path = path

            # Prompt to start live traces. Don't auto-start — the user may
            # want to load camera acquisition first, or inspect the file.
            reply = QtWidgets.QMessageBox.question(
                self,
                "Start live traces?",
                f"Loaded {n_rois} ROIs.\nStart live trace extraction now?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                try:
                    self.start_live_traces()
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self, "start_live_traces", f"Failed to start: {e}")
        except Exception as e:
            print(f"[UI] Load ROI file error: {e}")

    def _run_discover_rois(self, method="OTSU"):
        if method in ("CNMF", "Custom"):
            QtWidgets.QMessageBox.information(
                self,
                f"{method} Segmentation",
                f"{method} segmentation is not yet implemented — coming soon.",
            )
            return
        self._discover_method = method
        threading.Thread(target=self._thread_discover_rois, daemon=True).start()

    def _thread_discover_rois(self):
        print("Discovering ROIs…")

        self.requestStopLiveTraces.emit()


        try:
            save_npz_components = None
            if self._discover_method == "OTSU":
                movie = np.load(self.memmap_path, mmap_mode="r")
                from otsu_thresh import compute_mean_projection, denoise_and_threshold_gpu

                mean = compute_mean_projection(movie, calib_frames=5400, chunk_size=200)
                mean = cv2.resize(mean, (1936, 1096), interpolation=cv2.INTER_NEAREST)
                masks, sizes = denoise_and_threshold_gpu(
                    mean, gauss_ksize=(3, 3), gauss_sigma=1.5, min_area=60, max_area=300
                )
                if not masks:
                    print("ROI discovery produced no masks; aborting live traces/recording.")
                    return

                labeled = np.zeros_like(masks[0], dtype=np.int32)
                labeled = labeled.astype(np.int32, copy=False)

                for i, m in enumerate(masks, start=1):
                    labeled[m] = i

                save_npz_components = (np.asarray(masks, dtype=np.uint8), np.asarray(sizes, dtype=np.int32), labeled)

            elif self._discover_method == "Cellpose":
                if not self.video_path or not os.path.exists(self.video_path):
                    print("No valid video file selected")
                    return

                runner = os.path.join(os.path.dirname(__file__), "cellpose_runner.py")
                if not os.path.exists(runner):
                    raise FileNotFoundError(f"cellpose_runner.py not found at {runner}")

                # Prefer user's dedicated Cellpose venv if present
                venv_python = os.path.expanduser("~/cellpose_env/bin/python")
                python_exe = venv_python if os.path.exists(venv_python) else sys.executable

                # Optional custom model paths from the user's Cellpose repo
                cp_base = os.path.expanduser("~/U-Net_GPU_Analysis")
                model_path = os.path.join(cp_base, "cytotorch_0")
                size_path = os.path.join(cp_base, "size_cytotorch_0.npy")

                cmd = [python_exe, runner, "--video", self.video_path, "--out", self.rois_path]
                if os.path.exists(model_path):
                    cmd += ["--model", model_path]
                if os.path.exists(size_path):
                    cmd += ["--size", size_path]

                print(f"Running Cellpose via: {' '.join(cmd)}")
                try:
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    print(res.stdout)
                    if res.returncode != 0:
                        raise RuntimeError(f"Cellpose runner failed with code {res.returncode}")
                except Exception as e:
                    print(f"Cellpose execution failed: {e}")
                    raise

                try:
                    roi_data = np.load(self.rois_path)
                    if 'labels' in roi_data:
                        labeled = roi_data['labels'].astype(np.int32)
                    else:
                        labeled = np.load(self.rois_path)["labels"].astype(np.int32)
                except Exception:
                    labeled = np.load(self.rois_path)["labels"].astype(np.int32)

                # Build masks/sizes for consistency with OTSU path
                max_id = int(labeled.max(initial=0))
                masks = [(labeled == i) for i in range(1, max_id + 1)]
                sizes = [int(m.sum()) for m in masks]
                save_npz_components = (np.asarray(masks, dtype=np.uint8), np.asarray(sizes, dtype=np.int32), labeled)

            else:
                raise ValueError(f"Unknown ROI method: {self._discover_method}")


            try:
                from projection import ProjectDisplay
                from PyQt5.QtGui import QGuiApplication

                # Build binary union mask and display as grayscale (0/255)
                binary = (labeled > 0).astype(np.uint8)
                img_gray = (binary * 255).astype(np.uint8)

                screens = QGuiApplication.screens()
                scr = screens[1] if len(screens) > 1 else screens[0]
                size = scr.size()
                tgt_w, tgt_h = size.width(), size.height()
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
                        img_gray = np.pad(
                            img_gray,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0
                        )
                else:
                    img_gray = cv2.resize(img_gray, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)

                # Save the actually displayed (padded/resized) discovery mask.
                # Try primary path under CellposeRepo/cellpose_outputs, and fall back to rois dir and CWD.
                try:
                    from pathlib import Path
                    # Prefer tifffile; fall back to PIL or OpenCV if unavailable
                    def _save_tiff(img_arr, path_str):
                        try:
                            import tifffile as _tif
                            _tif.imwrite(path_str, img_arr.astype(np.uint8))
                            return True
                        except Exception:
                            try:
                                from PIL import Image as _PIL_Image
                                _PIL_Image.fromarray(img_arr.astype(np.uint8)).save(path_str, format="TIFF")
                                return True
                            except Exception:
                                try:
                                    import cv2 as _cv2
                                    # OpenCV supports TIFF on most builds; write as 8-bit
                                    return bool(_cv2.imwrite(path_str, img_arr.astype(np.uint8)))
                                except Exception:
                                    return False

                    repo_root = Path(__file__).resolve().parent.parent.parent
                    save_dir = (repo_root / "CellposeRepo" / "cellpose_outputs")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    primary_path = str((save_dir / "discover_mask_presented.tiff").resolve())
                    saved = _save_tiff(img_gray, primary_path)
                    if not saved:
                        # Fallback to the directory containing rois.npz (if resolvable)
                        try:
                            rois_dir = Path(self.rois_path).resolve().parent
                        except Exception:
                            rois_dir = Path.cwd()
                        fallback1 = str((rois_dir / "discover_mask_presented.tiff").resolve())
                        saved = _save_tiff(img_gray, fallback1)
                        if saved:
                            print(f"💾 Saved discovery presented mask to: {fallback1}")
                        else:
                            # Final fallback: current working directory
                            fallback2 = str((Path.cwd() / "discover_mask_presented.tiff").resolve())
                            if _save_tiff(img_gray, fallback2):
                                print(f"💾 Saved discovery presented mask to: {fallback2}")
                            else:
                                raise RuntimeError("All save methods failed (tifffile/PIL/OpenCV)")
                    else:
                        print(f"💾 Saved discovery presented mask to: {primary_path}")
                except Exception as _e:
                    print(f"⚠️ Failed to save discovery presented mask: {_e}")

                if self.proj_display:
                    try:
                        self.proj_display.close()
                    except Exception:
                        pass
                self.proj_display = ProjectDisplay(scr)

                H = getattr(self.camera, "translation_matrix", None)
                self.proj_display.show_image_fullscreen_on_second_monitor(img_gray, H)
                print("✅ Mask projection displayed")
            except Exception as e:
                print(f"Failed to project mask: {e}")


            if save_npz_components is not None:
                masks, sizes, labeled = save_npz_components
            binary = (labeled > 0).astype(np.uint8)
            np.savez_compressed(self.rois_path, masks=masks, sizes=sizes, labels=labeled, binary=binary)
            print(f"ROIs written to {self.rois_path}")


            self.requestStartLiveTraces.emit()
            print("Requested (queued) start of recording and live traces.")

        except Exception as e:
            print(f"ROI discovery failed: {e}")
            self._handle_error(e, "ROI discovery")

    def _run_refine_rois(self):
        threading.Thread(target=self._thread_refine_rois, daemon=True).start()

    def _thread_refine_rois(self):


        self.requestStopLiveTraces.emit()
        print("Manual Mask Generation…")
        try:
            from otsu_thresh import compute_mean_projection, load_movie
            mean = compute_mean_projection(load_movie(self.video_path), calib_frames=5400)
            mean = cv2.resize(mean, (1936, 1096), interpolation=cv2.INTER_NEAREST)
            masks = np.load(self.rois_path)["masks"]
            self.refineRequested.emit(mean, masks)
        except Exception as e:
            self._handle_error(e, "ROI refinement")

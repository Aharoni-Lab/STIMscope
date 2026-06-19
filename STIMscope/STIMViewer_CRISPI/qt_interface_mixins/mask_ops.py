"""MaskOpsMixin — extracted from qt_interface.py per L5 §0.5 decomposition.

Cluster 6 (mask-pattern operations + projector binary build).
5 methods, ~225 LOC.

Methods:
- ``_maybe_build_projector(proj_dir)``      — Build the C++ projector binary
  if missing or older than main.cpp. Idempotent; returns True on success.
- ``_helper_python_path_for_masks()``       — Resolve the Python interpreter
  to use for spawning zmq_mask_sender.py (venv → conda → sys.executable).
- ``_on_mask_pattern_changed(text)``        — Enable/disable the Browse
  button depending on which mask pattern is selected.
- ``_browse_mask_pattern_path()``           — File/folder dialog for Image,
  Folder, and Custom mask patterns; writes _mask_pattern_path.
- ``_toggle_send_masks()``                  — Start/stop the mask-sender
  QProcess. Builds the argv vector from the dropdown selection (Moving Bar,
  Checkerboard, Solid, Circle, Gradient, Image, Folder, Seg Mask, Custom),
  applies flip flags, applies stim-mode flags, and launches the subprocess.

Mixin contract — subclass provides:
    self._proc_masks                  : QProcess | None
    self._button_send_masks           : QPushButton
    self._mask_pattern_browse         : QPushButton
    self._mask_pattern_dropdown       : QComboBox
    self._mask_pattern_path           : str
    self._mask_flip_h, self._mask_flip_v : bool
    self._stim_mode_dropdown          : QComboBox (optional)
    self._proj_warp_mode              : str (optional, defaults "H")
    self._camera                      : OptimizedCamera-like
    self._ensure_qprocess()           : Interface helper returning QProcess
    self._attach_proc_signals(proc, tag) : Interface helper
    self._on_proc_finished(which)     : LEDAndProcessMixin slot

Pure hoist — no behavior change vs. monolith.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class MaskOpsMixin:
    """Cluster 6 — mask-pattern operations + projector binary build."""

    def _maybe_build_projector(self, proj_dir: str) -> bool:
        try:
            import subprocess
            exe = f"{proj_dir}/projector"
            src = f"{proj_dir}/main.cpp"
            need_build = (not os.path.exists(exe))
            if not need_build:
                try:
                    need_build = os.path.getmtime(exe) < os.path.getmtime(src)
                except Exception:
                    need_build = False
            if not need_build:
                return True
            print(f"[PROJ] Building projector in {proj_dir}...")
            cmd = [
                "g++", "-O2", "-std=c++17", "main.cpp", "-o", "projector",
                # Link order matters: GLEW before GL on Linux
                "-lglfw", "-lGLEW", "-lGL", "-lzmq", "-lgpiod", "-lpthread"
            ]
            res = subprocess.run(cmd, cwd=proj_dir, capture_output=True, text=True)
            if res.returncode != 0:
                print("[PROJ] Build failed:\n" + (res.stderr or res.stdout))
                return False
            print("[PROJ] Build succeeded")
            return True
        except Exception as e:
            print(f"[PROJ] Build error: {e}")
            return False

    def _helper_python_path_for_masks(self) -> str:
        # Prefer local venv (contains pyzmq), then active conda, then current python
        try:
            venv_py = (Path(__file__).resolve().parents[2] / "my_UARTvenv" / "bin" / "python").resolve()
            if venv_py.exists():
                return str(venv_py)
        except Exception:
            pass
        try:
            conda_pref = os.environ.get("CONDA_PREFIX")
            if conda_pref:
                cand = Path(conda_pref) / "bin" / "python"
                if cand.exists():
                    return str(cand)
        except Exception:
            pass
        return sys.executable or "/usr/bin/python3"

    def _on_mask_pattern_changed(self, text: str):
        # Enable browse button only for patterns that need a path
        need_path = text in ("Image", "Folder", "Custom", "Seg Mask")
        try:
            self._mask_pattern_browse.setEnabled(need_path)
        except Exception:
            pass

    def _browse_mask_pattern_path(self):
        try:
            from PyQt5.QtWidgets import QFileDialog
            # Start the browser at the operator's mounted save dir so recordings,
            # masks, and other persistent artifacts are surfaced. Falls back to
            # the home dir only when no save dir is configured. The launcher
            # sets STIM_SAVE_DIR to a host-mounted path so files survive
            # container restarts (--rm cleanup would otherwise wipe them).
            default_dir = os.environ.get("STIM_SAVE_DIR") or str(Path.home())
            try:
                os.makedirs(default_dir, exist_ok=True)
            except Exception:
                pass
            typ = self._mask_pattern_dropdown.currentText()
            if typ == "Image":
                fp, _ = QFileDialog.getOpenFileName(self, "Select Image", default_dir,
                                                    "Images (*.png *.jpg *.jpeg *.bmp)")
                if fp:
                    self._mask_pattern_path = fp
            elif typ == "Folder":
                dirp = QFileDialog.getExistingDirectory(self, "Select Folder", default_dir)
                if dirp:
                    self._mask_pattern_path = dirp
            elif typ == "Seg Mask":
                # Browse for a saved ROI/segmentation .npz to project. Defaults
                # to STIM_SAVE_DIR (/data) where Offline Setup saves rois.npz.
                fp, _ = QFileDialog.getOpenFileName(self, "Select ROI / segmentation NPZ", default_dir,
                                                    "ROI archives (*.npz);;All files (*)")
                if fp:
                    self._mask_pattern_path = fp
            elif typ == "Custom":
                # Allow selecting either a Python sender or a compiled custom sender (including no extension)
                fp, _ = QFileDialog.getOpenFileName(self, "Select Sender (Python or Executable)", default_dir,
                                                    "All Files (*)")
                if fp:
                    self._mask_pattern_path = fp
        except Exception as e:
            print(f"Browse failed: {e}")

    def _toggle_send_masks(self):
        QProcess = self._ensure_qprocess()
        try:
            # Guard against double-launch: check if process is alive
            if self._proc_masks is not None:
                try:
                    state = self._proc_masks.state()
                    if state != QProcess.NotRunning:
                        self._proc_masks.kill()
                        return
                except Exception:
                    pass
                try:
                    self._proc_masks.deleteLater()
                except Exception:
                    pass
                self._proc_masks = None

            if self._proc_masks is None:
                self._proc_masks = QProcess(self)
                self._proc_masks.finished.connect(lambda *_: self._on_proc_finished('masks'))
                self._proc_masks.errorOccurred.connect(lambda *_: self._on_proc_finished('masks'))
                self._attach_proc_signals(self._proc_masks, 'masks')
                self._button_send_masks.setText("Stop Sending Masks")

                work_dir = str(Path(__file__).resolve().parents[2])
                self._proc_masks.setWorkingDirectory(work_dir)
                py = self._helper_python_path_for_masks()
                # Resolve sender script according to dropdown
                script_path = str(Path(__file__).resolve().parent.parent.parent / "ZMQ_sender_mask" / "zmq_mask_sender.py")
                args = []
                pat = self._mask_pattern_dropdown.currentText()
                if pat == "Moving Bar":
                    args = []  # defaults
                elif pat == "Checkerboard":
                    args = ["--pattern", "checkerboard"]
                elif pat == "Solid":
                    args = ["--pattern", "solid"]
                elif pat == "Circle":
                    args = ["--pattern", "circle"]
                elif pat == "Gradient":
                    # Use sane defaults for visibility (60 Hz, 6 steps, 20-frame holds, gamma 2.2)
                    args = [
                        "--pattern", "gradient",
                        "--fps", "60",
                        "--gradient-steps", "3",
                        "--gradient-hold", "30",
                        "--gradient-gamma", "2.2"
                    ]
                elif pat == "Image":
                    args = ["--pattern", "image", "--image", self._mask_pattern_path]
                elif pat == "Folder":
                    args = ["--pattern", "folder", "--folder", self._mask_pattern_path]
                elif pat == "Seg Mask":
                    # Send latest segmentation labels/masks from rois.npz
                    try:
                        # Search multiple locations for rois.npz. STIM_SAVE_DIR
                        # (/data) comes FIRST because that's where Offline Setup
                        # and ROI discovery now save it; the legacy locations
                        # remain as fallbacks for older saves.
                        _save_dir = os.environ.get("STIM_SAVE_DIR")
                        _roi_candidates = []
                        if _save_dir:
                            _roi_candidates.append(Path(_save_dir) / "rois.npz")
                        _roi_candidates += [
                            Path("/data") / "rois.npz",
                            Path.cwd() / "rois.npz",
                            Path(__file__).resolve().parent / "CS" / "data" / "rois.npz",
                            Path.cwd() / "data" / "rois.npz",
                            Path(__file__).resolve().parent / "rois.npz",
                        ]
                        # Prefer an explicitly browsed .npz (Browse button with
                        # the "Seg Mask" pattern selected) over the auto-search.
                        roi_path = None
                        _picked = getattr(self, "_mask_pattern_path", None)
                        if _picked and str(_picked).lower().endswith(".npz") and Path(_picked).exists():
                            roi_path = str(Path(_picked).resolve())
                        if roi_path is None:
                            for _rp in _roi_candidates:
                                if _rp.exists():
                                    roi_path = str(_rp.resolve())
                                    break
                        if roi_path is None:
                            roi_path = str(_roi_candidates[0].resolve())
                            print("[MASK] WARNING: rois.npz not found in any known location")
                        # Save the actually presented segmask (post flips/prewarp) to CellposeRepo/cellpose_outputs
                        try:
                            repo_root = Path(__file__).resolve().parent.parent.parent
                            save_dir = (repo_root / "CellposeRepo" / "cellpose_outputs")
                            save_dir.mkdir(parents=True, exist_ok=True)
                            save_tiff = str((save_dir / "segmask_presented.tiff").resolve())
                        except Exception:
                            save_tiff = str((Path.cwd() / "segmask_presented.tiff").resolve())
                        args = ["--pattern", "segmask", "--roi-npz", roi_path, "--save-segmask-to", save_tiff]
                    except Exception:
                        args = ["--pattern", "segmask", "--roi-npz", "rois.npz"]
                elif pat == "Custom":
                    script_path = self._mask_pattern_path or script_path
                    args = []
                    # If file endswith.py, run with Python; else treat as executable
                    try:
                        if script_path.lower().endswith('.py'):
                            cmd_prog = py
                            cmd_args = [script_path] + args
                            print(f"[MASK] Launch (python): {cmd_prog} {' '.join(cmd_args)}")
                            self._proc_masks.start(cmd_prog, cmd_args)
                        else:
                            from PyQt5.QtCore import QFileInfo
                            fi = QFileInfo(script_path)
                            cmd_prog = fi.absoluteFilePath()
                            print(f"[MASK] Launch (exec): {cmd_prog} {' '.join(args)}")
                            self._proc_masks.start(cmd_prog, args)
                        return
                    except Exception as e:
                        print(f"Custom sender launch failed: {e}")

                # If LUT mode is active, pass prewarp dir
                try:
                    if getattr(self, '_proj_warp_mode', 'H') == 'LUT':
                        asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent / "Assets" / "Generated").resolve()))
                        args += ["--prewarp-lut-dir", asset_dir]
                        # Ensure engine H is cleared
                        try:
                            import zmq as _zmq
                            _ctx = _zmq.Context.instance(); _s = _ctx.socket(_zmq.REQ)
                            _s.setsockopt(_zmq.LINGER, 0)
                            _s.connect("tcp://127.0.0.1:5560"); _s.send(b"IDENTITY"); _ = _s.recv(); _s.close()
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    from PyQt5.QtCore import QProcessEnvironment
                    env = QProcessEnvironment.systemEnvironment()
                    env.insert("PYTHONUNBUFFERED", "1")
                    self._proc_masks.setProcessEnvironment(env)
                except Exception:
                    pass

                # Projection-mask flips (independent of camera flips). Applied
                # inside zmq_mask_sender.py via --flip-x / --flip-y. Mask flip
                # state lives on self._mask_flip_h/v (persisted in
                # camera_orientation.json). Re-click Send Masks after toggling
                # for changes to take effect.
                if getattr(self, "_mask_flip_h", False):
                    args.append("--flip-x")
                if getattr(self, "_mask_flip_v", False):
                    args.append("--flip-y")

                stim_sel = self._stim_mode_dropdown.currentText() if hasattr(self, "_stim_mode_dropdown") else ""
                if "Temporal" in stim_sel:
                    args.extend(["--temporal-alternate", "--fps", "60"])
                elif "Simultaneous" in stim_sel:
                    args.append("--composite-rgb")

                cmd = [script_path] + args
                print(f"[MASK] Launch: {py} {' '.join(cmd)}")
                self._proc_masks.start(py, cmd)
            else:
                self._proc_masks.kill()
        except Exception as e:
            print(f"Failed to toggle masks: {e}")
            self._on_proc_finished('masks')

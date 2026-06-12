
import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Optional

import numpy as np
import cv2

from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer





def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def _get_env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default

TARGET_PIXEL_FORMAT = {
    "MONO8": ids_peak_ipl.PixelFormatName_Mono8,
    "BGRA8": ids_peak_ipl.PixelFormatName_BGRa8,
    "BGR8":  ids_peak_ipl.PixelFormatName_BGR8,
    "RGBA8": ids_peak_ipl.PixelFormatName_RGBa8,
    "RGB8":  ids_peak_ipl.PixelFormatName_RGB8,
}.get(_get_env_str("STIM_PIXEL_FORMAT", "MONO8").upper(), ids_peak_ipl.PixelFormatName_Mono8)

DEFAULT_FPS       = _get_env_int("STIM_CAMERA_FPS", 60)
MAX_GUI_FPS       = _get_env_int("STIM_MAX_GUI_FPS", 30)  # hard cap on FPS exposed to GUI/recording paths
DEFAULT_BUFFERS   = max(4, _get_env_int("STIM_PEAK_BUFFERS", 16))
DEFAULT_TRIG_LINE = _get_env_str("STIM_TRIGGER_LINE", "Line0")
DEFAULT_RT_START  = _get_env_int("STIM_RT_DEFAULT", 1) == 1 

ASSETS_DIR  = _get_env_str("STIM_ASSETS_DIR", None)
CRISPI_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_FALLBACK = os.path.join(CRISPI_ROOT, "Assets")

def _assets_path(*parts) -> str:
    base = ASSETS_DIR if ASSETS_DIR else ASSETS_FALLBACK
    return os.path.join(base, *parts)




class OptimizedCamera(QObject):
 
    frame_ready = pyqtSignal(object)
    recordingStarted = pyqtSignal()
    recordingStopped = pyqtSignal()
    performance_metrics = pyqtSignal(dict)
    autoStartRecording = pyqtSignal()  # Signal to auto-start recording from acquisition thread
    # Emitted on the worker thread when calibration finishes successfully —
    # GUI connects this to a slot that pokes the camera (re-emit cached frame)
    # so the live preview reflects the new calibration without needing the user
    # to touch a slider/button to trigger a refresh.
    calibrationFinished = pyqtSignal()
    

    def __init__(self, device_manager, interface):
        super().__init__()
        if interface is None:
            raise ValueError("Interface is None")


        self._interface = interface
        # frame_ready → on_image_received is connected in start_window()
        # with QueuedConnection for proper cross-thread Qt safety.


        self.device_manager = device_manager
        self._device = None
        self._datastream = None
        self.node_map = None

        self._last_acq_err_ts = 0.0
        self._acq_err_interval = 1.0

        self._snapshot_path: Optional[str] = None



        self._state_lock = threading.Lock()
        self.acquisition_mode = 0  # 0: RT, 1: HW
        self.acquisition_running = False
        self._acq_thread: Optional[threading.Thread] = None
        self.acquisition_thread = None   # legacy alias
        self._acq_stop = threading.Event()


        self._buffer_list = []
        self._image_converter = ids_peak_ipl.ImageConverter()


        self.killed = False
        self.is_recording = False
        self.is_armed = False  # New state for hardware trigger armed mode
        self._auto_start_pending = False  # HW-1: one-shot gate for autoStartRecording signal
        self.save_image = False
        self.hardware_trigger_line = DEFAULT_TRIG_LINE


        self.target_gain = 1.0
        self.max_gain = 1.0
        self.target_dgain = 1.0


        self.frame_times = deque(maxlen=120)
        self.GUIfps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.performance_stats = {
            "fps": 0.0,
            "frame_processing_time": 0.0,
            "memory_usage": 0.0,
            "thread_pool_usage": 0.0,
        }


        self.translation_matrix = np.eye(3, dtype=np.float64)
        self.calibration_running = False
        self.calibration_lock = threading.Lock()

        self._dest_pf = None


        self.asset_dir = _assets_path("Generated")
        self.save_dir = _get_env_str("STIM_SAVE_DIR",
                                     os.path.join(CRISPI_ROOT, "Saved_Media"))
        os.makedirs(self.asset_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)


        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="CameraWorker")
        # recording_queue: buffer between camera acquisition thread (producer)
        # and VideoRecorder's writer thread (consumer). Size=60 matches upstream
        # Aharoni-Lab/STIMscope. Our prior value of 24 was insufficient
        # for sustained 30 Hz recording when the TIFF writer fell behind (user
        # earlier observation showed long recordings dropping silent frames).
        # Gives ~2 s of burst buffer at 30 fps.
        self.recording_queue: queue.Queue = queue.Queue(maxsize=60)
        # Silent-drop counter for frames that couldn't enqueue to
        # recording_queue because the writer thread fell behind.
        # Previously invisible → user saw 21 fps with VideoRecorder
        # reporting dropped=0. Exposed to VideoRecorder for finalize.
        self._recording_queue_drops: int = 0
        self.save_queue: queue.Queue = queue.Queue(maxsize=24)
        self.pipeline_queue: queue.Queue = queue.Queue(maxsize=24)
        self._pipeline_active = False  # Only populate queue when pipeline is running
        self.recording_worker_running = False
        self.save_worker_running = False


        self._open_device()
        self._apply_defaults()
        self._init_data_stream()
        self._interface.set_camera(self)


        from video_recorder import VideoRecorder

        self.video_recorder = VideoRecorder(interface)


        self._start_background_workers()



    def start(self, start_rt: bool = DEFAULT_RT_START):
       
        if start_rt:
            self.start_realtime_acquisition()
        self._start_acquisition_thread()

    def _pick_dest_pf(self, ipl_src):
        try:
            outs = self._image_converter.SupportedOutputPixelFormatNames(ipl_src.PixelFormat())

            pref = [
                ids_peak_ipl.PixelFormatName_BGRa8,
                ids_peak_ipl.PixelFormatName_BGR8,
                ids_peak_ipl.PixelFormatName_RGBa8,
                ids_peak_ipl.PixelFormatName_RGB8,
            ]
            for p in pref:
                if p in outs:
                    return p
            return outs[0] if outs else TARGET_PIXEL_FORMAT
        except Exception:
            return TARGET_PIXEL_FORMAT


    def _pause_stream_for_change(self):
       
        was_running   = bool(self.acquisition_running)
        was_recording = bool(self.is_recording)
        prev_mode     = self.acquisition_mode  # 0: RT, 1: HW


        critical_change = False
        

        r = getattr(self, "video_recorder", None)
        if was_recording and r is not None and critical_change:
            try: 
                r.stop_recording()
                print("⏸️ Recording paused for critical parameter change")
            except Exception: 
                pass


        if was_running and critical_change:
            try:
                if prev_mode == 0:
                    self.stop_realtime_acquisition()
                else:
                    self.stop_hardware_acquisition()
                print("⏸️ Acquisition paused for critical parameter change")
            except Exception:
                pass
        elif was_running:

            try:

                if self._datastream:
                    self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                time.sleep(0.001) 
            except Exception:
                pass

        return was_running, was_recording, prev_mode

    def _resume_stream_after_change(self, was_running, was_recording, prev_mode):
        try:
            if was_running:
                if prev_mode == 0:
                    self.start_realtime_acquisition()
                else:
                    self.start_hardware_acquisition()
                    
            if was_recording and getattr(self, "video_recorder", None):
                self.start_recording() 
        except Exception:
            pass


    def _rebuild_converter_and_buffers(self):
        try:

            try:
                self._payload_size = int(self.node_map.FindNode("PayloadSize").Value())
            except Exception:
                self._payload_size = None


            self.revoke_and_allocate_buffer()


            self.frame_times.clear()
            self.frame_count = 0
            self.start_time = time.time()
            self._dest_pf = None
        except Exception as e:
            print(f"Failed to rebuild buffers after setting: {e}")



    def change_pixel_format(self, symbolic: str) -> bool:
        was_running, was_recording, prev_mode = self._pause_stream_for_change()
        ok = False
        err = None
        try:
            node = self.node_map.FindNode("PixelFormat")

            setter = getattr(node, "FromString", None)
            if callable(setter):
                setter(symbolic)
            else:
                entries = node.Entries()
                chosen = None
                for e in entries:
                    if e.AccessStatus() in (
                        ids_peak.NodeAccessStatus_NotAvailable,
                        ids_peak.NodeAccessStatus_NotImplemented
                    ):
                        continue
                    if e.SymbolicValue() == symbolic:
                        chosen = e
                        break
                if not chosen:
                    raise RuntimeError(f"PixelFormat '{symbolic}' not available")
                node.SetCurrentEntry(chosen)
            ok = True
        except Exception as e:
            err = e
            ok = False
        finally:
            self._rebuild_converter_and_buffers()
            self._resume_stream_after_change(was_running, was_recording, prev_mode)
        if ok:
            print(f"✅ PixelFormat set to {symbolic} — converter rebuilt, stream resumed")
        else:
            print(f"❌ PixelFormat change to {symbolic} failed: {err}")
        return ok


    def set_fps(self, fps: int) -> bool:

        try:
            was_running, was_recording, prev_mode = self._pause_stream_for_change()

            node = self.node_map.FindNode("AcquisitionFrameRate")
            if node is None:
                print("AcquisitionFrameRate node not found")
                return False

            try:
                mn, mx = node.Minimum(), node.Maximum()
                fps = max(mn, min(mx, fps))
            except Exception:
                pass

            node.SetValue(float(fps))
            print(f"Camera frame rate set to {fps} FPS")

            self._resume_stream_after_change(was_running, was_recording, prev_mode)
            return True

        except Exception as e:
            print(f"FPS setting error: {e}")
            return False

    def set_gain(self, value: float) -> bool:
        """
        Optimized gain setter that minimizes FPS impact.
        Gain changes usually don't require stopping acquisition.
        """
        try:
            node = self.node_map.FindNode("Gain")
            if node is None:
                print("❌ Gain node not found")
                return False
            

            try:
                access_status = node.AccessStatus()
                if access_status not in (ids_peak.NodeAccessStatus_ReadWrite,):
                    print("⚠️ Gain node not writable during acquisition")

                    return self._set_gain_with_pause(value)
            except Exception:
                pass
            

            try:
                mn, mx = node.Minimum(), node.Maximum()
                value = max(mn, min(mx, value))
            except Exception:
                pass
            

            self.target_gain = value
            

            try:
                node.SetValue(float(value))
                print(f"✅ Gain set to {value:.2f} (live change)")
                return True
            except Exception as e:
                print(f"⚠️ Live gain change failed: {e}, using safe method")
                return self._set_gain_with_pause(value)
                
        except Exception as e:
            print(f"❌ Gain setting error: {e}")
            return False

    def _set_gain_with_pause(self, value: float) -> bool:
       
        was_running, was_recording, prev_mode = self._pause_stream_for_change()
        ok = False
        try:
            node = self.node_map.FindNode("Gain")
            node.SetValue(float(value))
            self.target_gain = value
            ok = True
            print(f"✅ Gain set to {value:.2f} (with pause)")
        except Exception as e:
            print(f"❌ Cannot set gain: {e}")
            ok = False
        finally:

            if not ok:
                self._rebuild_converter_and_buffers()
            self._resume_stream_after_change(was_running, was_recording, prev_mode)
        return ok

    def set_dgain(self, value: float) -> bool:
        """
        Set digital gain with FPS preservation.
        
        Args:
            value: Digital gain value
            
        Returns:
            True if successful, False otherwise
        """
        try:

            node = self.node_map.FindNode("DigitalGain")
            if node is None:
                print("❌ DigitalGain node not found")
                return False
            

            try:
                access_status = node.AccessStatus()
                if access_status in (ids_peak.NodeAccessStatus_ReadWrite,):

                    try:
                        mn, mx = node.Minimum(), node.Maximum()
                        value = max(mn, min(mx, value))
                    except Exception:
                        pass
                    
                    node.SetValue(float(value))
                    self.target_dgain = value
                    print(f"✅ Digital gain set to {value:.2f} (live change)")
                    return True
            except Exception:
                pass
            

            return self._set_dgain_with_pause(value)
            
        except Exception as e:
            print(f"❌ Digital gain setting error: {e}")
            return False

    def _set_dgain_with_pause(self, value: float) -> bool:
       
        was_running, was_recording, prev_mode = self._pause_stream_for_change()
        ok = False
        try:
            node = self.node_map.FindNode("DigitalGain")
            node.SetValue(float(value))
            self.target_dgain = value
            ok = True
            print(f"✅ Digital gain set to {value:.2f} (with pause)")
        except Exception as e:
            print(f"❌ Cannot set digital gain: {e}")
            ok = False
        finally:
            self._resume_stream_after_change(was_running, was_recording, prev_mode)
        return ok


    def snapshot(self, path: str) -> bool:
    
        try:

            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            
            if self.acquisition_running:

                img = self._get_latest_frame_for_snapshot()
                if img is not None:
                    try:
                        ids_peak_ipl.ImageWriter.WriteAsPNG(path, img)
                        print(f"✅ Snapshot saved: {path}")
                        return True
                    except Exception as e:
                        print(f"❌ Snapshot save failed: {e}")
                        return False
                else:
                    print("❌ No frame available for snapshot")
                    return False


            print("Starting temporary acquisition for snapshot...")
            started = self.start_realtime_acquisition()
            if not started:
                print("❌ Snapshot failed: could not start acquisition")
                return False
            
            try:

                time.sleep(0.001)
                

                t0 = time.time()
                while time.time() - t0 < 2.0:
                    img = self.get_data_stream_image()
                    if img is not None:
                        try:
                            ids_peak_ipl.ImageWriter.WriteAsPNG(path, img)
                            print(f"✅ Snapshot saved: {path}")
                            return True
                        except Exception as e:
                            print(f"❌ Snapshot save failed: {e}")
                            return False
                    time.sleep(0.001)  
                
                print("❌ Snapshot failed: no frame captured within timeout")
                return False
                
            finally:

                self.stop_realtime_acquisition()
                print("Temporary acquisition stopped")
                
        except Exception as e:
            print(f"❌ Snapshot error: {e}")
            return False

    def _get_latest_frame_for_snapshot(self):
    
        try:

            for _ in range(3):
                try:
                    self._datastream.KillWait()
                except Exception:
                    pass
            

            for attempt in range(5):
                img = self.get_data_stream_image()
                if img is not None:
                    return img
                time.sleep(0.001) 
            
            return None
            
        except Exception as e:
            print(f"Error getting latest frame: {e}")
            return None


    def shutdown(self):
        """Idempotent shutdown — safe to call from any state.

        D-cam-28fix: None-guard every attribute
        access. Pre-fix, calling `close()` before `__init__` completed
        (e.g., during cleanup of a failed device-open) raised
        `AttributeError: 'NoneType' object has no attribute 'set'`
        because `self._acq_stop` was None. Now every access is guarded
        so partial-init state degrades gracefully to a no-op shutdown.
        """
        self.killed = True

        # D-cam-28: guard against partial-init where _acq_stop is None
        if getattr(self, '_acq_stop', None) is not None:
            try:
                self._acq_stop.set()
            except Exception:
                pass

        try:
            self.stop_recording()
        except Exception:
            pass

        try:
            self.stop_realtime_acquisition()
        except Exception:
            pass
        try:
            self.stop_hardware_acquisition()
        except Exception:
            pass

        # D-cam-28: also guard the background worker stop
        try:
            self._stop_background_workers()
        except Exception:
            pass

        # D-cam-28: also guard the device teardown
        try:
            self._teardown_stream_and_device()
        except Exception:
            pass

    def close(self):
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass



    def _open_device(self):
        self.device_manager.Update()
        if self.device_manager.Devices().empty():
            raise RuntimeError("No IDS Peak device found")

        self._device = self.device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        self.node_map = self._device.RemoteDevice().NodeMaps()[0]


        try:
            self.node_map.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
            self.max_gain = self.node_map.FindNode("Gain").Maximum()
        except Exception:
            self.max_gain = 1.0
        try:
            self.node_map.FindNode("UserSetSelector").SetCurrentEntry("Default")
            self.node_map.FindNode("UserSetLoad").Execute()
            self.node_map.FindNode("UserSetLoad").WaitUntilDone()
        except Exception:
            pass

    def _apply_defaults(self):

        self._find_and_set_enum("GainAuto", "Off")
        self._find_and_set_enum("ExposureAuto", "Off")

        # Default operating point: 30 fps + 33333 µs exposure. This is the
        # canonical STIMscope mode (matches the 30 Hz MCU trigger) and gives
        # a stable, non-flickering live preview that operators expect. Either
        # can be overridden via Sensor Settings during the session (e.g. set
        # exposure 15000 µs for safe HW-trigger margin). Tunable via env vars:
        #   STIM_DEFAULT_FPS_HZ   (default 30)
        #   STIM_DEFAULT_EXP_US   (default 33333.33)
        # Order matters in IDS Peak: AcquisitionFrameRate caps the max
        # ExposureTime — set FPS first, then exposure, so the 33 ms exposure
        # fits under the 30 fps period.
        try:
            default_fps = float(os.environ.get("STIM_DEFAULT_FPS_HZ", "30"))
        except Exception:
            default_fps = 30.0
        try:
            default_exp = float(os.environ.get("STIM_DEFAULT_EXP_US", "33333.33"))
        except Exception:
            default_exp = 33333.33
        try:
            fps_node = self.node_map.FindNode("AcquisitionFrameRate")
            mn, mx = fps_node.Minimum(), fps_node.Maximum()
            fps_node.SetValue(max(mn, min(mx, default_fps)))
            print(f"AcquisitionFrameRate set to {default_fps:.1f} FPS (default)")
        except Exception as _e:
            print(f"AcquisitionFrameRate default-set skipped: {_e}")
        try:
            exp_node = self.node_map.FindNode("ExposureTime")
            mn, mx = exp_node.Minimum(), exp_node.Maximum()
            exp_node.SetValue(max(mn, min(mx, default_exp)))
            print(f"ExposureTime set to {default_exp:.2f} µs (default; matches {default_fps:.1f} fps period)")
        except Exception as _e:
            print(f"ExposureTime default-set skipped: {_e}")

    def _init_data_stream(self):
        self._datastream = self._device.DataStreams()[0].OpenDataStream()
        self.revoke_and_allocate_buffer()   

    def _teardown_stream_and_device(self):
        t = self._acq_thread
        self._acq_thread = None
        self.acquisition_thread = None 
        if t and t.is_alive():
            try: t.join(timeout=2.0)
            except Exception: pass


        if self._datastream is not None:
            try:
                for b in list(self._datastream.AnnouncedBuffers()):
                    self._datastream.RevokeBuffer(b)
            except Exception:
                pass
            try:
                self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            except Exception:
                pass
            try:
                self._datastream.Close()
            except Exception:
                pass
            self._datastream = None


        if self._device is not None:
            try:
                self._device.Close()
            except Exception:
                pass
            self._device = None



    def _start_background_workers(self):
        if not self.recording_worker_running:
            self.recording_worker_running = True
            self.thread_pool.submit(self._recording_worker)
        if not self.save_worker_running:
            self.save_worker_running = True
            self.thread_pool.submit(self._save_worker)

    def _stop_background_workers(self):

        try: self.recording_queue.put_nowait(None)
        except Exception: pass
        try: self.save_queue.put_nowait(None)
        except Exception: pass


        try:
            self.thread_pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self.thread_pool.shutdown(wait=True)
        except Exception:
            pass

        self.recording_worker_running = False
        self.save_worker_running = False


    def _recording_worker(self):
        while True:
            item = self.recording_queue.get()
            try:
                if item is None:
                    self.recording_queue.task_done()
                    break
                self.video_recorder.add_frame(item)
            except Exception as e:
                print(f"Recording worker error: {e}")
            finally:
                if item is not None:
                    self.recording_queue.task_done()

    def _save_worker(self):
        while True:
            item = self.save_queue.get()
            try:
                if item is None:
                    self.save_queue.task_done()
                    break
                save_path, ipl_img = item
                ids_peak_ipl.ImageWriter.WriteAsPNG(save_path, ipl_img)
            except Exception as e:
                print(f"Save worker error: {e}")
            finally:
                if item is not None:
                    self.save_queue.task_done()




    def _queue_all_buffers(self):
        for b in self._buffer_list:
            try:
                self._datastream.QueueBuffer(b)
            except Exception:
                pass

    def start_realtime_acquisition(self) -> bool:
        if self._device is None or self.acquisition_running:
            return False
        if self._datastream is None:
            self._init_data_stream()
        self.acquisition_mode = 0
        self._queue_all_buffers()
        try:
            self._select_trigger("Off", None)
            try:
                self.node_map.FindNode("TLParamsLocked").SetValue(1)
            except Exception:
                pass
            self._datastream.StartAcquisition()
            self.node_map.FindNode("AcquisitionStart").Execute()
            self.acquisition_running = True
            return True
        except Exception as e:
            print(f"start_realtime_acquisition failed: {e}")
            return False

    def stop_realtime_acquisition(self):
        if self._device is None or not self.acquisition_running or self.acquisition_mode != 0:
            return
        self._stop_acquisition_stream("RT")

    def start_hardware_acquisition(self) -> bool:
        if self._device is None or self.acquisition_running:
            print("❌ Cannot start acquisition: device missing or already running")
            return False

        if self._datastream is None:
            self._init_data_stream()

        self.acquisition_mode = 1
        self._queue_all_buffers()

        try:
            # Use currently-selected trigger line from GUI/env (falls back to Line0)
            trig_line = getattr(self, "hardware_trigger_line", None) or "Line0"

            # --- 1. Select trigger ---
            self._select_trigger("On", trig_line)  # TriggerMode = On, TriggerSource = <trig_line>

            # --- 2. Lock parameters ---
            try:
                self.node_map.FindNode("TLParamsLocked").SetValue(1)
            except Exception:
                print("⚠️ TLParamsLocked not writable, proceeding anyway")

            # --- 3. Configure selected line for input ---
            line_selector_node = self.node_map.FindNode("LineSelector")
            if line_selector_node and line_selector_node.AccessStatus() == ids_peak.NodeAccessStatus_ReadWrite:
                entry = line_selector_node.FindEntry(trig_line)
                if entry:
                    line_selector_node.SetCurrentEntry(entry)
                else:
                    print(f"⚠️ {trig_line} not found in LineSelector")
            else:
                print(f"⚠️ LineSelector node not writable or missing: {line_selector_node}")

            line_mode_node = self.node_map.FindNode("LineMode")
            if line_mode_node and line_mode_node.AccessStatus() == ids_peak.NodeAccessStatus_ReadWrite:
                entry = line_mode_node.FindEntry("Input")
                if entry:
                    line_mode_node.SetCurrentEntry(entry)
                    print(f"✅ {trig_line} configured as Input for external trigger")
                else:
                    print("⚠️ 'Input' entry not found in LineMode")
            else:
                print(f"⚠️ LineMode node not writable or missing: {line_mode_node}")

            # --- 4. Start datastream and acquisition ---
            self._datastream.StartAcquisition()

            acq_start_node = self.node_map.FindNode("AcquisitionStart")
            if acq_start_node:
                try:
                    acq_start_node.Execute()
                except Exception as e:
                    print(f"⚠️ Failed to execute AcquisitionStart: {e}")

            self.acquisition_running = True
            print(f"📡 Hardware Acquisition started! Waiting for external trigger on {trig_line}")
            return True

        except Exception as e:
            print(f"❌ start_hardware_acquisition failed: {e}")
            return False







    def stop_hardware_acquisition(self):
        if self._device is None or not self.acquisition_running or self.acquisition_mode != 1:
            return
        self._stop_acquisition_stream("HW")

    def _stop_acquisition_stream(self, label: str):
        try: self.node_map.FindNode("AcquisitionStop").Execute()
        except Exception: pass
        try: self._datastream.KillWait()
        except Exception: pass
        try: self._datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        except Exception: pass
        try: self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        except Exception: pass

        self.acquisition_running = False
        try:
            self.node_map.FindNode("TLParamsLocked").SetValue(0)
        except Exception:
            pass
        self.revoke_and_allocate_buffer()
        print(f"Closed {label} Acq")

    def _select_trigger(self, mode: str, source: Optional[str]):

        try:
            entries = self.node_map.FindNode("TriggerSelector").Entries()
            symbols = [e.SymbolicValue() for e in entries
                       if e.AccessStatus() not in (ids_peak.NodeAccessStatus_NotAvailable,
                                                   ids_peak.NodeAccessStatus_NotImplemented)]
            sel = "ExposureStart" if "ExposureStart" in symbols else (symbols[0] if symbols else None)
            if sel:
                self.node_map.FindNode("TriggerSelector").SetCurrentEntry(sel)
        except Exception:
            pass


        try:
            self.node_map.FindNode("TriggerMode").SetCurrentEntry(mode)
        except Exception:
            pass


        if mode == "On" and source:
            try:
                self.node_map.FindNode("TriggerSource").SetCurrentEntry(source)
                self.node_map.FindNode("TriggerActivation").SetCurrentEntry("RisingEdge")
            except Exception:
                pass

    def revoke_and_allocate_buffer(self):
        if self._datastream is None:
            return
        try:
            for b in list(self._datastream.AnnouncedBuffers()):
                self._datastream.RevokeBuffer(b)
        except Exception:
            pass

        try:
            payload_size = int(self.node_map.FindNode("PayloadSize").Value())
        except Exception:
            payload_size = 0

        try:
            min_required = self._datastream.NumBuffersAnnouncedMinRequired()
        except Exception:
            min_required = 4

        nbuf = max(min_required, DEFAULT_BUFFERS)
        self._buffer_list = []
        for _ in range(nbuf):
            if payload_size > 0:
                b = self._datastream.AllocAndAnnounceBuffer(payload_size)
            else:

                b = self._datastream.AllocAndAnnounceBuffer()
            self._buffer_list.append(b)


    def conversion_supported(self, source_pixel_format: int) -> bool:
        try:
            outs = self._image_converter.SupportedOutputPixelFormatNames(source_pixel_format)
            return any(TARGET_PIXEL_FORMAT == pf for pf in outs)
        except Exception:
            return False

    def _wait_for_live_fps(self, min_frames: int = 8, timeout: float = 3.0) -> int:
        """Wait until at least `min_frames` frames arrive, then estimate FPS.
        Returns 0 if no valid FPS can be estimated within timeout."""
        start_count = self.frame_count
        t0 = time.time()
        while time.time() - t0 < timeout:
            arrived = self.frame_count - start_count
            if arrived >= min_frames:
                fps = self.get_actual_fps()
                if fps > 0:
                    return fps
            time.sleep(0.005)
        return 0



    @pyqtSlot()
    @pyqtSlot(int)
    def start_recording(self, fps: Optional[int] = None):
        if self.is_recording:
            self._auto_start_pending = False
            return
        if self._datastream is None:
            self._init_data_stream()

        # Determine the recording FPS by MEASURING the true frame-arrival rate —
        # never assume it. Earlier code hardcoded fps=30 in HW-trigger mode on
        # the assumption the DMD MCU divides 60 Hz HDMI by 2 -> 30 Hz on
        # TRIG_OUT_2. That is NOT guaranteed: when the DMD pattern cycle is slow
        # the trigger arrives well below 30 Hz (observed ~11 Hz), and hardcoding
        # 30 MIS-TAGS the TIFF — the file claims 30 fps while frames actually
        # arrive slower, so playback runs too fast and the timeline is
        # temporally aliased (the operator "loses" frames relative to the tag).
        # get_actual_fps() measures arrival times over a trailing 2 s window, so
        # it is honest in BOTH free-run and HW-trigger modes — unlike the
        # AcquisitionFrameRate node, which reports the sensor's exposure-limited
        # max in HW mode, not the trigger rate. An explicitly-passed fps (caller
        # override) is still respected.
        if fps is None or fps <= 0:
            print("⏳ Measuring live frame rate...")
            est = self._wait_for_live_fps(min_frames=8, timeout=3.0)
            if est > 0:
                fps = est
                _mode = "HW-trigger" if self.acquisition_mode == 1 else "free-run"
                print(f"🎯 Using measured FPS ≈ {fps:.1f} ({_mode})")
                # Fail-loud guard: in HW-trigger mode a rate far below the camera's
                # free-run ceiling means TRIG_OUT_2 (DMD pattern cycle) is the
                # bottleneck, not the camera — the recording is undersampling.
                if self.acquisition_mode == 1 and fps < 25:
                    print(
                        f"⚠️ HW-trigger rate {fps:.1f} Hz is well below 30 Hz — the DMD "
                        f"is triggering slowly (check DMD pattern cycle / TRIG_OUT_2 "
                        f"config). The file is tagged at the TRUE rate, but the camera "
                        f"is undersampling the scene."
                    )
            else:
                print("🛑 No frames detected. Recording aborted.")
                self._auto_start_pending = False  # let next trigger retry
                return

        try:
            rec_fps = int(round(fps))  # round, don't truncate (29.96 -> 30, not 29)
            self.video_recorder.start_recording(rec_fps)
            self.is_recording = True
            # Reset silent-drop counter for this recording session.
            self._recording_queue_drops = 0
            # Clear armed/pending state only after successful start.
            self.is_armed = False
            self._auto_start_pending = False
            self.recordingStarted.emit()
            print(f"🔴 Recording started at {rec_fps} FPS (measured {fps:.1f})")
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self._auto_start_pending = False  # let next trigger retry



    @pyqtSlot()
    def stop_recording(self):
        if not self.is_recording:
            return
        try:
            self.video_recorder.stop_recording()
        except Exception:
            pass
        self.is_recording = False
        self.recordingStopped.emit()

    @pyqtSlot()
    def arm_recording(self):
        """Arm the system for hardware trigger recording"""
        print(f"🔫 Attempting to arm - mode: {self.acquisition_mode}, running: {self.acquisition_running}, recording: {self.is_recording}")
        if self.acquisition_mode == 1 and self.acquisition_running and not self.is_recording:
            self.is_armed = True
            self._auto_start_pending = False  # ensure fresh auto-start gate
            print("🔫 Recording armed - waiting for hardware trigger")
            return True
        print("❌ Cannot arm recording - conditions not met")
        return False

    @pyqtSlot()
    def disarm_recording(self):
        """Disarm the system"""
        self.is_armed = False
        self._auto_start_pending = False
        print("🔓 Recording disarmed")



    def start_calibration(self):
        with self.calibration_lock:
            if self.calibration_running:
                print("⚠️ Calibration already in progress"); return
            self.calibration_running = True

        def delayed_capture():
            try:
                save_path = os.path.join(self.asset_dir, "calibration_capture_image.png")
                latest = None
                for _ in range(20):
                    latest = self.get_data_stream_image()
                    if latest is not None: break
                    time.sleep(0.005)
                if latest is None:
                    print("❌ Failed to capture image for calibration")
                    return
                ids_peak_ipl.ImageWriter.WriteAsPNG(save_path, latest)
                self.thread_pool.submit(compute_h)
            finally:
                pass

        def compute_h():
            try:
                from calibration import find_homography_aruco

                # L3 calibration audit: find_homography_aruco
                # now returns CalibrationResult, not a raw ndarray. The
                # pre-audit `if H is not None` check passed on every
                # silent-success np.eye(3) return, so the "✅ Success!"
                # popup fired regardless of actual outcome. We now gate on
                # result.valid and surface result.message on failure.
                # Reference = the registration image that was actually projected
                # (built by _calibrate from the ChArUco board / generated), not
                # the source board file which may be a different size.
                result = find_homography_aruco(
                    registration_path=_assets_path("Generated", "custom_registration_image.png")
                )
                if not result.valid:
                    print(f"❌ Calibration failed: {result.message}")
                    return
                H = result.H
                self.translation_matrix = H  # keep raw H
                # Send H to projector engine via ZMQ
                try:
                    self._send_h_to_projector(H)
                except Exception as esend:
                    print(f"⚠️ Could not send H to projector: {esend}")
                # Also write H to a text file for preloading at projector startup
                try:
                    self._write_h_txt(H)
                except Exception as ewrite:
                    print(f"⚠️ Could not write H txt: {ewrite}")
                img_path = _assets_path("Generated", "custom_registration_image.png")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    try:
                        # Always project using H (not inverse)
                        Hn = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
                        print("📽️ Projecting with H for confirmation...")
                        self._safe_project(img, Hn)
                    except Exception as ewarp:
                        print(f"⚠️ Projection with H failed ({ewarp}); projecting image without warp")
                        self._safe_project(img, None)
                print(f"✅ Homography computed successfully: {result.message}")
                # Notify the GUI so the live preview can refresh without the
                # user needing to touch digital gain to wake it up.
                try:
                    self.calibrationFinished.emit()
                except Exception:
                    pass
            except Exception as e:
                print(f"❌ Homography error: {e}")
            finally:
                with self.calibration_lock:
                    self.calibration_running = False


        try:
            img_path = _assets_path("Generated", "custom_registration_image.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                self._safe_project(img, None)
            QTimer.singleShot(80, delayed_capture)
        except Exception as e:
            print(f"❌ Error starting calibration: {e}")
            with self.calibration_lock:
                self.calibration_running = False

    def _safe_project(self, img, H):

        try:
            self._interface.on_projection_received(img, H)
        except Exception:
            pass

    def _send_h_to_projector(self, H):
        """Send 3x3 homography to projector engine via the L3-audited helper.

        Stage-4 fix: replace inline ZMQ with
        delegation to ``core.projector._send_homography_inline`` — the
        audited helper that handles RCVTIMEO + WARNING-level logging
        on no-ACK + try/finally socket cleanup. Hardware verify Test 4
        (commit 06bc197) showed the inline path silently swallowed
        "no ACK" failures; this delegation surfaces them via the
        audited contract.

        Returns
        -------
        bool
            True on send+ACK success, False on timeout or error.
        """
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _cs = _Path(__file__).resolve().parent / "CS"
            if _cs.is_dir() and str(_cs) not in _sys.path:
                _sys.path.insert(0, str(_cs))
            from core.projector import _send_homography_inline
        except Exception as e:
            print(f"❌ Could not import audited send_homography helper: {e}")
            return False

        H_arr = np.asarray(H, dtype=np.float64).reshape(3, 3)
        success = _send_homography_inline(H_arr, "tcp://127.0.0.1:5560")
        if success:
            print("✅ Sent H to projector")
        else:
            print("⚠️ H delivery to projector failed — see log for ZMQ error")
        return success

    def _write_h_txt(self, H):
        import numpy as np
        import os
        arr = np.asarray(H, dtype=np.float64).reshape(3, 3)
        out_path = os.path.join(self.asset_dir, "homography_cam2proj.txt")
        with open(out_path, "w") as f:
            vals = arr.reshape(-1)
            f.write(" ".join(f"{float(v):.17g}" for v in vals))
        print(f"💾 Wrote H text: {out_path}")



    def _find_and_set_enum(self, name: str, value: str):
        try:
            node = self.node_map.FindNode(name)
            entries = node.Entries()
            vals = [e.SymbolicValue() for e in entries
                    if e.AccessStatus() not in (ids_peak.NodeAccessStatus_NotAvailable,
                                                ids_peak.NodeAccessStatus_NotImplemented)]
            if value in vals:
                node.SetCurrentEntry(value)
        except Exception:
            pass

    def set_remote_device_value(self, name: str, value):
        try:
            self.node_map.FindNode(name).SetValue(value)
        except ids_peak.Exception:
            try:
                self._interface.warning(f"Could not set value for {name}!")
            except Exception:
                pass



    def _start_acquisition_thread(self):
        if self._acq_thread and self._acq_thread.is_alive():
            return
        self._acq_stop.clear()
        t = threading.Thread(target=self._acquisition_loop,
                             name="AcquisitionLoop", daemon=True)
        self._acq_thread = t
        t.start()


    def acquisition_thread(self):
        self._acquisition_loop()

    def _ui_alive(self) -> bool:

        try:
            import sip
            return not sip.isdeleted(self._interface)
        except Exception:
            return True  

    def _acquisition_loop(self):
        print("Camera acquisition thread started")
        while not self._acq_stop.is_set() and not self.killed:
            try:
                self.get_data_stream_image()
            except Exception as e:
                now = time.time()
                if now - self._last_acq_err_ts > self._acq_err_interval:
                    try:
                        self._interface.warning(f"Acquisition error: {str(e)}")
                    except Exception:
                        pass
                    self._last_acq_err_ts = now
                self.save_image = False

    def _record_frame_arrival(self) -> None:
        """Record that a frame just arrived. Call once per delivered frame."""
        self.frame_times.append(time.time())

    def get_actual_fps(self) -> float:
        """Read current FPS as a pure function — safe to call from timers.
        Returns frames-per-second over a trailing 2-second window. Decays to 0
        when no frames arrive."""
        now = time.time()
        cutoff = now - 2.0
        while self.frame_times and self.frame_times[0] < cutoff:
            self.frame_times.popleft()
        if len(self.frame_times) < 2:
            self.GUIfps = 0.0
            return 0.0
        window_span = self.frame_times[-1] - self.frame_times[0]
        if window_span <= 0:
            self.GUIfps = 0.0
            return 0.0
        # Use span-based FPS (N-1 intervals over span) so recent arrivals weigh correctly
        fps = (len(self.frame_times) - 1) / window_span
        self.GUIfps = fps
        return fps

    def _update_performance_metrics(self):
        dur = max(1e-6, time.time() - self.start_time)
        self.performance_stats["fps"] = float(self.frame_count) / dur
        try:
            self.performance_metrics.emit(self.performance_stats)
        except Exception:
            pass

    def get_data_stream_image(self):

        if not self.acquisition_running or self._datastream is None or self.killed:
            time.sleep(0.001)
            return None

        timeout = 500 if self.acquisition_mode == 0 else 2000
        try:
            buffer = self._datastream.WaitForFinishedBuffer(timeout)
        except ids_peak.Exception as e:
            s = str(e)
            if "GC_ERR_TIMEOUT" in s or "GC_ERR_ABORT" in s:
                return None
            return None

        if buffer is None:
            if self.acquisition_mode == 1:
                time.sleep(0.001)
            return None

        # Auto-start recording if armed and hardware trigger detected.
        # HW-1 fix: don't clear is_armed here. start_recording() clears it on
        # success (camera.py:933). If start_recording fails (e.g. FPS estimation
        # couldn't complete), keeping is_armed=True lets subsequent trigger
        # frames retry instead of leaving the user silently disarmed.
        # Edge: start_recording is idempotent (bails if is_recording already
        # True), so multiple frames racing in while setup runs is safe.
        if self.acquisition_mode == 1 and self.is_armed and not self.is_recording:
            if not getattr(self, '_auto_start_pending', False):
                self._auto_start_pending = True
                print("🎯 Hardware trigger detected while armed - starting recording automatically")
                self.autoStartRecording.emit()

        try:
            ipl = ids_peak_ipl_extension.BufferToImage(buffer)
            if self._dest_pf is None:
                self._dest_pf = self._pick_dest_pf(ipl)
            converted = self._image_converter.Convert(ipl, self._dest_pf)
            try:
                converted_independent = converted.Clone()
            except Exception:
                converted_independent = converted

        finally:
            try:
                self._datastream.QueueBuffer(buffer)
            except Exception:
                pass


        if self._ui_alive():
            try:
                self.frame_ready.emit(converted_independent)
                # DEBUG (off unless STIM_FRAME_DEBUG=1): trace frame delivery
                # through the camera → Interface → Display chain. Logs once
                # per 30 frames (~1 s at 30 fps) to confirm the camera side
                # is alive without flooding the log.
                if os.environ.get("STIM_FRAME_DEBUG") == "1" and self.frame_count % 30 == 0:
                    try:
                        w = converted_independent.Width() if hasattr(converted_independent, "Width") else "?"
                        h = converted_independent.Height() if hasattr(converted_independent, "Height") else "?"
                        print(f"[FRAME-DEBUG cam] emitted frame_ready #{self.frame_count} ({w}x{h})")
                    except Exception:
                        print(f"[FRAME-DEBUG cam] emitted frame_ready #{self.frame_count}")
            except Exception:
                pass


        self.frame_count += 1
        # HW-1 fix: record arrival timestamp so get_actual_fps() /
        # _wait_for_live_fps() work. Previously _record_frame_arrival() was
        # orphaned, so HW-mode start_recording always aborted with
        # "No frames detected" because FPS estimation timed out.
        self._record_frame_arrival()
        if (self.frame_count % 60) == 0:
            try:
                pf = converted.PixelFormat() if hasattr(converted, "PixelFormat") else "?"
            except Exception:
                print(f"[camera] emitted frame #{self.frame_count}")

        rec_img = converted_independent
        if self.is_recording:
            try:
                self.recording_queue.put_nowait(rec_img)
            except queue.Full:
                self._recording_queue_drops += 1
                # Rate-limited log to flag sustained disk-I/O bottleneck.
                if self._recording_queue_drops in (1, 10, 100) or \
                   (self._recording_queue_drops % 100 == 0):
                    print(f"[CAM] ⚠ recording_queue full — silent drop #{self._recording_queue_drops} "
                          f"(writer thread falling behind; avg_fps will be < 30)")

        if self._pipeline_active:
            try:
                self.pipeline_queue.put_nowait((time.monotonic(), converted_independent))
            except queue.Full:
                try:
                    self.pipeline_queue.get_nowait()  # Drop oldest
                    self.pipeline_queue.put_nowait((time.monotonic(), converted_independent))
                except queue.Empty:
                    pass


        if self.save_image:
            save_path = self._snapshot_path or self._valid_name(os.path.join(self.save_dir, "image"), ".png")
            try:
                try:
                    save_img = converted_independent.Clone()
                except Exception:
                    save_img = converted_independent
                self.save_queue.put_nowait((save_path, save_img))
                self.save_image = False
                self._snapshot_path = None
            except queue.Full:
                pass



        if (self.frame_count % 120) == 0:
            self._update_performance_metrics()

        return converted_independent

    def _valid_name(self, base: str, ext: str) -> str:
        num = 0
        while True:
            p = f"{base}_{num}{ext}"
            if not os.path.exists(p):
                return p
            num += 1




    def change_hardware_trigger_line(self, new_line: str):
        self.hardware_trigger_line = new_line
        if self.acquisition_running and self.acquisition_mode == 1:
            self.stop_hardware_acquisition()
            QTimer.singleShot(200, self.start_hardware_acquisition)
        return new_line


    def start_pipeline_feed(self):
        """Enable frame delivery to pipeline_queue."""
        while not self.pipeline_queue.empty():
            try:
                self.pipeline_queue.get_nowait()
            except queue.Empty:
                break
        self._pipeline_active = True

    def stop_pipeline_feed(self):
        """Disable frame delivery to pipeline_queue."""
        self._pipeline_active = False
        while not self.pipeline_queue.empty():
            try:
                self.pipeline_queue.get_nowait()
            except queue.Empty:
                break

    def grab_frame_for_pipeline(self, after_timestamp=None, timeout_s=2.0):
        """Grab a frame from pipeline_queue, optionally waiting for one after a given timestamp.
        Returns (timestamp, numpy_array) or raises TimeoutError.
        """
        import numpy as np
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                ts, ipl_img = self.pipeline_queue.get(timeout=0.01)
                arr = ipl_img.get_numpy_3D() if hasattr(ipl_img, 'get_numpy_3D') else ipl_img.get_numpy_2D()
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                frame = arr.astype(np.float32)
                if after_timestamp is None or ts >= after_timestamp:
                    return ts, frame
            except queue.Empty:
                continue
        raise TimeoutError(f"No frame received within {timeout_s}s")

    def join_workers(self, timeout: float = 2.0):
        t = self._acq_thread
        if t and t.is_alive():
            try: t.join(timeout=timeout)
            except Exception: pass


Camera = OptimizedCamera

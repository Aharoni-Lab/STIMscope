"""Standalone IDS Peak camera recorder for the demo bundle.

Captures the IDS Peak camera and writes three artifacts in lockstep:
  - demo_camera.mp4              — H.264/mp4v review track (lossy)
  - tiff_frames/frame_NNNNNN.tif — lossless single-page TIFFs at native
                                   bit-depth (publication-grade verifiable raw)
  - demo_frames.csv camera_meta rows — one per frame, both host monotonic
                                       and (in slave mode) the IDS buffer's
                                       hardware timestamp (hw_ts_ns)

CAPTURE MODES — match how STIMscope is supposed to operate per the preprint
(Trig out 1/2 from the DMD → MCU → image sensor sync lines, sensor in
slave mode integrating over a single coherent pattern presentation):

  slave (DEFAULT, publication-grade)
    TriggerSelector = ExposureStart
    TriggerMode     = On
    TriggerSource   = $STIM_TRIGGER_LINE (default Line0)
    TriggerActivation = RisingEdge
    No AcquisitionFrameRate (clock comes from the trigger line).
    Eliminates rolling-shutter banding and missed-pattern frames.
    REQUIRES: DMD has been booted + is issuing Trig out 1/2 on the
              configured GPIO line. If no trigger ever arrives, we
              surface "no trigger detected after Ns" rather than
              hanging silently.

  freerun (--freerun, development only)
    TriggerMode = Off, AcquisitionFrameRate set explicitly. Sensor
    samples on its own clock, NOT phase-locked to the DMD. Use this
    for camera-only smoke tests; do NOT use for demo recordings the
    preprint relies on.

Reference implementation: STIMscope/STIMViewer_CRISPI/camera.py:862–887
(_select_trigger). This file mirrors that pattern.

Designed to be run as a subprocess by tools/demo/run_demo.py (launched via
scripts/run_demo.sh / `make demo`). Handles SIGINT/SIGTERM cleanly (finalizes
mp4, flushes TIFF writes, closes camera).

Usage:
    python3 tools/demo/camera_recorder.py \\
        --out /path/to/demo_camera.mp4 \\
        --log /path/to/demo_frames.csv \\
        --fps 30
    # development:
    python3 tools/demo/camera_recorder.py … --freerun
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Ensure the demo helpers + IDS Peak shim are importable
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "STIMscope" / "STIMViewer_CRISPI"))
sys.path.insert(0, str(_HERE))

from logger import DemoLogger  # noqa: E402

_STOP = False


def _signal_handler(signum, frame):
    global _STOP
    _STOP = True


def main(argv=None):
    import os
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--fps", type=int, default=30,
                   help="Target FPS. In freerun, used for AcquisitionFrameRate "
                        "and the mp4 header. In slave mode, used only as the "
                        "mp4 header rate; actual rate is driven by the trigger.")
    p.add_argument("--max-seconds", type=int, default=600,
                   help="Hard cap on recording duration")
    p.add_argument("--freerun", action="store_true",
                   help="DEVELOPMENT ONLY: ignore the DMD trigger and let the "
                        "sensor sample on its own clock. Use for camera-only "
                        "smoke tests. Slave mode is the default and the only "
                        "mode that produces publication-grade recordings.")
    p.add_argument("--trigger-source", default=os.environ.get("STIM_TRIGGER_LINE", "Line0"),
                   help="GenICam TriggerSource for slave mode (default: Line0, "
                        "overridable via $STIM_TRIGGER_LINE)")
    p.add_argument("--trigger-wait-sec", type=float, default=10.0,
                   help="If slave mode and no trigger arrives within this "
                        "many seconds, abort with a diagnostic so a silent "
                        "DMD-not-issuing-triggers failure surfaces fast.")
    p.add_argument("--no-tiff", action="store_true",
                   help="Skip writing per-frame TIFFs (mp4 only). The default "
                        "writes lossless TIFFs to <out_dir>/tiff_frames/ for "
                        "scientific verification.")
    args = p.parse_args(argv)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Try to import the platform's IDS Peak backend first (audited path);
    # fall back to direct ids_peak SDK call if backend isn't importable.
    try:
        from ids_peak_backend import IDSPeakBackend  # noqa: F401
        backend_mode = "audited_backend"
    except Exception:
        backend_mode = "ids_peak_direct"

    print(f"[camera_recorder] backend={backend_mode}, fps={args.fps}, out={args.out}")

    # Match the API pattern used by the working camera.py + video_recorder.py:
    # the ipl_extension provides BufferToImage; the image then has get_numpy_*
    # accessors. The Image_CreateFromSizeAndBuffer API used previously does
    # not exist in this SDK version (per CLAUDE.md "API changes between SDK
    # versions").
    from ids_peak import ids_peak
    from ids_peak_ipl import ids_peak_ipl  # noqa: F401 (imported for side-effect)
    from ids_peak import ids_peak_ipl_extension
    import cv2

    ids_peak.Library.Initialize()
    try:
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()
        if device_manager.Devices().empty():
            raise SystemExit("[camera_recorder] No IDS Peak device found")
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        node_map = device.RemoteDevice().NodeMaps()[0]

        # ── Trigger configuration ──────────────────────────────────────────
        # Default: slave mode (publication-grade). DMD's Trig out 1/2 → MCU → image
        # sensor sync line → camera ExposureStart fires once per projected
        # pattern. This is the operating mode the STIMscope preprint relies on.
        #
        # Pattern mirrors STIMscope/STIMViewer_CRISPI/camera.py:862–887
        # (_select_trigger). Each step is in its own try/except so a missing
        # node on an SDK variant degrades gracefully rather than aborting.
        if args.freerun:
            try:
                node_map.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
                node_map.FindNode("TriggerMode").SetCurrentEntry("Off")
                print("[camera_recorder] TriggerMode=Off (freerun — DEVELOPMENT ONLY)")
            except Exception as _e:
                print(f"[camera_recorder] WARN: could not force TriggerMode=Off: {_e}")
        else:
            try:
                # Probe which selectors the SDK exposes; ExposureStart is the
                # right one for per-frame trigger, but fall back to whatever
                # is available so we don't hard-fail on an SDK variant.
                entries = node_map.FindNode("TriggerSelector").Entries()
                symbols = [e.SymbolicValue() for e in entries
                           if e.AccessStatus() not in (
                               ids_peak.NodeAccessStatus_NotAvailable,
                               ids_peak.NodeAccessStatus_NotImplemented)]
                sel = "ExposureStart" if "ExposureStart" in symbols else (symbols[0] if symbols else None)
                if sel:
                    node_map.FindNode("TriggerSelector").SetCurrentEntry(sel)
                    print(f"[camera_recorder] TriggerSelector={sel}")
            except Exception as _e:
                print(f"[camera_recorder] WARN: could not set TriggerSelector: {_e}")
            try:
                node_map.FindNode("TriggerMode").SetCurrentEntry("On")
                node_map.FindNode("TriggerSource").SetCurrentEntry(args.trigger_source)
                node_map.FindNode("TriggerActivation").SetCurrentEntry("RisingEdge")
                print(f"[camera_recorder] SLAVE MODE: TriggerMode=On  "
                      f"TriggerSource={args.trigger_source}  Activation=RisingEdge")
            except Exception as _e:
                raise SystemExit(
                    f"[camera_recorder] FATAL: slave-mode trigger setup failed: {_e}\n"
                    f"  Either the trigger source '{args.trigger_source}' is not "
                    f"available on this sensor, or the SDK is missing trigger "
                    f"nodes. Try a different --trigger-source (e.g. Line1) or "
                    f"--freerun for a development capture without DMD sync."
                )
        # Camera-side TriggerDelay (µs): the deterministic phase control for the
        # half-black problem. The camera is slave-triggered at 30 Hz off the DMD
        # TRIG_OUT; within each HDMI frame the DMD shows R / G(dead) / B sub-
        # frames. TriggerDelay offsets exposure-start from the trigger edge so
        # the window lands on the intended R+B illumination (docs §10.4). IDS
        # Peak exposes TriggerDelay in µs (0..16.7e6, edge-only; pulses arriving
        # during the delay are ignored). The value is rig-specific and must be
        # bench-tuned; default 0. Only meaningful in slave mode.
        if not args.freerun:
            try:
                trig_delay_us = float(os.environ.get("STIM_TRIG_DELAY_US", "0"))
                if trig_delay_us > 0:
                    node_map.FindNode("TriggerDelay").SetValue(trig_delay_us)
                    print(f"[camera_recorder] TriggerDelay = {trig_delay_us:.0f}µs "
                          "(phase-align exposure to DMD illumination)")
                else:
                    print("[camera_recorder] TriggerDelay = 0µs (no phase offset; "
                          "set STIM_TRIG_DELAY_US — run_demo's --trig-delay-us — "
                          "to tune)")
            except Exception as _e:
                print(f"[camera_recorder] WARN: could not set TriggerDelay "
                      f"(node may be unavailable on this sensor): {_e}")
        try:
            current = float(node_map.FindNode("ExposureTime").Value())
            # Exposure MUST be ≤ one 60 Hz HDMI frame (16.7 ms) or the camera
            # integrates light from MULTIPLE DMD pattern presentations into a
            # single frame — visible as banding/blending across the projected
            # shape. At 30 ms (previous default) the camera saw ~2 HDMI
            # frames = up to 6 patterns (num_patterns=3 path) blended into
            # one capture. 15 ms gives ~1.7 ms margin under one HDMI frame.
            # Env-overridable for tuning per rig.
            import os as _os
            target = float(_os.environ.get("CAMERA_EXPOSURE_US", "15000"))
            if target > 16000:
                print(f"[camera_recorder] WARN: CAMERA_EXPOSURE_US={target:.0f}µs > 16000µs "
                      f"— camera will integrate across multiple HDMI frames, "
                      f"expect banding artifacts in projection capture.")
            node_map.FindNode("ExposureTime").SetValue(target)
            print(f"[camera_recorder] ExposureTime: {current:.0f}µs → {target:.0f}µs")
        except Exception as _e:
            print(f"[camera_recorder] WARN: could not set ExposureTime: {_e}")
        # Guard: in slave mode, trigger_delay + exposure (+ readout) must fit one
        # trigger period or the NEXT edge arrives mid-exposure and is ignored —
        # the sensor captures every other trigger (~half fps), SILENTLY (an
        # ignored edge produces no buffer, so it is not an sdk_lost/write drop).
        # This is the failure class commit 72e2898 killed via the exposure cap;
        # the new TriggerDelay knob can re-introduce it, so warn loud.
        if not args.freerun and args.fps > 0:
            try:
                period_us = 1e6 / float(args.fps)
                _td = float(os.environ.get("STIM_TRIG_DELAY_US", "0"))
                _exp = float(os.environ.get("CAMERA_EXPOSURE_US", "15000"))
                if _td + _exp > period_us - 2000:   # ~2 ms readout margin
                    print(f"[camera_recorder] *** WARNING: trig_delay({_td:.0f}) + "
                          f"exposure({_exp:.0f}) = {_td + _exp:.0f}µs exceeds the "
                          f"{period_us:.0f}µs trigger period − 2ms readout. The sensor "
                          "will MISS every other trigger (silent ~half fps). Lower "
                          "--exposure-us / --trig-delay-us. ***")
            except Exception:
                pass
        try:
            # Gain is a secondary brightness lever (primary is exposure +
            # trig-delay phase). In 8-bit-RGB each color is sub-framed (lit ~1/3
            # of the HDMI frame), so raise STIM_GAIN if captures are dark even
            # after phase tuning. Default 1.0 (deterministic; amplifies noise).
            gain = float(os.environ.get("STIM_GAIN", "1.0"))
            node_map.FindNode("GainAuto").SetCurrentEntry("Off")
            node_map.FindNode("Gain").SetValue(gain)
            print(f"[camera_recorder] GainAuto=Off, Gain={gain}")
        except Exception as _e:
            print(f"[camera_recorder] WARN: could not set Gain: {_e}")
        # Only set AcquisitionFrameRate in freerun. In slave mode the rate is
        # determined by the trigger and forcing AcquisitionFrameRate can
        # actually rate-limit the sensor below the trigger arrival rate.
        if args.freerun:
            try:
                node_map.FindNode("AcquisitionFrameRate").SetValue(float(args.fps))
            except Exception:
                pass
        # Determine width/height
        try:
            width = int(node_map.FindNode("Width").Value())
            height = int(node_map.FindNode("Height").Value())
        except Exception:
            width, height = 1936, 1096

        # Open data stream
        data_stream = device.DataStreams()[0].OpenDataStream()
        payload_size = node_map.FindNode("PayloadSize").Value()
        # Buffer pool sizing — critical for slave-mode capture under
        # any disk-write contention. With only the SDK-reported minimum
        # (~4 on this IDS Peak USB3 sensor), two slow iterations of the
        # receive loop exhaust the pool and subsequent triggers fire
        # into nothing — the GPIO line strobes but no image is stored.
        # The demo (run_demo.launch_camera) sets STIM_PEAK_BUFFERS=96 (~3 s of
        # slack at 30 Hz, ~400 MB) for zero-drop capture; 32 is only the
        # standalone fallback below. Mirrors the reference camera.py
        # DEFAULT_BUFFERS pattern; env-overridable.
        min_required = data_stream.NumBuffersAnnouncedMinRequired()
        nbuf = max(min_required, int(os.environ.get("STIM_PEAK_BUFFERS", "32")))
        print(f"[camera_recorder] buffer pool: {nbuf} (min_required={min_required})")
        for _ in range(nbuf):
            buf = data_stream.AllocAndAnnounceBuffer(payload_size)
            data_stream.QueueBuffer(buf)

        # Start acquisition EARLY (right after buffers are queued) so the camera
        # latches triggers immediately and the first WaitForFinishedBuffer returns
        # a pre-buffered frame — reliable slave-trigger detection. (Starting it
        # late, after the slow encoder init, made the first wait block on a live
        # trigger and time out → "no trigger detected".) The benign frames the SDK
        # drops during the encoder init are excluded from the reported sdk_lost by
        # baselining the counter just before the receive loop (see below).
        node_map.FindNode("TLParamsLocked").SetValue(1)
        data_stream.StartAcquisition()
        node_map.FindNode("AcquisitionStart").Execute()
        try:
            # AcquisitionStart is a fire-and-return SFNC command; WaitUntilDone is
            # redundant and the reference camera.py doesn't call it. Guard it so an
            # SDK variant that rejects it can't abort an already-armed stream.
            node_map.FindNode("AcquisitionStart").WaitUntilDone()
        except Exception:
            pass

        # mp4 writer — OPTIONAL. The per-frame software mp4 encode (~10 ms) is the
        # heaviest hot-path op besides LZW; on a long run it pushes the writer
        # thread over the 33 ms budget → the write queue backs up → drops + the
        # SDK starves. The lossless TIFFs are the scientific output and the
        # composer regenerates a (composite) mp4 from them, so the raw camera mp4
        # is redundant. The demo disables it (STIM_DISABLE_MP4=1) to keep the
        # writer well under budget; set STIM_DISABLE_MP4=0 to restore it.
        mp4_disabled = os.environ.get("STIM_DISABLE_MP4", "0").strip() in ("1", "true", "yes")
        writer = None
        if mp4_disabled:
            print("[camera_recorder] mp4 output disabled (STIM_DISABLE_MP4=1) — "
                  "TIFFs are the output; composer regenerates the review mp4")
        else:
            # H.264 (avc1) for broad player support; fall back to mp4v.
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(args.out), fourcc, float(args.fps), (width, height), isColor=False)
            if not writer.isOpened():
                print("[camera_recorder] avc1 not available, falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.out), fourcc, float(args.fps), (width, height), isColor=False)
            if not writer.isOpened():
                raise SystemExit(f"[camera_recorder] Failed to open mp4 writer at {args.out} (even mp4v failed)")

        logger = DemoLogger(args.log)
        logger.set_segment("camera_recorder")
        mode_label = "freerun" if args.freerun else f"slave({args.trigger_source})"
        logger.metric("camera_recorder_start",
                      f"width={width};height={height};fps={args.fps};mode={mode_label}")

        # TIFF output (lossless, native bit-depth) — sibling to the mp4.
        # Disabled if --no-tiff or STIM_DISABLE_TIFF=1 (env-overridable for
        # high-rate runs where TIFF compression would saturate disk and
        # back-pressure the SDK).
        tiff_disabled_env = os.environ.get("STIM_DISABLE_TIFF", "0").strip() in ("1", "true", "yes")
        tiff_dir = None
        if not args.no_tiff and not tiff_disabled_env:
            tiff_dir = args.out.parent / "tiff_frames"
            tiff_dir.mkdir(parents=True, exist_ok=True)
            print(f"[camera_recorder] TIFF frames -> {tiff_dir}")
        elif tiff_disabled_env:
            print("[camera_recorder] TIFF output disabled (STIM_DISABLE_TIFF=1)")
        else:
            print("[camera_recorder] TIFF output disabled (--no-tiff)")

        # cv2.imwrite's default TIFF encoder uses LZW compression, which at
        # 1936×1096 uint16 costs ~10-15 ms of CPU per frame on this Orin and
        # is the dominant term in writer-thread latency. NONE (uncompressed)
        # is ~2× the bytes on disk but ~10× faster — measured ~80 MB/s
        # sustained on the supplementary_data ext4 volume here, comfortable
        # for 30 Hz capture. Override with STIM_TIFF_COMPRESSION=lzw to
        # restore old behavior if disk space matters more than throughput.
        _tiff_comp = os.environ.get("STIM_TIFF_COMPRESSION", "none").lower()
        _TIFF_COMPRESSION_NONE = 1  # libtiff COMPRESSION_NONE
        _TIFF_COMPRESSION_LZW = 5
        _tiff_comp_code = (_TIFF_COMPRESSION_LZW if _tiff_comp == "lzw"
                           else _TIFF_COMPRESSION_NONE)
        _tiff_params = [cv2.IMWRITE_TIFF_COMPRESSION, _tiff_comp_code]
        if tiff_dir is not None:
            print(f"[camera_recorder] TIFF compression: {_tiff_comp}")

        # Per-buffer wait timeout strategy:
        #   - BEFORE first frame in slave mode: trigger_wait_sec (long enough
        #     to differentiate "slow trigger" from "no trigger at all" and
        #     fire the watchdog with a useful diagnostic).
        #   - AFTER first frame (any mode): 1 s. The shorter wait makes
        #     SIGTERM/SIGINT-driven shutdown responsive within ≤1 s, which
        #     matters because `docker stop --time N` SIGKILLs at N seconds
        #     and the mp4 writer needs writer.release() to complete cleanly
        #     for the moov atom to be written. A 10 s wait would mean the
        #     mp4 is finalized only intermittently — a recurring source of
        #     "Cannot open mp4: moov atom not found" composer failures.
        wait_ms_initial = int(max(1.0, args.trigger_wait_sec) * 1000) if not args.freerun else 1000
        wait_ms_running = 1000

        # Writer thread + bounded queue — keeps slow disk writes off the
        # receive loop. The receive loop only does (extract numpy,.copy(),
        # queue.put_nowait); TIFF + mp4 + CSV writes run in the writer.
        # The demo runs STIM_WRITE_QUEUE=360 (~12 s of slack at 30 Hz); 240 is
        # only the standalone fallback below. Even with LZW TIFF an occasional
        # sync stall can buffer here without back-pressuring the SDK.
        # Env-overridable for tuning.
        wq_max = int(os.environ.get("STIM_WRITE_QUEUE", "240"))
        write_q: "queue.Queue" = queue.Queue(maxsize=wq_max)
        write_drops = {"n": 0}

        def writer_loop():
            while True:
                item = write_q.get()
                if item is None:
                    write_q.task_done()
                    return
                np_image, frame_id_local, hw_ts_ns_local = item
                try:
                    if tiff_dir is not None:
                        cv2.imwrite(str(tiff_dir / f"frame_{frame_id_local:06d}.tif"),
                                    np_image, _tiff_params)
                    if writer is not None:
                        if np_image.dtype != np.uint8:
                            np_image_mp4 = ((np_image >> 8).astype(np.uint8)
                                            if np_image.dtype == np.uint16
                                            else np_image.astype(np.uint8, copy=False))
                        else:
                            np_image_mp4 = np_image
                        writer.write(np_image_mp4)
                    logger.camera_meta(
                        frame_id=frame_id_local,
                        hw_ts_ns=hw_ts_ns_local,
                        extra=f"shape={np_image.shape};dtype={np_image.dtype}",
                    )
                except Exception as e:
                    print(f"[camera_recorder] writer error on frame {frame_id_local}: {e}")
                finally:
                    write_q.task_done()

        writer_thread = threading.Thread(target=writer_loop, name="cam-writer", daemon=True)
        writer_thread.start()

        # Read the SDK lost-frame counter from whichever source THIS SDK build
        # exposes. The convenience method NumLostFrames() is often absent on this
        # IDS Peak build (it was — that's why the earlier baseline read 0 while
        # teardown read the node cumulatively); the GenTL stream node map carries
        # StreamLostFrameCount. Used for BOTH the pre-loop baseline and the
        # teardown read, so the reported value is the DELTA during capture.
        def _read_lost():
            try:
                return int(data_stream.NumLostFrames())
            except Exception:
                pass
            try:
                _snm = data_stream.NodeMaps()[0]
            except Exception:
                _snm = None
            for _nm in (_snm, node_map):
                if _nm is None:
                    continue
                for _nn in ("StreamLostFrameCount", "StreamDroppedFrameCount",
                            "StreamUnderrunCount", "LostFrameCount",
                            "StreamFailedBufferCount"):
                    try:
                        return int(_nm.FindNode(_nn).Value())
                    except Exception:
                        continue
            return None

        # Baseline just before the receive loop (after the slow encoder init) so
        # the benign startup-init losses are excluded; only a real mid-stream loss
        # DURING capture is counted in the teardown delta. The captured stream is
        # gap-free (std=0 ms inter-frame), so a correct baseline reports ~0.
        sdk_lost_base = _read_lost() or 0
        print(f"[camera_recorder] sdk-lost baseline at capture start: {sdk_lost_base}")

        frame_id = 0
        t0 = time.monotonic()
        first_frame_received = False
        while not _STOP:
            if time.monotonic() - t0 > args.max_seconds:
                print(f"[camera_recorder] Max duration ({args.max_seconds}s) reached")
                break
            try:
                wait_ms = wait_ms_initial if not first_frame_received else wait_ms_running
                buffer = data_stream.WaitForFinishedBuffer(wait_ms)
            except Exception as e:
                # In slave mode a timeout before the first frame almost
                # always means the DMD isn't issuing triggers (i2c boot
                # didn't ACK, projector engine isn't claiming the GPIO line,
                # etc.). Surface that distinctly from generic buffer errors.
                if not first_frame_received and not args.freerun:
                    elapsed = time.monotonic() - t0
                    if elapsed >= args.trigger_wait_sec:
                        raise SystemExit(
                            f"[camera_recorder] FATAL: no trigger detected after "
                            f"{elapsed:.1f}s in slave mode (source={args.trigger_source}).\n"
                            f"  The DMD is configured for slave-mode capture but no "
                            f"rising edge has arrived on the trigger line. Likely causes:\n"
                            f"    1. DMD i2c boot failed (check {args.log.parent}/i2c_boot.log)\n"
                            f"    2. Projector engine not claiming the GPIO trigger line\n"
                            f"    3. Trigger source mismatch — try --trigger-source Line1\n"
                            f"    4. Hardware cabling between MCU and sensor sync input\n"
                            f"  For a development capture without DMD sync, use --freerun."
                        )
                print(f"[camera_recorder] buffer wait error: {e}")
                continue

            # Capture both timestamps as close to buffer-receive as possible.
            # host_ts_ns is the cross-process clock the composer/run_demo share.
            # hw_ts_ns is the IDS buffer's hardware timestamp (sensor clock
            # domain) — used by metrics to verify trigger lock.
            hw_ts_ns = None
            try:
                # IDS Peak SDK: buffer.Timestamp_ns() is preferred; older
                # variants expose Timestamp() (in nanoseconds already).
                if hasattr(buffer, "Timestamp_ns"):
                    hw_ts_ns = int(buffer.Timestamp_ns())
                elif hasattr(buffer, "Timestamp"):
                    hw_ts_ns = int(buffer.Timestamp())
            except Exception:
                hw_ts_ns = None  # not fatal — just no jitter metric for this frame

            try:
                # API matches camera.py:1288 — BufferToImage from ipl_extension.
                ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                # Try shaped getters first (matches video_recorder.py pattern);
                # fall back to 1D + reshape if shaped getters aren't available.
                np_image = None
                for attr in ("get_numpy_2D", "get_numpy_3D"):
                    fn = getattr(ipl_image, attr, None)
                    if callable(fn):
                        try:
                            np_image = fn().copy()  #.copy() — break IDS Peak buffer aliasing per L3 video_recorder fix
                            break
                        except Exception:
                            continue
                if np_image is None:
                    # Last resort: 1D + manual reshape
                    for attr in ("get_numpy_1D", "get_numpy"):
                        fn = getattr(ipl_image, attr, None)
                        if callable(fn):
                            try:
                                flat = fn().copy()
                                np_image = flat.reshape(height, width)
                                break
                            except Exception:
                                continue
                if np_image is None:
                    print(f"[camera_recorder] could not extract numpy from buffer; skipping frame")
                    continue
                # Reduce to grayscale if multi-channel
                if np_image.ndim == 3:
                    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
                # Hand the native-bit-depth image off to the writer thread
                # for TIFF/mp4/CSV. The receive loop must not block on disk;
                # if the writer falls behind we drop the frame (and count it)
                # rather than back-pressure into SDK buffer exhaustion.
                try:
                    write_q.put_nowait((np_image, frame_id, hw_ts_ns))
                    frame_id += 1
                    first_frame_received = True
                except queue.Full:
                    write_drops["n"] += 1
                    if write_drops["n"] in (1, 10, 100) or write_drops["n"] % 500 == 0:
                        print(f"[camera_recorder] WARN: write queue full, "
                              f"dropped frame (cumulative drops={write_drops['n']}, "
                              f"q={write_q.qsize()}/{wq_max}). Disk is slower "
                              f"than trigger rate — raise STIM_WRITE_QUEUE or "
                              f"lower trigger rate.")
            finally:
                data_stream.QueueBuffer(buffer)

        # Drain the writer queue before tearing down anything it touches.
        # writer_loop closes itself on sentinel; join with a generous bound
        # so we don't hang shutdown if the disk is wedged.
        print(f"[camera_recorder] draining write queue (q={write_q.qsize()})…")
        write_q.put(None)
        writer_thread.join(timeout=30.0)
        if writer_thread.is_alive():
            print("[camera_recorder] WARN: writer thread did not drain in 30 s — "
                  "TIFF/mp4 may be truncated")

        # SDK-side lost-frame counter, if exposed by this build. Probe several
        # APIs/nodes so "every frame captured" can be VERIFIED rather than left
        # "unknown": (1) the DataStream convenience method, (2) the GenTL data
        # stream's own node map (the authoritative source — counts buffers the
        # SDK dropped before our receive loop ever saw them), (3) the remote
        # device node map. First hit wins.
        # Delta from the pre-loop baseline (same source) → only losses that
        # occurred DURING capture. A gap-free captured stream reports ~0 here;
        # the larger raw counter is dominated by benign startup-init losses.
        _final_lost = _read_lost()
        sdk_lost = (max(0, _final_lost - sdk_lost_base)
                    if _final_lost is not None else None)
        if _final_lost is not None:
            print(f"[camera_recorder] sdk-lost: final={_final_lost} - baseline="
                  f"{sdk_lost_base} = {sdk_lost} during capture")

        # Teardown
        logger.metric("camera_recorder_total_frames", str(frame_id))
        logger.metric("camera_recorder_duration_sec", f"{time.monotonic() - t0:.3f}")
        logger.metric("camera_recorder_mode", mode_label)
        logger.metric("camera_recorder_write_drops", str(write_drops["n"]))
        if sdk_lost is not None:
            logger.metric("camera_recorder_sdk_lost_frames", str(sdk_lost))
        if tiff_dir is not None:
            logger.metric("camera_recorder_tiff_dir", str(tiff_dir))
        logger.close()
        _lost = (sdk_lost or 0)
        if write_drops["n"] > 0 or _lost > 0:
            print("[camera_recorder] *** DROPS DETECTED — recording is NOT "
                  f"frame-complete: write-queue drops={write_drops['n']}, "
                  f"SDK lost={sdk_lost if sdk_lost is not None else 'unknown'}. "
                  "Raise STIM_PEAK_BUFFERS / STIM_WRITE_QUEUE or lower the data "
                  "rate (STIM_TIFF_COMPRESSION=lzw). ***")
        else:
            print(f"[camera_recorder] captured {frame_id} frames, 0 write-queue "
                  f"drops, SDK lost={sdk_lost if sdk_lost is not None else 'unknown'}")

        if writer is not None:
            writer.release()
        try:
            node_map.FindNode("AcquisitionStop").Execute()
            data_stream.StopAcquisition()
            data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            node_map.FindNode("TLParamsLocked").SetValue(0)
        except Exception as e:
            print(f"[camera_recorder] teardown warn: {e}")
        print(f"[camera_recorder] Wrote {frame_id} frames to {args.out}")
    finally:
        ids_peak.Library.Close()


if __name__ == "__main__":
    main()

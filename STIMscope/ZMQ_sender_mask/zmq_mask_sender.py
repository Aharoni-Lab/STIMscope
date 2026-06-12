import os, time, zmq, json, numpy as np, argparse, glob
from PIL import Image

W, H = 1920, 1080

def _to_rgb_wh(img: np.ndarray, w: int, h: int) -> np.ndarray:
    gray = _to_gray_wh(img, w, h)
    return np.stack([gray, gray, gray], axis=-1)


def _to_gray_wh(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        # simple luminance
        img = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)
    elif img.ndim == 2:
        pass
    else:
        img = np.zeros((h, w), np.uint8)
    if img.shape[0] != h or img.shape[1] != w:
        img = np.array(Image.fromarray(img).resize((w, h), resample=Image.BILINEAR))
    return img.astype(np.uint8, copy=False)

# ---------------------------------------------------------------------------
# Stage-5 closure-extraction refactor (iter 30):
# The following helpers were originally defined inside main() as closures.
# Hoisting them to module level enables direct testing + bumps coverage
# without changing runtime behavior. The closures captured `inv_x`/`inv_y`
# (for prewarp) and `args.flip_x`/`args.flip_y` (for flips) — those are
# now explicit parameters.
# ---------------------------------------------------------------------------


def pack_r_only(gray: np.ndarray, h: int = None, w: int = None) -> np.ndarray:
    """Pack a single-channel gray frame into the R channel of an HxWx3 RGB
    output (G=0, B=0). Used by --temporal-alternate stim frames."""
    if h is None: h = H
    if w is None: w = W
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = gray
    return rgb


def pack_b_only(gray: np.ndarray, h: int = None, w: int = None) -> np.ndarray:
    """Pack a single-channel gray frame into the B channel of an HxWx3 RGB
    output (R=0, G=0). Used by --temporal-alternate observe frames."""
    if h is None: h = H
    if w is None: w = W
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 2] = gray
    return rgb


def pack_composite_rgb(observe_gray: np.ndarray, stim_gray: np.ndarray,
                       h: int = None, w: int = None) -> np.ndarray:
    """Pack observe + stim into B + R channels (Mode B simultaneous-RGB)."""
    if h is None: h = H
    if w is None: w = W
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = stim_gray
    rgb[:, :, 2] = observe_gray
    return rgb


def apply_flips(img: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    """Apply horizontal/vertical flips. Returns the same array if no flips."""
    try:
        if flip_x and flip_y:
            return np.flipud(np.fliplr(img))
        if flip_x:
            return np.fliplr(img)
        if flip_y:
            return np.flipud(img)
    except Exception:
        pass
    return img


def apply_prewarp(img_gray: np.ndarray, inv_x, inv_y,
                  h: int = None, w: int = None) -> np.ndarray:
    """Apply LUT-based prewarp via cv2.remap. Returns unmodified image if
    inv_x or inv_y is None (no LUTs loaded)."""
    if inv_x is None or inv_y is None:
        return img_gray
    if h is None: h = H
    if w is None: w = W
    try:
        import cv2 as _cv2
        if inv_x.shape != (h, w):
            _ix = _cv2.resize(inv_x, (w, h), interpolation=_cv2.INTER_LINEAR)
            _iy = _cv2.resize(inv_y, (w, h), interpolation=_cv2.INTER_LINEAR)
        else:
            _ix, _iy = inv_x, inv_y
        warped = _cv2.remap(img_gray, _ix, _iy, interpolation=_cv2.INTER_LINEAR,
                            borderMode=_cv2.BORDER_CONSTANT, borderValue=0)
        return warped
    except Exception as _e:
        print(f"⚠️  LUT prewarp failed: {_e}")
        return img_gray


def load_segmask_from_npz(npz_path: str, h: int = None, w: int = None) -> np.ndarray:
    """Load a segmask from an.npz file with 'binary' or 'labels' keys.
    Returns a blank frame on any failure. Extracted from the inline
    main() loader for testability."""
    if h is None: h = H
    if w is None: w = W
    blank = np.zeros((h, w), np.uint8)
    try:
        data = np.load(npz_path, allow_pickle=False)
        if 'binary' in data:
            return (data['binary'] > 0).astype(np.uint8) * 255
        if 'labels' in data:
            return (data['labels'] > 0).astype(np.uint8) * 255
        return blank
    except Exception:
        return blank


def build_patterns(args):
    def blank(val=0):
        return np.full((H, W), val, np.uint8) if val else np.zeros((H, W), np.uint8)

    def moving_bar(t):
        img = blank()
        speed = args.speed
        w = max(1, args.bar_width)
        val = args.value
        x = int((t * speed) % (W + w)) - w
        x0, x1 = max(0, x), min(W, x + w)
        if x1 > x0: img[:, x0:x1] = val
        return img

    def checkerboard(_):
        sz = max(2, args.checker_size)
        img = blank()
        for y in range(0, H, sz):
            for x in range(0, W, sz):
                c = ((x//sz) + (y//sz)) & 1
                if c:
                    img[y:y+sz, x:x+sz] = args.value
        return img

    def solid(_):
        return blank(args.value)

    def circle(_):
        r = max(1, args.radius)
        img = blank()
        cy, cx = H//2, W//2
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        mask = (x - cx)**2 + (y - cy)**2 <= r*r
        img[mask] = args.value
        return img

    def gradient_sequence():
        # Steps from black to white with optional gamma and hold per step
        n = max(2, int(getattr(args, 'gradient_steps', 6)))
        g = float(getattr(args, 'gradient_gamma', 1.0))
        hold = max(1, int(getattr(args, 'gradient_hold', 10)))
        vals = []
        for i in range(n):
            x = i / float(n - 1)
            if g != 1.0:
                x = x ** g
            v = int(round(x * 255.0))
            vals.append(v)
        seq = []
        for v in vals:
            frame = blank(v)
            for _ in range(hold):
                seq.append(frame.copy())
        return seq

    seq = []
    if args.pattern == "folder":
        files = sorted(glob.glob(os.path.join(args.folder, "*.png")) +
                       glob.glob(os.path.join(args.folder, "*.jpg")) +
                       glob.glob(os.path.join(args.folder, "*.jpeg")) +
                       glob.glob(os.path.join(args.folder, "*.bmp")))
        for fp in files:
            try:
                arr = np.array(Image.open(fp).convert("RGB"))
                seq.append(_to_gray_wh(arr, W, H))
            except Exception:
                pass
        if not seq:
            seq.append(blank())
        return None, seq
    elif args.pattern == "image":
        if os.path.isfile(args.image):
            try:
                arr = np.array(Image.open(args.image).convert("RGB"))
                seq.append(_to_gray_wh(arr, W, H))
            except Exception:
                seq.append(blank())
        else:
            seq.append(blank())
        return None, seq
    elif args.pattern == "segmask":
        # Load binary (preferred) or labels/masks from NPZ and create a single grayscale frame
        fp = getattr(args, 'roi_npz', '') or os.path.join(os.getcwd(), "rois.npz")
        try:
            data = np.load(fp, allow_pickle=False)
            if 'binary' in data:
                b = data['binary'].astype(np.uint8)
                img = (b > 0).astype(np.uint8) * 255
            elif 'labels' in data:
                labels = data['labels'].astype(np.int32)
                img = (labels > 0).astype(np.uint8) * 255
            elif 'masks' in data:
                masks = data['masks']
                # Union all masks if 3D array, else try first mask
                if isinstance(masks, np.ndarray) and masks.ndim == 3 and masks.shape[0] > 0:
                    union = np.any(masks.astype(bool), axis=0)
                    img = union.astype(np.uint8) * 255
                elif isinstance(masks, list) and len(masks) > 0:
                    union = np.zeros_like(np.array(masks[0]).astype(bool))
                    for m in masks:
                        union |= np.array(m).astype(bool)
                    img = union.astype(np.uint8) * 255
                else:
                    img = blank()
            else:
                img = blank()
            # Pad to projector size without scaling if smaller
            ih, iw = img.shape[:2]
            if ih <= H and iw <= W:
                pad_top = (H - ih) // 2
                pad_bottom = H - ih - pad_top
                pad_left = (W - iw) // 2
                pad_right = W - iw - pad_left
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            else:
                img = np.array(Image.fromarray(img).resize((W, H), resample=Image.NEAREST))
            return None, [img]
        except Exception as _e:
            print(f"⚠️  segmask load failed: {_e}")
            return None, [blank()]
    elif args.pattern == "checkerboard":
        return checkerboard, None
    elif args.pattern == "solid":
        return solid, None
    elif args.pattern == "circle":
        return circle, None
    elif args.pattern == "gradient":
        return None, gradient_sequence()
    else:
        return moving_bar, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5558")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pattern", default="moving_bar",
                    choices=["moving_bar", "checkerboard", "solid", "circle", "gradient", "image", "folder", "segmask"]) 
    ap.add_argument("--speed", type=float, default=400.0)
    ap.add_argument("--bar-width", dest="bar_width", type=int, default=40)
    ap.add_argument("--value", type=int, default=255)
    ap.add_argument("--checker-size", dest="checker_size", type=int, default=64)
    ap.add_argument("--radius", type=int, default=200)
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--folder", type=str, default="")
    ap.add_argument("--gradient-steps", dest="gradient_steps", type=int, default=6)
    ap.add_argument("--gradient-hold", dest="gradient_hold", type=int, default=20)
    ap.add_argument("--gradient-gamma", dest="gradient_gamma", type=float, default=2.2)
    ap.add_argument("--prewarp-lut-dir", type=str, default="",
                    help="If set, load cam_from_proj_{x,y}.npy from this dir and prewarp frames")
    ap.add_argument("--roi-npz", type=str, default="",
                    help="Path to rois.npz containing 'labels' or 'masks'")
    ap.add_argument("--flip-x", action="store_true", help="Flip frames horizontally before send")
    ap.add_argument("--flip-y", action="store_true", help="Flip frames vertically before send")
    ap.add_argument("--save-segmask-to", type=str, default="",
                    help="If pattern=segmask: save the actually presented frame (after flips/prewarp) to this TIFF path")
    ap.add_argument("--composite-rgb", action="store_true",
                    help="Mode B: pack stim mask into R channel, observe mask into B channel, G=0. Sends H*W*3 bytes.")
    ap.add_argument("--stim-source", type=str, default="same",
                    help="Stim mask source for --composite-rgb: 'same' (duplicate main pattern), or path to image/npz")
    ap.add_argument("--temporal-alternate", action="store_true",
                    help="Mode A: alternate frames between R-only (stim) and B-only (observe). "
                         "At 60 Hz HDMI, each color gets ~16.6 ms. Uses External Pattern Streaming.")
    ap.add_argument("--obs-source", type=str, default="same",
                    help="Observe mask source for --temporal-alternate: 'same' (duplicate main pattern), or path to image/npz")
    args = ap.parse_args()

    global W, H
    # Allow W/H override via env if needed
    try:
        W = int(os.getenv("MASK_W", W))
        H = int(os.getenv("MASK_H", H))
    except Exception:
        pass

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUSH)
    s.setsockopt(zmq.SNDHWM, 4)
    s.setsockopt(zmq.IMMEDIATE, 1)
    s.setsockopt(zmq.SNDTIMEO, 0)
    s.connect(args.endpoint)

    # Optional LUT prewarp (projector expects prewarped content when H is cleared)
    inv_x = inv_y = None
    if args.prewarp_lut_dir:
        try:
            import numpy as _np
            import os as _os
            inv_x = _np.load(_os.path.join(args.prewarp_lut_dir, "cam_from_proj_x.npy")).astype(np.float32)
            inv_y = _np.load(_os.path.join(args.prewarp_lut_dir, "cam_from_proj_y.npy")).astype(np.float32)
        except Exception as _e:
            print(f"⚠️  Failed to load LUTs from {args.prewarp_lut_dir}: {_e}")
            inv_x = inv_y = None

    # iter-30: closures now delegate to module-level functions
    # (defined at top of file). Local lambdas preserved for clarity.
    def _prewarp(img_gray: np.ndarray) -> np.ndarray:
        return apply_prewarp(img_gray, inv_x, inv_y)

    def _apply_flips(img: np.ndarray) -> np.ndarray:
        return apply_flips(img, args.flip_x, args.flip_y)

    stim_mask_static = None
    if args.composite_rgb and args.stim_source != "same":
        try:
            p = args.stim_source
            if p.endswith(".npz"):
                data = np.load(p, allow_pickle=False)
                if "binary" in data:
                    stim_mask_static = (data["binary"] > 0).astype(np.uint8) * 255
                elif "labels" in data:
                    stim_mask_static = (data["labels"] > 0).astype(np.uint8) * 255
                else:
                    stim_mask_static = np.zeros((H, W), np.uint8)
            else:
                arr = np.array(Image.open(p).convert("RGB"))
                stim_mask_static = _to_gray_wh(arr, W, H)
            stim_mask_static = _to_gray_wh(stim_mask_static, W, H)
            print(f"Loaded stim mask from {p} ({stim_mask_static.shape})")
        except Exception as e:
            print(f"Failed to load stim source {args.stim_source}: {e}, falling back to 'same'")
            stim_mask_static = None

    obs_mask_static = None
    if args.temporal_alternate and args.obs_source != "same":
        try:
            p = args.obs_source
            if p.endswith(".npz"):
                data = np.load(p, allow_pickle=False)
                if "binary" in data:
                    obs_mask_static = (data["binary"] > 0).astype(np.uint8) * 255
                elif "labels" in data:
                    obs_mask_static = (data["labels"] > 0).astype(np.uint8) * 255
                else:
                    obs_mask_static = np.zeros((H, W), np.uint8)
            else:
                arr = np.array(Image.open(p).convert("RGB"))
                obs_mask_static = _to_gray_wh(arr, W, H)
            obs_mask_static = _to_gray_wh(obs_mask_static, W, H)
            print(f"Loaded observe mask from {p} ({obs_mask_static.shape})")
        except Exception as e:
            print(f"Failed to load obs source {args.obs_source}: {e}, falling back to 'same'")
            obs_mask_static = None

    saved_presented_once = False

    # iter-30: pack_* now at module level (see top of file).
    # Local thin wrappers preserved for the existing call sites below.
    def _pack_r_only(gray):
        return pack_r_only(gray)

    def _pack_b_only(gray):
        return pack_b_only(gray)

    def _pack_composite_rgb(observe_gray, stim_gray):
        return pack_composite_rgb(observe_gray, stim_gray)

    def send_mask(mid, img):
        meta = json.dumps({"id": int(mid)}).encode()
        try:
            img2 = _apply_flips(img)
            frame = _prewarp(img2)
            nonlocal saved_presented_once
            if (not saved_presented_once) and args.pattern == "segmask":
                try:
                    out_path = args.save_segmask_to.strip() or os.path.join(os.getcwd(), "segmask_presented.tiff")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    Image.fromarray(frame.astype(np.uint8)).save(out_path, format="TIFF")
                    print(f"Saved presented segmask to: {out_path}")
                except Exception as _e:
                    print(f"Failed saving presented segmask: {_e}")
                finally:
                    saved_presented_once = True
            if args.temporal_alternate:
                is_stim_frame = (mid % 2) == 1
                if is_stim_frame:
                    rgb_frame = _pack_r_only(frame)
                else:
                    obs = obs_mask_static if obs_mask_static is not None else frame
                    rgb_frame = _pack_b_only(obs)
                payload = rgb_frame.tobytes()
            elif args.composite_rgb:
                stim = stim_mask_static if stim_mask_static is not None else frame
                rgb_frame = _pack_composite_rgb(frame, stim)
                payload = rgb_frame.tobytes()
            else:
                payload = frame.tobytes()
            s.send_multipart([meta, payload], flags=zmq.DONTWAIT)
            return True
        except zmq.Again:
            return False

    gen_fn, seq = build_patterns(args)

    if args.temporal_alternate:
        obs_desc = args.obs_source if obs_mask_static is not None else "same as stim"
        print(f"Mode A temporal-alternate: odd frames=R(stim), even frames=B(observe={obs_desc}). "
              f"At {args.fps} Hz → {args.fps/2:.1f} Hz per color. Sending {H}x{W}x3 = {H*W*3} bytes/frame")
    elif args.composite_rgb:
        stim_desc = args.stim_source if stim_mask_static is not None else "same as observe"
        print(f"Mode B composite-RGB: R=stim({stim_desc}), G=0, B=observe. Sending {H}x{W}x3 = {H*W*3} bytes/frame")
    print("Streaming; Ctrl-C to stop")
    t0 = time.perf_counter()
    next_t = t0
    mid = 0
    INTERVAL = 1.0 / max(1e-6, float(args.fps))

    import csv
    csv_path = os.path.join(os.getcwd(), "sent_masks.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["mask_id", "timestamp", "status"])
        prev_t = t0
        try:
            idx = 0
            while True:
                if gen_fn is not None:
                    t = time.perf_counter() - t0
                    img = gen_fn(t)
                else:
                    if not seq:
                        img = np.zeros((H, W), np.uint8)
                    else:
                        img = seq[idx % len(seq)]
                        idx += 1

                mid += 1
                timestamp = time.perf_counter()
                ok = send_mask(mid, img)
                csv_writer.writerow([mid, timestamp, ("sent" if ok else "dropped")])
                csv_file.flush()

                dt_ms = (timestamp - prev_t) * 1000 if mid > 1 else 0.0
                prev_t = timestamp
                if args.temporal_alternate:
                    color = "RED " if (mid % 2) == 1 else "BLUE"
                    status = "sent" if ok else "DROP"
                    print(f"#{mid:5d} {color} {status}  dt={dt_ms:6.2f}ms", flush=True)
                elif mid % 60 == 0:
                    print(f"#{mid} sent={ok} dt={dt_ms:.2f}ms", flush=True)

                next_t += INTERVAL
                current_t = time.perf_counter()
                sleep_s = next_t - current_t
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif sleep_s < -INTERVAL:
                    drift_frames = int(-sleep_s / INTERVAL)
                    print(f"WARNING: {drift_frames} frames behind at mask {mid}")
                    next_t = current_t
        except KeyboardInterrupt:
            print(f"\nStopped by user. Sent masks log saved to: {csv_path}")
        finally:
            s.close()
            ctx.term()

if __name__ == "__main__":
    main()

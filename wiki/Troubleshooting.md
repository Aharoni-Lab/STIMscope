# Troubleshooting

Common problems, sorted by symptom.

## X11 / GUI won't open

### `Could not connect to display`, `X Error of failed request`, or the GUI silently fails to launch

```bash
export DISPLAY=:0
xhost +local:docker
sudo -E docker-compose up gui    # -E preserves DISPLAY through sudo
```

The `xhost +local:docker` must be re-run once per shell session.
The `-E` flag is what passes `DISPLAY` through `sudo`.

### `Authorization required, but no authorization protocol specified` (GDM 3.x)

GDM stores its X auth cookie at
`/run/user/<UID>/gdm/Xauthority`, not `~/.Xauthority`. `make fresh`
handles this automatically; if you're launching with raw
`docker run` instead:

```bash
DISPLAY=:0 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority \
  xhost +SI:localuser:root
cp /run/user/$(id -u)/gdm/Xauthority /tmp/docker.xauth
chmod 644 /tmp/docker.xauth
# then mount /tmp/docker.xauth into the container as /tmp/docker.xauth
# and set XAUTHORITY=/tmp/docker.xauth in the container's env.
```

## GPU not detected

```bash
sudo docker run --rm --runtime=nvidia \
  nvcr.io/nvidia/l4t-jetpack:r36.2.0 nvidia-smi
```

If this fails, the NVIDIA container toolkit isn't installed
correctly. Re-run [Install steps 2 + 3](Install#prerequisites).

If `nvidia-smi` works in the base image but CRISPI doesn't see
the GPU, check that `runtime: nvidia` is still in
`docker-compose.yml` (any `version:` downgrade can drop it).

## IDS Peak camera not detected

1. Verify the SDK is installed on the host:
   ```bash
   ls /opt/ids-peak/lib/
   # Should show arm64 .so files
   ```

2. If your SDK is elsewhere, point at it:
   ```bash
   export IDS_PEAK_PATH=/your/path
   # then sudo -E docker-compose up gui
   ```

3. Check USB:
   ```bash
   lsusb | grep IDS
   ```

   If nothing shows, the camera isn't enumerating. Try a different
   USB3 port (some hub-isolated ports on Jetson are unreliable),
   and confirm the red+green LEDs on the camera body are lit.

4. If `lsusb` shows the device but the GUI dropdown is empty, the
   Python bindings probably didn't install on first run. Re-launch
   with logs visible:
   ```bash
   sudo -E docker-compose up gui 2>&1 | grep -iE "ids_peak|peak"
   ```

## Camera was working but stopped after disconnect/reconnect

Common — USB renumeration plus the GenICam transport-layer cache
sometimes hold stale device handles. Stop and restart:

```bash
make fresh
```

`make fresh` is the canonical "I'm having a bad time" restart —
it brings the GUI container fully down and back up rather than
restarting in place, which fixes most stuck-handle issues.

## Build failed

### `COPY failed: ids-peak_*.deb: no such file or directory`

You ran `docker build` directly instead of `./build.sh`. Two fixes:

- Re-run via `./build.sh` (which creates a 0-byte stub if the
  `.deb` is missing, so hardware-free builds succeed)
- Or download the real `.deb` (see [Install step 4](Install#prerequisites))
  and place it at the repo root.

### CuPy install fails

`build.sh` picks `cupy-cuda11x` for JP5 and `cupy-cuda12x` for
JP6. If you're building outside `build.sh`, the `CUPY_PACKAGE`
build-arg must match your JetPack's CUDA version — see
[Install / Build for a specific JetPack version](Install#build-for-a-specific-jetpack-version).

### Build hangs at "Installing collected packages"

Sometimes the IDS Peak Python bindings (`ids_peak`, `ids_peak_ipl`,
`ids_peak_afl`) take 10+ minutes to install on the first run. They
build C extensions from the SDK headers. Subsequent rebuilds are
cached.

## Tests fail

Smoke-check the image's core imports:

```bash
docker run --rm --entrypoint python3 crispi:latest -c \
  "import sys; sys.path.insert(0, '/app/STIMViewer_CRISPI/CS'); \
   from core import projector, structured_light, paths, logging_config; \
   print('core imports OK')"
```

If this prints `core imports OK`, the platform's core modules are available; missing GPU / camera / GPIO are runtime-optional and fall back.

For test-level details, see the
[`docs/IMPLEMENTATION_NOTES.md` test-layer table](https://github.com/Aharoni-Lab/STIMscope/blob/main/docs/IMPLEMENTATION_NOTES.md).

## Logs

`make logs-tail` starts a background tail of the GUI container log,
written to `/tmp/crispi-<TS>.log` with a symlink at
`/tmp/crispi-latest.log`. Useful summary commands:

```bash
make logs           # follow GUI logs (foreground)
make logs-tail      # background capture
make logs-summary   # grep the latest capture for milestones
make logs-stop-tail # kill the background tail
```

## Data files end up root-owned

The container runs as root. Reclaim ownership:

```bash
sudo chown -R $(id -u):$(id -g) data/
```

## Filing a bug

Use the [bug-report issue
template](https://github.com/Aharoni-Lab/STIMscope/issues/new?template=bug_report.yml)
— it collects the layer, JetPack version, Jetson model, commit SHA,
and hardware mode without requiring you to remember which fields are
needed.

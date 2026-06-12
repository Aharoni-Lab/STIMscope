# Docker Image

The current distribution path is **build from source** — the Dockerfile
at the repo root, driven by `./build.sh`, produces an image tagged
`crispi:latest` on the host. A pre-built published image is not
currently available.

## Build from source

```bash
git clone https://github.com/Aharoni-Lab/STIMscope.git
cd STIMscope
./build.sh                   # auto-detects JetPack 5 vs 6
sudo -E docker-compose up gui
```

The full prerequisite walkthrough (NVIDIA Container Toolkit, IDS Peak
SDK download path, JetPack-specific build args) is on the
[Install](Install) page.

## Verifying what a local image was built from

Every image bakes its build provenance into `/app/build_info.txt`.
To confirm which commit an image came from:

```bash
docker run --rm --entrypoint cat <image> /app/build_info.txt
```

It reports `git_sha`, `build_date`, the JetPack base, CUDA / CuPy
package, and the projector binary's `sha256`. To verify the baked
source actually matches that commit (rather than trusting the SHA
field alone), checksum a file inside the image against the same path
in git:

```bash
docker run --rm --entrypoint sha256sum <image> \
  /app/STIMViewer_CRISPI/camera.py
git show <git_sha>:STIMscope/STIMViewer_CRISPI/camera.py | sha256sum
```

A discriminating match — pick a file that *differs* between the
candidate commits — is the tamper-evident check.

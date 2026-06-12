#!/bin/bash
# Launch the local STIMViewer GUI from this repo, using the active Python env if available
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer conda env python if CONDA_PREFIX set; else fall back to python3 in PATH; then /usr/bin/python3
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PY="$CONDA_PREFIX/bin/python"
else
  PY="$(command -v python3 || true)"
  if [ -z "$PY" ]; then PY="/usr/bin/python3"; fi
fi

exec "$PY" main_gui.pyw
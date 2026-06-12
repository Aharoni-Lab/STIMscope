"""CSV frame logger for the base-platform demo recording (pure stdlib,
no inference-module coupling).

Writes every projection event + camera-snapshot event with monotonic-ns
timestamp + mask metadata to a CSV the verifier/analysis consume.

Header:
  ts_ns,wall_iso,event,segment,mask_name,mask_color,mask_sha256,frame_id,hw_ts_ns,extra

event types:
  - "segment_start" / "segment_end" — segment markers
  - "projection_send" — mask sent to the projector via ProjectorClient
  - "camera_meta" — captured camera frame_id (host ts_ns; when slave-triggered,
                    the IDS buffer's hardware timestamp in hw_ts_ns proves lock)
  - "metric" — computed value (fps, drop_count, etc.)

Append-only, line-buffered so a Ctrl-C never loses prior log lines.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional


class DemoLogger:
    """Thread-safe append-only CSV logger using monotonic_ns timestamps."""

    HEADER = [
        "ts_ns", "wall_iso", "event", "segment", "mask_name",
        "mask_color", "mask_sha256", "frame_id", "hw_ts_ns", "extra",
    ]

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._segment: str = "init"
        new_file = not path.exists()
        self._fh = open(path, "a", buffering=1, newline="")
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow(self.HEADER)
            self._fh.flush()

    def set_segment(self, name: str) -> None:
        with self._lock:
            self._segment = name

    def segment_start(self, name: str, intent: str = "") -> None:
        self.set_segment(name)
        self._row("segment_start", "", "", "", "", intent)

    def segment_end(self, name: str) -> None:
        self._row("segment_end", "", "", "", "", "")

    def projection_send(
        self,
        mask_name: str,
        mask_color: str,
        mask_sha256: str,
        frame_id: Optional[int] = None,
        extra: str = "",
    ) -> None:
        self._row(
            "projection_send",
            mask_name=mask_name,
            mask_color=mask_color,
            mask_sha256=mask_sha256,
            frame_id="" if frame_id is None else str(frame_id),
            extra=extra,
        )

    def camera_meta(
        self,
        frame_id: int,
        hw_ts_ns: Optional[int] = None,
        extra: str = "",
    ) -> None:
        self._row(
            "camera_meta",
            mask_name="",
            mask_color="",
            mask_sha256="",
            frame_id=str(frame_id),
            hw_ts_ns="" if hw_ts_ns is None else str(hw_ts_ns),
            extra=extra,
        )

    def metric(self, name: str, value: str) -> None:
        self._row(
            "metric",
            mask_name=name,
            mask_color="",
            mask_sha256="",
            frame_id="",
            extra=value,
        )

    def _row(
        self,
        event: str,
        mask_name: str = "",
        mask_color: str = "",
        mask_sha256: str = "",
        frame_id: str = "",
        hw_ts_ns: str = "",
        extra: str = "",
    ) -> None:
        ts_ns = time.monotonic_ns()
        wall_iso = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._lock:
            self._writer.writerow([
                ts_ns, wall_iso, event, self._segment,
                mask_name, mask_color, mask_sha256, frame_id, hw_ts_ns, extra,
            ])
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
            finally:
                self._fh.close()

    def __enter__(self) -> "DemoLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["DemoLogger"]

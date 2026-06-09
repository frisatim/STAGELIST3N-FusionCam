"""Live metadata export for Phase 3.

The publisher keeps video transport separate from AI metadata. It can write a
JSONL trace for offline checks and/or POST live envelopes to a local dashboard.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def _round_list(values: Any, digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def detection_to_metadata(det: Any, zones: list[str] | None = None) -> dict[str, Any]:
    position_m = _round_list(det.foot_point_m, 3) if det.foot_point_m else None
    return {
        "camera_id": det.cam_id,
        "track_id": int(det.track_id),
        "global_id": det.global_id,
        "class_id": int(det.class_id),
        "class_name": det.class_name,
        "confidence": round(float(det.confidence), 4),
        "bbox_px": _round_list(det.bbox_px, 2),
        "foot_point_px": _round_list(det.foot_point_px, 2),
        "position_m": position_m,
        "zones": zones or [],
        "timestamp": round(float(det.timestamp), 6),
    }


def alert_to_metadata(alert: Any) -> dict[str, Any]:
    return {
        "alert_id": alert.alert_id,
        "alert_type": alert.alert_type,
        "alert_level": getattr(alert, "alert_level", "confirmed"),
        "global_id": alert.global_id,
        "zone_id": alert.zone_id,
        "class_name": alert.class_name,
        "position_m": _round_list(alert.position_m, 3),
        "cameras": list(alert.cameras),
        "confidence": round(float(alert.confidence), 4),
        "timestamp": round(float(alert.timestamp), 6),
    }


class MetadataPublisher:
    """Publish Phase 3 frame metadata without blocking the AI loop."""

    def __init__(
        self,
        jsonl_path: Path | None = None,
        http_url: str | None = None,
        every_n_frames: int = 1,
        http_timeout_s: float = 0.05,
        queue_size: int = 200,
    ) -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.http_url = http_url
        self.every_n_frames = max(1, int(every_n_frames))
        self.http_timeout_s = max(0.001, float(http_timeout_s))
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._sent = 0
        self._thread: threading.Thread | None = None

        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.http_url:
            self._thread = threading.Thread(target=self._http_worker, daemon=True)
            self._thread.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def sent(self) -> int:
        return self._sent

    def publish(
        self,
        frame_idx: int,
        run_label: str,
        model_version: str,
        model_name: str,
        format_label: str,
        detections: list[dict[str, Any]],
        alerts: list[dict[str, Any]],
    ) -> None:
        if frame_idx % self.every_n_frames != 0:
            return

        now_ms = time.time() * 1000.0
        envelope = {
            "schema": "benchmarkingai.phase3.metadata.v1",
            "created_epoch_ms": now_ms,
            "created_epoch_s": round(now_ms / 1000.0, 6),
            "frame": int(frame_idx),
            "run_label": run_label,
            "model_version": model_version,
            "model": model_name,
            "format": format_label,
            "detections": detections,
            "alerts": alerts,
        }

        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n")

        if self.http_url:
            try:
                self._queue.put_nowait(envelope)
            except queue.Full:
                self._dropped += 1

    def close(self) -> None:
        if self._thread:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=1.0)

    def _http_worker(self) -> None:
        while True:
            envelope = self._queue.get()
            if envelope is None:
                return
            data = json.dumps(envelope).encode("utf-8")
            request = urllib.request.Request(
                self.http_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.http_timeout_s):
                    self._sent += 1
            except Exception:
                self._dropped += 1

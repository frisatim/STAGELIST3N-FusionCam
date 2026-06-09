from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ENVELOPE_FIELDS = {
    "schema",
    "created_epoch_ms",
    "created_epoch_s",
    "frame",
    "run_label",
    "model_version",
    "model",
    "format",
    "detections",
    "alerts",
}

DETECTION_FIELDS = {
    "camera_id",
    "track_id",
    "global_id",
    "class_id",
    "class_name",
    "confidence",
    "bbox_px",
    "foot_point_px",
    "position_m",
    "zones",
    "timestamp",
}

ALERT_FIELDS = {
    "alert_id",
    "alert_type",
    "alert_level",
    "global_id",
    "zone_id",
    "class_name",
    "position_m",
    "cameras",
    "confidence",
    "timestamp",
}


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[ERREUR] JSON invalide ligne {line_no}: {exc}") from exc
    return rows


def analyze(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    missing_envelope = 0
    missing_detection = 0
    missing_alert = 0
    detections = 0
    alerts = 0
    cameras: set[str] = set()
    classes: set[str] = set()
    alert_types: set[str] = set()
    global_ids = 0
    bbox_ok = 0
    frame_backwards = 0
    intervals_ms: list[float] = []
    metadata_lags_ms: list[float] = []
    previous_frame: int | None = None
    previous_created_ms: float | None = None
    example_with_detection: dict[str, Any] | None = None
    example_with_alert: dict[str, Any] | None = None

    for row in rows:
        if ENVELOPE_FIELDS - row.keys():
            missing_envelope += 1

        frame = row.get("frame")
        if isinstance(frame, int):
            if previous_frame is not None and frame < previous_frame:
                frame_backwards += 1
            previous_frame = frame

        created_ms = row.get("created_epoch_ms")
        if isinstance(created_ms, (int, float)):
            if previous_created_ms is not None:
                intervals_ms.append(float(created_ms) - previous_created_ms)
            previous_created_ms = float(created_ms)

        newest_source_ts = 0.0
        for det in row.get("detections", []) or []:
            detections += 1
            if DETECTION_FIELDS - det.keys():
                missing_detection += 1
            camera_id = det.get("camera_id")
            if camera_id:
                cameras.add(str(camera_id))
            class_name = det.get("class_name")
            if class_name:
                classes.add(str(class_name))
            if det.get("global_id") is not None:
                global_ids += 1
            bbox = det.get("bbox_px")
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox_ok += 1
            ts = det.get("timestamp")
            if isinstance(ts, (int, float)):
                newest_source_ts = max(newest_source_ts, float(ts))
            if example_with_detection is None:
                example_with_detection = row

        for alert in row.get("alerts", []) or []:
            alerts += 1
            if ALERT_FIELDS - alert.keys():
                missing_alert += 1
            alert_type = alert.get("alert_type")
            if alert_type:
                alert_types.add(str(alert_type))
            for camera_id in alert.get("cameras", []) or []:
                cameras.add(str(camera_id))
            ts = alert.get("timestamp")
            if isinstance(ts, (int, float)):
                newest_source_ts = max(newest_source_ts, float(ts))
            if example_with_alert is None:
                example_with_alert = row

        created_s = row.get("created_epoch_s")
        if newest_source_ts and isinstance(created_s, (int, float)):
            metadata_lags_ms.append((float(created_s) - newest_source_ts) * 1000.0)

    return {
        "path": str(path),
        "envelopes": len(rows),
        "detections": detections,
        "alerts": alerts,
        "cameras": sorted(cameras),
        "classes": sorted(classes),
        "alert_types": sorted(alert_types),
        "bbox_ok": bbox_ok,
        "global_ids": global_ids,
        "missing_envelope": missing_envelope,
        "missing_detection": missing_detection,
        "missing_alert": missing_alert,
        "frame_backwards": frame_backwards,
        "publish_interval_mean_ms": round(_mean(intervals_ms), 3),
        "publish_interval_p95_ms": round(_pct(intervals_ms, 0.95), 3),
        "metadata_lag_mean_ms": round(_mean(metadata_lags_ms), 3),
        "metadata_lag_p95_ms": round(_pct(metadata_lags_ms, 0.95), 3),
        "example_with_detection": example_with_detection,
        "example_with_alert": example_with_alert,
    }


def print_summary(result: dict[str, Any], print_example: bool = False) -> None:
    print(f"[INFO] JSONL: {result['path']}")
    print(f"[INFO] Enveloppes: {result['envelopes']}")
    print(f"[INFO] Detections: {result['detections']} | bbox OK: {result['bbox_ok']} | global_id presents: {result['global_ids']}")
    print(f"[INFO] Alertes: {result['alerts']}")
    print(f"[INFO] Cameras: {', '.join(result['cameras']) or '-'}")
    print(f"[INFO] Classes: {', '.join(result['classes']) or '-'}")
    print(f"[INFO] Types alertes: {', '.join(result['alert_types']) or '-'}")
    print(
        "[INFO] Intervalle publication: "
        f"mean={result['publish_interval_mean_ms']}ms "
        f"p95={result['publish_interval_p95_ms']}ms"
    )
    print(
        "[INFO] Lag metadata estime: "
        f"mean={result['metadata_lag_mean_ms']}ms "
        f"p95={result['metadata_lag_p95_ms']}ms"
    )
    problems = {
        "enveloppes_incompletes": result["missing_envelope"],
        "detections_incompletes": result["missing_detection"],
        "alertes_incompletes": result["missing_alert"],
        "frames_en_arriere": result["frame_backwards"],
    }
    if any(problems.values()):
        print(f"[WARN] Problemes detectes: {problems}")
    else:
        print("[INFO] Structure JSONL OK")

    if print_example:
        example = result["example_with_alert"] or result["example_with_detection"]
        if example:
            print("[INFO] Exemple:")
            print(json.dumps(example, indent=2, ensure_ascii=False))
        else:
            print("[INFO] Aucun exemple avec detection/alerte trouve.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 3 live metadata JSONL.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--print-example", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.jsonl.exists():
        raise SystemExit(f"[ERREUR] Fichier introuvable: {args.jsonl}")
    print_summary(analyze(args.jsonl), print_example=args.print_example)


if __name__ == "__main__":
    main()

"""
Audit calibration quality from Phase 3 alert logs.

The script parses pipeline console logs containing lines like:
  [ALERT] Alert[abcd1234] zone_violation_person zone=zone_2 gid=1 pos=(5.23m,13.21m) cams=cam_04+cam_01 conf=0.93

It checks each alert position against the forbidden-zone polygons from config.yaml,
adds nearest-zone distance and per-camera calibration error metadata, and writes a CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


ALERT_RE = re.compile(
    r"Alert\[(?P<alert_id>[^\]]+)\]\s+"
    r"(?P<alert_type>\w+)"
    r"(?::(?P<alert_level>\w+))?"
    r"(?:\s+zone=(?P<declared_zone>\w+))?\s+"
    r"gid=(?P<gid>-?\d+)\s+"
    r"pos=\((?P<x>-?\d+(?:\.\d+)?)m,(?P<y>-?\d+(?:\.\d+)?)m\)\s+"
    r"cams=(?P<cams>[^\s]+)\s+"
    r"conf=(?P<conf>-?\d+(?:\.\d+)?)"
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_zones(config: dict[str, Any]) -> dict[str, np.ndarray]:
    zones: dict[str, np.ndarray] = {}
    for zone_id, zone_data in config.get("zones_interdites", {}).items():
        coords = zone_data.get("coordonnees_metres") or []
        if len(coords) < 3:
            continue
        zones[zone_id] = np.array(coords, dtype=np.float32).reshape((-1, 1, 2))
    return zones


def calibration_errors(config: dict[str, Any]) -> dict[str, tuple[float | None, float | None]]:
    errors: dict[str, tuple[float | None, float | None]] = {}
    for cam_id, data in config.get("homographie", {}).items():
        if not isinstance(data, dict):
            continue
        mean_err = data.get("mean_error_cm")
        max_err = data.get("max_error_cm")
        errors[cam_id] = (
            float(mean_err) if mean_err is not None else None,
            float(max_err) if max_err is not None else None,
        )
    return errors


def point_in_zone(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0


def distance_point_to_segment(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def distance_to_polygon(point: tuple[float, float], polygon: np.ndarray) -> float:
    pts = polygon.reshape((-1, 2)).astype(float)
    if point_in_zone(point, polygon):
        return 0.0
    distances = []
    for i in range(len(pts)):
        a = (float(pts[i][0]), float(pts[i][1]))
        b = (float(pts[(i + 1) % len(pts)][0]), float(pts[(i + 1) % len(pts)][1]))
        distances.append(distance_point_to_segment(point, a, b))
    return min(distances) if distances else float("nan")


def parse_alerts(log_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(log_text.splitlines(), start=1):
        match = ALERT_RE.search(line)
        if not match:
            continue
        cams = [cam for cam in match.group("cams").split("+") if cam]
        rows.append(
            {
                "line_no": line_no,
                "raw_line": line.strip(),
                "alert_id": match.group("alert_id"),
                "alert_type": match.group("alert_type"),
                "alert_level": match.group("alert_level") or "confirmed",
                "declared_zone": match.group("declared_zone") or "",
                "global_id": int(match.group("gid")),
                "x_m": float(match.group("x")),
                "y_m": float(match.group("y")),
                "cameras": cams,
                "confidence": float(match.group("conf")),
            }
        )
    return rows


def audit_rows(
    alerts: list[dict[str, Any]],
    zones: dict[str, np.ndarray],
    cam_errors: dict[str, tuple[float | None, float | None]],
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for alert in alerts:
        point = (alert["x_m"], alert["y_m"])
        inside = [zone_id for zone_id, poly in zones.items() if point_in_zone(point, poly)]
        distances = {
            zone_id: distance_to_polygon(point, poly)
            for zone_id, poly in zones.items()
        }
        nearest_zone = min(distances, key=distances.get) if distances else ""
        nearest_distance = distances.get(nearest_zone, float("nan")) if nearest_zone else float("nan")

        declared_zone = alert["declared_zone"]
        if alert["alert_type"] == "zone_violation_person":
            expected_ok = bool(declared_zone and declared_zone in inside)
        else:
            expected_ok = True

        camera_mean_errors = []
        camera_max_errors = []
        for cam_id in alert["cameras"]:
            mean_err, max_err = cam_errors.get(cam_id, (None, None))
            if mean_err is not None:
                camera_mean_errors.append(mean_err)
            if max_err is not None:
                camera_max_errors.append(max_err)

        audited.append(
            {
                **alert,
                "cameras": "+".join(alert["cameras"]),
                "inside_zones": "+".join(inside),
                "inside_any_zone": bool(inside),
                "expected_zone_ok": expected_ok,
                "nearest_zone": nearest_zone,
                "distance_to_nearest_zone_m": round(nearest_distance, 4),
                "mean_calibration_error_cm_max_camera": (
                    round(max(camera_mean_errors), 2) if camera_mean_errors else ""
                ),
                "max_calibration_error_cm_max_camera": (
                    round(max(camera_max_errors), 2) if camera_max_errors else ""
                ),
            }
        )
    return audited


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "line_no",
        "alert_id",
        "alert_type",
        "alert_level",
        "global_id",
        "declared_zone",
        "x_m",
        "y_m",
        "cameras",
        "confidence",
        "inside_any_zone",
        "inside_zones",
        "expected_zone_ok",
        "nearest_zone",
        "distance_to_nearest_zone_m",
        "mean_calibration_error_cm_max_camera",
        "max_calibration_error_cm_max_camera",
        "raw_line",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows: list[dict[str, Any]],
    cam_errors: dict[str, tuple[float | None, float | None]],
) -> None:
    print(f"[INFO] Alertes analysées : {len(rows)}")
    if not rows:
        return

    by_type = Counter(row["alert_type"] for row in rows)
    print(f"[INFO] Par type : {dict(by_type)}")

    bad_zone = [
        row for row in rows
        if row["alert_type"] == "zone_violation_person" and not row["expected_zone_ok"]
    ]
    print(f"[INFO] Violations personne hors zone déclarée : {len(bad_zone)}")

    by_cam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for cam_id in str(row["cameras"]).split("+"):
            if cam_id:
                by_cam[cam_id].append(row)

    print("[INFO] Résumé par caméra :")
    for cam_id in sorted(by_cam):
        cam_rows = by_cam[cam_id]
        mean_err, max_err = cam_errors.get(cam_id, ("", ""))
        outside = sum(1 for row in cam_rows if not row["inside_any_zone"])
        print(
            f"  {cam_id}: alerts={len(cam_rows)} outside_any_zone={outside} "
            f"mean_error_cm={mean_err if mean_err is not None else ''} "
            f"max_error_cm={max_err if max_err is not None else ''}"
        )

    if bad_zone:
        print("[WARN] Exemples suspects :")
        for row in bad_zone[:10]:
            print(
                f"  line={row['line_no']} {row['alert_type']} gid={row['global_id']} "
                f"pos=({row['x_m']:.2f},{row['y_m']:.2f}) declared={row['declared_zone']} "
                f"inside={row['inside_zones'] or '-'} nearest={row['nearest_zone']} "
                f"dist={row['distance_to_nearest_zone_m']}m cams={row['cameras']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 3 alert positions against forbidden-zone polygons."
    )
    parser.add_argument("--log", required=True, help="Text file containing pipeline logs.")
    parser.add_argument("--config", default="config.yaml", help="Phase 3 config.yaml path.")
    parser.add_argument(
        "--out",
        default="reports/calibration_alert_audit.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    zones = load_zones(config)
    cam_errors = calibration_errors(config)
    log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")

    alerts = parse_alerts(log_text)
    rows = audit_rows(alerts, zones, cam_errors)
    write_csv(rows, Path(args.out))
    print_summary(rows, cam_errors)
    print(f"[INFO] CSV écrit : {args.out}")


if __name__ == "__main__":
    main()

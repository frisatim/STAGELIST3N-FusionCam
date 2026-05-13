"""
Ablation du seuil D pour l'association inter-cameras Phase 3.

Le plan de recherche demande de tester D in {50, 100, 150, 200} cm et de
mesurer les fausses correspondances et correspondances manquees. Ce script
fait exactement cette evaluation, sans avoir besoin des cameras live.

Deux modes:
  1. Synthetic: jeu de detections controle, utile tout de suite.
  2. CSV: detections exportees plus tard depuis les videos, avec truth_id.

Format CSV attendu:
  frame,timestamp,cam_id,track_id,x_m,y_m,truth_id,confidence,class_name

Exemples:
  python ablate_fusion_threshold.py
  python ablate_fusion_threshold.py --input detections_fusion_gt.csv --output reports/fusion_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from detection import Detection
from fusion import MultiCameraFusion


DEFAULT_THRESHOLDS_CM = [50, 100, 150, 200]
DEFAULT_OUTPUT = Path("reports/fusion_threshold_ablation.csv")


def _make_detection(row: dict[str, Any]) -> Detection:
    x_m = float(row["x_m"])
    y_m = float(row["y_m"])
    return Detection(
        cam_id=str(row["cam_id"]),
        track_id=int(row["track_id"]),
        global_id=None,
        timestamp=float(row["timestamp"]),
        bbox_px=(0.0, 0.0, 0.0, 0.0),
        foot_point_px=(0.0, 0.0),
        foot_point_m=(x_m, y_m),
        confidence=float(row.get("confidence", 1.0)),
        class_id=0,
        class_name=str(row.get("class_name", "person")),
    )


def _synthetic_rows() -> list[dict[str, Any]]:
    """Small labelled dataset that makes the D trade-off visible.

    - person_A is seen by cam_03/cam_05/cam_07 with 0.42m and 0.85m gaps.
    - person_B is seen by cam_03/cam_07 with a 1.35m gap.
    - person_C and person_D are different people but close enough to become
      false matches when D is too permissive.
    """
    rows: list[dict[str, Any]] = []

    def add(frame: int, cam_id: str, track_id: int, x: float, y: float, truth_id: str):
        rows.append(
            {
                "frame": frame,
                "timestamp": frame / 25.0,
                "cam_id": cam_id,
                "track_id": track_id,
                "x_m": x,
                "y_m": y,
                "truth_id": truth_id,
                "confidence": 0.9,
                "class_name": "person",
            }
        )

    for frame in range(10):
        drift = frame * 0.02
        add(frame, "cam_03", 1, 5.00 + drift, 2.00, "person_A")
        add(frame, "cam_05", 8, 5.42 + drift, 2.03, "person_A")
        add(frame, "cam_07", 4, 5.85 + drift, 2.02, "person_A")

        add(frame, "cam_03", 2, 8.00, 1.00 + drift, "person_B")
        add(frame, "cam_07", 5, 9.35, 1.05 + drift, "person_B")

        add(frame, "cam_05", 9, 2.00, 5.00 + drift, "person_C")
        add(frame, "cam_07", 6, 2.95, 5.05 + drift, "person_D")

    return rows


def _load_rows(input_path: Path | None) -> list[dict[str, Any]]:
    if input_path is None:
        return _synthetic_rows()

    with open(input_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"frame", "timestamp", "cam_id", "track_id", "x_m", "y_m", "truth_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV invalide, colonnes manquantes: {sorted(missing)}")
        return list(reader)


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _group_rows_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame"])].append(row)
    return dict(sorted(grouped.items()))


def _rows_to_detections_by_cam(rows: list[dict[str, Any]]) -> tuple[dict[str, list[Detection]], dict[tuple[str, int], str]]:
    detections_by_cam: dict[str, list[Detection]] = defaultdict(list)
    truth_by_key: dict[tuple[str, int], str] = {}

    for row in rows:
        det = _make_detection(row)
        detections_by_cam[det.cam_id].append(det)
        truth_by_key[(det.cam_id, det.track_id)] = str(row["truth_id"])

    return dict(detections_by_cam), truth_by_key


def _score_frame(detections: list[Detection], truth_by_key: dict[tuple[str, int], str]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "predicted_links": 0, "truth_links": 0}

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            if a.cam_id == b.cam_id:
                continue

            truth_a = truth_by_key[(a.cam_id, a.track_id)]
            truth_b = truth_by_key[(b.cam_id, b.track_id)]
            truth_link = truth_a != "" and truth_a == truth_b
            pred_link = (
                a.global_id is not None
                and b.global_id is not None
                and a.global_id == b.global_id
            )

            if pred_link:
                counts["predicted_links"] += 1
            if truth_link:
                counts["truth_links"] += 1

            if pred_link and truth_link:
                counts["tp"] += 1
            elif pred_link and not truth_link:
                counts["fp"] += 1
            elif not pred_link and truth_link:
                counts["fn"] += 1
            else:
                counts["tn"] += 1

    return counts


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _summarize(threshold_cm: int, counts: dict[str, int], frames: int, global_ids: set[int]) -> dict[str, Any]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold_cm": threshold_cm,
        "threshold_m": threshold_cm / 100.0,
        "frames": frames,
        "tp": tp,
        "fp_false_matches": fp,
        "fn_missed_matches": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_match_rate": round(_safe_div(fp, fp + tn), 4),
        "missed_match_rate": round(_safe_div(fn, tp + fn), 4),
        "predicted_links": counts["predicted_links"],
        "truth_links": counts["truth_links"],
        "unique_global_ids": len(global_ids),
    }


def run_ablation(
    rows: list[dict[str, Any]],
    config: dict,
    thresholds_cm: list[int],
    time_window_s: float,
) -> list[dict[str, Any]]:
    grouped = _group_rows_by_frame(rows)
    summaries: list[dict[str, Any]] = []

    for threshold_cm in thresholds_cm:
        fusion = MultiCameraFusion(
            config=config,
            distance_threshold_m=threshold_cm / 100.0,
            time_window_s=time_window_s,
        )
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "predicted_links": 0, "truth_links": 0}
        global_ids: set[int] = set()

        for frame_rows in grouped.values():
            detections_by_cam, truth_by_key = _rows_to_detections_by_cam(frame_rows)
            fused = fusion.associate(detections_by_cam)
            for det in fused:
                if det.global_id is not None:
                    global_ids.add(det.global_id)
            frame_counts = _score_frame(fused, truth_by_key)
            for key, value in frame_counts.items():
                counts[key] += value

        summaries.append(_summarize(threshold_cm, counts, len(grouped), global_ids))

    return summaries


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_thresholds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation du seuil D pour la fusion multi-cameras Phase 3."
    )
    parser.add_argument("--input", type=Path, default=None, help="CSV de detections avec truth_id.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds-cm", default="50,100,150,200")
    parser.add_argument("--time-window-ms", type=float, default=500.0)
    args = parser.parse_args()

    rows = _load_rows(args.input)
    config = _load_config(args.config)
    summaries = run_ablation(
        rows=rows,
        config=config,
        thresholds_cm=_parse_thresholds(args.thresholds_cm),
        time_window_s=args.time_window_ms / 1000.0,
    )
    _write_csv(summaries, args.output)

    print(f"[INFO] Ablation terminee: {args.output}")
    for row in summaries:
        print(
            "D={threshold_cm:>3}cm | precision={precision:.4f} | recall={recall:.4f} | "
            "F1={f1:.4f} | FP={fp_false_matches} | FN={fn_missed_matches}".format(**row)
        )


if __name__ == "__main__":
    main()

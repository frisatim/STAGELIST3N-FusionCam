"""Run recorded-video Phase 2 vs Phase 3 campaign for zone_1."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from campaign_utils import (
    DEFAULT_CONFIG,
    DEFAULT_GT_OBJECTS_TAD,
    PERSON_CLASS_ID,
    PHASE2_DIR,
    PHASE3_DIR,
    PROJECT_ROOT,
    ZONE1_CAMERAS,
    ZONE1_GT_ID_MAP,
    ZONE1_VIDEO_MAP,
    Phase3Campaign,
    compute_phase3_metrics,
    discover_model_specs,
    merge_csvs,
    parse_csv_arg,
    precheck_campaign_inputs,
    run_audit,
    run_operational_ablation,
    run_subprocess,
    run_truth_ablation,
    stdout_to_log,
    timestamped_campaign_dir,
    write_csv,
    write_manifest,
    write_phase2_specs,
)


COMPARISON_FIELDS = [
    "phase", "task", "model_version", "model", "format", "camera", "gt_camera",
    "metric_median", "metric_p95", "metric_mean", "precision", "recall", "f1",
    "n_faux_positifs", "far_per_hour", "fps", "latency_mean_ms",
    "alerts_duplicated", "global_id_switches", "source_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Campaign recorded videos: Phase 2 TAD/TRD vs Phase 3 fusion."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cameras", default=",".join(ZONE1_CAMERAS))
    parser.add_argument(
        "--versions",
        "--dataset-versions",
        "--dataset-version",
        dest="versions",
        default="V2,V3",
        help="Comma-separated trained dataset/model versions to evaluate, e.g. V4.",
    )
    parser.add_argument(
        "--dataset-variant",
        choices=["person_objects", "objects_only"],
        default="person_objects",
        help="Nested V4 training variant to use. Ignored for flat V2/V3 layouts.",
    )
    parser.add_argument("--formats", default="pt,fp32_engine")
    parser.add_argument("--models", default=None, help="Comma-separated model names.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default=None, help="YOLO device for Phase 3, e.g. cuda:0 or cpu.")
    parser.add_argument("--phase2-device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--phase2-conf", type=float, default=0.5)
    parser.add_argument("--phase2-trd-conf", type=float, default=0.4)
    parser.add_argument(
        "--phase2-imgsz",
        type=int,
        default=None,
        help=(
            "Inference size for Phase 2 child evaluators. If omitted, V4 TensorRT "
            "engines use the training/export imgsz from args.yaml; other formats keep "
            "their evaluator default."
        ),
    )
    parser.add_argument("--tad-gt", type=Path, default=DEFAULT_GT_OBJECTS_TAD)
    parser.add_argument(
        "--video-map",
        default=None,
        help=(
            "Override recorded videos, format: cam_02=path,cam_03=path. "
            "Useful for replaying newly recorded live captures."
        ),
    )
    parser.add_argument(
        "--gt-id-map",
        default=None,
        help=(
            "Override GT camera IDs, format: cam_02=cam_02_live01,cam_03=cam_03_live01. "
            "The values must match id_camera in GT JSON files."
        ),
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=5,
        help="Phase 2 TRD frame step kept for backward compatibility. Use 1 for exhaustive TRD.",
    )
    parser.add_argument(
        "--phase2-trd-skip",
        type=int,
        default=None,
        help="Frame step for Phase 2 TRD. Defaults to --skip.",
    )
    parser.add_argument("--tad-skip", type=int, default=5)
    parser.add_argument(
        "--phase2-timeout-min",
        type=float,
        default=45.0,
        help="Timeout per Phase 2 child command. Use 0 to disable.",
    )
    parser.add_argument("--fusion-distance-m", type=float, default=1.0)
    parser.add_argument("--fusion-time-window-ms", type=float, default=500.0)
    parser.add_argument("--person-zone-vote-ratio", type=float, default=None)
    parser.add_argument("--person-zone-min-votes", type=int, default=None)
    parser.add_argument("--object-min-camera-votes", type=int, default=None)
    parser.add_argument(
        "--no-weak-object-alerts",
        action="store_true",
        help="Only export confirmed multi-camera object alerts.",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument(
        "--include-rtdetr-phase2-engine",
        action="store_true",
        help="Run RT-DETR TensorRT engines in Phase 2. Skipped by default because they can hang or emit invalid boxes.",
    )
    parser.add_argument(
        "--include-rtdetr-phase3",
        action="store_true",
        help="Run RT-DETR engines in Phase 3 too. RT-DETR .pt is included by default.",
    )
    parser.add_argument("--no-strict-tad", action="store_true")
    parser.add_argument("--fusion-truth-csv", type=Path, default=None)
    return parser.parse_args()


def parse_video_map_overrides(raw: str | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    if not raw:
        return overrides
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                "[ERREUR] --video-map attend cam_id=path, ex: cam_02=recordings/foo.mp4"
            )
        cam_id, path = item.split("=", 1)
        cam_id = cam_id.strip()
        path = path.strip().strip('"')
        if not cam_id or not path:
            raise SystemExit(f"[ERREUR] entree --video-map invalide: {item}")
        video_path = Path(path)
        if not video_path.is_absolute():
            video_path = (PROJECT_ROOT / video_path).resolve()
        overrides[cam_id] = video_path
    return overrides


def parse_gt_id_map_overrides(raw: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not raw:
        return overrides
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                "[ERREUR] --gt-id-map attend cam_id=gt_id, ex: cam_02=cam_02_live01"
            )
        cam_id, gt_id = item.split("=", 1)
        cam_id = cam_id.strip()
        gt_id = gt_id.strip()
        if not cam_id or not gt_id:
            raise SystemExit(f"[ERREUR] entree --gt-id-map invalide: {item}")
        overrides[cam_id] = gt_id
    return overrides


def group_specs_for_phase2(specs):
    grouped = defaultdict(list)
    for spec in specs:
        grouped[(spec.version, spec.format_label, spec.format_arg, spec.suffix_arg)].append(spec)
    return grouped


def infer_training_imgsz(spec, default: int = 960) -> int:
    """Read Ultralytics args.yaml next to a trained model when available."""
    args_yaml = spec.weights_path.parent.parent / "args.yaml"
    if not args_yaml.exists():
        return default
    for line in args_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("imgsz:"):
            _, value = line.split(":", 1)
            try:
                return int(float(value.strip()))
            except ValueError:
                return default
    return default


def phase2_imgsz_args(args, spec) -> list[str]:
    if args.phase2_imgsz is not None:
        return ["--imgsz", str(args.phase2_imgsz)]
    if spec.format_arg == "engine" and spec.version.upper() == "V4":
        return ["--imgsz", str(infer_training_imgsz(spec, default=960))]
    return []


def run_phase2(args, specs, cameras, gt_id_map, video_map, campaign_dir: Path) -> list[Path]:
    output_csvs: list[Path] = []
    phase2_log = campaign_dir / "logs" / "phase2.log"
    specs_dir = campaign_dir / "model_specs"
    trd_skip = args.phase2_trd_skip if args.phase2_trd_skip is not None else args.skip
    timeout_s = None if args.phase2_timeout_min <= 0 else args.phase2_timeout_min * 60.0

    for spec in specs:
        if (
            spec.model_type == "rtdetr"
            and spec.format_label != "pt"
            and not args.include_rtdetr_phase2_engine
        ):
            print(
                f"[SKIP] Phase 2 {spec.run_label}: RT-DETR engine ignore par defaut "
                "(risque de blocage/boites invalides). RT-DETR .pt reste inclus."
            )
            continue

        runs_dir = PHASE2_DIR / "Modelstrained" / spec.version
        specs_path = specs_dir / f"{spec.run_label}.json"
        write_phase2_specs([spec], specs_path)

        for cam_id in cameras:
            gt_id = gt_id_map[cam_id]
            video = video_map[cam_id]
            tad_out = campaign_dir / "phase2" / "tad" / f"{spec.version}_{spec.format_label}" / cam_id / spec.name
            trd_out = campaign_dir / "phase2" / "trd" / f"{spec.version}_{spec.format_label}" / cam_id / spec.name

            print(f"\n[INFO] Phase 2 TAD {spec.run_label} {cam_id}", flush=True)

            tad_cmd = [
                sys.executable,
                str(PHASE2_DIR / "evaluate_tad.py"),
                "--video",
                str(video),
                "--camera",
                gt_id,
                "--gt",
                str(args.tad_gt),
                "--runs-dir",
                str(runs_dir),
                "--format",
                spec.format_arg,
                "--model-specs",
                str(specs_path),
                "--conf",
                str(args.phase2_conf),
                "--device",
                args.phase2_device,
                "--skip",
                str(args.tad_skip),
                "--output",
                str(tad_out),
            ]
            if spec.suffix_arg:
                tad_cmd.extend(["--suffix", spec.suffix_arg])
            tad_cmd.extend(phase2_imgsz_args(args, spec))
            tad_rc = run_subprocess(tad_cmd, phase2_log, PROJECT_ROOT, timeout_s=timeout_s)
            if tad_rc != 0:
                print(f"[WARN] Phase 2 TAD {spec.run_label} {cam_id} termine avec code {tad_rc}", flush=True)
            output_csvs.extend(tad_out.rglob("*.csv"))

            print(f"\n[INFO] Phase 2 TRD {spec.run_label} {cam_id} (skip={trd_skip})", flush=True)

            trd_cmd = [
                sys.executable,
                str(PHASE2_DIR / "evaluate_trd.py"),
                "--video",
                str(video),
                "--camera",
                gt_id,
                "--config",
                str(args.config),
                "--runs-dir",
                str(runs_dir),
                "--format",
                spec.format_arg,
                "--model-specs",
                str(specs_path),
                "--person-class",
                str(PERSON_CLASS_ID),
                "--conf",
                str(args.phase2_trd_conf),
                "--device",
                args.phase2_device,
                "--skip",
                str(trd_skip),
                "--output",
                str(trd_out),
            ]
            if spec.suffix_arg:
                trd_cmd.extend(["--suffix", spec.suffix_arg])
            trd_cmd.extend(phase2_imgsz_args(args, spec))
            trd_rc = run_subprocess(trd_cmd, phase2_log, PROJECT_ROOT, timeout_s=timeout_s)
            if trd_rc != 0:
                print(f"[WARN] Phase 2 TRD {spec.run_label} {cam_id} termine avec code {trd_rc}", flush=True)
            output_csvs.extend(trd_out.rglob("*.csv"))

    return output_csvs


def run_phase3(args, specs, cameras, gt_id_map, video_map, campaign_dir: Path) -> list[Path]:
    output_csvs: list[Path] = []
    phase3_root = campaign_dir / "phase3"
    summary_rows = []

    for spec in specs:
        if (
            spec.model_type == "rtdetr"
            and spec.format_label != "pt"
            and not args.include_rtdetr_phase3
        ):
            print(
                f"[SKIP] Phase 3 {spec.run_label}: RT-DETR engine ignore par defaut "
                "(ByteTrack/TensorRT produit des bbox NaN). RT-DETR .pt reste inclus."
            )
            summary_rows.append(
                {
                    "model_version": spec.version,
                    "model": spec.name,
                    "format": spec.format_label,
                    "frames": 0,
                    "detections": 0,
                    "alerts": 0,
                    "fusion_links": 0,
                    "unique_global_ids": 0,
                    "global_id_switches": 0,
                    "latency_mean_ms": 0.0,
                    "latency_median_ms": 0.0,
                    "latency_p95_ms": 0.0,
                    "latency_max_ms": 0.0,
                }
            )
            continue
        run_dir = phase3_root / spec.run_label
        log_path = campaign_dir / "logs" / f"phase3_{spec.run_label}.txt"
        print(f"\n[INFO] Phase 3 {spec.run_label}")
        campaign = Phase3Campaign(
            config_path=args.config,
            out_dir=run_dir,
            cameras=cameras,
            model_spec=spec,
            device=args.device,
            fusion_distance_m=args.fusion_distance_m,
            fusion_time_window_ms=args.fusion_time_window_ms,
            confidence=args.conf,
            alerting_overrides={
                "person_zone_min_camera_ratio": args.person_zone_vote_ratio,
                "person_zone_min_camera_votes": args.person_zone_min_votes,
                "object_min_camera_votes": args.object_min_camera_votes,
                "object_emit_weak_alerts": not args.no_weak_object_alerts,
            },
        )
        log_file, redirect_ctx = stdout_to_log(log_path)
        try:
            with redirect_ctx:
                summary = campaign.run_recorded(
                    video_map=video_map,
                    max_frames=args.max_frames,
                    display=not args.no_display,
                    log_path=log_path,
                )
        finally:
            log_file.close()

        campaign.write_outputs(run_dir)
        trd_rows, tad_rows = compute_phase3_metrics(
            campaign,
            cameras,
            gt_id_map,
            run_dir,
            tad_gt_path=args.tad_gt,
        )
        run_operational_ablation(
            campaign=campaign,
            config=campaign.config,
            out_csv=run_dir / "ablation" / "fusion_threshold_ablation.csv",
        )
        run_audit(log_path, args.config, run_dir / "audit" / "calibration_alert_audit.csv")
        summary_rows.append(summary)
        output_csvs.extend(
            [
                run_dir / "detections.csv",
                run_dir / "alerts.csv",
                run_dir / "fusion_links.csv",
                run_dir / "track_stability.csv",
                run_dir / "phase3_trd.csv",
                run_dir / "phase3_tad.csv",
            ]
        )

    write_csv(
        phase3_root / "summary.csv",
        summary_rows,
        [
            "model_version", "model", "format", "frames", "detections", "alerts",
            "fusion_links", "unique_global_ids", "global_id_switches",
            "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_max_ms",
        ],
    )

    for name in [
        "detections.csv",
        "alerts.csv",
        "fusion_links.csv",
        "track_stability.csv",
        "phase3_trd.csv",
        "phase3_tad.csv",
    ]:
        merge_csvs(phase3_root.glob(f"*/{name}"), phase3_root / name)

    if args.fusion_truth_csv and args.fusion_truth_csv.exists():
        run_truth_ablation(
            args.fusion_truth_csv,
            args.config,
            campaign_dir / "ablation" / "fusion_threshold_ablation.csv",
            campaign_dir / "logs" / "ablation_truth.log",
        )
    else:
        merge_csvs(
            phase3_root.glob("*/ablation/fusion_threshold_ablation.csv"),
            campaign_dir / "ablation" / "fusion_threshold_ablation.csv",
        )
    merge_csvs(
        phase3_root.glob("*/audit/calibration_alert_audit.csv"),
        campaign_dir / "audit" / "calibration_alert_audit.csv",
    )
    return output_csvs


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def phase2_path_context(path: Path, task: str, campaign_dir: Path) -> tuple[str, str, str]:
    rel = path.relative_to(campaign_dir)
    parts = rel.parts
    # phase2/{task}/{version_format}/{camera}/[{model}/]file.csv
    if len(parts) < 5 or parts[0] != "phase2" or parts[1] != task:
        return "", "", ""
    version_format = parts[2]
    fmt = version_format.split("_", 1)[-1]
    camera = parts[3]
    model_hint = parts[4] if len(parts) > 5 else ""
    return fmt, camera, model_hint


def write_comparison(campaign_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in (campaign_dir / "phase2" / "tad").rglob("*.csv"):
        fmt, camera, _model_hint = phase2_path_context(path, "tad", campaign_dir)
        for row in read_rows(path):
            rows.append(
                {
                    "phase": "phase2",
                    "task": "tad",
                    "model": row.get("model", ""),
                    "format": fmt,
                    "camera": camera,
                    "metric_median": row.get("tad_median", ""),
                    "metric_p95": row.get("tad_p95", ""),
                    "metric_mean": row.get("tad_mean", ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "f1": row.get("f1", ""),
                    "n_faux_positifs": row.get("n_faux_positifs", ""),
                    "far_per_hour": row.get("far_per_hour", ""),
                    "fps": row.get("fps_moyen", ""),
                    "latency_mean_ms": row.get("latency_mean_ms", ""),
                    "source_csv": str(path),
                }
            )
    for path in (campaign_dir / "phase2" / "trd").rglob("*.csv"):
        fmt, camera, _model_hint = phase2_path_context(path, "trd", campaign_dir)
        for row in read_rows(path):
            rows.append(
                {
                    "phase": "phase2",
                    "task": "trd",
                    "model": row.get("model", ""),
                    "format": fmt,
                    "camera": camera,
                    "metric_median": row.get("trd_median", ""),
                    "metric_p95": row.get("trd_p95", ""),
                    "metric_mean": row.get("trd_mean", ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "f1": row.get("f1", ""),
                    "n_faux_positifs": row.get("n_faux_positifs", ""),
                    "far_per_hour": row.get("far_per_hour", ""),
                    "fps": row.get("fps_inference", ""),
                    "latency_mean_ms": row.get("latency_mean_ms", ""),
                    "source_csv": str(path),
                }
            )
    for path, task, median_col, p95_col, mean_col in [
        (campaign_dir / "phase3" / "phase3_tad.csv", "tad", "tad_median", "tad_p95", "tad_mean"),
        (campaign_dir / "phase3" / "phase3_trd.csv", "trd", "trd_median", "trd_p95", "trd_mean"),
    ]:
        for row in read_rows(path):
            rows.append(
                {
                    "phase": "phase3",
                    "task": task,
                    "model_version": row.get("model_version", ""),
                    "model": row.get("model", ""),
                    "format": row.get("format", ""),
                    "camera": row.get("camera", ""),
                    "gt_camera": row.get("gt_camera", ""),
                    "metric_median": row.get(median_col, ""),
                    "metric_p95": row.get(p95_col, ""),
                    "metric_mean": row.get(mean_col, ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "f1": row.get("f1", ""),
                    "n_faux_positifs": row.get("n_faux_positifs", ""),
                    "source_csv": str(path),
                }
            )
    write_csv(campaign_dir / "comparison_phase2_phase3.csv", rows, COMPARISON_FIELDS)


def main() -> None:
    args = parse_args()
    campaign_dir = args.out_dir or timestamped_campaign_dir()
    cameras = parse_csv_arg(args.cameras, ZONE1_CAMERAS)
    versions = parse_csv_arg(args.versions, ("V2", "V3"))
    formats = parse_csv_arg(args.formats, ("pt", "fp32_engine"))
    models = parse_csv_arg(args.models, ()) if args.models else None
    dataset_variants = [args.dataset_variant] if args.dataset_variant else None
    video_map = {cam: ZONE1_VIDEO_MAP[cam] for cam in cameras}
    video_map.update(
        {
            cam: path
            for cam, path in parse_video_map_overrides(args.video_map).items()
            if cam in cameras
        }
    )
    gt_id_map = {cam: ZONE1_GT_ID_MAP[cam] for cam in cameras}
    gt_id_map.update(
        {
            cam: gt_id
            for cam, gt_id in parse_gt_id_map_overrides(args.gt_id_map).items()
            if cam in cameras
        }
    )

    precheck_campaign_inputs(
        cameras=cameras,
        video_map=video_map,
        gt_id_map=gt_id_map,
        tad_gt_path=args.tad_gt,
        strict_tad=not args.no_strict_tad,
    )
    specs = discover_model_specs(
        versions=versions,
        formats=formats,
        model_names=models,
        dataset_variants=dataset_variants,
    )
    if not specs:
        raise SystemExit("[ERREUR] Aucun modele decouvert.")
    discovered_formats = {spec.format_label for spec in specs}
    missing_formats = [fmt for fmt in formats if fmt not in discovered_formats]
    if missing_formats:
        print(
            "[WARN] Aucun poids decouvert pour les formats demandes: "
            + ", ".join(missing_formats)
        )

    campaign_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        campaign_dir / "manifest.json",
        {
            "mode": "recorded",
            "config": str(args.config),
            "cameras": cameras,
            "gt_id_map": gt_id_map,
            "tad_gt": str(args.tad_gt),
            "video_map": {cam: str(path) for cam, path in video_map.items()},
            "dataset_versions": versions,
            "dataset_variant": args.dataset_variant,
            "models": [spec.manifest_dict() for spec in specs],
            "max_frames": args.max_frames,
        },
    )

    if not args.skip_phase2:
        run_phase2(args, specs, cameras, gt_id_map, video_map, campaign_dir)
    if not args.skip_phase3:
        run_phase3(args, specs, cameras, gt_id_map, video_map, campaign_dir)
    write_comparison(campaign_dir)
    print(f"\n[INFO] Campagne terminee: {campaign_dir}")


if __name__ == "__main__":
    main()

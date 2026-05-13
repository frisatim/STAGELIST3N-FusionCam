"""Run bounded RTSP Phase 3 campaign for zone_1."""

from __future__ import annotations

import argparse
from pathlib import Path

from campaign_utils import (
    DEFAULT_CONFIG,
    DEFAULT_GT_OBJECTS_TAD,
    ZONE1_CAMERAS,
    Phase3Campaign,
    compute_phase3_metrics,
    discover_model_specs,
    merge_csvs,
    parse_csv_arg,
    run_audit,
    run_operational_ablation,
    run_truth_ablation,
    stdout_to_log,
    timestamped_campaign_dir,
    write_csv,
    write_manifest,
)
from video_capture import CaptureOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded live RTSP Phase 3 campaign.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cameras", default=",".join(ZONE1_CAMERAS))
    parser.add_argument("--versions", default="V2,V3")
    parser.add_argument("--formats", default="pt,fp32_engine")
    parser.add_argument("--models", default=None, help="Comma-separated model names.")
    parser.add_argument("--duration-min", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Do not save RTSP videos during the live campaign.",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=25.0,
        help="FPS written to recorded live videos.",
    )
    parser.add_argument("--tad-gt", type=Path, default=DEFAULT_GT_OBJECTS_TAD)
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
    parser.add_argument(
        "--display-mode",
        choices=["annotated", "raw"],
        default="annotated",
        help="Live display mode when --no-display is not used.",
    )
    parser.add_argument(
        "--include-rtdetr-phase3",
        action="store_true",
        help="Run RT-DETR engines in Phase 3 too. RT-DETR .pt is included by default.",
    )
    parser.add_argument("--capture-backend", choices=["opencv", "gstreamer"], default="gstreamer")
    parser.add_argument("--gst-latency-ms", type=int, default=100)
    parser.add_argument("--gst-protocol", choices=["tcp", "udp"], default="tcp")
    parser.add_argument("--gst-codec", choices=["h264", "h265"], default="h265")
    parser.add_argument("--gst-decoder", default="decodebin")
    parser.add_argument("--gst-pipeline", choices=["decodebin", "fixed", "both"], default="decodebin")
    parser.add_argument("--no-ffmpeg-fallback", action="store_true")
    parser.add_argument("--fusion-truth-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_dir = args.out_dir or timestamped_campaign_dir("campaign_zone1_live")
    cameras = parse_csv_arg(args.cameras, ZONE1_CAMERAS)
    versions = parse_csv_arg(args.versions, ("V2", "V3"))
    formats = parse_csv_arg(args.formats, ("pt", "fp32_engine"))
    models = parse_csv_arg(args.models, ()) if args.models else None
    specs = discover_model_specs(versions=versions, formats=formats, model_names=models)
    if not specs:
        raise SystemExit("[ERREUR] Aucun modele decouvert.")

    campaign_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        campaign_dir / "manifest.json",
        {
            "mode": "live",
            "config": str(args.config),
            "cameras": cameras,
            "duration_min": args.duration_min,
            "tad_gt": str(args.tad_gt),
            "models": [spec.manifest_dict() for spec in specs],
            "capture_backend": args.capture_backend,
            "gst_codec": args.gst_codec,
            "record_video": not args.no_record_video,
            "record_fps": args.record_fps,
            "display_mode": args.display_mode,
            "alerting": {
                "person_zone_min_camera_ratio": args.person_zone_vote_ratio,
                "person_zone_min_camera_votes": args.person_zone_min_votes,
                "object_min_camera_votes": args.object_min_camera_votes,
                "object_emit_weak_alerts": not args.no_weak_object_alerts,
            },
        },
    )

    capture_options = CaptureOptions(
        backend=args.capture_backend,
        gst_latency_ms=args.gst_latency_ms,
        gst_protocol=args.gst_protocol,
        gst_codec=args.gst_codec,
        gst_decoder=args.gst_decoder,
        gst_pipeline=args.gst_pipeline,
        fallback_ffmpeg=not args.no_ffmpeg_fallback,
    )
    phase3_root = campaign_dir / "phase3"
    summary_rows = []

    for spec in specs:
        if (
            spec.model_type == "rtdetr"
            and spec.format_label != "pt"
            and not args.include_rtdetr_phase3
        ):
            print(
                f"[SKIP] Phase 3 live {spec.run_label}: RT-DETR engine ignore par defaut "
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
        log_path = campaign_dir / "logs" / f"phase3_live_{spec.run_label}.txt"
        print(f"\n[INFO] Live Phase 3 {spec.run_label}")
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
                summary = campaign.run_live(
                    duration_s=args.duration_min * 60.0,
                    display=not args.no_display,
                    log_path=log_path,
                    capture_options=capture_options,
                    record_dir=None if args.no_record_video else campaign_dir / "recordings" / spec.run_label,
                    record_fps=args.record_fps,
                    display_mode=args.display_mode,
                )
        finally:
            log_file.close()

        campaign.write_outputs(run_dir)
        compute_phase3_metrics(
            campaign,
            cameras,
            {cam: cam for cam in cameras},
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

    for name in [
        "detections.csv",
        "alerts.csv",
        "fusion_links.csv",
        "track_stability.csv",
        "phase3_trd.csv",
        "phase3_tad.csv",
        "sync_events.csv",
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
    write_csv(
        phase3_root / "summary.csv",
        summary_rows,
        [
            "model_version", "model", "format", "frames", "detections", "alerts",
            "fusion_links", "unique_global_ids", "global_id_switches",
            "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_max_ms",
        ],
    )
    print(f"\n[INFO] Campagne live terminee: {campaign_dir}")


if __name__ == "__main__":
    main()

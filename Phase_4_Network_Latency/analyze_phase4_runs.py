from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Phase_4_Network_Latency.latency_metrics import summarize_phase3_run


def _read_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def analyze_runs(run_dirs: list[Path], expected_cameras: list[str] | None = None) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "phase3" / "summary.csv"
        sync_path = run_dir / "phase3" / "sync_events.csv"
        if not summary_path.exists() or not sync_path.exists():
            continue

        manifest = _read_manifest(run_dir)
        health = summarize_phase3_run(run_dir, expected_cameras=expected_cameras)
        rows.append(
            {
                "run_dir": str(run_dir),
                "mode": manifest.get("mode", ""),
                "capture_backend": manifest.get("capture_backend", ""),
                "gst_protocol": manifest.get("gst_protocol", ""),
                "gst_latency_ms": manifest.get("gst_latency_ms", ""),
                "gst_pipeline": manifest.get("gst_pipeline", ""),
                "ffmpeg_fallback": manifest.get("ffmpeg_fallback", ""),
                "record_video": manifest.get("record_video", ""),
                "record_fps": manifest.get("record_fps", ""),
                "frames": health.campaign_frames,
                "duration_s": round(health.duration_s, 3),
                "effective_fps": round(health.effective_fps, 3),
                "latency_mean_ms": health.latency_mean_ms,
                "latency_p95_ms": health.latency_p95_ms,
                "latency_max_ms": health.latency_max_ms,
                "fusion_links": health.fusion_links,
                "fusion_active": int(health.has_fusion),
                "alerts": health.alerts,
                "dropped_cameras": ",".join(health.dropped_cameras),
                "camera_count": len(health.cameras),
            }
        )
    return rows


def write_rows(rows: list[dict[str, str | int | float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["run_dir"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 4/Phase 3 campaign health.")
    parser.add_argument(
        "--runs-glob",
        default="Phase_3_Fusion_MultiCam/reports/campaign_zone1_live_*",
        help="Glob matching campaign directories.",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("Phase_4_Network_Latency/phase4_run_health.csv"))
    parser.add_argument("--cameras", default="cam_02,cam_03,cam_05,cam_07")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = sorted(Path(path) for path in glob.glob(args.runs_glob))
    cameras = [part.strip() for part in args.cameras.split(",") if part.strip()]
    rows = analyze_runs(run_dirs, expected_cameras=cameras)
    write_rows(rows, args.out_csv)
    print(f"[INFO] Wrote {len(rows)} run summaries to {args.out_csv}")


if __name__ == "__main__":
    main()

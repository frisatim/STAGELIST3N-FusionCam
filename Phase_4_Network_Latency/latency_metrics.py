from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class CameraSyncStats:
    cam_id: str
    frames: int
    first_elapsed_s: float
    last_elapsed_s: float
    missing_campaign_frames: int
    effective_fps: float
    dropped_early: bool


@dataclass(frozen=True)
class RunHealth:
    campaign_frames: int
    duration_s: float
    effective_fps: float
    latency_mean_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    fusion_links: int
    alerts: int
    cameras: tuple[CameraSyncStats, ...]

    @property
    def has_fusion(self) -> bool:
        return self.fusion_links > 0

    @property
    def dropped_cameras(self) -> tuple[str, ...]:
        return tuple(cam.cam_id for cam in self.cameras if cam.dropped_early)


def _float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def analyze_sync_events(sync_events_csv: Path, expected_cameras: list[str] | None = None) -> tuple[CameraSyncStats, ...]:
    rows = read_csv_rows(sync_events_csv)
    if not rows:
        return ()

    campaign_frames = {_int(row.get("campaign_frame")) for row in rows}
    max_campaign_frame = max(campaign_frames)
    total_campaign_frames = max_campaign_frame + 1
    run_last_elapsed_s = max(_float(row.get("elapsed_s")) for row in rows)
    cameras = expected_cameras or sorted({row["cam_id"] for row in rows})
    stats: list[CameraSyncStats] = []

    for cam_id in cameras:
        cam_rows = [row for row in rows if row.get("cam_id") == cam_id]
        if not cam_rows:
            stats.append(CameraSyncStats(cam_id, 0, 0.0, 0.0, total_campaign_frames, 0.0, True))
            continue

        elapsed = [_float(row.get("elapsed_s")) for row in cam_rows]
        present_frames = {_int(row.get("campaign_frame")) for row in cam_rows}
        duration = max(elapsed) - min(elapsed)
        effective_fps = len(cam_rows) / duration if duration > 0 else 0.0
        missing = total_campaign_frames - len(present_frames)
        dropped_early = max(elapsed) < run_last_elapsed_s - 1.0 or missing > 0

        stats.append(
            CameraSyncStats(
                cam_id=cam_id,
                frames=len(cam_rows),
                first_elapsed_s=min(elapsed),
                last_elapsed_s=max(elapsed),
                missing_campaign_frames=missing,
                effective_fps=effective_fps,
                dropped_early=dropped_early,
            )
        )

    return tuple(stats)


def summarize_phase3_run(run_dir: Path, expected_cameras: list[str] | None = None) -> RunHealth:
    phase3_dir = run_dir / "phase3"
    summary_rows = read_csv_rows(phase3_dir / "summary.csv")
    if not summary_rows:
        raise ValueError(f"No rows found in {phase3_dir / 'summary.csv'}")

    summary = summary_rows[0]
    sync_stats = analyze_sync_events(phase3_dir / "sync_events.csv", expected_cameras=expected_cameras)
    campaign_frames = _int(summary.get("frames"))
    duration_s = max((cam.last_elapsed_s for cam in sync_stats), default=0.0) - min(
        (cam.first_elapsed_s for cam in sync_stats if cam.frames > 0),
        default=0.0,
    )
    effective_fps = campaign_frames / duration_s if duration_s > 0 else 0.0

    return RunHealth(
        campaign_frames=campaign_frames,
        duration_s=duration_s,
        effective_fps=effective_fps,
        latency_mean_ms=_float(summary.get("latency_mean_ms")),
        latency_p95_ms=_float(summary.get("latency_p95_ms")),
        latency_max_ms=_float(summary.get("latency_max_ms")),
        fusion_links=_int(summary.get("fusion_links")),
        alerts=_int(summary.get("alerts")),
        cameras=sync_stats,
    )


def compare_runs(run_health: list[RunHealth]) -> dict[str, float]:
    if not run_health:
        return {
            "runs": 0.0,
            "mean_effective_fps": 0.0,
            "mean_latency_p95_ms": 0.0,
            "drop_rate": 0.0,
            "fusion_enabled_rate": 0.0,
        }

    camera_count = sum(len(run.cameras) for run in run_health)
    dropped_count = sum(len(run.dropped_cameras) for run in run_health)
    return {
        "runs": float(len(run_health)),
        "mean_effective_fps": mean(run.effective_fps for run in run_health),
        "mean_latency_p95_ms": mean(run.latency_p95_ms for run in run_health),
        "drop_rate": dropped_count / camera_count if camera_count else 0.0,
        "fusion_enabled_rate": sum(1 for run in run_health if run.has_fusion) / len(run_health),
    }

"""Tests des métriques de latence et de synchronisation caméras sur des CSV de campagne minimaux."""

from pathlib import Path

from Phase_4_Network_Latency.latency_metrics import (
    analyze_fusion_links,
    analyze_sync_events,
    compare_runs,
    summarize_phase3_run,
)


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_analyze_sync_events_detects_camera_drop(tmp_path: Path):
    """Une caméra qui cesse d'émettre en cours de run est marquée décrochée, avec ses trames manquantes."""
    sync_csv = tmp_path / "sync_events.csv"
    _write(
        sync_csv,
        """
model_version,model,format,campaign_frame,cam_id,video_frame,timestamp_epoch,timestamp_iso,elapsed_s,recorded_video,capture_backend
V4,yolov8s,fp32_engine,0,cam_02,0,0,,0.0,,FFMPEG
V4,yolov8s,fp32_engine,0,cam_07,0,0,,0.0,,FFMPEG
V4,yolov8s,fp32_engine,1,cam_02,1,0,,0.1,,FFMPEG
V4,yolov8s,fp32_engine,1,cam_07,1,0,,0.1,,FFMPEG
V4,yolov8s,fp32_engine,2,cam_02,2,0,,0.2,,FFMPEG
V4,yolov8s,fp32_engine,3,cam_02,3,0,,0.3,,FFMPEG
V4,yolov8s,fp32_engine,4,cam_02,4,0,,0.4,,FFMPEG
""",
    )

    stats = analyze_sync_events(sync_csv, expected_cameras=["cam_02", "cam_07"])
    by_cam = {stat.cam_id: stat for stat in stats}

    assert by_cam["cam_02"].dropped_early is False
    assert by_cam["cam_02"].missing_campaign_frames == 0
    assert by_cam["cam_07"].dropped_early is True
    assert by_cam["cam_07"].missing_campaign_frames == 3


def test_summarize_phase3_run_reports_latency_fps_and_fusion(tmp_path: Path):
    """Le bilan d'un run reprend correctement cadence effective, latences et métriques de fusion."""
    _write(
        tmp_path / "phase3" / "summary.csv",
        """
model_version,model,format,frames,detections,alerts,fusion_links,unique_global_ids,global_id_switches,latency_mean_ms,latency_median_ms,latency_p95_ms,latency_max_ms
V4,yolov8s,fp32_engine,5,20,2,3,4,0,18.0,17.0,25.0,200.0
""",
    )
    _write(
        tmp_path / "phase3" / "fusion_links.csv",
        """
model_version,model,format,frame,timestamp,global_id,cam_a,track_a,cam_b,track_b,distance_m
V4,yolov8s,fp32_engine,1,0.1,1,cam_02,10,cam_03,20,0.8
V4,yolov8s,fp32_engine,1,0.1,2,cam_02,11,cam_03,21,0.9
V4,yolov8s,fp32_engine,3,0.3,3,cam_02,12,cam_03,22,0.7
""",
    )
    _write(
        tmp_path / "phase3" / "sync_events.csv",
        """
model_version,model,format,campaign_frame,cam_id,video_frame,timestamp_epoch,timestamp_iso,elapsed_s,recorded_video,capture_backend
V4,yolov8s,fp32_engine,0,cam_02,0,0,,0.0,,FFMPEG
V4,yolov8s,fp32_engine,0,cam_03,0,0,,0.0,,FFMPEG
V4,yolov8s,fp32_engine,1,cam_02,1,0,,0.1,,FFMPEG
V4,yolov8s,fp32_engine,1,cam_03,1,0,,0.1,,FFMPEG
V4,yolov8s,fp32_engine,2,cam_02,2,0,,0.2,,FFMPEG
V4,yolov8s,fp32_engine,2,cam_03,2,0,,0.2,,FFMPEG
V4,yolov8s,fp32_engine,3,cam_02,3,0,,0.3,,FFMPEG
V4,yolov8s,fp32_engine,3,cam_03,3,0,,0.3,,FFMPEG
V4,yolov8s,fp32_engine,4,cam_02,4,0,,0.4,,FFMPEG
V4,yolov8s,fp32_engine,4,cam_03,4,0,,0.4,,FFMPEG
""",
    )

    health = summarize_phase3_run(tmp_path, expected_cameras=["cam_02", "cam_03"])

    assert health.campaign_frames == 5
    assert health.effective_fps == 12.5
    assert health.latency_p95_ms == 25.0
    assert health.has_fusion is True
    assert health.detections == 20
    assert health.frames_with_fusion == 2
    assert health.fusion_frame_rate_pct == 40.0
    assert health.fusion_links_per_frame == 0.6
    assert health.fusion_links_per_1000_detections == 150.0
    assert health.fusion_camera_pairs == 1
    assert health.top_fusion_pair == "cam_02+cam_03"
    assert health.top_fusion_pair_links == 3
    assert health.dropped_cameras == ()


def test_analyze_fusion_links_counts_frames_not_links(tmp_path: Path):
    """L'analyse des liens de fusion compte les trames distinctes, pas les liens, et identifie la paire dominante."""
    fusion_csv = tmp_path / "fusion_links.csv"
    _write(
        fusion_csv,
        """
model_version,model,format,frame,timestamp,global_id,cam_a,track_a,cam_b,track_b,distance_m
V4,yolov8s,fp32_engine,10,1.0,1,cam_02,1,cam_07,2,0.5
V4,yolov8s,fp32_engine,10,1.0,2,cam_02,3,cam_07,4,0.6
V4,yolov8s,fp32_engine,11,1.1,3,cam_03,5,cam_05,6,0.7
""",
    )

    stats = analyze_fusion_links(fusion_csv)

    assert stats.frames_with_fusion == 2
    assert stats.camera_pairs == 2
    assert stats.top_pair == "cam_02+cam_07"
    assert stats.top_pair_links == 2


def test_compare_runs_aggregates_phase4_health_metrics(tmp_path: Path):
    """La comparaison de deux runs moyenne les p95 de latence et calcule le taux de runs avec fusion."""
    _write(
        tmp_path / "run_a" / "phase3" / "summary.csv",
        """
model_version,model,format,frames,detections,alerts,fusion_links,unique_global_ids,global_id_switches,latency_mean_ms,latency_median_ms,latency_p95_ms,latency_max_ms
V4,yolov8s,fp32_engine,2,10,1,1,2,0,10.0,10.0,20.0,40.0
""",
    )
    _write(
        tmp_path / "run_a" / "phase3" / "sync_events.csv",
        """
model_version,model,format,campaign_frame,cam_id,video_frame,timestamp_epoch,timestamp_iso,elapsed_s,recorded_video,capture_backend
V4,yolov8s,fp32_engine,0,cam_02,0,0,,0.0,,FFMPEG
V4,yolov8s,fp32_engine,1,cam_02,1,0,,1.0,,FFMPEG
""",
    )
    _write(
        tmp_path / "run_b" / "phase3" / "summary.csv",
        """
model_version,model,format,frames,detections,alerts,fusion_links,unique_global_ids,global_id_switches,latency_mean_ms,latency_median_ms,latency_p95_ms,latency_max_ms
V4,yolov8s,fp32_engine,2,10,1,0,2,0,20.0,20.0,30.0,60.0
""",
    )
    _write(
        tmp_path / "run_b" / "phase3" / "sync_events.csv",
        """
model_version,model,format,campaign_frame,cam_id,video_frame,timestamp_epoch,timestamp_iso,elapsed_s,recorded_video,capture_backend
V4,yolov8s,fp32_engine,0,cam_02,0,0,,0.0,,FFMPEG
V4,yolov8s,fp32_engine,1,cam_02,1,0,,1.0,,FFMPEG
""",
    )

    comparison = compare_runs(
        [
            summarize_phase3_run(tmp_path / "run_a", expected_cameras=["cam_02"]),
            summarize_phase3_run(tmp_path / "run_b", expected_cameras=["cam_02"]),
        ]
    )

    assert comparison["runs"] == 2.0
    assert comparison["mean_latency_p95_ms"] == 25.0
    assert comparison["fusion_enabled_rate"] == 0.5

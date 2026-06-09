from pathlib import Path

from Phase_4_Network_Latency.latency_metrics import (
    analyze_sync_events,
    compare_runs,
    summarize_phase3_run,
)


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_analyze_sync_events_detects_camera_drop(tmp_path: Path):
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
    _write(
        tmp_path / "phase3" / "summary.csv",
        """
model_version,model,format,frames,detections,alerts,fusion_links,unique_global_ids,global_id_switches,latency_mean_ms,latency_median_ms,latency_p95_ms,latency_max_ms
V4,yolov8s,fp32_engine,5,20,2,3,4,0,18.0,17.0,25.0,200.0
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
    assert health.dropped_cameras == ()


def test_compare_runs_aggregates_phase4_health_metrics(tmp_path: Path):
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

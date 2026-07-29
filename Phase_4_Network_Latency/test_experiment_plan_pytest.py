"""Tests du générateur de matrice d'expériences : contenu de la matrice, commandes et export CSV."""

from pathlib import Path

from Phase_4_Network_Latency.experiment_plan import (
    generate_experiment_cases,
    write_cases_csv,
)


def test_generate_experiment_cases_keeps_matrix_focused():
    """La matrice Linux/Ethernet contient exactement les 9 cas attendus (6 vidéo + 3 alertes)."""
    cases = generate_experiment_cases(
        interface="eth-test",
        os_targets=("linux",),
        link_types=("ethernet",),
        duration_min=3.0,
    )

    run_ids = {case.run_id for case in cases}

    assert "video_linux_ethernet_clean_rtsp_tcp" in run_ids
    assert "video_linux_ethernet_severe_rtsp_udp" in run_ids
    assert "alert_linux_ethernet_websocket" in run_ids
    assert "alert_linux_ethernet_mqtt_qos1" in run_ids
    assert len(cases) == 9


def test_video_case_uses_gstreamer_protocol_without_ffmpeg_fallback():
    """Un cas vidéo UDP force le backend GStreamer, le protocole udp et l'absence de repli FFmpeg."""
    cases = generate_experiment_cases(interface="eth-test")
    udp_case = next(case for case in cases if case.run_id == "video_linux_ethernet_clean_rtsp_udp")

    assert "--capture-backend gstreamer" in udp_case.command
    assert "--gst-protocol udp" in udp_case.command
    assert "--no-ffmpeg-fallback" in udp_case.command


def test_write_cases_csv(tmp_path: Path):
    """L'export CSV écrit l'en-tête attendu et l'identifiant du premier cas."""
    out = tmp_path / "matrix.csv"
    cases = generate_experiment_cases(interface="eth-test")

    write_cases_csv(cases[:1], out)

    text = out.read_text(encoding="utf-8")
    assert "run_id,family" in text
    assert "video_linux_ethernet_clean_rtsp_tcp" in text

#!/usr/bin/env python3
"""Generate presentation-ready Phase 3 / Phase 4 figures from local CSV results.

The script intentionally uses only the Python standard library plus matplotlib
and numpy so it can run in the lightweight WSL Python environment used during
analysis.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OUT_DIR = REPO_ROOT / "docs" / "figures" / "phase3_phase4_20260623"


BLUE = "#245999"
ORANGE = "#eb8c00"
GREEN = "#2f7d32"
RED = "#c9342c"
PURPLE = "#6b4ea0"
CYAN = "#27a6d9"
GRAY = "#5c6670"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def numeric_column(path: Path, column: str) -> list[float]:
    rows = read_rows(path)
    values: list[float] = []
    for row in rows:
        value = to_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def savefig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}", pad: float = 3.0) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def figure_latency_by_step() -> None:
    trace_path = first_existing(
        [
            WORKSPACE_ROOT
            / "STAGELIST3N-FusionCam-data"
            / "reports"
            / "Phase_3_Fusion_MultiCam"
            / "pc_latency_internal_2cam_trace.csv",
            WORKSPACE_ROOT
            / "Phase_3_Fusion_MultiCam"
            / "reports"
            / "pc_latency_internal_2cam_trace.csv",
        ]
    )
    if trace_path is None:
        print("[SKIP] latency trace not found")
        return

    columns = [
        ("Capture", "capture_read_ms"),
        ("Inference + tracking", "inference_tracking_ms"),
        ("Fusion", "fusion_ms"),
        ("Alertes", "alerts_ms"),
        ("Metadata", "metadata_ms"),
    ]
    means: list[float] = []
    p95s: list[float] = []
    labels: list[str] = []
    for label, col in columns:
        values = numeric_column(trace_path, col)
        labels.append(label)
        means.append(statistics.fmean(values) if values else float("nan"))
        p95s.append(percentile(values, 0.95) if values else float("nan"))

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - width / 2, means, width, label="Moyenne", color=BLUE)
    b2 = ax.bar(x + width / 2, p95s, width, label="P95", color=ORANGE)
    ax.set_title("Latence interne par etape - Phase 3 live 2 cameras", fontsize=15, pad=12)
    ax.set_ylabel("Latence (ms, echelle log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_yscale("log")
    ax.legend()
    style_axes(ax)
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.text(
        0.01,
        -0.22,
        "Lecture : l'inference/tracking domine. Fusion, alertes et metadata restent sous 1 ms en P95.",
        transform=ax.transAxes,
        fontsize=10,
    )
    savefig("01_latence_interne_par_etape.png")


def summarize_latency_file(path: Path) -> tuple[int, float, float, float, float] | None:
    values = numeric_column(path, "latency_ms")
    if not values:
        return None
    return (
        len(values),
        statistics.fmean(values),
        percentile(values, 0.5),
        percentile(values, 0.95),
        max(values),
    )


def figure_transport_latency() -> None:
    candidates = {
        "Queue": [
            WORKSPACE_ROOT / "Phase_4_Network_Latency" / "runs" / "today_queue" / "alert_latency.csv",
            WORKSPACE_ROOT / "Phase_4_Network_Latency" / "runs" / "final_queue" / "alert_latency.csv",
        ],
        "HTTP POST": [
            WORKSPACE_ROOT / "Phase_4_Network_Latency" / "runs" / "today_http_post" / "alert_latency.csv",
            WORKSPACE_ROOT / "Phase_4_Network_Latency" / "runs" / "final_http_post" / "alert_latency.csv",
        ],
        "WebSocket": [
            WORKSPACE_ROOT / "Phase_4_Network_Latency" / "runs" / "today_websocket" / "alert_latency.csv",
            WORKSPACE_ROOT / "Phase_4_Network_Latency" / "runs" / "final_websocket" / "alert_latency.csv",
        ],
        "MQTT QoS0": [
            WORKSPACE_ROOT / "Phase_4_Network_Latency" / "runs" / "today_mqtt_qos0" / "alert_latency.csv",
            REPO_ROOT / "Phase_4_Network_Latency" / "runs" / "final_mqtt_qos0" / "alert_latency.csv",
        ],
        "MQTT QoS1": [
            WORKSPACE_ROOT / "Phase_4_Network_Latency" / "runs" / "today_mqtt_qos1" / "alert_latency.csv",
            REPO_ROOT / "Phase_4_Network_Latency" / "runs" / "final_mqtt_qos1" / "alert_latency.csv",
        ],
    }

    labels: list[str] = []
    means: list[float] = []
    p95s: list[float] = []
    for label, paths in candidates.items():
        path = first_existing(paths)
        if path is None:
            continue
        summary = summarize_latency_file(path)
        if summary is None:
            continue
        _, mean, _, p95, _ = summary
        labels.append(label)
        means.append(mean)
        p95s.append(p95)

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - width / 2, means, width, label="Moyenne", color=BLUE)
    b2 = ax.bar(x + width / 2, p95s, width, label="P95", color=ORANGE)
    ax.set_title("Latence de transport des metadonnees - Phase 4", fontsize=15, pad=12)
    ax.set_ylabel("Latence (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.legend()
    style_axes(ax)
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.text(
        0.01,
        -0.22,
        "Lecture : WebSocket est le meilleur choix pour le dashboard; MQTT reste tres faible et decouple mieux les services.",
        transform=ax.transAxes,
        fontsize=10,
    )
    savefig("02_transport_metadata_http_websocket_mqtt.png")


def load_summary(path: Path) -> dict[str, float | str] | None:
    rows = read_rows(path)
    if not rows:
        return None
    row = rows[0]
    out: dict[str, float | str] = {}
    for key, value in row.items():
        num = to_float(value)
        out[key] = num if num is not None else value
    return out


def figure_camera_scaling() -> None:
    configs = [
        (
            "1 cam",
            1,
            120.0,
            WORKSPACE_ROOT / "server_relay_runs" / "server_relay_1cam_tcp100_2min" / "phase3" / "summary.csv",
        ),
        (
            "2 cams",
            2,
            300.0,
            WORKSPACE_ROOT / "server_relay_runs" / "server_relay_2cam_tcp100_5min" / "phase3" / "summary.csv",
        ),
        (
            "4 cams",
            4,
            600.0,
            WORKSPACE_ROOT / "server_relay_runs" / "server_relay_4cam_tcp100_10min" / "phase3" / "summary.csv",
        ),
    ]

    labels: list[str] = []
    fps: list[float] = []
    lat_mean: list[float] = []
    lat_p95: list[float] = []
    detections: list[float] = []
    fusion_links: list[float] = []
    alerts: list[float] = []

    for label, _, duration, path in configs:
        summary = load_summary(path)
        if not summary:
            continue
        labels.append(label)
        frames = float(summary.get("frames", 0) or 0)
        fps.append(frames / duration)
        lat_mean.append(float(summary.get("latency_mean_ms", 0) or 0))
        lat_p95.append(float(summary.get("latency_p95_ms", 0) or 0))
        detections.append(float(summary.get("detections", 0) or 0))
        fusion_links.append(float(summary.get("fusion_links", 0) or 0))
        alerts.append(float(summary.get("alerts", 0) or 0))

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    width = 0.36
    b1 = ax1.bar(x - width / 2, lat_mean, width, label="Lat. moyenne", color=BLUE)
    b2 = ax1.bar(x + width / 2, lat_p95, width, label="Lat. P95", color=ORANGE)
    ax1.set_title("Latence selon le nombre de cameras")
    ax1.set_ylabel("Latence (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    style_axes(ax1)
    annotate_bars(ax1, b1)
    annotate_bars(ax1, b2)

    ax1b = ax1.twinx()
    ax1b.plot(x, fps, color=GREEN, marker="o", linewidth=2.5, label="FPS campagne")
    ax1b.set_ylabel("FPS campagne")
    ax1b.set_ylim(0, max(fps) * 1.25 if fps else 1)
    ax1b.spines["top"].set_visible(False)

    b3 = ax2.bar(x - width, detections, width, label="Detections", color=BLUE)
    b4 = ax2.bar(x, fusion_links, width, label="Liens fusion", color=GREEN)
    b5 = ax2.bar(x + width, alerts, width, label="Alertes", color=ORANGE)
    ax2.set_title("Activite IA/fusion")
    ax2.set_ylabel("Nombre")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    style_axes(ax2)
    annotate_bars(ax2, b3, "{:.0f}")
    annotate_bars(ax2, b4, "{:.0f}")
    annotate_bars(ax2, b5, "{:.0f}")

    fig.suptitle("Montee en charge serveur - 1, 2 et 4 cameras", fontsize=15, y=1.03)
    savefig("03_montee_charge_1_2_4_cameras.png")


def count_alerts(path: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in read_rows(path):
        alert_type = row.get("alert_type") or "?"
        alert_level = row.get("alert_level") or "?"
        counts[(alert_type, alert_level)] += 1
    return counts


def figure_alerts_by_type() -> None:
    runs = [
        (
            "Server 2 cams",
            WORKSPACE_ROOT / "server_relay_runs" / "server_relay_2cam_tcp100_5min" / "phase3" / "alerts.csv",
        ),
        (
            "Server 4 cams",
            WORKSPACE_ROOT / "server_relay_runs" / "server_relay_4cam_tcp100_10min" / "phase3" / "alerts.csv",
        ),
        (
            "Recorded V4",
            WORKSPACE_ROOT
            / "Phase_3_Fusion_MultiCam"
            / "reports"
            / "campaign_zone1_20260527_135237"
            / "phase3"
            / "V4_person_objects_yolov8s_fp32_engine"
            / "alerts.csv",
        ),
    ]
    categories = [
        ("Zone/person confirmed", ("zone_violation_person", "confirmed"), BLUE),
        ("Object weak", ("forbidden_object", "weak"), ORANGE),
        ("Object confirmed", ("forbidden_object", "confirmed"), GREEN),
    ]

    labels: list[str] = []
    values = {name: [] for name, _, _ in categories}
    for label, path in runs:
        if not path.exists():
            continue
        labels.append(label)
        counts = count_alerts(path)
        for name, key, _ in categories:
            values[name].append(counts.get(key, 0))

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for name, _, color in categories:
        bars = ax.bar(x, values[name], bottom=bottom, label=name, color=color)
        bottom += np.array(values[name])
        annotate_bars(ax, bars, "{:.0f}", pad=2.0)
    ax.set_title("Alertes par type et niveau", fontsize=15, pad=12)
    ax.set_ylabel("Nombre d'alertes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    style_axes(ax)
    ax.text(
        0.01,
        -0.18,
        "Lecture : les alertes personnes/zones sont plus robustes; les objets produisent souvent des alertes faibles.",
        transform=ax.transAxes,
        fontsize=10,
    )
    savefig("04_alertes_par_type.png")


def figure_object_weak_confirmed() -> None:
    runs = [
        (
            "Server 2 cams",
            WORKSPACE_ROOT / "server_relay_runs" / "server_relay_2cam_tcp100_5min" / "phase3" / "alerts.csv",
        ),
        (
            "Server 4 cams",
            WORKSPACE_ROOT / "server_relay_runs" / "server_relay_4cam_tcp100_10min" / "phase3" / "alerts.csv",
        ),
        (
            "Recorded V4",
            WORKSPACE_ROOT
            / "Phase_3_Fusion_MultiCam"
            / "reports"
            / "campaign_zone1_20260527_135237"
            / "phase3"
            / "V4_person_objects_yolov8s_fp32_engine"
            / "alerts.csv",
        ),
    ]
    labels: list[str] = []
    weak: list[int] = []
    confirmed: list[int] = []
    for label, path in runs:
        if not path.exists():
            continue
        counts = count_alerts(path)
        labels.append(label)
        weak.append(counts.get(("forbidden_object", "weak"), 0))
        confirmed.append(counts.get(("forbidden_object", "confirmed"), 0))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x, weak, label="Weak", color=ORANGE)
    b2 = ax.bar(x, confirmed, bottom=weak, label="Confirmed", color=GREEN)
    ax.set_title("Objets interdits : alertes weak vs confirmed", fontsize=15, pad=12)
    ax.set_ylabel("Nombre d'alertes objets")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    style_axes(ax)
    annotate_bars(ax, b1, "{:.0f}", pad=2.0)
    annotate_bars(ax, b2, "{:.0f}", pad=2.0)
    ax.text(
        0.01,
        -0.18,
        "Lecture : confirmed only est plus propre mais peut manquer beaucoup d'objets si le recouvrement camera est faible.",
        transform=ax.transAxes,
        fontsize=10,
    )
    savefig("05_objets_weak_vs_confirmed.png")


def group_comparison() -> dict[tuple[str, str], dict[str, float]]:
    path = (
        WORKSPACE_ROOT
        / "Phase_3_Fusion_MultiCam"
        / "reports"
        / "campaign_zone1_20260527_135237"
        / "comparison_phase2_phase3.csv"
    )
    rows = read_rows(path)
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row.get("phase", ""), row.get("task", ""))
        for col in ["precision", "recall", "f1", "n_faux_positifs", "far_per_hour", "fps", "latency_mean_ms"]:
            value = to_float(row.get(col))
            if value is not None:
                grouped[key][col].append(value)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        out[key] = {col: statistics.fmean(vals) for col, vals in values.items() if vals}
    return out


def figure_phase2_phase3_scores() -> None:
    grouped = group_comparison()
    order = [
        ("phase2", "tad", "P2 TAD"),
        ("phase3", "tad", "P3 TAD"),
        ("phase2", "trd", "P2 TRD"),
        ("phase3", "trd", "P3 TRD"),
    ]
    labels = [label for _, _, label in order]
    metrics = [("precision", "Precision", ORANGE), ("recall", "Recall", GREEN), ("f1", "F1", BLUE)]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, (metric, name, color) in enumerate(metrics):
        vals = [grouped.get((phase, task), {}).get(metric, float("nan")) for phase, task, _ in order]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        annotate_bars(ax, bars)
    ax.set_title("Phase 2 vs Phase 3 - Precision, Recall, F1", fontsize=15, pad=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    style_axes(ax)
    ax.text(
        0.01,
        -0.20,
        "Lecture : la Phase 3 est plus interessante pour TRD/personnes; TAD/objets reste penalise par les faux positifs.",
        transform=ax.transAxes,
        fontsize=10,
    )
    savefig("06_phase2_vs_phase3_scores.png")


def figure_false_positives() -> None:
    grouped = group_comparison()
    order = [
        ("phase2", "tad", "P2 TAD"),
        ("phase3", "tad", "P3 TAD"),
        ("phase2", "trd", "P2 TRD"),
        ("phase3", "trd", "P3 TRD"),
    ]
    labels = [label for _, _, label in order]
    values = [grouped.get((phase, task), {}).get("n_faux_positifs", 0.0) for phase, task, _ in order]
    colors = [BLUE, RED, BLUE, RED]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(x, values, color=colors)
    ax.set_title("Faux positifs moyens avec verite terrain - recorded campaign", fontsize=15, pad=12)
    ax.set_ylabel("Faux positifs moyens par camera")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axes(ax)
    annotate_bars(ax, bars, "{:.1f}", pad=3.0)
    ax.text(
        0.01,
        -0.20,
        "Lecture : seuls les runs avec GT permettent de parler de vrais faux positifs. En live sans GT, parler d'alertes non verifiees.",
        transform=ax.transAxes,
        fontsize=10,
    )
    savefig("07_faux_positifs_phase2_phase3_gt.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure_latency_by_step()
    figure_transport_latency()
    figure_camera_scaling()
    figure_alerts_by_type()
    figure_object_weak_confirmed()
    figure_phase2_phase3_scores()
    figure_false_positives()


if __name__ == "__main__":
    main()

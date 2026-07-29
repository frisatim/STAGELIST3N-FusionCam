"""Agrégation des runs Phase 4 en un CSV de santé.

Ce script balaie les dossiers de runs correspondant à un motif glob et produit
une ligne de synthèse par run dans un CSV unique. Deux types de runs sont
reconnus automatiquement :

- run de benchmark d'alertes (présence d'un ``alert_latency.csv``) : résumé des
  latences de livraison par transport ;
- run de campagne live Phase 3 (présence de ``phase3/summary.csv`` et
  ``phase3/sync_events.csv``) : bilan de santé complet (cadence, latences,
  fusion, décrochages caméras) enrichi des paramètres du ``manifest.json``.

Exemple::

    python Phase_4_Network_Latency/analyze_phase4_runs.py --runs-glob "Phase_4_Network_Latency/runs/*"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

# Le script est lancé directement (et non comme module du paquet) : on ajoute
# la racine du dépôt au sys.path pour que l'import absolu
# "Phase_4_Network_Latency.latency_metrics" fonctionne quel que soit le
# répertoire courant d'exécution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Phase_4_Network_Latency.latency_metrics import summarize_phase3_run


def percentile(values: list[float], pct: float) -> float:
    """Retourne le percentile ``pct`` (0 à 100) par la méthode du rang le plus proche, 0.0 si vide."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def _read_manifest(run_dir: Path) -> dict:
    """Lit le ``manifest.json`` d'un run (paramètres de capture), ou un dictionnaire vide s'il est absent."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _summarize_alert_latency(run_dir: Path) -> dict[str, str | int | float] | None:
    """Résume un run de benchmark d'alertes à partir de son ``alert_latency.csv``.

    Retourne une ligne de synthèse (transport, nombre d'événements, latences
    moyenne/médiane/p95/max en ms) ou None si le CSV est absent, auquel cas le
    run est traité comme une campagne Phase 3. Les lignes dont la latence est
    illisible sont ignorées.
    """
    csv_path = run_dir / "alert_latency.csv"
    if not csv_path.exists():
        return None

    latencies: list[float] = []
    transport = run_dir.name
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("transport"):
                transport = row["transport"]
            try:
                latencies.append(float(row["latency_ms"]))
            except (TypeError, ValueError):
                continue

    return {
        "run_dir": str(run_dir),
        "summary_type": "alert_delivery",
        "transport": transport,
        "events": len(latencies),
        "latency_mean_ms": round(statistics.mean(latencies), 6) if latencies else 0.0,
        "latency_median_ms": round(statistics.median(latencies), 6) if latencies else 0.0,
        "latency_p95_ms": round(percentile(latencies, 95.0), 6),
        "latency_max_ms": round(max(latencies), 6) if latencies else 0.0,
    }


def analyze_runs(run_dirs: list[Path], expected_cameras: list[str] | None = None) -> list[dict[str, str | int | float]]:
    """Produit une ligne de synthèse par dossier de run.

    Chaque dossier est d'abord testé comme run de benchmark d'alertes
    (``alert_latency.csv``), sinon comme campagne Phase 3 (bilan
    :func:`summarize_phase3_run` enrichi du manifeste). Les fichiers simples et
    les dossiers sans CSV exploitables sont ignorés.
    """
    rows: list[dict[str, str | int | float]] = []
    for run_dir in run_dirs:
        if run_dir.is_file():
            continue
        # Cas 1 : run de benchmark de livraison d'alertes.
        alert_summary = _summarize_alert_latency(run_dir)
        if alert_summary:
            rows.append(alert_summary)
            continue

        # Cas 2 : campagne live Phase 3 (les deux CSV sont requis).
        summary_path = run_dir / "phase3" / "summary.csv"
        sync_path = run_dir / "phase3" / "sync_events.csv"
        if not summary_path.exists() or not sync_path.exists():
            continue

        manifest = _read_manifest(run_dir)
        health = summarize_phase3_run(run_dir, expected_cameras=expected_cameras)
        rows.append(
            {
                "run_dir": str(run_dir),
                "summary_type": "phase3_campaign",
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
                "detections": health.detections,
                "latency_mean_ms": health.latency_mean_ms,
                "latency_p95_ms": health.latency_p95_ms,
                "latency_max_ms": health.latency_max_ms,
                "fusion_links": health.fusion_links,
                "frames_with_fusion": health.frames_with_fusion,
                "fusion_frame_rate_pct": round(health.fusion_frame_rate_pct, 3),
                "fusion_links_per_frame": round(health.fusion_links_per_frame, 6),
                "fusion_links_per_1000_detections": round(health.fusion_links_per_1000_detections, 3),
                "fusion_camera_pairs": health.fusion_camera_pairs,
                "top_fusion_pair": health.top_fusion_pair,
                "top_fusion_pair_links": health.top_fusion_pair_links,
                "fusion_active": int(health.has_fusion),
                "alerts": health.alerts,
                "dropped_cameras": ",".join(health.dropped_cameras),
                "camera_count": len(health.cameras),
            }
        )
    return rows


def write_rows(rows: list[dict[str, str | int | float]], out_csv: Path) -> None:
    """Écrit les lignes de synthèse dans le CSV de sortie.

    L'en-tête est l'union triée des champs de toutes les lignes (les deux types
    de runs n'ont pas les mêmes colonnes), avec ``run_dir`` remis en première
    position ; les cellules absentes restent vides.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({field for row in rows for field in row.keys()})
        if "run_dir" in fields:
            fields.remove("run_dir")
            fields.insert(0, "run_dir")
    else:
        fields = ["run_dir"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Analyse les options de la ligne de commande de l'agrégateur."""
    parser = argparse.ArgumentParser(description="Synthèse de santé des campagnes Phase 4 / Phase 3.")
    parser.add_argument(
        "--runs-glob",
        default="Phase_3_Fusion_MultiCam/reports/campaign_zone1_live_*",
        help="Motif glob des dossiers de runs à agréger.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("Phase_4_Network_Latency/phase4_run_health.csv"),
        help="Chemin du CSV de synthèse (une ligne par run).",
    )
    parser.add_argument(
        "--cameras",
        default="cam_02,cam_03,cam_05,cam_07",
        help="Caméras attendues, séparées par des virgules (les absentes sont comptées comme décrochées).",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : résout le glob, analyse chaque run et écrit le CSV de synthèse."""
    args = parse_args()
    run_dirs = sorted(Path(path) for path in glob.glob(args.runs_glob))
    cameras = [part.strip() for part in args.cameras.split(",") if part.strip()]
    rows = analyze_runs(run_dirs, expected_cameras=cameras)
    write_rows(rows, args.out_csv)
    print(f"[INFO] Wrote {len(rows)} run summaries to {args.out_csv}")


if __name__ == "__main__":
    main()

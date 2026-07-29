"""
Phase 3, campagne live RTSP bornée dans le temps (zone_1).

Point d'entrée des campagnes en conditions réelles : le pipeline complet de la
Phase 3 (capture RTSP, détection YOLO, suivi ByteTrack, fusion multi-caméras,
alertes) tourne pendant une durée fixée (--duration-min) pour chaque modèle
découvert, sur les caméras de la zone_1.

Par rapport à la campagne enregistrée (run_recorded_campaign.py), ce script
ajoute les aspects propres au direct : choix du backend de capture
(GStreamer ou OpenCV, options --gst-*), enregistrement des flux pour rejeu
ultérieur, publication de métadonnées temps réel (JSONL ou HTTP) et trace de
latence détaillée par frame. Les mêmes post-traitements sont ensuite
appliqués : métriques TAD/TRD, ablation du seuil de fusion D et audit de
calibration, agrégés dans un dossier de campagne horodaté.

Exemples :
  python run_live_campaign.py --duration-min 10 --no-display
  python run_live_campaign.py --versions V4 --formats pt --duration-min 5 \\
      --metadata-jsonl reports/live_metadata.jsonl --latency-trace-csv reports/latency.csv
"""

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
from metadata_publisher import MetadataPublisher
from video_capture import CaptureOptions


def parse_args() -> argparse.Namespace:
    """Déclare et analyse les options de la campagne live RTSP."""
    parser = argparse.ArgumentParser(
        description="Bounded live RTSP Phase 3 campaign.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run_live_campaign.py --duration-min 10 --no-display
  python run_live_campaign.py --versions V4 --formats pt --duration-min 5 \\
      --metadata-jsonl reports/live_metadata.jsonl --latency-trace-csv reports/latency.csv
""",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Chemin du config.yaml Phase 3 (caméras RTSP, homographies, zones interdites). Défaut : config.yaml du dossier Phase 3.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Dossier de sortie de la campagne. Défaut : dossier horodaté reports/campaign_zone1_live_<date>.",
    )
    parser.add_argument(
        "--cameras",
        default=",".join(ZONE1_CAMERAS),
        help="Caméras à utiliser, séparées par des virgules. Défaut : les caméras de la zone_1 (cam_02,cam_03,cam_05,cam_07).",
    )
    parser.add_argument(
        "--versions",
        default="V2,V3",
        help="Versions de datasets/modèles entraînés à évaluer, séparées par des virgules. Défaut : V2,V3.",
    )
    parser.add_argument(
        "--formats",
        default="pt,fp32_engine",
        help="Formats de poids à évaluer, séparés par des virgules (pt, fp32_engine). Défaut : pt,fp32_engine.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Noms de modèles à évaluer, séparés par des virgules (ex. yolo11n,yolo11s). Défaut : tous les modèles découverts.",
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=10.0,
        help="Durée de capture live par modèle, en minutes. Défaut : 10.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device YOLO, ex. cuda:0 ou cpu. Défaut : sélection automatique d'Ultralytics.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Seuil de confiance des détections. Défaut : 0.5.",
    )
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Ne pas enregistrer les flux RTSP pendant la campagne (par défaut, ils sont sauvegardés pour rejeu).",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=25.0,
        help="FPS d'écriture des vidéos live enregistrées. Défaut : 25.",
    )
    parser.add_argument(
        "--tad-gt",
        type=Path,
        default=DEFAULT_GT_OBJECTS_TAD,
        help="JSON de vérité terrain TAD (apparitions d'objets) utilisé pour les métriques a posteriori.",
    )
    parser.add_argument(
        "--fusion-distance-m",
        type=float,
        default=1.0,
        help="Seuil de distance D de l'association inter-caméras, en mètres. Défaut : 1.0.",
    )
    parser.add_argument(
        "--fusion-time-window-ms",
        type=float,
        default=500.0,
        help="Fenêtre temporelle de la fusion, en millisecondes : deux détections plus éloignées dans le temps ne sont pas associées. Défaut : 500.",
    )
    parser.add_argument(
        "--person-zone-vote-ratio",
        type=float,
        default=None,
        help="Ratio minimal de caméras votantes pour confirmer une violation de zone par une personne (0 à 1). Défaut : valeur du config.yaml.",
    )
    parser.add_argument(
        "--person-zone-min-votes",
        type=int,
        default=None,
        help="Nombre minimal de caméras votantes pour confirmer une violation de zone par une personne. Défaut : valeur du config.yaml.",
    )
    parser.add_argument(
        "--object-min-camera-votes",
        type=int,
        default=None,
        help="Nombre minimal de caméras devant voir un objet pour confirmer son alerte. Défaut : valeur du config.yaml.",
    )
    parser.add_argument(
        "--no-weak-object-alerts",
        action="store_true",
        help="N'exporter que les alertes objet confirmées multi-caméras (supprime les alertes faibles mono-caméra).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Désactive l'affichage vidéo pendant la campagne (recommandé sur machine sans écran).",
    )
    parser.add_argument(
        "--display-mode",
        choices=["annotated", "raw"],
        default="annotated",
        help="Mode d'affichage live quand --no-display n'est pas utilisé : annotated (boîtes et alertes incrustées) ou raw (flux brut). Défaut : annotated.",
    )
    parser.add_argument(
        "--include-rtdetr-phase3",
        action="store_true",
        help="Inclure aussi les moteurs TensorRT RT-DETR. Le format .pt de RT-DETR reste inclus par défaut.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=["opencv", "gstreamer"],
        default="gstreamer",
        help="Backend de capture des flux RTSP : gstreamer (recommandé, latence maîtrisée) ou opencv. Défaut : gstreamer.",
    )
    parser.add_argument(
        "--gst-latency-ms",
        type=int,
        default=100,
        help="Latence du tampon rtspsrc GStreamer, en millisecondes. Plus bas réduit le retard mais fragilise le flux. Défaut : 100.",
    )
    parser.add_argument(
        "--gst-protocol",
        choices=["tcp", "udp"],
        default="tcp",
        help="Protocole de transport RTSP côté GStreamer : tcp (fiable) ou udp (plus léger, pertes possibles). Défaut : tcp.",
    )
    parser.add_argument(
        "--gst-codec",
        choices=["h264", "h265"],
        default="h265",
        help="Codec vidéo attendu du flux RTSP (dépayload et décodage). Défaut : h265, codec des caméras du site.",
    )
    parser.add_argument(
        "--gst-decoder",
        default="decodebin",
        help="Élément décodeur GStreamer à utiliser dans le pipeline fixe, ex. decodebin, avdec_h265, nvh265dec. Défaut : decodebin.",
    )
    parser.add_argument(
        "--gst-pipeline",
        choices=["decodebin", "fixed", "both"],
        default="decodebin",
        help="Stratégie de construction du pipeline GStreamer : decodebin (négociation automatique), fixed (chaîne explicite depay/parse/décodeur) ou both (essaie les deux dans l'ordre). Défaut : decodebin.",
    )
    parser.add_argument(
        "--no-ffmpeg-fallback",
        action="store_true",
        help="Désactive le repli automatique sur FFmpeg/OpenCV quand l'ouverture GStreamer échoue.",
    )
    parser.add_argument(
        "--fusion-truth-csv",
        type=Path,
        default=None,
        help="CSV de détections annotées (colonne truth_id) : si fourni et existant, l'ablation du seuil D est rejouée sur cette vérité terrain plutôt que sur les liens prédits.",
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        default=None,
        help="Écrit les enveloppes de métadonnées live Phase 3 (détections, alertes) dans ce fichier JSONL. Défaut : pas d'export.",
    )
    parser.add_argument(
        "--metadata-http-url",
        default=None,
        help="Envoie les enveloppes de métadonnées live Phase 3 en POST vers cette URL HTTP. Défaut : pas d'envoi.",
    )
    parser.add_argument(
        "--metadata-every-n-frames",
        type=int,
        default=1,
        help="Publie une enveloppe de métadonnées toutes les N frames traitées. Défaut : 1 (chaque frame).",
    )
    parser.add_argument(
        "--latency-trace-csv",
        type=Path,
        default=None,
        help=(
            "Écrit une trace de latence par frame dans ce CSV : temps de lecture capture, "
            "inférence/suivi, fusion, alertes, métadonnées, enregistrement et affichage. "
            "Défaut : pas de trace."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Orchestre la campagne live : préparation, run par modèle, agrégation.

    Déroulé : découverte des modèles, écriture du manifeste, puis pour chaque
    modèle un run live borné (capture RTSP, fusion, alertes) suivi des
    post-traitements (métriques, ablation, audit). Les CSV des différents
    runs sont enfin fusionnés au niveau campagne.
    """
    args = parse_args()
    campaign_dir = args.out_dir or timestamped_campaign_dir("campaign_zone1_live")
    # Résolution des listes (caméras, versions, formats, modèles) depuis les
    # arguments CSV, avec les valeurs zone_1 par défaut.
    cameras = parse_csv_arg(args.cameras, ZONE1_CAMERAS)
    versions = parse_csv_arg(args.versions, ("V2", "V3"))
    formats = parse_csv_arg(args.formats, ("pt", "fp32_engine"))
    models = parse_csv_arg(args.models, ()) if args.models else None
    specs = discover_model_specs(versions=versions, formats=formats, model_names=models)
    if not specs:
        raise SystemExit("[ERREUR] Aucun modele decouvert.")

    # Le manifeste fige la configuration exacte de la campagne (modèles,
    # options de capture, alerting, métadonnées) pour audit ultérieur.
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
            "gst_protocol": args.gst_protocol,
            "gst_latency_ms": args.gst_latency_ms,
            "gst_decoder": args.gst_decoder,
            "gst_pipeline": args.gst_pipeline,
            "ffmpeg_fallback": not args.no_ffmpeg_fallback,
            "record_video": not args.no_record_video,
            "record_fps": args.record_fps,
            "display_mode": args.display_mode,
            "alerting": {
                "person_zone_min_camera_ratio": args.person_zone_vote_ratio,
                "person_zone_min_camera_votes": args.person_zone_min_votes,
                "object_min_camera_votes": args.object_min_camera_votes,
                "object_emit_weak_alerts": not args.no_weak_object_alerts,
            },
            "metadata": {
                "jsonl": str(args.metadata_jsonl) if args.metadata_jsonl else "",
                "http_url": args.metadata_http_url or "",
                "every_n_frames": args.metadata_every_n_frames,
            },
            "latency_trace_csv": str(args.latency_trace_csv) if args.latency_trace_csv else "",
        },
    )

    # Options de capture partagées par toutes les caméras du run.
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
        # Les moteurs TensorRT RT-DETR sont exclus par défaut (ByteTrack +
        # TensorRT produit des bbox NaN). Une ligne de résumé neutre est tout
        # de même émise pour garder un summary.csv complet.
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
        # Quand plusieurs modèles sont évalués, chaque run écrit dans son
        # propre fichier de métadonnées (suffixe = label du run) pour éviter
        # d'entremêler les enveloppes.
        metadata_jsonl = args.metadata_jsonl
        if metadata_jsonl and len(specs) > 1:
            metadata_jsonl = metadata_jsonl.with_name(
                f"{metadata_jsonl.stem}_{spec.run_label}{metadata_jsonl.suffix or '.jsonl'}"
            )
        metadata_publisher = None
        if metadata_jsonl or args.metadata_http_url:
            metadata_publisher = MetadataPublisher(
                jsonl_path=metadata_jsonl,
                http_url=args.metadata_http_url,
                every_n_frames=args.metadata_every_n_frames,
            )
        # Même dédoublonnage pour la trace de latence par frame.
        latency_trace_csv = args.latency_trace_csv
        if latency_trace_csv and len(specs) > 1:
            latency_trace_csv = latency_trace_csv.with_name(
                f"{latency_trace_csv.stem}_{spec.run_label}{latency_trace_csv.suffix or '.csv'}"
            )
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
            metadata_publisher=metadata_publisher,
            latency_trace_path=latency_trace_csv,
        )
        # La sortie standard du run est redirigée vers un log dédié : ce log
        # sert ensuite d'entrée à l'audit de calibration (lignes [ALERT]).
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
        # Métriques TAD/TRD Phase 3 : en live, l'identifiant de caméra de la
        # vérité terrain est le même que celui du pipeline (carte identité).
        compute_phase3_metrics(
            campaign,
            cameras,
            {cam: cam for cam in cameras},
            run_dir,
            tad_gt_path=args.tad_gt,
        )
        # Ablation opérationnelle : rejoue la fusion sur les détections du run
        # pour D dans {50, 100, 150, 200} cm (sans vérité terrain).
        run_operational_ablation(
            campaign=campaign,
            config=campaign.config,
            out_csv=run_dir / "ablation" / "fusion_threshold_ablation.csv",
        )
        run_audit(log_path, args.config, run_dir / "audit" / "calibration_alert_audit.csv")
        summary_rows.append(summary)

    # Concatène les CSV homonymes de chaque run de modèle au niveau campagne
    # (sync_events.csv est propre au mode live : événements de synchronisation).
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

    # Ablation du seuil D : sur vérité terrain annotée si --fusion-truth-csv
    # est fourni, sinon simple fusion des ablations opérationnelles par run.
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

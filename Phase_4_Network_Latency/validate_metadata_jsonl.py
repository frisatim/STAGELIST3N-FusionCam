"""Validation du schéma des métadonnées JSONL produites par la campagne live Phase 3.

Le pipeline Phase 3 publie une enveloppe JSON par trame (schéma
``benchmarkingai.phase3.metadata.v1``) contenant les détections et les alertes.
Ce script relit un fichier JSONL de campagne et vérifie sa structure :

- présence des champs attendus dans chaque enveloppe, détection et alerte ;
- validité des boîtes englobantes et présence des identifiants globaux ;
- monotonie des numéros de trame (détection des retours en arrière) ;
- régularité de l'intervalle de publication (moyenne, p95) ;
- estimation du retard entre les horodatages sources et l'enveloppe.

Exemple::

    python Phase_4_Network_Latency/validate_metadata_jsonl.py chemin/vers/metadata.jsonl --print-example

Le script affiche un résumé et signale par un [WARN] tout écart de structure.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


# Champs attendus au niveau de l'enveloppe (une enveloppe = une trame publiée).
ENVELOPE_FIELDS = {
    "schema",
    "created_epoch_ms",
    "created_epoch_s",
    "frame",
    "run_label",
    "model_version",
    "model",
    "format",
    "detections",
    "alerts",
}

# Champs attendus pour chaque détection de la liste "detections".
DETECTION_FIELDS = {
    "camera_id",
    "track_id",
    "global_id",
    "class_id",
    "class_name",
    "confidence",
    "bbox_px",
    "foot_point_px",
    "position_m",
    "zones",
    "timestamp",
}

# Champs attendus pour chaque alerte de la liste "alerts".
ALERT_FIELDS = {
    "alert_id",
    "alert_type",
    "alert_level",
    "global_id",
    "zone_id",
    "class_name",
    "position_m",
    "cameras",
    "confidence",
    "timestamp",
}


def _pct(values: list[float], pct: float) -> float:
    """Retourne le percentile demandé par la méthode du rang le plus proche.

    Attention : ``pct`` est ici une fraction entre 0 et 1 (0.95 pour le p95),
    contrairement aux autres scripts de la Phase 4 qui prennent une valeur sur
    100. Retourne 0.0 sur une liste vide.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def _mean(values: list[float]) -> float:
    """Moyenne arithmétique, ou 0.0 sur une liste vide."""
    return statistics.fmean(values) if values else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lit un fichier JSONL et retourne la liste des objets, une entrée par ligne non vide.

    Arrête le script avec un message d'erreur (numéro de ligne inclus) à la
    première ligne dont le JSON est invalide.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[ERREUR] JSON invalide ligne {line_no}: {exc}") from exc
    return rows


def analyze(path: Path) -> dict[str, Any]:
    """Analyse un fichier JSONL de métadonnées et retourne un dictionnaire de statistiques.

    Parcourt chaque enveloppe (une par trame publiée) et cumule : les comptages
    (détections, alertes, caméras, classes, types d'alerte), les contrôles de
    structure (champs manquants, boîtes valides, identifiants globaux, trames
    en arrière) et les métriques temporelles (intervalle de publication en ms,
    retard estimé des métadonnées en ms). Conserve aussi une enveloppe
    d'exemple contenant une détection et une contenant une alerte, pour
    l'option ``--print-example``.
    """
    rows = _read_jsonl(path)
    # Compteurs d'écarts de structure.
    missing_envelope = 0        # enveloppes auxquelles il manque au moins un champ
    missing_detection = 0       # détections incomplètes
    missing_alert = 0           # alertes incomplètes
    # Comptages globaux.
    detections = 0
    alerts = 0
    cameras: set[str] = set()   # identifiants de caméras rencontrés
    classes: set[str] = set()   # noms de classes rencontrés
    alert_types: set[str] = set()
    global_ids = 0              # détections portant un global_id (fusion multi-caméras)
    bbox_ok = 0                 # détections dont la bbox_px est une liste de 4 valeurs
    frame_backwards = 0         # numéros de trame qui reculent (désordre de publication)
    # Séries temporelles pour les statistiques de cadence et de retard.
    intervals_ms: list[float] = []       # écarts entre created_epoch_ms consécutifs
    metadata_lags_ms: list[float] = []   # retard enveloppe vs horodatage source le plus récent
    previous_frame: int | None = None
    previous_created_ms: float | None = None
    example_with_detection: dict[str, Any] | None = None
    example_with_alert: dict[str, Any] | None = None

    for row in rows:
        # Contrôle 1 : tous les champs d'enveloppe attendus sont présents
        # (différence ensembliste non vide = au moins un champ manquant).
        if ENVELOPE_FIELDS - row.keys():
            missing_envelope += 1

        # Contrôle 2 : monotonie des numéros de trame. Un numéro inférieur au
        # précédent trahit un désordre de publication ou un redémarrage.
        frame = row.get("frame")
        if isinstance(frame, int):
            if previous_frame is not None and frame < previous_frame:
                frame_backwards += 1
            previous_frame = frame

        # Contrôle 3 : intervalle entre publications successives, mesuré sur
        # l'horodatage de création de l'enveloppe (en ms).
        created_ms = row.get("created_epoch_ms")
        if isinstance(created_ms, (int, float)):
            if previous_created_ms is not None:
                intervals_ms.append(float(created_ms) - previous_created_ms)
            previous_created_ms = float(created_ms)

        # Horodatage source le plus récent de l'enveloppe (détections et
        # alertes confondues), pour estimer le retard de publication plus bas.
        newest_source_ts = 0.0

        # Contrôles par détection.
        for det in row.get("detections", []) or []:
            detections += 1
            # Champs de détection manquants.
            if DETECTION_FIELDS - det.keys():
                missing_detection += 1
            # Inventaire des caméras vues.
            camera_id = det.get("camera_id")
            if camera_id:
                cameras.add(str(camera_id))
            # Inventaire des classes détectées.
            class_name = det.get("class_name")
            if class_name:
                classes.add(str(class_name))
            # Présence d'un identifiant global (résultat de la fusion Phase 3).
            if det.get("global_id") is not None:
                global_ids += 1
            # Boîte englobante bien formée : liste [x1, y1, x2, y2].
            bbox = det.get("bbox_px")
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox_ok += 1
            # Mise à jour de l'horodatage source le plus récent.
            ts = det.get("timestamp")
            if isinstance(ts, (int, float)):
                newest_source_ts = max(newest_source_ts, float(ts))
            # Première enveloppe contenant une détection, gardée comme exemple.
            if example_with_detection is None:
                example_with_detection = row

        # Contrôles par alerte.
        for alert in row.get("alerts", []) or []:
            alerts += 1
            # Champs d'alerte manquants.
            if ALERT_FIELDS - alert.keys():
                missing_alert += 1
            # Inventaire des types d'alerte vus.
            alert_type = alert.get("alert_type")
            if alert_type:
                alert_types.add(str(alert_type))
            # Les caméras citées par l'alerte alimentent le même inventaire.
            for camera_id in alert.get("cameras", []) or []:
                cameras.add(str(camera_id))
            # Mise à jour de l'horodatage source le plus récent.
            ts = alert.get("timestamp")
            if isinstance(ts, (int, float)):
                newest_source_ts = max(newest_source_ts, float(ts))
            # Première enveloppe contenant une alerte, gardée comme exemple.
            if example_with_alert is None:
                example_with_alert = row

        # Contrôle 4 : retard de l'enveloppe par rapport à sa source la plus
        # récente (created_epoch_s en secondes, converti en ms). Une valeur
        # élevée signale un pipeline qui publie en retard sur la capture.
        created_s = row.get("created_epoch_s")
        if newest_source_ts and isinstance(created_s, (int, float)):
            metadata_lags_ms.append((float(created_s) - newest_source_ts) * 1000.0)

    return {
        "path": str(path),
        "envelopes": len(rows),
        "detections": detections,
        "alerts": alerts,
        "cameras": sorted(cameras),
        "classes": sorted(classes),
        "alert_types": sorted(alert_types),
        "bbox_ok": bbox_ok,
        "global_ids": global_ids,
        "missing_envelope": missing_envelope,
        "missing_detection": missing_detection,
        "missing_alert": missing_alert,
        "frame_backwards": frame_backwards,
        "publish_interval_mean_ms": round(_mean(intervals_ms), 3),
        "publish_interval_p95_ms": round(_pct(intervals_ms, 0.95), 3),
        "metadata_lag_mean_ms": round(_mean(metadata_lags_ms), 3),
        "metadata_lag_p95_ms": round(_pct(metadata_lags_ms, 0.95), 3),
        "example_with_detection": example_with_detection,
        "example_with_alert": example_with_alert,
    }


def print_summary(result: dict[str, Any], print_example: bool = False) -> None:
    """Affiche le résumé de l'analyse sur la sortie standard.

    Les comptages et métriques sont listés en [INFO] ; les écarts de structure
    (champs manquants, trames en arrière) déclenchent une ligne [WARN]. Avec
    ``print_example``, affiche aussi une enveloppe d'exemple complète (alerte
    de préférence, sinon détection).
    """
    print(f"[INFO] JSONL: {result['path']}")
    print(f"[INFO] Enveloppes: {result['envelopes']}")
    print(f"[INFO] Detections: {result['detections']} | bbox OK: {result['bbox_ok']} | global_id presents: {result['global_ids']}")
    print(f"[INFO] Alertes: {result['alerts']}")
    print(f"[INFO] Cameras: {', '.join(result['cameras']) or '-'}")
    print(f"[INFO] Classes: {', '.join(result['classes']) or '-'}")
    print(f"[INFO] Types alertes: {', '.join(result['alert_types']) or '-'}")
    print(
        "[INFO] Intervalle publication: "
        f"mean={result['publish_interval_mean_ms']}ms "
        f"p95={result['publish_interval_p95_ms']}ms"
    )
    print(
        "[INFO] Lag metadata estime: "
        f"mean={result['metadata_lag_mean_ms']}ms "
        f"p95={result['metadata_lag_p95_ms']}ms"
    )
    problems = {
        "enveloppes_incompletes": result["missing_envelope"],
        "detections_incompletes": result["missing_detection"],
        "alertes_incompletes": result["missing_alert"],
        "frames_en_arriere": result["frame_backwards"],
    }
    if any(problems.values()):
        print(f"[WARN] Problemes detectes: {problems}")
    else:
        print("[INFO] Structure JSONL OK")

    if print_example:
        example = result["example_with_alert"] or result["example_with_detection"]
        if example:
            print("[INFO] Exemple:")
            print(json.dumps(example, indent=2, ensure_ascii=False))
        else:
            print("[INFO] Aucun exemple avec detection/alerte trouve.")


def parse_args() -> argparse.Namespace:
    """Analyse les options de la ligne de commande du validateur."""
    parser = argparse.ArgumentParser(description="Valide le JSONL de métadonnées live de la Phase 3.")
    parser.add_argument("jsonl", type=Path, help="Chemin du fichier JSONL de métadonnées à valider.")
    parser.add_argument(
        "--print-example",
        action="store_true",
        help="Affiche en plus une enveloppe d'exemple complète (avec alerte de préférence).",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : vérifie l'existence du fichier puis affiche le résumé de l'analyse."""
    args = parse_args()
    if not args.jsonl.exists():
        raise SystemExit(f"[ERREUR] Fichier introuvable: {args.jsonl}")
    print_summary(analyze(args.jsonl), print_example=args.print_example)


if __name__ == "__main__":
    main()

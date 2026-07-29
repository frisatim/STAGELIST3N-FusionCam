"""
Ablation du seuil de distance D de l'association inter-caméras (Phase 3).

Le plan de recherche demande de tester D dans {50, 100, 150, 200} cm et de
mesurer les fausses correspondances (FP) et les correspondances manquées (FN).
Ce script fait exactement cette évaluation, sans avoir besoin des caméras
live : il rejoue la fusion (MultiCameraFusion) sur des détections déjà
positionnées au sol, pour chaque valeur de D, et compare les liens produits à
la vérité terrain portée par la colonne truth_id.

Deux modes :
  1. Synthétique (sans --input) : jeu de détections contrôlé, utilisable
     immédiatement pour vérifier le compromis FP/FN.
  2. CSV (--input) : détections exportées depuis les vidéos, annotées avec
     truth_id. C'est le mode utilisé par les campagnes via --fusion-truth-csv.

Format CSV attendu :
  frame,timestamp,cam_id,track_id,x_m,y_m,truth_id,confidence,class_name

Exemples :
  python ablate_fusion_threshold.py
  python ablate_fusion_threshold.py --input detections_fusion_gt.csv --output reports/fusion_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from detection import Detection
from fusion import MultiCameraFusion


# Valeurs de D testées par défaut (en centimètres), imposées par le plan de recherche.
DEFAULT_THRESHOLDS_CM = [50, 100, 150, 200]
DEFAULT_OUTPUT = Path("reports/fusion_threshold_ablation.csv")


def _make_detection(row: dict[str, Any]) -> Detection:
    """Convertit une ligne CSV en objet Detection minimal pour la fusion.

    Seule la position au sol en mètres compte ici : les champs pixels (bbox,
    point pied) sont mis à zéro car la fusion ne les utilise pas.
    """
    x_m = float(row["x_m"])
    y_m = float(row["y_m"])
    return Detection(
        cam_id=str(row["cam_id"]),
        track_id=int(row["track_id"]),
        global_id=None,
        timestamp=float(row["timestamp"]),
        bbox_px=(0.0, 0.0, 0.0, 0.0),
        foot_point_px=(0.0, 0.0),
        foot_point_m=(x_m, y_m),
        confidence=float(row.get("confidence", 1.0)),
        class_id=0,
        class_name=str(row.get("class_name", "person")),
    )


def _synthetic_rows() -> list[dict[str, Any]]:
    """Petit jeu de données étiqueté qui rend visible le compromis sur D.

    - person_A est vue par cam_03/cam_05/cam_07 avec des écarts de 0,42 m et 0,85 m.
    - person_B est vue par cam_03/cam_07 avec un écart de 1,35 m.
    - person_C et person_D sont deux personnes différentes mais assez proches
      pour devenir de fausses correspondances quand D est trop permissif.
    """
    rows: list[dict[str, Any]] = []

    def add(frame: int, cam_id: str, track_id: int, x: float, y: float, truth_id: str):
        rows.append(
            {
                "frame": frame,
                "timestamp": frame / 25.0,
                "cam_id": cam_id,
                "track_id": track_id,
                "x_m": x,
                "y_m": y,
                "truth_id": truth_id,
                "confidence": 0.9,
                "class_name": "person",
            }
        )

    # 10 frames avec une légère dérive pour simuler un déplacement lent.
    for frame in range(10):
        drift = frame * 0.02
        add(frame, "cam_03", 1, 5.00 + drift, 2.00, "person_A")
        add(frame, "cam_05", 8, 5.42 + drift, 2.03, "person_A")
        add(frame, "cam_07", 4, 5.85 + drift, 2.02, "person_A")

        add(frame, "cam_03", 2, 8.00, 1.00 + drift, "person_B")
        add(frame, "cam_07", 5, 9.35, 1.05 + drift, "person_B")

        add(frame, "cam_05", 9, 2.00, 5.00 + drift, "person_C")
        add(frame, "cam_07", 6, 2.95, 5.05 + drift, "person_D")

    return rows


def _load_rows(input_path: Path | None) -> list[dict[str, Any]]:
    """Charge les détections annotées : jeu synthétique si aucun CSV fourni.

    Vérifie la présence des colonnes obligatoires et échoue avec un message
    explicite sinon.
    """
    if input_path is None:
        return _synthetic_rows()

    with open(input_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"frame", "timestamp", "cam_id", "track_id", "x_m", "y_m", "truth_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV invalide, colonnes manquantes: {sorted(missing)}")
        return list(reader)


def _load_config(config_path: Path) -> dict:
    """Charge le config.yaml Phase 3 (dictionnaire vide si fichier vide)."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _group_rows_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Regroupe les lignes par numéro de frame, frames triées par ordre croissant."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame"])].append(row)
    return dict(sorted(grouped.items()))


def _rows_to_detections_by_cam(rows: list[dict[str, Any]]) -> tuple[dict[str, list[Detection]], dict[tuple[str, int], str]]:
    """Prépare l'entrée de la fusion pour une frame.

    Retourne les détections regroupées par caméra (format attendu par
    MultiCameraFusion.associate) et la table (cam_id, track_id) -> truth_id
    servant de vérité terrain lors du scoring.
    """
    detections_by_cam: dict[str, list[Detection]] = defaultdict(list)
    truth_by_key: dict[tuple[str, int], str] = {}

    for row in rows:
        det = _make_detection(row)
        detections_by_cam[det.cam_id].append(det)
        truth_by_key[(det.cam_id, det.track_id)] = str(row["truth_id"])

    return dict(detections_by_cam), truth_by_key


def _score_frame(detections: list[Detection], truth_by_key: dict[tuple[str, int], str]) -> dict[str, int]:
    """Compte TP/FP/FN/TN sur toutes les paires inter-caméras d'une frame.

    Pour chaque paire de détections issues de caméras différentes :
      - lien vérité : les deux track_id partagent le même truth_id non vide ;
      - lien prédit : la fusion leur a attribué le même global_id.
    Le croisement des deux donne la classification TP/FP/FN/TN.
    """
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "predicted_links": 0, "truth_links": 0}

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            # Les paires intra-caméra ne sont pas des associations : ignorées.
            if a.cam_id == b.cam_id:
                continue

            truth_a = truth_by_key[(a.cam_id, a.track_id)]
            truth_b = truth_by_key[(b.cam_id, b.track_id)]
            truth_link = truth_a != "" and truth_a == truth_b
            pred_link = (
                a.global_id is not None
                and b.global_id is not None
                and a.global_id == b.global_id
            )

            if pred_link:
                counts["predicted_links"] += 1
            if truth_link:
                counts["truth_links"] += 1

            if pred_link and truth_link:
                counts["tp"] += 1
            elif pred_link and not truth_link:
                counts["fp"] += 1
            elif not pred_link and truth_link:
                counts["fn"] += 1
            else:
                counts["tn"] += 1

    return counts


def _safe_div(num: float, den: float) -> float:
    """Division protégée : retourne 0.0 si le dénominateur est nul."""
    return num / den if den else 0.0


def _summarize(threshold_cm: int, counts: dict[str, int], frames: int, global_ids: set[int]) -> dict[str, Any]:
    """Construit la ligne de résumé d'un seuil D : métriques dérivées des comptes.

    En plus de précision/rappel/F1, expose le taux de fausses correspondances
    (FP / (FP + TN)) et le taux de correspondances manquées (FN / (TP + FN))
    demandés par le plan de recherche.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold_cm": threshold_cm,
        "threshold_m": threshold_cm / 100.0,
        "frames": frames,
        "tp": tp,
        "fp_false_matches": fp,
        "fn_missed_matches": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_match_rate": round(_safe_div(fp, fp + tn), 4),
        "missed_match_rate": round(_safe_div(fn, tp + fn), 4),
        "predicted_links": counts["predicted_links"],
        "truth_links": counts["truth_links"],
        "unique_global_ids": len(global_ids),
    }


def run_ablation(
    rows: list[dict[str, Any]],
    config: dict,
    thresholds_cm: list[int],
    time_window_s: float,
) -> list[dict[str, Any]]:
    """Rejoue la fusion pour chaque seuil D et agrège les scores sur toutes les frames.

    Pour chaque valeur de D : instancie une fusion neuve (état vierge), passe
    les frames dans l'ordre, cumule les comptes TP/FP/FN/TN par frame et
    collecte les global_id attribués. Retourne une ligne de résumé par seuil.
    """
    grouped = _group_rows_by_frame(rows)
    summaries: list[dict[str, Any]] = []

    for threshold_cm in thresholds_cm:
        fusion = MultiCameraFusion(
            config=config,
            distance_threshold_m=threshold_cm / 100.0,
            time_window_s=time_window_s,
        )
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "predicted_links": 0, "truth_links": 0}
        global_ids: set[int] = set()

        for frame_rows in grouped.values():
            detections_by_cam, truth_by_key = _rows_to_detections_by_cam(frame_rows)
            fused = fusion.associate(detections_by_cam)
            for det in fused:
                if det.global_id is not None:
                    global_ids.add(det.global_id)
            frame_counts = _score_frame(fused, truth_by_key)
            for key, value in frame_counts.items():
                counts[key] += value

        summaries.append(_summarize(threshold_cm, counts, len(grouped), global_ids))

    return summaries


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Écrit les lignes de résumé dans le CSV de sortie (dossier créé au besoin)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_thresholds(raw: str) -> list[int]:
    """Analyse la liste de seuils --thresholds-cm (entiers séparés par des virgules)."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    """Point d'entrée : charge les détections, lance l'ablation, écrit et affiche le résumé."""
    parser = argparse.ArgumentParser(
        description="Ablation du seuil D pour la fusion multi-cameras Phase 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python ablate_fusion_threshold.py
  python ablate_fusion_threshold.py --input detections_fusion_gt.csv --output reports/fusion_ablation.csv
""",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV de détections annotées avec truth_id. Défaut : jeu synthétique intégré.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Chemin du config.yaml Phase 3 transmis à la fusion. Défaut : config.yaml du dossier courant.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV de sortie (une ligne par seuil testé). Défaut : reports/fusion_threshold_ablation.csv.",
    )
    parser.add_argument(
        "--thresholds-cm",
        default="50,100,150,200",
        help="Seuils D à tester, en centimètres, séparés par des virgules. Défaut : 50,100,150,200.",
    )
    parser.add_argument(
        "--time-window-ms",
        type=float,
        default=500.0,
        help="Fenêtre temporelle de la fusion, en millisecondes. Défaut : 500.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    config = _load_config(args.config)
    summaries = run_ablation(
        rows=rows,
        config=config,
        thresholds_cm=_parse_thresholds(args.thresholds_cm),
        time_window_s=args.time_window_ms / 1000.0,
    )
    _write_csv(summaries, args.output)

    print(f"[INFO] Ablation terminee: {args.output}")
    for row in summaries:
        print(
            "D={threshold_cm:>3}cm | precision={precision:.4f} | recall={recall:.4f} | "
            "F1={f1:.4f} | FP={fp_false_matches} | FN={fn_missed_matches}".format(**row)
        )


if __name__ == "__main__":
    main()

"""
Évaluation des liens de fusion Phase 3 contre un mini ground truth annoté.

Compare le fichier fusion_links.csv produit par le pipeline Phase 3 (paires
de tracks associées entre caméras par la fusion) à un petit CSV de vérité
terrain annoté à la main : pour chaque paire (frame, cam_a/track_a,
cam_b/track_b), l'annotateur indique dans expected_same si les deux tracks
correspondent réellement à la même personne ou au même objet (1) ou non (0).

Le script en déduit TP/FP/FN/TN puis précision, rappel, F1 et exactitude de
l'association inter-caméras. C'est ce script qui produit le résultat sur les
50 cas annotés cité dans le rapport. Les liens prédits absents de la vérité
terrain sont comptés à part (ignored_predictions_outside_truth) et n'influent
pas sur les métriques.

Colonnes attendues dans le CSV de vérité terrain :
  frame, cam_a, track_a, cam_b, track_b, expected_same

Exemples :
  python evaluate_fusion_links.py --fusion-links reports/campagne/phase3/fusion_links.csv \\
      --truth-csv annotations/fusion_truth_50.csv
  python evaluate_fusion_links.py --fusion-links fusion_links.csv --truth-csv truth.csv \\
      --frame-tolerance 2 --out-csv reports/fusion_links_eval.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


# Valeurs textuelles acceptées pour l'annotation expected_same (vrai / faux).
TRUE_VALUES = {"1", "true", "yes", "y", "oui", "o", "same", "correct"}
FALSE_VALUES = {"0", "false", "no", "n", "non", "different", "incorrect"}


@dataclass(frozen=True)
class PairKey:
    """Clé canonique d'une paire de tracks inter-caméras.

    La paire est ordonnée (côté le plus petit en premier) pour que
    (cam_a, track_a, cam_b, track_b) et son inverse désignent la même clé,
    quel que soit l'ordre d'écriture dans les CSV.
    """

    cam_a: str
    track_a: str
    cam_b: str
    track_b: str

    @classmethod
    def from_values(cls, cam_a: str, track_a: str, cam_b: str, track_b: str) -> "PairKey":
        """Construit la clé en normalisant espaces et ordre des deux côtés."""
        left = (cam_a.strip(), str(track_a).strip())
        right = (cam_b.strip(), str(track_b).strip())
        if right < left:
            left, right = right, left
        return cls(left[0], left[1], right[0], right[1])


@dataclass(frozen=True)
class TruthAssociation:
    """Une ligne de vérité terrain : paire de tracks et verdict de l'annotateur."""

    frame: int
    key: PairKey
    expected_same: bool


@dataclass(frozen=True)
class PredictedAssociation:
    """Un lien prédit par la fusion (ligne de fusion_links.csv), ligne brute conservée."""

    frame: int
    key: PairKey
    row: dict[str, str]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Lit un CSV en liste de dictionnaires (encodage utf-8-sig, BOM toléré)."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_int(value: str | None, default: int = 0) -> int:
    """Convertit une valeur CSV en entier ; `default` si vide ou absente."""
    if value is None or value == "":
        return default
    return int(float(value))


def parse_bool(value: str | None) -> bool:
    """Convertit une annotation textuelle en booléen (voir TRUE_VALUES/FALSE_VALUES).

    Lève ValueError si la valeur n'appartient à aucun des deux ensembles.
    """
    normalized = (value or "").strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean truth value: {value!r}")


def truth_flag(row: dict[str, str]) -> bool:
    """Retourne le verdict annoté d'une ligne, quel que soit le nom de colonne utilisé.

    Plusieurs alias sont acceptés (expected_same, same_person, same_object,
    is_correct, correct, same) pour tolérer différents gabarits d'annotation.
    """
    for column in ("expected_same", "same_person", "same_object", "is_correct", "correct", "same"):
        if column in row and row[column] != "":
            return parse_bool(row[column])
    raise ValueError(
        "Truth CSV must contain one of: expected_same, same_person, same_object, is_correct, correct, same"
    )


def has_truth_flag(row: dict[str, str]) -> bool:
    """Vrai si la ligne contient une annotation non vide (ligne réellement annotée)."""
    for column in ("expected_same", "same_person", "same_object", "is_correct", "correct", "same"):
        if column in row and row[column] != "":
            return True
    return False


def load_truth(path: Path) -> list[TruthAssociation]:
    """Charge la vérité terrain en ignorant les lignes non annotées."""
    truth: list[TruthAssociation] = []
    for row in read_csv(path):
        if not has_truth_flag(row):
            continue
        truth.append(
            TruthAssociation(
                frame=parse_int(row.get("frame")),
                key=PairKey.from_values(
                    row.get("cam_a", ""),
                    row.get("track_a", ""),
                    row.get("cam_b", ""),
                    row.get("track_b", ""),
                ),
                expected_same=truth_flag(row),
            )
        )
    return truth


def load_predictions(path: Path) -> list[PredictedAssociation]:
    """Charge les liens prédits depuis fusion_links.csv (une association par ligne)."""
    predictions: list[PredictedAssociation] = []
    for row in read_csv(path):
        predictions.append(
            PredictedAssociation(
                frame=parse_int(row.get("frame")),
                key=PairKey.from_values(
                    row.get("cam_a", ""),
                    row.get("track_a", ""),
                    row.get("cam_b", ""),
                    row.get("track_b", ""),
                ),
                row=row,
            )
        )
    return predictions


def has_prediction(
    truth: TruthAssociation,
    predictions: list[PredictedAssociation],
    frame_tolerance: int,
) -> PredictedAssociation | None:
    """Cherche un lien prédit pour cette paire, à frame_tolerance frames près."""
    for pred in predictions:
        if pred.key == truth.key and abs(pred.frame - truth.frame) <= frame_tolerance:
            return pred
    return None


def evaluate(
    fusion_links_csv: Path,
    truth_csv: Path,
    frame_tolerance: int = 0,
) -> tuple[dict[str, float], list[dict[str, str | int | float]]]:
    """Confronte les liens prédits à la vérité terrain et calcule les métriques.

    Pour chaque ligne annotée, cherche un lien prédit portant la même paire
    de tracks à frame_tolerance frames près, puis classe le cas :
      - TP : paire annotée identique et lien prédit trouvé ;
      - FP : paire annotée différente mais lien prédit (fausse association) ;
      - FN : paire annotée identique mais aucun lien prédit (association manquée) ;
      - TN : paire annotée différente et aucun lien prédit.

    Retourne (métriques globales, détail par ligne de vérité terrain). Les
    liens prédits jamais appariés à une ligne annotée sont seulement comptés
    dans ignored_predictions_outside_truth.
    """
    truth_rows = load_truth(truth_csv)
    if not truth_rows:
        raise ValueError(
            "Truth CSV contains no annotated rows. Fill expected_same with 1 or 0 for at least one row."
        )
    predictions = load_predictions(fusion_links_csv)

    tp = fp = fn = tn = ignored_predictions = 0
    evaluated: list[dict[str, str | int | float]] = []
    matched_prediction_indexes: set[int] = set()

    for truth in truth_rows:
        # Recherche du premier lien prédit compatible (même paire, frame proche).
        matched_index = None
        matched_prediction = None
        for index, pred in enumerate(predictions):
            if pred.key == truth.key and abs(pred.frame - truth.frame) <= frame_tolerance:
                matched_index = index
                matched_prediction = pred
                break

        predicted_same = matched_prediction is not None
        if matched_index is not None:
            matched_prediction_indexes.add(matched_index)

        # Classification TP/FP/FN/TN du cas annoté.
        if truth.expected_same and predicted_same:
            result = "tp"
            tp += 1
        elif not truth.expected_same and predicted_same:
            result = "fp"
            fp += 1
        elif truth.expected_same and not predicted_same:
            result = "fn"
            fn += 1
        else:
            result = "tn"
            tn += 1

        evaluated.append(
            {
                "frame": truth.frame,
                "cam_a": truth.key.cam_a,
                "track_a": truth.key.track_a,
                "cam_b": truth.key.cam_b,
                "track_b": truth.key.track_b,
                "expected_same": int(truth.expected_same),
                "predicted_same": int(predicted_same),
                "matched_frame": matched_prediction.frame if matched_prediction else "",
                "result": result,
            }
        )

    # Liens prédits hors du périmètre annoté : comptés mais non évalués.
    ignored_predictions = len(predictions) - len(matched_prediction_indexes)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(truth_rows) if truth_rows else 0.0

    metrics = {
        "truth_rows": float(len(truth_rows)),
        "predicted_links": float(len(predictions)),
        "evaluated_predictions": float(len(matched_prediction_indexes)),
        "ignored_predictions_outside_truth": float(ignored_predictions),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
    return metrics, evaluated


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Écrit le détail d'évaluation en CSV, colonnes dans l'ordre de première apparition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["result"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Déclare et analyse les options de l'évaluation des liens de fusion."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted fusion_links.csv against a small manual association truth CSV. "
            "Truth columns: frame, cam_a, track_a, cam_b, track_b, expected_same."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python evaluate_fusion_links.py --fusion-links reports/campagne/phase3/fusion_links.csv \\
      --truth-csv annotations/fusion_truth_50.csv
  python evaluate_fusion_links.py --fusion-links fusion_links.csv --truth-csv truth.csv \\
      --frame-tolerance 2 --out-csv reports/fusion_links_eval.csv
""",
    )
    parser.add_argument(
        "--fusion-links",
        type=Path,
        required=True,
        help="Chemin du fusion_links.csv produit par la Phase 3 (liens prédits). Obligatoire.",
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        required=True,
        help="CSV de vérité terrain annoté à la main (colonne expected_same à 1 ou 0). Obligatoire.",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=0,
        help="Écart de frames toléré entre un lien prédit et la ligne annotée lors de l'appariement. Défaut : 0 (même frame).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV optionnel du détail par ligne annotée (résultat tp/fp/fn/tn). Défaut : pas d'export.",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : évalue les liens, affiche les métriques, exporte le détail si demandé."""
    args = parse_args()
    try:
        metrics, evaluated = evaluate(args.fusion_links, args.truth_csv, frame_tolerance=args.frame_tolerance)
    except FileNotFoundError as exc:
        raise SystemExit(f"[ERROR] Missing CSV: {exc.filename}") from exc
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    print("[INFO] Fusion association quality")
    for key in (
        "truth_rows",
        "predicted_links",
        "evaluated_predictions",
        "ignored_predictions_outside_truth",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ):
        value = metrics[key]
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    if args.out_csv:
        write_csv(args.out_csv, evaluated)
        print(f"[INFO] Wrote {args.out_csv}")


if __name__ == "__main__":
    main()

"""
Boîte à outils partagée des campagnes d'évaluation Phase 2 / Phase 3.

Ce module est le socle des deux points d'entrée officiels de la Phase 3 :
run_recorded_campaign.py (rejeu de vidéos enregistrées) et
run_live_campaign.py (flux RTSP en direct). Il regroupe quatre grandes parties :

- Découverte des modèles entraînés : parcours de
  Phase_2_Baseline_MonoCam/Modelstrained (organisation à plat V2/V3, ou par
  variante de dataset en V4), sélection des poids PyTorch (.pt) et des moteurs
  TensorRT FP32 (ModelSpec, discover_model_specs, iter_model_dirs).

- Exécution des campagnes : la classe Phase3Campaign instancie la chaîne
  complète (un tracker par caméra, fusion multi-caméras, détecteur de
  violations) et déroule la boucle recorded ou live en accumulant les lignes
  CSV (détections, alertes, liens de fusion, synchronisation, stabilité des
  tracks, trace de latence).

- Métriques TRD / TAD : appariement des alertes aux événements de vérité
  terrain avec tolérance temporelle (match_alerts_to_gt,
  compute_phase3_metrics), regroupement des faux positifs TAD aligné sur la
  méthodologie de la Phase 2 (TAD_FP_MERGE_WINDOW_S).

- Écriture des sorties et utilitaires : CSV (write_csv, merge_csvs),
  manifestes JSON, journalisation (append_log, Tee, stdout_to_log), lancement
  de sous-processus (audit de calibration, ablations du seuil de fusion).
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


# Racines du projet et emplacements par défaut : poids Phase 2, config et rapports Phase 3.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE2_DIR = PROJECT_ROOT / "Phase_2_Baseline_MonoCam"
PHASE3_DIR = PROJECT_ROOT / "Phase_3_Fusion_MultiCam"
MODELSTRAINED_DIR = PHASE2_DIR / "Modelstrained"
DEFAULT_CONFIG = PHASE3_DIR / "config.yaml"
DEFAULT_REPORTS_DIR = PHASE3_DIR / "reports"
# Fenêtre (s) de regroupement des faux positifs TAD, alignée sur FP_MERGE_WINDOW_S
# de Phase_2_Baseline_MonoCam/evaluate_tad.py pour garder les chiffres comparables.
TAD_FP_MERGE_WINDOW_S = 3.0


def first_existing_path(*paths: Path) -> Path:
    """Retourne le premier chemin existant, ou le premier de la liste (défaut historique) si aucun n'existe."""
    for path in paths:
        if path.exists():
            return path
    return paths[0]


# Vérités terrain par défaut : plusieurs emplacements historiques sont tolérés,
# le premier fichier existant l'emporte.
DEFAULT_GT_OBJECTS_TAD = first_existing_path(
    PROJECT_ROOT / "dataset_objets_HD" / "gt_objects_tad.json",
    PROJECT_ROOT / "ground_truth" / "gt_objects_tad_dataset_objets_HD.json",
    PROJECT_ROOT / "ground_truth" / "gt_objects_tad.json",
)
DEFAULT_GT_PEOPLE = first_existing_path(
    PROJECT_ROOT / "gt_people.json",
    PROJECT_ROOT / "ground_truth" / "gt_people.json",
)
PERSON_CLASS_ID = 11  # identifiant de la classe "personne" dans les datasets entraînés

# Zone 1 (salle de référence) : caméras couvrantes, vidéos enregistrées associées
# et identifiants correspondants dans les fichiers de vérité terrain.
ZONE1_CAMERAS = ["cam_02", "cam_03", "cam_05", "cam_07"]
ZONE1_VIDEO_MAP = {
    "cam_02": PROJECT_ROOT / "recordings/recordings/Camera_2_2.3_20260506_131002.mp4",
    "cam_03": PROJECT_ROOT / "recordings/recordings/Camera_3_2.4_20260506_131002.mp4",
    "cam_05": PROJECT_ROOT / "recordings/recordings/Camera_5_2.6_20260506_131002.mp4",
    "cam_07": PROJECT_ROOT / "recordings/recordings/Camera_7_2.11_20260506_131002.mp4",
}
ZONE1_GT_ID_MAP = {
    cam_id: f"{cam_id}_20260506_131002" for cam_id in ZONE1_CAMERAS
}


@dataclass(frozen=True)
class ModelSpec:
    """Modèle entraîné découvert dans Modelstrained, prêt à être exécuté.

    Combine la version d'entraînement (V2, V3...), le nom du modèle, le format
    d'exécution (poids PyTorch "pt" ou moteur TensorRT "fp32_engine") et le
    chemin des poids. Sert à lancer les campagnes Phase 3 et à générer la
    liste de modèles consommée par les scripts d'évaluation de la Phase 2.
    """

    version: str
    name: str
    subdir: str
    model_type: str
    format_label: str
    format_arg: str
    suffix_arg: str
    weights_path: Path
    dataset_variant: str = ""

    def phase2_dict(self) -> dict[str, str]:
        """Sous-ensemble minimal attendu par les scripts d'évaluation de la Phase 2."""
        return {"name": self.name, "subdir": self.subdir, "type": self.model_type}

    def manifest_dict(self) -> dict[str, Any]:
        """Représentation complète pour le manifeste JSON de campagne (chemin
        des poids en texte, avec indicateur d'existence sur disque)."""
        data = asdict(self)
        data["weights_path"] = str(self.weights_path)
        data["exists"] = self.weights_path.exists()
        return data

    @property
    def run_label(self) -> str:
        """Étiquette unique du run (version, variante éventuelle, nom, format),
        utilisée pour nommer les dossiers et fichiers de sortie."""
        parts = [self.version]
        if self.dataset_variant:
            parts.append(self.dataset_variant)
        parts.extend([self.name, self.format_label])
        return "_".join(parts)


def parse_csv_arg(raw: str | None, default: Iterable[str]) -> list[str]:
    """Découpe un argument CLI de la forme "a,b,c" en liste, ou retourne la valeur par défaut si vide."""
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def timestamped_campaign_dir(prefix: str = "campaign_zone1") -> Path:
    """Construit un chemin de dossier de campagne horodaté sous reports/ (ex: campaign_zone1_20260506_131002)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_REPORTS_DIR / f"{prefix}_{stamp}"


def infer_model_type(name: str) -> str:
    """Déduit le type de modèle ("rtdetr" ou "yolo") à partir de son nom de dossier."""
    return "rtdetr" if name.lower().startswith("rtdetr") else "yolo"


def find_fp32_engine(weights_dir: Path, version: str) -> tuple[Path, str] | None:
    """Cherche le moteur TensorRT FP32 dans le dossier de poids d'un modèle.

    L'ordre des candidats dépend de la version d'entraînement : en V3,
    best_fp32_best.engine est prioritaire sur best.engine ; ailleurs c'est
    l'inverse (conventions d'export différentes selon les versions).

    Returns:
        Tuple (chemin du moteur, suffixe Phase 2 associé), ou None si absent.
    """
    version_key = version.upper()
    if version_key == "V3":
        candidates = [
            ("best_fp32_best.engine", "fp32_best"),
            ("best.engine", ""),
        ]
    else:
        candidates = [
            ("best.engine", ""),
            ("best_fp32_best.engine", "fp32_best"),
        ]

    for filename, suffix in candidates:
        engine_path = weights_dir / filename
        if engine_path.exists():
            return engine_path, suffix
    return None


def iter_model_dirs(
    version_dir: Path,
    dataset_variants: Iterable[str] | None = None,
) -> Iterable[tuple[Path, str, str]]:
    """Énumère les dossiers de modèles d'une version d'entraînement.

    Gère deux organisations de Modelstrained : V2/V3 à plat (un dossier par
    modèle contenant weights/) et V4 imbriquée par variante de dataset
    (person_objects, objects_only).

    Yields:
        Tuples (dossier du modèle, sous-chemin relatif, variante de dataset).
    """
    # Organisation à plat (V2/V3) : tout sous-dossier contenant weights/ est un modèle.
    direct_dirs = sorted(
        p for p in version_dir.iterdir()
        if p.is_dir() and (p / "weights").is_dir()
    )
    for model_dir in direct_dirs:
        yield model_dir, model_dir.name, ""

    # Une organisation à plat exclut l'organisation par variantes de dataset.
    if direct_dirs:
        return

    # Organisation imbriquée (V4) : version/<variante>/<modèle>. Sans variante
    # demandée, la première variante préférée présente est retenue.
    requested_variants = list(dataset_variants or [])
    if not requested_variants:
        for preferred in ("person_objects", "objects_only"):
            if (version_dir / preferred).is_dir():
                requested_variants = [preferred]
                break

    for variant in requested_variants:
        variant_dir = version_dir / variant
        if not variant_dir.is_dir():
            continue
        for model_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            yield model_dir, f"{variant}/{model_dir.name}", variant


def discover_model_specs(
    versions: Iterable[str] = ("V2", "V3"),
    formats: Iterable[str] = ("pt", "fp32_engine"),
    model_names: Iterable[str] | None = None,
    modelstrained_dir: Path = MODELSTRAINED_DIR,
    dataset_variants: Iterable[str] | None = None,
) -> list[ModelSpec]:
    """Découvre les poids entraînés disponibles pour une campagne.

    Parcourt Modelstrained/<version>/ et produit un ModelSpec par couple
    (modèle, format) demandé : poids PyTorch (best.pt) et/ou moteur TensorRT
    FP32. Les filtres versions, formats et model_names restreignent la
    sélection ; model_names vide signifie "tous les modèles".

    Returns:
        Liste de ModelSpec dont les poids existent sur disque.
    """
    selected = set(model_names or [])
    wanted_formats = set(formats)
    specs: list[ModelSpec] = []

    for version in versions:
        version_dir = modelstrained_dir / version
        if not version_dir.exists():
            continue
        for model_dir, subdir, dataset_variant in iter_model_dirs(version_dir, dataset_variants):
            name = model_dir.name
            if selected and name not in selected:
                continue
            weights_dir = model_dir / "weights"
            if "pt" in wanted_formats:
                pt_path = weights_dir / "best.pt"
                if pt_path.exists():
                    specs.append(
                        ModelSpec(
                            version=version,
                            name=name,
                            subdir=subdir,
                            model_type=infer_model_type(name),
                            format_label="pt",
                            format_arg="pt",
                            suffix_arg="",
                            weights_path=pt_path,
                            dataset_variant=dataset_variant,
                        )
                    )
            if "fp32_engine" in wanted_formats:
                engine_spec = find_fp32_engine(weights_dir, version)
                if engine_spec:
                    engine_path, suffix = engine_spec
                    specs.append(
                        ModelSpec(
                            version=version,
                            name=name,
                            subdir=subdir,
                            model_type=infer_model_type(name),
                            format_label="fp32_engine",
                            format_arg="engine",
                            suffix_arg=suffix,
                            weights_path=engine_path,
                            dataset_variant=dataset_variant,
                        )
                    )

    return specs


def write_phase2_specs(specs: list[ModelSpec], path: Path) -> None:
    """Écrit la liste des modèles au format JSON consommé par les scripts de la
    Phase 2, dédoublonnée par nom (le format pt/engine n'y figure pas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: dict[str, dict[str, str]] = {}
    for spec in specs:
        unique[spec.name] = spec.phase2_dict()
    path.write_text(
        json.dumps(list(unique.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_gt_events(gt_path: Path, camera_id: str) -> list[dict[str, Any]]:
    """Charge la vérité terrain et ne garde que les événements de la caméra demandée (champ id_camera)."""
    with open(gt_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [row for row in data if row.get("id_camera") == camera_id]


def precheck_campaign_inputs(
    cameras: list[str],
    video_map: dict[str, Path],
    gt_id_map: dict[str, str],
    tad_gt_path: Path = DEFAULT_GT_OBJECTS_TAD,
    strict_tad: bool = True,
) -> None:
    """Vérifie avant campagne que vidéos et vérités terrain sont disponibles.

    Échoue immédiatement (SystemExit) si une vidéo manque ou si une caméra n'a
    aucun événement GT TRD. Le contrôle TAD peut être désactivé (strict_tad à
    False) quand la GT objets n'est pas encore annotée pour ces caméras.
    """
    missing_videos = [str(video_map[cam]) for cam in cameras if not video_map[cam].exists()]
    if missing_videos:
        raise SystemExit("[ERREUR] Videos introuvables:\n  " + "\n  ".join(missing_videos))

    gt_people = DEFAULT_GT_PEOPLE
    missing_trd = [
        gt_id_map[cam] for cam in cameras if not load_gt_events(gt_people, gt_id_map[cam])
    ]
    if missing_trd:
        raise SystemExit("[ERREUR] GT TRD manquante pour: " + ", ".join(missing_trd))

    if strict_tad:
        missing_tad = [
            gt_id_map[cam] for cam in cameras if not load_gt_events(tad_gt_path, gt_id_map[cam])
        ]
        if missing_tad:
            raise SystemExit(
                "[ERREUR] GT TAD manquante pour: "
                + ", ".join(missing_tad)
                + f"\nFichier verifie: {tad_gt_path}"
                + "\nAjoutez ces cles ou relancez avec --no-strict-tad."
            )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Écrit une liste de dicts en CSV UTF-8, en ignorant les clés hors fieldnames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_log(log_path: Path, line: str) -> None:
    """Ajoute une ligne au fichier de log (créé au besoin)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def normalize_class(name: str) -> str:
    """Normalise un nom de classe (suppression des accents, minuscules) pour
    comparer GT et détections malgré les variations d'orthographe."""
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def safe_div(num: float, den: float) -> float:
    """Division protégée : retourne 0.0 si le dénominateur est nul."""
    return num / den if den else 0.0


def summarize_latencies(latencies_s: list[float]) -> dict[str, float]:
    """Résume les latences d'inférence (moyenne, médiane, p95, max) en millisecondes."""
    if not latencies_s:
        return {
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_max_ms": 0.0,
        }
    arr = np.array(latencies_s, dtype=np.float64) * 1000.0
    return {
        "latency_mean_ms": round(float(arr.mean()), 3),
        "latency_median_ms": round(float(np.median(arr)), 3),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 3),
        "latency_max_ms": round(float(arr.max()), 3),
    }


def _id_to_color(track_id: int) -> tuple[int, int, int]:
    """Couleur pseudo-aléatoire stable pour un identifiant de track (affichage)."""
    rng = np.random.default_rng(track_id * 97 + 13)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Dessine une étiquette texte sur fond coloré au-dessus d'une bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    top = max(y - th - baseline - 4, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - baseline - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def cluster_unmatched_times(times: list[float], merge_window_s: float) -> int:
    """Regroupe les alertes non appariées en événements de faux positif distincts.

    Une détection parasite persistante émettrait une alerte par frame, soit des
    centaines de FP pour un seul événement réel. Les instants séparés de moins
    de merge_window_s sont donc comptés comme un seul événement, comme le fait
    evaluate_tad.py en Phase 2.
    """
    if not times:
        return 0
    ordered = sorted(times)
    clusters = 1
    for prev, cur in zip(ordered, ordered[1:]):
        if cur - prev > merge_window_s:
            clusters += 1
    return clusters


def match_alerts_to_gt(
    gt_events: list[dict[str, Any]],
    alert_times: list[float],
    gt_frame_key: str,
    fps: float,
    tolerance_s: float = 10.0,
    fp_merge_window_s: float = 0.0,
) -> dict[str, Any]:
    """Apparie les alertes aux événements GT et calcule les métriques associées.

    Chaque événement GT est converti en secondes (trame gt_frame_key / fps),
    puis apparié à l'alerte non consommée la plus proche dans une fenêtre de
    plus ou moins tolerance_s (10 s par défaut). Le délai signé (alerte moins
    GT) alimente les statistiques de réactivité TRD/TAD. Les alertes restantes
    comptent en faux positifs, regroupées par fenêtre temporelle si
    fp_merge_window_s > 0 (cas TAD, pour ne pas compter un FP par frame).

    Returns:
        Dict avec les comptes (GT, appariés, manqués, FP), précision, rappel,
        F1 et les délais metric_median / metric_p95 / metric_mean (None si
        aucun appariement).
    """
    gt_times = [float(ev[gt_frame_key]) / fps for ev in gt_events]
    unmatched_alerts = set(range(len(alert_times)))
    delays: list[float] = []
    matched = 0
    # Appariement glouton : chaque événement GT consomme l'alerte disponible la
    # plus proche dans la fenêtre de tolérance (une alerte ne sert qu'une fois).
    for gt_time in gt_times:
        candidates = [
            (idx, alert_times[idx] - gt_time)
            for idx in unmatched_alerts
            if abs(alert_times[idx] - gt_time) <= tolerance_s
        ]
        if not candidates:
            continue
        idx, delay = min(candidates, key=lambda item: abs(item[1]))
        unmatched_alerts.remove(idx)
        matched += 1
        delays.append(delay)
    # Les alertes jamais appariées comptent en faux positifs, regroupées en
    # événements si une fenêtre de fusion est fournie (cas TAD).
    if fp_merge_window_s > 0:
        fp = cluster_unmatched_times([alert_times[idx] for idx in unmatched_alerts], fp_merge_window_s)
    else:
        fp = len(unmatched_alerts)
    fn = len(gt_events) - matched
    precision = safe_div(matched, matched + fp)
    recall = safe_div(matched, matched + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "n_gt_events": len(gt_events),
        "n_matched": matched,
        "n_missed": fn,
        "n_faux_positifs": fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "metric_median": round(float(np.median(delays)), 3) if delays else None,
        "metric_p95": round(float(np.percentile(delays, 95)), 3) if delays else None,
        "metric_mean": round(float(np.mean(delays)), 3) if delays else None,
    }


class Tee:
    """Duplique chaque écriture vers plusieurs flux (console et fichier de log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class Phase3Campaign:
    """Exécute une campagne Phase 3 complète pour un modèle donné.

    Instancie la chaîne de traitement par caméra (détection + ByteTrack +
    homographie via CameraTracker), la fusion multi-caméras et le détecteur de
    violations, puis déroule la boucle en mode recorded (rejeu de vidéos
    synchronisées) ou live (flux RTSP). Les données produites (détections,
    alertes, liens de fusion, synchronisation, trace de latence) sont
    accumulées en mémoire puis écrites en CSV par write_outputs().
    """

    def __init__(
        self,
        config_path: Path,
        out_dir: Path,
        cameras: list[str],
        model_spec: ModelSpec,
        device: str | None,
        fusion_distance_m: float,
        fusion_time_window_ms: float,
        confidence: float | None = None,
        alerting_overrides: dict[str, Any] | None = None,
        metadata_publisher: Any | None = None,
        latency_trace_path: Path | None = None,
    ) -> None:
        """Charge la configuration, applique les surcharges et construit la chaîne.

        Args:
            confidence: seuil de confiance imposé au détecteur (None conserve la config).
            alerting_overrides: surcharges de la section alerting (les valeurs None sont ignorées).
            metadata_publisher: éditeur optionnel de métadonnées temps réel (mode live).
            latency_trace_path: si fourni, active la trace de latence détaillée par frame.
        """
        from fusion import MultiCameraFusion
        from geometry_fix import load_aspect_fix
        from tracker import CameraTracker
        from violation_detector import ViolationDetector

        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh) or {}
        # La classe personne est imposée pour que TRD et fusion restent
        # cohérents quel que soit le modèle chargé.
        self.config.setdefault("detection", {})["person_class_ids"] = [PERSON_CLASS_ID]
        if confidence is not None:
            self.config.setdefault("detection", {})["confidence"] = confidence
        if alerting_overrides:
            self.config.setdefault("alerting", {}).update(
                {key: value for key, value in alerting_overrides.items() if value is not None}
            )

        self.out_dir = out_dir
        self.cameras = cameras
        self.model_spec = model_spec
        self.device = device
        self.fusion_distance_m = fusion_distance_m
        self.fusion_time_window_ms = fusion_time_window_ms
        self.metadata_publisher = metadata_publisher
        self.latency_trace_path = latency_trace_path
        self.ar_fix = load_aspect_fix(self.config)
        self.violation_detector = ViolationDetector(self.config)
        self.fusion = MultiCameraFusion(
            self.config,
            distance_threshold_m=fusion_distance_m,
            time_window_s=fusion_time_window_ms / 1000.0,
        )
        self.trackers = {
            cam_id: CameraTracker(
                cam_id=cam_id,
                model_path=str(model_spec.weights_path),
                config=self.config,
                conf_threshold=self.config.get("detection", {}).get("confidence", 0.5),
                target_classes=None,
                device=device,
            )
            for cam_id in cameras
        }

        # Accumulateurs des lignes CSV et état interne (suivi des tracks,
        # traînées de points au sol pour l'affichage, latences d'inférence).
        self.detection_rows: list[dict[str, Any]] = []
        self.alert_rows: list[dict[str, Any]] = []
        self.fusion_rows: list[dict[str, Any]] = []
        self.sync_rows: list[dict[str, Any]] = []
        self.raw_detection_rows: list[dict[str, Any]] = []
        self.track_state: dict[tuple[str, int], dict[str, Any]] = {}
        self.last_detections_by_cam: dict[str, list[Any]] = {}
        self.track_trails: dict[tuple[str, int], list[tuple[int, int]]] = {}
        self.latencies: list[float] = []
        self.latency_trace_rows: list[dict[str, Any]] = []
        self.frame_count = 0

    def _record_detections(self, frame_idx: int, detections: list[Any]) -> None:
        """Enregistre les détections fusionnées d'une frame et les liens inter-caméras.

        Met à jour l'état de chaque track local (âge, changements de global_id)
        et produit une ligne CSV par détection, plus une version brute allégée
        (raw_detection_rows) réutilisée par l'ablation opérationnelle.
        """
        by_gid: dict[int, list[Any]] = {}
        for det in detections:
            key = (det.cam_id, det.track_id)
            state = self.track_state.setdefault(
                key,
                {
                    "cam_id": det.cam_id,
                    "track_id": det.track_id,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "age": 0,
                    "global_ids": [],
                    "global_id_switches": 0,
                    "last_global_id": None,
                },
            )
            state["age"] += 1
            state["last_frame"] = frame_idx
            # Un changement de global_id sur un même track local compte comme
            # un switch d'identité (indicateur d'instabilité de la fusion).
            if det.global_id != state["last_global_id"]:
                if state["last_global_id"] is not None and det.global_id is not None:
                    state["global_id_switches"] += 1
                state["last_global_id"] = det.global_id
            if det.global_id is not None:
                state["global_ids"].append(det.global_id)
                by_gid.setdefault(det.global_id, []).append(det)

            x_m, y_m = det.foot_point_m if det.foot_point_m else (None, None)
            zones = self.violation_detector.zones_containing(det)
            row = {
                "model_version": self.model_spec.version,
                "model": self.model_spec.name,
                "format": self.model_spec.format_label,
                "frame": frame_idx,
                "timestamp": round(float(det.timestamp), 6),
                "cam_id": det.cam_id,
                "track_id": det.track_id,
                "track_age": state["age"],
                "global_id": det.global_id,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": round(det.confidence, 6),
                "x1": round(det.bbox_px[0], 3),
                "y1": round(det.bbox_px[1], 3),
                "x2": round(det.bbox_px[2], 3),
                "y2": round(det.bbox_px[3], 3),
                "foot_u": round(det.foot_point_px[0], 3),
                "foot_v": round(det.foot_point_px[1], 3),
                "x_m": round(x_m, 6) if x_m is not None else "",
                "y_m": round(y_m, 6) if y_m is not None else "",
                "inside_zones": "+".join(zones),
            }
            self.detection_rows.append(row)
            self.raw_detection_rows.append(
                {
                    "frame": frame_idx,
                    "timestamp": det.timestamp,
                    "cam_id": det.cam_id,
                    "track_id": det.track_id,
                    "x_m": x_m,
                    "y_m": y_m,
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                }
            )

        # Liens de fusion : pour chaque global_id vu par au moins deux caméras
        # sur la frame, une ligne par paire de caméras distinctes (écart
        # temporel et distance au sol observés entre les deux détections).
        for gid, dets in by_gid.items():
            if len({d.cam_id for d in dets}) < 2:
                continue
            dets = sorted(dets, key=lambda d: d.cam_id)
            for i, a in enumerate(dets):
                for b in dets[i + 1 :]:
                    if a.cam_id == b.cam_id:
                        continue
                    dist = ""
                    if a.foot_point_m and b.foot_point_m:
                        dist = math.dist(a.foot_point_m, b.foot_point_m)
                    self.fusion_rows.append(
                        {
                            "model_version": self.model_spec.version,
                            "model": self.model_spec.name,
                            "format": self.model_spec.format_label,
                            "frame": frame_idx,
                            "timestamp": round(float(a.timestamp), 6),
                            "global_id": gid,
                            "cam_a": a.cam_id,
                            "track_a": a.track_id,
                            "class_a": a.class_name,
                            "cam_b": b.cam_id,
                            "track_b": b.track_id,
                            "class_b": b.class_name,
                            "time_delta_ms": round(
                                abs(float(a.timestamp) - float(b.timestamp)) * 1000.0,
                                3,
                            ),
                            "distance_m": round(dist, 6) if dist != "" else "",
                        }
                    )

    def _record_alerts(self, frame_idx: int, alerts: list[Any], log_path: Path) -> None:
        """Journalise chaque alerte et l'ajoute aux lignes CSV.

        Le niveau d'alerte est conservé dans alert_level : "weak" (objet vu par
        une seule caméra) ou "confirmed" (confirmation multi-caméras), avec
        "confirmed" par défaut pour les alertes sans ce champ.
        """
        for alert in alerts:
            append_log(log_path, f"[ALERT] {alert}")
            x_m, y_m = alert.position_m
            self.alert_rows.append(
                {
                    "model_version": self.model_spec.version,
                    "model": self.model_spec.name,
                    "format": self.model_spec.format_label,
                    "frame": frame_idx,
                    "timestamp": round(float(alert.timestamp), 6),
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "alert_level": getattr(alert, "alert_level", "confirmed"),
                    "global_id": alert.global_id,
                    "zone_id": alert.zone_id or "",
                    "class_name": alert.class_name,
                    "x_m": round(x_m, 6),
                    "y_m": round(y_m, 6),
                    "cameras": "+".join(alert.cameras),
                    "confidence": round(alert.confidence, 6),
                }
            )

    def run_recorded(
        self,
        video_map: dict[str, Path],
        max_frames: int | None,
        display: bool,
        log_path: Path,
    ) -> dict[str, Any]:
        """Boucle de campagne sur vidéos enregistrées.

        Lit les vidéos des caméras en pas à pas synchronisé (une frame par
        caméra et par itération) ; les timestamps sont reconstruits à partir de
        l'index de frame et du FPS de chaque vidéo.

        Returns:
            Le résumé de campagne (voir summary()).
        """
        caps: dict[str, cv2.VideoCapture] = {}
        fps_map: dict[str, float] = {}
        for cam_id in self.cameras:
            cap = cv2.VideoCapture(str(video_map[cam_id]))
            if not cap.isOpened():
                raise SystemExit(f"[ERREUR] Impossible d'ouvrir {video_map[cam_id]}")
            caps[cam_id] = cap
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_map[cam_id] = fps if fps > 0 else 25.0

        try:
            # Avance synchronisée : une frame par caméra et par itération ; la
            # campagne s'arrête dès qu'un flux est épuisé (vidéo la plus courte)
            # ou que max_frames est atteint.
            while True:
                if max_frames is not None and self.frame_count >= max_frames:
                    break
                frames_raw: dict[str, Any] = {}
                exhausted = False
                for cam_id, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        exhausted = True
                        break
                    frames_raw[cam_id] = frame
                if exhausted or not frames_raw:
                    break
                self._process_frames(frames_raw, fps_map, log_path, live=False)
                if display and self._display(frames_raw):
                    break
                self.frame_count += 1
        finally:
            for cap in caps.values():
                cap.release()
            if display:
                cv2.destroyAllWindows()
        return self.summary()

    def run_live(
        self,
        duration_s: float,
        display: bool,
        log_path: Path,
        capture_options: Any,
        record_dir: Path | None = None,
        record_fps: float = 25.0,
        display_mode: str = "annotated",
    ) -> dict[str, Any]:
        """Boucle de campagne sur flux RTSP en direct, bornée par duration_s.

        À chaque itération : lecture d'une frame par caméra (avec reconnexion
        automatique en cas d'échec), réenregistrement vidéo optionnel
        accompagné d'événements de synchronisation, traitement complet
        (_process_frames), affichage optionnel. Chaque étape est horodatée pour
        alimenter la trace de latence et les métadonnées temps réel.

        Args:
            capture_options: options passées à open_capture_with_info (backend, transport...).
            record_dir: si fourni, chaque caméra est réenregistrée en MP4 dans ce dossier.
            record_fps: FPS d'écriture des vidéos réenregistrées.
            display_mode: "annotated" (zones, boîtes, traînées) ou affichage brut.

        Returns:
            Le résumé de campagne (voir summary()).
        """
        from video_capture import open_capture_with_info

        caps: dict[str, cv2.VideoCapture] = {}
        capture_backends: dict[str, str] = {}
        # Ouverture des flux RTSP : une caméra en échec est journalisée sans
        # bloquer la campagne, mais au moins un flux ouvert est requis.
        for cam_id in self.cameras:
            source = self.config.get("cameras", {}).get(cam_id, {}).get("rtsp_url")
            result = open_capture_with_info(str(source), live=True, options=capture_options)
            if result.cap.isOpened():
                result.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                caps[cam_id] = result.cap
                capture_backends[cam_id] = result.backend
                append_log(log_path, f"[INFO] {cam_id} opened via {result.backend}: {result.source}")
            else:
                append_log(log_path, f"[WARN] {cam_id} open failed: {result.error}")
                result.cap.release()
        if not caps:
            raise SystemExit("[ERREUR] Aucun flux RTSP ouvert.")

        started = time.time()
        # FPS nominal (inutilisé en live : les timestamps sont des horloges
        # murales) ; les écrivains vidéo sont créés à la première frame reçue.
        fps_map = {cam_id: 25.0 for cam_id in caps}
        writers: dict[str, cv2.VideoWriter] = {}
        video_paths: dict[str, Path] = {}
        recorded_counts = {cam_id: 0 for cam_id in caps}
        record_dir = Path(record_dir) if record_dir else None
        if record_dir:
            record_dir.mkdir(parents=True, exist_ok=True)
            append_log(log_path, f"[INFO] Enregistrement video live actif: {record_dir}")
        try:
            while time.time() - started < duration_s:
                loop_start_perf = time.perf_counter()
                loop_start_epoch = time.time()
                frames_raw = {}
                latency_trace_by_cam: dict[str, dict[str, Any]] = {}
                # Lecture d'une frame par caméra, chronométrée deux fois :
                # perf_counter pour les durées, epoch pour recaler les étapes
                # entre elles dans la trace de latence.
                for cam_id, cap in caps.items():
                    read_start_perf = time.perf_counter()
                    read_start_epoch = time.time()
                    ret, frame = cap.read()
                    read_end_perf = time.perf_counter()
                    read_end_epoch = time.time()
                    if self.latency_trace_path or self.metadata_publisher:
                        latency_trace_by_cam[cam_id] = {
                            "loop_start_epoch": loop_start_epoch,
                            "capture_read_start_epoch": read_start_epoch,
                            "capture_read_end_epoch": read_end_epoch,
                            "capture_read_ms": (read_end_perf - read_start_perf) * 1000.0,
                            "capture_success": bool(ret),
                            "record_write_ms": 0.0,
                            "capture_backend": capture_backends.get(cam_id, ""),
                        }
                    if ret:
                        frames_raw[cam_id] = frame
                        continue

                    # Échec de lecture : tentative de reconnexion immédiate.
                    # Même en échec, le flux reste dans caps pour être retenté
                    # au tour de boucle suivant.
                    append_log(log_path, f"[WARN] {cam_id} frame read failed, reconnecting...")
                    cap.release()
                    source = self.config.get("cameras", {}).get(cam_id, {}).get("rtsp_url")
                    result = open_capture_with_info(str(source), live=True, options=capture_options)
                    if result.cap.isOpened():
                        result.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        caps[cam_id] = result.cap
                        capture_backends[cam_id] = result.backend
                        append_log(
                            log_path,
                            f"[INFO] {cam_id} reconnected via {result.backend}: {result.source}",
                        )
                    else:
                        caps[cam_id] = result.cap
                        append_log(log_path, f"[WARN] {cam_id} reconnect failed: {result.error}")
                if frames_raw:
                    # Instant où le lot de frames est complet : référence de la
                    # latence bout en bout (capture -> métadonnées).
                    frame_wall_time = time.time()
                    batch_ready_epoch = frame_wall_time
                    # Réenregistrement optionnel : l'écrivain est créé à la
                    # première frame (dimensions alors connues) et chaque
                    # écriture produit un événement de synchronisation reliant
                    # frame de campagne, frame vidéo et horodatage mur.
                    if record_dir:
                        for cam_id, frame in frames_raw.items():
                            if cam_id not in writers:
                                h, w = frame.shape[:2]
                                video_path = record_dir / f"{cam_id}_{self.model_spec.run_label}.mp4"
                                writer = cv2.VideoWriter(
                                    str(video_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    record_fps,
                                    (w, h),
                                )
                                if writer.isOpened():
                                    writers[cam_id] = writer
                                    video_paths[cam_id] = video_path
                                    append_log(log_path, f"[INFO] {cam_id} recording: {video_path}")
                                else:
                                    append_log(log_path, f"[WARN] {cam_id} recording open failed: {video_path}")
                                    writer.release()
                            if cam_id in writers:
                                record_start_perf = time.perf_counter()
                                writers[cam_id].write(frame)
                                record_end_perf = time.perf_counter()
                                if cam_id in latency_trace_by_cam:
                                    latency_trace_by_cam[cam_id]["record_write_ms"] = (
                                        record_end_perf - record_start_perf
                                    ) * 1000.0
                                recorded_counts[cam_id] += 1
                                self.sync_rows.append(
                                    {
                                        "model_version": self.model_spec.version,
                                        "model": self.model_spec.name,
                                        "format": self.model_spec.format_label,
                                        "campaign_frame": self.frame_count,
                                        "cam_id": cam_id,
                                        "video_frame": recorded_counts[cam_id] - 1,
                                        "timestamp_epoch": round(frame_wall_time, 6),
                                        "timestamp_iso": datetime.fromtimestamp(frame_wall_time).isoformat(timespec="milliseconds"),
                                        "elapsed_s": round(frame_wall_time - started, 6),
                                        "recorded_video": str(video_paths.get(cam_id, "")),
                                        "capture_backend": capture_backends.get(cam_id, ""),
                                    }
                                )
                    self._process_frames(
                        frames_raw,
                        fps_map,
                        log_path,
                        live=True,
                        latency_trace_by_cam=latency_trace_by_cam,
                        batch_ready_epoch=batch_ready_epoch,
                    )
                    # display_ms et total_loop_ms sont mesurés ici puis
                    # réinjectés a posteriori dans les lignes de trace de la frame.
                    display_ms = 0.0
                    stop_display = False
                    if display:
                        display_start_perf = time.perf_counter()
                        stop_display = self._display(frames_raw, display_mode=display_mode)
                        display_ms = (time.perf_counter() - display_start_perf) * 1000.0
                    total_loop_ms = (time.perf_counter() - loop_start_perf) * 1000.0
                    self._finalize_latency_trace_frame(self.frame_count, display_ms, total_loop_ms)
                    self.frame_count += 1
                    if stop_display:
                        break
        finally:
            if self.metadata_publisher:
                self.metadata_publisher.close()
            for writer in writers.values():
                writer.release()
            for cap in caps.values():
                cap.release()
            if display:
                cv2.destroyAllWindows()
        return self.summary()

    def _process_frames(
        self,
        frames_raw: dict[str, Any],
        fps_map: dict[str, float],
        log_path: Path,
        live: bool,
        latency_trace_by_cam: dict[str, dict[str, Any]] | None = None,
        batch_ready_epoch: float | None = None,
    ) -> None:
        """Traite un lot de frames synchronisées : détection, fusion, alertes.

        Pour chaque caméra, le tracker produit ses détections ; la fusion
        associe ensuite les détections inter-caméras en global_id, puis le
        détecteur de violations est évalué sur le résultat fusionné. Chaque
        étape est chronométrée pour la trace de latence et les métadonnées.
        """
        detections_by_cam: dict[str, list[Any]] = {}
        tracker_timings: dict[str, dict[str, Any]] = {}
        for cam_id, frame in frames_raw.items():
            # En live, le timestamp est l'horloge murale ; en recorded, il est
            # reconstruit depuis l'index de frame et le FPS de la vidéo.
            timestamp = time.time() if live else self.frame_count / fps_map.get(cam_id, 25.0)
            t0 = time.perf_counter()
            inference_start_epoch = time.time()
            detections_by_cam[cam_id] = self.trackers[cam_id].process_frame(frame, timestamp)
            inference_end_perf = time.perf_counter()
            inference_end_epoch = time.time()
            latency_s = inference_end_perf - t0
            self.latencies.append(latency_s)
            tracker_timings[cam_id] = {
                "frame_timestamp_epoch": timestamp if live else "",
                "inference_start_epoch": inference_start_epoch if live else "",
                "inference_end_epoch": inference_end_epoch if live else "",
                "inference_tracking_ms": latency_s * 1000.0,
                "detections_camera": len(detections_by_cam[cam_id]),
            }
        fusion_start_perf = time.perf_counter()
        fusion_start_epoch = time.time()
        fused = self.fusion.associate(detections_by_cam)
        fusion_end_perf = time.perf_counter()
        fusion_end_epoch = time.time()
        # Mémorise les dernières détections par caméra et les traînées de
        # points au sol (limitées à 60 points) pour l'affichage annoté.
        self.last_detections_by_cam = {}
        for det in fused:
            self.last_detections_by_cam.setdefault(det.cam_id, []).append(det)
            if det.foot_point_px and np.isfinite(det.foot_point_px).all():
                key = (det.cam_id, det.track_id)
                trail = self.track_trails.setdefault(key, [])
                trail.append((int(det.foot_point_px[0]), int(det.foot_point_px[1])))
                if len(trail) > 60:
                    del trail[:-60]
        alerts_start_perf = time.perf_counter()
        alerts_start_epoch = time.time()
        alerts = self.violation_detector.check(fused)
        alerts_end_perf = time.perf_counter()
        alerts_end_epoch = time.time()
        self._record_detections(self.frame_count, fused)
        self._record_alerts(self.frame_count, alerts, log_path)
        timing_payload = self._build_metadata_timing(
            trace_by_cam=latency_trace_by_cam or {},
            tracker_timings=tracker_timings,
            batch_ready_epoch=batch_ready_epoch,
            fusion_ms=(fusion_end_perf - fusion_start_perf) * 1000.0,
            alerts_ms=(alerts_end_perf - alerts_start_perf) * 1000.0,
            metadata_start_epoch=time.time(),
        )
        metadata_start_perf = time.perf_counter()
        metadata_start_epoch = time.time()
        self._publish_metadata(self.frame_count, fused, alerts, timing=timing_payload)
        metadata_end_perf = time.perf_counter()
        metadata_end_epoch = time.time()
        if self.latency_trace_path:
            self._record_latency_trace(
                frame_idx=self.frame_count,
                frames_raw=frames_raw,
                trace_by_cam=latency_trace_by_cam or {},
                tracker_timings=tracker_timings,
                batch_ready_epoch=batch_ready_epoch,
                fusion_start_epoch=fusion_start_epoch,
                fusion_end_epoch=fusion_end_epoch,
                fusion_ms=(fusion_end_perf - fusion_start_perf) * 1000.0,
                fused_detections=len(fused),
                alerts_start_epoch=alerts_start_epoch,
                alerts_end_epoch=alerts_end_epoch,
                alerts_ms=(alerts_end_perf - alerts_start_perf) * 1000.0,
                alerts_count=len(alerts),
                metadata_start_epoch=metadata_start_epoch,
                metadata_end_epoch=metadata_end_epoch,
                metadata_ms=(metadata_end_perf - metadata_start_perf) * 1000.0,
            )

    def _build_metadata_timing(
        self,
        trace_by_cam: dict[str, dict[str, Any]],
        tracker_timings: dict[str, dict[str, Any]],
        batch_ready_epoch: float | None,
        fusion_ms: float,
        alerts_ms: float,
        metadata_start_epoch: float,
    ) -> dict[str, Any]:
        """Agrège les chronométrages d'une frame pour le payload de métadonnées.

        Résume les durées de capture, d'inférence, de fusion et d'alerting,
        ainsi que la latence bout en bout capture -> métadonnées : pire cas
        (depuis la capture la plus ancienne) et dernier cas (depuis la plus
        récente).
        """
        def clean_ms(value: float | int | None) -> float:
            return round(float(value or 0.0), 3)

        def clean_epoch(value: float | int | None) -> float | None:
            if value in ("", None):
                return None
            return round(float(value), 6)

        capture_read_ms = [
            float(row.get("capture_read_ms") or 0.0)
            for row in trace_by_cam.values()
            if row.get("capture_success", True)
        ]
        capture_read_end_epochs = [
            float(row.get("capture_read_end_epoch"))
            for row in trace_by_cam.values()
            if row.get("capture_success", True) and row.get("capture_read_end_epoch")
        ]
        inference_ms = [
            float(row.get("inference_tracking_ms") or 0.0)
            for row in tracker_timings.values()
        ]
        inference_start_epochs = [
            float(row.get("inference_start_epoch"))
            for row in tracker_timings.values()
            if row.get("inference_start_epoch")
        ]
        inference_end_epochs = [
            float(row.get("inference_end_epoch"))
            for row in tracker_timings.values()
            if row.get("inference_end_epoch")
        ]
        first_capture_end = min(capture_read_end_epochs) if capture_read_end_epochs else None
        last_capture_end = max(capture_read_end_epochs) if capture_read_end_epochs else None

        return {
            "camera_count": len(trace_by_cam),
            "capture_read_ms_mean": clean_ms(sum(capture_read_ms) / len(capture_read_ms)) if capture_read_ms else 0.0,
            "capture_read_ms_max": clean_ms(max(capture_read_ms)) if capture_read_ms else 0.0,
            "capture_read_end_epoch_min": clean_epoch(first_capture_end),
            "capture_read_end_epoch_max": clean_epoch(last_capture_end),
            "batch_ready_epoch": clean_epoch(batch_ready_epoch),
            "inference_start_epoch_min": clean_epoch(min(inference_start_epochs) if inference_start_epochs else None),
            "inference_end_epoch_max": clean_epoch(max(inference_end_epochs) if inference_end_epochs else None),
            "inference_tracking_ms_mean": clean_ms(sum(inference_ms) / len(inference_ms)) if inference_ms else 0.0,
            "inference_tracking_ms_max": clean_ms(max(inference_ms)) if inference_ms else 0.0,
            "fusion_ms": clean_ms(fusion_ms),
            "alerts_ms": clean_ms(alerts_ms),
            "capture_to_metadata_ms_worst": clean_ms(
                (metadata_start_epoch - first_capture_end) * 1000.0 if first_capture_end else 0.0
            ),
            "capture_to_metadata_ms_latest": clean_ms(
                (metadata_start_epoch - last_capture_end) * 1000.0 if last_capture_end else 0.0
            ),
        }

    def _publish_metadata(
        self,
        frame_idx: int,
        detections: list[Any],
        alerts: list[Any],
        timing: dict[str, Any] | None = None,
    ) -> None:
        """Publie détections et alertes de la frame vers l'éditeur de métadonnées, s'il est configuré."""
        if not self.metadata_publisher:
            return
        from metadata_publisher import alert_to_metadata, detection_to_metadata

        detection_payload = [
            detection_to_metadata(det, zones=self.violation_detector.zones_containing(det))
            for det in detections
        ]
        alert_payload = [alert_to_metadata(alert) for alert in alerts]
        self.metadata_publisher.publish(
            frame_idx=frame_idx,
            run_label=self.model_spec.run_label,
            model_version=self.model_spec.version,
            model_name=self.model_spec.name,
            format_label=self.model_spec.format_label,
            detections=detection_payload,
            alerts=alert_payload,
            timing=timing,
        )

    def _record_latency_trace(
        self,
        frame_idx: int,
        frames_raw: dict[str, Any],
        trace_by_cam: dict[str, dict[str, Any]],
        tracker_timings: dict[str, dict[str, Any]],
        batch_ready_epoch: float | None,
        fusion_start_epoch: float,
        fusion_end_epoch: float,
        fusion_ms: float,
        fused_detections: int,
        alerts_start_epoch: float,
        alerts_end_epoch: float,
        alerts_ms: float,
        alerts_count: int,
        metadata_start_epoch: float,
        metadata_end_epoch: float,
        metadata_ms: float,
    ) -> None:
        """Ajoute une ligne de trace de latence par caméra pour la frame courante.

        Chaque ligne détaille horodatages et durées de toutes les étapes
        (capture, inférence, fusion, alertes, métadonnées, écriture vidéo).
        display_ms et total_loop_ms sont remplis ensuite par
        _finalize_latency_trace_frame, une fois l'affichage effectué.
        """
        def ms(value: float | int | None) -> float:
            return round(float(value or 0.0), 3)

        def ts(value: float | int | str | None) -> float | str:
            if value in ("", None):
                return ""
            return round(float(value), 6)

        for cam_id, frame in frames_raw.items():
            read = trace_by_cam.get(cam_id, {})
            infer = tracker_timings.get(cam_id, {})
            h, w = frame.shape[:2]
            read_end_epoch = read.get("capture_read_end_epoch")
            metadata_created_epoch = metadata_end_epoch
            self.latency_trace_rows.append(
                {
                    "model_version": self.model_spec.version,
                    "model": self.model_spec.name,
                    "format": self.model_spec.format_label,
                    "campaign_frame": frame_idx,
                    "cam_id": cam_id,
                    "capture_backend": read.get("capture_backend", ""),
                    "frame_width": w,
                    "frame_height": h,
                    "capture_success": read.get("capture_success", True),
                    "scene_time_epoch_manual": "",
                    "loop_start_epoch": ts(read.get("loop_start_epoch")),
                    "capture_read_start_epoch": ts(read.get("capture_read_start_epoch")),
                    "capture_read_end_epoch": ts(read_end_epoch),
                    "capture_read_ms": ms(read.get("capture_read_ms")),
                    "batch_ready_epoch": ts(batch_ready_epoch),
                    "frame_timestamp_epoch": ts(infer.get("frame_timestamp_epoch")),
                    "inference_start_epoch": ts(infer.get("inference_start_epoch")),
                    "inference_end_epoch": ts(infer.get("inference_end_epoch")),
                    "inference_tracking_ms": ms(infer.get("inference_tracking_ms")),
                    "detections_camera": infer.get("detections_camera", 0),
                    "fusion_start_epoch": ts(fusion_start_epoch),
                    "fusion_end_epoch": ts(fusion_end_epoch),
                    "fusion_ms": ms(fusion_ms),
                    "fused_detections": fused_detections,
                    "alerts_start_epoch": ts(alerts_start_epoch),
                    "alerts_end_epoch": ts(alerts_end_epoch),
                    "alerts_ms": ms(alerts_ms),
                    "alerts_count": alerts_count,
                    "metadata_start_epoch": ts(metadata_start_epoch),
                    "metadata_end_epoch": ts(metadata_end_epoch),
                    "metadata_ms": ms(metadata_ms),
                    "record_write_ms": ms(read.get("record_write_ms")),
                    "display_ms": 0.0,
                    "internal_after_read_ms": ms(
                        (metadata_created_epoch - float(read_end_epoch)) * 1000.0
                        if read_end_epoch
                        else 0.0
                    ),
                    "total_loop_ms": 0.0,
                }
            )

    def _finalize_latency_trace_frame(
        self,
        frame_idx: int,
        display_ms: float,
        total_loop_ms: float,
    ) -> None:
        """Complète a posteriori display_ms et total_loop_ms sur les lignes de
        trace de la frame donnée (parcours arrière jusqu'à la frame précédente)."""
        if not self.latency_trace_rows:
            return
        for row in reversed(self.latency_trace_rows):
            if row.get("campaign_frame") != frame_idx:
                break
            row["display_ms"] = round(display_ms, 3)
            row["total_loop_ms"] = round(total_loop_ms, 3)

    def _display(self, frames_raw: dict[str, Any], display_mode: str = "annotated") -> bool:
        """Affiche la mosaïque des caméras (frames corrigées, annotées selon le
        mode). Retourne True si l'utilisateur demande l'arrêt (touche q)."""
        from geometry_fix import fix_frame

        cells = []
        for cam_id in self.cameras:
            frame = frames_raw.get(cam_id)
            if frame is None:
                continue
            fixed = fix_frame(frame, cam_id, self.ar_fix)
            if display_mode == "annotated":
                fixed = self._draw_live_annotations(cam_id, fixed)
            cells.append(cv2.resize(fixed, (384, 288)))
        if cells:
            cv2.imshow("Phase3 Campaign", np.hstack(cells))
        return cv2.waitKey(1) & 0xFF == ord("q")

    def _draw_live_annotations(self, cam_id: str, frame: np.ndarray) -> np.ndarray:
        """Dessine boîtes, points au sol, traînées et étiquettes des détections
        courantes d'une caméra (couleur stable par identifiant global)."""
        vis = frame.copy()
        self._draw_zones_px(cam_id, vis)
        detections = self.last_detections_by_cam.get(cam_id, [])

        for det in detections:
            if not np.isfinite(det.bbox_px).all():
                continue
            x1, y1, x2, y2 = [int(v) for v in det.bbox_px]
            color = _id_to_color(det.global_id or det.track_id)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            fu, fv = int(det.foot_point_px[0]), int(det.foot_point_px[1])
            cv2.circle(vis, (fu, fv), 5, color, -1)
            trail = self.track_trails.get((cam_id, det.track_id), [])
            if len(trail) >= 2:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], False, color, 2)

            label = f"G{det.global_id or '?'} #{det.track_id} {det.class_name} {det.confidence:.2f}"
            if det.foot_point_m is not None:
                label += f" {det.foot_point_m[0]:.2f},{det.foot_point_m[1]:.2f}m"
            _draw_label(vis, label, x1, y1, color)

        cv2.putText(
            vis,
            cam_id,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    def _draw_zones_px(self, cam_id: str, frame: np.ndarray) -> None:
        """Projette les zones interdites (définies en mètres) dans l'image caméra.

        Utilise l'inverse de l'homographie du tracker pour ramener les
        polygones du plan sol vers les pixels, puis les dessine en
        surimpression rouge semi-transparente.
        """
        tracker = self.trackers.get(cam_id)
        if tracker is None or tracker.homography is None:
            return
        try:
            h_inv = np.linalg.inv(tracker.homography)
        except np.linalg.LinAlgError:
            return

        for zone_id, zone_data in self.config.get("zones_interdites", {}).items():
            if cam_id not in zone_data.get("cameras_concernees", []):
                continue
            coords_m = zone_data.get("coordonnees_metres") or []
            if len(coords_m) < 3:
                continue
            pts_m = np.array(coords_m, dtype=np.float32).reshape(-1, 1, 2)
            pts_px = cv2.perspectiveTransform(pts_m, h_inv).reshape(-1, 2)
            if not np.isfinite(pts_px).all():
                continue
            contour = np.rint(pts_px).astype(np.int32).reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [contour], (0, 0, 255))
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [contour], True, (0, 0, 255), 2)
            anchor = tuple(contour.reshape(-1, 2)[0])
            cv2.putText(
                frame,
                zone_id,
                (int(anchor[0]), int(anchor[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    def summary(self) -> dict[str, Any]:
        """Résumé chiffré de la campagne : volumes (frames, détections, alertes,
        liens de fusion), identifiants globaux uniques, switches d'identité et
        statistiques de latence d'inférence."""
        switches = sum(int(s["global_id_switches"]) for s in self.track_state.values())
        return {
            "model_version": self.model_spec.version,
            "model": self.model_spec.name,
            "format": self.model_spec.format_label,
            "frames": self.frame_count,
            "detections": len(self.detection_rows),
            "alerts": len(self.alert_rows),
            "fusion_links": len(self.fusion_rows),
            "unique_global_ids": len(
                {
                    row["global_id"]
                    for row in self.detection_rows
                    if row.get("global_id") not in ("", None)
                }
            ),
            "global_id_switches": switches,
            **summarize_latencies(self.latencies),
        }

    def write_outputs(self, out_dir: Path) -> None:
        """Écrit tous les CSV de la campagne dans out_dir : détections, alertes,
        liens de fusion, synchronisation, trace de latence (si activée) et
        stabilité des tracks."""
        write_csv(out_dir / "detections.csv", self.detection_rows, DETECTION_FIELDS)
        write_csv(out_dir / "alerts.csv", self.alert_rows, ALERT_FIELDS)
        write_csv(out_dir / "fusion_links.csv", self.fusion_rows, FUSION_FIELDS)
        write_csv(out_dir / "sync_events.csv", self.sync_rows, SYNC_FIELDS)
        if self.latency_trace_rows:
            trace_path = self.latency_trace_path or out_dir / "latency_trace.csv"
            write_csv(trace_path, self.latency_trace_rows, LATENCY_TRACE_FIELDS)
        track_rows = []
        for state in self.track_state.values():
            gids = state["global_ids"]
            track_rows.append(
                {
                    "model_version": self.model_spec.version,
                    "model": self.model_spec.name,
                    "format": self.model_spec.format_label,
                    "cam_id": state["cam_id"],
                    "track_id": state["track_id"],
                    "first_frame": state["first_frame"],
                    "last_frame": state["last_frame"],
                    "age": state["age"],
                    "global_id_switches": state["global_id_switches"],
                    "global_ids": "+".join(str(gid) for gid in sorted(set(gids))),
                }
            )
        write_csv(out_dir / "track_stability.csv", track_rows, TRACK_FIELDS)


# Colonnes des CSV produits par Phase3Campaign.write_outputs().
DETECTION_FIELDS = [
    "model_version", "model", "format", "frame", "timestamp", "cam_id", "track_id",
    "track_age", "global_id", "class_id", "class_name", "confidence", "x1", "y1",
    "x2", "y2", "foot_u", "foot_v", "x_m", "y_m", "inside_zones",
]
ALERT_FIELDS = [
    "model_version", "model", "format", "frame", "timestamp", "alert_id",
    "alert_type", "alert_level", "global_id", "zone_id", "class_name", "x_m", "y_m",
    "cameras", "confidence",
]
FUSION_FIELDS = [
    "model_version", "model", "format", "frame", "timestamp", "global_id",
    "cam_a", "track_a", "class_a", "cam_b", "track_b", "class_b",
    "time_delta_ms", "distance_m",
]
TRACK_FIELDS = [
    "model_version", "model", "format", "cam_id", "track_id", "first_frame",
    "last_frame", "age", "global_id_switches", "global_ids",
]
SYNC_FIELDS = [
    "model_version", "model", "format", "campaign_frame", "cam_id",
    "video_frame", "timestamp_epoch", "timestamp_iso", "elapsed_s",
    "recorded_video", "capture_backend",
]
LATENCY_TRACE_FIELDS = [
    "model_version", "model", "format", "campaign_frame", "cam_id",
    "capture_backend", "frame_width", "frame_height", "capture_success",
    "scene_time_epoch_manual", "loop_start_epoch", "capture_read_start_epoch",
    "capture_read_end_epoch", "capture_read_ms", "batch_ready_epoch",
    "frame_timestamp_epoch", "inference_start_epoch", "inference_end_epoch",
    "inference_tracking_ms", "detections_camera", "fusion_start_epoch",
    "fusion_end_epoch", "fusion_ms", "fused_detections", "alerts_start_epoch",
    "alerts_end_epoch", "alerts_ms", "alerts_count", "metadata_start_epoch",
    "metadata_end_epoch", "metadata_ms", "record_write_ms", "display_ms",
    "internal_after_read_ms", "total_loop_ms",
]


def compute_phase3_metrics(
    campaign: Phase3Campaign,
    cameras: list[str],
    gt_id_map: dict[str, str],
    out_dir: Path,
    tad_gt_path: Path = DEFAULT_GT_OBJECTS_TAD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calcule les métriques TRD et TAD par caméra à partir d'une campagne.

    TRD (temps de réaction, intrusions de personnes) : les alertes
    zone_violation_person impliquant la caméra sont appariées aux événements
    de gt_people (champ trame_violation) avec la tolérance de 10 s de
    match_alerts_to_gt.

    TAD (temps d'apparition, objets abandonnés) : faute d'alerte dédiée aux
    objets, chaque détection d'objet (classe autre que personne) tient lieu de
    candidat. Les timestamps sont regroupés par classe normalisée, restreints
    aux classes présentes dans la GT de la caméra, puis les faux positifs sont
    regroupés par fenêtre de TAD_FP_MERGE_WINDOW_S (3 s) pour reproduire la
    méthodologie d'evaluate_tad.py de la Phase 2 (sinon chaque frame non
    appariée compterait comme un FP et écraserait la précision).

    Écrit phase3_trd.csv et phase3_tad.csv dans out_dir.

    Returns:
        Tuple (lignes TRD, lignes TAD), une ligne par caméra.
    """
    gt_people_path = DEFAULT_GT_PEOPLE
    gt_tad_path = tad_gt_path
    trd_rows: list[dict[str, Any]] = []
    tad_rows: list[dict[str, Any]] = []

    for cam_id in cameras:
        gt_id = gt_id_map[cam_id]
        cam_detections = [r for r in campaign.detection_rows if r["cam_id"] == cam_id]
        fps = 25.0  # les vidéos Zone 1 sont enregistrées à 25 FPS (conversion trame GT -> secondes)

        # TRD : alertes d'intrusion de personne attribuées à la caméra (le
        # champ cameras vaut "cam_a+cam_b" ; vide signifie caméra inconnue,
        # l'alerte est alors comptée pour toutes les caméras).
        gt_people = load_gt_events(gt_people_path, gt_id)
        trd_alert_times = [
            float(r["timestamp"])
            for r in campaign.alert_rows
            if r["alert_type"] == "zone_violation_person"
            and (cam_id in str(r["cameras"]).split("+") or not r["cameras"])
        ]
        trd = match_alerts_to_gt(gt_people, trd_alert_times, "trame_violation", fps)
        trd_rows.append(
            {
                "model_version": campaign.model_spec.version,
                "model": campaign.model_spec.name,
                "format": campaign.model_spec.format_label,
                "camera": cam_id,
                "gt_camera": gt_id,
                "trd_median": trd["metric_median"],
                "trd_p95": trd["metric_p95"],
                "trd_mean": trd["metric_mean"],
                **{k: v for k, v in trd.items() if not k.startswith("metric_")},
            }
        )

        # TAD : les détections d'objets (classe autre que personne) tiennent
        # lieu d'alertes, regroupées par classe normalisée (sans accents).
        gt_tad = load_gt_events(gt_tad_path, gt_id)
        tad_alert_times_by_class: dict[str, list[float]] = {}
        for row in cam_detections:
            if int(row["class_id"]) == PERSON_CLASS_ID:
                continue
            tad_alert_times_by_class.setdefault(normalize_class(row["class_name"]), []).append(
                float(row["timestamp"])
            )
        # Seules les classes présentes dans la GT de la caméra sont retenues,
        # puis dédupliquées et triées avant appariement.
        all_tad_alert_times = []
        for ev in gt_tad:
            cls = normalize_class(ev.get("classe_objet", ""))
            all_tad_alert_times.extend(tad_alert_times_by_class.get(cls, []))
        # Le regroupement des FP sur 3 s aligne la Phase 3 sur la méthodologie
        # d'evaluate_tad.py (Phase 2) et rend les précisions comparables.
        tad = match_alerts_to_gt(
            gt_tad,
            sorted(set(all_tad_alert_times)),
            "trame_apparition",
            fps,
            fp_merge_window_s=TAD_FP_MERGE_WINDOW_S,
        )
        tad_rows.append(
            {
                "model_version": campaign.model_spec.version,
                "model": campaign.model_spec.name,
                "format": campaign.model_spec.format_label,
                "camera": cam_id,
                "gt_camera": gt_id,
                "tad_median": tad["metric_median"],
                "tad_p95": tad["metric_p95"],
                "tad_mean": tad["metric_mean"],
                **{k: v for k, v in tad.items() if not k.startswith("metric_")},
            }
        )

    write_csv(out_dir / "phase3_trd.csv", trd_rows, PHASE3_TRD_FIELDS)
    write_csv(out_dir / "phase3_tad.csv", tad_rows, PHASE3_TAD_FIELDS)
    return trd_rows, tad_rows


PHASE3_TRD_FIELDS = [
    "model_version", "model", "format", "camera", "gt_camera", "trd_median",
    "trd_p95", "trd_mean", "n_gt_events", "n_matched", "n_missed",
    "n_faux_positifs", "precision", "recall", "f1",
]
PHASE3_TAD_FIELDS = [
    "model_version", "model", "format", "camera", "gt_camera", "tad_median",
    "tad_p95", "tad_mean", "n_gt_events", "n_matched", "n_missed",
    "n_faux_positifs", "precision", "recall", "f1",
]


def run_subprocess(
    command: list[str],
    log_path: Path,
    cwd: Path,
    timeout_s: float | None = None,
) -> int:
    """Lance une commande en sous-processus avec sortie dupliquée console + log.

    La sortie est lue ligne à ligne (stderr fusionné dans stdout) et écrite au
    fil de l'eau. Codes de retour particuliers : 124 en cas de dépassement de
    timeout_s (convention de timeout(1)), 125 sur exception interne ; un
    Ctrl+C tue le fils puis se propage.

    Returns:
        Le code de retour du sous-processus (ou 124/125, voir ci-dessus).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Force l'UTF-8 et la sortie non bufferisée du fils pour un streaming
    # propre des logs (notamment sous Windows).
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with open(log_path, "a", encoding="utf-8") as log:
        command_text = "$ " + " ".join(command)
        print(command_text, flush=True)
        log.write(command_text + "\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        start = time.monotonic()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if timeout_s is not None and time.monotonic() - start > timeout_s:
                    proc.kill()
                    log.write(f"[TIMEOUT] killed after {timeout_s:.0f}s\n")
                    print(f"[TIMEOUT] killed after {timeout_s:.0f}s", flush=True)
                    return 124
            returncode = proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            log.write("[INTERRUPTED] child process killed\n")
            log.flush()
            raise
        except Exception as exc:
            proc.kill()
            log.write(f"[ERROR] child process killed: {type(exc).__name__}: {exc}\n")
            log.flush()
            return 125
        log.write(f"[EXIT] {returncode}\n")
        return returncode


def run_audit(log_path: Path, config_path: Path, out_csv: Path) -> int:
    """Lance audit_calibration_alerts.py sur le log de campagne (vérification
    de la calibration des alertes) et retourne son code de sortie."""
    return run_subprocess(
        [
            sys.executable,
            str(PHASE3_DIR / "audit_calibration_alerts.py"),
            "--log",
            str(log_path),
            "--config",
            str(config_path),
            "--out",
            str(out_csv),
        ],
        log_path.with_name(log_path.stem + "_audit.log"),
        PHASE3_DIR,
    )


def run_truth_ablation(truth_csv: Path, config_path: Path, out_csv: Path, log_path: Path) -> int:
    """Lance ablate_fusion_threshold.py sur le CSV de vérité (positions
    annotées) pour balayer les seuils de fusion 50/100/150/200 cm. Retourne le
    code de sortie du sous-processus."""
    return run_subprocess(
        [
            sys.executable,
            str(PHASE3_DIR / "ablate_fusion_threshold.py"),
            "--input",
            str(truth_csv),
            "--config",
            str(config_path),
            "--output",
            str(out_csv),
            "--thresholds-cm",
            "50,100,150,200",
        ],
        log_path,
        PHASE3_DIR,
    )


def run_operational_ablation(
    campaign: Phase3Campaign,
    config: dict[str, Any],
    out_csv: Path,
    thresholds_cm: Iterable[int] = (50, 100, 150, 200),
    time_window_ms: float = 500.0,
) -> list[dict[str, Any]]:
    """Ablation opérationnelle du seuil de fusion, sans vérité terrain.

    Rejoue les détections brutes de la campagne (personnes avec position sol
    valide) dans une fusion neuve pour chaque seuil testé, et compte les liens
    inter-caméras prédits et les global_id uniques. Contrairement à
    run_truth_ablation, aucune identité de référence n'est utilisée : la
    métrique est purement structurelle (mode operational_no_truth_id).

    Écrit le résultat dans out_csv et retourne les lignes produites.
    """
    from detection import Detection
    from fusion import MultiCameraFusion

    rows: list[dict[str, Any]] = []
    # Regroupe par frame les détections de personnes disposant d'une position
    # métrique valide (les autres ne peuvent pas être fusionnées).
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in campaign.raw_detection_rows:
        if row["x_m"] in ("", None) or row["y_m"] in ("", None):
            continue
        if int(row["class_id"]) != PERSON_CLASS_ID:
            continue
        grouped.setdefault(int(row["frame"]), []).append(row)

    for threshold_cm in thresholds_cm:
        # Une fusion neuve par seuil : son état temporel interne ne doit pas
        # fuir d'un seuil à l'autre.
        fusion = MultiCameraFusion(
            config=config,
            distance_threshold_m=threshold_cm / 100.0,
            time_window_s=time_window_ms / 1000.0,
        )
        predicted_links = 0
        unique_gids: set[int] = set()
        for frame_rows in grouped.values():
            detections_by_cam: dict[str, list[Detection]] = {}
            for row in frame_rows:
                det = Detection(
                    cam_id=str(row["cam_id"]),
                    track_id=int(row["track_id"]),
                    global_id=None,
                    timestamp=float(row["timestamp"]),
                    bbox_px=(0.0, 0.0, 0.0, 0.0),
                    foot_point_px=(0.0, 0.0),
                    foot_point_m=(float(row["x_m"]), float(row["y_m"])),
                    confidence=float(row["confidence"]),
                    class_id=int(row["class_id"]),
                    class_name=str(row["class_name"]),
                )
                detections_by_cam.setdefault(det.cam_id, []).append(det)
            fused = fusion.associate(detections_by_cam)
            by_gid: dict[int, set[str]] = {}
            for det in fused:
                if det.global_id is not None:
                    unique_gids.add(det.global_id)
                    by_gid.setdefault(det.global_id, set()).add(det.cam_id)
            # Un lien prédit = un global_id vu par au moins deux caméras sur la frame.
            predicted_links += sum(1 for cams in by_gid.values() if len(cams) >= 2)
        rows.append(
            {
                "threshold_cm": threshold_cm,
                "threshold_m": threshold_cm / 100.0,
                "frames": len(grouped),
                "predicted_links": predicted_links,
                "unique_global_ids": len(unique_gids),
                "global_id_switches": campaign.summary()["global_id_switches"],
                "alerts": len(campaign.alert_rows),
                "mode": "operational_no_truth_id",
            }
        )
    write_csv(out_csv, rows, OPERATIONAL_ABLATION_FIELDS)
    return rows


OPERATIONAL_ABLATION_FIELDS = [
    "threshold_cm", "threshold_m", "frames", "predicted_links",
    "unique_global_ids", "global_id_switches", "alerts", "mode",
]


def merge_csvs(paths: Iterable[Path], output_path: Path) -> None:
    """Concatène plusieurs CSV en un seul, en unifiant les en-têtes.

    Les fichiers absents sont ignorés ; les colonnes sont l'union ordonnée des
    en-têtes rencontrés. Sans aucune donnée, un fichier vide est produit.
    """
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(dict(row) for row in reader)
    if not fieldnames:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return
    write_csv(output_path, rows, fieldnames)


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    """Écrit un manifeste JSON indenté (UTF-8, accents conservés)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def stdout_to_log(log_path: Path):
    """Prépare la duplication de stdout vers un fichier de log.

    Returns:
        Tuple (fichier de log ouvert, contexte redirect_stdout à utiliser dans
        un with) ; l'appelant est responsable de fermer le fichier.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")
    return log, redirect_stdout(Tee(sys.stdout, log))

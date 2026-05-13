"""
Phase 2 - Evaluation du TRD (Temps de Reponse de Detection)

Evalue le delai entre l'instant reel d'une intrusion de personne dans une
zone interdite (verite terrain gt_people.json) et le moment ou le modele
detecte pour la premiere fois un pied dans cette zone.

Regles scientifiques implementees :
  1. Filtrage Classe : seule la classe "Personne" (ID configurable, defaut=14)
  2. Logique Spatiale : alerte ssi le point bas-centre (pieds) du bounding box
     tombe a l'interieur du polygone de la zone interdite (cv2.pointPolygonTest)
  3. Association Bipartite : matching glouton GT <-> alertes avec tolerance +-10s
  4. Metriques : TRD median, moyen, 95e percentile + FPS d'inference pur

Formats supportes : .pt (PyTorch), .onnx (ONNX/CPU), .engine (TensorRT/GPU)

Usage :
  python evaluate_trd.py --video video.mkv --camera cam_03
  python evaluate_trd.py --video video.mkv --camera cam_07 --models yolo11n yolo11s
  python evaluate_trd.py --video video.mkv --camera cam_05 --format onnx --conf 0.4
  python evaluate_trd.py --video video.mkv --camera cam_03 --person-class 0  # COCO
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION PAR DEFAUT
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Chemin vers le config.yaml contenant les zones interdites en pixels
CONFIG_PATH = PROJECT_ROOT / "ExperimentsCAS" / "config.yaml"
ZONES_JSON_PATH = SCRIPT_DIR / "zones_config.json"

# Verite terrain (un evenement = une intrusion reelle datee)
GT_PATH = PROJECT_ROOT / "gt_people.json"

# Dossier racine des poids entraines
RUNS_DIR = SCRIPT_DIR / "Modelstrained" / "V2"

# Dossier de sortie des resultats TRD
OUTPUT_DIR = SCRIPT_DIR / "Modelstrained" / "TRD"

# Liste des modeles a evaluer : (nom_court, sous_dossier, type_modele)
MODELS = [
    ("yolov8n",  "yolov8n",  "yolo"),
    ("yolov8s",  "yolov8s",  "yolo"),
    ("yolo26n",  "yolo26n",  "yolo"),
    ("yolo11n",  "yolo11n",  "yolo"),
    ("yolo11s",  "yolo11s",  "yolo"),
    ("rtdetr-l", "rtdetr-l", "rtdetr"),
]

# Parametres d'inference
CONFIDENCE   = 0.40
IMGSZ        = 640
FRAME_SKIP   = 1          # 1 = toutes les frames
ZONE_BASE_W  = 1280
ZONE_BASE_H  = 720

# Classe "Personne" dans le modele unifie
# Mapping actuel (runtime_data_for_training.yaml): personne = index 11
PERSON_CLASS_ID = 11

# Fenetre de tolerance pour le matching bipartite (en secondes)
TOLERANCE_SEC = 10.0
FP_MERGE_WINDOW_S = 3.0
TRD_ACTIVE_WINDOW_S = 30.0

# Nombre de frames consecutives pour confirmer une alerte (anti-bruit)
MIN_PERSISTENCE = 3


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DU CONFIG.YAML
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> dict:
    """Charge le fichier config.yaml."""
    if not config_path.exists():
        sys.exit(f"[ERREUR] config.yaml introuvable : {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_camera_config(config: dict, camera_id: str) -> dict | None:
    """Extrait la configuration d'une camera.

    Supporte deux formats :
      - dict : cameras: {cam_03: {...}}               (Phase_3 style)
      - liste : cameras: [{camera_id: cam_03, ...}]   (ExperimentsCAS style)
    """
    cameras = config.get("cameras", {})

    if isinstance(cameras, dict):
        return cameras.get(camera_id)

    if isinstance(cameras, list):
        for cam in cameras:
            if cam.get("camera_id") == camera_id:
                return cam

    return None


def base_camera_id(camera_id: str) -> str:
    """Return cam_XX from IDs like cam_XX_20260506_131002."""
    match = re.match(r"^(cam_\d{2})", camera_id)
    return match.group(1) if match else camera_id


def select_homography_matrix(config: dict, camera_id: str) -> np.ndarray | None:
    """Load Phase 3 homography matrix for a camera if present."""
    homo = config.get("homographie", {})
    cam_entry = homo.get(camera_id)
    if isinstance(cam_entry, dict) and cam_entry.get("matrix"):
        return np.array(cam_entry["matrix"], dtype=np.float32)
    flat = homo.get(f"{camera_id}_matrix")
    if flat:
        return np.array(flat, dtype=np.float32)
    return None


def load_zone_from_phase3_projection(config: dict, camera_id: str) -> np.ndarray | None:
    """Project Phase 3 floor-plane forbidden-zone metres to camera pixels."""
    h_matrix = select_homography_matrix(config, camera_id)
    if h_matrix is None:
        return None

    zones = config.get("zones_interdites", {})
    for _zone_id, zone_data in zones.items():
        cameras = zone_data.get("cameras_concernees") or []
        coords_m = zone_data.get("coordonnees_metres") or []
        if camera_id not in cameras or len(coords_m) < 3:
            continue
        try:
            h_inv = np.linalg.inv(h_matrix)
        except np.linalg.LinAlgError:
            return None
        pts_m = np.array(coords_m, dtype=np.float32).reshape(-1, 1, 2)
        pts_px = cv2.perspectiveTransform(pts_m, h_inv).reshape(-1, 2)
        return np.rint(pts_px).astype(np.int32)
    return None


def phase3_projection_base_resolution(config: dict, camera_id: str) -> tuple[int, int] | None:
    """Return the pixel reference size used by Phase 3 projected zones."""
    if load_zone_from_phase3_projection(config, camera_id) is None:
        return None

    ar_fix = config.get("aspect_ratio_fix", {})
    if ar_fix.get("enabled") and camera_id in ar_fix.get("distorted_cameras", []):
        corrected = ar_fix.get("corrected_resolution")
        if corrected and len(corrected) == 2:
            return int(corrected[0]), int(corrected[1])

    video_cfg = config.get("video", {}).get("resolution")
    if video_cfg and len(video_cfg) == 2:
        return int(video_cfg[0]), int(video_cfg[1])

    return None


def load_zone_from_config(config: dict, camera_id: str) -> np.ndarray:
    """Charge le polygone de la zone interdite (coordonnees pixel) depuis config.yaml.

    Cherche la cle 'forbidden_zone_pixels' dans la configuration de la camera.
    Retourne un np.ndarray de shape (N, 2) dtype int32.
    """
    cam_cfg = get_camera_config(config, camera_id)
    base_id = base_camera_id(camera_id)
    if cam_cfg is None and base_id != camera_id:
        cam_cfg = get_camera_config(config, base_id)

    phase3_zone = load_zone_from_phase3_projection(config, base_id)
    if phase3_zone is not None:
        return phase3_zone

    if cam_cfg is None:
        sys.exit(f"[ERREUR] Camera '{camera_id}' introuvable dans le config.yaml.\n"
                 f"         Cameras disponibles : {list_available_cameras(config)}")

    zone_pts = cam_cfg.get("forbidden_zone_pixels")
    if zone_pts is None or len(zone_pts) < 3:
        sys.exit(f"[ERREUR] Pas de 'forbidden_zone_pixels' pour '{camera_id}' dans le config.\n"
                 f"         Un polygone de 3 sommets minimum est requis.")

    return np.array(zone_pts, dtype=np.int32)


def load_zone_from_zones_json(zones_json_path: Path, camera_id: str) -> np.ndarray | None:
    """Charge le polygone depuis zones_config.json (Phase_2), si present."""
    if not zones_json_path.exists():
        return None

    with open(zones_json_path, "r", encoding="utf-8") as f:
        zones = json.load(f) or {}

    zone_pts = zones.get(camera_id) or zones.get(base_camera_id(camera_id))
    if zone_pts is None or len(zone_pts) < 3:
        return None
    return np.array(zone_pts, dtype=np.int32)


def list_available_cameras(config: dict) -> list[str]:
    """Liste les identifiants de cameras disponibles."""
    cameras = config.get("cameras", {})
    if isinstance(cameras, dict):
        return list(cameras.keys())
    if isinstance(cameras, list):
        return [c.get("camera_id", "?") for c in cameras]
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DE LA VERITE TERRAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_gt(gt_path: Path, camera_id: str) -> list[dict]:
    """Charge les evenements de violation GT pour une camera donnee.

    Format attendu dans le JSON :
      [{"id_camera": "cam_XX", "id_evenement": N,
        "trame_violation": F, "horodatage_violation": T}, ...]

    Retourne la liste triee par trame_violation (croissant).
    """
    if not gt_path.exists():
        sys.exit(f"[ERREUR] Fichier GT introuvable : {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        all_events = json.load(f)

    events = [e for e in all_events if e["id_camera"] == camera_id]
    events.sort(key=lambda x: x["trame_violation"])

    if not events:
        cams = sorted(set(e["id_camera"] for e in all_events))
        cam_counts = {c: sum(1 for e in all_events if e["id_camera"] == c) for c in cams}
        print(f"[ERREUR] Aucun evenement GT pour la camera '{camera_id}' dans {gt_path.name}")
        print(f"         Cameras disponibles :")
        for c in cams:
            print(f"           {c} ({cam_counts[c]} evenements)")
        sys.exit(1)

    return events


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def detect_device(device_mode: str = "auto") -> tuple[str, str]:
    """Selectionne le device d'inference (GPU/CPU).
    Retourne (device_id, device_label) pour affichage.
    """
    mode = device_mode.lower().strip()

    if mode == "cpu":
        return "cpu", "CPU"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "0", name

    if mode == "gpu":
        print("  [ATTENTION] GPU force mais CUDA indisponible -- fallback CPU.")
    return "cpu", "CPU (pas de GPU CUDA)"


def load_model(weights: str, model_type: str):
    """Instancie un modele Ultralytics (YOLO ou RTDETR).

    Supporte les formats .pt, .onnx et .engine (TensorRT) de maniere
    transparente via l'API Ultralytics.
    """
    if model_type == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    else:
        from ultralytics import YOLO
        return YOLO(weights)


def compute_feet_point(box) -> tuple[int, int]:
    """Calcule le point bas-centre (pieds) d'une bounding box [x1, y1, x2, y2].

    x_pieds = (x_min + x_max) / 2
    y_pieds = y_max
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def point_in_polygon(x: float, y: float, contour: np.ndarray) -> bool:
    """Teste si (x, y) est a l'interieur du polygone via cv2.pointPolygonTest."""
    return cv2.pointPolygonTest(contour, (float(x), float(y)), measureDist=False) >= 0


def rescale_polygon(polygon: np.ndarray,
                    base_w: int, base_h: int,
                    target_w: int, target_h: int) -> np.ndarray:
    """Rescale un polygone pixel d'une resolution de reference vers la video cible."""
    if base_w <= 0 or base_h <= 0:
        return polygon.astype(np.int32)

    if base_w == target_w and base_h == target_h:
        return polygon.astype(np.int32)

    sx = target_w / float(base_w)
    sy = target_h / float(base_h)

    scaled = polygon.astype(np.float32).copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    return np.rint(scaled).astype(np.int32)


def format_duration(seconds: float) -> str:
    """Formate une duree en h/m/s lisible."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _finalize_fp_cluster(alerts: list[dict]) -> dict:
    """Transforme des alertes FP brutes proches en un evenement FP."""
    confidences = [float(a.get("confidence", 0.0)) for a in alerts]
    return {
        "start_time": round(float(alerts[0]["timestamp"]), 3),
        "end_time": round(float(alerts[-1]["timestamp"]), 3),
        "duration_s": round(float(alerts[-1]["timestamp"] - alerts[0]["timestamp"]), 3),
        "n_detections": len(alerts),
        "class_name": alerts[0].get("class_name", "personne"),
        "avg_confidence": round(float(sum(confidences) / len(confidences)), 4),
    }


def cluster_false_positives(unmatched_alerts: list[dict],
                            merge_window_s: float = FP_MERGE_WINDOW_S) -> list[dict]:
    """Regroupe les fausses alertes proches temporellement en evenements FP."""
    if not unmatched_alerts:
        return []

    sorted_alerts = sorted(unmatched_alerts, key=lambda a: a["timestamp"])
    clusters = []
    current_cluster = [sorted_alerts[0]]

    for alert in sorted_alerts[1:]:
        if alert["timestamp"] - current_cluster[-1]["timestamp"] <= merge_window_s:
            current_cluster.append(alert)
        else:
            clusters.append(_finalize_fp_cluster(current_cluster))
            current_cluster = [alert]

    clusters.append(_finalize_fp_cluster(current_cluster))
    return clusters


def compute_extended_metrics(tp: int, fn: int, fp_clusters: list[dict],
                             video_duration_s: float) -> dict:
    """Calcule precision, recall, F1 et FAR sans modifier TP/FN historiques."""
    fp = len(fp_clusters)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    video_duration_h = video_duration_s / 3600.0
    far = fp / video_duration_h if video_duration_h > 0 else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "fp_raw_detections": sum(c["n_detections"] for c in fp_clusters),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "far_per_hour": round(far, 2),
        "video_duration_s": round(video_duration_s, 1),
        "detected": f"{tp}/{tp + fn}",
    }


def compute_latency_metrics(frame_times: list[float]) -> dict:
    """Retourne les metriques de latence du pipeline d'inference."""
    if not frame_times:
        return {
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_max_ms": 0.0,
            "latency_jitter_ms": 0.0,
        }

    arr_ms = np.array(frame_times, dtype=np.float64) * 1000.0
    return {
        "latency_mean_ms": round(float(arr_ms.mean()), 1),
        "latency_median_ms": round(float(np.median(arr_ms)), 1),
        "latency_p95_ms": round(float(np.percentile(arr_ms, 95)), 1),
        "latency_max_ms": round(float(arr_ms.max()), 1),
        "latency_jitter_ms": round(float(arr_ms.std()), 1),
    }


def gt_active_end_time(ev: dict, vid_fps: float,
                       active_window_s: float = TRD_ACTIVE_WINDOW_S) -> float:
    """Retourne la fin de presence GT si disponible, sinon entree + 30s."""
    for key in ("horodatage_sortie", "t_sortie_sec", "temps_sortie"):
        if key in ev and ev[key] is not None:
            return float(ev[key])
    for key in ("trame_sortie", "frame_sortie"):
        if key in ev and ev[key] is not None:
            return float(ev[key]) / vid_fps
    return (ev["trame_violation"] / vid_fps) + active_window_s


def is_trd_alert_inside_active_gt(alert: dict, gt_events: list[dict],
                                  vid_fps: float) -> bool:
    """Vrai si une alerte intrusion tombe pendant une presence GT active."""
    alert_time = alert["timestamp"]
    for ev in gt_events:
        gt_start = ev["trame_violation"] / vid_fps
        gt_end = gt_active_end_time(ev, vid_fps)
        if gt_start <= alert_time <= gt_end:
            return True
    return False


def print_fp_details(fp_clusters: list[dict]) -> None:
    """Affiche les evenements de faux positifs pour debug."""
    if not fp_clusters:
        print("    Aucun faux positif clusterise.")
        return

    print("\n  Detail des Faux Positifs :")
    for idx, fp in enumerate(fp_clusters, 1):
        print(
            f"    FP #{idx} : t={fp['start_time']:.3f}s-{fp['end_time']:.3f}s "
            f"({fp['duration_s']:.3f}s) | {fp['n_detections']} detections | "
            f"conf_moy={fp['avg_confidence']:.3f} | classe={fp['class_name']}"
        )


def write_results_csv(csv_path: Path, results: list[dict]) -> None:
    """Exporte les metriques principales dans un CSV lisible par tableur."""
    fieldnames = [
        "model", "weights", "fps_inference", "detected", "n_gt_events",
        "n_matched", "n_missed", "n_alertes", "n_faux_positifs",
        "fp_raw_detections", "precision", "recall", "f1", "far_per_hour",
        "trd_median", "trd_p95", "trd_mean", "trd_min", "trd_max",
        "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
        "latency_max_ms", "latency_jitter_ms", "frames_analysees",
        "total_frames", "frame_skip", "video_duration_s",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({key: r.get(key) for key in fieldnames})


# ══════════════════════════════════════════════════════════════════════════════
#  MATCHING BIPARTITE GT <-> ALERTES
# ══════════════════════════════════════════════════════════════════════════════

def bipartite_matching(gt_times: list[float], alert_times: list[float],
                       tolerance: float) -> list[dict]:
    """Association bipartite gloutonne entre violations GT et alertes detectees.

    Algorithme :
      1. Enumere toutes les paires (gt_i, alert_j) ou |t_alert - t_gt| <= tolerance
      2. Trie ces paires par distance temporelle absolue croissante
      3. Assigne de maniere gloutonne (chaque GT et chaque alerte au plus une fois)

    Retourne une liste (une entree par GT) :
      {"gt_idx": i, "alert_idx": j ou None, "trd": float ou None}
    """
    n_gt = len(gt_times)

    # Enumerer les paires candidates
    candidates = []
    for i, t_gt in enumerate(gt_times):
        for j, t_al in enumerate(alert_times):
            delta = t_al - t_gt
            if abs(delta) <= tolerance:
                candidates.append((abs(delta), delta, i, j))

    # Trier par proximite
    candidates.sort(key=lambda c: c[0])

    # Assignation gloutonne
    matched_gt = set()
    matched_al = set()
    matches = {}

    for _, delta, i, j in candidates:
        if i in matched_gt or j in matched_al:
            continue
        matches[i] = {"gt_idx": i, "alert_idx": j, "trd": delta}
        matched_gt.add(i)
        matched_al.add(j)

    # Completer avec les GT non apparies
    results = []
    for i in range(n_gt):
        if i in matches:
            results.append(matches[i])
        else:
            results.append({"gt_idx": i, "alert_idx": None, "trd": None})

    return results


def count_detected_so_far(gt_frames: list[int], alert_frames: list[int],
                          tolerance_frames: int) -> int:
    """Compte combien de GT sont couvertes par au moins une alerte (pour la progression)."""
    count = 0
    used_alerts = set()
    for gt_f in gt_frames:
        for j, al_f in enumerate(alert_frames):
            if j in used_alerts:
                continue
            if abs(al_f - gt_f) <= tolerance_frames:
                count += 1
                used_alerts.add(j)
                break
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION D'UN MODELE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name: str, weights_path: Path, model_type: str,
                   video_path: str, gt_events: list[dict],
                   zone_polygon: np.ndarray, person_class: int,
                   conf: float, device: str,
                   tolerance_sec: float,
                   frame_skip: int = 1,
                   view: bool = False,
                   min_persistence: int = MIN_PERSISTENCE,
                   zone_base_w: int = ZONE_BASE_W,
                   zone_base_h: int = ZONE_BASE_H) -> dict | None:
    """Evalue un modele en TRD sur une video complete."""

    if not weights_path.exists():
        print(f"  [SKIP] Poids introuvables : {weights_path}")
        return None

    n_gt = len(gt_events)

    print(f"\n{'=' * 60}")
    print(f"  MODELE : {model_name}")
    print(f"  Poids  : {weights_path}")
    print(f"{'=' * 60}\n")

    # -- 1. Chargement du modele --
    model = load_model(str(weights_path), model_type)
    model_names = getattr(model, "names", {})
    person_class_name = model_names.get(person_class, "personne")

    use_half = (device != "cpu")
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model.predict(dummy, device=device, conf=conf, half=use_half, verbose=False)

    # -- 2. Ouverture de la video --
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERREUR] Impossible d'ouvrir la video : {video_path}")
        return None

    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tolerance_frames = int(tolerance_sec * vid_fps)
    gt_violation_frames = [ev["trame_violation"] for ev in gt_events]

    print(f"  Video  : {Path(video_path).name} ({total_frames} frames, {vid_fps:.1f} fps)")
    print(f"  GT     : {n_gt} violations a detecter")
    print(f"  Conf   : {conf}")
    skip_label = f"1/{frame_skip}" if frame_skip > 1 else "toutes"
    print(f"  Skip   : {skip_label} frames analysees")

    # -- 3. Zone : rescaling auto vers la resolution video --
    zone_polygon_scaled = rescale_polygon(
        polygon=zone_polygon,
        base_w=zone_base_w,
        base_h=zone_base_h,
        target_w=src_w,
        target_h=src_h,
    )
    zone_contour = zone_polygon_scaled.reshape(-1, 1, 2).astype(np.float32)

    if zone_base_w != src_w or zone_base_h != src_h:
        print(f"  Zone   : rescale auto {zone_base_w}x{zone_base_h} -> {src_w}x{src_h}")
    else:
        print(f"  Zone   : sans rescale ({src_w}x{src_h})")

    # -- 4. Boucle d'inference --
    alert_frames: list[int] = []
    alert_system_indices: list[int] = []
    system_alerts: list[dict] = []
    inference_times: list[float] = []
    consecutive_in_zone = 0
    frame_idx = 0

    print(f"  Persist  : {min_persistence} frames consecutives pour confirmer")

    window_name = None
    if view:
        window_name = f"TRD View - {model_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Inference
        t0 = time.perf_counter()
        results = model.predict(
            frame,
            classes=[person_class],
            conf=conf,
            device=device,
            imgsz=IMGSZ,
            half=use_half,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        inference_times.append(dt)

        # Pieds dans la zone ?
        boxes = results[0].boxes
        anyone_in_zone = False
        best_in_zone_conf = 0.0
        best_in_zone_position: tuple[int, int] | None = None

        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            x_pieds, y_pieds = compute_feet_point(box)
            in_zone = point_in_polygon(x_pieds, y_pieds, zone_contour)
            if in_zone:
                anyone_in_zone = True
                conf_value = float(boxes.conf[i].cpu().item())
                if conf_value >= best_in_zone_conf:
                    best_in_zone_conf = conf_value
                    best_in_zone_position = (x_pieds, y_pieds)

            if view:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if in_zone else (0, 200, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (x_pieds, y_pieds), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x_pieds, y_pieds), 10, (0, 0, 0), 2)
                if in_zone:
                    cv2.putText(frame, "PIED IN ZONE",
                                (x_pieds + 14, y_pieds - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)

            if anyone_in_zone and not view:
                break

        # Compteur de persistence (anti-bruit)
        if anyone_in_zone:
            consecutive_in_zone += 1
        else:
            consecutive_in_zone = 0

        # Alerte brute a chaque frame confirmee; alert_frames garde l'ancien
        # comportement TRD (premiere frame confirmee de chaque episode).
        if consecutive_in_zone >= min_persistence:
            system_alerts.append({
                "timestamp": frame_idx / vid_fps,
                "frame": frame_idx,
                "class_id": person_class,
                "class_name": person_class_name,
                "confidence": best_in_zone_conf,
                "position": best_in_zone_position,
                "matched_gt": None,
            })
            if consecutive_in_zone == min_persistence:
                alert_frames.append(frame_idx)
                alert_system_indices.append(len(system_alerts) - 1)

        frame_idx += 1

        # Progression toutes les 2000 frames
        if frame_idx % 2000 == 0:
            pct = frame_idx / total_frames * 100
            n_det = count_detected_so_far(gt_violation_frames, alert_frames,
                                          tolerance_frames)
            print(f"    frame {frame_idx}/{total_frames} ({pct:.0f}%)"
                  f" -- {n_det}/{n_gt} detectes")

        if view:
            # Zone avec remplissage semi-transparent
            intrusion_active = consecutive_in_zone >= min_persistence
            zone_color = (0, 0, 255) if intrusion_active else (0, 180, 0)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_polygon_scaled], zone_color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [zone_polygon_scaled], True, zone_color, 2)

            # HUD
            cv2.putText(frame, f"Frame: {frame_idx - 1}/{total_frames}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"Alertes: {len(alert_frames)}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"Persist: {consecutive_in_zone}/{min_persistence}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            if intrusion_active:
                txt = "INTRUSION CONFIRMEE"
                (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                              1.0, 3)
                tx = (src_w - tw) // 2
                cv2.putText(frame, txt, (tx, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    if view:
        cv2.destroyAllWindows()

    # -- 5. FPS d'inference pur --
    fps_moyen = 1.0 / np.mean(inference_times) if inference_times else 0.0
    frames_analysees = len(inference_times)

    # -- 6. Matching bipartite --
    gt_times_sec    = [ev["trame_violation"] / vid_fps for ev in gt_events]
    alert_times_sec = [f / vid_fps for f in alert_frames]

    matches = bipartite_matching(gt_times_sec, alert_times_sec, tolerance_sec)
    for match in matches:
        if match["alert_idx"] is not None:
            alert_idx = alert_system_indices[match["alert_idx"]]
            system_alerts[alert_idx]["matched_gt"] = gt_events[match["gt_idx"]]["id_evenement"]

    # -- 7. Calcul des metriques --
    trd_values: list[float] = []
    missed = 0
    details: list[dict] = []

    for match in matches:
        i = match["gt_idx"]
        ev = gt_events[i]

        detail = {
            "id_evenement":    ev["id_evenement"],
            "trame_violation": ev["trame_violation"],
            "t_violation_sec": round(gt_times_sec[i], 3),
        }

        if match["trd"] is not None:
            j = match["alert_idx"]
            trd_sec = match["trd"]
            trd_values.append(trd_sec)
            delai_frames = alert_frames[j] - ev["trame_violation"]

            detail["trame_alerte"]  = alert_frames[j]
            detail["t_alerte_sec"]  = round(alert_times_sec[j], 3)
            detail["delai_trames"]  = delai_frames
            detail["trd_sec"]       = round(trd_sec, 3)
        else:
            missed += 1
            detail["trame_alerte"] = None
            detail["t_alerte_sec"] = None
            detail["delai_trames"] = None
            detail["trd_sec"]      = None

        details.append(detail)

    trd_arr = np.array(trd_values) if trd_values else np.array([])
    video_duration_s = (total_frames if total_frames > 0 else frame_idx) / vid_fps
    unmatched_alerts = [
        a for a in system_alerts
        if a["matched_gt"] is None
        and not is_trd_alert_inside_active_gt(a, gt_events, vid_fps)
    ]
    fp_clusters = cluster_false_positives(unmatched_alerts)
    extended_metrics = compute_extended_metrics(
        tp=len(trd_values),
        fn=missed,
        fp_clusters=fp_clusters,
        video_duration_s=video_duration_s,
    )
    latency_metrics = compute_latency_metrics(inference_times)

    result = {
        "model":            model_name,
        "weights":          str(weights_path),
        "video":            video_path,
        "resolution":       f"{src_w}x{src_h}",
        "vid_fps":          vid_fps,
        "total_frames":     frame_idx,
        "frames_analysees": frames_analysees,
        "frame_skip":       frame_skip,
        "fps_inference":    round(fps_moyen, 1),
        "n_gt_events":      n_gt,
        "n_alertes":        len(alert_frames),
        "n_matched":        len(trd_values),
        "n_missed":         missed,
        "n_faux_positifs":  extended_metrics["fp"],
        "fp_raw_detections": extended_metrics["fp_raw_detections"],
        "recall":           extended_metrics["recall"],
        "precision":        extended_metrics["precision"],
        "f1":               extended_metrics["f1"],
        "far_per_hour":     extended_metrics["far_per_hour"],
        "video_duration_s": extended_metrics["video_duration_s"],
        "detected":         extended_metrics["detected"],
        "latency_mean_ms":  latency_metrics["latency_mean_ms"],
        "latency_median_ms": latency_metrics["latency_median_ms"],
        "latency_p95_ms":   latency_metrics["latency_p95_ms"],
        "latency_max_ms":   latency_metrics["latency_max_ms"],
        "latency_jitter_ms": latency_metrics["latency_jitter_ms"],
        "trd_median": round(float(np.median(trd_arr)), 3) if trd_arr.size > 0 else None,
        "trd_p95":    round(float(np.percentile(trd_arr, 95)), 3) if trd_arr.size > 0 else None,
        "trd_mean":   round(float(np.mean(trd_arr)), 3) if trd_arr.size > 0 else None,
        "trd_min":    round(float(np.min(trd_arr)), 3) if trd_arr.size > 0 else None,
        "trd_max":    round(float(np.max(trd_arr)), 3) if trd_arr.size > 0 else None,
        "details":    details,
        "fp_clusters": fp_clusters,
    }

    # -- 8. Tableau par evenement --
    print(f"\n  {'Evt':>4s}  {'Violation':>10s}  {'Alerte':>10s}  "
          f"{'Delai':>8s}  {'TRD (s)':>10s}")
    print(f"  {'----':>4s}  {'----------':>10s}  {'----------':>10s}  "
          f"{'--------':>8s}  {'----------':>10s}")

    for d in details:
        evt  = d["id_evenement"]
        viol = str(d["trame_violation"])
        if d["trame_alerte"] is not None:
            al    = str(d["trame_alerte"])
            delai = str(d["delai_trames"])
            trd_s = f"{d['trd_sec']:.3f}"
        else:
            al    = "MANQUE"
            delai = "--"
            trd_s = "--"
        print(f"  {evt:>4d}  {viol:>10s}  {al:>10s}  "
              f"{delai:>8s}  {trd_s:>10s}")

    # -- 9. Resume du modele --
    print(f"\n  Resultats {model_name} :")
    print(f"    Frames       : {frames_analysees}/{frame_idx} analysees (skip={frame_skip})")
    print(f"    FPS moyen    : {fps_moyen:.1f}")
    if trd_arr.size > 0:
        print(f"    TRD median   : {result['trd_median']:.3f}s")
        print(f"    TRD 95e pct  : {result['trd_p95']:.3f}s")
        print(f"    TRD moyen    : {result['trd_mean']:.3f}s")
        print(f"    TRD min/max  : {result['trd_min']:.3f}s / {result['trd_max']:.3f}s")
    print(f"    Detectes     : {result['n_matched']}/{n_gt}")
    print(f"    Manques      : {missed}")
    print(f"    Faux positifs: {result['n_faux_positifs']} "
          f"({result['fp_raw_detections']} detections brutes)")
    print(f"    Recall       : {result['recall']:.4f}")
    print(f"    Precision    : {result['precision']:.4f}")
    print(f"    F1-Score     : {result['f1']:.4f}")
    print(f"    FAR          : {result['far_per_hour']:.2f} fausses alertes / heure")
    print(f"    Latence moy. : {result['latency_mean_ms']:.1f} ms")
    print(f"    Latence 95%  : {result['latency_p95_ms']:.1f} ms")
    print(f"    Jitter       : {result['latency_jitter_ms']:.1f} ms")
    print_fp_details(fp_clusters)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TABLEAU RECAPITULATIF MULTI-MODELES
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], camera_id: str):
    """Affiche un tableau comparatif final de tous les modeles evalues."""

    print(f"\n\n{'=' * 112}")
    print(f"  RECAPITULATIF TRD -- Camera : {camera_id}")
    print(f"{'=' * 112}")
    print(f"  {'Modele':<12s}  {'FPS':>6s}  {'Detect':>7s}  "
          f"{'Precision':>9s}  {'Recall':>7s}  {'F1':>7s}  {'FAR(/h)':>8s}  "
          f"{'FP':>4s}  {'TRD Med':>8s}  {'TRD 95%':>8s}  {'TRD Moy':>8s}")
    print(f"  {'------------':<12s}  {'------':>6s}  {'-------':>7s}  "
          f"{'---------':>9s}  {'-------':>7s}  {'-------':>7s}  {'--------':>8s}  "
          f"{'----':>4s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}")

    for r in results:
        fps = f"{r['fps_inference']:.1f}"
        med = f"{r['trd_median']:.3f}" if r["trd_median"] is not None else "--"
        p95 = f"{r['trd_p95']:.3f}"    if r["trd_p95"]    is not None else "--"
        moy = f"{r['trd_mean']:.3f}"   if r["trd_mean"]   is not None else "--"
        det = r.get("detected", f"{r['n_matched']}/{r['n_gt_events']}")
        precision = f"{r['precision']:.4f}"
        recall = f"{r['recall']:.4f}"
        f1 = f"{r['f1']:.4f}"
        far = f"{r['far_per_hour']:.2f}"
        fp = str(r["n_faux_positifs"])
        print(f"  {r['model']:<12s}  {fps:>6s}  {det:>7s}  "
              f"{precision:>9s}  {recall:>7s}  {f1:>7s}  {far:>8s}  "
              f"{fp:>4s}  {med:>8s}  {p95:>8s}  {moy:>8s}")

    print(f"{'=' * 112}")

    # Meilleur modele
    valid = [r for r in results if r["trd_median"] is not None]
    if valid:
        best_trd = min(valid, key=lambda r: r["trd_median"])
        fastest  = max(valid, key=lambda r: r["fps_inference"])
        print(f"\n  Meilleur TRD median : {best_trd['model']} "
              f"({best_trd['trd_median']:.3f}s)")
        print(f"  Plus rapide (FPS)   : {fastest['model']} "
              f"({fastest['fps_inference']:.1f} fps)")


def infer_camera_id_from_video(video_path: Path) -> str | None:
    """Infer cam_XX from a video name like Camera_3_..."""
    match = re.search(r"camera_(\d+)_", video_path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return f"cam_{int(match.group(1)):02d}"


def resolve_jobs(args) -> list[tuple[Path, str]]:
    """Build a list of (video_path, camera_id) jobs."""
    videos: list[Path] = []
    cameras: list[str] = []

    if args.videos:
        videos = [Path(v) for v in args.videos]
        if args.cameras:
            if len(args.cameras) != len(videos):
                sys.exit("[ERREUR] --cameras doit avoir le meme nombre d'elements que --videos.")
            cameras = [c.strip() for c in args.cameras]
        else:
            for video in videos:
                inferred = infer_camera_id_from_video(video)
                if inferred is None:
                    sys.exit(
                        "[ERREUR] Impossible d'inferer la camera depuis le nom de video. "
                        "Utilisez --cameras explicitement."
                    )
                cameras.append(inferred)
    else:
        if not args.video:
            sys.exit("[ERREUR] Fournissez --video (mode simple) ou --videos (mode batch).")
        video = Path(args.video)
        camera = args.camera.strip() if args.camera else infer_camera_id_from_video(video)
        if not camera:
            sys.exit(
                "[ERREUR] Camera manquante. Fournissez --camera ou utilisez un nom "
                "de video compatible (ex: Camera_3_...)."
            )
        videos = [video]
        cameras = [camera]

    for video in videos:
        if not video.exists():
            sys.exit(f"[ERREUR] Video introuvable : {video}")

    return list(zip(videos, cameras))


def load_model_specs(specs_path: Path) -> list[tuple[str, str, str]]:
    """Load dynamic model specs written by campaign scripts."""
    if not specs_path.exists():
        sys.exit(f"[ERREUR] Model specs introuvable : {specs_path}")
    with open(specs_path, "r", encoding="utf-8") as f:
        raw_specs = json.load(f)
    specs: list[tuple[str, str, str]] = []
    for raw in raw_specs:
        name = str(raw["name"])
        subdir = str(raw.get("subdir", name))
        model_type = str(raw.get("type", "rtdetr" if name.startswith("rtdetr") else "yolo"))
        specs.append((name, subdir, model_type))
    if not specs:
        sys.exit(f"[ERREUR] Aucun modele dans {specs_path}")
    return specs


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENTS CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluation TRD multi-modeles -- Detection d'intrusion par les pieds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python evaluate_trd.py --video cam3.mkv --camera cam_03
  python evaluate_trd.py --video cam7.mkv --camera cam_07 --models yolo11n yolo11s
  python evaluate_trd.py --video cam5.mkv --camera cam_05 --format onnx
  python evaluate_trd.py --video cam3.mkv --camera cam_03 --person-class 0  # COCO
    python evaluate_trd.py --videos v1.mp4 v2.mp4 --cameras cam_03 cam_05 --formats pt engine
""")

    p.add_argument("--video", "-v", required=False,
                   help="Chemin vers la video de test (mode simple)")
    p.add_argument("--camera", "-c", required=False,
                   help="ID camera (ex: cam_03, cam_07) (mode simple)")
    p.add_argument("--videos", nargs="+", default=None,
                   help="Liste de videos a enchainer (mode batch)")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="Liste des cameras associees a --videos (meme taille)")
    p.add_argument("--gt", default=str(GT_PATH),
                   help=f"Chemin vers le fichier GT (defaut: {GT_PATH.name})")
    p.add_argument("--config", default=str(CONFIG_PATH),
                   help=f"Chemin vers le config.yaml avec les zones pixel (defaut: {CONFIG_PATH})")
    p.add_argument("--zones-json", default=str(ZONES_JSON_PATH),
                   help=f"Chemin vers zones_config.json (prioritaire) (defaut: {ZONES_JSON_PATH})")
    p.add_argument("--runs-dir", default=str(RUNS_DIR),
                   help=f"Dossier racine des poids entraines (defaut: {RUNS_DIR})")
    p.add_argument("--conf", type=float, default=CONFIDENCE,
                   help=f"Seuil de confiance (defaut: {CONFIDENCE})")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                   help="Device d'inference (defaut: auto)")
    p.add_argument("--skip", "-s", type=int, default=FRAME_SKIP,
                   help=f"Analyser 1 frame sur N (defaut: {FRAME_SKIP}, 1=toutes)")
    p.add_argument("--format", "-f", choices=["pt", "onnx", "engine"], default="pt",
                   help="Format des poids : pt, onnx ou engine (defaut: pt)")
    p.add_argument("--formats", nargs="+", choices=["pt", "onnx", "engine"], default=None,
                   help="Formats a enchainer (ex: --formats pt engine)")
    p.add_argument("--suffix", default="",
                   help="Suffixe du fichier de poids (ex: --suffix pruned -> best_pruned.onnx)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Sous-ensemble de modeles (ex: --models yolo11n rtdetr-l)")
    p.add_argument("--model-specs", default=None,
                   help="JSON de specs dynamiques: [{name, subdir, type}]")
    p.add_argument("--person-class", type=int, default=PERSON_CLASS_ID,
                   help=f"ID de la classe Personne (defaut: {PERSON_CLASS_ID}, COCO=0)")
    p.add_argument("--tolerance", type=float, default=TOLERANCE_SEC,
                   help=f"Fenetre de tolerance pour le matching en sec (defaut: {TOLERANCE_SEC})")
    p.add_argument("--output", "-o", default=None,
                   help="Override du chemin de sortie JSON (defaut: Modelstrained/TRD/)")
    p.add_argument("--view", action="store_true",
                   help="Affiche la video + point pieds en direct (debug visuel)")
    p.add_argument("--min-persistence", type=int, default=MIN_PERSISTENCE,
                   help=f"Frames consecutives pour confirmer une alerte (defaut: {MIN_PERSISTENCE})")
    p.add_argument("--zone-base-width", type=int, default=ZONE_BASE_W,
                   help=f"Largeur de reference des points zone (defaut: {ZONE_BASE_W})")
    p.add_argument("--zone-base-height", type=int, default=ZONE_BASE_H,
                   help=f"Hauteur de reference des points zone (defaut: {ZONE_BASE_H})")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.skip < 1:
        sys.exit(f"[ERREUR] --skip doit etre >= 1 (recu: {args.skip})")

    gt_path     = Path(args.gt)
    config_path = Path(args.config)
    runs_dir    = Path(args.runs_dir)
    jobs        = resolve_jobs(args)
    formats     = args.formats if args.formats else [args.format]

    # --- Charger config ---
    config = load_config(config_path)

    # --- Device ---
    device, device_label = detect_device(args.device)

    # --- Filtrer les modeles ---
    models = load_model_specs(Path(args.model_specs)) if args.model_specs else MODELS
    if args.models:
        selected = set(args.models)
        models = [m for m in models if m[0] in selected]
        unknown = selected - {m[0] for m in models}
        if unknown:
            print(f"  [ATTENTION] Modeles inconnus ignores : {unknown}")
        if not models:
            sys.exit("[ERREUR] Aucun modele valide selectionne.")

    # --- En-tete global ---
    print(f"\n  Jobs     : {len(jobs)}")
    print(f"  Formats  : {formats}")
    print(f"  GT       : {gt_path.name}")
    print(f"  Runs     : {runs_dir}")
    print(f"  Conf     : {args.conf}")
    print(f"  Modeles  : {[m[0] for m in models]}")
    if device != "cpu":
        print(f"  GPU      : {device_label}")
    else:
        print(f"  Device   : {device_label}")

    # --- Boucle batch : jobs x formats x modeles ---
    total_t0 = time.time()
    run_index = 0
    total_runs = len(jobs) * len(formats)

    for video_path, camera_id in jobs:
        gt_events = load_gt(gt_path, camera_id)
        zone_base_w = args.zone_base_width
        zone_base_h = args.zone_base_height

        zone_polygon = load_zone_from_zones_json(Path(args.zones_json), camera_id)
        if zone_polygon is not None:
            zone_source = f"{Path(args.zones_json).name} ({camera_id})"
        else:
            zone_polygon = load_zone_from_config(config, camera_id)
            zone_source = f"{Path(args.config).name} ({camera_id})"
            projected_base = phase3_projection_base_resolution(config, base_camera_id(camera_id))
            if projected_base is not None:
                zone_base_w, zone_base_h = projected_base

        for fmt in formats:
            run_index += 1
            all_results: list[dict] = []

            print(f"\n\n{'#' * 80}")
            print(f"  RUN {run_index}/{total_runs} -- Camera: {camera_id} -- Video: {video_path.name} -- Format: {fmt}")
            print(f"{'#' * 80}")
            print(f"  Zone source : {zone_source}")
            print(f"  GT events   : {len(gt_events)}")

            for idx, (name, subdir, mtype) in enumerate(models, 1):
                print(f"\n{'=' * 60}")
                print(f"  [{idx}/{len(models)}] Evaluation de {name}...")
                print(f"{'=' * 60}")

                suffix_part  = f"_{args.suffix}" if args.suffix else ""
                weights_file = f"best{suffix_part}.{fmt}"
                weights_path = runs_dir / subdir / "weights" / weights_file

                result = evaluate_model(
                    model_name=name,
                    weights_path=weights_path,
                    model_type=mtype,
                    video_path=str(video_path),
                    gt_events=gt_events,
                    zone_polygon=zone_polygon,
                    person_class=args.person_class,
                    conf=args.conf,
                    device=device,
                    tolerance_sec=args.tolerance,
                    frame_skip=args.skip,
                    view=args.view,
                    min_persistence=args.min_persistence,
                    zone_base_w=zone_base_w,
                    zone_base_h=zone_base_h,
                )

                if result is not None:
                    all_results.append(result)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if all_results:
                print_summary(all_results, camera_id)

                output_data = {
                    "camera":        camera_id,
                    "video":         str(video_path),
                    "confidence":    args.conf,
                    "person_class":  args.person_class,
                    "tolerance_sec": args.tolerance,
                    "frame_skip":    args.skip,
                    "format":        fmt,
                    "models":        [],
                }
                for r in all_results:
                    r_copy = {k: v for k, v in r.items() if k != "details"}
                    output_data["models"].append(r_copy)

                if args.output:
                    output_path = Path(args.output)
                    if not output_path.suffix:
                        video_tag = video_path.stem
                        output_path = output_path / f"trd_{camera_id}_{video_tag}_{fmt}.json"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    video_tag = video_path.stem
                    output_path = OUTPUT_DIR / f"trd_{camera_id}_{video_tag}_{fmt}.json"

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"  Resultats sauvegardes : {output_path}")

                csv_path = output_path.with_suffix(".csv")
                write_results_csv(csv_path, all_results)
                print(f"  CSV metriques sauvegarde : {csv_path}")

                for r in all_results:
                    detail_dir = OUTPUT_DIR / r["model"]
                    detail_dir.mkdir(parents=True, exist_ok=True)
                    video_tag = video_path.stem
                    detail_path = detail_dir / f"trd_details_{camera_id}_{video_tag}_{fmt}.json"
                    detail_data = {
                        "model":      r["model"],
                        "camera":     camera_id,
                        "video":      str(video_path),
                        "confidence": args.conf,
                        "metriques": {
                            "fps_inference":   r["fps_inference"],
                            "trd_median":      r["trd_median"],
                            "trd_p95":         r["trd_p95"],
                            "trd_mean":        r["trd_mean"],
                            "trd_min":         r["trd_min"],
                            "trd_max":         r["trd_max"],
                            "n_matched":       r["n_matched"],
                            "n_missed":        r["n_missed"],
                            "n_faux_positifs": r["n_faux_positifs"],
                            "fp_raw_detections": r["fp_raw_detections"],
                            "precision":       r["precision"],
                            "recall":          r["recall"],
                            "f1":              r["f1"],
                            "far_per_hour":    r["far_per_hour"],
                            "latency_mean_ms": r["latency_mean_ms"],
                            "latency_median_ms": r["latency_median_ms"],
                            "latency_p95_ms":  r["latency_p95_ms"],
                            "latency_max_ms":  r["latency_max_ms"],
                            "latency_jitter_ms": r["latency_jitter_ms"],
                        },
                        "details": r.get("details", []),
                        "fp_clusters": r.get("fp_clusters", []),
                    }
                    with open(detail_path, "w", encoding="utf-8") as f:
                        json.dump(detail_data, f, indent=2, ensure_ascii=False)
                    print(f"  Details {r['model']} : {detail_path}")
            else:
                print("\n  [!!] Aucun modele n'a pu etre evalue pour ce run.")

    total_elapsed = time.time() - total_t0
    print(f"\n  Duree totale batch : {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()

"""
Phase 2 - Evaluation du TAD (Temps d'Alerte a la Detection)

Pour chaque modele entraine, lit une video trame par trame, compare les
detections (conf > seuil) aux apparitions annotees dans gt_objects_tad.json,
et calcule :
  - TAD median (s) et 95e percentile
  - FPS moyen d'inference
  - Nombre d'objets manques

Usage :
  python evaluate_tad.py --video path/to/video.mp4 --camera cam_07_record2
  python evaluate_tad.py --video video.mp4 --camera cam_05_rec3.1 --conf 0.4
  python evaluate_tad.py --video video.mp4 --camera cam_07_rec3.1 --models yolov8n yolo11s
  python evaluate_tad.py --video video.mp4 --camera cam_07_rec3.1 --format onnx
"""

import argparse
import csv
import json
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "Phase_2_Baseline_MonoCam" / "Modelstrained" / "V2"
GT_PATH = next(
    (
        path
        for path in (
            PROJECT_ROOT / "dataset_objets_HD" / "gt_objects_tad.json",
            PROJECT_ROOT / "ground_truth" / "gt_objects_tad_dataset_objets_HD.json",
            PROJECT_ROOT / "ground_truth" / "gt_objects_tad.json",
        )
        if path.exists()
    ),
    PROJECT_ROOT / "dataset_objets_HD" / "gt_objects_tad.json",
)
OUTPUT_DIR = SCRIPT_DIR / "Modelstrained" / "TAD"

# Modeles a evaluer : (nom court, sous-dossier runs, type)
MODELS = [
    ("yolov8n", "yolov8n", "yolo"),
    ("yolov8s", "yolov8s", "yolo"),
    ("yolo26n", "yolo26n", "yolo"),
    ("yolo11n", "yolo11n", "yolo"),
    ("yolo11s", "yolo11s", "yolo"),
    ("rtdetr-l", "rtdetr-l", "rtdetr"),
]

CONFIDENCE = 0.5
IMGSZ = 640
FRAME_SKIP = 5  # 1 = toutes les frames, 5 = 1 frame sur 5 analysee
GT_MATCH_WINDOW_S = 10.0
FP_MERGE_WINDOW_S = 3.0
TAD_IGNORED_CLASSES = {"personne", "person", "persons", "people"}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def strip_accents(s: str) -> str:
    """Supprime les accents pour comparer 'niveau à bulle' == 'niveau a bulle'."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_class(name: str) -> str:
    """Normalise un nom de classe pour comparaison."""
    return strip_accents(name).lower().strip()


def is_tad_class_ignored(class_name: str) -> bool:
    """Ignore les classes non objet pour l'evaluation TAD."""
    return normalize_class(class_name) in TAD_IGNORED_CLASSES


def detect_device(device_mode: str = "auto") -> str:
    """Choisit le device d'inference.

    device_mode:
      - auto : GPU si dispo, sinon CPU
      - gpu  : force GPU (fallback CPU si indisponible)
      - cpu  : force CPU
    """
    mode = device_mode.lower().strip()

    if mode == "cpu":
        print("  Device force : CPU")
        return "cpu"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  GPU  : {name}")
        return "0"

    if mode == "gpu":
        print("  [ATTENTION] GPU force mais CUDA indisponible - fallback CPU.")
    else:
        print("  [!!] Pas de GPU CUDA - inference sur CPU.")
    return "cpu"


def load_model(weights: str, model_type: str):
    """Instancie YOLO ou RTDETR selon le type."""
    if model_type == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(weights)
    else:
        from ultralytics import YOLO
        return YOLO(weights)


def format_duration(seconds: float) -> str:
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
        "class_name": alerts[0].get("class_name", "unknown"),
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


def is_tad_alert_inside_gt_window(alert: dict, gt_events: list[dict],
                                  vid_fps: float) -> bool:
    """Vrai si une alerte objet tombe dans une fenetre temporelle GT compatible."""
    alert_time = alert["timestamp"]
    alert_class = alert.get("class_norm")
    for ev in gt_events:
        if alert_class != ev.get("classe_norm"):
            continue
        gt_time = ev["trame_apparition"] / vid_fps
        if abs(alert_time - gt_time) <= GT_MATCH_WINDOW_S:
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
        "model", "weights", "fps_moyen", "detected", "n_gt_events",
        "n_detected", "n_missed", "n_faux_positifs", "fp_raw_detections",
        "precision", "recall", "f1", "far_per_hour", "tad_median",
        "tad_p95", "tad_mean", "tad_min", "tad_max", "latency_mean_ms",
        "latency_median_ms", "latency_p95_ms", "latency_max_ms",
        "latency_jitter_ms", "frames_analysees", "total_frames",
        "frame_skip", "video_duration_s",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({key: r.get(key) for key in fieldnames})


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT GT
# ══════════════════════════════════════════════════════════════════════════════

def load_gt(gt_path: Path, camera_id: str) -> list[dict]:
    """Charge les evenements GT pour une camera donnee.

    Retourne une liste triee par trame_apparition, chaque element enrichi
    d'un champ 'classe_norm' pour la comparaison.
    """
    if not gt_path.exists():
        sys.exit(f"[ERREUR] GT introuvable : {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        all_events = json.load(f)

    events = []
    for e in all_events:
        if e["id_camera"] != camera_id:
            continue
        ev = dict(e)
        ev["classe_norm"] = normalize_class(e["classe_objet"])
        events.append(ev)

    events.sort(key=lambda x: x["trame_apparition"])
    return events


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
#  EVALUATION D'UN MODELE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name: str, weights_path: Path, model_type: str,
                   video_path: str, gt_events: list[dict],
                   conf: float, device: str,
                   frame_skip: int = 1) -> dict | None:
    """Evalue un modele sur la video et retourne les metriques TAD."""

    if not weights_path.exists():
        print(f"  [SKIP] Poids introuvables : {weights_path}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  MODELE : {model_name}")
    print(f"  Poids  : {weights_path}")
    print(f"{'=' * 60}")

    # Charger le modele
    model = load_model(str(weights_path), model_type)

    # Warmup
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model.predict(dummy, device=device, conf=conf, verbose=False)

    # Construire le mapping class_name normalisé -> class_id du modele
    model_names = model.names  # {0: "maillet", 1: "marteau", ...}
    norm_to_ids: dict[str, list[int]] = {}
    for cid, cname in model_names.items():
        norm = normalize_class(cname)
        norm_to_ids.setdefault(norm, []).append(cid)

    # Preparer le suivi des detections GT
    # Chaque evenement a un index unique; on traque la premiere detection
    pending: dict[int, dict] = {}
    for i, ev in enumerate(gt_events):
        pending[i] = {
            "event": ev,
            "trame_alerte": None,
            "detected": False,
        }

    # Determiner les class_ids a surveiller pour chaque evenement
    for idx, entry in pending.items():
        ev_norm = entry["event"]["classe_norm"]
        entry["target_class_ids"] = set(norm_to_ids.get(ev_norm, []))
        if not entry["target_class_ids"]:
            print(f"  [ATTENTION] Classe GT '{entry['event']['classe_objet']}' "
                  f"(norm: '{ev_norm}') non trouvee dans le modele")

    # Ouvrir la video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERREUR] Impossible d'ouvrir : {video_path}")
        return None

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_label = f"1/{frame_skip}" if frame_skip > 1 else "toutes"
    print(f"  Video  : {Path(video_path).name} ({total_frames} frames, {vid_fps:.1f} fps)")
    print(f"  GT     : {len(gt_events)} evenements a detecter")
    print(f"  Conf   : {conf}")
    print(f"  Skip   : {skip_label} frames analysees")

    # Boucle d'inference
    inference_times: list[float] = []
    system_alerts: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame skipping : on lit quand meme la frame (pour avancer le
        # compteur cv2) mais on ne lance l'inference que 1 frame sur N.
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter()
        results = model.predict(
            frame, conf=conf, device=device, imgsz=IMGSZ, verbose=False
        )
        dt = time.perf_counter() - t0
        inference_times.append(dt)

        # Extraire les classes detectees dans cette frame
        boxes = results[0].boxes
        frame_alert_indices_by_class: dict[int, list[int]] = {}
        if len(boxes) > 0:
            cls_values = [int(c) for c in boxes.cls.cpu().numpy()]
            conf_values = [float(c) for c in boxes.conf.cpu().numpy()]
            xyxy_values = boxes.xyxy.cpu().numpy()
            detected_cls = set(cls_values)

            for det_idx, (cid, det_conf, box) in enumerate(
                zip(cls_values, conf_values, xyxy_values)
            ):
                class_name = model_names.get(cid, str(cid))
                if is_tad_class_ignored(class_name):
                    continue

                x1, y1, x2, y2 = box
                alert = {
                    "timestamp": frame_idx / vid_fps,
                    "frame": frame_idx,
                    "class_id": cid,
                    "class_name": class_name,
                    "class_norm": normalize_class(class_name),
                    "confidence": det_conf,
                    "position": (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
                    "matched_gt": None,
                }
                system_alerts.append(alert)
                frame_alert_indices_by_class.setdefault(cid, []).append(
                    len(system_alerts) - 1
                )
        else:
            detected_cls = set()

        # Verifier chaque evenement GT non encore detecte
        for idx, entry in pending.items():
            if entry["detected"]:
                continue
            ev = entry["event"]
            # Ne compter que si on est apres (ou a) la trame d'apparition
            if frame_idx < ev["trame_apparition"]:
                continue
            # Verifier si la classe cible est detectee
            if entry["target_class_ids"] & detected_cls:
                entry["detected"] = True
                entry["trame_alerte"] = frame_idx
                matching_ids = entry["target_class_ids"] & detected_cls
                for cid in matching_ids:
                    if frame_alert_indices_by_class.get(cid):
                        alert_idx = frame_alert_indices_by_class[cid][0]
                        system_alerts[alert_idx]["matched_gt"] = ev["id_evenement"]
                        entry["matched_alert_idx"] = alert_idx
                        break

        frame_idx += 1

        # Progression (basee sur frame_idx reel, pas seulement les traitees)
        if frame_idx % 2000 == 0:
            pct = frame_idx / total_frames * 100
            n_det = sum(1 for e in pending.values() if e["detected"])
            print(f"    frame {frame_idx}/{total_frames} ({pct:.0f}%) "
                  f"- {n_det}/{len(pending)} detectes")

    cap.release()

    # Calculer les TAD
    tad_values: list[float] = []
    missed = 0
    details: list[dict] = []

    for idx in sorted(pending.keys()):
        entry = pending[idx]
        ev = entry["event"]
        detail = {
            "id_evenement": ev["id_evenement"],
            "classe": ev["classe_objet"],
            "trame_apparition": ev["trame_apparition"],
        }

        if entry["detected"]:
            delai_trames = entry["trame_alerte"] - ev["trame_apparition"]
            tad_sec = delai_trames / vid_fps
            tad_values.append(tad_sec)
            detail["trame_alerte"] = entry["trame_alerte"]
            detail["delai_trames"] = delai_trames
            detail["tad_sec"] = round(tad_sec, 3)
        else:
            missed += 1
            detail["trame_alerte"] = None
            detail["tad_sec"] = None

        details.append(detail)

    # Metriques
    fps_moyen = 1.0 / np.mean(inference_times) if inference_times else 0.0
    tad_arr = np.array(tad_values) if tad_values else np.array([])
    video_duration_s = (total_frames if total_frames > 0 else frame_idx) / vid_fps
    unmatched_alerts = [
        a for a in system_alerts
        if a["matched_gt"] is None
        and not is_tad_alert_inside_gt_window(a, gt_events, vid_fps)
    ]
    fp_clusters = cluster_false_positives(unmatched_alerts)
    extended_metrics = compute_extended_metrics(
        tp=len(tad_values),
        fn=missed,
        fp_clusters=fp_clusters,
        video_duration_s=video_duration_s,
    )
    latency_metrics = compute_latency_metrics(inference_times)

    frames_analysees = len(inference_times)

    result = {
        "model": model_name,
        "weights": str(weights_path),
        "total_frames": frame_idx,
        "frames_analysees": frames_analysees,
        "frame_skip": frame_skip,
        "fps_moyen": round(fps_moyen, 1),
        "n_gt_events": len(gt_events),
        "n_detected": len(tad_values),
        "n_missed": missed,
        "n_faux_positifs": extended_metrics["fp"],
        "fp_raw_detections": extended_metrics["fp_raw_detections"],
        "recall": extended_metrics["recall"],
        "precision": extended_metrics["precision"],
        "f1": extended_metrics["f1"],
        "far_per_hour": extended_metrics["far_per_hour"],
        "video_duration_s": extended_metrics["video_duration_s"],
        "detected": extended_metrics["detected"],
        "latency_mean_ms": latency_metrics["latency_mean_ms"],
        "latency_median_ms": latency_metrics["latency_median_ms"],
        "latency_p95_ms": latency_metrics["latency_p95_ms"],
        "latency_max_ms": latency_metrics["latency_max_ms"],
        "latency_jitter_ms": latency_metrics["latency_jitter_ms"],
        "tad_median": round(float(np.median(tad_arr)), 3) if len(tad_arr) > 0 else None,
        "tad_p95": round(float(np.percentile(tad_arr, 95)), 3) if len(tad_arr) > 0 else None,
        "tad_mean": round(float(np.mean(tad_arr)), 3) if len(tad_arr) > 0 else None,
        "tad_min": round(float(np.min(tad_arr)), 3) if len(tad_arr) > 0 else None,
        "tad_max": round(float(np.max(tad_arr)), 3) if len(tad_arr) > 0 else None,
        "details": details,
        "fp_clusters": fp_clusters,
    }

    # Affichage par evenement
    print(f"\n  {'Evt':>4s}  {'Classe':<18s}  {'Apparition':>10s}  "
          f"{'Alerte':>10s}  {'Delai':>8s}  {'TAD (s)':>8s}")
    print(f"  {'-' * 4}  {'-' * 18}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

    for d in details:
        cls_str = d["classe"][:18]
        app = str(d["trame_apparition"])
        if d["trame_alerte"] is not None:
            al = str(d["trame_alerte"])
            dl = str(d["delai_trames"])
            tad = f"{d['tad_sec']:.3f}"
        else:
            al = "MANQUE"
            dl = "-"
            tad = "-"
        print(f"  {d['id_evenement']:>4d}  {cls_str:<18s}  {app:>10s}  "
              f"{al:>10s}  {dl:>8s}  {tad:>8s}")

    print(f"\n  Resultats {model_name} :")
    print(f"    Frames       : {frames_analysees}/{frame_idx} analysees (skip={frame_skip})")
    print(f"    FPS moyen    : {fps_moyen:.1f}  (inference uniquement)")
    if tad_arr.size > 0:
        print(f"    TAD median   : {result['tad_median']:.3f}s")
        print(f"    TAD 95e pct  : {result['tad_p95']:.3f}s")
        print(f"    TAD moyen    : {result['tad_mean']:.3f}s")
        print(f"    TAD min/max  : {result['tad_min']:.3f}s / {result['tad_max']:.3f}s")
    print(f"    Detectes     : {result['n_detected']}/{result['n_gt_events']}")
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
#  TABLEAU RECAPITULATIF
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], camera_id: str):
    """Affiche un tableau comparatif de tous les modeles."""
    print(f"\n\n{'=' * 112}")
    print(f"  RECAPITULATIF TAD - Camera : {camera_id}")
    print(f"{'=' * 112}")
    print(f"  {'Modele':<12s}  {'FPS':>6s}  {'Detect':>7s}  "
          f"{'Precision':>9s}  {'Recall':>7s}  {'F1':>7s}  {'FAR(/h)':>8s}  "
          f"{'FP':>4s}  {'TAD Med':>8s}  {'TAD 95%':>8s}  {'TAD Moy':>8s}")
    print(f"  {'-' * 12}  {'-' * 6}  {'-' * 7}  {'-' * 9}  {'-' * 7}  "
          f"{'-' * 7}  {'-' * 8}  {'-' * 4}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for r in results:
        fps = f"{r['fps_moyen']:.1f}"
        med = f"{r['tad_median']:.3f}" if r["tad_median"] is not None else "-"
        p95 = f"{r['tad_p95']:.3f}" if r["tad_p95"] is not None else "-"
        moy = f"{r['tad_mean']:.3f}" if r["tad_mean"] is not None else "-"
        det = r.get("detected", f"{r['n_detected']}/{r['n_gt_events']}")
        precision = f"{r['precision']:.4f}"
        recall = f"{r['recall']:.4f}"
        f1 = f"{r['f1']:.4f}"
        far = f"{r['far_per_hour']:.2f}"
        fp = str(r["n_faux_positifs"])
        print(f"  {r['model']:<12s}  {fps:>6s}  {det:>7s}  "
              f"{precision:>9s}  {recall:>7s}  {f1:>7s}  {far:>8s}  "
              f"{fp:>4s}  {med:>8s}  {p95:>8s}  {moy:>8s}")

    print(f"{'=' * 112}")

    # Meilleur modele par TAD median
    valid = [r for r in results if r["tad_median"] is not None]
    if valid:
        best = min(valid, key=lambda r: r["tad_median"])
        print(f"\n  Meilleur TAD median : {best['model']} ({best['tad_median']:.3f}s)")
        fastest = max(valid, key=lambda r: r["fps_moyen"])
        print(f"  Plus rapide (FPS)   : {fastest['model']} ({fastest['fps_moyen']:.1f} fps)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluation TAD multi-modeles pour la detection d'objets.")
    p.add_argument("--video", "-v", required=True,
                   help="Chemin vers la video de test")
    p.add_argument("--camera", "-c", required=True,
                   help="ID camera dans le GT (ex: cam_07_record2, cam_05_rec3.1)")
    p.add_argument("--gt", default=str(GT_PATH),
                   help=f"Chemin vers gt_objects_tad.json (defaut: {GT_PATH})")
    p.add_argument("--runs-dir", default=str(RUNS_DIR),
                   help=f"Dossier racine des modeles entraines (defaut: {RUNS_DIR})")
    p.add_argument("--conf", type=float, default=CONFIDENCE,
                   help=f"Seuil de confiance (defaut: {CONFIDENCE})")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                   help="Choix du device d'inference (defaut: auto)")
    p.add_argument("--skip", "-s", type=int, default=FRAME_SKIP,
                   help=f"Analyser 1 frame sur N (defaut: {FRAME_SKIP}, 1=toutes)")
    p.add_argument("--format", "-f", choices=["pt", "onnx", "engine"], default="pt",
                   help="Format des poids a utiliser : pt, onnx ou engine (defaut: pt)")
    p.add_argument("--suffix", default="",
                   help="Suffixe du fichier de poids (ex: --suffix pruned30 -> best_pruned30.onnx)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Sous-ensemble de modeles (ex: --models yolov8n rtdetr-l)")
    p.add_argument("--model-specs", default=None,
                   help="JSON de specs dynamiques: [{name, subdir, type}]")
    p.add_argument("--output", "-o", default=None,
                   help="Sauvegarder les resultats en JSON")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"[ERREUR] Video introuvable : {video_path}")

    gt_path = Path(args.gt)
    runs_dir = Path(args.runs_dir)

    # Charger GT pour cette camera
    gt_events = load_gt(gt_path, args.camera)
    if not gt_events:
        print(f"[ERREUR] Aucun evenement GT pour la camera '{args.camera}'")
        print(f"         Cameras disponibles dans {gt_path.name} :")
        with open(gt_path, "r", encoding="utf-8") as f:
            all_ev = json.load(f)
        cams = sorted(set(e["id_camera"] for e in all_ev))
        for c in cams:
            n = sum(1 for e in all_ev if e["id_camera"] == c)
            print(f"           {c} ({n} evenements)")
        sys.exit(1)

    # Filtrer les modeles
    models = load_model_specs(Path(args.model_specs)) if args.model_specs else MODELS
    if args.models:
        selected = set(args.models)
        models = [m for m in models if m[0] in selected]
        unknown = selected - {m[0] for m in models}
        if unknown:
            print(f"[ATTENTION] Modeles inconnus ignores : {unknown}")
        if not models:
            sys.exit("[ERREUR] Aucun modele valide selectionne.")

    print("=" * 60)
    print("  EVALUATION TAD - Detection d'Objets Interdits")
    print("=" * 60)
    print(f"  Video    : {video_path}")
    print(f"  Camera   : {args.camera}")
    print(f"  GT       : {gt_path.name} ({len(gt_events)} evenements)")
    print(f"  Runs     : {runs_dir}")
    print(f"  Format   : {args.format}")
    print(f"  Conf     : {args.conf}")
    print(f"  Device   : {args.device}")
    skip_label = f"1/{args.skip}" if args.skip > 1 else "toutes"
    print(f"  Skip     : {skip_label} frames")
    print(f"  Modeles  : {[m[0] for m in models]}")

    device = detect_device(args.device)
    print("=" * 60)

    results: list[dict] = []
    total_t0 = time.time()

    for i, (name, subdir, mtype) in enumerate(models, 1):
        print(f"\n{'-' * 60}")
        print(f"  [{i}/{len(models)}] Evaluation de {name}...")
        print(f"{'-' * 60}")

        suffix_part = f"_{args.suffix}" if args.suffix else ""
        weights_file = f"best{suffix_part}.{args.format}"
        weights_path = runs_dir / subdir / "weights" / weights_file

        result = evaluate_model(
            model_name=name,
            weights_path=weights_path,
            model_type=mtype,
            video_path=str(video_path),
            gt_events=gt_events,
            conf=args.conf,
            device=device,
            frame_skip=args.skip,
        )

        if result is not None:
            results.append(result)

        # Liberer la VRAM entre les modeles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_t0

    if results:
        print_summary(results, args.camera)
        print(f"\n  Duree totale : {format_duration(total_elapsed)}")

        video_tag = video_path.stem
        if args.output:
            base_output_path = Path(args.output)
            if base_output_path.suffix:
                csv_path = base_output_path.with_suffix(".csv")
                fp_dir = base_output_path.with_suffix("").parent / f"{base_output_path.stem}_fp_details"
            else:
                csv_path = base_output_path / f"tad_{args.camera}_{video_tag}_{args.format}.csv"
                fp_dir = base_output_path / f"tad_fp_details_{args.camera}_{video_tag}_{args.format}"
        else:
            csv_path = OUTPUT_DIR / f"tad_{args.camera}_{video_tag}_{args.format}.csv"
            fp_dir = OUTPUT_DIR / f"tad_fp_details_{args.camera}_{video_tag}_{args.format}"

        write_results_csv(csv_path, results)
        print(f"  CSV metriques sauvegarde : {csv_path}")

        fp_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            fp_path = fp_dir / f"{r['model']}_fp_clusters.json"
            with open(fp_path, "w", encoding="utf-8") as f:
                json.dump(r.get("fp_clusters", []), f, indent=2, ensure_ascii=False)
            print(f"  Details FP {r['model']} : {fp_path}")

        # Sauvegarde JSON optionnelle
        if args.output:
            output_path = Path(args.output)
            if not output_path.suffix:
                output_path = output_path / f"tad_{args.camera}_{video_tag}_{args.format}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_data = {
                "camera": args.camera,
                "video": str(video_path),
                "confidence": args.conf,
                "models": [
                    {k: v for k, v in r.items() if k != "details"}
                    for r in results
                ],
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"  Resultats sauvegardes : {output_path}")
    else:
        print("\n  [!!] Aucun modele n'a pu etre evalue.")


if __name__ == "__main__":
    main()

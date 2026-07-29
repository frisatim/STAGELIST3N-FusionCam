"""
visual_check_fusion.py : harnais de test interactif de la fusion multi-caméras.

Charge les instances CameraTracker et MultiCameraFusion, lit une frame par
caméra à chaque itération, exécute ByteTrack + la fusion, journalise les
associations inter-caméras dans la console et affiche une vue annotée
côte à côte.

Usage :
    python visual_check_fusion.py --cameras cam_03,cam_05,cam_07 --model ../yolov8n.pt
    python visual_check_fusion.py --cameras cam_03,cam_07 --model ../yolov8n.pt --max-frames 300
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from detection import Detection
from fusion import MultiCameraFusion
from geometry_fix import fix_frame, load_aspect_fix
from tracker import CameraTracker


# ----------------------------------------------------------------------
# Utilitaires couleur
# ----------------------------------------------------------------------

def _global_id_to_color(gid: int) -> tuple[int, int, int]:
    """Couleur BGR stable dérivée de l'identifiant global (hachage MD5).

    Les composantes sont bornées à 80 minimum pour rester lisibles à l'écran.
    """
    h = int(hashlib.md5(str(gid).encode()).hexdigest()[:6], 16)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = (h & 0x0000FF)
    r = max(80, r)
    g = max(80, g)
    b = max(80, b)
    return (b, g, r)


# ----------------------------------------------------------------------
# Annotation
# ----------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    cam_id: str,
) -> np.ndarray:
    """Annote la frame avec les détections de la caméra : boîte englobante,
    étiquette (id global, id piste, classe, confiance, position en mètres)
    et point pied. Renvoie une copie annotée de la frame."""
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1

    for det in detections:
        if det.cam_id != cam_id:
            continue

        gid = det.global_id if det.global_id is not None else 0
        color = _global_id_to_color(gid)

        x1, y1, x2, y2 = [int(v) for v in det.bbox_px]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"G{gid}|#{det.track_id} {det.class_name} {det.confidence:.2f}"
        if det.foot_point_m is not None:
            xm, ym = det.foot_point_m
            label += f" ({xm:.2f}m,{ym:.2f}m)"

        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(vis, (x1, top), (x1 + tw + 2, y1), color, -1)
        cv2.putText(vis, label, (x1 + 1, y1 - baseline - 1), font, scale, (0, 0, 0), thick)

        fu, fv = int(det.foot_point_px[0]), int(det.foot_point_px[1])
        cv2.circle(vis, (fu, fv), 4, color, -1)

    cv2.putText(vis, cam_id, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(vis, cam_id, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)

    return vis


# ----------------------------------------------------------------------
# Boucle principale
# ----------------------------------------------------------------------

def run(
    cam_ids: list[str],
    model_path: str,
    config: dict,
    max_frames: int | None,
    show_window: bool,
    fusion_distance_threshold_m: float,
    fusion_time_window_ms: float,
    device: str | None,
) -> None:
    """Boucle principale du harnais : ouvre les vidéos des caméras, exécute le
    tracking par caméra puis la fusion, journalise les associations
    inter-caméras et affiche la mosaïque annotée.

    S'arrête après max_frames, à l'épuisement d'une vidéo ou sur la touche 'q',
    puis imprime un résumé (frames traitées, ids globaux uniques, associations).
    """
    script_dir = Path(__file__).parent
    base_dir = script_dir  # video_path dans config est déjà relatif à Phase_3/

    ar_fix = load_aspect_fix(config)
    cameras_cfg = config.get("cameras", {})

    trackers: dict[str, CameraTracker] = {}
    for cam_id in cam_ids:
        trackers[cam_id] = CameraTracker(
            cam_id=cam_id,
            model_path=model_path,
            config=config,
            conf_threshold=0.5,
            target_classes=[0],
            device=device,
        )

    fusion = MultiCameraFusion(
        config,
        distance_threshold_m=fusion_distance_threshold_m,
        time_window_s=fusion_time_window_ms / 1000.0,
    )

    caps: dict[str, cv2.VideoCapture] = {}
    for cam_id in cam_ids:
        cam_cfg = cameras_cfg.get(cam_id)
        if cam_cfg is None:
            print(f"[ERR] Camera {cam_id} not found in config.yaml, arret.")
            sys.exit(1)
        video_path_raw: str = cam_cfg.get("video_path", "")
        if not video_path_raw:
            print(f"[ERR] No video_path for {cam_id} in config.yaml, arret.")
            sys.exit(1)
        resolved = (base_dir / video_path_raw).resolve()
        if not resolved.exists():
            print(f"[WARN] Video file not found for {cam_id}: {resolved}")
        cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            print(f"[ERR] Cannot open video for {cam_id}: {resolved}")
            sys.exit(1)
        caps[cam_id] = cap
        print(f"[INFO] Opened video for {cam_id}: {resolved}")

    frame_count: int = 0
    total_global_ids: set[int] = set()
    total_cross_associations: int = 0

    while True:
        if max_frames is not None and frame_count >= max_frames:
            print(f"[INFO] Reached max_frames={max_frames}, stopping.")
            break

        raw_frames: dict[str, np.ndarray] = {}
        display_frames: dict[str, np.ndarray] = {}
        all_done = False

        for cam_id in cam_ids:
            ret, frame = caps[cam_id].read()
            if not ret:
                all_done = True
                break
            raw_frames[cam_id] = frame
            display_frames[cam_id] = fix_frame(frame, cam_id, ar_fix)

        if all_done:
            print("[INFO] One or more cameras terminees, arret.")
            break

        timestamp = frame_count / 25.0

        detections_by_cam: dict[str, list[Detection]] = {}
        for cam_id in cam_ids:
            dets = trackers[cam_id].process_frame(raw_frames[cam_id], timestamp)
            detections_by_cam[cam_id] = dets

        all_dets: list[Detection] = fusion.associate(detections_by_cam)

        by_gid: dict[int, list[Detection]] = defaultdict(list)
        for det in all_dets:
            if det.global_id is not None:
                by_gid[det.global_id].append(det)
                total_global_ids.add(det.global_id)

        for gid, dets in by_gid.items():
            cams_seen = {d.cam_id for d in dets}
            if len(cams_seen) >= 2:
                sorted_dets = sorted(dets, key=lambda d: d.cam_id)
                for i in range(len(sorted_dets)):
                    for j in range(i + 1, len(sorted_dets)):
                        da, db = sorted_dets[i], sorted_dets[j]
                        if da.cam_id == db.cam_id:
                            continue
                        dist_str = "N/A"
                        if da.foot_point_m is not None and db.foot_point_m is not None:
                            dx = da.foot_point_m[0] - db.foot_point_m[0]
                            dy = da.foot_point_m[1] - db.foot_point_m[1]
                            dist_val = math.sqrt(dx * dx + dy * dy)
                            dist_str = f"{dist_val:.2f}m"
                        pos_a = (
                            f"({da.foot_point_m[0]:.2f}m,{da.foot_point_m[1]:.2f}m)"
                            if da.foot_point_m is not None
                            else "pos=N/A"
                        )
                        pos_b = (
                            f"({db.foot_point_m[0]:.2f}m,{db.foot_point_m[1]:.2f}m)"
                            if db.foot_point_m is not None
                            else "pos=N/A"
                        )
                        print(
                            f"[INFO] Frame {frame_count} | G{gid} linked: "
                            f"{da.cam_id}:tid={da.track_id} {pos_a} + "
                            f"{db.cam_id}:tid={db.track_id} {pos_b} | dist={dist_str}"
                        )
                        total_cross_associations += 1
            else:
                for det in dets:
                    print(
                        f"[DEBUG] Frame {frame_count} | G{gid} solo: "
                        f"{det.cam_id}:tid={det.track_id}"
                    )

        if show_window:
            cells: list[np.ndarray] = []
            for cam_id in cam_ids:
                annotated = _annotate_frame(
                    display_frames[cam_id], all_dets, cam_id
                )
                cells.append(cv2.resize(annotated, (480, 360)))
            combined = np.hstack(cells)
            cv2.imshow("Fusion Test", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] touche q, arret.")
                break

        frame_count += 1

    for cap in caps.values():
        cap.release()
    if show_window:
        cv2.destroyAllWindows()

    print(
        f"\n[INFO] Summary: "
        f"frames={frame_count} | "
        f"unique_global_ids={len(total_global_ids)} | "
        f"cross_camera_associations={total_cross_associations}"
    )


# ----------------------------------------------------------------------
# Point d'entrée
# ----------------------------------------------------------------------

def main() -> None:
    """Analyse les arguments, charge config.yaml puis lance la boucle de test."""
    parser = argparse.ArgumentParser(
        description="Harnais de test de la fusion multi-caméras."
    )
    parser.add_argument(
        "--cameras",
        required=True,
        help="Identifiants caméras séparés par des virgules, ex. cam_03,cam_05,cam_07",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Chemin du modèle YOLO .pt, ex. ../yolov8n.pt",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="S'arrêter après N frames (défaut : jusqu'à épuisement ou touche 'q')",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Désactiver l'affichage cv2.imshow (mode sans fenêtre)",
    )
    parser.add_argument(
        "--fusion-distance-m",
        type=float,
        default=1.0,
        help="Seuil de distance de fusion D en mètres.",
    )
    parser.add_argument(
        "--fusion-time-window-ms",
        type=float,
        default=500.0,
        help="Fenêtre temporelle de fusion en millisecondes.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Périphérique YOLO, ex. cuda:0 ou cpu. Défaut : detection.device de la config.",
    )
    args = parser.parse_args()

    cam_ids = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cam_ids:
        print("[ERR] No cameras specified.")
        sys.exit(1)

    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    if not config_path.exists():
        print(f"[ERR] config.yaml not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    model_path = str((script_dir / args.model).resolve())

    run(
        cam_ids=cam_ids,
        model_path=model_path,
        config=config,
        max_frames=args.max_frames,
        show_window=not args.no_window,
        fusion_distance_threshold_m=args.fusion_distance_m,
        fusion_time_window_ms=args.fusion_time_window_ms,
        device=args.device,
    )


if __name__ == "__main__":
    main()

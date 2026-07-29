"""
Script de test du suivi mono-caméra avec ByteTrack (Étape 3.2).

Ce script permet de :
  1. Lancer le tracker sur une vidéo enregistrée d'une caméra calibrée
  2. Visualiser les bboxes + track_ids + positions en mètres sur la vidéo
  3. Dessiner les trajectoires (trail) sur le plan de sol en temps réel
  4. Sauvegarder la vidéo annotée si --save est spécifié

Usage :
  python visual_check_tracker.py --cam cam_03 --model ../yolov8n.pt
  python visual_check_tracker.py --cam cam_03 --model ../yolov8n.pt --save --max-frames 500

Critère de validation (à vérifier visuellement) :
  - Une même personne garde le même track_id (même couleur) tout au long de la vidéo
  - Les positions en mètres affichées sont cohérentes avec la réalité (±20 cm)
  - Le plan de sol montre des trajectoires lisses (pas de sauts)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ajouter le dossier courant au path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent))

from tracker import CameraTracker, _id_to_color


# ----------------------------------------------------------------------
# Chargement du plan de sol
# ----------------------------------------------------------------------

def load_floor_plan(config: dict) -> np.ndarray | None:
    """Charge l'image du plan de sol pour la visualisation des trajectoires."""
    plan_path = Path(__file__).parent / config.get("geometrie_2d", {}).get("plan_image", "plan_rdc.png")
    if not plan_path.exists():
        print(f"[WARN] Plan de sol introuvable : {plan_path}")
        return None
    plan = cv2.imread(str(plan_path))
    if plan is None:
        print(f"[WARN] Impossible de lire le plan de sol : {plan_path}")
    return plan


def metres_to_plan_px(
    x_m: float, y_m: float, config: dict
) -> tuple[int, int]:
    """Convertit des coordonnées sol en mètres en pixels sur l'image du plan de sol."""
    geo = config.get("geometrie_2d", {})
    scale = geo.get("echelle_px_par_metre", 42.3)
    ox = geo.get("origine_pixel_x", 82)
    oy = geo.get("origine_pixel_y", 45)
    px = int(ox + x_m * scale)
    py = int(oy + y_m * scale)
    return (px, py)


# ----------------------------------------------------------------------
# Tracé des zones interdites sur le plan de sol
# ----------------------------------------------------------------------

def draw_forbidden_zones(plan: np.ndarray, config: dict) -> np.ndarray:
    """Dessine les zones interdites (définies en mètres) sur l'image du plan de sol."""
    plan_vis = plan.copy()
    geo = config.get("geometrie_2d", {})
    scale = geo.get("echelle_px_par_metre", 42.3)
    ox = geo.get("origine_pixel_x", 82)
    oy = geo.get("origine_pixel_y", 45)

    for zone_id, zone_data in config.get("zones_interdites", {}).items():
        pts_m = zone_data.get("coordonnees_metres", [])
        if not pts_m:
            continue
        pts_px = np.array(
            [[int(ox + x * scale), int(oy + y * scale)] for x, y in pts_m],
            dtype=np.int32,
        )
        cv2.polylines(plan_vis, [pts_px], isClosed=True, color=(0, 0, 220), thickness=2)
        cv2.fillPoly(
            plan_vis, [pts_px], color=(0, 0, 80)
        )  # remplissage semi-transparent (approximation)
        # Label de zone
        cx, cy = pts_px.mean(axis=0).astype(int)
        cv2.putText(
            plan_vis, zone_id, (cx - 20, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return plan_vis


# ----------------------------------------------------------------------
# Boucle principale
# ----------------------------------------------------------------------

def run(
    cam_id: str,
    model_path: str,
    config: dict,
    video_path: str,
    save_path: str | None = None,
    max_frames: int | None = None,
    target_classes: list[int] | None = None,
    show_window: bool = True,
    device: str | None = None,
) -> None:
    """Exécute le tracker sur une vidéo enregistrée et affiche les résultats
    (vidéo annotée, plan de sol avec trajectoires, sauvegarde optionnelle)."""

    # -- Tracker --
    tracker = CameraTracker(
        cam_id=cam_id,
        model_path=model_path,
        config=config,
        conf_threshold=0.4,
        target_classes=target_classes,
        device=device,
    )

    # -- Vidéo source --
    if not os.path.exists(video_path):
        print(f"[ERR] Vidéo introuvable : {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Vidéo : {video_path} | {total_frames} frames à {fps:.1f} FPS")

    # -- Plan de sol --
    plan_base = load_floor_plan(config)
    if plan_base is not None:
        plan_base = draw_forbidden_zones(plan_base, config)

    # -- Writer de sortie --
    writer = None
    if save_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Si aspect-ratio fix, la largeur devient 768 pour les caméras concernées
        ar_cams = config.get("aspect_ratio_fix", {}).get("distorted_cameras", [])
        if cam_id in ar_cams:
            w = config["aspect_ratio_fix"]["corrected_resolution"][0]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        print(f"[INFO] Sauvegarde vidéo → {save_path}")

    # -- Historique des trajectoires (trail) --
    # trails[track_id] = liste de (px, py) sur le plan de sol
    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
    TRAIL_LEN = 60  # nombre de positions à conserver par track

    # -- Statistiques --
    frame_idx = 0
    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # -- Tracking --
        detections = tracker.process_frame(frame, timestamp=ts)

        # -- Affichage vidéo --
        # Note : fix_frame() est appelé dans process_frame(), mais frame ici
        # est l'image ORIGINALE. Pour draw_detections, on doit passer la frame corrigée.
        from geometry_fix import fix_frame as _fix
        frame_fixed = _fix(frame, cam_id, tracker.ar_fix)
        vis_frame = tracker.draw_detections(frame_fixed, detections, show_metres=True)

        # FPS overlay
        elapsed = time.time() - t_start
        fps_real = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(
            vis_frame,
            f"Frame {frame_idx} | FPS {fps_real:.1f} | {cam_id}",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # -- Plan de sol avec trajectoires --
        if plan_base is not None:
            plan_vis = plan_base.copy()
            for det in detections:
                if det.foot_point_m is None:
                    continue
                px, py = metres_to_plan_px(det.foot_point_m[0], det.foot_point_m[1], config)
                color = _id_to_color(det.track_id)

                # Mettre à jour le trail
                trail = trails[det.track_id]
                trail.append((px, py))
                if len(trail) > TRAIL_LEN:
                    trail.pop(0)

                # Dessiner le trail
                for k in range(1, len(trail)):
                    alpha = k / len(trail)
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(plan_vis, trail[k - 1], trail[k], c, 2)

                # Position actuelle
                cv2.circle(plan_vis, (px, py), 6, color, -1)
                cv2.putText(
                    plan_vis, f"#{det.track_id}",
                    (px + 8, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

        # -- Affichage --
        if show_window:
            cv2.imshow(f"Tracker - {cam_id}", vis_frame)
            if plan_base is not None:
                cv2.imshow(f"Plan de sol - {cam_id}", plan_vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Arrêt sur touche 'q'")
                break

        # -- Sauvegarde --
        if writer:
            writer.write(vis_frame)

        frame_idx += 1

        # Log toutes les 100 frames
        if frame_idx % 100 == 0:
            n_tracks = len({d.track_id for d in detections})
            print(f"[INFO] Frame {frame_idx}/{total_frames or '?'} | "
                  f"{n_tracks} track(s) actif(s) | FPS={fps_real:.1f}")

    # -- Nettoyage --
    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n[INFO] Terminé : {frame_idx} frames en {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS réel)")


# ----------------------------------------------------------------------
# Point d'entrée CLI
# ----------------------------------------------------------------------

def main() -> None:
    """Analyse les arguments, résout le modèle et la vidéo, puis lance run()."""
    parser = argparse.ArgumentParser(
        description="Test du tracker mono-caméra avec ByteTrack"
    )
    parser.add_argument(
        "--cam", default="cam_03",
        help="ID de la caméra à tester (défaut: cam_03)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Chemin vers le modèle .pt (défaut: auto-détection)",
    )
    parser.add_argument(
        "--video", default=None,
        help="Chemin vers la vidéo (défaut: video_path dans config.yaml pour cette caméra)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Sauvegarder la vidéo annotée",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Nombre maximum de frames à traiter",
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="Ne pas afficher de fenêtre OpenCV (mode headless)",
    )
    parser.add_argument(
        "--all-classes", action="store_true",
        help="Tracker toutes les classes COCO (défaut: personnes seulement)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, ex: cuda:0 ou cpu. Défaut: detection.device dans config.yaml.",
    )
    args = parser.parse_args()

    # -- Config --
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # -- Modèle --
    model_path = args.model
    if model_path is None:
        parent = Path(__file__).parent.parent
        for candidate in ["yolov8n.pt", "yolo11n.pt", "yolov8s.pt"]:
            p = parent / candidate
            if p.exists():
                model_path = str(p)
                break
        if model_path is None:
            print("[ERR] Aucun modèle trouvé. Utilisez --model <chemin>")
            sys.exit(1)
        print(f"[INFO] Modèle auto-détecté : {model_path}")

    # -- Vidéo --
    video_path = args.video
    if video_path is None:
        cam_cfg = config.get("cameras", {}).get(args.cam, {})
        video_path = cam_cfg.get("video_path", None)
        if video_path is None:
            print(f"[ERR] Pas de video_path pour {args.cam} dans config.yaml. "
                  f"Utilisez --video <chemin>")
            sys.exit(1)
        # Résoudre par rapport au répertoire du script (le ".." du config est relatif à Phase_3/)
        if not os.path.isabs(video_path):
            video_path = str((Path(__file__).parent / video_path).resolve())
        print(f"[INFO] Vidéo depuis config : {video_path}")

    # -- Sortie --
    save_path = None
    if args.save:
        save_path = str(
            Path(__file__).parent / f"test_tracker_{args.cam}_{int(time.time())}.mp4"
        )

    # -- Classes cibles --
    target_classes = None if args.all_classes else [0]  # 0 = person

    run(
        cam_id=args.cam,
        model_path=model_path,
        config=config,
        video_path=video_path,
        save_path=save_path,
        max_frames=args.max_frames,
        target_classes=target_classes,
        show_window=not args.no_window,
        device=args.device,
    )


if __name__ == "__main__":
    main()

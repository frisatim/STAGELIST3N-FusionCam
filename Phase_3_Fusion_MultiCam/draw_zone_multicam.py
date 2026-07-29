"""
Outil de dessin de zone interdite avec projection multi-caméras.

Dessine une zone sur une caméra de référence → la zone est automatiquement
projetée sur toutes les caméras en overlap via les homographies.

Workflow :
  1. Choisir la caméra de référence
  2. Dessiner le polygone de la zone (clics successifs)
  3. L'outil convertit les pixels en mètres (via H de la caméra ref)
  4. L'outil projette les mètres en pixels sur chaque caméra en overlap (via H_inv)
  5. Affichage simultané de toutes les caméras avec la zone projetée
  6. Sauvegarde des coordonnées mètres dans config.yaml

Usage :
    python draw_zone_multicam.py --ref-camera cam_03
    python draw_zone_multicam.py --ref-camera cam_06 --zone zone_2
    python draw_zone_multicam.py --ref-camera cam_03 --source 2  (video locale)
"""

import os
import sys
import argparse
import glob
import re
import yaml
import numpy as np
import cv2
from copy import deepcopy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Couleurs par caméra (BGR)
CAM_COLORS = {
    "cam_01": (255, 100, 100),
    "cam_02": (100, 255, 100),
    "cam_03": (100, 100, 255),
    "cam_04": (255, 255, 100),
    "cam_05": (255, 100, 255),
    "cam_06": (100, 255, 255),
    "cam_07": (200, 150, 50),
    "cam_08": (50, 150, 200),
}
ZONE_COLOR = (0, 0, 255)       # Rouge pour la zone
ZONE_FILL_ALPHA = 0.25
POINT_RADIUS = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".m4v")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(path, config):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def resolve_path(rel):
    return rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)


def fix_aspect_ratio(frame, cam_id, config):
    ar_fix = config.get("aspect_ratio_fix", {})
    if not ar_fix.get("enabled", False):
        return frame
    if cam_id not in ar_fix.get("distorted_cameras", []):
        return frame
    distorted_w, distorted_h = ar_fix.get("distorted_resolution", [704, 576])
    tw, th = ar_fix["corrected_resolution"]
    h, w = frame.shape[:2]
    if not (w == distorted_w and h == distorted_h):
        return frame
    if w == tw and h == th:
        return frame
    return cv2.resize(frame, (tw, th))


def grab_frame(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


def _res_matches(a, b, tol=2):
    if not a or not b or len(a) != 2 or len(b) != 2:
        return False
    return abs(int(a[0]) - int(b[0])) <= tol and abs(int(a[1]) - int(b[1])) <= tol


def _extract_cam_number(cam_id, cam_cfg):
    name = str(cam_cfg.get("name", ""))
    m = re.search(r"Camera\s+(\d+)", name)
    if m:
        return int(m.group(1))

    m = re.search(r"cam_(\d+)", str(cam_id))
    if m:
        return int(m.group(1))

    return None


def _extract_cam_suffix(cam_cfg):
    # Exemple attendu: "2.11" depuis l'IP <CAMERA_IP>
    ip = str(cam_cfg.get("ip", "")).strip()
    parts = ip.split(".")
    if len(parts) == 4 and parts[-1].isdigit() and parts[-2].isdigit():
        return f"{parts[-2]}.{parts[-1]}"

    # Fallback depuis le nom "Camera 7 (2.11)"
    name = str(cam_cfg.get("name", ""))
    m = re.search(r"\((\d+\.\d+)\)", name)
    if m:
        return m.group(1)

    return None


def _hhmmss_to_seconds(hhmmss):
    if not hhmmss or len(hhmmss) != 6 or not hhmmss.isdigit():
        return None
    h = int(hhmmss[0:2])
    m = int(hhmmss[2:4])
    s = int(hhmmss[4:6])
    return h * 3600 + m * 60 + s


def _parse_preferred_times(preferred_time, preferred_times_csv):
    times = []

    # Keep explicit single prefer-time first when provided.
    if preferred_time and preferred_time.lower() != "auto":
        times.append(preferred_time.strip())

    if preferred_times_csv:
        for t in str(preferred_times_csv).split(","):
            t = t.strip()
            if t and t not in times:
                times.append(t)

    # Retain only valid HHMMSS values.
    valid = []
    for t in times:
        if len(t) == 6 and t.isdigit():
            valid.append(t)
    return valid


def _find_preferred_local_video(cam_id, cam_cfg, preferred_date, preferred_time, preferred_times_csv):
    rec_dir = os.path.join(PROJECT_ROOT, "recordings", "recordings")
    if not os.path.isdir(rec_dir):
        return None

    cam_num = _extract_cam_number(cam_id, cam_cfg)
    cam_suffix = _extract_cam_suffix(cam_cfg)
    if cam_num is None or cam_suffix is None:
        return None

    preferred_times = _parse_preferred_times(preferred_time, preferred_times_csv)

    # 1) Match exact voulu par l'utilisateur (1..N heures candidates)
    exact_candidates = []
    for target_time in preferred_times:
        for ext in VIDEO_EXTENSIONS:
            p = os.path.join(
                rec_dir,
                f"Camera_{cam_num}_{cam_suffix}_{preferred_date}_{target_time}{ext}",
            )
            if os.path.isfile(p):
                exact_candidates.append((target_time, p))
    if exact_candidates:
        # Preserve preferred_times order, then deterministic path order.
        exact_candidates.sort(key=lambda item: (preferred_times.index(item[0]), item[1]))
        return exact_candidates[0][1]

    # 2) Fallback: même date, prendre l'heure la plus proche de l'heure cible
    same_day = []
    pattern = os.path.join(rec_dir, f"Camera_{cam_num}_{cam_suffix}_{preferred_date}_*")
    for p in glob.glob(pattern):
        ext = os.path.splitext(p)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            continue

        base = os.path.basename(p)
        m = re.match(
            rf"^Camera_{cam_num}_{re.escape(cam_suffix)}_{preferred_date}_(\d{{6}})",
            base,
            flags=re.IGNORECASE,
        )
        if not m:
            continue

        t = m.group(1)
        same_day.append((p, t))

    if same_day:
        # If no exact match, fallback to nearest from first preferred target when possible.
        target_time = preferred_times[0] if preferred_times else preferred_time
        target_sec = _hhmmss_to_seconds(target_time)
        if target_sec is not None:
            def key_fn(item):
                p, t = item
                ts = _hhmmss_to_seconds(t)
                if ts is None:
                    return (10**9, -os.path.getmtime(p))
                return (abs(ts - target_sec), -os.path.getmtime(p))

            same_day.sort(key=key_fn)
        else:
            same_day.sort(key=lambda item: os.path.getmtime(item[0]), reverse=True)

        return same_day[0][0]

    return None


def select_homography_for_resolution(config, cam_id, effective_res):
    hom = config.get("homographie", {})
    cam_data = hom.get(cam_id, {}) if isinstance(hom.get(cam_id, {}), dict) else {}

    matrix_sub = hom.get(f"{cam_id}_matrix") or cam_data.get("matrix")
    matrix_hd = cam_data.get("matrix_hd")
    hd_res = cam_data.get("hd_resolution")
    sub_res = cam_data.get("substream_resolution")

    if matrix_hd and _res_matches(effective_res, hd_res):
        return np.array(matrix_hd, dtype=np.float64), "matrix_hd"
    if matrix_sub and _res_matches(effective_res, sub_res):
        return np.array(matrix_sub, dtype=np.float64), f"{cam_id}_matrix"

    if matrix_sub:
        return np.array(matrix_sub, dtype=np.float64), f"{cam_id}_matrix (fallback)"
    if matrix_hd:
        return np.array(matrix_hd, dtype=np.float64), "matrix_hd (fallback)"

    return None, "none"


def get_source(cam_id, cam_cfg, config, source_type, preferred_date, preferred_time, preferred_times_csv):
    if source_type == "1":
        return cam_cfg["rtsp_url"]
    else:
        # Priorite: videos synchronisees de type Camera_x_2.x_20260420_103914
        preferred = _find_preferred_local_video(
            cam_id,
            cam_cfg,
            preferred_date,
            preferred_time,
            preferred_times_csv,
        )
        if preferred:
            return preferred

        # Prefer the exact video used during calibration when available.
        cam_h = config.get("homographie", {}).get(cam_id, {})
        if isinstance(cam_h, dict):
            calib_src = cam_h.get("calibration_source")
            if calib_src:
                candidates = []
                if os.path.isabs(calib_src):
                    candidates.append(calib_src)
                candidates.append(os.path.join(PROJECT_ROOT, "recordings", "recordings", calib_src))
                for c in candidates:
                    if os.path.isfile(c):
                        return c

        vp = cam_cfg.get("video_path")
        if not vp:
            return None
        path = resolve_path(vp)
        return path if os.path.isfile(path) else None


def get_overlap_cameras(config, ref_cam):
    """Trouve toutes les caméras en overlap avec ref_cam."""
    overlaps = config.get("camera_overlaps", {})
    partners = set()
    for room_id, pairs in overlaps.items():
        if pairs is None:
            continue
        for pair in pairs:
            if ref_cam in pair:
                other = pair[0] if pair[1] == ref_cam else pair[1]
                partners.add(other)
    return sorted(partners)


def pixels_to_metres(pts_px, H):
    """Convertit des points pixels en mètres via l'homographie H."""
    pts = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    pts_m = cv2.perspectiveTransform(pts, H)
    return pts_m.reshape(-1, 2)


def metres_to_pixels(pts_m, H):
    """Convertit des points mètres en pixels via l'homographie inverse H_inv."""
    H_inv = np.linalg.inv(H)
    pts = np.array(pts_m, dtype=np.float32).reshape(-1, 1, 2)
    pts_px = cv2.perspectiveTransform(pts, H_inv)
    return pts_px.reshape(-1, 2).astype(np.int32)


def draw_zone_on_frame(frame, pts_px, color=ZONE_COLOR, alpha=ZONE_FILL_ALPHA, label=None):
    """Dessine un polygone semi-transparent sur la frame."""
    if len(pts_px) < 3:
        # Dessiner juste les points et lignes
        for i, pt in enumerate(pts_px):
            cv2.circle(frame, tuple(pt), POINT_RADIUS, color, -1)
            if i > 0:
                cv2.line(frame, tuple(pts_px[i-1]), tuple(pt), color, 2)
        return frame

    pts = np.array(pts_px, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    if label:
        cx = int(np.mean([p[0] for p in pts_px]))
        cy = int(np.mean([p[1] for p in pts_px]))
        cv2.putText(frame, label, (cx - 30, cy), FONT, 0.5, color, 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Dessiner une zone avec projection multi-cameras")
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "config.yaml"),
        help="Fichier de configuration a lire/ecrire (defaut: config.yaml)",
    )
    parser.add_argument("--ref-camera", required=True, help="Camera de reference pour dessiner (ex: cam_03)")
    parser.add_argument("--zone", default="zone_new", help="Nom de la zone (ex: zone_1, zone_2)")
    parser.add_argument("--source", default="2", choices=["1", "2"], help="1=RTSP, 2=Video locale")
    parser.add_argument("--prefer-date", default="20260429", help="Date preferee des videos locales (AAAAMMJJ)")
    parser.add_argument("--prefer-time", default="auto", help="Heure preferee unique (HHMMSS) ou auto")
    parser.add_argument(
        "--prefer-times",
        default="094812,095835",
        help="Liste d'heures candidates separees par des virgules (ordre de priorite)",
    )
    parser.add_argument("--save", action="store_true", help="Sauvegarder la zone dans config.yaml")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    ref_cam = args.ref_camera
    if ref_cam not in config["cameras"]:
        sys.exit(f"[ERR] Camera '{ref_cam}' inconnue.")

    # --- Trouver les caméras en overlap ---
    overlap_cams = get_overlap_cameras(config, ref_cam)
    print(f"\n[INFO] Camera de reference : {ref_cam}")
    print(f"[INFO] Cameras en overlap  : {overlap_cams}")

    # --- Charger frames + homographies adaptées à chaque résolution ---
    cam_data = {}  # {cam_id: {"H": matrix, "frame": frame, "matrix_name": str, "source": str}}

    for cam_id in [ref_cam] + overlap_cams:
        cam_data[cam_id] = {
            "H": None,
            "frame": None,
            "matrix_name": "none",
            "source": None,
        }

        # Charger une frame
        source = get_source(
            cam_id,
            config["cameras"][cam_id],
            config,
            args.source,
            args.prefer_date,
            args.prefer_time,
            args.prefer_times,
        )
        cam_data[cam_id]["source"] = source
        if source:
            frame = grab_frame(source)
            if frame is not None:
                frame = fix_aspect_ratio(frame, cam_id, config)
                cam_data[cam_id]["frame"] = frame

                effective_res = (frame.shape[1], frame.shape[0])
                H, matrix_name = select_homography_for_resolution(config, cam_id, effective_res)
                cam_data[cam_id]["H"] = H
                cam_data[cam_id]["matrix_name"] = matrix_name

                if H is None:
                    print(f"[WARN] {cam_id} : frame {effective_res[0]}x{effective_res[1]} | matrice absente")
                else:
                    print(
                        f"[INFO] {cam_id} : frame {effective_res[0]}x{effective_res[1]} | "
                        f"matrice={matrix_name}"
                    )
            else:
                print(f"[WARN] {cam_id} : impossible de lire la frame")
        else:
            print(f"[WARN] {cam_id} : source non disponible")

    if cam_data[ref_cam]["frame"] is None:
        sys.exit(f"[ERR] Impossible de charger la frame de {ref_cam}")
    if cam_data[ref_cam]["H"] is None:
        sys.exit(f"[ERR] Impossible de choisir une homographie valide pour {ref_cam}.")

    H_ref = cam_data[ref_cam]["H"]

    # --- Interface de dessin ---
    zone_points_px = []  # Points en pixels sur la caméra de référence
    ref_frame_original = cam_data[ref_cam]["frame"].copy()
    drawing_done = False

    def update_all_views():
        """Met à jour l'affichage de toutes les caméras avec la zone projetée."""
        # Convertir les points pixels ref -> mètres uniquement si le polygone est valide.
        pts_metres = None
        if len(zone_points_px) >= 3:
            pts_px = np.array(zone_points_px, dtype=np.float32)
            pts_metres = pixels_to_metres(pts_px, H_ref)

        # --- Afficher la caméra de référence ---
        ref_display = ref_frame_original.copy()
        # Dessiner les points cliqués
        for i, pt in enumerate(zone_points_px):
            cv2.circle(ref_display, tuple(pt), POINT_RADIUS, (0, 255, 0), -1)
            cv2.putText(ref_display, str(i+1), (pt[0]+8, pt[1]-8), FONT, 0.4, (255,255,255), 1)
            if i > 0:
                cv2.line(ref_display, tuple(zone_points_px[i-1]), tuple(pt), (0, 255, 0), 2)
        if len(zone_points_px) >= 3:
            # Fermer le polygone
            cv2.line(ref_display, tuple(zone_points_px[-1]), tuple(zone_points_px[0]), (0, 255, 0), 2)
            draw_zone_on_frame(
                ref_display,
                np.array(zone_points_px, dtype=np.int32),
                ZONE_COLOR,
                ZONE_FILL_ALPHA,
                args.zone,
            )

        # Bandeau info
        n = len(zone_points_px)
        info = f"{ref_cam} (REF) | {n} points | Clic=ajouter | Z=undo | ENTREE=valider | Q=quitter"
        cv2.rectangle(ref_display, (0, 0), (ref_display.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(ref_display, info, (5, 20), FONT, 0.42, (255, 255, 255), 1)
        if n == 0:
            cv2.putText(
                ref_display,
                "Cliquez pour ajouter des points de zone",
                (8, 48),
                FONT,
                0.55,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            ref_display,
            f"H={cam_data[ref_cam]['matrix_name']}",
            (5, ref_display.shape[0] - 8),
            FONT,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.imshow(f"REF: {ref_cam}", ref_display)

        # --- Afficher les caméras en overlap avec projection ---
        for cam_id in overlap_cams:
            frame = cam_data[cam_id]["frame"]
            H = cam_data[cam_id]["H"]
            if frame is None:
                continue

            display = frame.copy()
            if H is None:
                cv2.putText(display, "PAS CALIBREE", (10, 30), FONT, 0.7, (0, 0, 255), 2)
            elif pts_metres is not None and len(pts_metres) >= 3:
                projected_px = metres_to_pixels(pts_metres, H)
                draw_zone_on_frame(display, projected_px, ZONE_COLOR, ZONE_FILL_ALPHA, args.zone)

                # Afficher les coordonnées mètres des coins
                for i, (pm, pp) in enumerate(zip(pts_metres, projected_px)):
                    cv2.circle(display, tuple(pp), 4, (0, 255, 0), -1)
                    txt = f"({pm[0]:.1f},{pm[1]:.1f})m"
                    cv2.putText(display, txt, (pp[0]+5, pp[1]-5), FONT, 0.35, (255,255,255), 1)

            # Bandeau
            cv2.rectangle(display, (0, 0), (display.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(display, f"{cam_id} (projection depuis {ref_cam})", (5, 16),
                        FONT, 0.42, (255, 255, 255), 1)
            cv2.putText(
                display,
                f"H={cam_data[cam_id]['matrix_name']}",
                (5, display.shape[0] - 8),
                FONT,
                0.45,
                (255, 255, 255),
                1,
            )
            if pts_metres is None:
                cv2.putText(
                    display,
                    "En attente du polygone (>=3 points)",
                    (8, 44),
                    FONT,
                    0.5,
                    (0, 255, 255),
                    1,
                )
            cv2.imshow(f"OVERLAP: {cam_id}", display)

    def on_click(event, x, y, flags, param):
        nonlocal drawing_done
        if drawing_done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            zone_points_px.append([x, y])
            print(f"  Point {len(zone_points_px)} : pixel ({x}, {y})")
            update_all_views()

    # --- Setup fenêtres ---
    ref_win = f"REF: {ref_cam}"
    cv2.namedWindow(ref_win)
    cv2.setMouseCallback(ref_win, on_click)

    print(f"\n{'='*60}")
    print(f"  DESSIN DE ZONE — {args.zone}")
    print(f"  Camera de reference : {ref_cam}")
    print(f"{'='*60}")
    print(f"  Cliquez les coins de la zone sur la camera {ref_cam}.")
    print(f"  Minimum 3 points pour former un polygone.")
    print(f"  Z = annuler le dernier point")
    print(f"  ENTREE = valider la zone")
    print(f"  Q = quitter sans sauvegarder")
    print()

    update_all_views()

    # --- Boucle principale ---
    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q') or key == 27:
            print("[INFO] Abandon.")
            cv2.destroyAllWindows()
            return

        if key == ord('z') and len(zone_points_px) > 0 and not drawing_done:
            removed = zone_points_px.pop()
            print(f"  [UNDO] Point retire : ({removed[0]}, {removed[1]})")
            update_all_views()

        if (key == 13 or key == 10) and len(zone_points_px) >= 3:
            drawing_done = True
            break

    # --- Zone validée ---
    pts_px = np.array(zone_points_px, dtype=np.float32)
    pts_metres = pixels_to_metres(pts_px, H_ref)

    print(f"\n{'='*60}")
    print(f"  ZONE VALIDEE : {args.zone}")
    print(f"{'='*60}")
    print(f"  {len(zone_points_px)} points")
    print(f"\n  Coordonnees metres (origine = haut-gauche image) :")
    for i, (pm, pp) in enumerate(zip(pts_metres, zone_points_px)):
        print(f"    P{i+1} : ({pm[0]:.3f}, {pm[1]:.3f}) m  ← pixel ({pp[0]}, {pp[1]})")

    print(f"\n  Projection sur les cameras en overlap :")
    for cam_id in overlap_cams:
        H = cam_data[cam_id]["H"]
        if H is not None:
            proj_px = metres_to_pixels(pts_metres, H)
            print(f"    {cam_id} ({cam_data[cam_id]['matrix_name']}) :")
            for i, (pm, pp) in enumerate(zip(pts_metres, proj_px)):
                print(f"      P{i+1} : pixel ({pp[0]}, {pp[1]})")
        else:
            print(f"    {cam_id} : PAS CALIBREE")

    # --- Sauvegarde ---
    if args.save:
        coords_list = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in pts_metres]

        if "zones_interdites" not in config:
            config["zones_interdites"] = {}

        # Déterminer les caméras concernées
        cams_concernees = [ref_cam] + [c for c in overlap_cams if cam_data[c]["H"] is not None]

        config["zones_interdites"][args.zone] = {
            "nom": args.zone,
            "type_alerte": "violation_zone",
            "cameras_concernees": cams_concernees,
            "coordonnees_metres": coords_list,
        }

        save_config(config_path, config)
        print(f"\n  [OK] Zone '{args.zone}' sauvegardee dans config.yaml")
        print(f"       Cameras : {cams_concernees}")
    else:
        print(f"\n  [INFO] Zone NON sauvegardee. Relancez avec --save pour sauvegarder.")
        print(f"         Ou copiez ces coordonnees dans config.yaml manuellement :")
        print(f"\n  coordonnees_metres:")
        for p in pts_metres:
            print(f"    - [{p[0]:.3f}, {p[1]:.3f}]")

    # --- Attendre avant de fermer ---
    print(f"\n  Appuyez sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

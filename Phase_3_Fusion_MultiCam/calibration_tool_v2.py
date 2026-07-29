"""
Outil de calibration par homographie (Phase 3.1 - Multi-résolution).

Amélioration clé par rapport à la version précédente (conservée dans
legacy/calibration_tool_v1.py) :
  Support de la calibration en HAUTE RÉSOLUTION (flux principal / mainstream)
  ou directement sur le flux secondaire corrigé 768x576.
  Si la source est une autre résolution, conversion automatique vers la
  RÉSOLUTION D'INFÉRENCE (flux secondaire / substream).

  On clique les points sur une frame source (idéalement une vidéo substream
  enregistrée en 768x576 pour correspondre à l'inférence), et l'outil stocke
  DEUX matrices :
    - H_calibration : utilisable avec la frame source
    - H_inference   : utilisable avec la frame substream (apres fix ratio)

Principe mathématique :
  Soit H la matrice calculée avec des pixels HD : pts_metres = H @ pts_HD
  Soit S la matrice de scale substream -> HD : pts_HD = S @ pts_substream
  Alors : pts_metres = (H @ S) @ pts_substream
  Donc : H_inference = H_calibration @ S

Usage :
  # Calibrer avec une vidéo substream corrigée 768x576
  python calibration_tool_v2.py --camera cam_03 --video recordings/recordings/cam_03_calib.mp4 \\
                                --ref-file ref_points_template.yaml

  # Après calibration, verify_calibration.py utilisera automatiquement H_inference
  # sur les frames du substream (avec fix ratio 768x576)
"""

import os
import sys
import argparse
import glob
import re
import yaml
import numpy as np
import cv2
from datetime import datetime


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
COLORS = {
    "point_new":    (0, 255, 0),
    "point_reproj": (0, 165, 255),
    "error_line":   (0, 0, 255),
    "text":         (255, 255, 255),
    "text_bg":      (0, 0, 0),
    "good":         (0, 255, 0),
    "warn":         (0, 200, 255),
    "bad":          (0, 0, 255),
}
RADIUS = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 2
MIN_POINTS = 4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RECORDINGS_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "recordings", "recordings")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".m4v")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(path, config):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def resolve_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


def resolve_hd_video_path(video_input=None, cam_id=None):
    """Resolve un chemin de video HD avec fallback automatique.

    Supporte:
      - chemin absolu
      - chemin relatif (cwd, script dir, project root)
      - nom de fichier sans extension
      - nom de base retrouve dans recordings/recordings
      - auto-detection de la derniere video d'une camera si video_input est vide
    """
    candidate_paths = []

    if video_input:
        if os.path.isabs(video_input):
            candidate_paths.append(os.path.normpath(video_input))
        else:
            candidate_paths.extend([
                os.path.normpath(os.path.abspath(video_input)),
                os.path.normpath(os.path.join(SCRIPT_DIR, video_input)),
                os.path.normpath(os.path.join(PROJECT_ROOT, video_input)),
            ])

        # De-dup en preservant l'ordre
        unique = []
        seen = set()
        for c in candidate_paths:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        # 1) Match direct
        for c in unique:
            if os.path.isfile(c):
                return c

        # 2) Auto-completion d'extension
        roots = []
        for c in unique:
            root, ext = os.path.splitext(c)
            if ext.lower() in VIDEO_EXTENSIONS:
                continue
            roots.append(c)      # conserve les noms avec points (ex: Camera_3_2.4_...)
            roots.append(root)   # fallback splitext classique

        roots_unique = []
        seen_roots = set()
        for r in roots:
            if r not in seen_roots:
                seen_roots.add(r)
                roots_unique.append(r)

        for r in roots_unique:
            for ext in VIDEO_EXTENSIONS:
                p = r + ext
                if os.path.isfile(p):
                    print(f"  [INFO] Video resolue automatiquement : {p}")
                    return p

            for p in sorted(glob.glob(r + ".*")):
                if os.path.isfile(p) and os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS:
                    print(f"  [INFO] Video resolue automatiquement : {p}")
                    return p

        # 3) Recherche par basename dans recordings/recordings
        if os.path.isdir(RECORDINGS_DIR_DEFAULT):
            base = os.path.basename(video_input)
            base_no_ext = os.path.splitext(base)[0]
            matches = []
            for ext in VIDEO_EXTENSIONS:
                pattern = os.path.join(RECORDINGS_DIR_DEFAULT, "**", base_no_ext + ext)
                matches.extend(glob.glob(pattern, recursive=True))
            if matches:
                chosen = sorted(set(os.path.normpath(m) for m in matches))[0]
                print(f"  [INFO] Video resolue automatiquement : {chosen}")
                return chosen

    # 4) Si pas de video explicite, prendre la plus recente de la camera
    if cam_id and os.path.isdir(RECORDINGS_DIR_DEFAULT):
        cam_low = cam_id.lower()
        m = re.search(r"(\d+)$", cam_low)
        cam_num = str(int(m.group(1))) if m else ""

        tokens = []
        if cam_num:
            tokens = [
                f"camera_{cam_num}_",
                f"camera {cam_num} ",
                f"cam_{cam_num}",
                f"cam{cam_num}",
            ]

        all_files = []
        for root, _, files in os.walk(RECORDINGS_DIR_DEFAULT):
            for name in files:
                if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                    all_files.append(os.path.join(root, name))

        def match_score(path):
            low = os.path.basename(path).lower()
            s = 0
            for tk in tokens:
                if tk and tk in low:
                    s += 1
            return s

        scored = [(match_score(p), p) for p in all_files]
        scored = [sp for sp in scored if sp[0] > 0]
        if scored:
            # Meilleur score, puis plus recent
            best_score = max(s for s, _ in scored)
            best = [p for s, p in scored if s == best_score]
            best.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            chosen = os.path.normpath(best[0])
            print(f"  [INFO] Video auto-detectee pour {cam_id} : {chosen}")
            return chosen

    return None


def load_ref_points(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("ref_points", {})


def validate_ref_points_metres(ref_points):
    """Valide les coordonnees metres pour eviter une homographie degeneree."""
    if len(ref_points) < MIN_POINTS:
        return False, f"Minimum {MIN_POINTS} points requis."

    pts = np.array(ref_points, dtype=np.float64)
    unique_pts = np.unique(np.round(pts, 6), axis=0)

    if len(unique_pts) < MIN_POINTS:
        return False, (
            "Points metres invalides: il faut au moins 4 points distincts "
            f"(actuel: {len(unique_pts)})."
        )

    if np.allclose(pts, 0.0):
        return False, (
            "Points metres invalides: toutes les coordonnees sont a (0,0). "
            "Remplissez ref_points_room1.yaml avec les vraies positions en metres."
        )

    return True, ""


# ---------------------------------------------------------------------------
# Capture frame depuis source HD
# ---------------------------------------------------------------------------
def grab_frame_hd(source, frame_number=None):
    """Capture une frame depuis une source HD.

    Si frame_number est donné : saute à cette frame (utile pour une vidéo
    où les gobelets sont posés après quelques secondes).
    Sinon : première frame valide.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERREUR] Impossible d'ouvrir : {source}")

    if frame_number is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        sys.exit(f"[ERREUR] Aucune frame depuis : {source}")
    return frame


def browse_video_for_frame(source):
    """Permet à l'utilisateur de parcourir la vidéo pour choisir la bonne frame
    (là où tous les gobelets sont bien placés et visibles)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERREUR] Impossible d'ouvrir : {source}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"\n  Video : {n_frames} frames, {fps:.1f} FPS, duree {n_frames/fps:.1f}s")

    window = "Choisissez la frame (gauche/droite: +/- 1s | espace: avance 5s | ENTREE: valider)"
    current = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, current)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        sys.exit("[ERREUR] Lecture frame impossible")

    step_1s = int(fps)
    step_5s = int(fps * 5)

    while True:
        # Afficher info
        display = frame.copy()
        # Scale down for display if too big
        h, w = display.shape[:2]
        scale = 1.0
        if w > 1280:
            scale = 1280 / w
            display = cv2.resize(display, None, fx=scale, fy=scale)

        info = f"Frame {current}/{n_frames}  ({current/fps:.1f}s)"
        cv2.rectangle(display, (0, 0), (display.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(display, info, (10, 22), FONT, 0.6, (255, 255, 255), 2)
        cv2.imshow(window, display)

        key = cv2.waitKey(0) & 0xFF
        if key == 13 or key == 10:  # ENTREE
            break
        elif key == 27:  # ECHAP
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("Annule par l'utilisateur")
        elif key == 81 or key == ord('a'):  # GAUCHE
            current = max(0, current - step_1s)
        elif key == 83 or key == ord('d'):  # DROITE
            current = min(n_frames - 1, current + step_1s)
        elif key == 32:  # ESPACE = +5s
            current = min(n_frames - 1, current + step_5s)
        elif key == ord('q'):
            current = max(0, current - step_5s)

        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, frame = cap.read()
        if not ret:
            print("  [WARN] Fin de video")
            break

    cap.release()
    cv2.destroyWindow(window)
    print(f"  Frame selectionnee : {current}")
    return frame, current


# ---------------------------------------------------------------------------
# Collecte de clics avec support du zoom pour HD
# ---------------------------------------------------------------------------
def collect_points_hd(image, window_name, labels, n_points, ref_coords_metres=None):
    """Collecte N clics sur une image HD avec :
       - Zoom fenêtré (molette ou touches +/-) pour cliquer avec précision
       - Affichage de coordonnées mètres attendues
       - Undo (z)
    """
    points = []
    display_scale = 1.0
    offset = [0, 0]  # décalage pour le pan
    h_img, w_img = image.shape[:2]

    # Calcul scale initial pour fit dans 1280x720
    max_w = 1600
    max_h = 900
    init_scale = min(max_w / w_img, max_h / h_img, 1.0)
    display_scale = init_scale
    current_scale = [display_scale]
    current_offset = [0.0, 0.0]

    def render():
        scale = current_scale[0]
        ox, oy = current_offset
        view_w = int(w_img * scale)
        view_h = int(h_img * scale)
        display = cv2.resize(image, (view_w, view_h), interpolation=cv2.INTER_AREA)

        # Dessiner les points déjà placés
        for i, pt in enumerate(points):
            px = int(pt[0] * scale)
            py = int(pt[1] * scale)
            label = labels[i] if i < len(labels) else f"P{i+1}"
            cv2.circle(display, (px, py), RADIUS, COLORS["point_new"], -1)
            cv2.circle(display, (px, py), RADIUS + 1, (0, 0, 0), 1)
            if ref_coords_metres and i < len(ref_coords_metres):
                mx, my = ref_coords_metres[i]
                info = f"{label} ({mx:.2f},{my:.2f})m"
            else:
                info = label
            cv2.putText(display, info, (px+10, py-10), FONT, 0.5, (0,0,0), 3)
            cv2.putText(display, info, (px+10, py-10), FONT, 0.5, (255,255,255), 1)

        # Bandeau info
        n_done = len(points)
        remaining = n_points - n_done
        if remaining > 0:
            next_label = labels[n_done] if n_done < len(labels) else f"P{n_done+1}"
            if ref_coords_metres and n_done < len(ref_coords_metres):
                mx, my = ref_coords_metres[n_done]
                info = f"[{n_done+1}/{n_points}] Cliquez : {next_label}  ({mx:.2f}, {my:.2f})m  |  Zoom: +/- | Undo: Z"
            else:
                info = f"[{n_done+1}/{n_points}] Cliquez : {next_label}  |  Zoom: +/- | Undo: Z"
        else:
            info = f"[TERMINE] {n_points} points collectes. Touche pour continuer."

        cv2.rectangle(display, (0, 0), (display.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(display, info, (10, 25), FONT, FONT_SCALE, (255,255,255), FONT_THICK)
        return display

    def on_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            # Convertir les coordonnées display vers les coordonnées image originales
            scale = current_scale[0]
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            points.append([orig_x, orig_y])
            idx = len(points)
            label = labels[idx-1] if idx <= len(labels) else f"P{idx}"
            print(f"  Point {idx}/{n_points} [{label}] : pixel HD ({orig_x}, {orig_y})")
            cv2.imshow(window_name, render())

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom molette (pas toujours supporté selon plateforme)
            if flags > 0:  # molette up
                current_scale[0] = min(current_scale[0] * 1.25, 3.0)
            else:
                current_scale[0] = max(current_scale[0] * 0.8, 0.2)
            cv2.imshow(window_name, render())

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(w_img * display_scale), int(h_img * display_scale) + 35)
    cv2.imshow(window_name, render())
    cv2.setMouseCallback(window_name, on_event)

    print(f"\n>> Cliquez les {n_points} points de reference. Touches : + (zoom in) | - (zoom out) | z (undo).")
    print("   Appuyez sur une touche apres le dernier clic.\n")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if len(points) >= n_points and key not in (255, 0):
            break
        if key == ord('+') or key == ord('='):
            current_scale[0] = min(current_scale[0] * 1.25, 3.0)
            cv2.imshow(window_name, render())
        elif key == ord('-') or key == ord('_'):
            current_scale[0] = max(current_scale[0] * 0.8, 0.2)
            cv2.imshow(window_name, render())
        elif key == ord('z') and len(points) > 0:
            removed = points.pop()
            print(f"  [UNDO] Point retire : ({removed[0]}, {removed[1]})")
            cv2.imshow(window_name, render())

    cv2.destroyWindow(window_name)
    return points


# ---------------------------------------------------------------------------
# Calcul homographie + diagnostics
# ---------------------------------------------------------------------------
def compute_homography(pts_pixels, pts_metres, use_ransac=True):
    n = len(pts_pixels)
    method = cv2.RANSAC if (use_ransac and n >= 5) else 0
    ransac_thresh = 3.0 if use_ransac else 0

    matrix, mask = cv2.findHomography(pts_pixels, pts_metres, method, ransac_thresh)
    if matrix is None:
        return None, None, {"error": "Calcul echoue"}

    pts_px_h = pts_pixels.reshape(-1, 1, 2).astype(np.float32)
    pts_reproj = cv2.perspectiveTransform(pts_px_h, matrix).reshape(-1, 2)
    errors_cm = np.linalg.norm(pts_reproj - pts_metres, axis=1) * 100

    h_inv = np.linalg.inv(matrix)
    pts_m_h = pts_metres.reshape(-1, 1, 2).astype(np.float32)
    pts_reproj_px = cv2.perspectiveTransform(pts_m_h, h_inv).reshape(-1, 2)
    errors_px = np.linalg.norm(pts_reproj_px - pts_pixels, axis=1)

    diagnostics = {
        "n_points": n,
        "method": "RANSAC" if method == cv2.RANSAC else "EXACT",
        "inliers": int(mask.sum()) if mask is not None else n,
        "errors_cm": errors_cm.tolist(),
        "errors_px": errors_px.tolist(),
        "mean_error_cm": float(errors_cm.mean()),
        "max_error_cm": float(errors_cm.max()),
        "mean_error_px": float(errors_px.mean()),
    }
    return matrix, mask, diagnostics


# ---------------------------------------------------------------------------
# Conversion matrice HD → matrice substream
# ---------------------------------------------------------------------------
def convert_homography_to_substream(H_hd, hd_resolution, substream_resolution):
    """Convertit une homographie calibrée en HD vers une homographie utilisable
    sur le substream.

    Args:
        H_hd: matrice 3x3 calibrée avec des pixels HD
        hd_resolution: (width_HD, height_HD) ex: (1920, 1080)
        substream_resolution: (width_sub, height_sub) ex: (768, 576)

    Returns:
        H_substream: matrice 3x3 utilisable avec des pixels substream
    """
    w_hd, h_hd = hd_resolution
    w_sub, h_sub = substream_resolution

    # Matrice de scale : substream -> HD
    # Si substream est plus petit, les pixels du substream correspondent à des
    # pixels "plus grands" en HD (facteur de scale > 1)
    sx = w_hd / w_sub
    sy = h_hd / h_sub

    S = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=np.float64)

    # Composition : H_substream = H_hd @ S
    # (pixel_substream -> pixel_hd -> metres)
    H_substream = H_hd @ S

    # Normaliser (la dernière valeur doit être 1)
    H_substream = H_substream / H_substream[2, 2]

    return H_substream


# ---------------------------------------------------------------------------
# Rapport diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(diag, labels, context=""):
    print("\n" + "=" * 70)
    print(f"  RAPPORT DE CALIBRATION {context}")
    print("=" * 70)
    print(f"  Points : {diag['n_points']}  |  Methode : {diag['method']}")

    print(f"\n  {'Point':<25} {'Err (cm)':<12} {'Err (px)':<12} {'Statut'}")
    print("  " + "-" * 60)

    for i in range(diag['n_points']):
        label = labels[i] if i < len(labels) else f"P{i+1}"
        err_cm = diag['errors_cm'][i]
        err_px = diag['errors_px'][i]
        if err_cm < 5:
            status = "OK"
        elif err_cm < 15:
            status = "MOYEN"
        else:
            status = "MAUVAIS !"
        print(f"  {label[:25]:<25} {err_cm:<12.2f} {err_px:<12.1f} {status}")

    print(f"\n  Erreur moyenne : {diag['mean_error_cm']:.2f} cm")
    print(f"  Erreur max     : {diag['max_error_cm']:.2f} cm")

    mean = diag['mean_error_cm']
    if mean < 5:
        verdict = "EXCELLENT"
    elif mean < 10:
        verdict = "BON"
    elif mean < 20:
        verdict = "MOYEN"
    else:
        verdict = "MAUVAIS"
    print(f"  Verdict        : {verdict}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Visualisation reprojection
# ---------------------------------------------------------------------------
def draw_reprojection_on_image(image, pts_pixels, H, pts_metres, labels, max_display_w=1280):
    """Dessine les points originaux + reprojetés + lignes d'erreur."""
    vis = image.copy()
    h_inv = np.linalg.inv(H)

    pts_m_h = pts_metres.reshape(-1, 1, 2).astype(np.float32)
    reproj_px = cv2.perspectiveTransform(pts_m_h, h_inv).reshape(-1, 2).astype(np.int32)

    pts_px_h = pts_pixels.reshape(-1, 1, 2).astype(np.float32)
    pts_reproj_m = cv2.perspectiveTransform(pts_px_h, H).reshape(-1, 2)
    errors_cm = np.linalg.norm(pts_reproj_m - pts_metres, axis=1) * 100

    for i in range(len(pts_pixels)):
        px_orig = tuple(pts_pixels[i].astype(int))
        px_repr = tuple(reproj_px[i])
        err_cm = errors_cm[i]
        label = labels[i] if i < len(labels) else f"P{i+1}"

        if err_cm < 5:
            color = COLORS["good"]
        elif err_cm < 15:
            color = COLORS["warn"]
        else:
            color = COLORS["bad"]

        cv2.circle(vis, px_orig, RADIUS + 2, color, -1)
        cv2.circle(vis, px_repr, RADIUS, COLORS["point_reproj"], 3)
        cv2.line(vis, px_orig, px_repr, COLORS["error_line"], 2)

        info = f"{label}: {err_cm:.1f}cm"
        cv2.putText(vis, info, (px_orig[0]+12, px_orig[1]-12), FONT, 0.6, (0,0,0), 3)
        cv2.putText(vis, info, (px_orig[0]+12, px_orig[1]-12), FONT, 0.6, color, 2)

    # Scale down pour affichage si HD
    h, w = vis.shape[:2]
    if w > max_display_w:
        scale = max_display_w / w
        vis = cv2.resize(vis, None, fx=scale, fy=scale)

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calibration homographique v3 (multi-resolution)")
    parser.add_argument("--camera", required=True, help="Camera (ex: cam_03)")
    parser.add_argument("--hd-video", default=None,
                        help="Chemin ou nom de la video source (historique: HD; substream 768x576 recommande)")
    parser.add_argument("--video", default=None,
                        help="Alias de --hd-video")
    parser.add_argument("--ref-file", required=True, help="YAML des points de reference")
    parser.add_argument("--frame", type=int, default=None,
                        help="Numero de frame a utiliser (sinon parcours interactif)")
    parser.add_argument("--substream-resolution", nargs=2, type=int, default=None,
                        help="Resolution substream (largeur hauteur). Par defaut : lu depuis config.yaml aspect_ratio_fix")
    args = parser.parse_args()

    # --- Chargement config ---
    config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    config = load_config(config_path)

    if args.camera not in config["cameras"]:
        sys.exit(f"[ERREUR] Camera '{args.camera}' inconnue.")

    # --- Points de référence ---
    ref_data = load_ref_points(args.ref_file)
    if args.camera not in ref_data:
        sys.exit(f"[ERREUR] Pas de points de reference pour {args.camera} dans {args.ref_file}")

    ref_labels = [p["label"] for p in ref_data[args.camera]]
    ref_metres = [p["metres"] for p in ref_data[args.camera]]
    n_points = len(ref_labels)

    ok_ref, ref_msg = validate_ref_points_metres(ref_metres)
    if not ok_ref:
        sys.exit(f"[ERREUR] {ref_msg}")

    if n_points < MIN_POINTS:
        sys.exit(f"[ERREUR] Minimum {MIN_POINTS} points requis, {n_points} fournis.")

    selected_video_arg = args.hd_video if args.hd_video else args.video

    print(f"\n{'='*70}")
    print(f"  CALIBRATION : {args.camera}")
    print(f"  Video source (arg) : {selected_video_arg if selected_video_arg else '[auto]'}")
    print(f"  Points   : {n_points}")
    print(f"{'='*70}")

    # --- Source de calibration ---
    hd_source = resolve_hd_video_path(selected_video_arg, cam_id=args.camera)
    if not hd_source or not os.path.isfile(hd_source):
        hint = selected_video_arg if selected_video_arg else f"auto ({args.camera})"
        sys.exit(f"[ERREUR] Video source introuvable pour '{hint}'.")

    # --- Récupérer une frame source ---
    if args.frame is not None:
        print(f"\n  Frame demandee : {args.frame}")
        frame_hd = grab_frame_hd(hd_source, frame_number=args.frame)
    else:
        print(f"\n  Parcours interactif pour choisir la frame...")
        frame_hd, chosen_frame = browse_video_for_frame(hd_source)
        print(f"  Frame choisie : {chosen_frame}")

    hd_h, hd_w = frame_hd.shape[:2]
    print(f"\n  Resolution source : {hd_w}x{hd_h}")

    # --- Résolution substream (cible pour inférence) ---
    ar_fix = config.get("aspect_ratio_fix", {})
    if args.substream_resolution:
        sub_w, sub_h = args.substream_resolution
    elif ar_fix.get("enabled") and args.camera in ar_fix.get("distorted_cameras", []):
        sub_w, sub_h = ar_fix["corrected_resolution"]
        print(f"  Resolution substream (fix ratio) : {sub_w}x{sub_h}")
    else:
        video_res = config.get("video", {}).get("resolution")
        if video_res:
            sub_w, sub_h = video_res[0], video_res[1]
        else:
            sys.exit("[ERREUR] Impossible de determiner la resolution substream. Utilisez --substream-resolution.")
        print(f"  Resolution substream (depuis config.video) : {sub_w}x{sub_h}")

    # --- Afficher les points à cliquer ---
    print(f"\n  Points a cliquer (dans cet ordre) :")
    for i, (label, (mx, my)) in enumerate(zip(ref_labels, ref_metres)):
        desc = ref_data[args.camera][i].get("description", "")
        print(f"    {i+1}. {label} : ({mx:.3f}, {my:.3f}) m  ({desc})")

    # --- Collecte clics en HD ---
    pts_pixels_hd = collect_points_hd(
        frame_hd,
        f"Calibration HD {args.camera} - {hd_w}x{hd_h}",
        ref_labels,
        n_points,
        ref_coords_metres=ref_metres,
    )

    if len(pts_pixels_hd) < MIN_POINTS:
        sys.exit(f"[ERREUR] Seulement {len(pts_pixels_hd)} points collectes.")

    # --- Calcul homographie HD ---
    pts_px_hd = np.array(pts_pixels_hd, dtype=np.float32)
    pts_m = np.array(ref_metres, dtype=np.float32)

    H_hd, mask, diag_hd = compute_homography(pts_px_hd, pts_m, use_ransac=(n_points >= 5))
    if H_hd is None:
        sys.exit(f"[ERREUR] {diag_hd.get('error')}")

    print_diagnostics(diag_hd, ref_labels, context=f"(HD {hd_w}x{hd_h})")

    # --- Visualisation reprojection sur HD ---
    vis = draw_reprojection_on_image(frame_hd, pts_px_hd, H_hd, pts_m, ref_labels)
    cv2.imshow("Reprojection sur frame HD (touche pour continuer)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- Conversion vers substream ---
    H_substream = convert_homography_to_substream(
        H_hd,
        (hd_w, hd_h),
        (sub_w, sub_h),
    )

    # Vérifier la qualité après conversion : reprojection des points substream équivalents
    # pts_substream équivalents = pts_hd / scale
    sx = hd_w / sub_w
    sy = hd_h / sub_h
    pts_px_sub_equivalent = pts_px_hd / np.array([sx, sy], dtype=np.float32)

    _, _, diag_sub = compute_homography(pts_px_sub_equivalent, pts_m, use_ransac=False)
    # Vérif directe avec H_substream :
    pts_sub_h = pts_px_sub_equivalent.reshape(-1, 1, 2).astype(np.float32)
    pts_reproj_from_sub = cv2.perspectiveTransform(pts_sub_h, H_substream).reshape(-1, 2)
    errors_sub_cm = np.linalg.norm(pts_reproj_from_sub - pts_m, axis=1) * 100

    print(f"\n  Verification conversion HD -> substream ({sub_w}x{sub_h}) :")
    print(f"    Erreur moyenne avec H_substream : {errors_sub_cm.mean():.3f} cm")
    print(f"    Erreur max avec H_substream     : {errors_sub_cm.max():.3f} cm")
    print(f"    (Ces erreurs doivent etre quasi-identiques au HD, sinon bug)")

    # --- Confirmation ---
    if diag_hd['mean_error_cm'] > 20:
        print(f"\n  [ATTENTION] Erreur HD elevee ({diag_hd['mean_error_cm']:.1f} cm).")
        confirm = input("  Sauvegarder quand meme ? (o/n) : ").strip().lower()
        if confirm != 'o':
            sys.exit("Calibration annulee.")

    # --- Sauvegarde dans config.yaml ---
    if "homographie" not in config:
        config["homographie"] = {}

    # On stocke les DEUX matrices, mais c'est H_substream qui sera utilisee
    # par le pipeline d'inference.
    config["homographie"][f"{args.camera}_matrix"] = H_substream.tolist()  # matrice PRINCIPALE (substream)

    config["homographie"][args.camera] = {
        "matrix": H_substream.tolist(),
        "matrix_hd": H_hd.tolist(),
        "src_points_px_hd": pts_pixels_hd,
        "dst_points_metres": pts_m.tolist(),
        "ref_labels": ref_labels,
        "n_points": n_points,
        "hd_resolution": [hd_w, hd_h],
        "substream_resolution": [sub_w, sub_h],
        "mean_error_cm": round(diag_hd['mean_error_cm'], 2),
        "max_error_cm": round(diag_hd['max_error_cm'], 2),
        "method": diag_hd['method'],
        "calibrated_at": datetime.now().isoformat(),
        "calibration_source": os.path.basename(hd_source),
    }

    save_config(config_path, config)

    print(f"\n  [OK] Calibration sauvegardee :")
    print(f"       homographie.{args.camera}_matrix        <- matrice SUBSTREAM (pour inference)")
    print(f"       homographie.{args.camera}.matrix_hd     <- matrice HD (pour reference)")
    print(f"       Erreur moyenne : {diag_hd['mean_error_cm']:.2f} cm")


if __name__ == "__main__":
    main()

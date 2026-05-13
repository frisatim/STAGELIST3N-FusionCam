"""
Outil interactif de calibration par homographie (Phase 3.1).

Methode : Coordonnees Absolues.
Workflow :
  1. Choisir une camera et une source (RTSP live ou video locale depuis config)
  2. Capturer une frame depuis la source
  3. Cliquer les 4 coins de la zone interdite sur la frame camera (pixels)
  4. Les coordonnees metres sont lues directement depuis config.yaml (zone_1)
  5. Calculer la matrice d'homographie 3x3 et la sauvegarder dans config.yaml
"""

import os
import sys
import yaml
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
COLOR_POINT = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
RADIUS = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def fix_aspect_ratio(frame: np.ndarray, cam_id: str, config: dict) -> np.ndarray:
    """Correctif ratio d'aspect pour caméras Dahua dont le sub-stream 704x576
    est déformé horizontalement par le firmware. Redimensionne en 768x576 (4:3)
    pour restaurer la géométrie réelle avant tout calcul d'homographie ou
    de projection spatiale."""
    ar_fix = config.get("aspect_ratio_fix", {})
    if not ar_fix.get("enabled", False):
        return frame
    if cam_id not in ar_fix.get("distorted_cameras", []):
        return frame
    target_w, target_h = ar_fix["corrected_resolution"]
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    print(f"  [FIX RATIO] {cam_id} : {w}x{h} -> {target_w}x{target_h} (correction 4:3)")
    return cv2.resize(frame, (target_w, target_h))


def grab_frame(source: str) -> np.ndarray:
    """Capture la premiere frame valide depuis un flux RTSP ou un fichier video."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERREUR] Impossible d'ouvrir la source : {source}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        sys.exit(f"[ERREUR] Aucune frame recue depuis : {source}")
    return frame


def collect_points(image: np.ndarray, window_name: str, labels: list[str], n_points: int = 4) -> list[list[int]]:
    """Affiche une image et collecte n_points clics souris avec labels."""
    points: list[list[int]] = []
    display = image.copy()

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= n_points:
            return
        points.append([x, y])
        idx = len(points)
        label = labels[idx - 1] if idx <= len(labels) else str(idx)
        cv2.circle(display, (x, y), RADIUS, COLOR_POINT, -1)
        cv2.putText(display, label, (x + 10, y - 10), FONT, 0.6, COLOR_TEXT, 2)
        cv2.imshow(window_name, display)
        print(f"  Point {idx}/{n_points} ({label}) : ({x}, {y})")

    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, on_click)
    print(f"\n>> Cliquez sur {n_points} coins de la zone dans \"{window_name}\"")
    print("   Ordre : Haut-Gauche, Haut-Droite, Bas-Droite, Bas-Gauche")
    print("   Appuyez sur une touche apres le dernier clic.\n")

    while True:
        key = cv2.waitKey(100) & 0xFF
        if len(points) >= n_points and key != 255:
            break

    cv2.destroyWindow(window_name)
    return points


def resolve_path(relative_path: str) -> str:
    """Resout un chemin relatif par rapport a la racine du projet."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    if not os.path.isfile(config_path):
        sys.exit(f"[ERREUR] config.yaml introuvable : {config_path}")

    config = load_config(config_path)
    cameras = config["cameras"]

    # --- Banniere ---
    print("=" * 60)
    print("  CALIBRATION PAR COORDONNEES ABSOLUES  (Phase 3.1)")
    print("=" * 60)

    # --- Zone interdite (coordonnees metres) ---
    zone_key = "zone_1"
    zone_data = config["zones_interdites"].get(zone_key)
    if not zone_data:
        sys.exit(f"[ERREUR] Zone '{zone_key}' non trouvee dans config.yaml.")

    coords_metres = zone_data["coordonnees_metres"]
    pts_plan_metres = np.array(coords_metres, dtype=np.float32)
    print(f"\nZone '{zone_data['nom']}' - {len(coords_metres)} points en metres :")
    corner_labels = ["HG", "HD", "BD", "BG"]
    for i, (pt, label) in enumerate(zip(coords_metres, corner_labels)):
        print(f"  {label} : ({pt[0]:.3f}, {pt[1]:.3f}) m")

    # --- Selection de la camera ---
    print("\nCameras disponibles :")
    for cam_id, cam_data in cameras.items():
        status = "ON" if cam_data.get("enabled") else "OFF"
        print(f"  {cam_id}  {cam_data['name']}  [{status}]")

    cam_choice = input("\nQuelle camera voulez-vous calibrer ? (ex: cam_01) : ").strip()
    if cam_choice not in cameras:
        sys.exit(f"[ERREUR] Camera '{cam_choice}' inconnue.")

    cam_cfg = cameras[cam_choice]

    # --- Choix de la source ---
    print(f"\nSource pour {cam_choice} :")
    print(f"  1 - RTSP en direct  ({cam_cfg['rtsp_url']})")
    print(f"  2 - Video locale    ({cam_cfg.get('video_path', 'non defini')})")
    source_choice = input("Votre choix (1 ou 2) : ").strip()

    if source_choice == "1":
        source = cam_cfg["rtsp_url"]
    elif source_choice == "2":
        video_path = cam_cfg.get("video_path")
        if not video_path:
            sys.exit(f"[ERREUR] Pas de 'video_path' dans le config pour {cam_choice}.")
        source = resolve_path(video_path)
        if not os.path.isfile(source):
            sys.exit(f"[ERREUR] Fichier introuvable : {source}")
    else:
        sys.exit("[ERREUR] Choix invalide. Tapez 1 ou 2.")

    print(f"\nSource selectionnee : {source}")

    # --- Capture frame ---
    frame = grab_frame(source)
    print(f"Frame brute capturee : {frame.shape[1]}x{frame.shape[0]}")

    # --- Correctif ratio d'aspect (AVANT affichage et clics) ---
    frame = fix_aspect_ratio(frame, cam_choice, config)
    print(f"Frame utilisee pour calibration : {frame.shape[1]}x{frame.shape[0]}")

    # === Etape unique : 4 clics sur la camera ===
    print(f"\n>> Cliquez sur les 4 coins de la zone '{zone_data['nom']}' sur la camera.")
    print("   Meme ordre que les coordonnees metres : HG, HD, BD, BG")
    pts_camera = collect_points(frame, f"Calibration {cam_choice} - Cliquez les 4 coins de la zone", corner_labels)
    print(f"\nPoints camera (px) : {pts_camera}")

    # === Homographie : pixels camera -> metres ===
    src = np.array(pts_camera, dtype=np.float32)
    matrix, status = cv2.findHomography(src, pts_plan_metres)
    if matrix is None:
        sys.exit("[ERREUR] Calcul de l'homographie echoue (points degeneres ?).")

    print("\n" + "=" * 60)
    print(f"  MATRICE D'HOMOGRAPHIE — {cam_choice}")
    print("=" * 60)
    for row in matrix:
        print(f"  [{row[0]:+.10f}  {row[1]:+.10f}  {row[2]:+.10f}]")

    # === Sauvegarde config.yaml ===
    matrix_list = matrix.tolist()

    if not config.get("homographie"):
        config["homographie"] = {}

    matrix_key = cam_choice + "_matrix"
    config["homographie"][matrix_key] = matrix_list
    config["homographie"][cam_choice] = {
        "matrix": matrix_list,
        "src_points_px": pts_camera,
        "dst_points_metres": pts_plan_metres.tolist(),
    }

    save_config(config_path, config)

    print(f"\n[OK] Calibration directe reussie !")
    print(f"     Sauvegarde dans config.yaml :")
    print(f"     homographie.{matrix_key}")
    print(f"     homographie.{cam_choice}")


if __name__ == "__main__":
    main()

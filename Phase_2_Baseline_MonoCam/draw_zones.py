"""
Phase 2 - Outil de tracage interactif des zones interdites (pixels).

Ouvre la premiere frame d'une video camera et permet de tracer un polygone
a la souris. Les coordonnees sont sauvegardees dans zones_config.json,
associees a l'ID de la camera.

Logique de dessin reutilisee depuis ExperimentsCAS/zone_drawer.py.

Usage :
  python draw_zones.py --camera cam_03              # video auto depuis config.yaml
  python draw_zones.py --camera cam_03 --video v.mp4  # video specifique
  python draw_zones.py --video video.mp4 --id cam_99  # source libre + id custom
  python draw_zones.py --list                        # afficher les zones deja sauvees

Controles :
  Clic gauche : ajouter un point
  U           : annuler le dernier point
  C           : effacer tous les points
  S / Entree  : valider et sauvegarder
  Q / Echap   : quitter sans sauvegarder
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Chemins ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "Phase_3_Fusion_MultiCam" / "config.yaml"
ZONES_JSON = SCRIPT_DIR / "zones_config.json"

# ── Resolution cible 16:9 (doit correspondre a evaluate_trd.py) ─────────────
TARGET_W = 1280
TARGET_H = 720

# ── Couleurs (BGR) ───────────────────────────────────────────────────────────
COLOR_POINT = (0, 255, 255)       # jaune
COLOR_LINE_OPEN = (255, 255, 0)   # cyan
COLOR_POLY_FILL = (0, 180, 255)   # orange
COLOR_POLY_BORDER = (0, 180, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_EXISTING = (0, 255, 0)      # vert pour zone existante


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURE FRAME STABLE (repris de ExperimentsCAS/zone_drawer.py)
# ══════════════════════════════════════════════════════════════════════════════

def read_stable_frame(cap: cv2.VideoCapture, timeout_s: float = 8.0) -> np.ndarray | None:
    """Lit des frames jusqu'a obtenir une image avec un contraste suffisant
    (evite les frames noires/corrompues en debut de flux RTSP)."""
    deadline = time.time() + timeout_s
    last_valid = None

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        last_valid = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray)) >= 12.0:
            return frame

    return last_valid


# ══════════════════════════════════════════════════════════════════════════════
#  GESTION JSON
# ══════════════════════════════════════════════════════════════════════════════

def load_zones_json() -> dict:
    """Charge zones_config.json ou retourne un dict vide si inexistant."""
    if ZONES_JSON.exists():
        with open(ZONES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_zones_json(data: dict) -> None:
    """Sauvegarde zones_config.json avec indentation lisible."""
    with open(ZONES_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
#  RESOLUTION VIDEO DEPUIS CONFIG.YAML
# ══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_video(config: dict, cam_id: str) -> str | None:
    """Resout le chemin video pour une camera depuis config.yaml."""
    cameras = config.get("cameras", {})
    cam_cfg = cameras.get(cam_id)
    if not cam_cfg:
        return None

    video_rel = cam_cfg.get("video_path", "")
    if video_rel:
        for base in (CONFIG_PATH.parent, PROJECT_ROOT):
            resolved = (base / video_rel).resolve()
            if resolved.exists():
                return str(resolved)

    # Fallback : glob dans recordings/
    recordings = PROJECT_ROOT / "recordings" / "recordings"
    if recordings.exists():
        ip = cam_cfg.get("ip", "")
        if ip:
            for ext in ("mp4", "mkv"):
                matches = sorted(recordings.glob(f"*{ip.replace('.', '_')}*.{ext}"))
                if not matches:
                    matches = sorted(recordings.glob(f"*{ip.replace('.', '.')}*.{ext}"))
                if matches:
                    return str(matches[-1])
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  DESSIN INTERACTIF (coeur repris de ExperimentsCAS/zone_drawer.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_drawer(source: str, cam_id: str) -> None:
    """Ouvre la source video, capture une frame, et lance l'interface de tracage."""

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERREUR] Impossible d'ouvrir : {source}")

    print("  Lecture d'une frame stable...")
    frame = read_stable_frame(cap)
    cap.release()
    if frame is None:
        sys.exit("[ERREUR] Aucune frame lisible depuis la source.")

    src_h, src_w = frame.shape[:2]
    print(f"  Frame brute : {src_w}x{src_h}")

    # Redimensionner en 16:9 (meme resolution que evaluate_trd.py)
    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    h, w = TARGET_H, TARGET_W
    print(f"  Affichage 16:9 : {w}x{h} (les coordonnees seront en 16:9)")

    # Charger la zone existante si elle existe (pour l'afficher en fond)
    zones_data = load_zones_json()
    existing_zone = zones_data.get(cam_id)

    points: list[list[int]] = []
    win_name = f"Tracer zone 16:9 - {cam_id}"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"  Point {len(points)} : [{x}, {y}]")

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, TARGET_W, TARGET_H)
    cv2.setMouseCallback(win_name, on_mouse)

    print(f"\n  Clic gauche = ajouter point | U = annuler | C = effacer")
    print(f"  S ou Entree = sauvegarder   | Q ou Echap = quitter\n")

    while True:
        view = frame.copy()

        # Afficher la zone existante en vert si elle existe
        if existing_zone and len(existing_zone) >= 3:
            old_pts = np.array(existing_zone, dtype=np.int32)
            overlay = view.copy()
            cv2.fillPoly(overlay, [old_pts], COLOR_EXISTING)
            cv2.addWeighted(overlay, 0.15, view, 0.85, 0, view)
            cv2.polylines(view, [old_pts], isClosed=True, color=COLOR_EXISTING, thickness=1)
            cv2.putText(view, "Zone actuelle (vert)", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EXISTING, 1)

        # Dessiner les points en cours
        for px, py in points:
            cv2.circle(view, (px, py), 5, COLOR_POINT, -1)

        # Lignes entre les points
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(view, [pts], isClosed=False, color=COLOR_LINE_OPEN, thickness=2)

        # Polygone ferme + remplissage semi-transparent (3+ points)
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            overlay = view.copy()
            cv2.fillPoly(overlay, [pts], COLOR_POLY_FILL)
            cv2.addWeighted(overlay, 0.25, view, 0.75, 0, view)
            cv2.polylines(view, [pts], isClosed=True, color=COLOR_POLY_BORDER, thickness=2)

        # HUD
        cv2.putText(
            view,
            f"{cam_id} | Points: {len(points)} | Clic:ajouter U:annuler C:effacer S:sauver Q:quitter",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2,
        )

        cv2.imshow(win_name, view)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("u") and points:
            removed = points.pop()
            print(f"  Annule : [{removed[0]}, {removed[1]}] (reste {len(points)})")

        elif key == ord("c"):
            points.clear()
            print("  Tous les points effaces.")

        elif key == ord("s") or key == 13:  # S ou Entree
            if len(points) < 3:
                print("  [!] Il faut au moins 3 points pour former un polygone.")
                continue

            # Sauvegarder dans zones_config.json (merge avec l'existant)
            zones_data = load_zones_json()
            zones_data[cam_id] = points
            save_zones_json(zones_data)

            print(f"\n  [OK] Zone sauvegardee dans {ZONES_JSON.name}")
            print(f"       Camera : {cam_id}")
            print(f"       Points : {len(points)}")
            for i, pt in enumerate(points):
                print(f"         P{i+1} : [{pt[0]}, {pt[1]}]")

            # Afficher aussi le contenu complet du JSON
            print(f"\n  Contenu actuel de {ZONES_JSON.name} :")
            for cid, pts in zones_data.items():
                print(f"    {cid} : {len(pts)} points")
            break

        elif key == ord("q") or key == 27:  # Q ou Echap
            print("  Annule. Aucune sauvegarde.")
            break

    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE DES ZONES EXISTANTES
# ══════════════════════════════════════════════════════════════════════════════

def list_zones() -> None:
    """Affiche toutes les zones deja sauvegardees dans zones_config.json."""
    zones = load_zones_json()
    if not zones:
        print(f"  Aucune zone sauvegardee ({ZONES_JSON}).")
        return

    print(f"\n  Zones dans {ZONES_JSON.name} :")
    print("  " + "-" * 50)
    for cam_id, pts in zones.items():
        print(f"  {cam_id} : {len(pts)} points")
        for i, pt in enumerate(pts):
            print(f"    P{i+1} : [{pt[0]}, {pt[1]}]")
    print("  " + "-" * 50)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Tracage interactif de zones interdites par camera (Phase 2)")
    parser.add_argument("--camera", type=str, default=None,
                        help="ID camera (ex: cam_03). Resout la video depuis config.yaml.")
    parser.add_argument("--video", type=str, default=None,
                        help="Chemin video ou URL RTSP (override la resolution auto)")
    parser.add_argument("--id", type=str, default=None,
                        help="ID camera custom (utilise avec --video sans --camera)")
    parser.add_argument("--list", action="store_true",
                        help="Afficher les zones deja sauvegardees")
    args = parser.parse_args()

    print("=" * 60)
    print("  DRAW ZONES - Tracage de zones interdites (Phase 2)")
    print("=" * 60)

    if args.list:
        list_zones()
        return

    config = load_config()

    # Determiner l'ID camera
    if args.camera:
        cam_id = args.camera
        cameras = config.get("cameras", {})
        if cam_id not in cameras:
            available = list(cameras.keys())
            sys.exit(f"[ERREUR] Camera '{cam_id}' inconnue.\n  Disponibles : {available}")
        cam_name = cameras[cam_id].get("name", cam_id)
        print(f"\n  Camera : {cam_id} ({cam_name})")
    elif args.id:
        cam_id = args.id
        print(f"\n  Camera (custom) : {cam_id}")
    else:
        sys.exit("[ERREUR] Specifiez --camera cam_XX ou --id mon_id.\n"
                 "  Exemple : python draw_zones.py --camera cam_03")

    # Determiner la source video
    if args.video:
        source = args.video
    elif args.camera and config:
        source = resolve_video(config, cam_id)
        if not source:
            sys.exit(f"[ERREUR] Aucune video trouvee pour {cam_id}. Utilise --video.")
        print(f"  Video  : {Path(source).name}")
    else:
        sys.exit("[ERREUR] Specifiez --video ou utilisez --camera (resout depuis config).")

    if not Path(source).exists() and not source.startswith("rtsp"):
        sys.exit(f"[ERREUR] Source introuvable : {source}")

    # Zone existante ?
    zones = load_zones_json()
    if cam_id in zones:
        n = len(zones[cam_id])
        print(f"  [INFO] Zone existante pour {cam_id} ({n} points) : sera ecrasee si tu sauvegardes.")

    print()
    run_drawer(source, cam_id)


if __name__ == "__main__":
    main()

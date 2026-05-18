"""
Outil d'annotation de vérité terrain pour surveillance multi-caméras.

Produit TROIS sorties séparées :
  - gt_people.json                      : violations de zone (TRD) - racine projet
  - <output-dir>/gt_objects_tad.json    : apparitions d'objets interdits pour le TAD
  - <output-dir>/images/ + labels/      : images + labels YOLO pour fine-tuning

Le dossier de sortie est configurable via --output-dir (défaut: dataset_objets_HD).

Raccourcis :
  V .......... Marquer violation personne       -> gt_people.json
  O .......... Marquer apparition objet (TAD)   -> gt_objects_tad.json
  B .......... Dessiner BBox objet (YOLO)       -> dataset/images + labels
  N .......... Image suivante (mode images)
  P .......... Image precedente (mode images)
  S .......... Passer l'image sans annotation (mode images)
  U .......... Annuler dernière annotation
  Suppr ...... Supprimer annotation proche
  L .......... Lister les annotations (console)
  Espace ..... Lecture / Pause
  Flèches .... ±1 frame
  Q / Echap .. Quitter et sauvegarder
"""

import argparse
import cv2
import json
import numpy as np
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

# ── Dossier de sortie par défaut ─────────────────────────────────────────────
# Modifie cette variable OU passe --output-dir en ligne de commande.
BASE_OUTPUT_DIR = "dataset_objets_HD"

# ── Chemins ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings" / "recordings"
GT_PEOPLE_PATH = PROJECT_ROOT / "gt_people.json"
DEFAULT_PHASE3_CONFIG_PATH = PROJECT_ROOT / "Phase_3_Fusion_MultiCam" / "config.yaml"

# Ces chemins sont recalculés dans main() après parsing des arguments.
# Valeurs par défaut (écrasées par --output-dir).
_OUTPUT_ROOT = PROJECT_ROOT / BASE_OUTPUT_DIR
GT_OBJECTS_TAD_PATH = _OUTPUT_ROOT / "gt_objects_tad.json"
DATASET_IMAGES_DIR = _OUTPUT_ROOT / "images"
DATASET_LABELS_DIR = _OUTPUT_ROOT / "labels"

# ── Classes YOLO ─────────────────────────────────────────────────────────────
# Touche clavier -> (class_id, nom).
# Par defaut, l'ordre suit dataset_objets_HD/data.yaml.
DEFAULT_YOLO_CLASSES = [
    (ord("1"), 0, "marteau"),
    (ord("2"), 1, "niveau a bulle"),
    (ord("3"), 2, "scie"),
    (ord("4"), 3, "verre"),
    (ord("5"), 4, "perceuse"),
    (ord("6"), 5, "bouteille"),
    (ord("7"), 6, "pince"),
    (ord("8"), 7, "cutter"),
    (ord("9"), 8, "metre"),
    (ord("t"), 9, "tournevis"),
    (ord("k"), 10, "cle allen"),
    (ord("p"), 11, "personne"),
]
YOLO_CLASSES = DEFAULT_YOLO_CLASSES.copy()

# ── Objectifs du plan de recherche ───────────────────────────────────────────
GOAL_VIOLATIONS = 50
GOAL_OBJECTS_MIN = 200
GOAL_OBJECTS_MAX = 300

# ── Codes clavier OpenCV (cross-platform) ────────────────────────────────────
KEY_SPACE = 32
KEY_ESC = 27
KEY_LEFT = 81              # Linux
KEY_RIGHT = 83             # Linux
KEY_LEFT_WIN = 2424832     # Windows
KEY_RIGHT_WIN = 2555904    # Windows
KEY_DELETE = 255            # Linux
KEY_DELETE_WIN = 3014656    # Windows
KEY_Q = ord("q")
KEY_V = ord("v")
KEY_O = ord("o")
KEY_U = ord("u")
KEY_L = ord("l")
KEY_B = ord("b")
KEY_N = ord("n")
KEY_P = ord("p")
KEY_S = ord("s")

PROXIMITY_THRESHOLD = 15
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".ts"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Utilitaires ──────────────────────────────────────────────────────────────

def format_time_ms(timestamp_ms: float) -> str:
    """Formate un timestamp en MM:SS.mmm."""
    total_s = timestamp_ms / 1000.0
    mins, secs = divmod(total_s, 60)
    whole_s = int(secs)
    millis = int(round((secs - whole_s) * 1000))
    return f"{int(mins):02d}:{whole_s:02d}.{millis:03d}"


def extract_camera_id(filename: str) -> str:
    """Retourne un identifiant unique par vidéo basé sur le stem complet.

    Exemples :
      Camera_1_2.2_20260303_092034.mp4 -> cam_01_20260303_092034
      Camera_1_2.2_record2.mp4         -> cam_01_record2
      Camera_7_2.11_record2.mp4        -> cam_07_record2
    """
    stem = Path(filename).stem
    # Extraire le numéro de caméra
    m = re.search(r"[Cc]am(?:era)?[\s_]*(\d+)", stem)
    if not m:
        return stem.replace(" ", "_").lower()

    cam_num = int(m.group(1))

    # Extraire un suffixe discriminant après le pattern "X.XX_"
    # Ex: "Camera_1_2.2_record2" -> "record2"
    # Ex: "Camera_1_2.2_20260303_092034" -> "20260303_092034"
    suffix_match = re.search(r"[Cc]am(?:era)?[\s_]*\d+[\s_]+\d+\.\d+[\s_]+(.*)", stem)
    if suffix_match and suffix_match.group(1):
        suffix = suffix_match.group(1).replace(" ", "_").lower()
        return f"cam_{cam_num:02d}_{suffix}"

    return f"cam_{cam_num:02d}"


def extract_base_camera_id(cam_id: str) -> str:
    """Return cam_XX from cam_XX_suffix or a video filename-derived ID."""
    match = re.search(r"cam_(\d{2})", cam_id)
    if match:
        return f"cam_{int(match.group(1)):02d}"
    match = re.search(r"[Cc]am(?:era)?[\s_]*(\d+)", cam_id)
    if match:
        return f"cam_{int(match.group(1)):02d}"
    return cam_id


def list_videos(recordings_dir: Path,
                pattern: str = "",
                recursive: bool = True) -> list[dict]:
    """Liste les vidéos annotables depuis un dossier recordings.

    - pattern=""  -> toutes les vidéos
    - recursive=True -> scan récursif des sous-dossiers
    """
    if not recordings_dir.exists():
        print(f"[ERREUR] Dossier introuvable : {recordings_dir}")
        sys.exit(1)

    walker = recordings_dir.rglob("*") if recursive else recordings_dir.iterdir()
    videos = []
    seen: set[Path] = set()
    pattern_l = pattern.lower().strip()

    for f in sorted(walker):
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if pattern_l and pattern_l not in f.stem.lower():
            continue

        f_resolved = f.resolve()
        if f_resolved in seen:
            continue
        seen.add(f_resolved)

        videos.append({
            "path": str(f_resolved),
            "filename": f.name,
            "id_camera": extract_camera_id(f.name),
        })

    if not videos:
        if pattern_l:
            print(f"[ERREUR] Aucun fichier vidéo '{pattern}' trouvé dans {recordings_dir}")
        else:
            print(f"[ERREUR] Aucune vidéo trouvée dans {recordings_dir}")
        sys.exit(1)
    return videos


def list_images(images_dir: Path,
                pattern: str = "",
                recursive: bool = True) -> list[Path]:
    """Liste les images annotables depuis un dossier."""
    if not images_dir.exists():
        print(f"[ERREUR] Dossier images introuvable : {images_dir}")
        sys.exit(1)

    walker = images_dir.rglob("*") if recursive else images_dir.iterdir()
    pattern_l = pattern.lower().strip()
    images = []

    for f in sorted(walker):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if pattern_l and pattern_l not in f.stem.lower():
            continue
        images.append(f.resolve())

    if not images:
        if pattern_l:
            print(f"[ERREUR] Aucune image '{pattern}' trouvée dans {images_dir}")
        else:
            print(f"[ERREUR] Aucune image trouvée dans {images_dir}")
        sys.exit(1)
    return images


def load_yolo_classes_from_yaml(data_yaml: Path) -> list[tuple[int, int, str]]:
    """Charge l'ordre des classes depuis un data.yaml YOLO.

    Les touches restent ergonomiques: 1..9 puis T/K/P si besoin.
    """
    if not data_yaml.exists():
        return DEFAULT_YOLO_CLASSES.copy()

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names")
    if isinstance(names, dict):
        ordered_names = [str(names[i]) for i in sorted(names)]
    elif isinstance(names, list):
        ordered_names = [str(name) for name in names]
    else:
        return DEFAULT_YOLO_CLASSES.copy()

    keys = list("123456789") + ["t", "k", "p", "a", "c", "m", "v", "x", "z"]
    classes = []
    for class_id, name in enumerate(ordered_names):
        if class_id >= len(keys):
            raise ValueError(
                f"Trop de classes dans {data_yaml}: {len(ordered_names)} "
                f"(max supporte par raccourcis: {len(keys)})"
            )
        classes.append((ord(keys[class_id]), class_id, name))
    return classes


def apply_yolo_classes(classes: list[tuple[int, int, str]]) -> None:
    global YOLO_CLASSES
    YOLO_CLASSES = classes


def write_data_yaml(output_root: Path) -> None:
    """Ecrit un data.yaml cohérent avec les classes actives."""
    output_root.mkdir(parents=True, exist_ok=True)
    names = {cid: name for _, cid, name in YOLO_CLASSES}
    payload = {
        "path": ".",
        "train": "images",
        "val": "images",
        "nc": len(names),
        "names": names,
    }
    with open(output_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


# ── Zones interdites Phase 3 ────────────────────────────────────────────────

def load_phase3_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[WARN] Config Phase 3 introuvable : {config_path}")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_cli_path(path_arg: str, fallback_base: Path) -> Path:
    """Resolve a CLI path relative to cwd first, then fallback_base.

    PowerShell users usually pass paths relative to the directory where they run
    the command. Some defaults in this script are project-root-relative, so this
    helper supports both without forcing absolute paths.
    """
    path = Path(path_arg)
    if path.is_absolute():
        return path.resolve()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (fallback_base / path).resolve()


def get_homography_matrix(config: dict, cam_id: str) -> np.ndarray | None:
    """Load pixels->metres homography for cam_id from Phase 3 config."""
    homo = config.get("homographie", {})

    nested = homo.get(cam_id)
    if isinstance(nested, dict) and nested.get("matrix"):
        return np.array(nested["matrix"], dtype=np.float32)

    flat = homo.get(f"{cam_id}_matrix")
    if flat:
        return np.array(flat, dtype=np.float32)

    return None


def zones_for_camera(config: dict, cam_id: str) -> list[dict]:
    zones = []
    for zone_id, zone_data in config.get("zones_interdites", {}).items():
        cam_ids = zone_data.get("cameras_concernees") or []
        coords = zone_data.get("coordonnees_metres") or []
        if cam_id in cam_ids and len(coords) >= 3:
            zones.append({
                "zone_id": zone_id,
                "name": zone_data.get("nom", zone_id),
                "coords_m": coords,
            })
    return zones


def project_zone_to_pixels(zone: dict, homography_px_to_m: np.ndarray) -> np.ndarray | None:
    """Project a floor-plane zone polygon from metres to image pixels."""
    try:
        inv_h = np.linalg.inv(homography_px_to_m)
    except np.linalg.LinAlgError:
        return None

    pts_m = np.array(zone["coords_m"], dtype=np.float32).reshape((-1, 1, 2))
    pts_px = cv2.perspectiveTransform(pts_m, inv_h)
    pts_px = pts_px.reshape((-1, 2))
    if not np.isfinite(pts_px).all():
        return None
    return np.round(pts_px).astype(np.int32).reshape((-1, 1, 2))


def build_projected_zones_for_camera(config: dict, cam_id: str) -> list[dict]:
    homography = get_homography_matrix(config, cam_id)
    if homography is None:
        return []

    projected = []
    for zone in zones_for_camera(config, cam_id):
        pts_px = project_zone_to_pixels(zone, homography)
        if pts_px is None:
            continue
        projected.append({
            "zone_id": zone["zone_id"],
            "name": zone["name"],
            "pts_px": pts_px,
        })
    return projected


def draw_forbidden_zones(frame: np.ndarray, projected_zones: list[dict]) -> None:
    """Draw projected forbidden zones on the video frame in-place."""
    if not projected_zones:
        return

    overlay = frame.copy()
    for zone in projected_zones:
        pts = zone["pts_px"]
        cv2.fillPoly(overlay, [pts], (0, 0, 130))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        center = pts.reshape((-1, 2)).mean(axis=0).astype(int)
        cv2.putText(
            frame,
            zone["zone_id"],
            (int(center[0]) - 24, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)


# ── Chargement / sauvegarde des deux GT ──────────────────────────────────────

def load_json(path: Path) -> list[dict]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_json(data: list[dict], path: Path, label: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [SAUVEGARDE] {len(data)} entrées -> {path.name}  ({label})")


def load_gt_people() -> list[dict]:
    return load_json(GT_PEOPLE_PATH)


def load_gt_objects_tad() -> list[dict]:
    return load_json(GT_OBJECTS_TAD_PATH)


def save_gt_people(data: list[dict]) -> None:
    save_json(data, GT_PEOPLE_PATH, "violations personnes")


def save_gt_objects_tad(data: list[dict]) -> None:
    save_json(data, GT_OBJECTS_TAD_PATH, "objets TAD")


def next_event_id(gt_list: list[dict]) -> int:
    if not gt_list:
        return 1
    return max(e.get("id_evenement", 0) for e in gt_list) + 1


# ── Sélection vidéo ─────────────────────────────────────────────────────────

def select_video(videos: list[dict]) -> dict | None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║       SÉLECTION DE LA VIDÉO                  ║")
    print("╠══════════════════════════════════════════════╣")
    for i, v in enumerate(videos, 1):
        print(f"║  {i:2d}. {v['id_camera']:>8s}  │  {v['filename']:<25s} ║")
    print("║   0. Quitter                                 ║")
    print("╚══════════════════════════════════════════════╝")
    while True:
        try:
            choice = input("\nChoix [numéro] : ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                return videos[idx]
        except (ValueError, EOFError):
            pass
        print("  Choix invalide, réessayez.")


# ── HUD ──────────────────────────────────────────────────────────────────────

PANEL_HEIGHT = 120


def draw_hud(frame, cam_id: str, frame_idx: int, fps: float,
             timestamp_ms: float, paused: bool, total_frames: int,
             gt_people: list[dict], gt_objects_tad: list[dict]):
    """Crée un panneau noir de 120px sous la vidéo avec toutes les infos HUD.

    Retourne le panneau (np.ndarray) à concaténer via np.vstack.
    La frame vidéo n'est PAS modifiée -> coordonnées BBox intactes.
    """
    h, w = frame.shape[:2]
    panel = np.zeros((PANEL_HEIGHT, w, 3), dtype=np.uint8)

    time_str = format_time_ms(timestamp_ms)
    status = "PAUSE" if paused else "LECTURE"

    # Ligne 1 (y~22) : caméra + frame + temps + FPS + status
    cv2.putText(panel, f"{cam_id}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(panel, f"Frame: {frame_idx}/{total_frames}",
                (150, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"T: {time_str}", (380, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"FPS: {fps:.1f}", (550, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    color = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(panel, status, (w - 130, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Ligne 2 (y~50) : compteurs de progression + indicateurs annotation
    n_people = len(gt_people)
    n_objects_tad = len(gt_objects_tad)
    ppl_color = (0, 255, 0) if n_people >= GOAL_VIOLATIONS else (0, 180, 255)
    obj_color = (0, 255, 0) if n_objects_tad >= GOAL_OBJECTS_MIN else (0, 180, 255)

    cv2.putText(panel, f"[V] Violations: {n_people}/{GOAL_VIOLATIONS}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ppl_color, 1)
    cv2.putText(panel, f"[O] Objets TAD: {n_objects_tad}/{GOAL_OBJECTS_MIN}-{GOAL_OBJECTS_MAX}",
                (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 1)

    # Indicateur si frame annotée (violations)
    for ann in gt_people:
        if ann["id_camera"] == cam_id and ann["trame_violation"] == frame_idx:
            cv2.rectangle(panel, (w // 2 - 100, 35), (w // 2 + 100, 58),
                          (0, 0, 180), -1)
            cv2.putText(panel, "VIOLATION MARQUEE", (w // 2 - 92, 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            break

    # Indicateur si frame annotée (objets TAD)
    for ann in gt_objects_tad:
        if ann["id_camera"] == cam_id and ann["trame_apparition"] == frame_idx:
            cv2.rectangle(panel, (w // 2 + 110, 35), (w // 2 + 270, 58),
                          (255, 0, 200), -1)
            cv2.putText(panel, "OBJET TAD", (w // 2 + 118, 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            break

    # Ligne 3 (y~78) : aide rapide
    cv2.putText(panel, "[V] Violation Personne | [O] Apparition Objet (TAD) | [B] Box Objet (YOLO) | Espace=Pause  Q=Quitter",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    # Barre de progression (y~90)
    bar_y = 90
    bar_h = 8
    cv2.rectangle(panel, (10, bar_y), (w - 10, bar_y + bar_h), (60, 60, 60), -1)
    if total_frames > 0:
        progress = frame_idx / total_frames
        bar_w = int((w - 20) * progress)
        cv2.rectangle(panel, (10, bar_y), (10 + bar_w, bar_y + bar_h),
                      (0, 180, 255), -1)

    # Marqueurs violations (rouge) sur la barre
    for ann in gt_people:
        if ann["id_camera"] == cam_id and total_frames > 0:
            x = 10 + int((w - 20) * ann["trame_violation"] / total_frames)
            cv2.line(panel, (x, bar_y - 5), (x, bar_y + bar_h + 5),
                     (0, 0, 255), 2)

    # Marqueurs objets TAD (magenta) sur la barre
    for ann in gt_objects_tad:
        if ann["id_camera"] == cam_id and total_frames > 0:
            x = 10 + int((w - 20) * ann["trame_apparition"] / total_frames)
            cv2.line(panel, (x, bar_y - 5), (x, bar_y + bar_h + 5),
                     (255, 0, 200), 2)

    return panel


def draw_flash(frame, label: str, color: tuple):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h // 2 - 30), (w, h // 2 + 30), color, -1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    tx = (w - text_size[0]) // 2
    cv2.putText(frame, label, (tx, h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def draw_image_hud(frame: np.ndarray, image_path: Path, index: int, total: int,
                   stem: str, label_count: int, auto_next: bool) -> np.ndarray:
    """HUD pour le mode annotation d'images."""
    h, w = frame.shape[:2]
    panel = np.zeros((PANEL_HEIGHT, w, 3), dtype=np.uint8)
    cv2.putText(panel, f"Image {index + 1}/{total}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(panel, image_path.name[:80], (180, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(panel, f"Labels: {label_count}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(panel, f"Stem: {stem[:60]}", (150, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    mode = "AUTO-NEXT apres BBox" if auto_next else "MULTI-BOX: N pour image suivante"
    cv2.putText(panel, mode, (w - 360, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1)
    cv2.putText(panel, "[B] bbox | [N] suivante | [P] precedente | [S] skip | [U] suppr dernier label | Q/Echap quitter",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    bar_y = 98
    cv2.rectangle(panel, (10, bar_y), (w - 10, bar_y + 8), (60, 60, 60), -1)
    if total > 1:
        progress = index / (total - 1)
        cv2.rectangle(panel, (10, bar_y), (10 + int((w - 20) * progress), bar_y + 8),
                      (0, 180, 255), -1)

    return np.vstack([frame, panel])


# ── Trackbar callback ───────────────────────────────────────────────────────

def on_trackbar(val):
    """Callback vide, la logique est dans la boucle principale."""
    pass


# ── Annotation : sélection bounding box via callback souris ──────────────────

class BBoxDrawer:
    """Dessin interactif de bounding box via callback souris.

    Remplace cv2.selectROI qui freeze la fenêtre sur Windows
    quand une trackbar est présente.
    """

    def __init__(self):
        self.drawing = False
        self.done = False
        self.cancelled = False
        self.x0 = self.y0 = 0
        self.x1 = self.y1 = 0

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x1, self.y1 = x, y
            self.done = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.cancelled = True

    def get_bbox(self) -> list[int] | None:
        """Retourne [x, y, w, h] ou None si annulé/vide."""
        if self.cancelled:
            return None
        x = min(self.x0, self.x1)
        y = min(self.y0, self.y1)
        w = abs(self.x1 - self.x0)
        h = abs(self.y1 - self.y0)
        if w < 5 or h < 5:
            return None
        return [x, y, w, h]


def annotate_object_bbox(frame, window_name: str) -> tuple[int, str, list[int]] | None:
    """Workflow 100% OpenCV (zéro input() terminal).

    Étape 1 : Dessin de la BBox via BBoxDrawer (clic gauche + glisser).
    Étape 2 : Menu HUD de sélection de classe via touches 1-7 / ESC.

    Retourne (class_id, class_name, [x,y,w,h]) ou None si annulé.
    """
    # ── Étape 1 : Dessin de la BBox ──────────────────────────────────────
    drawer = BBoxDrawer()
    cv2.setMouseCallback(window_name, drawer.mouse_cb)

    base_frame = frame.copy()

    while not drawer.done and not drawer.cancelled:
        display = base_frame.copy()

        h, w = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w, 35), (200, 0, 200), -1)
        cv2.putText(display, "DESSINER BBOX : clic gauche + glisser | clic droit = annuler",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if drawer.drawing:
            cv2.rectangle(display,
                          (drawer.x0, drawer.y0), (drawer.x1, drawer.y1),
                          (0, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF
        if key == KEY_ESC:
            drawer.cancelled = True

    cv2.setMouseCallback(window_name, lambda *args: None)

    bbox = drawer.get_bbox()
    if bbox is None:
        print("  [ANNULÉ] Sélection vide ou annulée.")
        return None

    # ── Étape 2 : Menu HUD de sélection de classe ───────────────────────
    # Construire le menu texte
    menu_parts = []
    for key_code, cid, name in YOLO_CLASSES:
        key_label = chr(key_code).upper()
        menu_parts.append(f"[{key_label}] {cid}:{name}")
    menu_line = " | ".join(menu_parts)

    # Mapping touche -> (class_id, class_name)
    key_to_class = {k: (cid, name) for k, cid, name in YOLO_CLASSES}

    while True:
        preview = base_frame.copy()
        h, w = preview.shape[:2]

        # Dessiner la bbox validée
        cv2.rectangle(preview,
                      (bbox[0], bbox[1]),
                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 255, 0), 2)

        # Bandeau noir en haut pour le menu
        cv2.rectangle(preview, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(preview, "CLASSE :", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.putText(preview, menu_line, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(preview, "[ESC] Annuler", (w - 160, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(0) & 0xFF

        if key == KEY_ESC:
            print("  [ANNULÉ] Sélection de classe annulée.")
            return None

        if key in key_to_class:
            class_id, class_name = key_to_class[key]
            return class_id, class_name, bbox


def save_yolo_image(frame, cam_id: str, frame_idx: int) -> str:
    """Sauvegarde la frame complète (propre, sans dessin) en .jpg
    dans dataset/images/train/. Retourne le stem du fichier (sans extension)."""
    DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{cam_id}_frame{frame_idx}"
    filepath = DATASET_IMAGES_DIR / f"{stem}.jpg"
    cv2.imwrite(str(filepath), frame)
    print(f"  [IMAGE] {filepath.relative_to(PROJECT_ROOT)}")
    return stem


def save_yolo_source_image(frame: np.ndarray, source_path: Path) -> str:
    """Sauvegarde une image source dans le nouveau dataset.

    Le nom est conservé autant que possible pour faciliter le lien avec
    dataset_objets_HD/images. Si le fichier existe déjà, il n'est pas réécrit.
    """
    DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem.replace(" ", "_")
    filepath = DATASET_IMAGES_DIR / f"{stem}.jpg"
    if not filepath.exists():
        cv2.imwrite(str(filepath), frame)
        print(f"  [IMAGE] {filepath.relative_to(PROJECT_ROOT)}")
    return stem


def save_yolo_label(stem: str, class_id: int, bbox: list[int],
                    img_w: int, img_h: int) -> None:
    """Sauvegarde le label YOLO normalisé dans dataset/labels/train/.

    Conversion pixels absolus -> format YOLO :
      x_center = (x + w/2) / img_w
      y_center = (y + h/2) / img_h
      width    = w / img_w
      height   = h / img_h
    """
    DATASET_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    filepath = DATASET_LABELS_DIR / f"{stem}.txt"

    # Append pour supporter plusieurs objets sur la même frame
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}\n")

    print(f"  [LABEL] {filepath.relative_to(PROJECT_ROOT)}  "
          f"-> {class_id} {x_center:.4f} {y_center:.4f} "
          f"{w_norm:.4f} {h_norm:.4f}")


def label_file_for_stem(stem: str) -> Path:
    return DATASET_LABELS_DIR / f"{stem}.txt"


def count_labels_for_stem(stem: str) -> int:
    path = label_file_for_stem(stem)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ── Fonctions utilitaires annotations ────────────────────────────────────────

def find_nearest_people(gt_people: list[dict], cam_id: str,
                        frame_idx: int) -> tuple[int, dict] | None:
    best_idx, best_dist = None, float("inf")
    for i, a in enumerate(gt_people):
        if a["id_camera"] != cam_id:
            continue
        dist = abs(a["trame_violation"] - frame_idx)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_idx is not None:
        return best_idx, gt_people[best_idx]
    return None


def find_nearest_objects(gt_objects_tad: list[dict], cam_id: str,
                         frame_idx: int) -> tuple[int, dict] | None:
    best_idx, best_dist = None, float("inf")
    for i, a in enumerate(gt_objects_tad):
        if a["id_camera"] != cam_id:
            continue
        dist = abs(a["trame_apparition"] - frame_idx)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_idx is not None:
        return best_idx, gt_objects_tad[best_idx]
    return None


def print_annotation_list(gt_people: list[dict], gt_objects_tad: list[dict],
                           cam_id: str) -> None:
    cam_p = sorted(
        [a for a in gt_people if a["id_camera"] == cam_id],
        key=lambda a: a["trame_violation"],
    )
    cam_o = sorted(
        [a for a in gt_objects_tad if a["id_camera"] == cam_id],
        key=lambda a: a["trame_apparition"],
    )

    print(f"\n  ═══ VIOLATIONS PERSONNES ({cam_id}) : {len(cam_p)} ═══")
    for a in cam_p:
        t = format_time_ms(a["horodatage_violation"])
        print(f"    #{a['id_evenement']:3d}  frame {a['trame_violation']:>7d}  {t}")

    print(f"\n  ═══ OBJETS TAD ({cam_id}) : {len(cam_o)} ═══")
    for a in cam_o:
        t = format_time_ms(a["horodatage_apparition"])
        print(f"    #{a['id_evenement']:3d}  frame {a['trame_apparition']:>7d}  "
              f"{t}  [{a['classe_objet']}]")

    print(f"\n  ═══ TOTAUX GLOBAUX ═══")
    print(f"    Violations : {len(gt_people)}/{GOAL_VIOLATIONS}")
    print(f"    Objets TAD : {len(gt_objects_tad)}/{GOAL_OBJECTS_MIN}-{GOAL_OBJECTS_MAX}")


# ── Cache de frames (evite les seeks lents en arriere) ────────────────────────

class FrameCache:
    """Ring buffer LRU qui garde les N dernieres frames en memoire.
    Reculer d'une frame pioche dans le cache au lieu de faire un seek codec."""

    def __init__(self, maxsize: int = 600):
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def put(self, idx: int, frame: np.ndarray) -> None:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[idx] = frame

    def get(self, idx: int) -> np.ndarray | None:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        return None

    def __contains__(self, idx: int) -> bool:
        return idx in self._cache


# ── Boucle principale ────────────────────────────────────────────────────────

def annotate_video(video_info: dict,
                   gt_people: list[dict],
                   gt_objects_tad: list[dict],
                   phase3_config: dict | None = None) -> tuple[list[dict], list[dict]]:
    path = video_info["path"]
    cam_id = video_info["id_camera"]
    base_cam_id = extract_base_camera_id(cam_id)
    phase3_config = phase3_config or {}
    projected_zones = build_projected_zones_for_camera(phase3_config, base_cam_id)
    available_zone_ids = [zone["zone_id"] for zone in projected_zones]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir : {path}")
        return gt_people, gt_objects_tad

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    paused = True
    flash_label = None
    flash_color = (0, 0, 0)
    flash_counter = 0
    fcache = FrameCache(maxsize=600)

    count_p_before = len([a for a in gt_people if a["id_camera"] == cam_id])
    count_o_before = len([a for a in gt_objects_tad if a["id_camera"] == cam_id])

    window_name = f"Annotation - {cam_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720 + PANEL_HEIGHT)

    # Trackbar pour navigation rapide
    cv2.createTrackbar("Frame", window_name, 0, max(total_frames - 1, 1),
                       on_trackbar)

    print(f"\n[LECTURE] {cam_id} | {video_info['filename']}")
    print(f"  Caméra de base Phase 3 : {base_cam_id}")
    print(f"  Frames: {total_frames} | FPS: {fps:.1f} | "
          f"Durée: {format_time_ms(total_frames / fps * 1000)}")
    print(f"  Violations existantes (cam): {count_p_before} | "
          f"Objets existants (cam): {count_o_before}")
    if projected_zones:
        print(f"  Zones projetées : {', '.join(available_zone_ids)}")
    else:
        print("  [WARN] Aucune zone projetée pour cette caméra.")

    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Vidéo vide.")
        cap.release()
        return gt_people, gt_objects_tad
    fcache.put(frame_idx, frame)

    while True:
        # Synchronise trackbar -> frame_idx si l'utilisateur a bougé la trackbar
        tb_val = cv2.getTrackbarPos("Frame", window_name)
        if tb_val != frame_idx:
            frame_idx = tb_val
            cached = fcache.get(frame_idx)
            if cached is not None:
                frame = cached
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, new_frame = cap.read()
                if ret:
                    frame = new_frame
                    fcache.put(frame_idx, frame)
                # Repositionner pour la prochaine lecture séquentielle
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        timestamp_ms = (frame_idx / fps) * 1000.0

        display = frame.copy()
        draw_forbidden_zones(display, projected_zones)

        if flash_counter > 0:
            draw_flash(display, flash_label, flash_color)
            flash_counter -= 1

        panel = draw_hud(frame, cam_id, frame_idx, fps, timestamp_ms, paused,
                         total_frames, gt_people, gt_objects_tad)
        display = np.vstack([display, panel])

        cv2.imshow(window_name, display)

        wait_ms = 30 if paused else max(1, int(1000 / fps))
        key = cv2.waitKeyEx(wait_ms)

        # ── Touches ──────────────────────────────────────────────────────
        if key == KEY_ESC or key == KEY_Q:
            break

        elif key == KEY_SPACE:
            paused = not paused

        elif key in (KEY_RIGHT, KEY_RIGHT_WIN):
            paused = True
            if frame_idx < total_frames - 1:
                cached = fcache.get(frame_idx + 1)
                if cached is not None:
                    frame_idx += 1
                    frame = cached
                else:
                    ret, new_frame = cap.read()
                    if ret:
                        frame_idx += 1
                        frame = new_frame
                        fcache.put(frame_idx, frame)
                cv2.setTrackbarPos("Frame", window_name, frame_idx)

        elif key in (KEY_LEFT, KEY_LEFT_WIN):
            paused = True
            if frame_idx > 0:
                frame_idx -= 1
                cached = fcache.get(frame_idx)
                if cached is not None:
                    frame = cached
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, new_frame = cap.read()
                    if ret:
                        frame = new_frame
                        fcache.put(frame_idx, frame)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                cv2.setTrackbarPos("Frame", window_name, frame_idx)

        # ── V : Violation personne (avec zone + id_evenement global) ───
        elif key == KEY_V:
            paused = True
            ts = round(timestamp_ms, 2)

            # Afficher un bandeau pour indiquer le mode
            prompt_disp = frame.copy()
            h_f, w_f = prompt_disp.shape[:2]
            cv2.rectangle(prompt_disp, (0, h_f // 2 - 30), (w_f, h_f // 2 + 30),
                          (0, 0, 200), -1)
            cv2.putText(prompt_disp, "MODE VIOLATION : repondre dans le terminal",
                        (w_f // 2 - 280, h_f // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, prompt_disp)
            cv2.waitKey(1)

            # Demander id_evenement global et zone dans le terminal
            print(f"\n  ── VIOLATION PERSONNE @ frame {frame_idx} "
                  f"(t={format_time_ms(ts)}) ──")
            try:
                eid_str = input("  ID événement global (ex: 1, 2...) : ").strip()
                if len(available_zone_ids) == 1:
                    default_zone = available_zone_ids[0]
                    zone_str = input(f"  Zone [{default_zone}] : ").strip() or default_zone
                else:
                    choices = ", ".join(available_zone_ids) if available_zone_ids else "zone_1, zone_2"
                    zone_str = input(f"  Zone ({choices}) : ").strip()
            except EOFError:
                print("  [ANNULÉ]")
                continue

            if not eid_str or not zone_str:
                print("  [ANNULÉ] Champs vides.")
                continue

            try:
                eid = int(eid_str)
            except ValueError:
                print("  [ANNULÉ] ID événement doit être un entier.")
                continue

            if available_zone_ids and zone_str not in available_zone_ids:
                print(f"  [ANNULÉ] Zone invalide : '{zone_str}' "
                      f"(attendu: {', '.join(available_zone_ids)}).")
                continue

            gt_people.append({
                "id_camera": cam_id,
                "id_evenement": eid,
                "zone_id": zone_str,
                "trame_violation": frame_idx,
                "horodatage_violation": ts,
            })
            flash_label = f"VIOLATION #{eid} {zone_str}"
            flash_color = (0, 0, 255)
            flash_counter = 8
            print(f"  [V] Événement #{eid} {zone_str} | "
                  f"{cam_id} frame {frame_idx} "
                  f"[{len(gt_people)}/{GOAL_VIOLATIONS}]")

        # ── O : Apparition objet pour le TAD (temporel uniquement) ────
        elif key == KEY_O:
            paused = True
            ts = round(timestamp_ms, 2)

            # Bandeau pour indiquer le mode
            prompt_disp = frame.copy()
            h_f, w_f = prompt_disp.shape[:2]
            cv2.rectangle(prompt_disp, (0, h_f // 2 - 30), (w_f, h_f // 2 + 30),
                          (200, 0, 200), -1)
            cv2.putText(prompt_disp, "MODE OBJET TAD : repondre dans le terminal",
                        (w_f // 2 - 280, h_f // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, prompt_disp)
            cv2.waitKey(1)

            print(f"\n  ── APPARITION OBJET (TAD) @ frame {frame_idx} "
                  f"(t={format_time_ms(ts)}) ──")
            try:
                classe = input("  Classe de l'objet (ex: telephone) : ").strip()
                eid_str = input("  ID de l'événement global : ").strip()
            except EOFError:
                print("  [ANNULÉ]")
                continue

            if not classe or not eid_str:
                print("  [ANNULÉ] Champs vides.")
                continue

            try:
                eid = int(eid_str)
            except ValueError:
                print("  [ANNULÉ] ID événement doit être un entier.")
                continue

            gt_objects_tad.append({
                "id_camera": cam_id,
                "id_evenement": eid,
                "classe_objet": classe,
                "trame_apparition": frame_idx,
                "horodatage_apparition": ts,
            })

            flash_label = f"OBJET TAD: {classe} #{eid}"
            flash_color = (255, 0, 200)
            flash_counter = 8
            print(f"  [O] TAD #{eid} '{classe}' @ frame {frame_idx} "
                  f"[{len(gt_objects_tad)}/{GOAL_OBJECTS_MIN}]")

        # ── B : Bounding Box objet pour dataset YOLO (fine-tuning) ────
        elif key == KEY_B:
            paused = True

            bbox_result = annotate_object_bbox(frame, window_name)
            if bbox_result is None:
                continue

            class_id, class_name, bbox = bbox_result
            img_h, img_w = frame.shape[:2]

            # 1. Sauvegarder l'image propre dans dataset/images/train/
            stem = save_yolo_image(frame, cam_id, frame_idx)

            # 2. Sauvegarder le label YOLO normalisé dans dataset/labels/train/
            save_yolo_label(stem, class_id, bbox, img_w, img_h)

            flash_label = f"YOLO: {class_name} (id:{class_id})"
            flash_color = (0, 200, 0)
            flash_counter = 8
            print(f"  [B] YOLO '{class_name}'(id:{class_id}) @ frame {frame_idx} "
                  f"bbox={bbox}")

        # ── U : Annuler dernière annotation ──────────────────────────────
        elif key == KEY_U:
            # Cherche la dernière annotation (people ou objects) pour cette cam
            last_p = None
            for i in range(len(gt_people) - 1, -1, -1):
                if gt_people[i]["id_camera"] == cam_id:
                    last_p = ("people", i, gt_people[i])
                    break
            last_o = None
            for i in range(len(gt_objects_tad) - 1, -1, -1):
                if gt_objects_tad[i]["id_camera"] == cam_id:
                    last_o = ("objects", i, gt_objects_tad[i])
                    break

            # Supprime la plus récente (dernier index le plus grand)
            to_remove = None
            if last_p and last_o:
                to_remove = last_p if last_p[1] >= last_o[1] else last_o
            elif last_p:
                to_remove = last_p
            elif last_o:
                to_remove = last_o

            if to_remove:
                kind, idx, ann = to_remove
                if kind == "people":
                    removed = gt_people.pop(idx)
                    t = format_time_ms(removed["horodatage_violation"])
                    print(f"  [U] Annulé: violation #{removed['id_evenement']} "
                          f"@ frame {removed['trame_violation']} ({t})")
                else:
                    removed = gt_objects_tad.pop(idx)
                    t = format_time_ms(removed["horodatage_apparition"])
                    print(f"  [U] Annulé: objet #{removed['id_evenement']} "
                          f"'{removed['classe_objet']}' @ frame "
                          f"{removed['trame_apparition']} ({t})")
                flash_label, flash_color = "ANNOTATION SUPPRIMEE", (80, 80, 80)
                flash_counter = 6
            else:
                print("  [U] Aucune annotation à annuler pour cette caméra.")

        # ── Suppr : supprimer annotation la plus proche ──────────────────
        elif key in (KEY_DELETE, KEY_DELETE_WIN):
            res_p = find_nearest_people(gt_people, cam_id, frame_idx)
            res_o = find_nearest_objects(gt_objects_tad, cam_id, frame_idx)

            dist_p = abs(res_p[1]["trame_violation"] - frame_idx) if res_p else float("inf")
            dist_o = abs(res_o[1]["trame_apparition"] - frame_idx) if res_o else float("inf")

            if dist_p <= dist_o and res_p:
                removed = gt_people.pop(res_p[0])
                t = format_time_ms(removed["horodatage_violation"])
                print(f"  [Suppr] Violation #{removed['id_evenement']} "
                      f"@ frame {removed['trame_violation']} ({t}) "
                      f"[distance: {dist_p} frames]")
            elif res_o:
                removed = gt_objects_tad.pop(res_o[0])
                t = format_time_ms(removed["horodatage_apparition"])
                print(f"  [Suppr] Objet #{removed['id_evenement']} "
                      f"'{removed['classe_objet']}' @ frame "
                      f"{removed['trame_apparition']} ({t}) "
                      f"[distance: {int(dist_o)} frames]")
            else:
                print("  [Suppr] Aucune annotation à supprimer.")

            flash_label, flash_color = "SUPPRIME", (80, 80, 80)
            flash_counter = 8

        # ── L : Liste récapitulative ─────────────────────────────────────
        elif key == KEY_L:
            print_annotation_list(gt_people, gt_objects_tad, cam_id)

        # ── Avance automatique ───────────────────────────────────────────
        if not paused and key == -1:
            if frame_idx < total_frames - 1:
                ret, new_frame = cap.read()
                if ret:
                    frame_idx += 1
                    frame = new_frame
                    fcache.put(frame_idx, frame)
                    cv2.setTrackbarPos("Frame", window_name, frame_idx)
                else:
                    paused = True
                    print("  [FIN] Fin de la vidéo.")
            else:
                paused = True
                print("  [FIN] Fin de la vidéo.")

    cap.release()
    cv2.destroyWindow(window_name)

    count_p_after = len([a for a in gt_people if a["id_camera"] == cam_id])
    count_o_after = len([a for a in gt_objects_tad if a["id_camera"] == cam_id])
    print(f"  [{cam_id}] +{count_p_after - count_p_before} violation(s), "
          f"+{count_o_after - count_o_before} objet(s) TAD")

    return gt_people, gt_objects_tad


def annotate_image_folder(images: list[Path], auto_next: bool = True) -> None:
    """Annotation YOLO depuis un dossier d'images.

    Contrairement au mode vidéo, ce mode ne touche pas aux GT TAD/TRD. Il sert à
    refaire un dataset objet: image source -> output/images + output/labels.
    """
    if not images:
        return

    ensure_dataset_dirs()
    write_data_yaml(DATASET_IMAGES_DIR.parent)

    window_name = "Annotation images - YOLO"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720 + PANEL_HEIGHT)

    index = 0
    print(f"\n[MODE IMAGES] {len(images)} image(s) à annoter")
    print("  B = dessiner une bbox | N = suivante | P = précédente | S = skip | U = supprimer dernier label")
    print(f"  Auto-next après bbox : {'ON' if auto_next else 'OFF'}")

    while 0 <= index < len(images):
        image_path = images[index]
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  [WARN] Image illisible, skip : {image_path}")
            index += 1
            continue

        stem = image_path.stem.replace(" ", "_")
        copied_stem = save_yolo_source_image(frame, image_path)
        if copied_stem != stem:
            stem = copied_stem

        while True:
            label_count = count_labels_for_stem(stem)
            display = draw_image_hud(frame.copy(), image_path, index, len(images),
                                     stem, label_count, auto_next)
            cv2.imshow(window_name, display)
            key = cv2.waitKeyEx(0)

            if key in (KEY_ESC, KEY_Q):
                cv2.destroyWindow(window_name)
                return

            if key == KEY_N or key == KEY_S:
                index += 1
                break

            if key == KEY_P:
                index = max(0, index - 1)
                break

            if key == KEY_U:
                label_path = label_file_for_stem(stem)
                if not label_path.exists():
                    print("  [U] Aucun label à supprimer pour cette image.")
                    continue
                lines = label_path.read_text(encoding="utf-8").splitlines()
                lines = [line for line in lines if line.strip()]
                if not lines:
                    print("  [U] Aucun label à supprimer pour cette image.")
                    continue
                removed = lines.pop()
                label_path.write_text(("\n".join(lines) + "\n") if lines else "",
                                      encoding="utf-8")
                print(f"  [U] Label supprimé : {removed}")
                continue

            if key == KEY_B:
                bbox_result = annotate_object_bbox(frame, window_name)
                if bbox_result is None:
                    continue

                class_id, class_name, bbox = bbox_result
                img_h, img_w = frame.shape[:2]
                save_yolo_label(stem, class_id, bbox, img_w, img_h)
                print(f"  [B] {image_path.name} -> '{class_name}' id={class_id} bbox={bbox}")

                if auto_next:
                    index += 1
                    break

        if index >= len(images):
            break

    cv2.destroyWindow(window_name)
    n_images = len(list(DATASET_IMAGES_DIR.glob("*.jpg")))
    n_labels = len(list(DATASET_LABELS_DIR.glob("*.txt")))
    print(f"\n[TERMINÉ MODE IMAGES]")
    print(f"  Images dataset : {DATASET_IMAGES_DIR} ({n_images})")
    print(f"  Labels dataset : {DATASET_LABELS_DIR} ({n_labels})")
    print(f"  data.yaml      : {DATASET_IMAGES_DIR.parent / 'data.yaml'}")


# ── Contrôles ────────────────────────────────────────────────────────────────

def print_controls():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║       OUTIL D'ANNOTATION - GROUND TRUTH MULTI-CAMÉRAS        ║
║   Sortie : --output-dir (défaut: dataset_objets_HD)          ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  NAVIGATION :                                                 ║
║    Trackbar .......... Navigation rapide dans la timeline     ║
║    Espace ............ Lecture / Pause                        ║
║    Flèche Droite ..... Avancer d'1 frame                     ║
║    Flèche Gauche ..... Reculer d'1 frame                     ║
║    Q / Echap ......... Quitter la vidéo et sauvegarder       ║
║                                                               ║
║  ANNOTATIONS :                                                ║
║    V ..... Violation Personne          -> gt_people.json     ║
║            Zone choisie depuis config.yaml Phase 3            ║
║    O ..... Apparition Objet (TAD)      -> gt_objects_tad.json║
║    B ..... Box Objet (YOLO) :                                ║
║            1. Dessiner la bbox (clic gauche + glisser)       ║
║            2. Choisir la classe via le menu HUD              ║
║               touches 1..9 puis T/K/P selon data.yaml        ║
║               [ESC] annuler                                  ║
║            -> Sauvegarde image + label YOLO automatiquement  ║
║    U ..... Annuler la dernière annotation                    ║
║    Suppr . Supprimer l'annotation la plus proche             ║
║    L ..... Lister les annotations (console)                  ║
║                                                               ║
║  MODE IMAGES (--source images) :                              ║
║    B ..... Dessiner une bbox et choisir la classe             ║
║    N ..... Image suivante                                     ║
║    P ..... Image précédente                                   ║
║    S ..... Skip image courante                                ║
║    U ..... Supprimer le dernier label de l'image courante     ║
║    Par défaut, après BBox validée, l'outil passe              ║
║    automatiquement à l'image suivante.                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def ensure_dataset_dirs():
    """Crée l'arborescence YOLO si elle n'existe pas."""
    DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[YOLO] Arborescence dataset prête :")
    print(f"  images  -> {DATASET_IMAGES_DIR}/")
    print(f"  labels  -> {DATASET_LABELS_DIR}/")
    print(f"  gt TAD  -> {GT_OBJECTS_TAD_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Outil d'annotation de vérité terrain multi-caméras.")
    parser.add_argument(
        "--source",
        choices=["videos", "images"],
        default="videos",
        help="Source à annoter: vidéos historiques ou dossier d'images YOLO "
             "(défaut: videos)")
    parser.add_argument(
        "--output-dir", "-o",
        default=BASE_OUTPUT_DIR,
        help=f"Dossier de sortie pour les images YOLO, labels et "
             f"gt_objects_tad.json (défaut: {BASE_OUTPUT_DIR})")
    parser.add_argument(
        "--pattern", "-p",
        default="",
        help="Pattern de filtrage des vidéos (défaut: vide = toutes)")
    parser.add_argument(
        "--recordings-dir",
        default=str(RECORDINGS_DIR),
        help="Dossier recordings à scanner (défaut: recordings/recordings)")
    parser.add_argument(
        "--images-dir",
        default=str(PROJECT_ROOT / BASE_OUTPUT_DIR / "images"),
        help="Dossier d'images à scanner en mode --source images "
             "(défaut: dataset_objets_HD/images)")
    parser.add_argument(
        "--classes-yaml",
        default="",
        help="data.yaml contenant l'ordre des classes. Par défaut, utilise "
             "<images-dir>/../data.yaml si présent, sinon les classes internes.")
    parser.add_argument(
        "--phase3-config",
        default=str(DEFAULT_PHASE3_CONFIG_PATH),
        help="Chemin du config.yaml Phase 3 contenant zones_interdites + homographies")
    parser.add_argument(
        "--no-zones",
        action="store_true",
        help="Désactive l'affichage des zones interdites projetées")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Désactive le scan récursif des sous-dossiers")
    parser.add_argument(
        "--multi-box-per-image",
        action="store_true",
        help="Mode images: rester sur la même image après une bbox pour annoter "
             "plusieurs objets. Sans cette option, BBox validée => image suivante.")
    return parser.parse_args()


def apply_output_dir(output_dir: str):
    """Recalcule les chemins globaux selon le dossier de sortie choisi."""
    global GT_OBJECTS_TAD_PATH, DATASET_IMAGES_DIR, DATASET_LABELS_DIR

    output_path = Path(output_dir)
    output_root = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
    output_root = output_root.resolve()
    GT_OBJECTS_TAD_PATH = output_root / "gt_objects_tad.json"
    DATASET_IMAGES_DIR = output_root / "images"
    DATASET_LABELS_DIR = output_root / "labels"


def main():
    args = parse_args()
    apply_output_dir(args.output_dir)

    print(f"\n[CONFIG] Dossier de sortie : {DATASET_IMAGES_DIR.parent}")

    images_dir = resolve_cli_path(args.images_dir, PROJECT_ROOT)
    classes_yaml = None
    if args.classes_yaml:
        classes_yaml = resolve_cli_path(args.classes_yaml, PROJECT_ROOT)
    else:
        candidate = images_dir.parent / "data.yaml"
        if candidate.exists():
            classes_yaml = candidate
    if classes_yaml:
        apply_yolo_classes(load_yolo_classes_from_yaml(classes_yaml))
        print(f"[CONFIG] Classes YOLO : {classes_yaml}")
    else:
        print("[CONFIG] Classes YOLO : classes internes par défaut")

    print_controls()

    if args.source == "images":
        print(f"[CONFIG] Dossier images source : {images_dir}")
        ensure_dataset_dirs()
        write_data_yaml(DATASET_IMAGES_DIR.parent)
        images = list_images(
            images_dir=images_dir,
            pattern=args.pattern,
            recursive=not args.no_recursive,
        )
        annotate_image_folder(images, auto_next=not args.multi_box_per_image)
        cv2.destroyAllWindows()
        return

    recordings_dir = Path(args.recordings_dir)
    if not recordings_dir.is_absolute():
        recordings_dir = PROJECT_ROOT / recordings_dir
    recordings_dir = recordings_dir.resolve()
    print(f"[CONFIG] Dossier recordings : {recordings_dir}")

    phase3_config = {}
    if not args.no_zones:
        phase3_config_path = resolve_cli_path(args.phase3_config, PROJECT_ROOT)
        phase3_config = load_phase3_config(phase3_config_path)
        print(f"[CONFIG] Zones Phase 3 : {phase3_config_path}")
        zone_count = len(phase3_config.get("zones_interdites", {}))
        print(f"[CONFIG] {zone_count} zone(s) interdite(s) chargée(s)")
    else:
        print("[CONFIG] Affichage des zones désactivé (--no-zones)")

    ensure_dataset_dirs()
    write_data_yaml(DATASET_IMAGES_DIR.parent)

    videos = list_videos(
        recordings_dir=recordings_dir,
        pattern=args.pattern,
        recursive=not args.no_recursive,
    )

    gt_people = load_gt_people()
    gt_objects_tad = load_gt_objects_tad()

    if gt_people:
        print(f"[INFO] {len(gt_people)} violations chargées depuis {GT_PEOPLE_PATH.name}")
    if gt_objects_tad:
        print(f"[INFO] {len(gt_objects_tad)} objets TAD chargés depuis {GT_OBJECTS_TAD_PATH.name}")
    print(f"[OBJECTIFS] Violations: {len(gt_people)}/{GOAL_VIOLATIONS} | "
          f"Objets TAD: {len(gt_objects_tad)}/{GOAL_OBJECTS_MIN}-{GOAL_OBJECTS_MAX}")

    while True:
        video_info = select_video(videos)
        if video_info is None:
            break

        gt_people, gt_objects_tad = annotate_video(
            video_info,
            gt_people,
            gt_objects_tad,
            phase3_config=phase3_config,
        )
        save_gt_people(gt_people)
        save_gt_objects_tad(gt_objects_tad)

    save_gt_people(gt_people)
    save_gt_objects_tad(gt_objects_tad)

    # Compter les fichiers YOLO générés
    n_images = len(list(DATASET_IMAGES_DIR.glob("*.jpg")))
    n_labels = len(list(DATASET_LABELS_DIR.glob("*.txt")))

    print(f"\n[TERMINÉ] Fichiers sauvegardés :")
    print(f"  - {GT_PEOPLE_PATH}  ({len(gt_people)} violations)")
    print(f"  - {GT_OBJECTS_TAD_PATH}  ({len(gt_objects_tad)} objets TAD)")
    print(f"  - YOLO images : {DATASET_IMAGES_DIR}/  ({n_images} .jpg)")
    print(f"  - YOLO labels : {DATASET_LABELS_DIR}/  ({n_labels} .txt)")
    classes_str = ", ".join(f"{cid}:{name}" for _, cid, name in YOLO_CLASSES)
    print(f"\n  Classes YOLO : {classes_str}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

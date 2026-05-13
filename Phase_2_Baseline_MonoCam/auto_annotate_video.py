"""
auto_annotate_video.py
Extrait des frames d'une vidéo à intervalle régulier et génère
des pré-annotations YOLO (.txt) pour les classes COCO sélectionnées.
"""

import os
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# PARAMÈTRES — modifier ici avant de lancer le script
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

VIDEO_PATH = "recordings/recordings/Camera_7_2.11_record2.mp4"
OUTPUT_DIR = "dataset_output"
VIDEOS_DIR = PROJECT_ROOT / "recordings" / "recordings"
VIDEO_PATTERN = "Camera_*_record2.mp4"
BATCH_OUTPUT_ROOT = PROJECT_ROOT / "dataset_output_batch"
FRAME_INTERVAL_SEC = 2.0
# Classes custom du projet :
#  0: maillet       1: marteau      2: niveau a bulle  3: scie
#  4: tabouret      5: telephone    6: verre           7: perceuse
#  8: bouteille     9: pince       10: cutter         11: metre
# 12: tournevis    13: cle allen 14: personne
COCO_TO_CUSTOM = {39: 8, 67: 5}
TARGET_CLASSES = list(COCO_TO_CUSTOM.keys())  # [39, 67]
MODEL_PATH = "yolov8n.pt"
# ============================================================


def auto_annotate(
    video_path: str,
    output_dir: str,
    frame_interval_sec: float = 2.0,
    target_classes: list[int] | None = None,
    coco_to_custom: dict[int, int] | None = None,
    model_path: str = "yolov8n.pt",
) -> None:
    if target_classes is None:
        target_classes = TARGET_CLASSES
    if coco_to_custom is None:
        coco_to_custom = COCO_TO_CUSTOM

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    # Créer les sous-dossiers de sortie
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Charger le modèle YOLO
    model = YOLO(model_path)

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(fps * frame_interval_sec))

    # Nombre estimé de frames à traiter (pour la barre de progression)
    estimated_steps = total_frames // frame_skip
    video_stem = video_path.stem

    images_saved = 0
    labels_saved = 0
    frame_idx = 0

    with tqdm(total=estimated_steps, desc="Extraction & annotation", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Calculer le timestamp pour le nom de fichier
                total_sec = frame_idx / fps
                minutes = int(total_sec // 60)
                seconds = int(total_sec % 60)
                filename = f"{video_stem}_{minutes:02d}m{seconds:02d}s"

                # Inférence YOLO
                results = model(frame, classes=target_classes, verbose=False)

                # Sauvegarder l'image
                img_path = images_dir / f"{filename}.jpg"
                cv2.imwrite(str(img_path), frame)
                images_saved += 1

                # Sauvegarder les annotations si des objets sont détectés
                # Convertir les IDs COCO en IDs custom du projet
                boxes = results[0].boxes
                if len(boxes) > 0:
                    label_path = labels_dir / f"{filename}.txt"
                    with open(label_path, "w") as f:
                        for cls_id, xywhn in zip(boxes.cls, boxes.xywhn):
                            coco_id = int(cls_id.item())
                            custom_id = coco_to_custom.get(coco_id)
                            if custom_id is None:
                                continue
                            x_c, y_c, w, h = xywhn.tolist()
                            f.write(f"{custom_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    labels_saved += 1

                pbar.update(1)

            frame_idx += 1

    cap.release()

    print(f"\n{'='*50}")
    print(f"{images_saved} images extraites, {labels_saved} fichiers d'annotations générés.")
    print(f"Images : {images_dir}")
    print(f"Labels : {labels_dir}")
    print(f"{'='*50}")


def process_videos_one_by_one(
    videos_dir: str,
    video_pattern: str,
    batch_output_root: str,
    frame_interval_sec: float,
    target_classes: list[int] | None,
    model_path: str,
) -> None:
    videos_root = Path(videos_dir)
    if not videos_root.is_dir():
        raise FileNotFoundError(f"Dossier vidéos introuvable : {videos_root}")

    video_paths = sorted(videos_root.glob(video_pattern))
    if not video_paths:
        raise FileNotFoundError(
            f"Aucune vidéo trouvée dans {videos_root} avec le pattern '{video_pattern}'"
        )

    output_root = Path(batch_output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{len(video_paths)} vidéo(s) détectée(s).")
    for index, video_path in enumerate(video_paths, start=1):
        per_video_output = output_root / video_path.stem
        print(f"\n[{index}/{len(video_paths)}] Traitement de : {video_path.name}")
        auto_annotate(
            video_path=str(video_path),
            output_dir=str(per_video_output),
            frame_interval_sec=frame_interval_sec,
            target_classes=target_classes,
            model_path=model_path,
        )

    print(f"\nTraitement terminé. Résultats disponibles dans : {output_root}")


if __name__ == "__main__":
    process_videos_one_by_one(
        videos_dir=VIDEOS_DIR,
        video_pattern=VIDEO_PATTERN,
        batch_output_root=BATCH_OUTPUT_ROOT,
        frame_interval_sec=FRAME_INTERVAL_SEC,
        target_classes=TARGET_CLASSES,
        model_path=MODEL_PATH,
    )

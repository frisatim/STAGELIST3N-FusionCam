"""
Convertit gt_objects.json en dataset YOLO (images + labels normalisés).

Gère :
  - La correction automatique des typos dans les noms de classe
  - Le nommage mixte des images (cam_XX_frameN.jpg et cam_XX_obj_XXXX.jpg)
  - Les annotations multiples sur une même image (plusieurs lignes par .txt)
  - La normalisation des coordonnées bbox [x, y, w, h] -> YOLO [xc, yc, w, h]

Sortie : yolo_dataset/images/train/ + yolo_dataset/labels/train/ + classes.txt
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_JSON = PROJECT_ROOT / "gt_objects.json"
SOURCE_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
OUTPUT_DIR = PROJECT_ROOT / "yolo_dataset"
OUTPUT_IMAGES = OUTPUT_DIR / "images" / "train"
OUTPUT_LABELS = OUTPUT_DIR / "labels" / "train"

# ── Classes valides (ordre = class_id) ────────────────────────────────────
VALID_CLASSES = [
    "maillet",
    "marteau",
    "niveau à bulle",
    "scie",
    "tabouret",
    "téléphone",
    "verre",
]

# ── Correction automatique des typos ──────────────────────────────────────
TYPO_FIXES = {
    "amrteau": "marteau",
    "mailet": "maillet",
    "maillety": "maillet",
    "mailley": "maillet",
    "télé^phone": "téléphone",
    "5": "verre",
}


def fix_class_name(raw: str) -> str:
    return TYPO_FIXES.get(raw, raw)


def build_class_map() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(VALID_CLASSES)}


def resolve_image_path(cam_id: str, frame_idx: int,
                       obj_indices: dict[str, int]) -> Path | None:
    """Cherche l'image correspondante dans SOURCE_IMAGES_DIR.

    Essaie d'abord le pattern cam_XX_frameN.jpg,
    puis le pattern legacy cam_XX_obj_XXXX.jpg (séquentiel).
    """
    # Pattern principal
    candidate = SOURCE_IMAGES_DIR / f"{cam_id}_frame{frame_idx}.jpg"
    if candidate.exists():
        return candidate

    # Pattern legacy (cam_01_obj_0001.jpg etc.) - attribution séquentielle
    if cam_id not in obj_indices:
        obj_indices[cam_id] = 0
    obj_indices[cam_id] += 1
    seq = obj_indices[cam_id]
    candidate = SOURCE_IMAGES_DIR / f"{cam_id}_obj_{seq:04d}.jpg"
    if candidate.exists():
        return candidate

    return None


def main():
    # ── Chargement ────────────────────────────────────────────────────────
    if not INPUT_JSON.exists():
        print(f"[ERREUR] Fichier introuvable : {INPUT_JSON}")
        sys.exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"[LECTURE] {INPUT_JSON.name} : {len(annotations)} annotations")

    # ── Création de l'arborescence ────────────────────────────────────────
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)
    print(f"[DOSSIERS] {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")

    # ── Mapping des classes ───────────────────────────────────────────────
    class_map = build_class_map()
    print(f"[CLASSES] {len(class_map)} classes :")
    for name, cid in class_map.items():
        print(f"  {cid}: {name}")

    # Correction des typos
    typo_count = 0
    for ann in annotations:
        original = ann["classe_objet"]
        fixed = fix_class_name(original)
        if fixed != original:
            typo_count += 1
            ann["classe_objet"] = fixed

    if typo_count:
        print(f"[TYPOS] {typo_count} correction(s) appliquée(s)")

    # Vérifier qu'il ne reste pas de classe inconnue
    unknown = set()
    for ann in annotations:
        if ann["classe_objet"] not in class_map:
            unknown.add(ann["classe_objet"])
    if unknown:
        print(f"[ATTENTION] Classes inconnues ignorées : {unknown}")

    # ── Regrouper les annotations par image (cam + frame) ────────────────
    # Clé = (cam_id, trame_apparition), Valeur = liste d'annotations
    image_groups: dict[tuple, list[dict]] = defaultdict(list)
    for ann in annotations:
        if ann["classe_objet"] not in class_map:
            continue
        key = (ann["id_camera"], ann["trame_apparition"])
        image_groups[key].append(ann)

    print(f"[IMAGES] {len(image_groups)} images uniques à traiter")

    # ── Tri par (cam, frame) pour que le pattern obj_XXXX soit séquentiel
    sorted_keys = sorted(image_groups.keys(), key=lambda k: (k[0], k[1]))

    # ── Conversion ────────────────────────────────────────────────────────
    obj_indices: dict[str, int] = {}  # compteur séquentiel par caméra
    n_ok = 0
    n_missing = 0
    n_labels_written = 0

    for cam_id, frame_idx in sorted_keys:
        anns = image_groups[(cam_id, frame_idx)]
        src_path = resolve_image_path(cam_id, frame_idx, obj_indices)

        if src_path is None:
            n_missing += 1
            print(f"  [MANQUANTE] {cam_id}_frame{frame_idx}.jpg "
                  f"({len(anns)} annotation(s) perdues)")
            continue

        # Lire les dimensions réelles de l'image
        with Image.open(src_path) as img:
            img_w, img_h = img.size

        # Nom de sortie unifié
        out_stem = f"{cam_id}_frame{frame_idx}"
        out_img = OUTPUT_IMAGES / f"{out_stem}.jpg"
        out_lbl = OUTPUT_LABELS / f"{out_stem}.txt"

        # Copier l'image
        shutil.copy2(str(src_path), str(out_img))

        # Générer le label YOLO (une ligne par objet)
        lines = []
        for ann in anns:
            class_id = class_map[ann["classe_objet"]]
            bx, by, bw, bh = ann["bbox"]

            x_center = (bx + bw / 2) / img_w
            y_center = (by + bh / 2) / img_h
            w_norm = bw / img_w
            h_norm = bh / img_h

            # Clamp entre 0 et 1
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} "
                         f"{w_norm:.6f} {h_norm:.6f}")

        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        n_ok += 1
        n_labels_written += len(lines)

    # ── classes.txt ───────────────────────────────────────────────────────
    classes_file = OUTPUT_DIR / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for name in VALID_CLASSES:
            f.write(name + "\n")

    # ── Bilan ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  CONVERSION TERMINÉE")
    print(f"{'=' * 55}")
    print(f"  Images copiées  : {n_ok}")
    print(f"  Images manquantes : {n_missing}")
    print(f"  Labels générés  : {n_ok} fichiers ({n_labels_written} lignes)")
    print(f"  classes.txt     : {classes_file.relative_to(PROJECT_ROOT)}")
    print(f"{'=' * 55}")

    if n_missing:
        print(f"\n  [!] {n_missing} image(s) introuvable(s) dans "
              f"{SOURCE_IMAGES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()

"""
Pré-annotation automatique de la classe 'personne' sur un dataset existant.

Passe YOLOv8n (poids COCO) sur toutes les images du premier dataset valide
trouvé parmi DATASET_CANDIDATES et ajoute (en mode append, sans écraser
les annotations existantes) une ligne YOLO par personne détectée
(confiance >= CONFIDENCE_THRESHOLD). L'ID de la classe personne est lu
dans le data.yaml du dataset ; à défaut, PERSON_NEW_ID_FALLBACK est
utilisé. Utilisé pendant la construction du dataset pour éviter
d'annoter les personnes à la main ; les boîtes restent à vérifier.

Pas d'arguments CLI : les chemins et seuils sont des constantes en tête
de fichier (DATASET_CANDIDATES, CONFIDENCE_THRESHOLD...) à adapter avant
lancement.

Lancement :
  python auto_annotate_person.py
"""

import os
import glob
from ultralytics import YOLO

try:
    import yaml
except ImportError:
    yaml = None

# ==========================================
# CONFIGURATION
# ==========================================
# Dossiers tries dans l'ordre: le premier dataset valide est utilise.
DATASET_CANDIDATES = ("dataset", "dataset_objets_HD", "dataset_outils_720p")

# La resolution se fait depuis la racine du projet pour etre independant du cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Fallback utilise uniquement si data.yaml est absent/invalide.
PERSON_NEW_ID_FALLBACK = 14

CONFIDENCE_THRESHOLD = 0.6

# Extensions d'images a traiter
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
# ==========================================


def _resolve_split_dir(base_dir):
    """Retourne base_dir/train si present, sinon base_dir."""
    train_dir = os.path.join(base_dir, "train")
    if os.path.isdir(train_dir):
        return train_dir
    return base_dir


def _find_dataset_dirs():
    """Detecte automatiquement un dataset avec images + labels."""
    for dataset_name in DATASET_CANDIDATES:
        dataset_dir = os.path.join(PROJECT_ROOT, dataset_name)
        images_base = os.path.join(dataset_dir, "images")
        labels_base = os.path.join(dataset_dir, "labels")
        if os.path.isdir(images_base) and os.path.isdir(labels_base):
            images_dir = _resolve_split_dir(images_base)
            labels_dir = _resolve_split_dir(labels_base)
            yaml_path = os.path.join(dataset_dir, "data.yaml")
            return dataset_name, images_dir, labels_dir, yaml_path
    return None, None, None, None


def _resolve_person_class_id(yaml_path):
    """Recupere l'ID de la classe personne depuis data.yaml."""
    if yaml is None:
        print("AVERTISSEMENT: PyYAML non installe. Utilisation de PERSON_NEW_ID_FALLBACK.")
        return PERSON_NEW_ID_FALLBACK

    if not os.path.isfile(yaml_path):
        print(f"AVERTISSEMENT: data.yaml introuvable ({yaml_path}). Utilisation du fallback.")
        return PERSON_NEW_ID_FALLBACK

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names", [])
    if isinstance(names, dict):
        normalized = {}
        for k, v in names.items():
            try:
                normalized[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        names_dict = normalized
    elif isinstance(names, list):
        names_dict = {i: str(v) for i, v in enumerate(names)}
    else:
        names_dict = {}

    for idx, class_name in names_dict.items():
        label = class_name.strip().lower()
        if label in ("person", "personne", "people"):
            return idx

    nc = data.get("nc")
    if isinstance(nc, int):
        print(
            "AVERTISSEMENT: Classe 'personne' absente de data.yaml. "
            f"ID propose = {nc}. Mets a jour data.yaml (nc + nom de classe)."
        )
        return nc

    print("AVERTISSEMENT: data.yaml incomplet. Utilisation de PERSON_NEW_ID_FALLBACK.")
    return PERSON_NEW_ID_FALLBACK


def auto_annotate():
    """Détecte les personnes sur chaque image et complète les labels YOLO."""
    dataset_dir, images_dir, labels_dir, yaml_path = _find_dataset_dirs()
    if not dataset_dir:
        print(
            "ERREUR : Aucun dataset valide trouve. "
            "Attendu: <dataset>/images et <dataset>/labels."
        )
        return

    person_new_id = _resolve_person_class_id(yaml_path)

    print(f"Dataset detecte: {dataset_dir}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"ID classe personne utilise: {person_new_id}")

    print("Chargement du modele YOLOv8n (COCO)...")
    model = YOLO("yolov8n.pt")

    # Collecte de toutes les images avec les extensions supportees
    images_paths = []
    for ext in IMAGE_EXTENSIONS:
        images_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    images_paths.sort()

    if not images_paths:
        print(f"Aucune image trouvee dans '{images_dir}'.")
        return

    print(f"{len(images_paths)} images trouvees. Debut de l'analyse...")

    personnes_ajoutees = 0
    images_modifiees = 0

    for i, img_path in enumerate(images_paths, 1):
        nom_fichier = os.path.basename(img_path)
        # Remplace l'extension par .txt quel que soit le format d'image
        nom_txt = os.path.splitext(nom_fichier)[0] + ".txt"
        label_path = os.path.join(labels_dir, nom_txt)

        # Inference sans logs console
        results = model(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)

        nouvelles_lignes = []

        for r in results:
            for box in r.boxes:
                # Classe 0 dans COCO = person
                if int(box.cls[0]) == 0:
                    x, y, w, h = box.xywhn[0].tolist()
                    nouvelles_lignes.append(
                        f"{person_new_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                    )

        # Mode 'a' (append) : on ajoute sans ecraser les annotations existantes
        if nouvelles_lignes:
            with open(label_path, "a") as f:
                f.writelines(nouvelles_lignes)
            personnes_ajoutees += len(nouvelles_lignes)
            images_modifiees += 1

        # Progression toutes les 50 images
        if i % 50 == 0:
            print(f"  [{i}/{len(images_paths)}] images traitees...")

    print("==========================================")
    print("AUTO-ANNOTATION TERMINEE")
    print(f"  {personnes_ajoutees} boites 'personne' ajoutees")
    print(f"  {images_modifiees}/{len(images_paths)} images contenaient une personne")
    print("==========================================")

if __name__ == "__main__":
    auto_annotate()

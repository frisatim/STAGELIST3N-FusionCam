"""Prépare l'arborescence externe de livraison des données lourdes.

Ce script crée volontairement les répertoires seulement (avec .gitkeep) : il ne
copie ni datasets, ni vidéos, ni poids de modèles, car ces fichiers très
volumineux doivent être sélectionnés délibérément avant la livraison. Seules
les vérités terrain (JSON légers) sont copiées si elles sont présentes.

Exemple :
    python scripts/prepare_delivery_layout.py --data-dir ../STAGELIST3N-FusionCam-data
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


DEFAULT_DIRS = [
    "datasets/dataset_objets_V4",
    "datasets/dataset_objets_HD",
    "datasets/dataset",
    "recordings/recordings",
    "models/V2",
    "models/V3",
    "models/V4",
    "reports/Phase_2_Baseline_MonoCam",
    "reports/Phase_3_Fusion_MultiCam",
    "reports/Phase_4_Network_Latency",
    "exports",
]


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Prépare l'arborescence de livraison FusionCam.")
    parser.add_argument(
        "--data-dir",
        default="../STAGELIST3N-FusionCam-data",
        help="Répertoire externe de données lourdes à créer.",
    )
    return parser.parse_args()


def main() -> None:
    """Crée l'arborescence de livraison, un README_DATA.md décrivant le contenu
    attendu, puis copie les vérités terrain si elles existent dans le dépôt."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root = Path(args.data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Créer chaque répertoire attendu avec un .gitkeep pour le conserver vide.
    for rel in DEFAULT_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        keep.touch(exist_ok=True)

    readme = root / "README_DATA.md"
    if not readme.exists():
        readme.write_text(
            "# STAGELIST3N FusionCam Heavy Data\n\n"
            "This folder is intentionally kept outside Git.\n\n"
            "Expected content:\n\n"
            "- `datasets/`: training and evaluation datasets.\n"
            "- `recordings/recordings/`: MP4/AVI camera recordings.\n"
            "- `models/`: trained `.pt` weights and optional TensorRT `.engine` files,\n"
            "  using the same internal layout as `Phase_2_Baseline_MonoCam/Modelstrained`.\n"
            "- `reports/`: large campaign outputs and logs.\n"
            "- `exports/`: packaged results for delivery.\n",
            encoding="utf-8",
        )

    # Copier les vérités terrain (fichiers JSON légers) sans écraser l'existant.
    gt_people_src = repo_root / "ground_truth" / "gt_people.json"
    gt_objects_src = repo_root / "ground_truth" / "gt_objects_tad_dataset_objets_HD.json"
    if not gt_objects_src.exists():
        gt_objects_src = repo_root / "ground_truth" / "gt_objects_tad.json"

    gt_people_dst = root / "ground_truth" / "gt_people.json"
    gt_objects_dst = root / "datasets" / "dataset_objets_HD" / "gt_objects_tad.json"
    gt_people_dst.parent.mkdir(parents=True, exist_ok=True)
    gt_objects_dst.parent.mkdir(parents=True, exist_ok=True)
    if gt_people_src.exists() and not gt_people_dst.exists():
        shutil.copy2(gt_people_src, gt_people_dst)
    if gt_objects_src.exists() and not gt_objects_dst.exists():
        shutil.copy2(gt_objects_src, gt_objects_dst)

    print(f"Prepared heavy-data layout: {root}")
    for rel in DEFAULT_DIRS:
        print(f"  - {rel}/")


if __name__ == "__main__":
    main()

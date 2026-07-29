import argparse
import shutil
from pathlib import Path

# ==========================================
# CONFIGURATION PAR DEFAUT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source Roboflow: chemin racine (contenant train/valid/test) ou un split direct.
SOURCE_DEFAULT = PROJECT_ROOT / "cup.v1i.yolov8"

# Dataset cible de ce repository.
TARGET_DEFAULT = PROJECT_ROOT / "dataset"

# Mapping classes Roboflow -> classes cible.
# Exemple: la classe 0 Roboflow devient la classe 6 dans le dataset cible.
CLASS_MAPPING_DEFAULT = {
    0: 6,
}

# Extensions image supportees pour retrouver le fichier correspondant au label.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_mapping(mapping_text: str | None) -> dict[int, int]:
    """Parse un mapping sous forme 0:6,1:4 depuis la CLI."""
    if not mapping_text:
        return dict(CLASS_MAPPING_DEFAULT)

    result: dict[int, int] = {}
    chunks = [c.strip() for c in mapping_text.split(",") if c.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            raise ValueError(f"Mapping invalide '{chunk}'. Format attendu: src:dst")
        src, dst = [x.strip() for x in chunk.split(":", 1)]
        result[int(src)] = int(dst)
    return result


def detect_source_splits(source_root: Path) -> list[tuple[str, Path]]:
    """Retourne la liste des splits detectes au format (nom_split, dossier_split)."""
    # Cas 1: source_root est deja un split (contient images/labels).
    if (source_root / "images").is_dir() and (source_root / "labels").is_dir():
        split_name = source_root.name.lower()
        if split_name not in {"train", "valid", "test"}:
            split_name = "train"
        return [(split_name, source_root)]

    # Cas 2: source_root contient des sous-dossiers train/valid/test.
    splits: list[tuple[str, Path]] = []
    for name in ("train", "valid", "test"):
        split_dir = source_root / name
        if (split_dir / "images").is_dir() and (split_dir / "labels").is_dir():
            splits.append((name, split_dir))
    return splits


def resolve_source_path(source_root: Path) -> Path:
    """Trouve une source valide meme si le chemin par defaut n'existe pas."""
    if source_root.exists():
        return source_root

    downloads_dir = Path.home() / "Downloads"
    if not downloads_dir.is_dir():
        return source_root

    candidates = sorted(
        [p for p in downloads_dir.glob("*.yolov8") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        if detect_source_splits(candidate):
            print(f"INFO: source introuvable ({source_root}).")
            print(f"INFO: source detectee automatiquement: {candidate}")
            return candidate

    return source_root


def get_target_split_dirs(target_root: Path, split_name: str, merge_to_train: bool) -> tuple[Path, Path]:
    """Construit les dossiers images/labels de destination pour un split."""
    images_root = target_root / "images"
    labels_root = target_root / "labels"

    if merge_to_train:
        dst_split = "train"
    else:
        dst_split = split_name

    images_dir = images_root / dst_split
    labels_dir = labels_root / dst_split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def find_matching_image(images_dir: Path, stem: str) -> Path | None:
    """Trouve l'image correspondant a un label sans supposer une extension unique."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_unique_stem(preferred_stem: str, labels_target_dir: Path, images_target_dir: Path, ext: str) -> str:
    """Evite d'ecraser des donnees existantes en ajoutant un suffixe si necessaire."""
    candidate = preferred_stem
    i = 1
    while True:
        label_exists = (labels_target_dir / f"{candidate}.txt").exists()
        image_exists = (images_target_dir / f"{candidate}{ext}").exists()
        if not label_exists and not image_exists:
            return candidate
        candidate = f"{preferred_stem}__rf{i}"
        i += 1


def remap_label_file(label_path: Path, class_mapping: dict[int, int]) -> tuple[list[str], int, int]:
    """Remap classes d'un label YOLO et retourne (lignes, total_objs, mapped_objs)."""
    output_lines: list[str] = []
    total_objs = 0
    mapped_objs = 0

    with open(label_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            total_objs += 1
            try:
                old_class = int(parts[0])
            except ValueError:
                continue

            if old_class not in class_mapping:
                continue

            parts[0] = str(class_mapping[old_class])
            output_lines.append(" ".join(parts) + "\n")
            mapped_objs += 1

    return output_lines, total_objs, mapped_objs


def merge_datasets(source: Path, target: Path, class_mapping: dict[int, int], merge_to_train: bool) -> int:
    """Fusionne des labels/images Roboflow dans un dataset YOLO cible."""
    source = resolve_source_path(source).resolve()
    target = target.resolve()

    print("Demarrage de la fusion Roboflow -> dataset cible...")
    print(f"Source : {source}")
    print(f"Target : {target}")
    print(f"Mapping: {class_mapping}")
    print(f"Mode split: {'merge->train' if merge_to_train else 'preserve'}")

    if not source.exists():
        print(f"ERREUR: source introuvable: {source}")
        return 1

    splits = detect_source_splits(source)
    if not splits:
        print("ERREUR: aucun split valide detecte dans la source.")
        print("Attendu: source/images+labels ou source/train|valid|test/images+labels")
        return 1

    (target / "images").mkdir(parents=True, exist_ok=True)
    (target / "labels").mkdir(parents=True, exist_ok=True)

    files_seen = 0
    files_imported = 0
    files_skipped_no_image = 0
    files_skipped_no_mapped = 0
    files_collisions = 0
    objects_seen = 0
    objects_mapped = 0

    for split_name, split_dir in splits:
        src_images = split_dir / "images"
        src_labels = split_dir / "labels"
        dst_images, dst_labels = get_target_split_dirs(target, split_name, merge_to_train)

        label_files = sorted(src_labels.glob("*.txt"))
        if not label_files:
            continue

        print(f"\nSplit {split_name}: {len(label_files)} labels detectes")

        for label_path in label_files:
            files_seen += 1
            stem = label_path.stem
            image_path = find_matching_image(src_images, stem)
            if image_path is None:
                files_skipped_no_image += 1
                continue

            converted_lines, total_obj, mapped_obj = remap_label_file(label_path, class_mapping)
            objects_seen += total_obj
            objects_mapped += mapped_obj

            if not converted_lines:
                files_skipped_no_mapped += 1
                continue

            ext = image_path.suffix.lower()
            final_stem = stem
            if (dst_labels / f"{stem}.txt").exists() or (dst_images / f"{stem}{ext}").exists():
                files_collisions += 1
                final_stem = get_unique_stem(stem, dst_labels, dst_images, ext)

            dst_label_path = dst_labels / f"{final_stem}.txt"
            dst_image_path = dst_images / f"{final_stem}{ext}"

            with open(dst_label_path, "w", encoding="utf-8") as f:
                f.writelines(converted_lines)
            shutil.copy2(image_path, dst_image_path)
            files_imported += 1

    print("\n==========================================")
    print("Fusion terminee")
    print(f"Fichiers vus                 : {files_seen}")
    print(f"Fichiers importes            : {files_imported}")
    print(f"Ignores (image introuvable)  : {files_skipped_no_image}")
    print(f"Ignores (classes non mappees): {files_skipped_no_mapped}")
    print(f"Collisions renommees         : {files_collisions}")
    print(f"Objets vus                   : {objects_seen}")
    print(f"Objets mappes                : {objects_mapped}")
    print("==========================================")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Fusion de dataset Roboflow vers dataset YOLO local")
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DEFAULT,
        help=f"Source Roboflow (racine ou split direct). Defaut: {SOURCE_DEFAULT}",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=TARGET_DEFAULT,
        help=f"Dataset cible (contient images/ et labels/). Defaut: {TARGET_DEFAULT}",
    )
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Mapping classes au format src:dst,src:dst (ex: 0:6,1:4)",
    )
    parser.add_argument(
        "--preserve-splits",
        action="store_true",
        help="Garde train/valid/test separes au lieu de tout fusionner dans train",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        class_mapping = parse_mapping(args.map)
    except ValueError as e:
        print(f"ERREUR mapping: {e}")
        return 1

    return merge_datasets(
        source=args.source,
        target=args.target,
        class_mapping=class_mapping,
        merge_to_train=not args.preserve_splits,
    )


if __name__ == "__main__":
    raise SystemExit(main())
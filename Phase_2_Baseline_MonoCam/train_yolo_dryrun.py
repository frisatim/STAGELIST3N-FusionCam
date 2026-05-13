

import tempfile
from multiprocessing import freeze_support
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "yolo_dataset" / "atelier.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "train"


def get_device() -> str:
    print("=" * 60)
    print("  VERIFICATION MATERIELLE")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        total_memory = getattr(device_props, "total_memory", None)
        if total_memory is None:
            total_memory = getattr(device_props, "total_mem")
        vram = total_memory / (1024 ** 3)
        print(f"  [OK] GPU détecté : {gpu_name}")
        print(f"       VRAM : {vram:.1f} Go")
        print(f"       CUDA : {torch.version.cuda}")
        device = "0"
    else:
        print("  [!!] Aucun GPU CUDA détecté.")
        print("       L'entraînement se fera sur CPU (très lent).")
        device = "cpu"

    print("=" * 60)
    return device


def build_resolved_dataset_yaml(dataset_yaml: Path) -> Path:
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = dataset_yaml.parent.resolve()
    yaml_root = Path(data.get("path", "."))
    if not yaml_root.is_absolute():
        yaml_root = (dataset_root / yaml_root).resolve()

    resolved_train = (yaml_root / data["train"]).resolve()
    resolved_val = (yaml_root / data["val"]).resolve()

    if not resolved_train.exists():
        raise FileNotFoundError(f"Dossier train introuvable : {resolved_train}")
    if not resolved_val.exists():
        raise FileNotFoundError(f"Dossier val introuvable : {resolved_val}")

    data["path"] = str(yaml_root)
    data["train"] = str(resolved_train)
    data["val"] = str(resolved_val)

    temp_dir = Path(tempfile.mkdtemp(prefix="yolo_dryrun_", dir=PROJECT_ROOT))
    resolved_yaml = temp_dir / dataset_yaml.name
    with open(resolved_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"  Train   : {resolved_train}")
    print(f"  Val     : {resolved_val}")
    return resolved_yaml


def main() -> None:
    device = get_device()

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATASET_YAML}")

    print(f"\n  Dataset : {DATASET_YAML}")
    print(f"  Sortie  : {RUNS_DIR / 'dry_run_test'}")

    resolved_dataset_yaml = build_resolved_dataset_yaml(DATASET_YAML)

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    print("\n" + "=" * 60)
    print("  LANCEMENT DU DRY RUN (20 epochs, batch=8, imgsz=640)")
    print("=" * 60 + "\n")

    model.train(
        data=str(resolved_dataset_yaml),
        epochs=20,
        imgsz=640,
        batch=8,
        device=device,
        project=str(RUNS_DIR),
        name="dry_run_test",
        exist_ok=True,
    )

    weights_path = RUNS_DIR / "dry_run_test" / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("  DRY RUN TERMINE")
    print("=" * 60)
    print(f"  Poids best.pt : {weights_path}")
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"  Taille        : {size_mb:.1f} Mo")
        print("  [OK] Pipeline validé. Prêt pour l'entraînement complet.")
    else:
        print("  [!!] best.pt non trouvé — vérifier les logs ci-dessus.")
    print("=" * 60)


if __name__ == "__main__":
    freeze_support()
    main()

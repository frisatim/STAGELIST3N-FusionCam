"""
Entraînement comparatif multi-architectures pour la détection d'objets (outils).

Lance séquentiellement l'entraînement de plusieurs modèles Ultralytics
sur le même dataset, chacun dans son propre sous-dossier de résultats.

Usage :
  python train_models.py
  python train_models.py --data mon_dataset/data.yaml --epochs 100 --batch 8
  python train_models.py --models yolov8n yolov8s       # sous-ensemble
"""

import argparse
import gc
import shutil
import time
from multiprocessing import freeze_support
from pathlib import Path

import torch
import yaml

# ── Configuration par défaut ─────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = Path(__file__).resolve().parent
AUTO_COMBINED_DATA = "__AUTO_COMBINED__"
DEFAULT_DATA_YAML = PROJECT_ROOT / "dataset_objets_V4" / "data.yaml"
LEGACY_DATA_YAML = PROJECT_ROOT / "dataset_objets_HD" / "data.yaml"
FALLBACK_DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
DEFAULT_PROJECT = str(BASELINE_DIR / "Modelstrained" / "V4")

FINAL_CLASS_NAMES = [
    "marteau",
    "niveau a bulle",
    "scie",
    "verre",
    "perceuse",
    "bouteille",
    "pince",
    "cutter",
    "metre",
    "tournevis",
    "cle allen",
    "personne",
]

MODELS = [
    # (nom court pour le dossier, fichier poids, type)
    ("yolov8n", "yolov8n.pt", "yolo"),
    ("yolov8s", "yolov8s.pt", "yolo"),
    # ("yolo11n", "yolo11n.pt", "yolo"),
    ("yolo11s", "yolo11s.pt", "yolo"),
    # ("yolo26n", "yolo26n.pt", "yolo"),
    ("rtdetr-l", "rtdetr-l.pt", "rtdetr"),
]


# ── Utilitaires ──────────────────────────────────────────────────────────────

def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / (1024 ** 3)
        print(f"  GPU  : {name}  ({vram:.1f} Go VRAM)")
        print(f"  CUDA : {torch.version.cuda}")
        return "0"
    print("  [!!] Aucun GPU CUDA — entraînement sur CPU (lent).")
    return "cpu"


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def load_model(weights: str, model_type: str):
    """Instancie le bon objet selon le type de modèle.

    Si un checkpoint local .pt est corrompu, on le supprime puis on retente
    (Ultralytics le re-téléchargera automatiquement pour les poids connus).
    """
    if model_type == "rtdetr":
        from ultralytics import RTDETR
        loader = RTDETR
    else:
        from ultralytics import YOLO
        loader = YOLO

    if model_type == "rtdetr":
        pass

    try:
        return loader(weights)
    except Exception as e:
        msg = str(e).lower()
        wpath = Path(weights)
        if "failed finding central directory" in msg and wpath.exists() and wpath.is_file():
            print(f"\n  [ATTENTION] Checkpoint local corrompu: {wpath}")
            print("             Suppression puis nouvelle tentative de chargement...")
            try:
                wpath.unlink()
            except Exception as unlink_err:
                print(f"             Impossible de supprimer {wpath}: {unlink_err}")
                raise
            return loader(weights)
        raise


def _resolve_split_path(split_value, root_dir: Path) -> list[Path]:
    """Normalise train/val/test vers une liste de chemins potentiels."""
    if split_value is None:
        return []
    if isinstance(split_value, (list, tuple)):
        values = split_value
    else:
        values = [split_value]

    paths: list[Path] = []
    for v in values:
        p = Path(str(v))
        if not p.is_absolute():
            p = (root_dir / p).resolve()
        paths.append(p)
    return paths


def validate_dataset_yaml(yaml_path: Path) -> tuple[bool, str]:
    """Vérifie que le YAML existe et que les chemins train/val pointent vers des dossiers/fichiers existants."""
    if not yaml_path.exists():
        return False, f"introuvable: {yaml_path}"

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        return False, f"lecture impossible: {e}"

    data_root = cfg.get("path", ".")
    data_root_path = Path(str(data_root))
    if not data_root_path.is_absolute():
        data_root_path = (yaml_path.parent / data_root_path).resolve()

    train_paths = _resolve_split_path(cfg.get("train"), data_root_path)
    val_paths = _resolve_split_path(cfg.get("val"), data_root_path)

    if not train_paths:
        return False, "champ 'train' manquant ou vide"
    if not val_paths:
        return False, "champ 'val' manquant ou vide"

    train_ok = any(p.exists() for p in train_paths)
    val_ok = any(p.exists() for p in val_paths)

    if not train_ok:
        return False, f"chemin train introuvable: {[str(p) for p in train_paths]}"
    if not val_ok:
        return False, f"chemin val introuvable: {[str(p) for p in val_paths]}"

    return True, "ok"


def read_data_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def class_id_for_name(cfg: dict, class_name: str) -> int | None:
    for idx, name in enumerate(normalize_names(cfg.get("names"))):
        if name == class_name:
            return idx
    return None


def resolve_data_root(source_yaml: Path, cfg: dict) -> Path:
    data_root = cfg.get("path", ".")
    data_root_path = Path(str(data_root))
    if not data_root_path.is_absolute():
        data_root_path = (source_yaml.parent / data_root_path).resolve()
    return data_root_path


def image_to_label_path(image_path: Path, data_root: Path) -> Path:
    """Convertit dataset/images/foo.jpg -> dataset/labels/foo.txt."""
    try:
        rel = image_path.resolve().relative_to(data_root.resolve())
        parts = list(rel.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return data_root.joinpath(*parts).with_suffix(".txt")
    except ValueError:
        pass
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def iter_image_files(paths: list[Path]) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".txt":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    img = Path(line.strip())
                    if img.exists():
                        images.append(img.resolve())
        elif p.is_file() and p.suffix.lower() in exts:
            images.append(p.resolve())
        elif p.is_dir():
            for ext in exts:
                images.extend(sorted(p.rglob(f"*{ext}")))
    return sorted(set(img.resolve() for img in images))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def build_variant_dataset(source_yaml: Path, variant: str,
                          excluded_classes: set[str] | None = None) -> Path:
    """Construit un dataset runtime pour une variante d'entraînement.

    - person_objects: copie/hardlink images + labels identiques.
    - objects_only: retire les lignes des classes exclues et retire ces classes
      du data.yaml. Les IDs restants sont conservés si la classe exclue est la
      dernière, ce qui est le cas de `personne` dans dataset_objets_V4.
    """
    excluded_classes = excluded_classes or set()
    cfg = read_data_yaml(source_yaml)
    data_root = resolve_data_root(source_yaml, cfg)
    names = normalize_names(cfg.get("names"))
    excluded_ids = {
        idx for idx, name in enumerate(names)
        if name in excluded_classes
    }
    kept_names = [name for idx, name in enumerate(names) if idx not in excluded_ids]

    train_paths = _resolve_split_path(cfg.get("train"), data_root)
    val_paths = _resolve_split_path(cfg.get("val"), data_root)
    image_paths = iter_image_files(train_paths + val_paths)

    out_root = PROJECT_ROOT / "00_Workspace_Admin" / "temp_root" / "datasets" / variant
    images_out = out_root / "images"
    labels_out = out_root / "labels"
    if out_root.exists():
        shutil.rmtree(out_root)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    removed_lines = 0
    kept_lines = 0
    missing_labels = 0

    for image_path in image_paths:
        out_image = images_out / image_path.name
        link_or_copy(image_path, out_image)

        src_label = image_to_label_path(image_path, data_root)
        dst_label = labels_out / f"{image_path.stem}.txt"
        if not src_label.exists():
            missing_labels += 1
            dst_label.write_text("", encoding="utf-8")
            continue

        out_lines = []
        with open(src_label, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    continue
                if cls_id in excluded_ids:
                    removed_lines += 1
                    continue
                out_lines.append(stripped)
                kept_lines += 1
        dst_label.write_text(("\n".join(out_lines) + "\n") if out_lines else "",
                             encoding="utf-8")

    out_yaml = out_root / "data.yaml"
    out_cfg = {
        "path": str(out_root.resolve()),
        "train": "images",
        "val": "images",
        "nc": len(kept_names),
        "names": kept_names,
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Dataset runtime {variant}: {out_yaml}")
    print(f"       images={len(image_paths)} labels_keep={kept_lines} "
          f"labels_removed={removed_lines} missing_labels={missing_labels}")
    return out_yaml


def build_combined_data_yaml() -> Path:
    """Construit un YAML combiné (dataset + dataset_objets_HD) pour entraîner sur toutes les images."""
    dataset_train = (PROJECT_ROOT / "dataset" / "images" / "train").resolve()
    dataset_hd_train = (PROJECT_ROOT / "dataset_objets_HD" / "images").resolve()

    cfg = {
        "path": str(PROJECT_ROOT.resolve()),
        "train": [str(dataset_train), str(dataset_hd_train)],
        "val": [str(dataset_train), str(dataset_hd_train)],
        "nc": 12,
        "names": FINAL_CLASS_NAMES,
    }

    runtime_yaml = PROJECT_ROOT / "00_Workspace_Admin" / "temp_root" / "runtime_data_for_training_v3_combined.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(runtime_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return runtime_yaml


def resolve_data_yaml(requested_path: str) -> Path | None:
    """Résout un YAML valide. Fallback auto vers dataset/data.yaml si besoin."""
    if requested_path == AUTO_COMBINED_DATA:
        combined_yaml = build_combined_data_yaml()
        ok, reason = validate_dataset_yaml(combined_yaml)
        if ok:
            print(f"[INFO] Dataset combiné auto : {combined_yaml}")
            return combined_yaml.resolve()
        print(f"[ERREUR] Dataset combiné invalide : {reason}")
        return None

    data_path = Path(requested_path)

    ok, reason = validate_dataset_yaml(data_path)
    if ok:
        return data_path.resolve()

    # Fallback automatique uniquement si l'utilisateur utilise le défaut.
    if data_path.resolve() == DEFAULT_DATA_YAML.resolve():
        for fallback in (LEGACY_DATA_YAML, FALLBACK_DATA_YAML):
            if not fallback.exists():
                continue
            ok_fallback, reason_fallback = validate_dataset_yaml(fallback)
            if ok_fallback:
                print(f"[ATTENTION] Dataset par défaut invalide ({reason}).")
                print(f"            Bascule automatique vers : {fallback}")
                return fallback.resolve()
            print(f"[WARN] Fallback invalide {fallback}: {reason_fallback}")
        print(f"[ERREUR] Dataset par défaut invalide ({reason}).")
        return None

    print(f"[ERREUR] Dataset invalide : {data_path}")
    print(f"         Détail : {reason}")
    return None


def build_runtime_data_yaml(source_yaml: Path) -> Path:
    """Écrit un YAML temporaire avec `path` absolu pour éviter les ambiguïtés Ultralytics."""
    with open(source_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_root = cfg.get("path", ".")
    data_root_path = Path(str(data_root))
    if not data_root_path.is_absolute():
        data_root_path = (source_yaml.parent / data_root_path).resolve()

    cfg["path"] = str(data_root_path)

    runtime_yaml = PROJECT_ROOT / "00_Workspace_Admin" / "temp_root" / "runtime_data_for_training.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(runtime_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return runtime_yaml


def is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda err" in msg or "acceleratorerror" in msg


def safe_cuda_cleanup() -> None:
    """Nettoyage best-effort après OOM CUDA.

    Sur Windows, après certains `torch.AcceleratorError`, même
    torch.cuda.empty_cache() peut échouer. Le nettoyage ne doit donc jamais
    faire planter le script d'orchestration.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"             [WARN] torch.cuda.empty_cache() a échoué: {e}")


def parse_batch_overrides(raw: str) -> dict[str, int]:
    """Parse `yolo11s=2,rtdetr-l=1` vers un dict."""
    overrides: dict[str, int] = {}
    if not raw:
        return overrides
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override batch invalide: {item!r}, attendu modele=batch")
        name, value = item.split("=", 1)
        overrides[name.strip()] = int(value.strip())
    return overrides


# ── Entraînement ─────────────────────────────────────────────────────────────

def train_one(name: str, weights: str, model_type: str,
              data: str, epochs: int, imgsz: int, batch: int,
              project: str, device: str, min_batch: int = 2,
              oom_retries: int = 3, export_engine: bool = False,
              engine_workspace: int | None = None,
              skip_existing: bool = False) -> dict:
    """Entraîne un modèle et retourne un résumé."""
    print(f"\n{'=' * 60}")
    print(f"  MODÈLE : {name}  ({weights})")
    print(f"  Type   : {model_type.upper()}")
    print(f"  Sortie : {project}/{name}/")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    attempts = 0
    current_batch = batch
    train_error = None
    best_pt = Path(project) / name / "weights" / "best.pt"
    skipped_training = False

    if skip_existing and best_pt.exists():
        skipped_training = True
        print(f"  [SKIP] best.pt existe déjà : {best_pt}")

    while not skipped_training and attempts <= oom_retries:
        attempts += 1
        model = None
        try:
            model = load_model(weights, model_type)
            model.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                batch=current_batch,
                device=device,
                project=project,
                name=name,
                exist_ok=True,
                # ── Data Augmentation agressive (Anti-Domain Shift) ─
                mosaic=1.0,            # Mosaic 4-en-1 systématique : plus de contextes variés
                mixup=0.2,             # MixUp 20% : force le modèle à trouver des formes dans le bruit
                degrees=15.0,          # Rotation ±15° : les outils ne sont jamais parfaitement droits
                hsv_s=0.2,             # Variation saturation : reflets métalliques, éclairage variable
                hsv_v=0.2,             # Variation luminosité : compense le domain shift RTSP sombre
                # ── Sécurité ────────────────────────────────────────
                patience=30,           # Early Stopping : arrêt si pas d'amélioration pendant 30 epochs
            )
            train_error = None
            break
        except Exception as e:
            train_error = e
            if is_cuda_oom_error(e) and device != "cpu" and current_batch > min_batch:
                next_batch = max(min_batch, current_batch // 2)
                print(f"\n  [ATTENTION] OOM sur {name} avec batch={current_batch}.")
                print(f"             Nouvelle tentative avec batch={next_batch}...")
                current_batch = next_batch
                try:
                    del model
                except Exception:
                    pass
                safe_cuda_cleanup()
                continue
            break

    elapsed = time.time() - t0

    result = {
        "name": name,
        "weights": weights,
        "duration": elapsed,
        "batch_used": current_batch,
        "attempts": attempts,
        "best_pt": str(best_pt),
        "best_exists": best_pt.exists(),
        "status": "OK" if train_error is None and best_pt.exists() else "ECHEC",
        "error": "" if train_error is None else str(train_error),
        "engine_path": "",
        "engine_exists": False,
        "engine_error": "",
        "skipped_training": skipped_training,
    }

    if skipped_training and best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 ** 2)
        result["size_mb"] = size_mb
        print(f"\n  [OK] {name} réutilisé — best.pt = {size_mb:.1f} Mo")
    elif train_error is not None:
        print(f"\n  [ECHEC] {name} après {format_duration(elapsed)}")
        print(f"          Raison: {train_error}")
    elif best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 ** 2)
        result["size_mb"] = size_mb
        print(f"\n  [OK] {name} terminé en {format_duration(elapsed)} "
              f"— best.pt = {size_mb:.1f} Mo (batch={current_batch})")
    else:
        result["status"] = "ECHEC"
        print(f"\n  [!!] {name} terminé en {format_duration(elapsed)} "
              f"— best.pt NON TROUVÉ")

    if export_engine and result["status"] == "OK" and best_pt.exists():
        try:
            print(f"\n  [EXPORT] TensorRT FP32 engine pour {name}...")
            model = load_model(str(best_pt), model_type)
            export_kwargs = {
                "format": "engine",
                "imgsz": imgsz,
                "device": device,
                "half": False,
            }
            if engine_workspace is not None:
                export_kwargs["workspace"] = engine_workspace
            exported = model.export(**export_kwargs)
            engine_path = Path(exported) if exported else best_pt.with_suffix(".engine")
            result["engine_path"] = str(engine_path)
            result["engine_exists"] = engine_path.exists()
            if engine_path.exists():
                size_mb = engine_path.stat().st_size / (1024 ** 2)
                print(f"  [OK] engine exporté : {engine_path} ({size_mb:.1f} Mo)")
            else:
                print(f"  [WARN] Export terminé mais engine introuvable : {engine_path}")
        except Exception as e:
            result["engine_error"] = str(e)
            print(f"  [WARN] Export engine échoué pour {name}: {e}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Entraînement comparatif multi-architectures (Ultralytics).")
    p.add_argument("--data", "-d", default=str(DEFAULT_DATA_YAML),
                   help=f"Chemin vers data.yaml (défaut: {DEFAULT_DATA_YAML})")
    p.add_argument("--epochs", "-e", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", "-b", type=int, default=16)
    p.add_argument("--min-batch", type=int, default=2,
                   help="Batch minimal en cas de retry OOM (défaut: 2)")
    p.add_argument("--oom-retries", type=int, default=3,
                   help="Nombre max de retries OOM par modèle (défaut: 3)")
    p.add_argument("--project", default=DEFAULT_PROJECT,
                   help=f"Dossier racine des résultats (défaut: {DEFAULT_PROJECT})")
    p.add_argument("--models", nargs="+", default=None,
                   help="Sous-ensemble de modèles à entraîner "
                        "(ex: --models yolov8n yolo11s rtdetr-l)")
    p.add_argument("--variants", nargs="+",
                   default=["person_objects", "objects_only"],
                   choices=["person_objects", "objects_only"],
                   help="Variantes à entraîner: toutes classes ou objets seuls")
    p.add_argument("--no-export-engine", action="store_true",
                   help="Ne pas exporter en TensorRT .engine après entraînement")
    p.add_argument("--engine-workspace", type=int, default=None,
                   help="Workspace TensorRT en GiB si supporté par Ultralytics")
    p.add_argument("--skip-existing", action="store_true",
                   help="Ne réentraîne pas les modèles dont weights/best.pt existe déjà; "
                        "exporte l'engine si nécessaire.")
    p.add_argument("--batch-overrides", default="",
                   help="Batch par modèle, ex: yolo11s=2,rtdetr-l=1. "
                        "Remplace --batch pour ces modèles.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        batch_overrides = parse_batch_overrides(args.batch_overrides)
    except ValueError as e:
        print(f"[ERREUR] {e}")
        return

    data_path = resolve_data_yaml(args.data)
    if data_path is None:
        print("         Passe --data <chemin_vers_un_data.yaml_valide>")
        return

    variant_data = {}
    if "person_objects" in args.variants:
        variant_data["person_objects"] = build_variant_dataset(
            data_path,
            "person_objects",
            excluded_classes=set(),
        )
    if "objects_only" in args.variants:
        cfg = read_data_yaml(data_path)
        if class_id_for_name(cfg, "personne") is None:
            print("[WARN] Classe 'personne' absente du dataset, objects_only = person_objects")
            excluded = set()
        else:
            excluded = {"personne"}
        variant_data["objects_only"] = build_variant_dataset(
            data_path,
            "objects_only",
            excluded_classes=excluded,
        )

    # Filtrer les modèles si --models est utilisé
    models = MODELS
    if args.models:
        selected = set(args.models)
        models = [m for m in MODELS if m[0] in selected]
        unknown = selected - {m[0] for m in models}
        if unknown:
            print(f"[ATTENTION] Modèles inconnus ignorés : {unknown}")
            print(f"  Disponibles : {[m[0] for m in MODELS]}")
        if not models:
            print("[ERREUR] Aucun modèle valide sélectionné.")
            return

    print("\n" + "=" * 60)
    print("  BENCHMARK ENTRAÎNEMENT — DÉTECTION D'OBJETS (OUTILS)")
    print("=" * 60)
    print(f"  Dataset  : {data_path}")
    print(f"  Sortie   : {args.project}/")
    print(f"  Epochs   : {args.epochs}")
    print(f"  ImgSize  : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    if batch_overrides:
        print(f"  Batch/model : {batch_overrides}")
    print(f"  OOM      : retries={args.oom_retries}, min_batch={args.min_batch}")
    print(f"  Modèles  : {[m[0] for m in models]}")
    print(f"  Variants : {list(variant_data)}")
    print(f"  Export   : {'best.pt + fp32.engine' if not args.no_export_engine else 'best.pt seulement'}")

    device = detect_device()
    print("=" * 60)

    results = []
    total_t0 = time.time()

    total_jobs = len(models) * len(variant_data)
    job_idx = 0
    for variant, variant_yaml in variant_data.items():
        runtime_data_yaml = build_runtime_data_yaml(variant_yaml)
        variant_project = str(Path(args.project) / variant)
        for name, weights, mtype in models:
            job_idx += 1
            print(f"\n{'─' * 60}")
            print(f"  [{job_idx}/{total_jobs}] Lancement de {variant}/{name}...")
            print(f"{'─' * 60}")

            model_batch = batch_overrides.get(name, args.batch)
            result = train_one(
                name=name,
                weights=weights,
                model_type=mtype,
                data=str(runtime_data_yaml.resolve()),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=model_batch,
                project=variant_project,
                device=device,
                min_batch=args.min_batch,
                oom_retries=args.oom_retries,
                export_engine=not args.no_export_engine,
                engine_workspace=args.engine_workspace,
                skip_existing=args.skip_existing,
            )
            result["variant"] = variant
            results.append(result)

    total_elapsed = time.time() - total_t0

    # ── Récapitulatif final ──────────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print("  RÉCAPITULATIF")
    print(f"{'=' * 60}")
    print(f"  {'Variant':<15s}  {'Modèle':<12s}  {'Durée':>10s}  {'best.pt':>10s}  {'Engine':>8s}  {'Batch':>5s}  {'Status'}")
    print(f"  {'─' * 15}  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 8}  {'─' * 5}  {'─' * 8}")

    for r in results:
        dur = format_duration(r["duration"])
        if r["best_exists"] and r["status"] == "OK":
            size = f"{r['size_mb']:.1f} Mo"
            status = "OK"
        else:
            size = "—"
            status = "ECHEC"
        engine = "OK" if r.get("engine_exists") else ("skip" if args.no_export_engine else "—")
        print(f"  {r.get('variant',''):<15s}  {r['name']:<12s}  {dur:>10s}  {size:>10s}  {engine:>8s}  {r['batch_used']:>5d}  {status}")

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print("\n  Détail des échecs :")
        for r in failed:
            err = r.get("error", "")
            short_err = err.splitlines()[0][:140] if err else "Erreur inconnue"
            print(f"    - {r.get('variant','')}/{r['name']}: {short_err}")

    engine_failed = [r for r in results if r["status"] == "OK"
                     and not args.no_export_engine
                     and not r.get("engine_exists")]
    if engine_failed:
        print("\n  Exports engine non produits :")
        for r in engine_failed:
            err = r.get("engine_error", "")
            short_err = err.splitlines()[0][:140] if err else "engine introuvable"
            print(f"    - {r.get('variant','')}/{r['name']}: {short_err}")

    print(f"\n  Durée totale : {format_duration(total_elapsed)}")
    print(f"  Résultats    : {args.project}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    freeze_support()
    main()

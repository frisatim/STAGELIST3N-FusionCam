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
import time
from multiprocessing import freeze_support
from pathlib import Path

import torch
import yaml

# ── Configuration par défaut ─────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = Path(__file__).resolve().parent
AUTO_COMBINED_DATA = "__AUTO_COMBINED__"
DEFAULT_DATA_YAML = PROJECT_ROOT / "dataset_objets_HD" / "data.yaml"
FALLBACK_DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
DEFAULT_PROJECT = str(BASELINE_DIR / "Modelstrained" / "V3")

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
    # ("yolov8n", "yolov8n.pt", "yolo"),
    # ("yolov8s", "yolov8s.pt", "yolo"),
    # ("yolo11n", "yolo11n.pt", "yolo"),
    # ("yolo11s", "yolo11s.pt", "yolo"),
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
    if data_path.resolve() == DEFAULT_DATA_YAML.resolve() and FALLBACK_DATA_YAML.exists():
        ok_fallback, reason_fallback = validate_dataset_yaml(FALLBACK_DATA_YAML)
        if ok_fallback:
            print(f"[ATTENTION] Dataset par défaut invalide ({reason}).")
            print(f"            Bascule automatique vers : {FALLBACK_DATA_YAML}")
            return FALLBACK_DATA_YAML.resolve()
        print(f"[ERREUR] Dataset par défaut invalide ({reason}).")
        print(f"         Fallback invalide aussi ({reason_fallback}).")
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


# ── Entraînement ─────────────────────────────────────────────────────────────

def train_one(name: str, weights: str, model_type: str,
              data: str, epochs: int, imgsz: int, batch: int,
              project: str, device: str, min_batch: int = 2,
              oom_retries: int = 3) -> dict:
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

    while attempts <= oom_retries:
        attempts += 1
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            break

    elapsed = time.time() - t0
    best_pt = Path(project) / name / "weights" / "best.pt"

    result = {
        "name": name,
        "weights": weights,
        "duration": elapsed,
        "batch_used": current_batch,
        "attempts": attempts,
        "best_pt": str(best_pt),
        "best_exists": best_pt.exists(),
        "status": "OK" if train_error is None else "ECHEC",
        "error": "" if train_error is None else str(train_error),
    }

    if train_error is not None:
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

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Entraînement comparatif multi-architectures (Ultralytics).")
    p.add_argument("--data", "-d", default=AUTO_COMBINED_DATA,
                   help=("Chemin vers data.yaml. Défaut: dataset combiné auto "
                         "(dataset/images/train + dataset_objets_HD/images)."))
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
    return p.parse_args()


def main():
    args = parse_args()

    data_path = resolve_data_yaml(args.data)
    if data_path is None:
        print("         Passe --data <chemin_vers_un_data.yaml_valide>")
        return

    runtime_data_yaml = build_runtime_data_yaml(data_path)

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
    print(f"  Runtime  : {runtime_data_yaml}")
    print(f"  Sortie   : {args.project}/")
    print(f"  Epochs   : {args.epochs}")
    print(f"  ImgSize  : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  OOM      : retries={args.oom_retries}, min_batch={args.min_batch}")
    print(f"  Modèles  : {[m[0] for m in models]}")

    device = detect_device()
    print("=" * 60)

    results = []
    total_t0 = time.time()

    for i, (name, weights, mtype) in enumerate(models, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(models)}] Lancement de {name}...")
        print(f"{'─' * 60}")

        result = train_one(
            name=name,
            weights=weights,
            model_type=mtype,
            data=str(runtime_data_yaml.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            device=device,
            min_batch=args.min_batch,
            oom_retries=args.oom_retries,
        )
        results.append(result)

    total_elapsed = time.time() - total_t0

    # ── Récapitulatif final ──────────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print("  RÉCAPITULATIF")
    print(f"{'=' * 60}")
    print(f"  {'Modèle':<12s}  {'Durée':>10s}  {'best.pt':>10s}  {'Batch':>5s}  {'Status'}")
    print(f"  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 5}  {'─' * 8}")

    for r in results:
        dur = format_duration(r["duration"])
        if r["best_exists"] and r["status"] == "OK":
            size = f"{r['size_mb']:.1f} Mo"
            status = "OK"
        else:
            size = "—"
            status = "ECHEC"
        print(f"  {r['name']:<12s}  {dur:>10s}  {size:>10s}  {r['batch_used']:>5d}  {status}")

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print("\n  Détail des échecs :")
        for r in failed:
            err = r.get("error", "")
            short_err = err.splitlines()[0][:140] if err else "Erreur inconnue"
            print(f"    - {r['name']}: {short_err}")

    print(f"\n  Durée totale : {format_duration(total_elapsed)}")
    print(f"  Résultats    : {args.project}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    freeze_support()
    main()

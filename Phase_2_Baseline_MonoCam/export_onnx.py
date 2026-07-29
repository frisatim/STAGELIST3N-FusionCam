"""
Export et optimisation des modeles entraines (.pt -> .onnx / .engine).

Parcourt automatiquement le dossier Modelstrained/V2/ (ou un dossier custom)
et exporte chaque best.pt dans le format choisi. Les fichiers .pt sont
toujours conserves intacts.

Formats supportes :
  - onnx    : graphe statique optimise, portable CPU/GPU (defaut)
  - engine  : TensorRT, optimise pour GPU Nvidia (RTX, Jetson...)

Optimisations optionnelles :
  - --half   : precision FP16 (divise la taille, accelere l'inference GPU)
  - --prune  : elagage des poids proches de zero avant export

Usage :
  python export_onnx.py
  python export_onnx.py --format engine --half
  python export_onnx.py --prune 0.3
  python export_onnx.py --format engine --half --prune 0.2 --models yolov8n
  python export_onnx.py --project ../runs_comparaison_outils --imgsz 1280
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT = BASELINE_DIR / "Modelstrained" / "V2"

# Noms de dossiers correspondant a des modeles RT-DETR (pas YOLO)
RTDETR_NAMES = {"rtdetr-l", "rtdetr-x", "rtdetr-resnet50", "rtdetr-resnet101"}

# Extensions produites par chaque format d'export
FORMAT_EXTENSIONS = {
    "onnx": ".onnx",
    "engine": ".engine",
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def discover_models(project: Path, filter_names: list[str] | None = None) -> list[Path]:
    """Trouve tous les best.pt dans les sous-dossiers du projet."""
    candidates = sorted(project.glob("*/weights/best.pt"))
    if filter_names:
        selected = set(filter_names)
        candidates = [p for p in candidates if p.parent.parent.name in selected]
    return candidates


def is_rtdetr(model_dir_name: str) -> bool:
    return model_dir_name.lower() in RTDETR_NAMES


def build_suffix(suffix_arg: str | None, half: bool, prune_fraction: float | None) -> str:
    """Construit le suffixe de fichier a partir des options.

    - suffix_arg=None          -> "" (pas de renommage)
    - suffix_arg="auto"        -> "_pruned30_fp16" (genere depuis les flags)
    - suffix_arg="mon_test"    -> "_mon_test" (custom)
    """
    if suffix_arg is None:
        return ""

    if suffix_arg != "auto":
        return f"_{suffix_arg}"

    # Generation automatique depuis les flags actifs
    parts = []
    if prune_fraction is not None and prune_fraction > 0:
        parts.append(f"pruned{int(prune_fraction * 100)}")
    if half:
        parts.append("fp16")

    if not parts:
        parts.append("export")  # fallback si --suffix sans --half ni --prune

    return "_" + "_".join(parts)


def load_model(pt_path: Path):
    """Instancie le bon objet selon le nom du dossier parent."""
    model_name = pt_path.parent.parent.name
    if is_rtdetr(model_name):
        from ultralytics import RTDETR
        return RTDETR(str(pt_path))
    else:
        from ultralytics import YOLO
        return YOLO(str(pt_path))


# ══════════════════════════════════════════════════════════════════════════════
#  PRUNING (ELAGAGE)
# ══════════════════════════════════════════════════════════════════════════════
#
# Le pruning met a zero les poids les plus faibles du reseau avant l'export.
# Concretement, pour chaque couche convolutive (Conv2d), on classe tous les
# poids par valeur absolue et on force a zero les X% les plus petits.
#
# Resultat : le reseau devient "sparse" (creux). Les multiplications par zero
# sont triviales, ce qui peut accelerer l'inference sur des runtimes qui
# exploitent la sparsite (TensorRT avec structured sparsity sur Ampere+,
# ONNX Runtime avec sparseML, etc.).
#
# Attention : le pruning degrade legerement la precision. 20-30% est un bon
# compromis ; au-dela de 50% la qualite de detection chute significativement.
# ══════════════════════════════════════════════════════════════════════════════

def apply_pruning(model, fraction: float) -> tuple[int, int]:
    """Applique un elagage L1 non-structure sur toutes les couches Conv2d.

    Parcourt le reseau PyTorch interne du modele Ultralytics, identifie
    chaque Conv2d, et met a zero les poids dont la valeur absolue est
    dans les `fraction` % les plus bas.

    Retourne (nb_couches_elaguees, nb_total_poids_mis_a_zero).
    """
    # Le module PyTorch brut est dans model.model
    pytorch_module = model.model

    pruned_layers = 0
    total_zeroed = 0
    total_params = 0

    for name, module in pytorch_module.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        # Nombre de poids dans cette couche avant pruning
        n_params = module.weight.nelement()
        total_params += n_params

        # Appliquer le pruning L1 non-structure :
        # classe les poids par |valeur| et masque les `fraction` plus faibles
        prune.l1_unstructured(module, name="weight", amount=fraction)

        # Rendre le pruning permanent : retire le masque et ecrit les zeros
        # directement dans le tenseur .weight (necessaire pour que l'export
        # ONNX/TensorRT voie les vrais zeros, pas un hook PyTorch)
        prune.remove(module, "weight")

        n_zeros = int((module.weight == 0).sum().item())
        total_zeroed += n_zeros
        pruned_layers += 1

    return pruned_layers, total_zeroed


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION TENSORRT
# ══════════════════════════════════════════════════════════════════════════════
#
# TensorRT est le moteur d'inference exclusif Nvidia. Il analyse l'architecture
# physique du GPU (nombre de cores, taille des caches, bande passante memoire)
# pour fusionner les couches du reseau et generer un plan d'execution optimal.
#
# Le fichier .engine produit est donc specifique a la carte GPU sur laquelle
# il est genere un .engine cree sur RTX 4070 ne fonctionnera pas sur Jetson
# et inversement. Il faut re-exporter sur chaque cible.
#
# Prerequis : pip install tensorrt (+ CUDA toolkit installe)
# ══════════════════════════════════════════════════════════════════════════════

def check_tensorrt_available() -> bool:
    """Verifie que TensorRT est installe et utilisable."""
    try:
        import tensorrt  # noqa: F401
        print(f"  TensorRT : v{tensorrt.__version__} detecte")
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT D'UN MODELE
# ══════════════════════════════════════════════════════════════════════════════

def export_one(pt_path: Path, fmt: str, imgsz: int,
               half: bool, prune_fraction: float | None,
               suffix: str = "") -> dict:
    """Charge un modele, applique les optimisations, et exporte.

    Pipeline :
      1. Chargement du .pt (YOLO ou RTDETR selon le dossier)
      2. Pruning optionnel (si prune_fraction > 0)
      3. Export dans le format demande (onnx ou engine)
    """
    model_name = pt_path.parent.parent.name
    ext = FORMAT_EXTENSIONS[fmt]
    t0 = time.time()

    try:
        # ── Etape 1 : Chargement ─────────────────────────────────────────
        print(f"    Chargement de {pt_path.name}...")
        model = load_model(pt_path)

        # ── Etape 2 : Pruning optionnel ──────────────────────────────────
        if prune_fraction is not None and prune_fraction > 0:
            print(f"    Elagage en cours ({prune_fraction * 100:.0f}% des poids)...")
            n_layers, n_zeros = apply_pruning(model, prune_fraction)
            print(f"    Elagage termine : {n_layers} couches Conv2d, "
                  f"{n_zeros:,} poids mis a zero")

        # ── Etape 3 : Export ─────────────────────────────────────────────
        export_kwargs = {"format": fmt, "imgsz": imgsz}

        if fmt == "onnx":
            # Export ONNX : graphe statique, portable partout
            if half:
                export_kwargs["half"] = True
                print(f"    Export ONNX (FP16)...")
            else:
                print(f"    Export ONNX (FP32)...")

        elif fmt == "engine":
            # Export TensorRT : optimise pour le GPU Nvidia present
            if half:
                export_kwargs["half"] = True
                print(f"    Export TensorRT (FP16)...")
            else:
                print(f"    Export TensorRT (FP32)...")

        exported_path = model.export(**export_kwargs)

        # ── Etape 4 : Renommage optionnel (--suffix) ────────────────────
        # Ultralytics exporte toujours vers best.onnx / best.engine.
        # Si un suffixe est demande, on renomme pour preserver l'ancien fichier.
        # Ex: best.onnx -> best_pruned30.onnx
        out_file = Path(exported_path)
        if suffix and out_file.exists():
            new_name = f"{out_file.stem}{suffix}{out_file.suffix}"
            new_path = out_file.parent / new_name
            out_file.rename(new_path)
            out_file = new_path
            exported_path = str(new_path)

        elapsed = time.time() - t0
        size_mb = out_file.stat().st_size / (1024 ** 2) if out_file.exists() else 0

        suffix = ""
        if prune_fraction:
            suffix += f" pruned={prune_fraction * 100:.0f}%"
        if half:
            suffix += " FP16"

        print(f"  [OK] {model_name} -> {out_file.name} "
              f"({size_mb:.1f} Mo, {format_duration(elapsed)}{suffix})")

        return {
            "name": model_name,
            "pt_path": str(pt_path),
            "export_path": str(exported_path),
            "format": fmt,
            "size_mb": size_mb,
            "duration": elapsed,
            "half": half,
            "pruned": prune_fraction or 0,
            "status": "OK",
            "error": "",
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ECHEC] {model_name} -- {e}")
        return {
            "name": model_name,
            "pt_path": str(pt_path),
            "export_path": "",
            "format": fmt,
            "size_mb": 0,
            "duration": elapsed,
            "half": half,
            "pruned": prune_fraction or 0,
            "status": "ECHEC",
            "error": str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Export et optimisation des modeles entraines (.pt -> onnx/engine).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python export_onnx.py                          # ONNX FP32 (defaut)
  python export_onnx.py --half                   # ONNX FP16
  python export_onnx.py --format engine --half   # TensorRT FP16
  python export_onnx.py --prune 0.3              # Elagage 30%% + ONNX
  python export_onnx.py --format engine --half --prune 0.2 --models yolov8n
  python export_onnx.py --prune 0.3 --suffix          # -> best_pruned30.onnx
  python export_onnx.py --half --suffix                # -> best_fp16.onnx
  python export_onnx.py --prune 0.3 --half --suffix    # -> best_pruned30_fp16.onnx
  python export_onnx.py --suffix mon_test              # -> best_mon_test.onnx
""")
    p.add_argument("--project", "-p", default=str(DEFAULT_PROJECT),
                   help=f"Dossier racine contenant les modeles (defaut: {DEFAULT_PROJECT})")
    p.add_argument("--models", nargs="+", default=None,
                   help="Sous-ensemble de modeles a exporter (ex: --models yolov8n yolo11s)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Resolution d'image pour l'export (defaut: 640)")
    p.add_argument("--format", "-f", choices=["onnx", "engine"], default="onnx",
                   help="Format de sortie : onnx (defaut) ou engine (TensorRT)")
    p.add_argument("--half", action="store_true",
                   help="Activer la precision FP16 (reduit la taille, accelere sur GPU)")
    p.add_argument("--prune", type=float, default=None, metavar="FRACTION",
                   help="Fraction d'elagage (0.0-1.0). Ex: 0.3 = coupe 30%% des poids faibles")
    p.add_argument("--suffix", nargs="?", const="auto", default=None,
                   help="Renomme la sortie pour ne pas ecraser l'existant. "
                        "Sans valeur : auto-genere (ex: best_pruned30_fp16.onnx). "
                        "Avec valeur : suffixe custom (ex: --suffix mon_test -> best_mon_test.onnx)")
    return p.parse_args()


def main():
    args = parse_args()
    project = Path(args.project)
    fmt = args.format
    ext = FORMAT_EXTENSIONS[fmt]

    # ── Validations ──────────────────────────────────────────────────────

    if not project.exists():
        print(f"[ERREUR] Dossier introuvable : {project}")
        return

    # Verifier TensorRT si necessaire
    if fmt == "engine":
        if not torch.cuda.is_available():
            print("[ERREUR] Export TensorRT impossible : aucun GPU CUDA detecte.")
            print("         TensorRT necessite un GPU Nvidia avec CUDA.")
            print("         Utilisez --format onnx pour un export CPU-compatible.")
            return
        if not check_tensorrt_available():
            print("[ERREUR] Le module 'tensorrt' n'est pas installe.")
            print("         Pour l'installer :")
            print("           pip install tensorrt")
            print("         Prerequis : CUDA Toolkit + cuDNN installes.")
            print("         Ou utilisez --format onnx pour exporter sans TensorRT.")
            return

    # Valider la fraction de pruning
    if args.prune is not None:
        if not 0.0 < args.prune < 1.0:
            print(f"[ERREUR] --prune doit etre entre 0.0 et 1.0 (recu: {args.prune})")
            return
        if args.prune > 0.5:
            print(f"  [ATTENTION] Pruning a {args.prune * 100:.0f}% : la precision de")
            print(f"              detection sera probablement degradee. 20-30%% recommande.")

    # Construire le suffixe de renommage (vide si --suffix pas utilise)
    suffix = build_suffix(args.suffix, args.half, args.prune)

    # ── Decouverte des modeles ───────────────────────────────────────────

    models = discover_models(project, args.models)

    if not models:
        print(f"[ERREUR] Aucun best.pt trouve dans {project}/*/weights/")
        if args.models:
            print(f"         Filtre applique : {args.models}")
        return

    # ── Affichage de la configuration ────────────────────────────────────

    format_label = "ONNX" if fmt == "onnx" else "TensorRT (.engine)"
    prune_label = f"{args.prune * 100:.0f}%" if args.prune else "non"
    precision = "FP16" if args.half else "FP32"

    print("\n" + "=" * 60)
    print(f"  EXPORT .PT -> {ext.upper()}")
    print("=" * 60)
    print(f"  Dossier   : {project}")
    print(f"  Format    : {format_label}")
    print(f"  Precision : {precision}")
    print(f"  Pruning   : {prune_label}")
    print(f"  ImgSize   : {args.imgsz}")
    print(f"  Modeles   : {[p.parent.parent.name for p in models]}")
    if suffix:
        print(f"  Suffixe   : best{suffix}{ext}")
    print(f"  Note      : les fichiers .pt sont conserves intacts")
    print("=" * 60)

    # ── Boucle d'export ──────────────────────────────────────────────────

    results = []
    total_t0 = time.time()

    for i, pt_path in enumerate(models, 1):
        name = pt_path.parent.parent.name
        print(f"\n  [{i}/{len(models)}] Export de {name}...")
        result = export_one(
            pt_path=pt_path,
            fmt=fmt,
            imgsz=args.imgsz,
            half=args.half,
            prune_fraction=args.prune,
            suffix=suffix,
        )
        results.append(result)

    total_elapsed = time.time() - total_t0

    # ── Recapitulatif ────────────────────────────────────────────────────

    col_format = fmt.upper()
    print(f"\n\n{'=' * 60}")
    print("  RECAPITULATIF")
    print(f"{'=' * 60}")
    print(f"  {'Modele':<12s}  {'Duree':>10s}  {col_format:>10s}  {'Status'}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    for r in results:
        dur = format_duration(r["duration"])
        if r["status"] == "OK":
            size = f"{r['size_mb']:.1f} Mo"
        else:
            size = "---"
        print(f"  {r['name']:<12s}  {dur:>10s}  {size:>10s}  {r['status']}")

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print("\n  Detail des echecs :")
        for r in failed:
            err = r.get("error", "")
            short_err = err.splitlines()[0][:140] if err else "Erreur inconnue"
            print(f"    - {r['name']}: {short_err}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\n  Exportes : {ok_count}/{len(results)}")
    print(f"  Duree totale : {format_duration(total_elapsed)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""Verify that the repository and external data layout can run experiments.

The project intentionally keeps heavy assets outside Git. This script checks
both the standard external layout and the historical in-repository paths used by
the research scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_V4_RECORDINGS = {
    "cam_02": "Camera_2_2.3_20260506_131002.mp4",
    "cam_03": "Camera_3_2.4_20260506_131002.mp4",
    "cam_05": "Camera_5_2.6_20260506_131002.mp4",
    "cam_07": "Camera_7_2.11_20260506_131002.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FusionCam data layout.")
    parser.add_argument(
        "--data-dir",
        default="../STAGELIST3N-FusionCam-data",
        help="External heavy-data directory.",
    )
    parser.add_argument("--model-version", default="V4")
    parser.add_argument("--model", default="yolov8s")
    parser.add_argument(
        "--require-engine",
        action="store_true",
        help="Require TensorRT best.engine in addition to best.pt.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_path(label: str, path: Path, *, required: bool = True) -> bool:
    ok = path.exists()
    marker = "OK" if ok else ("MISS" if required else "optional")
    print(f"[{marker:8}] {label}: {path}")
    return ok or not required


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> int:
    args = parse_args()
    root = repo_root()
    data_root = (root / args.data_dir).resolve()
    failures: list[str] = []

    print(f"Repository: {root}")
    print(f"Data root : {data_root}")
    print("")

    def require(label: str, path: Path) -> None:
        if not check_path(label, path, required=True):
            failures.append(label)

    def optional(label: str, path: Path) -> None:
        check_path(label, path, required=False)

    print("Ground truths")
    require("ground_truth/gt_people.json", root / "ground_truth" / "gt_people.json")
    gt_tad = first_existing(
        [
            root / "dataset_objets_HD" / "gt_objects_tad.json",
            root / "ground_truth" / "gt_objects_tad_dataset_objets_HD.json",
            root / "ground_truth" / "gt_objects_tad.json",
        ]
    )
    if gt_tad:
        check_path("TAD GT usable by scripts", gt_tad)
    else:
        failures.append("TAD GT usable by scripts")
        check_path("TAD GT usable by scripts", root / "dataset_objets_HD" / "gt_objects_tad.json")
    print("")

    print("External data layout")
    require("external data dir", data_root)
    optional("datasets/dataset", data_root / "datasets" / "dataset")
    optional("datasets/dataset_objets_HD", data_root / "datasets" / "dataset_objets_HD")
    optional("datasets/dataset_objets_V4", data_root / "datasets" / "dataset_objets_V4")
    optional("recordings/recordings", data_root / "recordings" / "recordings")
    optional("models", data_root / "models")
    print("")

    print("Recorded V4 videos")
    for camera, filename in REQUIRED_V4_RECORDINGS.items():
        require(
            f"{camera} external recording",
            data_root / "recordings" / "recordings" / filename,
        )
        require(
            f"{camera} legacy recording",
            root / "recordings" / "recordings" / filename,
        )
    print("")

    print("Model artifacts")
    model_rel = Path(args.model_version) / "person_objects" / args.model / "weights"
    require(
        f"{args.model_version}/{args.model} best.pt external",
        data_root / "models" / model_rel / "best.pt",
    )
    require(
        f"{args.model_version}/{args.model} best.pt legacy",
        root / "Phase_2_Baseline_MonoCam" / "Modelstrained" / model_rel / "best.pt",
    )
    engine_external = data_root / "models" / model_rel / "best.engine"
    engine_legacy = root / "Phase_2_Baseline_MonoCam" / "Modelstrained" / model_rel / "best.engine"
    if args.require_engine:
        require(f"{args.model_version}/{args.model} best.engine external", engine_external)
        require(f"{args.model_version}/{args.model} best.engine legacy", engine_legacy)
    else:
        optional(f"{args.model_version}/{args.model} best.engine external", engine_external)
        optional(f"{args.model_version}/{args.model} best.engine legacy", engine_legacy)
    print("")

    print("Legacy paths expected by scripts")
    optional("dataset", root / "dataset")
    optional("dataset_objets_HD", root / "dataset_objets_HD")
    optional("dataset_objets_V4", root / "dataset_objets_V4")
    require("recordings", root / "recordings")
    require("Phase_2_Baseline_MonoCam/Modelstrained", root / "Phase_2_Baseline_MonoCam" / "Modelstrained")
    require("Phase_3_Fusion_MultiCam/reports", root / "Phase_3_Fusion_MultiCam" / "reports")
    print("")

    if failures:
        print("Missing required items:")
        for failure in failures:
            print(f"  - {failure}")
        print("")
        print("Typical fix on Windows:")
        print(r"  .\scripts\link_external_data_windows.ps1 -DataDir ..\STAGELIST3N-FusionCam-data")
        return 1

    print("Data layout is ready for recorded Phase 2/Phase 3 experiments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

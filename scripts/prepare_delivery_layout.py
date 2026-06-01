"""Create the external heavy-data delivery layout.

This script intentionally creates directories only. It does not copy datasets,
videos or model weights automatically because those assets can be very large and
should be selected deliberately before delivery.
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Prepare FusionCam delivery layout.")
    parser.add_argument(
        "--data-dir",
        default="../STAGELIST3N-FusionCam-data",
        help="External heavy-data directory to create.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

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

    print(f"Prepared heavy-data layout: {root}")
    for rel in DEFAULT_DIRS:
        print(f"  - {rel}/")


if __name__ == "__main__":
    main()

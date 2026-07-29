"""Tests Pytest des utilitaires de campagne (campaign_utils).

Couvre la découverte des modèles entraînés (V2/V3, formats .pt et TensorRT)
et l'écriture des CSV de détections à en-tête stable, indispensables à la
reproductibilité des campagnes de benchmark.
"""

from pathlib import Path

from campaign_utils import (
    DETECTION_FIELDS,
    ModelSpec,
    discover_model_specs,
    write_csv,
)


def test_discover_model_specs_v2_and_v3(tmp_path: Path):
    """La découverte doit lister chaque modèle V2/V3 dans ses deux formats (.pt et moteur TensorRT), y compris les moteurs à nom suffixé."""
    root = tmp_path / "Modelstrained"
    (root / "V2" / "yolo26n" / "weights").mkdir(parents=True)
    (root / "V2" / "yolo26n" / "weights" / "best.pt").write_text("pt")
    (root / "V2" / "yolo26n" / "weights" / "best.engine").write_text("engine")
    (root / "V3" / "yolov8n" / "weights").mkdir(parents=True)
    (root / "V3" / "yolov8n" / "weights" / "best.pt").write_text("pt")
    (root / "V3" / "yolov8n" / "weights" / "best_fp32_best.engine").write_text("engine")

    specs = discover_model_specs(modelstrained_dir=root)
    labels = {(s.version, s.name, s.format_label, s.weights_path.name) for s in specs}

    assert ("V2", "yolo26n", "pt", "best.pt") in labels
    assert ("V2", "yolo26n", "fp32_engine", "best.engine") in labels
    assert ("V3", "yolov8n", "pt", "best.pt") in labels
    assert ("V3", "yolov8n", "fp32_engine", "best_fp32_best.engine") in labels


def test_discover_model_specs_filters_models(tmp_path: Path):
    """Les filtres version/format/nom doivent restreindre la découverte aux seuls modèles demandés."""
    root = tmp_path / "Modelstrained"
    for name in ["yolov8n", "yolo11s"]:
        (root / "V2" / name / "weights").mkdir(parents=True)
        (root / "V2" / name / "weights" / "best.pt").write_text("pt")

    specs = discover_model_specs(
        versions=["V2"],
        formats=["pt"],
        model_names=["yolo11s"],
        modelstrained_dir=root,
    )

    assert [s.name for s in specs] == ["yolo11s"]


def test_model_spec_phase2_dict():
    """ModelSpec doit exposer le dictionnaire attendu par le pipeline Phase 2 et un run_label unique version_modele_format."""
    spec = ModelSpec(
        version="V3",
        name="rtdetr-l",
        subdir="rtdetr-l",
        model_type="rtdetr",
        format_label="fp32_engine",
        format_arg="engine",
        suffix_arg="fp32_best",
        weights_path=Path("weights/best_fp32_best.engine"),
    )

    assert spec.phase2_dict() == {
        "name": "rtdetr-l",
        "subdir": "rtdetr-l",
        "type": "rtdetr",
    }
    assert spec.run_label == "V3_rtdetr-l_fp32_engine"


def test_write_csv_stable_header(tmp_path: Path):
    """write_csv doit conserver un en-tête de colonnes stable et ignorer les champs hors schéma."""
    out = tmp_path / "detections.csv"
    write_csv(
        out,
        [{"model": "yolov8n", "frame": 1, "extra": "ignored"}],
        DETECTION_FIELDS,
    )

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("model_version,model,format,frame")
    assert "ignored" not in lines[1]

"""Pytest checks for the Phase 3 CameraTracker.

These tests keep the existing manual script test_tracker.py untouched.
They mock Ultralytics so pytest can validate the tracker wiring without
loading a YOLO model, CUDA, or video files.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeBoxRow:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeBoxes:
    id = [_FakeScalar(7)]
    conf = [_FakeScalar(0.83)]
    cls = [_FakeScalar(0)]
    xyxy = [_FakeBoxRow([10.0, 20.0, 50.0, 100.0])]

    def __len__(self):
        return 1


class _FakeResult:
    boxes = _FakeBoxes()


class _FakeYOLO:
    instances: list["_FakeYOLO"] = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.init_kwargs = kwargs
        self.names = {0: "person"}
        self.track_calls = []
        _FakeYOLO.instances.append(self)

    def track(self, frame, **kwargs):
        self.track_calls.append({"frame_shape": frame.shape, **kwargs})
        return [_FakeResult()]


def _import_tracker_with_fake_yolo(monkeypatch):
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    phase3_dir = Path(__file__).resolve().parent
    monkeypatch.syspath_prepend(str(phase3_dir))
    sys.modules.pop("tracker", None)
    sys.modules.pop("detection", None)
    sys.modules.pop("geometry_fix", None)
    _FakeYOLO.instances.clear()
    return importlib.import_module("tracker")


def _config():
    return {
        "aspect_ratio_fix": {"enabled": False},
        "cameras": {
            "cam_03": {"room": "room_1"},
            "cam_05": {"room": "room_1"},
        },
        "homographie": {
            "cam_03_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "cam_05_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }


def test_camera_tracker_uses_bytetrack_yaml_and_builds_detection(tmp_path, monkeypatch):
    tracker_module = _import_tracker_with_fake_yolo(monkeypatch)
    model_path = tmp_path / "model.pt"
    model_path.write_text("fake model")

    camera_tracker = tracker_module.CameraTracker(
        cam_id="cam_03",
        model_path=str(model_path),
        config=_config(),
        conf_threshold=0.42,
        target_classes=[0],
    )

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = camera_tracker.process_frame(frame, timestamp=12.5)

    assert len(detections) == 1
    det = detections[0]
    assert det.cam_id == "cam_03"
    assert det.track_id == 7
    assert det.class_name == "person"
    assert det.bbox_px == (10.0, 20.0, 50.0, 100.0)
    assert det.foot_point_px == (30.0, 100.0)
    assert det.foot_point_m == (30.0, 100.0)
    assert det.room_id == "room_1"

    yolo = _FakeYOLO.instances[0]
    assert yolo.init_kwargs["task"] == "detect"
    call = yolo.track_calls[0]
    assert call["persist"] is True
    assert call["conf"] == 0.42
    assert call["classes"] == [0]
    assert Path(call["tracker"]).name == "bytetrack.yaml"
    assert Path(call["tracker"]).exists()


def test_each_camera_tracker_gets_its_own_yolo_instance(tmp_path, monkeypatch):
    tracker_module = _import_tracker_with_fake_yolo(monkeypatch)
    model_path = tmp_path / "model.pt"
    model_path.write_text("fake model")

    cam03 = tracker_module.CameraTracker("cam_03", str(model_path), _config())
    cam05 = tracker_module.CameraTracker("cam_05", str(model_path), _config())

    assert cam03.model is not cam05.model
    assert len(_FakeYOLO.instances) == 2


def test_camera_tracker_passes_explicit_gpu_device(tmp_path, monkeypatch):
    tracker_module = _import_tracker_with_fake_yolo(monkeypatch)
    model_path = tmp_path / "model.pt"
    model_path.write_text("fake model")

    camera_tracker = tracker_module.CameraTracker(
        cam_id="cam_03",
        model_path=str(model_path),
        config=_config(),
        device="cuda:0",
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    camera_tracker.process_frame(frame, timestamp=1.0)

    call = _FakeYOLO.instances[0].track_calls[0]
    assert call["device"] == "cuda:0"


def test_camera_tracker_filters_invalid_nan_bbox(tmp_path, monkeypatch):
    tracker_module = _import_tracker_with_fake_yolo(monkeypatch)
    model_path = tmp_path / "model.pt"
    model_path.write_text("fake model")

    class BadBoxes(_FakeBoxes):
        xyxy = [_FakeBoxRow([float("nan"), 20.0, 50.0, 100.0])]

    class BadResult:
        boxes = BadBoxes()

    def bad_track(self, frame, **kwargs):
        self.track_calls.append({"frame_shape": frame.shape, **kwargs})
        return [BadResult()]

    monkeypatch.setattr(_FakeYOLO, "track", bad_track)
    camera_tracker = tracker_module.CameraTracker(
        cam_id="cam_03",
        model_path=str(model_path),
        config=_config(),
    )

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = camera_tracker.process_frame(frame, timestamp=1.0)

    assert detections == []


def test_camera_tracker_recovers_when_bytetrack_raises(tmp_path, monkeypatch):
    tracker_module = _import_tracker_with_fake_yolo(monkeypatch)
    model_path = tmp_path / "model.pt"
    model_path.write_text("fake model")

    def crashing_track(self, frame, **kwargs):
        self.track_calls.append({"frame_shape": frame.shape, **kwargs})
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(_FakeYOLO, "track", crashing_track)
    camera_tracker = tracker_module.CameraTracker(
        cam_id="cam_03",
        model_path=str(model_path),
        config=_config(),
    )

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = camera_tracker.process_frame(frame, timestamp=1.0)

    assert detections == []
    assert len(_FakeYOLO.instances) == 2
    assert camera_tracker.model is _FakeYOLO.instances[1]

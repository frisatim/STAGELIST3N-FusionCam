"""Tests Pytest de la détection de violations de zones interdites.

Couvre l'intrusion de personnes en zone interdite (décision par class_id) et
la confirmation multi-caméras des alertes (personnes et objets interdits),
au cœur de la fiabilité des alertes émises en production.
"""

from __future__ import annotations

import time

from detection import Detection
from violation_detector import ViolationDetector


def _config():
    return {
        "zones_interdites": {
            "zone_1": {
                "nom": "zone_1",
                "cameras_concernees": ["cam_02"],
                "coordonnees_metres": [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [2.0, 2.0],
                    [0.0, 2.0],
                ],
            }
        }
    }


def _detection(
    class_id: int,
    class_name: str,
    foot_point_m: tuple[float, float],
    cam_id: str = "cam_02",
    track_id: int = 12,
    global_id: int = 5,
):
    return Detection(
        cam_id=cam_id,
        track_id=track_id,
        global_id=global_id,
        timestamp=time.time(),
        bbox_px=(10.0, 10.0, 50.0, 100.0),
        foot_point_px=(30.0, 100.0),
        foot_point_m=foot_point_m,
        confidence=0.91,
        class_id=class_id,
        class_name=class_name,
    )


def test_person_zone_violation_uses_class_id_not_only_label():
    """Une personne en zone interdite déclenche l'alerte même avec un label atypique : la décision repose sur class_id, pas sur le nom de classe."""
    detector = ViolationDetector(_config())

    alerts = detector.check([_detection(0, "0", (1.0, 1.0))])

    assert len(alerts) == 1
    assert alerts[0].alert_type == "zone_violation_person"
    assert alerts[0].zone_id == "zone_1"
    assert alerts[0].global_id == 5


def test_bottom_center_outside_zone_does_not_trigger_violation():
    """Une personne dont le point pied est hors de la zone interdite ne doit déclencher aucune alerte."""
    detector = ViolationDetector(_config())

    alerts = detector.check([_detection(0, "person", (3.0, 3.0))])

    assert alerts == []


def test_person_class_ids_can_be_configured_for_custom_models():
    """Un modèle personnalisé peut déclarer d'autres class_id de personne via la config (ex. classe 'worker')."""
    config = _config()
    config["detection"] = {"person_class_ids": [2]}
    detector = ViolationDetector(config)

    alerts = detector.check([_detection(2, "worker", (1.0, 1.0))])

    assert len(alerts) == 1
    assert alerts[0].alert_type == "zone_violation_person"


def test_person_zone_violation_defaults_to_single_camera_safety_mode():
    """Par défaut (mode sécurité), une seule caméra suffit à confirmer une intrusion de personne."""
    config = _config()
    config["zones_interdites"]["zone_1"]["cameras_concernees"] = [
        "cam_02",
        "cam_03",
        "cam_05",
        "cam_07",
    ]
    detector = ViolationDetector(config)

    alerts = detector.check([_detection(0, "person", (1.0, 1.0), cam_id="cam_02")])

    assert len(alerts) == 1
    assert alerts[0].alert_type == "zone_violation_person"
    assert alerts[0].cameras == ["cam_02"]


def test_person_zone_violation_can_require_half_zone_cameras():
    """Avec un ratio configuré, l'alerte personne exige la confirmation par plusieurs caméras couvrant la zone."""
    config = _config()
    config["zones_interdites"]["zone_1"]["cameras_concernees"] = [
        "cam_02",
        "cam_03",
        "cam_05",
        "cam_07",
    ]
    config["alerting"] = {
        "person_zone_min_camera_ratio": 0.5,
        "person_zone_min_camera_votes": 1,
    }
    detector = ViolationDetector(config)

    one_camera = [_detection(0, "person", (1.0, 1.0), cam_id="cam_02")]
    two_cameras = [
        _detection(0, "person", (1.0, 1.0), cam_id="cam_02", track_id=1),
        _detection(0, "person", (1.1, 1.0), cam_id="cam_03", track_id=2),
    ]

    assert detector.check(one_camera) == []
    alerts = detector.check(two_cameras)
    assert len(alerts) == 1
    assert sorted(alerts[0].cameras) == ["cam_02", "cam_03"]


def test_forbidden_object_can_require_multi_camera_confirmation():
    """Un objet interdit vu par une seule caméra reste une alerte faible ; deux caméras la confirment, sans ré-émission en double."""
    config = _config()
    config["alerting"] = {"object_min_camera_votes": 2}
    detector = ViolationDetector(config)

    single_camera = [_detection(4, "perceuse", (1.0, 1.0), cam_id="cam_02")]
    two_cameras = [
        _detection(4, "perceuse", (1.0, 1.0), cam_id="cam_02", track_id=1),
        _detection(4, "perceuse", (1.1, 1.0), cam_id="cam_03", track_id=2),
    ]

    weak_alerts = detector.check(single_camera)
    assert len(weak_alerts) == 1
    assert weak_alerts[0].alert_level == "weak"
    alerts = detector.check(two_cameras)
    assert len(alerts) == 1
    assert alerts[0].alert_type == "forbidden_object"
    assert alerts[0].alert_level == "confirmed"
    assert sorted(alerts[0].cameras) == ["cam_02", "cam_03"]
    assert detector.check(two_cameras) == []


def test_forbidden_object_defaults_to_weak_then_confirmed_mode():
    """Par défaut, une alerte objet passe de 'weak' (une caméra) à 'confirmed' (deux caméras)."""
    detector = ViolationDetector(_config())

    single_camera = [_detection(4, "perceuse", (1.0, 1.0), cam_id="cam_02")]
    two_cameras = [
        _detection(4, "perceuse", (1.0, 1.0), cam_id="cam_02", track_id=1),
        _detection(4, "perceuse", (1.1, 1.0), cam_id="cam_03", track_id=2),
    ]

    weak_alerts = detector.check(single_camera)
    assert len(weak_alerts) == 1
    assert weak_alerts[0].alert_level == "weak"
    confirmed_alerts = detector.check(two_cameras)
    assert len(confirmed_alerts) == 1
    assert confirmed_alerts[0].alert_level == "confirmed"


def test_weak_object_alerts_can_be_disabled():
    """Les alertes objet faibles peuvent être désactivées : une seule caméra ne produit alors aucune alerte."""
    config = _config()
    config["alerting"] = {
        "object_min_camera_votes": 2,
        "object_emit_weak_alerts": False,
    }
    detector = ViolationDetector(config)

    single_camera = [_detection(4, "perceuse", (1.0, 1.0), cam_id="cam_02")]

    assert detector.check(single_camera) == []

"""Tests Pytest de la compatibilité de classes dans la fusion multi-caméras.

Vérifie que deux détections ne partagent un identifiant global que si leurs
classes sont compatibles : cela évite de fusionner une personne avec un objet
proche, source directe de fausses associations inter-caméras.
"""

from detection import Detection
from fusion import MultiCameraFusion


def _det(
    cam_id: str,
    track_id: int,
    class_name: str,
    class_id: int,
    timestamp: float = 1000.0,
) -> Detection:
    return Detection(
        cam_id=cam_id,
        track_id=track_id,
        global_id=None,
        timestamp=timestamp,
        bbox_px=(10.0, 20.0, 30.0, 40.0),
        foot_point_px=(20.0, 40.0),
        foot_point_m=(5.0, 5.0),
        confidence=0.9,
        class_id=class_id,
        class_name=class_name,
    )


def _fusion(time_window_s: float = 1.0) -> MultiCameraFusion:
    return MultiCameraFusion(
        {"camera_overlaps": {"room_1": [["cam_a", "cam_b"]]}},
        distance_threshold_m=1.0,
        time_window_s=time_window_s,
    )


def test_same_class_inside_time_window_shares_global_id():
    """Deux détections de même classe, proches et dans la fenêtre temporelle, doivent partager le même identifiant global."""
    fusion = _fusion(time_window_s=2.0)
    a = _det("cam_a", 1, "bouteille", 5, timestamp=1000.0)
    b = _det("cam_b", 2, "bouteille", 5, timestamp=1001.2)

    fusion.associate({"cam_a": [a], "cam_b": [b]})

    assert a.global_id == b.global_id


def test_different_classes_do_not_share_global_id_even_if_close():
    """Deux classes différentes ne doivent jamais être fusionnées, même à position quasi identique."""
    fusion = _fusion()
    a = _det("cam_a", 1, "bouteille", 5)
    b = _det("cam_b", 2, "personne", 11)

    fusion.associate({"cam_a": [a], "cam_b": [b]})

    assert a.global_id != b.global_id


def test_person_aliases_are_compatible():
    """Les alias de la classe personne ('personne' et 'person') doivent être considérés comme compatibles."""
    fusion = _fusion()
    a = _det("cam_a", 1, "personne", 11)
    b = _det("cam_b", 2, "person", 0)

    fusion.associate({"cam_a": [a], "cam_b": [b]})

    assert a.global_id == b.global_id


def test_same_local_track_gets_new_global_id_when_class_changes():
    """Si la classe d'une piste locale change, elle doit recevoir un nouvel identifiant global au lieu d'hériter de l'ancien."""
    fusion = _fusion()
    bottle = _det("cam_a", 1, "bouteille", 5)
    fusion.associate({"cam_a": [bottle]})
    bottle_gid = bottle.global_id

    person = _det("cam_a", 1, "personne", 11, timestamp=1000.5)
    fusion.associate({"cam_a": [person]})

    assert person.global_id is not None
    assert person.global_id != bottle_gid

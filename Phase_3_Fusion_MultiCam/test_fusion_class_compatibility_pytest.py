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
    fusion = _fusion(time_window_s=2.0)
    a = _det("cam_a", 1, "bouteille", 5, timestamp=1000.0)
    b = _det("cam_b", 2, "bouteille", 5, timestamp=1001.2)

    fusion.associate({"cam_a": [a], "cam_b": [b]})

    assert a.global_id == b.global_id


def test_different_classes_do_not_share_global_id_even_if_close():
    fusion = _fusion()
    a = _det("cam_a", 1, "bouteille", 5)
    b = _det("cam_b", 2, "personne", 11)

    fusion.associate({"cam_a": [a], "cam_b": [b]})

    assert a.global_id != b.global_id


def test_person_aliases_are_compatible():
    fusion = _fusion()
    a = _det("cam_a", 1, "personne", 11)
    b = _det("cam_b", 2, "person", 0)

    fusion.associate({"cam_a": [a], "cam_b": [b]})

    assert a.global_id == b.global_id


def test_same_local_track_gets_new_global_id_when_class_changes():
    fusion = _fusion()
    bottle = _det("cam_a", 1, "bouteille", 5)
    fusion.associate({"cam_a": [bottle]})
    bottle_gid = bottle.global_id

    person = _det("cam_a", 1, "personne", 11, timestamp=1000.5)
    fusion.associate({"cam_a": [person]})

    assert person.global_id is not None
    assert person.global_id != bottle_gid

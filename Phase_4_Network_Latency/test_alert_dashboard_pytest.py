"""Tests du dashboard d'alertes : vérifie le format des alertes simulées."""

from Phase_4_Network_Latency.alert_dashboard import make_simulated_alert


def test_make_simulated_alert_contains_dashboard_fields():
    """Une alerte simulée contient tous les champs attendus par le dashboard, avec des valeurs cohérentes."""
    alert = make_simulated_alert(5)

    assert alert["event_id"] == 5
    assert alert["alert_level"] == "confirmed"
    assert alert["alert_type"] in {"forbidden_object", "zone_violation_person"}
    assert alert["camera"].startswith("cam_")
    assert alert["created_epoch_ms"] > 0

import os
import yaml
import numpy as np
import cv2


def load_zones(config_path: str) -> dict[str, np.ndarray]:
    """Charge les zones interdites depuis le config.yaml.

    Retourne un dict {nom_zone: np.array de shape (N,1,2) en float32}
    compatible avec cv2.pointPolygonTest.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    zones = {}
    for zone_id, zone_data in config.get("zones_interdites", {}).items():
        coords = zone_data.get("coordonnees_metres")
        if not coords or len(coords) < 3:
            print(f"[WARN] Zone ignorée '{zone_id}' (coordonnées absentes/invalides).")
            continue

        points = np.array(coords, dtype=np.float32)
        # OpenCV attend un contour de shape (N, 1, 2)
        zones[zone_id] = points.reshape(-1, 1, 2)

    return zones


def is_point_in_zone(pt_x: float, pt_y: float, zone_polygon: np.ndarray) -> bool:
    """Teste si le point (pt_x, pt_y) est a l'interieur du polygone.

    Utilise cv2.pointPolygonTest :
      >= 0  -> interieur ou sur le bord -> True
      <  0  -> exterieur               -> False
    """
    result = cv2.pointPolygonTest(zone_polygon, (pt_x, pt_y), measureDist=False)
    return result >= 0


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    zones = load_zones(config_path)

    zone_1 = zones["zone_1"]
    print(f"Zone 1 chargee : {zone_1.shape[0]} sommets")
    print(f"  Coordonnees : {zone_1.reshape(-1, 2).tolist()}")

    # Test 1 : point interieur (4.5, 4.0) — dans la plage [3.67-5.40, 3.29-4.81]
    x_in, y_in = 4.5, 4.0
    inside = is_point_in_zone(x_in, y_in, zone_1)
    print(f"\nPoint ({x_in}, {y_in}) dans zone_1 ? {inside}")  # attendu : True

    # Test 2 : point exterieur (1.0, 1.0) — hors de la zone
    x_out, y_out = 1.0, 1.0
    outside = is_point_in_zone(x_out, y_out, zone_1)
    print(f"Point ({x_out}, {y_out}) dans zone_1 ? {outside}")  # attendu : False

"""
Correctif geometrique pour cameras Dahua a ratio d'aspect deforme.

Probleme :
  Certaines cameras Dahua produisent un sub-stream en 704x576 (ratio ~1.22:1)
  au lieu du 4:3 natif du capteur. L'image est ecrasee horizontalement par le
  firmware, ce qui fausse toute la geometrie spatiale : homographie, distances,
  projections sur le plan de sol.

Solution :
  Redimensionner ces frames en 768x576 pour restaurer le ratio 4:3.
  768/576 = 1.333 = 4/3.

Configuration :
  Les cameras concernees sont definies dans config.yaml > aspect_ratio_fix.

Usage dans un script d'inference :
  from geometry_fix import load_aspect_fix, fix_frame

  config = yaml.safe_load(open("config.yaml"))
  ar_fix = load_aspect_fix(config)

  # Dans la boucle de traitement :
  frame = fix_frame(frame, cam_id, ar_fix)  # AVANT YOLO, ByteTrack, homographie
  detections = model(frame)
  ...
"""

import cv2
import numpy as np


def load_aspect_fix(config: dict) -> dict:
    """Charge la configuration du correctif ratio d'aspect depuis le config."""
    ar_fix = config.get("aspect_ratio_fix", {})
    if not ar_fix.get("enabled", False):
        return {"enabled": False}
    return {
        "enabled": True,
        "cameras": set(ar_fix.get("distorted_cameras", [])),
        "target_w": ar_fix["corrected_resolution"][0],
        "target_h": ar_fix["corrected_resolution"][1],
    }


def fix_frame(frame: np.ndarray, cam_id: str, ar_fix: dict) -> np.ndarray:
    """Applique le correctif ratio d'aspect si necessaire.

    DOIT etre appele AVANT :
      - Passage a YOLO / ByteTrack (les bounding boxes doivent etre en coordonnees corrigees)
      - cv2.perspectiveTransform (l'homographie a ete calculee sur l'image corrigee)
      - Tout calcul de distance ou de position spatiale

    Args:
        frame: Image brute lue par cap.read()
        cam_id: Identifiant camera (ex: "cam_03")
        ar_fix: Dict retourne par load_aspect_fix()

    Returns:
        Frame corrigee (768x576) ou frame originale si pas de correction necessaire.
    """
    if not ar_fix.get("enabled", False):
        return frame
    if cam_id not in ar_fix["cameras"]:
        return frame
    h, w = frame.shape[:2]
    tw, th = ar_fix["target_w"], ar_fix["target_h"]
    if w == tw and h == th:
        return frame
    return cv2.resize(frame, (tw, th))

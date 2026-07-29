"""
Module de structures de données pour les détections Phase 3.

Contient la dataclass Detection qui représente une détection unique
(personne ou objet) produite par un CameraTracker à un instant donné.

Place dans le pipeline : Detection est le format d'échange entre le suivi
mono-caméra (tracker.py), la fusion inter-caméras (fusion.py, qui renseigne
global_id) et la détection de violations (violation_detector.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    """Représente une détection produite par un tracker mono-caméra.

    Attributes:
        cam_id: Identifiant de la caméra (ex. "cam_03").
        track_id: Identifiant local ByteTrack (unique par caméra, remis à zéro
                  entre sessions).
        global_id: Identifiant inter-caméras attribué par MultiCameraFusion.
                   None tant que la fusion n'est pas passée.
        timestamp: Horodatage Unix de la frame (secondes, float).
        bbox_px: Boîte englobante en pixels (x1, y1, x2, y2) après correctif
                 de ratio d'aspect.
        foot_point_px: Bas-centre de la bbox en pixels (u, v).
        foot_point_m: Point au sol projeté en coordonnées mètres (x, y).
                      (0, 0) est le coin haut-gauche du plan de sol.
                      None en l'absence d'homographie.
        confidence: Score de confiance de la détection dans [0, 1].
        class_id: Indice de classe COCO (0 = personne).
        class_name: Nom de classe lisible (ex. "person").
    """

    cam_id: str
    track_id: int
    global_id: int | None
    timestamp: float
    bbox_px: tuple[float, float, float, float]
    foot_point_px: tuple[float, float]
    foot_point_m: tuple[float, float] | None
    confidence: float
    class_id: int
    class_name: str

    # Champs calculés après construction (non requis à l'init)
    room_id: str | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    @property
    def is_person(self) -> bool:
        """Vrai si la détection est une personne (classe COCO 0)."""
        return self.class_id == 0

    @property
    def bbox_center_px(self) -> tuple[float, float]:
        """Retourne le centre géométrique de la boîte englobante en pixels."""
        x1, y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bbox_wh_px(self) -> tuple[float, float]:
        """Retourne (largeur, hauteur) de la boîte englobante en pixels."""
        x1, y1, x2, y2 = self.bbox_px
        return (x2 - x1, y2 - y1)

    def has_floor_position(self) -> bool:
        """Vrai si foot_point_m est disponible (l'homographie a été appliquée)."""
        return self.foot_point_m is not None

    def __repr__(self) -> str:
        gid = f"gid={self.global_id}" if self.global_id is not None else "gid=?"
        pos = (
            f"({self.foot_point_m[0]:.2f}m, {self.foot_point_m[1]:.2f}m)"
            if self.foot_point_m
            else "pos=N/A"
        )
        return (
            f"Detection({self.cam_id} tid={self.track_id} {gid} "
            f"{self.class_name} conf={self.confidence:.2f} {pos})"
        )


# ----------------------------------------------------------------------
# Test basique
# ----------------------------------------------------------------------

if __name__ == "__main__":
    det = Detection(
        cam_id="cam_03",
        track_id=5,
        global_id=None,
        timestamp=1743000000.0,
        bbox_px=(100.0, 200.0, 180.0, 400.0),
        foot_point_px=(140.0, 400.0),
        foot_point_m=(4.2, 3.8),
        confidence=0.87,
        class_id=0,
        class_name="person",
    )

    print("[INFO] Detection créée :")
    print(f"  {det}")
    print(f"  is_person      : {det.is_person}")
    print(f"  bbox_center_px : {det.bbox_center_px}")
    print(f"  bbox_wh_px     : {det.bbox_wh_px}")
    print(f"  has_floor_pos  : {det.has_floor_position()}")
    print(f"  room_id        : {det.room_id}")

    # Simule l'attribution du global_id par la fusion
    det.global_id = 1
    print(f"\n[INFO] Après fusion : {det}")

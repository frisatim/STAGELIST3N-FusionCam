"""
Module de structures de données pour les détections Phase 3.

Contient la dataclass Detection qui représente une détection unique
(personne ou objet) produite par un CameraTracker à un instant donné.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    """Représente une détection produite par un tracker mono-caméra.

    Attributes:
        cam_id: Camera identifier (e.g. "cam_03").
        track_id: ByteTrack local ID (unique per camera, resets between sessions).
        global_id: Cross-camera ID assigned by MultiCameraFusion. None until fusion.
        timestamp: Unix timestamp of the frame (seconds, float).
        bbox_px: Bounding box in pixels (x1, y1, x2, y2) after aspect-ratio fix.
        foot_point_px: Bottom-center of bbox in pixels (u, v).
        foot_point_m: Foot point projected to floor coordinates in metres (x, y).
                      (0, 0) is top-left of the floor plan. None if no homography.
        confidence: Detection confidence score in [0, 1].
        class_id: COCO class index (0 = person).
        class_name: Human-readable class label (e.g. "person").
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
        """Return True if this detection is a person (COCO class 0)."""
        return self.class_id == 0

    @property
    def bbox_center_px(self) -> tuple[float, float]:
        """Return the geometric center of the bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bbox_wh_px(self) -> tuple[float, float]:
        """Return (width, height) of the bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox_px
        return (x2 - x1, y2 - y1)

    def has_floor_position(self) -> bool:
        """Return True if foot_point_m is available (homography was applied)."""
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

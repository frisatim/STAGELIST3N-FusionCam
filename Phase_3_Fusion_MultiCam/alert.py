"""
Module de structures de données pour les alertes Phase 3.

Contient la dataclass Alert générée par ViolationDetector quand
une violation de zone est détectée ou qu'un objet interdit est vu.

Place dans le pipeline : Alert est le format de sortie de la détection de
violations, consommé par l'affichage de pipeline.py et sérialisé vers la
Phase 4 par metadata_publisher.py.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal


AlertType = Literal["zone_violation_person", "forbidden_object"]
AlertLevel = Literal["weak", "confirmed"]


@dataclass
class Alert:
    """Représente une alerte de violation de sécurité.

    Attributes:
        alert_id: Identifiant unique de l'alerte (UUID tronqué à 8 caractères).
        alert_type: "zone_violation_person" ou "forbidden_object".
        global_id: Identifiant inter-caméras de l'entité concernée.
        zone_id: Identifiant de la zone interdite (ex. "zone_1").
                 None pour les alertes objet.
        position_m: Position sur le plan de sol en mètres (x, y).
        cameras: Identifiants des caméras ayant observé l'entité.
        timestamp: Horodatage Unix de l'alerte (secondes, float).
        confidence: Confiance de la détection déclenchante.
        class_name: "person" ou nom de l'objet interdit.
        alert_level: Niveau de l'alerte : "weak" (une seule caméra) ou
                     "confirmed" (vote multi-caméras atteint).
        acknowledged: Vrai une fois l'alerte journalisée ou transmise.
    """

    alert_type: AlertType
    global_id: int
    position_m: tuple[float, float]
    cameras: list[str]
    timestamp: float
    confidence: float
    class_name: str
    zone_id: str | None = None
    alert_level: AlertLevel = "confirmed"
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    acknowledged: bool = False

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    @classmethod
    def person_in_zone(
        cls,
        global_id: int,
        zone_id: str,
        position_m: tuple[float, float],
        cameras: list[str],
        timestamp: float,
        confidence: float,
        alert_level: AlertLevel = "confirmed",
    ) -> "Alert":
        """Construit une alerte 'personne en zone interdite'."""
        return cls(
            alert_type="zone_violation_person",
            global_id=global_id,
            zone_id=zone_id,
            position_m=position_m,
            cameras=cameras,
            timestamp=timestamp,
            confidence=confidence,
            class_name="person",
            alert_level=alert_level,
        )

    @classmethod
    def forbidden_object(
        cls,
        global_id: int,
        class_name: str,
        position_m: tuple[float, float],
        cameras: list[str],
        timestamp: float,
        confidence: float,
        alert_level: AlertLevel = "confirmed",
    ) -> "Alert":
        """Construit une alerte 'objet interdit' (zone_id volontairement à None)."""
        return cls(
            alert_type="forbidden_object",
            global_id=global_id,
            zone_id=None,
            position_m=position_m,
            cameras=cameras,
            timestamp=timestamp,
            confidence=confidence,
            class_name=class_name,
            alert_level=alert_level,
        )

    def to_dict(self) -> dict:
        """Sérialise l'alerte en dictionnaire simple pour journalisation JSON."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "global_id": self.global_id,
            "zone_id": self.zone_id,
            "position_m": list(self.position_m),
            "cameras": self.cameras,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4),
            "class_name": self.class_name,
            "alert_level": self.alert_level,
        }

    def __repr__(self) -> str:
        zone = f" zone={self.zone_id}" if self.zone_id else ""
        cams = "+".join(self.cameras)
        return (
            f"Alert[{self.alert_id}] {self.alert_type}:{self.alert_level}{zone} "
            f"gid={self.global_id} pos=({self.position_m[0]:.2f}m,{self.position_m[1]:.2f}m) "
            f"cams={cams} conf={self.confidence:.2f}"
        )


# ----------------------------------------------------------------------
# Test basique
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Alerte personne en zone interdite
    a1 = Alert.person_in_zone(
        global_id=3,
        zone_id="zone_1",
        position_m=(4.5, 4.0),
        cameras=["cam_03", "cam_05"],
        timestamp=time.time(),
        confidence=0.91,
    )
    print("[INFO] Alerte personne :")
    print(f"  {a1}")
    print(f"  JSON : {json.dumps(a1.to_dict(), indent=2)}")

    # Alerte objet interdit
    a2 = Alert.forbidden_object(
        global_id=7,
        class_name="fire_extinguisher",
        position_m=(5.1, 3.5),
        cameras=["cam_07"],
        timestamp=time.time(),
        confidence=0.76,
    )
    print("\n[INFO] Alerte objet :")
    print(f"  {a2}")

"""
Détection de violations de sécurité pour le système multi-caméras (Phase 3).

Place dans le pipeline : dernier maillon de l'analyse. Consomme les détections
fusionnées (global_id attribué par MultiCameraFusion) et produit des alertes
de deux types :
  - "zone_violation_person" : une personne dont le point au sol entre dans une
    zone interdite, définie en mètres dans config.yaml ;
  - "forbidden_object" : un objet non autorisé vu dans l'atelier, avec deux
    niveaux de confiance ("weak" : une seule caméra, "confirmed" : vote
    multi-caméras atteint).

Le détecteur maintient une machine à états par entité afin de dédupliquer les
alertes : une entrée en zone ne déclenche qu'une seule alerte tant que la
personne reste dans la zone, et une nouvelle alerte est émise si elle ressort
puis rentre à nouveau (re-entry). Même principe pour les objets interdits,
avec en plus une escalade possible du niveau "weak" vers "confirmed".
"""

from __future__ import annotations

import sys
import time
from typing import Any

import cv2
import numpy as np

from alert import Alert
from detection import Detection

# Classes de véhicules tolérées dans l'atelier : jamais traitées comme objets interdits
_ALLOWED_VEHICLE_CLASSES: frozenset[str] = frozenset(
    {"car", "truck", "bicycle", "motorcycle"}
)


class ViolationDetector:
    """Détecte violations de zones et objets interdits, avec vote multi-caméras.

    Paramètres lus dans config.yaml (section alerting) :
      - person_zone_min_camera_votes : nombre minimal de caméras (parmi celles
        couvrant la zone) devant voir la personne dans la zone pour confirmer
        la violation.
      - person_zone_min_camera_ratio : fraction minimale des caméras de la
        zone devant voter ; le seuil effectif est le maximum des deux
        critères, avec un plancher de 1.
      - object_min_camera_votes : nombre de caméras requis pour qu'un objet
        interdit passe au niveau "confirmed".
      - object_emit_weak_alerts : si True, un objet vu par une seule caméra
        émet une alerte de niveau "weak".

    État interne (machine à états des alertes) :
      - _active_violations : couples (global_id, zone_id) actuellement en
        violation. Sert à dédupliquer (aucune nouvelle alerte tant que la
        violation persiste) et à détecter les re-entries.
      - _active_object_levels : niveau d'alerte courant par objet interdit
        ("weak" ou "confirmed"). Permet l'escalade weak vers confirmed sans
        jamais réémettre ni rétrograder une alerte confirmée active.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.zones: dict[str, dict] = self._load_zones(config)
        self.person_class_ids: set[int] = set(
            config.get("detection", {}).get("person_class_ids", [0])
        )
        alerting_cfg = config.get("alerting", {})
        self.person_zone_min_camera_ratio = float(
            alerting_cfg.get("person_zone_min_camera_ratio", 0.0)
        )
        self.person_zone_min_camera_votes = int(
            alerting_cfg.get("person_zone_min_camera_votes", 1)
        )
        self.object_min_camera_votes = int(
            alerting_cfg.get("object_min_camera_votes", 2)
        )
        self.object_emit_weak_alerts = bool(
            alerting_cfg.get("object_emit_weak_alerts", True)
        )
        self._active_violations: set[tuple[int, str]] = set()
        self._active_object_levels: dict[tuple[int, str], str] = {}
        print(f"[INFO] ViolationDetector initialisé avec {len(self.zones)} zone(s) interdite(s).")
        print(
            "[INFO] Règles alertes : "
            f"zone_person votes>={self.person_zone_min_camera_votes}, "
            f"ratio>={self.person_zone_min_camera_ratio:.2f}; "
            f"object cams>={self.object_min_camera_votes}, "
            f"weak_objects={self.object_emit_weak_alerts}"
        )

    def check(self, detections: list[Detection]) -> list[Alert]:
        """Analyse une frame fusionnée et retourne les NOUVELLES alertes.

        Déroulé :
          1. Personnes : regroupe les détections par global_id, teste chaque
             point au sol contre les polygones de zones, et agrège par couple
             (global_id, zone_id) les caméras votantes et la meilleure
             observation (confiance maximale).
          2. Vote multi-caméras : une violation n'est retenue que si assez de
             caméras couvrant la zone la confirment
             (_has_enough_person_zone_votes).
          3. Cycle de vie : par différence avec l'état de la frame précédente,
             seules les violations nouvelles émettent une alerte
             (déduplication) ; les violations disparues sont marquées résolues,
             ce qui réarme l'alerte en cas de re-entry.
          4. Objets interdits : même logique de vote par (global_id, classe),
             avec niveaux weak / confirmed et escalade sans réémission.

        Args:
            detections: Détections de la frame courante, global_id renseigné
                        par la fusion.

        Returns:
            Liste des alertes nouvellement déclenchées (vide si rien de neuf).
        """
        alerts: list[Alert] = []
        current_violations: set[tuple[int, str]] = set()

        # --- 1. Regroupement des personnes par identifiant global ---
        # Seules les détections avec position au sol et global_id sont
        # exploitables : le test de zone se fait en mètres et le vote se
        # compte par entité fusionnée, pas par caméra isolée.
        person_detections: dict[int, list[Detection]] = {}
        for det in detections:
            if self.is_person_detection(det) and det.foot_point_m is not None and det.global_id is not None:
                person_detections.setdefault(det.global_id, []).append(det)

        # --- 2. Test point-dans-polygone et agrégation par (global_id, zone) ---
        # Pour chaque entité et chaque zone, on retient la meilleure
        # observation (confiance maximale) et l'ensemble des caméras votantes.
        # Seules les caméras déclarées comme couvrant la zone comptent dans le
        # vote : une caméra d'une autre salle ne peut pas confirmer.
        violation_meta: dict[tuple[int, str], dict] = {}

        for global_id, dets in person_detections.items():
            for zone_id, zone in self.zones.items():
                for det in dets:
                    pt = (float(det.foot_point_m[0]), float(det.foot_point_m[1]))
                    result = cv2.pointPolygonTest(zone["polygon"], pt, measureDist=False)
                    if result >= 0:
                        key = (global_id, zone_id)
                        if key not in violation_meta:
                            violation_meta[key] = {
                                "position_m": det.foot_point_m,
                                "timestamp": det.timestamp,
                                "max_confidence": det.confidence,
                                "cameras": set(),
                            }
                        meta = violation_meta[key]
                        if det.confidence > meta["max_confidence"]:
                            meta["max_confidence"] = det.confidence
                            meta["position_m"] = det.foot_point_m
                            meta["timestamp"] = det.timestamp
                        if det.cam_id in zone["cameras"]:
                            meta["cameras"].add(det.cam_id)

        # --- 3. Vote multi-caméras : ne retient que les violations assez confirmées ---
        for key, meta in violation_meta.items():
            _global_id, zone_id = key
            zone = self.zones[zone_id]
            if self._has_enough_person_zone_votes(meta["cameras"], zone):
                current_violations.add(key)

        # --- 4. Machine à états : différence avec la frame précédente ---
        # new_violations : entrées en zone (une alerte est émise).
        # resolved_violations : sorties de zone (réarme l'alerte pour une re-entry).
        new_violations = current_violations - self._active_violations
        resolved_violations = self._active_violations - current_violations

        for key in new_violations:
            global_id, zone_id = key
            meta = violation_meta[key]
            cameras = sorted(meta["cameras"]) if meta["cameras"] else []
            alert = Alert.person_in_zone(
                global_id=global_id,
                zone_id=zone_id,
                position_m=meta["position_m"],
                cameras=cameras,
                timestamp=meta["timestamp"],
                confidence=meta["max_confidence"],
            )
            alerts.append(alert)
            print(f"[WARN] Nouvelle violation détectée : {alert}")

        for key in resolved_violations:
            global_id, zone_id = key
            print(f"[INFO] Violation résolue : gid={global_id} zone={zone_id}")

        self._active_violations = current_violations

        # --- 5. Objets interdits : regroupement par (global_id, classe canonique) ---
        # Sont exclus : les personnes (gérées ci-dessus) et les véhicules
        # tolérés. Comme pour les personnes, on agrège les caméras votantes et
        # la meilleure observation de chaque objet.
        object_groups: dict[tuple[int, str], dict] = {}
        for det in detections:
            if self.is_person_detection(det) or det.class_name in _ALLOWED_VEHICLE_CLASSES:
                continue
            if det.foot_point_m is None or det.global_id is None:
                continue
            class_key = self._class_key(det.class_name)
            key = (det.global_id, class_key)
            group = object_groups.setdefault(
                key,
                {
                    "class_name": det.class_name,
                    "timestamp": det.timestamp,
                    "max_confidence": det.confidence,
                    "best_position_m": det.foot_point_m,
                    "cameras": set(),
                },
            )
            group["cameras"].add(det.cam_id)
            if det.confidence > group["max_confidence"]:
                group["max_confidence"] = det.confidence
                group["best_position_m"] = det.foot_point_m
                group["timestamp"] = det.timestamp

        # Niveau d'alerte selon le vote : "confirmed" si assez de caméras
        # (object_min_camera_votes), sinon "weak" (une seule caméra) lorsque
        # les alertes faibles sont activées.
        current_object_levels: dict[tuple[int, str], str] = {}
        for key, group in object_groups.items():
            n_cameras = len(group["cameras"])
            if n_cameras >= self.object_min_camera_votes:
                current_object_levels[key] = "confirmed"
            elif self.object_emit_weak_alerts and n_cameras >= 1:
                current_object_levels[key] = "weak"

        # Émission avec escalade : on n'émet que si le niveau change, et une
        # alerte "confirmed" active n'est jamais réémise ni rétrogradée
        # (la perte momentanée d'une caméra ne doit pas regénérer d'alertes).
        for key, alert_level in current_object_levels.items():
            previous_level = self._active_object_levels.get(key)
            if previous_level == alert_level or previous_level == "confirmed":
                continue
            global_id, _class_key = key
            group = object_groups[key]
            alert = Alert.forbidden_object(
                global_id=global_id,
                class_name=group["class_name"],
                position_m=group["best_position_m"],
                cameras=sorted(group["cameras"]),
                timestamp=group["timestamp"],
                confidence=group["max_confidence"],
                alert_level=alert_level,
            )
            alerts.append(alert)
            if alert_level == "confirmed":
                print(f"[WARN] Objet interdit confirmé : {alert}")
            else:
                print(f"[WARN] Objet interdit faible : {alert}")

        # Objets disparus de la frame : réarme l'alerte (une réapparition réémettra)
        resolved_objects = set(self._active_object_levels) - set(current_object_levels)
        for global_id, class_key in resolved_objects:
            print(f"[INFO] Objet interdit résolu : gid={global_id} classe={class_key}")
        self._active_object_levels = current_object_levels

        return alerts

    def _has_enough_person_zone_votes(self, cameras: set[str], zone: dict) -> bool:
        """Vérifie le vote multi-caméras d'une violation de zone.

        Le seuil requis est le maximum entre le nombre minimal absolu de votes
        (person_zone_min_camera_votes) et le nombre dérivé du ratio
        (ceil(person_zone_min_camera_ratio * nombre de caméras couvrant la
        zone)), avec un plancher de 1 pour ne jamais exiger zéro vote.
        """
        zone_cameras = set(zone.get("cameras") or [])
        votes = len(cameras)
        required_by_ratio = int(np.ceil(len(zone_cameras) * self.person_zone_min_camera_ratio))
        required = max(self.person_zone_min_camera_votes, required_by_ratio, 1)
        return votes >= required

    @staticmethod
    def _class_key(class_name: str) -> str:
        """Normalise un nom de classe (minuscules, espaces retirés) en clé de regroupement."""
        return class_name.strip().lower()

    def zones_containing(self, det: Detection) -> list[str]:
        """Retourne les zones interdites contenant le point au sol de cette détection."""
        if det.foot_point_m is None:
            return []
        pt = (float(det.foot_point_m[0]), float(det.foot_point_m[1]))
        if not np.isfinite(pt).all():
            return []
        hits: list[str] = []
        for zone_id, zone in self.zones.items():
            if cv2.pointPolygonTest(zone["polygon"], pt, measureDist=False) >= 0:
                hits.append(zone_id)
        return hits

    def is_person_detection(self, det: Detection) -> bool:
        """Vrai si la détection est une personne (class_id configuré ou nom de classe)."""
        if det.class_id in self.person_class_ids:
            return True
        return det.class_name.strip().lower() in {"person", "personne", "human", "humain"}

    def _load_zones(self, config: dict[str, Any]) -> dict[str, dict]:
        """Charge les zones interdites depuis config.yaml.

        Chaque zone devient un polygone OpenCV en mètres (coordonnées du plan
        de sol) accompagné de la liste des caméras qui la couvrent, base du
        vote multi-caméras.
        """
        zones: dict[str, dict] = {}
        zones_config: dict[str, Any] = config.get("zones_interdites", {})
        for zone_id, zone_data in zones_config.items():
            coords = zone_data["coordonnees_metres"]
            polygon = np.array(coords, dtype=np.float32).reshape((-1, 1, 2))
            zones[zone_id] = {
                "nom": zone_data["nom"],
                "polygon": polygon,
                "cameras": zone_data.get("cameras_concernees", []),
            }
            print(f"[INFO] Zone chargée : {zone_id} ({len(coords)} points, caméras: {zones[zone_id]['cameras']})")
        return zones


if __name__ == "__main__":
    from pathlib import Path
    import yaml

    config_path = Path(__file__).with_name("config.yaml")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERR] config.yaml introuvable à {config_path}, utilisation d'une config de test.")
        config = {
            "zones_interdites": {
                "zone_1": {
                    "nom": "zone_1",
                    "cameras_concernees": ["cam_07", "cam_02", "cam_03", "cam_05"],
                    "coordonnees_metres": [
                        [7.832, 0.632],
                        [7.438, 2.084],
                        [6.247, 1.953],
                        [5.772, 0.308],
                    ],
                },
                "zone_2": {
                    "nom": "zone_2",
                    "cameras_concernees": ["cam_04", "cam_01", "cam_06"],
                    "coordonnees_metres": [
                        [4.405, 12.318],
                        [6.401, 11.603],
                        [7.789, 13.879],
                        [4.532, 12.952],
                    ],
                },
            }
        }

    detector = ViolationDetector(config)

    now = time.time()

    det_in_zone1 = Detection(
        cam_id="cam_03",
        track_id=1,
        global_id=42,
        timestamp=now,
        bbox_px=(100.0, 200.0, 180.0, 400.0),
        foot_point_px=(140.0, 400.0),
        foot_point_m=(6.8, 1.0),
        confidence=0.92,
        class_id=0,
        class_name="person",
    )

    det_outside = Detection(
        cam_id="cam_01",
        track_id=2,
        global_id=99,
        timestamp=now,
        bbox_px=(300.0, 100.0, 380.0, 300.0),
        foot_point_px=(340.0, 300.0),
        foot_point_m=(1.0, 1.0),
        confidence=0.78,
        class_id=0,
        class_name="person",
    )

    det_forbidden_obj = Detection(
        cam_id="cam_07",
        track_id=5,
        global_id=7,
        timestamp=now,
        bbox_px=(50.0, 50.0, 120.0, 150.0),
        foot_point_px=(85.0, 150.0),
        foot_point_m=(7.0, 1.5),
        confidence=0.85,
        class_id=76,
        class_name="scissors",
    )

    print("\n--- Frame 1 : entrée en zone_1 + objet interdit ---")
    alerts = detector.check([det_in_zone1, det_outside, det_forbidden_obj])
    print(f"[INFO] {len(alerts)} alerte(s) déclenchée(s)")
    for a in alerts:
        print(f"  {a}")

    print("\n--- Frame 2 : même position (pas de nouvelle alerte attendue) ---")
    alerts2 = detector.check([det_in_zone1, det_outside])
    print(f"[INFO] {len(alerts2)} alerte(s) déclenchée(s)")

    print("\n--- Frame 3 : personne sort de zone_1 ---")
    alerts3 = detector.check([det_outside])
    print(f"[INFO] {len(alerts3)} alerte(s) déclenchée(s)")

    print("\n--- Frame 4 : re-entrée en zone_1 (nouvelle alerte attendue) ---")
    alerts4 = detector.check([det_in_zone1])
    print(f"[INFO] {len(alerts4)} alerte(s) déclenchée(s)")
    for a in alerts4:
        print(f"  {a}")

    print("\n[INFO] Smoke test terminé avec succès.")
    sys.exit(0)

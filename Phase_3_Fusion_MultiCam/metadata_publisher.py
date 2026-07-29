"""Export des métadonnées temps réel de la Phase 3 vers la Phase 4.

Place dans le pipeline : interface de sortie de la Phase 3. Le transport
vidéo reste séparé des métadonnées IA : ce module sérialise, pour chaque
frame traitée, les détections fusionnées et les alertes dans une enveloppe
JSON qui peut être :
  - écrite en JSONL sur disque (rejeu et vérifications hors ligne) ;
  - envoyée en POST HTTP au tableau de bord de la Phase 4.

Schéma de l'enveloppe publiée (une par frame) :
  schema           : identifiant de version ("benchmarkingai.phase3.metadata.v1")
  created_epoch_ms : date de création (epoch, millisecondes)
  created_epoch_s  : date de création (epoch, secondes)
  frame            : index de la frame dans la session
  run_label        : étiquette de la session (traçabilité des runs)
  model_version    : version du modèle de détection
  model            : nom du modèle
  format           : format d'export du modèle (pt, engine, ...)
  detections       : liste de détections (voir detection_to_metadata)
  alerts           : liste d'alertes (voir alert_to_metadata)
  timing           : mesures de temps optionnelles (clé absente sinon)
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def _round_list(values: Any, digits: int = 3) -> list[float]:
    """Arrondit une séquence de valeurs numériques en liste de floats (JSON compact)."""
    return [round(float(value), digits) for value in values]


def detection_to_metadata(det: Any, zones: list[str] | None = None) -> dict[str, Any]:
    """Sérialise une Detection en dictionnaire publiable vers la Phase 4.

    Champs produits :
      camera_id     : identifiant de la caméra source
      track_id      : identifiant de piste local (ByteTrack)
      global_id     : identifiant global inter-caméras (None avant fusion)
      class_id      : indice de classe COCO
      class_name    : nom de classe lisible
      confidence    : confiance de détection, arrondie à 4 décimales
      bbox_px       : boîte englobante (x1, y1, x2, y2) en pixels
      foot_point_px : point bas-centre de la bbox en pixels
      position_m    : point au sol en mètres, ou None sans homographie
      zones         : zones interdites contenant ce point (liste vide sinon)
      timestamp     : horodatage de la frame en secondes
    """
    position_m = _round_list(det.foot_point_m, 3) if det.foot_point_m else None
    return {
        "camera_id": det.cam_id,
        "track_id": int(det.track_id),
        "global_id": det.global_id,
        "class_id": int(det.class_id),
        "class_name": det.class_name,
        "confidence": round(float(det.confidence), 4),
        "bbox_px": _round_list(det.bbox_px, 2),
        "foot_point_px": _round_list(det.foot_point_px, 2),
        "position_m": position_m,
        "zones": zones or [],
        "timestamp": round(float(det.timestamp), 6),
    }


def alert_to_metadata(alert: Any) -> dict[str, Any]:
    """Sérialise une Alert en dictionnaire publiable vers la Phase 4.

    Reprend les champs de l'alerte : identifiant, type, niveau
    (weak / confirmed), entité globale, zone, classe, position en mètres,
    caméras votantes, confiance et horodatage, avec arrondis pour un JSON
    compact.
    """
    return {
        "alert_id": alert.alert_id,
        "alert_type": alert.alert_type,
        "alert_level": getattr(alert, "alert_level", "confirmed"),
        "global_id": alert.global_id,
        "zone_id": alert.zone_id,
        "class_name": alert.class_name,
        "position_m": _round_list(alert.position_m, 3),
        "cameras": list(alert.cameras),
        "confidence": round(float(alert.confidence), 4),
        "timestamp": round(float(alert.timestamp), 6),
    }


class MetadataPublisher:
    """Publie les métadonnées par frame de la Phase 3 sans bloquer la boucle IA.

    L'écriture JSONL est synchrone (locale et rapide). L'envoi HTTP passe par
    une file bornée vidée par un thread démon : si la Phase 4 est lente ou
    absente, les enveloppes excédentaires sont abandonnées (compteur dropped)
    plutôt que de ralentir le traitement vidéo temps réel.
    """

    def __init__(
        self,
        jsonl_path: Path | None = None,
        http_url: str | None = None,
        every_n_frames: int = 1,
        http_timeout_s: float = 0.5,
        queue_size: int = 200,
    ) -> None:
        """Initialise le publieur.

        Args:
            jsonl_path: Fichier JSONL de sortie (dossiers créés au besoin),
                        ou None pour désactiver l'écriture disque.
            http_url: URL du POST HTTP vers la Phase 4, ou None pour désactiver.
            every_n_frames: Ne publie qu'une frame sur N (1 = toutes).
            http_timeout_s: Délai maximal d'un POST avant abandon de l'enveloppe.
            queue_size: Taille de la file HTTP ; au-delà, les enveloppes sont
                        comptées comme abandonnées.
        """
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.http_url = http_url
        self.every_n_frames = max(1, int(every_n_frames))
        self.http_timeout_s = max(0.001, float(http_timeout_s))
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._sent = 0
        self._thread: threading.Thread | None = None

        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.http_url:
            self._thread = threading.Thread(target=self._http_worker, daemon=True)
            self._thread.start()

    @property
    def dropped(self) -> int:
        """Nombre d'enveloppes abandonnées (file pleine ou échec HTTP)."""
        return self._dropped

    @property
    def sent(self) -> int:
        """Nombre d'enveloppes envoyées avec succès en HTTP."""
        return self._sent

    def publish(
        self,
        frame_idx: int,
        run_label: str,
        model_version: str,
        model_name: str,
        format_label: str,
        detections: list[dict[str, Any]],
        alerts: list[dict[str, Any]],
        timing: dict[str, Any] | None = None,
    ) -> None:
        """Construit l'enveloppe de la frame et la publie (JSONL et/ou HTTP).

        Les frames dont l'index n'est pas multiple de every_n_frames sont
        ignorées (sous-échantillonnage). L'enveloppe suit le schéma décrit
        dans la docstring de module. L'envoi HTTP est non bloquant : file
        pleine = enveloppe comptée comme abandonnée, jamais d'attente.

        Args:
            frame_idx: Index de la frame dans la session.
            run_label: Étiquette de la session (traçabilité des runs).
            model_version: Version du modèle de détection.
            model_name: Nom du modèle.
            format_label: Format d'export du modèle (pt, engine, ...).
            detections: Détections sérialisées via detection_to_metadata().
            alerts: Alertes sérialisées via alert_to_metadata().
            timing: Mesures de temps optionnelles ajoutées à l'enveloppe.
        """
        if frame_idx % self.every_n_frames != 0:
            return

        now_ms = time.time() * 1000.0
        envelope = {
            "schema": "benchmarkingai.phase3.metadata.v1",
            "created_epoch_ms": now_ms,
            "created_epoch_s": round(now_ms / 1000.0, 6),
            "frame": int(frame_idx),
            "run_label": run_label,
            "model_version": model_version,
            "model": model_name,
            "format": format_label,
            "detections": detections,
            "alerts": alerts,
        }
        if timing:
            envelope["timing"] = timing

        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n")

        if self.http_url:
            try:
                self._queue.put_nowait(envelope)
            except queue.Full:
                self._dropped += 1

    def close(self) -> None:
        """Arrête proprement le thread HTTP (sentinelle None) et affiche les compteurs."""
        if self._thread:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=1.0)
        if self.http_url:
            print(
                "[INFO] Metadata publisher: "
                f"sent={self._sent} dropped={self._dropped} "
                f"queued={self._queue.qsize()}"
            )

    def _http_worker(self) -> None:
        """Boucle du thread démon : dépile et poste chaque enveloppe vers la Phase 4.

        Une sentinelle None arrête le thread. Tout échec réseau est compté
        comme abandonné, jamais propagé à la boucle IA.
        """
        while True:
            envelope = self._queue.get()
            if envelope is None:
                return
            data = json.dumps(envelope).encode("utf-8")
            request = urllib.request.Request(
                self.http_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.http_timeout_s):
                    self._sent += 1
            except Exception:
                self._dropped += 1

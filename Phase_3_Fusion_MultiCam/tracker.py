"""
Module de suivi par caméra avec ByteTrack (Étape 3.2).

Place dans le pipeline : premier maillon de la Phase 3. Chaque caméra possède
son CameraTracker ; les détections produites (avec position au sol en mètres)
sont ensuite associées entre caméras par fusion.py.

CameraTracker encapsule :
  - Le chargement du modèle YOLO/RT-DETR
  - Le correctif ratio d'aspect Dahua (geometry_fix)
  - Le suivi ByteTrack frame par frame (model.track + persist=True)
  - La projection homographique foot_point pixels → mètres

Usage :
  from tracker import CameraTracker
  import yaml, cv2

  config = yaml.safe_load(open("config.yaml"))
  tracker = CameraTracker("cam_03", "../yolov8n.pt", config)

  cap = cv2.VideoCapture(video_path)
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      detections = tracker.process_frame(frame, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
      for det in detections:
          print(det)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from detection import Detection
from geometry_fix import fix_frame, load_aspect_fix


# Chemin de la configuration ByteTrack, relatif à ce fichier
_BYTETRACK_CFG = str(Path(__file__).parent / "bytetrack.yaml")


class CameraTracker:
    """Tracker mono-caméra : détection YOLO/RT-DETR + suivi ByteTrack + projection sol.

    Attributes:
        cam_id: Identifiant caméra (ex. "cam_03").
        model: Instance Ultralytics YOLO/RT-DETR chargée.
        homography: Matrice 3x3 numpy float32 (pixels → mètres), ou None.
        ar_fix: Configuration du correctif ratio d'aspect.
        room_id: Salle à laquelle appartient la caméra.
        conf_threshold: Seuil de confiance minimal pour filtrer les résultats.
        target_classes: Liste d'indices de classes à suivre. [0] = personnes uniquement.
    """

    def __init__(
        self,
        cam_id: str,
        model_path: str,
        config: dict,
        conf_threshold: float = 0.5,
        target_classes: list[int] | None = None,
        device: str | None = None,
    ) -> None:
        """Initialise le tracker de la caméra.

        Args:
            cam_id: Identifiant caméra correspondant à la clé de config.yaml
                    (ex. "cam_03").
            model_path: Chemin du fichier modèle .pt ou .engine.
            config: Dictionnaire issu du parsing de config.yaml.
            conf_threshold: Confiance minimale pour conserver une détection.
            target_classes: Indices de classes COCO à suivre. None suit toutes
                           les classes. Utiliser [0] pour les personnes seules.
            device: Périphérique d'inférence ("cuda:0", "cpu", ...). None
                    reprend la valeur du config (detection.device).
        """
        self.cam_id = cam_id
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes  # None = toutes les classes
        self._track_failures = 0
        self._disabled_after_failures = False
        configured_device = config.get("detection", {}).get("device", "auto")
        self.device = device if device is not None else configured_device
        if self.device in ("auto", "", None):
            self.device = None

        # -- Charger le modèle --
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERR] Modèle introuvable : {model_path}")
        print(f"[INFO] {cam_id} : Chargement modèle : {model_path}")
        self.model = self._load_model()

        # -- Correctif ratio d'aspect --
        self.ar_fix = load_aspect_fix(config)

        # -- Homographie (pixels → mètres) --
        self.homography: Optional[np.ndarray] = None
        self._load_homography(config)

        # -- Identifiant de salle (room) --
        cam_cfg = config.get("cameras", {}).get(cam_id, {})
        self.room_id: str | None = cam_cfg.get("room", None)

        print(
            f"[INFO] {cam_id} prêt | room={self.room_id} | "
            f"homographie={'OK' if self.homography is not None else 'ABSENTE'} | "
            f"ar_fix={'ON' if cam_id in self.ar_fix.get('cameras', set()) else 'OFF'} | "
            f"device={self.device or 'auto'}"
        )

    # ------------------------------------------------------------------
    # Initialisation interne
    # ------------------------------------------------------------------

    def _load_model(self):
        return YOLO(self.model_path, task="detect")

    def _reset_model_after_track_error(self, exc: Exception) -> None:
        """Réinitialise l'état Ultralytics/ByteTrack après un échec interne du tracker.

        Au troisième échec consécutif, la caméra est désactivée : mieux vaut
        perdre une caméra que bloquer tout le pipeline multi-caméras.
        """
        self._track_failures += 1
        if self._track_failures >= 3:
            self._disabled_after_failures = True
            print(
                f"[ERR] {self.cam_id} : ByteTrack désactivé pour ce modèle après "
                f"{self._track_failures} échecs consécutifs "
                f"({type(exc).__name__}: {exc})."
            )
            return
        print(
            f"[WARN] {self.cam_id} : ByteTrack/Ultralytics a échoué "
            f"({type(exc).__name__}: {exc}); modèle rechargé et frame ignorée."
        )
        self.model = self._load_model()

    def _load_homography(self, config: dict) -> None:
        """Charge la matrice d'homographie de cette caméra depuis le config.

        Essaie d'abord le format imbriqué (config['homographie'][cam_id]['matrix']),
        puis se rabat sur le format plat (config['homographie'][cam_id + '_matrix']).
        """
        homo_section = config.get("homographie", {})

        # Format imbriqué : homographie > cam_03 > matrix
        cam_entry = homo_section.get(self.cam_id, None)
        if isinstance(cam_entry, dict):
            matrix = cam_entry.get("matrix", None)
            if matrix:
                self.homography = np.array(matrix, dtype=np.float32)
                print(f"[INFO] {self.cam_id} : Homographie chargée (format imbriqué)")
                return

        # Format plat : homographie > cam_03_matrix
        flat_key = f"{self.cam_id}_matrix"
        matrix = homo_section.get(flat_key, None)
        if matrix:
            arr = np.array(matrix, dtype=np.float32)
            if arr.size > 0:
                self.homography = arr
                print(f"[INFO] {self.cam_id} : Homographie chargée (format plat)")
                return

        print(f"[WARN] {self.cam_id} : Pas d'homographie dans config.yaml. "
              f"foot_point_m sera None jusqu'à calibration.")

    # ------------------------------------------------------------------
    # Traitement de frame
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, timestamp: float | None = None
    ) -> list[Detection]:
        """Traite une frame et retourne la liste des détections suivies.

        Enchaînement :
          1. fix_frame() : correctif ratio d'aspect
          2. model.track(persist=True) : détection + ByteTrack
          3. Pour chaque résultat : extraire bbox, foot_point, projeter en mètres
          4. Créer et retourner des objets Detection

        Args:
            frame: Frame brute issue de cv2.VideoCapture.read() (BGR, HxWxC).
            timestamp: Horodatage de la frame en secondes. time.time() par défaut.

        Returns:
            Liste d'objets Detection. Liste vide si aucune détection.
        """
        if self._disabled_after_failures:
            return []

        if timestamp is None:
            timestamp = time.time()

        # 1. Correctif ratio d'aspect : OBLIGATOIRE avant toute inférence.
        # L'homographie a été calibrée sur l'image corrigée ; détecter ou
        # projeter sur l'image déformée fausserait toutes les positions au sol.
        frame = fix_frame(frame, self.cam_id, self.ar_fix)

        # 2. Suivi ByteTrack (persist=True conserve l'état du tracker entre frames)
        track_kwargs = {
            "persist": True,
            "tracker": _BYTETRACK_CFG,
            "conf": self.conf_threshold,
            "classes": self.target_classes,
            "verbose": False,
        }
        if self.device is not None:
            track_kwargs["device"] = self.device

        try:
            results = self.model.track(frame, **track_kwargs)
        except Exception as exc:
            self._reset_model_after_track_error(exc)
            return []
        self._track_failures = 0

        detections: list[Detection] = []

        if results is None or len(results) == 0:
            return detections

        result = results[0]

        # Pas de boîtes détectées
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes

        # Pas de track_ids disponibles (première frame ou ByteTrack non initialisé)
        if boxes.id is None:
            return detections

        for i in range(len(boxes)):
            track_id_tensor = boxes.id[i]
            if track_id_tensor is None:
                continue

            track_id = int(track_id_tensor.item())
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            cls_name = self.model.names.get(cls_id, str(cls_id))

            # Coordonnées bbox en pixels (x1, y1, x2, y2) dans l'image corrigée
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox_px = (float(x1), float(y1), float(x2), float(y2))
            if not self._is_valid_bbox(bbox_px):
                print(f"[WARN] {self.cam_id} : bbox invalide ignorée: {bbox_px}")
                continue
            if not np.isfinite(conf):
                print(f"[WARN] {self.cam_id} : confidence invalide ignorée: {conf}")
                continue

            # Foot point = bas-centre de la bbox, approximation du point de
            # contact avec le sol. C'est le seul point de la boîte censé être
            # sur le plan du sol : l'homographie n'est valable que pour les
            # points de ce plan, projeter le centre de la bbox (à mi-hauteur
            # du corps) donnerait une position en mètres aberrante.
            foot_u = (x1 + x2) / 2.0
            foot_v = float(y2)
            foot_point_px = (foot_u, foot_v)

            # Projection homographique : pixels → mètres
            foot_point_m = self._project_to_metres(foot_u, foot_v)

            det = Detection(
                cam_id=self.cam_id,
                track_id=track_id,
                global_id=None,
                timestamp=timestamp,
                bbox_px=bbox_px,
                foot_point_px=foot_point_px,
                foot_point_m=foot_point_m,
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
                room_id=self.room_id,
            )
            detections.append(det)

        return detections

    # ------------------------------------------------------------------
    # Projection homographique
    # ------------------------------------------------------------------

    def _project_to_metres(
        self, u: float, v: float
    ) -> tuple[float, float] | None:
        """Projette un point pixel en coordonnées sol (mètres) via l'homographie.

        Args:
            u: Coordonnée x en pixels (après correctif ratio d'aspect).
            v: Coordonnée y en pixels (après correctif ratio d'aspect).

        Returns:
            Coordonnées sol (x_m, y_m) en mètres, ou None sans homographie.
        """
        if self.homography is None:
            return None
        if not np.isfinite([u, v]).all():
            return None

        # cv2.perspectiveTransform attend shape (1, 1, 2) en float32
        pt = np.array([[[u, v]]], dtype=np.float32)
        pt_m = cv2.perspectiveTransform(pt, self.homography)
        x_m = float(pt_m[0, 0, 0])
        y_m = float(pt_m[0, 0, 1])
        if not np.isfinite([x_m, y_m]).all():
            return None
        return (x_m, y_m)

    @staticmethod
    def _is_valid_bbox(bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        if not np.isfinite([x1, y1, x2, y2]).all():
            return False
        return x2 > x1 and y2 > y1

    # ------------------------------------------------------------------
    # Dessin (utilitaire de visualisation)
    # ------------------------------------------------------------------

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        show_metres: bool = True,
    ) -> np.ndarray:
        """Dessine les boîtes englobantes et leurs labels sur une frame (visualisation).

        IMPORTANT : frame doit DÉJÀ avoir subi fix_frame() (même frame que celle
        passée à process_frame). Ne pas appeler fix_frame() une seconde fois ici.

        Args:
            frame: Frame déjà corrigée en ratio d'aspect.
            detections: Liste d'objets Detection issus de process_frame().
            show_metres: Si True, affiche les coordonnées sol en mètres sur la bbox.

        Returns:
            Frame annotée (BGR).
        """
        vis = frame.copy()
        for det in detections:
            if not self._is_valid_bbox(det.bbox_px):
                continue
            x1, y1, x2, y2 = [int(v) for v in det.bbox_px]

            # Couleur basée sur track_id (reproductible, visuellement distincte)
            color = _id_to_color(det.track_id)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label principal
            label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
            if show_metres and det.foot_point_m is not None:
                xm, ym = det.foot_point_m
                label += f" ({xm:.2f}m,{ym:.2f}m)"

            _draw_label(vis, label, x1, y1, color)

            # Point de contact au sol (bas-centre bbox)
            fu, fv = int(det.foot_point_px[0]), int(det.foot_point_px[1])
            cv2.circle(vis, (fu, fv), 4, color, -1)

        return vis


# ----------------------------------------------------------------------
# Utilitaires de dessin
# ----------------------------------------------------------------------

def _id_to_color(track_id: int) -> tuple[int, int, int]:
    """Associe à un track_id une couleur BGR reproductible."""
    np.random.seed(track_id * 97 + 13)
    return tuple(int(c) for c in np.random.randint(80, 255, 3))  # type: ignore[return-value]


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Dessine un label sur fond plein au-dessus d'une boîte englobante."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    top = max(y - th - baseline - 4, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 2, y), color, -1)
    cv2.putText(frame, text, (x + 1, y - baseline - 1), font, scale, (0, 0, 0), thick)


# ----------------------------------------------------------------------
# Test basique (sans GPU, smoke test seulement)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("[INFO] Smoke test CameraTracker : chargement config + vérification homographie")

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Chercher un modèle dans le répertoire parent
    parent = Path(__file__).parent.parent
    model_candidates = list(parent.glob("yolov8n.pt")) + list(parent.glob("yolo11n.pt"))
    if not model_candidates:
        print("[ERR] Aucun modèle .pt trouvé dans", parent)
        print("      Placez yolov8n.pt dans", parent, "pour lancer ce test.")
        sys.exit(1)

    model_path = str(model_candidates[0])
    print(f"[INFO] Modèle trouvé : {model_path}")

    # Instancier un tracker pour cam_03
    tracker = CameraTracker(
        cam_id="cam_03",
        model_path=model_path,
        config=cfg,
        conf_threshold=0.5,
        target_classes=[0],  # personnes seulement
    )

    # Test sur une frame noire synthétique (pas de vraie vidéo requise)
    fake_frame = np.zeros((576, 704, 3), dtype=np.uint8)
    dets = tracker.process_frame(fake_frame, timestamp=0.0)
    print(f"[INFO] Frame noire → {len(dets)} détection(s) (attendu : 0)")

    # Test de la projection homographique
    if tracker.homography is not None:
        pt_m = tracker._project_to_metres(140.0, 400.0)
        print(f"[INFO] Projection (140px, 400px) → {pt_m} m")
    else:
        print("[WARN] Pas d'homographie disponible pour cam_03 : calibrer d'abord.")

    print("[INFO] Smoke test OK")

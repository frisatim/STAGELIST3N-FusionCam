"""
Pipeline de surveillance multi-caméras du système industriel (Phase 3).

Point d'entrée du coeur de la Phase 3 : orchestre les instances CameraTracker
(une par caméra), MultiCameraFusion (identifiants globaux inter-caméras) et
ViolationDetector (zones interdites, objets interdits), sur des vidéos
enregistrées ou des flux RTSP en direct.

Déroulé par frame :
  1. Lecture d'une frame par caméra (fichiers ou RTSP).
  2. Suivi mono-caméra (YOLO + ByteTrack) et projection au sol, par caméra.
  3. Fusion inter-caméras : attribution des identifiants globaux.
  4. Détection de violations et émission d'alertes.
  5. Affichage optionnel : mosaïque annotée et bandeau d'alertes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import deque
from pathlib import Path

# À garder avant l'import de cv2 : sous Windows, OpenCV ne voit les DLL
# GStreamer que si elles sont déjà sur le PATH au chargement de cv2.
# _GST_BIN pointe vers l'installation GStreamer Windows ; si ce chemin
# n'existe pas (Linux, machine sans GStreamer), le bloc est simplement ignoré.
_GST_BIN = r"C:\gstreamer\1.0\msvc_x86_64\bin"
if os.path.exists(_GST_BIN):
    os.environ["PATH"] = _GST_BIN + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_GST_BIN)
    except (AttributeError, OSError):
        pass

import cv2
import numpy as np
import yaml

from alert import Alert
from detection import Detection
from fusion import MultiCameraFusion
from geometry_fix import fix_frame, load_aspect_fix
from tracker import CameraTracker
from video_capture import CaptureOptions, open_capture_with_info
from violation_detector import ViolationDetector

# Nombre de cellules par ligne dans la mosaïque d'affichage
COLS_PER_ROW: int = 4


# ----------------------------------------------------------------------
# Utilitaires de dessin (niveau module)
# ----------------------------------------------------------------------

def _global_id_to_color(gid: int) -> tuple[int, int, int]:
    """Dérive du global_id une couleur BGR stable (hachage MD5).

    La même entité garde ainsi la même couleur sur toutes les caméras ;
    le plancher à 80 par canal évite les couleurs trop sombres sur la vidéo.
    """
    h = int(hashlib.md5(str(gid).encode()).hexdigest()[:6], 16)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = (h & 0x0000FF)
    r = max(80, r)
    g = max(80, g)
    b = max(80, b)
    return (b, g, r)


def _draw_detections_global(
    frame: np.ndarray,
    detections: list[Detection],
    cam_id: str,
    zones_px: list[tuple[str, np.ndarray]] | None = None,
    violation_detector: ViolationDetector | None = None,
    track_trails: dict[tuple[str, int], deque[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """Annote une frame avec les détections d'une caméra (vue fusion).

    Pour chaque détection de la caméra : boîte englobante colorée par
    identifiant global (rouge si le point au sol est dans une zone interdite),
    label (global_id, track_id, classe, confiance, position en mètres),
    traînée du centre de la boîte, marqueur du point au sol pour les
    personnes et croix pour les objets.
    """
    vis = frame.copy()
    _draw_projected_zones(vis, zones_px or [])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1

    for det in detections:
        if det.cam_id != cam_id:
            continue
        if not _is_valid_bbox(det.bbox_px):
            continue

        gid = det.global_id if det.global_id is not None else 0
        zone_hits = (
            violation_detector.zones_containing(det)
            if violation_detector is not None
            else []
        )
        color = (0, 0, 255) if zone_hits else _global_id_to_color(gid)

        x1, y1, x2, y2 = [int(v) for v in det.bbox_px]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"G{gid}|#{det.track_id} c{det.class_id}:{det.class_name} {det.confidence:.2f}"
        if det.foot_point_m is not None:
            xm, ym = det.foot_point_m
            label += f" ({xm:.2f}m,{ym:.2f}m)"
        if zone_hits:
            label += f" IN {','.join(zone_hits)}"

        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(vis, (x1, top), (x1 + tw + 2, y1), color, -1)
        cv2.putText(vis, label, (x1 + 1, y1 - baseline - 1), font, scale, (0, 0, 0), thick)

        is_person = (
            violation_detector.is_person_detection(det)
            if violation_detector is not None
            else det.is_person
        )
        fu, fv = int(det.foot_point_px[0]), int(det.foot_point_px[1])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if track_trails is not None:
            trail = track_trails.setdefault((det.cam_id, det.track_id), deque(maxlen=60))
            trail.append((cx, cy))
            if len(trail) >= 2:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=2)
            cv2.putText(
                vis,
                f"age={len(trail)}",
                (cx + 8, cy - 8),
                font,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"age={len(trail)}",
                (cx + 8, cy - 8),
                font,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.circle(vis, (cx, cy), 4, color, -1)
        if is_person:
            cv2.circle(vis, (fu, fv), 6 if zone_hits else 4, color, -1)
            cv2.circle(vis, (fu, fv), 8, (255, 255, 255), 1)
        else:
            cv2.drawMarker(
                vis,
                (cx, cy),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )

    return vis


def _is_valid_bbox(bbox: tuple[float, float, float, float]) -> bool:
    """Vrai si la bbox est finie et de surface strictement positive."""
    x1, y1, x2, y2 = bbox
    return bool(np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1)


def _draw_projected_zones(
    frame: np.ndarray,
    zones_px: list[tuple[str, np.ndarray]],
) -> None:
    """Dessine les polygones de zones interdites déjà projetés en pixels caméra."""
    if not zones_px:
        return

    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for zone_id, pts in zones_px:
        if pts.size == 0:
            continue
        cv2.fillPoly(overlay, [pts], color=(0, 0, 130))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
        cx, cy = pts.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(
            frame,
            zone_id,
            (int(cx) - 24, int(cy)),
            font,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def _build_grid(
    frames_by_cam: dict[str, np.ndarray],
    cam_ids: list[str] | None = None,
    status_by_cam: dict[str, str] | None = None,
    cell_w: int = 480,
    cell_h: int = 360,
) -> np.ndarray:
    """Assemble les frames caméra en une mosaïque à COLS_PER_ROW colonnes.

    Les caméras sans frame reçoivent une cellule de statut ; les cellules
    manquantes de la dernière ligne sont remplies de noir.
    """
    cam_ids = cam_ids or sorted(frames_by_cam.keys())
    status_by_cam = status_by_cam or {}
    cells: list[np.ndarray] = []

    for cam_id in cam_ids:
        frame = frames_by_cam.get(cam_id)
        if frame is None:
            resized = _make_status_cell(
                cam_id,
                status_by_cam.get(cam_id, "NO FRAME"),
                cell_w,
                cell_h,
            )
        else:
            resized = cv2.resize(frame, (cell_w, cell_h))
            status = status_by_cam.get(cam_id)
            if status:
                _draw_status_text(resized, cam_id, status)
        cells.append(resized)

    n = len(cells)
    n_cols = COLS_PER_ROW
    n_rows = (n + n_cols - 1) // n_cols
    total_cells = n_rows * n_cols

    black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    while len(cells) < total_cells:
        cells.append(black_cell)

    rows: list[np.ndarray] = []
    for r in range(n_rows):
        row_cells = cells[r * n_cols : (r + 1) * n_cols]
        rows.append(np.hstack(row_cells))

    return np.vstack(rows)


def _make_status_cell(cam_id: str, status: str, cell_w: int, cell_h: int) -> np.ndarray:
    """Crée une cellule noire portant l'identifiant caméra et son statut."""
    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    _draw_status_text(cell, cam_id, status)
    return cell


def _draw_status_text(frame: np.ndarray, cam_id: str, status: str) -> None:
    """Écrit l'identifiant caméra et son statut en haut de la cellule."""
    cv2.putText(
        frame,
        cam_id,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 255) if status == "OK" else (0, 80, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_alert_banner(grid: np.ndarray, active_alerts: list[Alert]) -> np.ndarray:
    """Ajoute sous la mosaïque un bandeau listant les trois dernières alertes."""
    banner_h = 40
    banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    recent = active_alerts[-3:] if len(active_alerts) > 3 else active_alerts
    for i, alert in enumerate(recent):
        text = f"[ALERT] {alert}"
        x = 6
        y = 14 + i * 13
        cv2.putText(banner, text, (x, y), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([grid, banner])


# ----------------------------------------------------------------------
# Classe pipeline
# ----------------------------------------------------------------------

class MultiCameraPipeline:
    """Orchestrateur de la Phase 3 : trackers par caméra, fusion et violations.

    À la construction : charge config.yaml, instancie un CameraTracker par
    caméra activée et calibrée (homographie présente), projette les zones
    interdites dans chaque image caméra pour l'affichage, puis crée la fusion
    et le détecteur de violations partagés par toutes les caméras.
    """

    def __init__(
        self,
        config_path: str,
        model_path: str = "yolov8n.pt",
        fusion_distance_threshold_m: float | None = None,
        fusion_time_window_ms: float | None = None,
        camera_ids: list[str] | None = None,
        device: str | None = None,
        capture_options: CaptureOptions | None = None,
        target_classes: list[int] | None = None,
        person_class_ids: list[int] | None = None,
    ) -> None:
        """Charge la configuration et construit trackers, fusion et détecteur.

        Args:
            config_path: Chemin de config.yaml (caméras, homographies, zones, fusion).
            model_path: Poids YOLO/RT-DETR communs à toutes les caméras.
            fusion_distance_threshold_m: Seuil de distance de fusion en mètres
                (défaut : valeur du config, sinon 1.0).
            fusion_time_window_ms: Fenêtre temporelle de fusion en millisecondes
                (défaut : valeur du config, sinon 500).
            camera_ids: Sous-ensemble de caméras à lancer (None = toutes les actives).
            device: Périphérique d'inférence ("cuda:0", "cpu", ...), sinon config.
            capture_options: Options d'ouverture des flux (backend, GStreamer).
            target_classes: Classes COCO à suivre (None = toutes).
            person_class_ids: Classes considérées comme personnes pour les zones.
        """
        with open(config_path, "r", encoding="utf-8") as fh:
            self._config: dict = yaml.safe_load(fh)
        if person_class_ids is not None:
            self._config.setdefault("detection", {})["person_class_ids"] = person_class_ids

        self._config_path = config_path
        self._model_path = model_path
        self._ar_fix = load_aspect_fix(self._config)
        fusion_cfg = self._config.get("fusion", {})
        distance_threshold_m = (
            fusion_distance_threshold_m
            if fusion_distance_threshold_m is not None
            else fusion_cfg.get("distance_threshold_m", 1.0)
        )
        time_window_ms = (
            fusion_time_window_ms
            if fusion_time_window_ms is not None
            else fusion_cfg.get("time_window_ms", 500)
        )

        conf_threshold: float = self._config.get("detection", {}).get("confidence", 0.5)
        self._capture_options = capture_options or CaptureOptions()
        self._device = device if device is not None else self._config.get("detection", {}).get("device", "auto")
        if self._device in ("auto", "", None):
            self._device = None
        self._zone_polygons_px = self._build_zone_polygons_by_cam(self._config)
        self._track_trails: dict[tuple[str, int], deque[tuple[int, int]]] = {}

        self._trackers: dict[str, CameraTracker] = {}
        self._camera_order: list[str] = []
        self._capture_status: dict[str, str] = {}
        cameras_section: dict = self._config.get("cameras", {})
        requested_cameras = set(camera_ids or [])

        # Ne conserve que les caméras demandées, activées dans le config et
        # calibrées : sans homographie, aucune position au sol n'est possible,
        # donc ni fusion ni détection de violation pour cette caméra.
        for cam_id, cam_cfg in cameras_section.items():
            if requested_cameras and cam_id not in requested_cameras:
                print(f"[WARN] {cam_id} : not requested, skipping.")
                continue
            if not cam_cfg.get("enabled", False):
                print(f"[WARN] {cam_id} : disabled in config, skipping.")
                continue
            if not self._has_homography(self._config, cam_id):
                print(f"[WARN] {cam_id} : no homography matrix found, skipping.")
                continue
            tracker = CameraTracker(
                cam_id=cam_id,
                model_path=model_path,
                config=self._config,
                conf_threshold=conf_threshold,
                target_classes=target_classes,
                device=self._device,
            )
            self._trackers[cam_id] = tracker
            self._camera_order.append(cam_id)
            self._capture_status[cam_id] = "NOT OPENED"
            print(f"[INFO] {cam_id} : tracker loaded.")

        self._fusion = MultiCameraFusion(
            config=self._config,
            distance_threshold_m=distance_threshold_m,
            time_window_s=time_window_ms / 1000.0,
        )

        self._violation_detector = ViolationDetector(self._config)

    # ------------------------------------------------------------------
    # Méthodes d'exécution publiques
    # ------------------------------------------------------------------

    def run_on_recordings(
        self,
        max_frames: int | None = None,
        display: bool = True,
    ) -> None:
        """Exécute le pipeline sur les vidéos enregistrées du config.

        L'horodatage de chaque frame est reconstruit à partir de l'index de
        frame et du FPS de la vidéo, afin que la fenêtre temporelle de fusion
        reste cohérente entre caméras.

        Args:
            max_frames: Arrêt après ce nombre de frames (None = jusqu'à épuisement).
            display: Affiche la mosaïque OpenCV annotée ('q' pour quitter).
        """
        captures = self._open_captures(use_rtsp=False)
        if not captures:
            print("[ERR] No video captures could be opened.")
            return

        fps_map: dict[str, float] = {}
        for cam_id, cap in captures.items():
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_map[cam_id] = fps if fps > 0 else 25.0

        self._fusion.reset()

        frame_count = 0
        alert_log: list[Alert] = []

        try:
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    print(f"[INFO] Reached max_frames={max_frames}, stopping.")
                    break

                # Une frame par caméra ; les vidéos épuisées sont absentes du lot
                frames_raw: dict[str, np.ndarray] = {}
                for cam_id, cap in captures.items():
                    ret, frame = cap.read()
                    if ret:
                        frames_raw[cam_id] = frame

                if not frames_raw:
                    print("[INFO] All cameras exhausted, stopping.")
                    break

                detections_by_cam: dict[str, list[Detection]] = {}
                frames_fixed: dict[str, np.ndarray] = {}

                for cam_id, frame in frames_raw.items():
                    fps = fps_map[cam_id]
                    timestamp = frame_count / fps

                    # fix_frame ne sert ici qu'à l'affichage : process_frame
                    # applique lui-même le correctif avant l'inférence, et les
                    # bbox retournées sont exprimées dans l'image corrigée.
                    fixed = fix_frame(frame, cam_id, self._ar_fix)
                    frames_fixed[cam_id] = fixed

                    dets = self._trackers[cam_id].process_frame(frame, timestamp)
                    detections_by_cam[cam_id] = dets

                # Fusion inter-caméras puis détection de violations sur les
                # détections porteuses d'identifiants globaux
                all_detections = self._fusion.associate(detections_by_cam)
                new_alerts = self._violation_detector.check(all_detections)

                for alert in new_alerts:
                    alert_log.append(alert)
                    print(f"[ALERT] {alert}")

                if display:
                    annotated: dict[str, np.ndarray] = {}
                    for cam_id, fixed_frame in frames_fixed.items():
                        cam_dets = detections_by_cam.get(cam_id, [])
                        annotated[cam_id] = _draw_detections_global(
                            fixed_frame,
                            cam_dets,
                            cam_id,
                            zones_px=self._zone_polygons_px.get(cam_id, []),
                            violation_detector=self._violation_detector,
                            track_trails=self._track_trails,
                        )

                    status_by_cam = {
                        cam_id: "OK" if cam_id in annotated else "NO FRAME"
                        for cam_id in self._camera_order
                    }
                    grid = _build_grid(
                        annotated,
                        cam_ids=self._camera_order,
                        status_by_cam=status_by_cam,
                    )
                    grid = _draw_alert_banner(grid, alert_log)
                    cv2.imshow("Pipeline", grid)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] User pressed 'q', stopping.")
                        break

                frame_count += 1

        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt received, stopping.")
        finally:
            for cap in captures.values():
                cap.release()
            if display:
                cv2.destroyAllWindows()

    def run_live(self) -> None:
        """Exécute le pipeline en direct sur les flux RTSP du config.

        Contrairement au mode enregistré, l'horodatage est l'heure de lecture
        réelle (time.time()) et une caméra muette n'arrête pas la boucle :
        sa dernière frame annotée reste affichée avec le statut NO FRAME.
        """
        captures = self._open_captures(use_rtsp=True)
        if not captures:
            print("[ERR] No RTSP captures could be opened.")
            return

        self._fusion.reset()
        alert_log: list[Alert] = []
        last_annotated: dict[str, np.ndarray] = {}
        live_status: dict[str, str] = {
            cam_id: self._capture_status.get(cam_id, "NOT OPENED")
            for cam_id in self._camera_order
        }

        try:
            while True:
                frames_raw: dict[str, np.ndarray] = {}
                frame_timestamps: dict[str, float] = {}
                for cam_id, cap in captures.items():
                    ret, frame = cap.read()
                    if ret:
                        frames_raw[cam_id] = frame
                        frame_timestamps[cam_id] = time.time()
                        live_status[cam_id] = "OK"
                    else:
                        live_status[cam_id] = "NO FRAME"

                if not frames_raw:
                    # Aucune frame ce tour : on continue d'afficher le dernier
                    # état connu plutôt que d'arrêter le pipeline
                    grid = _build_grid(
                        last_annotated,
                        cam_ids=self._camera_order,
                        status_by_cam=live_status,
                    )
                    grid = _draw_alert_banner(grid, alert_log)
                    cv2.imshow("Pipeline", grid)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        print("[INFO] User pressed 'q', stopping.")
                        break
                    continue

                detections_by_cam: dict[str, list[Detection]] = {}
                frames_fixed: dict[str, np.ndarray] = {}

                for cam_id, frame in frames_raw.items():
                    timestamp = frame_timestamps[cam_id]
                    # Comme en mode enregistré : fix_frame ne sert qu'à
                    # l'affichage, process_frame réapplique le correctif
                    fixed = fix_frame(frame, cam_id, self._ar_fix)
                    frames_fixed[cam_id] = fixed

                    dets = self._trackers[cam_id].process_frame(frame, timestamp)
                    detections_by_cam[cam_id] = dets

                all_detections = self._fusion.associate(detections_by_cam)
                new_alerts = self._violation_detector.check(all_detections)

                for alert in new_alerts:
                    alert_log.append(alert)
                    print(f"[ALERT] {alert}")

                annotated: dict[str, np.ndarray] = {}
                for cam_id, fixed_frame in frames_fixed.items():
                    cam_dets = detections_by_cam.get(cam_id, [])
                    annotated[cam_id] = _draw_detections_global(
                        fixed_frame,
                        cam_dets,
                        cam_id,
                        zones_px=self._zone_polygons_px.get(cam_id, []),
                        violation_detector=self._violation_detector,
                        track_trails=self._track_trails,
                    )
                last_annotated.update(annotated)

                grid = _build_grid(
                    last_annotated,
                    cam_ids=self._camera_order,
                    status_by_cam=live_status,
                )
                grid = _draw_alert_banner(grid, alert_log)
                cv2.imshow("Pipeline", grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User pressed 'q', stopping.")
                    break

        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt received, stopping.")
        finally:
            for cap in captures.values():
                cap.release()
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _open_captures(self, use_rtsp: bool) -> dict[str, cv2.VideoCapture]:
        """Ouvre une capture par caméra (vidéo enregistrée ou flux RTSP).

        Les caméras sans source configurée ou dont l'ouverture échoue sont
        écartées avec un statut explicite ; le pipeline tourne avec le reste.
        """
        captures: dict[str, cv2.VideoCapture] = {}
        cameras_section: dict = self._config.get("cameras", {})

        for cam_id in self._trackers:
            cam_cfg = cameras_section.get(cam_id, {})
            source = cam_cfg.get("rtsp_url") if use_rtsp else cam_cfg.get("video_path")

            if not source:
                label = "rtsp_url" if use_rtsp else "video_path"
                print(f"[WARN] {cam_id} : no {label} in config, skipping.")
                continue

            open_result = open_capture_with_info(
                str(source),
                live=use_rtsp,
                options=self._capture_options,
            )
            cap = open_result.cap
            if not cap.isOpened():
                print(
                    f"[WARN] {cam_id} : failed to open "
                    f"{'RTSP stream' if use_rtsp else 'video'}: {source}"
                )
                if open_result.error:
                    print(f"[WARN] {cam_id} : open attempts: {open_result.error}")
                self._capture_status[cam_id] = "OPEN FAILED"
                cap.release()
                continue

            captures[cam_id] = cap
            backend = open_result.backend
            self._capture_status[cam_id] = "OK"
            print(
                f"[INFO] {cam_id} : opened {'RTSP stream' if use_rtsp else 'video'} "
                f"via {backend}: {open_result.source}"
            )

        return captures

    @staticmethod
    def _build_zone_polygons_by_cam(config: dict) -> dict[str, list[tuple[str, np.ndarray]]]:
        """Projette les zones interdites (plan de sol) dans chaque image caméra.

        Les zones sont définies en mètres ; l'homographie inverse (mètres vers
        pixels) permet de les dessiner dans chaque vue concernée. Utilisé
        uniquement pour l'affichage : la détection de violation se fait en
        mètres, jamais en pixels.
        """
        zones_by_cam: dict[str, list[tuple[str, np.ndarray]]] = {}
        for zone_id, zone_data in config.get("zones_interdites", {}).items():
            coords_m = zone_data.get("coordonnees_metres") or []
            cam_ids = zone_data.get("cameras_concernees") or []
            if not coords_m or not cam_ids:
                continue
            pts_m = np.array(coords_m, dtype=np.float32).reshape(-1, 1, 2)
            for cam_id in cam_ids:
                h_matrix = MultiCameraPipeline._select_homography_matrix(config, cam_id)
                if h_matrix is None:
                    print(f"[WARN] Zone {zone_id} non projetée pour {cam_id}: homographie absente.")
                    continue
                try:
                    h_inv = np.linalg.inv(h_matrix)
                except np.linalg.LinAlgError:
                    print(f"[WARN] Zone {zone_id} non projetée pour {cam_id}: homographie non inversible.")
                    continue
                pts_px = cv2.perspectiveTransform(pts_m, h_inv).astype(np.int32)
                zones_by_cam.setdefault(cam_id, []).append((zone_id, pts_px))
                print(f"[INFO] Zone {zone_id} projetée sur {cam_id}.")
        return zones_by_cam

    @staticmethod
    def _select_homography_matrix(config: dict, cam_id: str) -> np.ndarray | None:
        """Retourne la matrice d'homographie de la caméra (format imbriqué ou plat), ou None."""
        homo_section = config.get("homographie", {})
        cam_entry = homo_section.get(cam_id)
        if isinstance(cam_entry, dict):
            matrix = cam_entry.get("matrix")
            if matrix:
                return np.array(matrix, dtype=np.float32)

        flat_key = f"{cam_id}_matrix"
        matrix = homo_section.get(flat_key)
        if matrix:
            return np.array(matrix, dtype=np.float32)

        return None

    @staticmethod
    def _has_homography(config: dict, cam_id: str) -> bool:
        """Vrai si une homographie exploitable existe pour cette caméra dans le config."""
        homo_section = config.get("homographie", {})

        cam_entry = homo_section.get(cam_id, None)
        if isinstance(cam_entry, dict) and cam_entry.get("matrix"):
            return True

        flat_key = f"{cam_id}_matrix"
        matrix = homo_section.get(flat_key, None)
        if matrix:
            arr = np.array(matrix, dtype=np.float32)
            if arr.size > 0:
                return True

        return False


# ----------------------------------------------------------------------
# Point d'entrée CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de surveillance multi-caméras (Phase 3)."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Chemin de config.yaml (défaut : config.yaml à côté de pipeline.py)",
    )
    parser.add_argument(
        "--model",
        default="../yolov8n.pt",
        help="Chemin des poids du modèle YOLO (défaut : ../yolov8n.pt)",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="Identifiants de caméras à lancer, séparés par des virgules, ex. cam_03,cam_05,cam_07.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Nombre maximal de frames à traiter en mode enregistré.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Désactive la fenêtre d'affichage OpenCV.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Utilise les flux RTSP au lieu des vidéos enregistrées.",
    )
    parser.add_argument(
        "--fusion-distance-m",
        type=float,
        default=None,
        help="Surcharge le seuil de distance de fusion D, en mètres.",
    )
    parser.add_argument(
        "--fusion-time-window-ms",
        type=float,
        default=None,
        help="Surcharge la fenêtre temporelle de fusion, en millisecondes.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Périphérique YOLO, ex. cuda:0 ou cpu. Défaut : detection.device du config.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Indices de classes à suivre, séparés par des virgules, ex. 0 pour personnes seules. Défaut : toutes les classes.",
    )
    parser.add_argument(
        "--person-classes",
        default=None,
        help="Indices de classes traités comme personnes pour les violations de zone. Défaut : 0.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=["opencv", "gstreamer"],
        default="opencv",
        help="Backend de capture RTSP pour le mode live.",
    )
    parser.add_argument(
        "--gst-latency-ms",
        type=int,
        default=100,
        help="Latence rtspsrc GStreamer, en millisecondes.",
    )
    parser.add_argument(
        "--gst-protocol",
        choices=["tcp", "udp"],
        default="tcp",
        help="Protocole de transport RTSP pour GStreamer.",
    )
    parser.add_argument(
        "--gst-codec",
        choices=["h264", "h265"],
        default="h264",
        help="Codec vidéo RTSP utilisé par les caméras.",
    )
    parser.add_argument(
        "--gst-decoder",
        default="avdec_h264",
        help="Élément décodeur GStreamer, ex. avdec_h264 ou nvh264dec.",
    )
    parser.add_argument(
        "--gst-pipeline",
        choices=["decodebin", "fixed", "both"],
        default="decodebin",
        help="Style de pipeline GStreamer. decodebin correspond à crash_test_streams.py.",
    )
    parser.add_argument(
        "--no-ffmpeg-fallback",
        action="store_true",
        help="Désactive le repli FFmpeg/OpenCV après échec GStreamer.",
    )
    args = parser.parse_args()

    camera_ids = (
        [cam.strip() for cam in args.cameras.split(",") if cam.strip()]
        if args.cameras
        else None
    )
    target_classes = (
        [int(cls.strip()) for cls in args.classes.split(",") if cls.strip()]
        if args.classes
        else None
    )
    person_class_ids = (
        [int(cls.strip()) for cls in args.person_classes.split(",") if cls.strip()]
        if args.person_classes
        else None
    )

    pipeline = MultiCameraPipeline(
        config_path=args.config,
        model_path=args.model,
        fusion_distance_threshold_m=args.fusion_distance_m,
        fusion_time_window_ms=args.fusion_time_window_ms,
        camera_ids=camera_ids,
        device=args.device,
        target_classes=target_classes,
        person_class_ids=person_class_ids,
        capture_options=CaptureOptions(
            backend=args.capture_backend,
            gst_latency_ms=args.gst_latency_ms,
            gst_protocol=args.gst_protocol,
            gst_codec=args.gst_codec,
            gst_decoder=args.gst_decoder,
            gst_pipeline=args.gst_pipeline,
            fallback_ffmpeg=not args.no_ffmpeg_fallback,
        ),
    )

    if args.live:
        pipeline.run_live()
    else:
        pipeline.run_on_recordings(
            max_frames=args.max_frames,
            display=not args.no_display,
        )

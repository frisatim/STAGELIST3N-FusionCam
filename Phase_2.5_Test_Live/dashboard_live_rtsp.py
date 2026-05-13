"""
Phase 2.5 - Dashboard Live RTSP : Securite Industrielle Temps Reel
===================================================================

Modes disponibles:
  1) backend ultralytics (comportement historique)
  2) backend gstreamer (capture OpenCV + frame skip inference)

Regles metier:
  A) TAD  -- Objet interdit (classes 0..13) detecte n'importe ou
    B) TRD  -- Personne (classe 14), test du point pieds bas-centre
                         contre le polygone de zone
"""

import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np

# ============================================================================
#  CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# -- Modele TensorRT --
MODEL_PATH = str(
    PROJECT_ROOT / "Phase_2_Baseline_MonoCam" / "Modelstrained"
    / "V3" / "yolo11s" / "weights" / "best.engine"
)

# -- Zones interdites (polygones pixel) --
ZONES_JSON = str(PROJECT_ROOT / "Phase_2_Baseline_MonoCam" / "zones_config.json")

# -- Hotes RTSP des 8 cameras --
RTSP_HOSTS = {
    # "cam_01": "<CAMERA_IP>",
    # "cam_02": "<CAMERA_IP>",
    # "cam_03": "<CAMERA_IP>",
    # "cam_04": "<CAMERA_IP>",
    # "cam_05": "<CAMERA_IP>",
    # "cam_06": "<CAMERA_IP>",
    "cam_07": "<CAMERA_IP>",
    # "cam_08": "<CAMERA_IP>",
}

# -- Classes du modele unifie --
PERSON_CLASS = 11
OBJECT_CLASS_MAX = 12

# -- Inference --
DEFAULT_CONF = 0.4
DEFAULT_IMGSZ = 960
DEFAULT_INFER_EVERY = 1
DEFAULT_STALE_MS = 800
DEFAULT_LATENCY_REPORT_SEC = 2.0
DEFAULT_SYNC_TOLERANCE_MS = 200
DEFAULT_USE_TRACKER = False
DEFAULT_TRACKER = "botsort.yaml"
DEFAULT_TRACK_CONF = 0.3
RECONNECT_DELAY = 2

# -- Resolution de reference des polygones zones_config.json --
ZONE_REF_W = 1280
ZONE_REF_H = 720

# -- Couleurs BGR --
GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_RED = (0, 0, 180)


# ============================================================================
#  CHARGEMENT DES ZONES
# ============================================================================

def load_zones(path: str) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERREUR] Fichier zones introuvable : {p}")

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    zones: dict[str, np.ndarray] = {}
    for cam_id, pts in raw.items():
        if pts and len(pts) >= 3:
            zones[cam_id] = np.array(pts, dtype=np.int32)
    return zones


def rescale_polygon(poly: np.ndarray,
                    ref_w: int, ref_h: int,
                    dst_w: int, dst_h: int) -> np.ndarray:
    if ref_w == dst_w and ref_h == dst_h:
        return poly.copy()

    scaled = poly.astype(np.float64)
    scaled[:, 0] *= dst_w / ref_w
    scaled[:, 1] *= dst_h / ref_h
    return np.rint(scaled).astype(np.int32)


# ============================================================================
#  LOGIQUE METIER
# ============================================================================

def foot_contact_points(x1: int, y1: int,
                        x2: int, y2: int) -> list[tuple[int, int]]:
    # TRD: utiliser uniquement le point bas-centre des pieds.
    return [
        (int((x1 + x2) / 2), int(y2)),
    ]


def any_point_in_polygon(points: list[tuple[int, int]],
                         contour: np.ndarray) -> bool:
    for px, py in points:
        if cv2.pointPolygonTest(contour, (float(px), float(py)),
                                measureDist=False) >= 0:
            return True
    return False


# ============================================================================
#  ETAT PAR CAMERA
# ============================================================================

class CamState:
    PERSIST_THRESHOLD = 3

    def __init__(self, cam_id: str, zone_ref: np.ndarray):
        self.cam_id = cam_id
        self._zone_ref = zone_ref
        self.zone_poly: np.ndarray | None = None
        self.zone_contour: np.ndarray | None = None
        self._scaled = False

        self._intrusion_ctr = 0
        self._object_ctr = 0
        self.intrusion = False
        self.object = False

        self.frame_index = 0
        self.last_snapshot: DetectionSnapshot | None = None
        self.last_capture_ts = 0.0
        self.last_latency_ms = 0.0
        self.last_skew_ms = 0.0

    def ensure_scale(self, frame_w: int, frame_h: int):
        if self._scaled:
            return
        self.zone_poly = rescale_polygon(self._zone_ref,
                                         ZONE_REF_W, ZONE_REF_H,
                                         frame_w, frame_h)
        self.zone_contour = self.zone_poly.reshape(-1, 1, 2).astype(np.float32)
        self._scaled = True
        ratio_x = frame_w / ZONE_REF_W
        ratio_y = frame_h / ZONE_REF_H
        if abs(ratio_x - 1.0) > 0.01 or abs(ratio_y - 1.0) > 0.01:
            print(f"  [SCALE] {self.cam_id} : {ZONE_REF_W}x{ZONE_REF_H}"
                  f" -> {frame_w}x{frame_h} (x{ratio_x:.2f}, y{ratio_y:.2f})")

    def update(self, has_intrusion: bool, has_object: bool):
        self._intrusion_ctr = self._intrusion_ctr + 1 if has_intrusion else 0
        self.intrusion = self._intrusion_ctr >= self.PERSIST_THRESHOLD

        self._object_ctr = self._object_ctr + 1 if has_object else 0
        self.object = self._object_ctr >= self.PERSIST_THRESHOLD


@dataclass
class DetectionSnapshot:
    timestamp: float
    person_boxes: list[tuple[int, int, int, int, float, bool, list[tuple[int, int]]]]
    object_boxes: list[tuple[int, int, int, int, int, float]]
    has_intrusion: bool
    has_object: bool


# ============================================================================
#  RENDU VISUEL
# ============================================================================

def draw_zone(frame: np.ndarray, poly: np.ndarray, alert: bool):
    if alert:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], RED)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [poly], True, RED if alert else GREEN, 2)


def draw_person(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                in_zone: bool, feet: list[tuple[int, int]], conf: float):
    color = RED if in_zone else GREEN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for fx, fy in feet:
        cv2.circle(frame, (fx, fy), 7, CYAN, -1)
        cv2.circle(frame, (fx, fy), 7, BLACK, 2)

    label = f"Personne {conf:.0%}"
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)


def draw_object(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                cls_id: int, conf: float, blink_on: bool):
    if not blink_on:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, 3)
    cv2.putText(frame, f"Objet[{cls_id}] {conf:.0%}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, ORANGE, 2)


def draw_banners(frame: np.ndarray, state: CamState):
    h, w = frame.shape[:2]
    alerts: list[str] = []
    if state.intrusion:
        alerts.append("INTRUSION ZONE")
    if state.object:
        alerts.append("OBJET INTERDIT")
    if not alerts:
        return

    banner_h = 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), DARK_RED, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    text = "  |  ".join(alerts)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(frame, text, ((w - tw) // 2, (banner_h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)


def draw_fps(frame: np.ndarray, fps: float, model_label: str):
    text = f"FPS: {fps:.1f} - {model_label}"
    cv2.putText(frame, text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED, 2)


def draw_status(frame: np.ndarray, cam_id: str, state: CamState):
    h = frame.shape[0]
    parts = []
    if state.intrusion:
        parts.append("INTRUSION")
    if state.object:
        parts.append("OBJET")
    status = " | ".join(parts) if parts else "OK"
    color = RED if parts else GREEN
    cv2.putText(frame, f"{cam_id}  {status}", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_latency(frame: np.ndarray, state: CamState):
    """Affiche latence camera->affichage et desynchronisation inter-cameras."""
    text = f"LAT {state.last_latency_ms:.0f}ms | SKEW {state.last_skew_ms:.0f}ms"
    cv2.putText(frame, text, (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, RED, 2)


def draw_snapshot(frame: np.ndarray, snapshot: DetectionSnapshot):
    blink_on = int(time.perf_counter() * 6) % 2 == 0
    for x1, y1, x2, y2, conf, in_zone, feet in snapshot.person_boxes:
        draw_person(frame, x1, y1, x2, y2, in_zone, feet, conf)
    for x1, y1, x2, y2, cls_id, conf in snapshot.object_boxes:
        draw_object(frame, x1, y1, x2, y2, cls_id, conf, blink_on)


# ============================================================================
#  RTSP HELPERS
# ============================================================================

def build_rtsp_url(host: str, subtype: int) -> str:
    return (
        f"rtsp://<USER>:<PASSWORD>@{host}:554/"
        f"cam/realmonitor?channel=1&subtype={subtype}"
    )


def build_gst_pipeline(rtsp_url: str, latency: int, force_tcp: bool) -> str:
    protocol_clause = " protocols=tcp" if force_tcp else ""
    return (
        f'rtspsrc location="{rtsp_url}" latency={latency} drop-on-latency=true'
        f"{protocol_clause} ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "appsink sync=false max-buffers=1 drop=true"
    )


def build_rtsp_variants(rtsp_url: str) -> list[str]:
    """Retourne URL courante puis variante subtype=1 pour fallback robuste."""
    variants = [rtsp_url]
    try:
        parts = urlsplit(rtsp_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if query.get("subtype") != "1":
            query["subtype"] = "1"
            alt = urlunsplit(
                (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
            )
            if alt not in variants:
                variants.append(alt)
    except Exception:
        pass
    return variants


def extract_max_imgsz_from_exception(exc: Exception) -> int | None:
    """Extrait la taille max H/W d'un engine statique depuis un message d'assertion Ultralytics."""
    msg = str(exc)
    m = re.search(r"max model size \(\d+,\s*\d+,\s*(\d+),\s*(\d+)\)", msg)
    if not m:
        return None

    h = int(m.group(1))
    w = int(m.group(2))
    if h <= 0 or w <= 0:
        return None
    return max(h, w)


class CameraCapture:
    def __init__(self, cam_id: str, source: str,
                 backend: str,
                 gst_latency: int,
                 force_tcp: bool,
                 fallback_ffmpeg: bool):
        self.cam_id = cam_id
        self.source = source
        self.backend = backend
        self.gst_latency = gst_latency
        self.force_tcp = force_tcp
        self.fallback_ffmpeg = fallback_ffmpeg
        self.cap: cv2.VideoCapture | None = None
        self.backend_used = ""

    def open(self) -> bool:
        if self.backend == "gstreamer":
            pipeline = build_gst_pipeline(self.source, self.gst_latency, self.force_tcp)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                self.backend_used = "gstreamer"
                return True

            if not self.fallback_ffmpeg:
                return False

            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.backend_used = "ffmpeg-fallback"
                return True
            return False

        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.backend_used = "ffmpeg"
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()


class CameraReader:
    """Lecteur RTSP asynchrone avec reconnexion et fallback backend."""

    def __init__(self, cam_id: str, source: str,
                 gst_latency: int,
                 force_tcp: bool,
                 fallback_ffmpeg: bool):
        self.cam_id = cam_id
        self.source = source
        self.gst_latency = gst_latency
        self.force_tcp = force_tcp
        self.fallback_ffmpeg = fallback_ffmpeg
        self._source_candidates = build_rtsp_variants(source)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._capture_ts = 0.0
        self._connected = False
        self._backend_used = ""
        self._active_source = ""
        self._last_error = "initialisation"
        self._running = True

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _open_capture(self):
        for candidate in self._source_candidates:
            pipeline = build_gst_pipeline(candidate, self.gst_latency, self.force_tcp)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap, "gstreamer", candidate, ""
            cap.release()

            if self.fallback_ffmpeg:
                cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap, "ffmpeg-fallback", candidate, ""
                cap.release()

        return None, "", "", "ouverture RTSP impossible"

    def _reader_loop(self):
        cap = None
        while self._running:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()

                with self._lock:
                    self._connected = False
                    self._last_error = "connexion en cours"

                cap, backend, active_source, err = self._open_capture()
                if cap is None:
                    with self._lock:
                        self._backend_used = ""
                        self._active_source = ""
                        self._last_error = err
                    time.sleep(RECONNECT_DELAY)
                    continue

                with self._lock:
                    self._connected = True
                    self._backend_used = backend
                    self._active_source = active_source
                    self._last_error = ""

            ret, frame = cap.read()
            if not ret or frame is None:
                with self._lock:
                    self._connected = False
                    self._last_error = "lecture frame echouee"
                cap.release()
                cap = None
                time.sleep(RECONNECT_DELAY)
                continue

            with self._lock:
                self._frame = frame
                self._capture_ts = time.perf_counter()

        if cap is not None:
            cap.release()

    def get_latest(self):
        with self._lock:
            return (
                self._frame,
                self._capture_ts,
                self._connected,
                self._backend_used,
                self._active_source,
                self._last_error,
            )

    def stop(self):
        self._running = False


# ============================================================================
#  INFERENCE HELPERS
# ============================================================================

def infer_snapshot(result, state: CamState) -> DetectionSnapshot:
    boxes = result.boxes
    has_intrusion = False
    has_object = False
    person_boxes = []
    object_boxes = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())

        if cls == PERSON_CLASS:
            feet = foot_contact_points(x1, y1, x2, y2)
            in_zone = any_point_in_polygon(feet, state.zone_contour)
            if in_zone:
                has_intrusion = True
            person_boxes.append((x1, y1, x2, y2, conf, in_zone, feet))

        elif cls <= OBJECT_CLASS_MAX:
            has_object = True
            object_boxes.append((x1, y1, x2, y2, cls, conf))

    return DetectionSnapshot(
        timestamp=time.perf_counter(),
        person_boxes=person_boxes,
        object_boxes=object_boxes,
        has_intrusion=has_intrusion,
        has_object=has_object,
    )


# ============================================================================
#  BOUCLE ULTRALYTICS (historique)
# ============================================================================

def run_ultralytics(model_path: str, sources: list[str], camera_ids: list[str],
                    zones: dict[str, np.ndarray], conf: float, imgsz: int,
                    use_tracker: bool, tracker_name: str, track_conf: float):
    from ultralytics import YOLO

    n_cams = len(sources)
    print(f"\n  Cameras   : {n_cams}")
    for i, (cid, src) in enumerate(zip(camera_ids, sources)):
        safe = src.split("@", 1)[-1] if "@" in src else src
        print(f"    [{i+1}] {cid} -> {safe}")

    mpath = Path(model_path)
    fmt_label = {"engine": "TensorRT", "onnx": "ONNX", "pt": "PyTorch"}
    model_label = (mpath.parents[1].name + " "
                   + fmt_label.get(mpath.suffix.lstrip("."), mpath.suffix))

    print("\n  Chargement du modele...")
    model = YOLO(model_path)
    model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                  conf=conf, verbose=False)
    print(f"  Modele pret : {model_label}\n")

    states: dict[str, CamState] = {}
    for cid in camera_ids:
        if cid in zones:
            states[cid] = CamState(cid, zones[cid])
        else:
            states[cid] = CamState(cid, np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int32))
            print(f"  [ATTENTION] Pas de zone pour {cid}")

    windows = {}
    for cid in camera_ids:
        wname = f"Dashboard - {cid}"
        windows[cid] = wname
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wname, 960, 540)

    frame_count = 0
    fps_clock = time.perf_counter()
    fps_frames = 0
    fps_display = 0.0

    tracker_active = use_tracker

    try:
        src = sources[0] if n_cams == 1 else sources
        if tracker_active:
            print(f"  Tracking active : {tracker_name} (conf={track_conf})")
            try:
                stream_iter = model.track(
                    source=src,
                    stream=True,
                    conf=track_conf,
                    imgsz=imgsz,
                    persist=True,
                    tracker=tracker_name,
                    verbose=False,
                )
            except Exception as e:
                print(f"  [ATTENTION] Tracking indisponible ({e}). Fallback predict().")
                tracker_active = False
                stream_iter = model.predict(
                    source=src,
                    stream=True,
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )
        else:
            stream_iter = model.predict(
                source=src,
                stream=True,
                conf=conf,
                imgsz=imgsz,
                verbose=False,
            )

        for result in stream_iter:
            cam_id = camera_ids[frame_count % n_cams]
            state = states[cam_id]
            frame = result.orig_img
            if frame is None:
                frame_count += 1
                continue

            h_frame, w_frame = frame.shape[:2]
            state.ensure_scale(w_frame, h_frame)

            snapshot = infer_snapshot(result, state)
            state.update(snapshot.has_intrusion, snapshot.has_object)
            draw_snapshot(frame, snapshot)

            if cam_id in zones:
                draw_zone(frame, state.zone_poly, state.intrusion)
            draw_banners(frame, state)
            draw_fps(frame, fps_display, model_label)
            draw_status(frame, cam_id, state)
            cv2.imshow(windows[cam_id], frame)

            frame_count += 1
            fps_frames += 1
            now = time.perf_counter()
            dt = now - fps_clock
            if dt >= 1.0:
                fps_display = fps_frames / dt
                fps_frames = 0
                fps_clock = now

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\n  Arret (Ctrl+C).")
    finally:
        cv2.destroyAllWindows()
        print(f"  Total frames : {frame_count}")
        print("  Dashboard ferme.")


# ============================================================================
#  BOUCLE GSTREAMER + TIMESKIP INFERENCE
# ============================================================================

def run_gstreamer(model_path: str,
                  sources: list[str],
                  camera_ids: list[str],
                  zones: dict[str, np.ndarray],
                  conf: float,
                  imgsz: int,
                  infer_every: int,
                  stale_ms: int,
                  use_tracker: bool,
                  tracker_name: str,
                  track_conf: float,
                  gst_latency: int,
                  force_tcp: bool,
                  fallback_ffmpeg: bool,
                  show_latency: bool,
                  latency_report_sec: float,
                  sync_cameras: bool,
                  sync_tolerance_ms: int):
    from ultralytics import YOLO

    print("\n  Chargement du modele...")
    model = YOLO(model_path)
    model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), conf=conf, verbose=False)
    print("  Modele pret.\n")

    requested_imgsz = int(imgsz)

    states: dict[str, CamState] = {}
    readers: dict[str, CameraReader] = {}
    windows: dict[str, str] = {}

    for cam_id, source in zip(camera_ids, sources):
        if cam_id in zones:
            states[cam_id] = CamState(cam_id, zones[cam_id])
        else:
            states[cam_id] = CamState(cam_id, np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int32))
            print(f"  [ATTENTION] Pas de zone pour {cam_id}")

        reader = CameraReader(
            cam_id=cam_id,
            source=source,
            gst_latency=gst_latency,
            force_tcp=force_tcp,
            fallback_ffmpeg=fallback_ffmpeg,
        )
        readers[cam_id] = reader

        wname = f"Dashboard - {cam_id}"
        windows[cam_id] = wname
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wname, 960, 540)

    if not readers:
        cv2.destroyAllWindows()
        sys.exit("[ERREUR] Aucun flux ouvert. Verifiez RTSP/GStreamer.")

    print("\n  Demarrage des flux...")
    print("  Frame skip inference actif:")
    print(f"    - infer every      : {infer_every}")
    print(f"    - stale max (ms)   : {stale_ms}")
    if use_tracker:
        print(f"    - tracker          : {tracker_name} (conf={track_conf}, persist=True)")
    else:
        print("    - tracker          : off")
    print(f"    - sync cameras     : {sync_cameras}")
    print(f"    - sync tolerance   : {sync_tolerance_ms} ms")
    print("  Appuyer sur 'q' pour quitter.\n")

    frame_count = 0
    fps_clock = time.perf_counter()
    fps_frames = 0
    fps_display = 0.0
    last_latency_report = time.perf_counter()
    announced_cams: set[str] = set()
    tracker_disabled = False

    try:
        while True:
            latest_by_cam: dict[str, tuple[np.ndarray, float]] = {}
            for cam_id, reader in readers.items():
                frame, capture_ts, connected, backend, active_source, _ = reader.get_latest()
                if connected and backend and cam_id not in announced_cams:
                    safe = active_source.split("@", 1)[-1] if "@" in active_source else active_source
                    print(f"  [{cam_id}] backend={backend} source={safe}")
                    announced_cams.add(cam_id)
                if frame is None or capture_ts <= 0:
                    continue
                latest_by_cam[cam_id] = (frame, capture_ts)

            had_frame = len(latest_by_cam) > 0
            if sync_cameras and len(latest_by_cam) >= 2:
                ts_values = [t for _, t in latest_by_cam.values()]
                global_skew_ms = (max(ts_values) - min(ts_values)) * 1000.0
                if global_skew_ms > sync_tolerance_ms:
                    time.sleep(0.002)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
                    continue
            else:
                global_skew_ms = 0.0

            for cam_id in camera_ids:
                if cam_id not in latest_by_cam:
                    continue
                frame, t_capture = latest_by_cam[cam_id]

                state = states[cam_id]
                state.last_capture_ts = t_capture
                h_frame, w_frame = frame.shape[:2]
                state.ensure_scale(w_frame, h_frame)

                state.frame_index += 1
                do_infer = (state.frame_index % infer_every == 0) or (state.last_snapshot is None)

                if do_infer:
                    if use_tracker and not tracker_disabled:
                        try:
                            results = model.track(
                                source=frame,
                                conf=track_conf,
                                imgsz=imgsz,
                                persist=True,
                                tracker=tracker_name,
                                verbose=False,
                            )
                        except Exception as e:
                            max_imgsz = extract_max_imgsz_from_exception(e)
                            if max_imgsz is not None and max_imgsz != imgsz:
                                print(
                                    f"  [AUTO-IMGSZ] Engine statique detecte: {imgsz} -> {max_imgsz} "
                                    f"(demande initiale: {requested_imgsz})."
                                )
                                imgsz = max_imgsz
                            print(f"  [ATTENTION] Tracking indisponible ({e}). Fallback predict().")
                            tracker_disabled = True
                            results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
                    else:
                        try:
                            results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
                        except Exception as e:
                            max_imgsz = extract_max_imgsz_from_exception(e)
                            if max_imgsz is not None and max_imgsz != imgsz:
                                print(
                                    f"  [AUTO-IMGSZ] Engine statique detecte: {imgsz} -> {max_imgsz} "
                                    f"(demande initiale: {requested_imgsz})."
                                )
                                imgsz = max_imgsz
                                results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
                            else:
                                raise
                    result = results[0]
                    snapshot = infer_snapshot(result, state)
                    state.last_snapshot = snapshot
                    state.update(snapshot.has_intrusion, snapshot.has_object)
                else:
                    snapshot = state.last_snapshot

                is_fresh = False
                if snapshot is not None:
                    age_ms = (time.perf_counter() - snapshot.timestamp) * 1000.0
                    is_fresh = age_ms <= stale_ms

                if cam_id in zones:
                    draw_zone(frame, state.zone_poly, state.intrusion)

                if snapshot is not None and is_fresh:
                    draw_snapshot(frame, snapshot)
                elif snapshot is None:
                    state.update(False, False)

                draw_banners(frame, state)
                draw_fps(frame, fps_display, Path(model_path).name)
                draw_status(frame, cam_id, state)

                now_draw = time.perf_counter()
                state.last_latency_ms = max(0.0, (now_draw - state.last_capture_ts) * 1000.0)

                # Desynchronisation inter-cameras = ecart entre timestamps de capture
                state.last_skew_ms = global_skew_ms

                if show_latency:
                    draw_latency(frame, state)

                cv2.imshow(windows[cam_id], frame)

                frame_count += 1
                fps_frames += 1

            if not had_frame:
                time.sleep(0.002)

            now = time.perf_counter()
            dt = now - fps_clock
            if dt >= 1.0:
                fps_display = fps_frames / dt if dt > 0 else 0.0
                fps_frames = 0
                fps_clock = now

            if latency_report_sec > 0 and (now - last_latency_report) >= latency_report_sec:
                active_states = [s for s in states.values() if s.last_capture_ts > 0]
                if active_states:
                    avg_lat = sum(s.last_latency_ms for s in active_states) / len(active_states)
                    max_lat = max(s.last_latency_ms for s in active_states)
                    max_skew = max(s.last_skew_ms for s in active_states)
                    print(f"  [LATENCY] avg={avg_lat:.0f}ms max={max_lat:.0f}ms sync_skew={max_skew:.0f}ms")
                last_latency_report = now

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\n  Arret (Ctrl+C).")
    finally:
        for reader in readers.values():
            reader.stop()
        cv2.destroyAllWindows()
        print(f"  Total frames affichees : {frame_count}")
        print("  Dashboard ferme.")


# ============================================================================
#  CLI
# ============================================================================

def resolve_camera_ids(args, zones: dict[str, np.ndarray]) -> list[str]:
    all_ids = sorted(RTSP_HOSTS.keys())

    if args.cameras:
        camera_ids = args.cameras
    else:
        camera_ids = [cid for cid in all_ids if cid in zones]
        if not camera_ids:
            camera_ids = all_ids

    if args.num_cameras is not None:
        n = max(1, min(8, args.num_cameras))
        camera_ids = camera_ids[:n]

    return camera_ids


def main():
    import argparse

    p = argparse.ArgumentParser(description="Dashboard Live RTSP")
    p.add_argument("--model", "-m", default=MODEL_PATH)
    p.add_argument("--zones", "-z", default=ZONES_JSON)
    p.add_argument("--cameras", nargs="+", default=None,
                   help="Liste explicite de cameras (ex: cam_01 cam_05 cam_07)")
    p.add_argument("--num-cameras", type=int, default=None,
                   help="Selection rapide des N premieres cameras (1..8)")
    p.add_argument("--sources", nargs="+", default=None,
                   help="Sources custom (meme taille que --cameras finales)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--use-tracker", action="store_true", default=DEFAULT_USE_TRACKER,
                   help="Utilise model.track() avec persist=True pour lisser les detections")
    p.add_argument("--tracker", type=str, default=DEFAULT_TRACKER,
                   help="Tracker Ultralytics (ex: botsort.yaml, bytetrack.yaml)")
    p.add_argument("--track-conf", type=float, default=DEFAULT_TRACK_CONF,
                   help="Seuil de confiance utilise en mode tracking")

    p.add_argument("--backend", choices=["ultralytics", "gstreamer"], default="gstreamer",
                   help="ultralytics: historique, gstreamer: capture low-latency + timeskip")

    p.add_argument("--infer-every", type=int, default=DEFAULT_INFER_EVERY,
                   help="Inference 1 frame sur N (N>=1)")
    p.add_argument("--stale-ms", type=int, default=DEFAULT_STALE_MS,
                   help="Age max des detections reutilisees en mode skip")

    p.add_argument("--rtsp-subtype", type=int, default=1,
                   help="Subtype RTSP (0=main stream, 1=substream)")
    p.add_argument("--gst-latency", type=int, default=0,
                   help="Latence GStreamer (ms) sur rtspsrc")
    p.add_argument("--force-tcp", action="store_true",
                   help="Force TCP pour GStreamer (rtspsrc protocols=4)")
    p.add_argument("--no-ffmpeg-fallback", action="store_true",
                   help="Desactive fallback FFMPEG si ouverture GStreamer echoue")
    p.add_argument("--show-latency", action="store_true",
                   help="Affiche LAT/SKEW dans le HUD (mode gstreamer)")
    p.add_argument("--latency-report-sec", type=float, default=DEFAULT_LATENCY_REPORT_SEC,
                   help="Intervalle du log latence console en secondes (0=off)")
    p.add_argument("--sync-cameras", action="store_true",
                   help="Active une barriere de synchronisation soft entre cameras (gstreamer)")
    p.add_argument("--sync-tolerance-ms", type=int, default=DEFAULT_SYNC_TOLERANCE_MS,
                   help="Tolerance max de desync pour la barriere soft")

    args = p.parse_args()

    infer_every = max(1, int(args.infer_every))
    stale_ms = max(50, int(args.stale_ms))

    zones = load_zones(args.zones)
    print(f"  Zones chargees : {list(zones.keys())}")

    camera_ids = resolve_camera_ids(args, zones)
    if not camera_ids:
        sys.exit("[ERREUR] Aucune camera selectionnee.")

    for cid in camera_ids:
        if cid not in RTSP_HOSTS:
            sys.exit(f"[ERREUR] Camera inconnue: {cid}")

    if args.sources:
        if len(args.sources) != len(camera_ids):
            sys.exit("[ERREUR] --sources doit avoir la meme taille que les cameras selectionnees.")
        sources = args.sources
    else:
        sources = [build_rtsp_url(RTSP_HOSTS[cid], args.rtsp_subtype) for cid in camera_ids]

    if not Path(args.model).exists():
        sys.exit(f"[ERREUR] Modele introuvable : {args.model}")

    print("\n  Configuration:")
    print(f"    - backend       : {args.backend}")
    print(f"    - cameras       : {camera_ids}")
    print(f"    - imgsz         : {args.imgsz}")
    print(f"    - infer_every   : {infer_every}")
    print(f"    - stale_ms      : {stale_ms}")
    print(f"    - use_tracker   : {args.use_tracker}")
    if args.use_tracker:
        print(f"    - tracker_cfg   : {args.tracker} (conf={args.track_conf})")
    print(f"    - rtsp_subtype  : {args.rtsp_subtype}")
    print(f"    - show_latency  : {args.show_latency}")
    print(f"    - sync_cameras  : {args.sync_cameras}")

    if args.backend == "gstreamer":
        run_gstreamer(
            model_path=str(Path(args.model)),
            sources=sources,
            camera_ids=camera_ids,
            zones=zones,
            conf=args.conf,
            imgsz=args.imgsz,
            infer_every=infer_every,
            stale_ms=stale_ms,
            use_tracker=bool(args.use_tracker),
            tracker_name=str(args.tracker),
            track_conf=float(args.track_conf),
            gst_latency=max(0, int(args.gst_latency)),
            force_tcp=bool(args.force_tcp),
            fallback_ffmpeg=not args.no_ffmpeg_fallback,
            show_latency=bool(args.show_latency),
            latency_report_sec=max(0.0, float(args.latency_report_sec)),
            sync_cameras=bool(args.sync_cameras),
            sync_tolerance_ms=max(20, int(args.sync_tolerance_ms)),
        )
    else:
        if infer_every > 1:
            print("  [INFO] --infer-every est ignore en backend ultralytics.")
        run_ultralytics(
            model_path=str(Path(args.model)),
            sources=sources,
            camera_ids=camera_ids,
            zones=zones,
            conf=args.conf,
            imgsz=args.imgsz,
            use_tracker=bool(args.use_tracker),
            tracker_name=str(args.tracker),
            track_conf=float(args.track_conf),
        )


if __name__ == "__main__":
    main()

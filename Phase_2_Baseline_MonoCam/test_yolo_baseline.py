"""
Phase 2 - Test YOLO Baseline (out-of-the-box)
8 caméras simultanées en grille 2x4 avec inférence YOLO sur chaque flux.
Utilise GStreamer pour des lectures RTSP zéro-latence en parallèle.
Supporte aussi les fichiers vidéo locaux.

Appuyer sur Q pour quitter.
"""

# ── Chargement GStreamer Windows (AVANT import cv2) ──────────────────────────
import os

gst_path = r'C:\gstreamer\1.0\msvc_x86_64\bin'
if os.path.exists(gst_path):
    os.environ['PATH'] = gst_path + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(gst_path)
    except AttributeError:
        pass

import sys
import re
import yaml
import cv2
import numpy as np
import threading
import time
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# --- Configuration ---
CONFIG_PATH = Path(__file__).resolve().parent.parent / "Phase_3_Fusion_MultiCam" / "config.yaml"

# Classes COCO ciblées
TARGET_CLASSES = [0, 39, 41, 67]  # person, bottle, cup, cell phone
CONFIDENCE = 0.35

# Grille 2 lignes x 4 colonnes
CELL_W = 640
CELL_H = 360
GRID_COLS = 4
GRID_ROWS = 2

RECONNECT_DELAY = 2  # secondes avant tentative de reconnexion

# ── Paramètres de performance ────────────────────────────────────────────────
# Taille d'entrée YOLO : 640 (précis) | 480 (équilibré) | 320 (rapide)
# Pour de la sécurité industrielle, 480 suffit largement pour détecter un humain.
INFERENCE_SIZE = 480

# Frame skipping : traiter 1 frame sur N (le reste réutilise les dernières détections)
# 1 = chaque frame | 2 = 1/2 (~12 FPS) | 3 = 1/3 (~8 FPS) | 5 = 1/5 (~5 FPS)
# Pour la sécurité, ~8 FPS est largement suffisant.
PROCESS_EVERY_N = 3

# FP16 (half precision) : double le débit GPU, perte de précision négligeable
USE_HALF = True


def get_device() -> str:
    """Détecte le meilleur device disponible (CUDA > CPU)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"  [GPU] {gpu_name} ({vram_mb:.0f} MB VRAM)")
        print(f"  [GPU] CUDA {torch.version.cuda}")
        return "cuda:0"
    print("  [CPU] Aucun GPU CUDA détecté, fallback CPU.")
    return "cpu"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_nvdec_available() -> bool:
    """Vérifie si le décodage GPU (nvdec) est disponible via GStreamer."""
    try:
        test_pipe = cv2.VideoCapture(
            'videotestsrc num-buffers=1 ! nvh264enc ! nvh264dec ! '
            'videoconvert ! appsink', cv2.CAP_GSTREAMER)
        ok = test_pipe.isOpened()
        test_pipe.release()
        return ok
    except Exception:
        return False


def build_gst_pipeline(rtsp_url: str, use_nvdec: bool = False) -> str:
    """Pipeline GStreamer zéro-latence pour flux RTSP.
    Utilise nvdec (décodage GPU NVIDIA) si disponible."""
    if use_nvdec:
        # Décodage matériel GPU via NVIDIA nvdec
        return (
            f'rtspsrc location="{rtsp_url}" latency=0 protocols=tcp ! '
            f'rtph264depay ! h264parse ! nvh264dec ! '
            f'videoconvert ! appsink max-buffers=1 drop=true sync=false'
        )
    # Fallback: decodebin (auto-détection codec, décodage CPU)
    return (
        f'rtspsrc location="{rtsp_url}" latency=0 protocols=tcp ! '
        f'decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false'
    )


def resolve_video_path(cam_cfg, config_dir):
    """Résout le chemin vidéo local d'une caméra."""
    video_path = cam_cfg.get("video_path", "")
    if not video_path:
        return None

    resolved = (config_dir / video_path).resolve()
    if resolved.exists():
        return str(resolved)

    # Fallback: chercher dans recordings avec pattern
    recordings_dir = (config_dir.parent / "recordings" / "recordings").resolve()
    cam_name = cam_cfg.get("name", "")
    m = re.search(r"Camera\s*(\d+)\s*\(([^)]+)\)", cam_name)
    if recordings_dir.exists() and m:
        cam_num = m.group(1)
        room_code = m.group(2)
        for ext in ("mp4", "mkv"):
            matches = sorted(recordings_dir.glob(f"Camera_{cam_num}_{room_code}*.{ext}"))
            if matches:
                return str(matches[-1])

    return None


class CameraReader:
    """Lecteur vidéo asynchrone avec thread dédié.
    Supporte RTSP via GStreamer et fichiers vidéo locaux."""

    def __init__(self, name: str, source: str, use_gstreamer: bool,
                 use_nvdec: bool = False):
        self.name = name
        self.source = source
        self.use_gstreamer = use_gstreamer
        self.use_nvdec = use_nvdec

        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = None
        self._fps = 0.0
        self._connected = False
        self._running = True
        self._last_error = "initialisation"

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _open_capture(self) -> cv2.VideoCapture:
        if self.use_gstreamer:
            pipeline = build_gst_pipeline(self.source, self.use_nvdec)
            return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            return cv2.VideoCapture(self.source)

    def _reader_loop(self):
        cap = None
        frame_count = 0
        fps_timer = time.monotonic()

        while self._running:
            if cap is None or not cap.isOpened():
                with self._lock:
                    self._connected = False
                    self._last_error = "connexion en cours"
                if cap is not None:
                    cap.release()
                cap = self._open_capture()
                if not cap.isOpened():
                    with self._lock:
                        self._last_error = "ouverture impossible"
                    print(f"  [{self.name}] Connexion échouée, retry dans {RECONNECT_DELAY}s...")
                    time.sleep(RECONNECT_DELAY)
                    continue
                print(f"  [{self.name}] Connecté.")
                with self._lock:
                    self._connected = True
                    self._last_error = ""
                frame_count = 0
                fps_timer = time.monotonic()

            ret, frame = cap.read()

            if not ret:
                if not self.use_gstreamer:
                    # Fichier vidéo terminé -> boucler
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                with self._lock:
                    self._connected = False
                    self._last_error = "lecture frame échouée"
                print(f"  [{self.name}] Flux perdu, reconnexion...")
                cap.release()
                cap = None
                time.sleep(RECONNECT_DELAY)
                continue

            # Certains backends peuvent retourner ret=True avec une image vide.
            if frame is None or frame.size == 0:
                with self._lock:
                    self._connected = False
                    self._last_error = "frame vide"
                time.sleep(0.01)
                continue

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            frame_count += 1

            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.monotonic()
            else:
                fps = self._fps

            with self._lock:
                self._frame = frame
                self._timestamp = ts
                self._fps = fps

        if cap is not None:
            cap.release()

    def get_frame(self):
        """Retourne (frame, timestamp, fps, connected, last_error) thread-safe."""
        with self._lock:
            return (
                self._frame.copy() if self._frame is not None else None,
                self._timestamp,
                self._fps,
                self._connected,
                self._last_error,
            )

    def stop(self):
        self._running = False


def build_cell(name: str, frame, annotated, fps: float,
               connected: bool, last_error: str, ts: str,
               inference_ms: float) -> np.ndarray:
    """Construit une cellule de la grille avec l'image annotée YOLO."""

    if frame is None:
        cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        label = "DECONNECTE" if not connected else "EN ATTENTE..."
        color = (0, 0, 200) if not connected else (0, 140, 255)
        cv2.putText(cell, name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        cv2.putText(cell, label, (10, CELL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if last_error:
            msg = f"Cause: {last_error}"[:58]
            cv2.putText(cell, msg, (10, CELL_H // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
        return cell

    # Avec le frame skipping, l'annotation peut être absente; fallback frame brute.
    render = annotated if annotated is not None else frame
    if render is None or getattr(render, "size", 0) == 0:
        cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    else:
        cell = cv2.resize(render, (CELL_W, CELL_H))

    # Bande semi-transparente en haut
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (CELL_W, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, cell, 0.4, 0, cell)

    # Nom caméra (cyan)
    cv2.putText(cell, name, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    # FPS + inference time (vert)
    cv2.putText(cell, f"{fps:.1f} FPS | {inference_ms:.0f}ms", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Timestamp (blanc)
    if ts:
        cv2.putText(cell, ts, (CELL_W - 200, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return cell


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  YOLO BASELINE - 8 Caméras Simultanées (Grille 2x4)     ║")
    print("║  GStreamer (RTSP) / OpenCV (fichiers locaux)             ║")
    print("║  Q = Quitter                                             ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Détecter le device GPU/CPU
    device = get_device()
    use_gpu = device.startswith("cuda")

    # Vérifier GStreamer
    backends = [cv2.videoio_registry.getBackendName(b)
                for b in cv2.videoio_registry.getBackends()]
    gst_ok = "GSTREAMER" in backends
    print(f"\n  Backends OpenCV : {', '.join(backends)}")
    if gst_ok:
        print("  [OK] GStreamer détecté.")
    else:
        print("  [INFO] GStreamer non détecté. Les flux RTSP utiliseront FFmpeg/OpenCV.")

    # Vérifier décodage GPU (nvdec) via GStreamer
    nvdec_ok = False
    if gst_ok and use_gpu:
        nvdec_ok = check_nvdec_available()
        if nvdec_ok:
            print("  [OK] nvdec (décodage vidéo GPU) disponible.")
        else:
            print("  [INFO] nvdec non disponible, décodage vidéo sur CPU.")

    # Charger config
    config = load_config()
    config_dir = CONFIG_PATH.parent
    cameras_cfg = config["cameras"]

    # Charger le modèle YOLO sur GPU
    use_half = USE_HALF and use_gpu  # FP16 uniquement sur GPU
    precision = "FP16" if use_half else "FP32"
    print(f"\n  Chargement du modèle YOLOv8s sur {device} ({precision}, imgsz={INFERENCE_SIZE})...")
    model = YOLO("yolov8s.pt")
    # Warmup : premier appel lent (compilation CUDA kernels), les suivants seront rapides
    model.predict(np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8),
                  device=device, conf=CONFIDENCE, half=use_half, verbose=False)
    print(f"  [OK] Modèle prêt. Skip={PROCESS_EVERY_N} (~{25//PROCESS_EVERY_N} FPS effectif/cam)\n")

    # Lancer les readers pour les 8 caméras
    readers = []
    reader_names = []
    for cam_id, cam_cfg in cameras_cfg.items():
        name = cam_cfg.get("name", cam_id)
        rtsp_url = cam_cfg.get("rtsp_url", "")
        video_path = resolve_video_path(cam_cfg, config_dir)

        # Préférer le fichier local s'il existe, sinon RTSP
        if video_path:
            print(f"  {name} -> fichier local : {Path(video_path).name}")
            readers.append(CameraReader(name, video_path, use_gstreamer=False))
        elif rtsp_url:
            use_gst = gst_ok
            backend = "GStreamer+nvdec" if (use_gst and nvdec_ok) else (
                "GStreamer" if use_gst else "OpenCV/FFmpeg")
            print(f"  {name} -> RTSP ({backend})")
            readers.append(CameraReader(name, rtsp_url,
                                        use_gstreamer=use_gst,
                                        use_nvdec=nvdec_ok))
        else:
            print(f"  {name} -> AUCUNE SOURCE, ignorée")
            continue

        reader_names.append(name)

    print(f"\n  [{len(readers)} caméras lancées] Affichage grille {GRID_ROWS}x{GRID_COLS}")
    print("  Appuyer sur Q pour quitter.\n")

    window_name = "YOLO Baseline - 8 Cameras (Grille 2x4)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CELL_W * GRID_COLS, CELL_H * GRID_ROWS)

    # Tracking de l'inférence par caméra
    inference_times = [0.0] * len(readers)
    last_annotated = [None] * len(readers)
    batch_inference_ms = 0.0
    frame_counter = 0

    start_time = time.monotonic()

    while True:
        # Récupérer toutes les frames (instantané grâce aux threads)
        frames_data = [r.get_frame() for r in readers]
        frame_counter += 1

        # Frame skipping : on ne lance YOLO que toutes les N frames
        # Entre-temps on réutilise les dernières annotations (last_annotated)
        run_inference = (frame_counter % PROCESS_EVERY_N == 0)

        if run_inference:
            # Batch inference YOLO : toutes les frames en un seul appel GPU
            batch_frames = []
            batch_indices = []
            for i, (frame, ts, fps, connected, err) in enumerate(frames_data):
                if frame is not None and frame.size > 0:
                    batch_frames.append(frame)
                    batch_indices.append(i)
                else:
                    last_annotated[i] = None

            if batch_frames:
                t0 = time.monotonic()
                results = model.predict(
                    batch_frames,
                    classes=TARGET_CLASSES,
                    conf=CONFIDENCE,
                    device=device,
                    imgsz=INFERENCE_SIZE,
                    half=use_half,
                    verbose=False,
                )
                batch_inference_ms = (time.monotonic() - t0) * 1000
                per_frame_ms = batch_inference_ms / len(batch_frames)

                for j, idx in enumerate(batch_indices):
                    inference_times[idx] = per_frame_ms
                    last_annotated[idx] = results[j].plot()

        # Construire la grille
        rows = []
        for r in range(GRID_ROWS):
            row_cells = []
            for c in range(GRID_COLS):
                idx = r * GRID_COLS + c
                if idx < len(readers):
                    frame, ts, fps, connected, err = frames_data[idx]
                    cell = build_cell(
                        readers[idx].name, frame, last_annotated[idx],
                        fps, connected, err, ts, inference_times[idx]
                    )
                    row_cells.append(cell)
                else:
                    row_cells.append(np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
            rows.append(np.hstack(row_cells))
        grid = np.vstack(rows)

        # Barre d'info en bas
        elapsed_s = time.monotonic() - start_time
        elapsed_min = int(elapsed_s // 60)
        elapsed_sec = int(elapsed_s % 60)
        h_grid, w_grid = grid.shape[:2]

        cv2.putText(grid, f"Duree: {elapsed_min:02d}:{elapsed_sec:02d}",
                    (10, h_grid - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Batch inférence timing + config
        n_cams = len(readers)
        cv2.putText(grid,
                    f"Batch {n_cams}x: {batch_inference_ms:.0f}ms | "
                    f"{batch_inference_ms/max(n_cams,1):.0f}ms/cam | "
                    f"{device.upper()} {precision} | "
                    f"imgsz={INFERENCE_SIZE} skip={PROCESS_EVERY_N}",
                    (250, h_grid - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Connexions
        connected_count = sum(1 for f, _, _, c, _ in frames_data if c)
        cv2.putText(grid, f"Connectees: {connected_count}/{len(readers)}",
                    (w_grid - 280, h_grid - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        cv2.imshow(window_name, grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # Cleanup
    print("\n  Arrêt des threads...")
    for reader in readers:
        reader.stop()
    cv2.destroyAllWindows()
    print("  Terminé.")


if __name__ == "__main__":
    main()

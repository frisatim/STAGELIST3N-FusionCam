"""
Stream Live - Phase 1 Infrastructure

Deux modes :
  python streamlive.py              -> Replay : re-stream les fichiers existants via MediaMTX
  python streamlive.py --record     -> Record : enregistre les 8 caméras RTSP en nouveaux fichiers
  python streamlive.py --record 300 -> Record pendant 300 secondes (5 min) puis arrêt auto
  python streamlive.py --record --gpu          -> Encodage GPU (NVENC) au lieu de CPU
  python streamlive.py --record 300 --gpu      -> Record 5min avec GPU

Les nouvelles vidéos sont sauvegardées dans recordings/recordings/ avec le même format
de nommage (Camera_X_IP_YYYYMMDD_HHMMSS.mp4), prêtes à être utilisées par le pipeline.

Ctrl+C ou Q pour arrêter.
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

import subprocess
import sys
import cv2
import numpy as np
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

# ── Configuration commune ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings" / "recordings"

CAMERAS = [
{'name': 'Camera_1_2.2',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
{'name': 'Camera_2_2.3',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
{'name': 'Camera_3_2.4',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
{'name': 'Camera_4_2.5',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
{'name': 'Camera_5_2.6',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
{'name': 'Camera_6_2.7',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
{'name': 'Camera_7_2.11', 'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
{'name': 'Camera_8_2.13', 'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
]

CAMERA_IDS = [f"cam_{i:02d}" for i in range(1, 9)]
CAMERAS_BY_ID = dict(zip(CAMERA_IDS, CAMERAS))

# Fichiers existants pour le mode replay (mapping ancien nommage)
REPLAY_FILES = {
    "cam_01": "Camera 1 (2.2).mkv",
    "cam_02": "Camera 2 (2.3).mkv",
    "cam_03": "Camera_3_2.4.mkv",
    "cam_04": "Camera 4 (2.5).mkv",
    "cam_05": "Camera 5 (2.6).mkv",
    "cam_06": "Camera 6 (2.7).mkv",
    "cam_07": "Camera 7 (2.11).mkv",
    "cam_08": "Camera 8 (2.13).mkv",
}

MEDIAMTX_URL = "rtsp://localhost:8554"

# Paramètres d'enregistrement.
# Les caméras Dahua sortent le flux secondaire en 704x576. On l'enregistre en
# 768x576 pour restaurer le ratio 4:3 utilisé par la calibration/inférence.
WRITE_W = 704
WRITE_H = 576
TARGET_FPS = 25
FRAME_INTERVAL = 1.0 / TARGET_FPS
RECONNECT_DELAY = 3
USE_GPU = False  # Activé par --gpu

# Flag global d'arrêt
stop_event = threading.Event()


def signal_handler(sig, frame):
    print("\n\n  [CTRL+C] Arrêt demandé...")
    stop_event.set()


# ══════════════════════════════════════════════════════════════════════════════
#  MODE REPLAY (comportement original)
# ══════════════════════════════════════════════════════════════════════════════

def mode_replay():
    """Re-stream les fichiers existants via MediaMTX (fake live)."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  STREAM LIVE - Mode REPLAY (Fake Live)                    ║")
    print("║  Re-stream les enregistrements existants via MediaMTX     ║")
    print("║  Ctrl+C = Arrêter                                         ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # Chercher aussi les fichiers .mp4 plus récents
    footages = {}
    for cam_id, cam_cfg in zip(
        [f"cam_{i:02d}" for i in range(1, 9)], CAMERAS
    ):
        # D'abord chercher le fichier le plus récent (.mp4 ou .mkv)
        patterns = [f"{cam_cfg['name']}*.mp4", f"{cam_cfg['name']}*.mkv"]
        found = None
        for pattern in patterns:
            matches = sorted(RECORDINGS_DIR.glob(pattern))
            if matches:
                found = str(matches[-1])  # le plus récent

        # Fallback sur les anciens noms
        if not found and cam_id in REPLAY_FILES:
            old_path = RECORDINGS_DIR / REPLAY_FILES[cam_id]
            if old_path.exists():
                found = str(old_path)

        if found:
            footages[cam_id] = found
        else:
            print(f"  [!] Aucun fichier trouvé pour {cam_id} ({cam_cfg['name']})")

    processes = []
    for name, filepath in footages.items():
        cmd = [
            'ffmpeg',
            '-re',
            '-stream_loop', '-1',
            '-i', filepath,
            '-c', 'copy',
            '-f', 'rtsp',
            f"{MEDIAMTX_URL}/{name}"
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        print(f"  Stream actif : {MEDIAMTX_URL}/{name} <- {Path(filepath).name}")

    print(f"\n  {len(processes)} flux en ligne.")
    print(f"  Connexion : rtsp://localhost:8554/cam_01 ... cam_08")
    print("  Ctrl+C pour arrêter.\n")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        print("\n  Flux arrêtés.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE RECORD (nouvel enregistrement depuis les caméras réelles)
# ══════════════════════════════════════════════════════════════════════════════

def build_gst_pipelines(rtsp_url: str) -> list:
    """Retourne une liste de pipelines GStreamer à essayer dans l'ordre.
    Certaines caméras (ex: cam 1/7) peuvent streamer en H.265 ou avoir
    des paramètres incompatibles avec decodebin. On tente plusieurs variantes."""
    return [
        # 1) Pipeline standard (decodebin auto-détecte H.264/H.265)
        (
            "GStreamer decodebin (auto)",
            f'rtspsrc location="{rtsp_url}" latency=0 protocols=tcp ! '
            f'decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false'
        ),
        # 2) Pipeline avec latency plus élevée + buffer (caméras lentes à répondre)
        (
            "GStreamer decodebin (latency=300)",
            f'rtspsrc location="{rtsp_url}" latency=300 protocols=tcp '
            f'tcp-timeout=5000000 ! '
            f'decodebin ! videoconvert ! appsink max-buffers=2 drop=true sync=false'
        ),
        # 3) Force H.264 explicitement (si decodebin échoue sur du H.264)
        (
            "GStreamer H.264 explicit",
            f'rtspsrc location="{rtsp_url}" latency=200 protocols=tcp ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! '
            f'videoconvert ! appsink max-buffers=1 drop=true sync=false'
        ),
        # 4) Force H.265 explicitement (Dahua main stream souvent en H.265)
        (
            "GStreamer H.265 explicit",
            f'rtspsrc location="{rtsp_url}" latency=200 protocols=tcp ! '
            f'rtph265depay ! h265parse ! avdec_h265 ! '
            f'videoconvert ! appsink max-buffers=1 drop=true sync=false'
        ),
        # 5) UDP au lieu de TCP (certaines caméras ont un bug TCP)
        (
            "GStreamer decodebin (UDP)",
            f'rtspsrc location="{rtsp_url}" latency=200 ! '
            f'decodebin ! videoconvert ! appsink max-buffers=2 drop=true sync=false'
        ),
    ]


def check_nvenc_available() -> bool:
    """Vérifie si FFmpeg supporte h264_nvenc (GPU NVIDIA)."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        return 'h264_nvenc' in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def parse_camera_ids(raw: str | None) -> list[str] | None:
    """Parse --cameras cam_02,cam_03 into normalized camera IDs."""
    if not raw:
        return None
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    normalized: list[str] = []
    for cam_id in ids:
        if cam_id.isdigit():
            cam_id = f"cam_{int(cam_id):02d}"
        if cam_id not in CAMERAS_BY_ID:
            valid = ", ".join(CAMERA_IDS)
            raise ValueError(f"Caméra inconnue: {cam_id}. Valides: {valid}")
        if cam_id not in normalized:
            normalized.append(cam_id)
    return normalized


def select_cameras(camera_ids: list[str] | None) -> list[dict]:
    """Return camera configs filtered by camera ID, preserving requested order."""
    selected_ids = camera_ids or CAMERA_IDS
    selected = []
    for cam_id in selected_ids:
        cam = dict(CAMERAS_BY_ID[cam_id])
        cam["id"] = cam_id
        selected.append(cam)
    return selected


class GPUWriter:
    """Écrit des frames via FFmpeg + NVENC (h264_nvenc) au lieu de cv2.VideoWriter.
    Décharge l'encodage sur le GPU NVIDIA."""

    def __init__(self, output_path: str, fps: int, width: int, height: int):
        self.output_path = output_path
        self.width = width
        self.height = height
        self._proc = subprocess.Popen(
            [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',                      # stdin
                '-c:v', 'h264_nvenc',           # GPU NVIDIA
                '-preset', 'p4',                # bon compromis vitesse/qualité
                '-rc', 'vbr',                   # variable bitrate
                '-cq', '23',                    # qualité (plus bas = meilleur)
                '-g', str(fps),                 # keyframe chaque seconde (seek rapide)
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def write(self, frame: np.ndarray):
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait(timeout=10)


class CameraRecorder:
    """Enregistre un flux RTSP avec frame pacing strict (vitesse 1:1).
    2 threads : lecteur (cap.read) + écrivain (tick à TARGET_FPS).
    Essaie plusieurs pipelines GStreamer en fallback, puis FFmpeg en dernier recours."""

    def __init__(self, name: str, url: str, output_path: Path):
        self.name = name
        self.url = url
        self.output_path = output_path
        self._pipelines = build_gst_pipelines(url)

        self._lock = threading.Lock()
        self._latest_raw = None
        self._new_frame_flag = False
        self._frames_written = 0
        self._frames_duplicated = 0
        self._connected = False
        self._last_error = ""
        self._reconnections = 0
        self._backend_used = ""
        self._warned_resolution = False

        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True)
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True)
        self._reader_thread.start()
        self._writer_thread.start()

    def _try_open_capture(self) -> cv2.VideoCapture | None:
        """Essaie chaque pipeline GStreamer, puis fallback FFmpeg."""
        # Essai des pipelines GStreamer
        for label, pipeline in self._pipelines:
            print(f"  [{self.name}] Essai: {label}...")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self._backend_used = label
                    return cap
            cap.release()
            print(f"  [{self.name}]   -> Échec ({label})")

        # Dernier recours : FFmpeg backend d'OpenCV (pas GStreamer)
        print(f"  [{self.name}] Essai: FFmpeg backend (dernier recours)...")
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                self._backend_used = "FFmpeg backend (fallback)"
                return cap
        cap.release()
        print(f"  [{self.name}]   -> Échec (FFmpeg)")

        return None

    def _reader_loop(self):
        cap = None
        while not stop_event.is_set():
            if cap is None or not cap.isOpened():
                with self._lock:
                    self._connected = False
                    self._last_error = "connexion en cours"
                if cap is not None:
                    cap.release()

                cap = self._try_open_capture()
                if cap is None:
                    with self._lock:
                        self._last_error = "tous les pipelines ont échoué"
                        self._reconnections += 1
                    print(f"  [{self.name}] Tous les pipelines échoués, "
                          f"retry dans {RECONNECT_DELAY}s...")
                    for _ in range(RECONNECT_DELAY * 10):
                        if stop_event.is_set():
                            break
                        time.sleep(0.1)
                    continue

                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  [{self.name}] Connecté via [{self._backend_used}]. "
                      f"Flux ouvert avec résolution : {actual_w} x {actual_h}")
                with self._lock:
                    self._connected = True
                    self._last_error = ""

            ret, frame = cap.read()
            if not ret:
                with self._lock:
                    self._connected = False
                    self._last_error = "lecture frame échouée"
                    self._reconnections += 1
                print(f"  [{self.name}] Flux perdu, reconnexion...")
                cap.release()
                cap = None
                for _ in range(RECONNECT_DELAY * 10):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            if w != WRITE_W or h != WRITE_H:
                if (w, h) != (704, 576) and not self._warned_resolution:
                    print(f"  [{self.name}] [WARN] Résolution source inattendue: "
                          f"{w}x{h}; sortie forcée en {WRITE_W}x{WRITE_H}")
                    self._warned_resolution = True
                frame = cv2.resize(frame, (WRITE_W, WRITE_H))
            with self._lock:
                self._latest_raw = frame
                self._new_frame_flag = True

        if cap is not None:
            cap.release()

    def _writer_loop(self):
        if USE_GPU:
            writer = GPUWriter(
                str(self.output_path), TARGET_FPS, WRITE_W, WRITE_H)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(self.output_path), fourcc, TARGET_FPS, (WRITE_W, WRITE_H))

        last_written_frame = None
        next_tick = time.monotonic()

        while not stop_event.is_set():
            now = time.monotonic()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += FRAME_INTERVAL

            with self._lock:
                raw = self._latest_raw
                is_new = self._new_frame_flag
                self._new_frame_flag = False

            if raw is not None and is_new:
                last_written_frame = raw.copy()
                writer.write(raw)
                with self._lock:
                    self._frames_written += 1
            elif last_written_frame is not None:
                writer.write(last_written_frame)
                with self._lock:
                    self._frames_written += 1
                    self._frames_duplicated += 1

            if time.monotonic() > next_tick + FRAME_INTERVAL:
                next_tick = time.monotonic()

        writer.release()

    def get_status(self) -> dict:
        with self._lock:
            return {
                'name': self.name,
                'connected': self._connected,
                'frames': self._frames_written,
                'duplicated': self._frames_duplicated,
                'reconnections': self._reconnections,
                'error': self._last_error,
                'backend': self._backend_used,
            }


def mode_record(max_duration: float = 0, camera_ids: list[str] | None = None):
    """Enregistre les 8 caméras RTSP en fichiers .mp4."""
    global USE_GPU
    signal.signal(signal.SIGINT, signal_handler)

    selected_cameras = select_cameras(camera_ids)

    duration_str = f"{int(max_duration)}s" if max_duration > 0 else "illimité"
    enc_str = "GPU (NVENC)" if USE_GPU else "CPU (mp4v)"
    cam_count = len(selected_cameras)

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  STREAM LIVE - Mode RECORD (Enregistrement réel)        ║")
    print(f"║  {cam_count} cam | {WRITE_W}x{WRITE_H} | {TARGET_FPS}FPS | {enc_str} | Durée: {duration_str:<7s}║")
    print("║  Ctrl+C ou Q = Arrêt propre                              ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"\n  Caméras sélectionnées : {', '.join(cam['id'] for cam in selected_cameras)}")

    # Vérifier GStreamer
    backends = [cv2.videoio_registry.getBackendName(b)
                for b in cv2.videoio_registry.getBackends()]
    gst_ok = "GSTREAMER" in backends
    print(f"\n  Backends OpenCV : {', '.join(backends)}")
    if gst_ok:
        print("  [OK] GStreamer détecté.")
    else:
        print("  [ATTENTION] GStreamer NON détecté ! (FFmpeg fallback disponible)")

    # Vérifier GPU si demandé
    if USE_GPU:
        if check_nvenc_available():
            print("  [OK] NVENC (GPU) détecté : encodage matériel activé.")
        else:
            print("  [ERREUR] NVENC non disponible ! Vérifier drivers NVIDIA + FFmpeg.")
            print("           Fallback sur CPU (mp4v)...")
            USE_GPU = False

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Timestamp pour nommer les fichiers (même format que les existants)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Lister les enregistrements existants
    existing = sorted(RECORDINGS_DIR.glob("*.mp4")) + sorted(RECORDINGS_DIR.glob("*.mkv"))
    if existing:
        print(f"\n  Enregistrements existants ({len(existing)} fichiers) :")
        for f in existing:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name} ({size_mb:.0f} MB)")
    print()

    # Lancer les recorders
    recorders = []
    print("  Nouveaux fichiers :")
    for cam in selected_cameras:
        filename = f"{cam['name']}_{session_ts}.mp4"
        output_path = RECORDINGS_DIR / filename
        print(f"    {cam['name']} -> {filename}")
        recorders.append(CameraRecorder(cam['name'], cam['url'], output_path))

    print(f"\n  [{len(recorders)} caméras lancées] "
          f"Enregistrement en cours...")
    print("  Q dans la fenêtre ou Ctrl+C pour arrêter.\n")

    start_time = time.monotonic()

    # Fenêtre de monitoring
    monitor_name = "Enregistrement - Monitoring"
    cv2.namedWindow(monitor_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(monitor_name, 650, 420)

    while not stop_event.is_set():
        elapsed = time.monotonic() - start_time

        # Arrêt auto si durée max atteinte
        if max_duration > 0 and elapsed >= max_duration:
            print(f"\n  [TIMER] Durée max atteinte ({max_duration:.0f}s), arrêt...")
            stop_event.set()
            break

        elapsed_h = int(elapsed // 3600)
        elapsed_m = int((elapsed % 3600) // 60)
        elapsed_s = int(elapsed % 60)

        monitor = np.zeros((420, 650, 3), dtype=np.uint8)

        # Titre
        cv2.putText(monitor, "ENREGISTREMENT EN COURS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Durée
        if max_duration > 0:
            remaining = max(0, max_duration - elapsed)
            rem_m = int(remaining // 60)
            rem_s = int(remaining % 60)
            time_txt = (f"Duree: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"
                        f"  |  Restant: {rem_m:02d}:{rem_s:02d}")
        else:
            time_txt = f"Duree: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"
        cv2.putText(monitor, time_txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        enc_label = "NVENC (GPU)" if USE_GPU else "mp4v (CPU)"
        cv2.putText(monitor,
                    f"{WRITE_W}x{WRITE_H} | {TARGET_FPS} FPS | {enc_label} | {session_ts}",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        # Statut par caméra
        y = 115
        total_frames = 0
        connected_count = 0
        for rec in recorders:
            status = rec.get_status()
            total_frames += status['frames']
            is_conn = status['connected']
            if is_conn:
                connected_count += 1

            dot_color = (0, 255, 0) if is_conn else (0, 0, 255)
            cv2.circle(monitor, (20, y - 5), 6, dot_color, -1)

            cv2.putText(monitor, status['name'], (35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            frames_txt = f"{status['frames']:>7d} fr"
            cv2.putText(monitor, frames_txt, (280, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

            # Taille estimée du fichier
            est_mb = (status['frames'] * WRITE_W * WRITE_H * 3) / (1024**2) * 0.03
            cv2.putText(monitor, f"~{est_mb:.0f}MB", (400, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            dup_pct = (status['duplicated'] / max(status['frames'], 1)) * 100
            dup_color = (150, 150, 150) if dup_pct < 5 else (0, 165, 255)
            cv2.putText(monitor, f"dup:{dup_pct:.0f}%", (470, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, dup_color, 1)

            if status['reconnections'] > 0:
                cv2.putText(monitor, f"re:{status['reconnections']}",
                            (550, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

            if not is_conn and status['error']:
                cv2.putText(monitor, status['error'][:45], (35, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)
                y += 15

            y += 30

        # Résumé en bas
        cv2.putText(monitor,
                    f"Connectees: {connected_count}/{len(recorders)} | "
                    f"Total: {total_frames} frames | "
                    f"Sortie: recordings/recordings/",
                    (10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        cv2.imshow(monitor_name, monitor)

        key = cv2.waitKey(500) & 0xFF
        if key == ord('q') or key == 27:
            print("\n  [Q] Arrêt demandé...")
            stop_event.set()

    cv2.destroyAllWindows()

    print("\n  Fermeture des Writers...")
    time.sleep(2)

    # Résumé final
    elapsed = time.monotonic() - start_time
    print(f"\n╔═══════════════════════════════════════════════════════════╗")
    print(f"║  RÉSUMÉ DE L'ENREGISTREMENT                              ║")
    print(f"╠═══════════════════════════════════════════════════════════╣")
    for rec in recorders:
        s = rec.get_status()
        duration_est = s['frames'] / TARGET_FPS if s['frames'] > 0 else 0
        dur_m = int(duration_est // 60)
        dur_s = int(duration_est % 60)
        dup_pct = (s['duplicated'] / s['frames'] * 100) if s['frames'] > 0 else 0
        print(f"║  {s['name']:<18s} | {s['frames']:>7d} fr | "
              f"~{dur_m:02d}:{dur_s:02d} | dup: {dup_pct:.1f}%")
    print(f"╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Session  : {session_ts}")
    print(f"║  Durée    : {elapsed / 60:.1f} min")
    print(f"║  Dossier  : {RECORDINGS_DIR}/")
    print(f"╚═══════════════════════════════════════════════════════════╝")

    # Lister les fichiers produits
    new_files = sorted(RECORDINGS_DIR.glob(f"*_{session_ts}.mp4"))
    if new_files:
        print(f"\n  Fichiers créés ({len(new_files)}) :")
        for f in new_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name}  ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global USE_GPU
    args = sys.argv[1:]

    if '--gpu' in args:
        USE_GPU = True

    if '--record' in args:
        # Durée optionnelle en secondes après --record
        max_duration = 0
        idx = args.index('--record')
        if idx + 1 < len(args):
            try:
                max_duration = float(args[idx + 1])
            except ValueError:
                pass

        camera_ids = None
        if '--cameras' in args:
            cam_idx = args.index('--cameras')
            if cam_idx + 1 >= len(args):
                print("[ERREUR] --cameras demande une liste, ex: cam_02,cam_03")
                sys.exit(2)
            try:
                camera_ids = parse_camera_ids(args[cam_idx + 1])
            except ValueError as exc:
                print(f"[ERREUR] {exc}")
                sys.exit(2)

        mode_record(max_duration, camera_ids=camera_ids)

    elif '--help' in args or '-h' in args:
        print("Usage :")
        print("  python streamlive.py                  Replay (fake live)")
        print("  python streamlive.py --record          Enregistre les 8 caméras RTSP")
        print("  python streamlive.py --record 300      Enregistre pendant 300s (5 min)")
        print("  python streamlive.py --record --gpu    Enregistre avec GPU (NVENC)")
        print("  python streamlive.py --record 300 --gpu  Record 5min avec GPU")
        print("  python streamlive.py --record 300 --gpu --cameras cam_02,cam_03,cam_05,cam_07")
        print(f"\n  Les fichiers sont sauvegardés dans : {RECORDINGS_DIR}/")
        print(f"\n  --gpu  Utilise h264_nvenc (GPU NVIDIA) au lieu de mp4v (CPU)")
        print(f"         Requiert : drivers NVIDIA + FFmpeg avec support NVENC")

    else:
        mode_replay()


if __name__ == "__main__":
    main()

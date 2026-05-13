"""
Enregistrement multi-caméras RTSP via GStreamer + OpenCV.

- 8 flux simultanés, 2 threads par caméra (lecteur + écrivain)
- Frame Pacing : écriture stricte à TARGET_FPS via horloge monotonic
  Si aucune nouvelle frame n'arrive, la dernière est dupliquée.
  Résultat : vitesse 1:1 avec la réalité, zéro avance rapide.
- Horodatage incrusté sur chaque frame (pour annotation TRD)
- Codec mp4v (.mp4), 768x576, 25 FPS d'écriture
- Reconnexion automatique sans corruption
- Arrêt propre : Ctrl+C ou 'q' dans la fenêtre de monitoring

Fichiers produits dans recordings/recordings/
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

import cv2
import numpy as np
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
CAMERAS = [
    {'name': 'Camera 1 (2.2)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 2 (2.3)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera_3_2.4',    'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 4 (2.5)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 5 (2.6)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 6 (2.7)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 7 (2.11)', 'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 8 (2.13)', 'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
]

# Résolution et FPS d'écriture (substream 704x576 corrigé en 4:3)
WRITE_W = 768
WRITE_H = 576
TARGET_FPS = 25
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 0.04s entre chaque write

RECONNECT_DELAY = 3  # secondes entre tentatives de reconnexion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "recordings" / "recordings"

# Flag global d'arrêt
stop_event = threading.Event()


def build_gst_pipeline(rtsp_url: str) -> str:
    return (
        f'rtspsrc location="{rtsp_url}" latency=0 protocols=tcp ! '
        f'decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false'
    )


def stamp_frame(frame, cam_name: str, timestamp: str):
    """Incruste le nom de la caméra et l'horodatage sur la frame."""
    # Fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (480, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, cam_name, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    cv2.putText(frame, timestamp, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


class CameraRecorder:
    """Enregistre un flux RTSP avec frame pacing strict (vitesse 1:1).

    Architecture 2 threads :
      - Thread LECTEUR : cap.read() en boucle, stocke la dernière frame brute
      - Thread ÉCRIVAIN : tick à TARGET_FPS via time.monotonic(), écrit
        la nouvelle frame (horodatée) ou duplique la précédente si rien de neuf
    """

    def __init__(self, name: str, url: str, output_path: Path):
        self.name = name
        self.url = url
        self.output_path = output_path
        self.pipeline = build_gst_pipeline(url)

        # Shared state (protégé par lock)
        self._lock = threading.Lock()
        self._latest_raw = None       # dernière frame brute du lecteur
        self._new_frame_flag = False   # True si le lecteur a posé une frame non consommée
        self._frames_written = 0
        self._frames_duplicated = 0
        self._connected = False
        self._last_error = ""
        self._reconnections = 0

        # Thread lecteur (cap.read en boucle)
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True)
        # Thread écrivain (tick strict à TARGET_FPS)
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True)

        self._reader_thread.start()
        self._writer_thread.start()

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

    # ── Thread LECTEUR : lit GStreamer aussi vite que possible ────────────
    def _reader_loop(self):
        cap = None

        while not stop_event.is_set():
            # Connexion / reconnexion
            if cap is None or not cap.isOpened():
                with self._lock:
                    self._connected = False
                    self._last_error = "connexion en cours"
                if cap is not None:
                    cap.release()
                cap = self._open_capture()
                if not cap.isOpened():
                    with self._lock:
                        self._last_error = "ouverture GStreamer impossible"
                        self._reconnections += 1
                    print(f"  [{self.name}] Connexion échouée, "
                          f"retry dans {RECONNECT_DELAY}s...")
                    for _ in range(RECONNECT_DELAY * 10):
                        if stop_event.is_set():
                            break
                        time.sleep(0.1)
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  [{self.name}] Connecté via GStreamer.")
                print(f"  [{self.name}] Flux ouvert avec résolution : {w} x {h}")
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

            # Stocker la frame brute (redimensionnée) pour le thread écrivain.
            # 704x576 -> 768x576 restaure le ratio 4:3 attendu par la calibration.
            h, w = frame.shape[:2]
            if (w, h) != (704, 576) and (w, h) != (WRITE_W, WRITE_H):
                print(f"  [{self.name}] [WARN] Résolution source inattendue: "
                      f"{w}x{h}; sortie forcée en {WRITE_W}x{WRITE_H}")
            resized = cv2.resize(frame, (WRITE_W, WRITE_H))
            with self._lock:
                self._latest_raw = resized
                self._new_frame_flag = True

        if cap is not None:
            cap.release()

    # ── Thread ÉCRIVAIN : tick strict toutes les FRAME_INTERVAL sec ──────
    def _writer_loop(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(self.output_path), fourcc, TARGET_FPS, (WRITE_W, WRITE_H))

        last_written_frame = None  # dernière frame déjà horodatée et écrite
        next_tick = time.monotonic()

        while not stop_event.is_set():
            # Attendre le prochain tick
            now = time.monotonic()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            next_tick += FRAME_INTERVAL

            # Récupérer la frame du lecteur
            with self._lock:
                raw = self._latest_raw
                is_new = self._new_frame_flag
                self._new_frame_flag = False

            if raw is not None and is_new:
                # Nouvelle frame : horodater et écrire
                ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                stamp_frame(raw, self.name, ts)
                last_written_frame = raw.copy()
                writer.write(raw)
                with self._lock:
                    self._frames_written += 1
            elif last_written_frame is not None:
                # Pas de nouvelle frame : dupliquer la précédente telle quelle
                writer.write(last_written_frame)
                with self._lock:
                    self._frames_written += 1
                    self._frames_duplicated += 1
            # else: pas encore de frame du tout (caméra pas encore connectée)

            # Rattraper le retard si on a accumulé du drift
            if time.monotonic() > next_tick + FRAME_INTERVAL:
                next_tick = time.monotonic()

        writer.release()
        print(f"  [{self.name}] Enregistrement terminé. "
              f"{self._frames_written} frames écrites "
              f"({self._frames_duplicated} dupliquées).")

    def get_status(self) -> dict:
        with self._lock:
            return {
                'name': self.name,
                'connected': self._connected,
                'frames': self._frames_written,
                'duplicated': self._frames_duplicated,
                'reconnections': self._reconnections,
                'error': self._last_error,
            }


def signal_handler(sig, frame):
    """Gestion propre de Ctrl+C."""
    print("\n\n  [CTRL+C] Arrêt demandé, fermeture des enregistrements...")
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, signal_handler)

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  ENREGISTREMENT MULTI-CAMÉRAS RTSP (GStreamer + OpenCV)  ║")
    print(f"║  8 flux | mp4v .mp4 | {WRITE_W}x{WRITE_H} | 25 FPS | Horodaté     ║")
    print("║  Ctrl+C ou Q = Arrêt propre                              ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Vérifier GStreamer
    backends = [cv2.videoio_registry.getBackendName(b)
                for b in cv2.videoio_registry.getBackends()]
    gst_ok = "GSTREAMER" in backends
    print(f"\n  Backends OpenCV : {', '.join(backends)}")
    if gst_ok:
        print("  [OK] GStreamer détecté.")
    else:
        print("  [ATTENTION] GStreamer NON détecté !")

    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Timestamp de session pour nommer les fichiers
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Lancer les recorders
    recorders = []
    for cam in CAMERAS:
        safe_name = cam['name'].replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"{safe_name}_{session_ts}.mp4"
        output_path = OUTPUT_DIR / filename
        print(f"  Lancement {cam['name']} -> {filename}")
        recorders.append(CameraRecorder(cam['name'], cam['url'], output_path))

    print(f"\n  [{len(recorders)} threads démarrés]")
    print("  Appuyer sur Q dans la fenêtre de monitoring pour arrêter.\n")

    start_time = time.monotonic()

    # Fenêtre de monitoring (petite, statut texte uniquement)
    monitor_name = "Enregistrement - Monitoring"
    cv2.namedWindow(monitor_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(monitor_name, 600, 400)

    while not stop_event.is_set():
        elapsed = time.monotonic() - start_time
        elapsed_h = int(elapsed // 3600)
        elapsed_m = int((elapsed % 3600) // 60)
        elapsed_s = int(elapsed % 60)

        # Construire l'image de monitoring
        monitor = np.zeros((400, 600, 3), dtype=np.uint8)

        cv2.putText(monitor, "ENREGISTREMENT EN COURS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(monitor,
                    f"Duree: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(monitor,
                    f"Resolution: {WRITE_W}x{WRITE_H} | FPS: {TARGET_FPS} | "
                    f"Codec: mp4v | Frame Pacing: ON",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Statut par caméra
        y = 120
        total_frames = 0
        connected_count = 0
        for rec in recorders:
            status = rec.get_status()
            total_frames += status['frames']
            is_conn = status['connected']
            if is_conn:
                connected_count += 1

            # Indicateur couleur
            dot_color = (0, 255, 0) if is_conn else (0, 0, 255)
            cv2.circle(monitor, (20, y - 5), 6, dot_color, -1)

            # Nom + frames
            info = f"{status['name']}"
            cv2.putText(monitor, info, (35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            frames_txt = f"{status['frames']:>7d} fr"
            cv2.putText(monitor, frames_txt, (280, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

            dup_txt = f"dup:{status['duplicated']}"
            dup_color = (150, 150, 150) if status['duplicated'] == 0 else (0, 165, 255)
            cv2.putText(monitor, dup_txt, (390, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, dup_color, 1)

            if status['reconnections'] > 0:
                cv2.putText(monitor,
                            f"reconn:{status['reconnections']}",
                            (490, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

            if not is_conn and status['error']:
                cv2.putText(monitor, status['error'][:40], (35, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)
                y += 15

            y += 30

        # Résumé en bas
        cv2.putText(monitor,
                    f"Connectees: {connected_count}/{len(recorders)} | "
                    f"Total frames: {total_frames}",
                    (10, 385),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow(monitor_name, monitor)

        key = cv2.waitKey(500) & 0xFF
        if key == ord('q') or key == 27:
            print("\n  [Q] Arrêt demandé...")
            stop_event.set()

    cv2.destroyAllWindows()

    # Attendre que tous les threads terminent proprement
    print("\n  Fermeture des Writers en cours...")
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
        print(f"║  {s['name']:<22s} | {s['frames']:>7d} fr | "
              f"~{dur_m:02d}:{dur_s:02d} | dup: {dup_pct:.1f}% | "
              f"reconn: {s['reconnections']}")
    print(f"╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Durée totale session : {elapsed / 60:.1f} min")
    print(f"║  Fichiers dans : {OUTPUT_DIR}/")
    print(f"╚═══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()

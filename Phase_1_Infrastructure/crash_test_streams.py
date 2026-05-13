"""
Crash Test Streams - Phase 3.1 : Stabilité multi-caméras RTSP.

Test de stabilité : 8 flux RTSP simultanés pendant 30 min via GStreamer.
Vérifie la synchronisation douce (décalage < 200 ms entre caméras).

Pipeline zéro-latence : rtspsrc latency=0 -> h264 decode -> appsink drop=1
Aucune écriture disque. Affichage grille 2x4 avec timestamps.
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

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# ── Configuration des 8 caméras ─────────────────────────────────────────────
CAMERAS = [
    {'name': 'Camera 1 (2.2)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 2 (2.3)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 3 (2.4)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 4 (2.5)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 5 (2.6)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 6 (2.7)',  'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 7 (2.11)', 'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
    {'name': 'Camera 8 (2.13)', 'url': 'rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/cam/realmonitor?channel=1&subtype=1'},
]

# Taille de chaque cellule dans la grille
CELL_W = 720
CELL_H = 405

# Grille 2 lignes x 4 colonnes
GRID_COLS = 4
GRID_ROWS = 2

RECONNECT_DELAY = 2  # secondes avant tentative de reconnexion


def build_gst_pipeline(rtsp_url: str) -> str:
    """Construit un pipeline GStreamer intelligent et zéro-latence."""
    # On utilise decodebin pour qu'il détecte seul si c'est du H.264 ou H.265
    return (
        f'rtspsrc location="{rtsp_url}" latency=0 protocols=tcp ! '
        f'decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false'
    )


def build_rtsp_variants(rtsp_url: str) -> list[str]:
    """Retourne l'URL initiale puis une variante subtype=1 si possible."""
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
        # En cas d'URL non standard, on garde uniquement l'URL d'origine.
        pass
    return variants

class CameraReader:
    """Lecteur RTSP asynchrone via GStreamer avec thread dédié
    et reconnexion automatique."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.pipeline = build_gst_pipeline(url)
        self._url_candidates = build_rtsp_variants(url)

        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = None
        self._fps = 0.0
        self._connected = False
        self._running = True
        self._last_error = "initialisation"
        self._backend = ""
        self._active_url = ""

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _open_capture(self):
        """Ouvre la capture avec fallback: GST main -> GST sub -> FFMPEG."""
        errors = []

        for candidate_url in self._url_candidates:
            gst_pipeline = build_gst_pipeline(candidate_url)
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap, "GSTREAMER", candidate_url, ""
            cap.release()
            errors.append(f"GST KO ({candidate_url})")

            cap = cv2.VideoCapture(candidate_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                return cap, "FFMPEG", candidate_url, ""
            cap.release()
            errors.append(f"FFMPEG KO ({candidate_url})")

        return None, "", "", " | ".join(errors[-4:])

    def _reader_loop(self):
        cap = None
        frame_count = 0
        fps_timer = time.monotonic()

        while self._running:
            # Connexion / reconnexion
            if cap is None or not cap.isOpened():
                with self._lock:
                    self._connected = False
                    self._last_error = "connexion en cours"
                if cap is not None:
                    cap.release()
                cap, backend, active_url, open_error = self._open_capture()
                if cap is None:
                    with self._lock:
                        self._backend = ""
                        self._active_url = ""
                        self._last_error = (
                            f"ouverture RTSP impossible ({open_error})"[:120]
                        )
                    print(f"  [{self.name}] Connexion échouée, retry dans "
                          f"{RECONNECT_DELAY}s...")
                    time.sleep(RECONNECT_DELAY)
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                subtype = "?"
                try:
                    q = dict(parse_qsl(urlsplit(active_url).query, keep_blank_values=True))
                    subtype = q.get("subtype", "?")
                except Exception:
                    pass
                print(f"  [{self.name}] Connecté via {backend} (subtype={subtype}).")
                print(f"  [{self.name}] Flux ouvert avec résolution : {w} x {h}")
                with self._lock:
                    self._connected = True
                    self._last_error = ""
                    self._backend = backend
                    self._active_url = active_url
                frame_count = 0
                fps_timer = time.monotonic()

            ret, frame = cap.read()

            if not ret:
                with self._lock:
                    self._connected = False
                    self._last_error = "lecture frame échouée"
                print(f"  [{self.name}] Flux perdu, reconnexion...")
                cap.release()
                cap = None
                time.sleep(RECONNECT_DELAY)
                continue

            # Horodatage exact de réception
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            frame_count += 1

            # Calcul FPS (reset chaque seconde)
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.monotonic()
            else:
                fps = self._fps

            # Ne stocker que la dernière frame (vide le buffer)
            with self._lock:
                self._frame = frame
                self._timestamp = ts
                self._fps = fps

        if cap is not None:
            cap.release()

    def get_frame(self):
        """Retourne (frame, timestamp, fps, connected) thread-safe."""
        with self._lock:
            return (
                self._frame,
                self._timestamp,
                self._fps,
                self._connected,
                self._last_error,
            )

    def stop(self):
        self._running = False


def build_cell(reader: CameraReader) -> np.ndarray:
    """Construit une cellule 640x360 avec overlay informatif."""
    frame, ts, fps, connected, last_error = reader.get_frame()

    if frame is None:
        cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        label = "DECONNECTE" if not connected else "EN ATTENTE..."
        color = (0, 0, 200) if not connected else (0, 140, 255)
        cv2.putText(cell, reader.name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        cv2.putText(cell, label, (10, CELL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if last_error:
            msg = f"Cause: {last_error}"[:58]
            cv2.putText(cell, msg, (10, CELL_H // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
        return cell

    cell = cv2.resize(frame, (CELL_W, CELL_H))

    # Bande semi-transparente en haut
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (CELL_W, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, cell, 0.4, 0, cell)

    # Nom caméra (cyan)
    cv2.putText(cell, reader.name, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    # FPS (vert)
    cv2.putText(cell, f"{fps:.1f} FPS", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Timestamp de réception (blanc, pour vérifier la synchro)
    cv2.putText(cell, ts, (CELL_W - 200, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return cell


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  CRASH TEST STREAMS - Phase 3.1 Stabilité RTSP          ║")
    print("║  8 caméras | GStreamer zéro-latence | Grille 2x4        ║")
    print("║  Q = Quitter                                             ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Vérifier le backend GStreamer
    backends = [cv2.videoio_registry.getBackendName(b)
                for b in cv2.videoio_registry.getBackends()]
    gst_ok = "GSTREAMER" in backends
    print(f"\n  Backends OpenCV : {', '.join(backends)}")
    if gst_ok:
        print("  [OK] GStreamer détecté.")
    else:
        print("  [ATTENTION] GStreamer NON détecté. Le pipeline risque d'échouer.")
        print("  Vérifiez l'installation de GStreamer et le chemin des DLLs.\n")

    # Lancer les readers depuis la liste CAMERAS
    readers = []
    for cam in CAMERAS:
        print(f"  Lancement {cam['name']} -> {cam['url'][:55]}...")
        readers.append(CameraReader(cam['name'], cam['url']))

    print(f"\n  [{len(readers)} threads démarrés] Affichage grille "
          f"{GRID_ROWS}x{GRID_COLS}")
    print("  Appuyer sur Q pour quitter.\n")

    window_name = "Crash Test - 8 Cameras RTSP (GStreamer)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CELL_W * GRID_COLS, CELL_H * GRID_ROWS)

    start_time = time.monotonic()

    while True:
        # Construire la grille 2x4
        rows = []
        for r in range(GRID_ROWS):
            row_cells = []
            for c in range(GRID_COLS):
                idx = r * GRID_COLS + c
                if idx < len(readers):
                    row_cells.append(build_cell(readers[idx]))
                else:
                    row_cells.append(
                        np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
            rows.append(np.hstack(row_cells))
        grid = np.vstack(rows)

        # ── Barre d'info en bas ──────────────────────────────────────────
        elapsed_s = time.monotonic() - start_time
        elapsed_min = int(elapsed_s // 60)
        elapsed_sec = int(elapsed_s % 60)

        h_grid = grid.shape[0]
        w_grid = grid.shape[1]

        # Durée écoulée
        cv2.putText(grid, f"Duree: {elapsed_min:02d}:{elapsed_sec:02d} / 30:00",
                    (10, h_grid - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Vérification synchro : collecter les timestamps de réception
        timestamps = []
        for reader in readers:
            _, ts, _, connected, _ = reader.get_frame()
            if ts and connected:
                try:
                    t = datetime.strptime(ts, "%H:%M:%S.%f")
                    timestamps.append(t)
                except ValueError:
                    pass

        if len(timestamps) >= 2:
            deltas_ms = [
                abs((t - timestamps[0]).total_seconds() * 1000)
                for t in timestamps[1:]
            ]
            max_delta = max(deltas_ms)
            if max_delta < 200:
                sync_text = f"Synchro OK ({max_delta:.0f} ms)"
                sync_color = (0, 255, 0)
            else:
                sync_text = f"DESYNC! ({max_delta:.0f} ms > 200 ms)"
                sync_color = (0, 0, 255)
            cv2.putText(grid, sync_text, (400, h_grid - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, sync_color, 2)

        # Compteur connexions
        connected_count = sum(1 for r in readers if r.get_frame()[3])
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

    elapsed_s = time.monotonic() - start_time
    print(f"\n  Test terminé après {elapsed_s / 60:.1f} minutes.")
    print("  Résultat : vérifiez la console pour les déconnexions/reconnexions.")


if __name__ == "__main__":
    main()

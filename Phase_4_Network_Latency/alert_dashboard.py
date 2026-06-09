from __future__ import annotations

import argparse
import json
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


CLIENTS: list[queue.Queue[dict]] = []
METADATA_CLIENTS: list[queue.Queue[dict]] = []
CLIENTS_LOCK = threading.Lock()
METADATA_CLIENTS_LOCK = threading.Lock()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 4 Live Dashboard</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #f4f6f8; color: #101820; }
    header { padding: 18px 24px; background: #101820; color: #f7f9fb; }
    main { padding: 20px 24px; max-width: 1280px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 12px; margin-bottom: 18px; }
    .metric { background: #fff; border: 1px solid #d9e0e7; border-radius: 8px; padding: 14px; }
    .metric span { display: block; color: #5d6b78; font-size: 13px; }
    .metric strong { display: block; font-size: 26px; margin-top: 4px; }
    .viewer { display: none; margin-bottom: 18px; background: #fff; border: 1px solid #d9e0e7; border-radius: 8px; overflow: hidden; }
    .stage { position: relative; background: #0b1117; aspect-ratio: 16 / 9; }
    video, iframe, canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
    video { object-fit: contain; }
    iframe { border: 0; background: #0b1117; }
    canvas { pointer-events: none; }
    .viewer footer { display: flex; justify-content: space-between; gap: 12px; padding: 10px 12px; font-size: 13px; color: #5d6b78; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9e0e7; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e6ebf0; text-align: left; font-size: 14px; }
    th { color: #5d6b78; font-weight: 600; }
    .weak { color: #9a6200; font-weight: 700; }
    .confirmed { color: #b42318; font-weight: 700; }
    @media (max-width: 720px) { .grid { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <header><h1>Phase 4 Live Dashboard</h1></header>
  <main>
    <section class="grid">
      <div class="metric"><span>Total</span><strong id="total">0</strong></div>
      <div class="metric"><span>Weak</span><strong id="weak">0</strong></div>
      <div class="metric"><span>Confirmed</span><strong id="confirmed">0</strong></div>
      <div class="metric"><span>Last latency</span><strong id="latency">0 ms</strong></div>
    </section>
    <section class="viewer" id="viewer">
      <div class="stage">
        <video id="video" autoplay muted playsinline controls></video>
        <iframe id="videoFrame" allow="autoplay; fullscreen; camera; microphone"></iframe>
        <canvas id="overlay"></canvas>
      </div>
      <footer>
        <span id="cameraLabel">camera</span>
        <span id="frameLabel">frame -</span>
      </footer>
    </section>
    <table>
      <thead><tr><th>Time</th><th>Level</th><th>Type</th><th>Camera</th><th>Latency</th></tr></thead>
      <tbody id="events"></tbody>
    </table>
  </main>
  <script>
    const counts = { total: 0, weak: 0, confirmed: 0 };
    const body = document.getElementById("events");
    const params = new URLSearchParams(window.location.search);
    const selectedCamera = params.get("camera") || "cam_02";
    const videoUrl = params.get("video");
    const videoMode = params.get("video_mode") || params.get("mode") || "video";
    const viewer = document.getElementById("viewer");
    const video = document.getElementById("video");
    const videoFrame = document.getElementById("videoFrame");
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");
    let lastMetadata = null;

    video.style.display = "none";
    videoFrame.style.display = "none";
    if (videoUrl) {
      viewer.style.display = "block";
      if (videoMode === "iframe") {
        videoFrame.style.display = "block";
        videoFrame.src = videoUrl;
      } else {
        video.style.display = "block";
        video.src = videoUrl;
      }
      document.getElementById("cameraLabel").textContent = selectedCamera;
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width));
      canvas.height = Math.max(1, Math.round(rect.height));
    }

    function drawOverlay(metadata) {
      if (!videoUrl) return;
      resizeCanvas();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const detections = (metadata.detections || []).filter((det) => det.camera_id === selectedCamera);
      const videoW = videoMode === "iframe" ? 1280 : (video.videoWidth || canvas.width);
      const videoH = videoMode === "iframe" ? 720 : (video.videoHeight || canvas.height);
      const scale = Math.min(canvas.width / videoW, canvas.height / videoH);
      const offsetX = (canvas.width - videoW * scale) / 2;
      const offsetY = (canvas.height - videoH * scale) / 2;
      ctx.lineWidth = 2;
      ctx.font = "13px Segoe UI, Arial";
      for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox_px;
        const x = offsetX + x1 * scale;
        const y = offsetY + y1 * scale;
        const w = (x2 - x1) * scale;
        const h = (y2 - y1) * scale;
        const color = det.class_name === "personne" || det.class_name === "person" ? "#21a67a" : "#d97706";
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(x, y, w, h);
        const label = `${det.class_name} #${det.global_id ?? det.track_id} ${(det.confidence * 100).toFixed(0)}%`;
        const tw = ctx.measureText(label).width + 8;
        ctx.fillRect(x, Math.max(0, y - 20), tw, 18);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x + 4, Math.max(13, y - 6));
      }
      document.getElementById("frameLabel").textContent = `frame ${metadata.frame} · ${detections.length} bbox`;
    }

    window.addEventListener("resize", () => lastMetadata && drawOverlay(lastMetadata));

    function addAlertRow(event) {
      counts.total += 1;
      counts[event.alert_level] = (counts[event.alert_level] || 0) + 1;
      document.getElementById("total").textContent = counts.total;
      document.getElementById("weak").textContent = counts.weak;
      document.getElementById("confirmed").textContent = counts.confirmed;
      document.getElementById("latency").textContent = `${event.delivery_latency_ms.toFixed(1)} ms`;
      const row = document.createElement("tr");
      row.innerHTML = `<td>${new Date(event.received_epoch_ms).toLocaleTimeString()}</td>
        <td class="${event.alert_level}">${event.alert_level}</td>
        <td>${event.alert_type}</td>
        <td>${event.camera}</td>
        <td>${event.delivery_latency_ms.toFixed(1)} ms</td>`;
      body.prepend(row);
      while (body.children.length > 100) body.removeChild(body.lastChild);
    }

    const source = new EventSource("/events");
    source.onmessage = (message) => {
      const event = JSON.parse(message.data);
      addAlertRow(event);
    };

    const metadataSource = new EventSource("/metadata-events");
    metadataSource.onmessage = (message) => {
      lastMetadata = JSON.parse(message.data);
      drawOverlay(lastMetadata);
    };
  </script>
</body>
</html>
"""


def broadcast_alert(alert: dict) -> None:
    now_ms = time.time() * 1000.0
    created_ms = float(alert.get("created_epoch_ms", now_ms))
    alert["received_epoch_ms"] = now_ms
    alert["delivery_latency_ms"] = max(0.0, now_ms - created_ms)
    with CLIENTS_LOCK:
        clients = list(CLIENTS)
    for client in clients:
        client.put(alert)


def broadcast_metadata(metadata: dict) -> None:
    now_ms = time.time() * 1000.0
    created_ms = float(metadata.get("created_epoch_ms", now_ms))
    metadata["received_epoch_ms"] = now_ms
    metadata["delivery_latency_ms"] = max(0.0, now_ms - created_ms)
    with METADATA_CLIENTS_LOCK:
        clients = list(METADATA_CLIENTS)
    for client in clients:
        client.put(metadata)
    for alert in metadata.get("alerts", []):
        broadcast_alert(
            {
                "event_id": alert.get("alert_id"),
                "alert_level": alert.get("alert_level", "confirmed"),
                "alert_type": alert.get("alert_type", ""),
                "camera": "+".join(alert.get("cameras", [])),
                "created_epoch_ms": metadata["created_epoch_ms"],
            }
        )


def make_simulated_alert(event_id: int) -> dict:
    return {
        "event_id": event_id,
        "alert_level": "confirmed" if event_id % 5 == 0 else "weak",
        "alert_type": "forbidden_object" if event_id % 3 else "zone_violation_person",
        "camera": random.choice(["cam_02", "cam_03", "cam_05", "cam_07"]),
        "created_epoch_ms": time.time() * 1000.0,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return

        if parsed.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            channel: queue.Queue[dict] = queue.Queue()
            with CLIENTS_LOCK:
                CLIENTS.append(channel)
            try:
                while True:
                    alert = channel.get(timeout=15.0)
                    payload = json.dumps(alert)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, queue.Empty):
                pass
            finally:
                with CLIENTS_LOCK:
                    if channel in CLIENTS:
                        CLIENTS.remove(channel)
            return

        if parsed.path == "/metadata-events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            channel: queue.Queue[dict] = queue.Queue()
            with METADATA_CLIENTS_LOCK:
                METADATA_CLIENTS.append(channel)
            try:
                while True:
                    metadata = channel.get(timeout=15.0)
                    payload = json.dumps(metadata)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, queue.Empty):
                pass
            finally:
                with METADATA_CLIENTS_LOCK:
                    if channel in METADATA_CLIENTS:
                        METADATA_CLIENTS.remove(channel)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/alerts", "/metadata"}:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if parsed.path == "/metadata":
            broadcast_metadata(payload)
        else:
            broadcast_alert(payload)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return


def start_simulator(rate_hz: float) -> None:
    def run() -> None:
        event_id = 0
        spacing_s = 1.0 / rate_hz if rate_hz > 0 else 1.0
        while True:
            broadcast_alert(make_simulated_alert(event_id))
            event_id += 1
            time.sleep(spacing_s)

    threading.Thread(target=run, daemon=True).start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Phase 4 alert dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simulate:
        start_simulator(args.rate_hz)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[INFO] Dashboard: http://{args.host}:{args.port}")
    print("[INFO] POST alerts to /alerts or use --simulate")
    server.serve_forever()


if __name__ == "__main__":
    main()

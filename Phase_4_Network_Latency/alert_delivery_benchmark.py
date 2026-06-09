from __future__ import annotations

import argparse
import asyncio
import csv
import json
import queue
import random
import statistics
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class AlertEvent:
    event_id: int
    alert_level: str
    created_ns: int

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "event_id": self.event_id,
                "alert_level": self.alert_level,
                "created_ns": self.created_ns,
            }
        ).encode("utf-8")


@dataclass(frozen=True)
class AlertLatency:
    event_id: int
    transport: str
    latency_ms: float


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize_latencies(rows: list[AlertLatency]) -> dict[str, float]:
    latencies = [row.latency_ms for row in rows]
    return {
        "events": float(len(rows)),
        "mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_ms": percentile(latencies, 95.0),
        "max_ms": max(latencies) if latencies else 0.0,
    }


def run_queue_benchmark(events: int, rate_hz: float) -> list[AlertLatency]:
    channel: queue.Queue[AlertEvent | None] = queue.Queue()
    rows: list[AlertLatency] = []

    def receiver() -> None:
        while True:
            event = channel.get()
            if event is None:
                return
            rows.append(
                AlertLatency(
                    event_id=event.event_id,
                    transport="queue",
                    latency_ms=(time.perf_counter_ns() - event.created_ns) / 1_000_000.0,
                )
            )

    thread = threading.Thread(target=receiver, daemon=True)
    thread.start()
    spacing_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
    for event_id in range(events):
        level = "confirmed" if event_id % 5 == 0 else "weak"
        channel.put(AlertEvent(event_id=event_id, alert_level=level, created_ns=time.perf_counter_ns()))
        if spacing_s:
            time.sleep(spacing_s)
    channel.put(None)
    thread.join(timeout=5.0)
    return rows


class _AlertPostHandler(BaseHTTPRequestHandler):
    received: list[AlertLatency] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        _AlertPostHandler.received.append(
            AlertLatency(
                event_id=int(payload["event_id"]),
                transport="http_post",
                latency_ms=(time.perf_counter_ns() - int(payload["created_ns"])) / 1_000_000.0,
            )
        )
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return


def run_http_post_benchmark(events: int, rate_hz: float) -> list[AlertLatency]:
    _AlertPostHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AlertPostHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/alerts"
    spacing_s = 1.0 / rate_hz if rate_hz > 0 else 0.0

    try:
        for event_id in range(events):
            level = "confirmed" if event_id % 5 == 0 else "weak"
            event = AlertEvent(event_id=event_id, alert_level=level, created_ns=time.perf_counter_ns())
            request = Request(url, data=event.to_json(), headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=2.0):
                pass
            if spacing_s:
                time.sleep(spacing_s)
    finally:
        server.shutdown()
        thread.join(timeout=5.0)

    return list(_AlertPostHandler.received)


async def _run_websocket_benchmark_async(events: int, rate_hz: float) -> list[AlertLatency]:
    try:
        import websockets
    except ModuleNotFoundError as exc:
        raise SystemExit("WebSocket benchmark requires: pip install websockets") from exc

    rows: list[AlertLatency] = []

    async def handler(websocket) -> None:
        async for message in websocket:
            payload = json.loads(message)
            rows.append(
                AlertLatency(
                    event_id=int(payload["event_id"]),
                    transport="websocket",
                    latency_ms=(time.perf_counter_ns() - int(payload["created_ns"])) / 1_000_000.0,
                )
            )

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    spacing_s = 1.0 / rate_hz if rate_hz > 0 else 0.0

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as websocket:
            for event_id in range(events):
                level = "confirmed" if event_id % 5 == 0 else "weak"
                event = AlertEvent(event_id=event_id, alert_level=level, created_ns=time.perf_counter_ns())
                await websocket.send(event.to_json().decode("utf-8"))
                if spacing_s:
                    await asyncio.sleep(spacing_s)

        deadline = time.monotonic() + 5.0
        while len(rows) < events and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
    finally:
        server.close()
        await server.wait_closed()

    return rows


def run_websocket_benchmark(events: int, rate_hz: float) -> list[AlertLatency]:
    return asyncio.run(_run_websocket_benchmark_async(events, rate_hz))


def run_mqtt_benchmark(
    events: int,
    rate_hz: float,
    qos: int,
    host: str,
    port: int,
    topic: str,
) -> list[AlertLatency]:
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MQTT benchmark requires: pip install paho-mqtt and a local broker such as mosquitto"
        ) from exc

    rows: list[AlertLatency] = []
    ready = threading.Event()

    def on_connect(client, userdata, flags, rc, properties=None) -> None:
        if rc != 0:
            return
        client.subscribe(topic, qos=qos)
        ready.set()

    def on_message(client, userdata, message) -> None:
        payload = json.loads(message.payload.decode("utf-8"))
        rows.append(
            AlertLatency(
                event_id=int(payload["event_id"]),
                transport=f"mqtt_qos{qos}",
                latency_ms=(time.perf_counter_ns() - int(payload["created_ns"])) / 1_000_000.0,
            )
        )

    subscriber = mqtt.Client()
    subscriber.on_connect = on_connect
    subscriber.on_message = on_message
    subscriber.connect(host, port, keepalive=30)
    subscriber.loop_start()
    if not ready.wait(timeout=5.0):
        subscriber.loop_stop()
        subscriber.disconnect()
        raise SystemExit(f"MQTT subscriber could not connect to {host}:{port}")

    publisher = mqtt.Client()
    publisher.connect(host, port, keepalive=30)
    publisher.loop_start()
    spacing_s = 1.0 / rate_hz if rate_hz > 0 else 0.0

    try:
        for event_id in range(events):
            level = "confirmed" if event_id % 5 == 0 else "weak"
            event = AlertEvent(event_id=event_id, alert_level=level, created_ns=time.perf_counter_ns())
            publisher.publish(topic, event.to_json(), qos=qos)
            if spacing_s:
                time.sleep(spacing_s)

        deadline = time.monotonic() + 5.0
        while len(rows) < events and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        publisher.loop_stop()
        publisher.disconnect()
        subscriber.loop_stop()
        subscriber.disconnect()

    return rows


def write_latency_rows(rows: list[AlertLatency], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "transport", "latency_ms"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "event_id": row.event_id,
                    "transport": row.transport,
                    "latency_ms": round(row.latency_ms, 6),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic alert delivery latency benchmark.")
    parser.add_argument("--transport", choices=["queue", "http_post", "websocket", "mqtt"], default="queue")
    parser.add_argument("--qos", type=int, default=0)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", default="phase4/alerts")
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--rate-hz", type=float, default=25.0)
    parser.add_argument("--out", type=Path, default=Path("Phase_4_Network_Latency/runs/alert_latency.csv"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if args.transport == "queue":
        rows = run_queue_benchmark(args.events, args.rate_hz)
    elif args.transport == "http_post":
        rows = run_http_post_benchmark(args.events, args.rate_hz)
    elif args.transport == "websocket":
        rows = run_websocket_benchmark(args.events, args.rate_hz)
    elif args.transport == "mqtt":
        rows = run_mqtt_benchmark(
            events=args.events,
            rate_hz=args.rate_hz,
            qos=args.qos,
            host=args.mqtt_host,
            port=args.mqtt_port,
            topic=args.mqtt_topic,
        )
    else:
        raise ValueError(f"Unsupported transport: {args.transport}")

    write_latency_rows(rows, args.out)
    summary = summarize_latencies(rows)
    print(
        "[INFO] "
        f"events={summary['events']:.0f} "
        f"mean={summary['mean_ms']:.3f}ms "
        f"median={summary['median_ms']:.3f}ms "
        f"p95={summary['p95_ms']:.3f}ms "
        f"max={summary['max_ms']:.3f}ms"
    )
    print(f"[INFO] CSV written: {args.out}")


if __name__ == "__main__":
    main()

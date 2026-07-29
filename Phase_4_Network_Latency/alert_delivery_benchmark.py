"""Benchmark synthétique de latence de livraison d'alertes selon le transport.

Quatre transports sont comparés. Dans chaque cas, la latence est mesurée côté
récepteur avec ``time.perf_counter_ns`` entre la création de l'événement et sa
réception (aller simple, émetteur et récepteur dans le même processus) :

- ``queue``     : file locale ``queue.Queue`` entre deux threads, sans réseau
  (borne basse de référence) ;
- ``http_post`` : requêtes POST JSON vers un mini serveur HTTP sur localhost ;
- ``websocket`` : messages sur une connexion WebSocket locale ;
- ``mqtt``      : publication/souscription via un broker MQTT externe (QoS 0 ou 1).

Dépendances optionnelles : le transport ``websocket`` requiert le paquet
``websockets`` (pip install websockets) ; le transport ``mqtt`` requiert
``paho-mqtt`` (pip install paho-mqtt) ainsi qu'un broker accessible, par
exemple mosquitto en local. Les transports ``queue`` et ``http_post`` ne
dépendent que de la bibliothèque standard.

Exemple::

    python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport queue --events 500 --rate-hz 25

Le script écrit un CSV avec une ligne par événement (event_id, transport,
latency_ms) et affiche un résumé (moyenne, médiane, p95, max).
"""

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
    """Alerte synthétique émise par le benchmark.

    Attributs :
        event_id : numéro séquentiel de l'événement ;
        alert_level : niveau simulé (``confirmed`` ou ``weak``) ;
        created_ns : horodatage de création en nanosecondes (``time.perf_counter_ns``),
            transporté avec l'événement pour calculer la latence à la réception.
    """

    event_id: int
    alert_level: str
    created_ns: int

    def to_json(self) -> bytes:
        """Sérialise l'événement en JSON encodé UTF-8 (charge utile des transports réseau)."""
        return json.dumps(
            {
                "event_id": self.event_id,
                "alert_level": self.alert_level,
                "created_ns": self.created_ns,
            }
        ).encode("utf-8")


@dataclass(frozen=True)
class AlertLatency:
    """Mesure de latence d'un événement livré : identifiant, transport utilisé, latence en ms."""

    event_id: int
    transport: str
    latency_ms: float


def percentile(values: list[float], pct: float) -> float:
    """Retourne le percentile ``pct`` (0 à 100) par la méthode du rang le plus proche.

    Exemple : ``percentile(latences, 95.0)`` pour le p95. Retourne 0.0 sur une
    liste vide.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize_latencies(rows: list[AlertLatency]) -> dict[str, float]:
    """Agrège les mesures en statistiques : nombre d'événements, moyenne, médiane, p95 et max (ms)."""
    latencies = [row.latency_ms for row in rows]
    return {
        "events": float(len(rows)),
        "mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_ms": percentile(latencies, 95.0),
        "max_ms": max(latencies) if latencies else 0.0,
    }


def run_queue_benchmark(events: int, rate_hz: float) -> list[AlertLatency]:
    """Benchmark du transport ``queue`` : file locale entre un thread émetteur et un thread récepteur.

    L'émetteur pousse ``events`` événements espacés de ``1 / rate_hz`` secondes
    (aucune pause si la cadence est nulle) ; un ``None`` final sert de signal
    d'arrêt au récepteur. Mesure le coût minimal de remise inter-threads, sans
    sérialisation ni réseau.
    """
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
    """Récepteur HTTP du benchmark ``http_post``.

    Chaque POST est horodaté à la réception et la mesure est accumulée dans
    l'attribut de classe ``received`` (réinitialisé à chaque exécution du
    benchmark).
    """

    received: list[AlertLatency] = []

    def do_POST(self) -> None:
        """Décode l'événement JSON reçu, calcule sa latence et répond 204."""
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
        """Désactive le journal d'accès HTTP pour ne pas fausser les mesures."""
        return


def run_http_post_benchmark(events: int, rate_hz: float) -> list[AlertLatency]:
    """Benchmark du transport ``http_post`` : POST JSON vers un serveur HTTP local.

    Démarre un serveur éphémère sur 127.0.0.1 (port choisi par l'OS), envoie
    ``events`` requêtes synchrones espacées de ``1 / rate_hz`` secondes, puis
    arrête le serveur. La latence inclut la sérialisation JSON, la pile TCP
    locale et le traitement de la requête.
    """
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
    """Cœur asynchrone du benchmark ``websocket`` (serveur et client dans la même boucle asyncio).

    Ouvre un serveur WebSocket local sur un port libre, y connecte un client,
    envoie ``events`` messages JSON espacés de ``1 / rate_hz`` secondes, puis
    attend (au plus 5 s) que toutes les réceptions soient comptabilisées.

    Dépendance optionnelle : le paquet ``websockets`` est importé ici à la
    demande ; s'il est absent, le script s'arrête avec un message expliquant
    comment l'installer.
    """
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
    """Enveloppe synchrone du benchmark ``websocket`` (exécute la version asynchrone)."""
    return asyncio.run(_run_websocket_benchmark_async(events, rate_hz))


def run_mqtt_benchmark(
    events: int,
    rate_hz: float,
    qos: int,
    host: str,
    port: int,
    topic: str,
) -> list[AlertLatency]:
    """Benchmark du transport ``mqtt`` : publication/souscription via un broker externe.

    Connecte un abonné puis un éditeur au broker ``host:port``, publie
    ``events`` messages JSON sur ``topic`` avec le QoS demandé (0 ou 1),
    espacés de ``1 / rate_hz`` secondes, puis attend (au plus 5 s) la réception
    complète. La latence inclut l'aller-retour par le broker.

    Dépendance optionnelle : le paquet ``paho-mqtt`` est importé ici à la
    demande ; il faut aussi un broker MQTT joignable (par exemple mosquitto en
    local). Le script s'arrête avec un message explicite si le paquet manque ou
    si la connexion échoue.
    """
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
    """Écrit les mesures brutes dans un CSV (event_id, transport, latency_ms), en créant les dossiers au besoin."""
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
    """Analyse les options de la ligne de commande du benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark synthétique de latence de livraison d'alertes.")
    parser.add_argument(
        "--transport",
        choices=["queue", "http_post", "websocket", "mqtt"],
        default="queue",
        help="Transport à mesurer (défaut : queue). websocket et mqtt requièrent des dépendances optionnelles.",
    )
    parser.add_argument("--qos", type=int, default=0, help="Niveau de QoS MQTT, 0 ou 1 (transport mqtt uniquement).")
    parser.add_argument("--mqtt-host", default="127.0.0.1", help="Adresse du broker MQTT (transport mqtt uniquement).")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="Port du broker MQTT (défaut : 1883).")
    parser.add_argument("--mqtt-topic", default="phase4/alerts", help="Sujet MQTT de publication des alertes.")
    parser.add_argument("--events", type=int, default=500, help="Nombre d'événements à émettre (défaut : 500).")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=25.0,
        help="Cadence d'émission en Hz ; 0 pour émettre sans pause (défaut : 25).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Phase_4_Network_Latency/runs/alert_latency.csv"),
        help="Chemin du CSV de sortie (une ligne par événement).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Graine du générateur aléatoire (reproductibilité).")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : exécute le benchmark du transport choisi, écrit le CSV et affiche le résumé."""
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

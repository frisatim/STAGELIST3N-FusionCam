"""Génération de la matrice d'expériences réseau de la Phase 4.

Ce script énumère les combinaisons (OS cible x type de lien x condition réseau
x transport vidéo x transport d'alertes) et produit deux fichiers : un CSV
exploitable par script et un plan Markdown lisible. Chaque cas contient la
commande de run à lancer (campagne live Phase 3 ou benchmark d'alertes Phase 4)
ainsi que les commandes de dégradation réseau à appliquer avant le run et à
retirer après.

Exemple::

    python Phase_4_Network_Latency/experiment_plan.py --interface eth0 --duration-min 10

Avertissement : sous Linux, les commandes générées contiennent ``sudo tc qdisc
... netem`` et modifient réellement la configuration réseau de l'interface
indiquée (délai, gigue, perte). Elles ne doivent être exécutées que sur la
machine de test dédiée, jamais sur un poste de production. Le script lui-même
n'exécute rien : il ne fait qu'écrire les commandes dans les fichiers de
sortie. Pour une cible Windows, des lignes ``REM`` rappellent de configurer
manuellement un émulateur réseau (type NetLimiter ou clumsy).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NetworkCondition:
    """Condition réseau émulée : nom, délai ajouté (ms), gigue (ms) et taux de perte (%)."""

    name: str
    delay_ms: int
    jitter_ms: int
    loss_pct: float

    def tc_apply_command(self, interface: str) -> str:
        """Retourne la commande ``tc netem`` (Linux) appliquant cette condition à l'interface.

        Pour la condition ``clean``, retourne la commande de suppression de la
        discipline de file (aucune dégradation). Attention : ces commandes
        utilisent ``sudo tc`` et sont destinées à la machine de test uniquement.
        """
        if self.name == "clean":
            return f"sudo tc qdisc del dev {interface} root"
        return (
            f"sudo tc qdisc replace dev {interface} root netem "
            f"delay {self.delay_ms}ms {self.jitter_ms}ms loss {self.loss_pct}%"
        )


@dataclass(frozen=True)
class VideoTransport:
    """Transport vidéo testé : backend de capture, protocole RTSP (tcp/udp) et latence GStreamer (ms)."""

    name: str
    capture_backend: str
    gst_protocol: str
    gst_latency_ms: int
    gst_pipeline: str = "decodebin"


@dataclass(frozen=True)
class AlertTransport:
    """Transport d'alertes testé : nom, protocole (websocket/mqtt/none) et QoS MQTT éventuel."""

    name: str
    protocol: str
    qos: int | None = None


@dataclass(frozen=True)
class ExperimentCase:
    """Cas d'expérience complet de la matrice Phase 4.

    Regroupe l'identifiant du run, sa famille (``video_network`` ou
    ``alert_transport``), le contexte (OS cible, type de lien), la condition
    réseau et les transports testés, ainsi que les trois commandes associées :
    ``tc_apply`` (dégradation réseau avant le run), ``command`` (le run
    lui-même) et ``tc_clear`` (retour à la normale après le run).
    """

    run_id: str
    family: str
    os_target: str
    link_type: str
    network: NetworkCondition
    video: VideoTransport
    alert: AlertTransport
    command: str
    tc_apply: str
    tc_clear: str

    def csv_row(self) -> dict[str, str | int | float]:
        """Aplati le cas en une ligne de dictionnaire pour l'écriture CSV."""
        return {
            "run_id": self.run_id,
            "family": self.family,
            "os_target": self.os_target,
            "link_type": self.link_type,
            "network_condition": self.network.name,
            "delay_ms": self.network.delay_ms,
            "jitter_ms": self.network.jitter_ms,
            "loss_pct": self.network.loss_pct,
            "video_transport": self.video.name,
            "capture_backend": self.video.capture_backend,
            "gst_protocol": self.video.gst_protocol,
            "gst_latency_ms": self.video.gst_latency_ms,
            "alert_transport": self.alert.name,
            "alert_protocol": self.alert.protocol,
            "alert_qos": "" if self.alert.qos is None else self.alert.qos,
            "tc_apply": self.tc_apply,
            "tc_clear": self.tc_clear,
            "command": self.command,
        }


# Les trois conditions réseau étudiées : réseau propre (référence), dégradation
# modérée et dégradation sévère (délai/gigue/perte croissants).
NETWORK_CONDITIONS = (
    NetworkCondition("clean", delay_ms=0, jitter_ms=0, loss_pct=0.0),
    NetworkCondition("moderate", delay_ms=80, jitter_ms=25, loss_pct=1.0),
    NetworkCondition("severe", delay_ms=200, jitter_ms=75, loss_pct=5.0),
)

# Transports vidéo comparés : RTSP sur TCP (fiable) et sur UDP (faible latence).
VIDEO_TRANSPORTS = (
    VideoTransport("rtsp_tcp", capture_backend="gstreamer", gst_protocol="tcp", gst_latency_ms=50),
    VideoTransport("rtsp_udp", capture_backend="gstreamer", gst_protocol="udp", gst_latency_ms=50),
)

# Transports d'alertes comparés : WebSocket, MQTT QoS 0 et MQTT QoS 1.
ALERT_TRANSPORTS = (
    AlertTransport("websocket", protocol="websocket"),
    AlertTransport("mqtt_qos0", protocol="mqtt", qos=0),
    AlertTransport("mqtt_qos1", protocol="mqtt", qos=1),
)


def network_apply_command(os_target: str, network: NetworkCondition, interface: str) -> str:
    """Retourne la commande d'application de la condition réseau selon l'OS cible.

    Linux : commande ``sudo tc netem`` réelle (machine de test uniquement).
    Windows : simple ligne ``REM`` rappelant les paramètres à reproduire dans
    un émulateur réseau (NetLimiter, clumsy, etc.), aucune commande exécutable.
    """
    if os_target.lower() == "linux":
        return network.tc_apply_command(interface)
    if network.name == "clean":
        return "REM no network degradation"
    return (
        "REM configure Windows network emulator "
        f"delay={network.delay_ms}ms jitter={network.jitter_ms}ms loss={network.loss_pct}%"
    )


def network_clear_command(os_target: str, interface: str) -> str:
    """Retourne la commande de retrait de toute dégradation réseau selon l'OS cible."""
    if os_target.lower() == "linux":
        return f"sudo tc qdisc del dev {interface} root"
    return "REM disable Windows network emulator"


def build_live_command(
    *,
    python_exe: str,
    version: str,
    model: str,
    fmt: str,
    cameras: str,
    duration_min: float,
    device: str,
    record_fps: float,
    video: VideoTransport,
    out_dir: str,
) -> str:
    """Construit la ligne de commande d'une campagne live Phase 3 pour un transport vidéo donné.

    La commande force le backend GStreamer (``--no-ffmpeg-fallback``) afin que
    le protocole RTSP testé (tcp/udp) soit réellement celui utilisé.
    """
    return (
        f"{python_exe} Phase_3_Fusion_MultiCam/run_live_campaign.py "
        f"--versions {version} --models {model} --formats {fmt} "
        f"--cameras {cameras} --duration-min {duration_min:g} --device {device} "
        f"--record-fps {record_fps:g} --object-min-camera-votes 2 "
        f"--capture-backend {video.capture_backend} --gst-protocol {video.gst_protocol} "
        f"--gst-latency-ms {video.gst_latency_ms} --gst-pipeline {video.gst_pipeline} "
        f"--no-ffmpeg-fallback --no-display --out-dir {out_dir}"
    )


def generate_experiment_cases(
    *,
    interface: str = "eth0",
    os_targets: tuple[str, ...] = ("linux",),
    link_types: tuple[str, ...] = ("ethernet",),
    python_exe: str = "python",
    version: str = "V4",
    model: str = "yolov8s",
    fmt: str = "fp32_engine",
    cameras: str = "cam_02,cam_03,cam_05,cam_07",
    duration_min: float = 10.0,
    device: str = "cuda:0",
    record_fps: float = 25.0,
) -> list[ExperimentCase]:
    """Génère la liste complète des cas d'expérience de la matrice Phase 4.

    Deux familles de cas sont produites pour chaque couple (OS cible, type de
    lien) :

    - ``video_network`` : produit cartésien des conditions réseau et des
      transports vidéo (campagne live Phase 3 sous réseau dégradé) ;
    - ``alert_transport`` : un cas par transport d'alertes, toujours en réseau
      propre et avec le premier transport vidéo (mesure isolée du transport).

    Chaque cas embarque sa commande de run et ses commandes réseau apply/clear.
    """
    cases: list[ExperimentCase] = []

    for os_target in os_targets:
        for link_type in link_types:
            # Famille video_network : conditions réseau x transports vidéo.
            for network in NETWORK_CONDITIONS:
                for video in VIDEO_TRANSPORTS:
                    run_id = f"video_{os_target}_{link_type}_{network.name}_{video.name}"
                    out_dir = f"Phase_4_Network_Latency/runs/{run_id}"
                    cases.append(
                        ExperimentCase(
                            run_id=run_id,
                            family="video_network",
                            os_target=os_target,
                            link_type=link_type,
                            network=network,
                            video=video,
                            alert=AlertTransport("none", protocol="none"),
                            command=build_live_command(
                                python_exe=python_exe,
                                version=version,
                                model=model,
                                fmt=fmt,
                                cameras=cameras,
                                duration_min=duration_min,
                                device=device,
                                record_fps=record_fps,
                                video=video,
                                out_dir=out_dir,
                            ),
                            tc_apply=network_apply_command(os_target, network, interface),
                            tc_clear=network_clear_command(os_target, interface),
                        )
                    )

            # Famille alert_transport : un cas par transport d'alertes, en
            # réseau propre, pour isoler le coût du transport lui-même.
            clean = NETWORK_CONDITIONS[0]
            video = VIDEO_TRANSPORTS[0]
            for alert in ALERT_TRANSPORTS:
                run_id = f"alert_{os_target}_{link_type}_{alert.name}"
                cases.append(
                    ExperimentCase(
                        run_id=run_id,
                        family="alert_transport",
                        os_target=os_target,
                        link_type=link_type,
                        network=clean,
                        video=video,
                        alert=alert,
                        command=(
                            f"{python_exe} Phase_4_Network_Latency/alert_delivery_benchmark.py "
                            f"--transport {alert.protocol} "
                            f"--qos {0 if alert.qos is None else alert.qos} "
                            f"--events 500 --rate-hz 25 "
                            f"--out Phase_4_Network_Latency/runs/{run_id}/alert_latency.csv"
                        ),
                        tc_apply=network_apply_command(os_target, clean, interface),
                        tc_clear=network_clear_command(os_target, interface),
                    )
                )

    return cases


def write_cases_csv(cases: list[ExperimentCase], out_csv: Path) -> None:
    """Écrit la matrice au format CSV (une ligne par cas), en créant les dossiers au besoin."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(cases[0].csv_row().keys()) if cases else ["run_id"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(case.csv_row())


def write_cases_markdown(cases: list[ExperimentCase], out_md: Path) -> None:
    """Écrit le plan d'expériences au format Markdown lisible.

    Chaque cas devient une section avec ses caractéristiques puis trois blocs de
    code : mise en place réseau, commande de run et nettoyage réseau.
    """
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 experiment matrix",
        "",
        "Run one case at a time and keep the generated report directory.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case.run_id}",
                "",
                f"- family: `{case.family}`",
                f"- network: `{case.network.name}`",
                f"- video: `{case.video.name}`",
                f"- alert: `{case.alert.name}`",
                "",
                "Network setup:",
                "",
                f"```bash\n{case.tc_apply}\n```",
                "",
                "Run:",
                "",
                f"```powershell\n{case.command}\n```",
                "",
                "Network cleanup:",
                "",
                f"```bash\n{case.tc_clear}\n```",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Analyse les options de la ligne de commande du générateur de matrice."""
    parser = argparse.ArgumentParser(
        description="Génère la matrice d'expériences réseau de la Phase 4 (CSV + Markdown).",
        epilog=(
            "Les commandes réseau générées pour Linux contiennent sudo tc netem : "
            "à exécuter uniquement sur la machine de test dédiée."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("Phase_4_Network_Latency/phase4_experiment_matrix.csv"),
        help="Chemin du CSV de sortie (une ligne par cas d'expérience).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("Phase_4_Network_Latency/phase4_experiment_matrix.md"),
        help="Chemin du plan Markdown de sortie (un cas par section).",
    )
    parser.add_argument("--interface", default="eth0", help="Interface réseau visée par les commandes tc netem.")
    parser.add_argument("--python-exe", default="python", help="Exécutable Python inséré dans les commandes de run.")
    parser.add_argument(
        "--os-targets",
        default="linux",
        help="OS cibles séparés par des virgules (ex. linux,windows) ; détermine tc netem ou REM.",
    )
    parser.add_argument(
        "--link-types",
        default="ethernet",
        help="Types de lien séparés par des virgules (ex. ethernet,wifi) ; sert au nommage des runs.",
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=10.0,
        help="Durée de chaque campagne live Phase 3 en minutes (défaut : 10).",
    )
    parser.add_argument("--device", default="cuda:0", help="Périphérique d'inférence inséré dans les commandes de run.")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée : génère les cas puis écrit la matrice CSV et le plan Markdown."""
    args = parse_args()
    cases = generate_experiment_cases(
        interface=args.interface,
        os_targets=tuple(part.strip() for part in args.os_targets.split(",") if part.strip()),
        link_types=tuple(part.strip() for part in args.link_types.split(",") if part.strip()),
        python_exe=args.python_exe,
        duration_min=args.duration_min,
        device=args.device,
    )
    write_cases_csv(cases, args.out_csv)
    write_cases_markdown(cases, args.out_md)
    print(f"[INFO] Wrote {len(cases)} cases to {args.out_csv}")
    print(f"[INFO] Wrote markdown plan to {args.out_md}")


if __name__ == "__main__":
    main()

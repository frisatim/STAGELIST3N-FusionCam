from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NetworkCondition:
    name: str
    delay_ms: int
    jitter_ms: int
    loss_pct: float

    def tc_apply_command(self, interface: str) -> str:
        if self.name == "clean":
            return f"sudo tc qdisc del dev {interface} root"
        return (
            f"sudo tc qdisc replace dev {interface} root netem "
            f"delay {self.delay_ms}ms {self.jitter_ms}ms loss {self.loss_pct}%"
        )


@dataclass(frozen=True)
class VideoTransport:
    name: str
    capture_backend: str
    gst_protocol: str
    gst_latency_ms: int
    gst_pipeline: str = "decodebin"


@dataclass(frozen=True)
class AlertTransport:
    name: str
    protocol: str
    qos: int | None = None


@dataclass(frozen=True)
class ExperimentCase:
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


NETWORK_CONDITIONS = (
    NetworkCondition("clean", delay_ms=0, jitter_ms=0, loss_pct=0.0),
    NetworkCondition("moderate", delay_ms=80, jitter_ms=25, loss_pct=1.0),
    NetworkCondition("severe", delay_ms=200, jitter_ms=75, loss_pct=5.0),
)

VIDEO_TRANSPORTS = (
    VideoTransport("rtsp_tcp", capture_backend="gstreamer", gst_protocol="tcp", gst_latency_ms=50),
    VideoTransport("rtsp_udp", capture_backend="gstreamer", gst_protocol="udp", gst_latency_ms=50),
)

ALERT_TRANSPORTS = (
    AlertTransport("websocket", protocol="websocket"),
    AlertTransport("mqtt_qos0", protocol="mqtt", qos=0),
    AlertTransport("mqtt_qos1", protocol="mqtt", qos=1),
)


def network_apply_command(os_target: str, network: NetworkCondition, interface: str) -> str:
    if os_target.lower() == "linux":
        return network.tc_apply_command(interface)
    if network.name == "clean":
        return "REM no network degradation"
    return (
        "REM configure Windows network emulator "
        f"delay={network.delay_ms}ms jitter={network.jitter_ms}ms loss={network.loss_pct}%"
    )


def network_clear_command(os_target: str, interface: str) -> str:
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
    cases: list[ExperimentCase] = []

    for os_target in os_targets:
        for link_type in link_types:
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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(cases[0].csv_row().keys()) if cases else ["run_id"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(case.csv_row())


def write_cases_markdown(cases: list[ExperimentCase], out_md: Path) -> None:
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
    parser = argparse.ArgumentParser(description="Generate Phase 4 experiment matrix.")
    parser.add_argument("--out-csv", type=Path, default=Path("Phase_4_Network_Latency/phase4_experiment_matrix.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("Phase_4_Network_Latency/phase4_experiment_matrix.md"))
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--python-exe", default="python")
    parser.add_argument("--os-targets", default="linux")
    parser.add_argument("--link-types", default="ethernet")
    parser.add_argument("--duration-min", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
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

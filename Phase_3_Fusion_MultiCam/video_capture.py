"""Video capture helpers for Phase 3.

OpenCV's default RTSP reader can buffer stale frames. For live multi-camera
fusion, GStreamer with appsink/drop=true keeps the newest frame and reduces
timestamp skew between cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Keep this before importing cv2. On Windows, OpenCV only sees GStreamer DLLs
# if they are already on PATH when cv2 is loaded.
_GST_BIN = r"C:\gstreamer\1.0\msvc_x86_64\bin"
if os.path.exists(_GST_BIN):
    os.environ["PATH"] = _GST_BIN + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_GST_BIN)
    except (AttributeError, OSError):
        pass
import cv2


@dataclass(frozen=True)
class CaptureOptions:
    backend: str = "opencv"  # opencv | gstreamer
    gst_latency_ms: int = 100
    gst_protocol: str = "tcp"  # tcp | udp
    gst_codec: str = "h264"  # h264 | h265
    gst_decoder: str = "avdec_h264"
    gst_pipeline: str = "decodebin"  # decodebin | fixed | both
    fallback_ffmpeg: bool = True


@dataclass(frozen=True)
class CaptureOpenResult:
    cap: cv2.VideoCapture
    backend: str
    source: str
    error: str = ""


def build_gstreamer_rtsp_pipeline(
    url: str,
    latency_ms: int = 100,
    protocol: str = "tcp",
    codec: str = "h264",
    decoder: str = "avdec_h264",
) -> str:
    """Build a low-latency RTSP -> BGR appsink pipeline for OpenCV."""
    protocols = "tcp" if protocol.lower() == "tcp" else "udp"
    codec = codec.lower()
    if codec == "h265":
        depay = "rtph265depay"
        parser = "h265parse"
    else:
        depay = "rtph264depay"
        parser = "h264parse"
    return (
        f'rtspsrc location="{url}" latency={latency_ms} protocols={protocols} '
        f"! {depay} "
        f"! {parser} "
        f"! {decoder} "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=true max-buffers=1 sync=false"
    )


def build_gstreamer_decodebin_pipeline(
    url: str,
    latency_ms: int = 100,
    protocol: str = "tcp",
) -> str:
    """Build the tolerant RTSP pipeline used by the crash test."""
    protocols = "tcp" if protocol.lower() == "tcp" else "udp"
    return (
        f'rtspsrc location="{url}" latency={latency_ms} protocols={protocols} '
        "! decodebin "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=true max-buffers=1 sync=false"
    )


def build_rtsp_variants(rtsp_url: str) -> list[str]:
    """Return original URL, then subtype=1 variant when applicable."""
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


def _try_open(source: str, backend: int | None = None) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, backend)


def open_capture_with_info(source: str, live: bool, options: CaptureOptions) -> CaptureOpenResult:
    """Open a video source with GStreamer variants and FFmpeg fallback."""
    if not live:
        cap = cv2.VideoCapture(source)
        return CaptureOpenResult(cap=cap, backend="OPENCV", source=source)

    errors: list[str] = []
    candidates = build_rtsp_variants(source)

    if live and options.backend == "gstreamer":
        for candidate in candidates:
            if options.gst_pipeline in ("decodebin", "both"):
                pipeline = build_gstreamer_decodebin_pipeline(
                    candidate,
                    latency_ms=options.gst_latency_ms,
                    protocol=options.gst_protocol,
                )
                cap = _try_open(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return CaptureOpenResult(cap=cap, backend="GSTREAMER/decodebin", source=candidate)
                cap.release()
                errors.append("GST decodebin KO")

            if options.gst_pipeline in ("fixed", "both"):
                pipeline = build_gstreamer_rtsp_pipeline(
                    candidate,
                    latency_ms=options.gst_latency_ms,
                    protocol=options.gst_protocol,
                    codec=options.gst_codec,
                    decoder=options.gst_decoder,
                )
                cap = _try_open(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return CaptureOpenResult(cap=cap, backend="GSTREAMER/fixed", source=candidate)
                cap.release()
                errors.append("GST fixed KO")

    if options.fallback_ffmpeg or options.backend == "opencv":
        for candidate in candidates:
            cap = _try_open(candidate, cv2.CAP_FFMPEG)
            if cap.isOpened():
                return CaptureOpenResult(cap=cap, backend="FFMPEG", source=candidate)
            cap.release()
            errors.append("FFMPEG KO")

            cap = _try_open(candidate)
            if cap.isOpened():
                return CaptureOpenResult(cap=cap, backend="OPENCV", source=candidate)
            cap.release()
            errors.append("OPENCV KO")

    return CaptureOpenResult(
        cap=cv2.VideoCapture(),
        backend="",
        source=source,
        error=" | ".join(errors[-6:]),
    )


def open_capture(source: str, live: bool, options: CaptureOptions) -> cv2.VideoCapture:
    """Backward-compatible wrapper returning only the capture object."""
    return open_capture_with_info(source, live, options).cap

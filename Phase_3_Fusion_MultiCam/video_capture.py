"""Utilitaires de capture vidéo pour la Phase 3.

Place dans le pipeline : ouvre les sources vidéo (fichiers enregistrés ou
flux RTSP en direct) consommées par pipeline.py.

Le lecteur RTSP par défaut d'OpenCV peut accumuler des frames périmées dans
son tampon. Pour la fusion multi-caméras en direct, un pipeline GStreamer
terminé par appsink drop=true ne conserve que la frame la plus récente, ce
qui réduit la latence et l'écart d'horodatage entre caméras.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# À garder avant l'import de cv2 : sous Windows, OpenCV ne voit les DLL
# GStreamer que si elles sont déjà sur le PATH au chargement de cv2.
# _GST_BIN pointe vers l'installation GStreamer Windows ; si ce chemin
# n'existe pas (Linux, machine sans GStreamer), le bloc est simplement
# ignoré et OpenCV se rabat sur ses backends disponibles.
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
    """Options d'ouverture d'un flux : backend, latence et pipeline GStreamer."""

    backend: str = "opencv"  # opencv | gstreamer
    gst_latency_ms: int = 100
    gst_protocol: str = "tcp"  # tcp | udp
    gst_codec: str = "h264"  # h264 | h265
    gst_decoder: str = "avdec_h264"
    gst_pipeline: str = "decodebin"  # decodebin | fixed | both
    fallback_ffmpeg: bool = True


@dataclass(frozen=True)
class CaptureOpenResult:
    """Résultat d'une ouverture : capture, backend retenu, source et erreurs cumulées."""

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
    """Construit un pipeline RTSP vers BGR appsink à faible latence pour OpenCV.

    Chaîne figée depay/parse/décodeur selon le codec. La terminaison
    appsink drop=true max-buffers=1 sync=false ne conserve que la frame la
    plus récente : c'est le coeur de la stratégie anti-buffering.
    """
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
    """Construit le pipeline RTSP tolérant (decodebin) validé par le crash test.

    decodebin négocie lui-même depay/parse/décodeur, ce qui absorbe les
    variations de codec d'une caméra à l'autre.
    """
    protocols = "tcp" if protocol.lower() == "tcp" else "udp"
    return (
        f'rtspsrc location="{url}" latency={latency_ms} protocols={protocols} '
        "! decodebin "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=true max-buffers=1 sync=false"
    )


def build_rtsp_variants(rtsp_url: str) -> list[str]:
    """Retourne l'URL d'origine, puis sa variante subtype=1 le cas échéant.

    Sur les caméras Dahua, subtype=1 désigne le sub-stream (résolution
    réduite), utile en repli quand le flux principal refuse de s'ouvrir.
    """
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
    """Ouvre une VideoCapture avec le backend demandé (ou celui par défaut)."""
    if backend is None:
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, backend)


def open_capture_with_info(source: str, live: bool, options: CaptureOptions) -> CaptureOpenResult:
    """Ouvre une source vidéo avec variantes GStreamer puis repli FFmpeg/OpenCV.

    En mode enregistré (live=False), une VideoCapture OpenCV standard suffit.
    En mode live, essaie dans l'ordre : pipelines GStreamer (decodebin et/ou
    fixe) sur chaque variante d'URL, puis FFmpeg et OpenCV par défaut si le
    repli est autorisé. Retourne toujours un CaptureOpenResult ; en cas
    d'échec total, cap est une capture fermée et error résume les tentatives.
    """
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
    """Enveloppe rétro-compatible ne retournant que l'objet capture."""
    return open_capture_with_info(source, live, options).cap

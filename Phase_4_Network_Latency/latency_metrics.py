"""Métriques de latence et de synchronisation caméras pour les runs Phase 3/4.

Ce module relit les CSV produits par une campagne Phase 3 (``summary.csv``,
``sync_events.csv``, ``fusion_links.csv``) et en tire des indicateurs de santé
par run : cadence effective, latences d'inférence, activité de la fusion
multi-caméras et décrochages de caméras.

Il est importé par ``analyze_phase4_runs.py`` (agrégation des runs Phase 4)
mais aussi par ``Phase_2_Baseline_MonoCam/evaluate_tad.py`` et
``evaluate_trd.py`` : toute modification de ses signatures impacte donc aussi
les évaluations TAD/TRD.

Conventions d'unités : les durées sont en secondes (suffixe ``_s``), les
latences en millisecondes (suffixe ``_ms``), les cadences en images par
seconde (fps) et les taux en pourcentage (suffixe ``_pct``) ou en fraction
de 0 à 1 (suffixe ``_rate``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from collections import Counter


@dataclass(frozen=True)
class CameraSyncStats:
    """Statistiques de synchronisation d'une caméra sur un run.

    Attributs :
        cam_id : identifiant de la caméra (ex. ``cam_02``) ;
        frames : nombre de trames de campagne où la caméra a fourni une image ;
        first_elapsed_s : instant (en s depuis le début du run) de sa première trame ;
        last_elapsed_s : instant (en s) de sa dernière trame ;
        missing_campaign_frames : nombre de trames de campagne où la caméra est absente ;
        effective_fps : cadence effective de la caméra (trames / durée de présence, en fps) ;
        dropped_early : True si la caméra a décroché (dernière trame plus d'une
            seconde avant la fin du run, ou au moins une trame manquante).
    """

    cam_id: str
    frames: int
    first_elapsed_s: float
    last_elapsed_s: float
    missing_campaign_frames: int
    effective_fps: float
    dropped_early: bool


@dataclass(frozen=True)
class RunHealth:
    """Bilan de santé d'un run de campagne Phase 3.

    Attributs :
        campaign_frames : nombre de trames de campagne annoncé par ``summary.csv`` ;
        duration_s : durée du run en secondes (de la première trame reçue à la dernière) ;
        effective_fps : cadence globale effective (campaign_frames / duration_s, en fps) ;
        detections : nombre total de détections du run ;
        latency_mean_ms / latency_p95_ms / latency_max_ms : latence d'inférence
            par trame (moyenne, 95e percentile, maximum), en millisecondes,
            reprise de ``summary.csv`` ;
        fusion_links : nombre total d'appariements inter-caméras (liens de fusion) ;
        frames_with_fusion : nombre de trames distinctes comportant au moins un lien ;
        fusion_frame_rate_pct : part des trames avec fusion, en % des trames de campagne ;
        fusion_links_per_frame : liens de fusion par trame de campagne ;
        fusion_links_per_1000_detections : liens de fusion pour 1000 détections
            (mesure d'activité indépendante de la densité de détections) ;
        fusion_camera_pairs : nombre de paires de caméras distinctes appariées ;
        top_fusion_pair : paire la plus fréquente, au format ``cam_a+cam_b`` ;
        top_fusion_pair_links : nombre de liens de cette paire ;
        alerts : nombre total d'alertes émises pendant le run ;
        cameras : statistiques de synchronisation par caméra (:class:`CameraSyncStats`).
    """

    campaign_frames: int
    duration_s: float
    effective_fps: float
    detections: int
    latency_mean_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    fusion_links: int
    frames_with_fusion: int
    fusion_frame_rate_pct: float
    fusion_links_per_frame: float
    fusion_links_per_1000_detections: float
    fusion_camera_pairs: int
    top_fusion_pair: str
    top_fusion_pair_links: int
    alerts: int
    cameras: tuple[CameraSyncStats, ...]

    @property
    def has_fusion(self) -> bool:
        """True si le run comporte au moins un lien de fusion inter-caméras."""
        return self.fusion_links > 0

    @property
    def dropped_cameras(self) -> tuple[str, ...]:
        """Identifiants des caméras ayant décroché pendant le run (``dropped_early``)."""
        return tuple(cam.cam_id for cam in self.cameras if cam.dropped_early)


def _float(value: str | None, default: float = 0.0) -> float:
    """Convertit une cellule CSV en float, avec valeur par défaut pour None ou chaîne vide."""
    if value is None or value == "":
        return default
    return float(value)


def _int(value: str | None, default: int = 0) -> int:
    """Convertit une cellule CSV en int (en passant par float pour accepter ``"3.0"``)."""
    if value is None or value == "":
        return default
    return int(float(value))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Lit un CSV en liste de dictionnaires (encodage utf-8-sig pour absorber un éventuel BOM)."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class FusionLinkStats:
    """Statistiques agrégées des liens de fusion d'un run.

    Attributs :
        frames_with_fusion : nombre de trames distinctes avec au moins un lien ;
        camera_pairs : nombre de paires de caméras distinctes appariées ;
        top_pair : paire la plus fréquente au format ``cam_a+cam_b`` (chaîne vide si aucune) ;
        top_pair_links : nombre de liens portés par cette paire.
    """

    frames_with_fusion: int
    camera_pairs: int
    top_pair: str
    top_pair_links: int


def analyze_fusion_links(fusion_links_csv: Path) -> FusionLinkStats:
    """Analyse ``fusion_links.csv`` (un lien inter-caméras par ligne).

    Compte les trames distinctes avec fusion (et non les liens : plusieurs
    liens peuvent partager une trame) et recense les paires de caméras
    appariées, ordonnées alphabétiquement pour que (a, b) et (b, a) soient
    confondues. Retourne des statistiques nulles si le fichier est absent
    (run sans fusion).
    """
    if not fusion_links_csv.exists():
        return FusionLinkStats(0, 0, "", 0)

    rows = read_csv_rows(fusion_links_csv)
    frames = {_int(row.get("frame")) for row in rows}
    camera_pairs: Counter[tuple[str, str]] = Counter()

    for row in rows:
        cam_a = row.get("cam_a", "")
        cam_b = row.get("cam_b", "")
        if cam_a and cam_b:
            camera_pairs[tuple(sorted((cam_a, cam_b)))] += 1

    if camera_pairs:
        top_pair_tuple, top_pair_links = camera_pairs.most_common(1)[0]
        top_pair = "+".join(top_pair_tuple)
    else:
        top_pair = ""
        top_pair_links = 0

    return FusionLinkStats(
        frames_with_fusion=len(frames),
        camera_pairs=len(camera_pairs),
        top_pair=top_pair,
        top_pair_links=top_pair_links,
    )


def analyze_sync_events(sync_events_csv: Path, expected_cameras: list[str] | None = None) -> tuple[CameraSyncStats, ...]:
    """Analyse ``sync_events.csv`` (une ligne par trame reçue et par caméra).

    Calcule pour chaque caméra sa présence trame par trame, sa cadence
    effective (fps) et son éventuel décrochage : dernière trame plus d'une
    seconde avant la fin du run, ou au moins une trame de campagne manquante.
    Si ``expected_cameras`` est fourni, les caméras attendues mais absentes du
    CSV sont rapportées avec zéro trame et ``dropped_early=True``. Retourne un
    tuple vide si le CSV est vide.
    """
    rows = read_csv_rows(sync_events_csv)
    if not rows:
        return ()

    campaign_frames = {_int(row.get("campaign_frame")) for row in rows}
    max_campaign_frame = max(campaign_frames)
    total_campaign_frames = max_campaign_frame + 1
    run_last_elapsed_s = max(_float(row.get("elapsed_s")) for row in rows)
    cameras = expected_cameras or sorted({row["cam_id"] for row in rows})
    stats: list[CameraSyncStats] = []

    for cam_id in cameras:
        cam_rows = [row for row in rows if row.get("cam_id") == cam_id]
        if not cam_rows:
            stats.append(CameraSyncStats(cam_id, 0, 0.0, 0.0, total_campaign_frames, 0.0, True))
            continue

        elapsed = [_float(row.get("elapsed_s")) for row in cam_rows]
        present_frames = {_int(row.get("campaign_frame")) for row in cam_rows}
        duration = max(elapsed) - min(elapsed)
        effective_fps = len(cam_rows) / duration if duration > 0 else 0.0
        missing = total_campaign_frames - len(present_frames)
        dropped_early = max(elapsed) < run_last_elapsed_s - 1.0 or missing > 0

        stats.append(
            CameraSyncStats(
                cam_id=cam_id,
                frames=len(cam_rows),
                first_elapsed_s=min(elapsed),
                last_elapsed_s=max(elapsed),
                missing_campaign_frames=missing,
                effective_fps=effective_fps,
                dropped_early=dropped_early,
            )
        )

    return tuple(stats)


def summarize_phase3_run(run_dir: Path, expected_cameras: list[str] | None = None) -> RunHealth:
    """Construit le :class:`RunHealth` d'un run à partir de son dossier ``<run_dir>/phase3``.

    Combine ``summary.csv`` (première ligne : trames, détections, latences,
    alertes), ``sync_events.csv`` (présence par caméra) et ``fusion_links.csv``
    (activité de la fusion). La durée du run est déduite des instants extrêmes
    observés dans les événements de synchronisation ; les taux de fusion sont
    normalisés par trame et pour 1000 détections. Lève ValueError si
    ``summary.csv`` est vide.
    """
    phase3_dir = run_dir / "phase3"
    summary_rows = read_csv_rows(phase3_dir / "summary.csv")
    if not summary_rows:
        raise ValueError(f"No rows found in {phase3_dir / 'summary.csv'}")

    summary = summary_rows[0]
    sync_stats = analyze_sync_events(phase3_dir / "sync_events.csv", expected_cameras=expected_cameras)
    fusion_stats = analyze_fusion_links(phase3_dir / "fusion_links.csv")
    campaign_frames = _int(summary.get("frames"))
    detections = _int(summary.get("detections"))
    fusion_links = _int(summary.get("fusion_links"))
    duration_s = max((cam.last_elapsed_s for cam in sync_stats), default=0.0) - min(
        (cam.first_elapsed_s for cam in sync_stats if cam.frames > 0),
        default=0.0,
    )
    effective_fps = campaign_frames / duration_s if duration_s > 0 else 0.0

    return RunHealth(
        campaign_frames=campaign_frames,
        duration_s=duration_s,
        effective_fps=effective_fps,
        detections=detections,
        latency_mean_ms=_float(summary.get("latency_mean_ms")),
        latency_p95_ms=_float(summary.get("latency_p95_ms")),
        latency_max_ms=_float(summary.get("latency_max_ms")),
        fusion_links=fusion_links,
        frames_with_fusion=fusion_stats.frames_with_fusion,
        fusion_frame_rate_pct=(fusion_stats.frames_with_fusion / campaign_frames * 100.0) if campaign_frames else 0.0,
        fusion_links_per_frame=(fusion_links / campaign_frames) if campaign_frames else 0.0,
        fusion_links_per_1000_detections=(fusion_links / detections * 1000.0) if detections else 0.0,
        fusion_camera_pairs=fusion_stats.camera_pairs,
        top_fusion_pair=fusion_stats.top_pair,
        top_fusion_pair_links=fusion_stats.top_pair_links,
        alerts=_int(summary.get("alerts")),
        cameras=sync_stats,
    )


def compare_runs(run_health: list[RunHealth]) -> dict[str, float]:
    """Agrège plusieurs :class:`RunHealth` en indicateurs de comparaison.

    Retourne : ``runs`` (nombre de runs), ``mean_effective_fps`` (moyenne des
    cadences effectives, en fps), ``mean_latency_p95_ms`` (moyenne des p95 de
    latence, en ms), ``drop_rate`` (fraction 0-1 de caméras ayant décroché,
    toutes caméras de tous les runs confondues) et ``fusion_enabled_rate``
    (fraction 0-1 des runs avec au moins un lien de fusion). Toutes les valeurs
    sont nulles si la liste est vide.
    """
    if not run_health:
        return {
            "runs": 0.0,
            "mean_effective_fps": 0.0,
            "mean_latency_p95_ms": 0.0,
            "drop_rate": 0.0,
            "fusion_enabled_rate": 0.0,
        }

    camera_count = sum(len(run.cameras) for run in run_health)
    dropped_count = sum(len(run.dropped_cameras) for run in run_health)
    return {
        "runs": float(len(run_health)),
        "mean_effective_fps": mean(run.effective_fps for run in run_health),
        "mean_latency_p95_ms": mean(run.latency_p95_ms for run in run_health),
        "drop_rate": dropped_count / camera_count if camera_count else 0.0,
        "fusion_enabled_rate": sum(1 for run in run_health if run.has_fusion) / len(run_health),
    }

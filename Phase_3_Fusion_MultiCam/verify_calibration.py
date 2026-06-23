"""
Vérification visuelle de la calibration homographique.
Projette la zone interdite (en mètres) sur la vidéo caméra via l'homographie inverse.

Usage:
    python verify_calibration.py              # utilise cam_01 par défaut
    python verify_calibration.py --camera cam_03
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".m4v")


def _res_matches(a, b, tol=2):
    if not a or not b or len(a) != 2 or len(b) != 2:
        return False
    return abs(int(a[0]) - int(b[0])) <= tol and abs(int(a[1]) - int(b[1])) <= tol


def _convert_h_to_resolution(H, src_res, dst_res):
    """Adapte une homographie pixels->metres d'une resolution src vers dst.

    Si H mappe des pixels src vers les metres, alors pour des pixels dst :
      p_src = S * p_dst  =>  H_dst = H_src @ S
    """
    sw, sh = int(src_res[0]), int(src_res[1])
    dw, dh = int(dst_res[0]), int(dst_res[1])
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return H

    sx = sw / dw
    sy = sh / dh
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    H2 = np.asarray(H, dtype=np.float32) @ S
    if abs(float(H2[2, 2])) > 1e-9:
        H2 = H2 / H2[2, 2]
    return H2


def _res_distance(a, b):
    if not a or not b or len(a) != 2 or len(b) != 2:
        return float("inf")
    aw, ah = float(a[0]), float(a[1])
    bw, bh = float(b[0]), float(b[1])
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return float("inf")
    return abs(np.log(aw / bw)) + abs(np.log(ah / bh))


def fix_aspect_ratio(frame: np.ndarray, cam_id: str, config: dict) -> np.ndarray:
    """Correctif ratio d'aspect pour caméras Dahua dont le sub-stream 704x576
    est déformé horizontalement par le firmware. Redimensionne en 768x576 (4:3)
    pour restaurer la géométrie réelle avant toute projection homographique."""
    ar_fix = config.get("aspect_ratio_fix", {})
    if not ar_fix.get("enabled", False):
        return frame
    if cam_id not in ar_fix.get("distorted_cameras", []):
        return frame
    h, w = frame.shape[:2]
    target_w, target_h = ar_fix["corrected_resolution"]
    distorted_w, distorted_h = ar_fix.get("distorted_resolution", [704, 576])

    # N'appliquer le correctif que si la frame correspond vraiment au sub-stream deforme.
    if not (w == distorted_w and h == distorted_h):
        return frame

    if w == target_w and h == target_h:
        return frame
    return cv2.resize(frame, (target_w, target_h))


def resolve_video_source(raw_source: str, config_path: Path) -> str:
    """Resolve les chemins locaux en absolu pour eviter les erreurs backend GStreamer."""
    if not raw_source:
        return raw_source
    if "://" in raw_source:
        return raw_source

    source_path = Path(raw_source)
    candidates = [
        source_path,
        config_path.parent / source_path,
        config_path.parent.parent / source_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((config_path.parent / source_path).resolve())


def _extract_cam_number(cam_id: str, cam_cfg: dict):
    name = str(cam_cfg.get("name", ""))
    m = re.search(r"Camera\s+(\d+)", name)
    if m:
        return int(m.group(1))

    m = re.search(r"cam_(\d+)", cam_id)
    if m:
        return int(m.group(1))
    return None


def _extract_cam_suffix(cam_cfg: dict):
    # Ex: <CAMERA_IP> -> 2.11
    ip = str(cam_cfg.get("ip", "")).strip()
    parts = ip.split(".")
    if len(parts) == 4 and parts[-2].isdigit() and parts[-1].isdigit():
        return f"{parts[-2]}.{parts[-1]}"

    name = str(cam_cfg.get("name", ""))
    m = re.search(r"\((\d+\.\d+)\)", name)
    if m:
        return m.group(1)
    return None


def _hhmmss_to_seconds(hhmmss: str):
    if not hhmmss or len(hhmmss) != 6 or not hhmmss.isdigit():
        return None
    h = int(hhmmss[0:2])
    m = int(hhmmss[2:4])
    s = int(hhmmss[4:6])
    return h * 3600 + m * 60 + s


def _find_preferred_local_video(project_root: Path, cam_id: str, cam_cfg: dict, prefer_date: str, prefer_time: str):
    rec_dir = project_root / "recordings" / "recordings"
    if not rec_dir.is_dir():
        return None

    cam_num = _extract_cam_number(cam_id, cam_cfg)
    cam_suffix = _extract_cam_suffix(cam_cfg)
    if cam_num is None or cam_suffix is None:
        return None

    # 1) Exact match: Camera_x_2.x_YYYYMMDD_HHMMSS
    for ext in VIDEO_EXTENSIONS:
        p = rec_dir / f"Camera_{cam_num}_{cam_suffix}_{prefer_date}_{prefer_time}{ext}"
        if p.is_file():
            return str(p.resolve())

    # 2) Same date fallback with closest HHMMSS
    same_day = []
    pattern = str(rec_dir / f"Camera_{cam_num}_{cam_suffix}_{prefer_date}_*")
    for p in glob.glob(pattern):
        ext = Path(p).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            continue

        base = Path(p).name
        m = re.match(
            rf"^Camera_{cam_num}_{re.escape(cam_suffix)}_{prefer_date}_(\d{{6}})",
            base,
            flags=re.IGNORECASE,
        )
        if m:
            same_day.append((p, m.group(1)))

    if not same_day:
        return None

    target_sec = _hhmmss_to_seconds(prefer_time)
    if target_sec is None:
        same_day.sort(key=lambda it: os.path.getmtime(it[0]), reverse=True)
        return str(Path(same_day[0][0]).resolve())

    def sort_key(item):
        p, t = item
        ts = _hhmmss_to_seconds(t)
        if ts is None:
            return (10**9, -os.path.getmtime(p))
        return (abs(ts - target_sec), -os.path.getmtime(p))

    same_day.sort(key=sort_key)
    return str(Path(same_day[0][0]).resolve())


def build_source_candidates(
    cam_id: str,
    cam_cfg: dict,
    config: dict,
    config_path: Path,
    source_mode: str,
    prefer_date: str,
    prefer_time: str,
):
    project_root = config_path.parent.parent
    candidates = []

    rtsp_url = cam_cfg.get("rtsp_url")
    if rtsp_url:
        candidates.append(rtsp_url)

    preferred_video = _find_preferred_local_video(project_root, cam_id, cam_cfg, prefer_date, prefer_time)
    if preferred_video:
        candidates.append(preferred_video)

    cam_h = config.get("homographie", {}).get(cam_id, {})
    if isinstance(cam_h, dict):
        calib_src = cam_h.get("calibration_source")
        if calib_src:
            if os.path.isabs(calib_src):
                candidates.append(calib_src)
            else:
                rec_candidate = project_root / "recordings" / "recordings" / calib_src
                if rec_candidate.is_file():
                    candidates.append(str(rec_candidate.resolve()))
                else:
                    candidates.append(resolve_video_source(calib_src, config_path))

    raw_video_path = cam_cfg.get("video_path")
    if raw_video_path:
        candidates.append(resolve_video_source(raw_video_path, config_path))

    # Reorder by source mode preference.
    if source_mode == "rtsp":
        ordered = [c for c in candidates if "://" in c] + [c for c in candidates if "://" not in c]
    elif source_mode == "video":
        ordered = [c for c in candidates if "://" not in c] + [c for c in candidates if "://" in c]
    else:
        ordered = candidates  # auto

    dedup = []
    seen = set()
    for c in ordered:
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        dedup.append(c)
    return dedup


def open_first_working_source(candidates):
    for src in candidates:
        is_local_file = "://" not in src
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if is_local_file else cv2.VideoCapture(src)
        if is_local_file and not cap.isOpened():
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"[WARN] Source non ouverte: {src}")
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Source ouverte mais frame illisible: {src}")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return src, cap

    return None, None


def resolve_zone(config: dict, cam_id: str, zone_arg: str):
    zones = config.get("zones_interdites", {})

    if zone_arg != "auto":
        zone = zones.get(zone_arg)
        if not zone:
            sys.exit(f"ERREUR: Zone '{zone_arg}' non trouvée dans config.yaml.")
        coords = zone.get("coordonnees_metres")
        if not coords:
            sys.exit(f"ERREUR: Zone '{zone_arg}' est vide (coordonnees_metres null/vide).")
        return zone_arg, zone

    # Auto: prendre la zone où la caméra est explicitement listée.
    candidates = []
    for zone_id, zone_data in zones.items():
        coords = zone_data.get("coordonnees_metres")
        cams = zone_data.get("cameras_concernees") or []
        if coords and cam_id in cams:
            candidates.append((zone_id, zone_data))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"[WARN] Plusieurs zones correspondent à {cam_id}: {[z[0] for z in candidates]}")
        print("       Utilisation de la première. Vous pouvez forcer avec --zone <id>.")
        return candidates[0]

    details = []
    for zone_id, zone_data in zones.items():
        cams = zone_data.get("cameras_concernees")
        coords = zone_data.get("coordonnees_metres")
        details.append(f"  - {zone_id}: cameras={cams}, coords={'OK' if coords else 'VIDE'}")
    sys.exit(
        f"ERREUR: Aucune zone auto-valide pour {cam_id}.\n"
        "Définissez cameras_concernees/coordonnees_metres ou utilisez --zone explicitement.\n"
        + "\n".join(details)
    )


def resolve_zones(config: dict, cam_id: str, zone_arg: str):
    zones = config.get("zones_interdites", {})

    if zone_arg in {"all", "*"}:
        candidates = []
        for zone_id, zone_data in zones.items():
            coords = zone_data.get("coordonnees_metres")
            cams = zone_data.get("cameras_concernees") or []
            if coords and (not cams or cam_id in cams):
                candidates.append((zone_id, zone_data))
        if candidates:
            return candidates
        details = []
        for zone_id, zone_data in zones.items():
            cams = zone_data.get("cameras_concernees")
            coords = zone_data.get("coordonnees_metres")
            details.append(f"  - {zone_id}: cameras={cams}, coords={'OK' if coords else 'VIDE'}")
        sys.exit(
            f"ERREUR: Aucune zone affichable pour {cam_id} avec --zone all.\n"
            + "\n".join(details)
        )

    return [resolve_zone(config, cam_id, zone_arg)]


def select_homography(config: dict, cam_id: str, effective_res):
    hom = config.get("homographie", {})
    cam_data = hom.get(cam_id, {}) if isinstance(hom.get(cam_id, {}), dict) else {}

    matrix_sub = hom.get(f"{cam_id}_matrix") or cam_data.get("matrix")
    matrix_hd = cam_data.get("matrix_hd")
    hd_res = cam_data.get("hd_resolution")
    sub_res = cam_data.get("substream_resolution")

    if not matrix_sub and not matrix_hd:
        sys.exit(f"ERREUR: Aucune matrice trouvée pour {cam_id} dans config.yaml.")

    # Priorité: matrice dont la résolution de calibration match la frame réelle.
    if matrix_hd and _res_matches(effective_res, hd_res):
        return np.array(matrix_hd, dtype=np.float32), "matrix_hd", True, cam_data
    if matrix_sub and _res_matches(effective_res, sub_res):
        return np.array(matrix_sub, dtype=np.float32), f"{cam_id}_matrix", False, cam_data

    # Si pas de match exact, convertir depuis la resolution la plus proche.
    candidates = []
    if matrix_hd and hd_res:
        candidates.append((
            _res_distance(effective_res, hd_res),
            np.array(matrix_hd, dtype=np.float32),
            hd_res,
            "matrix_hd",
            True,
        ))
    if matrix_sub and sub_res:
        candidates.append((
            _res_distance(effective_res, sub_res),
            np.array(matrix_sub, dtype=np.float32),
            sub_res,
            f"{cam_id}_matrix",
            False,
        ))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, H_base, src_res, base_name, base_is_hd = candidates[0]
        H_scaled = _convert_h_to_resolution(H_base, src_res, effective_res)
        print(
            f"[WARN] Résolution frame {effective_res[0]}x{effective_res[1]} ne matche pas exactement "
            f"les résolutions calibrées. Conversion de {base_name} ({src_res[0]}x{src_res[1]}) "
            f"-> {effective_res[0]}x{effective_res[1]}."
        )
        return H_scaled, f"{base_name}_scaled", base_is_hd, cam_data

    # Fallback sûr: matrice substream (utilisée en prod), sinon HD.
    if matrix_sub:
        print(
            f"[WARN] Résolution frame {effective_res[0]}x{effective_res[1]} ne correspond pas exactement "
            f"à hd={hd_res} ni sub={sub_res}. Fallback sur matrice substream."
        )
        return np.array(matrix_sub, dtype=np.float32), f"{cam_id}_matrix", False, cam_data

    print(
        f"[WARN] Résolution frame {effective_res[0]}x{effective_res[1]} ne correspond pas exactement "
        f"à hd={hd_res}. Fallback sur matrice HD."
    )
    return np.array(matrix_hd, dtype=np.float32), "matrix_hd", True, cam_data


def get_display_src_points(calib_data: dict, use_hd_matrix: bool):
    if not calib_data:
        return None

    if use_hd_matrix:
        pts_hd = calib_data.get("src_points_px_hd")
        if pts_hd:
            return np.array(pts_hd, dtype=np.int32)
        pts_legacy = calib_data.get("src_points_px")
        if pts_legacy:
            return np.array(pts_legacy, dtype=np.int32)
        return None

    # Matrice substream: utiliser points legacy si présents, sinon conversion HD -> substream.
    pts_legacy = calib_data.get("src_points_px")
    if pts_legacy:
        return np.array(pts_legacy, dtype=np.int32)

    pts_hd = calib_data.get("src_points_px_hd")
    hd_res = calib_data.get("hd_resolution")
    sub_res = calib_data.get("substream_resolution")
    if pts_hd and hd_res and sub_res and len(hd_res) == 2 and len(sub_res) == 2:
        sx = float(hd_res[0]) / float(sub_res[0])
        sy = float(hd_res[1]) / float(sub_res[1])
        pts_sub = np.array(pts_hd, dtype=np.float32) / np.array([sx, sy], dtype=np.float32)
        return pts_sub.astype(np.int32)

    return None


def main():
    parser = argparse.ArgumentParser(description="Vérification visuelle de la calibration")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Fichier de configuration à lire (défaut: config.yaml)",
    )
    parser.add_argument("--camera", default="cam_01", help="Identifiant caméra (ex: cam_01, cam_03)")
    parser.add_argument("--zone", default="auto", help="Zone interdite à projeter ('auto', 'all' ou un id)")
    parser.add_argument("--source", choices=["auto", "rtsp", "video"], default="auto",
                        help="Source à privilégier: auto, rtsp ou video")
    parser.add_argument("--prefer-date", default="20260420", help="Date préférée des vidéos locales (AAAAMMJJ)")
    parser.add_argument("--prefer-time", default="103914", help="Heure préférée des vidéos locales (HHMMSS)")
    args = parser.parse_args()

    # --- Chargement config ---
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- Ouverture vidéo ---
    if args.camera not in config.get("cameras", {}):
        sys.exit(f"ERREUR: Camera '{args.camera}' inconnue dans config.yaml.")

    cam_config = config["cameras"][args.camera]
    candidates = build_source_candidates(
        args.camera,
        cam_config,
        config,
        config_path,
        args.source,
        args.prefer_date,
        args.prefer_time,
    )
    if not candidates:
        sys.exit("ERREUR: Aucune source vidéo candidate disponible.")

    print("Sources candidates (ordre de priorité):")
    for i, src in enumerate(candidates, start=1):
        print(f"  {i}. {src}")

    video_source, cap = open_first_working_source(candidates)
    if cap is None:
        sys.exit("ERREUR: Impossible d'ouvrir une source RTSP/vidéo valide.")

    print(f"Source utilisée : {video_source}")

    # Lire une frame pour déterminer la résolution réellement utilisée après fix ratio.
    ret, first_frame = cap.read()
    if not ret:
        print("ERREUR: Impossible de lire la première frame.")
        sys.exit(1)

    first_frame_fixed = fix_aspect_ratio(first_frame, args.camera, config)
    raw_h, raw_w = first_frame.shape[:2]
    eff_h, eff_w = first_frame_fixed.shape[:2]

    # Revenir au début pour la boucle vidéo.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"Résolution source : {raw_w}x{raw_h}")
    if (raw_w, raw_h) != (eff_w, eff_h):
        print(f"Résolution après fix ratio : {eff_w}x{eff_h}")

    # --- Zones en mètres ---
    zones_to_draw = resolve_zones(config, args.camera, args.zone)
    print("Zones utilisées :")
    for zone_id, zone_data in zones_to_draw:
        print(f"  - {zone_id} ({zone_data['nom']})")

    # --- Matrice d'homographie (auto selon résolution réelle) ---
    h_matrix, matrix_name, use_hd_matrix, calib_data = select_homography(config, args.camera, (eff_w, eff_h))
    print(f"Matrice utilisée : {matrix_name} ({'HD' if use_hd_matrix else 'SUBSTREAM'})")

    # --- Homographie inverse : mètres -> pixels caméra ---
    try:
        h_inv = np.linalg.inv(h_matrix)
    except np.linalg.LinAlgError:
        sys.exit(f"ERREUR: Matrice {matrix_name} non inversible.")

    projected_zones = []
    for zone_id, zone_data in zones_to_draw:
        coords_metres = zone_data["coordonnees_metres"]
        pts_metres = np.array(coords_metres, dtype=np.float32).reshape(-1, 1, 2)
        pixels_cam = cv2.perspectiveTransform(pts_metres, h_inv).astype(np.int32)
        projected_zones.append((zone_id, zone_data, pixels_cam))
        print(f"Points projetés en pixels ({zone_id}) :\n{pixels_cam.reshape(-1, 2)}")

    # --- Diagnostic : reprojection des points de calibration ---
    src_pts_for_diag = get_display_src_points(calib_data, use_hd_matrix)
    dst_pts_m = calib_data.get("dst_points_metres") if isinstance(calib_data, dict) else None
    if src_pts_for_diag is not None and dst_pts_m:
        print("\n--- DIAGNOSTIC REPROJECTION ---")
        src_px_orig = src_pts_for_diag.astype(np.float32).reshape(-1, 1, 2)
        dst_m_orig = np.array(dst_pts_m, dtype=np.float32).reshape(-1, 1, 2)

        dst_m_calc = cv2.perspectiveTransform(src_px_orig, h_matrix)
        print("Aller (px -> m) :")
        for i in range(len(src_px_orig)):
            orig = dst_m_orig[i][0]
            calc = dst_m_calc[i][0]
            err = np.linalg.norm(orig - calc)
            print(
                f"  P{i+1}: attendu ({orig[0]:.3f}, {orig[1]:.3f}) | "
                f"calculé ({calc[0]:.3f}, {calc[1]:.3f}) | erreur: {err:.4f} m"
            )

        src_px_calc = cv2.perspectiveTransform(dst_m_orig, h_inv)
        print("Retour (m -> px) :")
        for i in range(len(dst_m_orig)):
            orig = src_px_orig[i][0]
            calc = src_px_calc[i][0]
            err = np.linalg.norm(orig - calc)
            print(
                f"  P{i+1}: attendu ({orig[0]:.0f}, {orig[1]:.0f}) | "
                f"calculé ({calc[0]:.0f}, {calc[1]:.0f}) | erreur: {err:.1f} px"
            )
        print("--- FIN DIAGNOSTIC ---\n")

    # Points de calibration originaux pour affichage
    calib_src_pts = src_pts_for_diag

    window_name = f"Verification Calibration - {cam_config['name']}"
    print(f"Appuyez sur 'q' pour quitter.")

    # --- Info fix ratio ---
    ar_fix = config.get("aspect_ratio_fix", {})
    if ar_fix.get("enabled") and args.camera in ar_fix.get("distorted_cameras", []):
        target = ar_fix["corrected_resolution"]
        distorted = ar_fix.get("distorted_resolution", [704, 576])
        print(
            f"[FIX RATIO] {args.camera} : mode auto {distorted[0]}x{distorted[1]} -> {target[0]}x{target[1]}"
        )

    # --- Boucle vidéo ---
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Correctif ratio d'aspect (AVANT tout dessin ou projection)
        frame = fix_aspect_ratio(frame, args.camera, config)

        # Dessiner les zones interdites.
        colors = [(0, 0, 255), (0, 140, 255), (255, 0, 255), (255, 120, 0), (0, 200, 200)]
        for idx, (zone_id, zone_data, pixels_cam) in enumerate(projected_zones):
            color = colors[idx % len(colors)]
            cv2.polylines(frame, [pixels_cam], isClosed=True, color=color, thickness=3)

            pt_text = tuple(pixels_cam[0][0])
            cv2.putText(
                frame,
                zone_data["nom"],
                (pt_text[0] + 10, pt_text[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Dessiner les points de calibration originaux (vert) pour référence
        if calib_src_pts is not None:
            for i, pt in enumerate(calib_src_pts):
                cv2.circle(frame, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"C{i+1}", (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


import argparse
from collections import OrderedDict
from pathlib import Path
import sys
import time

import cv2
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ZONES_JSON_PATH = SCRIPT_DIR / "zones_config.json"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".m4v"}

KEY_SPACE = 32
KEY_ESC = 27
KEY_Q = ord("q")
KEY_LEFT = 81
KEY_RIGHT = 83
KEY_LEFT_WIN = 2424832
KEY_RIGHT_WIN = 2555904


def detect_device() -> tuple[str, bool]:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU] {gpu_name} ({vram_gb:.1f} Go VRAM)")
        print(f"[GPU] CUDA {torch.version.cuda}")
        return "cuda:0", True
    print("[CPU] Aucun GPU CUDA detecte, fallback CPU.")
    return "cpu", False


def parse_classes(value: str | None) -> list[int] | None:
    if not value:
        return None
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out if out else None


def list_videos(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        return [source]

    if source.is_dir():
        videos = [p for p in sorted(source.iterdir()) if p.suffix.lower() in VIDEO_EXTENSIONS]
        return videos

    return []


def choose_video(videos: list[Path]) -> Path | None:
    if not videos:
        return None
    if len(videos) == 1:
        return videos[0]

    print("\nVideos disponibles:")
    for i, v in enumerate(videos, 1):
        print(f"  {i:2d}. {v.name}")
    print("   0. Quitter")

    while True:
        raw = input("Choix [numero]: ").strip()
        if raw == "0":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(videos):
                return videos[idx]
        except ValueError:
            pass
        print("Choix invalide.")


class FrameCache:
    """Petit cache LRU pour accelerer les aller-retours sur la timeline."""

    def __init__(self, maxsize: int = 300):
        self._cache: OrderedDict[int, tuple] = OrderedDict()
        self._maxsize = maxsize

    def get(self, idx: int):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        return None

    def put(self, idx: int, value):
        if idx in self._cache:
            self._cache.move_to_end(idx)
        self._cache[idx] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


def load_model(model_path: Path, model_type: str):
    model_type = model_type.lower().strip()
    if model_type == "auto":
        if "rtdetr" in model_path.stem.lower():
            model_type = "rtdetr"
        else:
            model_type = "yolo"

    if model_type == "rtdetr":
        from ultralytics import RTDETR
        model = RTDETR(str(model_path))
        return model, "RTDETR"

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    return model, "YOLO"


def load_zone_polygon(camera_id: str,
                      zones_json_path: Path,
                      config_path: Path):
    # Priorite : zones_config.json (Phase_2)
    if zones_json_path.exists():
        try:
            import json
            with open(zones_json_path, "r", encoding="utf-8") as f:
                zones = json.load(f) or {}
            zone = zones.get(camera_id)
            if zone and len(zone) >= 3:
                print(f"[ZONE]   source: {zones_json_path.name}")
                return __import__("numpy").array(zone, dtype=__import__("numpy").int32)
        except Exception as e:
            print(f"[ATTENTION] Lecture {zones_json_path} impossible: {e}")

    # Fallback : config YAML
    if not config_path.exists():
        print(f"[ERREUR] Config introuvable: {config_path}")
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    cameras = config.get("cameras", [])
    if isinstance(cameras, dict):
        cam_cfg = cameras.get(camera_id)
    else:
        cam_cfg = next((c for c in cameras if c.get("camera_id") == camera_id), None)

    if not cam_cfg:
        print(f"[ATTENTION] Camera '{camera_id}' absente du config: pas de zone affichee.")
        return None

    zone = cam_cfg.get("forbidden_zone_pixels")

    # Fallback Phase_3_Fusion_MultiCam: homographie.cam_xx.src_points_px
    if (not zone or len(zone) < 3) and isinstance(config.get("homographie"), dict):
        homographie = config.get("homographie", {})
        cam_h = homographie.get(camera_id, {})
        if isinstance(cam_h, dict):
            zone = cam_h.get("src_points_px")

    if not zone or len(zone) < 3:
        print(
            f"[ATTENTION] Pas de zone pixel exploitable pour '{camera_id}' "
            f"(forbidden_zone_pixels ou homographie.{camera_id}.src_points_px)."
        )
        return None

    print(f"[ZONE]   source: {config_path.name}")
    return __import__("numpy").array(zone, dtype=__import__("numpy").int32)


def run_predict(model, frame, conf: float, classes: list[int] | None,
                device: str, imgsz: int, use_half: bool,
                zone_polygon=None):
    t0 = time.monotonic()
    results = model.predict(
        frame,
        conf=conf,
        classes=classes,
        device=device,
        imgsz=imgsz,
        half=use_half,
        verbose=False,
    )
    infer_ms = (time.monotonic() - t0) * 1000.0
    boxes = results[0].boxes
    n_boxes = len(boxes) if boxes is not None else 0
    annotated = frame.copy()

    if zone_polygon is not None:
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [zone_polygon], (0, 140, 255))
        cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
        cv2.polylines(annotated, [zone_polygon], True, (0, 140, 255), 2)

    model_names = getattr(model, "names", {})
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            conf_det = float(boxes.conf[i].cpu()) if boxes.conf is not None else 0.0
            cls_id = int(boxes.cls[i].cpu()) if boxes.cls is not None else -1
            cls_name = model_names.get(cls_id, str(cls_id)) if isinstance(model_names, dict) else str(cls_id)

            # Boite englobante
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)

            # Etiquette classe + confiance
            label = f"{cls_name} {conf_det:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(0, y1 - th - 6)
            cv2.rectangle(annotated, (x1, y_label), (x1 + tw + 4, y_label + th + 6), (0, 200, 255), -1)
            cv2.putText(annotated, label, (x1 + 2, y_label + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Point pieds (milieu bas de la bbox), comme dans TRD
            x_feet = int((x1 + x2) / 2)
            y_feet = int(y2)
            cv2.circle(annotated, (x_feet, y_feet), 7, (0, 255, 0), -1)
            cv2.circle(annotated, (x_feet, y_feet), 7, (255, 255, 255), 2)

    return annotated, infer_ms, n_boxes


def draw_hud(frame, frame_idx: int, total_frames: int, fps: float,
             infer_ms: float, n_boxes: int, paused: bool):
    h, w = frame.shape[:2]
    panel_h = 68
    panel = cv2.rectangle(
        img=(
            __import__("numpy").zeros((panel_h, w, 3), dtype=__import__("numpy").uint8)
        ),
        pt1=(0, 0),
        pt2=(w, panel_h),
        color=(0, 0, 0),
        thickness=-1,
    )

    status = "PAUSE" if paused else "LECTURE"
    cv2.putText(panel, f"Frame: {frame_idx}/{total_frames}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, f"FPS video: {fps:.1f}", (230, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, f"Inference: {infer_ms:.0f} ms", (390, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    cv2.putText(panel, f"Boxes: {n_boxes}", (610, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
    cv2.putText(panel, status, (760, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if not paused else (0, 0, 255), 2)

    cv2.putText(panel,
                "Espace=Pause/Play | Fleches=+/-1 frame | Trackbar=Navigation | Q/Echap=Quitter",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return __import__("numpy").vstack([frame, panel])


def main():
    default_model = (
        Path(__file__).resolve().parent
        / "Modelstrained"
        / "V2"
        / "yolo11s"
        / "weights"
        / "best.pt"
    )
    parser = argparse.ArgumentParser(
        description="Visualiseur de modeles entraines YOLO/RT-DETR avec trackbar")
    default_config = Path(__file__).resolve().parent.parent / "ExperimentsCAS" / "config.yaml"
    default_zones_json = ZONES_JSON_PATH
    parser.add_argument("--model", default=str(default_model),
                        help=(
                            "Chemin vers le poids (.pt), ex: runs_comparaison_outils/yolo11n/weights/best.pt "
                            f"(defaut: {default_model})"
                        ))
    parser.add_argument("--source", required=True,
                        help="Fichier video ou dossier contenant des videos")
    parser.add_argument("--model-type", default="auto", choices=["auto", "yolo", "rtdetr"],
                        help="Type de modele (auto par defaut)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Seuil de confiance")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Taille d'entree inference")
    parser.add_argument("--classes", type=str, default=None,
                        help="Filtre classes (ids) separees par virgule, ex: 0,39,41")
    parser.add_argument("--cache-size", type=int, default=300,
                        help="Nombre max de frames annotees en cache")
    parser.add_argument("--camera", default="cam_07",
                        help="Camera pour afficher la zone (defaut: cam_07)")
    parser.add_argument("--zones-json", default=str(default_zones_json),
                        help=f"Zones JSON (prioritaire) (defaut: {default_zones_json})")
    parser.add_argument("--config", default=str(default_config),
                        help=f"Config YAML contenant forbidden_zone_pixels (defaut: {default_config})")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[ERREUR] Modele introuvable: {model_path}")
        sys.exit(1)

    source_path = Path(args.source).resolve()
    videos = list_videos(source_path)
    if not videos:
        print(f"[ERREUR] Aucune video valide trouvee dans: {source_path}")
        sys.exit(1)

    video_path = choose_video(videos)
    if video_path is None:
        print("Arret utilisateur.")
        return

    classes = parse_classes(args.classes)
    zone_polygon = load_zone_polygon(
        camera_id=args.camera,
        zones_json_path=Path(args.zones_json).resolve(),
        config_path=Path(args.config).resolve(),
    )
    if zone_polygon is not None:
        print(f"[ZONE]   {args.camera}: {len(zone_polygon)} points charges")
    device, has_cuda = detect_device()
    use_half = has_cuda

    model, model_label = load_model(model_path, args.model_type)
    print(f"\n[MODELE] {model_label} charge: {model_path.name}")
    print(f"[VIDEO]  {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = max(total_frames, 1)

    window_name = f"Detections - {model_path.stem}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 760)

    def on_trackbar(_):
        pass

    cv2.createTrackbar("Frame", window_name, 0, max(total_frames - 1, 1), on_trackbar)

    cache = FrameCache(maxsize=max(50, args.cache_size))
    paused = True
    frame_idx = 0

    def get_annotated(i: int):
        cached = cache.get(i)
        if cached is not None:
            return cached

        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        annotated, infer_ms, n_boxes = run_predict(
            model=model,
            frame=frame,
            conf=args.conf,
            classes=classes,
            device=device,
            imgsz=args.imgsz,
            use_half=use_half,
            zone_polygon=zone_polygon,
        )
        payload = (annotated, infer_ms, n_boxes)
        cache.put(i, payload)
        return payload

    while True:
        # Sync position depuis la trackbar
        tb_val = cv2.getTrackbarPos("Frame", window_name)
        if tb_val != frame_idx:
            frame_idx = tb_val
            paused = True

        payload = get_annotated(frame_idx)
        if payload is None:
            break

        annotated, infer_ms, n_boxes = payload
        display = draw_hud(annotated, frame_idx, total_frames - 1, fps,
                           infer_ms, n_boxes, paused)
        cv2.imshow(window_name, display)

        wait_ms = 30 if paused else max(1, int(1000 / fps))
        key = cv2.waitKeyEx(wait_ms)

        if key in (KEY_Q, KEY_ESC):
            break
        if key == KEY_SPACE:
            paused = not paused
        elif key in (KEY_RIGHT, KEY_RIGHT_WIN):
            paused = True
            frame_idx = min(frame_idx + 1, total_frames - 1)
            cv2.setTrackbarPos("Frame", window_name, frame_idx)
        elif key in (KEY_LEFT, KEY_LEFT_WIN):
            paused = True
            frame_idx = max(frame_idx - 1, 0)
            cv2.setTrackbarPos("Frame", window_name, frame_idx)
        elif not paused and key == -1:
            if frame_idx < total_frames - 1:
                frame_idx += 1
                cv2.setTrackbarPos("Frame", window_name, frame_idx)
            else:
                paused = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

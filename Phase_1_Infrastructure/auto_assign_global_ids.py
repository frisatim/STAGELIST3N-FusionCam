"""
Clustering temporel des annotations d'objets interdits.

Lit un fichier JSON d'annotations (gt_objects_tad.json par defaut),
regroupe les apparitions par proximite temporelle + classe d'objet,
et assigne un id_evenement par salle (suite independante par salle).

Regles :
  - Meme salle + meme classe + ecart <= FUSION_WINDOW_MS -> meme evenement
  - Une camera ne peut pas apparaitre deux fois dans le meme evenement
  - Sauvegarde dans gt_objects_grouped.json (ne modifie pas l'original)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "gt_objects_tad.json"
OUTPUT_PATH = PROJECT_ROOT / "gt_objects_grouped.json"

FUSION_WINDOW_MS = 5000.0  # 5 secondes

# Definition des salles
# Salle 1 : cam_03, cam_05, cam_07
# Salle 2 : cam_01, cam_02, cam_04, cam_06, cam_08
ROOM_CAMERAS = {
    1: {"cam_03", "cam_05", "cam_07"},
    2: {"cam_01", "cam_02", "cam_04", "cam_06", "cam_08"},
}


def get_room_id(camera_id: str) -> int:
    for room_id, cameras in ROOM_CAMERAS.items():
        if camera_id in cameras:
            return room_id
    raise ValueError(f"Camera non mappee a une salle : {camera_id}")


def format_time(ms: float) -> str:
    total_s = ms / 1000.0
    mins, secs = divmod(total_s, 60)
    return f"{int(mins):02d}:{secs:06.3f}"


def load_annotations(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"[ERREUR] Le fichier {path} ne contient pas une liste.")
        sys.exit(1)
    return data


def cluster_annotations(annotations: list[dict]) -> tuple[list[dict], list[dict]]:
    """Assigne un id_evenement par salle via clustering temporel + classe.

    Chaque groupe (evenement) est defini par :
      - une salle
      - une classe_objet
      - un horodatage de reference (premiere apparition du groupe)
      - un ensemble de cameras deja presentes dans le groupe
    """
    sorted_anns = sorted(annotations, key=lambda a: a["horodatage_apparition"])

    # Chaque groupe :
    # {"id": int, "room_id": int, "classe": str, "t_ref": float, "cameras": set}
    groups: list[dict] = []
    # Compteur independant par salle
    current_room_ids: dict[int, int] = {room_id: 1 for room_id in ROOM_CAMERAS}

    for ann in sorted_anns:
        classe = ann["classe_objet"]
        t = ann["horodatage_apparition"]
        cam = ann["id_camera"]
        room_id = get_room_id(cam)

        ann["id_salle"] = room_id

        merged = False
        for group in groups:
            if group["room_id"] != room_id:
                continue
            if group["classe"] != classe:
                continue
            if abs(t - group["t_ref"]) > FUSION_WINDOW_MS:
                continue
            if cam in group["cameras"]:
                continue

            ann["id_evenement"] = group["id"]
            group["cameras"].add(cam)
            merged = True
            break

        if not merged:
            new_id = current_room_ids[room_id]
            ann["id_evenement"] = new_id
            groups.append(
                {
                    "id": new_id,
                    "room_id": room_id,
                    "classe": classe,
                    "t_ref": t,
                    "cameras": {cam},
                }
            )
            current_room_ids[room_id] += 1

    return sorted_anns, groups


def print_summary(groups: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("  BILAN DU CLUSTERING TEMPOREL")
    print(f"  Fenetre de fusion : {FUSION_WINDOW_MS / 1000:.0f} secondes")
    print(f"{'=' * 60}")

    total_events = 0
    for room_id in sorted(ROOM_CAMERAS):
        room_groups = [g for g in groups if g["room_id"] == room_id]
        room_groups = sorted(room_groups, key=lambda x: x["id"])
        total_events += len(room_groups)

        cams_in_room = ", ".join(sorted(ROOM_CAMERAS[room_id]))
        print(f"\n  Salle {room_id} ({cams_in_room})")

        for g in room_groups:
            cams = ", ".join(sorted(g["cameras"]))
            t_str = format_time(g["t_ref"])
            print(
                f"    Evenement {g['id']:3d} ({g['classe']:<16s}) : "
                f"vu par {cams}  [t_ref={t_str}]"
            )

    print(f"\n  Total : {total_events} evenements distincts (toutes salles)")
    print(f"{'=' * 60}\n")


def main() -> None:
    input_path = DEFAULT_INPUT

    # Accepter un chemin en argument optionnel
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"[ERREUR] Fichier introuvable : {input_path}")
        sys.exit(1)

    print(f"[LECTURE] {input_path}")
    annotations = load_annotations(input_path)
    print(f"  {len(annotations)} annotations chargees")

    try:
        grouped, groups = cluster_annotations(annotations)
    except ValueError as exc:
        print(f"[ERREUR] {exc}")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2, ensure_ascii=False)
    print(f"[SAUVEGARDE] {OUTPUT_PATH}")

    print_summary(groups)


if __name__ == "__main__":
    main()

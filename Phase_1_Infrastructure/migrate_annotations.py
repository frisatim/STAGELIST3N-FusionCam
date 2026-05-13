"""
Migration des annotations gt_people.json existantes.

Parcourt chaque entrée et demande interactivement :
  - Le VRAI id_evenement global (pour grouper les caméras d'un même événement)
  - La zone (A ou B)

Les entrées déjà migrées (avec zone_id) sont ignorées.
Sauvegarde le résultat dans gt_people.json.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_PEOPLE_PATH = PROJECT_ROOT / "gt_people.json"


def format_time_ms(timestamp_ms: float) -> str:
    total_s = timestamp_ms / 1000.0
    mins, secs = divmod(total_s, 60)
    whole_s = int(secs)
    millis = int(round((secs - whole_s) * 1000))
    return f"{int(mins):02d}:{whole_s:02d}.{millis:03d}"


def main():
    if not GT_PEOPLE_PATH.exists():
        print(f"[ERREUR] Fichier introuvable : {GT_PEOPLE_PATH}")
        return

    with open(GT_PEOPLE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[ERREUR] Le fichier ne contient pas une liste.")
        return

    # Séparer les entrées à migrer et celles déjà faites
    to_migrate = [e for e in data if "zone_id" not in e]
    already_done = [e for e in data if "zone_id" in e]

    print(f"╔═══════════════════════════════════════════════════════════╗")
    print(f"║  MIGRATION GT_PEOPLE.JSON                                ║")
    print(f"║  Ajout de zone_id + correction id_evenement global       ║")
    print(f"╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Total entrées     : {len(data):>4d}                              ║")
    print(f"║  Déjà migrées      : {len(already_done):>4d}                              ║")
    print(f"║  À migrer          : {len(to_migrate):>4d}                              ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")

    if not to_migrate:
        print("\n[OK] Toutes les entrées ont déjà un zone_id. Rien à faire.")
        return

    print("\nPour chaque entrée, entrez : <id_evenement_global> <zone>")
    print("Exemple : 1 A")
    print("Tapez 's' pour sauter une entrée, 'q' pour quitter et sauvegarder.\n")

    migrated_count = 0

    for i, entry in enumerate(to_migrate):
        cam = entry["id_camera"]
        old_eid = entry.get("id_evenement", "?")
        frame = entry["trame_violation"]
        ts = entry.get("horodatage_violation", 0)
        ts_str = format_time_ms(ts) if isinstance(ts, (int, float)) else str(ts)

        print(f"  [{i + 1}/{len(to_migrate)}] {cam} | "
              f"ancien id_evenement={old_eid} | "
              f"frame={frame} | t={ts_str}")

        while True:
            try:
                resp = input("    -> id_evenement zone (ex: 1 A) : ").strip()
            except EOFError:
                resp = "q"

            if resp.lower() == "q":
                print("\n  [QUIT] Sauvegarde en cours...")
                # Sauvegarder ce qu'on a déjà fait
                _save(data)
                return

            if resp.lower() == "s":
                print("    Sauté.")
                break

            parts = resp.split()
            if len(parts) != 2:
                print("    Format invalide. Attendu : <entier> <A ou B>")
                continue

            try:
                new_eid = int(parts[0])
            except ValueError:
                print("    L'id_evenement doit être un entier.")
                continue

            zone = parts[1].upper()
            if zone not in ("A", "B"):
                print(f"    Zone invalide : '{zone}'. Attendu A ou B.")
                continue

            # Appliquer la migration
            entry["id_evenement"] = new_eid
            entry["zone_id"] = zone
            migrated_count += 1
            print(f"    -> Événement #{new_eid}, Zone {zone}")
            break

    _save(data)

    print(f"\n╔═══════════════════════════════════════════════════════════╗")
    print(f"║  MIGRATION TERMINÉE                                      ║")
    print(f"║  {migrated_count:>4d} entrées migrées                              ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")

    # Résumé par événement
    events = {}
    for e in data:
        if "zone_id" in e:
            key = (e["id_evenement"], e["zone_id"])
            events.setdefault(key, []).append(e["id_camera"])

    if events:
        print("\n  Résumé des événements :")
        for (eid, zone), cams in sorted(events.items()):
            unique_cams = sorted(set(cams))
            print(f"    Événement #{eid} Zone {zone} : "
                  f"{', '.join(unique_cams)} ({len(cams)} annotations)")


def _save(data: list[dict]) -> None:
    with open(GT_PEOPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [SAUVEGARDE] {len(data)} entrées -> {GT_PEOPLE_PATH.name}")


if __name__ == "__main__":
    main()

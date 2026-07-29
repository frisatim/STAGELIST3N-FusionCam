"""
Vérification visuelle des coordonnées machines sur le plan de sol.

Dessine les machines (depuis machines.yaml) et les zones interdites
(depuis config.yaml) sur le plan de sol pour vérifier que tout est
correctement positionné.

Usage :
    python verify_machines.py
    python verify_machines.py --machines machines.yaml
    python verify_machines.py --show-labels        # affiche les noms des machines
    python verify_machines.py --show-grid           # affiche une grille en mètres
    python verify_machines.py --machines-coords-unit pixels
    python verify_machines.py --save output.png     # sauvegarde l'image
"""

import os
import sys
import argparse
import yaml
import numpy as np
import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Couleurs (BGR)
COLOR_MACHINE = (0, 0, 200)       # Rouge
COLOR_MACHINE_FILL = (0, 0, 200)
COLOR_ZONE = (0, 0, 255)          # Rouge vif pour les zones interdites
COLOR_ZONE_FILL = (0, 0, 255)
COLOR_GRID = (200, 200, 200)      # Gris clair
COLOR_LABEL = (0, 0, 0)           # Noir
COLOR_CAMERA = (255, 0, 0)        # Bleu pour les positions caméras
COLOR_ORIGIN = (0, 180, 0)        # Vert pour l'origine


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metres_to_pixels(x_m, y_m, echelle, origin_px_x, origin_px_y):
    """Convertit des coordonnées mètres en pixels sur le plan."""
    px = int(origin_px_x + x_m * echelle)
    py = int(origin_px_y + y_m * echelle)
    return px, py


def draw_polygon_metres(image, coins_metres, echelle, ox, oy, color, thickness=2, fill_alpha=0.15):
    """Dessine un polygone dont les coins sont en mètres."""
    pts_px = []
    for (x_m, y_m) in coins_metres:
        px, py = metres_to_pixels(x_m, y_m, echelle, ox, oy)
        pts_px.append([px, py])
    pts = np.array(pts_px, dtype=np.int32)

    # Remplissage semi-transparent
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, image)

    # Contour
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

    return pts_px


def draw_polygon_pixels(image, coins_pixels, color, thickness=2, fill_alpha=0.15):
    """Dessine un polygone dont les coins sont deja en pixels."""
    pts = np.array(coins_pixels, dtype=np.int32)

    # Remplissage semi-transparent
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, image)

    # Contour
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

    return pts.tolist()


def infer_machines_coords_unit(machines):
    """Heuristique simple: des valeurs tres grandes sont probablement en pixels."""
    all_vals = []
    for machine in machines:
        for x, y in machine.get("coins_metres", []):
            all_vals.extend([abs(float(x)), abs(float(y))])

    if not all_vals:
        return "metres"

    # Dans ce projet, les coordonnees en metres sont petites (<~20),
    # alors que les pixels atteignent facilement plusieurs centaines.
    return "pixels" if max(all_vals) > 50 else "metres"


def main():
    parser = argparse.ArgumentParser(description="Vérification visuelle machines + zones sur plan")
    parser.add_argument("--machines", default=os.path.join(SCRIPT_DIR, "machines.yaml"),
                        help="Fichier YAML des machines")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"),
                        help="Fichier config.yaml")
    parser.add_argument(
        "--machines-coords-unit",
        choices=["auto", "metres", "pixels"],
        default="auto",
        help="Unite des coordonnees machines dans le YAML",
    )
    parser.add_argument("--show-labels", action="store_true", help="Afficher les noms des machines")
    parser.add_argument("--show-grid", action="store_true", help="Afficher grille métrique")
    parser.add_argument("--save", type=str, help="Sauvegarder l'image (ex: output.png)")
    args = parser.parse_args()

    # --- Charger config ---
    config = load_yaml(args.config)
    geo = config["geometrie_2d"]
    echelle = geo["echelle_px_par_metre"]
    ox = geo["origine_pixel_x"]
    oy = geo["origine_pixel_y"]

    # --- Charger le plan de sol ---
    plan_path = os.path.join(SCRIPT_DIR, geo["plan_image"])
    plan = cv2.imread(plan_path)
    if plan is None:
        sys.exit(f"[ERREUR] Plan introuvable : {plan_path}")

    print(f"Plan charge : {plan.shape[1]}x{plan.shape[0]} px")
    print(f"Echelle : {echelle} px/m | Origine : ({ox}, {oy}) px")

    # --- Grille métrique ---
    if args.show_grid:
        h, w = plan.shape[:2]
        # Grille tous les mètres
        max_x_m = int((w - ox) / echelle) + 1
        max_y_m = int((h - oy) / echelle) + 1
        for x_m in range(max_x_m + 1):
            px, _ = metres_to_pixels(x_m, 0, echelle, ox, oy)
            cv2.line(plan, (px, 0), (px, h), COLOR_GRID, 1)
            cv2.putText(plan, f"{x_m}m", (px + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_GRID, 1)
        for y_m in range(max_y_m + 1):
            _, py = metres_to_pixels(0, y_m, echelle, ox, oy)
            cv2.line(plan, (0, py), (w, py), COLOR_GRID, 1)
            cv2.putText(plan, f"{y_m}m", (2, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_GRID, 1)

    # --- Dessiner l'origine ---
    cv2.circle(plan, (ox, oy), 6, COLOR_ORIGIN, -1)
    cv2.putText(plan, "(0,0)", (ox + 8, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ORIGIN, 1)

    # --- Dessiner les zones interdites ---
    zones = config.get("zones_interdites", {})
    for zone_id, zone_data in zones.items():
        coords = zone_data.get("coordonnees_metres")
        if not coords or len(coords) < 3:
            print(f"[WARN] Zone '{zone_id}' ignorée (coordonnées absentes/invalides).")
            continue

        pts_px = draw_polygon_metres(plan, coords, echelle, ox, oy,
                                     COLOR_ZONE, thickness=3, fill_alpha=0.2)
        # Label de la zone
        cx = int(np.mean([p[0] for p in pts_px]))
        cy = int(np.mean([p[1] for p in pts_px]))
        cv2.putText(plan, zone_data["nom"], (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_ZONE, 1)
        print(f"Zone '{zone_data['nom']}' dessinee ({len(coords)} sommets)")
        print(f"  - coordonnees_metres: {coords}")
        print(f"  - coordonnees_pixels: {pts_px}")

    # --- Dessiner les machines ---
    if os.path.isfile(args.machines):
        machines_data = load_yaml(args.machines)
        machines = machines_data.get("machines", [])

        coords_unit = args.machines_coords_unit
        if coords_unit == "auto":
            coords_unit = infer_machines_coords_unit(machines)
        print(f"Unite coordonnees machines: {coords_unit}")

        n_valid = 0
        for i, machine in enumerate(machines):
            coins = machine.get("coins_metres", [])
            nom = machine.get("nom", f"Machine_{i}")

            # Ignorer les machines non remplies (tous les coins à 0,0)
            if all(c[0] == 0.0 and c[1] == 0.0 for c in coins):
                continue

            if coords_unit == "pixels":
                pts_px = draw_polygon_pixels(plan, coins, COLOR_MACHINE, thickness=2, fill_alpha=0.25)
            else:
                pts_px = draw_polygon_metres(plan, coins, echelle, ox, oy,
                                             COLOR_MACHINE, thickness=2, fill_alpha=0.25)

            if args.show_labels:
                cx = int(np.mean([p[0] for p in pts_px]))
                cy = int(np.mean([p[1] for p in pts_px]))
                cv2.putText(plan, nom, (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_LABEL, 1)

            n_valid += 1

        print(f"Machines dessinees : {n_valid}/{len(machines)}")
    else:
        print(f"[WARN] Fichier machines introuvable : {args.machines}")
        print(f"       Seules les zones interdites sont affichees.")

    # --- Affichage ---
    window_name = "Plan de sol : Machines + Zones (Q pour quitter)"

    if args.save:
        cv2.imwrite(args.save, plan)
        print(f"[OK] Image sauvegardee : {args.save}")

    cv2.imshow(window_name, plan)
    print("\nAppuyez sur 'Q' pour quitter.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
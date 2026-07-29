"""
Module de fusion multi-caméras du système de surveillance industriel (Phase 3).

Place dans le pipeline : après le suivi mono-caméra (tracker.py), ce module
associe les détections vues simultanément par plusieurs caméras dont les
champs de vision se recouvrent, afin d'attribuer à chaque entité physique un
identifiant global unique et stable dans le temps. Ces identifiants globaux
alimentent ensuite la détection de violations (violation_detector.py).

Principe :
  1. Pour chaque paire de caméras en recouvrement (déclarée dans config.yaml,
     salle par salle), un algorithme hongrois apparie les détections selon
     leur distance au sol en mètres.
  2. Les appariements retenus sont fusionnés dans une structure Union-Find
     avec compression de chemin : chaque composante connexe représente une
     même entité physique vue par une ou plusieurs caméras.
  3. Chaque composante reçoit un identifiant global persistant d'une frame à
     l'autre (l'identifiant le plus ancien de la composante est conservé).

L'association est volontairement limitée aux paires en recouvrement d'une
même salle : deux caméras sans recouvrement physique ne peuvent pas observer
la même personne, et restreindre les paires évite de fusionner à tort deux
entités situées à des coordonnées sol proches mais dans des salles
différentes.

Depuis le correctif "class-aware", deux détections ne sont appariables que si
leurs classes sont compatibles : sans cette contrainte, une personne et un
objet posé à ses pieds pouvaient partager un identifiant global, ce qui
polluait le vote multi-caméras et générait des faux positifs.
"""

from __future__ import annotations

import math
import unicodedata

import numpy as np
from scipy.optimize import linear_sum_assignment

from detection import Detection

# Clé unique d'une piste locale : (cam_id, track_id, classe canonique)
TrackKey = tuple[str, int, str]
# Coût sentinelle : rend un appariement inéligible sans casser l'algorithme hongrois
INVALID_COST = 1e6


class MultiCameraFusion:
    """Associe les détections inter-caméras et gère les identifiants globaux.

    Args:
        config: Dictionnaire issu de config.yaml (section camera_overlaps).
        distance_threshold_m: Distance au sol maximale (en mètres) pour
            apparier deux détections de caméras différentes.
        time_window_s: Écart temporel maximal (en secondes) entre deux
            détections pour qu'elles restent appariables.
        require_class_match: Si True, seules des détections de classes
            compatibles peuvent être associées (correctif class-aware,
            voir la docstring de module).
    """

    def __init__(
        self,
        config: dict,
        distance_threshold_m: float = 1.0,
        time_window_s: float = 0.1,
        require_class_match: bool = True,
    ) -> None:
        self._distance_threshold_m = distance_threshold_m
        self._time_window_s = time_window_s
        self._require_class_match = require_class_match
        self._overlap_pairs: list[tuple[str, str]] = self._load_overlap_pairs(config)

        # Table des parents Union-Find, indexée par (cam_id, track_id, classe canonique)
        self._parent: dict[TrackKey, TrackKey] = {}

        # Registre persistant des identifiants globaux : survit d'un appel associate() à l'autre
        self._track_to_global: dict[TrackKey, int] = {}
        self._next_global_id: int = 1

        print(
            f"[INFO] MultiCameraFusion initialised: "
            f"{len(self._overlap_pairs)} overlap pairs, "
            f"dist_thresh={distance_threshold_m}m, "
            f"time_window={time_window_s}s, "
            f"class_match={'ON' if require_class_match else 'OFF'}"
        )

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def associate(
        self, detections_by_cam: dict[str, list[Detection]]
    ) -> list[Detection]:
        """Associe les détections des paires de caméras en recouvrement.

        Args:
            detections_by_cam: Dictionnaire caméra vers liste des détections
                               du lot de frames courant.

        Returns:
            Liste à plat de toutes les détections, avec global_id renseigné.
        """
        all_dets: list[Detection] = [
            det for dets in detections_by_cam.values() for det in dets
        ]

        # Initialise l'Union-Find avec tous les noeuds présents sur cette frame
        self._parent = {}
        for det in all_dets:
            key = self._detection_key(det)
            if key not in self._parent:
                self._parent[key] = key

        # Étape 1 : association hongroise pour chaque paire de caméras en recouvrement
        for cam_a, cam_b in self._overlap_pairs:
            dets_a = detections_by_cam.get(cam_a, [])
            dets_b = detections_by_cam.get(cam_b, [])

            # Ignore les paires dont au moins une caméra n'a aucune détection
            if not dets_a or not dets_b:
                continue

            # Ne garde que les détections disposant d'une position au sol :
            # sans projection en mètres, aucune distance inter-caméras n'est calculable
            valid_a = [d for d in dets_a if d.foot_point_m is not None]
            valid_b = [d for d in dets_b if d.foot_point_m is not None]

            if not valid_a or not valid_b:
                continue

            C = self._build_cost_matrix(valid_a, valid_b)
            row_inds, col_inds = linear_sum_assignment(C)

            # Seuls les appariements sous le seuil de distance fusionnent :
            # les coûts sentinelles (classes incompatibles, hors fenêtre
            # temporelle) sont éliminés au passage
            for r, c in zip(row_inds, col_inds):
                if C[r, c] < self._distance_threshold_m:
                    self._union(
                        self._detection_key(valid_a[r]),
                        self._detection_key(valid_b[c]),
                    )

        # Étape 2 : regroupe les composantes connexes et attribue / propage
        # les identifiants globaux
        components: dict[TrackKey, list[TrackKey]] = {}
        for key in self._parent:
            root = self._find(key)
            components.setdefault(root, []).append(key)

        for members in components.values():
            # Récupère les identifiants globaux déjà connus des membres de la composante
            existing_ids = [
                self._track_to_global[m] for m in members if m in self._track_to_global
            ]
            # L'identifiant le plus ancien (le plus petit) gagne : c'est le
            # plus stable dans le temps
            gid = min(existing_ids) if existing_ids else self._allocate_global_id()
            for m in members:
                self._track_to_global[m] = gid

        # Étape 3 : les détections sans position au sol reçoivent aussi un
        # identifiant global persistant
        for det in all_dets:
            key = self._detection_key(det)
            if key not in self._track_to_global:
                # Noeud jamais atteint par l'Union-Find (foot_point_m est None)
                self._track_to_global[key] = self._allocate_global_id()

            det.global_id = self._track_to_global[key]

        return all_dets

    def reset(self) -> None:
        """Réinitialise l'état de suivi persistant entre deux sessions vidéo."""
        self._parent.clear()
        self._track_to_global.clear()
        self._next_global_id = 1
        print("[INFO] MultiCameraFusion state reset.")

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _load_overlap_pairs(self, config: dict) -> list[tuple[str, str]]:
        """Extrait du config toutes les paires de caméras en recouvrement.

        La section camera_overlaps est organisée par salle : seules les
        paires déclarées y figurent, ce qui borne l'association aux caméras
        d'une même salle (voir la docstring de module).
        """
        pairs: list[tuple[str, str]] = []
        overlap_section = config.get("camera_overlaps", {})
        for room_pairs in overlap_section.values():
            for pair in room_pairs:
                if len(pair) == 2:
                    pairs.append((pair[0], pair[1]))
                else:
                    # Paire mal formée dans le YAML : ignorée plutôt que de planter
                    print(f"[WARN] Malformed overlap pair ignored: {pair}")
        return pairs

    def _build_cost_matrix(
        self, dets_a: list[Detection], dets_b: list[Detection]
    ) -> np.ndarray:
        """Construit la matrice de coûts (len_a x len_b) des distances au sol.

        Chaque coût est la distance euclidienne en mètres entre les points au
        sol des deux détections. Les paires inéligibles (classes incompatibles
        ou écart temporel supérieur à la fenêtre) reçoivent le coût sentinelle
        1e6 : l'algorithme hongrois ne les retiendra jamais comme appariement
        optimal, et le seuil de distance les écarterait de toute façon.
        """
        n, m = len(dets_a), len(dets_b)
        C = np.empty((n, m), dtype=np.float64)
        for i, da in enumerate(dets_a):
            for j, db in enumerate(dets_b):
                if self._require_class_match and not self._classes_compatible(da, db):
                    C[i, j] = INVALID_COST
                elif abs(da.timestamp - db.timestamp) > self._time_window_s:
                    C[i, j] = INVALID_COST
                else:
                    dx = da.foot_point_m[0] - db.foot_point_m[0]  # type: ignore[index]
                    dy = da.foot_point_m[1] - db.foot_point_m[1]  # type: ignore[index]
                    C[i, j] = math.sqrt(dx * dx + dy * dy)
        return C

    def _find(self, x: TrackKey) -> TrackKey:
        """Recherche la racine avec compression de chemin (halving en une passe)."""
        while self._parent[x] != x:
            # Compression de chemin : rattache x à son grand-parent à chaque itération
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def _union(self, a: TrackKey, b: TrackKey) -> None:
        """Fusionne les composantes contenant a et b."""
        # Les noeuds issus de détections sans position au sol peuvent être absents de la table
        for node in (a, b):
            if node not in self._parent:
                self._parent[node] = node

        root_a = self._find(a)
        root_b = self._find(b)
        if root_a != root_b:
            # Rattache root_b sous root_a (pas d'union par rang : le nombre
            # de caméras reste faible, l'arbre reste court)
            self._parent[root_b] = root_a

    def _detection_key(self, det: Detection) -> TrackKey:
        """Clé Union-Find d'une détection : (caméra, piste locale, classe canonique)."""
        return (det.cam_id, det.track_id, self._class_key(det))

    def _classes_compatible(self, a: Detection, b: Detection) -> bool:
        """Vrai si les deux détections partagent la même classe canonique."""
        return self._class_key(a) == self._class_key(b)

    def _class_key(self, det: Detection) -> str:
        """Normalise le nom de classe en une clé canonique.

        Supprime accents, casse et underscores, puis rabat les synonymes de
        "personne" (français / anglais, singulier / pluriel) sur la clé unique
        "person". Garantit que la compatibilité de classes ne dépend ni de la
        langue ni du modèle de détection utilisé.
        """
        raw = det.class_name or str(det.class_id)
        normalized = unicodedata.normalize("NFKD", raw)
        ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
        key = " ".join(ascii_name.lower().strip().replace("_", " ").split())
        if key in {"person", "persons", "personne", "personnes", "human", "humain"}:
            return "person"
        return key or str(det.class_id)

    def _allocate_global_id(self) -> int:
        """Retourne le prochain identifiant global libre et avance le compteur."""
        gid = self._next_global_id
        self._next_global_id += 1
        return gid


# ----------------------------------------------------------------------
# Test rapide (smoke test)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    config_path = "config.yaml"
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    fusion = MultiCameraFusion(cfg, distance_threshold_m=1.0, time_window_s=0.1)

    t = 1743000000.0

    det_cam03_1 = Detection(
        cam_id="cam_03",
        track_id=1,
        global_id=None,
        timestamp=t,
        bbox_px=(100.0, 200.0, 180.0, 400.0),
        foot_point_px=(140.0, 400.0),
        foot_point_m=(5.0, 2.0),
        confidence=0.92,
        class_id=0,
        class_name="person",
    )
    det_cam07_1 = Detection(
        cam_id="cam_07",
        track_id=1,
        global_id=None,
        timestamp=t + 0.03,
        bbox_px=(200.0, 210.0, 280.0, 410.0),
        foot_point_px=(240.0, 410.0),
        foot_point_m=(5.1, 2.05),   # proche de det_cam03_1 : doit fusionner
        confidence=0.88,
        class_id=0,
        class_name="person",
    )
    det_cam07_2 = Detection(
        cam_id="cam_07",
        track_id=2,
        global_id=None,
        timestamp=t,
        bbox_px=(400.0, 210.0, 480.0, 410.0),
        foot_point_px=(440.0, 410.0),
        foot_point_m=(8.0, 0.5),   # loin des détections de cam_03 : identifiant distinct
        confidence=0.75,
        class_id=0,
        class_name="person",
    )
    det_cam05_no_floor = Detection(
        cam_id="cam_05",
        track_id=3,
        global_id=None,
        timestamp=t,
        bbox_px=(50.0, 50.0, 100.0, 200.0),
        foot_point_px=(75.0, 200.0),
        foot_point_m=None,          # pas d'homographie : identifiant propre
        confidence=0.60,
        class_id=0,
        class_name="person",
    )

    result = fusion.associate(
        {
            "cam_03": [det_cam03_1],
            "cam_07": [det_cam07_1, det_cam07_2],
            "cam_05": [det_cam05_no_floor],
        }
    )

    print(f"\n[INFO] Frame 1 : {len(result)} detections after fusion:")
    for d in result:
        print(f"  {d}")

    # cam_03:1 et cam_07:1 doivent partager le même global_id
    assert det_cam03_1.global_id == det_cam07_1.global_id, (
        f"[ERR] Expected cam_03:1 and cam_07:1 to share a global_id, "
        f"got {det_cam03_1.global_id} vs {det_cam07_1.global_id}"
    )
    assert det_cam07_2.global_id != det_cam03_1.global_id, (
        "[ERR] cam_07:2 should have a different global_id"
    )
    assert det_cam05_no_floor.global_id is not None, (
        "[ERR] Detection without floor point should still get a global_id"
    )

    # Deuxième frame : même personne physique, les identifiants doivent rester stables
    det_cam03_1b = Detection(
        cam_id="cam_03",
        track_id=1,
        global_id=None,
        timestamp=t + 0.5,
        bbox_px=(101.0, 201.0, 181.0, 401.0),
        foot_point_px=(141.0, 401.0),
        foot_point_m=(5.02, 2.01),
        confidence=0.91,
        class_id=0,
        class_name="person",
    )
    det_cam07_1b = Detection(
        cam_id="cam_07",
        track_id=1,
        global_id=None,
        timestamp=t + 0.53,
        bbox_px=(201.0, 211.0, 281.0, 411.0),
        foot_point_px=(241.0, 411.0),
        foot_point_m=(5.12, 2.06),
        confidence=0.87,
        class_id=0,
        class_name="person",
    )

    result2 = fusion.associate(
        {"cam_03": [det_cam03_1b], "cam_07": [det_cam07_1b]}
    )

    print(f"\n[INFO] Frame 2 : {len(result2)} detections after fusion:")
    for d in result2:
        print(f"  {d}")

    assert det_cam03_1b.global_id == det_cam03_1.global_id, (
        "[ERR] global_id must be stable across frames for the same track"
    )

    print("\n[INFO] All assertions passed.")

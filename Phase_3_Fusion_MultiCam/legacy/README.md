# Scripts legacy Phase 3

Scripts conserves pour la tracabilite du travail, mais remplaces par des
versions plus recentes. Ne pas les utiliser pour reproduire les resultats.

- `calibration_tool_v1.py` (ancien nom : `calibration_tool.py`) : premiere
  calibration par 4 coins en coordonnees absolues. Ecrit la cle
  `src_points_px` qui n'est plus lue par le pipeline. Remplace par
  `../calibration_tool_v2.py` (multi-resolution, diagnostics d'erreur en cm).
- `violation_checker.py` : prototype de verification point-dans-zone.
  Reecrit et etendu dans `../violation_detector.py` (votes multi-cameras,
  niveaux weak/confirmed, machine a etats des alertes).

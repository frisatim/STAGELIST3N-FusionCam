# Scripts legacy Phase 2

Scripts conserves pour la tracabilite du travail, mais remplaces par des
versions plus recentes. Ne pas les utiliser pour reproduire les resultats.

- `train_yolo_dryrun.py` : dry-run d'entrainement 1 epoch utilise au tout
  debut pour valider l'installation. Pointe un dataset (`yolo_dataset/atelier.yaml`)
  qui n'est plus genere. Remplace par `../train_models.py`.

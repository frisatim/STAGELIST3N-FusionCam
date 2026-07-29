# Phase 2 - Baseline mono-camera

Reference du projet : detection et alertes en coordonnees image, camera
par camera, sans fusion. Cette phase entraine les modeles (YOLOv8,
YOLO11, RT-DETR) et fournit l'evaluation TAD/TRD a laquelle la Phase 3
est comparee. Definitions des metriques : `docs/METRIQUES.md`. Versions
des modeles et datasets : `docs/VERSIONS_MODELES.md`.

## Scripts principaux

| Script | Role |
|---|---|
| `train_models.py` | Entrainement comparatif multi-architectures sur `dataset_objets_V4` (80 epochs, imgsz 960 par defaut). |
| `evaluate_trd.py` | Evaluation TRD (intrusion en zone interdite) sur enregistrements, appariement evenementiel avec tolerance 10 s. |
| `evaluate_tad.py` | Evaluation TAD (objet interdit) avec regroupement des faux positifs (fenetre 3 s). |
| `export_onnx.py` | Export des poids `.pt` vers `.onnx` puis `.engine` TensorRT (FP16/pruning en option). |
| `draw_zones.py` | Tracage interactif des zones interdites en pixels (`zones_config.json`). |
| `view_trained_detector.py` | Visualiseur des modeles entraines sur video (trackbar). |
| `demo_yolo_baseline.py` | Demo YOLO sur les 8 cameras live (grille 2x4), sans argparse. |

## Scripts d'annotation et de dataset

| Script | Role |
|---|---|
| `auto_annotate_video.py` | Extraction de frames + pre-annotation YOLO depuis une video. |
| `auto_annotate_person.py` | Pre-annotation de la classe personne sur un dataset existant. |
| `merge_roboflow_dataset.py` | Fusion d'un export Roboflow dans le dataset cible avec remap des classes. |

## Fichiers

- `zones_config.json` : zones interdites en pixels par camera (produit
  par `draw_zones.py`).
- `Modelstrained/` : poids entraines. Dossier non versionne, fourni par
  le dossier de donnees externe via jonction Windows ou montage Docker
  (voir `docs/DATA_LAYOUT.md`).
- `legacy/` : scripts remplaces (voir `legacy/README.md`).

## Sorties d'evaluation

`evaluate_trd.py` et `evaluate_tad.py` produisent des CSV par camera et
par modele (TP/FP/FN, precision, rappel, delais median/p95, FAR). Ils
sont aussi appeles automatiquement par
`Phase_3_Fusion_MultiCam/run_recorded_campaign.py` pour la partie
Phase 2 de la comparaison.

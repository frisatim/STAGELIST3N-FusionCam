# Phase 1 - Infrastructure d'acquisition

Capture des 8 flux cameras RTSP de l'atelier, enregistrement synchronise,
annotation de la verite terrain et conversion vers le format YOLO.
Cette phase produit les enregistrements video et les ground truths
(`ground_truth/`) utilises par toutes les evaluations suivantes.

## Scripts

| Script | Role |
|---|---|
| `streamlive.py` | Visualisation ou enregistrement des 8 cameras (grille), replay MediaMTX. |
| `records.py` | Enregistrement RTSP des 8 flux avec cadence de frames stricte (frame pacing). |
| `crash_test_streams.py` | Test de stabilite des 8 flux pendant 30 minutes (grille 2x4). |
| `annotation_tool.py` | Outil interactif d'annotation : evenements TRD (personnes/zones), TAD (objets) et labels YOLO. |
| `auto_assign_global_ids.py` | Regroupement temporel des annotations objets en evenements (`id_evenement`). |
| `convert_json_to_yolo.py` | Conversion de `gt_objects.json` en dataset YOLO (images + labels). |
| `migrate_annotations.py` | Migration one-shot de `gt_people.json` (ajout du champ `zone_id`). |

## Remarques

- Les URLs RTSP reelles ont ete remplacees par des placeholders
  (`rtsp://admin:<PASSWORD>@<CAMERA_IP>:554/...`). Pour executer ces
  scripts sur une installation reelle, renseigner les URLs dans la liste
  `CAMERAS` en tete de script (ou dans `Phase_3_Fusion_MultiCam/config.yaml`
  pour le pipeline Phase 3).
- Les cameras cam_03, cam_05 et cam_07 diffusent en 704x576 avec un
  aspect ratio errone ; le correctif est applique plus tard par
  `Phase_3_Fusion_MultiCam/geometry_fix.py`.
- `legacy/` contient les scripts remplaces (voir `legacy/README.md`).

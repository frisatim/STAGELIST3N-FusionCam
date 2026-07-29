# Phase 3 - Fusion multi-cameras

Coeur du projet : projection des detections sur le plan sol par
homographie, association inter-cameras, detection de violations de zones
en coordonnees metres et campagnes d'evaluation comparees a la Phase 2.

Pipeline complet : lecture des flux, detection (YOLO/RT-DETR), suivi
ByteTrack par camera, correction d'aspect ratio, projection du point
bas-centre au sol, fusion inter-cameras, verification des zones, alertes
et exports.

## Points d'entree

| Script | Role |
|---|---|
| `run_recorded_campaign.py` | Campagne complete Phase 2 vs Phase 3 sur videos enregistrees (le script de reference des comparaisons du rapport). |
| `run_live_campaign.py` | Campagne live RTSP a duree bornee : performances, alertes, export metadonnees JSONL/HTTP, trace de latence. |

Exemples de commandes : `README.md` racine et `docs/REPRODUCTION.md`.

## Coeur du pipeline

| Module | Role |
|---|---|
| `pipeline.py` | Orchestration : cameras, tracking, fusion, violations, affichage, modes recorded et live. |
| `campaign_utils.py` | Socle des campagnes : decouverte des modeles, execution, CSV, metriques, comparaison Phase 2/3. |
| `detection.py` / `alert.py` | Structures de donnees Detection et Alert. |
| `tracker.py` | ByteTrack mono-camera + projection homographique du point bas-centre. |
| `fusion.py` | Association inter-cameras : algorithme hongrois par paire de cameras en overlap + Union-Find, classes compatibles exigees, IDs globaux persistants. |
| `violation_detector.py` | Zones interdites en metres, vote multi-cameras, niveaux weak/confirmed, machine a etats des alertes. |
| `geometry_fix.py` | Correctif d'aspect ratio des cameras Dahua 704x576 (a appliquer avant toute inference/homographie). |
| `video_capture.py` | Capture RTSP anti-buffering (GStreamer ou OpenCV/FFmpeg). |
| `metadata_publisher.py` | Publication des metadonnees par frame (JSONL, HTTP POST) vers la Phase 4. |

## Calibration et outils

| Script | Role |
|---|---|
| `calibration_tool_v2.py` | Calibration homographique multi-resolution image vers plan sol (points de reference, erreur en cm). |
| `verify_calibration.py` | Verification visuelle : reprojection des zones sur la video. |
| `verify_machines.py` | Verification des coordonnees machines sur le plan de sol. |
| `draw_zone_multicam.py` | Dessin d'une zone projetee automatiquement sur toutes les cameras en overlap. |
| `audit_calibration_alerts.py` | Audit qualite de calibration a partir des logs d'alertes. |
| `ablate_fusion_threshold.py` | Ablation du seuil de distance d'association D (resultats dans `reports/ablation/`). |
| `evaluate_fusion_links.py` | Evaluation de la qualite des liens de fusion contre un mini GT annote. |

## Validation

| Script | Role |
|---|---|
| `visual_check_tracker.py` | Harnais visuel du tracking + projection (1 camera). |
| `visual_check_fusion.py` | Harnais visuel de la fusion (2-3 cameras, log des associations). |
| `*_pytest.py` | Tests automatises (tracker, violations, campagne, compatibilite de classes). Lances par `python -m pytest -q` a la racine. |

## Fichiers de configuration

- `config.yaml` : cameras (URLs assainies), homographies, salles, zones.
- `config_real_zones.yaml` : configuration des zones reelles utilisee
  pour la session finale du rapport.
- `machines.yaml`, `ref_points_room1.yaml` : coordonnees machines et
  points de reference de calibration.
- `bytetrack.yaml` : parametres du tracker ByteTrack.
- `plan_rdc.png` : plan de sol de l'atelier (echelle 42,3 px/m, origine
  au coin haut-gauche).

## Regles importantes

- `geometry_fix.py` doit etre applique avant inference et homographie
  pour cam_03, cam_05 et cam_07 (fait par le pipeline).
- Le point projete est le bas-centre de la bounding box.
- Association inter-cameras uniquement entre cameras de la meme salle,
  et uniquement entre classes compatibles.
- `model.track(persist=True)` est obligatoire pour des IDs stables.
- `legacy/` contient les prototypes remplaces (voir `legacy/README.md`).

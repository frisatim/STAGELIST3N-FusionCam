# Guide de lecture du depot

Ce document donne l'ordre de lecture conseille pour comprendre le projet,
le role de chaque phase, et la correspondance entre les chapitres du
rapport de stage et les dossiers du depot.

## Question de recherche

Construire et evaluer un systeme de surveillance multi-cameras pour
detecter automatiquement l'entree d'une personne dans une zone
interdite (metrique TRD) et l'apparition d'un objet interdit dans
l'atelier (metrique TAD). Le travail vise une comparaison
experimentale rigoureuse entre une baseline mono-camera en coordonnees
image et une fusion multi-cameras projetee sur le plan sol, pas
seulement l'execution d'un modele de detection.

Hypothese centrale : la fusion doit etre plus pertinente
geometriquement pour les violations de zone, car elle raisonne sur le
sol reel en metres, mais elle depend fortement de la qualite des
calibrations, du tracking et de la fusion. Pour les objets interdits,
la fusion brute peut creer beaucoup de repetitions, d'ou la logique
d'alertes a deux niveaux : `weak` pour une camera seule, `confirmed`
pour une confirmation multi-camera.

## Logique generale

Le projet suit une progression en 4 phases. Chaque phase repond a une
question et fournit les briques de la suivante :

1. **Phase 1 - Infrastructure** : comment capter, enregistrer et annoter
   les 8 flux cameras RTSP de l'atelier. Produit les enregistrements et
   les ground truths utilises par toutes les evaluations.
2. **Phase 2 - Baseline mono-camera** : que vaut une detection classique
   en coordonnees image, camera par camera ? Entrainement des modeles
   (YOLOv8, YOLO11, RT-DETR) et evaluation TAD/TRD de reference.
3. **Phase 2.5 - Test live** : etape intermediaire, premier dashboard
   temps reel sur flux RTSP pour valider la faisabilite avant la fusion.
4. **Phase 3 - Fusion multi-cameras** : le coeur du projet. Calibration
   par homographie, projection des detections sur le plan sol, association
   inter-cameras, detection de violations en coordonnees metres, campagnes
   d'evaluation comparees a la Phase 2.
5. **Phase 4 - Reseau et latence** : l'architecture temps reel complete.
   Separation video / IA / metadonnees, mesure de latence par etape,
   benchmarks de transport (HTTP, WebSocket, MQTT), dashboard web.

## Ordre de lecture conseille

1. `README.md` (racine) : vue d'ensemble et commandes principales.
2. `docs/METRIQUES.md` : definitions TRD, TAD, FAR utilisees partout.
3. `docs/DONNEES.md` : layout des donnees et versions V1 a V4 des
   modeles et datasets.
4. Le `README.md` de chaque phase, dans l'ordre 1, 2, 2.5, 3, 4.
5. `docs/REPRODUCTION.md` : pour relancer les campagnes.

L'analyse detaillee des resultats et la lecture critique de l'interet
de la fusion sont dans le rapport de stage ; le depot conserve les
figures (`docs/figures/phase3_phase4_20260707/`) et les resumes CSV
(`reports/`).

## Correspondance rapport de stage / depot

| Chapitre du rapport | Contenu | Dossiers et fichiers principaux |
|---|---|---|
| 1. Presentation de l'organisme d'accueil | Contexte du stage | (pas de code) |
| 2. Contexte scientifique et problematique | Question de recherche, etat de l'art | ce document (section Question de recherche) |
| 3. Infrastructure d'acquisition et constitution du jeu de donnees | Cameras, enregistrement, annotation, datasets | `Phase_1_Infrastructure/`, `ground_truth/`, `docs/DONNEES.md` |
| 4. Evaluation mono-camera : reference (baseline) | Entrainement, evaluation TAD/TRD par camera | `Phase_2_Baseline_MonoCam/` (`train_models.py`, `evaluate_trd.py`, `evaluate_tad.py`) |
| 5. Approche multi-cameras a fusion au sol | Calibration, homographie, fusion, violations, campagnes | `Phase_3_Fusion_MultiCam/` (`calibration_tool_v2.py`, `fusion.py`, `violation_detector.py`, `run_recorded_campaign.py`), `reports/` |
| 6. Architecture temps reel : separation video / IA / metadonnees | Latence par etape, transport, dashboard, session finale 4 cameras | `Phase_4_Network_Latency/`, `Phase_3_Fusion_MultiCam/run_live_campaign.py`, `metadata_publisher.py`, `docs/REPRODUCTION.md` (section latence) |
| 7. Discussion | Interet de la fusion, limites | (rapport de stage ; figures dans `docs/figures/`, resumes CSV dans `reports/`) |
| 8. Enseignements / 9. Competences | Bilan personnel | (pas de code) |

Les figures du rapport (latence par etape, transport des metadonnees,
montee en charge, Phase 2 vs Phase 3, faux positifs) sont dans
`docs/figures/phase3_phase4_20260707/` et se regenerent avec
`python scripts/generate_phase3_phase4_figures.py`.

## Ou sont les donnees et les resultats complets

- Dans Git : ground truths legeres (`ground_truth/`), resumes CSV
  (`reports/`), figures (`docs/figures/`).
- Hors Git, dans le dossier externe `STAGELIST3N-FusionCam-data` :
  datasets, enregistrements video, poids entraines (V2 a V4, courbes
  d'entrainement et matrices de confusion incluses), engines TensorRT,
  rapports de campagnes complets dont la session finale 4 cameras
  30 minutes du rapport. Voir `docs/DONNEES.md`.

## Points d'attention pour la review du code

- Les cameras cam_03, cam_05 et cam_07 diffusent en 704x576 avec un
  mauvais aspect ratio : `geometry_fix.py` doit etre applique avant toute
  inference ou homographie (c'est fait par le pipeline).
- Le point projete au sol est le bas-centre de la bounding box, pas le
  centre.
- L'association inter-cameras n'a lieu qu'entre cameras de la meme salle,
  et exige des classes compatibles depuis le correctif class-aware.
- Les scripts `visual_check_*.py` et `demo_*.py` ouvrent des fenetres
  OpenCV ou des flux RTSP : ce sont des outils de validation visuelle,
  pas des tests automatises. Les tests pytest sont les `*_pytest.py`.

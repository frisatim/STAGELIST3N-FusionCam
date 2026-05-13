# Rapport d'analyse - Campagne Zone 1 du 11 mai 2026

## 1. Synthese executive

Cette campagne a traite les quatre cameras de la zone 1: `cam_02`, `cam_03`, `cam_05` et `cam_07`, sur les videos `*_20260506_131002.mp4`.

Les resultats exploitables de ce run concernent surtout la Phase 3: tracking ByteTrack, projection au sol, fusion multi-cameras, violation de zone, audit calibration et export des detections/alertes. La comparaison Phase 2 vs Phase 3 n'est pas exploitable dans ce run, car les scripts Phase 2 n'ont pas produit de CSV valides.

Chiffres globaux Phase 3:

| Indicateur | Valeur |
|---|---:|
| Lignes de detections exportees | 495 459 |
| Alertes exportees | 141 737 |
| Liens de fusion inter-cameras | 331 530 |
| Tracks ByteTrack exportes | 6 849 |
| Alertes audit calibration analysees | 283 474 |
| Modeles/configurations Phase 3 executes | 18 actifs + 2 RT-DETR engine ignores |

Conclusion principale: le pipeline Phase 3 fonctionne et produit des donnees riches, mais les alertes `forbidden_object` sont trop nombreuses et doivent etre dedupliquees avant interpretation scientifique. Pour les violations de zone personne, les meilleurs resultats TRD sont obtenus par `V3 yolov8s fp32_engine`, suivi de `V3 yolov8n fp32_engine`.

## 2. Validite du run

### Phase 3

La Phase 3 a bien produit les exports attendus:

- `phase3/detections.csv`
- `phase3/alerts.csv`
- `phase3/fusion_links.csv`
- `phase3/track_stability.csv`
- `phase3/phase3_trd.csv`
- `phase3/phase3_tad.csv`
- `audit/calibration_alert_audit.csv`
- `ablation/fusion_threshold_ablation.csv`

### Phase 2

Les resultats Phase 2 ne sont pas disponibles dans cette campagne. Le dossier `phase2/` ne contient aucun CSV exploitable.

Causes observees dans `logs/phase2.log`:

- `evaluate_tad.py` a plante sous Windows avec `UnicodeEncodeError` lors de l'affichage des caracteres de separation Unicode.
- `evaluate_trd.py` a cherche des cameras comme `cam_05_20260506_131002` dans le fichier de zones pixel, alors que la config Phase 2 contient des IDs comme `cam_03`, `cam_05`, `cam_07`.

Consequence: le fichier `comparison_phase2_phase3.csv` contient uniquement des lignes Phase 3. Il ne faut pas l'utiliser comme preuve comparative Phase 2 vs Phase 3 pour ce run.

## 3. Performance Phase 3 par modele

Chaque modele actif a traite 14 963 frames. Les engines RT-DETR FP32 ont ete ignores en Phase 3, car ils produisaient des bounding boxes invalides avec ByteTrack.

| Version | Modele | Format | Detections | Alertes | Liens fusion | G uniques | Switch G | Latence mediane ms | Latence p95 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| V3 | yolov8s | fp32_engine | 19 475 | 8 862 | 6 459 | 262 | 66 | 9.36 | 12.16 |
| V2 | yolo11n | fp32_engine | 32 574 | 11 586 | 24 623 | 225 | 49 | 11.51 | 16.87 |
| V2 | yolov8n | fp32_engine | 31 343 | 10 495 | 22 941 | 232 | 50 | 12.26 | 16.81 |
| V2 | yolo26n | fp32_engine | 20 215 | 1 063 | 16 647 | 120 | 25 | 12.37 | 16.13 |
| V3 | yolov8n | fp32_engine | 14 510 | 3 463 | 7 736 | 172 | 40 | 12.38 | 17.32 |
| V3 | yolo11s | fp32_engine | 18 865 | 4 801 | 10 726 | 238 | 39 | 12.66 | 16.31 |
| V2 | yolo11s | fp32_engine | 30 604 | 9 946 | 20 475 | 196 | 38 | 13.67 | 17.62 |
| V2 | yolo11n | pt | 41 098 | 10 001 | 28 815 | 228 | 54 | 21.04 | 29.47 |
| V2 | yolov8n | pt | 32 433 | 10 270 | 26 565 | 225 | 60 | 20.35 | 28.48 |
| V2 | yolo11s | pt | 41 991 | 9 108 | 26 313 | 171 | 51 | 23.16 | 32.20 |
| V2 | yolo26n | pt | 22 285 | 2 822 | 17 827 | 111 | 22 | 23.49 | 33.14 |
| V2 | yolov8s | pt | 23 915 | 1 864 | 23 401 | 204 | 37 | 19.61 | 27.86 |
| V3 | yolov8n | pt | 12 972 | 1 772 | 6 951 | 214 | 43 | 22.89 | 30.55 |
| V3 | yolo11s | pt | 23 262 | 9 355 | 9 716 | 271 | 38 | 22.27 | 32.76 |
| V3 | yolov8s | pt | 17 563 | 6 638 | 6 788 | 268 | 57 | 23.07 | 37.75 |
| V2 | rtdetr-l | pt | 46 967 | 18 017 | 32 184 | 582 | 73 | 46.21 | 62.58 |
| V3 | rtdetr-l | pt | 38 441 | 19 845 | 22 739 | 514 | 78 | 61.47 | 73.85 |

Observations:

- Les engines YOLO sont nettement plus rapides que les `.pt`.
- `V3 yolov8s fp32_engine` est le plus rapide: latence mediane 9.36 ms, p95 12.16 ms.
- `V2 yolo26n fp32_engine` produit beaucoup moins d'alertes que les autres modeles: 1 063 alertes seulement.
- `RT-DETR .pt` fonctionne sans crash, mais il est lent et genere beaucoup d'alertes et de `global_id` differents. Il n'est pas le meilleur choix pour la Phase 3 temps reel.
- `V2 yolov8s fp32_engine` a une latence max anormale de 573 098 ms. Sa mediane reste correcte, mais cette valeur max doit etre traitee comme anomalie de mesure ou blocage ponctuel.

## 4. Resultats TRD Phase 3

Le TRD mesure la detection d'entree de personne dans la zone interdite. Les meilleurs resultats sont classes ici par F1 moyen sur les quatre cameras.

| Rang | Modele | Matched / GT | FP | Precision | Recall | F1 | TRD median moyen |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | V3 yolov8s fp32_engine | 15 / 27 | 12 | 0.605 | 0.554 | 0.547 | 0.005 s |
| 2 | V3 yolov8n fp32_engine | 15 / 27 | 16 | 0.515 | 0.554 | 0.522 | 0.095 s |
| 3 | V2 yolov8s fp32_engine | 12 / 27 | 7 | 0.642 | 0.446 | 0.517 | 1.060 s |
| 4 | V3 yolov8n pt | 14 / 27 | 14 | 0.522 | 0.512 | 0.508 | 0.760 s |
| 5 | V3 yolov8s pt | 15 / 27 | 16 | 0.600 | 0.560 | 0.482 | 1.040 s |
| 6 | V3 yolo11s fp32_engine | 11 / 27 | 7 | 0.656 | 0.399 | 0.464 | -0.015 s |
| 7 | V2 yolov8n fp32_engine | 11 / 27 | 9 | 0.626 | 0.405 | 0.462 | 0.570 s |
| 8 | V2 yolo26n fp32_engine | 10 / 27 | 6 | 0.717 | 0.363 | 0.460 | -0.085 s |
| 9 | V2 rtdetr-l pt | 13 / 27 | 13 | 0.583 | 0.476 | 0.452 | 2.610 s |
| 10 | V2 yolo11n fp32_engine | 11 / 27 | 8 | 0.625 | 0.393 | 0.447 | 0.367 s |

Interpretation:

- Le meilleur compromis TRD est `V3 yolov8s fp32_engine`.
- `V3 yolov8n fp32_engine` est tres proche et reste un bon candidat.
- Les valeurs de TRD legerement negatives indiquent que certaines alertes sont detectees avant la frame GT annotee. Cela peut venir d'une annotation GT un peu tardive ou d'un seuil spatial plus sensible.
- Le recall reste limite: le meilleur modele ne detecte que 15 evenements sur 27. Il faut donc revoir les annotations GT, la logique de matching temporel et la sensibilite zone/personne.

## 5. Resultats TAD Phase 3

Le TAD objet mesure la detection d'objets interdits. Les resultats montrent un probleme clair: le systeme produit beaucoup trop d'alertes objet. Les FP sont tres eleves, ce qui fait chuter la precision.

| Rang | Modele | Matched / GT | FP | Precision | Recall | F1 | TAD median moyen |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | V2 yolo26n fp32_engine | 26 / 49 | 979 | 0.108 | 0.525 | 0.155 | 2.045 s |
| 2 | V2 yolo26n pt | 19 / 49 | 2 756 | 0.200 | 0.372 | 0.135 | 1.820 s |
| 3 | V3 yolov8n fp32_engine | 45 / 49 | 3 318 | 0.074 | 0.906 | 0.120 | 0.225 s |
| 4 | V2 yolo11n pt | 43 / 49 | 9 565 | 0.055 | 0.874 | 0.097 | 0.150 s |
| 5 | V3 yolov8n pt | 46 / 49 | 1 651 | 0.044 | 0.937 | 0.081 | 0.180 s |
| 6 | V2 yolov8s pt | 44 / 49 | 1 734 | 0.039 | 0.891 | 0.073 | 0.305 s |
| 7 | V3 yolov8s fp32_engine | 44 / 49 | 7 254 | 0.042 | 0.891 | 0.073 | 0.250 s |
| 8 | V2 yolov8s fp32_engine | 45 / 49 | 1 697 | 0.034 | 0.919 | 0.065 | 0.295 s |
| 9 | V2 yolov8n pt | 42 / 49 | 9 723 | 0.036 | 0.825 | 0.060 | 0.430 s |
| 10 | V3 yolo11s fp32_engine | 47 / 49 | 4 596 | 0.032 | 0.954 | 0.059 | 0.075 s |

Interpretation:

- `V2 yolo26n fp32_engine` est le meilleur selon F1 TAD dans ce run.
- `V3 yolo11s fp32_engine` et `V3 yolov8n pt` ont un excellent recall, mais trop de FP.
- Les metriques TAD Phase 3 doivent etre interpretees avec prudence: le systeme compte beaucoup de detections objet frame par frame. Il faut ajouter une deduplication temporelle/spatiale par objet avant d'en faire une metrique de rapport scientifique.

## 6. Alertes et faux positifs

Repartition des alertes:

| Type d'alerte | Nombre |
|---|---:|
| `forbidden_object` | 141 338 |
| `zone_violation_person` | 399 |

Classes detectees les plus frequentes:

| Classe | Detections |
|---|---:|
| personne | 349 133 |
| perceuse | 56 689 |
| pince | 51 977 |
| bouteille | 13 619 |
| marteau | 11 752 |
| metre | 5 017 |
| verre | 3 704 |
| cutter | 1 281 |
| scie | 943 |
| tournevis | 823 |

Interpretation:

- La majorite massive des alertes vient des objets interdits, pas des violations de zone personne.
- Le pipeline declenche actuellement une alerte `forbidden_object` pour de nombreuses detections successives. Cela explique les 141 338 alertes objet.
- Pour le rapport final, il faut distinguer:
  - detections brutes frame par frame;
  - alertes evenementielles dedupliquees.
- Sans deduplication, les valeurs de FP objet surestiment fortement le probleme.

## 7. Audit calibration

L'audit calibration a analyse 283 474 lignes d'alertes.

| Critere | Nombre |
|---|---:|
| `expected_zone_ok = True` | 283 462 |
| `expected_zone_ok = False` | 12 |
| `inside_any_zone = True` | 25 252 |
| `inside_any_zone = False` | 258 222 |

Les 12 lignes suspectes correspondent a des violations `zone_1` situees quasiment sur la bordure:

- distances a la zone la plus proche entre 0.0004 m et 0.0027 m;
- cameras concernees: `cam_02`, `cam_03`, `cam_07`;
- les lignes apparaissent souvent en double car le log contient a la fois `[WARN] Nouvelle violation détectée` et `[ALERT]`.

Interpretation:

- Les alertes de zone sont geometriquement coherentes.
- Les quelques alertes marquees suspectes sont des cas de bordure, pas des erreurs grossieres de calibration.
- La calibration zone 1 semble suffisamment coherente pour continuer les tests live, mais il faut inspecter visuellement les cas bordure.

## 8. Fusion multi-cameras et stabilite des IDs

Liens de fusion les plus nombreux:

| Modele | Liens fusion |
|---|---:|
| V2 rtdetr-l pt | 32 184 |
| V2 yolo11n pt | 28 815 |
| V2 yolov8n pt | 26 565 |
| V2 yolo11s pt | 26 313 |
| V2 yolo11n fp32_engine | 24 623 |
| V2 yolov8s pt | 23 401 |
| V2 yolov8n fp32_engine | 22 941 |
| V3 rtdetr-l pt | 22 739 |

Stabilite ByteTrack:

| Modele | Tracks | Age moyen | Age median | Tracks >= 80 frames | Tracks <= 5 frames | Switch G |
|---|---:|---:|---:|---:|---:|---:|
| V2 yolo11s pt | 319 | 131.6 | 13 | 75 | 113 | 51 |
| V2 yolo11n pt | 343 | 119.8 | 12 | 93 | 121 | 54 |
| V2 yolo26n pt | 207 | 107.7 | 17 | 63 | 78 | 22 |
| V2 yolo11n fp32_engine | 342 | 95.2 | 10 | 85 | 118 | 49 |
| V2 yolo11s fp32_engine | 319 | 95.9 | 12 | 76 | 115 | 38 |
| V2 yolo26n fp32_engine | 214 | 94.5 | 12 | 59 | 84 | 25 |
| V3 yolo11s pt | 398 | 58.4 | 11 | 72 | 136 | 38 |
| V3 yolov8n fp32_engine | 289 | 50.2 | 10 | 51 | 99 | 40 |

Interpretation:

- Les meilleurs ages moyens sont obtenus par les modeles V2, surtout `V2 yolo11s pt`, `V2 yolo11n pt` et `V2 yolo26n pt`.
- `V2 yolo26n pt` et `V2 yolo26n fp32_engine` ont peu de changements de global ID, ce qui est positif pour la fusion.
- Beaucoup de tracks ont un age <= 5 frames. Cela montre que le tracking reste fragmente, surtout avec les detections objet ou les cas difficiles.

## 9. Ablation du seuil D

Le fichier `ablation/fusion_threshold_ablation.csv` est en mode `operational_no_truth_id`, donc il ne mesure pas TP/FP/FN reels. Il donne seulement des indicateurs operationnels: nombre de liens predits, nombre de global IDs et changements de global IDs.

Pour les modeles V3 qui ont des lignes exploitables, l'augmentation de D augmente les liens predits et reduit le nombre de global IDs:

Exemple `V3 rtdetr-l pt`:

| D | Liens predits | Global IDs |
|---:|---:|---:|
| 50 cm | 5 356 | 67 |
| 100 cm | 6 000 | 48 |
| 150 cm | 6 132 | 41 |
| 200 cm | 6 147 | 35 |

Exemple `V3 yolov8n fp32_engine`:

| D | Liens predits | Global IDs |
|---:|---:|---:|
| 50 cm | 3 019 | 83 |
| 100 cm | 3 439 | 48 |
| 150 cm | 3 580 | 43 |
| 200 cm | 3 601 | 34 |

Interpretation:

- D = 50 cm est probablement trop strict: beaucoup de global IDs restent separes.
- D = 150 ou 200 cm fusionne davantage, mais augmente le risque de fausses associations quand deux personnes sont proches.
- D = 100 cm reste le meilleur compromis par defaut en l'absence de `truth_id`.
- Pour prouver scientifiquement le meilleur D, il faut un CSV de fusion annote avec `truth_id`.

Limite importante: le CSV d'ablation merge ne contient pas les colonnes `model/version/format`, ce qui rend l'analyse par modele difficile. Il faut corriger l'export pour le prochain run ou utiliser les CSV d'ablation dans chaque sous-dossier modele.

## 10. Choix recommande des modeles

### Si l'objectif prioritaire est TRD / violation personne

Choix recommande:

1. `V3 yolov8s fp32_engine`
2. `V3 yolov8n fp32_engine`
3. `V2 yolov8s fp32_engine`

Raison: meilleurs F1 TRD et latence faible.

### Si l'objectif prioritaire est de limiter les alertes objet

Choix recommande:

1. `V2 yolo26n fp32_engine`
2. `V2 yolov8s fp32_engine`
3. `V3 yolov8n pt`

Raison: moins d'alertes et meilleure precision relative sur TAD, meme si le recall varie.

### Si l'objectif prioritaire est stabilite ByteTrack/fusion

Choix recommande:

1. `V2 yolo26n pt`
2. `V2 yolo26n fp32_engine`
3. `V2 yolo11s fp32_engine`

Raison: moins de changements de global ID et tracks plus stables que plusieurs modeles V3.

### RT-DETR

`rtdetr-l.pt` fonctionne en Phase 3, mais n'est pas recommande pour le pipeline final:

- latence mediane elevee: 46.21 ms en V2, 61.47 ms en V3;
- beaucoup d'alertes;
- beaucoup de global IDs;
- beaucoup de tracks courts.

Les versions RT-DETR `.engine` ne doivent pas etre utilisees avec ByteTrack dans l'etat actuel.

## 11. Limites du run

1. Phase 2 absente: les CSV Phase 2 n'ont pas ete produits. La comparaison Phase 2 vs Phase 3 doit etre relancee apres correction.
2. TAD objet trop brut: les detections objet sont comptees frame par frame, ce qui gonfle les FP.
3. Ablation D non annotee: sans `truth_id`, elle ne prouve pas les fausses associations.
4. Les logs dupliquent certaines alertes: `[WARN]` et `[ALERT]` peuvent representer le meme evenement.
5. Les class IDs different entre V2 et V3: V2 encode `personne` comme classe 14, V3 comme classe 11. Les analyses doivent utiliser `class_name`, pas seulement `class_id`.
6. Certaines mesures de latence max sont aberrantes, probablement dues a des blocages ponctuels ou au chargement TensorRT.

## 12. Actions recommandees avant rapport final

1. Corriger Phase 2:
   - remplacer les caracteres Unicode problematiques dans `evaluate_tad.py`;
   - mapper les IDs GT `cam_XX_20260506_131002` vers les IDs camera config `cam_XX` pour les zones Phase 2.
2. Ajouter une deduplication des alertes objets:
   - regrouper par modele, camera, classe, zone spatiale et fenetre temporelle;
   - exporter `object_events.csv`.
3. Refaire l'ablation D avec un CSV annote `truth_id`.
4. Garder D = 100 cm comme seuil par defaut provisoire.
5. Pour les tests live, commencer avec:
   - `V3 yolov8s fp32_engine` pour TRD rapide;
   - `V2 yolo26n fp32_engine` pour TAD/alertes plus conservatrices.

## 13. Fichiers a transmettre pour le classeur Excel

Fichiers principaux:

- `phase3/summary.csv`
- `phase3/phase3_trd.csv`
- `phase3/phase3_tad.csv`
- `phase3/detections.csv`
- `phase3/alerts.csv`
- `phase3/fusion_links.csv`
- `phase3/track_stability.csv`
- `audit/calibration_alert_audit.csv`
- `ablation/fusion_threshold_ablation.csv`

Feuilles recommandees dans Excel:

1. `Résumé modèles`
2. `TRD Phase 3`
3. `TAD Phase 3`
4. `Alertes`
5. `Audit calibration`
6. `Fusion`
7. `Tracking stability`
8. `Ablation D`
9. `Limites du run`


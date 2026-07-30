# Metriques d'evaluation

Ce document definit les metriques utilisees dans tout le projet
(rapport, CSV de `reports/`, scripts d'evaluation). Les implementations
de reference sont `Phase_2_Baseline_MonoCam/evaluate_trd.py`,
`Phase_2_Baseline_MonoCam/evaluate_tad.py` et
`Phase_3_Fusion_MultiCam/campaign_utils.py`.

## Evenements evalues

L'evaluation est evenementielle, pas frame par frame : une violation
annotee dans la ground truth est un evenement, une alerte emise par le
systeme est une prediction. Les deux familles d'evenements sont
independantes et ne doivent pas etre melangees :

- **Violation de zone (personnes)** : une personne entre dans une zone
  interdite. Ground truth : `ground_truth/gt_people.json`.
- **Objet interdit (outils)** : un objet d'une classe interdite apparait
  dans la scene. Ground truth : `ground_truth/gt_objects_tad*.json`.

## Appariement alerte / ground truth

Une alerte est appariee a un evenement annote si elle tombe dans une
fenetre de tolerance de **10 secondes** (`TOLERANCE_SEC = 10.0` dans
`evaluate_trd.py`, `tolerance_s = 10.0` dans `campaign_utils.py`).
L'appariement est un couplage un-pour-un : un evenement annote ne peut
pas etre credite par deux alertes, une alerte ne peut pas couvrir deux
evenements.

- Alerte appariee : vrai positif (TP).
- Evenement annote sans alerte : faux negatif (FN).
- Alerte sans evenement : faux positif (FP).

### Regroupement des faux positifs

Les alertes non appariees temporellement proches sont regroupees en un
seul evenement FP si elles sont separees de moins de
**3 secondes** (`FP_MERGE_WINDOW_S = 3.0`). Sans ce regroupement, une
meme fausse detection persistante compterait un FP par frame et
ecraserait artificiellement la precision. Ce regroupement est applique
de la meme maniere en Phase 2 et en Phase 3
(`TAD_FP_MERGE_WINDOW_S = 3.0` dans `campaign_utils.py`) pour que la
comparaison soit equitable.

## TRD - Time to Restricted-zone Detection

Delai entre l'instant annote ou la personne franchit la limite de la
zone interdite et l'instant de l'alerte correspondante, en secondes.

- Rapporte en mediane et en p95 sur les evenements apparies.
- **Un TRD negatif est possible et normal** : l'alerte peut partir avant
  l'horodatage annote quand le systeme declenche a l'approche de la
  frontiere de zone (le point bas-centre projete touche la zone avant
  l'instant retenu par l'annotateur).

## TAD - Time to Abandoned-object Detection

Delai entre l'apparition annotee d'un objet interdit
(`trame_apparition` dans la ground truth) et la premiere alerte objet
de la classe correspondante, en secondes. Rapporte en mediane et p95.

## FAR - False Alarm Rate

Nombre d'evenements faux positifs (apres regroupement 3 s) divise par
la duree evaluee, exprime en **fausses alertes par heure**
(`far_per_hour` dans les CSV).

## Precision et rappel evenementiels

- precision = TP / (TP + FP)
- rappel = TP / (TP + FN)

Calcules sur les evenements apres appariement et regroupement des FP.
Attention en comparant des chiffres entre documents anciens et recents :
avant le correctif de regroupement des FP en Phase 3, la precision TAD
etait calculee avec un FP par alerte non appariee, ce qui donnait des
valeurs artificiellement catastrophiques (par exemple 0,12) non
comparables a la Phase 2.

## Alertes weak / confirmed (objets, Phase 3)

- `weak` : objet vu par une seule camera.
- `confirmed` : objet confirme par au moins `--object-min-camera-votes`
  cameras dans la fenetre temporelle de fusion.

Pour les personnes, la violation de zone est confirmee par un vote
multi-cameras controle par `person_zone_min_camera_ratio` /
`person_zone_min_camera_votes` (voir `violation_detector.py`).

## Metriques de fusion (Phase 3)

- `fusion_links` : nombre d'associations inter-cameras creees.
- `global_id_switches` : changements d'identite globale d'un track.
- Qualite d'association : evaluee sur un mini echantillon annote avec
  `evaluate_fusion_links.py` (colonnes TP/FP/FN/TN sur les liens).
- Seuil de distance d'association `D` au sol : etudie par ablation dans
  `reports/ablation/` (voir `ablate_fusion_threshold.py`).

## Latences (Phases 3 et 4)

- Latence d'inference : temps du forward modele par frame (mediane, p95).
- Latence de boucle complete : capture, inference, tracking, projection,
  fusion, verification de zone, publication des metadonnees (p95).
- Latence de transport : delai de livraison des metadonnees/alertes par
  canal (queue locale, HTTP POST, WebSocket, MQTT), mesure par
  `Phase_4_Network_Latency/alert_delivery_benchmark.py`.
- Decomposition par etape : `docs/REPRODUCTION.md` (section latence).

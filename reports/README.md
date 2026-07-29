# Reports (resultats synthetiques versionnes)

Ce dossier contient les exports legers des campagnes conserves dans Git :
resumes CSV, manifestes et un rapport d'analyse. Les CSV frame par frame
complets (detections, alertes, liens de fusion, traces de latence) et les
sorties de la session finale sont livres dans le dossier de donnees externe
(`docs/DATA_LAYOUT.md`).

Les definitions des metriques (TRD, TAD, FAR, precision/rappel evenementiels,
regroupement des faux positifs) sont dans `docs/METRIQUES.md`. La
signification des versions de modeles (V2, V3, V4) est dans
`docs/VERSIONS_MODELES.md`.

## AVERTISSEMENT de perimetre

La campagne `phase2_phase3_comparison/campaign_zone1_20260511_091303/` est une
**campagne intermediaire** (modeles V2/V3, avant les modeles finaux V4). Son
rapport d'analyse (`RAPPORT_ANALYSE_CAMPAGNE_ZONE1.md`) note que la partie
Phase 2 de ce run avait plante (erreurs d'encodage et de mapping d'identifiants
camera) ; le fichier `comparison_phase2_phase3.csv` livre ici contient bien des
lignes `phase2` obtenues apres corrections, mais le texte du rapport d'analyse
n'a pas ete mis a jour et decrit l'etat initial. Ce run sert donc a illustrer
la methodologie et la selection de modeles, pas les chiffres finaux.

**Les resultats finaux du rapport de stage** proviennent de la session live
finale (4 cameras, 30 minutes, yolov8s V4), livree dans le dossier de donnees
externe :
`STAGELIST3N-FusionCam-data/reports/Phase_3_Fusion_MultiCam/final_real_zones_4cam_30min/`
(voir `docs/REPRODUCTION.md`, section 1).

Note : les colonnes `source_csv` et les chemins des manifestes sont au format
Windows et pointent vers `Phase_3_Fusion_MultiCam/reports/...`, l'emplacement
d'origine des sorties de campagne ; les chemins sensibles ont ete remplaces
par `<PROJECT_ROOT>` (voir `docs/DATA_AND_SECURITY.md`).

## Contenu des sous-dossiers

### `phase2_phase3_comparison/campaign_zone1_20260511_091303/`

Campagne sur videos enregistrees (session `20260506_131002`, cameras cam_02,
cam_03, cam_05, cam_07 de la zone 1), produite par
`Phase_3_Fusion_MultiCam/run_recorded_campaign.py` :

- `manifest.json` : parametres exacts du run (mode `recorded`, cameras,
  `gt_id_map` cam_XX -> cam_XX_20260506_131002, GT TAD utilisee, videos,
  liste des 16 modeles V2/V3 avec chemins de poids).
- `RAPPORT_ANALYSE_CAMPAGNE_ZONE1.md` : analyse detaillee du run (voir
  l'avertissement ci-dessus sur sa partie Phase 2).
- `summary.csv` : resume Phase 3 par modele.
- `phase3_trd.csv` / `phase3_tad.csv` : metriques evenementielles Phase 3 par
  modele et par camera.
- `comparison_phase2_phase3.csv` : tableau long fusionnant les resultats
  Phase 2 et Phase 3.

### `live_smoke_tests/`

Trois smoke tests live du 12 mai 2026 (`run_live_campaign.py`, modele
V3 yolov8s fp32_engine, 4 cameras zone 1, 5 minutes, capture GStreamer h265).
Chaque run a son `manifest_*.json` et son `summary_*.csv` (memes colonnes que
`summary.csv` ci-dessous).

Le prefixe `vote1` / `vote2` correspond au parametre
`alerting.object_min_camera_votes` visible dans les manifestes :

- `vote1` : `object_min_camera_votes = 1`, une detection objet vue par une
  seule camera suffit a declencher une alerte.
- `vote2` : `object_min_camera_votes = 2`, une alerte objet `confirmed` exige
  au moins 2 cameras dans la fenetre de fusion (voir la section alertes
  weak/confirmed de `docs/METRIQUES.md`).

L'effet du vote est visible dans les resumes : 54 alertes en vote1 contre 15
et 10 alertes dans les deux runs vote2.

### `ablation/`

Etude du seuil de distance d'association au sol `D` de la fusion
multi-cameras (D dans {50, 100, 150, 200} cm). **Les deux CSV n'ont pas du
tout le meme statut** :

- `fusion_threshold_ablation.csv` : **donnees reelles, mode
  `operational_no_truth_id`**. Produit par `run_operational_ablation`
  (`campaign_utils.py`) : les detections personne enregistrees pendant la
  campagne du 20260511 sont rejouees dans `MultiCameraFusion` pour chaque
  valeur de D. Sans identite verite (`truth_id`), aucune notion de TP/FP
  n'est calculable : le CSV ne donne que des indicateurs operationnels
  (liens predits, global IDs, switches). Limites a connaitre :
  - le fichier concatene un bloc de 4 lignes (une par seuil) **par run de
    modele, sans colonne identifiant le modele** ; on ne peut re-associer un
    bloc a son modele que par recoupement de la colonne `alerts` avec
    `summary.csv` ;
  - les blocs des modeles V2 ont `frames = 0` ou `predicted_links = 0` :
    l'ablation filtre la classe personne par `class_id == 11`, or les modeles
    V2 encodent `personne` avec l'id 14 (voir `docs/VERSIONS_MODELES.md`) ;
    seuls les blocs V3 sont exploitables.
- `fusion_threshold_ablation_synthetic.csv` : **donnees synthetiques, a ne
  PAS citer comme des mesures reelles**. Produit par
  `ablate_fusion_threshold.py` en mode par defaut : un mini jeu de detections
  fabrique a la main (4 personnes fictives, 10 frames, ecarts de 0.42 a
  1.35 m) avec `truth_id` connu, ce qui permet de calculer TP /
  `fp_false_matches` / `fn_missed_matches` / TN, precision, rappel, F1,
  `false_match_rate`, `missed_match_rate`, `predicted_links`, `truth_links`.
  Il sert uniquement a illustrer le compromis attendu sur D (D petit = liens
  manques, D grand = fausses associations). Pour une preuve sur donnees
  reelles, il faudrait un CSV de detections annote avec `truth_id` (option
  `--input` du script).

## Dictionnaire des colonnes

Toutes les significations ci-dessous ont ete verifiees dans le code qui ecrit
les CSV (`campaign_utils.py`, `run_recorded_campaign.py`,
`ablate_fusion_threshold.py`, `evaluate_tad.py`, `evaluate_trd.py`).

### `summary.csv` et `summary_vote*.csv` (resume Phase 3 par modele)

Ecrit par `run_recorded_campaign.py` / `run_live_campaign.py` a partir de
`Phase3Campaign.summary()`.

| Colonne | Signification |
|---|---|
| `model_version` | Version d'entrainement du modele : V2, V3 ou V4 (`docs/VERSIONS_MODELES.md`). |
| `model` | Nom du modele (yolov8s, yolo11n, rtdetr-l...). |
| `format` | Format des poids : `pt` (PyTorch) ou `fp32_engine` (TensorRT FP32). |
| `frames` | Nombre de lots de frames synchronisees traites (un lot = 1 frame par camera). |
| `detections` | Nombre de lignes de detections exportees (une par track et par frame). |
| `alerts` | Nombre d'alertes emises (tous types : `zone_violation_person` + `forbidden_object`, niveaux weak et confirmed confondus). |
| `fusion_links` | Nombre de paires de detections inter-cameras partageant un meme `global_id` sur une meme frame (lignes de `fusion_links.csv`). |
| `unique_global_ids` | Nombre de `global_id` distincts attribues par la fusion. |
| `global_id_switches` | Somme des changements de `global_id` subis par les tracks mono-camera (instabilite d'identite globale). |
| `latency_mean_ms` / `latency_median_ms` / `latency_p95_ms` / `latency_max_ms` | Latence inference + tracking par frame et par camera (duree de `process_frame`), en millisecondes. Une ligne `0.0` partout avec `frames = 0` signale un modele saute (RT-DETR engine). |

### `phase3_trd.csv` / `phase3_tad.csv`

Ecrits par `compute_phase3_metrics` (`campaign_utils.py`), une ligne par
(modele, format, camera).

| Colonne | Signification |
|---|---|
| `model_version`, `model`, `format` | Comme dans `summary.csv`. |
| `camera` | Identifiant camera du pipeline (`cam_02`...). |
| `gt_camera` | Identifiant `id_camera` interroge dans la GT (via le `gt_id_map`, ex. `cam_02_20260506_131002`) ; voir `ground_truth/README.md`. |
| `trd_median` / `trd_p95` / `trd_mean` (ou `tad_*`) | Delai TRD (ou TAD) en secondes sur les evenements apparies ; un TRD negatif est normal (`docs/METRIQUES.md`). |
| `n_gt_events` | Nombre d'evenements annotes pour cette camera GT. |
| `n_matched` | Evenements apparies a une alerte (TP). |
| `n_missed` | Evenements sans alerte (FN). |
| `n_faux_positifs` | Alertes non appariees. Attention : pour le TAD elles sont regroupees par fenetre de 3 s (evenements FP) ; pour le TRD de ce fichier elles ne le sont pas (comptage brut). |
| `precision`, `recall`, `f1` | Metriques evenementielles calculees sur TP/FP/FN ci-dessus. |

### `comparison_phase2_phase3.csv`

Ecrit par `write_comparison` (`run_recorded_campaign.py`) : concatenation des
CSV Phase 2 (par camera et modele) et des lignes `phase3_trd.csv` /
`phase3_tad.csv`. Toutes les colonnes ne sont pas remplies pour toutes les
lignes.

| Colonne | Signification |
|---|---|
| `phase` | `phase2` (mono-camera pixels) ou `phase3` (fusion plan sol). |
| `task` | `tad` ou `trd`. **Le sens de `metric_*` en depend.** |
| `model_version` | Renseigne pour les lignes phase3 uniquement (les CSV Phase 2 ne portent pas la version ; elle est deductible du chemin `source_csv`, ex. `V2_fp32_engine`). |
| `model`, `format`, `camera` | Comme ci-dessus. |
| `gt_camera` | Lignes phase3 uniquement. |
| `metric_median` / `metric_p95` / `metric_mean` | Delai en secondes : TAD si `task = tad`, TRD si `task = trd`. |
| `precision`, `recall`, `f1`, `n_faux_positifs` | Metriques evenementielles de la tache. |
| `far_per_hour` | Lignes phase2 uniquement : evenements FP (apres regroupement 3 s) divises par la duree video, en fausses alertes par heure. |
| `fps` | Lignes phase2 uniquement : FPS d'inference pur (`fps_moyen` pour le TAD, `fps_inference` pour le TRD). |
| `latency_mean_ms` | Lignes phase2 uniquement : latence moyenne d'inference par frame, en ms. |
| `alerts_duplicated`, `global_id_switches` | Colonnes reservees du schema, **jamais remplies** dans cet export (vides sur les 256 lignes) ; pour les switches Phase 3, voir `summary.csv`. |
| `source_csv` | Chemin du CSV source de la ligne (permet de retrouver le run exact, la version et le format). |

### `ablation/fusion_threshold_ablation.csv` (reel, operationnel)

| Colonne | Signification |
|---|---|
| `threshold_cm` / `threshold_m` | Seuil de distance d'association D teste (centimetres / metres). |
| `frames` | Nombre de frames contenant au moins une detection personne projetee au sol rejouee dans la fusion. |
| `predicted_links` | Nombre de liens inter-cameras predits (un `global_id` vu par >= 2 cameras sur une frame). |
| `unique_global_ids` | Nombre de `global_id` distincts crees pendant le rejeu. |
| `global_id_switches` | Recopie du total du run d'origine (identique pour les 4 seuils d'un bloc ; ne varie pas avec D). |
| `alerts` | Recopie du nombre d'alertes du run d'origine (idem ; sert de cle de recoupement avec `summary.csv`). |
| `mode` | `operational_no_truth_id` : rejeu sans identite verite, donc pas de TP/FP. |

### `ablation/fusion_threshold_ablation_synthetic.csv` (synthetique)

Colonnes `threshold_cm`, `threshold_m`, `frames`, puis scoring des paires
inter-cameras contre le `truth_id` synthetique : `tp`, `fp_false_matches`
(fausses associations), `fn_missed_matches` (associations manquees), `tn`,
`precision`, `recall`, `f1`, `false_match_rate` (= FP / (FP + TN)),
`missed_match_rate` (= FN / (TP + FN)), `predicted_links`, `truth_links`,
`unique_global_ids`. Rappel : jeu de donnees fabrique, illustratif uniquement.

## Voir aussi

- `docs/METRIQUES.md` : definitions des metriques et fenetres de tolerance.
- `docs/VERSIONS_MODELES.md` : versions V2/V3/V4 et piege des `class_id`.
- `docs/REPRODUCTION.md` : rejouer une campagne et retrouver la session finale.
- `ground_truth/README.md` : fichiers de verite terrain et mapping `gt_id_map`.

# Ground truths (verites terrain)

Ce dossier contient les verites terrain evenementielles legeres, versionnees
dans Git, utilisees par les evaluations Phase 2 (`evaluate_trd.py`,
`evaluate_tad.py`) et par les campagnes Phase 3 (`campaign_utils.py`,
`run_recorded_campaign.py`). Les definitions des metriques calculees a partir
de ces fichiers (TRD, TAD, FAR, precision/rappel evenementiels, fenetres de
tolerance) sont dans `docs/METRIQUES.md` et ne sont pas repetees ici.

Tous les fichiers sont des listes JSON d'annotations : une entree = une
observation d'un evenement par une camera. Un meme evenement physique vu par
plusieurs cameras donne plusieurs entrees qui partagent le meme
`id_evenement`.

## Role de chaque fichier

| Fichier | Evenement | Role |
|---|---|---|
| `gt_people.json` | Violation de zone (personne) | Reference TRD. Une entree = l'instant ou une personne franchit la limite d'une zone interdite, vue par une camera. Lu par `evaluate_trd.py` et `compute_phase3_metrics` (`campaign_utils.py`). |
| `gt_objects_tad.json` | Apparition d'objet interdit | Reference TAD generale : annotations temporelles sur l'ensemble des sessions d'enregistrement (8 cameras, plus les sessions suffixees `_record2`, `_rec3.1`, `_rec3.2`). Lu par `evaluate_tad.py` et les campagnes en dernier recours (voir ordre de priorite ci-dessous). |
| `gt_objects_tad_dataset_objets_HD.json` | Apparition d'objet interdit | Reference TAD de la session `20260506_131002` (les 4 videos zone 1 : cam_02, cam_03, cam_05, cam_07). Copie conforme de `dataset_objets_HD/gt_objects_tad.json` (verifie identique), versionnee ici pour que le depot Git reste autonome. C'est la GT TAD effectivement utilisee par les campagnes zone 1. |
| `gt_objects.json` | Annotation bbox pour entrainement | N'est PAS une GT d'evaluation : ce sont les annotations bbox historiques des images de `dataset/`, consommees par `Phase_1_Infrastructure/convert_json_to_yolo.py` pour produire un dataset YOLO (images + labels normalises). Contient des fautes de frappe brutes (`amrteau`, `mailet`...) corrigees a la conversion. |
| `gt_objects_grouped.json` | Apparition d'objet interdit (groupee) | Sortie de `Phase_1_Infrastructure/auto_assign_global_ids.py` : les annotations objets sont regroupees en evenements par clustering temporel (meme salle + meme classe + ecart <= 5 s), avec ajout de `id_salle` et re-numerotation de `id_evenement` par salle. Aucun script d'evaluation ne le lit : c'est un artefact intermediaire conserve pour tracabilite. Il ne couvre que les identifiants camera sans suffixe (le script ne mappe que `cam_01` a `cam_08` vers une salle). |
| `ground_truth.json` | Violation de zone (personne) | Prototype historique (4 entrees, format `type`/`frame_id`/`timestamp_ms`). Aucun code actuel ne le lit ; conserve pour l'historique du format d'annotation initial. |

### Ordre de priorite dans le code

Les scripts cherchent les GT a plusieurs emplacements (premier existant gagne,
voir `first_existing_path` dans `campaign_utils.py` et `GT_PATH` dans les deux
evaluateurs Phase 2) :

- TRD : `gt_people.json` (racine projet), puis `ground_truth/gt_people.json`.
- TAD : `dataset_objets_HD/gt_objects_tad.json`, puis
  `ground_truth/gt_objects_tad_dataset_objets_HD.json`, puis
  `ground_truth/gt_objects_tad.json`. Remplacable par `--gt`
  (`evaluate_tad.py`) ou `--tad-gt` (campagnes).

## Dictionnaire des champs

| Champ | Type / unite | Signification |
|---|---|---|
| `id_camera` | chaine | Identifiant de la camera OU de la video annotee. Attention : deux conventions coexistent, voir le piege ci-dessous. |
| `id_evenement` | entier | Identifiant global d'evenement : les entrees de plusieurs cameras qui observent le meme evenement physique partagent la meme valeur. Dans `gt_objects_grouped.json`, la numerotation repart de 1 dans chaque salle. |
| `classe_objet` | chaine | Classe de l'objet interdit, en francais, saisie a la main. Peut contenir accents et fautes de frappe ; les evaluateurs normalisent accents et casse (`normalize_class`) mais ne corrigent pas les fautes (une classe mal orthographiee ne matchera jamais une prediction). |
| `zone_id` | chaine | Zone interdite concernee. Ancienne convention d'annotation : `A` ou `B` ; entrees de la session `20260506_131002` : `zone_1` (identifiant de zone de la config Phase 3). |
| `id_salle` | entier | Salle de la camera (`gt_objects_grouped.json` uniquement). Salle 1 = cam_03, cam_05, cam_07 ; salle 2 = cam_01, cam_02, cam_04, cam_06, cam_08. |
| `trame_apparition` | entier (index de frame, base 0) | Frame de la video ou l'objet apparait. C'est ce champ que le matching utilise (converti en secondes via le FPS : `trame / fps`, avec fps = 25 dans `compute_phase3_metrics`). |
| `horodatage_apparition` | flottant (millisecondes) | Meme instant en ms depuis le debut de la video (`trame * 1000 / fps`). Champ d'affichage et de tracabilite ; le matching passe par la trame. |
| `trame_violation` | entier (index de frame, base 0) | Frame ou la personne franchit la limite de la zone interdite (`gt_people.json`). |
| `horodatage_violation` | flottant (millisecondes) | Meme instant en ms. |
| `bbox` | `[x, y, w, h]` en pixels | `gt_objects.json` uniquement. `x`, `y` = coin superieur gauche ; `w`, `h` = largeur et hauteur. Format verifie dans `convert_json_to_yolo.py` (conversion YOLO : `x_center = (x + w/2) / largeur_image`). |
| `type` | chaine | `ground_truth.json` (legacy) uniquement : `person_violation`. |
| `frame_id` | entier (index de frame) | Legacy, equivalent de `trame_violation`. |
| `timestamp_ms` | flottant (millisecondes) | Legacy, equivalent de `horodatage_violation`. |

## PIEGE : deux conventions d'`id_camera`

Deux familles d'identifiants coexistent :

1. **Identifiant camera de base** : `cam_01` ... `cam_08`, eventuellement
   suffixe par le nom de la session (`cam_01_record2`, `cam_03_rec3.1`).
   Utilise dans `gt_people.json` (266 entrees sur 293), `gt_objects.json`,
   `gt_objects_tad.json`, `gt_objects_grouped.json`.
2. **Identifiant estampille session** : `cam_02_20260506_131002`.
   Utilise dans `gt_objects_tad_dataset_objets_HD.json` (et les 27 entrees
   `zone_1` de `gt_people.json`). Il est derive automatiquement du nom du
   fichier video par `extract_camera_id` dans `annotation_tool.py`
   (`Camera_2_2.3_20260506_131002.mp4` -> `cam_02_20260506_131002`).

Le pipeline Phase 3 fait tourner les cameras sous leur identifiant de base
(`cam_02`...), mais doit interroger la GT avec l'identifiant estampille. Le
pont entre les deux est le **gt_id_map** :

- Par defaut, `ZONE1_GT_ID_MAP` (`campaign_utils.py`) mappe
  `cam_XX -> cam_XX_20260506_131002` pour les 4 cameras de la zone 1.
- L'option `--gt-id-map` de `run_recorded_campaign.py` (lignes ~95-102)
  permet de remplacer ce mapping, au format
  `cam_02=cam_02_live01,cam_03=cam_03_live01`. Les valeurs doivent
  correspondre exactement au champ `id_camera` des JSON de GT ; sinon
  `precheck_campaign_inputs` arrete la campagne avec
  `[ERREUR] GT TRD manquante pour: ...`.

Ce mapping apparait dans le `manifest.json` de chaque campagne (cle
`gt_id_map`) et dans la colonne `gt_camera` des CSV `phase3_trd.csv` /
`phase3_tad.csv`.

## Comment ces fichiers ont ete produits

Outils dans `Phase_1_Infrastructure/` :

- `annotation_tool.py` : outil interactif d'annotation sur les videos
  enregistrees. Touche `V` : marque une violation personne (demande
  `id_evenement` global + zone) et ecrit dans `gt_people.json` a la racine du
  projet. Touche `O` : marque une apparition d'objet (demande classe +
  `id_evenement`) et ecrit dans `<output-dir>/gt_objects_tad.json` (defaut :
  `dataset_objets_HD/`). Touche `B` : dessine une bbox et ecrit directement
  images + labels YOLO (sans passer par un JSON).
- `migrate_annotations.py` : migration one-shot de `gt_people.json` pour
  ajouter `zone_id` et corriger l'`id_evenement` global des anciennes entrees
  (interactif, les entrees deja migrees sont ignorees).
- `auto_assign_global_ids.py` : lit `gt_objects_tad.json`, regroupe les
  apparitions en evenements (meme salle + meme classe + ecart <= 5 s, une
  camera au plus par evenement) et sauvegarde `gt_objects_grouped.json` sans
  modifier l'original.
- `convert_json_to_yolo.py` : consomme `gt_objects.json` (annotations bbox
  historiques) pour generer un dataset YOLO ; corrige les fautes de frappe de
  classes a la volee.

Les copies presentes dans ce dossier ont ete rassemblees ici pour la
livraison (voir `docs/DATA_LAYOUT.md`, section Ground truths) : les scripts
utilisent d'abord les chemins historiques (racine projet,
`dataset_objets_HD/`) et retombent sur `ground_truth/` quand le depot Git est
utilise seul.

# Versions des modeles et des datasets (V1 a V4)

Les scripts et les CSV font reference a des versions V2, V3, V4. Ce
document explique ce que recouvre chaque version. Les poids sont livres
dans le dossier externe `STAGELIST3N-FusionCam-data/models/` (voir
`docs/DATA_LAYOUT.md`), chaque dossier de modele contenant les poids
(`best.pt`, et selon les cas `best.onnx` / `best.engine`), la courbe
d'entrainement `results.csv`/`results.png` et les matrices de confusion.

## Classes

Les datasets finaux comptent 12 classes : 11 outils (marteau, niveau a
bulle, scie, verre, perceuse, bouteille, pince, cutter, metre,
tournevis, cle allen) et `personne`.

Attention : **les identifiants de classes different entre versions**.
`personne` porte l'id 14 dans les modeles V2 (ancien jeu de classes) et
l'id 11 dans les modeles V3 et V4. C'est la raison du champ `class_id: 11`
dans les metadonnees JSONL de la Phase 4, et un piege connu si on
compare des sorties brutes V2 et V3/V4.

## Historique des versions

- **V1** : tout premiers essais d'entrainement, non utilises dans le
  rapport. Conserves uniquement en local.
- **V2** : premiere campagne d'entrainement complete (yolov8n, yolov8s,
  yolo11n, yolo11s, yolo26n, rtdetr-l) sur les premieres annotations.
  Ancien jeu de classes (`personne` = 14). Utilises dans les premieres
  campagnes comparatives et pour l'evaluation TAD detaillee du chapitre
  baseline du rapport (courbes TAD cam_07, poids `.pt`).
- **V3** : re-entrainement sur annotations consolidees, jeu de classes
  final (`personne` = 11), 50 epochs, imgsz 960. Modeles : yolov8n,
  yolov8s, yolo11s, rtdetr-l.
- **V4** : version finale utilisee pour la session finale du rapport.
  Dataset `dataset_objets_V4`, 80 epochs, imgsz 960, avec deux
  variantes :
  - `person_objects` : detection personnes + objets (variante utilisee
    par le pipeline Phase 3 complet) ;
  - `objects_only` : objets seuls (pour isoler la detection d'outils).

## Datasets associes

Livres dans `STAGELIST3N-FusionCam-data/datasets/` :

- `dataset/` : dataset historique (annotations initiales).
- `dataset_objets_HD/` : annotations 720p des objets interdits, sert
  aussi de reference TAD (`gt_objects_tad_dataset_objets_HD.json`).
- `dataset_objets_V4/` : dataset final des entrainements V4
  (`train_models.py` pointe dessus par defaut).

## Formats de poids

- `.pt` : poids PyTorch, source de verite portable. A privilegier sur
  une nouvelle machine.
- `.onnx` : export intermediaire (`export_onnx.py`).
- `.engine` : engine TensorRT compile pour le GPU qui l'a genere
  (`fp32_engine` dans les campagnes). Non portable : si l'engine ne se
  charge pas sur un autre GPU, le regenerer depuis le `.pt` avec
  `Phase_2_Baseline_MonoCam/export_onnx.py`.

## Selection dans les campagnes

`run_recorded_campaign.py` et `run_live_campaign.py` selectionnent les
modeles par `--versions` (ou l'alias `--dataset-version`), `--models` et
`--formats`. Les poids sont cherches dans
`Phase_2_Baseline_MonoCam/Modelstrained/<version>/...` (jonction vers le
dossier de donnees externe, creee par
`scripts/link_external_data_windows.ps1` ou montee par Docker Compose).
Exemple pour la configuration finale du rapport :

```bash
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py \
  --dataset-version V4 --models yolov8s --formats pt \
  --no-display --phase2-imgsz 960
```

# Donnees, modeles et securite

Le depot Git reste leger et clonable rapidement : il contient le code,
les ground truths JSON legeres, les configurations de calibration
assainies, les petits resumes CSV et la documentation. Les videos,
datasets complets, poids entraines et gros rapports vivent dans un
dossier externe standard :

```text
delivery/
  STAGELIST3N-FusionCam/              # depot Git
  STAGELIST3N-FusionCam-data/         # donnees lourdes, hors Git
```

## 1. Layout du dossier de donnees externe

```text
STAGELIST3N-FusionCam-data/
  datasets/
    dataset/                          # dataset personnes, si utilise
    dataset_objets_HD/                # dataset objets HD + gt_objects_tad.json
      gt_objects_tad.json
      data.yaml
      images/
      labels/
    dataset_objets_V4/                # dataset final V4
      data.yaml
      images/
      labels/
  recordings/
    recordings/
      Camera_2_2.3_20260506_131002.mp4
      Camera_3_2.4_20260506_131002.mp4
      Camera_5_2.6_20260506_131002.mp4
      Camera_7_2.11_20260506_131002.mp4
  models/
    V2/
    V3/
    V4/
      person_objects/
        yolov8s/
          weights/
            best.pt
            best.engine              # optionnel, machine-dependent
  reports/
    Phase_3_Fusion_MultiCam/
  exports/
```

Creer le squelette depuis la racine du repo :

```powershell
python scripts\prepare_delivery_layout.py --data-dir ..\STAGELIST3N-FusionCam-data
```

Le script cree l'arborescence et copie les GT legeres utiles dans le
dossier data lorsque c'est possible. Copier ensuite les datasets,
enregistrements, modeles et rapports (helper USB :
`scripts/copy_heavy_data_from_usb.ps1`, voir `docs/INSTALLATION.md`).

## 2. Chemins legacy et jonctions

Les scripts de recherche utilisent encore des chemins courts dans le
repo. Le dossier data externe y est relie par des jonctions Windows ou
par des volumes Docker :

```text
dataset_objets_HD/                         -> ../STAGELIST3N-FusionCam-data/datasets/dataset_objets_HD
dataset_objets_V4/                         -> ../STAGELIST3N-FusionCam-data/datasets/dataset_objets_V4
recordings/                                -> ../STAGELIST3N-FusionCam-data/recordings
Phase_2_Baseline_MonoCam/Modelstrained/    -> ../STAGELIST3N-FusionCam-data/models
Phase_3_Fusion_MultiCam/reports/           -> ../STAGELIST3N-FusionCam-data/reports/Phase_3_Fusion_MultiCam
```

Sur Windows :

```powershell
.\scripts\link_external_data_windows.ps1 -DataDir ..\STAGELIST3N-FusionCam-data
```

Dans Docker, `docker-compose.yml` monte automatiquement les memes
dossiers dans les chemins legacy du conteneur.

## 3. Ground truths

Les GT legeres sont versionnees dans Git :

```text
ground_truth/gt_people.json
ground_truth/gt_objects_tad_dataset_objets_HD.json
ground_truth/gt_objects_tad.json
```

Les scripts Phase 2 / Phase 3 utilisent ces fichiers comme fallback si
les anciens chemins (`gt_people.json` a la racine,
`dataset_objets_HD/gt_objects_tad.json`) ne sont pas presents.

## 4. Contenu minimal pour rejouer les campagnes

Pour refaire les tests V4 sur videos, le dossier data doit contenir au
minimum :

```text
recordings/recordings/Camera_2_2.3_20260506_131002.mp4
recordings/recordings/Camera_3_2.4_20260506_131002.mp4
recordings/recordings/Camera_5_2.6_20260506_131002.mp4
recordings/recordings/Camera_7_2.11_20260506_131002.mp4
models/V4/person_objects/yolov8s/weights/best.pt
models/V4/person_objects/yolov8s/weights/best.engine       # optionnel
datasets/dataset_objets_HD/gt_objects_tad.json
```

Les `.engine` sont optionnels : s'ils ne se chargent pas sur une autre
machine, repartir du `.pt` et regenerer l'engine localement.

## 5. Verification avant un run

Toujours lancer :

```powershell
python scripts\verify_data_layout.py --data-dir ..\STAGELIST3N-FusionCam-data --model-version V4 --model yolov8s
```

Ajouter `--require-engine` si le test doit utiliser TensorRT. Si la
verification echoue, la correction normale sur Windows est de recreer
les jonctions (`scripts\link_external_data_windows.ps1`).

## 6. Versions des modeles et des datasets (V1 a V4)

Les scripts et les CSV font reference a des versions V2, V3, V4. Les
poids sont livres dans `STAGELIST3N-FusionCam-data/models/`, chaque
dossier de modele contenant les poids (`best.pt`, et selon les cas
`best.onnx` / `best.engine`), la courbe d'entrainement
`results.csv`/`results.png` et les matrices de confusion.

### Classes

Les datasets finaux comptent 12 classes : 11 outils (marteau, niveau a
bulle, scie, verre, perceuse, bouteille, pince, cutter, metre,
tournevis, cle allen) et `personne`.

Attention : **les identifiants de classes different entre versions**.
`personne` porte l'id 14 dans les modeles V2 (ancien jeu de classes) et
l'id 11 dans les modeles V3 et V4. C'est la raison du champ
`class_id: 11` dans les metadonnees JSONL de la Phase 4, et un piege
connu si on compare des sorties brutes V2 et V3/V4.

### Historique des versions

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

### Datasets associes

Livres dans `STAGELIST3N-FusionCam-data/datasets/` :

- `dataset/` : dataset historique (annotations initiales).
- `dataset_objets_HD/` : annotations 720p des objets interdits, sert
  aussi de reference TAD (`gt_objects_tad_dataset_objets_HD.json`).
- `dataset_objets_V4/` : dataset final des entrainements V4
  (`train_models.py` pointe dessus par defaut).

### Formats de poids

- `.pt` : poids PyTorch, source de verite portable. A privilegier sur
  une nouvelle machine.
- `.onnx` : export intermediaire (`export_onnx.py`).
- `.engine` : engine TensorRT compile pour le GPU qui l'a genere
  (`fp32_engine` dans les campagnes). Non portable : si l'engine ne se
  charge pas sur un autre GPU, le regenerer depuis le `.pt` avec
  `Phase_2_Baseline_MonoCam/export_onnx.py`.

### Selection dans les campagnes

`run_recorded_campaign.py` et `run_live_campaign.py` selectionnent les
modeles par `--versions` (ou l'alias `--dataset-version`), `--models`
et `--formats`. Les poids sont cherches dans
`Phase_2_Baseline_MonoCam/Modelstrained/<version>/...` (jonction vers
le dossier de donnees externe, ou montage Docker).

## 7. Donnees exclues de Git et assainissement

Volontairement exclus du depot :

- videos RTSP et videos de test ;
- images et labels des datasets complets ;
- poids de modeles `.pt`, `.pth`, engines TensorRT `.engine` ;
- exports frame-par-frame volumineux et logs complets ;
- vraies URLs RTSP et identifiants camera.

Raison : GitHub devient vite trop lourd, les engines TensorRT ne sont
pas portables, et les videos/URLs RTSP peuvent contenir des donnees
sensibles. Ces fichiers sont livres hors Git dans
`STAGELIST3N-FusionCam-data`.

Les valeurs sensibles ont ete remplacees par des placeholders dans le
code, les configurations et la documentation (y compris
`docs/archive/`) :

```text
<USER>
<PASSWORD>
<CAMERA_IP>
<CAMERA_NET>
<GATEWAY_IP>
<SERVER_IP>
```

Avant d'executer les scripts sur une installation reelle, creer une
configuration locale non versionnee, par exemple `config.local.yaml`,
et y renseigner les vraies URLs RTSP.

### Verification avant publication

Avant de rendre public un nouveau snapshot du depot :

```powershell
git status
git grep -n -I -E "F[F]CA|172\.16\.|rtsp://admin[:][^<]|github_p[a]t|gh[p]_|\bs[k]-"
git ls-files | Select-String -Pattern "\.pt$|\.engine$|\.onnx$|\.mp4$|\.mkv$|\.avi$"
```

Resultat attendu : aucune sortie pour le `git grep`, et aucun gros
artefact modele/video suivi par Git.

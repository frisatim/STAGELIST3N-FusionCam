# Architecture officielle des donnees

Ce projet garde volontairement le code et les donnees lourdes separes.

Le depot Git doit rester clonable rapidement. Les videos, datasets complets,
modeles entraines et gros rapports doivent etre places dans un dossier externe
standard :

```text
STAGELIST3N-FusionCam-data/
```

Les scripts historiques utilisent encore des chemins courts dans le repo
(`recordings/`, `dataset_objets_HD/`, `Phase_2_Baseline_MonoCam/Modelstrained/`).
Pour eviter de modifier tous les scripts de recherche, le dossier data externe
est relie a ces chemins par des jonctions Windows ou par des volumes Docker.

## Vue globale

```text
delivery/
  STAGELIST3N-FusionCam/              # depot Git
  STAGELIST3N-FusionCam-data/         # donnees lourdes, hors Git
```

Le dossier Git contient :

```text
STAGELIST3N-FusionCam/
  Phase_1_Infrastructure/
  Phase_2_Baseline_MonoCam/
  Phase_3_Fusion_MultiCam/
  Phase_4_Network_Latency/
  ground_truth/                       # GT legeres versionnees
  reports/                            # resultats legers versionnes
  docs/
  scripts/
  docker-compose.yml
  requirements.txt
```

Le dossier data externe doit contenir :

```text
STAGELIST3N-FusionCam-data/
  datasets/
    dataset/                          # dataset personnes, si utilise
    dataset_objets_HD/                # dataset objets HD + gt_objects_tad.json
      gt_objects_tad.json
      data.yaml
      images/
      labels/
    dataset_objets_V4/                # nouveau dataset V4
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

## Chemins legacy attendus par les scripts

Les scripts de recherche utilisent encore ces chemins :

```text
dataset/
dataset_objets_HD/
dataset_objets_V4/
recordings/
Phase_2_Baseline_MonoCam/Modelstrained/
Phase_3_Fusion_MultiCam/reports/
```

Sur Windows, ces chemins doivent pointer vers le dossier data externe via :

```powershell
.\scripts\link_external_data_windows.ps1 -DataDir ..\STAGELIST3N-FusionCam-data
```

Apres cette commande, les scripts voient :

```text
dataset_objets_HD/                         -> ../STAGELIST3N-FusionCam-data/datasets/dataset_objets_HD
dataset_objets_V4/                         -> ../STAGELIST3N-FusionCam-data/datasets/dataset_objets_V4
recordings/                                -> ../STAGELIST3N-FusionCam-data/recordings
Phase_2_Baseline_MonoCam/Modelstrained/    -> ../STAGELIST3N-FusionCam-data/models
Phase_3_Fusion_MultiCam/reports/           -> ../STAGELIST3N-FusionCam-data/reports/Phase_3_Fusion_MultiCam
```

Dans Docker, `docker-compose.yml` monte automatiquement les memes dossiers dans
les chemins legacy du conteneur.

## Ground truths

Les GT legeres sont versionnees dans Git :

```text
ground_truth/gt_people.json
ground_truth/gt_objects_tad_dataset_objets_HD.json
ground_truth/gt_objects_tad.json
```

Les scripts Phase 2 / Phase 3 utilisent maintenant ces fichiers comme fallback
si les anciens chemins ne sont pas presents :

```text
gt_people.json                          # ancien chemin racine
ground_truth/gt_people.json             # fallback Git

dataset_objets_HD/gt_objects_tad.json   # ancien chemin dataset
ground_truth/gt_objects_tad_*.json      # fallback Git
```

Pour les tuteurs, il faut donc au minimum :

- le repo Git ;
- `ground_truth/` deja inclus dans Git ;
- les videos recorded si on veut tester les campagnes offline ;
- les modeles `.pt` et optionnellement `.engine`.

## Creation du dossier data externe

Depuis la racine du repo :

```powershell
python scripts\prepare_delivery_layout.py --data-dir ..\STAGELIST3N-FusionCam-data
```

Le script cree l'arborescence et copie les GT legeres utiles dans le dossier
data lorsque c'est possible.

## Verification avant un run

Toujours lancer :

```powershell
python scripts\verify_data_layout.py --data-dir ..\STAGELIST3N-FusionCam-data --model-version V4 --model yolov8s
```

Si le test doit utiliser TensorRT :

```powershell
python scripts\verify_data_layout.py --data-dir ..\STAGELIST3N-FusionCam-data --model-version V4 --model yolov8s --require-engine
```

Si la verification echoue, la correction normale sur Windows est :

```powershell
.\scripts\link_external_data_windows.ps1 -DataDir ..\STAGELIST3N-FusionCam-data
```

## Commande recorded minimale

Avec poids `.pt` :

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats pt --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960
```

Avec TensorRT :

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats fp32_engine --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960
```

## Commande Docker minimale

Preparer `.env` :

```text
FUSIONCAM_DATA_DIR=../STAGELIST3N-FusionCam-data
```

Puis :

```bash
docker compose run --rm fusioncam \
  python3 scripts/verify_data_layout.py \
    --data-dir /workspace/data \
    --model-version V4 \
    --model yolov8s
```

Run recorded :

```bash
docker compose run --rm fusioncam \
  python3 Phase_3_Fusion_MultiCam/run_recorded_campaign.py \
    --dataset-version V4 \
    --models yolov8s \
    --formats pt \
    --no-display \
    --device cuda:0 \
    --phase2-device gpu \
    --phase2-imgsz 960
```

## Ce qu'il faut fournir aux tuteurs

Livraison recommandee :

```text
STAGELIST3N-FusionCam/                  # repo Git
STAGELIST3N-FusionCam-data/             # donnees lourdes privees
```

Dans `STAGELIST3N-FusionCam-data`, inclure au minimum pour refaire les tests V4
sur videos :

```text
recordings/recordings/Camera_2_2.3_20260506_131002.mp4
recordings/recordings/Camera_3_2.4_20260506_131002.mp4
recordings/recordings/Camera_5_2.6_20260506_131002.mp4
recordings/recordings/Camera_7_2.11_20260506_131002.mp4
models/V4/person_objects/yolov8s/weights/best.pt
models/V4/person_objects/yolov8s/weights/best.engine       # optionnel
datasets/dataset_objets_HD/gt_objects_tad.json
```

Les `.engine` sont optionnels pour la portabilite. Si un `.engine` ne marche pas
sur une autre machine, repartir du `.pt` et regenerer l'engine localement.

## Pourquoi ne pas tout mettre dans Git

Ne pas versionner :

- videos ;
- datasets complets ;
- modeles `.pt` ;
- engines `.engine` ;
- rapports frame-par-frame volumineux ;
- URLs RTSP et identifiants camera.

Raison :

- GitHub devient vite trop lourd ;
- les engines TensorRT ne sont pas toujours portables ;
- les videos et RTSP peuvent contenir des donnees sensibles ;
- Docker peut monter ces donnees sans les mettre dans l'image.


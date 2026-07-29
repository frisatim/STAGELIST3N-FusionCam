# Checklist de livraison finale

Cette checklist decrit le package de livraison propre a donner aux tuteurs ou
a installer sur une nouvelle machine de test.

## 1. Depot

Livrer le depot Git :

```text
STAGELIST3N-FusionCam/
```

Le depot contient le code, les ground truths legeres, les petits resumes CSV,
la documentation, les fichiers Docker et les scripts d'installation.

La reference des versions d'environnement (GPU, CUDA, Python, Ultralytics)
est documentee dans `docs/ENVIRONNEMENT_REFERENCE.md` et
`requirements.lock.txt`.

Ne pas ajouter ces fichiers a Git :

- vraies URLs RTSP, mots de passe camera ou adresses IP internes ;
- datasets ;
- enregistrements ;
- poids `.pt` entraines ;
- fichiers TensorRT `.engine` ;
- gros rapports et logs generes.

## 2. Dossier de donnees lourdes

Livrer le dossier de donnees lourdes separement, par exemple sur une cle USB :

```text
STAGELIST3N-FusionCam-data/
  datasets/
    dataset/
    dataset_objets_HD/
    dataset_objets_V4/
  recordings/
    recordings/
      Camera_*.mp4
  models/
    V2/
    V3/
    V4/
  reports/
    Phase_3_Fusion_MultiCam/
  exports/
```

Exemple de modeles importants :

```text
models/V4/person_objects/yolov8s/weights/best.pt
models/V4/person_objects/yolov8s/weights/best.engine
```

Si des fichiers TensorRT `.engine` echouent sur un autre GPU, les regenerer
depuis les `.pt` sur la machine cible.

## 3. Nouvelle machine Windows

Cloner le repo :

```powershell
git clone https://github.com/frisatim/STAGELIST3N-FusionCam.git
cd STAGELIST3N-FusionCam
```

Lancer le setup :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -InstallOptionalNetwork
```

Copier les donnees lourdes depuis l'USB :

```powershell
.\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\STAGELIST3N-FusionCam-data
```

Creer les jonctions natives Windows :

```powershell
.\scripts\link_external_data_windows.ps1
```

Activer Python :

```powershell
& "$env:USERPROFILE\Desktop\aivenv\Scripts\Activate.ps1"
```

## 4. Machine Docker

Creer `.env` :

```powershell
Copy-Item .env.example .env
```

Build (utiliser `docker compose`, Docker Compose v2 ; voir
`docs/DOCKER_DELIVERY.md`) :

```powershell
docker compose build fusioncam
```

Ouvrir un shell :

```powershell
docker compose run --rm fusioncam bash
```

Test de fumee dans Docker :

```bash
python3 -m pytest Phase_3_Fusion_MultiCam/test_campaign_utils_pytest.py -q
```

## 5. Commandes de validation

Python/GPU :

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "from ultralytics import YOLO; YOLO(r'Phase_2_Baseline_MonoCam\Modelstrained\V4\person_objects\yolov8s\weights\best.pt'); print('model ok')"
```

Test de fumee Phase 3 recorded :

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats pt --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960
```

Test de fumee performance Phase 3 live :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_07 --duration-min 2 --device cuda:0 --no-display --no-record-video
```

Utiliser `--formats fp32_engine` seulement apres avoir confirme que l'engine
TensorRT local se charge sur le GPU cible.

## 6. Verification securite avant publication

Avant de rendre public un nouveau snapshot du depot, verifier :

```powershell
git status
git grep -n -I -E "F[F]CA|172\.16\.|rtsp://admin[:][^<]|github_p[a]t|gh[p]_|\bs[k]-"
git ls-files | Select-String -Pattern "\.pt$|\.engine$|\.onnx$|\.mp4$|\.mkv$|\.avi$"
```

Resultat attendu pour la commande `git grep` : aucune sortie sur un depot
propre.

Resultat attendu global : pas de vrais identifiants, pas d'IP internes, pas de
chemins locaux prives et pas de gros artefacts modeles/videos suivis par Git.

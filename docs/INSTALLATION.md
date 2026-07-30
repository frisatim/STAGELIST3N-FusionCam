# Installation

Deux voies d'installation sont possibles : Windows natif (la voie
utilisee pendant le stage) ou Docker. Dans les deux cas, les donnees
lourdes (datasets, videos, poids) vivent dans un dossier externe
`STAGELIST3N-FusionCam-data` decrit dans `docs/DONNEES.md`.

## 1. Environnement de reference

Environnement exact utilise pour les entrainements et les campagnes
finales du rapport. En cas d'ecart de resultats lors d'une
reproduction, comparer d'abord les versions ci-dessous.

### Materiel et OS

- Driver NVIDIA : 591.74
- OS : Windows 11 + PowerShell (venv `aivenv` sur le Bureau, cree par
  `scripts/setup_new_pc_windows.ps1 -UseDesktopAivenv`)

### Logiciels cles

| Composant | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.10.0+cu128 (CUDA 12.8) |
| torchvision | 0.25.0+cu128 |
| Ultralytics | 8.4.14 |
| TensorRT (pip) | 10.16.0.72 |
| onnxruntime-gpu | 1.24.1 |
| onnx | 1.20.1 |
| OpenCV (opencv-python) | 4.13.0.92 |
| NumPy | 2.4.2 |
| SciPy | 1.17.0 |
| matplotlib | 3.10.8 |

La liste complete et exacte des paquets est figee dans
`requirements.lock.txt` (sortie de `pip freeze` de cet environnement).
Pour une reproduction stricte, l'utiliser a la place de
`requirements.txt` (necessite le meme index PyTorch cu128 pour les
paquets torch).

Remarques :

- Les engines TensorRT `.engine` livres ont ete generes avec TensorRT
  10.16 sur le GPU d'origine : ils ne se chargent pas forcement sur un
  autre GPU ou une autre version de TensorRT. Les regenerer depuis les
  `.pt` avec `Phase_2_Baseline_MonoCam/export_onnx.py` si besoin.
- Le re-entrainement n'est pas bit-a-bit deterministe sur GPU : on
  attend des performances comparables, pas des poids identiques.

## 2. Installation Windows native

Prerequis manuels : driver NVIDIA du GPU, acces GitHub / cle SSH si on
veut pousser, et en option le CUDA Toolkit et TensorRT correspondant a
la stack si on veut exporter ou charger des `.engine`.

Cloner le depot :

```powershell
cd $env:USERPROFILE\Desktop
git clone https://github.com/frisatim/STAGELIST3N-FusionCam.git
cd STAGELIST3N-FusionCam
```

Autoriser les scripts PowerShell locaux puis lancer l'installation
(venv externe `aivenv` sur le Bureau, comme la machine de reference) :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -InstallOptionalNetwork
```

Variantes :

```powershell
# venv .venv dans le repo plutot que sur le Bureau
.\scripts\setup_new_pc_windows.ps1 -InstallOptionalNetwork

# forcer un index de wheels PyTorch CUDA specifique
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -TorchIndexUrl https://download.pytorch.org/whl/cu121
```

Copier ensuite les donnees externes (datasets, enregistrements, poids)
et creer les jonctions vers les chemins attendus par les scripts :

```powershell
.\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\STAGELIST3N-FusionCam-data
.\scripts\link_external_data_windows.ps1
```

Le layout de donnees et le detail des jonctions sont dans
`docs/DONNEES.md`.

## 3. Installation Docker

L'image Docker fournit l'environnement d'execution (memes versions
epinglees de torch/onnx/onnxruntime/tensorrt sur une base
`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`). Les donnees restent
hors image et sont montees comme volumes : l'image reste petite et les
donnees peuvent etre mises a jour sans rebuild.

Copier `.env.example` vers `.env` et ajuster si le dossier de donnees
n'est pas a cote du repo :

```text
FUSIONCAM_DATA_DIR=../STAGELIST3N-FusionCam-data
```

Important : utiliser la commande `docker compose` (Docker Compose v2),
pas l'ancien binaire `docker-compose` (v1). L'acces GPU declare via
`deploy.resources` dans `docker-compose.yml` n'est honore que par la
v2 ; avec l'ancien binaire, le conteneur demarre sans GPU.

```bash
docker compose build fusioncam
docker compose run --rm fusioncam bash
```

Docker Compose monte le dossier de donnees externe dans
`/workspace/data` et dans les chemins legacy attendus par les scripts
(`/workspace/code/dataset*`, `/workspace/code/recordings`, etc., voir
`docs/DONNEES.md`).

Pour republier les enregistrements en RTSP local (tests de type live
sans acces aux cameras) :

```bash
docker compose up -d mediamtx
```

puis suivre `docker/replay_rtsp_examples.md`. Par defaut les ports
MediaMTX sont lies a `127.0.0.1` ; ne les exposer que sur un reseau de
confiance.

## 4. Valider l'installation

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pytest -q
python scripts\verify_data_layout.py --model-version V4 --model yolov8s
```

Test de fumee Phase 3 sur videos enregistrees :

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats pt --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960
```

Test de fumee Phase 3 live (necessite des flux RTSP, reels ou rejoues) :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_07 --duration-min 2 --device cuda:0 --no-display --no-record-video
```

Utiliser `--formats fp32_engine` seulement apres avoir confirme que
l'engine TensorRT local se charge sur le GPU cible. Les options de
capture GStreamer (`--capture-backend gstreamer --gst-protocol tcp
--gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin`) sont
documentees dans `Phase_3_Fusion_MultiCam/README.md`.

## 5. Problemes courants

- `torch.cuda.is_available() == False` : reinstaller le driver NVIDIA
  et verifier l'index de wheels PyTorch CUDA.
- Le `.engine` TensorRT ne charge pas : regenerer l'engine sur la
  machine cible depuis le `.pt`.
- `python` pointe vers `C:\Program Files (x86)\wapt\python.exe` :
  utiliser `py -3.12` pour creer le venv, puis activer le venv avant
  d'installer les dependances.
- GStreamer n'ouvre pas le flux RTSP : verifier que
  `C:\gstreamer\1.0\msvc_x86_64\bin` est dans le `PATH`, puis relancer
  PowerShell.
- FPS bas a 4/8 cameras avec l'affichage active : utiliser
  `--no-display --no-record-video` pour les benchmarks, et
  afficher/enregistrer la video dans un processus separe.

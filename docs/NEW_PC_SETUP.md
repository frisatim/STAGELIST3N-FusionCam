# Installation d'un nouveau PC Windows

Ce guide prepare un nouveau portable/poste de travail Windows pour les
experiences des Phases 2, 3 et 4.

## 1. Prerequis manuels

Installer d'abord :

- le driver NVIDIA du GPU du portable ;
- l'acces GitHub / la cle SSH si on veut pousser depuis le nouveau PC ;
- optionnel mais recommande : le CUDA Toolkit NVIDIA correspondant a la stack
  PyTorch/TensorRT ;
- optionnel pour l'export/l'inference TensorRT `.engine` : TensorRT installe
  pour la version GPU/CUDA du nouveau PC.

Les engines TensorRT ne sont pas totalement portables entre machines. Si un
`best.engine` existant echoue sur le nouveau GPU, regenerer l'engine sur le
nouveau PC.

## 2. Cloner le depot

```powershell
cd $env:USERPROFILE\Desktop
git clone git@github.com:frisatim/STAGELIST3N-FusionCam.git
cd STAGELIST3N-FusionCam
```

Si SSH n'est pas configure :

```powershell
git clone https://github.com/frisatim/STAGELIST3N-FusionCam.git
cd STAGELIST3N-FusionCam
```

## 3. Lancer le script d'installation

Autoriser les scripts PowerShell locaux :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Installation recommandee avec le meme style de venv externe que la machine
actuelle :

```powershell
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -InstallOptionalNetwork
```

Alternative : creer `.venv` dans le repo :

```powershell
.\scripts\setup_new_pc_windows.ps1 -InstallOptionalNetwork
```

Si l'index de wheels PyTorch CUDA par defaut ne correspond pas a la machine,
le forcer :

```powershell
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -TorchIndexUrl https://download.pytorch.org/whl/cu121
```

## 4. Copier les donnees externes et les artefacts modeles

Ces fichiers ne sont volontairement pas stockes dans Git :

- `dataset/`
- `dataset_objets_HD/`
- `dataset_objets_V4/`
- `recordings/`
- `Phase_2_Baseline_MonoCam/Modelstrained/`
- les fichiers modeles comme `*.pt`, `*.engine`, `*.onnx`

Le layout cible officiel est documente dans :

```text
docs/DATA_LAYOUT.md
```

Les copier depuis l'ancien PC, un disque externe ou un lecteur partage vers le
dossier de donnees externe standard :

```text
..\STAGELIST3N-FusionCam-data
```

Workflow USB recommande :

```powershell
.\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\BenchmarkingAI
```

Si la cle USB contient deja un dossier `STAGELIST3N-FusionCam-data` prepare :

```powershell
.\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\
```

Exemples attendus :

```text
..\STAGELIST3N-FusionCam-data\datasets\dataset_objets_HD\gt_objects_tad.json
..\STAGELIST3N-FusionCam-data\datasets\dataset_objets_V4\data.yaml
..\STAGELIST3N-FusionCam-data\recordings\recordings\*.mp4
..\STAGELIST3N-FusionCam-data\models\V4\person_objects\yolov8s\weights\best.pt
..\STAGELIST3N-FusionCam-data\models\V4\person_objects\yolov8s\weights\best.engine
```

Le setup Docker remonte ce dossier de donnees externe dans les chemins legacy
utilises par les scripts. Pour une execution native Windows, soit garder les
fichiers dans les chemins historiques du repo, soit creer des jonctions de
repertoires. L'option native la plus simple est de creer les jonctions apres
la copie :

```powershell
.\scripts\link_external_data_windows.ps1
```

Cela expose les donnees externes aux chemins legacy attendus par les scripts :

```text
dataset
dataset_objets_HD
dataset_objets_V4
recordings
Phase_2_Baseline_MonoCam\Modelstrained
Phase_3_Fusion_MultiCam\reports
```

Pour une execution Docker, garder le layout externe standard ci-dessus.

## 5. Valider l'environnement

Activer l'environnement :

```powershell
C:\Users\$env:USERNAME\Desktop\aivenv\Scripts\Activate.ps1
```

Lancer les verifications :

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import cv2; print(cv2.__version__); print('GStreamer' in cv2.getBuildInformation())"
python -m pytest Phase_3_Fusion_MultiCam/test_campaign_utils_pytest.py -q
python scripts\verify_data_layout.py --model-version V4 --model yolov8s
Test-Path ..\STAGELIST3N-FusionCam-data\models\V4
Test-Path ..\STAGELIST3N-FusionCam-data\recordings\recordings
Test-Path ..\STAGELIST3N-FusionCam-data\datasets\dataset_objets_V4
```

## 6. Tests de fumee Phase 3

Test visuel live a deux cameras :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 3 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --display-mode annotated --no-record-video
```

Test de performance a quatre cameras sans affichage/enregistrement :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --no-display --no-record-video
```

Test de performance a huit cameras :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_01,cam_02,cam_03,cam_04,cam_05,cam_06,cam_07,cam_08 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --no-display --no-record-video
```

## 7. Test metadonnees/dashboard Phase 4

Demarrer le dashboard :

```powershell
python Phase_4_Network_Latency/alert_dashboard.py --host 127.0.0.1 --port 8765
```

Lancer la Phase 3 avec export des metadonnees :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --no-display --no-record-video --metadata-http-url http://127.0.0.1:8765/metadata --metadata-jsonl Phase_3_Fusion_MultiCam/reports/live_metadata.jsonl
```

Ouvrir :

```text
http://127.0.0.1:8765/
```

## 8. Problemes courants

- `torch.cuda.is_available() == False` : reinstaller le driver NVIDIA et
  verifier l'index de wheels PyTorch CUDA.
- Le `.engine` TensorRT ne charge pas : regenerer l'engine sur le nouveau PC.
- `python` pointe vers `C:\Program Files (x86)\wapt\python.exe` : utiliser
  `py -3.12` pour creer le venv, puis activer le venv avant d'installer les
  dependances.
- GStreamer n'ouvre pas le flux RTSP : verifier que
  `C:\gstreamer\1.0\msvc_x86_64\bin` est dans le `PATH`, puis relancer
  PowerShell.
- FPS bas a 4/8 cameras avec l'affichage active : utiliser `--no-display
  --no-record-video` pour les benchmarks, et enregistrer/afficher la video
  dans un processus separe.

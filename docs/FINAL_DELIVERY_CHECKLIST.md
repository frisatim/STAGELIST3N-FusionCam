# Final Delivery Checklist

This checklist describes the clean handoff package to give to supervisors or to
install on a new test machine.

## 1. Repository

Deliver the Git repository:

```text
STAGELIST3N-FusionCam/
```

The repository contains code, lightweight ground truths, small CSV summaries,
documentation, Docker files and setup scripts.

Do not add these files to Git:

- real RTSP URLs, camera passwords or internal IP addresses;
- datasets;
- recordings;
- trained `.pt` weights;
- TensorRT `.engine` files;
- large generated reports and logs.

## 2. Heavy Data Folder

Deliver the heavy-data folder separately, for example on a USB drive:

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

Important model example:

```text
models/V4/person_objects/yolov8s/weights/best.pt
models/V4/person_objects/yolov8s/weights/best.engine
```

If TensorRT `.engine` files fail on another GPU, regenerate them from `.pt` on
the target machine.

## 3. New Windows Machine

Clone the repo:

```powershell
git clone https://github.com/frisatim/STAGELIST3N-FusionCam.git
cd STAGELIST3N-FusionCam
```

Run setup:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -InstallOptionalNetwork
```

Copy heavy data from USB:

```powershell
.\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\STAGELIST3N-FusionCam-data
```

Create native Windows junctions:

```powershell
.\scripts\link_external_data_windows.ps1
```

Activate Python:

```powershell
& "$env:USERPROFILE\Desktop\aivenv\Scripts\Activate.ps1"
```

## 4. Docker Machine

Create `.env`:

```powershell
Copy-Item .env.example .env
```

Build:

```powershell
docker-compose build fusioncam
```

Open a shell:

```powershell
docker-compose run --rm fusioncam bash
```

Smoke test inside Docker:

```bash
python3 -m pytest Phase_3_Fusion_MultiCam/test_campaign_utils_pytest.py -q
```

## 5. Validation Commands

Python/GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "from ultralytics import YOLO; YOLO(r'Phase_2_Baseline_MonoCam\Modelstrained\V4\person_objects\yolov8s\weights\best.pt'); print('model ok')"
```

Phase 3 recorded smoke test:

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats pt --no-display --device cuda:0 --skip-phase2
```

Phase 3 live performance smoke test:

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_07 --duration-min 2 --device cuda:0 --no-display --no-record-video
```

Use `--formats fp32_engine` only after confirming that the local TensorRT engine
loads on the target GPU.

## 6. Security Check Before Public Release

Before making a new repository snapshot public, check:

```powershell
git status
git grep -n -I -E "FFCA|172\.16\.|rtsp://admin:[^<]|github_pat|ghp_|sk-"
git ls-files | Select-String -Pattern "\.pt$|\.engine$|\.onnx$|\.mp4$|\.mkv$|\.avi$"
```

Expected result: no real credentials, no internal IPs, no private local paths and
no heavy model/video artifacts tracked by Git.

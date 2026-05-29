# New Windows PC Setup

This guide prepares a fresh Windows laptop/workstation for Phase 2, Phase 3 and
Phase 4 experiments.

## 1. Manual prerequisites

Install these first:

- NVIDIA driver for the laptop GPU.
- GitHub access/SSH key if you want to push from the new PC.
- Optional but recommended: NVIDIA CUDA Toolkit matching your PyTorch/TensorRT
  stack.
- Optional for TensorRT `.engine` export/inference: TensorRT installed for the
  GPU/CUDA version on the new PC.

TensorRT engines are not fully portable between machines. If an existing
`best.engine` fails on the new GPU, regenerate the engine on the new PC.

## 2. Clone the repository

```powershell
cd $env:USERPROFILE\Desktop
git clone git@github.com:frisatim/STAGELIST3N-FusionCam.git
cd STAGELIST3N-FusionCam
```

If SSH is not configured:

```powershell
git clone https://github.com/frisatim/STAGELIST3N-FusionCam.git
cd STAGELIST3N-FusionCam
```

## 3. Run the setup script

Allow local PowerShell scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Recommended setup using the same external venv style as the current machine:

```powershell
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -InstallOptionalNetwork
```

Alternative: create `.venv` inside the repo:

```powershell
.\scripts\setup_new_pc_windows.ps1 -InstallOptionalNetwork
```

If the default PyTorch CUDA wheel index does not match the machine, override it:

```powershell
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -TorchIndexUrl https://download.pytorch.org/whl/cu121
```

## 4. Copy external data and model artifacts

These files are intentionally not stored in Git:

- `dataset/`
- `dataset_objets_HD/`
- `dataset_objets_V4/`
- `recordings/`
- `Phase_2_Baseline_MonoCam/Modelstrained/`
- model files such as `*.pt`, `*.engine`, `*.onnx`

Copy them from the old PC, an external disk, or a shared drive.

Expected examples:

```text
dataset_objets_HD/gt_objects_tad.json
gt_people.json
recordings/recordings/*.mp4
Phase_2_Baseline_MonoCam/Modelstrained/V4/person_objects/yolov8s/weights/best.pt
Phase_2_Baseline_MonoCam/Modelstrained/V4/person_objects/yolov8s/weights/best.engine
```

## 5. Validate the environment

Activate the environment:

```powershell
C:\Users\$env:USERNAME\Desktop\aivenv\Scripts\Activate.ps1
```

Run checks:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import cv2; print(cv2.__version__); print('GStreamer' in cv2.getBuildInformation())"
python -m pytest Phase_4_Network_Latency
```

## 6. Phase 3 smoke tests

Two-camera visual live test:

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 3 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --display-mode annotated --no-record-video
```

Four-camera performance test without display/recording:

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --no-display --no-record-video
```

Eight-camera performance test:

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_01,cam_02,cam_03,cam_04,cam_05,cam_06,cam_07,cam_08 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --no-display --no-record-video
```

## 7. Phase 4 metadata/dashboard test

Start dashboard:

```powershell
python Phase_4_Network_Latency/alert_dashboard.py --host 127.0.0.1 --port 8765
```

Run Phase 3 with metadata export:

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --no-display --no-record-video --metadata-http-url http://127.0.0.1:8765/metadata --metadata-jsonl Phase_3_Fusion_MultiCam/reports/live_metadata.jsonl
```

Open:

```text
http://127.0.0.1:8765/
```

## 8. Common issues

- `torch.cuda.is_available() == False`: reinstall the NVIDIA driver and verify
  the PyTorch CUDA wheel index.
- TensorRT `.engine` fails to load: regenerate the engine on the new PC.
- GStreamer does not open RTSP: verify `C:\gstreamer\1.0\msvc_x86_64\bin` is in
  `PATH`, then restart PowerShell.
- 4/8 cameras have low FPS with display enabled: use `--no-display
  --no-record-video` for benchmarks, and record/display video in a separate
  process.

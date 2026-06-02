# Docker and Heavy Data Delivery

The Git repository contains code, documentation, lightweight ground truths and
small result summaries. Heavy assets are kept outside Git and mounted as Docker
volumes.

## Recommended Delivery Layout

The canonical data layout is documented in `docs/DATA_LAYOUT.md`.

```text
delivery/
  STAGELIST3N-FusionCam/          # Git repository
  STAGELIST3N-FusionCam-data/     # Heavy assets, not tracked by Git
    datasets/
      dataset_objets_V4/
      dataset_objets_HD/
      dataset/
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

The Docker image provides the execution environment. The data directory provides
datasets, recordings, trained weights, TensorRT engines and large reports.

## Why Data Is Not Baked Into The Image

Keeping datasets and models outside the image is more practical:

- the Docker image stays smaller and faster to rebuild;
- datasets and results can be updated without rebuilding the image;
- `.engine` TensorRT files are often tied to the GPU, CUDA and TensorRT version;
- `.pt` weights remain the portable source of truth and engines can be rebuilt.

## First Setup

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` if the data folder is not next to the repository:

```text
FUSIONCAM_DATA_DIR=../STAGELIST3N-FusionCam-data
```

Create the heavy-data folder skeleton:

```bash
python3 scripts/prepare_delivery_layout.py \
  --data-dir ../STAGELIST3N-FusionCam-data
```

Then manually copy datasets, recordings, models and reports into the created
folders, or use the USB copy helper on Windows:

```powershell
.\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\BenchmarkingAI
```

This command accepts either the original full project folder or an already
prepared `STAGELIST3N-FusionCam-data` folder on the USB drive.

For native Windows runs outside Docker, create local junctions after the copy:

```powershell
.\scripts\link_external_data_windows.ps1
```

The junctions expose the external data at the historical paths used by the
Python scripts without duplicating large files inside Git.

## Build

```bash
docker-compose build fusioncam
```

## Open A Shell

```bash
docker-compose run --rm fusioncam bash
```

Inside the container:

```bash
python3 -m pytest Phase_3_Fusion_MultiCam/test_campaign_utils_pytest.py -q
```

## Run Recorded Campaign

Example after placing recordings and models in the mounted data folder:

```bash
docker-compose run --rm fusioncam \
  python3 Phase_3_Fusion_MultiCam/run_recorded_campaign.py \
    --dataset-version V4 \
    --models yolov8s \
    --formats pt \
    --no-display \
    --device cuda:0 \
    --phase2-device gpu \
    --phase2-imgsz 960
```

Before running campaigns, verify the mounts:

```bash
docker-compose run --rm fusioncam \
  python3 scripts/verify_data_layout.py \
    --data-dir /workspace/data \
    --model-version V4 \
    --model yolov8s
```

Paths in configs may need to be adapted to the mounted data directories:

```text
/workspace/data/datasets
/workspace/data/recordings
/workspace/data/models
/workspace/data/reports
```

For compatibility with the existing research scripts, Docker Compose also mounts
the same external data into the historical paths:

```text
/workspace/code/dataset
/workspace/code/dataset_objets_HD
/workspace/code/dataset_objets_V4
/workspace/code/recordings
/workspace/code/Phase_2_Baseline_MonoCam/Modelstrained
/workspace/code/Phase_3_Fusion_MultiCam/reports
```

## RTSP Replay From Recordings

Start MediaMTX:

```bash
docker-compose up -d mediamtx
```

By default the MediaMTX ports are bound to `127.0.0.1` in
`docker-compose.yml`. Keep this default for local tests. If video replay must be
reachable from another machine, expose the ports only on a trusted network and
add the appropriate firewall or MediaMTX authentication rules.

Then publish recordings with FFmpeg as documented in
`docker/replay_rtsp_examples.md`.

This allows repeatable live-like tests from annotated video files.

## What To Give To Tutors

Recommended files:

- the Git repository;
- `STAGELIST3N-FusionCam-data` as a `.zip`, `.tar` or external drive folder;
- this Docker setup;
- a short note with GPU, CUDA, Python and Ultralytics versions used for training.

Keep both `.pt` and `.engine` when available. If the `.engine` fails on another
machine, regenerate it from the `.pt`.

Do not place datasets, recordings, `.pt`, `.engine` or real RTSP credentials in
Git. Keep them in `STAGELIST3N-FusionCam-data` or another private storage
location.

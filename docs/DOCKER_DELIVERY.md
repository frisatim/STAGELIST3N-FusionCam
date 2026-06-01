# Docker and Heavy Data Delivery

The Git repository contains code, documentation, lightweight ground truths and
small result summaries. Heavy assets are kept outside Git and mounted as Docker
volumes.

## Recommended Delivery Layout

```text
delivery/
  STAGELIST3N-FusionCam/          # Git repository
  STAGELIST3N-FusionCam-data/     # Heavy assets, not tracked by Git
    datasets/
      dataset_objets_V4/
      dataset_objets_HD/
      dataset/
    recordings/
      Camera_*.mp4
    models/
      V2/
      V3/
      V4/
    reports/
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
folders.

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
    --versions V4 \
    --models yolov8s \
    --formats pt \
    --no-display \
    --skip-phase2
```

Paths in configs may need to be adapted to the mounted data directories:

```text
/workspace/data/datasets
/workspace/data/recordings
/workspace/data/models
/workspace/data/reports
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

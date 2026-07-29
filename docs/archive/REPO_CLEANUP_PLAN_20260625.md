# Repo Cleanup Plan - 2026-06-25

Goal: prepare the repository for final internship handoff without breaking the
current test scripts.

This document is intentionally conservative. The project is still used for live
experiments, so cleanup should focus on documentation, script classification and
small safety checks before moving or renaming files.

## Current Decision

Keep the current phase-based structure:

```text
Phase_1_Infrastructure/       RTSP capture, recording, annotation and dataset conversion.
Phase_2_Baseline_MonoCam/     Single-camera baseline, training and TAD/TRD evaluation.
Phase_2.5_Test_Live/          Legacy live tests and early dashboard experiments.
Phase_3_Fusion_MultiCam/      Calibration, tracking, fusion, alerts, campaigns.
Phase_4_Network_Latency/      Metadata, network latency, dashboard and transport tests.
scripts/                      Setup, delivery, replay and orchestration helpers.
docs/                         Methodology, results, setup and final handoff docs.
docker/                       MediaMTX and replay examples.
reports/                      Small curated reports only.
```

Do not reorganize these folders before the final camera tests are finished.
Several commands and docs still reference these paths directly.

## Immediate Rules

- Do not commit large assets: videos, datasets, `.pt`, `.engine`, `.onnx`.
- Do not commit real RTSP credentials or internal camera URLs.
- Do not commit unstable dashboard automation until it is visually validated.
- Keep final benchmark runs in `Phase_3_Fusion_MultiCam/reports/` locally or in
  the external data folder, not in Git.
- Keep only small summary CSV/MD/PNG artifacts in Git when they support the
  report or presentation.

## Current Unstable Items

These files should stay out of commits until the browser video relay works
reliably:

```text
scripts/start_phase4_dashboard_windows.ps1
scripts/start_phase4_full_dashboard_windows.ps1
```

Reason: metadata overlay works, but browser video relay through MediaMTX
HLS/WebRTC has not been fully validated yet.

## Script Classification

### Stable / Important

```text
scripts/setup_new_pc_windows.ps1
scripts/prepare_delivery_layout.py
scripts/link_external_data_windows.ps1
scripts/copy_heavy_data_from_usb.ps1
scripts/verify_data_layout.py
scripts/generate_phase3_phase4_figures.py
Phase_3_Fusion_MultiCam/run_recorded_campaign.py
Phase_3_Fusion_MultiCam/run_live_campaign.py
Phase_3_Fusion_MultiCam/calibration_tool_v2.py
Phase_3_Fusion_MultiCam/verify_calibration.py
Phase_3_Fusion_MultiCam/draw_zone-multicam.py
Phase_3_Fusion_MultiCam/evaluate_fusion_links.py
Phase_4_Network_Latency/alert_delivery_benchmark.py
Phase_4_Network_Latency/validate_metadata_jsonl.py
Phase_4_Network_Latency/analyze_phase4_runs.py
```

These scripts should be kept, documented and tested before final delivery.

### Experimental / To Review

```text
Phase_2.5_Test_Live/dashboard_live_rtsp.py
Phase_4_Network_Latency/alert_dashboard.py
scripts/start_phase4_dashboard_windows.ps1
scripts/start_phase4_full_dashboard_windows.ps1
scripts/start_replay_4cams_windows.ps1
docker/replay_rtsp_examples.md
```

These are useful for demos and Phase 4, but should be clearly marked as
experimental if the video part remains unstable.

### Legacy / Candidate Archive

```text
Phase_3_Fusion_MultiCam/calibration_tool.py
Phase_3_Fusion_MultiCam/test_fusion.py
Phase_3_Fusion_MultiCam/test_tracker.py
Phase_2_Baseline_MonoCam/train_yolo_dryrun.py
Phase_2_Baseline_MonoCam/test_yolo_baseline.py
```

Before moving or deleting them, check whether they are referenced in docs or by
other scripts.

## Documentation Cleanup

Recommended final docs to keep prominent:

```text
README.md
docs/DATA_LAYOUT.md
docs/NEW_PC_SETUP.md
docs/DOCKER_DELIVERY.md
docs/FINAL_DELIVERY_CHECKLIST.md
docs/DATA_AND_SECURITY.md
docs/RESEARCH_PLAN_SUMMARY.md
docs/LATENCE_END_TO_END_PAR_ETAPE.md
docs/AUDIT_INTERET_FUSION_MULTICAMERA_20260623.md
docs/ANALYSE_RESULTATS_GRAPHES_PHASE3_PHASE4_20260623.md
```

Candidate docs to archive later under `docs/archive/`:

```text
docs/ROADMAP_TESTS_MARDI_VENDREDI.md
docs/ROADMAP_TESTS_APRES_ANALYSE_LIVE_PHASE3_PHASE4.md
docs/ROADMAP_3_JOURS_FINAL_TESTS.md
docs/ROADMAP_ACCES_CAMERAS_50_50.md
docs/ROADMAP_PHASE4_DASHBOARD_WEB.md
docs/SYNTHESE_REUNION_SEMAINE_20260612.md
docs/ANALYSE_RESULTATS_REUNION_20260612.md
```

Do not archive them before the report is finished, because they still contain
useful context.

## Suggested Final README Updates

Before final handoff, update `README.md` with:

- one clean installation path for Windows;
- one clean recorded-campaign command;
- one clean live-campaign command;
- one clean Phase 4 metadata benchmark command;
- a short note explaining that the dashboard video relay is experimental unless
  it has been validated;
- a link to `docs/FINAL_DELIVERY_CHECKLIST.md`.

## Safety Checks Before Any Public Snapshot

Run from repo root:

```powershell
git status
git grep -n -I -E "F[F]CA|172\.16\.|rtsp://admin[:][^<]|github_p[a]t|gh[p]_|\bs[k]-"
git ls-files | Select-String -Pattern "\.pt$|\.engine$|\.onnx$|\.mp4$|\.mkv$|\.avi$"
```

Expected:

- no real credentials;
- no internal camera URLs with passwords;
- no model weights;
- no TensorRT engines;
- no videos.

## Cleanup Order

1. Finish final camera tests and copy results outside Git.
2. Decide whether dashboard video relay is stable or experimental.
3. Update README and final delivery docs.
4. Add a short script index if needed.
5. Move old roadmaps to `docs/archive/` only after the report is written.
6. Run security checks.
7. Commit only stable docs/scripts.

## Do Not Do Yet

- Do not rename phase folders.
- Do not move `run_live_campaign.py` or `run_recorded_campaign.py`.
- Do not delete calibration files.
- Do not delete old roadmaps until the report is complete.
- Do not commit dashboard orchestration scripts until the video path is stable.

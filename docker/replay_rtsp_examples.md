# RTSP Replay Examples

Start MediaMTX:

```bash
docker-compose up -d mediamtx
```

Publish one recorded video as one RTSP stream:

```bash
ffmpeg -re -stream_loop -1 \
  -i ../STAGELIST3N-FusionCam-data/recordings/Camera_2_2.3_20260506_131002.mp4 \
  -c copy -f rtsp rtsp://localhost:8554/cam_02
```

Publish the four Zone 1 cameras in four terminals:

```bash
ffmpeg -re -stream_loop -1 -i ../STAGELIST3N-FusionCam-data/recordings/Camera_2_2.3_20260506_131002.mp4 -c copy -f rtsp rtsp://localhost:8554/cam_02
ffmpeg -re -stream_loop -1 -i ../STAGELIST3N-FusionCam-data/recordings/Camera_3_2.4_20260506_131002.mp4 -c copy -f rtsp rtsp://localhost:8554/cam_03
ffmpeg -re -stream_loop -1 -i ../STAGELIST3N-FusionCam-data/recordings/Camera_5_2.6_20260506_131002.mp4 -c copy -f rtsp rtsp://localhost:8554/cam_05
ffmpeg -re -stream_loop -1 -i ../STAGELIST3N-FusionCam-data/recordings/Camera_7_2.11_20260506_131002.mp4 -c copy -f rtsp rtsp://localhost:8554/cam_07
```

Temporary config URLs:

```text
rtsp://localhost:8554/cam_02
rtsp://localhost:8554/cam_03
rtsp://localhost:8554/cam_05
rtsp://localhost:8554/cam_07
```

This is useful for repeatable "live" validation from annotated recordings.

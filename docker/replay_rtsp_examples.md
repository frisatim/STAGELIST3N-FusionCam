# Exemples de replay RTSP

Demarrer MediaMTX :

```bash
docker compose up -d mediamtx
```

Les videos suivent le layout officiel documente dans `docs/DONNEES.md` :
`STAGELIST3N-FusionCam-data/recordings/recordings/Camera_*.mp4` (deux niveaux
`recordings`).

Publier une video enregistree comme un flux RTSP (FFmpeg installe sur
l'hote) :

```bash
ffmpeg -re -stream_loop -1 \
  -i ../STAGELIST3N-FusionCam-data/recordings/recordings/Camera_2_2.3_20260506_131002.mp4 \
  -c copy -f rtsp rtsp://localhost:8554/cam_02
```

Publier les quatre cameras de la Zone 1 dans quatre terminaux :

```bash
ffmpeg -re -stream_loop -1 -i ../STAGELIST3N-FusionCam-data/recordings/recordings/Camera_2_2.3_20260506_131002.mp4 -c copy -f rtsp rtsp://localhost:8554/cam_02
ffmpeg -re -stream_loop -1 -i ../STAGELIST3N-FusionCam-data/recordings/recordings/Camera_3_2.4_20260506_131002.mp4 -c copy -f rtsp rtsp://localhost:8554/cam_03
ffmpeg -re -stream_loop -1 -i ../STAGELIST3N-FusionCam-data/recordings/recordings/Camera_5_2.6_20260506_131002.mp4 -c copy -f rtsp rtsp://localhost:8554/cam_05
ffmpeg -re -stream_loop -1 -i ../STAGELIST3N-FusionCam-data/recordings/recordings/Camera_7_2.11_20260506_131002.mp4 -c copy -f rtsp rtsp://localhost:8554/cam_07
```

Variante conteneur avec le service `rtsp-replay` : dans `docker-compose.yml`,
ce service monte `${FUSIONCAM_DATA_DIR}/recordings` sur `/recordings`, donc le
chemin dans le conteneur est `/recordings/recordings/Camera_*.mp4`. Depuis le
reseau Docker Compose, MediaMTX est joignable via `rtsp://mediamtx:8554/...`
(pas `localhost`).

```bash
docker compose --profile rtsp-replay up -d rtsp-replay
docker compose exec rtsp-replay ffmpeg -re -stream_loop -1 \
  -i /recordings/recordings/Camera_2_2.3_20260506_131002.mp4 \
  -c copy -f rtsp rtsp://mediamtx:8554/cam_02
```

URLs temporaires pour la config :

```text
rtsp://localhost:8554/cam_02
rtsp://localhost:8554/cam_03
rtsp://localhost:8554/cam_05
rtsp://localhost:8554/cam_07
```

C'est utile pour une validation "live" repetable a partir d'enregistrements
annotes.

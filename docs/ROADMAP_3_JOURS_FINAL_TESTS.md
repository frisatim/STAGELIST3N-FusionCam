# Roadmap finale 3 jours - serveur, nouveau PC, Phase 4, dashboard

Objectif : finir les tests prioritaires tant que le nouveau PC et le serveur sont disponibles.

## Priorites globales

1. Mettre les scripts a jour sur nouveau PC et serveur.
2. Faire un run final long serveur avec le meilleur modele.
3. Mesurer la latence par etape et la latence end-to-end.
4. Tester Phase 4 : JSONL, HTTP, WebSocket, MQTT.
5. Tester un dashboard concret video + metadata.
6. Verifier la synchronisation video / IA.
7. Revoir calibration/homographie seulement si les resultats montrent un probleme clair.

## Point critique avant de commencer

Les machines doivent avoir la version du repo contenant :

- `--metadata-jsonl`
- `--metadata-http-url`
- `--latency-trace-csv`
- `Phase_4_Network_Latency/alert_dashboard.py`
- `Phase_4_Network_Latency/alert_delivery_benchmark.py`
- `Phase_4_Network_Latency/validate_metadata_jsonl.py`

Verification :

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --help | findstr metadata
python Phase_3_Fusion_MultiCam\run_live_campaign.py --help | findstr latency
dir Phase_4_Network_Latency
```

Linux :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --help | grep -E "metadata|latency"
ls Phase_4_Network_Latency
```

## Jour 1 - Run final serveur et latence interne

### 1. Run final serveur 30 min

But : avoir le meilleur run final Phase 3.

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 30 --device cuda:0 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam/reports/final_server_4cam_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam/reports/final_server_4cam_latency_trace.csv --out-dir Phase_3_Fusion_MultiCam/reports/final_server_4cam_yolov8s_engine_30min
```

Verifier :

```bash
python Phase_4_Network_Latency/validate_metadata_jsonl.py Phase_3_Fusion_MultiCam/reports/final_server_4cam_metadata.jsonl --print-example
```

Livrables :

- `summary.csv`
- `alerts.csv`
- `fusion_links.csv`
- `track_stability.csv`
- `final_server_4cam_metadata.jsonl`
- `final_server_4cam_latency_trace.csv`

### 2. Analyse rapide du run final

Points a regarder :

- FPS effectif ;
- latence IA moyenne/p95 ;
- `capture_read_ms` ;
- `internal_after_read_ms` ;
- alertes personnes ;
- alertes objets weak/confirmed ;
- liens de fusion ;
- ID switches ;
- cameras reellement presentes.

## Jour 2 - Phase 4 transport et dashboard local

### 1. Benchmarks transport sans cameras

Queue :

```bash
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport queue --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/final_queue/alert_latency.csv
```

HTTP :

```bash
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport http_post --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/final_http_post/alert_latency.csv
```

WebSocket :

```bash
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport websocket --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/final_websocket/alert_latency.csv
```

MQTT :

Terminal 1 :

```bash
mosquitto -p 1883
```

Terminal 2 :

```bash
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 0 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/final_mqtt_qos0/alert_latency.csv
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 1 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/final_mqtt_qos1/alert_latency.csv
```

### 2. Dashboard local metadata seule

Terminal 1 :

```bash
python Phase_4_Network_Latency/alert_dashboard.py --host 0.0.0.0 --port 8765
```

Terminal 2 :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --metadata-http-url http://127.0.0.1:8765/metadata --metadata-jsonl Phase_3_Fusion_MultiCam/reports/dashboard_server_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam/reports/dashboard_server_latency_trace.csv --out-dir Phase_3_Fusion_MultiCam/reports/dashboard_server_2cam_metadata_10min
```

Ouvrir :

```text
http://<IP_SERVEUR>:8765/
```

## Jour 3 - Dashboard video + metadata et synchronisation

### 1. Test dashboard avec video

Ouvrir le dashboard avec video MediaMTX :

```text
http://<IP_SERVEUR>:8765/?camera=cam_02&video=http://<IP_MEDIAMTX>:8889/cam_02/&video_mode=iframe
```

Verifier :

- video visible ;
- bbox visibles ;
- alertes visibles ;
- `global_id` affiches ;
- overlay pas trop decale ;
- latence metadata affichee ;
- pas de coupure longue.

### 2. Mesure end-to-end avec chrono

Mettre un chrono visible dans la camera.

Pour chaque evenement :

```text
T0 = temps visible dans l'image quand l'evenement arrive
T1 = capture_read_end_epoch dans latency_trace
T2 = inference_end_epoch
T3 = fusion_end_epoch
T4 = alerts_end_epoch
T5 = metadata_end_epoch
```

Calculs :

```text
camera + reseau + buffers = T1 - T0
inference/tracking = T2 - inference_start_epoch
fusion = T3 - fusion_start_epoch
alertes = T4 - alerts_start_epoch
metadata = T5 - metadata_start_epoch
end-to-end alerte = T5 - T0
```

Faire au moins :

- 5 entrees de personne en zone ;
- 5 poses d'objet ;
- idealement sur PC nouveau et serveur.

## Vercel : point important

Vercel est bien pour heberger le frontend du dashboard.

Mais Vercel n'est pas le meilleur endroit pour :

- recevoir en continu `/metadata` depuis Phase 3 ;
- maintenir un backend Python long-running ;
- faire du WebSocket/SSE temps reel lourd ;
- relayer de la video.

Architecture recommandee pour une demo internet :

```text
Frontend React/Next.js sur Vercel
  -> lit la video WebRTC/HLS depuis MediaMTX public ou tunnel
  -> lit les metadata depuis backend serveur HTTPS/WSS

Backend temps reel sur serveur Linux
  -> recoit /metadata depuis Phase 3
  -> redistribue au frontend par WebSocket/SSE
```

Pour une demo rapide, utiliser :

- Cloudflare Tunnel ;
- Tailscale Funnel ;
- ngrok ;
- ou un reverse proxy HTTPS sur le serveur.

Ne pas exposer directement les RTSP cameras.

## Calibration / homographie

Ne pas refaire toutes les calibrations avant les runs prioritaires.

Verifier la calibration si :

- les bbox sont correctes mais les positions au sol sont incoherentes ;
- les objets/personnes visibles par deux cameras ne fusionnent presque jamais ;
- beaucoup de confirmations n'arrivent qu'a 1.5 m ou 2.0 m ;
- les alertes de zone sont visuellement fausses ;
- une camera produit beaucoup moins de detections que prevu.

Test rapide :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 3 --device cuda:0 --object-min-camera-votes 2 --capture-backend opencv --display-mode annotated --no-record-video --out-dir Phase_3_Fusion_MultiCam/reports/calibration_visual_check_4cam
```

Decision :

- si la projection semble correcte, continuer les tests ;
- si une camera est mauvaise, recalibrer seulement cette camera ;
- refaire ensuite un run court sur la paire concernee.

## Ce qui doit etre fini a la fin des 3 jours

- run final serveur 30 min ;
- metadata JSONL final valide ;
- latency trace CSV final ;
- tableau Phase 4 transports ;
- demo dashboard local fonctionnelle ;
- test dashboard avec video si MediaMTX/WebRTC OK ;
- decision sur serveur vs nouveau PC ;
- decision sur modele final ;
- decision sur calibration ;
- limites claires pour l'article.

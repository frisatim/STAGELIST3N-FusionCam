# Roadmap Phase 4 - Dashboard web concret

Objectif : valider l'architecture finale en conditions proches du reel :

```text
Cameras RTSP
  -> flux video separe via MediaMTX / FFmpeg / WebRTC
  -> pipeline IA Phase 3 sans affichage ni recording
  -> metadonnees : bbox, alertes, global_id, timestamp
  -> export JSONL puis HTTP/WebSocket/MQTT
  -> dashboard web qui superpose les bbox sur la video
```

Le point critique est la synchronisation video / metadata. Le dashboard ne doit pas afficher une bbox uniquement parce qu'elle vient d'arriver. Il doit l'associer a la frame video correspondante via `timestamp`, `frame` ou un buffer temporel.

## 1. Ce qui existe deja

Scripts disponibles :

- `Phase_3_Fusion_MultiCam/run_live_campaign.py`
  - inference live ;
  - fusion ;
  - alertes ;
  - export JSONL ;
  - export HTTP metadata ;
  - trace de latence interne via `--latency-trace-csv`.

- `Phase_4_Network_Latency/alert_delivery_benchmark.py`
  - benchmark queue locale ;
  - benchmark HTTP POST ;
  - benchmark WebSocket ;
  - benchmark MQTT QoS 0/1.

- `Phase_4_Network_Latency/alert_dashboard.py`
  - dashboard web local ;
  - reception `/alerts` ;
  - reception `/metadata` ;
  - flux SSE vers navigateur ;
  - overlay bbox sur une video fournie par URL.

## 2. Etape A - Tests Phase 4 sans cameras

But : mesurer le cout des transports alertes/metadonnees sans melanger avec RTSP, IA ou fusion.

### Queue locale

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport queue --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_queue\alert_latency.csv
```

### HTTP POST

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport http_post --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_http_post\alert_latency.csv
```

### WebSocket

Dependance :

```powershell
pip install websockets
```

Commande :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport websocket --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_websocket\alert_latency.csv
```

### MQTT

Dependance :

```powershell
pip install paho-mqtt
```

Demarrer Mosquitto dans un terminal :

```powershell
mosquitto -p 1883
```

Puis :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 0 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_mqtt_qos0\alert_latency.csv
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 1 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_mqtt_qos1\alert_latency.csv
```

Livrable :

| Transport | Events | Rate | Lat. moyenne | Lat. p95 | Pertes | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Queue | 1000 | 25 Hz | | | | |
| HTTP POST | 1000 | 25 Hz | | | | |
| WebSocket | 1000 | 25 Hz | | | | |
| MQTT QoS0 | 1000 | 25 Hz | | | | |
| MQTT QoS1 | 1000 | 25 Hz | | | | |

## 3. Etape B - Dashboard local avec metadata IA

But : verifier que la Phase 3 peut envoyer les bbox/alertes a un site web.

Terminal 1 : dashboard local

```powershell
python Phase_4_Network_Latency\alert_dashboard.py --host 127.0.0.1 --port 8765
```

Terminal 2 : IA Phase 3 sans affichage ni recording

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --metadata-http-url http://127.0.0.1:8765/metadata --metadata-jsonl Phase_3_Fusion_MultiCam\reports\phase4_dashboard_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam\reports\phase4_dashboard_latency_trace.csv
```

Ouvrir :

```text
http://127.0.0.1:8765/
```

Verification :

- les alertes apparaissent ;
- les compteurs weak/confirmed bougent ;
- les metadonnees arrivent avec une latence faible ;
- le JSONL est ecrit ;
- `latency_trace_csv` permet de decouper la latence interne.

## 4. Etape C - Dashboard local avec video separee

But : tester le principe complet video + overlay.

Architecture :

```text
Camera RTSP
  -> MediaMTX / WebRTC pour la video
  -> Phase 3 pour les metadonnees
  -> dashboard pour overlay
```

Le dashboard accepte une URL video :

```text
http://127.0.0.1:8765/?camera=cam_02&video=<URL_VIDEO>&video_mode=iframe
```

Exemple avec MediaMTX WebRTC expose en page web :

```text
http://127.0.0.1:8765/?camera=cam_02&video=http://127.0.0.1:8889/cam_02/&video_mode=iframe
```

Si une URL HLS/MP4 compatible `<video>` est disponible :

```text
http://127.0.0.1:8765/?camera=cam_02&video=http://127.0.0.1:8888/cam_02/index.m3u8
```

Limite actuelle :

- le dashboard dessine les dernieres bbox recues ;
- pour une demo stricte, il faudra ajouter un buffer timestamp pour aligner metadata et video.

## 5. Etape D - Test serveur avec PC passerelle

Architecture testee :

```text
Cameras IP <CAMERA_NET>.x
  -> PC passerelle Windows
     - Ethernet cameras : <GATEWAY_IP>
     - Wi-Fi / reseau ecole : <SERVER_IP>
  -> FFmpeg republie les flux
  -> MediaMTX sur le PC passerelle
     - rtsp://<SERVER_IP>:8554/cam_02
     - rtsp://<SERVER_IP>:8554/cam_03
  -> serveur Linux GPU
  -> run_live_campaign.py
  -> detection / tracking / fusion / alertes
```

Dashboard sur serveur :

```bash
python Phase_4_Network_Latency/alert_dashboard.py --host 0.0.0.0 --port 8765
```

IA sur serveur :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --metadata-http-url http://127.0.0.1:8765/metadata --metadata-jsonl Phase_3_Fusion_MultiCam/reports/server_dashboard_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam/reports/server_dashboard_latency_trace.csv
```

Ouvrir depuis un autre PC du reseau :

```text
http://<IP_SERVEUR>:8765/
```

## 6. Etape E - Dashboard accessible sur internet

Pour une vraie demo internet, ne pas exposer directement les RTSP cameras.

Architecture recommandee :

```text
Cameras privees
  -> PC passerelle / serveur
  -> MediaMTX pour video WebRTC/HLS
  -> dashboard web public HTTPS
  -> metadata via HTTPS/WebSocket/MQTT over WebSocket
```

Options d'exposition :

1. Reverse proxy HTTPS sur le serveur public.
2. Cloudflare Tunnel / Tailscale Funnel / ngrok pour demo temporaire.
3. VPN/Tailscale si demo reservee aux tuteurs.

Important :

- le navigateur bloque souvent le contenu mixte : un dashboard en `https://` ne doit pas charger une video en `http://` ;
- utiliser `https://` et `wss://` pour la demo internet ;
- ajouter une authentification simple avant exposition publique ;
- ne pas exposer `/metadata` sans token, sinon n'importe qui peut injecter de fausses alertes.

## 7. Synchronisation video / metadata

Probleme :

```text
video latency != metadata latency
```

Si le dashboard affiche la bbox des qu'elle arrive, l'overlay peut etre decale.

Solution a implementer pour la version finale :

```text
video buffer 300-800 ms
metadata buffer par camera
synchronisation par timestamp
affichage de la bbox dont timestamp ~= timestamp video affiche
```

Champs necessaires dans les metadata :

- `camera_id` ;
- `frame` ;
- `timestamp` ;
- `created_epoch_ms` ;
- `bbox_px` ;
- `global_id` ;
- `alert_level` ;
- `alert_type`.

Le JSONL actuel contient deja l'essentiel.

## 8. Definition d'un test reussi

Un test dashboard est reussi si :

- la video est fluide ;
- les bbox apparaissent sur la bonne camera ;
- les bbox suivent correctement les personnes/objets ;
- les alertes sont visibles dans le tableau ;
- les `global_id` sont affiches ;
- la latence metadata est mesuree ;
- la latence end-to-end est mesurable avec chrono visible ;
- l'overlay n'est pas visiblement decale.

## 9. Priorite de developpement

Ordre conseille :

1. Valider dashboard local sans video, metadata uniquement.
2. Valider dashboard local avec video MediaMTX.
3. Ajouter buffer/synchronisation timestamp.
4. Tester serveur avec PC passerelle.
5. Exposer temporairement sur internet avec tunnel ou reverse proxy.
6. Ajouter authentification/token.
7. Comparer HTTP vs WebSocket vs MQTT.

## 10. Message pour le rapport

La Phase 4 ne teste pas seulement un protocole reseau. Elle valide une architecture complete separant :

- le transport video ;
- le pipeline IA ;
- le transport des metadonnees ;
- l'interface de supervision.

Cette separation rend le systeme plus scalable, mais impose une synchronisation temporelle stricte entre flux video et metadonnees.

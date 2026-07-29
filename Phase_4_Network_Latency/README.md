# Phase 4 - Reseau et latence

Objectif: mesurer l'impact du reseau sur le systeme complet, sans melanger les problemes de modele avec les problemes de capture RTSP.

La Phase 4 est separee en deux familles d'experiences:

- video entrante: `RTSP/TCP` vs `RTSP/UDP` sous degradation reseau;
- alertes sortantes: dashboard web, WebSocket, MQTT QoS 0/1.

Le principe experimental reste simple: changer une seule variable a la fois.

## Outils disponibles sans cameras

Generer la matrice d'experiences (la matrice versionnee dans ce dossier
a ete generee avec la commande Linux/WSL ci-dessous). `--interface` est
le nom de l'interface reseau LINUX utilise dans les commandes tc netem
generees (eth0 en general), meme si la generation est lancee depuis
Windows :

```powershell
python Phase_4_Network_Latency/experiment_plan.py --interface eth0 --python-exe python3 --os-targets linux --link-types ethernet,wifi
```

Depuis Linux/WSL:

```bash
python3 Phase_4_Network_Latency/experiment_plan.py --interface eth0 --python-exe python3 --os-targets linux --link-types ethernet,wifi
```

Sorties:

- `Phase_4_Network_Latency/phase4_experiment_matrix.csv`
- `Phase_4_Network_Latency/phase4_experiment_matrix.md`

Analyser les runs live/recorded deja produits:

```powershell
python Phase_4_Network_Latency/analyze_phase4_runs.py --runs-glob "Phase_3_Fusion_MultiCam/reports/campaign_zone1_live_*"
```

Lancer un benchmark synthetique d'alertes sans camera:

```powershell
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport queue --events 500 --rate-hz 25
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport http_post --events 500 --rate-hz 25
```

Dashboard web local sans dependance externe:

```powershell
python Phase_4_Network_Latency/alert_dashboard.py --simulate --rate-hz 2
```

Puis ouvrir:

```text
http://127.0.0.1:8765
```

Pour pousser une alerte manuellement:

```powershell
curl -X POST http://127.0.0.1:8765/alerts -H "Content-Type: application/json" -d "{\"event_id\":1,\"alert_level\":\"confirmed\",\"alert_type\":\"forbidden_object\",\"camera\":\"cam_05\",\"created_epoch_ms\":0}"
```

## Test separe video et metadata IA

Architecture recommandee pour les tests live:

```text
RTSP cameras
  -> MediaMTX / relay video pour affichage et recording
  -> Phase 3 en basse latence, sans affichage ni recording
  -> metadata JSON: bbox, global_id, position au sol, alertes
  -> dashboard local avec overlay optionnel
```

Demarrer le dashboard metadata:

```powershell
python Phase_4_Network_Latency/alert_dashboard.py --host 127.0.0.1 --port 8765
```

Lancer Phase 3 en mode IA seule, avec export JSONL et POST HTTP vers le dashboard:

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-codec h264 --gst-pipeline decodebin --no-display --no-record-video --metadata-http-url http://127.0.0.1:8765/metadata --metadata-jsonl Phase_3_Fusion_MultiCam/reports/live_metadata.jsonl --metadata-every-n-frames 1
```

Ouvrir le dashboard alertes/metadata:

```text
http://127.0.0.1:8765/
```

Pour tester un overlay, ajouter une URL video compatible navigateur, par exemple
une sortie WebRTC/WHEP ou HLS exposee par MediaMTX:

```text
http://127.0.0.1:8765/?camera=cam_02&video=http://127.0.0.1:8889/cam_02/
```

Sans parametre `video`, le dashboard reste utile pour mesurer la latence de
livraison metadata et verifier les alertes.

## WebSocket et MQTT

Le benchmark contient les chemins WebSocket et MQTT, mais ils demandent des dependances optionnelles:

```powershell
pip install websockets paho-mqtt
```

Pour MQTT, il faut aussi un broker local, par exemple Mosquitto:

```powershell
mosquitto -p 1883
```

Commandes:

```powershell
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport websocket --events 500 --rate-hz 25
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 0 --events 500 --rate-hz 25
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 1 --events 500 --rate-hz 25
```

## Protocole video a executer quand les cameras reviennent

Cas principal:

- `RTSP/TCP`, reseau propre;
- `RTSP/UDP`, reseau propre;
- `RTSP/TCP`, degradation moderee;
- `RTSP/UDP`, degradation moderee;
- `RTSP/TCP`, degradation severe;
- `RTSP/UDP`, degradation severe.

Sur Linux, appliquer la degradation avec `tc netem`.

Exemple modere:

```bash
sudo tc qdisc replace dev eth0 root netem delay 80ms 25ms loss 1.0%
```

Exemple severe:

```bash
sudo tc qdisc replace dev eth0 root netem delay 200ms 75ms loss 5.0%
```

Nettoyage:

```bash
sudo tc qdisc del dev eth0 root
```

Pour chaque run, conserver le dossier de sortie et comparer:

- `phase3/summary.csv`: latence modele/pipeline, alertes, fusion links;
- `phase3/sync_events.csv`: FPS effectif, frames manquantes, camera drop;
- `phase3/alerts.csv`: stabilite weak/confirmed;
- `logs/*.txt`: reconnects, erreurs RTSP, warnings decode.

## Windows vs Linux, Wi-Fi vs Ethernet

Priorite:

1. Linux + Ethernet comme reference controlee.
2. Linux + Wi-Fi pour mesurer la degradation realiste.
3. Windows + Ethernet uniquement sur les meilleurs cas, pour verifier l'impact OS.
4. Windows + Wi-Fi seulement si le temps reste disponible.

Ne pas faire toute la matrice complete OS x lien x protocole x dashboard: elle serait trop grande et difficile a defendre.

## Tests

```powershell
python -m pytest Phase_4_Network_Latency
```

Depuis WSL:

```bash
python3 -m pytest Phase_4_Network_Latency
```

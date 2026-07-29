# Roadmap tests avec acces cameras 50/50

Date : 2026-06-23  
Contexte : cette semaine, l'acces aux cameras sera intermittent. L'objectif est de separer clairement les taches qui dependent du live des taches faisables hors camera.

## Regle generale

Quand les cameras sont accessibles :

- lancer les runs live ;
- mesurer la latence end-to-end ;
- tester les vraies zones ;
- tester quelques objets representatifs ;
- verifier le dashboard video + metadata.

Quand les cameras ne sont pas accessibles :

- analyser les CSV / JSONL ;
- generer les graphes ;
- faire les tests Phase 4 transport metadata ;
- rediger rapport / article / diapos ;
- nettoyer le repo et preparer serveur / nouveau PC.

Les commandes ci-dessous supposent que tu es dans :

```powershell
cd C:\Users\frisa\Desktop\BenchmarkingAI\STAGELIST3N-FusionCam
```

Pour les tests live sur ce PC, utilise la vraie config locale hors Git :

```powershell
..\Phase_3_Fusion_MultiCam\config_real_zones.yaml
```

Ne pas commit cette config si elle contient les vrais RTSP / mots de passe.

## Quand Tu As Acces Aux Cameras

### 1. Smoke Test 2 Cameras

But : verifier rapidement que les flux, le modele, les vraies zones et la metadata fonctionnent.

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --config ..\Phase_3_Fusion_MultiCam\config_real_zones.yaml --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 3 --device cuda:0 --conf 0.25 --object-min-camera-votes 1 --capture-backend opencv --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam\reports\smoke_live_2cam_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam\reports\smoke_live_2cam_trace.csv --out-dir Phase_3_Fusion_MultiCam\reports\smoke_live_2cam
```

Validation :

```powershell
python Phase_4_Network_Latency\validate_metadata_jsonl.py Phase_3_Fusion_MultiCam\reports\smoke_live_2cam_metadata.jsonl --print-example
```

### 2. Test Zones Reelles 4 Cameras

But : verifier que les trois vraies zones de salle 1 declenchent correctement les alertes personne/zone.

Scenario :

- entrer dans `salle1_zone_1`, puis sortir ;
- entrer dans `salle1_zone_2`, puis sortir ;
- entrer dans `salle1_zone_3`, puis sortir.

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --config ..\Phase_3_Fusion_MultiCam\config_real_zones.yaml --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --conf 0.25 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam\reports\real_zones_4cam_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam\reports\real_zones_4cam_trace.csv --out-dir Phase_3_Fusion_MultiCam\reports\real_zones_4cam
```

### 3. Test Objets Representatifs

Il n'est pas necessaire de tester tous les objets. Pour le final, mieux vaut un scenario clair avec quelques objets visibles.

Objets conseilles :

- bouteille ;
- marteau ;
- pince ou perceuse.

Ne mets pas `--no-weak-object-alerts` ici : on veut observer `weak` et `confirmed`.

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --config ..\Phase_3_Fusion_MultiCam\config_real_zones.yaml --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 10 --device cuda:0 --conf 0.25 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam\reports\real_objects_2cam_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam\reports\real_objects_2cam_trace.csv --out-dir Phase_3_Fusion_MultiCam\reports\real_objects_2cam
```

### 4. Latence End-To-End Experience Utilisateur

But : mesurer la latence percue par un utilisateur, pas seulement la latence IA.

Methode :

1. Ouvrir `https://time.is/` sur le telephone.
2. Mettre le telephone devant `cam_02`.
3. Ouvrir `https://time.is/` sur le PC.
4. Ouvrir le dashboard ou le flux video.
5. Faire une capture ecran ou on voit l'heure PC et l'heure du telephone dans la video.

Calcul :

```text
latence_video = heure_PC_visible - heure_telephone_visible_dans_la_video
```

Mesures a faire :

- video seule ;
- dashboard video + metadata ;
- entree en zone avec apparition d'alerte.

### 5. Dashboard Reel 2 Cameras

Le dashboard visuel doit rester en 2 cameras sur ce PC pour eviter les clignotements et la surcharge.

Terminal dashboard :

```powershell
python Phase_4_Network_Latency\alert_dashboard.py --host 127.0.0.1 --port 8765 --zones-config ..\Phase_3_Fusion_MultiCam\config_real_zones.yaml
```

URL dashboard :

```text
http://127.0.0.1:8765/?cameras=cam_02,cam_07&video_base=http://127.0.0.1:8888&video_mode=iframe&source_w=704&source_h=576&overlay_delay_ms=5000&v=demo
```

Run IA + metadata HTTP :

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --config ..\Phase_3_Fusion_MultiCam\config_real_zones.yaml --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --conf 0.25 --object-min-camera-votes 1 --capture-backend opencv --no-display --no-record-video --metadata-http-url http://127.0.0.1:8765/metadata --metadata-jsonl Phase_3_Fusion_MultiCam\reports\pc_dashboard_real_2cam_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam\reports\pc_dashboard_real_2cam_trace.csv --out-dir Phase_3_Fusion_MultiCam\reports\pc_dashboard_real_2cam
```

Debug dashboard :

```powershell
Invoke-RestMethod http://127.0.0.1:8765/debug.json
```

Il faut voir :

```text
metadata_posts > 0
last_detections > 0
last_cameras = cam_02, cam_07
```

### 6. Run Final 4 Cameras

A faire seulement quand les tests courts sont valides.

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --config ..\Phase_3_Fusion_MultiCam\config_real_zones.yaml --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 20 --device cuda:0 --conf 0.25 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam\reports\final_real_4cam_metadata.jsonl --latency-trace-csv Phase_3_Fusion_MultiCam\reports\final_real_4cam_trace.csv --out-dir Phase_3_Fusion_MultiCam\reports\final_real_4cam
```

Livrables a garder :

- `summary.csv` ;
- `detections.csv` ;
- `alerts.csv` ;
- `fusion_links.csv` ;
- `track_stability.csv` ;
- `*_trace.csv` ;
- `*_metadata.jsonl`.

## Quand Tu N'as Pas Acces Aux Cameras

### 1. Analyser Les Runs Existants

Validation metadata :

```powershell
python Phase_4_Network_Latency\validate_metadata_jsonl.py Phase_3_Fusion_MultiCam\reports\real_zones_4cam_metadata.jsonl --print-example
```

Analyse latence interne :

```powershell
python -c "import pandas as pd; p='Phase_3_Fusion_MultiCam/reports/pc_latency_internal_2cam_trace.csv'; df=pd.read_csv(p); cols=[c for c in ['capture_read_ms','inference_tracking_ms','fusion_ms','alerts_ms','metadata_ms','internal_after_read_ms','total_loop_ms'] if c in df.columns]; print(df[cols].describe(percentiles=[.5,.95,.99]).round(3))"
```

Detection des pics :

```powershell
python -c "import pandas as pd; p='Phase_3_Fusion_MultiCam/reports/pc_latency_internal_2cam_trace.csv'; df=pd.read_csv(p); print('inference > 100ms:', (df['inference_tracking_ms']>100).sum()); print('capture > 50ms:', (df['capture_read_ms']>50).sum()); print(df.nlargest(10,'inference_tracking_ms')[['campaign_frame','cam_id','capture_read_ms','inference_tracking_ms','total_loop_ms']])"
```

### 2. Tests Transport Metadata Phase 4

Queue locale :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport queue --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\final_queue\alert_latency.csv
```

HTTP POST :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport http_post --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\final_http_post\alert_latency.csv
```

WebSocket :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport websocket --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\final_websocket\alert_latency.csv
```

MQTT avec Docker :

Terminal 1 :

```powershell
docker run --rm -p 1883:1883 eclipse-mosquitto:2
```

Terminal 2 :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 0 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\final_mqtt_qos0\alert_latency.csv
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 1 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\final_mqtt_qos1\alert_latency.csv
```

Resume :

```powershell
python Phase_4_Network_Latency\analyze_phase4_runs.py --runs-glob "Phase_4_Network_Latency/runs/final_*" --out-csv Phase_4_Network_Latency\runs\final_transport_summary.csv
```

### 3. Charge Metadata : WebSocket / MQTT

But : voir si la latence p95 augmente quand le nombre de messages monte.

WebSocket :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport websocket --events 1000 --rate-hz 10 --out Phase_4_Network_Latency\runs\ws_10hz\alert_latency.csv
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport websocket --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\ws_25hz\alert_latency.csv
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport websocket --events 1000 --rate-hz 50 --out Phase_4_Network_Latency\runs\ws_50hz\alert_latency.csv
```

MQTT :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 0 --events 1000 --rate-hz 10 --out Phase_4_Network_Latency\runs\mqtt_10hz\alert_latency.csv
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 0 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\mqtt_25hz\alert_latency.csv
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 0 --events 1000 --rate-hz 50 --out Phase_4_Network_Latency\runs\mqtt_50hz\alert_latency.csv
```

### 4. Graphes A Produire

Graphes prioritaires :

- latence moyenne / p95 par etape (`capture`, `inference`, `fusion`, `alerts`, `metadata`) ;
- HTTP vs WebSocket vs MQTT ;
- FPS / latence selon 1, 2, 4 cameras ;
- nombre d'alertes par type ;
- objets `weak` vs `confirmed` ;
- Phase 2 vs Phase 3.

### 5. Rapport / Article / Diapos

Sections a avancer sans cameras :

- architecture globale ;
- protocole experimental ;
- metriques ;
- calibration / homographie ;
- Phase 2 vs Phase 3 ;
- Phase 4 : separation video / IA / metadata ;
- limites actuelles ;
- travaux futurs.

### 6. Preparation Serveur / Nouveau PC

A faire hors camera :

- verifier branche Git ;
- verifier presence de `config_real_zones.yaml` ;
- preparer config serveur avec RTSP relayes ;
- verifier poids `.pt` / `.engine` ;
- preparer commandes finales serveur.

## Planning Type

### Si acces camera court

Priorite :

1. smoke test 2 cameras ;
2. test zones ;
3. chrono end-to-end.

### Si acces camera long

Priorite :

1. smoke test ;
2. zones ;
3. objets ;
4. dashboard 2 cameras ;
5. run final 20 min.

### Si pas d'acces camera

Priorite :

1. analyse CSV / JSONL ;
2. Phase 4 transport ;
3. graphes ;
4. rapport / diapos ;
5. repo / serveur.

### Si serveur GPU libre

Priorite :

1. copier la config real zones locale vers serveur ;
2. smoke test serveur ;
3. run final serveur 20-30 min.

### Si serveur GPU occupe

Priorite :

1. tests PC local 2 cameras ;
2. tests Phase 4 ;
3. latence end-to-end utilisateur ;
4. analyse et redaction.

## Conclusions Attendues

A la fin de la semaine, il faut pouvoir conclure :

- nombre de cameras viable sur PC local ;
- nombre de cameras viable sur serveur ;
- latence interne IA ;
- latence end-to-end utilisateur ;
- transport metadata recommande ;
- architecture finale recommandee ;
- limites des objets `weak` / `confirmed` ;
- validite des vraies zones.

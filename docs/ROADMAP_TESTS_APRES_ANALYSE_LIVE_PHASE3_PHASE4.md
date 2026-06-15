# Roadmap de tests apres analyse live Phase 3 / Phase 4

Date : 2026-06-10  
Objectif : finir proprement les tests live, comparer PC actuel / nouveau PC / serveur, choisir l'architecture finale, puis passer aux tests Phase 4 reseau/metadonnees.

## Resume de la situation

Les resultats live deja analyses montrent :

- le serveur Linux est le meilleur candidat pour l'IA 4 cameras ;
- le nouveau PC est un bon fallback local pour 2 a 4 cameras sans affichage ;
- le PC actuel reste utile pour developpement, analyse, recorded campaign et debug ;
- 8 cameras ne sont pas encore validees proprement ;
- l'affichage et l'enregistrement dans `run_live_campaign.py` degradent fortement les performances ;
- la fusion fonctionne, mais elle reste sensible a la calibration, a la visibilite effective des objets/personnes et au seuil de fusion.

Conclusion actuelle :

```text
run_live_campaign.py = benchmark IA/fusion
MediaMTX / FFmpeg / GStreamer = affichage et enregistrement video
JSONL puis WebSocket/MQTT = metadonnees, bbox, alertes, global_id
```

## Regles generales pour tous les tests

Toujours noter :

- machine utilisee ;
- modele ;
- format (`pt` ou `fp32_engine`) ;
- cameras ;
- protocole RTSP (`tcp` ou `udp`) ;
- `gst-latency-ms` ;
- affichage active ou non ;
- recording actif ou non ;
- duree ;
- chemin du dossier de sortie ;
- problemes visibles : camera absente, flux coupe, grosse latence, alertes incoherentes.

Apres chaque run, regarder :

```text
phase3/summary.csv
phase3/alerts.csv
phase3/fusion_links.csv
phase3/track_stability.csv
ablation/fusion_threshold_ablation.csv
logs/*.txt
```

Metriques principales :

- FPS effectif ;
- latence moyenne IA ;
- latence p95 IA ;
- alertes totales ;
- alertes objets confirmees ;
- alertes personne/zone ;
- nombre de liens de fusion ;
- `global_id_switches` ;
- cameras reellement presentes dans `detections.csv` ;
- erreurs RTSP / GStreamer / decodeur dans les logs.

Important : `latency_mean_ms` mesure surtout le temps IA apres reception de frame. Ce n'est pas la latence complete camera -> passerelle -> serveur -> IA -> dashboard.

## Phase 0 - Verification setup

### PC actuel, nouveau PC et serveur

Verifier que les scripts sont a jour :

```powershell
git pull
python Phase_3_Fusion_MultiCam\run_live_campaign.py --help | findstr metadata
python Phase_3_Fusion_MultiCam\run_recorded_campaign.py --help | findstr dataset
dir Phase_4_Network_Latency
```

Sur Linux :

```bash
git pull
python Phase_3_Fusion_MultiCam/run_live_campaign.py --help | grep metadata
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --help | grep dataset
ls Phase_4_Network_Latency
```

Verifier GPU :

Windows :

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Linux :

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Livrable :

- capture ou note rapide du GPU ;
- version du commit Git ;
- confirmation que `--metadata-jsonl` existe.

## Phase 1 - Comparaison machines stricte

But : comparer PC actuel / nouveau PC / serveur avec le meme protocole.

Pour une comparaison juste, utiliser :

```text
4 cameras
yolov8s
format pt si possible partout
TCP100
GStreamer
no display
no record
10 min
```

Si `pt` est trop lent sur le PC actuel, faire aussi `fp32_engine`, mais ne pas melanger les conclusions.

### Test M1 - PC actuel, 4 cameras, reference

PowerShell :

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\pc1_m1_4cam_yolov8s_pt_tcp100_10min
```

Si `pt` n'est pas utilisable, run complementaire engine :

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\pc1_m1_4cam_yolov8s_engine_tcp100_10min
```

### Test M2 - Nouveau PC, 4 cameras, reference

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\pc2_m2_4cam_yolov8s_pt_tcp100_10min
```

### Test M3 - Serveur Linux, 4 cameras, reference

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam/reports/server_m3_4cam_yolov8s_pt_tcp100_10min
```

### Analyse attendue Phase 1

Creer un tableau :

| Machine | Format | FPS | Lat. moy. | Lat. p95 | Alertes | Objets confirmes | Fusion links | ID switches | Cameras detectees | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| PC actuel | pt/engine | | | | | | | | | |
| Nouveau PC | pt | | | | | | | | | |
| Serveur | pt | | | | | | | | | |

Decision attendue :

- machine finale pour IA 4 cameras ;
- machine fallback ;
- machine a garder uniquement pour developpement/analyse ;
- besoin ou non de retester 8 cameras.

## Phase 2 - Tests modeles live

But : verifier si d'autres modeles sont interessants en live.

Regle :

- faire les tests modeles sur une seule machine stable ;
- idealement serveur ;
- sinon nouveau PC ;
- ne pas changer cameras/protocole/duree pendant la comparaison.

Configuration conseillee :

```text
4 cameras
TCP100
GStreamer
no display
no record
5 min par modele
```

### Test modele yolov8n

PowerShell :

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8n --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\modeltest_yolov8n_pt_4cam_tcp100_5min
```

Linux :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8n --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam/reports/modeltest_yolov8n_pt_4cam_tcp100_5min
```

### Test modele yolov8s

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\modeltest_yolov8s_pt_4cam_tcp100_5min
```

Linux :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam/reports/modeltest_yolov8s_pt_4cam_tcp100_5min
```

### Test modele yolo11s

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolo11s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\modeltest_yolo11s_pt_4cam_tcp100_5min
```

Linux :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolo11s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam/reports/modeltest_yolo11s_pt_4cam_tcp100_5min
```

### Test rtdetr-l en smoke test uniquement

RT-DETR est potentiellement plus lourd. Commencer avec 2 cameras pendant 3 min :

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models rtdetr-l --formats pt --cameras cam_02,cam_07 --duration-min 3 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\modeltest_rtdetr_l_pt_2cam_tcp100_3min
```

Linux :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models rtdetr-l --formats pt --cameras cam_02,cam_07 --duration-min 3 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam/reports/modeltest_rtdetr_l_pt_2cam_tcp100_3min
```

### Analyse attendue Phase 2

Tableau :

| Modele | Format | FPS | Lat. moy. | Lat. p95 | Detections | Alertes | Objets confirmes | Fusion links | ID switches | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| yolov8n | pt | | | | | | | | | |
| yolov8s | pt | | | | | | | | | |
| yolo11s | pt | | | | | | | | | |
| rtdetr-l | pt | | | | | | | | | |

Decision attendue :

- modele live recommande ;
- modele rapide fallback ;
- modele qualite si meilleur ;
- modele a exclure si trop lent ou instable.

Hypothese actuelle :

- `yolov8s` = meilleur compromis ;
- `yolov8n` = utile si besoin de FPS ;
- `yolo11s` = a verifier ;
- `rtdetr-l` = probablement trop lourd en multi-camera live.

## Phase 3 - Decision calibration / homographie

But : ne pas refaire toutes les calibrations inutilement, mais ne pas ignorer une fusion incoherente.

### Quand verifier l'homographie

Verifier calibration/homographie si un de ces symptomes apparait :

- une camera detecte beaucoup mais ne fusionne presque jamais ;
- objets visibles par 2 cameras mais jamais confirmes ;
- beaucoup plus de fusion a 1.5 m ou 2.0 m qu'a 1.0 m ;
- beaucoup de `global_id_switches` ;
- positions au sol visiblement fausses ;
- alertes zone alors que la personne semble hors zone ;
- une camera apparait dans le manifeste mais pas dans `detections.csv`.

Cas deja suspect :

```text
Nouveau PC : cam_07 absente des detections sur certains runs 4 cameras TCP100/TCP50.
```

### Verification rapide avant recalibration

1. Lancer un live court avec affichage annote pour verifier zones, footpoints et bbox :

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 2 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --display-mode annotated --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\calib_visual_check_4cam_2min
```

2. Regarder si les points de pied projetes tombent au bon endroit.
3. Regarder si les zones rouges correspondent bien au sol reel.
4. Regarder si la meme personne a des positions proches dans plusieurs cameras.

### Decision

Si la projection est visiblement correcte :

```text
Continuer les tests live.
Ne pas perdre de temps sur une recalibration complete.
```

Si la projection est mauvaise :

```text
Corriger homographie/camera concernee.
Relancer un test 2 cameras sur la paire concernee.
Puis relancer le test 4 cameras final.
```

## Phase 4 - Tests objets

But : verifier le systeme weak/confirmed et la confirmation multi-camera des objets.

Probleme actuel :

- les objets sont plus difficiles a confirmer que les personnes ;
- ils sont petits ;
- ils sont souvent visibles par une seule camera ;
- `object_min_camera_votes=2` est strict ;
- augmenter le seuil de fusion peut aider mais augmente le risque de mauvaise association.

### Test O1 - Objets weak + confirmed

Utiliser 2 cameras avec recouvrement fort, par exemple `cam_02,cam_07` ou la paire la plus pertinente selon la salle.

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\objecttest_2cam_weak_confirmed_tcp100_10min
```

Ne pas mettre `--no-weak-object-alerts` ici. Le but est d'observer les alertes faibles et confirmees.

### Test O2 - Objets confirmed only

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\objecttest_2cam_confirmed_only_tcp100_5min
```

### Test O3 - Seuil de fusion objets a 1.5 m

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --fusion-distance-m 1.5 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --out-dir Phase_3_Fusion_MultiCam\reports\objecttest_2cam_confirmed_1m5_tcp100_5min
```

Analyse attendue :

| Test | Weak | Confirmed | Objets confirmes | Faux positifs visibles | Fusion links | Verdict |
|---|---:|---:|---:|---:|---:|---|
| O1 weak+confirmed | | | | | | |
| O2 confirmed only 1.0 m | | | | | | |
| O3 confirmed only 1.5 m | | | | | | |

Decision :

- garder weak+confirmed si confirmed only manque trop d'objets ;
- garder 1.0 m si 1.5 m cree trop d'associations douteuses ;
- utiliser 1.5 m uniquement si les objets sont clairement sous-fusionnes.

## Phase 5 - Test final long

But : produire un run final defensible pour rapport/article.

Faire sur la meilleure machine apres Phase 1 et Phase 2.

Configuration probable :

```text
serveur Linux
4 cameras zone_1
yolov8s pt ou meilleur modele trouve
TCP100
no display
no record
metadata JSONL active
20 a 30 min
```

### Commande Windows

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 30 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam\reports\final_4cam_metadata.jsonl --out-dir Phase_3_Fusion_MultiCam\reports\final_4cam_yolov8s_pt_tcp100_30min
```

### Commande Linux

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 30 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam/reports/final_4cam_metadata.jsonl --out-dir Phase_3_Fusion_MultiCam/reports/final_4cam_yolov8s_pt_tcp100_30min
```

Validation metadata :

Windows :

```powershell
python Phase_4_Network_Latency\validate_metadata_jsonl.py Phase_3_Fusion_MultiCam\reports\final_4cam_metadata.jsonl --print-example
```

Linux :

```bash
python Phase_4_Network_Latency/validate_metadata_jsonl.py Phase_3_Fusion_MultiCam/reports/final_4cam_metadata.jsonl --print-example
```

Livrable :

- summary final ;
- tableau performance ;
- exemple JSONL ;
- screenshot ou extrait log d'alerte ;
- conclusion machine/modele.

## Phase 6 - Latence complete camera -> serveur -> IA -> alerte

But : mesurer la vraie latence complete, pas seulement la latence IA.

Architecture actuelle :

```text
Cameras
  -> PC passerelle
  -> MediaMTX relay
  -> serveur Capsec
  -> inference IA
  -> fusion / alertes
  -> metadata / dashboard
```

### Test L1 - Latence video pure

Methode :

1. Placer un telephone/ecran avec chrono visible devant la camera.
2. Lire le flux relaye cote serveur/dashboard.
3. Filmer ou observer la difference entre l'heure visible dans l'image et l'heure reelle.

Mesure :

```text
latence_video = temps_reception_flux - temps_visible_dans_image
```

### Test L2 - Latence alerte complete

Methode :

1. Declencher une entree en zone ou poser un objet.
2. Noter le temps visible dans le chrono image.
3. Lire le timestamp `created_epoch_ms` dans le JSONL.

Mesure :

```text
latence_alerte_complete = created_epoch_ms - temps_evenement_visible
```

Livrable :

| Test | Camera | Evenement | Latence video | Latence IA | Latence alerte complete | Commentaire |
|---|---|---|---:|---:|---:|---|
| L1 | | | | | | |
| L2 | | | | | | |

## Phase 7 - Phase 4 metadata et transports alertes

But : comparer les transports de metadonnees/alertes une fois la Phase 3 stabilisee.

Ordre :

1. JSONL ;
2. queue locale ;
3. HTTP POST ;
4. WebSocket ;
5. MQTT ;
6. WebRTC/WHEP pour video, si temps disponible.

### Test P4-1 - JSONL Phase 3

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats pt --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam\reports\phase4_metadata_test.jsonl
python Phase_4_Network_Latency\validate_metadata_jsonl.py Phase_3_Fusion_MultiCam\reports\phase4_metadata_test.jsonl --print-example
```

### Test P4-2 - Queue locale

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport queue --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_queue\alert_latency.csv
```

Linux :

```bash
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport queue --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_queue/alert_latency.csv
```

### Test P4-3 - HTTP POST

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport http_post --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_http_post\alert_latency.csv
```

Linux :

```bash
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport http_post --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_http_post/alert_latency.csv
```

### Test P4-4 - WebSocket

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport websocket --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_websocket\alert_latency.csv
```

Linux :

```bash
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport websocket --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_websocket/alert_latency.csv
```

### Test P4-5 - MQTT

Demarrer Mosquitto :

Windows, terminal 1 :

```powershell
mosquitto -p 1883
```

Linux, terminal 1 :

```bash
mosquitto -p 1883
```

Terminal 2 :

```powershell
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 0 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_mqtt_qos0\alert_latency.csv
python Phase_4_Network_Latency\alert_delivery_benchmark.py --transport mqtt --qos 1 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency\runs\alert_mqtt_qos1\alert_latency.csv
```

Linux :

```bash
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 0 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_mqtt_qos0/alert_latency.csv
python Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 1 --events 1000 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_mqtt_qos1/alert_latency.csv
```

### Test P4-6 - WebRTC / WHEP video

Seulement apres les tests prioritaires.

But :

- verifier que la video peut etre affichee hors pipeline IA ;
- mesurer la latence video ;
- verifier la possibilite de superposer les bbox metadata.

Architecture visee :

```text
Camera RTSP -> MediaMTX -> WebRTC/WHEP -> navigateur
IA Phase 3 -> JSONL/WebSocket/MQTT -> dashboard overlay
```

Livrable Phase 4 :

| Transport | Machine | Events | Rate | Lat. moy. | Lat. p95 | Pertes | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Queue | | | | | | | |
| HTTP POST | | | | | | | |
| WebSocket | | | | | | | |
| MQTT QoS0 | | | | | | | |
| MQTT QoS1 | | | | | | | |
| WebRTC/WHEP | | | | | | | |

## Ordre recommande sur 2-3 jours

### Jour 1 - Comparaison machines et modele

1. Phase 0 setup.
2. M1, M2, M3 : comparaison 4 cameras.
3. Si le serveur est stable, faire les tests modeles sur serveur.
4. Sinon faire les tests modeles sur nouveau PC.

Livrable fin jour 1 :

- tableau machines ;
- tableau modeles ;
- machine recommandee ;
- modele recommande.

### Jour 2 - Objets, calibration et test final

1. Tests objets O1/O2/O3.
2. Controle calibration rapide si symptomes.
3. Correction homographie seulement si necessaire.
4. Test final long 20-30 min.

Livrable fin jour 2 :

- decision seuil fusion ;
- decision weak/confirmed objets ;
- run final long ;
- JSONL final valide.

### Jour 3 - Phase 4

1. JSONL.
2. Queue locale.
3. HTTP.
4. WebSocket.
5. MQTT.
6. WebRTC/WHEP si temps.

Livrable fin jour 3 :

- tableau latence transports ;
- schema architecture finale ;
- decision transport metadata ;
- decision video separee.

## Decision finale attendue

A la fin de cette roadmap, il faut pouvoir ecrire clairement :

```text
La solution finale recommandee est :
- IA sur serveur ou nouveau PC ;
- 4 cameras validees ;
- 8 cameras non validees ou validees selon nouveau test ;
- modele retenu ;
- protocole RTSP retenu ;
- seuil de fusion retenu ;
- affichage et recording separes ;
- metadonnees publiees via JSONL puis WebSocket/MQTT ;
- video distribuee via MediaMTX/WebRTC.
```


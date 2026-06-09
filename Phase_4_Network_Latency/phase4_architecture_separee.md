# Phase 4 - Architecture separee video / IA / metadonnees

## Objectif du jeudi

Valider une architecture plus propre que le mode live actuel, qui melange trop de roles dans `run_live_campaign.py` :

- acquisition RTSP ;
- inference IA ;
- fusion multi-camera ;
- affichage ;
- enregistrement ;
- audit ;
- generation de rapports.

L'objectif est de garder `run_live_campaign.py` comme outil de benchmark IA/fusion, puis de separer l'affichage video et les metadonnees IA.

## Architecture cible

```text
Cameras RTSP
  |
  +--> Flux video
  |      RTSP -> MediaMTX -> WebRTC/WHEP -> dashboard ou recording
  |
  +--> Flux IA
         RTSP -> Phase 3 -> detections / bbox / global_id / alertes
                           -> JSONL d'abord
                           -> WebSocket ou MQTT ensuite
```

Dans cette architecture, la video peut rester fluide a 25 FPS via MediaMTX/WebRTC, tandis que l'IA publie des metadonnees a une frequence plus basse, par exemple 4 a 10 FPS par camera. L'overlay peut donc etre moins fluide que la video, mais la video n'est plus bloquee par l'IA.

## Test 1 - Metadata JSONL

But : verifier que le pipeline IA peut produire des metadonnees exploitables sans dashboard complexe.

Avant le test, utiliser un fichier neuf ou supprimer l'ancien. Le publisher ecrit en append, donc un fichier reutilise peut contenir plusieurs runs et fausser l'analyse des frames.

```powershell
Remove-Item Phase_3_Fusion_MultiCam/reports/live_metadata_test.jsonl -ErrorAction SilentlyContinue
```

Commande 2 cameras :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam/reports/live_metadata_test.jsonl
```

Verification :

```powershell
python Phase_4_Network_Latency/validate_metadata_jsonl.py Phase_3_Fusion_MultiCam/reports/live_metadata_test.jsonl --print-example
```

Points a verifier :

- le fichier JSONL est cree ;
- les enveloppes ont le schema `benchmarkingai.phase3.metadata.v1` ;
- les timestamps `created_epoch_ms` et `created_epoch_s` sont presents ;
- les `camera_id` sont presents dans les detections ;
- les `bbox_px` sont presentes ;
- les `global_id` sont presents ;
- les alertes sont presentes si des violations sont detectees ;
- le lag metadata estime reste faible.

## Test 2 - IA seule + video separee

But : tester le principe ou l'IA tourne sans affichage ni enregistrement, pendant que la video est affichee ou enregistree par un autre outil.

Avant le test :

```powershell
Remove-Item Phase_3_Fusion_MultiCam/reports/live_metadata_cam02_cam07.jsonl -ErrorAction SilentlyContinue
```

Commande IA :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam/reports/live_metadata_cam02_cam07.jsonl
```

Verification :

```powershell
python Phase_4_Network_Latency/validate_metadata_jsonl.py Phase_3_Fusion_MultiCam/reports/live_metadata_cam02_cam07.jsonl --print-example
```

Pendant ce test, l'affichage ou l'enregistrement doit etre fait separement, par exemple avec MediaMTX, VLC, FFmpeg ou un futur dashboard WebRTC.

## Test 3 - Serveur Linux

But : preparer le serveur meme si les cameras ne sont pas encore accessibles.

Installation :

```bash
python3 -m venv ~/aivenv
source ~/aivenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
nvidia-smi
```

Si les videos recorded sont disponibles sur le serveur :

```bash
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats pt --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960 --object-min-camera-votes 2 --no-weak-object-alerts
```

Remarque : les engines TensorRT `.engine` sont souvent lies a la machine/GPU/TensorRT. Sur le serveur, commencer par `.pt`, puis regenerer les engines localement si besoin.

## Exemple JSONL attendu

Une ligne JSONL correspond a une enveloppe de metadonnees pour une frame de campagne :

```json
{
  "schema": "benchmarkingai.phase3.metadata.v1",
  "created_epoch_ms": 1780488828043.868,
  "created_epoch_s": 1780488828.043868,
  "frame": 1693,
  "run_label": "V4_person_objects_yolov8s_fp32_engine",
  "model_version": "V4",
  "model": "yolov8s",
  "format": "fp32_engine",
  "detections": [
    {
      "camera_id": "cam_02",
      "track_id": 12,
      "global_id": 2,
      "class_id": 11,
      "class_name": "person",
      "confidence": 0.6013,
      "bbox_px": [180.2, 91.4, 241.7, 288.9],
      "foot_point_px": [211.0, 288.9],
      "position_m": [6.98, 5.011],
      "zones": ["zone_1"],
      "timestamp": 1780488828.0201
    }
  ],
  "alerts": [
    {
      "alert_id": "4b4e432f",
      "alert_type": "zone_violation_person",
      "alert_level": "confirmed",
      "global_id": 2,
      "zone_id": "zone_1",
      "class_name": "person",
      "position_m": [6.98, 5.011],
      "cameras": ["cam_02"],
      "confidence": 0.6013,
      "timestamp": 1780488828.043868
    }
  ]
}
```

## Comparaison architectures

| Architecture | Video | IA | Metadonnees | Avantage | Probleme |
| --- | --- | --- | --- | --- | --- |
| Tout dans `run_live_campaign.py` | Meme process | Meme process | logs/CSV | Simple a lancer, bon benchmark | Trop lourd en live, affichage/recording ralentissent l'ensemble |
| IA seule + video separee | MediaMTX/RTSP/WebRTC | Phase 3 | JSONL | Plus stable, roles separes, video fluide possible | Synchronisation video/metadonnees a verifier |
| Serveur IA distant | Camera -> serveur | Serveur GPU | WebSocket/MQTT | Plus de puissance calcul, scalable | Latence reseau, acces cameras, securite, synchro |

## Decision recommandee

1. Garder `run_live_campaign.py` comme benchmark IA/fusion.
2. Utiliser une chaine separee pour afficher et enregistrer la video.
3. Publier les metadonnees d'abord en JSONL.
4. Valider le format, les timestamps, les bbox, les `global_id` et les alertes.
5. Passer a WebSocket ou MQTT seulement quand le JSONL est propre.

## Livrables du jeudi

- schema d'architecture separee ;
- tableau avantages / limites ;
- exemple JSONL de metadonnees ;
- validation du fichier JSONL avec `validate_metadata_jsonl.py` ;
- decision sur l'architecture Phase 4.

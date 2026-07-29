# Mesurer la latence end-to-end par etape

Objectif : distinguer la latence IA interne de la latence complete de l'architecture.

## 1. Ce que mesure deja le projet

Les champs `latency_mean_ms`, `latency_p95_ms`, etc. dans `summary.csv` mesurent surtout :

```text
frame deja recue par le programme
  -> inference / tracking camera
```

Ce n'est pas la latence complete camera -> alerte.

## 2. Nouvelle trace interne par etape

`run_live_campaign.py` accepte maintenant :

```text
--latency-trace-csv <chemin.csv>
```

Cette option ecrit une ligne par camera et par frame traitee avec :

- temps de lecture `cap.read()` ;
- temps inference/tracking ;
- temps fusion ;
- temps regles d'alertes ;
- temps publication metadata ;
- temps ecriture video si recording actif ;
- temps affichage si display actif ;
- temps total de boucle interne.

Exemple PC local :

```powershell
python Phase_3_Fusion_MultiCam\run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --latency-trace-csv Phase_3_Fusion_MultiCam\reports\latency_trace_pc.csv
```

Exemple serveur :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --capture-backend opencv --no-display --no-record-video --latency-trace-csv Phase_3_Fusion_MultiCam/reports/latency_trace_server.csv
```

## 3. Colonnes importantes

| Colonne | Signification |
|---|---|
| `capture_read_ms` | Temps passe dans `cap.read()` pour lire une frame depuis le flux RTSP |
| `inference_tracking_ms` | Detection + tracking + projection par camera |
| `fusion_ms` | Association inter-camera |
| `alerts_ms` | Verification zones/personnes/objets |
| `metadata_ms` | Ecriture/publication metadata |
| `record_write_ms` | Cout d'ecriture video si recording actif |
| `display_ms` | Cout d'affichage si display actif |
| `internal_after_read_ms` | Temps interne entre fin de lecture frame et metadata produite |
| `total_loop_ms` | Temps total d'une iteration live |

## 4. Ce que cette trace ne mesure pas seule

Elle ne mesure pas automatiquement :

```text
scene reelle
  -> camera encode RTSP
  -> reseau camera
  -> buffer camera / FFmpeg / MediaMTX
  -> moment ou cap.read() recupere enfin la frame
```

Pour cette partie, il faut une reference temporelle visible dans l'image.

## 5. Mesure complete scene -> alerte

Methode recommandee :

1. Mettre un telephone ou un ecran avec chrono visible dans la camera.
2. Faire un evenement clair : entree en zone, objet pose, passage devant une ligne.
3. Noter le temps visible dans l'image au moment de l'evenement.
4. Comparer avec `metadata_end_epoch`, `alerts_end_epoch` ou `created_epoch_ms` dans le JSONL.

Formule :

```text
latence_complete = temps_alerte_systeme - temps_visible_dans_image
```

## 6. Mesure architecture serveur avec passerelle

Architecture :

```text
Cameras IP <CAMERA_NET>.x
  -> PC passerelle Windows <GATEWAY_IP> / <SERVER_IP>
  -> FFmpeg
  -> MediaMTX rtsp://<SERVER_IP>:8554/cam_XX
  -> serveur Linux GPU
  -> run_live_campaign.py
  -> detection / tracking / fusion / alertes
```

Pour separer les etapes :

| Etape | Methode |
|---|---|
| Camera -> PC passerelle | chrono visible dans l'image lue sur le PC passerelle |
| PC passerelle -> MediaMTX -> serveur | comparer le meme chrono visible cote serveur |
| Serveur interne | `--latency-trace-csv` |
| Alerte complete | chrono visible vs timestamp alerte |

## 7. Interpretation

Si `inference_tracking_ms` est faible mais que l'alerte arrive tres tard, le probleme ne vient probablement pas de l'IA. Il vient plutot de :

- buffer RTSP ;
- FFmpeg/OpenCV qui lit des frames anciennes ;
- MediaMTX / relay ;
- reseau ;
- affichage ou recording dans le meme process ;
- machine trop chargee qui accumule du retard.

La trace CSV permet donc de separer :

```text
latence transport / buffering
vs
latence IA
vs
latence fusion / alertes
vs
latence affichage / recording
```

# Analyse des resultats live Phase 3 / Phase 4

Date : 2026-06-12  
Sources analysees :
- `Phase_3_Fusion_MultiCam/reports/` sur le PC principal ;
- `results_to_old_pc-20260609T130312Z-3-001/results_to_old_pc/` ;
- `DRIVE_UPLOAD_RESULTATS_REUNION_CLEAN/` ;
- archives PC2 et serveur extraites pour la reunion.

## 1. Synthese executive

Les resultats de la semaine confirment trois points importants.

Premier point : la Phase 3 apporte une vraie valeur par rapport a la Phase 2 pour les violations de zone/personne. La Phase 2 travaille camera par camera en image-space, alors que la Phase 3 projette les detections au sol, fait du tracking, fusionne les cameras et produit des alertes globales. Sur les videos V4, la Phase 3 ameliore surtout la precision et le F1 des violations personne/zone, tout en ajoutant les `global_id`, les liens de fusion et une logique multi-camera exploitable.

Deuxieme point : pour le live multi-camera, le serveur est nettement le meilleur candidat. Sur 4 cameras, il tient environ 25 FPS agreges avec une latence IA moyenne autour de 9 ms. Le nouveau PC tient correctement 2 a 4 cameras sans affichage, mais il n'est pas encore valide pour 8 cameras. Le PC actuel reste utile pour developpement, debug, tests courts et analyse, mais il est limite pour du live multi-camera fiable.

Troisieme point : les objets interdits restent le point faible. Le mode `weak + confirmed` detecte beaucoup d'objets, mais genere beaucoup d'alertes faibles. Le mode `confirmed only` est trop strict dans plusieurs runs : il supprime le bruit mais rate beaucoup d'objets. Le seuil de fusion a 1.5 m augmente les confirmations objets, mais doit etre justifie par une verification visuelle de la calibration/homographie pour eviter les mauvaises associations.

## 2. Phase 2 vs Phase 3 sur videos V4

Campagne de reference : `campaign_zone1_20260527_135237`.

### Objets interdits

| Systeme | Precision | Recall | F1 | FP moyens | Delai moyen |
|---|---:|---:|---:|---:|---:|
| Phase 2 TAD | 0.403 | 0.538 | 0.448 | 10.0 | 63.1 s |
| Phase 3 TAD | 0.120 | 0.567 | 0.147 | 2221.8 | 1.44 s |

Interpretation :
- la Phase 3 detecte plus vite les objets quand elle les voit ;
- le recall est legerement superieur a la Phase 2 ;
- la precision objet chute fortement a cause des nombreuses alertes faibles et faux positifs ;
- pour les objets, la Phase 3 est interessante comme systeme d'alerte sensible, mais pas encore comme systeme strictement fiable sans filtrage supplementaire.

Conclusion objets : il faut conserver le systeme `weak + confirmed`, mais presenter clairement que les alertes faibles servent de signal de suspicion. Les alertes confirmees sont plus fiables, mais trop rares avec `object_min_camera_votes=2`.

### Violations personne / zone

| Systeme | Precision | Recall | F1 | FP moyens | Delai moyen |
|---|---:|---:|---:|---:|---:|
| Phase 2 TRD | 0.312 | 0.524 | 0.386 | 6.0 | -1.33 s |
| Phase 3 TRD | 0.474 | 0.554 | 0.416 | 7.8 | 0.82 s |

Interpretation :
- la Phase 3 est meilleure en precision ;
- le recall reste comparable ;
- le F1 est meilleur ;
- les alertes deviennent globales au lieu d'etre separees par camera ;
- les `global_id` permettent de suivre un meme objet/personne entre cameras.

Conclusion Phase 2 vs Phase 3 : scientifiquement, la Phase 3 est plus defendable pour les personnes/zones. Pour les objets, elle doit etre presentee comme une base multi-camera prometteuse mais encore sensible a la calibration, a la visibilite multi-camera et au seuil de fusion.

## 3. Comparaison des machines

### PC principal

Runs representatifs :

| Run | Cameras | Backend effectif | FPS | Lat. moy. | Lat. p95 | Liens fusion | ID switches | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `pc1_smoke_gstreamer_cam02` | 1 | GStreamer | 24.72 | 21.5 ms | 23.6 ms | 0 | 0 | OK 1 camera |
| `pc1_m1_4cam_yolov8s_engine_opencv_10min` | 4 | FFmpeg | 14.27 | 15.1 ms | 19.1 ms | 333 | 3 | limite |
| `campaign_zone1_live_20260603_105924` | 4 | GStreamer TCP100 | 16.11 | 14.8 ms | 18.2 ms | 2648 | 11 | acceptable test |
| `campaign_zone1_live_20260603_112024` | 8 | GStreamer TCP100 | 8.35 | 14.3 ms | 16.8 ms | 628 | 4 | non valide pour 8 cams |

Le PC principal peut servir pour developper et debugger. Il peut faire tourner 1 ou 2 cameras proprement, 4 cameras de maniere limitee, mais 8 cameras ne sont pas validees. La latence IA affichee reste faible, mais elle ne represente pas la latence visuelle complete camera -> reception -> IA -> alerte. Le retard observe en vrai vient probablement de l'accumulation de retard dans les flux RTSP/FFmpeg/OpenCV et du faible FPS effectif.

### Nouveau PC

Runs representatifs :

| Run | Cameras | Backend effectif | FPS | Lat. moy. | Lat. p95 | Liens fusion | ID switches | Remarque |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `pc2_direct_2cam_tcp100_no_ui_5min` | 2 | FFmpeg | 23.83 | 17.2 ms | 22.3 ms | 1845 | 9 | bon |
| `pc2_direct_4cam_tcp100_no_ui_10min` | 4 | FFmpeg | 18.64 | 16.3 ms | 20.9 ms | 5238 | 20 | correct, cam_07 absente |
| `pc2_direct_4cam_udp50_no_ui_5min` | 4 | FFmpeg | 17.53 | 16.5 ms | 21.3 ms | 1653 | 15 | cam_07 presente |
| `pc2_direct_8cam_tcp100_no_ui_5min` | 8 | FFmpeg | 9.46 | 15.9 ms | 20.6 ms | 38 | 1 | non valide |
| `pc2_direct_2cam_tcp100_display_5min` | 2 | FFmpeg | 8.08 | 22.0 ms | 29.3 ms | 0 | 0 | affichage trop couteux |

Le nouveau PC est meilleur que le PC principal pour 2 a 4 cameras sans affichage. Il n'est pas valide pour 8 cameras. L'affichage dans `run_live_campaign.py` degrade fortement les performances. Point important : plusieurs runs demandaient GStreamer, mais les manifests indiquent souvent un backend effectif FFmpeg. Il faut donc eviter de conclure que GStreamer a ete compare proprement sur PC2 si le backend reel n'etait pas GStreamer.

### Serveur

Runs representatifs :

| Run | Cameras | Format | FPS | Lat. moy. | Lat. p95 | Liens fusion | ID switches | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `server_relay_2cam_tcp100_5min` | 2 | pt | 25.04 | 12.2 ms | 17.9 ms | 1727 | 10 | tres bon |
| `server_relay_4cam_tcp100_10min` | 4 | pt | 25.01 | 8.8 ms | 11.4 ms | 14280 | 59 | meilleur candidat |
| `server_metadata_4cam_yolov8s_engine_10min` | 4 | engine | 25.02 | 8.8 ms | 11.1 ms | 10135 | 27 | meilleur run live |
| `server_relay_8cam_tcp100_5min` | 8 | pt | n/a | n/a | n/a | n/a | n/a | incomplet |

Le serveur est le meilleur choix pour l'IA live 4 cameras. Il tient environ 25 FPS agreges avec la latence IA la plus faible. Il faut encore valider proprement 8 cameras si l'objectif final reste 8 cameras.

## 4. Conclusion machine

| Machine | Role recommande |
|---|---|
| Serveur Linux | IA live principale pour 4 cameras, et candidat pour 8 cameras si nouveau test valide |
| Nouveau PC | fallback local pour 2 a 4 cameras sans affichage |
| PC principal | developpement, analyse, recorded campaign, debug, tests courts |

Decision technique : il ne faut pas utiliser `run_live_campaign.py` pour tout faire en meme temps. L'architecture finale doit separer l'IA, la video, l'affichage, l'enregistrement et les metadonnees.

## 5. Comparaison des modeles live sur serveur

Tests serveur, principalement sur 2 cameras avec objets.

| Modele | Format | Cameras | FPS | Lat. moy. | Lat. p95 | Objets confirmes | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| yolov8n | engine | 2 | 25.02 | 11.0 ms | 16.3 ms | 0 | rapide, fallback |
| yolov8n | pt | 2 | 24.91 | 13.5 ms | 20.7 ms | 0 | rapide mais moins convaincant |
| yolov8s | engine | 2 | 24.91 | 12.8 ms | 18.5 ms | 4 | meilleur compromis |
| yolov8s | pt | 2 | 24.96 | 12.5 ms | 19.2 ms | 3 | bon compromis |
| yolo11s | engine | 2 | 24.84 | 12.4 ms | 17.0 ms | 0 | pas clairement meilleur |
| yolo11s | pt | 2 | 25.01 | 13.6 ms | 20.2 ms | 0 | pas clairement meilleur |
| rtdetr-l | pt | 2 | 15.16 | 32.3 ms | 37.3 ms | 1 | trop lourd |
| rtdetr-l | pt | 4 | 7.58 | 31.9 ms | 37.4 ms | 7 | trop lent pour live multi-camera |

Conclusion modeles :
- `yolov8s fp32_engine` est le meilleur choix live actuel ;
- `yolov8n fp32_engine` est le fallback si le FPS devient prioritaire ;
- `yolo11s` n'apporte pas encore assez de gain pour remplacer yolov8s ;
- `rtdetr-l` est trop lourd en live multi-camera, meme s'il peut detecter plus d'objets dans certains cas.

## 6. Objets interdits et systeme weak / confirmed

Runs serveur utiles :

| Run | Mode | Objets faibles | Objets confirmes | Interpretation |
|---|---|---:|---:|---|
| `server_objects_yolov8s_engine_weak_confirmed_10min` | weak + confirmed | 128 | 0 | sensible mais bruite |
| `server_objects_yolov8s_engine_confirmed_only_10min` | confirmed only | 0 | 0 | trop strict |
| `server_objects_yolov8s_engine_fusion_1m5_5min` | confirmed, seuil 1.5 m | 118 | 6 | confirme plus, mais risque d'association |
| `server_metadata_4cam_yolov8s_engine_10min` | 4 cameras | 215 | 3 | confirme quelques objets |

Le comportement observe est coherent avec le probleme : les objets sont petits, souvent visibles par une seule camera, et donc difficiles a confirmer avec `object_min_camera_votes=2`.

Decision recommandee :
- garder `weak + confirmed` pour les tests objets ;
- ne pas utiliser `confirmed only` comme seul mode, car il manque trop d'objets ;
- presenter les alertes faibles comme suspicion et les alertes confirmees comme signal plus fiable ;
- tester 1.5 m uniquement apres verification visuelle de la calibration.

## 7. Fusion et seuil de fusion

Les ablations montrent que plus le seuil augmente, plus le systeme fusionne de tracks.

Exemple serveur 4 cameras :

| Seuil | Liens predits | IDs uniques | ID switches |
|---:|---:|---:|---:|
| 0.5 m | 3221 | 59 | 27 |
| 1.0 m | 4180 | 41 | 27 |
| 1.5 m | 4490 | 32 | 27 |
| 2.0 m | 4534 | 26 | 27 |

Interpretation :
- 0.5 m est probablement trop strict ;
- 1.0 m est un bon seuil conservateur ;
- 1.5 m augmente les fusions et peut aider pour les objets ;
- 2.0 m risque de fusionner des detections qui ne devraient pas l'etre.

Decision recommandee :
- garder 1.0 m comme seuil par defaut dans le rapport ;
- mentionner 1.5 m comme variante utile pour les objets, sous reserve d'une calibration correcte ;
- ne pas choisir 2.0 m sans ground truth ou validation visuelle.

## 8. Calibration / homographie

La fusion multi-camera depend fortement de la calibration. Les symptomes observes justifient une verification, mais pas forcement une recalibration complete de toutes les cameras.

Symptomes importants :
- certaines cameras disparaissent de certains runs, par exemple cam_07 sur PC2 en TCP100/TCP50 ;
- les objets confirmes restent rares ;
- les `global_id_switches` montent fortement sur le serveur 4 cameras ;
- augmenter le seuil de fusion aide parfois, ce qui peut indiquer une erreur de projection ou simplement une visibilite imparfaite.

Action recommandee :
1. faire un test visuel court avec affichage annote sur les paires importantes ;
2. verifier les bounding boxes, les footpoints et la position projetee au sol ;
3. verifier que la meme personne vue par deux cameras tombe a une position proche ;
4. recalibrer uniquement les cameras/paires suspectes ;
5. refaire un test final avec les vraies zones.

Il faut surtout verifier les paires utilisees pour la zone finale et les objets, typiquement cam_02, cam_05, cam_07, cam_03 selon la zone reelle choisie.

## 9. Phase 4 et metadonnees

Etat actuel :
- l'export JSONL est implemente dans `run_live_campaign.py` via `--metadata-jsonl` ;
- le validateur `validate_metadata_jsonl.py` existe ;
- les benchmarks de transport existent ou doivent etre recuperes sur GitHub ;
- un smoke test queue local existe avec une latence tres faible, autour de 0.15 ms ;
- aucun resultat complet WebSocket / MQTT / HTTP n'a encore ete trouve dans les dossiers analyses ;
- aucun fichier `.jsonl` live n'a ete retrouve dans les archives serveur analysees.

Conclusion Phase 4 :
- les tests Phase 4 transport restent a faire ;
- Queue / HTTP / WebSocket / MQTT peuvent etre testes sans cameras ;
- le test JSONL Phase 3 necessite soit les cameras, soit un replay RTSP ;
- WebRTC/WHEP concerne surtout la video separee, pas l'inference IA.

Architecture finale recommandee :

```text
Cameras RTSP
  -> MediaMTX / FFmpeg / GStreamer pour affichage et recording video
  -> pipeline IA Phase 3 en basse latence
  -> export metadonnees JSONL puis WebSocket ou MQTT
  -> dashboard qui superpose bbox, global_id et alertes sur le flux video
```

Architecture reseau testee pour les runs serveur :

```text
Cameras IP <CAMERA_NET>.x
  |
  | RTSP cameras
  v
PC passerelle Windows
  - Ethernet cameras : <GATEWAY_IP>
  - Wi-Fi / reseau ecole : <SERVER_IP>
  |
  | FFmpeg republie les flux
  v
MediaMTX sur le PC passerelle
  - rtsp://<SERVER_IP>:8554/cam_02
  - rtsp://<SERVER_IP>:8554/cam_03
  - ...
  |
  | RTSP relaye
  v
Serveur Linux GPU
  - run_live_campaign.py
  - backend video selon disponibilite : OpenCV/FFmpeg ou GStreamer
  - CUDA / TensorRT
  |
  v
Detection / tracking / fusion / alertes
```

Cette architecture est plus robuste que le live actuel, car elle evite de melanger acquisition, inference, fusion, affichage, enregistrement et reporting dans un seul processus.

## 10. Points a dire aux tuteurs

1. La Phase 3 est validee conceptuellement : projection sol, homographie, tracking, fusion, global IDs et alertes globales fonctionnent.
2. Sur les personnes/zones, la Phase 3 apporte un gain mesurable par rapport a la Phase 2.
3. Sur les objets, la Phase 3 est sensible mais encore bruitee ; la confirmation multi-camera est difficile car les objets sont petits et peu visibles simultanement.
4. Le serveur est le meilleur candidat pour l'IA live 4 cameras.
5. Le nouveau PC est un bon fallback local pour 2 a 4 cameras sans affichage.
6. Le PC principal ne doit pas etre la machine finale pour du 4/8 cameras live.
7. L'affichage et l'enregistrement doivent etre separes de l'IA.
8. Le modele recommande pour la suite est `yolov8s fp32_engine`.
9. `rtdetr-l` est trop lourd pour du live multi-camera.
10. Il faut encore valider la Phase 4 : JSONL, WebSocket, MQTT, et video WebRTC/WHEP.

## 11. Priorites restantes

Priorite 1 : faire un run final serveur 4 cameras avec `yolov8s fp32_engine`, sans affichage, sans recording, avec JSONL.  
Priorite 2 : verifier visuellement calibration/homographie sur les cameras de la vraie zone.  
Priorite 3 : refaire un test final avec les vraies zones.  
Priorite 4 : lancer les tests Phase 4 transport : Queue, HTTP, WebSocket, MQTT.  
Priorite 5 : tester WebRTC/WHEP pour la video separee via MediaMTX.  
Priorite 6 : si le temps le permet, refaire un test 8 cameras serveur pour voir si l'objectif initial est atteignable.

## 12. Decision actuelle

La solution la plus defendable a ce stade est :

- IA : serveur Linux si disponible ;
- fallback : nouveau PC pour 2 a 4 cameras ;
- modele : `yolov8s fp32_engine` ;
- fusion : seuil 1.0 m par defaut, 1.5 m en test objets ;
- objets : garder `weak + confirmed`, ne pas utiliser `confirmed only` seul ;
- video : afficher/record hors du pipeline IA ;
- metadonnees : JSONL d'abord, puis WebSocket ou MQTT ;
- rapport : presenter la Phase 3 comme un gain clair sur les personnes/zones, et comme une base prometteuse mais encore a consolider pour les objets.

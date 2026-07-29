# Synthese reunion - resultats et avancement de la semaine

Date : 2026-06-12  
Sujet : avancement Phase 3 live multi-camera, comparaison machines/modeles, preparation Phase 4.

## 1. Objectif de la semaine

L'objectif principal etait de passer d'un prototype Phase 3 fonctionnel a un banc de test exploitable scientifiquement :

- comparer le PC principal, le nouveau PC et le serveur ;
- verifier si le live multi-camera est realiste ;
- tester plusieurs modeles IA en conditions live ;
- comprendre les limites de la fusion multi-camera ;
- preparer l'architecture Phase 4 video / IA / metadonnees ;
- produire des resultats defendables pour le rapport et la reunion.

## 2. Ce qui a ete mis au clair techniquement

### Backend video : OpenCV, FFmpeg et GStreamer

On a clarifie un point important : installer GStreamer sur une machine ne suffit pas. Il faut aussi que la version d'OpenCV utilisee par Python ait ete compilee avec `GStreamer: YES`.

Sur le nouveau PC, GStreamer etait installe au niveau systeme, mais OpenCV Python indiquait `GStreamer: NO`. Donc meme quand on demandait `--capture-backend gstreamer`, les runs passaient en pratique par FFmpeg/OpenCV.

Decision prise :

- pour comparer proprement PC principal / nouveau PC / serveur, on utilise `--capture-backend opencv` ;
- donc la comparaison principale se fait via FFmpeg/OpenCV ;
- GStreamer reste une piste d'optimisation separee, mais pas la base de comparaison entre machines.

### Serveur GPU

Sur le serveur, on a verifie que l'environnement GPU etait fonctionnel :

- 2 GPU NVIDIA L4 ;
- PyTorch avec CUDA disponible ;
- TensorRT disponible ;
- utilisation de `CUDA_VISIBLE_DEVICES=1` puis `--device cuda:0` pour cibler proprement un GPU.

Conclusion : le serveur est exploitable pour les tests IA live et devient le candidat principal pour les scenarios 4 cameras.

### Organisation des modeles

On a corrige le probleme de chemins de modeles. Certains scripts cherchaient les poids ici :

```text
Phase_2_Baseline_MonoCam/Modelstrained/V4/<model>/weights/best.pt
```

alors que les modeles etaient parfois ici :

```text
Phase_2_Baseline_MonoCam/Modelstrained/V4/person_objects/<model>/weights/best.pt
```

Des liens/symlinks ont ete crees pour rendre les modeles accessibles de maniere coherente. Les modeles concernes sont notamment :

- `yolov8n` ;
- `yolov8s` ;
- `yolo11s` ;
- `rtdetr-l`.

## 3. Resultats Phase 2 vs Phase 3

Le resultat scientifique principal reste la comparaison Phase 2 / Phase 3 sur les videos V4.

### Phase 2

La Phase 2 fonctionne en mono-camera :

- chaque camera est evaluee separement ;
- les alertes sont en image-space ;
- il n'y a pas de fusion inter-camera ;
- les doublons entre cameras ne sont pas resolus ;
- il n'y a pas de `global_id`.

### Phase 3

La Phase 3 ajoute :

- projection au sol par homographie ;
- tracking par camera ;
- fusion inter-camera par distance et fenetre temporelle ;
- `global_id` ;
- alertes globales ;
- gestion `weak` / `confirmed` pour les objets.

### Conclusion Phase 2 vs Phase 3

Pour les violations personne/zone, la Phase 3 apporte une amelioration claire :

- meilleure precision ;
- meilleur F1 ;
- alertes globales plus coherentes ;
- capacite de relier plusieurs cameras.

Pour les objets, la Phase 3 detecte vite mais reste bruitee :

- les alertes faibles sont nombreuses ;
- les confirmations multi-camera sont rares ;
- les objets sont petits et souvent visibles par une seule camera ;
- la precision chute si on garde toutes les alertes faibles.

Conclusion defendable :

> La Phase 3 est validee pour la fusion personne/zone. Pour les objets, elle constitue une base prometteuse mais depend fortement de la visibilite multi-camera, de la calibration et du seuil de fusion.

## 4. Comparaison des machines

### PC principal

Le PC principal permet de developper et debugger, mais il est limite pour du live multi-camera lourd.

Constats :

- 1 camera : OK ;
- 2 cameras : acceptable ;
- 4 cameras : possible mais limite ;
- 8 cameras : non valide ;
- affichage et enregistrement dans le meme process degradent fortement la fluidite.

Role recommande :

> PC principal = developpement, debug, analyse, tests courts, recorded campaign.

### Nouveau PC

Le nouveau PC est meilleur que le PC principal pour le live local.

Constats :

- 2 cameras sans affichage : bon ;
- 4 cameras sans affichage : utilisable ;
- 8 cameras : pas encore valide ;
- avec affichage : chute forte de performance ;
- certains runs ont montre des cameras absentes, par exemple cam_07 selon le protocole/backend.

Role recommande :

> Nouveau PC = fallback local pour 2 a 4 cameras sans affichage.

### Serveur

Le serveur est le meilleur candidat.

Constats :

- 4 cameras autour de 25 FPS agreges ;
- latence IA moyenne autour de 9 a 13 ms selon le modele ;
- meilleurs resultats de stabilite et de debit ;
- bon candidat pour tester le systeme final.

Role recommande :

> Serveur = machine principale pour l'IA live multi-camera.

## 5. Comparaison des modeles live

Les tests modeles serveur ont ete faits principalement sur 2 cameras, avec objets, en mode `weak + confirmed`.

### Resultats principaux

| Modele | Format | Frames | Detections | Alertes | Latence p95 | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `yolov8n` | engine | 7505 | 3110 | 96 | 16.3 ms | tres rapide, fallback leger |
| `yolov8n` | pt | 7474 | 2703 | 106 | 20.7 ms | correct mais moins interessant que engine |
| `yolov8s` | engine | 7474 | 7505 | 116 | 18.5 ms | meilleur compromis |
| `yolov8s` | pt | 7488 | 5068 | 134 | 19.2 ms | bon, mais engine preferable |
| `yolo11s` | engine | 7452 | 304 | 49 | 17.0 ms | sous-detecte dans ces conditions |
| `yolo11s` | pt | 7503 | 366 | 65 | 20.2 ms | sous-detecte aussi |
| `rtdetr-l` | pt | variable | plus lourd | variable | ~32-37 ms | interessant scientifiquement, trop lent live |
| `rtdetr-l` | engine | echec | bbox invalides | echec | n/a | exclu du live |

### Decision modele

Modele recommande :

> `yolov8s fp32_engine`

Raisons :

- detecte beaucoup plus que `yolo11s` dans les conditions testees ;
- reste rapide ;
- tient bien sur serveur ;
- meilleur compromis performance / qualite.

Fallback :

> `yolov8n fp32_engine`

Raison :

- plus leger ;
- utile si le nombre de cameras augmente ou si la machine est limitee.

Modele a garder pour comparaison scientifique :

> `rtdetr-l pt`

Raison :

- evite que l'etude soit uniquement un benchmark YOLO ;
- mais pas recommande en live multi-camera.

Modele exclu du live :

> `rtdetr-l engine`

Raison :

- bbox invalides / NaN ;
- erreurs ByteTrack du type `Singular matrix` ;
- instable en Phase 3 live.

## 6. Alertes objets : weak vs confirmed

On a clarifie les regles :

- sans `--no-weak-object-alerts` : alertes faibles + confirmees ;
- avec `--no-weak-object-alerts` : seulement objets confirmes ;
- `--object-min-camera-votes 2` : objet confirme seulement s'il est associe entre au moins 2 cameras.

Constat :

- `weak + confirmed` detecte beaucoup plus d'objets ;
- `confirmed only` est trop strict dans les conditions actuelles ;
- les objets confirmes restent rares ;
- augmenter `--fusion-distance-m` a 1.5 m peut aider, mais augmente le risque de mauvaises associations.

Decision recommandee :

> Garder `weak + confirmed` pour l'analyse objets, mais distinguer clairement les alertes faibles et les alertes confirmees dans le rapport.

## 7. Fusion multi-camera et homographie

La fusion fonctionne, mais elle est sensible a trois facteurs :

- calibration/homographie ;
- visibilite simultanee entre cameras ;
- seuil de fusion.

Constats :

- augmenter le seuil de fusion augmente le nombre d'associations ;
- 1.0 m reste le seuil par defaut le plus defendable ;
- 1.5 m peut etre teste pour les objets ;
- 2.0 m risque de fusionner des objets/personnes differents.

Decision :

> Garder 1.0 m par defaut, tester 1.5 m uniquement si les projections sont visuellement correctes.

Action a faire :

- verifier visuellement les footpoints ;
- verifier les positions projetees au sol ;
- verifier les paires de cameras importantes ;
- recalibrer seulement les cameras suspectes.

## 8. Architecture Phase 4

Le live actuel melange trop de roles :

- acquisition RTSP ;
- inference ;
- tracking ;
- fusion ;
- affichage ;
- enregistrement ;
- audit ;
- generation de rapports.

Pour tenir 4 a 8 cameras, il faut separer les flux.

Architecture recommandee :

```text
Cameras RTSP
  -> flux video separe via MediaMTX / FFmpeg / WebRTC
  -> pipeline IA Phase 3 sans affichage ni recording
  -> metadonnees : bbox, alertes, global_id, timestamp
  -> export JSONL puis WebSocket/MQTT
  -> dashboard qui superpose les bbox sur la video
```

Architecture reseau testee avec le serveur :

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

Decision :

> `run_live_campaign.py` doit rester le benchmark IA/fusion. L'affichage et le recording doivent etre geres par une chaine separee.

## 9. Phase 4 reseau / transport

On a commence a structurer la Phase 4 autour de deux axes :

1. transport video ;
2. transport metadonnees / alertes.

### Transport video

Pistes :

- RTSP via OpenCV/FFmpeg ;
- GStreamer si OpenCV est compile avec support GStreamer ;
- MediaMTX ;
- WebRTC/WHEP pour affichage navigateur.

### Transport metadonnees

Pistes :

- JSONL comme format de validation simple ;
- queue locale comme baseline ;
- HTTP POST ;
- WebSocket ;
- MQTT.

Etat actuel :

- l'export JSONL a ete ajoute a la Phase 3 ;
- le validateur JSONL existe ;
- les tests WebSocket / MQTT / HTTP restent a completer ;
- pour l'instant, Phase 4 peut etre presentee comme l'etape d'architecture reseau et de separation video/IA.

## 10. Ce qui est maintenant defendable en reunion

Points forts :

- la Phase 3 est implementee et testee ;
- la comparaison Phase 2 / Phase 3 existe ;
- les tests live multi-machines ont ete lances ;
- le serveur ressort clairement comme meilleure plateforme ;
- `yolov8s fp32_engine` ressort comme meilleur compromis live ;
- les limites des objets sont comprises ;
- l'architecture finale est claire.

Graphiques associes :

- voir `docs/GRAPHIQUES_REUNION_20260612.md` pour les figures pretes a reprendre dans la presentation ;
- les figures couvrent comparaison machines, montee en charge cameras, impact affichage, modeles serveur, objets weak/confirmed, seuil de fusion et Phase 2 vs Phase 3.

Limites a annoncer :

- 8 cameras pas encore validees proprement ;
- objets confirmes encore rares ;
- calibration/homographie a verifier sur les vraies zones ;
- GStreamer pas comparable partout car OpenCV n'a pas toujours le support ;
- latence IA mesuree differente de la latence complete camera -> alerte ;
- Phase 4 WebSocket/MQTT/WebRTC reste a finaliser.

## 11. Prochaines etapes

Priorite court terme :

1. Finaliser les runs serveur 4 cameras.
2. Faire un run final long 20-30 min avec `yolov8s fp32_engine`.
3. Verifier les vraies zones et les homographies.
4. Relancer un test final avec les vraies zones.
5. Tester JSONL / WebSocket / MQTT.
6. Tester WebRTC/WHEP pour la video separee.
7. Si le temps le permet, retester 8 cameras sur serveur.

## 12. Message global pour les tuteurs

Cette semaine, le projet est passe d'un pipeline experimental a un banc de benchmark plus structure. Les resultats montrent que la fusion multi-camera est pertinente, surtout pour les personnes et zones interdites. Le principal verrou n'est plus uniquement le modele IA, mais l'architecture live : il faut separer le flux video, l'inference, les metadonnees et l'affichage.

La suite logique est donc de stabiliser le scenario final sur serveur, verifier les vraies zones et la calibration, puis terminer la Phase 4 avec les transports metadonnees et video separee.

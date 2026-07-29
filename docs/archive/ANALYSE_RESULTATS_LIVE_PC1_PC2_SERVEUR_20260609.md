# Analyse des resultats live Phase 3 - PC actuel, nouveau PC et serveur

Date d'analyse : 2026-06-09  
Perimetre : tests Phase 3 live avec cameras RTSP, fusion multi-camera, alertes zone/personne et objets interdits.  
Hors perimetre volontaire : WebSocket, WebRTC et MQTT. Ces tests seront traites ensuite comme couche Phase 4 reseau/metadonnees.

## Sources analysees

Les resultats viennent de trois sources :

| Source | Dossier principal | Remarque |
|---|---|---|
| PC actuel | `Phase_3_Fusion_MultiCam/reports/campaign_zone1_live_*` | Principalement `yolov8s fp32_engine`, GStreamer, tests 1/2/4/8 cameras, TCP/UDP, affichage/enregistrement. |
| Nouveau PC | `results_to_old_pc-20260609T130312Z-3-001/results_to_old_pc/pc2_direct_runs` | `yolov8s pt`, tests directs cameras, sans WebSocket/MQTT/WebRTC. |
| Serveur Linux | `results_to_old_pc-20260609T130312Z-3-001/results_to_old_pc/server_relay_runs` | `yolov8s pt`, cameras via relais/RTSP serveur. |

Attention : les comparaisons ne sont pas parfaitement strictes, car le PC actuel utilise surtout `fp32_engine`, alors que le nouveau PC et le serveur utilisent ici `pt`. Les scenes ne sont pas strictement identiques non plus. Les tendances restent exploitables pour decider l'architecture.

## Point clef sur les metriques

La colonne `latency_mean_ms` / `latency_p95_ms` mesure surtout la latence de traitement IA dans la boucle Phase 3. Ce n'est pas la latence visuelle complete camera -> reseau -> decoding -> IA -> affichage. Pour mesurer la latence visuelle, il faudra une methode dediee : timestamp visuel, mire, LED, horloge affichee, ou comparaison frame source/frame dashboard.

Le `FPS effectif` ci-dessous est calcule comme :

```text
frames Phase 3 / duree theorique du test
```

Il represente donc la cadence de boucle globale du pipeline Phase 3, pas le FPS video brut de chaque camera.

## Synthese rapide

1. Le serveur est nettement le meilleur sur 4 cameras : environ 25 FPS de boucle avec `yolov8s pt`, latence moyenne IA 8.85 ms, p95 11.36 ms.
2. Le nouveau PC tient correctement 4 cameras sans affichage : environ 18.6 FPS, latence moyenne autour de 16.2 ms, p95 autour de 21 ms.
3. Le PC actuel tient 4 cameras, mais plutot entre 12 et 17 FPS selon le protocole et la duree. Le mode TCP50 est le meilleur compromis observe sur ce PC.
4. Les 8 cameras ne sont pas validees proprement pour l'instant. Le PC actuel et le nouveau PC tombent autour de 8-9 FPS, et le serveur a un run 8 cameras incomplet.
5. L'affichage dans le meme process degrade fortement les performances. Sur le nouveau PC, 2 cameras passent de 23.8 FPS sans UI a 8.1 FPS avec affichage.
6. L'enregistrement dans le meme process semble aussi couteux. Sur le PC actuel, un run 2 cameras avec recording descend a environ 14.9 FPS contre environ 21 FPS sans enregistrement.
7. La fusion est bien active et utile en 4 cameras, mais elle reste sensible a la calibration, a la visibilite effective par camera et au seuil de fusion.

## Tableau performance principal

| Machine | Run | Cams | Format | Protocole | UI/Record | FPS effectif | Lat. moy. | Lat. p95 | Alertes | Liens fusion | Verdict |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| PC actuel | `20260603_101235` | 1 | engine | TCP100 | no UI/no rec | 24.34 | 24.86 ms | 28.55 ms | 75 | 0 | OK 1 cam. |
| PC actuel | `20260603_102520` | 2 | engine | TCP100 | no UI/no rec | 21.30 | 21.49 ms | 24.74 ms | 23 | 1678 | OK 2 cams. |
| PC actuel | `20260603_105924` | 4 | engine | TCP100 | no UI/no rec | 16.11 | 14.80 ms | 18.21 ms | 59 | 2648 | Utilisable, mais pas 25 FPS. |
| PC actuel | `20260603_141101` | 4 | engine | TCP100 | no UI/no rec | 11.86 | 20.25 ms | 23.44 ms | 98 | 2932 | 10 min plus difficile, throughput bas. |
| PC actuel | `20260603_151644` | 4 | engine | TCP50 | no UI/no rec | 17.34 | 13.77 ms | 15.95 ms | 16 | 2340 | Meilleur setup PC actuel observe. |
| PC actuel | `20260603_151049` | 4 | engine | UDP50 | no UI/no rec | 14.37 | 15.17 ms | 17.60 ms | 42 | 2843 | Pas meilleur que TCP50. |
| PC actuel | `20260603_112024` | 8 | engine | TCP100 | no UI/no rec | 8.35 | 14.33 ms | 16.75 ms | 19 | 628 | 8 cams non viable sur ce PC dans ce setup. |
| PC actuel | `20260603_145503` | 2 | engine | TCP100 | recording | 14.86 | 23.85 ms | 27.92 ms | 87 | 1319 | Recording a separer. |
| Nouveau PC | `pc2_direct_smoke_1cam_tcp100_2min` | 1 | pt | TCP100 | no UI/no rec | 25.06 | 19.53 ms | 23.87 ms | 49 | 0 | OK 1 cam. |
| Nouveau PC | `pc2_direct_2cam_tcp100_no_ui_5min` | 2 | pt | TCP100 | no UI/no rec | 23.83 | 17.21 ms | 22.28 ms | 32 | 1845 | Tres bon 2 cams. |
| Nouveau PC | `pc2_direct_2cam_tcp100_display_5min` | 2 | pt | TCP100 | UI | 8.08 | 21.96 ms | 29.33 ms | 11 | 0 | Affichage trop couteux. |
| Nouveau PC | `pc2_direct_4cam_tcp100_no_ui_10min` | 4 | pt | TCP100 | no UI/no rec | 18.64 | 16.26 ms | 20.93 ms | 102 | 5238 | Bon 4 cams, mais cam_07 absente des detections. |
| Nouveau PC | `pc2_direct_4cam_tcp50_no_ui_5min` | 4 | pt | TCP50 | no UI/no rec | 18.60 | 16.22 ms | 20.71 ms | 19 | 235 | FPS bon, fusion faible, cam_07 absente. |
| Nouveau PC | `pc2_direct_4cam_udp50_no_ui_5min` | 4 | pt | UDP50 | no UI/no rec | 17.53 | 16.50 ms | 21.27 ms | 52 | 1653 | UDP marche, mais pas plus rapide. |
| Nouveau PC | `pc2_direct_8cam_tcp100_no_ui_5min` | 8 | pt | TCP100 | no UI/no rec | 9.46 | 15.88 ms | 20.59 ms | 30 | 38 | 8 cams non valide : fusion quasi absente. |
| Serveur Linux | `server_relay_1cam_tcp100_2min` | 1 | pt | TCP100 | no UI/no rec | 24.76 | 16.59 ms | 19.34 ms | 43 | 0 | OK 1 cam. |
| Serveur Linux | `server_relay_2cam_tcp100_5min` | 2 | pt | TCP100 | no UI/no rec | 25.04 | 12.15 ms | 17.92 ms | 67 | 1727 | Excellent 2 cams. |
| Serveur Linux | `server_relay_4cam_tcp100_10min` | 4 | pt | TCP100 | no UI/no rec | 25.01 | 8.85 ms | 11.36 ms | 103 | 14280 | Meilleur resultat global. |
| Serveur Linux | `server_relay_8cam_tcp100_5min` | 8 | pt | TCP100 | no UI/no rec | incomplet | - | - | - | - | Run interrompu/incomplet, 8 cams non valide. |

## Analyse par machine

### PC actuel

Le PC actuel est capable de faire tourner la Phase 3 en live, mais il montre rapidement une limite de cadence quand on monte a 4 ou 8 cameras.

Observations :

- 1 camera : environ 24 FPS, donc proche du temps reel.
- 2 cameras : environ 21 FPS sans enregistrement, acceptable.
- 4 cameras : entre 11.9 et 17.3 FPS selon le protocole et la duree.
- 8 cameras : environ 8.35 FPS, insuffisant pour un systeme live fluide.
- TCP50 donne le meilleur resultat observe en 4 cameras : 17.34 FPS, latence moyenne 13.77 ms, p95 15.95 ms.
- UDP50 n'apporte pas de gain clair : 14.37 FPS, p95 17.60 ms.
- L'enregistrement dans le meme process degrade la cadence : 2 cameras avec recording tombent a 14.86 FPS.

Conclusion PC actuel :

Le PC actuel reste utile pour les tests, la calibration, les comparaisons Phase 2/Phase 3 et les tests 2-4 cameras. En revanche, il ne faut pas le presenter comme plateforme capable de tenir 8 cameras IA + affichage + enregistrement dans un seul process.

### Nouveau PC

Le nouveau PC donne de meilleurs resultats que le PC actuel en mode sans affichage.

Observations :

- 1 camera : 25.06 FPS.
- 2 cameras sans UI : 23.83 FPS.
- 4 cameras TCP100 : 18.64 FPS sur 10 min, latence moyenne 16.26 ms.
- 4 cameras TCP50 : 18.60 FPS, presque identique au TCP100.
- 4 cameras UDP50 : 17.53 FPS, donc UDP ne donne pas de gain net.
- 8 cameras : 9.46 FPS, donc limite non acceptable pour un live fluide.
- 2 cameras avec affichage : chute a 8.08 FPS.

Point important :

Dans les runs PC2 `4cam_tcp100` et `4cam_tcp50`, `cam_07` est presente dans le manifeste mais absente des detections. La charge d'acquisition contient bien 4 flux, mais la fusion utile repose surtout sur `cam_02`, `cam_03` et `cam_05`. Dans le run UDP50, `cam_07` reapparait avec 458 detections. Il faut donc verifier si l'absence de `cam_07` vient :

- d'une scene sans objet/personne visible sur cette camera ;
- d'un probleme de flux ponctuel ;
- d'un probleme de seuil/confiance ;
- d'un probleme de calibration/homographie ;
- ou d'une difference de timing entre flux.

Conclusion nouveau PC :

Le nouveau PC est une bonne plateforme pour les tests live 2-4 cameras, surtout sans affichage ni enregistrement. Il ne valide pas encore 8 cameras. L'affichage doit etre separe du pipeline IA.

### Serveur Linux

Le serveur est le meilleur candidat pour la charge IA pure.

Observations :

- 1 camera : environ 24.8 FPS.
- 2 cameras : 25.0 FPS, latence moyenne 12.15 ms.
- 4 cameras : 25.0 FPS, latence moyenne 8.85 ms, p95 11.36 ms.
- 4 cameras : 31 918 detections et 14 280 liens de fusion sur 10 min, donc la fusion est tres active.
- 8 cameras : run incomplet, donc non valide.

Le serveur montre une forte capacite de calcul, mais le test 8 cameras n'est pas encore conclusif. Il faut distinguer deux choses :

- puissance GPU/CPU serveur : tres bonne sur 4 cameras ;
- robustesse reseau/RTSP/relais 8 cameras : pas encore demontree.

Conclusion serveur :

Le serveur est le meilleur choix pour l'IA centralisee 4 cameras. Pour 8 cameras, il faut relancer un test propre et verifier si le blocage vient du reseau, du relais RTSP, du nombre de flux, du decodeur ou du script.

## Impact affichage et enregistrement

Les resultats confirment fortement l'architecture separee.

### Affichage

Sur le nouveau PC :

| Test | FPS | Lat. moy. | Lat. p95 | Liens fusion |
|---|---:|---:|---:|---:|
| 2 cams TCP100 sans UI | 23.83 | 17.21 ms | 22.28 ms | 1845 |
| 2 cams TCP100 avec UI | 8.08 | 21.96 ms | 29.33 ms | 0 |

L'affichage dans le meme process degrade fortement la cadence et peut perturber la fusion. Ce resultat justifie de ne pas utiliser `run_live_campaign.py` comme dashboard final.

### Enregistrement

Sur le PC actuel :

| Test | FPS | Lat. moy. | Lat. p95 |
|---|---:|---:|---:|
| 2 cams TCP100 sans recording | 21.30 | 21.49 ms | 24.74 ms |
| 2 cams TCP100 avec recording | 14.86 | 23.85 ms | 27.92 ms |

L'enregistrement dans le meme process coute aussi cher. Il faut enregistrer les flux avec un outil separe : MediaMTX, FFmpeg, GStreamer, ou un service dedie.

## Analyse fusion multi-camera

### Fusion active

La fusion est active dans les runs 2 et 4 cameras :

- PC actuel 4 cams TCP100 : 2932 liens de fusion.
- PC actuel 4 cams TCP50 : 2340 liens de fusion.
- Nouveau PC 4 cams TCP100 : 5238 liens de fusion.
- Serveur 4 cams TCP100 : 14280 liens de fusion.

Cela montre que la Phase 3 ne se limite pas a faire plusieurs detections mono-camera : elle associe bien des pistes entre cameras et cree des `global_id`.

### Stabilite des IDs

| Source | Run 4 cams | Tracks | Switches global_id | Tracks avec switch | Switch / track |
|---|---|---:|---:|---:|---:|
| PC actuel | TCP100 10 min | 89 | 7 | 7 | 0.079 |
| Nouveau PC | TCP100 10 min | 93 | 20 | 18 | 0.215 |
| Serveur | TCP100 10 min | 305 | 59 | 58 | 0.193 |

Interpretation :

- Le PC actuel a la meilleure stabilite relative des IDs dans ce run, mais avec moins de detections.
- Le nouveau PC et le serveur ont davantage de switches, mais aussi plus de detections et plus de liens de fusion.
- La stabilite reste correcte pour un prototype, mais il faut eviter de pretendre que les IDs globaux sont parfaitement stables.

### Distribution par camera

Le detail des detections montre que toutes les cameras ne contribuent pas toujours de maniere equivalente.

| Run | Distribution detections |
|---|---|
| PC actuel 4 cams TCP50 | cam_07: 1345, cam_02: 929, cam_05: 569, cam_03: 216 |
| Nouveau PC 4 cams TCP100 | cam_02: 7253, cam_03: 3511, cam_05: 1229, cam_07: 0 |
| Nouveau PC 4 cams UDP50 | cam_02: 2512, cam_05: 820, cam_03: 642, cam_07: 458 |
| Serveur 4 cams TCP100 | cam_07: 22732, cam_02: 5212, cam_05: 2459, cam_03: 1515 |

Ce point est important pour l'article et le rapport : un test "4 cameras" peut etre techniquement lance avec 4 flux, mais la fusion utile depend de ce que chaque camera detecte reellement.

## Alertes personnes et objets

### PC actuel 4 cams TCP100

Alertes :

- 98 alertes au total.
- 73 alertes `forbidden_object`.
- 25 alertes `zone_violation_person`.
- 67 alertes faibles, 31 confirmees.

Interpretation :

Ce run conserve les alertes faibles pour les objets. Il montre que le mode weak/confirmed est utile pour observer les detections objet, mais il genere beaucoup d'alertes faibles.

### Nouveau PC 4 cams TCP100

Alertes :

- 102 alertes au total.
- 53 `zone_violation_person`.
- 49 `forbidden_object`.
- toutes confirmees.

Interpretation :

Ce resultat est interessant : avec `--no-weak-object-alerts` et `object_min_camera_votes=2`, il y a quand meme 49 objets interdits confirmes. Cela veut dire que le systeme peut confirmer des objets multi-camera, au moins dans certaines conditions. Le detail indique toutefois que beaucoup de confirmations viennent de `cam_02+cam_03`, donc il faut verifier si ces associations sont physiquement coherentes avec la calibration.

### Serveur 4 cams TCP100

Alertes :

- 103 alertes au total.
- 95 `zone_violation_person`.
- 8 `forbidden_object`.
- toutes confirmees.

Interpretation :

Le serveur detecte beaucoup plus de personnes et produit beaucoup de liens de fusion, mais peu d'objets confirmes. Cela peut venir de la scene testee, de la visibilite des objets par plusieurs cameras, ou d'une association objet plus difficile que personne.

## Ablation du seuil de fusion

Les fichiers `fusion_threshold_ablation.csv` confirment que le seuil change surtout le nombre de liens de fusion et le nombre d'IDs globaux.

### PC actuel 4 cams TCP100

| Seuil | Liens predits | IDs globaux |
|---:|---:|---:|
| 0.5 m | 1152 | 41 |
| 1.0 m | 1382 | 25 |
| 1.5 m | 1395 | 23 |
| 2.0 m | 1395 | 23 |

Ici, 1.0 m est un bon compromis : 0.5 m fragmente trop les tracks, tandis que 1.5 m et 2.0 m n'apportent presque plus de liens.

### Nouveau PC 4 cams TCP100

| Seuil | Liens predits | IDs globaux |
|---:|---:|---:|
| 0.5 m | 85 | 40 |
| 1.0 m | 205 | 36 |
| 1.5 m | 415 | 34 |
| 2.0 m | 561 | 33 |

Ici, augmenter le seuil augmente beaucoup les liens. Cela peut aider a confirmer davantage d'objets, mais augmente aussi le risque de mauvaises associations. Sans ground truth multi-camera, 1.0 m reste le seuil defendable par defaut. Pour les objets, un test cible a 1.5 m peut etre justifie.

### Serveur 4 cams TCP100

| Seuil | Liens predits | IDs globaux |
|---:|---:|---:|
| 0.5 m | 2641 | 114 |
| 1.0 m | 3983 | 80 |
| 1.5 m | 4336 | 64 |
| 2.0 m | 4444 | 54 |

Ici aussi, 1.0 m reduit fortement la fragmentation sans fusionner aussi agressivement que 1.5 m ou 2.0 m. Pour le rapport, 1.0 m reste le meilleur seuil par defaut.

## TCP vs UDP

Sur PC actuel, en 4 cameras :

| Protocole | FPS | Lat. moy. | Lat. p95 | Liens fusion |
|---|---:|---:|---:|---:|
| TCP100 | 16.11 | 14.80 ms | 18.21 ms | 2648 |
| TCP50 | 17.34 | 13.77 ms | 15.95 ms | 2340 |
| UDP50 | 14.37 | 15.17 ms | 17.60 ms | 2843 |

Sur nouveau PC, en 4 cameras :

| Protocole | FPS | Lat. moy. | Lat. p95 | Liens fusion |
|---|---:|---:|---:|---:|
| TCP100 | 18.64 | 16.26 ms | 20.93 ms | 5238 |
| TCP50 | 18.60 | 16.22 ms | 20.71 ms | 235 |
| UDP50 | 17.53 | 16.50 ms | 21.27 ms | 1653 |

Conclusion :

- UDP50 n'apporte pas de gain clair.
- TCP50 est bon sur le PC actuel.
- TCP100 est le choix le plus stable et le plus defendable pour les tests comparatifs.
- UDP peut rester un test de sensibilite reseau, mais pas le choix par defaut pour le pipeline final.

## Runs incomplets et limites

Runs incomplets identifies :

- `pc2_direct_4cam_final_tcp100_20min` : manifeste present, pas de `summary.csv`. Le test final 20 min doit etre relance.
- `server_relay_8cam_tcp100_5min` : logs presents mais pas de resume final. Le test 8 cameras serveur n'est pas valide.
- `server_relay_1cam_tcp100_clean` : logs presents mais pas de resume final.

Limites generales :

- Les scenes ne sont pas strictement identiques entre machines.
- Les formats modele ne sont pas identiques : engine sur PC actuel, pt sur PC2/serveur.
- Les metriques de latence ne mesurent pas la latence visuelle complete.
- L'absence de detections sur une camera peut venir de la scene ou du flux, pas seulement du modele.
- La validation des objets confirmes demande idealement un protocole dedie avec objets visibles simultanement par au moins deux cameras.

## Decision technique recommandee

### Pour les tests a court terme

1. Utiliser le serveur pour les tests 4 cameras les plus serieux.
2. Utiliser le nouveau PC pour confirmer les resultats 2-4 cameras et tester l'ergonomie Windows.
3. Garder le PC actuel pour analyses, calibration, generation de rapports, tests recorded et debug.
4. Ne pas utiliser l'affichage OpenCV dans `run_live_campaign.py` pour les mesures de performance.
5. Ne pas enregistrer la video dans le meme process que l'IA pour les mesures finales.

### Pour l'architecture Phase 4

Les resultats soutiennent fortement l'architecture separee :

```text
Cameras RTSP
  -> flux video affiche/enregistre separement avec MediaMTX / WebRTC / FFmpeg / GStreamer
  -> pipeline IA Phase 3 sans affichage ni recording
  -> export metadata JSONL puis WebSocket/MQTT
  -> dashboard qui superpose bbox/alertes/global_id sur la video
```

Raison :

- l'affichage dans le process IA fait chuter le FPS ;
- l'enregistrement dans le process IA coute aussi cher ;
- le serveur tient bien 4 cameras si on garde l'IA seule ;
- les metadonnees JSONL fonctionnent deja : 5999 detections avec bbox et `global_id`, p95 de lag metadata estime a environ 29 ms.

## Priorites de tests restantes

1. Relancer le test final 4 cameras TCP100 sur le nouveau PC pendant 20 min, sans UI et sans recording.
2. Relancer le test serveur 8 cameras, mais en ajoutant un monitoring systeme : `nvidia-smi`, CPU, RAM, debit reseau, logs RTSP.
3. Faire un test objets dedie avec objets visibles par au moins deux cameras, puis comparer :
   - weak + confirmed ;
   - confirmed only ;
   - seuil fusion 1.0 m vs 1.5 m.
4. Faire un test calibration/homographie cible sur les cameras qui fusionnent mal, notamment `cam_07` sur PC2.
5. Ensuite seulement, lancer les tests Phase 4 WebSocket, MQTT et WebRTC.

## Conclusion

Les resultats montrent que la Phase 3 est techniquement fonctionnelle : detection, tracking, projection sol, fusion inter-camera, alertes globales et metadonnees fonctionnent. Le probleme principal n'est plus l'existence de la fusion, mais sa robustesse selon les cameras, la calibration et la charge live.

Le serveur est le meilleur support pour une architecture IA centralisee 4 cameras. Le nouveau PC est suffisant pour des tests live 2-4 cameras sans UI. Le PC actuel reste utile, mais ne doit pas porter tout le pipeline live complet.

La conclusion architecturale est claire : pour un systeme realiste, il faut separer video, IA, metadata, affichage et enregistrement.

# Analyse des resultats et graphes Phase 3 / Phase 4

Date : 2026-06-23  
Objectif : preparer les graphes et l'interpretation pour la presentation, le rapport de stage et un eventuel article.

## 1. Message general a faire passer

Les resultats actuels montrent trois choses distinctes :

1. La partie IA/fusion est suffisamment rapide quand elle tourne sans affichage et sans enregistrement.
2. La partie metadonnees est tres legere : JSONL, WebSocket ou MQTT ajoutent peu de latence par rapport a l'inference.
3. La difficulte principale n'est pas uniquement la performance GPU : ce sont surtout la synchronisation video/metadonnees, la calibration/homographie et la definition des alertes objets.

Il faut donc eviter de presenter le systeme comme un simple benchmark de FPS. Le vrai resultat de recherche est l'architecture :

```text
Cameras RTSP
  -> video separee via MediaMTX / WebRTC / HLS / FFmpeg
  -> pipeline IA Phase 3 sans affichage ni recording
  -> metadonnees bbox / alertes / global_id / timestamps
  -> JSONL puis WebSocket ou MQTT
  -> dashboard qui superpose les metadonnees sur la video
```

Cette separation est defendable car les mesures montrent que l'IA et les metadonnees peuvent etre rapides, tandis que l'affichage, l'enregistrement et le streaming video doivent etre traites separement.

## 2. Graphe : latence moyenne / p95 par etape

Source principale :

```text
STAGELIST3N-FusionCam-data/reports/Phase_3_Fusion_MultiCam/pc_latency_internal_2cam_trace.csv
```

Resultats mesures sur PC avec 2 cameras :

| Etape | Moyenne | Mediane | P95 | Max |
|---|---:|---:|---:|---:|
| Lecture/capture frame | 3.75 ms | 2.54 ms | 17.63 ms | 42.52 ms |
| Inference + tracking | 15.77 ms | 15.40 ms | 18.17 ms | 1658.46 ms |
| Fusion multi-camera | 0.016 ms | 0.009 ms | 0.039 ms | 0.56 ms |
| Generation alertes | 0.015 ms | 0.006 ms | 0.034 ms | 0.76 ms |
| Ecriture metadata | 0.65 ms | 0.62 ms | 0.86 ms | 4.11 ms |
| Total apres lecture frame | 35.49 ms | 33.70 ms | 48.44 ms | 1822.94 ms |
| Boucle totale | 39.93 ms | 37.59 ms | 52.62 ms | 1847.26 ms |

### Interpretation

Le graphe doit montrer que l'inference/tracking est la partie dominante du pipeline IA. La fusion, la logique d'alerte et l'ecriture des metadonnees sont quasi negligeables a l'echelle du systeme : moins de 1 ms en p95 pour la metadata, et moins de 0.1 ms en p95 pour fusion/alertes.

Le point important est que la latence IA normale reste autour de quelques dizaines de millisecondes. Le maximum tres eleve de l'inference/tracking, environ 1.66 s, correspond a un outlier. Il ne faut pas le cacher, mais il faut le presenter comme un pic ponctuel. Pour le graphe, il est preferable d'afficher :

- un barplot moyenne + p95 ;
- et eventuellement un second graphe ou une annotation pour les maximums/outliers.

Message a mettre sur la diapo :

> La latence IA est surtout portee par l'inference/tracking. La fusion et les metadonnees ne sont pas le goulot d'etranglement.

## 3. Graphe : HTTP vs WebSocket vs MQTT

Sources :

```text
Phase_4_Network_Latency/runs/*/alert_latency.csv
STAGELIST3N-FusionCam/Phase_4_Network_Latency/runs/*/alert_latency.csv
```

Resultats representatifs :

| Transport | Run | Evenements | Moyenne | Mediane | P95 | Max |
|---|---|---:|---:|---:|---:|---:|
| Queue locale | today_queue | 1000 | 0.094 ms | 0.082 ms | 0.146 ms | 0.741 ms |
| WebSocket | today_websocket | 1000 | 0.587 ms | 0.541 ms | 0.901 ms | 2.474 ms |
| MQTT QoS 0 | today_mqtt_qos0 | 1000 | 2.298 ms | 2.133 ms | 2.893 ms | 42.097 ms |
| MQTT QoS 1 | today_mqtt_qos1 | 1000 | 2.387 ms | 2.290 ms | 2.932 ms | 29.688 ms |
| HTTP POST | today_http_post | 1000 | 14.384 ms | 20.444 ms | 24.807 ms | 42.594 ms |

Tests complementaires WebSocket :

| Transport | Frequence | Moyenne | P95 |
|---|---:|---:|---:|
| WebSocket | 10 Hz | 0.489 ms | 0.711 ms |
| WebSocket | 25 Hz | 0.501 ms | 0.715 ms |
| WebSocket | 50 Hz | 0.497 ms | 0.759 ms |

Tests complementaires MQTT :

| Transport | Frequence | Moyenne | P95 |
|---|---:|---:|---:|
| MQTT | 10 Hz | 1.862 ms | 2.249 ms |
| MQTT | 25 Hz | 1.890 ms | 2.273 ms |
| MQTT | 50 Hz | 1.868 ms | 2.125 ms |

### Interpretation

Le graphe doit montrer clairement trois niveaux :

1. La queue locale est la reference minimale, presque instantanee.
2. WebSocket est tres rapide et stable, avec un p95 inferieur a 1 ms dans les tests locaux.
3. MQTT est legerement plus lent que WebSocket, mais reste tres bon, autour de 2 a 3 ms en p95.
4. HTTP POST fonctionne, mais il est plus variable et moins adapte a un flux continu temps reel.

La conclusion recommandee est :

- WebSocket pour le dashboard temps reel et l'overlay bbox ;
- MQTT pour une architecture distribuee, decouplee, avec plusieurs consommateurs ;
- HTTP POST comme solution simple de debug ou d'integration, mais pas comme meilleur choix final pour les metadonnees temps reel.

Message a mettre sur la diapo :

> Les metadonnees sont peu couteuses. WebSocket est le plus adapte au dashboard, MQTT est pertinent pour une architecture distribuee.

## 4. Graphe : FPS / latence selon 1, 2 et 4 cameras

Sources :

```text
server_relay_runs/server_relay_1cam_tcp100/phase3/summary.csv
server_relay_runs/server_relay_2cam_tcp100_5min/phase3/summary.csv
server_relay_runs/server_relay_4cam_tcp100_10min/phase3/summary.csv
```

Resultats serveur relay :

| Cameras | Frames | Detections | Alertes | Liens fusion | IDs globaux | ID switches | Latence moyenne | Latence p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2992 | 0 | 0 | 0 | 0 | 0 | 17.05 ms | 17.90 ms |
| 2 | 7513 | 6025 | 67 | 1727 | 47 | 10 | 12.15 ms | 17.92 ms |
| 4 | 15004 | 31918 | 103 | 14280 | 207 | 59 | 8.85 ms | 11.36 ms |

### Interpretation

Le run 4 cameras serveur est le plus convaincant : il traite 15004 frames sur 10 minutes, avec 31918 detections, 14280 liens de fusion et une latence p95 de 11.36 ms. Cela montre que le serveur est capable de tenir une configuration 4 cameras dans de bonnes conditions.

Attention : il ne faut pas conclure que "plus il y a de cameras, plus la latence baisse". La baisse observee entre 1, 2 et 4 cameras vient probablement des conditions de run, du contenu video, du format de mesure et de la maniere dont les frames sont comptees. Le message correct est plutot :

> Le serveur ne se degrade pas sur le test 4 cameras, et il reste dans une plage de latence compatible temps reel.

Le run 1 camera n'est pas un bon run qualite car il n'a produit aucune detection. Il peut servir pour la latence technique, mais pas pour juger le modele.

Graphes conseilles :

- barplot latence moyenne/p95 par nombre de cameras ;
- barplot detections/alertes/fusion links par nombre de cameras ;
- eventuellement un graphe "liens de fusion" pour montrer la valeur ajoutee multi-camera.

## 5. Graphe : nombre d'alertes par type

Sources :

```text
server_relay_runs/server_relay_4cam_tcp100_10min/phase3/alerts.csv
server_relay_runs/server_relay_2cam_tcp100_5min/phase3/alerts.csv
Phase_3_Fusion_MultiCam/reports/campaign_zone1_20260527_135237/phase3/V4_person_objects_yolov8s_fp32_engine/alerts.csv
```

Alertes observees :

| Run | zone_violation_person confirmed | forbidden_object weak | forbidden_object confirmed |
|---|---:|---:|---:|
| Serveur 4 cameras TCP100 10 min | 95 | 0 | 8 |
| Serveur 2 cameras TCP100 5 min | 17 | 50 | 0 |
| Recorded campaign V4 yolov8s engine | 46 | 188 | 2 |

### Interpretation

Le graphe doit separer les personnes/zones et les objets. Les alertes personnes/zones sont plus robustes : elles sont confirmees et exploitables. Les objets sont plus difficiles : ils sont petits, visibles par peu de cameras, et souvent classes en `weak`.

Le systeme `weak + confirmed` est utile pour ne pas rater trop d'objets, mais il augmente le volume d'alertes. A l'inverse, `confirmed only` est plus propre mais peut manquer beaucoup d'objets si l'objet n'est pas visible par deux cameras en meme temps.

Message a mettre sur la diapo :

> La Phase 3 est plus mature pour les violations de zone/personnes que pour les objets interdits. Les objets demandent une calibration et une strategie d'alerte plus prudentes.

## 6. Graphe : objets weak vs confirmed

Le graphe le plus clair est un barplot empile :

```text
X = run
Y = nombre d'alertes objets
couleurs = weak / confirmed
```

Avec les chiffres actuels :

- serveur 2 cameras : 50 objets weak, 0 confirmed ;
- recorded campaign : 188 objets weak, 2 confirmed ;
- serveur 4 cameras : 0 weak, 8 confirmed.

### Interpretation

Ces resultats montrent que la confirmation objet depend fortement des conditions :

- nombre de cameras ;
- recouvrement reel entre cameras ;
- taille de l'objet ;
- calibration ;
- seuil de fusion ;
- regle `object_min_camera_votes`.

Il faut donc presenter les objets comme un axe de travail encore sensible. Pour un rapport scientifique, c'est un bon resultat : il montre que le multi-camera apporte de la valeur, mais que les objets exigent une validation plus stricte que les personnes.

## 7. Graphe : Phase 2 vs Phase 3

Source :

```text
Phase_3_Fusion_MultiCam/reports/campaign_zone1_20260527_135237/comparison_phase2_phase3.csv
```

Moyennes observees :

| Phase | Tache | Precision | Recall | F1 | Faux positifs moyens | FAR/h | FPS | Latence moyenne |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 2 | TAD objets | 0.403 | 0.538 | 0.448 | 10.00 | 60.15 | 55.60 | 18.05 ms |
| Phase 2 | TRD zone/personne | 0.312 | 0.524 | 0.386 | 6.00 | 36.09 | 55.53 | 18.03 ms |
| Phase 3 | TAD objets | 0.120 | 0.568 | 0.147 | 2221.75 | - | - | - |
| Phase 3 | TRD zone/personne | 0.474 | 0.554 | 0.416 | 7.75 | - | - | - |

### Interpretation

La comparaison Phase 2 vs Phase 3 doit etre nuancee.

Pour TRD, c'est-a-dire les violations de zone/personne, la Phase 3 est plus interessante : la precision passe d'environ 0.31 a 0.47, le recall augmente legerement, et le F1 passe d'environ 0.39 a 0.42. Ce n'est pas une amelioration spectaculaire, mais c'est coherent avec l'objectif multi-camera : projection sol, fusion, tracking global et alertes globales.

Pour TAD, c'est-a-dire les objets interdits, la Phase 3 actuelle est trop sensible. Le recall est legerement meilleur que la Phase 2, mais la precision chute fortement. Le nombre de faux positifs moyens devient tres eleve, notamment parce que le systeme genere beaucoup d'alertes objet faibles ou de detections objet non filtrees. Ce resultat ne doit pas etre cache : il indique que la Phase 3 objet doit etre reglee differemment des violations de zone/personne.

Message a mettre sur la diapo :

> La Phase 3 apporte surtout une valeur claire pour les violations de zone/personne. Pour les objets, elle augmente la sensibilite mais doit etre mieux filtree pour reduire les faux positifs.

## 8. Graphe : faux positifs

Graphes conseilles :

1. Faux positifs par phase et par tache :

```text
X = Phase 2 / Phase 3
Y = n_faux_positifs
groupes = TAD / TRD
```

2. Faux positifs par camera pour TAD :

```text
X = camera
Y = n_faux_positifs
couleur = phase
```

3. Precision vs recall :

```text
X = recall
Y = precision
point = phase/tache
```

Ce dernier graphe est utile parce qu'il montre le compromis :

- Phase 3 TAD : plus sensible, mais beaucoup moins precise ;
- Phase 3 TRD : meilleure precision que Phase 2, avec recall comparable.

## 9. Paragraphe a utiliser sur les faux positifs

Les faux positifs ne peuvent etre mesures proprement que lorsqu'une verite terrain est disponible. Sur les campagnes enregistrees, les annotations GT permettent de compter les alertes qui ne correspondent a aucun evenement attendu : ce sont donc de vrais faux positifs, exploitables scientifiquement. En revanche, sur les tests live sans annotation, il ne faut pas appeler automatiquement toutes les alertes supplementaires des faux positifs. Il vaut mieux parler de "volume d'alertes", "alertes non verifiees" ou "bruit d'alerte". Pour mesurer des faux positifs en live, il faut soit annoter manuellement une sequence, soit definir une periode controlee ou aucun evenement ne doit se produire, puis compter les alertes produites pendant cette periode.

## 10. Figures prioritaires a produire maintenant

### Figure 1 - Latence interne par etape

But : montrer que la fusion et les metadonnees ne sont pas le probleme.

Donnees :

```text
pc_latency_internal_2cam_trace.csv
```

A afficher :

- moyenne ;
- p95 ;
- idealement log scale ou figure separee car capture/inference sont beaucoup plus grandes que fusion/alertes.

### Figure 2 - Transport metadonnees

But : justifier WebSocket/MQTT.

Donnees :

```text
Phase_4_Network_Latency/runs/*/alert_latency.csv
```

A afficher :

- Queue ;
- HTTP POST ;
- WebSocket ;
- MQTT QoS0 ;
- MQTT QoS1.

Message :

> WebSocket est le meilleur pour le dashboard. MQTT est tres bon pour une architecture distribuee. HTTP est plus instable.

### Figure 3 - Montee en cameras serveur

But : montrer que le serveur tient 4 cameras.

Donnees :

```text
server_relay_runs/server_relay_1cam_tcp100/phase3/summary.csv
server_relay_runs/server_relay_2cam_tcp100_5min/phase3/summary.csv
server_relay_runs/server_relay_4cam_tcp100_10min/phase3/summary.csv
```

A afficher :

- latence p95 ;
- detections ;
- fusion links ;
- alertes.

### Figure 4 - Alertes par type

But : separer personnes/zones et objets.

Donnees :

```text
alerts.csv
```

A afficher :

- `zone_violation_person confirmed` ;
- `forbidden_object weak` ;
- `forbidden_object confirmed`.

### Figure 5 - Phase 2 vs Phase 3

But : comparaison scientifique centrale.

Donnees :

```text
comparison_phase2_phase3.csv
```

A afficher :

- precision ;
- recall ;
- F1 ;
- faux positifs.

Il faut separer TAD et TRD. Ne pas mettre tout dans un seul graphe sans distinction, sinon on donne l'impression que la Phase 3 est mauvaise partout, alors que le probleme concerne surtout les objets.

### Figure 6 - Faux positifs avec GT

But : repondre directement a la demande de la tutrice.

Donnees :

```text
comparison_phase2_phase3.csv
phase3_tad.csv
phase3_trd.csv
```

A afficher :

- faux positifs Phase 2 TAD ;
- faux positifs Phase 2 TRD ;
- faux positifs Phase 3 TAD ;
- faux positifs Phase 3 TRD.

Message :

> Les faux positifs montrent que le multi-camera doit etre evalue separement pour les personnes et les objets. La partie zone/personne est plus stable que la partie objets.

## 11. Conclusion exploitable

Les resultats actuels valident l'interet de l'architecture separee video / IA / metadonnees. La latence interne du pipeline IA est compatible avec le temps reel, surtout lorsque l'affichage et l'enregistrement sont retires du processus d'inference. Les transports de metadonnees sont suffisamment rapides pour un dashboard, avec un avantage clair pour WebSocket en visualisation directe et MQTT en architecture distribuee.

La Phase 3 est surtout convaincante pour la detection de violations de zone/personne : elle apporte la fusion multi-camera, les IDs globaux et une meilleure precision que la Phase 2 sur TRD. En revanche, les objets interdits restent plus fragiles. Les alertes objets faibles augmentent le recall mais degradent fortement la precision et les faux positifs. Cette limite doit etre presentee comme un resultat important : le systeme multi-camera fonctionne, mais la strategie objet doit etre ajustee par calibration, filtrage temporel, seuil de fusion et regles weak/confirmed.

La suite logique est donc :

1. produire les graphes ci-dessus ;
2. faire un run final long avec les vraies zones ;
3. mesurer la latence end-to-end avec le test chrono ;
4. presenter les faux positifs uniquement sur les sequences avec GT ou avec protocole manuel controle.

# Roadmap de tests - Mardi a Vendredi

## Objectif de la semaine

L'objectif principal est de produire une comparaison experimentale claire entre :

- Phase 2 : detection mono-camera, raisonnement dans l'image, alertes separees par camera.
- Phase 3 : detection multi-camera, projection au sol, tracking, fusion inter-cameras, alertes globales.

La priorite scientifique est la suivante :

1. Prouver si la Phase 3 apporte un gain reel par rapport a la Phase 2.
2. Mesurer la capacite live du systeme avec 1, 2, 4 puis eventuellement 8 cameras.
3. Valider le comportement des alertes objets faibles et confirmees.
4. Preparer la Phase 4 : separation flux video / pipeline IA / metadonnees.
5. Produire des tableaux et figures utilisables dans le rapport ou l'article.

## Etat deja acquis

### Runs recorded deja exploitables

Run V4 corrige avec `--phase2-imgsz 960` :

```text
Phase_3_Fusion_MultiCam/reports/campaign_zone1_20260527_135237
```

Ce run est le plus important pour comparer proprement Phase 2 et Phase 3 avec `yolov8s fp32_engine`.

Run V4 de comparaison multi-modeles :

```text
Phase_3_Fusion_MultiCam/reports/campaign_zone1_20260526_012308
```

Ce run sert a comparer les modeles et formats :

- `yolov8n`
- `yolov8s`
- `yolo11s`
- `rtdetr-l`
- `pt`
- `fp32_engine`

### Conclusion provisoire actuelle

Pour les violations personne / zone, la Phase 3 est interessante :

- Phase 2 TRD moyenne : precision environ `0.31`, recall environ `0.52`, F1 environ `0.39`.
- Phase 3 TRD moyenne : precision environ `0.47`, recall environ `0.55`, F1 environ `0.42`.
- Latence Phase 3 `yolov8s fp32_engine` : environ `19 ms` moyen et `23.6 ms` en p95 sur le run corrige.
- Fusion : `11149` liens de fusion, `128` IDs globaux, `31` switches.

Pour les objets interdits, les resultats actuels sont trop bruites :

- Phase 2 TAD moyenne : F1 environ `0.45`, recall environ `0.54`.
- Phase 3 TAD moyenne : F1 environ `0.15`, recall environ `0.57`.
- Les faux positifs Phase 3 objets explosent surtout sur `cam_02` et `cam_07`.
- Cause probable : alertes objets faibles mono-camera trop nombreuses et objets rarement visibles par plusieurs cameras.

Conclusion provisoire a defendre :

```text
La Phase 3 est pertinente pour les violations de zone/personne, car elle raisonne dans le plan sol et fournit des alertes globales. En revanche, la detection d'objets interdits necessite une logique de confirmation plus stricte, car les alertes faibles mono-camera produisent trop de faux positifs.
```

## Fichiers a analyser systematiquement

Pour chaque campagne recorded ou live, regarder :

```text
Phase_3_Fusion_MultiCam/reports/<campaign>/comparison_phase2_phase3.csv
Phase_3_Fusion_MultiCam/reports/<campaign>/phase3/summary.csv
Phase_3_Fusion_MultiCam/reports/<campaign>/phase3/<run>/alerts.csv
Phase_3_Fusion_MultiCam/reports/<campaign>/phase3/<run>/fusion_links.csv
Phase_3_Fusion_MultiCam/reports/<campaign>/phase3/<run>/track_stability.csv
Phase_3_Fusion_MultiCam/reports/<campaign>/phase3/<run>/ablation/fusion_threshold_ablation.csv
Phase_3_Fusion_MultiCam/reports/<campaign>/phase3/<run>/phase3_tad.csv
Phase_3_Fusion_MultiCam/reports/<campaign>/phase3/<run>/phase3_trd.csv
```

Les metriques a extraire :

- Precision.
- Recall.
- F1.
- Faux positifs.
- FAR, faux positifs par heure.
- TAD median, moyen et p95.
- TRD median, moyen et p95.
- FPS.
- Latence moyenne, mediane, p95 et max.
- Nombre total d'alertes.
- Nombre d'alertes `weak` et `confirmed`.
- Nombre de liens de fusion.
- Nombre d'IDs globaux.
- Nombre de switches d'IDs globaux.
- Seuil de fusion recommande.

## Mardi - Comparaison Phase 2 vs Phase 3 sur videos

### Objectif du mardi

Produire le socle scientifique principal :

- Phase 2 vs Phase 3 sur V4.
- `yolov8s` comme modele principal.
- Comparaison `pt` vs `fp32_engine` si possible.
- Validation du mode objets confirmes uniquement.

### Machine conseillee

Utiliser le nouveau PC si les videos, ground truths et modeles sont disponibles.

Sinon utiliser le PC principal, car les resultats existants y sont deja presents.

### Verification avant run sur le nouveau PC

Depuis la racine du repo, avec le venv active :

```powershell
Test-Path .\recordings\recordings\Camera_2_2.3_20260506_131002.mp4
Test-Path .\recordings\recordings\Camera_3_2.4_20260506_131002.mp4
Test-Path .\recordings\recordings\Camera_5_2.6_20260506_131002.mp4
Test-Path .\recordings\recordings\Camera_7_2.11_20260506_131002.mp4
Test-Path .\dataset_objets_HD\gt_objects_tad.json
Test-Path .\gt_people.json
Test-Path .\Phase_2_Baseline_MonoCam\Modelstrained\V4\person_objects\yolov8s\weights\best.pt
```

Si `gt_people.json` est absent :

```powershell
Copy-Item .\ground_truth\gt_people.json .\gt_people.json
```

Si les donnees sont dans un dossier Drive externe :

```powershell
.\scripts\link_external_data_windows.ps1 -DataDir "G:\Mon Drive\STAGELIST3N-FusionCam-data"
```

### Run recorded de base en `pt`

Commande sure, a lancer si l'engine n'est pas encore pret :

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats pt --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960
```

### Run recorded en `fp32_engine`

Commande a lancer si `best.engine` est present :

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats fp32_engine --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960
```

### Run prioritaire complementaire : objets confirmes uniquement

Ce test est important car les anciens resultats Phase 3 TAD generent trop de faux positifs objets.

En `fp32_engine` :

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats fp32_engine --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960 --object-min-camera-votes 2 --no-weak-object-alerts
```

En `pt` si l'engine n'est pas disponible :

```powershell
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats pt --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960 --object-min-camera-votes 2 --no-weak-object-alerts
```

### Analyse a faire mardi

Remplir un tableau avec :

| Systeme | Tache | Precision | Recall | F1 | FP | FAR | TAD/TRD median | Latence moy. | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 2 | TAD objets |  |  |  |  |  |  |  |  |
| Phase 3 | TAD objets weak+confirmed |  |  |  |  |  |  |  |  |
| Phase 3 | TAD objets confirmed-only |  |  |  |  |  |  |  |  |
| Phase 2 | TRD personne/zone |  |  |  |  |  |  |  |  |
| Phase 3 | TRD personne/zone |  |  |  |  |  |  |  |  |

### Questions a trancher mardi

- Est-ce que la Phase 3 ameliore vraiment les violations personne / zone ?
- Est-ce que les objets confirmes uniquement reduisent fortement les faux positifs ?
- Est-ce que `yolov8s fp32_engine` reste le meilleur compromis pour le live ?
- Est-ce que `yolo11s pt` merite encore d'etre teste en live ?
- Est-ce que le seuil de fusion `1.0 m` est toujours defensable ?

### Livrable de mardi

Produire :

- Un tableau Phase 2 vs Phase 3.
- Un paragraphe d'interpretation pour les personnes.
- Un paragraphe d'interpretation pour les objets.
- Une decision sur le seuil de fusion.
- Une decision sur le modele principal pour le live.

## Mercredi - Capacite live avec les cameras

### Objectif du mercredi

Mesurer combien de cameras peuvent tourner en live sur le PC principal :

- 1 camera.
- 2 cameras.
- 4 cameras.
- tentative 8 cameras si 4 cameras est stable.

Le but n'est pas encore de faire le meilleur affichage. Le but est de mesurer la capacite IA/fusion pure.

### Principe

Tester dans cet ordre :

1. Sans affichage.
2. Sans enregistrement.
3. Avec GStreamer.
4. Avec objets confirmes uniquement.
5. Puis seulement ensuite avec affichage ou enregistrement.

### Test 1 camera

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video
```

### Test 2 cameras

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video
```

### Test 4 cameras zone 1

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video
```

### Test 8 cameras

Lancer seulement si le test 4 cameras est stable.

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_01,cam_02,cam_03,cam_04,cam_05,cam_06,cam_07,cam_08 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video
```

### Test UDP si TCP est trop lent

Faire seulement apres les tests TCP.

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol udp --gst-latency-ms 50 --no-display --no-record-video
```

### Mesures a noter pour chaque run live

- Nombre de cameras.
- Backend : `gstreamer` ou `opencv`.
- Protocole : `tcp` ou `udp`.
- Affichage active ou non.
- Enregistrement active ou non.
- FPS ressenti ou mesure.
- Latence moyenne.
- Latence p95.
- Latence max.
- Nombre d'alertes.
- Erreurs H264.
- Coupures de cameras.
- Utilisation GPU avec `nvidia-smi`.

Commande utile pendant le run :

```powershell
nvidia-smi -l 1
```

### Test avec affichage

Ne lancer qu'apres les tests sans affichage.

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --display-mode annotated --no-record-video
```

### Test avec enregistrement

Ne lancer qu'apres avoir mesure sans enregistrement.

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display
```

### Livrable de mercredi

Produire un tableau :

| Cameras | Backend | Protocole | Display | Record | Latence moy. | Latence p95 | Alertes | Coupures | Verdict |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |  |  |  |

Decision attendue :

- Nombre de cameras utilisables sur PC principal.
- Backend recommande.
- Protocole recommande.
- Est-ce que l'affichage doit etre separe de l'IA ?
- Est-ce que l'enregistrement doit etre separe de l'IA ?

## Jeudi - Architecture separee video / IA / metadonnees

### Objectif du jeudi

Valider l'architecture la plus propre pour la Phase 4 :

```text
Cameras RTSP
  -> flux video affiche ou enregistre separement
  -> pipeline IA lit les flux en basse latence
  -> IA publie alertes + bbox + global_id
  -> dashboard superpose les bbox sur la video
```

Cette architecture est importante car le live actuel melange trop de roles :

- acquisition RTSP ;
- inference IA ;
- fusion ;
- affichage ;
- enregistrement ;
- audit ;
- generation de rapports.

Pour tenir 4 a 8 cameras, il faut separer ces roles.

### Test metadata JSONL

Tester d'abord la sortie metadonnees sans dashboard complexe :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam/reports/live_metadata_test.jsonl
```

A verifier :

- Le fichier JSONL est cree.
- Les timestamps sont coherents.
- Les `camera_id` sont presents.
- Les bbox sont presentes.
- Les `global_id` sont presents.
- Les alertes sont presentes.
- La latence metadata est faible.

### Test video separee

Tester le principe suivant :

1. Le pipeline IA tourne avec `--no-display --no-record-video`.
2. L'affichage ou l'enregistrement est gere par un autre outil.
3. Les metadonnees sont exportees en JSONL, puis plus tard WebSocket ou MQTT.

Commande IA :

```powershell
python Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_07 --duration-min 10 --device cuda:0 --object-min-camera-votes 2 --no-weak-object-alerts --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 100 --no-display --no-record-video --metadata-jsonl Phase_3_Fusion_MultiCam/reports/live_metadata_cam02_cam07.jsonl
```

### Test serveur

Sur le serveur Linux, preparer l'environnement meme si les cameras ne sont pas encore accessibles :

```bash
python3 -m venv ~/aivenv
source ~/aivenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Tester aussi :

```bash
nvidia-smi
```

Si les videos recorded sont disponibles sur le serveur, lancer un test offline :

```bash
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py --dataset-version V4 --models yolov8s --formats pt --no-display --device cuda:0 --phase2-device gpu --phase2-imgsz 960 --object-min-camera-votes 2 --no-weak-object-alerts
```

### Analyse attendue jeudi

Comparer trois architectures :

| Architecture | Video | IA | Metadonnees | Avantage | Probleme |
| --- | --- | --- | --- | --- | --- |
| Tout dans `run_live_campaign.py` | meme process | meme process | logs/CSV | simple | trop lourd en live |
| IA seule + video separee | MediaMTX/RTSP | Phase 3 | JSONL | plus stable | synchro a verifier |
| Serveur IA distant | camera -> serveur | serveur GPU | WebSocket/MQTT | grosse puissance | latence reseau |

Decision attendue :

- Garder `run_live_campaign.py` comme benchmark IA/fusion.
- Utiliser une chaine separee pour afficher et enregistrer la video.
- Publier les metadonnees d'abord en JSONL.
- Passer ensuite a WebSocket ou MQTT seulement si JSONL est propre.

### Livrable de jeudi

Produire :

- Un schema d'architecture propre.
- Un tableau avantages / limites.
- Un exemple JSONL de metadonnees.
- Une decision sur l'architecture Phase 4.

## Vendredi - Synthese, figures et plan de rapport

### Objectif du vendredi

Transformer les tests en resultats presentables :

- tableaux propres ;
- graphiques ;
- analyse ;
- limites ;
- choix techniques ;
- plan d'article scientifique.

### Tableaux a produire

#### Tableau 1 - Comparaison modeles V4

Colonnes :

- Modele.
- Format.
- Latence moyenne.
- Latence p95.
- Nombre de detections.
- Nombre d'alertes.
- Fusion links.
- Global IDs.
- Global ID switches.
- Verdict live.

Source principale :

```text
Phase_3_Fusion_MultiCam/reports/campaign_zone1_20260526_012308/phase3/summary.csv
```

#### Tableau 2 - Phase 2 vs Phase 3

Colonnes :

- Systeme.
- Tache.
- Precision.
- Recall.
- F1.
- FP.
- FAR.
- TAD/TRD median.
- Latence.
- Commentaire.

Source principale :

```text
Phase_3_Fusion_MultiCam/reports/campaign_zone1_20260527_135237/comparison_phase2_phase3.csv
```

#### Tableau 3 - Fusion

Colonnes :

- Seuil.
- Fusion links.
- Unique global IDs.
- Global ID switches.
- Alertes.
- Verdict.

Source :

```text
Phase_3_Fusion_MultiCam/reports/<campaign>/phase3/<run>/ablation/fusion_threshold_ablation.csv
```

#### Tableau 4 - Live capacite

Colonnes :

- Nombre de cameras.
- Backend.
- Protocole.
- Display.
- Record.
- Latence moyenne.
- Latence p95.
- Stabilite.
- Verdict.

Source :

```text
Phase_3_Fusion_MultiCam/reports/campaign_zone1_live_*/phase3/*/summary.csv
```

### Graphiques a produire

Priorite haute :

- F1 Phase 2 vs Phase 3 pour TRD.
- F1 Phase 2 vs Phase 3 pour TAD.
- Latence moyenne par modele.
- Latence p95 par modele.
- Nombre d'alertes par modele.
- Effet du seuil de fusion sur le nombre d'IDs globaux.

Priorite moyenne :

- Precision / recall par camera.
- Faux positifs par camera.
- Nombre de fusion links par modele.
- Switches d'IDs globaux par modele.

### Paragraphes a rediger vendredi

#### Resultat Phase 2 vs Phase 3

Dire clairement :

- La Phase 2 est une baseline mono-camera.
- La Phase 3 ajoute la geometrie reelle et les IDs globaux.
- Pour les personnes, la Phase 3 est plus coherente et legerement meilleure en F1.
- Pour les objets, la Phase 3 brute genere trop de faux positifs.
- Les objets doivent etre geres avec confirmation multi-camera ou logique weak/confirmed.

#### Resultat fusion

Dire clairement :

- La fusion cree des IDs globaux entre cameras.
- Le seuil `1.0 m` est un compromis raisonnable.
- `0.5 m` est plus conservateur mais fragmente plus.
- `1.5 m` et `2.0 m` fusionnent plus mais augmentent le risque de mauvaises associations.
- L'evaluation sans verite terrain d'association ne prouve pas parfaitement les mauvaises associations.

#### Resultat live

Dire clairement :

- Le live est limite par l'acquisition RTSP, l'affichage, l'enregistrement et la synchronisation, pas seulement par l'inference.
- L'inference TensorRT est rapide, mais le systeme complet peut avoir de la latence.
- Pour tenir 4 a 8 cameras, l'architecture doit separer video et metadonnees.

#### Limites

Inclure :

- Homographie suppose un sol plat.
- Calibration sensible aux points choisis.
- Objets rarement visibles par plusieurs cameras.
- GT live difficile a faire sans enregistrement.
- Latence visuelle difficile a mesurer sans reference temporelle.
- Les tests serveur dependent du reseau entre cameras, PC et serveur.

### Decision finale a prendre vendredi

Remplir cette matrice :

| Sujet | Decision |
| --- | --- |
| Modele live principal |  |
| Format principal |  |
| Nombre de cameras stable sur PC principal |  |
| Nombre de cameras vise sur nouveau PC |  |
| Seuil de fusion |  |
| Alertes objets |  |
| Backend capture |  |
| Protocole RTSP |  |
| Architecture Phase 4 |  |

Recommandation probable a confirmer :

```text
Modele live principal : yolov8s fp32_engine
Alternative : yolo11s pt si TensorRT pose probleme
Seuil de fusion : 1.0 m
Alertes objets : weak pour debug, confirmed-only pour evaluation finale
Capture : GStreamer prioritaire
Video : affichage/enregistrement separes du pipeline IA
Metadonnees : JSONL d'abord, puis WebSocket/MQTT
```

## Checklist finale de la semaine

### Mardi

- [ ] Verifier que le nouveau PC a videos, GT et modeles.
- [ ] Lancer au moins un run recorded V4 `yolov8s`.
- [ ] Lancer le run confirmed-only objets.
- [ ] Extraire Phase 2 vs Phase 3.
- [ ] Rediger la conclusion provisoire Phase 2 vs Phase 3.

### Mercredi

- [ ] Tester live 1 camera.
- [ ] Tester live 2 cameras.
- [ ] Tester live 4 cameras.
- [ ] Tester 8 cameras seulement si 4 cameras est stable.
- [ ] Comparer GStreamer TCP et UDP.
- [ ] Mesurer impact display.
- [ ] Mesurer impact recording.

### Jeudi

- [ ] Tester metadata JSONL.
- [ ] Valider le principe video separee / IA separee.
- [ ] Preparer la venv serveur.
- [ ] Verifier CUDA sur serveur.
- [ ] Faire un test recorded sur serveur si les videos sont disponibles.

### Vendredi

- [ ] Produire tableaux.
- [ ] Produire graphiques.
- [ ] Rediger analyse Phase 2 vs Phase 3.
- [ ] Rediger analyse fusion.
- [ ] Rediger analyse live.
- [ ] Rediger limites.
- [ ] Fixer les decisions techniques finales.


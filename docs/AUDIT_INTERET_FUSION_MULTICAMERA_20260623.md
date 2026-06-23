# Audit scientifique - Interet de l'architecture Phase 3 / Phase 4

Date : 2026-06-23  
Sujet : verifier si les resultats actuels permettent de defendre l'interet d'une architecture multi-camera fusionnee par rapport a une architecture mono-camera avec modeles separes par camera.

## 1. Question de recherche reformulee

Le plan de recherche donne la question centrale suivante :

> Comment le choix du modele, la plateforme materielle, les conditions reseau, et la fusion multi-cameras influencent-ils conjointement la capacite a detecter des violations de securite en temps reel dans un atelier industriel ?

Dans le contexte actuel du stage, la question operationnelle devient :

> Est-ce que l'architecture Phase 3 / Phase 4, avec projection au sol, tracking, fusion multi-camera, global_id et metadonnees separees, apporte une valeur mesurable par rapport a une architecture plus simple ou chaque camera execute son modele separement ?

La reponse actuelle est nuancee :

- oui, pour les violations de zone/personne ;
- oui, pour l'architecture systeme temps reel et la separation video / IA / metadonnees ;
- partiellement, pour la stabilite multi-camera et les global_id ;
- pas encore clairement, pour les objets interdits, car la Phase 3 TAD genere trop de faux positifs dans les resultats actuels.

## 2. Ce que le plan de recherche demande vraiment

Le plan distingue clairement :

1. Phase 2 : baseline mono-camera.
2. Phase 3 : systeme multi-camera avec homographie, tracking, association inter-cameras et alertes en coordonnees reelles.
3. Phase 4 : impact reseau, latence, RTSP/TCP vs RTSP/UDP et conditions de transmission.

Le plan precise aussi une distinction tres importante :

- pour les personnes, la fusion multi-camera doit apporter un gain majeur : localisation au sol, deduplication, reduction des erreurs liees a la perspective, meilleure detection de violation de zone ;
- pour les objets, le gain est plus modeste : localisation sur plan, deduplication si l'objet est visible par plusieurs cameras, mais la detection reste largement dependante du modele et de la visibilite dans une camera.

Cela veut dire que les resultats actuels ne doivent pas etre juges uniquement avec un score global. Il faut separer :

- TRD : personnes / zones ;
- TAD : objets interdits ;
- performances live ;
- stabilite de fusion ;
- latence reseau / metadonnees ;
- exploitabilite du dashboard.

## 3. Resultats actuels qui soutiennent l'interet de la Phase 3

### 3.1 Phase 2 vs Phase 3 sur TRD/personnes

Source principale :

```text
Phase_3_Fusion_MultiCam/reports/campaign_zone1_20260527_135237/comparison_phase2_phase3.csv
```

Moyennes observees :

| Systeme | Tache | Precision | Recall | F1 | Faux positifs moyens |
|---|---|---:|---:|---:|---:|
| Phase 2 | TRD zone/personne | 0.312 | 0.524 | 0.386 | 6.00 |
| Phase 3 | TRD zone/personne | 0.474 | 0.554 | 0.416 | 7.75 |

Interpretation :

- la Phase 3 ameliore la precision ;
- le recall augmente legerement ;
- le F1 augmente ;
- les faux positifs augmentent un peu, mais pas de maniere explosive comme pour TAD ;
- les alertes deviennent globales, avec position au sol et global_id.

Ce resultat est interessant pour la question de recherche. Il montre que l'approche au sol/fusionnee est plus pertinente que l'approche image-space mono-camera pour les violations de zone/personne.

Niveau de preuve actuel : **bon mais pas encore definitif**.

Pourquoi pas definitif :

- l'amelioration F1 reste modeste ;
- il faut verifier que les memes evenements et les memes conditions sont bien compares ;
- il faut renforcer la discussion avec des exemples qualitatifs : cas ou Phase 2 duplique ou rate une alerte, cas ou Phase 3 fusionne correctement.

### 3.2 Fusion active et global_id

Sur le serveur 4 cameras :

| Cameras | Frames | Detections | Alertes | Liens fusion | Frames avec fusion | Liens / frame | Liens / 1000 detections | IDs globaux | ID switches | Latence p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 15004 | 31918 | 103 | 14280 | 4212 | 0.95 | 447.4 | 207 | 59 | 11.36 ms |

Interpretation :

- `fusion_links = 14280` signifie 14280 associations inter-cameras, pas 14280 frames fusionnees ;
- `frames_with_fusion = 4212` signifie que 4212 frames ont au moins une association inter-camera ;
- cela fait environ 0.95 lien de fusion par frame en moyenne sur tout le run ;
- cela fait environ 447 liens de fusion pour 1000 detections ;
- la paire dominante observee sur ce run est `cam_02+cam_07` ;
- cela montre que le systeme ne fait pas simplement tourner 4 modeles independants : il associe effectivement des detections entre cameras ;
- la presence de 207 IDs globaux indique que le systeme reconstruit une couche d'identite globale.

Ce resultat soutient fortement l'existence fonctionnelle de la fusion multi-camera.

Niveau de preuve actuel : **bon pour montrer que la fusion est active**.

Limite :

- un `fusion_link` n'est pas automatiquement un lien correct ;
- il manque une evaluation de la qualite de ces associations ;
- il faudrait mesurer `frames_with_fusion`, `fusion_links_per_1000_detections`, et si possible un taux d'association correcte sur une petite sequence annotee.

### 3.3 Performance serveur 4 cameras

Sur serveur :

| Cameras | FPS approx. | Latence moyenne | Latence p95 | Verdict |
|---:|---:|---:|---:|---|
| 2 | ~25 FPS | 12.15 ms | 17.92 ms | tres bon |
| 4 | ~25 FPS | 8.85 ms | 11.36 ms | meilleur resultat actuel |

Interpretation :

Le serveur tient 4 cameras avec une latence IA compatible temps reel. C'est important car l'architecture multi-camera centralisee n'a de sens que si elle reste realisable en live.

Niveau de preuve actuel : **fort pour 4 cameras serveur**.

Limite :

- 8 cameras n'est pas encore valide proprement ;
- la latence mesuree est la latence IA interne, pas la latence end-to-end camera -> dashboard ;
- les tests serveur passent par une passerelle/relais, donc il faut documenter l'architecture reseau.

### 3.4 Architecture separee video / IA / metadonnees

Les tests Phase 4 montrent :

| Transport | Moyenne | P95 | Interpretation |
|---|---:|---:|---|
| Queue locale | ~0.09 ms | ~0.15 ms | reference minimale |
| WebSocket | ~0.59 ms | ~0.90 ms | tres bon pour dashboard |
| MQTT QoS0/QoS1 | ~2.3 ms | ~2.9 ms | tres bon pour architecture distribuee |
| HTTP POST | ~14.4 ms | ~24.8 ms | fonctionne, mais plus variable |

Interpretation :

La couche metadonnees n'est pas un goulot d'etranglement. Cela soutient l'architecture :

```text
video separee via MediaMTX / WebRTC / HLS
IA sans affichage ni recording
metadata via JSONL puis WebSocket/MQTT
dashboard overlay
```

Niveau de preuve actuel : **fort pour les metadonnees locales**.

Limite :

- il manque encore un test end-to-end avec video reelle synchronisee ;
- WebRTC/HLS video et overlay metadata doivent etre calibres temporellement ;
- les resultats MQTT/WebSocket locaux ne remplacent pas un test reseau complet PC passerelle -> serveur -> dashboard.

## 4. Resultats qui affaiblissent ou nuancent la conclusion

### 4.1 Phase 3 TAD objets tres faible

Moyennes observees :

| Systeme | Tache | Precision | Recall | F1 | Faux positifs moyens |
|---|---|---:|---:|---:|---:|
| Phase 2 | TAD objets | 0.403 | 0.538 | 0.448 | 10.00 |
| Phase 3 | TAD objets | 0.120 | 0.568 | 0.147 | 2221.75 |

Interpretation :

La Phase 3 TAD augmente legerement le recall, mais elle fait exploser les faux positifs. La precision et le F1 deviennent donc tres faibles.

Cela ne veut pas dire que la Phase 3 est inutile. Cela veut dire que la Phase 3, dans sa configuration objet actuelle, est trop permissive.

Causes probables :

- alertes `weak` tres nombreuses ;
- objets petits et instables ;
- tracking objet moins robuste que tracking personne ;
- GT evenementielle alors que le pipeline produit des detections/alertes continues ;
- deduplication objet insuffisante ;
- homographie/calibration plus critique pour les petits objets ;
- objets souvent visibles par une seule camera, donc la fusion multi-camera apporte moins.

Niveau de preuve actuel : **defavorable pour TAD objets en configuration actuelle**.

Message scientifique correct :

> La fusion multi-camera apporte un gain clair pour les personnes/zones, mais les objets interdits necessitent une strategie differente : filtrage temporel, confirmed-only, seuils separes et evaluation evenementielle plus stricte.

### 4.2 Les tests live ne remplacent pas les tests avec ground truth

Les tests live donnent :

- FPS ;
- latence IA ;
- nombre d'alertes ;
- nombre de detections ;
- liens de fusion ;
- global_id_switches.

Mais ils ne donnent pas directement :

- precision ;
- recall ;
- F1 ;
- vrais faux positifs ;
- vrais faux negatifs ;
- association inter-camera correcte ou incorrecte.

Pour parler de faux positifs en live, il faut une annotation ou un protocole controle.

Conclusion :

- les tests live prouvent la faisabilite et la performance ;
- les tests recorded avec GT prouvent la qualite detection/alerte ;
- il faut presenter les deux comme complementaires.

### 4.3 La fusion est active, mais sa qualite n'est pas encore completement mesuree

Les `fusion_links` montrent que le systeme fusionne. Mais pour prouver que la fusion est meilleure qu'une architecture mono-camera, il manque encore une mesure explicite de :

- combien d'alertes dupliquees sont supprimees ;
- combien de personnes gardent le meme global_id entre cameras ;
- combien de liens de fusion sont corrects ;
- combien de liens sont faux ;
- combien d'associations manquent.

Actuellement, on peut defendre :

> La fusion est active et exploitable.

Mais il est plus difficile de defendre :

> La fusion est quantitativement meilleure sur toutes les dimensions.

Pour dire cela, il manque une evaluation plus ciblee.

Depuis la mise a jour des scripts, les metriques suivantes peuvent etre extraites automatiquement :

```text
frames_with_fusion
fusion_frame_rate_pct
fusion_links_per_frame
fusion_links_per_1000_detections
fusion_camera_pairs
top_fusion_pair
top_fusion_pair_links
```

Commande exemple :

```powershell
python Phase_4_Network_Latency\analyze_phase4_runs.py --runs-glob "server_relay_runs/server_relay_4cam_tcp100_10min" --out-csv Phase_4_Network_Latency\runs\fusion_health_server_4cam.csv --cameras cam_02,cam_03,cam_05,cam_07
```

Pour mesurer un taux d'association correcte, un petit fichier de verite terrain manuel reste necessaire. Le script ajoute pour cela est :

```powershell
python Phase_3_Fusion_MultiCam\evaluate_fusion_links.py --fusion-links server_relay_runs\server_relay_4cam_tcp100_10min\phase3\fusion_links.csv --truth-csv mini_gt_fusion.csv --frame-tolerance 2 --out-csv fusion_link_eval.csv
```

Format minimal de `mini_gt_fusion.csv` :

```csv
frame,cam_a,track_a,cam_b,track_b,expected_same
837,cam_05,5,cam_07,1,1
840,cam_02,12,cam_07,1,0
```

`expected_same=1` signifie que les deux tracks doivent etre fusionnes. `expected_same=0` signifie qu'ils ne doivent pas etre fusionnes.

## 5. Est-ce que les resultats actuels sont interessants pour l'objectif global ?

### Reponse courte

Oui, les resultats sont deja interessants, mais ils doivent etre presentes avec prudence.

### Ce qui est solide

1. La Phase 3 est plus interessante que la Phase 2 pour les violations de zone/personne.
2. Le serveur peut executer une Phase 3 4 cameras en temps reel.
3. La fusion multi-camera est active : beaucoup de liens de fusion, global_id, alertes globales.
4. L'affichage et l'enregistrement doivent etre separes du pipeline IA.
5. Les metadonnees sont assez legeres pour une architecture dashboard/WebSocket/MQTT.

### Ce qui est encore fragile

1. Les objets TAD en Phase 3 ont trop de faux positifs.
2. La qualite exacte des associations inter-cameras n'est pas encore mesuree par GT.
3. La latence end-to-end video + IA + dashboard n'est pas encore totalement mesuree.
4. Les 8 cameras ne sont pas encore validees proprement.
5. Les vraies zones finales doivent etre testees avec les nouvelles calibrations.

### Conclusion d'audit

Les resultats actuels suffisent pour soutenir une conclusion de type :

> L'architecture multi-camera fusionnee est pertinente et techniquement faisable pour les violations de zone/personne en environnement industriel. Elle apporte une localisation au sol, une logique d'alerte globale et une capacite de deduplication que l'architecture mono-camera ne fournit pas. En revanche, pour les objets interdits, le gain de la fusion est plus limite et les resultats actuels montrent un besoin de filtrage supplementaire pour reduire les faux positifs.

Ils ne suffisent pas encore pour soutenir une conclusion trop forte comme :

> La Phase 3 est globalement meilleure que la Phase 2 pour toutes les taches.

La bonne conclusion est donc :

> Phase 3 meilleure pour TRD/personnes ; Phase 3 prometteuse mais non stabilisee pour TAD/objets ; Phase 4 valide l'interet de separer video, IA et metadonnees.

## 5.1 Limite de scalabilite : 4 cameras, 8 cameras, 50-60 cameras

Il faut etre explicite : l'architecture actuelle ne doit pas etre vendue comme une solution centralisee universelle pour 50 a 60 cameras avec un seul processus IA qui traite tout.

Dans un entrepot avec 50-60 cameras, une architecture naive :

```text
50-60 flux RTSP -> un seul serveur -> un seul pipeline IA/fusion global
```

serait probablement trop lourde, trop fragile et difficile a maintenir. Elle poserait plusieurs problemes :

- decodage video massif ;
- bande passante reseau elevee ;
- consommation GPU elevee ;
- synchronisation plus complexe ;
- fusion inter-camera potentiellement quadratique si toutes les cameras sont comparees entre elles ;
- risque qu'une panne reseau ou serveur impacte tout le systeme ;
- dashboard plus difficile a synchroniser et a exploiter.

Cela ne veut pas dire que la fusion multi-camera est inutile. Cela veut dire qu'elle doit etre appliquee localement, la ou elle apporte vraiment un gain.

Architecture plus realiste pour un grand site :

```text
Cameras independantes ou groupes de cameras
  -> inference locale par camera ou par petit groupe
  -> fusion uniquement entre cameras qui se recouvrent ou couvrent la meme zone
  -> publication d'evenements/metadonnees
  -> supervision centrale
```

La bonne conclusion est donc :

> La fusion multi-camera est pertinente pour des groupes de cameras avec recouvrement, des zones critiques et des scenarios ou la deduplication/localisation au sol apporte une vraie valeur. Pour un grand nombre de cameras independantes, une architecture mono-camera distribuee ou hybride est probablement plus scalable.

Il faut donc presenter trois architectures :

| Architecture | Quand elle est pertinente | Limites |
|---|---|---|
| Mono-camera separee | Cameras sans recouvrement, grande echelle, detection simple | Alertes dupliquees, pas de global_id, zones en image-space |
| Fusion centralisee petit groupe | 2 a 8 cameras autour d'une zone critique | Charge GPU/reseau, calibration, synchronisation |
| Hybride distribuee | Entrepot 50-60 cameras | Plus complexe, mais plus scalable |

Dans ton rapport, la phrase la plus juste serait :

> Les resultats valident l'interet de la fusion multi-camera pour des sous-ensembles de cameras couvrant une meme zone industrielle. L'approche n'a pas vocation a remplacer une architecture distribuee mono-camera pour toutes les cameras d'un grand site ; elle doit plutot etre combinee a celle-ci dans une architecture hybride.

## 6. Tests manquants hors roadmap principale

La roadmap contient deja les grands tests live, Phase 4 et final long. Hors cette roadmap, il manque surtout des tests de validation scientifique fine.

### 6.1 Test de deduplication d'alertes

Objectif :

Verifier si la Phase 3 reduit vraiment les alertes dupliquees par rapport a plusieurs modeles mono-camera independants.

Protocole :

1. Choisir une sequence recorded courte ou une personne est visible par au moins deux cameras.
2. Compter les alertes Phase 2 camera par camera.
3. Compter les alertes Phase 3 globales.
4. Mesurer :

```text
duplicated_alerts_phase2
global_alerts_phase3
duplicate_reduction_rate
```

Interet :

C'est probablement le test le plus directement lie a la question "fusion vs modeles separes".

### 6.2 Test frames avec fusion

Objectif :

Clarifier les `fusion_links`.

Aujourd'hui, on sait qu'il y a 14280 liens de fusion, mais on ne sait pas combien de frames ont au moins une fusion.

Metriques a ajouter :

```text
frames_with_fusion
fusion_links_per_frame
fusion_links_per_1000_detections
camera_pair_distribution
```

Interet :

Cela rend la fusion plus interpretable pour les tuteurs et pour le rapport.

### 6.3 Mini-GT d'association inter-camera

Objectif :

Verifier si les liens de fusion sont corrects.

Protocole minimal :

1. Choisir 5 a 10 evenements ou une personne est visible dans deux cameras.
2. Annoter manuellement :

```text
frame/time
camera_a_track
camera_b_track
same_person oui/non
```

3. Comparer avec `fusion_links.csv`.

Metriques :

```text
association_precision
association_recall
wrong_fusion_count
missed_fusion_count
```

Interet :

C'est la preuve la plus directe que la fusion multi-camera fonctionne correctement.

### 6.4 Test d'ablation "Phase 3 sans fusion"

Objectif :

Isoler l'effet de la fusion.

Il ne suffit pas de comparer Phase 2 et Phase 3, car Phase 3 change plusieurs choses :

- homographie ;
- tracking ;
- global_id ;
- zone au sol ;
- logique d'alerte ;
- fusion inter-camera.

Un test plus propre serait :

```text
Phase 3 avec homographie + tracking mais sans fusion inter-camera
vs
Phase 3 avec homographie + tracking + fusion inter-camera
```

Metriques :

- alertes dupliquees ;
- global_id_switches ;
- faux positifs ;
- TRD ;
- nombre d'alertes globales.

Interet :

C'est un vrai test d'ablation de la fusion.

### 6.5 Test TAD confirmed-only sur recorded

Objectif :

Verifier si les mauvais resultats TAD viennent surtout des alertes weak.

Runs a comparer :

```text
Phase 3 TAD weak + confirmed
Phase 3 TAD confirmed only
Phase 3 TAD confirmed only fusion-distance 1.5 m
```

Metriques :

- precision ;
- recall ;
- F1 ;
- faux positifs ;
- objets confirmes ;
- objets rates.

Interet :

Ce test permet de transformer le mauvais resultat TAD en analyse scientifique :

> Le systeme objet peut etre regle entre sensibilite et precision.

### 6.6 Test "no-event" pour faux positifs live

Objectif :

Mesurer les faux positifs live sans faire une annotation complete.

Protocole :

1. Filmer 5 a 10 minutes ou personne ne rentre dans les zones et aucun objet interdit n'est presente.
2. Lancer Phase 3.
3. Toutes les alertes produites pendant cette periode sont des faux positifs operationnels.

Metriques :

```text
false_alerts_per_hour
false_zone_alerts
false_object_alerts
weak_vs_confirmed_false_alerts
```

Interet :

Tres utile pour la tutrice, car cela donne un chiffre operationnel simple.

### 6.7 Test end-to-end utilisateur avec chrono

Objectif :

Mesurer la latence totale percue, pas seulement la latence IA.

Protocole :

1. Mettre un telephone avec horloge millisecondes devant la camera.
2. Afficher dashboard ou flux video sur le PC.
3. Faire une capture ou video ou l'on voit l'horloge source et l'heure PC.
4. Mesurer :

```text
latence_video_percue
latence_alerte_percue
latence_overlay_metadata
```

Interet :

C'est indispensable pour defendre l'architecture separee video / metadata.

### 6.8 Test vraies zones avec scenario controle

Objectif :

Verifier que les nouvelles zones correspondent au scenario reel voulu par les tuteurs.

Protocole minimal :

1. Pour chaque vraie zone, faire 3 entrees/personnes.
2. Faire 3 passages proches de la zone mais sans entrer.
3. Compter alertes correctes, alertes manquees et alertes fausses.

Interet :

Cela relie les resultats scientifiques aux conditions reelles de deploiement.

## 7. Ce qu'il ne faut pas refaire inutilement

Il n'est pas prioritaire de refaire :

- toutes les combinaisons modeles/formats si la conclusion live est deja claire ;
- tous les tests 1/2/4 cameras sur tous les PC si les tendances sont deja connues ;
- des tests objets live sans protocole clair ;
- des tests dashboard uniquement visuels sans mesure de synchronisation ;
- des tests 8 cameras sur PC principal si l'objectif final est serveur.

Le temps doit plutot aller vers :

1. tests qui isolent l'effet de la fusion ;
2. tests qui mesurent les faux positifs correctement ;
3. tests qui valident les vraies zones ;
4. tests qui prouvent la synchronisation video/metadonnees.

## 8. Audit final par rapport a la question "architecture fusionnee vs modeles separes"

| Critere | Evidence actuelle | Verdict |
|---|---|---|
| Meilleure detection zone/personne | Phase 3 TRD precision/F1 meilleurs que Phase 2 | Positif |
| Meilleure detection objets | Phase 3 TAD recall legerement meilleur mais precision/F1 tres faibles | Negatif/fragile |
| Deduplication multi-camera | Fusion links et global_id presents | Prometteur mais a quantifier |
| Latence IA temps reel | Serveur 4 cams p95 ~11 ms | Fort |
| Passage a 8 cameras | Pas encore valide | Manquant |
| Architecture video/IA separee | Phase 4 metadata rapide, dashboard avance | Positif |
| Synchronisation video/metadonnees | Encore imparfaite / a mesurer | Manquant |
| Robustesse reseau | Tests TCP/UDP partiels, Wireshark/tcpdump a completer | Partiel |
| Evaluation avec vraies zones | Zones preparees, tests finaux manquants | Manquant |

## 9. Recommandation pour le rapport / presentation

La these a defendre doit etre nuancee :

> Une architecture multi-camera fusionnee est pertinente pour un systeme de securite industrielle, surtout pour les violations de zone impliquant des personnes. Elle permet de travailler en coordonnees reelles, de fusionner les vues, de produire des alertes globales et de separer proprement video, IA et metadonnees. Les resultats live montrent que cette architecture est realisable en temps reel sur serveur pour 4 cameras. En revanche, les objets interdits restent plus difficiles : la fusion apporte une localisation et une deduplication potentielle, mais les resultats actuels montrent un compromis difficile entre sensibilite et faux positifs.

Ce positionnement est plus fort qu'une conclusion trop generale. Il montre que le systeme fonctionne, mais aussi que tu comprends ses limites.

## 10. Priorites hors roadmap

Si tu as du temps en plus de la roadmap :

1. faire le test de deduplication Phase 2 vs Phase 3 ;
2. calculer `frames_with_fusion` et `fusion_links_per_1000_detections` ;
3. faire une mini-GT de 5 a 10 associations inter-cameras ;
4. faire un test no-event pour faux positifs live ;
5. refaire TAD recorded en `confirmed only` ;
6. faire un test end-to-end chrono sur le dashboard ;
7. documenter un cas visuel ou Phase 3 corrige une limite de Phase 2.

Ces tests completeraient directement la question scientifique, sans refaire toute la matrice experimentale.

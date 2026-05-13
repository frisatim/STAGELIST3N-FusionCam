# Resume du plan de recherche

## Objectif

Construire et evaluer un systeme de surveillance multi-cameras pour detecter automatiquement :

- l'entree d'une personne dans une zone interdite ;
- l'apparition d'un objet interdit dans l'atelier.

Le travail vise une comparaison experimentale rigoureuse, pas seulement l'execution d'un modele de detection.

## Phases

### Phase 1 - Infrastructure

- Recuperer plusieurs flux RTSP simultanement.
- Horodater les images.
- Enregistrer des sequences multi-cameras.
- Preparer les annotations et ground truths.

### Phase 2 - Baseline mono-camera

- Evaluer plusieurs modeles en detection image.
- Declencher les violations dans l'espace pixel.
- Mesurer TAD, TRD, precision, recall, F1, faux positifs et FAR.

### Phase 3 - Fusion multi-cameras

- Calibrer les cameras par homographie.
- Suivre les detections avec ByteTrack.
- Projeter les detections en coordonnees metres.
- Fusionner les tracks inter-cameras avec des IDs globaux.
- Declencher les alertes en coordonnees reelles.

### Phase 4 - Reseau et latence

- Mesurer l'impact des conditions reseau.
- Comparer les modes de transport RTSP.
- Relier latence, FPS et qualite de detection.

## Metriques

- `TAD` : temps entre apparition reelle d'un objet interdit et detection.
- `TRD` : temps entre entree reelle d'une personne en zone interdite et alerte.
- Precision, recall, F1.
- Faux positifs et False Alarm Rate.
- Stabilite des IDs ByteTrack et `global_id`.
- Alertes dupliquees.
- Ablation du seuil de fusion `D`.

## Hypothese centrale

La Phase 3 doit etre plus pertinente geometriquement pour les violations de zone, car elle raisonne sur le sol reel en metres. Elle peut aussi reduire les doublons inter-cameras, mais elle depend fortement de la qualite des calibrations, du tracking et de la fusion.

Pour les objets interdits, la fusion brute peut creer beaucoup de repetitions. Le depot contient donc une logique d'alertes a deux niveaux : `weak` pour une camera seule, `confirmed` pour une confirmation multi-camera.


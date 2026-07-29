# Reproduire les resultats du rapport

Ce document donne le chemin unique pour verifier et rejouer les
resultats presentes dans le rapport de stage. Prerequis : avoir installe
l'environnement (`docs/NEW_PC_SETUP.md` ou `docs/DOCKER_DELIVERY.md`) et
copie le dossier de donnees externe (`docs/DATA_LAYOUT.md`).

## 0. Verifier la mise en place des donnees

```bash
python scripts/verify_data_layout.py --model-version V4 --model yolov8s
```

Le script controle la presence des datasets, enregistrements et poids
aux emplacements attendus. Ne pas aller plus loin tant qu'il signale des
manques.

Smoke test de l'environnement :

```bash
python -m pytest -q
```

## 1. Consulter les resultats de la session finale (sans GPU)

Les chiffres du chapitre temps reel du rapport proviennent de la session
live finale (4 cameras, 30 minutes, yolov8s V4). Cette session ne peut
pas etre relancee hors du laboratoire (elle necessite les cameras), mais
ses sorties completes sont livrees dans le dossier de donnees :

```text
STAGELIST3N-FusionCam-data/reports/Phase_3_Fusion_MultiCam/final_real_zones_4cam_30min/
```

Reperes pour la verification (voir le rapport pour le detail) :
environ 28 772 frames traitees, 35 627 detections, 516 alertes,
7 618 liens de fusion, latence d'inference p95 environ 16 ms, boucle
complete p95 environ 67 ms.

Les autres campagnes et runs Phase 4 (montee en charge, transports,
captures Wireshark) sont livres au meme endroit.

## 2. Rejouer une campagne sur videos enregistrees (GPU conseille)

C'est la voie reproductible sans cameras. Elle utilise les 4
enregistrements livres (cam_02, cam_03, cam_05, cam_07) et les ground
truths du depot :

```bash
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py \
  --dataset-version V4 \
  --models yolov8s \
  --formats pt \
  --no-display \
  --phase2-imgsz 960
```

Sorties dans `Phase_3_Fusion_MultiCam/reports/campaign_zone1_<date>/` :

- `summary.csv` : resume Phase 2 vs Phase 3 par modele ;
- `phase3_trd.csv` / `phase3_tad.csv` : metriques evenementielles
  (definitions dans `docs/METRIQUES.md`) ;
- `comparison_phase2_phase3.csv` : tableau de comparaison ;
- `manifest.json` : parametres exacts du run.

Notes :

- utiliser `--formats fp32_engine` seulement si l'engine TensorRT livre
  se charge sur le GPU local, sinon rester en `pt` (memes modeles, plus
  lent) ;
- les resultats dependent du GPU et des versions de bibliotheques :
  l'environnement de reference est fige dans `requirements.lock.txt` et
  decrit dans `docs/ENVIRONNEMENT_REFERENCE.md`.

## 3. Verifier la qualite d'association de la fusion (mini GT)

La qualite des liens inter-cameras (chapitre fusion du rapport, 50 cas
annotes) se verifie avec :

```bash
python Phase_3_Fusion_MultiCam/evaluate_fusion_links.py \
  --fusion-links <campagne>/fusion_links.csv \
  --truth-csv <mini_gt>.csv
```

Le CSV de verite doit contenir la colonne `expected_same` remplie (1 ou
0). Le mini GT utilise dans le rapport est fourni en annexe du rapport.

## 4. Regenerer les figures du rapport

```bash
python scripts/generate_phase3_phase4_figures.py
```

Les figures sont ecrites dans `docs/figures/` (voir
`docs/FIGURES_PHASE3_PHASE4_20260623.md` pour la legende de chacune).
Le script lit les resumes CSV livres ; il ne necessite pas de GPU.

## 5. Rejouer un flux live sans cameras (optionnel)

Pour tester le mode live sans acces au laboratoire, republier les
enregistrements en RTSP local avec MediaMTX :

```bash
docker compose up -d mediamtx
```

puis suivre `docker/replay_rtsp_examples.md` et lancer
`run_live_campaign.py` sur les URLs locales. Ce mode valide le
fonctionnement du pipeline live, mais les latences mesurees ne sont pas
comparables a celles du rapport (pas de reseau cameras reel).

## 6. Re-entrainer les modeles (optionnel, long)

```bash
python Phase_2_Baseline_MonoCam/train_models.py --models yolov8s
```

Entraine sur `dataset_objets_V4` (80 epochs, imgsz 960 par defaut, voir
`docs/VERSIONS_MODELES.md`). Compter plusieurs heures par modele sur un
GPU portable. Le re-entrainement ne redonne pas exactement les memes
poids (non determinisme CUDA), mais des performances comparables.

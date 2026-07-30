# STAGELIST3N FusionCam

Systeme de surveillance multi-cameras pour la securite industrielle.

Ce depot contient le code de recherche et les scripts d'evaluation developpes pour comparer une baseline mono-camera en coordonnees image avec une approche multi-cameras projetee sur un plan sol, puis pour mesurer le comportement temps reel de l'ensemble (latence, transport des metadonnees, dashboard).

## Question de recherche

Comment le choix du modele, la plateforme materielle, les conditions reseau et la fusion multi-cameras influencent-ils conjointement la detection temps reel de violations de securite dans un atelier industriel ?

Le projet traite deux evenements separes :

- `TRD` : detection d'une personne entrant dans une zone interdite.
- `TAD` : detection d'un objet interdit dans la scene.

Les definitions formelles des metriques (TRD, TAD, FAR, precision/rappel evenementiels) sont dans `docs/METRIQUES.md`.

## Par ou commencer

- `docs/GUIDE_LECTURE.md` : question de recherche, ordre de lecture des phases, correspondance avec les chapitres du rapport de stage.
- `docs/REPRODUCTION.md` : comment relancer les campagnes et retrouver les resultats du rapport.

## Organisation

```text
Phase_1_Infrastructure/       Capture RTSP, enregistrement, annotation, conversion dataset.
Phase_2_Baseline_MonoCam/     Baseline mono-camera en pixels/image, entrainement, evaluation TAD/TRD.
Phase_2.5_Test_Live/          Premier dashboard live RTSP (etape intermediaire).
Phase_3_Fusion_MultiCam/      Calibration, tracking, homographie, fusion multi-cameras, alertes, campagnes.
Phase_4_Network_Latency/      Latence reseau, transport des metadonnees, dashboard web, benchmarks.
scripts/                      Setup, livraison, replay, generation de figures.
ground_truth/                 Ground truths JSON legeres pour TAD/TRD.
reports/                      Resultats synthetiques legers conserves dans Git.
docs/                         Methode, metriques, resultats, setup, livraison.
docs/archive/                 Notes de travail internes (non maintenues).
docker/                       Dockerfile, MediaMTX, replay RTSP.
```

Chaque dossier de phase contient un `README.md` qui decrit ses scripts. Les scripts obsolets sont conserves dans des sous-dossiers `legacy/` avec explication.

Les datasets complets, videos, poids `.pt`, engines TensorRT `.engine` et sorties d'entrainement ne sont pas versionnes dans Git. Ils sont livres dans un dossier externe `STAGELIST3N-FusionCam-data` decrit dans `docs/DONNEES.md`.

## Installation

Guide complet (Windows natif ou Docker, avec l'environnement de reference des runs finaux) : `docs/INSTALLATION.md`

Setup Windows automatique :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -InstallOptionalNetwork
```

Copie des donnees lourdes depuis une cle USB, puis creation des jonctions et verification :

```powershell
.\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\STAGELIST3N-FusionCam-data
.\scripts\link_external_data_windows.ps1
python scripts\verify_data_layout.py --model-version V4 --model yolov8s
```

## Pipeline Phase 3

1. Lecture des flux camera ou videos enregistrees.
2. Detection par modele YOLO / RT-DETR (versions de modeles : `docs/DONNEES.md`).
3. Suivi mono-camera avec ByteTrack.
4. Projection du point bas-centre sur le plan sol par homographie.
5. Association inter-cameras avec `MultiCameraFusion` (compatibilite de classes requise).
6. Verification des zones interdites en coordonnees metres, vote multi-cameras.
7. Generation d'alertes (`weak` / `confirmed`), exports CSV et metadonnees JSONL.

## Scripts principaux

Campagne sur videos enregistrees (poids V4 livres) :

```bash
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py \
  --dataset-version V4 \
  --models yolov8s \
  --formats pt \
  --no-display \
  --phase2-imgsz 960
```

Campagne live RTSP a duree bornee :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py \
  --versions V4 \
  --models yolov8s \
  --formats pt \
  --duration-min 5 \
  --device cuda:0
```

Mode objets avec deux niveaux d'alertes :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py \
  --versions V4 \
  --models yolov8s \
  --formats pt \
  --duration-min 5 \
  --device cuda:0 \
  --object-min-camera-votes 2
```

Avec cette configuration, une detection objet vue par une seule camera est une alerte `weak`; une detection confirmee par au moins deux cameras dans la fenetre temporelle est une alerte `confirmed`.

Utiliser `--formats fp32_engine` seulement apres avoir verifie que l'engine TensorRT local se charge sur le GPU cible (les `.engine` sont lies au GPU qui les a generes, voir `docs/INSTALLATION.md`).

Tests automatises (les scripts `visual_check_*.py` et `demo_*.py` sont des harnais visuels, pas des tests pytest) :

```bash
python -m pytest -q
```

## Resultats inclus

- `reports/` : exports legers des campagnes (comparaison Phase 2 vs Phase 3, ablation du seuil de fusion, resumes live). Colonnes documentees dans `reports/README.md`.
- `docs/figures/phase3_phase4_20260707/` : figures finales du rapport, regenerables avec `python scripts/generate_phase3_phase4_figures.py`. L'interpretation detaillee des resultats est dans le rapport de stage.

Les CSV frame-par-frame complets, videos et logs volumineux sont livres dans le dossier de donnees externe (`docs/DONNEES.md`).

## Index des documents

| Document | Contenu |
|---|---|
| `docs/GUIDE_LECTURE.md` | Question de recherche, ordre de lecture, mapping rapport de stage vers depot |
| `docs/METRIQUES.md` | Definitions TRD, TAD, FAR, precision/rappel |
| `docs/INSTALLATION.md` | Installation Windows natif ou Docker, environnement de reference |
| `docs/DONNEES.md` | Donnees externes, versions V1 a V4 des modeles, securite |
| `docs/REPRODUCTION.md` | Reproduire les resultats du rapport, mesurer la latence |
| `docs/archive/` | Notes de travail internes, non maintenues |

## Securite des donnees

Les URLs RTSP, mots de passe camera, IPs internes, videos et datasets complets ont ete exclus ou remplaces par des placeholders. Voir `docs/DONNEES.md`.

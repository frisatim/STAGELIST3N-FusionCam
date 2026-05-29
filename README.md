# STAGELIST3N FusionCam

Systeme de surveillance multi-cameras pour la securite industrielle.

Ce depot contient le code de recherche et les scripts d'evaluation developpes pour comparer une baseline mono-camera en coordonnees image avec une approche multi-cameras projetee sur un plan sol.

## Question de recherche

Comment le choix du modele, la plateforme materielle, les conditions reseau et la fusion multi-cameras influencent-ils conjointement la detection temps reel de violations de securite dans un atelier industriel ?

Le projet traite deux evenements separes :

- `TRD` : detection d'une personne entrant dans une zone interdite.
- `TAD` : detection d'un objet interdit dans la scene.

## Organisation

```text
Phase_1_Infrastructure/       Capture RTSP, enregistrement, annotation, conversion dataset.
Phase_2_Baseline_MonoCam/     Baseline mono-camera en pixels/image, evaluation TAD/TRD.
Phase_2.5_Test_Live/          Dashboard et tests live RTSP intermediaires.
Phase_3_Fusion_MultiCam/      Tracking, homographie, fusion multi-cameras, alertes, campagnes.
Phase_4_Network_Latency/      Espace reserve pour les experiences reseau et latence.
ground_truth/                 Ground truths JSON legeres pour TAD/TRD.
reports/                      Resultats synthetiques legers conserves dans Git.
docs/                         Notes de methode et politique de donnees.
```

Les datasets complets, videos, poids `.pt`, engines TensorRT `.engine` et sorties d'entrainement ne sont pas versionnes dans Git. Ils doivent etre stockes dans des volumes externes ou conteneurs Docker.

## Installation sur un nouveau PC

Guide complet:

```text
docs/NEW_PC_SETUP.md
```

Setup Windows automatique:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -InstallOptionalNetwork
```

Le script installe les outils de base via `winget` quand disponible, cree un
environnement virtuel, installe PyTorch CUDA puis les dependances Python du
projet. Les datasets, videos, poids `.pt` et engines `.engine` restent a copier
manuellement car ils ne sont pas stockes dans Git.

## Pipeline Phase 3

1. Lecture des flux camera ou videos enregistrees.
2. Detection par modele YOLO / RT-DETR.
3. Suivi mono-camera avec ByteTrack.
4. Projection du point bas-centre sur le plan sol par homographie.
5. Association inter-cameras avec `MultiCameraFusion`.
6. Verification des zones interdites en coordonnees metres.
7. Generation d'alertes et exports CSV.

## Scripts principaux

Campagne sur videos enregistrees :

```bash
python Phase_3_Fusion_MultiCam/run_recorded_campaign.py \
  --models yolov8n,yolov8s,yolo11s \
  --formats pt,fp32_engine \
  --no-display
```

Campagne live RTSP a duree bornee :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py \
  --versions V3 \
  --models yolov8s \
  --formats fp32_engine \
  --duration-min 5 \
  --device cuda:0
```

Mode objets avec deux niveaux d'alertes :

```bash
python Phase_3_Fusion_MultiCam/run_live_campaign.py \
  --versions V3 \
  --models yolov8s \
  --formats fp32_engine \
  --duration-min 5 \
  --device cuda:0 \
  --object-min-camera-votes 2
```

Avec cette configuration, une detection objet vue par une seule camera est une alerte `weak`; une detection confirmee par au moins deux cameras dans la fenetre temporelle est une alerte `confirmed`.

## Resultats inclus

Le dossier `reports/` contient seulement des exports legers :

- comparaison Phase 2 vs Phase 3 sur la zone 1 ;
- resumes Phase 3 TAD/TRD ;
- ablation du seuil de fusion `D` ;
- petits resumes de campagnes live RTSP.

Les CSV frame-par-frame complets (`detections.csv`, `fusion_links.csv`, videos, logs volumineux) sont exclus du depot.

## Securite des donnees

Les URLs RTSP, mots de passe camera, IPs internes, videos et datasets complets ont ete exclus ou remplaces par des placeholders. Voir `docs/DATA_AND_SECURITY.md`.

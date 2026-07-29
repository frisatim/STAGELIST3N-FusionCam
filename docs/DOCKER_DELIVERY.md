# Livraison Docker et donnees lourdes

Le depot Git contient le code, la documentation, les ground truths legeres et
les petits resumes de resultats. Les assets lourds restent hors Git et sont
montes comme volumes Docker.

## Layout de livraison recommande

Le layout de donnees canonique est documente dans `docs/DATA_LAYOUT.md`.

```text
delivery/
  STAGELIST3N-FusionCam/          # depot Git
  STAGELIST3N-FusionCam-data/     # donnees lourdes, hors Git
    datasets/
      dataset_objets_V4/
      dataset_objets_HD/
      dataset/
    recordings/
      recordings/
        Camera_*.mp4
    models/
      V2/
      V3/
      V4/
    reports/
      Phase_3_Fusion_MultiCam/
    exports/
```

L'image Docker fournit l'environnement d'execution. Le dossier de donnees
fournit les datasets, les enregistrements, les poids entraines, les engines
TensorRT et les gros rapports.

## Pourquoi les donnees ne sont pas dans l'image

Garder les datasets et les modeles hors de l'image est plus pratique :

- l'image Docker reste plus petite et plus rapide a reconstruire ;
- les datasets et les resultats peuvent etre mis a jour sans reconstruire
  l'image ;
- les fichiers TensorRT `.engine` sont souvent lies au GPU et aux versions
  CUDA et TensorRT ;
- les poids `.pt` restent la source de verite portable et les engines peuvent
  etre regeneres.

## Premiere installation

Copier `.env.example` vers `.env` :

```bash
cp .env.example .env
```

Editer `.env` si le dossier de donnees n'est pas a cote du repo :

```text
FUSIONCAM_DATA_DIR=../STAGELIST3N-FusionCam-data
```

Creer le squelette du dossier de donnees lourdes :

```bash
python3 scripts/prepare_delivery_layout.py \
  --data-dir ../STAGELIST3N-FusionCam-data
```

Puis copier manuellement les datasets, enregistrements, modeles et rapports
dans les dossiers crees, ou utiliser le helper de copie USB sur Windows :

```powershell
.\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\BenchmarkingAI
```

Cette commande accepte soit le dossier projet complet d'origine, soit un
dossier `STAGELIST3N-FusionCam-data` deja prepare sur la cle USB.

Pour les runs natifs Windows hors Docker, creer les jonctions locales apres la
copie :

```powershell
.\scripts\link_external_data_windows.ps1
```

Les jonctions exposent les donnees externes aux chemins historiques utilises
par les scripts Python sans dupliquer de gros fichiers dans Git.

## Build

Important : utiliser la commande `docker compose` (Docker Compose v2), pas
l'ancien binaire `docker-compose` (v1). L'acces GPU declare via
`deploy.resources` dans `docker-compose.yml` n'est honore que par Docker
Compose v2 ; avec l'ancien binaire `docker-compose`, le conteneur demarre
sans GPU.

```bash
docker compose build fusioncam
```

## Ouvrir un shell

```bash
docker compose run --rm fusioncam bash
```

Dans le conteneur :

```bash
python3 -m pytest Phase_3_Fusion_MultiCam/test_campaign_utils_pytest.py -q
```

## Lancer une campagne recorded

Exemple apres avoir place les enregistrements et les modeles dans le dossier
de donnees monte :

```bash
docker compose run --rm fusioncam \
  python3 Phase_3_Fusion_MultiCam/run_recorded_campaign.py \
    --dataset-version V4 \
    --models yolov8s \
    --formats pt \
    --no-display \
    --device cuda:0 \
    --phase2-device gpu \
    --phase2-imgsz 960
```

Avant de lancer des campagnes, verifier les montages :

```bash
docker compose run --rm fusioncam \
  python3 scripts/verify_data_layout.py \
    --data-dir /workspace/data \
    --model-version V4 \
    --model yolov8s
```

Les chemins dans les configs peuvent devoir etre adaptes aux dossiers de
donnees montes :

```text
/workspace/data/datasets
/workspace/data/recordings
/workspace/data/models
/workspace/data/reports
```

Pour la compatibilite avec les scripts de recherche existants, Docker Compose
monte aussi les memes donnees externes dans les chemins historiques :

```text
/workspace/code/dataset
/workspace/code/dataset_objets_HD
/workspace/code/dataset_objets_V4
/workspace/code/recordings
/workspace/code/Phase_2_Baseline_MonoCam/Modelstrained
/workspace/code/Phase_3_Fusion_MultiCam/reports
```

## Replay RTSP depuis les enregistrements

Demarrer MediaMTX :

```bash
docker compose up -d mediamtx
```

Par defaut, les ports MediaMTX sont lies a `127.0.0.1` dans
`docker-compose.yml`. Garder ce defaut pour les tests locaux. Si le replay
video doit etre accessible depuis une autre machine, exposer les ports
uniquement sur un reseau de confiance et ajouter les regles de pare-feu ou
d'authentification MediaMTX appropriees.

Puis publier les enregistrements avec FFmpeg comme documente dans
`docker/replay_rtsp_examples.md`.

Cela permet des tests repetables de type live a partir de fichiers video
annotes.

## Ce qu'il faut donner aux tuteurs

Elements recommandes :

- le depot Git ;
- `STAGELIST3N-FusionCam-data` en `.zip`, `.tar` ou dossier sur disque
  externe ;
- ce setup Docker ;
- la reference des versions GPU, CUDA, Python et Ultralytics utilisees pour
  l'entrainement : voir `docs/ENVIRONNEMENT_REFERENCE.md` et
  `requirements.lock.txt`.

Garder a la fois les `.pt` et les `.engine` quand ils existent. Si le
`.engine` echoue sur une autre machine, le regenerer depuis le `.pt`.

Ne pas placer de datasets, d'enregistrements, de `.pt`, de `.engine` ni de
vrais identifiants RTSP dans Git. Les garder dans `STAGELIST3N-FusionCam-data`
ou dans un autre stockage prive.

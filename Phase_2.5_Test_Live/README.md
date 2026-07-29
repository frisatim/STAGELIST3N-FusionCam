# Phase 2.5 - Premier dashboard live

Etape intermediaire entre la baseline (Phase 2) et la fusion (Phase 3) :
premier dashboard temps reel sur flux RTSP, pour valider la faisabilite
du live (decodage, inference, affichage, regles TAD/TRD simples) avant
de construire le pipeline multi-cameras.

## Script

| Script | Role |
|---|---|
| `dashboard_live_rtsp.py` | Dashboard live RTSP : detection sur plusieurs cameras, regles TAD/TRD en pixels, backends OpenCV/Ultralytics et GStreamer. |

## Statut

Outil d'etape, conserve pour la tracabilite du cheminement. Le dashboard
final du projet est celui de la Phase 4
(`Phase_4_Network_Latency/alert_dashboard.py`), alimente par les
metadonnees du pipeline Phase 3.

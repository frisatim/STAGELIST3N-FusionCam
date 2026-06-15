# Phase 4 Dashboard — Design Spec
Date: 2026-06-15
Status: approved

## Contexte

Refonte complète de `Phase_4_Network_Latency/alert_dashboard.py` pour passer d'un viewer mono-caméra basique à une interface multi-caméras exploitable en supervision live. Architecture séparée : vidéo via RTSP/MediaMTX/WHEP, IA via Phase 3 `run_live_campaign.py`, metadata via HTTP POST `/metadata`.

## Approche retenue

Single-file monolith (Approche A) : tout dans `alert_dashboard.py`. HTML/CSS/JS inline dans la constante `INDEX_HTML`. Python standard library uniquement. Pas de build step, pas de dépendances externes.

## Layout global

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4 LIVE DASHBOARD          [connected] [last: 12ms]       │
├──────────┬────────────┬────────────┬────────────────────────────┤
│ Detects  │ Weak       │ Confirmed  │ Metadata age (per cam)     │
│  42      │   8        │    3       │ cam_02: 0.3s  cam_03: 1.2s │
├──────────┴────────────┴────────────┴────────────────────────────┤
│                                            │                    │
│   ┌──────────────────────────────────┐     │  Alertes           │
│   │                                  │     │  (scroll)          │
│   │     CAMÉRA PRINCIPALE (16:9)     │     │                    │
│   │     canvas overlay bbox          │     │  [CONF] zone_viol  │
│   │                                  │     │  [weak] forbidden  │
│   └──────────────────────────────────┘     │                    │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │                    │
│   │cam_02│ │cam_03│ │cam_05│ │cam_07│     │                    │
│   └──────┘ └──────┘ └──────┘ └──────┘     │                    │
└────────────────────────────────────────────┴────────────────────┘
```

- Caméras par défaut affichées : cam_02, cam_03, cam_05, cam_07
- Caméras supplémentaires (cam_01, cam_04, cam_06, cam_08) apparaissent dynamiquement si elles reçoivent des metadata
- Cliquer sur une miniature la place en principal ; l'ancienne principale redescend en miniature
- Caméra principale au démarrage : `?camera=cam_02` (défaut cam_02)
- Layout CSS grid. Panneau alertes en colonne droite (écran large) ou en bas (mobile)

## Tuiles caméra

Structure HTML d'une tuile :
```html
<div class="tile" data-cam="cam_02">
  <iframe> ou <video>          <!-- flux vidéo, hidden si pas d'URL -->
  <canvas class="overlay">     <!-- bbox overlay, pointer-events: none -->
  <div class="tile-info">      <!-- label, nb bbox, latence -->
  <div class="badge">          <!-- connected / stale / no-metadata -->
</div>
```

- Sans URL vidéo : fond sombre #0b1117, nom centré, bbox dessinées quand même
- Placeholder actif même en l'absence de flux — les overlay restent fonctionnels

## Modes vidéo et query params

| Param | Comportement |
|-------|-------------|
| `?video_base=http://host:8889&video_mode=iframe` | iframe src = `base/cam_XX/` pour chaque cam |
| `?video_base=http://host:8889&video_mode=video` | video src = `base/cam_XX/` |
| `?cam_02=http://...&cam_03=http://...` | URLs individuelles par cam |
| `?camera=cam_05` | caméra principale au démarrage |

## Overlay bbox

Un `CanvasRenderer` par tuile. Buffer `lastMetadata[cam_id]` côté JS.

Priorité couleur (la plus haute l'emporte) :
1. Alerte `confirmed` → rouge `#ef4444`
2. Alerte `weak` → jaune `#fbbf24`
3. `personne` / `person` → cyan `#21c997`
4. Autre objet → orange `#f59e0b`

Un `global_id` présent dans `alerts[]` du même frame déclenche la surcoloration.

Label : `{class_name} #{global_id} {confidence}%` — fond coloré semi-transparent, texte blanc.

Scaling : `ResizeObserver` sur chaque canvas. Iframes : résolution supposée 1280×720.

Redessin : à chaque SSE metadata reçu, et à chaque redimensionnement (ex: miniature → principal).

## Badges d'état par caméra

| Condition | Badge | Couleur |
|-----------|-------|---------|
| metadata reçue dans les 2 dernières secondes | `connected` | vert |
| 2s < délai ≤ 5s | `stale` | orange |
| délai > 5s ou jamais reçue | `no metadata` | rouge/gris |

Tick de vérification : `setInterval` toutes les secondes.

## Barre de métriques (header)

- Total detections (cumulé session)
- Alertes weak (cumulé)
- Alertes confirmed (cumulé)
- Dernière latence metadata globale (ms)
- Âge metadata par caméra : affiché uniquement pour les caméras ayant reçu au moins une metadata (pas les 8 en dur). Format compact : `cam_02: 0.3s  cam_07: 1.2s`. Si > 5s, valeur colorée en rouge.

## Panneau Alertes

- Source : SSE `/events` + alertes extraites des enveloppes SSE `/metadata-events`
- Déduplication côté frontend par clé composite : `alert_id` (alertes venant de metadata-events) ou `event_id` (alertes venant de /events). Ces deux champs ont la même valeur quand l'alerte est issue d'une enveloppe metadata (`broadcast_metadata` copie `alert_id` → `event_id`). La Map JS utilise la clé `alert_id ?? event_id`.
- Insertion en tête, max 200 entrées
- Colonnes : Time / Level / Type / Cameras / Global ID / Latence delivery

Chaque entrée :
```
[14:32:07] CONFIRMED  zone_violation_person  cam_02+cam_07  #4  12.3ms
```

Badge level : `confirmed` (rouge), `weak` (orange/jaune)

## Backend Python — changements

### Simulateur amélioré

`--simulate` génère des enveloppes metadata complètes via `broadcast_metadata()` (qui re-dispatche automatiquement les alertes sur `/events`) :
- 1-3 détections aléatoires par cam active (bbox aléatoire dans 1280×720, class_name `personne`/`objet_interdit`, global_id séquentiel)
- 0-1 alerte aléatoire (weak ou confirmed) référençant un global_id des détections du frame
- `make_simulated_alert()` reste identique (utilisé par le test existant) — le simulateur n'appelle plus `broadcast_alert()` directement, il appelle `broadcast_metadata()` qui s'en charge

### Keepalive SSE

Commentaire SSE `": keepalive\n\n"` envoyé toutes les 15s dans les deux boucles SSE pour éviter les coupures proxy.

### Gestion BrokenPipe

Déjà dans `handle()`. On renforce avec `try/except` autour de `wfile.flush()` dans chaque boucle SSE.

### API — aucun changement

Tous les endpoints conservés tels quels :
- `GET /` → HTML
- `GET /events` → SSE alertes
- `GET /metadata-events` → SSE metadata
- `POST /alerts`
- `POST /metadata`

## Tests

Le test existant `test_make_simulated_alert_contains_dashboard_fields` reste valide — `make_simulated_alert` garde sa signature.

Aucun test HTML/JS (pas de headless browser dans la stack actuelle). Le comportement frontend est validé manuellement :
```
python Phase_4_Network_Latency/alert_dashboard.py --host 127.0.0.1 --port 8765 --simulate --rate-hz 2
```

Vérification syntaxe :
```
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py
python -m pytest Phase_4_Network_Latency
```

## Contraintes

- Python standard library uniquement
- Pas de React/Vite/npm
- JS vanilla (closures, pas d'ES modules)
- Un seul fichier : `alert_dashboard.py`
- Ne pas casser les tests existants
- Utilisable en local et derrière reverse proxy

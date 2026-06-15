# Phase 4 Dashboard Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réécrire `alert_dashboard.py` pour produire un dashboard multi-caméras avec overlay bbox, panneau alertes, métriques de latence, et gestion de l'état par caméra (connected/stale/no-metadata).

**Architecture:** Single-file monolith — tout dans `alert_dashboard.py`. Le HTML/CSS/JS est une constante Python inline `INDEX_HTML`. Le JS est vanilla (closures, pas d'ES modules). Le backend Python ne change que pour le simulateur et le keepalive SSE.

**Tech Stack:** Python 3.12 stdlib (`http.server`, `threading`, `queue`, `json`), HTML5 + CSS Grid + Canvas API + EventSource + ResizeObserver.

---

## Structure des fichiers

| Fichier | Action | Rôle |
|---------|--------|------|
| `Phase_4_Network_Latency/alert_dashboard.py` | Modifier | Seul fichier modifié |
| `Phase_4_Network_Latency/test_alert_dashboard_pytest.py` | Modifier | Ajouter test pour `make_simulated_metadata` |

Le repo git est à `STAGELIST3N-FusionCam/`. Toutes les commandes git : `git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam <cmd>`.

Les fichiers sont accessibles via junction Windows à `/mnt/c/Users/frisa/Desktop/BenchmarkingAI/Phase_4_Network_Latency/`.

---

## Task 1 : Ajouter `make_simulated_metadata()` avec test

**Files:**
- Modify: `Phase_4_Network_Latency/alert_dashboard.py` (après `make_simulated_alert`)
- Modify: `Phase_4_Network_Latency/test_alert_dashboard_pytest.py`

- [ ] **Step 1.1 : Ajouter le test dans `test_alert_dashboard_pytest.py`**

Lire le fichier d'abord. Ajouter à la fin :

```python
from Phase_4_Network_Latency.alert_dashboard import make_simulated_metadata


def test_make_simulated_metadata_structure():
    meta = make_simulated_metadata(42)
    assert meta["schema"] == "benchmarkingai.phase3.metadata.v1"
    assert meta["frame"] == 42
    assert isinstance(meta["detections"], list)
    assert len(meta["detections"]) >= 1
    assert isinstance(meta["alerts"], list)
    for det in meta["detections"]:
        assert "camera_id" in det
        assert "bbox_px" in det
        assert len(det["bbox_px"]) == 4
        assert det["class_name"] in {"personne", "objet_interdit"}
        assert 0.0 <= det["confidence"] <= 1.0
    for al in meta["alerts"]:
        assert al["alert_level"] in {"weak", "confirmed"}
        assert "alert_id" in al
```

- [ ] **Step 1.2 : Vérifier que le test échoue**

```bash
python -m pytest Phase_4_Network_Latency/test_alert_dashboard_pytest.py::test_make_simulated_metadata_structure -v
```

Résultat attendu : `FAILED` avec `ImportError: cannot import name 'make_simulated_metadata'`.

- [ ] **Step 1.3 : Implémenter `make_simulated_metadata()` dans `alert_dashboard.py`**

Ajouter après la fonction `make_simulated_alert` (ligne ~214), avant la classe `DashboardHandler` :

```python
def make_simulated_metadata(frame_id: int) -> dict:
    now_ms = time.time() * 1000.0
    cameras = ["cam_02", "cam_03", "cam_05", "cam_07"]
    active_cams = random.sample(cameras, k=random.randint(1, len(cameras)))
    detections = []
    gid = frame_id * 10
    for cam in active_cams:
        for _ in range(random.randint(1, 2)):
            gid += 1
            x1 = random.uniform(50, 800)
            y1 = random.uniform(30, 500)
            x2 = min(x1 + random.uniform(80, 300), 1270.0)
            y2 = min(y1 + random.uniform(100, 350), 710.0)
            cls = random.choice(["personne", "objet_interdit"])
            detections.append({
                "camera_id": cam,
                "track_id": gid,
                "global_id": gid,
                "class_id": 11 if cls == "personne" else 5,
                "class_name": cls,
                "confidence": round(random.uniform(0.4, 0.95), 4),
                "bbox_px": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "foot_point_px": [round((x1 + x2) / 2, 2), round(y2, 2)],
                "position_m": [round(random.uniform(0, 10), 3), round(random.uniform(0, 8), 3)],
                "zones": ["zone_1"] if random.random() > 0.5 else [],
                "timestamp": round(now_ms / 1000.0, 6),
            })
    alerts = []
    if detections and random.random() > 0.6:
        det = random.choice(detections)
        alerts.append({
            "alert_id": f"sim_{frame_id:06x}_{det['global_id']}",
            "alert_type": random.choice(["zone_violation_person", "forbidden_object"]),
            "alert_level": "confirmed" if random.random() > 0.5 else "weak",
            "global_id": det["global_id"],
            "zone_id": "zone_1",
            "class_name": det["class_name"],
            "position_m": det["position_m"],
            "cameras": [det["camera_id"]],
            "confidence": det["confidence"],
            "timestamp": det["timestamp"],
        })
    return {
        "schema": "benchmarkingai.phase3.metadata.v1",
        "created_epoch_ms": now_ms,
        "created_epoch_s": round(now_ms / 1000.0, 6),
        "frame": frame_id,
        "run_label": "simulated",
        "model_version": "sim",
        "model": "simulate",
        "format": "none",
        "detections": detections,
        "alerts": alerts,
    }
```

- [ ] **Step 1.4 : Vérifier que le test passe**

```bash
python -m pytest Phase_4_Network_Latency/test_alert_dashboard_pytest.py -v
```

Résultat attendu : `2 passed`.

- [ ] **Step 1.5 : Commit**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam add Phase_4_Network_Latency/alert_dashboard.py Phase_4_Network_Latency/test_alert_dashboard_pytest.py
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam commit -m "feat(dashboard): add make_simulated_metadata() with bbox/alert generation"
```

---

## Task 2 : Mettre à jour `start_simulator()` et renforcer les boucles SSE

**Files:**
- Modify: `Phase_4_Network_Latency/alert_dashboard.py`

- [ ] **Step 2.1 : Remplacer `start_simulator()`**

Trouver la fonction `start_simulator` (~ligne 301) et la remplacer par :

```python
def start_simulator(rate_hz: float) -> None:
    def run() -> None:
        frame_id = 0
        spacing_s = 1.0 / rate_hz if rate_hz > 0 else 1.0
        while True:
            broadcast_metadata(make_simulated_metadata(frame_id))
            frame_id += 1
            time.sleep(spacing_s)

    threading.Thread(target=run, daemon=True).start()
```

Note : on supprime l'appel direct à `broadcast_alert` — `broadcast_metadata` s'en charge pour les alertes contenues dans l'enveloppe.

- [ ] **Step 2.2 : Remplacer les deux boucles SSE dans `do_GET`**

La boucle pour `/events` (chercher `if parsed.path == "/events":`) :

```python
        if parsed.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            channel: queue.Queue[dict] = queue.Queue()
            with CLIENTS_LOCK:
                CLIENTS.append(channel)
            try:
                while True:
                    try:
                        alert = channel.get(timeout=15.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    payload = json.dumps(alert)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                pass
            finally:
                with CLIENTS_LOCK:
                    if channel in CLIENTS:
                        CLIENTS.remove(channel)
            return
```

La boucle pour `/metadata-events` (chercher `if parsed.path == "/metadata-events":`) :

```python
        if parsed.path == "/metadata-events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            channel: queue.Queue[dict] = queue.Queue()
            with METADATA_CLIENTS_LOCK:
                METADATA_CLIENTS.append(channel)
            try:
                while True:
                    try:
                        metadata = channel.get(timeout=15.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    payload = json.dumps(metadata)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                pass
            finally:
                with METADATA_CLIENTS_LOCK:
                    if channel in METADATA_CLIENTS:
                        METADATA_CLIENTS.remove(channel)
            return
```

- [ ] **Step 2.3 : Vérifier syntaxe et tests**

```bash
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py && echo "OK"
python -m pytest Phase_4_Network_Latency -v
```

Résultat attendu : `OK` et `2 passed`.

- [ ] **Step 2.4 : Commit**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam add Phase_4_Network_Latency/alert_dashboard.py
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam commit -m "feat(dashboard): simulator uses broadcast_metadata, SSE keepalive every 15s"
```

---

## Task 3 : Remplacer `INDEX_HTML` — structure HTML + CSS complet

**Files:**
- Modify: `Phase_4_Network_Latency/alert_dashboard.py` (constante `INDEX_HTML`)

> Note : dans les Tasks 3 à 9, on construit `INDEX_HTML` en remplaçant l'ancienne constante. La stratégie est de remplacer entièrement `INDEX_HTML = """..."""` en une seule opération à Task 3, puis d'ajouter le JS dans les Tasks suivantes en remplaçant le bloc `<script>` vide.

- [ ] **Step 3.1 : Remplacer toute la constante `INDEX_HTML`**

Localiser `INDEX_HTML = """<!doctype html>` (ligne ~19) jusqu'à `"""` (ligne ~172) et remplacer par ce qui suit. Le bloc `<script>` est vide pour l'instant — on l'enrichira dans les Tasks suivantes.

```python
INDEX_HTML = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 4 — Live Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #21262d;
      --text:     #e6edf3;
      --muted:    #7d8590;
      --accent:   #21c997;
      --warn:     #f59e0b;
      --danger:   #ef4444;
      --weak-col: #fbbf24;
    }
    html, body {
      margin: 0; padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      font-size: 13px;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* ── Header ─────────────────────────────────────── */
    #header {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 8px 16px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      flex-wrap: wrap;
    }
    .dashboard-title {
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
      white-space: nowrap;
    }
    #sse-badge {
      font-size: 10px;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 99px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .badge-connecting  { background: #1c2a3a; color: #7cb9e8; }
    .badge-connected   { background: #0d2918; color: var(--accent); }
    .badge-error       { background: #2d0a0a; color: var(--danger); }

    #metrics-bar {
      display: flex;
      gap: 20px;
      margin-left: auto;
    }
    .metric { display: flex; flex-direction: column; align-items: flex-end; }
    .metric-label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .05em; }
    .metric-value { font-size: 18px; font-weight: 700; line-height: 1.1; }
    #m-weak      { color: var(--weak-col); }
    #m-confirmed { color: var(--danger); }
    #m-latency   { color: var(--accent); }

    #cam-ages {
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    /* ── Workspace ───────────────────────────────────── */
    #workspace {
      display: grid;
      grid-template-columns: 1fr 300px;
      flex: 1;
      min-height: 0;
    }

    #video-area {
      display: flex;
      flex-direction: column;
      min-height: 0;
      border-right: 1px solid var(--border);
    }

    #main-stage {
      flex: 1;
      min-height: 0;
      background: #010409;
      position: relative;
    }

    #thumbnails {
      display: flex;
      flex-wrap: wrap;
      gap: 3px;
      padding: 3px;
      background: #010409;
      border-top: 1px solid var(--border);
      flex-shrink: 0;
    }

    /* ── Tiles ───────────────────────────────────────── */
    .tile {
      position: relative;
      background: #0b1117;
      overflow: hidden;
    }
    .tile-main {
      width: 100%;
      height: 100%;
    }
    .tile-thumb {
      width: 160px;
      height: 90px;
      cursor: pointer;
      transition: outline .1s;
    }
    .tile-thumb:hover { outline: 2px solid var(--accent); outline-offset: -2px; }

    .tile video, .tile iframe {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      border: 0;
      background: #0b1117;
    }
    .tile canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
    .tile-cam-name {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: #2d3340;
      font-size: 18px;
      font-weight: 700;
      pointer-events: none;
      user-select: none;
    }
    .tile-info {
      position: absolute;
      bottom: 0; left: 0; right: 0;
      background: linear-gradient(transparent, rgba(0,0,0,.75));
      padding: 4px 6px 3px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 6px;
      font-size: 11px;
      color: #cdd5de;
    }
    .tile-cam-label { font-weight: 600; }
    .tile-bbox-count { color: var(--muted); }
    .tile-lat { color: var(--accent); white-space: nowrap; }
    .tile-badge {
      position: absolute;
      top: 4px; right: 4px;
      font-size: 9px;
      font-weight: 700;
      padding: 2px 5px;
      border-radius: 3px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .badge-connected   { background: #0d2918; color: var(--accent); }
    .badge-stale       { background: #2a1d00; color: var(--warn); }
    .badge-no-metadata { background: #1a0505; color: var(--muted); }

    /* ── Alert panel ─────────────────────────────────── */
    #alert-panel {
      display: flex;
      flex-direction: column;
      min-height: 0;
      background: var(--surface);
    }
    .panel-header {
      padding: 8px 12px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    #alert-list {
      overflow-y: auto;
      flex: 1;
      padding: 4px 0;
    }
    .alert-row {
      padding: 5px 12px;
      border-bottom: 1px solid var(--border);
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 2px 8px;
      font-size: 11px;
    }
    .alert-row:hover { background: rgba(255,255,255,.03); }
    .alert-time { color: var(--muted); grid-row: 1; }
    .alert-level { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 700; letter-spacing: .04em; }
    .level-confirmed { background: #2d0a0a; color: var(--danger); }
    .level-weak      { background: #2a1d00; color: var(--weak-col); }
    .alert-type { color: var(--text); }
    .alert-meta { color: var(--muted); }
    .alert-lat  { color: var(--accent); margin-left: 4px; }

    /* ── Responsive ──────────────────────────────────── */
    @media (max-width: 860px) {
      #workspace { grid-template-columns: 1fr; grid-template-rows: 1fr 200px; }
      #video-area { border-right: none; border-bottom: 1px solid var(--border); }
      #metrics-bar { gap: 12px; }
      .tile-thumb { width: 120px; height: 68px; }
    }
  </style>
</head>
<body>
  <div id="header">
    <span class="dashboard-title">Phase 4 Live Dashboard</span>
    <span id="sse-badge" class="badge-connecting">connecting</span>
    <div id="metrics-bar">
      <div class="metric">
        <span class="metric-label">Detections</span>
        <span class="metric-value" id="m-detections">0</span>
      </div>
      <div class="metric">
        <span class="metric-label">Weak</span>
        <span class="metric-value" id="m-weak">0</span>
      </div>
      <div class="metric">
        <span class="metric-label">Confirmed</span>
        <span class="metric-value" id="m-confirmed">0</span>
      </div>
      <div class="metric">
        <span class="metric-label">Latence</span>
        <span class="metric-value" id="m-latency">—</span>
      </div>
    </div>
    <div id="cam-ages"></div>
  </div>

  <div id="workspace">
    <div id="video-area">
      <div id="main-stage"></div>
      <div id="thumbnails"></div>
    </div>
    <div id="alert-panel">
      <div class="panel-header">Alertes</div>
      <div id="alert-list"></div>
    </div>
  </div>

  <script>
  // JS will be added in Tasks 4-9
  </script>
</body>
</html>
"""
```

- [ ] **Step 3.2 : Vérifier syntaxe Python**

```bash
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py && echo "OK"
```

Résultat attendu : `OK`.

- [ ] **Step 3.3 : Smoke test visuel**

```bash
python Phase_4_Network_Latency/alert_dashboard.py --host 127.0.0.1 --port 8765 --simulate --rate-hz 2
```

Ouvrir http://127.0.0.1:8765/ — on doit voir le header avec métriques, le layout en deux colonnes, et la zone vidéo vide. Le panneau Alertes est visible à droite. Aucune interaction JS pour l'instant.

- [ ] **Step 3.4 : Commit**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam add Phase_4_Network_Latency/alert_dashboard.py
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam commit -m "feat(dashboard): new HTML/CSS layout - header, grid, tiles, alert panel"
```

---

## Task 4 : JS — Config, state, et tile DOM builder

**Files:**
- Modify: `Phase_4_Network_Latency/alert_dashboard.py` (bloc `<script>`)

- [ ] **Step 4.1 : Remplacer le commentaire `// JS will be added` par le JS de config + state + tile builder**

Localiser `// JS will be added in Tasks 4-9` dans `INDEX_HTML` et remplacer par :

```javascript
  // ─── CONFIG ────────────────────────────────────────────────────────────────
  const params = new URLSearchParams(location.search);
  const VIDEO_BASE = params.get('video_base') || '';
  const VIDEO_MODE = params.get('video_mode') || 'none'; // 'iframe' | 'video' | 'none'
  const INITIAL_PRIMARY = params.get('camera') || 'cam_02';
  const DEFAULT_CAMS = ['cam_02', 'cam_03', 'cam_05', 'cam_07'];

  function videoUrl(camId) {
    const individual = params.get(camId);
    if (individual) return individual;
    if (!VIDEO_BASE) return null;
    return VIDEO_BASE.replace(/\\/+$/, '') + '/' + camId + '/';
  }

  // ─── STATE ─────────────────────────────────────────────────────────────────
  const state = {
    primaryCam: INITIAL_PRIMARY,
    activeCams: new Set(DEFAULT_CAMS),
    lastMetadata: {},       // cam_id → full metadata envelope
    lastMetaTime: {},       // cam_id → Date.now() at reception
    alertMap: new Map(),    // alert_id → true (dedup)
    counts: { detections: 0, weak: 0, confirmed: 0 },
  };

  // ─── TILE REGISTRY ─────────────────────────────────────────────────────────
  // tiles[camId] = { tile, canvas, ctx, mediaEl, badge, bboxCount, latEl, ro }
  const tiles = {};

  function makeTile(camId) {
    const tile = document.createElement('div');
    tile.className = 'tile tile-thumb';
    tile.dataset.cam = camId;

    const nameEl = document.createElement('div');
    nameEl.className = 'tile-cam-name';
    nameEl.textContent = camId;
    tile.appendChild(nameEl);

    const url = videoUrl(camId);
    let mediaEl = null;
    if (url && VIDEO_MODE === 'iframe') {
      mediaEl = document.createElement('iframe');
      mediaEl.src = url;
      mediaEl.allow = 'autoplay; fullscreen; camera; microphone';
      tile.appendChild(mediaEl);
    } else if (url && VIDEO_MODE === 'video') {
      mediaEl = document.createElement('video');
      mediaEl.src = url;
      mediaEl.autoplay = true;
      mediaEl.muted = true;
      mediaEl.playsInline = true;
      tile.appendChild(mediaEl);
    }

    const canvas = document.createElement('canvas');
    canvas.className = 'overlay';
    tile.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const ro = new ResizeObserver(() => {
      const r = canvas.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) {
        canvas.width = Math.round(r.width);
        canvas.height = Math.round(r.height);
        if (state.lastMetadata[camId]) drawBbox(camId);
      }
    });
    ro.observe(canvas);

    const info = document.createElement('div');
    info.className = 'tile-info';
    const labelEl = document.createElement('span');
    labelEl.className = 'tile-cam-label';
    labelEl.textContent = camId;
    const bboxCount = document.createElement('span');
    bboxCount.className = 'tile-bbox-count';
    const latEl = document.createElement('span');
    latEl.className = 'tile-lat';
    info.appendChild(labelEl);
    info.appendChild(bboxCount);
    info.appendChild(latEl);
    tile.appendChild(info);

    const badge = document.createElement('div');
    badge.className = 'tile-badge badge-no-metadata';
    badge.textContent = 'no metadata';
    tile.appendChild(badge);

    tile.addEventListener('click', () => {
      if (state.primaryCam !== camId) setPrimary(camId);
    });

    tiles[camId] = { tile, canvas, ctx, mediaEl, badge, bboxCount, latEl, ro, nameEl };
    return tiles[camId];
  }

  function setPrimary(camId) {
    state.primaryCam = camId;
    renderGrid();
  }

  function renderGrid() {
    const mainStage = document.getElementById('main-stage');
    const thumbsEl = document.getElementById('thumbnails');
    mainStage.innerHTML = '';
    thumbsEl.innerHTML = '';

    for (const camId of state.activeCams) {
      if (!tiles[camId]) makeTile(camId);
      const t = tiles[camId];
      if (camId === state.primaryCam) {
        t.tile.className = 'tile tile-main';
        mainStage.appendChild(t.tile);
      } else {
        t.tile.className = 'tile tile-thumb';
        thumbsEl.appendChild(t.tile);
      }
    }

    // Redraw bbox after DOM rearrangement (ResizeObserver fires, but force anyway)
    for (const camId of state.activeCams) {
      if (state.lastMetadata[camId]) drawBbox(camId);
    }
  }
```

- [ ] **Step 4.2 : Vérifier syntaxe Python**

```bash
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py && echo "OK"
```

- [ ] **Step 4.3 : Vérifier syntaxe JS**

```bash
node --input-type=module <<'EOF'
// quick parse check — no imports needed
const src = require('fs').readFileSync('Phase_4_Network_Latency/alert_dashboard.py', 'utf8');
const match = src.match(/<script>([\s\S]*?)<\/script>/);
if (!match) { console.error('no script block'); process.exit(1); }
new Function(match[1]);
console.log('JS syntax OK');
EOF
```

Si `node` n'est pas disponible, vérifier manuellement dans le navigateur.

- [ ] **Step 4.4 : Commit**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam add Phase_4_Network_Latency/alert_dashboard.py
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam commit -m "feat(dashboard): JS config, state, tile builder, renderGrid"
```

---

## Task 5 : JS — CanvasRenderer (drawBbox)

**Files:**
- Modify: `Phase_4_Network_Latency/alert_dashboard.py` (bloc `<script>`)

- [ ] **Step 5.1 : Ajouter `drawBbox` après `renderGrid`**

Localiser la fin de la fonction `renderGrid` (ligne finissant par `}`) dans le bloc `<script>` et ajouter immédiatement après :

```javascript
  // ─── BBOX DRAWING ──────────────────────────────────────────────────────────
  function getDetColor(det, alertedIds, alertLevels) {
    const gid = det.global_id;
    if (gid != null && alertedIds.has(gid)) {
      return alertLevels[gid] === 'confirmed' ? '#ef4444' : '#fbbf24';
    }
    return (det.class_name === 'personne' || det.class_name === 'person')
      ? '#21c997' : '#f59e0b';
  }

  function drawBbox(camId) {
    const t = tiles[camId];
    if (!t) return;
    const { canvas, ctx, mediaEl } = t;
    const W = canvas.width;
    const H = canvas.height;
    if (W === 0 || H === 0) return;

    const meta = state.lastMetadata[camId];
    if (!meta) { ctx.clearRect(0, 0, W, H); return; }

    const dets = (meta.detections || []).filter(d => d.camera_id === camId);

    // Build alert lookup
    const alertedIds = new Set();
    const alertLevels = {};
    for (const al of (meta.alerts || [])) {
      const gid = al.global_id;
      if (gid != null) {
        alertedIds.add(gid);
        if (!alertLevels[gid] || al.alert_level === 'confirmed') {
          alertLevels[gid] = al.alert_level;
        }
      }
    }

    // Scale to canvas
    let videoW = 1280, videoH = 720;
    if (mediaEl && mediaEl.tagName === 'VIDEO' && mediaEl.videoWidth) {
      videoW = mediaEl.videoWidth;
      videoH = mediaEl.videoHeight;
    }
    const scale = Math.min(W / videoW, H / videoH);
    const offX = (W - videoW * scale) / 2;
    const offY = (H - videoH * scale) / 2;

    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 2;
    ctx.font = 'bold 11px system-ui, sans-serif';

    for (const det of dets) {
      const [x1, y1, x2, y2] = det.bbox_px;
      const sx = offX + x1 * scale;
      const sy = offY + y1 * scale;
      const sw = (x2 - x1) * scale;
      const sh = (y2 - y1) * scale;
      const color = getDetColor(det, alertedIds, alertLevels);

      ctx.strokeStyle = color;
      ctx.strokeRect(sx, sy, sw, sh);

      const label = det.class_name + ' #' + (det.global_id != null ? det.global_id : det.track_id)
        + ' ' + Math.round(det.confidence * 100) + '%';
      const tw = ctx.measureText(label).width + 8;
      const ly = Math.max(0, sy - 16);

      ctx.globalAlpha = 0.8;
      ctx.fillStyle = color;
      ctx.fillRect(sx, ly, tw, 14);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, sx + 4, ly + 11);
    }

    // Update tile info
    t.bboxCount.textContent = dets.length + ' bbox';
  }
```

- [ ] **Step 5.2 : Vérifier syntaxe Python**

```bash
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py && echo "OK"
```

- [ ] **Step 5.3 : Commit**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam add Phase_4_Network_Latency/alert_dashboard.py
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam commit -m "feat(dashboard): CanvasRenderer - drawBbox with color priority"
```

---

## Task 6 : JS — Panneau Alertes (`addAlertRow`)

**Files:**
- Modify: `Phase_4_Network_Latency/alert_dashboard.py` (bloc `<script>`)

- [ ] **Step 6.1 : Ajouter `addAlertRow` après `drawBbox`**

```javascript
  // ─── ALERT PANEL ───────────────────────────────────────────────────────────
  function addAlertRow(al) {
    const list = document.getElementById('alert-list');
    const time = new Date(al.received_epoch_ms || Date.now()).toLocaleTimeString();
    const cameras = Array.isArray(al.cameras)
      ? al.cameras.join('+')
      : (al.camera || '—');
    const level = al.alert_level || 'unknown';
    const levelCls = level === 'confirmed' ? 'level-confirmed' : 'level-weak';
    const lat = al.delivery_latency_ms != null
      ? al.delivery_latency_ms.toFixed(1) + ' ms' : '—';
    const gid = al.global_id != null ? '#' + al.global_id : '';

    const row = document.createElement('div');
    row.className = 'alert-row';
    row.innerHTML =
      '<span class="alert-time">' + time + '</span>'
      + '<span><span class="alert-level ' + levelCls + '">' + level.toUpperCase() + '</span></span>'
      + '<span class="alert-type">' + (al.alert_type || '') + '</span>'
      + '<span class="alert-meta">' + cameras + ' ' + gid
        + '<span class="alert-lat">' + lat + '</span></span>';
    list.prepend(row);
    while (list.children.length > 200) list.removeChild(list.lastChild);

    // Update counts
    const countKey = level === 'confirmed' ? 'confirmed' : 'weak';
    state.counts[countKey] = (state.counts[countKey] || 0) + 1;
    document.getElementById('m-' + countKey).textContent = state.counts[countKey];
  }
```

- [ ] **Step 6.2 : Vérifier syntaxe**

```bash
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py && echo "OK"
```

- [ ] **Step 6.3 : Commit**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam add Phase_4_Network_Latency/alert_dashboard.py
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam commit -m "feat(dashboard): alert panel with dedup and level badges"
```

---

## Task 7 : JS — Handler metadata SSE (`handleMetadata`)

**Files:**
- Modify: `Phase_4_Network_Latency/alert_dashboard.py` (bloc `<script>`)

- [ ] **Step 7.1 : Ajouter `handleMetadata` après `addAlertRow`**

```javascript
  // ─── METADATA HANDLER ──────────────────────────────────────────────────────
  function handleMetadata(meta) {
    const now = Date.now();

    // Collect which cameras are referenced
    const camIds = new Set();
    for (const det of (meta.detections || [])) camIds.add(det.camera_id);
    for (const al of (meta.alerts || [])) {
      for (const cam of (al.cameras || [])) camIds.add(cam);
    }

    // Add unknown cameras dynamically
    for (const camId of camIds) {
      if (!state.activeCams.has(camId)) {
        state.activeCams.add(camId);
        makeTile(camId);
        document.getElementById('thumbnails').appendChild(tiles[camId].tile);
      }
    }

    // Update per-camera state
    for (const camId of camIds) {
      state.lastMetadata[camId] = meta;
      state.lastMetaTime[camId] = now;
      if (meta.delivery_latency_ms != null && tiles[camId]) {
        tiles[camId].latEl.textContent = meta.delivery_latency_ms.toFixed(1) + ' ms';
      }
    }

    // Metrics
    const detCount = (meta.detections || []).length;
    state.counts.detections += detCount;
    document.getElementById('m-detections').textContent = state.counts.detections;
    if (meta.delivery_latency_ms != null) {
      document.getElementById('m-latency').textContent =
        meta.delivery_latency_ms.toFixed(1) + ' ms';
    }

    // Alerts from metadata (deduplicated)
    for (const al of (meta.alerts || [])) {
      const key = al.alert_id || al.event_id;
      if (key && state.alertMap.has(key)) continue;
      if (key) state.alertMap.set(key, true);
      addAlertRow({
        alert_level: al.alert_level,
        alert_type: al.alert_type,
        cameras: al.cameras || [],
        global_id: al.global_id,
        delivery_latency_ms: meta.delivery_latency_ms,
        received_epoch_ms: now,
      });
    }

    // Redraw bbox for all affected cameras
    for (const camId of camIds) drawBbox(camId);
  }
```

- [ ] **Step 7.2 : Vérifier syntaxe**

```bash
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py && echo "OK"
```

- [ ] **Step 7.3 : Commit**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam add Phase_4_Network_Latency/alert_dashboard.py
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam commit -m "feat(dashboard): handleMetadata - dynamic cameras, metrics, alert dedup"
```

---

## Task 8 : JS — SSE connections, badge loop, et `init()`

**Files:**
- Modify: `Phase_4_Network_Latency/alert_dashboard.py` (bloc `<script>`)

- [ ] **Step 8.1 : Ajouter les connexions SSE + badge loop + init après `handleMetadata`**

```javascript
  // ─── BADGE UPDATE LOOP ─────────────────────────────────────────────────────
  function updateBadgesAndAges() {
    const now = Date.now();
    const agesEl = document.getElementById('cam-ages');
    const parts = [];

    for (const camId of state.activeCams) {
      const t = tiles[camId];
      if (!t) continue;
      const lastTime = state.lastMetaTime[camId];
      const ageMs = lastTime ? now - lastTime : Infinity;

      let cls, label;
      if (ageMs < 2000)      { cls = 'tile-badge badge-connected';   label = 'connected'; }
      else if (ageMs < 5000) { cls = 'tile-badge badge-stale';       label = 'stale'; }
      else                   { cls = 'tile-badge badge-no-metadata';  label = 'no metadata'; }
      t.badge.className = cls;
      t.badge.textContent = label;

      if (lastTime) {
        const s = (ageMs / 1000).toFixed(1);
        const color = ageMs > 5000 ? '#ef4444' : ageMs > 2000 ? '#f59e0b' : '#21c997';
        parts.push('<span style="color:' + color + '">' + camId + ': ' + s + 's</span>');
      }
    }
    agesEl.innerHTML = parts.join(' &nbsp; ');
  }

  // ─── SSE ───────────────────────────────────────────────────────────────────
  function connectSSE() {
    const sseBadge = document.getElementById('sse-badge');

    const metaSource = new EventSource('/metadata-events');
    metaSource.onmessage = (e) => handleMetadata(JSON.parse(e.data));
    metaSource.onopen = () => {
      sseBadge.className = 'badge-connected';
      sseBadge.textContent = 'connected';
    };
    metaSource.onerror = () => {
      sseBadge.className = 'badge-error';
      sseBadge.textContent = 'disconnected';
    };

    const eventsSource = new EventSource('/events');
    eventsSource.onmessage = (e) => {
      const al = JSON.parse(e.data);
      // Deduplicate: alerts from broadcast_metadata() already arrive via metadata-events
      const key = al.alert_id || al.event_id;
      if (key && state.alertMap.has(key)) return;
      if (key) state.alertMap.set(key, true);
      addAlertRow({
        alert_level: al.alert_level,
        alert_type: al.alert_type,
        cameras: al.camera ? [al.camera] : (al.cameras || []),
        global_id: al.global_id,
        delivery_latency_ms: al.delivery_latency_ms,
        received_epoch_ms: al.received_epoch_ms || Date.now(),
      });
    };
  }

  // ─── INIT ──────────────────────────────────────────────────────────────────
  (function init() {
    for (const camId of DEFAULT_CAMS) makeTile(camId);
    renderGrid();
    connectSSE();
    setInterval(updateBadgesAndAges, 1000);
  })();
```

- [ ] **Step 8.2 : Vérifier syntaxe Python**

```bash
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py && echo "OK"
```

- [ ] **Step 8.3 : Suite de tests complète**

```bash
python -m pytest Phase_4_Network_Latency -v
```

Résultat attendu : `2 passed`.

- [ ] **Step 8.4 : Commit**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam add Phase_4_Network_Latency/alert_dashboard.py
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam commit -m "feat(dashboard): SSE connections, badge loop, init - dashboard fully wired"
```

---

## Task 9 : Test d'intégration manuel complet

**Files:** (aucun fichier modifié)

- [ ] **Step 9.1 : Smoke test simulateur**

```bash
python Phase_4_Network_Latency/alert_dashboard.py --host 127.0.0.1 --port 8765 --simulate --rate-hz 2
```

Ouvrir http://127.0.0.1:8765/ et vérifier :
- [ ] 4 tuiles visibles (cam_02 en principal, cam_03/05/07 en miniatures)
- [ ] Métriques qui s'incrémentent (Detections, Weak, Confirmed)
- [ ] Bbox dessinées sur les tuiles (rectangles colorés avec labels)
- [ ] Badges `connected` en vert sur les caméras actives
- [ ] Panneau Alertes avec des entrées qui arrivent
- [ ] Cliquer sur une miniature → elle passe en principal
- [ ] Badge SSE "connected" en header

- [ ] **Step 9.2 : Test mode iframe (simulé, sans MediaMTX)**

```bash
# Ouvrir avec video_base et video_mode=iframe — les iframes pointent vers une URL qui ne répond pas
# mais les bbox doivent quand même se dessiner sur le canvas
```

Ouvrir http://127.0.0.1:8765/?video_base=http://127.0.0.1:8889&video_mode=iframe

Vérifier : les bbox sont dessinées même si les iframes affichent une erreur de connexion.

- [ ] **Step 9.3 : Test stabilité fermeture navigateur**

Fermer l'onglet, attendre 20s, rouvrir. Vérifier que le serveur Python n'a pas crashé et accepte la nouvelle connexion proprement (pas de traceback dans le terminal).

- [ ] **Step 9.4 : Test refresh rapide**

Recharger la page 5 fois rapidement. Vérifier que le serveur reste stable.

- [ ] **Step 9.5 : Commit final si tout est OK**

```bash
git -C /mnt/c/Users/frisa/Desktop/BenchmarkingAI/STAGELIST3N-FusionCam log --oneline -8
```

---

## Checklist de vérification finale

```bash
# Syntaxe Python
python -m py_compile Phase_4_Network_Latency/alert_dashboard.py && echo "py_compile OK"

# Tests unitaires
python -m pytest Phase_4_Network_Latency -v

# Commande de test complète (manuel)
python Phase_4_Network_Latency/alert_dashboard.py --host 127.0.0.1 --port 8765 --simulate --rate-hz 2
# → http://127.0.0.1:8765/

# Test avec MediaMTX (si disponible)
# → http://127.0.0.1:8765/?video_base=http://127.0.0.1:8889&video_mode=iframe

# Test avec Phase 3
# python Phase_3_Fusion_MultiCam/run_live_campaign.py \
#   --versions V4 --models yolov8s --formats fp32_engine \
#   --cameras cam_02,cam_07 --duration-min 5 --device cuda:0 \
#   --object-min-camera-votes 2 --capture-backend opencv \
#   --no-display --no-record-video \
#   --metadata-http-url http://127.0.0.1:8765/metadata \
#   --metadata-jsonl Phase_3_Fusion_MultiCam/reports/dashboard_metadata.jsonl
```

from __future__ import annotations

import argparse
import json
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml


CLIENTS: list[queue.Queue[dict]] = []
METADATA_CLIENTS: list[queue.Queue[dict]] = []
CLIENTS_LOCK = threading.Lock()
METADATA_CLIENTS_LOCK = threading.Lock()
ZONES_BY_CAMERA: dict[str, list[dict]] = {}
LAST_METADATA: dict | None = None
LAST_METADATA_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()
METADATA_POSTS = 0
ALERT_POSTS = 0


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 4 Live Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef2f5;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #64748b;
      --line: #d8e0e8;
      --dark: #111820;
      --dark2: #17212b;
      --green: #16a34a;
      --amber: #d97706;
      --red: #dc2626;
      --blue: #2563eb;
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 22px;
      background: var(--dark);
      color: #f8fafc;
      border-bottom: 1px solid #253241;
    }
    header h1 { margin: 0; font-size: 17px; font-weight: 700; letter-spacing: 0; }
    header .hint { color: #b6c2cf; font-size: 13px; }
    main {
      min-height: calc(100vh - 56px);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      padding: 16px;
    }
    .left, .right { min-width: 0; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 70px;
    }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { display: block; font-size: 25px; margin-top: 5px; line-height: 1; }
    .camera-area {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 620px;
    }
    .focus-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 250px;
      gap: 10px;
      height: calc(100vh - 176px);
      min-height: 520px;
    }
    .thumbs {
      display: grid;
      grid-auto-rows: minmax(120px, 1fr);
      gap: 10px;
      overflow: auto;
    }
    .grid-layout {
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 10px;
    }
    .tile {
      position: relative;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid #2f3c49;
      background: #111820;
      min-height: 160px;
      cursor: pointer;
    }
    .tile.focus { min-height: 520px; cursor: default; }
    .tile-stage { position: relative; width: 100%; height: 100%; aspect-ratio: 4 / 3; background: #101418; }
    .tile.focus .tile-stage { height: 100%; aspect-ratio: auto; }
    .tile video, .tile iframe, .tile canvas, .placeholder {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }
    .tile video { object-fit: contain; background: #101418; }
    .tile video, .tile iframe, .placeholder { z-index: 1; }
    .tile iframe { border: 0; background: #101418; }
    .tile canvas { pointer-events: none; z-index: 2; }
    .placeholder {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
      color: #98a6b3;
      text-align: center;
      font-size: 13px;
    }
    .tile-top {
      position: absolute;
      z-index: 3;
      top: 8px;
      left: 8px;
      right: 8px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      pointer-events: none;
    }
    .label, .badge {
      border-radius: 6px;
      padding: 5px 7px;
      font-size: 12px;
      line-height: 1;
      background: rgba(17, 24, 32, 0.86);
      color: #f8fafc;
      border: 1px solid rgba(255,255,255,0.12);
    }
    .badge.live { color: #d1fae5; border-color: rgba(22, 163, 74, .45); }
    .badge.stale { color: #fef3c7; border-color: rgba(217, 119, 6, .45); }
    .badge.off { color: #cbd5e1; border-color: rgba(148, 163, 184, .35); }
    .tile-bottom {
      position: absolute;
      z-index: 3;
      left: 8px;
      right: 8px;
      bottom: 8px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: #dbe5ef;
      font-size: 12px;
      pointer-events: none;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 12px;
    }
    .panel h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
    }
    .camera-status, .events { padding: 10px 12px; }
    .status-row {
      display: grid;
      grid-template-columns: 70px 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 7px 0;
      border-bottom: 1px solid #edf1f5;
      font-size: 13px;
    }
    .status-row:last-child { border-bottom: 0; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #94a3b8; display: inline-block; margin-right: 6px; }
    .dot.live { background: var(--green); }
    .dot.stale { background: var(--amber); }
    .dot.off { background: #94a3b8; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 9px 10px; border-bottom: 1px solid #edf1f5; text-align: left; font-size: 13px; }
    th { color: var(--muted); font-weight: 700; background: #f8fafc; }
    .weak { color: var(--amber); font-weight: 800; }
    .confirmed { color: var(--red); font-weight: 800; }
    .empty {
      color: var(--muted);
      font-size: 13px;
      padding: 14px;
    }
    @media (max-width: 1050px) {
      main { grid-template-columns: 1fr; }
      .focus-layout { grid-template-columns: 1fr; height: auto; }
      .thumbs { grid-template-columns: repeat(2, 1fr); grid-auto-rows: minmax(140px, 1fr); }
    }
    @media (max-width: 700px) {
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .grid-layout { grid-template-columns: 1fr; }
      .thumbs { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Phase 4 Live Dashboard</h1>
    <span class="hint">video stream + AI metadata overlay</span>
  </header>
  <main>
    <section class="left">
      <section class="metrics">
        <div class="metric"><span>Cameras</span><strong id="cameraCount">0</strong></div>
        <div class="metric"><span>Detections</span><strong id="detCount">0</strong></div>
        <div class="metric"><span>Alerts</span><strong id="total">0</strong></div>
        <div class="metric"><span>Metadata latency</span><strong id="latency">0 ms</strong></div>
      </section>
      <section class="camera-area" id="cameraArea"></section>
    </section>
    <aside class="right">
      <section class="panel">
        <h2>Camera Health</h2>
        <div class="camera-status" id="cameraStatus"></div>
      </section>
      <section class="panel">
        <h2>Alerts</h2>
        <table>
          <thead><tr><th>Time</th><th>Level</th><th>Type</th><th>Camera</th></tr></thead>
          <tbody id="events"></tbody>
        </table>
        <div class="empty" id="emptyAlerts">No alerts received yet.</div>
      </section>
    </aside>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/hls.js@1"></script>
  <script>
    const params = new URLSearchParams(window.location.search);
    const DEFAULT_CAMERAS = ["cam_02", "cam_03", "cam_05", "cam_07"];
    const requestedCameras = (params.get("cameras") || params.get("camera") || DEFAULT_CAMERAS.join(","))
      .split(",").map((x) => x.trim()).filter(Boolean);
    const videoBase = (params.get("video_base") || "").replace(/\\/$/, "");
    const explicitVideo = params.get("video") || "";
    const videoMode = params.get("video_mode") || params.get("mode") || (videoBase ? "iframe" : "none");
    const sourceW = Number(params.get("source_w") || 768);
    const sourceH = Number(params.get("source_h") || 576);
    const overlayDelayMs = Math.max(0, Number(params.get("overlay_delay_ms") || 0));
    const syncMode = params.get("sync_mode") || (videoMode === "hls" ? "video" : "delay");
    const syncOffsetMs = Number(params.get("sync_offset_ms") || 1000);
    const syncSafetyMs = Number(params.get("sync_safety_ms") || 250);
    const state = {
      cameras: new Map(),
      zones: new Map(),
      focusCam: params.get("focus") || requestedCameras[0] || "cam_02",
      alerts: new Map(),
      totalAlerts: 0,
      lastLatencyMs: 0,
      metadataBuffer: [],
      lastAppliedMetadataKey: "",
      layoutCameraKey: "",
    };
    const cameraArea = document.getElementById("cameraArea");
    const cameraStatus = document.getElementById("cameraStatus");
    const eventsBody = document.getElementById("events");
    const emptyAlerts = document.getElementById("emptyAlerts");

    function ensureCamera(camId) {
      if (!state.cameras.has(camId)) {
        state.cameras.set(camId, {
          camId,
          detections: [],
          frame: "-",
          lastMetadataMs: 0,
          deliveryLatencyMs: 0,
          videoLatencyMs: 0,
          mediaEl: null,
          hls: null,
          canvas: null,
          ctx: null,
        });
      }
      return state.cameras.get(camId);
    }

    for (const camId of requestedCameras) ensureCamera(camId);

    function videoUrlFor(camId) {
      if (explicitVideo && requestedCameras.length === 1) return explicitVideo;
      if (!videoBase) return "";
      if (videoMode === "hls") return `${videoBase}/${camId}/index.m3u8`;
      return `${videoBase}/${camId}/`;
    }

    function classColor(det) {
      const name = String(det.class_name || "").toLowerCase();
      if (name === "person" || name === "personne" || name === "human") return "#22c55e";
      return "#f59e0b";
    }

    function zoneColor(index) {
      return ["#ef4444", "#38bdf8", "#facc15", "#a855f7", "#14b8a6"][index % 5];
    }

    function cameraFreshness(cam) {
      if (!cam.lastMetadataMs) return "off";
      const age = Date.now() - cam.lastMetadataMs;
      if (age < 2200) return "live";
      if (age < 7000) return "stale";
      return "off";
    }

    function updateVideoLatency(cam) {
      if (!cam || !cam.mediaEl) return;
      let latencyMs = 0;
      if (cam.hls && Number.isFinite(cam.hls.latency)) {
        latencyMs = cam.hls.latency * 1000;
      } else {
        const video = cam.mediaEl;
        try {
          if (video.seekable && video.seekable.length > 0) {
            const liveEdge = video.seekable.end(video.seekable.length - 1);
            latencyMs = Math.max(0, (liveEdge - video.currentTime) * 1000);
          }
        } catch (_) {}
      }
      if (latencyMs > 0 && Number.isFinite(latencyMs)) {
        cam.videoLatencyMs = latencyMs;
      }
    }

    function effectiveOverlayDelayMs() {
      if (syncMode === "video") {
        const latencies = Array.from(state.cameras.values())
          .map((cam) => {
            updateVideoLatency(cam);
            return Number(cam.videoLatencyMs || 0);
          })
          .filter((value) => value > 0);
        if (latencies.length > 0) {
          return Math.max(...latencies) + syncOffsetMs + syncSafetyMs;
        }
      }
      return overlayDelayMs;
    }

    function setupHlsVideo(camId, cam, video, url) {
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.controls = true;
      video.preload = "auto";
      if (video.canPlayType("application/vnd.apple.mpegurl")) {
        video.src = url;
        video.addEventListener("timeupdate", () => updateVideoLatency(cam));
        return;
      }
      if (window.Hls && window.Hls.isSupported()) {
        const hls = new window.Hls({
          lowLatencyMode: true,
          liveSyncDurationCount: 3,
          maxLiveSyncPlaybackRate: 1.5,
        });
        cam.hls = hls;
        hls.loadSource(url);
        hls.attachMedia(video);
        hls.on(window.Hls.Events.FRAG_CHANGED, () => updateVideoLatency(cam));
        hls.on(window.Hls.Events.LEVEL_LOADED, () => updateVideoLatency(cam));
        hls.on(window.Hls.Events.ERROR, (_event, data) => {
          if (data && data.fatal) {
            try { hls.startLoad(); } catch (_) {}
          }
        });
        return;
      }
      video.src = url;
    }

    function buildMedia(camId, cam) {
      const url = videoUrlFor(camId);
      const stage = document.createElement("div");
      stage.className = "tile-stage";
      if (!url || videoMode === "none") {
        const ph = document.createElement("div");
        ph.className = "placeholder";
        ph.textContent = "No browser video URL configured. Use ?video_base=http://HOST:8889&video_mode=iframe";
        stage.appendChild(ph);
      } else if (videoMode === "iframe") {
        const iframe = document.createElement("iframe");
        iframe.src = url;
        iframe.allow = "autoplay; fullscreen; camera; microphone";
        cam.mediaEl = iframe;
        stage.appendChild(iframe);
      } else {
        const video = document.createElement("video");
        cam.mediaEl = video;
        if (videoMode === "hls") setupHlsVideo(camId, cam, video, url);
        else {
          video.src = url;
          video.autoplay = true;
          video.muted = true;
          video.playsInline = true;
          video.controls = true;
        }
        stage.appendChild(video);
      }
      const canvas = document.createElement("canvas");
      cam.canvas = canvas;
      cam.ctx = canvas.getContext("2d");
      stage.appendChild(canvas);
      return stage;
    }

    function buildTile(camId, focus=false) {
      const cam = ensureCamera(camId);
      const tile = document.createElement("div");
      tile.className = focus ? "tile focus" : "tile";
      tile.dataset.camera = camId;
      const stage = buildMedia(camId, cam);
      const top = document.createElement("div");
      top.className = "tile-top";
      top.innerHTML = `<span class="label">${camId}</span><span class="badge off" data-role="badge">no metadata</span>`;
      const bottom = document.createElement("div");
      bottom.className = "tile-bottom";
      bottom.innerHTML = `<span data-role="frame">frame -</span><span data-role="count">0 bbox</span>`;
      tile.appendChild(stage);
      tile.appendChild(top);
      tile.appendChild(bottom);
      tile.addEventListener("click", () => {
        if (state.focusCam !== camId) {
          state.focusCam = camId;
          const url = new URL(window.location.href);
          url.searchParams.set("focus", camId);
          window.history.replaceState(null, "", url.toString());
          renderLayout();
        }
      });
      cam.tile = tile;
      return tile;
    }

    function renderLayout() {
      cameraArea.innerHTML = "";
      const cams = Array.from(state.cameras.keys()).sort();
      state.layoutCameraKey = cams.join(",");
      document.getElementById("cameraCount").textContent = cams.length;
      if (cams.length > 1 && state.focusCam && state.cameras.has(state.focusCam)) {
        const layout = document.createElement("div");
        layout.className = "focus-layout";
        layout.appendChild(buildTile(state.focusCam, true));
        const thumbs = document.createElement("div");
        thumbs.className = "thumbs";
        for (const camId of cams) {
          if (camId !== state.focusCam) thumbs.appendChild(buildTile(camId, false));
        }
        layout.appendChild(thumbs);
        cameraArea.appendChild(layout);
      } else {
        const grid = document.createElement("div");
        grid.className = "grid-layout";
        for (const camId of cams) grid.appendChild(buildTile(camId, false));
        cameraArea.appendChild(grid);
      }
      for (const camId of cams) drawOverlay(camId);
      updateBadges();
    }

    function resizeCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(rect.width));
      const h = Math.max(1, Math.round(rect.height));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }

    function drawOverlay(camId) {
      const cam = ensureCamera(camId);
      const canvas = cam.canvas;
      const ctx = cam.ctx;
      if (!canvas || !ctx) return;
      resizeCanvas(canvas);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const isFocus = state.focusCam === camId || state.cameras.size <= 1;
      if (!isFocus) {
        const tile = cam.tile;
        if (tile) {
          const frameEl = tile.querySelector('[data-role="frame"]');
          const countEl = tile.querySelector('[data-role="count"]');
          if (frameEl) frameEl.textContent = `frame ${cam.frame}`;
          if (countEl) countEl.textContent = `${cam.detections.length} bbox`;
        }
        return;
      }
      const scale = Math.min(canvas.width / sourceW, canvas.height / sourceH);
      const offX = (canvas.width - sourceW * scale) / 2;
      const offY = (canvas.height - sourceH * scale) / 2;
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 420));
      ctx.font = `${Math.max(11, Math.round(canvas.width / 62))}px Segoe UI, Arial`;
      const zones = state.zones.get(camId) || [];
      zones.forEach((zone, index) => {
        const pts = zone.points_px || [];
        if (pts.length < 3) return;
        const color = zoneColor(index);
        ctx.save();
        ctx.beginPath();
        pts.forEach((pt, i) => {
          const x = offX + pt[0] * scale;
          const y = offY + pt[1] * scale;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.setLineDash([8, 6]);
        ctx.lineWidth = Math.max(2, Math.round(canvas.width / 360));
        ctx.strokeStyle = color;
        ctx.fillStyle = `${color}2a`;
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        const labelPt = pts[0];
        const lx = offX + labelPt[0] * scale;
        const ly = offY + labelPt[1] * scale;
        const label = zone.zone_id || zone.name || "zone";
        const tw = ctx.measureText(label).width + 10;
        ctx.fillStyle = color;
        ctx.fillRect(lx, Math.max(0, ly - 22), tw, 20);
        ctx.fillStyle = "#111820";
        ctx.fillText(label, lx + 5, Math.max(15, ly - 7));
        ctx.restore();
      });
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 420));
      for (const det of cam.detections) {
        if (!det.bbox_px || det.bbox_px.length < 4) continue;
        const [x1, y1, x2, y2] = det.bbox_px;
        const x = offX + x1 * scale;
        const y = offY + y1 * scale;
        const w = (x2 - x1) * scale;
        const h = (y2 - y1) * scale;
        const color = classColor(det);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(x, y, w, h);
        const conf = Number(det.confidence || 0) * 100;
        const label = `${det.class_name || "obj"} #${det.global_id ?? det.track_id ?? "-"} ${conf.toFixed(0)}%`;
        const tw = ctx.measureText(label).width + 10;
        const ty = Math.max(0, y - 22);
        ctx.fillRect(x, ty, tw, 20);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + 5, ty + 15);
      }
      const tile = cam.tile;
      if (tile) {
        const frameEl = tile.querySelector('[data-role="frame"]');
        const countEl = tile.querySelector('[data-role="count"]');
        if (frameEl) frameEl.textContent = `frame ${cam.frame}`;
        if (countEl) countEl.textContent = `${cam.detections.length} bbox`;
      }
    }

    function updateBadges() {
      const rows = [];
      for (const [camId, cam] of state.cameras) {
        const freshness = cameraFreshness(cam);
        const ageS = cam.lastMetadataMs ? ((Date.now() - cam.lastMetadataMs) / 1000).toFixed(1) : "-";
        const tile = cam.tile;
        if (tile) {
          const badge = tile.querySelector('[data-role="badge"]');
          if (badge) {
            badge.className = `badge ${freshness}`;
            badge.textContent = freshness === "live" ? "metadata live" : freshness === "stale" ? "stale" : "no metadata";
          }
        }
        updateVideoLatency(cam);
        const videoLag = cam.videoLatencyMs ? `${(cam.videoLatencyMs / 1000).toFixed(1)}s video` : "";
        rows.push(`<div class="status-row"><span><i class="dot ${freshness}"></i>${camId}</span><span>${cam.detections.length} bbox ${videoLag}</span><span>${ageS}s</span></div>`);
      }
      cameraStatus.innerHTML = rows.join("") || '<div class="empty">No cameras configured.</div>';
      document.getElementById("latency").textContent = `${state.lastLatencyMs.toFixed(1)} ms`;
    }

    function addAlertRow(event) {
      const key = event.alert_id || event.event_id || `${event.alert_type}-${event.camera}-${event.received_epoch_ms}`;
      if (state.alerts.has(key)) return;
      state.alerts.set(key, event);
      state.totalAlerts += 1;
      document.getElementById("total").textContent = state.totalAlerts;
      emptyAlerts.style.display = "none";
      const row = document.createElement("tr");
      row.innerHTML = `<td>${new Date(event.received_epoch_ms || Date.now()).toLocaleTimeString()}</td>
        <td class="${event.alert_level}">${event.alert_level || ""}</td>
        <td>${event.alert_type || ""}</td>
        <td>${event.camera || ""}</td>`;
      eventsBody.prepend(row);
      while (eventsBody.children.length > 80) eventsBody.removeChild(eventsBody.lastChild);
    }

    function metadataTimeMs(metadata) {
      return Number(metadata.received_epoch_ms || metadata.created_epoch_ms || (metadata.created_epoch_s || 0) * 1000 || Date.now());
    }

    function metadataSyncTimeMs(metadata) {
      const times = [];
      for (const det of metadata.detections || []) {
        if (det.timestamp) times.push(Number(det.timestamp) * 1000);
      }
      for (const alert of metadata.alerts || []) {
        if (alert.timestamp) times.push(Number(alert.timestamp) * 1000);
      }
      const valid = times.filter((value) => Number.isFinite(value) && value > 0);
      if (valid.length > 0) return Math.max(...valid);
      return Number(metadata.created_epoch_ms || (metadata.created_epoch_s || 0) * 1000 || Date.now());
    }

    function metadataKey(metadata) {
      return `${metadata.run_label || ""}:${metadata.frame ?? ""}:${metadataTimeMs(metadata)}`;
    }

    function receiveMetadata(metadata) {
      if (!metadata || !metadata.schema) return;
      state.metadataBuffer.push(metadata);
      if (state.metadataBuffer.length > 600) state.metadataBuffer.splice(0, state.metadataBuffer.length - 600);
      if (effectiveOverlayDelayMs() === 0) applyMetadata(metadata);
    }

    function applyBufferedMetadata() {
      const delayMs = effectiveOverlayDelayMs();
      if (delayMs <= 0 || state.metadataBuffer.length === 0) return;
      const targetTime = Date.now() - delayMs;
      let chosen = null;
      for (const metadata of state.metadataBuffer) {
        if (metadataSyncTimeMs(metadata) <= targetTime) chosen = metadata;
        else break;
      }
      if (!chosen) return;
      const key = metadataKey(chosen);
      if (key === state.lastAppliedMetadataKey) return;
      applyMetadata(chosen);
      state.metadataBuffer = state.metadataBuffer.filter((metadata) => metadataSyncTimeMs(metadata) >= targetTime - 5000);
    }

    function applyMetadata(metadata) {
      state.lastAppliedMetadataKey = metadataKey(metadata);
      const grouped = new Map();
      let needsLayout = false;
      for (const det of metadata.detections || []) {
        const camId = det.camera_id || "unknown";
        if (!grouped.has(camId)) grouped.set(camId, []);
        grouped.get(camId).push(det);
        if (!state.cameras.has(camId)) needsLayout = true;
        ensureCamera(camId);
      }
      for (const [camId, detections] of grouped) {
        const cam = ensureCamera(camId);
        cam.detections = detections;
        cam.frame = metadata.frame ?? "-";
        cam.lastMetadataMs = Number(metadata.received_epoch_ms || metadata.created_epoch_ms || Date.now());
        cam.deliveryLatencyMs = Number(metadata.delivery_latency_ms || 0);
      }
      state.lastLatencyMs = Number(metadata.delivery_latency_ms || 0);
      const detTotal = Array.from(state.cameras.values()).reduce((acc, cam) => acc + cam.detections.length, 0);
      document.getElementById("detCount").textContent = detTotal;
      for (const alert of metadata.alerts || []) {
        addAlertRow({
          alert_id: alert.alert_id,
          alert_level: alert.alert_level,
          alert_type: alert.alert_type,
          camera: (alert.cameras || []).join("+"),
          received_epoch_ms: metadata.received_epoch_ms,
        });
      }
      const cameraKey = Array.from(state.cameras.keys()).sort().join(",");
      if (needsLayout || cameraKey !== state.layoutCameraKey) {
        renderLayout();
      } else {
        for (const camId of state.cameras.keys()) drawOverlay(camId);
        updateBadges();
      }
    }

    window.addEventListener("resize", () => {
      for (const camId of state.cameras.keys()) drawOverlay(camId);
    });
    setInterval(updateBadges, 1000);
    setInterval(applyBufferedMetadata, 120);

    const alertSource = new EventSource("/events");
    alertSource.onmessage = (message) => addAlertRow(JSON.parse(message.data));

    const metadataSource = new EventSource("/metadata-events");
    metadataSource.onmessage = (message) => receiveMetadata(JSON.parse(message.data));
    metadataSource.onerror = () => fetchLatestMetadata();

    async function fetchLatestMetadata() {
      try {
        const response = await fetch("/metadata/latest", {cache: "no-store"});
        if (!response.ok) return;
        const metadata = await response.json();
        if (metadata && metadata.schema) receiveMetadata(metadata);
      } catch (_) {}
    }

    setInterval(fetchLatestMetadata, 1000);
    fetchLatestMetadata();

    fetch("/zones.json")
      .then((response) => response.ok ? response.json() : {})
      .then((zonesByCamera) => {
        for (const [camId, zones] of Object.entries(zonesByCamera || {})) {
          if (state.cameras.has(camId)) {
            state.zones.set(camId, zones || []);
          }
        }
        renderLayout();
      })
      .catch(() => {});

    renderLayout();
  </script>
</body>
</html>
"""


def _invert_3x3(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(det) < 1e-12:
        raise ValueError("Singular homography matrix")
    inv_det = 1.0 / det
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def _homography_for(config: dict, camera_id: str) -> list[list[float]] | None:
    homographies = config.get("homographie", {})
    matrix = homographies.get(f"{camera_id}_matrix")
    if matrix:
        return matrix
    nested = homographies.get(camera_id)
    if isinstance(nested, dict):
        return nested.get("matrix") or nested.get("matrix_substream")
    return None


def _project_meter_to_pixel(inv_h: list[list[float]], x_m: float, y_m: float) -> list[float] | None:
    u = inv_h[0][0] * x_m + inv_h[0][1] * y_m + inv_h[0][2]
    v = inv_h[1][0] * x_m + inv_h[1][1] * y_m + inv_h[1][2]
    w = inv_h[2][0] * x_m + inv_h[2][1] * y_m + inv_h[2][2]
    if abs(w) < 1e-9:
        return None
    return [round(u / w, 2), round(v / w, 2)]


def load_zones_by_camera(config_path: Path | None) -> dict[str, list[dict]]:
    if not config_path:
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    zones_by_camera: dict[str, list[dict]] = {}
    inverse_by_camera: dict[str, list[list[float]]] = {}
    for zone_id, zone_data in (config.get("zones_interdites") or {}).items():
        coords_m = zone_data.get("coordonnees_metres") or []
        cameras = zone_data.get("cameras_concernees") or []
        if len(coords_m) < 3 or not cameras:
            continue
        for camera_id in cameras:
            if camera_id not in inverse_by_camera:
                matrix = _homography_for(config, camera_id)
                if not matrix:
                    continue
                try:
                    inverse_by_camera[camera_id] = _invert_3x3(matrix)
                except ValueError:
                    continue
            points_px = []
            for point in coords_m:
                projected = _project_meter_to_pixel(inverse_by_camera[camera_id], float(point[0]), float(point[1]))
                if projected is not None:
                    points_px.append(projected)
            if len(points_px) >= 3:
                zones_by_camera.setdefault(camera_id, []).append(
                    {
                        "zone_id": zone_id,
                        "name": zone_data.get("nom", zone_id),
                        "points_px": points_px,
                        "points_m": coords_m,
                    }
                )
    return zones_by_camera


def broadcast_alert(alert: dict) -> None:
    global ALERT_POSTS
    now_ms = time.time() * 1000.0
    created_ms = float(alert.get("created_epoch_ms", now_ms))
    alert["received_epoch_ms"] = now_ms
    alert["delivery_latency_ms"] = max(0.0, now_ms - created_ms)
    with STATS_LOCK:
        ALERT_POSTS += 1
    with CLIENTS_LOCK:
        clients = list(CLIENTS)
    for client in clients:
        client.put(alert)


def broadcast_metadata(metadata: dict) -> None:
    global LAST_METADATA, METADATA_POSTS
    now_ms = time.time() * 1000.0
    created_ms = float(metadata.get("created_epoch_ms", now_ms))
    metadata["received_epoch_ms"] = now_ms
    metadata["delivery_latency_ms"] = max(0.0, now_ms - created_ms)
    with LAST_METADATA_LOCK:
        LAST_METADATA = metadata
    with STATS_LOCK:
        METADATA_POSTS += 1
    with METADATA_CLIENTS_LOCK:
        clients = list(METADATA_CLIENTS)
    for client in clients:
        client.put(metadata)
    for alert in metadata.get("alerts", []):
        broadcast_alert(
            {
                "event_id": alert.get("alert_id"),
                "alert_level": alert.get("alert_level", "confirmed"),
                "alert_type": alert.get("alert_type", ""),
                "camera": "+".join(alert.get("cameras", [])),
                "created_epoch_ms": metadata["created_epoch_ms"],
            }
        )


def make_simulated_alert(event_id: int) -> dict:
    return {
        "event_id": event_id,
        "alert_level": "confirmed" if event_id % 5 == 0 else "weak",
        "alert_type": "forbidden_object" if event_id % 3 else "zone_violation_person",
        "camera": random.choice(["cam_02", "cam_03", "cam_05", "cam_07"]),
        "created_epoch_ms": time.time() * 1000.0,
    }


def make_simulated_metadata(frame_id: int) -> dict:
    now_s = time.time()
    cameras = ["cam_02", "cam_03", "cam_05", "cam_07"]
    detections = []
    alerts = []
    for idx, cam_id in enumerate(cameras):
        x = 70 + ((frame_id * 9 + idx * 120) % 520)
        y = 80 + ((frame_id * 5 + idx * 55) % 310)
        person_gid = idx + 1
        detections.append(
            {
                "camera_id": cam_id,
                "track_id": person_gid,
                "global_id": person_gid,
                "class_id": 11,
                "class_name": "personne",
                "confidence": 0.72,
                "bbox_px": [x, y, x + 92, y + 190],
                "foot_point_px": [x + 46, y + 190],
                "position_m": [6.5 + idx * 0.35, 4.2 + idx * 0.25],
                "zones": ["zone_1"] if idx == 0 and frame_id % 8 == 0 else [],
                "timestamp": now_s,
            }
        )
        if idx % 2 == 0:
            ox = 210 + ((frame_id * 7 + idx * 140) % 380)
            oy = 290 + ((frame_id * 3 + idx * 45) % 160)
            detections.append(
                {
                    "camera_id": cam_id,
                    "track_id": 20 + idx,
                    "global_id": 20 + idx,
                    "class_id": 5,
                    "class_name": "bouteille",
                    "confidence": 0.61,
                    "bbox_px": [ox, oy, ox + 38, oy + 62],
                    "foot_point_px": [ox + 19, oy + 62],
                    "position_m": [7.0 + idx * 0.2, 4.8],
                    "zones": [],
                    "timestamp": now_s,
                }
            )

    if frame_id % 8 == 0:
        alerts.append(
            {
                "alert_id": f"sim-{frame_id}",
                "alert_type": "zone_violation_person",
                "alert_level": "confirmed",
                "global_id": 1,
                "zone_id": "zone_1",
                "class_name": "person",
                "position_m": [6.5, 4.2],
                "cameras": ["cam_02"],
                "confidence": 0.72,
                "timestamp": now_s,
            }
        )

    return {
        "schema": "benchmarkingai.phase3.metadata.v1",
        "created_epoch_ms": now_s * 1000.0,
        "created_epoch_s": now_s,
        "frame": frame_id,
        "run_label": "simulated_dashboard",
        "model_version": "V4",
        "model": "yolov8s",
        "format": "fp32_engine",
        "detections": detections,
        "alerts": alerts,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return

        if parsed.path == "/zones.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(ZONES_BY_CAMERA).encode("utf-8"))
            return

        if parsed.path == "/metadata/latest":
            with LAST_METADATA_LOCK:
                metadata = dict(LAST_METADATA or {})
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps(metadata).encode("utf-8"))
            return

        if parsed.path == "/debug.json":
            with CLIENTS_LOCK:
                alert_clients = len(CLIENTS)
            with METADATA_CLIENTS_LOCK:
                metadata_clients = len(METADATA_CLIENTS)
            with LAST_METADATA_LOCK:
                metadata = LAST_METADATA or {}
            with STATS_LOCK:
                metadata_posts = METADATA_POSTS
                alert_posts = ALERT_POSTS
            debug = {
                "alert_sse_clients": alert_clients,
                "metadata_sse_clients": metadata_clients,
                "metadata_posts": metadata_posts,
                "alert_posts": alert_posts,
                "has_last_metadata": bool(metadata),
                "last_frame": metadata.get("frame"),
                "last_detections": len(metadata.get("detections") or []),
                "last_alerts": len(metadata.get("alerts") or []),
                "last_cameras": sorted(
                    {
                        str(det.get("camera_id"))
                        for det in metadata.get("detections", [])
                        if det.get("camera_id")
                    }
                ),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps(debug, indent=2).encode("utf-8"))
            return

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
                    alert = channel.get(timeout=15.0)
                    payload = json.dumps(alert)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError, queue.Empty):
                pass
            finally:
                with CLIENTS_LOCK:
                    if channel in CLIENTS:
                        CLIENTS.remove(channel)
            return

        if parsed.path == "/metadata-events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            channel: queue.Queue[dict] = queue.Queue()
            with METADATA_CLIENTS_LOCK:
                METADATA_CLIENTS.append(channel)
            with LAST_METADATA_LOCK:
                latest_metadata = LAST_METADATA
            if latest_metadata:
                channel.put(latest_metadata)
            try:
                while True:
                    metadata = channel.get(timeout=15.0)
                    payload = json.dumps(metadata)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError, queue.Empty):
                pass
            finally:
                with METADATA_CLIENTS_LOCK:
                    if channel in METADATA_CLIENTS:
                        METADATA_CLIENTS.remove(channel)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/alerts", "/metadata"}:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if parsed.path == "/metadata":
            broadcast_metadata(payload)
        else:
            broadcast_alert(payload)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return


def start_simulator(rate_hz: float) -> None:
    def run() -> None:
        frame_id = 0
        spacing_s = 1.0 / rate_hz if rate_hz > 0 else 1.0
        while True:
            broadcast_metadata(make_simulated_metadata(frame_id))
            frame_id += 1
            time.sleep(spacing_s)

    threading.Thread(target=run, daemon=True).start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Phase 4 alert dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=2.0)
    parser.add_argument(
        "--zones-config",
        type=Path,
        default=None,
        help="Optional Phase 3 YAML config used to draw forbidden zones on the video overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global ZONES_BY_CAMERA
    if args.zones_config:
        ZONES_BY_CAMERA = load_zones_by_camera(args.zones_config)
        total_zones = sum(len(zones) for zones in ZONES_BY_CAMERA.values())
        print(f"[INFO] Zones overlay loaded: {total_zones} camera-zone projection(s) from {args.zones_config}")
    if args.simulate:
        start_simulator(args.rate_hz)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[INFO] Dashboard: http://{args.host}:{args.port}")
    print("[INFO] POST alerts to /alerts or use --simulate")
    server.serve_forever()


if __name__ == "__main__":
    main()

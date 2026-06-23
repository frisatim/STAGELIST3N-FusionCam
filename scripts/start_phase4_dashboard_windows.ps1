param(
    [string]$Cameras = "cam_02,cam_07",
    [int]$DurationMin = 5,
    [string]$Config = "Phase_3_Fusion_MultiCam\config_real_zones.yaml",
    [string]$Python = "python",
    [string]$Model = "yolov8s",
    [string]$Format = "fp32_engine",
    [string]$Device = "cuda:0",
    [string]$CaptureBackend = "opencv",
    [int]$ObjectMinCameraVotes = 2,
    [string]$DashboardHost = "127.0.0.1",
    [int]$DashboardPort = 8765,
    [switch]$WithVideo,
    [switch]$StartMediaMTX,
    [switch]$StartRelays,
    [switch]$TranscodeRelays,
    [string]$VideoBase = "http://127.0.0.1:8889",
    [switch]$NoWeakObjectAlerts,
    [switch]$NoIA,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

function Quote-Arg([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (!(Test-Path $Config)) {
    throw "Config introuvable: $Config"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$cameraSlug = $Cameras.Replace(",", "_").Replace(" ", "")
$runName = "dashboard_${cameraSlug}_${timestamp}"
$metadataJsonl = "Phase_3_Fusion_MultiCam\reports\${runName}_metadata.jsonl"
$latencyTrace = "Phase_3_Fusion_MultiCam\reports\${runName}_latency_trace.csv"
$outDir = "Phase_3_Fusion_MultiCam\reports\${runName}"

if ($StartMediaMTX) {
    if (Test-Path "STAGELIST3N-FusionCam\docker-compose.yml") {
        $mediaCmd = "cd " + (Quote-Arg (Join-Path $RepoRoot "STAGELIST3N-FusionCam")) + "; docker-compose up mediamtx"
        Start-Process powershell -ArgumentList @("-NoExit", "-Command", $mediaCmd) | Out-Null
        Start-Sleep -Seconds 3
    } elseif (Test-Path "docker-compose.yml") {
        $mediaCmd = "cd " + (Quote-Arg $RepoRoot) + "; docker-compose up mediamtx"
        Start-Process powershell -ArgumentList @("-NoExit", "-Command", $mediaCmd) | Out-Null
        Start-Sleep -Seconds 3
    } else {
        Write-Warning "docker-compose.yml introuvable: MediaMTX ne sera pas lance automatiquement."
    }
}

if ($StartRelays) {
    $extractScript = @"
import sys
import yaml

config_path = sys.argv[1]
cameras = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
for camera_id in cameras:
    camera = (config.get("cameras") or {}).get(camera_id) or {}
    rtsp_url = camera.get("rtsp_url") or ""
    if rtsp_url:
        print(f"{camera_id}\t{rtsp_url}")
"@
    $tmpExtract = New-TemporaryFile
    Set-Content -Path $tmpExtract -Value $extractScript -Encoding UTF8
    try {
        $relayLines = & $Python $tmpExtract $Config $Cameras
    } finally {
        Remove-Item $tmpExtract -Force -ErrorAction SilentlyContinue
    }

    foreach ($line in $relayLines) {
        if (!$line.Trim()) { continue }
        $parts = $line -split "`t", 2
        if ($parts.Count -lt 2) { continue }
        $cameraId = $parts[0]
        $rtspUrl = $parts[1]
        $relayUrl = "rtsp://127.0.0.1:8554/$cameraId"
        if ($TranscodeRelays) {
            $relayCmd = "ffmpeg -re -stream_loop -1 -i " + (Quote-Arg $rtspUrl) + " -an -vf scale=704:576,fps=25,format=yuv420p -c:v libx264 -preset veryfast -tune zerolatency -g 25 -keyint_min 25 -bf 0 -f rtsp -rtsp_transport tcp " + $relayUrl
        } else {
            $relayCmd = "ffmpeg -re -stream_loop -1 -rtsp_transport tcp -i " + (Quote-Arg $rtspUrl) + " -an -c:v copy -f rtsp -rtsp_transport tcp " + $relayUrl
        }
        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-Command",
            "cd " + (Quote-Arg $RepoRoot) + "; " + $relayCmd
        ) | Out-Null
    }
    Start-Sleep -Seconds 3
}

$dashboardCmd = @(
    $Python,
    "Phase_4_Network_Latency\alert_dashboard.py",
    "--host", $DashboardHost,
    "--port", $DashboardPort,
    "--zones-config", (Quote-Arg $Config)
) -join " "

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd " + (Quote-Arg $RepoRoot) + "; " + $dashboardCmd
) | Out-Null

Start-Sleep -Seconds 2

$dashboardUrl = "http://${DashboardHost}:${DashboardPort}/?cameras=$([uri]::EscapeDataString($Cameras))"
if ($WithVideo) {
    $dashboardUrl += "&video_base=$([uri]::EscapeDataString($VideoBase))&video_mode=iframe"
}

if (!$NoBrowser) {
    Start-Process $dashboardUrl | Out-Null
}

if (!$NoIA) {
    $iaArgs = @(
        $Python,
        "Phase_3_Fusion_MultiCam\run_live_campaign.py",
        "--config", (Quote-Arg $Config),
        "--versions", "V4",
        "--models", $Model,
        "--formats", $Format,
        "--cameras", $Cameras,
        "--duration-min", $DurationMin,
        "--device", $Device,
        "--object-min-camera-votes", $ObjectMinCameraVotes,
        "--capture-backend", $CaptureBackend,
        "--no-display",
        "--no-record-video",
        "--metadata-http-url", "http://${DashboardHost}:${DashboardPort}/metadata",
        "--metadata-jsonl", (Quote-Arg $metadataJsonl),
        "--latency-trace-csv", (Quote-Arg $latencyTrace),
        "--out-dir", (Quote-Arg $outDir)
    )
    if ($NoWeakObjectAlerts) {
        $iaArgs += "--no-weak-object-alerts"
    }
    $iaCmd = $iaArgs -join " "
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd " + (Quote-Arg $RepoRoot) + "; " + $iaCmd
    ) | Out-Null
}

Write-Host "[INFO] Dashboard URL: $dashboardUrl"
Write-Host "[INFO] Metadata JSONL: $metadataJsonl"
Write-Host "[INFO] Latency trace: $latencyTrace"
Write-Host "[INFO] Out dir: $outDir"
if ($WithVideo) {
    Write-Host "[INFO] Video mode active. Verifie que chaque flux existe: $VideoBase/cam_XX/"
} else {
    Write-Host "[INFO] Video mode inactive. Le dashboard affiche zones + bbox sur fond sombre."
}

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
    [int]$DashboardPort = 8776,
    [string]$VideoBase = "http://127.0.0.1:8889",
    [string]$RtspBase = "rtsp://127.0.0.1:8554",
    [switch]$NoIA,
    [switch]$NoBrowser,
    [switch]$NoTranscode,
    [switch]$NoWeakObjectAlerts,
    [int]$RelayReadyTimeoutSec = 45
)

$ErrorActionPreference = "Stop"

function New-ProcessLogPrefix([string]$LogDir, [string]$Name) {
    return Join-Path $LogDir $Name
}

function Join-ProcessArguments {
    param([string[]]$Arguments)

    $escaped = foreach ($arg in $Arguments) {
        if ($null -eq $arg) {
            '""'
            continue
        }
        $value = [string]$arg
        if ($value -notmatch '[\s"]') {
            $value
        } else {
            $escapedValue = $value -replace '\\', '\\'
            $escapedValue = $escapedValue -replace '"', '\"'
            '"' + $escapedValue + '"'
        }
    }
    return ($escaped -join " ")
}

function Start-LoggedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory,
        [string]$LogPrefix
    )

    $stdoutPath = "${LogPrefix}.out.log"
    $stderrPath = "${LogPrefix}.err.log"

    $stdoutWriter = [System.IO.StreamWriter]::new($stdoutPath, $false, [System.Text.Encoding]::UTF8)
    $stderrWriter = [System.IO.StreamWriter]::new($stderrPath, $false, [System.Text.Encoding]::UTF8)

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.Arguments = Join-ProcessArguments -Arguments $Arguments

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    $proc.add_OutputDataReceived({
        if ($EventArgs.Data -ne $null) {
            $stdoutWriter.WriteLine($EventArgs.Data)
            $stdoutWriter.Flush()
        }
    })
    $proc.add_ErrorDataReceived({
        if ($EventArgs.Data -ne $null) {
            $stderrWriter.WriteLine($EventArgs.Data)
            $stderrWriter.Flush()
        }
    })

    if (-not $proc.Start()) {
        throw "Impossible de demarrer $Name"
    }
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()

    [pscustomobject]@{
        Name = $Name
        Process = $proc
        Stdout = $stdoutPath
        Stderr = $stderrPath
        StdoutWriter = $stdoutWriter
        StderrWriter = $stderrWriter
    }
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath a echoue avec code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Test-RtspStream {
    param([string]$Url)
    $probeArgs = @(
        "-rtsp_transport", "tcp",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
        "-of", "default=nw=1",
        $Url
    )
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & ffprobe @probeArgs 2>&1
        $exitCode = $LASTEXITCODE
    } catch {
        $output = @($_.Exception.Message)
        $exitCode = 1
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    return [pscustomobject]@{
        Ok = ($exitCode -eq 0)
        Output = ($output -join "`n")
    }
}

function Wait-RtspStream {
    param(
        [string]$Url,
        [int]$TimeoutSec,
        [System.Diagnostics.Process]$RelayProcess
    )
    for ($i = 1; $i -le $TimeoutSec; $i++) {
        if ($RelayProcess.HasExited) {
            return [pscustomobject]@{ Ok = $false; Reason = "relay_exited"; Output = "" }
        }
        if (($i -eq 1) -or ($i % 5 -eq 0)) {
            Write-Host "[INFO] Waiting relay $Url ($i/$TimeoutSec sec)..."
        }
        $probe = Test-RtspStream -Url $Url
        if ($probe.Ok) {
            Write-Host "[INFO] Relay probe OK for $Url"
            return [pscustomobject]@{ Ok = $true; Reason = "ok"; Output = $probe.Output }
        }
        if (($i -eq 1) -or ($i % 10 -eq 0)) {
            $shortOutput = ($probe.Output -split "`n" | Select-Object -First 1)
            if ($shortOutput) {
                Write-Host "[INFO] Relay probe not ready: $shortOutput"
            }
        }
        Start-Sleep -Seconds 1
    }
    [pscustomobject]@{ Ok = $false; Reason = "timeout"; Output = "" }
}

function Get-CameraRtspUrls {
    param(
        [string]$PythonExe,
        [string]$ConfigPath,
        [string]$CameraCsv
    )
    $script = @"
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
    $tmp = New-TemporaryFile
    Set-Content -Path $tmp -Value $script -Encoding UTF8
    try {
        $lines = & $PythonExe $tmp $ConfigPath $CameraCsv
        if ($LASTEXITCODE -ne 0) {
            throw "Extraction des URLs RTSP impossible"
        }
        foreach ($line in $lines) {
            if (-not $line.Trim()) {
                continue
            }
            $parts = $line -split "`t", 2
            if ($parts.Count -ne 2) {
                throw "Ligne RTSP invalide: $line"
            }
            [pscustomobject]@{
                camera_id = $parts[0]
                rtsp_url = $parts[1]
            }
        }
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path $Config)) {
    throw "Config introuvable: $Config"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$cameraSlug = $Cameras.Replace(",", "_").Replace(" ", "")
$runName = "phase4_full_dashboard_${cameraSlug}_${timestamp}"
$runDir = Join-Path $Root "Phase_3_Fusion_MultiCam\reports\$runName"
$logDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$metadataJsonl = Join-Path $runDir "metadata.jsonl"
$latencyTrace = Join-Path $runDir "latency_trace.csv"
$iaOutDir = Join-Path $runDir "ia_run"

Write-Host "[INFO] Root     : $Root"
Write-Host "[INFO] Run dir  : $runDir"
Write-Host "[INFO] Cameras  : $Cameras"
Write-Host "[INFO] Dashboard: http://${DashboardHost}:${DashboardPort}/"
Write-Host "[INFO] VideoBase: $VideoBase"

$composeFile = Join-Path $Root "STAGELIST3N-FusionCam\docker-compose.yml"
if (-not (Test-Path $composeFile)) {
    $composeFile = Join-Path $Root "docker-compose.yml"
}
if (-not (Test-Path $composeFile)) {
    throw "docker-compose.yml introuvable pour MediaMTX"
}

Write-Host "[INFO] Starting MediaMTX..."
Invoke-Checked -FilePath "docker-compose" -Arguments @("-f", $composeFile, "up", "-d", "mediamtx") -WorkingDirectory $Root

$cameraEntries = @()
foreach ($entry in (Get-CameraRtspUrls -PythonExe $Python -ConfigPath $Config -CameraCsv $Cameras)) {
    $cameraEntries += $entry
}
if ($cameraEntries.Count -eq 0) {
    throw "Aucune URL RTSP trouvee dans $Config pour $Cameras"
}
Write-Host "[INFO] Camera entries extracted: $($cameraEntries.Count) -> $(@($cameraEntries | ForEach-Object { $_.camera_id }) -join ', ')"

$processes = @()

foreach ($entry in $cameraEntries) {
    $cameraId = [string]$entry.camera_id
    $inputUrl = [string]$entry.rtsp_url
    $relayUrl = "$RtspBase/$cameraId"

    Write-Host "[INFO] Testing source $cameraId..."
    $sourceProbe = Test-RtspStream -Url $inputUrl
    if (-not $sourceProbe.Ok) {
        Write-Host "[ERROR] Source RTSP inaccessible pour $cameraId"
        Write-Host $sourceProbe.Output
        throw "Source RTSP inaccessible: $cameraId"
    }
    Write-Host "[OK] Source $cameraId reachable"

    if ($NoTranscode) {
        $ffArgs = @(
            "-hide_banner",
            "-nostdin",
            "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", $inputUrl,
            "-an",
            "-c:v", "copy",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            $relayUrl
        )
    } else {
        $ffArgs = @(
            "-hide_banner",
            "-nostdin",
            "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", $inputUrl,
            "-an",
            "-vf", "scale=704:576,fps=25,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-g", "25",
            "-keyint_min", "25",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            $relayUrl
        )
    }

    $logPrefix = New-ProcessLogPrefix -LogDir $logDir -Name "relay_${cameraId}"
    Write-Host "[INFO] Publishing $cameraId -> $relayUrl"
    $relayProc = Start-LoggedProcess -Name "relay_$cameraId" -FilePath "ffmpeg" -Arguments $ffArgs -WorkingDirectory $Root -LogPrefix $logPrefix
    $processes += $relayProc

    $ready = Wait-RtspStream -Url $relayUrl -TimeoutSec $RelayReadyTimeoutSec -RelayProcess $relayProc.Process
    if (-not $ready.Ok) {
        Write-Host "[ERROR] Relay non disponible pour $cameraId ($($ready.Reason))"
        Write-Host "[INFO] Log erreur: $($relayProc.Stderr)"
        if (Test-Path $relayProc.Stderr) {
            Get-Content $relayProc.Stderr -Tail 30
        }
        throw "Relay MediaMTX non disponible: $cameraId"
    }
    Write-Host "[OK] Relay $cameraId ready"
    Write-Host $ready.Output
}

$dashboardArgs = @(
    "Phase_4_Network_Latency\alert_dashboard.py",
    "--host", $DashboardHost,
    "--port", [string]$DashboardPort,
    "--zones-config", $Config
)
$dashboardLog = New-ProcessLogPrefix -LogDir $logDir -Name "dashboard"
Write-Host "[INFO] Starting dashboard..."
$dashboardProc = Start-LoggedProcess -Name "dashboard" -FilePath $Python -Arguments $dashboardArgs -WorkingDirectory $Root -LogPrefix $dashboardLog
$processes += $dashboardProc
Start-Sleep -Seconds 2

$dashboardUrl = "http://${DashboardHost}:${DashboardPort}/?cameras=$([uri]::EscapeDataString($Cameras))&video_base=$([uri]::EscapeDataString($VideoBase))&video_mode=iframe&source_w=704&source_h=576&overlay_delay_ms=5000&v=$timestamp"
Write-Host "[INFO] Dashboard URL:"
Write-Host $dashboardUrl
if (-not $NoBrowser) {
    Start-Process $dashboardUrl | Out-Null
}

if (-not $NoIA) {
    $iaArgs = @(
        "Phase_3_Fusion_MultiCam\run_live_campaign.py",
        "--config", $Config,
        "--versions", "V4",
        "--models", $Model,
        "--formats", $Format,
        "--cameras", $Cameras,
        "--duration-min", [string]$DurationMin,
        "--device", $Device,
        "--object-min-camera-votes", [string]$ObjectMinCameraVotes,
        "--capture-backend", $CaptureBackend,
        "--no-display",
        "--no-record-video",
        "--metadata-http-url", "http://${DashboardHost}:${DashboardPort}/metadata",
        "--metadata-jsonl", $metadataJsonl,
        "--latency-trace-csv", $latencyTrace,
        "--out-dir", $iaOutDir
    )
    if ($NoWeakObjectAlerts) {
        $iaArgs += "--no-weak-object-alerts"
    }
    $iaLog = New-ProcessLogPrefix -LogDir $logDir -Name "ia"
    Write-Host "[INFO] Starting IA metadata publisher..."
    $iaProc = Start-LoggedProcess -Name "ia" -FilePath $Python -Arguments $iaArgs -WorkingDirectory $Root -LogPrefix $iaLog
    $processes += $iaProc
}

Write-Host ""
Write-Host "[OK] All components started."
Write-Host "[INFO] Direct WebRTC checks:"
foreach ($entry in $cameraEntries) {
    Write-Host "       $VideoBase/$($entry.camera_id)/"
}
Write-Host "[INFO] Logs: $logDir"
Write-Host "[INFO] Keep this PowerShell open. Press Ctrl+C to stop watching; child processes may keep running."
Write-Host ""

while ($true) {
    foreach ($p in $processes) {
        if ($p.Process.HasExited) {
            Write-Host "[WARN] Process exited: $($p.Name) code=$($p.Process.ExitCode)"
            Write-Host "       stderr: $($p.Stderr)"
        }
    }
    Start-Sleep -Seconds 5
}

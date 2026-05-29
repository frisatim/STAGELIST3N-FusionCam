<#
Setup a fresh Windows machine for STAGELIST3N FusionCam.

Run from the repository root:
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  .\scripts\setup_new_pc_windows.ps1 -UseDesktopAivenv -InstallOptionalNetwork

Notes:
  - NVIDIA driver must be installed separately first.
  - TensorRT engines are hardware/version-sensitive; regenerate them on the new PC.
  - If the CUDA wheel index changes, pass -TorchIndexUrl manually.
#>

[CmdletBinding()]
param(
    [string]$VenvPath = ".venv",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$UseDesktopAivenv,
    [switch]$SkipWinget,
    [switch]$InstallTensorRT,
    [switch]$InstallOptionalNetwork
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action,
        [switch]$ContinueOnError
    )
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    try {
        & $Action
    }
    catch {
        if ($ContinueOnError) {
            Write-Warning $_.Exception.Message
        }
        else {
            throw
        }
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if ($UseDesktopAivenv) {
    $VenvPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "aivenv"
}

if (-not $SkipWinget -and (Test-Command winget)) {
    $Packages = @(
        @{ Id = "Git.Git"; Name = "Git" },
        @{ Id = "Microsoft.VisualStudioCode"; Name = "Visual Studio Code" },
        @{ Id = "Python.Python.3.12"; Name = "Python 3.12" },
        @{ Id = "Gyan.FFmpeg"; Name = "FFmpeg" },
        @{ Id = "GStreamer.GStreamer"; Name = "GStreamer runtime" },
        @{ Id = "GStreamer.GStreamer.Devel"; Name = "GStreamer development files" }
    )

    foreach ($Package in $Packages) {
        Invoke-Step "Install $($Package.Name) with winget" {
            winget install --id $Package.Id --exact --silent --accept-source-agreements --accept-package-agreements
        } -ContinueOnError
    }
}
elseif (-not $SkipWinget) {
    Write-Warning "winget not found. Install Git, VS Code, Python, FFmpeg and GStreamer manually."
}

Invoke-Step "Expose GStreamer on user PATH when installed" {
    $GstBin = "C:\gstreamer\1.0\msvc_x86_64\bin"
    if (Test-Path $GstBin) {
        $CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($CurrentPath -notlike "*$GstBin*") {
            [Environment]::SetEnvironmentVariable("Path", "$GstBin;$CurrentPath", "User")
            $env:Path = "$GstBin;$env:Path"
        }
    }
    else {
        Write-Warning "GStreamer was not found at $GstBin. Check the installer path if RTSP/GStreamer fails."
    }
} -ContinueOnError

Invoke-Step "Create Python virtual environment at $VenvPath" {
    if (-not (Test-Path $VenvPath)) {
        if (Test-Command py) {
            py -3.12 -m venv $VenvPath
        }
        elseif (Test-Command python) {
            python -m venv $VenvPath
        }
        else {
            throw "Python not found. Install Python 3.12 and rerun this script."
        }
    }
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python not found: $VenvPython"
}

Invoke-Step "Upgrade pip tooling" {
    & $VenvPython -m pip install --upgrade pip setuptools wheel
}

Invoke-Step "Install PyTorch CUDA wheels" {
    & $VenvPython -m pip install torch torchvision torchaudio --index-url $TorchIndexUrl
}

Invoke-Step "Install project Python dependencies" {
    & $VenvPython -m pip install -r requirements.txt
}

if ($InstallTensorRT) {
    Invoke-Step "Install optional TensorRT Python package" {
        & $VenvPython -m pip install tensorrt
    } -ContinueOnError
}

if ($InstallOptionalNetwork) {
    Invoke-Step "Install optional network/dashboard packages" {
        & $VenvPython -m pip install websockets paho-mqtt fastapi uvicorn
    }
}

if (Test-Command code) {
    $Extensions = @(
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.debugpy",
        "ms-vscode.powershell",
        "redhat.vscode-yaml",
        "ms-toolsai.jupyter"
    )
    foreach ($Extension in $Extensions) {
        Invoke-Step "Install VS Code extension $Extension" {
            code --install-extension $Extension
        } -ContinueOnError
    }
}
else {
    Write-Warning "VS Code CLI 'code' not found yet. Open VS Code once, then rerun if extensions are missing."
}

Invoke-Step "Validate installation" {
    & $VenvPython -c "import sys; print('python', sys.version)"
    & $VenvPython -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    & $VenvPython -c "import cv2; print('opencv', cv2.__version__, 'gstreamer' if 'GStreamer' in cv2.getBuildInformation() else 'no_gstreamer_string')"
    & $VenvPython -c "from ultralytics import YOLO; print('ultralytics OK')"
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate the environment with:"
Write-Host "  $VenvPath\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Recommended smoke test:"
Write-Host "  python -m pytest Phase_4_Network_Latency"

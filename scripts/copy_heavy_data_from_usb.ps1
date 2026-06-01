<#
Copy heavy FusionCam assets from a USB drive or shared folder into the
standard external data directory.

Typical usage from the repository root:
  .\scripts\copy_heavy_data_from_usb.ps1 -SourceRoot E:\BenchmarkingAI

Accepted source layouts:
  1. A full old BenchmarkingAI folder:
     E:\BenchmarkingAI\dataset_objets_V4
     E:\BenchmarkingAI\dataset_objets_HD
     E:\BenchmarkingAI\recordings
     E:\BenchmarkingAI\Phase_2_Baseline_MonoCam\Modelstrained

  2. An already prepared delivery folder:
     E:\STAGELIST3N-FusionCam-data\datasets
     E:\STAGELIST3N-FusionCam-data\recordings
     E:\STAGELIST3N-FusionCam-data\models

The copied directory remains outside Git by default:
  ..\STAGELIST3N-FusionCam-data
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [string]$DataDir = "../STAGELIST3N-FusionCam-data",

    [switch]$SkipDatasets,
    [switch]$SkipRecordings,
    [switch]$SkipModels,
    [switch]$SkipReports,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function Invoke-Robocopy {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$Label
    )

    if (-not (Test-Path $Source)) {
        Write-Warning "Missing source for $Label`: $Source"
        return
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
    Write-Host "Source      : $Source"
    Write-Host "Destination : $Destination"

    if ($DryRun) {
        Write-Host "Dry run only; no files copied."
        return
    }

    robocopy $Source $Destination /E /R:2 /W:2 /NFL /NDL /NP
    $code = $LASTEXITCODE
    if ($code -gt 7) {
        throw "Robocopy failed for $Label with exit code $code"
    }
}

function Copy-IfPresent {
    param(
        [string[]]$Candidates,
        [string]$Destination,
        [string]$Label
    )

    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            Invoke-Robocopy -Source $Candidate -Destination $Destination -Label $Label
            return
        }
    }

    Write-Warning "No source found for $Label. Checked: $($Candidates -join '; ')"
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$SourceRoot = Resolve-FullPath $SourceRoot
$DataRoot = Resolve-FullPath $DataDir

if (-not (Test-Path $SourceRoot)) {
    throw "SourceRoot does not exist: $SourceRoot"
}

Write-Host "Source root : $SourceRoot"
Write-Host "Data root   : $DataRoot"

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.12 "$PSScriptRoot\prepare_delivery_layout.py" --data-dir $DataRoot
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python "$PSScriptRoot\prepare_delivery_layout.py" --data-dir $DataRoot
}
else {
    throw "Python was not found. Install Python 3.12 or run setup_new_pc_windows.ps1 first."
}

$PreparedData = Join-Path $SourceRoot "STAGELIST3N-FusionCam-data"
if (Test-Path $PreparedData) {
    $SourceDataRoot = $PreparedData
}
else {
    $SourceDataRoot = $SourceRoot
}

if (-not $SkipDatasets) {
    Copy-IfPresent `
        -Candidates @(
            (Join-Path $SourceDataRoot "datasets\dataset_objets_V4"),
            (Join-Path $SourceRoot "dataset_objets_V4")
        ) `
        -Destination (Join-Path $DataRoot "datasets\dataset_objets_V4") `
        -Label "dataset_objets_V4"

    Copy-IfPresent `
        -Candidates @(
            (Join-Path $SourceDataRoot "datasets\dataset_objets_HD"),
            (Join-Path $SourceRoot "dataset_objets_HD")
        ) `
        -Destination (Join-Path $DataRoot "datasets\dataset_objets_HD") `
        -Label "dataset_objets_HD"

    Copy-IfPresent `
        -Candidates @(
            (Join-Path $SourceDataRoot "datasets\dataset"),
            (Join-Path $SourceRoot "dataset")
        ) `
        -Destination (Join-Path $DataRoot "datasets\dataset") `
        -Label "dataset"
}

if (-not $SkipRecordings) {
    Copy-IfPresent `
        -Candidates @(
            (Join-Path $SourceDataRoot "recordings\recordings"),
            (Join-Path $SourceDataRoot "recordings"),
            (Join-Path $SourceRoot "recordings\recordings"),
            (Join-Path $SourceRoot "recordings")
        ) `
        -Destination (Join-Path $DataRoot "recordings\recordings") `
        -Label "recordings"
}

if (-not $SkipModels) {
    Copy-IfPresent `
        -Candidates @(
            (Join-Path $SourceDataRoot "models"),
            (Join-Path $SourceRoot "Phase_2_Baseline_MonoCam\Modelstrained")
        ) `
        -Destination (Join-Path $DataRoot "models") `
        -Label "trained models"
}

if (-not $SkipReports) {
    Copy-IfPresent `
        -Candidates @(
            (Join-Path $SourceDataRoot "reports\Phase_3_Fusion_MultiCam"),
            (Join-Path $SourceRoot "Phase_3_Fusion_MultiCam\reports")
        ) `
        -Destination (Join-Path $DataRoot "reports\Phase_3_Fusion_MultiCam") `
        -Label "Phase 3 reports"
}

Write-Host ""
Write-Host "Heavy data copy complete." -ForegroundColor Green
Write-Host "Expected model example:"
Write-Host "  $DataRoot\models\V4\person_objects\yolov8s\weights\best.pt"
Write-Host "Expected recording example:"
Write-Host "  $DataRoot\recordings\recordings\Camera_2_2.3_20260506_131002.mp4"

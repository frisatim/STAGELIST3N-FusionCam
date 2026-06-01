<#
Create Windows directory junctions from the repository legacy paths to the
external heavy-data directory.

Run from the repository root after copying heavy data:
  .\scripts\link_external_data_windows.ps1

Default external data directory:
  ..\STAGELIST3N-FusionCam-data

This keeps datasets, recordings and models outside Git while allowing the
historical Python scripts to run natively on Windows without path changes.
#>

[CmdletBinding()]
param(
    [string]$DataDir = "../STAGELIST3N-FusionCam-data",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function New-Junction {
    param(
        [string]$Link,
        [string]$Target,
        [string]$Label
    )

    if (-not (Test-Path $Target)) {
        Write-Warning "Missing target for $Label`: $Target"
        return
    }

    if (Test-Path $Link) {
        $Item = Get-Item $Link
        $IsReparse = [bool]($Item.Attributes -band [IO.FileAttributes]::ReparsePoint)
        if ($IsReparse -or $Force) {
            Remove-Item $Link -Force -Recurse
        }
        else {
            Write-Warning "Existing non-junction path left unchanged for $Label`: $Link"
            return
        }
    }

    $Parent = Split-Path -Parent $Link
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    New-Item -ItemType Junction -Path $Link -Target $Target | Out-Null
    Write-Host "$Label -> $Target" -ForegroundColor Green
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$DataRoot = Resolve-FullPath $DataDir
if (-not (Test-Path $DataRoot)) {
    throw "Data directory does not exist: $DataRoot"
}

New-Junction `
    -Label "dataset" `
    -Link (Join-Path $RepoRoot "dataset") `
    -Target (Join-Path $DataRoot "datasets\dataset")

New-Junction `
    -Label "dataset_objets_HD" `
    -Link (Join-Path $RepoRoot "dataset_objets_HD") `
    -Target (Join-Path $DataRoot "datasets\dataset_objets_HD")

New-Junction `
    -Label "dataset_objets_V4" `
    -Link (Join-Path $RepoRoot "dataset_objets_V4") `
    -Target (Join-Path $DataRoot "datasets\dataset_objets_V4")

New-Junction `
    -Label "recordings" `
    -Link (Join-Path $RepoRoot "recordings") `
    -Target (Join-Path $DataRoot "recordings")

New-Junction `
    -Label "Modelstrained" `
    -Link (Join-Path $RepoRoot "Phase_2_Baseline_MonoCam\Modelstrained") `
    -Target (Join-Path $DataRoot "models")

New-Junction `
    -Label "Phase 3 reports" `
    -Link (Join-Path $RepoRoot "Phase_3_Fusion_MultiCam\reports") `
    -Target (Join-Path $DataRoot "reports\Phase_3_Fusion_MultiCam")

Write-Host ""
Write-Host "External data links are ready." -ForegroundColor Green

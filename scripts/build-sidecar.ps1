<#
.SYNOPSIS
    SR Engine Sidecar Builder (Windows)
    Packages the Python backend into a standalone PyInstaller binary
    for bundling into the Tauri desktop app via externalBin.

.DESCRIPTION
    Layout assumed:
      envs\build.ps1                    -> build the dev .venv
      scripts\build-sidecar.ps1 (this)  -> package that .venv into a sidecar
      scripts\sidecar_entry.py          -> PyInstaller entrypoint

    This script never installs anything into your dev .venv.
    It hardlinks .venv into a scratch build venv, installs PyInstaller there,
    builds, then deletes the scratch venv. Your dev environment stays untouched.

.PARAMETER Backend
    Force torch backend: cpu or cuda (default: auto-detect from .venv).

.PARAMETER KeepTemp
    Don't delete the scratch build venv (debugging).

.PARAMETER Force
    Continue even if -Backend disagrees with what's installed in .venv.

.EXAMPLE
    .\scripts\build-sidecar.ps1                   # auto-detect
    .\scripts\build-sidecar.ps1 -Backend cpu
    .\scripts\build-sidecar.ps1 -Backend cuda -Force -KeepTemp
#>

param(
    [ValidateSet("cpu", "cuda")]
    [string]$Backend = "",

    [switch]$KeepTemp,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $ScriptDir
$VenvDir = "$ProjectDir\.venv"
$BinariesDir = "$ProjectDir\src-tauri\binaries"

$BuildVenv = ""
$BuildMode = "--onefile"
$ExtraBinaries = @()
$ExtraData = @()
$DetectedBackend = ""

function info    { Write-Host "`n==> $($args -join ' ')" -ForegroundColor Cyan }
function success { Write-Host "`n✓ $($args -join ' ')" -ForegroundColor Green }
function warn    { Write-Host "`n⚠ $($args -join ' ')" -ForegroundColor Yellow }
function die     { Write-Host "`n✗ $($args -join ' ')" -ForegroundColor Red; exit 1 }

# ── Requirements check ───────────────────────────────────────────────────
function check_requirements {
    $null = Get-Command uv -ErrorAction Stop

    if (-not (Test-Path $VenvDir)) {
        die "No .venv found at $VenvDir — run envs\build.ps1 -Backend <cpu|cuda> first."
    }

    $pythonExe = "$VenvDir\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        die ".venv looks broken (no Scripts\python.exe) — rebuild it with envs\build.ps1."
    }

    $torchDir = "$VenvDir\Lib\site-packages\torch"
    if (-not (Test-Path $torchDir)) {
        die "torch is not installed in .venv — run envs\build.ps1 first."
    }

    $entryPoint = "$ScriptDir\sidecar_entry.py"
    if (-not (Test-Path $entryPoint)) {
        die "sidecar_entry.py not found in $ScriptDir"
    }
}

# ── Detect / validate backend from .venv ────────────────────────────────
function detect_backend {
    info "Detecting installed torch backend in .venv..."

    $detectCode = @'
import torch
if getattr(torch.version, "hip", None):
    print("rocm")
elif getattr(torch.version, "cuda", None):
    print("cuda")
else:
    print("cpu")
'@

    $tempScript = [System.IO.Path]::GetTempFileName() + ".py"
    try {
        Set-Content -Path $tempScript -Value $detectCode
        $output = & "$VenvDir\Scripts\python.exe" $tempScript 2>$null
        $script:DetectedBackend = ($output | Select-Object -Last 1).Trim()
    } finally {
        Remove-Item -Force $tempScript -ErrorAction SilentlyContinue
    }

    "Detected in .venv: $DetectedBackend"

    if ([string]::IsNullOrEmpty($Backend)) {
        $script:Backend = $DetectedBackend
    } elseif ($Backend -ne $DetectedBackend) {
        if ($Force) {
            warn "-Backend=$Backend but .venv has '$DetectedBackend' torch installed. Continuing anyway (-Force)."
        } else {
            die "-Backend=$Backend but .venv has '$DetectedBackend' torch installed. Rebuild with envs\build.ps1 -Backend $Backend, or pass -Force."
        }
    }

    "Building for: $Backend"
}

# ── Scratch build venv (hardlink copy) ──────────────────────────────────
function create_build_venv {
    info "Hardlinking dev .venv → scratch build venv (dev .venv is left untouched)..."

    $buildDir = "$ProjectDir\build"
    $null = New-Item -ItemType Directory -Force -Path $buildDir

    $tmpParent = New-Item -ItemType Directory -Path "$buildDir\.venv-sidecar.$([System.IO.Path]::GetRandomFileName())"
    $script:BuildVenv = "$tmpParent\venv"

    $hardlinkCode = @'
import os, sys, shutil
src, dst = sys.argv[1], sys.argv[2]
def link(src, dst, *a, **kw):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
shutil.copytree(src, dst, symlinks=True, copy_function=link)
'@

    $tempScript = [System.IO.Path]::GetTempFileName() + ".py"
    try {
        Set-Content -Path $tempScript -Value $hardlinkCode
        & "$VenvDir\Scripts\python.exe" $tempScript $VenvDir $BuildVenv
        if (-not $?) { die "Failed to hardlink .venv into scratch directory." }
    } finally {
        Remove-Item -Force $tempScript -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path "$BuildVenv\Scripts\python.exe")) {
        die "Scratch venv hardlink copy is broken (no Scripts\python.exe)."
    }

    success "Scratch venv ready: $BuildVenv"
}

# ── Install PyInstaller into scratch venv ──────────────────────────────
function install_pyinstaller {
    info "Installing PyInstaller into scratch venv..."
    uv pip install --python "$BuildVenv\Scripts\python.exe" pyinstaller
    if (-not $?) { die "Failed to install PyInstaller into scratch venv." }
}

# ── Run PyInstaller ─────────────────────────────────────────────────────
function run_pyinstaller {
    info "Running PyInstaller ($BuildMode, backend=$Backend)..."

    Push-Location $ProjectDir

    $entryPoint = "$ScriptDir\sidecar_entry.py"
    $configData = "$ProjectDir\src\sr_engine\utils\configs;sr_engine\utils\configs"

    $pyinstallerArgs = @(
        "-m", "PyInstaller"
        $BuildMode
        "--name", "sr-engine"
        "--add-data", $configData
        "--hidden-import", "uvicorn"
        "--hidden-import", "uvicorn.logging"
        "--hidden-import", "uvicorn.loops.auto"
        "--hidden-import", "uvicorn.protocols.http.auto"
        "--hidden-import", "uvicorn.middleware.wsgi"
        "--hidden-import", "starlette"
        "--hidden-import", "starlette.applications"
        "--hidden-import", "sr_engine.api.app"
        "--hidden-import", "sr_engine.api.routes"
        "--hidden-import", "sr_engine.models.archs"
        "--hidden-import", "torchvision"
        "--hidden-import", "torchvision.models"
        "--hidden-import", "lpips"
        "--hidden-import", "safetensors"
        "--hidden-import", "cv2"
        "--hidden-import", "pydantic"
        $entryPoint
    )

    if ($ExtraBinaries.Count -gt 0) { $pyinstallerArgs += $ExtraBinaries }
    if ($ExtraData.Count -gt 0) { $pyinstallerArgs += $ExtraData }

    & "$BuildVenv\Scripts\python.exe" @pyinstallerArgs
    if (-not $?) { die "PyInstaller build failed. Re-run with -KeepTemp to inspect the scratch venv." }
}

# ── Target triple resolution ────────────────────────────────────────────
function resolve_triple {
    $rustc = Get-Command rustc -ErrorAction SilentlyContinue
    if ($rustc) {
        $versionOutput = & rustc -vV 2>&1 | Out-String
        $match = [regex]::Match($versionOutput, 'host:\s*(\S+)')
        if ($match.Success) {
            return $match.Groups[1].Value
        }
    }
    warn "rustc not found — falling back to default triple."
    return "x86_64-pc-windows-msvc"
}

# ── Install result into src-tauri\binaries\ ────────────────────────────
function package_output {
    $triple = resolve_triple
    "Target triple: ${triple}"

    $null = New-Item -ItemType Directory -Force -Path $BinariesDir
    Remove-Item -Recurse -Force "$BinariesDir\sr-engine-${triple}" -ErrorAction SilentlyContinue
    Remove-Item -Force "$BinariesDir\sr-engine-${triple}.exe" -ErrorAction SilentlyContinue

    $distDir = "$ProjectDir\dist"

    if ($BuildMode -eq "--onedir") {
        if (-not (Test-Path "$distDir\sr-engine")) {
            die "Expected PyInstaller output dir not found: dist\sr-engine"
        }
        Copy-Item -Recurse "$distDir\sr-engine" "$BinariesDir\sr-engine-${triple}"
        $sizeBytes = (Get-ChildItem "$BinariesDir\sr-engine-${triple}" -Recurse | Measure-Object -Property Length -Sum).Sum
    } else {
        if (-not (Test-Path "$distDir\sr-engine.exe")) {
            die "Expected PyInstaller binary not found: dist\sr-engine.exe"
        }
        Copy-Item "$distDir\sr-engine.exe" "$BinariesDir\sr-engine-${triple}.exe"
        $sizeBytes = (Get-Item "$BinariesDir\sr-engine-${triple}.exe").Length
    }

    $outSize = if ($sizeBytes -gt 1GB) { "{0:N2}G" -f ($sizeBytes / 1GB) }
               elseif ($sizeBytes -gt 1MB) { "{0:N0}M" -f ($sizeBytes / 1MB) }
               else { "{0:N0}K" -f ($sizeBytes / 1KB) }

    Remove-Item -Recurse -Force $distDir -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$ProjectDir\build\.venv-sidecar.*" -ErrorAction SilentlyContinue
    Remove-Item -Force "$ProjectDir\sr-engine.spec" -ErrorAction SilentlyContinue

    $binaryName = if ($BuildMode -eq "--onedir") { "sr-engine-${triple}" } else { "sr-engine-${triple}.exe" }
    success "Sidecar built (${outSize}): src-tauri\binaries\${binaryName}"
}

# ── Cleanup ──────────────────────────────────────────────────────────────
function cleanup {
    if (-not [string]::IsNullOrEmpty($BuildVenv) -and (Test-Path $BuildVenv)) {
        if ($KeepTemp) {
            warn "Keeping scratch build venv for inspection: $BuildVenv"
        } else {
            Remove-Item -Recurse -Force (Split-Path -Parent $BuildVenv) -ErrorAction SilentlyContinue
        }
    }
}

# ── Main ─────────────────────────────────────────────────────────────────
try {
    info "SR Engine Sidecar Builder"
    check_requirements
    detect_backend
    create_build_venv
    install_pyinstaller
    run_pyinstaller
    package_output

    Write-Host "`nDev mode:      cargo tauri dev"
    Write-Host "Release build: cargo tauri build`n"
} finally {
    cleanup
}

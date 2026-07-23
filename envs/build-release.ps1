<#
.SYNOPSIS
    SR Tuner Release Builder (Windows)
    Mirrors envs/build-release.sh for Windows.

.DESCRIPTION
    Checks prerequisites, installs frontend deps, builds the Tauri app
    (MSI + NSIS installers), and prints output paths.

.PARAMETER Backend
    PyTorch backend for sidecar: cpu, cuda, or rocm (default: cpu).

.EXAMPLE
    .\envs\build-release.ps1
    .\envs\build-release.ps1 -Backend cuda
#>

param(
    [ValidateSet("cpu", "cuda", "rocm")]
    [string]$Backend = "cpu"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $ScriptDir
$FrontendDir = "$ProjectDir\frontend"
$SrcTauriDir = "$ProjectDir\src-tauri"

# ── Color helpers ────────────────────────────────────────────────────────
function info    { Write-Host "`n==> $($args -join ' ')" -ForegroundColor Cyan }
function ok      { Write-Host "  ✓ $($args -join ' ')" -ForegroundColor Green }
function warn    { Write-Host "  ⚠ $($args -join ' ')" -ForegroundColor Yellow }
function err     { Write-Host "  ✗ $($args -join ' ')" -ForegroundColor Red }
function section { Write-Host "`n▶ $($args -join ' ')" -ForegroundColor Blue }
function step    { Write-Host "  → $($args -join ' ')" -ForegroundColor Cyan }
function die     { err $args; exit 1 }

# ── Prerequisites ────────────────────────────────────────────────────────
section "Checking prerequisites"
$failed = $false

# Rust / Cargo
$cargoVer = (cargo --version 2>$null | Select-String '\d+\.\d+\.\d+').Matches.Value
if ($cargoVer) {
    ok "cargo $cargoVer"
} else {
    err "cargo not found — install from https://rustup.rs/"
    $failed = $true
}

# Tauri CLI
$tauriVer = (cargo tauri --version 2>$null | Select-String '\d+\.\d+\.\d+').Matches.Value
if ($tauriVer) {
    ok "cargo-tauri $tauriVer"
} else {
    warn "cargo-tauri not found — installing via cargo..."
    cargo install tauri-cli --version "^2"
    if ($?) { ok "cargo-tauri installed" } else { err "Failed to install cargo-tauri"; $failed = $true }
}

# Node.js / npm
$nodeVer = node --version 2>$null
if ($nodeVer) {
    ok "node $nodeVer"
} else {
    err "node not found — install from https://nodejs.org/"
    $failed = $true
}
$npmVer = npm --version 2>$null
if ($npmVer) {
    ok "npm v$npmVer"
} else {
    err "npm not found"
    $failed = $true
}

# uv
$uvVer = (uv --version 2>$null | Select-String '\d+\.\d+\.\d+').Matches.Value
if ($uvVer) {
    ok "uv $uvVer"
} else {
    err "uv not found — install from https://docs.astral.sh/uv/"
    $failed = $true
}

# Visual Studio Build Tools (for Rust)
$clVer = & { cl.exe 2>$null | Select-String 'x86' } 2>$null
if ($?) {
    ok "Visual Studio Build Tools detected"
} else {
    err "Visual Studio Build Tools not found (cl.exe missing)"
    err "Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    err "Select the 'Desktop development with C++' workload"
    $failed = $true
}

Write-Host ""
if ($failed) {
    die "Prerequisites check failed. Install the above and retry."
}
ok "All prerequisites satisfied!"

# ── Frontend ─────────────────────────────────────────────────────────────
section "Installing frontend dependencies"
step "npm install..."
Push-Location $FrontendDir
npm install 2>&1 | Select-Object -Last 3
if (-not $?) { die "npm install failed" }
Pop-Location
ok "Frontend deps installed"

# ── Build Tauri ──────────────────────────────────────────────────────────
section "Building Tauri app (MSI + NSIS)"
step "cargo tauri build..."
Push-Location $ProjectDir
cargo tauri build 2>&1 | Select-Object -Last 5
if (-not $?) { die "Tauri build failed" }
Pop-Location

# ── Output ───────────────────────────────────────────────────────────────
$bundleDir = "$SrcTauriDir\target\release\bundle"
Write-Host ""
section "Build complete!"
Write-Host ""
Write-Host "    MSI installer:  $bundleDir\msi\"
Write-Host "    NSIS installer: $bundleDir\nsis\"
Write-Host "    Binary:         $SrcTauriDir\target\release\sr-tuner.exe"
Write-Host ""
ok "Done!"
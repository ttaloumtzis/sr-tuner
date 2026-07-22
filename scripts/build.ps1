<#
.SYNOPSIS
    SR Tuner Build Orchestrator (Windows)

.DESCRIPTION
    Orchestrates building the frontend, Python sidecar, and Tauri desktop app.

.PARAMETER Command
    Command to run: all, frontend, sidecar, tauri, rebuild, dev, clean, check,
    help. Default (empty) builds frontend + Tauri.

.PARAMETER Backend
    Sidecar backend: cpu or cuda (rocm is Linux-only).

.PARAMETER Parallel
    Build frontend and sidecar concurrently (-j also works).

.PARAMETER Help
    Show this help (-h also works).

.EXAMPLE
    .\scripts\build.ps1                    # frontend + Tauri
    .\scripts\build.ps1 all -Backend cpu   # clean + everything
    .\scripts\build.ps1 all -Backend cuda -Parallel
    .\scripts\build.ps1 all cuda -j        # bash-style
    .\scripts\build.ps1 dev                # dev mode
    .\scripts\build.ps1 check              # prerequisites
    .\scripts\build.ps1 clean              # remove artifacts
    .\scripts\build.ps1 help               # show help
#>

param(
    [Parameter(Position=0)]
    [string]$Command = "",

    [Parameter(Position=1)]
    [ValidateSet("cpu", "cuda")]
    [string]$Backend = "",

    [Alias("j")]
    [switch]$Parallel,

    [Alias("h")]
    [switch]$Help,

    # Capture any remaining bash-style args (e.g. -j as separate token)
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $ScriptDir
$FrontendDir = "$ProjectDir\frontend"
$SrcTauriDir = "$ProjectDir\src-tauri"

# Detect platform
$Platform = if ($env:OS -eq "Windows_NT") { "windows" } else { "linux" }

# ── Color / log helpers ─────────────────────────────────────────────────
$Cyan = [ConsoleColor]::Cyan
$Green = [ConsoleColor]::Green
$Yellow = [ConsoleColor]::Yellow
$Red = [ConsoleColor]::Red
$Blue = [ConsoleColor]::Blue

function info    { Write-Host "  $($args -join ' ')" }
function ok      { Write-Host "  ✓ $($args -join ' ')" -ForegroundColor $Green }
function warn    { Write-Host "  ⚠ $($args -join ' ')" -ForegroundColor $Yellow }
function err     { Write-Host "  ✗ $($args -join ' ')" -ForegroundColor $Red }
function section { Write-Host "`n▶ $($args -join ' ')" -ForegroundColor $Blue -NoNewline; Write-Host "" }
function step    { Write-Host "  → $($args -join ' ')" -ForegroundColor $Cyan }

function die     { err $args; exit 1 }

# ── Help ────────────────────────────────────────────────────────────────
function show_help {
    $me = ".\scripts\build.ps1"
@"
SR Tuner Build Script — ${Platform}

Usage:  ${me} [command] [options]

Commands:
  (default)   Build frontend + Tauri app
  all         Clean + build everything (including sidecar)
  frontend    Build frontend only
  tauri       Build Tauri app only (frontend must be pre-built)
  sidecar     Build Python sidecar
  dev         Start development server (Vite + Tauri hot-reload)
  clean       Delete all build artifacts
  rebuild     Clean then build (frontend + Tauri, no sidecar)
  check       Verify all build prerequisites
  help        Show this help

Options:
  -Backend cpu|cuda   Sidecar variant (default: auto-detect from .venv)
  -Parallel, -j       Build frontend and sidecar concurrently
  -Help, -h           Show this help

Examples:
  ${me}                          # frontend + Tauri
  ${me} dev                      # hot-reload dev mode
  ${me} check                    # verify prerequisites
  ${me} all -Backend cpu         # clean + everything
  ${me} all cuda -j              # clean + all CUDA, parallel
  ${me} sidecar cuda             # CUDA sidecar only
  ${me} rebuild                  # clean + frontend + Tauri
  ${me} clean                    # remove artifacts

"@
}

# ── Prerequisite check ──────────────────────────────────────────────────
function check_prereqs {
    section "Checking prerequisites"
    $failed = $false

    # Rust / Cargo
    $cargoVer = (cargo --version 2>$null | Select-String '\d+\.\d+\.\d+').Matches.Value
    if ($cargoVer) {
        ok "cargo $cargoVer"
    } else {
        err "cargo not found — install Rust from https://rustup.rs/"
        $failed = $true
    }

    # Tauri CLI
    $tauriVer = (cargo tauri --version 2>$null | Select-String '\d+\.\d+\.\d+').Matches.Value
    if ($tauriVer) {
        ok "cargo-tauri $tauriVer"
    } else {
        warn "cargo-tauri not found — attempting install via cargo..."
        cargo install tauri-cli --version "^2"
        if ($?) {
            ok "cargo-tauri installed"
        } else {
            err "Failed to install cargo-tauri"
            $failed = $true
        }
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
        err "uv not found — https://github.com/astral-sh/uv"
        $failed = $true
    }

    # Python venv
    if (Test-Path "$ProjectDir\.venv") {
        ok "Python virtual environment (.venv)"
    } else {
        warn "No .venv found — run: envs\build.ps1 -Backend cpu"
    }

    # Visual Studio Build Tools (for Rust)
    $clVer = & { cl.exe 2>$null | Select-String 'x86' } 2>$null
    if ($?) {
        ok "Visual Studio Build Tools detected"
    } else {
        warn "Visual Studio Build Tools not found (cl.exe missing)"
        warn "Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        warn "Select the 'Desktop development with C++' workload"
    }

    Write-Host ""
    if ($failed) {
        die "Prerequisites check failed. Install the above and retry."
    }
    ok "All prerequisites satisfied!"
}

# ── Clean ────────────────────────────────────────────────────────────────
function clean {
    section "Cleaning Build Artifacts"
    Remove-Item -Recurse -Force "$FrontendDir\dist" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$ProjectDir\dist" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$ProjectDir\build" -ErrorAction SilentlyContinue
    Remove-Item -Force "$ProjectDir\sr-engine.spec" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$SrcTauriDir\binaries" -ErrorAction SilentlyContinue
    $null = New-Item -ItemType Directory -Force -Path "$SrcTauriDir\binaries"
    ok "Clean complete."
}

# ── Build Frontend ──────────────────────────────────────────────────────
function build_frontend {
    section "Building Frontend"
    step "Installing npm dependencies..."
    Push-Location $FrontendDir
    npm install
    if (-not $?) { die "npm install failed" }

    step "Compiling TypeScript + bundling with Vite..."
    npm run build
    if (-not $?) { die "npm run build failed" }

    ok "Frontend ready → frontend\dist\"
}

# ── Build Sidecar ───────────────────────────────────────────────────────
function build_sidecar {
    param($Variant)

    section "Building Sidecar $(if ($Variant) { "($Variant)" })"

    if (-not (Test-Path "$ProjectDir\.venv")) {
        die "No .venv found — run envs\build.ps1 -Backend <cpu|cuda> first"
    }

    step "Running PyInstaller via build-sidecar.ps1..."
    $argsList = @()
    if ($Variant) { $argsList += "-Backend"; $argsList += $Variant }
    & "$ScriptDir\build-sidecar.ps1" @argsList
    if (-not $?) { die "Sidecar build failed" }
}

# ── Build Tauri App ─────────────────────────────────────────────────────
function build_tauri {
    section "Building Tauri App"
    step "Compiling Rust + creating platform bundles..."
    Push-Location $ProjectDir
    cargo tauri build
    if (-not $?) { die "Tauri build failed" }

    $bundleDir = "$SrcTauriDir\target\release\bundle"
    Write-Host ""
    ok "Build complete! Output:"
    Write-Host "    .msi    → $bundleDir\msi\"
    Write-Host "    .exe    → $bundleDir\nsis\"
    Write-Host "    Binary  → $SrcTauriDir\target\release\sr-tuner.exe"
}

# ── Development mode ────────────────────────────────────────────────────
function dev_mode {
    section "Starting Development Server"
    step "Starting Vite dev server + Tauri (hot-reload)..."
    warn "Press Ctrl+C to stop.`n"
    Push-Location $ProjectDir
    cargo tauri dev
}

# ── Parallel build ──────────────────────────────────────────────────────
function build_parallel {
    param($Variant)

    section "Parallel Build: Frontend + Sidecar $(if ($Variant) { "($Variant)" })"
    step "Starting both builds concurrently...`n"

    $frontendLog = "$env:TEMP\sr-tuner-frontend.log"
    $sidecarLog  = "$env:TEMP\sr-tuner-sidecar.log"
    Remove-Item -Force $frontendLog, $sidecarLog -ErrorAction SilentlyContinue

    # Build frontend job
    $frontendJob = Start-Job -Name "frontend" -ScriptBlock {
        param($Fd, $Pd)
        Push-Location $Fd
        npm install 2>&1 | Out-File -Append $using:frontendLog
        if (-not $?) { throw "npm install failed" }
        npm run build 2>&1 | Out-File -Append $using:frontendLog
        if (-not $?) { throw "npm run build failed" }
        Pop-Location
    } -ArgumentList $FrontendDir, $ProjectDir

    # Build sidecar job
    $sidecarJob = Start-Job -Name "sidecar" -ScriptBlock {
        param($Sd, $Pd, $V)
        $argsList = @()
        if ($V) { $argsList += "-Backend"; $argsList += $V }
        & "$Sd\build-sidecar.ps1" @argsList 2>&1 | Out-File -Append $using:sidecarLog
        if (-not $?) { throw "Sidecar build failed" }
    } -ArgumentList $ScriptDir, $ProjectDir, $Variant

    # Wait for both
    $null = $frontendJob | Wait-Job
    $null = $sidecarJob  | Wait-Job

    # Check results
    $frontendOk = $true
    $sidecarOk  = $true

    if ($frontendJob.State -eq "Failed") {
        $frontendOk = $false
        err "[frontend] FAILED"
        $frontendJob | Receive-Job | Write-Host
    } else {
        ok "[frontend] Done"
    }

    if ($sidecarJob.State -eq "Failed") {
        $sidecarOk = $false
        err "[sidecar] FAILED"
        $sidecarJob | Receive-Job | Write-Host
    } else {
        ok "[sidecar]  Done"
    }

    Remove-Job $frontendJob, $sidecarJob -Force

    if (-not $frontendOk) {
        err "Frontend build failed. Log:"
        if (Test-Path $frontendLog) { Get-Content $frontendLog | Write-Host }
        exit 1
    }
    if (-not $sidecarOk) {
        err "Sidecar build failed. Log:"
        if (Test-Path $sidecarLog) { Get-Content $sidecarLog | Write-Host }
        exit 1
    }

    Write-Host ""
    build_tauri
}

# ── Argument preprocessing ──────────────────────────────────────────────
# Handle bash-style flags in ExtraArgs (e.g., -j as separate token)
foreach ($arg in $ExtraArgs) {
    switch -Wildcard ($arg) {
        "-j"       { $Parallel  = $true }
        "--parallel" { $Parallel = $true }
        "-h"       { $Help      = $true }
        "--help"   { $Help      = $true }
        "cpu"      { if ([string]::IsNullOrEmpty($Backend)) { $Backend = "cpu" } }
        "cuda"     { if ([string]::IsNullOrEmpty($Backend)) { $Backend = "cuda" } }
        "rocm"     { warn "ROCm is Linux-only. Use 'cuda' or 'cpu' on Windows." }
    }
}

# ── Main dispatch ───────────────────────────────────────────────────────
if ($Help) { show_help; return }

switch -Wildcard ($Command) {
    "help"  { show_help }
    "check" { check_prereqs }
    "dev"   { dev_mode }
    "clean" { clean }
    "frontend" { build_frontend }
    "tauri" { build_tauri }
    "sidecar" { build_sidecar -Variant $Backend }
    "all" {
        clean
        if ($Parallel) {
            build_parallel -Variant $Backend
        } else {
            build_frontend
            build_sidecar -Variant $Backend
            build_tauri
        }
    }
    "rebuild" {
        clean
        build_frontend
        build_tauri
    }
    "" {
        build_frontend
        build_tauri
    }
    default {
        err "Unknown command: $Command"
        show_help
        exit 1
    }
}

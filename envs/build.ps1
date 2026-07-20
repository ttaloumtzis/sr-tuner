<#
.SYNOPSIS
    SR Engine Environment Builder (Windows)
.DESCRIPTION
    Creates a uv virtual environment, installs project dependencies, and
    installs the backend-specific PyTorch wheel (CPU or CUDA).
    ROCm is not supported on Windows.
.PARAMETER Backend
    PyTorch backend: "cpu" or "cuda".
.PARAMETER Clean
    Remove .venv and uv.lock before building.
.EXAMPLE
    .\envs\build.ps1 -Backend cpu
    .\envs\build.ps1 -Backend cuda
    .\envs\build.ps1 -Backend cpu -Clean
#>

param(
    [Parameter(Mandatory)]
    [ValidateSet("cpu", "cuda")]
    [string]$Backend,

    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$TorchIndex = @{
    cpu  = "https://download.pytorch.org/whl/cpu"
    cuda = "https://download.pytorch.org/whl/cu121"
}

$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

function Info  { Write-Host "`n==> $args" }
function Success { Write-Host "`n[v] $args" -ForegroundColor Green }
function Warn  { Write-Host "`n[!] $args" -ForegroundColor Yellow }
function Die   { Write-Host "`n[x] $args" -ForegroundColor Red; exit 1 }

# ── Requirements ─────────────────────────────────────────────────────────

function CheckRequirements {
    Info "Checking prerequisites..."

    $null = Get-Command uv -ErrorAction Stop
    $null = Get-Command python -ErrorAction Stop

    if ($Backend -eq "rocm") {
        Die "ROCm is not supported on Windows. Use -Backend cpu or -Backend cuda."
    }

    Success "Prerequisites found."
}

# ── Clean ─────────────────────────────────────────────────────────────────

function CleanEnvironment {
    if (-not $Clean) { return }

    Info "Removing existing environment..."
    $venv = Join-Path $ProjectDir ".venv"
    $lock = Join-Path $ProjectDir "uv.lock"
    if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
    if (Test-Path $lock) { Remove-Item -Force $lock }
}

# ── Create venv ───────────────────────────────────────────────────────────

function CreateEnvironment {
    Info "Creating virtual environment..."
    Push-Location $ProjectDir
    try {
        uv venv
    } finally {
        Pop-Location
    }
}

# ── Install base deps ─────────────────────────────────────────────────────

function InstallBase {
    Info "Installing project dependencies (excluding dev group)..."
    Push-Location $ProjectDir
    try {
        uv sync --no-dev
    } finally {
        Pop-Location
    }
}

# ── Retry helper ──────────────────────────────────────────────────────────

function Retry {
    param([scriptblock]$Block)
    $attempts = 3
    $delay = 5
    for ($i = 1; $i -le $attempts; $i++) {
        try {
            & $Block
            return
        } catch {
            if ($i -lt $attempts) {
                Warn "Attempt $i failed. Retrying in ${delay}s..."
                Start-Sleep -Seconds $delay
            } else {
                throw
            }
        }
    }
}

# ── Install torch ─────────────────────────────────────────────────────────

function InstallBackend {
    $index = $TorchIndex[$Backend]

    Info "Installing PyTorch backend: $Backend"
    Write-Host "Index: $index"

    Push-Location $ProjectDir
    try {
        Retry -Block {
            uv pip install --index-url $index torch torchvision
        }
    } finally {
        Pop-Location
    }

    Success "PyTorch installed."
}

# ── Verify ─────────────────────────────────────────────────────────────────

function VerifyEnvironment {
    Info "Running backend verification..."
    Push-Location $ProjectDir
    try {
        uv run python envs/verify_env.py
    } finally {
        Pop-Location
    }
}

# ── Main ──────────────────────────────────────────────────────────────────

function Main {
    Write-Host "`n=== SR Engine Environment Builder ===" -ForegroundColor Cyan
    Write-Host "Backend : $Backend"

    CheckRequirements
    CleanEnvironment
    CreateEnvironment
    InstallBase
    InstallBackend
    VerifyEnvironment

    Write-Host "`n[v] Environment successfully created." -ForegroundColor Green
    Write-Host "`nActivate with:" -ForegroundColor Yellow
    Write-Host "`n    .venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
}

Main

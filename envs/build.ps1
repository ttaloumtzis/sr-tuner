<#
.SYNOPSIS
    SR Engine Environment Builder (Windows)
    Creates a uv virtual environment, installs project deps + PyTorch,
    and verifies the installation.

.PARAMETER Backend
    PyTorch backend: cpu or cuda (rocm is Linux-only).

.PARAMETER Clean
    Remove .venv and uv.lock before building.

.EXAMPLE
    .\envs\build.ps1 -Backend cpu
    .\envs\build.ps1 -Backend cuda -Clean
#>

param(
    [Parameter(Mandatory)]
    [ValidateSet("cpu", "cuda")]
    [string]$Backend,

    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$TorchIndex = @{
    cpu  = "https://download.pytorch.org/whl/cpu"
    cuda = "https://download.pytorch.org/whl/cu121"
}

$ScriptDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $ScriptDir

function info  { Write-Host "`n==> $($args -join ' ')" -ForegroundColor Cyan }
function ok    { Write-Host "`n✓ $($args -join ' ')" -ForegroundColor Green }
function warn  { Write-Host "`n⚠ $($args -join ' ')" -ForegroundColor Yellow }
function die   { Write-Host "`n✗ $($args -join ' ')" -ForegroundColor Red; exit 1 }

# ── Banner ────────────────────────────────────────────────────────────────
info "SR Engine Environment Builder"
"Backend : $Backend"

# ── Prerequisites ─────────────────────────────────────────────────────────
$null = Get-Command uv -ErrorAction Stop

# ── Clean ─────────────────────────────────────────────────────────────────
if ($Clean) {
    info "Removing existing environment..."
    Remove-Item -Recurse -Force "$ProjectDir\.venv" -ErrorAction SilentlyContinue
    Remove-Item -Force "$ProjectDir\uv.lock" -ErrorAction SilentlyContinue
}

# ── Create venv ───────────────────────────────────────────────────────────
info "Creating virtual environment..."
Push-Location $ProjectDir
uv venv
if (-not $?) { die "uv venv failed" }

# ── Install base deps ─────────────────────────────────────────────────────
info "Installing project dependencies (excluding dev group)..."
uv sync --no-dev
if (-not $?) { die "uv sync failed" }

# ── Install PyTorch ───────────────────────────────────────────────────────
$index = $TorchIndex[$Backend]
info "Installing PyTorch backend: $Backend"
"Index: $index"

$installArgs = @("pip", "install", "--index-url", $index)
if ($Backend -eq "cuda") {
    $installArgs += "--reinstall"
}
$installArgs += @("torch", "torchvision")

$attempts = 3
for ($i = 1; $i -le $attempts; $i++) {
    uv @installArgs
    if ($?) { break }
    if ($i -lt $attempts) {
        warn "Attempt $i failed. Retrying in 5s..."
        Start-Sleep -Seconds 5
    } else {
        die "Failed to install PyTorch after $attempts attempts"
    }
}
ok "PyTorch installed."

# ── Install LPIPS ─────────────────────────────────────────────────────────
info "Installing optional LPIPS dependency..."
uv pip install lpips

# ── Verify ────────────────────────────────────────────────────────────────
info "Running backend verification..."
$verifyCode = @'
import torch, sys
backend = sys.argv[1]
print()
print('=' * 60)
print('Torch Version:', torch.__version__)
print('Backend      :', backend)
print()
if backend == 'cpu':
    if torch.cuda.is_available():
        raise RuntimeError('GPU backend detected while CPU backend requested.')
    print('✓ CPU backend verified')
elif backend == 'cuda':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    if torch.version.cuda is None:
        raise RuntimeError('CUDA wheel not installed.')
    print('✓ CUDA backend verified')
    print('CUDA Version:', torch.version.cuda)
    print('GPU         :', torch.cuda.get_device_name(0))
print('=' * 60)
'@

$tempScript = [System.IO.Path]::GetTempFileName() + ".py"
try {
    Set-Content -Path $tempScript -Value $verifyCode
    uv run python $tempScript $Backend
    if (-not $?) { die "Backend verification failed" }
} finally {
    Remove-Item -Force $tempScript -ErrorAction SilentlyContinue
}

ok "Environment successfully created."
Write-Host "`nActivate with:`n"
Write-Host "    .venv\Scripts\Activate.ps1`n"

# bootstrap.ps1
# Sets up the reproducibility environment for the icra2027 research line.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
#
# Requirements:
#   - `uv` (https://docs.astral.sh/uv) for Python environment management.
#   - A populated `third_party/` (fetched here) with the G1 assets.
# Snowflake note: no GPU is required; everything runs on CPU.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "== icra2027 bootstrap ==" -ForegroundColor Cyan

# ---- 0. Tools ------------------------------------------------------------
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw "uv not found. Install it from https://docs.astral.sh/uv and retry."
}

# ---- 1. Third-party assets ------------------------------------------------
# Used by 01-snn (MuJoCo model zoo) and 02-cl (Unitree RL Gym deployment +
# the G1 motion.pt checkpoint), and shared by the surrogate repo.
$tp = Join-Path $root "third_party"
New-Item -ItemType Directory -Force -Path $tp | Out-Null

$jobs = @(
    @{ dir = Join-Path $tp "unitree_rl_gym";   repo = "https://github.com/unitreerobotics/unitree_rl_gym.git" },
    @{ dir = Join-Path $tp "mujoco_menagerie"; repo = "https://github.com/google-deepmind/mujoco_menagerie.git" }
)
foreach ($j in $jobs) {
    if (Test-Path (Join-Path $j.dir ".git")) {
        Write-Host " [skip] $($j.dir) already present"
    } else {
        Write-Host " [clone] $($j.repo)"
        git clone --depth 1 $j.repo $j.dir
    }
}
# Note: the deployment configs (`deploy/deploy_mujoco/configs/g1.yaml` in the
# RLGym tree) resolve `xml_path`/`policy_path` against `{LEGGED_GYM_ROOT_DIR}`;
# a pre-trained G1 `motion.pt` under `deploy/pre_train/g1/` is required to run
# the demos. See 01-snn/README.md.

# ---- 2. Shared Python environment ---------------------------------------
# The contract environment (02-cl) is reused by the surrogate repository via a
# junction; black-box reproducibility pins live in the per-phase lock files.
$phaseEnv = Join-Path $root "02_cl/.venv312"
if (Test-Path $phaseEnv) {
    Write-Host " [reuse] $phaseEnv (existing contract venv)"
} else {
    Write-Host " [create] shared venv (Python 3.12)"
    & $uv.Source "python install 3.12"
    & $uv.Source "venv" (Join-Path $root ".venv") --python 3.12
    & $uv.Source "pip" "install" "--python" (Join-Path $root ".venv/Scripts/python.exe") "-r" (Join-Path $root "02_cl/requirements.lock.txt")
}

Write-Host ""
Write-Host "Environment ready. Per-phase commands:" -ForegroundColor Green
Write-Host "   01-snn       : see 01_snn/README.md (deploy12.py, run_eval.py)"
Write-Host "   02-cl        : see 02_cl/README.md (bridge_g1.py, ablate_loop.py, demo_walk.py)"
Write-Host "   03-union     : see 03_union/README.md (BL-1 substrate, bl1_venv)"
Write-Host "   surrogate-cl : separate repo, see its README.md (task_tracking.py, signal_analysis.py)"
Write-Host ""
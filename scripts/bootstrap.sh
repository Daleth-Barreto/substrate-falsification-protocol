#!/usr/bin/env bash
# bootstrap.sh
# Sets up the reproducibility environment for the icra2027 research line.
# Usage:  ./scripts/bootstrap.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it from https://docs.astral.sh/uv and retry." >&2
  exit 1
fi

TP="$ROOT/third_party"
mkdir -p "$TP"

clone_if_missing() {
  if [ -d "$1/.git" ]; then
    echo " [skip] $1 already present"
  else
    echo " [clone] $2"
    git clone --depth 1 "$2" "$1"
  fi
}

clone_if_missing "$TP/unitree_rl_gym"   "https://github.com/unitreerobotics/unitree_rl_gym.git"
clone_if_missing "$TP/mujoco_menagerie" "https://github.com/google-deepmind/mujoco_menagerie.git"

if [ -d "$ROOT/02_cl/.venv312" ]; then
  echo " [reuse] $ROOT/02_cl/.venv312 (existing contract venv)"
else
  echo " [create] shared venv (Python 3.12)"
  uv python install 3.12
  uv venv "$ROOT/.venv" --python 3.12
  uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/02_cl/requirements.lock.txt"
fi

echo ""
echo "Environment ready. Per-phase commands:"
echo "   01-snn       : see 01_snn/README.md"
echo "   02-cl        : see 02_cl/README.md"
echo "   03-union     : see 03_union/README.md"
echo "   surrogate-cl : separate repo, see its README.md"
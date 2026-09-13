#!/usr/bin/env bash
# Sets up a virtual environment (if not already present), installs
# dependencies, and runs the full Part 3/4/5 experiment suite.
#
# Usage:
#   ./run.sh              # run the full experiment pipeline
#   ./run.sh kinematics    # just run the FK/Jacobian self-check
#   ./run.sh visualize     # run the real-time visualization
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

cd src

if [ "${1:-}" == "kinematics" ]; then
    echo "Running kinematics self-check (FK/Jacobian vs finite differences)..."
    python3 kinematics.py
elif [ "${1:-}" == "visualize" ]; then
    shift
    echo "Running visualization..."
    python3 visualize.py "$@"
else
    echo "Running full experiment suite (Parts 3, 4, 5)..."
    python3 experiments.py
    echo ""
    echo "Done. Plots and summary.txt were written to ../results/"
fi

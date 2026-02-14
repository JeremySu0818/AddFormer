#!/usr/bin/env bash
set -e

echo "[1/3] Checking virtual environment (.venv)"
if [ ! -d ".venv" ]; then
    echo "    .venv not found, calling install_env.sh to setup environment..."
    bash install_env.sh
else
    echo "    .venv already exists, skipping installation"
fi

echo
echo "[2/3] Activating virtual environment"
source .venv/bin/activate

echo
echo "[3/3] Generating data"
python prepare_data.py
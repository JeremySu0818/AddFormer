#!/usr/bin/env bash
set -e

echo "[1/6] Checking virtual environment (.venv)"
if [ ! -d ".venv" ]; then
    echo "    .venv not found, calling install_env.sh to setup environment..."
    bash install_env.sh
else
    echo "    .venv already exists, skipping installation"
fi

echo
echo "[2/6] Activating virtual environment"
source .venv/bin/activate

echo
echo "[3/6] Generating data"
python prepare_data.py

echo
read -p "Press Enter to continue..."

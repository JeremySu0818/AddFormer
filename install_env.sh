#!/usr/bin/env bash

# Ensure script exits on any error
set -e

echo "[1/6] Checking and creating virtual environment (.venv)"
if [ ! -d ".venv" ]; then
    python -m venv .venv
else
    echo "    .venv already exists, skipping creation"
fi

echo

echo "[2/6] Activating virtual environment"
source .venv/bin/activate

echo

echo "[3/6] Upgrading pip"
python -m pip install -U pip

echo

echo "[4/6] Installing PyTorch (torch, CUDA 13.0)"
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130

echo

echo "[5/6] Installing transformers / tqdm / numpy / datasets / matplotlib"
python -m pip install transformers tqdm numpy datasets matplotlib

echo

echo "[6/6] Installing accelerate"
python -m pip install "accelerate>=0.26.0"

echo

echo "===== All steps completed ====="
read -p "Press any key to continue..." -n1 -s

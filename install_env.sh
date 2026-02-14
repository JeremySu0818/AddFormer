#!/usr/bin/env bash
set -e

echo "[1/6] Checking and creating virtual environment (.venv)"
if [ ! -d ".venv" ]; then
    uv venv
else
    echo "    .venv already exists, skipping creation"
fi

echo
echo "[2/6] Activating virtual environment"
source .venv/bin/activate

echo
echo "[3/6] Upgrading pip (using uv)"
uv pip install -U pip

echo
echo "[4/6] Installing PyTorch (torch, CUDA 13.0)"
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

echo
echo "[5/6] Installing transformers / tqdm / numpy / datasets"
uv pip install transformers tqdm numpy datasets

echo
echo "[6/6] Installing accelerate"
uv pip install "accelerate>=0.26.0"

echo
echo "All steps completed"
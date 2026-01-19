#!/usr/bin/env bash

# 確保腳本在出錯時立即停止
set -e

echo "[1/6] Checking and creating virtual environment (.venv)"
# 檢查虛擬環境是否存在，若不存在則使用 uv 建立 
if [ ! -d ".venv" ]; then
    uv venv
else
    echo "    .venv already exists, skipping creation"
fi

echo

echo "[2/6] Activating virtual environment"
# 啟動虛擬環境 
source .venv/bin/activate

echo

echo "[3/6] Upgrading pip (using uv)"
# 使用 uv 升級 pip
uv pip install -U pip

echo

echo "[4/6] Installing PyTorch (torch, CUDA 13.0)"
# 使用 uv 安裝支援 CUDA 13.0 的 PyTorch 
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

echo

echo "[5/6] Installing transformers / tqdm / numpy / datasets / matplotlib"
# 使用 uv 安裝指定的函式庫 [cite: 4]
uv pip install transformers tqdm numpy datasets matplotlib

echo

echo "[6/6] Installing accelerate"
# 使用 uv 安裝指定版本以上的 accelerate [cite: 5]
uv pip install "accelerate>=0.26.0"

echo

echo "===== All steps completed ====="
read -p "Press any key to continue..." -n1 -s
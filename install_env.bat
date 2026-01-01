@echo off
chcp 65001 >nul

echo [1/6] 檢查並建立虛擬環境 (.venv)
if not exist ".venv" (
    py -m venv .venv
) else (
    echo     .venv 已存在，略過建立
)

echo.
echo [2/6] 啟動虛擬環境
call .venv\Scripts\activate.bat

echo.
echo [3/6] 升級 pip
py -m pip install -U pip

echo.
echo [4/6] 安裝 PyTorch (torch, CUDA 13.0)
py -m pip install torch --index-url https://download.pytorch.org/whl/cu130

echo.
echo [5/6] 安裝 transformers / tqdm / numpy / datasets
py -m pip install transformers tqdm numpy datasets

echo.
echo [6/6] 安裝 accelerate
py -m pip install "accelerate>=0.26.0"

echo.
echo ===== 所有步驟完成 =====
pause

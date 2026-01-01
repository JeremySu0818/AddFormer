@echo off
chcp 65001 >nul

echo [1/6] Checking and creating virtual environment (.venv)
if not exist ".venv" (
    py -m venv .venv
) else (
    echo     .venv already exists, skipping creation
)

echo.
echo [2/6] Activating virtual environment
call .venv\Scripts\activate.bat

echo.
echo [3/6] Upgrading pip
py -m pip install -U pip

echo.
echo [4/6] Installing PyTorch (torch, CUDA 13.0)
py -m pip install torch --index-url https://download.pytorch.org/whl/cu130
    
echo.
echo [5/6] Installing transformers / tqdm / numpy / datasets / matplotlib
py -m pip install transformers tqdm numpy datasets matplotlib

echo.
echo [6/6] Installing accelerate
py -m pip install "accelerate>=0.26.0"

echo.
echo All steps completed
pause

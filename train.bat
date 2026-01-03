@echo off
chcp 65001 >nul

if not exist ".venv" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat

python train_addformer.py
if %errorlevel% neq 0 (
    echo 訓練過程中出現錯誤。
    pause
    exit /b
)

pause
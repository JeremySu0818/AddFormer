@echo off
chcp 65001 >nul

if not exist ".venv" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat

python train_subformer.py
if %errorlevel% neq 0 (
    echo An error occurred during the training process.
    pause
    exit /b
)

pause

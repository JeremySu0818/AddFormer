@echo off
chcp 65001 >nul

echo [1/6] Checking virtual environment (.venv)
if not exist ".venv" (
    echo     .venv not found, calling install_env.bat to setup environment...
    call install_env.bat
) else (
    echo     .venv already exists, skipping installation
)

echo.
echo [2/6] Activating virtual environment
call .venv\Scripts\activate.bat

echo.
echo [3/6] Starting training
python train_addformer.py

if %errorlevel% neq 0 (
    echo.
    echo An error occurred during the training process.
    pause
    exit /b
)

echo.
echo Training completed.
pause

@echo off
chcp 65001 >nul

echo [1/3] Checking virtual environment (.venv)
if not exist ".venv" (
    echo     .venv not found, calling install_env.bat to setup environment...
    call install_env.bat
) else (
    echo     .venv already exists, skipping installation
)

echo.
echo [2/3] Activating virtual environment
call .venv\Scripts\activate.bat

echo.
echo [3/3] Generating data
python prepare_data.py

pause
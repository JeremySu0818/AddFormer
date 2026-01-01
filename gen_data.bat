@echo off
chcp 65001 >nul

python prepare_data.py --out_dir data --max_digits 3

pause
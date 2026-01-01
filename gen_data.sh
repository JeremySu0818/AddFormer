#!/usr/bin/env bash
set -e

python3 prepare_data.py \
  --out_dir data \
  --max_digits 3 

read -p "Press Enter to continue..."

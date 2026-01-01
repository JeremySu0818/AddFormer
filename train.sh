#!/usr/bin/env bash
set -e

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

python train_addformer.py

read -p "Press Enter to continue..."

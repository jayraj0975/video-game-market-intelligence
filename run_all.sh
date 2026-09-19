#!/usr/bin/env bash
# Reproduce every figure and report in this repository from the raw data.
set -euo pipefail
cd "$(dirname "$0")/src"
python download_data.py   # no-op if data/video_game_sales.csv already exists
python data_prep.py
python eda.py
python train.py

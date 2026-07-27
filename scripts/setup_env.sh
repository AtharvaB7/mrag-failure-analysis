#!/usr/bin/env bash
# One-time environment setup. Run from the project root.
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install qwen-vl-utils  # process_vision_info helper used by models/qwen2vl.py

echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Next: bash scripts/download_data.sh"

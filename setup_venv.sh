#!/usr/bin/env bash
# Isolated env: avoids the system cv2 (built for NumPy 1) that crashes SB3 under NumPy 2.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps

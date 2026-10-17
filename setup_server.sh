#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"
python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11 or 3.12"'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
echo 'Setup complete. Next: .venv/bin/python run.py download'

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python3 -c 'import shutil; assert shutil.disk_usage(".").free > 15*1024**3, "At least 15 GB free space required"'
bash setup_server.sh
.venv/bin/python -u run.py download
if [[ -f data/prepared/manifest.csv ]]; then
  echo 'Existing manifest preserved. Checking patient and duplicate separation.'
  .venv/bin/python -c 'from neurofusion.data import load_rows,ROOT; print(len(load_rows(ROOT / "data/prepared/manifest.csv")), "rows checked")'
else
  .venv/bin/python -u run.py prepare
fi
echo 'DATA READY. Training has not started. Select a free GPU with nvidia-smi.'

#!/usr/bin/env bash
# bash train_queue.sh PHYSICAL_GPU_INDEX [variant ...]
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu="${1:?Supply the physical GPU index from nvidia-smi}"
shift
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be an integer'; exit 2; }
uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
util="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
if (( free < 12000 )); then
  echo "GPU $gpu has less than 12 GB free; nothing started. Choose an available GPU."
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$uuid"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
[[ -x .venv/bin/python ]] || { echo 'Run setup_server.sh first'; exit 1; }
[[ -f data/prepared/manifest.csv ]] || { echo 'Run download and prepare first'; exit 1; }
if (( $# == 0 )); then set -- densenet full xception fusion convlstm; fi
for variant in "$@"; do
  case "$variant" in densenet|xception|fusion|convlstm|full) ;; *) echo "Unknown variant: $variant"; exit 2;; esac
done
mkdir -p runs
exec 9>runs/.training-queue.lock
flock -n 9 || { echo 'Another project queue is already running'; exit 1; }
queue="runs/queue-$(date +%Y%m%d-%H%M%S)"
mkdir "$queue"
echo "QUEUE: $queue; physical GPU: $gpu; UUID: $uuid; visible device: 0"
for variant in "$@"; do
  echo "START $variant $(date -Is)"
  .venv/bin/python -u run.py train --config "configs/$variant.yaml" --device 0 --output "$queue/$variant" > "$queue/$variant-train.log" 2>&1
  # Deliberately do not use the held-out test set for architecture decisions.
  .venv/bin/python -u run.py evaluate --weights "$queue/$variant/best.pt" --split val --device 0 --output "$queue/$variant-validation" > "$queue/$variant-validation.log" 2>&1
  echo "FINISHED $variant $(date -Is)"
done
echo "QUEUE COMPLETE: $queue. Compare validation results before final test evaluation."

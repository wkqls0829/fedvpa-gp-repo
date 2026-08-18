#!/bin/bash
# Stage 1 — federated variational preference selector.
#
# Usage:  bash scripts/run_selector.sh <10|100> [gpu_index]
# Example: bash scripts/run_selector.sh 100 0
#
# Override the data and checkpoint locations with DATA_ROOT / CKPT_DIR:
#   DATA_ROOT=/mnt/data CKPT_DIR=/mnt/ckpt bash scripts/run_selector.sh 10

set -euo pipefail

N="${1:-}"
GPU="${2:-0}"

if [[ "$N" != "10" && "$N" != "100" ]]; then
    echo "usage: $0 <10|100> [gpu_index]" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/outputs}"

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$DATA_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CFG="cfg/selector_n${N}.yaml"
LOG="$LOG_DIR/selector_n${N}.log"

echo "Stage 1 selector: K=$N clients, GPU $GPU"
echo "  config     : $CFG"
echo "  checkpoint : $CKPT_DIR/final_selector_n${N}.ckpt"
echo "  log        : $LOG"

python -u federatedscope/main.py \
    --cfg "$CFG" \
    device "$GPU" \
    data.root "$DATA_ROOT" \
    federate.save_to "$CKPT_DIR/selector_n${N}.ckpt" \
    2>&1 | tee "$LOG"

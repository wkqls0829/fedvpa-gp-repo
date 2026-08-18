#!/bin/bash
# Stage 2 — z-conditioned DPO on top of a trained Stage-1 selector.
#
# Usage:  bash scripts/run_rl.sh <10|100> [gpu_index]
# Example: bash scripts/run_rl.sh 100 0
#
# Requires OPENAI_API_KEY in the environment (or in .env) for the GPT
# win-rate evaluation. Run scripts/run_selector.sh with the same client
# count first.

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

SELECTOR_CKPT="$CKPT_DIR/final_selector_n${N}.ckpt"
if [[ ! -f "$SELECTOR_CKPT" ]]; then
    echo "Stage-1 checkpoint not found: $SELECTOR_CKPT" >&2
    echo "Run: bash scripts/run_selector.sh $N $GPU" >&2
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" && -f "$PROJECT_ROOT/.env" ]]; then
    set -a; . "$PROJECT_ROOT/.env"; set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "warning: OPENAI_API_KEY is unset; GPT win-rate evaluation will fail." >&2
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CFG="cfg/rl_n${N}.yaml"
SELECTOR_CFG="cfg/selector_n${N}.yaml"
LOG="$LOG_DIR/rl_n${N}.log"

echo "Stage 2 DPO: K=$N clients, GPU $GPU"
echo "  config       : $CFG"
echo "  selector cfg : $SELECTOR_CFG"
echo "  selector ckpt: $SELECTOR_CKPT"
echo "  log          : $LOG"

python -u federatedscope/llm/rlhf/main.py \
    --cfg "$CFG" \
    --selector-cfg-file "$SELECTOR_CFG" \
    device "$GPU" \
    data.root "$DATA_ROOT" \
    federate.save_to "$CKPT_DIR/rl_n${N}.ckpt" \
    llm.rlhf_selector_checkpoint "$SELECTOR_CKPT" \
    2>&1 | tee "$LOG"

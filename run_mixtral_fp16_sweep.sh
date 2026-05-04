#!/bin/bash
# Fill in the missing Mixtral fp16 cells in figures/full_sweep/.
# Mixtral fp16 is ~95 GB and doesn't fit on one 80 GB GPU, so we shard
# across two GPUs via device_map="auto".
#
# Usage:
#   ./run_mixtral_fp16_sweep.sh [GPU_PAIR]
#   default GPU_PAIR=6,7

set -u
cd "$(dirname "$0")"

GPU_PAIR="${1:-6,7}"
TRANSFER_MS=50.0
NUM_EXAMPLES=32
MAX_LENGTH=256

VENVPY="$PWD/venv_cxl/bin/python"
[[ -x "$VENVPY" ]] || VENVPY="$PWD/venv_cxl/bin/python3"
echo "interpreter: $VENVPY"
echo "GPUs: $GPU_PAIR (sharded)"

OUT_DIR="figures/full_sweep"
mkdir -p "$OUT_DIR"

GB_CAPS=(4 8 16 32)
BENCHMARKS=(wikitext_test mmlu gsm8k)

run_cell() {
    local gb=$1 bench=$2
    local tag="mixtral_8x7b_fp16_${gb}gb_${bench}"
    local out="$OUT_DIR/${tag}.json"
    local log="$OUT_DIR/${tag}.log"
    if [[ -f "$out" ]]; then
        echo "[skip] $tag (already done)"; return
    fi
    echo "[run  $(date +%H:%M:%S)] $tag (sharded across GPU_PAIR=$GPU_PAIR)"
    CUDA_VISIBLE_DEVICES=$GPU_PAIR "$VENVPY" prefetch_constrained.py \
        --model mixtral_8x7b --gpu 0 \
        --gpu_memory_gb "$gb" --quantization fp16 \
        --shard \
        --predict_topk_extra 4 \
        --num_examples $NUM_EXAMPLES --max_length $MAX_LENGTH \
        --benchmark "$bench" \
        --simulate_transfer_ms $TRANSFER_MS \
        --out "$out" \
        > "$log" 2>&1
    rc=$?
    [[ $rc -ne 0 ]] && echo "[fail] $tag (rc=$rc) — see $log"
}

for gb in "${GB_CAPS[@]}"; do
    for bench in "${BENCHMARKS[@]}"; do
        run_cell "$gb" "$bench"
    done
done

echo "[$(date)] Mixtral fp16 sweep complete"
ls -1 figures/full_sweep/mixtral_8x7b_fp16_*.json 2>/dev/null | wc -l | xargs -I {} echo "{} Mixtral fp16 JSONs total"

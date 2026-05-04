#!/bin/bash
# Full prefetch study: 3 models × 3 quantizations × 4 GPU caps × 3 benchmarks × 1 NVMe tier
# = up to 108 runs. Skips Mixtral fp16 (needs multi-GPU sharding).
#
# Usage:
#   ./run_full_sweep.sh [GPU_ID]
#
# Output: figures/full_sweep/<model>_<quant>_<gpu>gb_<bench>.json
# Master log: figures/full_sweep/master.log

set -u
cd "$(dirname "$0")"

GPU_ID="${1:-0}"
TRANSFER_MS=50.0          # NVMe-tier; change to 5.0 for CXL or omit for PCIe
NUM_EXAMPLES=32
MAX_LENGTH=256

VENVPY="$PWD/venv_cxl/bin/python"
[[ -x "$VENVPY" ]] || VENVPY="$PWD/venv_cxl/bin/python3"
[[ -x "$VENVPY" ]] || { echo "no venv interpreter"; exit 1; }
echo "interpreter: $VENVPY"
echo "GPU: $GPU_ID"

OUT_DIR="figures/full_sweep"
mkdir -p "$OUT_DIR"
MASTER="$OUT_DIR/master_gpu${GPU_ID}.log"
echo "[$(date)] starting full sweep" | tee -a "$MASTER"

run_one() {
    local model="$1"
    local quant="$2"
    local gb="$3"
    local bench="$4"
    local tag="${model}_${quant}_${gb}gb_${bench}"
    local out="$OUT_DIR/${tag}.json"
    local log="$OUT_DIR/${tag}.log"

    if [[ -f "$out" ]]; then
        echo "[skip $(date +%H:%M:%S)] $tag (already exists)" | tee -a "$MASTER"
        return
    fi

    echo "[run  $(date +%H:%M:%S)] $tag" | tee -a "$MASTER"
    CUDA_VISIBLE_DEVICES=$GPU_ID "$VENVPY" prefetch_constrained.py \
        --model "$model" --gpu 0 \
        --gpu_memory_gb "$gb" \
        --quantization "$quant" \
        --predict_topk_extra 4 \
        --num_examples $NUM_EXAMPLES --max_length $MAX_LENGTH \
        --benchmark "$bench" \
        --simulate_transfer_ms $TRANSFER_MS \
        --out "$out" \
        > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[fail $(date +%H:%M:%S)] $tag (rc=$rc)  -- see $log" | tee -a "$MASTER"
    fi
}

# Skip Mixtral fp16 (won't fit on single GPU). Skip combos where the quantized
# model can't fit on the host GPU at all (Mixtral 8bit needs ~47 GB).
MIXTRAL_QUANTS=("4bit" "8bit")          # fp16 dropped — needs multi-GPU
DEEPSEEK_QUANTS=("4bit" "8bit" "fp16")
QWEN_QUANTS=("4bit" "8bit" "fp16")

# Per model, per quantization: GB caps to study
GB_CAPS=(4 8 16 32)
BENCHMARKS=(wikitext_test mmlu gsm8k)

# === Mixtral ===
for q in "${MIXTRAL_QUANTS[@]}"; do
    for gb in "${GB_CAPS[@]}"; do
        # Skip nonsense combos (Mixtral 8bit needs ~47 GB; smaller caps just stress prefetcher more)
        for bench in "${BENCHMARKS[@]}"; do
            run_one mixtral_8x7b "$q" "$gb" "$bench"
        done
    done
done

# === DeepSeek ===
for q in "${DEEPSEEK_QUANTS[@]}"; do
    for gb in "${GB_CAPS[@]}"; do
        for bench in "${BENCHMARKS[@]}"; do
            run_one deepseek_moe_16b "$q" "$gb" "$bench"
        done
    done
done

# === Qwen ===
for q in "${QWEN_QUANTS[@]}"; do
    for gb in "${GB_CAPS[@]}"; do
        for bench in "${BENCHMARKS[@]}"; do
            run_one qwen1_5_moe_a2_7b "$q" "$gb" "$bench"
        done
    done
done

echo "[$(date)] sweep complete" | tee -a "$MASTER"
ls -lh "$OUT_DIR"/*.json | wc -l | xargs -I {} echo "{} JSON files written" | tee -a "$MASTER"

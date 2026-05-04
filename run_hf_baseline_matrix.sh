#!/bin/bash
# HF auto-offload baseline only — 144 cells.
# Each cell loads model with device_map="auto" + max_memory cap and runs real inference.
# Output: figures/real_baseline/<model>_<quant>_<gb>gb_<bench>.json
#
# Usage:  ./run_hf_baseline_matrix.sh [GPU_ID]

set -u
cd "$(dirname "$0")"

GPU_ID="${1:-0}"
NUM_EXAMPLES=16
MAX_LENGTH=256

VENVPY="$PWD/venv_cxl/bin/python"
[[ -x "$VENVPY" ]] || VENVPY="$PWD/venv_cxl/bin/python3"
echo "interpreter: $VENVPY  GPU: $GPU_ID"

OUT_DIR="figures/real_baseline"
mkdir -p "$OUT_DIR"
MASTER="$OUT_DIR/master_gpu${GPU_ID}.log"
echo "[$(date)] starting HF baseline sweep" | tee -a "$MASTER"

GB_CAPS=(4 8 16 32)
BENCHMARKS=(wikitext_test mmlu gsm8k)

# Approximate quantized-model GPU footprint (GiB). bnb 4/8-bit cannot split across
# CPU/disk and GPU, so cells where this exceeds the cap are infeasible.
# (fp16/bf16 freely split, so they don't need this check.)
declare -A MIN_GB
MIN_GB[mixtral_8x7b_4bit]=24
MIN_GB[mixtral_8x7b_8bit]=47
MIN_GB[deepseek_moe_16b_4bit]=10
MIN_GB[deepseek_moe_16b_8bit]=16
MIN_GB[qwen1_5_moe_a2_7b_4bit]=9
MIN_GB[qwen1_5_moe_a2_7b_8bit]=14

run_cell() {
    local model="$1"; local quant="$2"; local gb="$3"; local bench="$4"
    local tag="${model}_${quant}_${gb}gb_${bench}"
    local out="$OUT_DIR/${tag}.json"
    local log="$OUT_DIR/${tag}.log"
    if [[ -f "$out" ]]; then
        echo "[skip] $tag (already done)" | tee -a "$MASTER"; return
    fi
    # Infeasibility check for quantized models
    local key="${model}_${quant}"
    local min_gb="${MIN_GB[$key]:-0}"
    if [[ "$min_gb" != "0" ]] && (( gb < min_gb )); then
        echo "[skip-infeasible] $tag (needs >= ${min_gb} GiB; bnb $quant can't split)" | tee -a "$MASTER"
        return
    fi
    # bnb 4bit/8bit cannot offload to disk; use CPU. bf16/fp16 use disk.
    local target="disk"
    if [[ "$quant" == "4bit" || "$quant" == "8bit" ]]; then
        target="cpu"
    fi
    echo "[run  $(date +%H:%M:%S)] $tag (offload=$target)" | tee -a "$MASTER"
    local cache_dir="./hf_offload_cache_${tag}"
    CUDA_VISIBLE_DEVICES=$GPU_ID "$VENVPY" prefetch_real.py \
        --model "$model" --gpu 0 \
        --mode hf_offload \
        --quantization "$quant" \
        --gpu_memory_gb "$gb" \
        --benchmark "$bench" \
        --offload_target "$target" \
        --offload_dir "$cache_dir" \
        --num_examples $NUM_EXAMPLES --max_length $MAX_LENGTH \
        --out "$out" \
        > "$log" 2>&1
    rc=$?
    [[ $rc -ne 0 ]] && echo "[fail] $tag (rc=$rc) — see $log" | tee -a "$MASTER"
    rm -rf "$cache_dir"
}

for model in mixtral_8x7b deepseek_moe_16b qwen1_5_moe_a2_7b; do
    for quant in 4bit 8bit fp16; do          # match our_system: skip bf16
        for gb in "${GB_CAPS[@]}"; do
            for bench in "${BENCHMARKS[@]}"; do
                run_cell "$model" "$quant" "$gb" "$bench"
            done
        done
    done
done

echo "[$(date)] HF baseline sweep complete" | tee -a "$MASTER"
ls -1 "$OUT_DIR"/*.json 2>/dev/null | wc -l | xargs -I {} echo "{} JSON files written" | tee -a "$MASTER"

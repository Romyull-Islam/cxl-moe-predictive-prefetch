#!/bin/bash
# Phase 1: trace MMLU and GSM8K for all 3 MoE models, in parallel across 6 GPUs.
# Each job pins itself to one GPU via CUDA_VISIBLE_DEVICES.
# Logs go to phase1_logs/{model}_{benchmark}.log so you can tail individually.

set -u
cd "$(dirname "$0")"

# Use the venv interpreter directly so this works whether or not the parent
# shell activated venv_cxl. Falls back to python3 if `python` shim is missing.
VENVPY="$PWD/venv_cxl/bin/python"
[[ -x "$VENVPY" ]] || VENVPY="$PWD/venv_cxl/bin/python3"
[[ -x "$VENVPY" ]] || { echo "no venv interpreter at $PWD/venv_cxl/bin/"; exit 1; }
echo "using interpreter: $VENVPY"

mkdir -p phase1_logs

run_one() {
    local gpu=$1
    local model=$2
    local bench=$3
    local logfile="phase1_logs/${model}_${bench}.log"
    echo "[GPU $gpu] $model / $bench  -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu "$VENVPY" trace_extract.py \
        --model "$model" \
        --benchmark "$bench" \
        --num_examples 2000 \
        > "$logfile" 2>&1 &
}

# 6 jobs, one per GPU. Mixtral=24GB 4-bit, DeepSeek/Qwen smaller — all fit on one A100-80GB.
run_one 0 mixtral_8x7b       mmlu
run_one 1 mixtral_8x7b       gsm8k
run_one 2 deepseek_moe_16b   mmlu
run_one 3 deepseek_moe_16b   gsm8k
run_one 4 qwen1_5_moe_a2_7b  mmlu
run_one 5 qwen1_5_moe_a2_7b  gsm8k

echo
echo "Started 6 jobs. Tail any with:"
echo "  tail -f phase1_logs/mixtral_8x7b_mmlu.log"
echo
echo "Waiting for all jobs to finish..."
wait
echo "ALL TRACE JOBS DONE."
ls -lh */(*all_layers_raw.npz) 2>/dev/null || true

#!/bin/bash
# Phase 2: train one predictor per model on the union of WikiText + MMLU + GSM8K.
# Runs the 3 model trainings sequentially because each uses 6 GPUs.
# Logs go to phase2_logs/{model}.log

set -u
cd "$(dirname "$0")"

VENVPY="$PWD/venv_cxl/bin/python"
[[ -x "$VENVPY" ]] || VENVPY="$PWD/venv_cxl/bin/python3"
[[ -x "$VENVPY" ]] || { echo "no venv interpreter at $PWD/venv_cxl/bin/"; exit 1; }
echo "using interpreter: $VENVPY"

mkdir -p phase2_logs

# ---------- Mixtral (small expert pool — original recipe is enough) ----------
echo "[$(date)] Mixtral training start"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$VENVPY" train_predictor_multi.py \
    --model_key mixtral_8x7b \
    --npz_paths \
        mixtral_8x7b_wikitext_logs/mixtral_8x7b_wikitext_all_layers_raw.npz \
        mixtral_8x7b_mmlu_logs/mixtral_8x7b_mmlu_all_layers_raw.npz \
        mixtral_8x7b_gsm8k_logs/mixtral_8x7b_gsm8k_all_layers_raw.npz \
    --top_k 2 --lookahead_depth 4 \
    --hidden_dim 1024 --num_layers_mlp 3 \
    --epochs 15 --batch_size 32768 --num_gpus 6 \
    2>&1 | tee phase2_logs/mixtral_8x7b.log

# ---------- DeepSeek + Qwen in PARALLEL on 3 GPUs each ----------
echo "[$(date)] DeepSeek + Qwen training start (parallel, 3 GPUs each)"

CUDA_VISIBLE_DEVICES=0,1,2 "$VENVPY" train_predictor_multi.py \
    --model_key deepseek_moe_16b \
    --npz_paths \
        deepseek_moe_16b_wikitext_logs/deepseek_moe_16b_wikitext_all_layers_raw.npz \
        deepseek_moe_16b_mmlu_logs/deepseek_moe_16b_mmlu_all_layers_raw.npz \
        deepseek_moe_16b_gsm8k_logs/deepseek_moe_16b_gsm8k_all_layers_raw.npz \
    --top_k 6 --lookahead_depth 4 \
    --hidden_dim 2048 --num_layers_mlp 4 \
    --epochs 30 --batch_size 8192 --lr 5e-4 --num_gpus 3 \
    > phase2_logs/deepseek_moe_16b.log 2>&1 &
DS_PID=$!
echo "  DeepSeek pid=$DS_PID on GPUs 0,1,2 -> phase2_logs/deepseek_moe_16b.log"

CUDA_VISIBLE_DEVICES=3,4,5 "$VENVPY" train_predictor_multi.py \
    --model_key qwen1_5_moe_a2_7b \
    --npz_paths \
        qwen1_5_moe_a2_7b_wikitext_logs/qwen1_5_moe_a2_7b_wikitext_all_layers_raw.npz \
        qwen1_5_moe_a2_7b_mmlu_logs/qwen1_5_moe_a2_7b_mmlu_all_layers_raw.npz \
        qwen1_5_moe_a2_7b_gsm8k_logs/qwen1_5_moe_a2_7b_gsm8k_all_layers_raw.npz \
    --top_k 4 --lookahead_depth 4 \
    --hidden_dim 2048 --num_layers_mlp 4 \
    --epochs 30 --batch_size 8192 --lr 5e-4 --num_gpus 3 \
    > phase2_logs/qwen1_5_moe_a2_7b.log 2>&1 &
QW_PID=$!
echo "  Qwen pid=$QW_PID on GPUs 3,4,5 -> phase2_logs/qwen1_5_moe_a2_7b.log"

wait $DS_PID $QW_PID
echo "[$(date)] DeepSeek + Qwen training done"

echo "[$(date)] ALL TRAINING DONE."
ls -lh *_multi_logs/*.pt 2>/dev/null || true

# Predictive Expert Activation Modeling for Efficient MoE Inference

A lightweight, decoupled predictor that forecasts expert activations up to four
transformer layers ahead in Mixture-of-Experts (MoE) Large Language Models, enabling
asynchronous prefetching of expert weights to overlap NVMe-to-GPU transfers with
computation.

Per-token latency speedups at NVMe-tier expert storage (50 ms transfer cost) on a
single GPU with 4–32 GiB unified memory:

| Backbone           | Best speedup | Best config              |
|--------------------|:------------:|--------------------------|
| Mixtral-8x7B       | **3.30×**    | 8-bit / 16 GiB / MMLU    |
| DeepSeek-MoE-16B   | **3.01×**    | FP16 / 16 GiB / GSM8K    |
| Qwen1.5-MoE-A2.7B  | **3.57×**    | FP16 / 8 GiB / GSM8K     |

Cache hit rates ≥ 0.84 at the recommended operating point (`K = top_k + 4`).

## Setup

```bash
python3.12 -m venv venv_cxl
source venv_cxl/bin/activate
pip install torch transformers datasets bitsandbytes accelerate matplotlib tqdm numpy
```

A single A100 80 GB is sufficient for all three backbones at 4-bit and 8-bit
quantization. Mixtral fp16 requires sharding across 2 GPUs.

## Predictor checkpoints

The three trained predictors are committed alongside the code, under their
canonical paths:

```
mixtral_8x7b_multi_logs/mixtral_8x7b_multi_predictor_topk2_d4.pt
deepseek_moe_16b_multi_logs/deepseek_moe_16b_multi_predictor_topk6_d4.pt
qwen1_5_moe_a2_7b_multi_logs/qwen1_5_moe_a2_7b_multi_predictor_topk4_d4.pt
```

You can run the demo (Section 4 below) directly after `git clone`, no retraining
needed. To regenerate the checkpoints from scratch, use steps 1 and 2 below.

## Reproducing the paper

### 1. Phase 1 — trace extraction (~30 min per (model, benchmark) on one A100)

Captures per-token routing traces for {Mixtral, DeepSeek, Qwen} × {WikiText, MMLU, GSM8K}.

```bash
./run_phase1_traces.sh
```

Outputs go to `{model}_{benchmark}_logs/` as compressed `.npz` files.

### 2. Phase 2 — predictor training (~30 min per backbone, multi-GPU)

```bash
./run_phase2_train.sh
```

Outputs the per-backbone predictor checkpoints listed above.

### 3. Comprehensive sweep (~30 min, single A100)

```bash
./run_full_sweep.sh                     # 96 cells (Mixtral 4-bit/8-bit + DeepSeek + Qwen)
./run_mixtral_fp16_sweep.sh 6,7         # 12 cells (Mixtral fp16 sharded across GPUs 6,7)
python aggregate_sweep.py               # produces figures + CSVs
```

Outputs:
- `figures/full_sweep/*.json` — 108 per-cell raw simulator results
- `results_sweep_summary.csv`, `results_sweep_summary_wide.csv`
- `figures/sweep_speedup_heatmap_nvme.png`
- `figures/sweep_hitrate_vs_cap_combined.png`

### 4. Single-cell demo (the recorded video)

Two separate runs that each show real DeepSeek inference + the chosen cache policy's
per-token latency, plus one decoded model output for visual confirmation that the
generated text is identical between runs (= no quality degradation).

```bash
# Baseline (LRU only, no predictor)
python prefetch_constrained.py \
    --model deepseek_moe_16b --gpu 0 \
    --gpu_memory_gb 16 --quantization fp16 \
    --predict_topk_extra 4 \
    --num_examples 8 --max_length 256 \
    --benchmark gsm8k --simulate_transfer_ms 50.0 \
    --policy baseline --show_generation --gen_tokens 64 \
    --out demo_baseline.json

# Prefetched (predictor + confidence-PQ + LRU + hot preload)
python prefetch_constrained.py \
    --model deepseek_moe_16b --gpu 0 \
    --gpu_memory_gb 16 --quantization fp16 \
    --predict_topk_extra 4 \
    --num_examples 8 --max_length 256 \
    --benchmark gsm8k --simulate_transfer_ms 50.0 \
    --policy prefetched --show_generation --gen_tokens 64 \
    --out demo_prefetched.json
```

Expected: baseline ≈ 5600 ms/token at 0.42 hit rate; prefetched K=10 ≈ 1830 ms/token
at 0.89 hit rate; **3.01× speedup**. Same generated text in both runs.

## Repository layout

| Path | Purpose |
|---|---|
| `prefetch_constrained.py` | Main simulator (real inference + LRU/prefetch cache replay) |
| `prefetch_real.py` | HF auto-offload baseline + predictor hooks |
| `expert_predictor_topk.py` | Predictor architecture (shared trunk + 4 horizon heads) |
| `train_predictor_multi.py` | Trains per-backbone predictor on multi-benchmark traces |
| `trace_extract.py` | Captures per-layer routing traces during real inference |
| `aggregate_sweep.py` | Aggregates `figures/full_sweep/*.json` into CSVs and figures |
| `evaluate_*.py` | Recall@k evaluation utilities |
| `benchmark_latency.py` | Predictor latency micro-benchmark (Table IV data) |
| `plot_*.py` | Figure generators |
| `run_*.sh` | End-to-end pipeline drivers |
| `figures/full_sweep/*.json` | All 108 simulator outputs |
| `results_*.{json,csv}` | Tables III / IV / V data |
| `*_multi_logs/*.pt` | Trained per-backbone predictor checkpoints |

## Citation

```bibtex
@misc{islam2026predictive_moe,
  title  = {Predictive Expert Activation Modeling for Efficient MoE Inference},
  author = {Islam, Md Romyull},
  year   = {2026},
  note   = {CS 8347 final report, Kennesaw State University},
  url    = {https://github.com/Romyull-Islam/cxl-moe-predictive-prefetch}
}
```

## License

Not yet selected. The intended license is MIT (permissive, academic-friendly);
will be added before public release.

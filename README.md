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

---

## 1. Clone and set up

```bash
git clone https://github.com/Romyull-Islam/cxl-moe-predictive-prefetch.git
cd cxl-moe-predictive-prefetch

python3.12 -m venv venv_cxl
source venv_cxl/bin/activate
pip install torch transformers datasets bitsandbytes accelerate matplotlib tqdm numpy
```

### Hardware requirements

The recorded demo runs DeepSeek-MoE-16B at FP16 and needs **~32 GB of GPU memory**
to load the model. A single A100 (40 GB or 80 GB) or any GPU with ≥40 GB free works.

Smaller consumer GPUs (24 GB or below) will OOM at FP16 — switch to
`--quantization 4bit` (~10 GB needed) or `--quantization 8bit` (~17 GB needed)
in the demo commands below for those.

The full sweep (`./run_full_sweep.sh`) can run on a single A100; **Mixtral FP16**
specifically requires sharding across 2 GPUs.

### First run downloads ~32 GB

The first invocation will pull DeepSeek-MoE-16B's checkpoint shards from
HuggingFace into `~/.cache/huggingface/`. Takes a few minutes on a fast network;
subsequent runs reuse the cache and load in ~30 s.

---

## 2. Quick demo (no retraining needed)

The repo ships with the trained DeepSeek predictor at
`deepseek_moe_16b_multi_logs/deepseek_moe_16b_multi_predictor_topk6_d4.pt`, so
the demo runs immediately after `git clone` + setup.

Run two separate inferences — one with the LRU baseline, one with our predictor —
and compare per-token latency:

```bash
# A) BASELINE: LRU expert cache, no predictor
./venv_cxl/bin/python prefetch_constrained.py \
    --model deepseek_moe_16b --gpu 0 \
    --gpu_memory_gb 16 --quantization fp16 \
    --predict_topk_extra 4 \
    --num_examples 8 --max_length 256 \
    --benchmark gsm8k --simulate_transfer_ms 50.0 \
    --policy baseline --show_generation --gen_tokens 64 \
    --out demo_baseline.json

# B) PREFETCHED: predictor + confidence-PQ + LRU + hot preload
./venv_cxl/bin/python prefetch_constrained.py \
    --model deepseek_moe_16b --gpu 0 \
    --gpu_memory_gb 16 --quantization fp16 \
    --predict_topk_extra 4 \
    --num_examples 8 --max_length 256 \
    --benchmark gsm8k --simulate_transfer_ms 50.0 \
    --policy prefetched --show_generation --gen_tokens 64 \
    --out demo_prefetched.json
```

Expected output:

| | Baseline | Prefetched (best K=10) |
|---|---:|---:|
| Per-token latency (mean) | ~5600 ms | ~1830 ms |
| Cache hit rate | 0.42 | 0.89 |
| **Speedup** | | **~3.0×** |

Both runs print one decoded model continuation at the end. The text is identical
in both runs (greedy decoding on the same model + post-hoc cache simulation),
which is the visual proof that the cache policy never affects model output.

---

## 3. Reproducing the paper (full pipeline, ~3 hours total)

The four phases below regenerate every number in the paper from scratch. Skip to
phase 3 if you only want to re-run the simulator with the included DeepSeek
checkpoint; phases 1 and 2 are needed for the Mixtral and Qwen results because
those checkpoints are not committed (size).

### 3.1 Phase 1 — trace extraction (~30 min per (model, benchmark) on one A100)

Captures per-token routing traces for {Mixtral, DeepSeek, Qwen} × {WikiText, MMLU, GSM8K}.

```bash
./run_phase1_traces.sh
```

Outputs go to `{model}_{benchmark}_logs/` as compressed `.npz` files.

### 3.2 Phase 2 — predictor training (~30 min per backbone, multi-GPU)

```bash
./run_phase2_train.sh
```

Outputs the per-backbone predictor checkpoints. Mixtral and Qwen checkpoints
land at `mixtral_8x7b_multi_logs/...pt` and `qwen1_5_moe_a2_7b_multi_logs/...pt`
respectively. The DeepSeek checkpoint already in the repo will be overwritten
with a fresh-trained one.

### 3.3 Phase 3 — comprehensive sweep (~30 min, single A100)

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

---

## 4. Repository layout

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
| `deepseek_moe_16b_multi_logs/*.pt` | Trained DeepSeek predictor checkpoint (demo) |

---



## License

Not yet selected. The intended license is MIT (permissive, academic-friendly);
will be added before public release.

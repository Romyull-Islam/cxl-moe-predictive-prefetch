"""
Evaluate each multi-benchmark predictor against each individual benchmark NPZ
to get a per-(model, benchmark) Recall@K breakdown. Combined "all benchmarks"
result already lives in results_multi.json from extract_test_results.py.

Usage:
  python evaluate_per_benchmark.py
  python evaluate_per_benchmark.py --max_tokens_per_layer 50000 --num_gpus 1
"""

import argparse
import json
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from expert_predictor_topk import GlobalMultiStepDataset, GlobalMultiStepPredictor


CONFIGS = {
    "mixtral_8x7b": dict(
        ckpt="mixtral_8x7b_multi_logs/mixtral_8x7b_multi_predictor_topk2_d4.pt",
        top_k=2, hidden_dim=1024, num_layers_mlp=3,
        benchmarks={
            "wikitext": "mixtral_8x7b_wikitext_logs/mixtral_8x7b_wikitext_all_layers_raw.npz",
            "mmlu":     "mixtral_8x7b_mmlu_logs/mixtral_8x7b_mmlu_all_layers_raw.npz",
            "gsm8k":    "mixtral_8x7b_gsm8k_logs/mixtral_8x7b_gsm8k_all_layers_raw.npz",
        },
    ),
    "deepseek_moe_16b": dict(
        ckpt="deepseek_moe_16b_multi_logs/deepseek_moe_16b_multi_predictor_topk6_d4.pt",
        top_k=6, hidden_dim=2048, num_layers_mlp=4,
        benchmarks={
            "wikitext": "deepseek_moe_16b_wikitext_logs/deepseek_moe_16b_wikitext_all_layers_raw.npz",
            "mmlu":     "deepseek_moe_16b_mmlu_logs/deepseek_moe_16b_mmlu_all_layers_raw.npz",
            "gsm8k":    "deepseek_moe_16b_gsm8k_logs/deepseek_moe_16b_gsm8k_all_layers_raw.npz",
        },
    ),
    "qwen1_5_moe_a2_7b": dict(
        ckpt="qwen1_5_moe_a2_7b_multi_logs/qwen1_5_moe_a2_7b_multi_predictor_topk4_d4.pt",
        top_k=4, hidden_dim=2048, num_layers_mlp=4,
        benchmarks={
            "wikitext": "qwen1_5_moe_a2_7b_wikitext_logs/qwen1_5_moe_a2_7b_wikitext_all_layers_raw.npz",
            "mmlu":     "qwen1_5_moe_a2_7b_mmlu_logs/qwen1_5_moe_a2_7b_mmlu_all_layers_raw.npz",
            "gsm8k":    "qwen1_5_moe_a2_7b_gsm8k_logs/qwen1_5_moe_a2_7b_gsm8k_all_layers_raw.npz",
        },
    ),
}


def evaluate(model, npz_path, top_k, lookahead_depth, max_tokens_per_layer,
             batch_size, device, predict_topk):
    dataset = GlobalMultiStepDataset(
        npz_path=npz_path,
        lookahead_depth=lookahead_depth,
        top_k=top_k,
        max_tokens_per_layer=max_tokens_per_layer,
        shuffle=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True,
                        multiprocessing_context="fork")

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = torch.zeros(1, device=device)
    total_samples = 0
    coverage = {d: {k: torch.zeros(1, device=device) for k in predict_topk}
                for d in range(lookahead_depth)}

    with torch.no_grad():
        for h, lid, y in tqdm(loader, desc=os.path.basename(npz_path), leave=False):
            h, lid, y = h.to(device, non_blocking=True), lid.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(h, lid)
            B, D, E = logits.shape
            tgt = torch.zeros(B, D, E, device=device)
            tgt.scatter_(2, y, 1.0)
            total_loss += criterion(logits, tgt) * B
            total_samples += B
            for d in range(D):
                for k in predict_topk:
                    pred = logits[:, d, :].topk(k, dim=-1).indices
                    matched = (pred.unsqueeze(1) == y[:, d, :].unsqueeze(2)).any(dim=2)
                    coverage[d][k] += matched.all(dim=1).sum()

    return dict(
        n_samples=int(total_samples),
        loss=float((total_loss / max(1, total_samples)).item()),
        recall={
            f"d+{d+1}": {f"@{k}": float((coverage[d][k] / max(1, total_samples)).item())
                        for k in predict_topk}
            for d in range(lookahead_depth)
        },
    )


def load_predictor(cfg, npz_for_meta, device):
    meta = np.load(npz_for_meta, mmap_mode="r")
    hidden_size = int(meta["hidden_size"][0])
    num_experts = int(meta["num_experts_per_layer"][0])
    num_layers = int(meta["num_layers"][0])
    model = GlobalMultiStepPredictor(
        d_model=hidden_size, num_experts=num_experts,
        num_layers_total=num_layers, lookahead_depth=4,
        layer_embed_dim=32,
        hidden_dim=cfg["hidden_dim"], num_layers_mlp=cfg["num_layers_mlp"],
    )
    model.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu"))
    return model.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_tokens_per_layer", type=int, default=50000,
                   help="Subsample to keep eval fast. Set to a large number or None for all.")
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results_per_benchmark.json")
    args = p.parse_args()

    all_results = []
    for model_key, cfg in CONFIGS.items():
        # First benchmark NPZ provides metadata for predictor init
        first_bench_npz = next(iter(cfg["benchmarks"].values()))
        if not os.path.exists(cfg["ckpt"]) or not os.path.exists(first_bench_npz):
            print(f"[skip] {model_key}: missing checkpoint or npz")
            continue

        predictor = load_predictor(cfg, first_bench_npz, args.device)
        predict_topk = (cfg["top_k"], cfg["top_k"] + 2, cfg["top_k"] + 4)
        print(f"\n{'='*78}\n{model_key}\n{'='*78}")

        per_bench = {}
        for bench_name, bench_npz in cfg["benchmarks"].items():
            if not os.path.exists(bench_npz):
                print(f"  [skip] {bench_name}: npz missing")
                continue
            print(f"  evaluating on {bench_name} ...")
            rec = evaluate(predictor, bench_npz, cfg["top_k"], 4,
                           args.max_tokens_per_layer, args.batch_size,
                           args.device, predict_topk)
            per_bench[bench_name] = rec
            print(f"    n={rec['n_samples']:,}  loss={rec['loss']:.4f}")
            for h, ks in rec["recall"].items():
                print(f"      {h}: " + "  ".join(f"@{k.split('@')[1]}={v:.4f}" for k, v in ks.items()))

        all_results.append(dict(
            model=model_key, top_k=cfg["top_k"], per_benchmark=per_bench,
        ))

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {args.out}")

    # Pretty cross-table: one row per (model, benchmark, horizon) at @top_k+2 (the headline metric)
    print("\n" + "=" * 96)
    print(f"{'Model':<22} {'Benchmark':<12} {'Horizon':<10} "
          f"{'Recall@k':<12} {'Recall@k+2':<14} {'Recall@k+4':<14}")
    print("=" * 96)
    for r in all_results:
        for bname, br in r["per_benchmark"].items():
            for h, ks in br["recall"].items():
                keys = sorted(ks.keys(), key=lambda s: int(s[1:]))
                vals = [ks[k] for k in keys]
                print(f"{r['model']:<22} {bname:<12} {h:<10} "
                      f"{vals[0]:<12.4f} {vals[1]:<14.4f} {vals[2]:<14.4f}")
            print("-" * 96)


if __name__ == "__main__":
    main()

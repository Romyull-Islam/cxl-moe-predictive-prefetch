"""
Compare two real inference systems running the same MoE model:

  - HF auto-offload: device_map='auto' with max_memory cap; accelerate manages
    CPU<->GPU expert paging on real hardware.
    Inputs: figures/real_baseline/<model>_<quant>_<gb>gb_<bench>.json
            (from prefetch_real.py --mode hf_offload)

  - Our system: real inference with predictor-driven prefetch and LRU cache;
    per-token latency derived from CUDA-measured per-layer/transfer timings
    walked through the real demand trace at a configured cache capacity.
    Inputs: figures/full_sweep/<model>_<quant>_<gb>gb_<bench>.json
            (from prefetch_constrained.py)

Both are real inference. The 'cache' difference is in implementation, not
methodology: HF actually pages weights via accelerate hooks; our system runs
the model fully on GPU and counts which experts a hypothetical N-slot cache
would hold (timings come from real CUDA events).

Outputs:
  - results_real_vs_sim.csv         — flat join with all metrics per cell
  - prints comparison table on stdout

For each (model, quant, gpu_memory_gb, benchmark) cell where both files exist:
  - HF tok/s, ms/token  (real engine baseline)
  - Our system: ms/token at predictor-best K  (real-measured timing constants)
  - Predicted speedup of our system over an equivalent on-demand baseline
  - real/our_baseline ratio  (≈1.0 means our LRU-baseline timing matches HF reality)
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


TAG_RE = re.compile(r"^(?P<model>[a-z0-9_]+?)_(?P<quant>4bit|8bit|fp16|bf16)_(?P<gb>\d+)gb_(?P<bench>wikitext_(?:train|test)|mmlu|gsm8k)$")


def parse_real_dir(path):
    out = {}
    for jp in sorted(glob.glob(os.path.join(path, "*.json"))):
        tag = os.path.splitext(os.path.basename(jp))[0]
        m = TAG_RE.match(tag)
        if not m:
            continue
        with open(jp) as f:
            data = json.load(f)
        wall = data.get("wall_seconds")
        toks = data.get("tokens")
        if wall is None or toks is None:
            # newer JSON may nest under results.hf_offload
            r = data.get("results", {}).get("hf_offload", {})
            wall = r.get("wall_seconds")
            toks = r.get("tokens")
        if wall is None or toks is None:
            continue
        key = (m.group("model"), m.group("quant"), int(m.group("gb")), m.group("bench"))
        out[key] = dict(real_wall_s=wall, real_tokens=toks,
                        real_tok_per_s=toks/wall,
                        real_ms_per_token=wall*1000.0/toks)
    return out


def parse_sim_dir(path):
    """For each cell, return (our_baseline_ms_per_tok, best our_prefetched_ms_per_tok)."""
    out = {}
    for jp in sorted(glob.glob(os.path.join(path, "*.json"))):
        tag = os.path.splitext(os.path.basename(jp))[0]
        m = TAG_RE.match(tag)
        if not m:
            continue
        with open(jp) as f:
            data = json.load(f)
        if "baseline" not in data or "prefetched_by_K" not in data:
            continue
        sim_base_ms = data["baseline"]["mean_latency_ms"]
        # best K = lowest prefetched_latency_ms
        best_K, best_ms = None, float("inf")
        for K_str, r in data["prefetched_by_K"].items():
            if r["mean_latency_ms"] < best_ms:
                best_ms = r["mean_latency_ms"]
                best_K = int(K_str)
        if best_K is None:
            continue
        sim_speedup = sim_base_ms / best_ms
        key = (m.group("model"), m.group("quant"), int(m.group("gb")), m.group("bench"))
        out[key] = dict(
            our_baseline_ms_per_tok=sim_base_ms,
            our_prefetched_ms_per_tok=best_ms,
            our_predicted_speedup=sim_speedup,
            our_best_K=best_K,
            our_best_hit_rate=data["prefetched_by_K"][str(best_K)]["cache_hit_rate"],
        )
    return out


def write_csv(rows, out_path):
    if not rows:
        print("No joined rows to write."); return
    cols = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows -> {out_path}")


def print_summary(rows):
    if not rows: return
    headers = ["Model", "Q", "GB", "Bench", "HF_tok/s", "HF_ms/tok",
               "Our_BL_ms/tok", "Our_Pref_ms/tok", "Our_speedup",
               "real/our_BL", "Proj_real_ms/tok"]
    widths = [22, 6, 4, 16, 11, 12, 14, 16, 12, 12, 18]
    print()
    print("=" * sum(widths))
    print("HF AUTO-OFFLOAD  vs  OUR SYSTEM (both real inference, different cache strategy)")
    print(f"  HF_ms/tok = real wall ms / token  (HF auto-offload baseline)")
    print(f"  Our_BL_ms/tok = our system's per-token latency without predictor (LRU only)")
    print(f"  Our_Pref_ms/tok = our system's per-token latency with predictor at best K")
    print(f"  Our_speedup = Our_BL / Our_Pref  (predictor's contribution in our cache)")
    print(f"  real/our_BL = HF_ms / Our_BL  (≈1.0 means our LRU baseline matches HF reality)")
    print(f"  Proj_real_ms/tok = HF_ms / Our_speedup  (HF latency × predictor speedup ratio)")
    print("=" * sum(widths))
    line = "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(line)
    print("-" * sum(widths))

    rows_sorted = sorted(rows, key=lambda r: (r["model"], r["quantization"],
                                                r["gpu_memory_gb"], r["benchmark"]))
    for r in rows_sorted:
        ratio = r["real_ms_per_token"] / max(0.001, r["our_baseline_ms_per_tok"])
        proj_real = r["real_ms_per_token"] / max(0.001, r["our_predicted_speedup"])
        cells = [
            r["model"], r["quantization"], str(r["gpu_memory_gb"]), r["benchmark"],
            f"{r['real_tok_per_s']:.1f}",
            f"{r['real_ms_per_token']:.2f}",
            f"{r['our_baseline_ms_per_tok']:.2f}",
            f"{r['our_prefetched_ms_per_tok']:.2f}",
            f"{r['our_predicted_speedup']:.2f}×",
            f"{ratio:.2f}",
            f"{proj_real:.2f}",
        ]
        print("".join(f"{c:<{w}}" for c, w in zip(cells, widths)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", default="figures/real_baseline")
    p.add_argument("--sim_dir", default="figures/full_sweep")
    p.add_argument("--out_csv", default="results_real_vs_sim.csv")
    args = p.parse_args()

    real = parse_real_dir(args.real_dir)
    sim = parse_sim_dir(args.sim_dir)

    rows = []
    for key, real_stats in real.items():
        sim_stats = sim.get(key)
        if sim_stats is None:
            continue
        m, q, gb, bench = key
        row = dict(model=m, quantization=q, gpu_memory_gb=gb, benchmark=bench)
        row.update(real_stats)
        row.update(sim_stats)
        # Derived
        row["real_to_sim_baseline_ratio"] = real_stats["real_ms_per_token"] / max(0.001, sim_stats["our_baseline_ms_per_tok"])
        row["projected_real_prefetched_ms_per_token"] = real_stats["real_ms_per_token"] / max(0.001, sim_stats["our_predicted_speedup"])
        rows.append(row)

    print(f"Real cells: {len(real)}, sim cells: {len(sim)}, joined: {len(rows)}")
    if not rows:
        print("No (model, quant, gb, bench) cells overlap — check directory contents.")
        return

    write_csv(rows, args.out_csv)
    print_summary(rows)


if __name__ == "__main__":
    main()

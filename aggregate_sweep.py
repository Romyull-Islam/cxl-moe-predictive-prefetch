"""
Aggregate prefetch sweep results from figures/full_sweep/*.json into:
  - results_sweep_summary.csv             — flat row per (model, quant, gb, bench, K)
  - results_sweep_summary_wide.csv        — wide form, one row per run
  - figures/sweep_speedup_heatmap_nvme.png — combined 3-panel speedup heatmap (NVMe-tier)
  - figures/sweep_hitrate_vs_cap_*.png     — per-model hit-rate vs GPU cap
  - figures/sweep_K_curves_*.png           — per-model speedup vs K, panels per (quant, cap)

Run after run_full_sweep.sh completes (or partially — script handles missing).
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


TAG_RE = re.compile(r"^(?P<model>[a-z0-9_]+?)_(?P<quant>4bit|8bit|fp16)_(?P<gb>\d+)gb_(?P<bench>wikitext_(?:train|test)|mmlu|gsm8k)$")

BENCHMARKS = ("wikitext_test", "mmlu", "gsm8k")
QUANT_ORDER = ("4bit", "8bit", "fp16")
MODEL_ORDER = ("mixtral_8x7b", "deepseek_moe_16b", "qwen1_5_moe_a2_7b")
MODEL_LABEL = {
    "mixtral_8x7b": "Mixtral-8x7B",
    "deepseek_moe_16b": "DeepSeek-MoE-16B",
    "qwen1_5_moe_a2_7b": "Qwen1.5-MoE-A2.7B",
}


def parse_results(in_dir):
    """Walk the JSON files and yield flat per-K records."""
    records = []
    for path in sorted(glob.glob(os.path.join(in_dir, "*.json"))):
        tag = os.path.splitext(os.path.basename(path))[0]
        m = TAG_RE.match(tag)
        if not m:
            continue
        with open(path) as f:
            data = json.load(f)
        base = data["baseline"]
        speedups = data.get("speedup_by_K", {})
        prefs = data.get("prefetched_by_K", {})
        for K_str, pref in prefs.items():
            speedup = float(speedups.get(K_str, base["mean_latency_ms"] / pref["mean_latency_ms"]))
            row = dict(
                model=m.group("model"),
                quantization=m.group("quant"),
                gpu_memory_gb=int(m.group("gb")),
                benchmark=m.group("bench"),
                prefetch_K=int(K_str),
                tokens=data.get("tokens", 0),
                cache_capacity=data.get("cache_capacity", 0),
                total_experts=data.get("total_experts", 0),
                cache_pct=data["cache_capacity"] / max(1, data["total_experts"]),
                # measured constants
                predictor_ms=data["measured"]["predictor_ms"],
                layer_compute_ms=data["measured"]["layer_compute_ms"],
                transfer_ms=data["measured"]["transfer_ms"],
                # baseline
                baseline_latency_ms=base["mean_latency_ms"],
                baseline_hit_rate=base["cache_hit_rate"],
                baseline_misses=base["cache_misses"],
                # prefetched at this K
                prefetched_latency_ms=pref["mean_latency_ms"],
                prefetched_hit_rate=pref["cache_hit_rate"],
                prefetched_misses=pref["cache_misses"],
                cache_evictions=pref["cache_evictions"],
                total_prefetches=pref["total_prefetches"],
                dedup_skipped=pref.get("dedup_skipped", 0),
                wasted_prefetches=pref["wasted_prefetches"],
                waste_rate=pref["wasted_prefetches"] / max(1, pref["total_prefetches"]),
                miss_reduction=base["cache_misses"] / max(1, pref["cache_misses"]),
                hit_rate_gain=pref["cache_hit_rate"] - base["cache_hit_rate"],
                speedup=speedup,
            )
            records.append(row)
    return records


def write_csv(records, out_path):
    if not records:
        print("No records to write.")
        return
    cols = list(records[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"Wrote {len(records)} rows -> {out_path}")


def best_K_records(records):
    """Return one record per (model, quant, gb, bench) — the K that maximizes speedup."""
    best = {}
    for r in records:
        key = (r["model"], r["quantization"], r["gpu_memory_gb"], r["benchmark"])
        if key not in best or r["speedup"] > best[key]["speedup"]:
            best[key] = r
    return list(best.values())


def plot_speedup_heatmap(records, out_dir):
    """Per model: rows=GB caps, cols=quant, value=speedup at best K (averaged across benchmarks)."""
    os.makedirs(out_dir, exist_ok=True)
    by_model = defaultdict(list)
    for r in best_K_records(records):
        by_model[r["model"]].append(r)

    for model, rows in by_model.items():
        gbs = sorted(set(r["gpu_memory_gb"] for r in rows))
        quants = sorted(set(r["quantization"] for r in rows), key=lambda q: ["4bit", "8bit", "fp16"].index(q))
        if not gbs or not quants:
            continue
        mat = np.full((len(gbs), len(quants)), np.nan)
        for r in rows:
            i = gbs.index(r["gpu_memory_gb"])
            j = quants.index(r["quantization"])
            cur = mat[i, j]
            mat[i, j] = r["speedup"] if np.isnan(cur) else (cur + r["speedup"]) / 2

        fig, ax = plt.subplots(figsize=(5 + 0.6 * len(quants), 4))
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(quants)), labels=quants)
        ax.set_yticks(range(len(gbs)), labels=[f"{g} GiB" for g in gbs])
        ax.set_xlabel("Quantization")
        ax.set_ylabel("GPU memory cap")
        ax.set_title(f"{model}: speedup at best K (mean over benchmarks)")
        for i in range(len(gbs)):
            for j in range(len(quants)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}×", ha="center", va="center",
                            color="white" if mat[i, j] < 2.0 else "black", fontsize=10)
        plt.colorbar(im, ax=ax, label="Speedup ×")
        plt.tight_layout()
        path = os.path.join(out_dir, f"sweep_speedup_heatmap_{model}.png")
        plt.savefig(path, dpi=200); plt.close()
        print(f"saved -> {path}")


def _hitrate_agg_for_model(rows):
    """Returns (agg_dict, quants_in_order). agg keys = (quant, gb), values = list of (base, pref)."""
    agg = defaultdict(list)
    for r in rows:
        agg[(r["quantization"], r["gpu_memory_gb"])].append(
            (r["baseline_hit_rate"], r["prefetched_hit_rate"])
        )
    quants = sorted(set(k[0] for k in agg), key=lambda q: ["4bit", "8bit", "fp16"].index(q))
    return agg, quants


def plot_hitrate_vs_cap(records, out_dir):
    by_model = defaultdict(list)
    for r in best_K_records(records):
        by_model[r["model"]].append(r)

    markers = {"4bit": "o", "8bit": "s", "fp16": "^"}

    # ---- per-model standalone panels ----
    for model, rows in by_model.items():
        agg, quants = _hitrate_agg_for_model(rows)
        if not agg:
            continue
        plt.figure(figsize=(7, 4.5))
        for q in quants:
            xs = sorted(k[1] for k in agg if k[0] == q)
            ys_pref = [np.mean([h[1] for h in agg[(q, x)]]) for x in xs]
            ys_base = [np.mean([h[0] for h in agg[(q, x)]]) for x in xs]
            plt.plot(xs, ys_pref, marker=markers[q], linestyle="-", label=f"{q} prefetched")
            plt.plot(xs, ys_base, marker=markers[q], linestyle="--", alpha=0.5, label=f"{q} baseline")
        plt.xlabel("GPU memory cap (GiB)")
        plt.ylabel("Cache hit rate")
        plt.ylim(0.0, 1.0)
        plt.title(f"{model}: hit rate vs GPU cap (mean over benchmarks)")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"sweep_hitrate_vs_cap_{model}.png")
        plt.savefig(path, dpi=200); plt.close()
        print(f"saved -> {path}")

    # ---- combined 3-panel figure in MODEL_ORDER ----
    panels = [m for m in MODEL_ORDER if m in by_model]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5 * len(panels), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    handles_by_label = {}
    for ax, model in zip(axes, panels):
        agg, quants = _hitrate_agg_for_model(by_model[model])
        for q in quants:
            xs = sorted(k[1] for k in agg if k[0] == q)
            ys_pref = [np.mean([h[1] for h in agg[(q, x)]]) for x in xs]
            ys_base = [np.mean([h[0] for h in agg[(q, x)]]) for x in xs]
            l_pref, = ax.plot(xs, ys_pref, marker=markers[q], linestyle="-",
                              label=f"{q} prefetched")
            l_base, = ax.plot(xs, ys_base, marker=markers[q], linestyle="--",
                              alpha=0.5, label=f"{q} baseline")
            handles_by_label[f"{q} prefetched"] = l_pref
            handles_by_label[f"{q} baseline"] = l_base
        ax.set_xlabel("GPU memory cap (GiB)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(MODEL_LABEL.get(model, model))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cache hit rate")

    # Single legend for the whole figure, ordered by quant then variant
    quant_order = ["4bit", "8bit", "fp16"]
    ordered_labels = [f"{q} {kind}" for q in quant_order for kind in ("prefetched", "baseline")]
    handles = [handles_by_label[l] for l in ordered_labels if l in handles_by_label]
    labels = [l for l in ordered_labels if l in handles_by_label]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cache hit rate vs GPU memory cap (mean over benchmarks, best K)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = os.path.join(out_dir, "sweep_hitrate_vs_cap_combined.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"saved -> {path}")


def plot_K_curves(records, out_dir):
    """For each model, a grid of (quant rows × GB cols) showing speedup vs K, line per benchmark."""
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    for model, rows in by_model.items():
        quants = sorted(set(r["quantization"] for r in rows), key=lambda q: ["4bit", "8bit", "fp16"].index(q))
        gbs = sorted(set(r["gpu_memory_gb"] for r in rows))
        if not quants or not gbs:
            continue
        fig, axes = plt.subplots(len(quants), len(gbs),
                                  figsize=(3 * len(gbs), 2.8 * len(quants)),
                                  sharey=True, squeeze=False)
        for i, q in enumerate(quants):
            for j, gb in enumerate(gbs):
                ax = axes[i][j]
                for bench in ["wikitext_test", "mmlu", "gsm8k"]:
                    cell = sorted([r for r in rows
                                    if r["quantization"] == q and r["gpu_memory_gb"] == gb
                                    and r["benchmark"] == bench],
                                   key=lambda r: r["prefetch_K"])
                    if not cell:
                        continue
                    ax.plot([r["prefetch_K"] for r in cell],
                             [r["speedup"] for r in cell],
                             marker="o", label=bench)
                ax.set_title(f"{q} / {gb} GiB", fontsize=9)
                ax.grid(alpha=0.3)
                if i == len(quants) - 1: ax.set_xlabel("Prefetch K")
                if j == 0: ax.set_ylabel("Speedup ×")
        axes[0][0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"{model}: speedup vs prefetch-K (per quant × cap × benchmark)")
        plt.tight_layout()
        path = os.path.join(out_dir, f"sweep_K_curves_{model}.png")
        plt.savefig(path, dpi=200); plt.close()
        print(f"saved -> {path}")


def print_summary_table(records):
    """Best-K only — the manuscript headline view."""
    best = best_K_records(records)
    print("\n" + "=" * 110)
    print("HEADLINE TABLE — best prefetch K per (model, quant, cap, benchmark)")
    print("=" * 110)
    print(f"{'Model':<22}{'Quant':<7}{'GB cap':<8}{'Bench':<16}{'Best K':<8}{'Hit-rate':<10}{'Speedup':<10}{'Waste':<8}")
    print("-" * 110)
    for r in sorted(best, key=lambda r: (r["model"], r["quantization"], r["gpu_memory_gb"], r["benchmark"])):
        print(f"{r['model']:<22}{r['quantization']:<7}{r['gpu_memory_gb']:<8}{r['benchmark']:<16}"
              f"{r['prefetch_K']:<8}{r['prefetched_hit_rate']:<10.4f}{r['speedup']:<10.2f}{r['waste_rate']:<8.2f}")


def print_detailed_per_K_table(records):
    """Per-K detail table — every metric × every K, ONE TABLE PER MODEL."""
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    for model in sorted(by_model.keys()):
        model_records = by_model[model]
        # Group by (quant, cap, bench)
        grouped = defaultdict(dict)
        for r in model_records:
            key = (r["quantization"], r["gpu_memory_gb"], r["benchmark"])
            grouped[key][r["prefetch_K"]] = r

        # Get the K values used for this model
        all_Ks = sorted(set(r["prefetch_K"] for r in model_records))
        K_labels = [f"K={k}" for k in all_Ks]

        header = (f"{'Q':<6}{'GB':<5}{'Bench':<16}"
                  f"{'BL_lat':<9}{'BL_hit':<8}")
        for kl in K_labels:
            header += f"{kl+'_lat':<10}{kl+'_hit':<9}{kl+'_miss':<9}{kl+'_sp':<8}{kl+'_wst':<8}"
        line_w = len(header)

        # Infer the routing top_k from the smallest K we tested (it equals top_k by convention)
        actual_top_k = min(all_Ks)
        print("\n" + "=" * line_w)
        print(f"DETAILED PER-K TABLE — {model}  (top_k={actual_top_k}, prefetched K ∈ {all_Ks})")
        print(f"  BL = baseline LRU; lat ms; hit = cache hit rate; miss = cache misses;")
        print(f"  sp = speedup vs baseline; wst = waste rate (predictor precision = 1-wst)")
        print("=" * line_w)
        print(header)
        print("-" * line_w)

        for key in sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2])):
            q, gb, bench = key
            any_K = next(iter(grouped[key].values()))
            line = (f"{q:<6}{gb:<5}{bench:<16}"
                    f"{any_K['baseline_latency_ms']:<9.1f}{any_K['baseline_hit_rate']:<8.3f}")
            for k in all_Ks:
                r = grouped[key].get(k)
                if r is None:
                    line += f"{'-':<10}{'-':<9}{'-':<9}{'-':<8}{'-':<8}"
                else:
                    line += (f"{r['prefetched_latency_ms']:<10.1f}"
                             f"{r['prefetched_hit_rate']:<9.3f}"
                             f"{r['prefetched_misses']:<9d}"
                             f"{r['speedup']:<8.2f}"
                             f"{r['waste_rate']:<8.2f}")
            print(line)


def write_wide_csv(records, out_path):
    """Wide CSV: one row per run, columns per K and per metric."""
    grouped = defaultdict(dict)
    for r in records:
        key = (r["model"], r["quantization"], r["gpu_memory_gb"], r["benchmark"])
        grouped[key][r["prefetch_K"]] = r

    if not grouped:
        return
    sample = next(iter(grouped.values()))
    Ks = sorted(sample.keys())

    fieldnames = ["model", "quantization", "gpu_memory_gb", "benchmark",
                  "tokens", "cache_capacity", "total_experts", "cache_pct",
                  "predictor_ms", "layer_compute_ms", "transfer_ms",
                  "baseline_latency_ms", "baseline_hit_rate", "baseline_misses"]
    for k in Ks:
        for metric in ("latency_ms", "hit_rate", "misses", "evictions",
                       "total_prefetches", "dedup_skipped", "wasted_prefetches",
                       "waste_rate", "speedup", "miss_reduction", "hit_rate_gain"):
            fieldnames.append(f"K{k}_{metric}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key, kdict in grouped.items():
            m, q, gb, bench = key
            any_K = next(iter(kdict.values()))
            row = dict(
                model=m, quantization=q, gpu_memory_gb=gb, benchmark=bench,
                tokens=any_K["tokens"], cache_capacity=any_K["cache_capacity"],
                total_experts=any_K["total_experts"], cache_pct=any_K["cache_pct"],
                predictor_ms=any_K["predictor_ms"],
                layer_compute_ms=any_K["layer_compute_ms"],
                transfer_ms=any_K["transfer_ms"],
                baseline_latency_ms=any_K["baseline_latency_ms"],
                baseline_hit_rate=any_K["baseline_hit_rate"],
                baseline_misses=any_K["baseline_misses"],
            )
            for k in Ks:
                r = kdict.get(k)
                if r is None:
                    continue
                row[f"K{k}_latency_ms"] = r["prefetched_latency_ms"]
                row[f"K{k}_hit_rate"] = r["prefetched_hit_rate"]
                row[f"K{k}_misses"] = r["prefetched_misses"]
                row[f"K{k}_evictions"] = r["cache_evictions"]
                row[f"K{k}_total_prefetches"] = r["total_prefetches"]
                row[f"K{k}_dedup_skipped"] = r["dedup_skipped"]
                row[f"K{k}_wasted_prefetches"] = r["wasted_prefetches"]
                row[f"K{k}_waste_rate"] = r["waste_rate"]
                row[f"K{k}_speedup"] = r["speedup"]
                row[f"K{k}_miss_reduction"] = r["miss_reduction"]
                row[f"K{k}_hit_rate_gain"] = r["hit_rate_gain"]
            w.writerow(row)
    print(f"Wrote {len(grouped)} runs (wide format) -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="figures/full_sweep")
    p.add_argument("--out_csv", default="results_sweep_summary.csv")
    p.add_argument("--figures_dir", default="figures")
    args = p.parse_args()

    records = parse_results(args.in_dir)
    if not records:
        print(f"No JSON files found in {args.in_dir}/. Did the sweep produce any?")
        return
    print(f"Loaded {len(records)} per-K records across "
          f"{len(set((r['model'], r['quantization'], r['gpu_memory_gb'], r['benchmark']) for r in records))} runs.")

    write_csv(records, args.out_csv)
    write_wide_csv(records, args.out_csv.replace(".csv", "_wide.csv"))
    plot_speedup_heatmap(records, args.figures_dir)
    plot_hitrate_vs_cap(records, args.figures_dir)
    plot_K_curves(records, args.figures_dir)
    print_summary_table(records)
    print_detailed_per_K_table(records)


if __name__ == "__main__":
    main()

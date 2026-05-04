"""
Speedup heatmaps at NVMe-tier transfer cost.

Reads results_sweep_summary.csv (long format from aggregate_sweep.py) and
produces ONE figure with three panels (one per model). Within each panel:
  - Rows = GPU memory cap (GiB)
  - Cols = quantization
  - Cell = mean speedup at best-K, averaged over the three held-out
           benchmarks (wikitext_test, mmlu, gsm8k)
  - Empty (gray) cell = infeasible: any benchmark missing from the sweep
    (model did not fit) OR mean best-K speedup <= 1.0 (overhead exhausted
    the budget).
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


BENCHMARKS = ("wikitext_test", "mmlu", "gsm8k")
QUANT_ORDER = ("4bit", "8bit", "fp16")
MODEL_ORDER = ("mixtral_8x7b", "deepseek_moe_16b", "qwen1_5_moe_a2_7b")
MODEL_LABEL = {
    "mixtral_8x7b": "Mixtral-8x7B",
    "deepseek_moe_16b": "DeepSeek-MoE-16B",
    "qwen1_5_moe_a2_7b": "Qwen1.5-MoE-A2.7B",
}


def load_records(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def best_K_speedup(records):
    """dict[(model, quant, gb, bench)] -> max speedup across K."""
    best = {}
    for r in records:
        if r["benchmark"] not in BENCHMARKS:
            continue
        key = (r["model"], r["quantization"], int(r["gpu_memory_gb"]), r["benchmark"])
        sp = float(r["speedup"])
        if key not in best or sp > best[key]:
            best[key] = sp
    return best


def cell_value(best, model, quant, gb):
    """Mean over the 3 benchmarks; NaN if any benchmark missing or mean <= 1.0."""
    vals = []
    for b in BENCHMARKS:
        v = best.get((model, quant, gb, b))
        if v is None:
            return np.nan
        vals.append(v)
    mean = float(np.mean(vals))
    return mean if mean > 1.0 else np.nan


def build_matrix(best, model, gbs, quants):
    M = np.full((len(gbs), len(quants)), np.nan)
    for i, gb in enumerate(gbs):
        for j, q in enumerate(quants):
            M[i, j] = cell_value(best, model, q, gb)
    return M


def plot(records, out_path):
    best = best_K_speedup(records)
    models_present = [m for m in MODEL_ORDER if any(k[0] == m for k in best)]
    if not models_present:
        print("No records for any known model.")
        return

    all_gbs = sorted({k[2] for k in best})
    all_quants = [q for q in QUANT_ORDER if any(k[1] == q for k in best)]

    mats = {m: build_matrix(best, m, all_gbs, all_quants) for m in models_present}
    finite = np.concatenate([m[np.isfinite(m)] for m in mats.values()])
    if finite.size == 0:
        print("All cells infeasible — nothing to plot.")
        return
    vmin, vmax = 1.0, float(np.nanmax(finite))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#dddddd")  # explicit infeasible color

    fig, axes = plt.subplots(
        1, len(models_present),
        figsize=(3.4 * len(models_present) + 1.2, 3.6),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    im = None
    midpoint = (vmin + vmax) / 2.0
    for ax, model in zip(axes, models_present):
        M = np.ma.masked_invalid(mats[model])
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(all_quants)))
        ax.set_xticklabels(all_quants)
        ax.set_yticks(range(len(all_gbs)))
        ax.set_yticklabels([str(g) for g in all_gbs])
        ax.set_xlabel("Quantization")
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=10)
        for i in range(len(all_gbs)):
            for j in range(len(all_quants)):
                v = mats[model][i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}×",
                            ha="center", va="center",
                            color="white" if v < midpoint else "black",
                            fontsize=9)
    axes[0].set_ylabel("GPU memory cap (GiB)")

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.035, pad=0.02, shrink=0.9)
    cbar.set_label("Speedup ×")

    fig.suptitle(
        "Speedup at best-$K$ — NVMe-tier transfer  (mean over wikitext‐test, mmlu, gsm8k)",
        fontsize=11, y=1.02,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results_sweep_summary.csv")
    p.add_argument("--out", default="figures/sweep_speedup_heatmap_nvme.png")
    args = p.parse_args()
    records = load_records(args.csv)
    if not records:
        print(f"No records in {args.csv}.")
        return
    plot(records, args.out)


if __name__ == "__main__":
    main()

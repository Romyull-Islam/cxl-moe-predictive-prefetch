"""
Plot Recall vs lookahead distance for all trained predictors.
Reads results_multi.json (from extract_test_results.py).
"""
import argparse
import json
import os
import matplotlib.pyplot as plt


def plot_per_model(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        horizons = sorted(r["recall"].keys(), key=lambda s: int(s.split("+")[1]))
        x = [int(h.split("+")[1]) for h in horizons]
        # Each model has 3 metric columns; pick whatever is in the dict
        metric_keys = sorted(
            r["recall"][horizons[0]].keys(),
            key=lambda s: int(s[1:])
        )
        plt.figure(figsize=(7, 4.5))
        markers = ["o", "s", "^"]
        for k, m in zip(metric_keys, markers):
            y = [r["recall"][h][k] for h in horizons]
            plt.plot(x, y, m + "-", label=f"Recall {k}")
        plt.xlabel("Lookahead distance (layers ahead)")
        plt.ylabel("Recall (test set)")
        plt.title(
            f"{r['model']}  ({r['num_experts']} experts, top-{r['top_k']})"
        )
        plt.ylim(0.0, 1.0)
        plt.xticks(x)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, f"recall_vs_distance_{r['model']}.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"saved -> {path}")


def plot_combined(results, out_path):
    """One panel per model so we can compare on the same x-axis."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5),
                              sharey=True, squeeze=False)
    axes = axes[0]
    markers = ["o", "s", "^"]
    for ax, r in zip(axes, results):
        horizons = sorted(r["recall"].keys(), key=lambda s: int(s.split("+")[1]))
        x = [int(h.split("+")[1]) for h in horizons]
        metric_keys = sorted(
            r["recall"][horizons[0]].keys(),
            key=lambda s: int(s[1:])
        )
        for k, mk in zip(metric_keys, markers):
            y = [r["recall"][h][k] for h in horizons]
            ax.plot(x, y, mk + "-", label=k)
        ax.set_title(f"{r['model']}\n(E={r['num_experts']}, top-{r['top_k']})")
        ax.set_xlabel("d (layers ahead)")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
        ax.legend(title="Recall")
    axes[0].set_ylabel("Recall (test set)")
    fig.suptitle("Predictor recall vs lookahead distance — multi-benchmark training (WikiText + MMLU + GSM8K)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_multi.json")
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    plot_per_model(results, args.out_dir)
    plot_combined(results, os.path.join(args.out_dir, "recall_vs_distance_all_models.png"))


if __name__ == "__main__":
    main()

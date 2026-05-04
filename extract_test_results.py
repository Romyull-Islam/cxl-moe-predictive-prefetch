"""
Parse the FINAL TEST RESULTS blocks from phase2 training logs and build a
JSON suitable for plot_recall_vs_distance.py and the manuscript Table II.
"""
import argparse
import json
import os
import re

LOGS = {
    "mixtral_8x7b":     ("phase2_logs/mixtral_8x7b.log",     8,  2, 32, 4096),
    "deepseek_moe_16b": ("phase2_logs/deepseek_moe_16b.log", 64, 6, 28, 2048),
    "qwen1_5_moe_a2_7b":("phase2_logs/qwen1_5_moe_a2_7b.log", 60, 4, 24, 2048),
}

LINE = re.compile(r"\s*d\+(\d+):\s+(.*)")
KV   = re.compile(r"@(\d+)=([\d.]+)")


def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()

    # Grab the FINAL TEST RESULTS block (last occurrence)
    block = text.split("FINAL TEST RESULTS")[-1]
    m_loss = re.search(r"test_loss\s*=\s*([\d.]+)", block)
    test_loss = float(m_loss.group(1)) if m_loss else None

    recall = {}
    for line in block.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        d_idx = int(m.group(1))
        recall[f"d+{d_idx}"] = {f"@{int(k)}": float(v)
                                for k, v in KV.findall(m.group(2))}
    if not recall:
        return None
    return test_loss, recall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_multi.json")
    args = p.parse_args()

    results = []
    for key, (log_path, num_experts, top_k, num_layers, hidden_size) in LOGS.items():
        parsed = parse_log(log_path)
        if parsed is None:
            print(f"[skip] {key}: no FINAL TEST RESULTS in {log_path}")
            continue
        test_loss, recall = parsed
        rec = dict(
            model=key,
            num_experts=num_experts,
            num_layers=num_layers,
            hidden_size=hidden_size,
            top_k=top_k,
            test_loss=test_loss,
            recall=recall,
        )
        results.append(rec)
        print(f"[ok] {key}: test_loss={test_loss:.4f}  horizons={list(recall.keys())}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")

    # Pretty table
    print("\n" + "=" * 90)
    print(f"{'Model':<22} {'Horizon':<8} " + " ".join(f"{k:>10}" for k in ["@k", "@k+2", "@k+4"]))
    print("=" * 90)
    for r in results:
        for h, ks in r["recall"].items():
            keys = sorted(ks.keys(), key=lambda s: int(s[1:]))
            row = f"{r['model']:<22} {h:<8} " + " ".join(f"{ks[k]:>10.4f}" for k in keys)
            print(row)
        print("-" * 90)


if __name__ == "__main__":
    main()

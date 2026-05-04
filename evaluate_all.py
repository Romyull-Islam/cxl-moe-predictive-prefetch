import os
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from expert_predictor_topk import GlobalMultiStepDataset, GlobalMultiStepPredictor


MODELS_SINGLE = {
    "mixtral_8x7b": {
        "dir": "mixtral_8x7b_wikitext_logs",
        "npz": "mixtral_8x7b_wikitext_all_layers_raw.npz",
        "ckpt": "mixtral_8x7b_wikitext_all_layers_raw_predictor_topk2_d4.pt",
        "top_k": 2, "hidden_dim": 1024, "num_layers_mlp": 3,
    },
    "deepseek_moe_16b": {
        "dir": "deepseek_moe_16b_wikitext_logs",
        "npz": "deepseek_moe_16b_wikitext_all_layers_raw.npz",
        "ckpt": "deepseek_moe_16b_wikitext_all_layers_raw_predictor_topk6_d4.pt",
        "top_k": 6, "hidden_dim": 1024, "num_layers_mlp": 3,
    },
    "qwen1_5_moe_a2_7b": {
        "dir": "qwen1_5_moe_a2_7b_wikitext_logs",
        "npz": "qwen1_5_moe_a2_7b_wikitext_all_layers_raw.npz",
        "ckpt": "qwen1_5_moe_a2_7b_wikitext_all_layers_raw_predictor_topk4_d4.pt",
        "top_k": 4, "hidden_dim": 1024, "num_layers_mlp": 3,
    },
}

MODELS_MULTI = {
    "mixtral_8x7b": {
        "dir": "mixtral_8x7b_multi_logs",
        "npz": "mixtral_8x7b_wikitext_logs/mixtral_8x7b_wikitext_all_layers_raw.npz",
        "ckpt": "mixtral_8x7b_multi_predictor_topk2_d4.pt",
        "top_k": 2, "hidden_dim": 1024, "num_layers_mlp": 3,
    },
    "deepseek_moe_16b": {
        "dir": "deepseek_moe_16b_multi_logs",
        "npz": "deepseek_moe_16b_wikitext_logs/deepseek_moe_16b_wikitext_all_layers_raw.npz",
        "ckpt": "deepseek_moe_16b_multi_predictor_topk6_d4.pt",
        "top_k": 6, "hidden_dim": 2048, "num_layers_mlp": 4,
    },
    "qwen1_5_moe_a2_7b": {
        "dir": "qwen1_5_moe_a2_7b_multi_logs",
        "npz": "qwen1_5_moe_a2_7b_wikitext_logs/qwen1_5_moe_a2_7b_all_layers_raw.npz",
        "ckpt": "qwen1_5_moe_a2_7b_multi_predictor_topk4_d4.pt",
        "top_k": 4, "hidden_dim": 2048, "num_layers_mlp": 4,
    },
}

MODELS = MODELS_MULTI  # default; overridden by --variant single


def evaluate(model_key, lookahead_depth, batch_size, num_gpus, device):
    cfg = MODELS[model_key]
    npz_path = os.path.join(cfg["dir"], cfg["npz"])
    ckpt_path = os.path.join(cfg["dir"], cfg["ckpt"])
    test_indices_path = os.path.join(cfg["dir"], f"test_indices_d{lookahead_depth}.npy")

    if not os.path.exists(ckpt_path):
        print(f"[skip] {model_key}: checkpoint missing → {ckpt_path}")
        return None
    if not os.path.exists(test_indices_path):
        print(f"[skip] {model_key}: test indices missing → {test_indices_path}")
        return None

    meta = np.load(npz_path, mmap_mode="r")
    hidden_size = int(meta["hidden_size"][0])
    num_experts = int(meta["num_experts_per_layer"][0])
    num_layers = int(meta["num_layers"][0])

    model = GlobalMultiStepPredictor(
        d_model=hidden_size,
        num_experts=num_experts,
        num_layers_total=num_layers,
        lookahead_depth=lookahead_depth,
        layer_embed_dim=32,
        hidden_dim=1024,
        num_layers_mlp=3,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        main_device = "cuda:0"
    else:
        main_device = device
    model = model.to(main_device).eval()

    dataset = GlobalMultiStepDataset(
        npz_path=npz_path,
        lookahead_depth=lookahead_depth,
        top_k=cfg["top_k"],
        max_tokens_per_layer=None,
        shuffle=False,
    )
    test_indices = np.load(test_indices_path)
    test_ds = Subset(dataset, test_indices)

    num_workers = 16 if num_gpus > 1 else 0
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )

    criterion = nn.BCEWithLogitsLoss()
    predict_topk = (2, 3, 4)
    total_loss, total_samples = 0.0, 0
    coverage = {d: {k: 0 for k in predict_topk} for d in range(lookahead_depth)}

    with torch.no_grad():
        for h, layer_ids, y in tqdm(test_loader, desc=f"{model_key}"):
            h = h.to(main_device, non_blocking=True)
            layer_ids = layer_ids.to(main_device, non_blocking=True)
            y = y.to(main_device, non_blocking=True)

            logits = model(h, layer_ids)
            B, D, E = logits.shape

            targets = torch.zeros(B, D, E, device=main_device)
            targets.scatter_(2, y, 1.0)
            loss = criterion(logits, targets)
            total_loss += loss.item() * B
            total_samples += B

            probs = torch.softmax(logits, dim=-1)
            for d in range(D):
                for k in predict_topk:
                    topk_preds = probs[:, d, :].topk(k, dim=-1).indices
                    matched = (topk_preds.unsqueeze(1) == y[:, d, :].unsqueeze(2)).any(dim=2)
                    coverage[d][k] += matched.all(dim=1).sum().item()

    test_loss = total_loss / total_samples
    rec = {
        "model": model_key,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "top_k": cfg["top_k"],
        "test_samples": total_samples,
        "test_loss": test_loss,
        "recall": {
            f"d+{d+1}": {f"@{k}": coverage[d][k] / total_samples for k in predict_topk}
            for d in range(lookahead_depth)
        },
    }
    return rec


def print_table(results):
    print("\n" + "=" * 90)
    print(f"{'Model':<22} {'Horizon':<10} {'Recall@2':<12} {'Recall@3':<12} {'Recall@4':<12} {'Loss':<10}")
    print("=" * 90)
    for r in results:
        if r is None:
            continue
        for h_label, ks in r["recall"].items():
            print(
                f"{r['model']:<22} {h_label:<10} "
                f"{ks['@2']:<12.4f} {ks['@3']:<12.4f} {ks['@4']:<12.4f} {r['test_loss']:<10.4f}"
            )
        print("-" * 90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lookahead_depth", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32768)
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="results_all_models.json")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = p.parse_args()

    results = []
    for key in args.models:
        rec = evaluate(key, args.lookahead_depth, args.batch_size, args.num_gpus, args.device)
        if rec is not None:
            results.append(rec)

    print_table(results)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()

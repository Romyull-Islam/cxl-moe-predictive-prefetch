"""
train_predictor_multi.py — train one predictor per model on the union of
multiple benchmark traces (e.g. WikiText + MMLU + GSM8K).

All NPZs passed in must come from the SAME backbone model — same num_layers,
same num_experts, same hidden_size. The script verifies this and concatenates
H_layer{l}/E_layer{l} arrays across benchmarks.

Example:
  python train_predictor_multi.py \\
      --model_key mixtral_8x7b \\
      --npz_paths \\
          mixtral_8x7b_wikitext_logs/mixtral_8x7b_wikitext_all_layers_raw.npz \\
          mixtral_8x7b_mmlu_logs/mixtral_8x7b_mmlu_all_layers_raw.npz \\
          mixtral_8x7b_gsm8k_logs/mixtral_8x7b_gsm8k_all_layers_raw.npz \\
      --top_k 2 --lookahead_depth 4 \\
      --hidden_dim 1024 --num_layers_mlp 3 \\
      --epochs 10 --batch_size 32768 --num_gpus 6 \\
      --out_dir mixtral_8x7b_multi_logs
"""

import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from expert_predictor_topk import GlobalMultiStepPredictor


class MultiNpzDataset(Dataset):
    """
    Pools tokens from multiple NPZs (different benchmarks, same model).

    For each NPZ k and each source layer l in [0, num_layers - lookahead):
      one sample = (H_npz_k_layer_l[j], layer_id=l, top_k experts at layers l+1..l+D)
    """

    def __init__(self, npz_paths, lookahead_depth, top_k, max_tokens_per_layer=None,
                 shuffle=True, seed=42):
        assert len(npz_paths) > 0
        self.lookahead_depth = lookahead_depth
        self.top_k = top_k

        # Verify metadata consistency, gather metadata
        first = np.load(npz_paths[0], mmap_mode="r")
        self.hidden_size = int(first["hidden_size"][0])
        self.num_layers = int(first["num_layers"][0])
        self.num_experts = int(first["num_experts_per_layer"][0])
        self.file_top_k = int(first["top_k"][0])

        # Per-NPZ caches
        # H_caches[k][l] -> tensor [N_l, hidden_size]   (float32 in RAM)
        # E_caches[k][l] -> tensor [N_l, top_k]         (long)
        self.H_caches = []
        self.E_caches = []

        for path in npz_paths:
            print(f"loading {path}")
            meta = np.load(path, mmap_mode="r")
            assert int(meta["hidden_size"][0]) == self.hidden_size, f"{path}: hidden_size mismatch"
            assert int(meta["num_layers"][0]) == self.num_layers, f"{path}: num_layers mismatch"
            assert int(meta["num_experts_per_layer"][0]) == self.num_experts, f"{path}: num_experts mismatch"
            assert int(meta["top_k"][0]) >= top_k, \
                f"{path}: file top_k={int(meta['top_k'][0])} < requested top_k={top_k}"

            H_per_layer = {}
            E_per_layer = {}
            for l in range(self.num_layers):
                key_h = f"H_layer{l}"
                key_e = f"E_layer{l}"
                if key_h in meta.files:
                    H_per_layer[l] = torch.from_numpy(meta[key_h]).to(torch.float32)
                    E_per_layer[l] = torch.from_numpy(meta[key_e][:, :top_k]).to(torch.long)
            self.H_caches.append(H_per_layer)
            self.E_caches.append(E_per_layer)

        # Build sample index: list of (npz_idx, layer_id, j_in_npz)
        rng = np.random.default_rng(seed)
        samples = []
        for k, H_pl in enumerate(self.H_caches):
            for l in range(self.num_layers - lookahead_depth):
                if l not in H_pl:
                    continue
                # Skip layer if any required future layer is missing in this NPZ
                if any((l + d) not in self.E_caches[k] for d in range(1, lookahead_depth + 1)):
                    continue
                N = H_pl[l].shape[0]
                if max_tokens_per_layer and max_tokens_per_layer < N:
                    idx = rng.choice(N, size=max_tokens_per_layer, replace=False)
                else:
                    idx = np.arange(N)
                for j in idx:
                    samples.append((k, l, int(j)))

        if shuffle:
            rng.shuffle(samples)
        self.samples = samples
        print(f"combined dataset: {len(samples):,} samples "
              f"({self.num_layers} layers, {self.num_experts} experts, "
              f"hidden={self.hidden_size}, top_k={top_k})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        k, l, j = self.samples[idx]
        h = self.H_caches[k][l][j]
        y = torch.stack([self.E_caches[k][l + d][j] for d in range(1, self.lookahead_depth + 1)])
        return h, torch.tensor(l, dtype=torch.long), y


def evaluate(model, loader, device, lookahead_depth, top_k, predict_topk):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = torch.zeros(1, device=device)
    total_samples = 0
    coverage = {d: {k: torch.zeros(1, device=device) for k in predict_topk}
                for d in range(lookahead_depth)}
    with torch.no_grad():
        for h, lid, y in tqdm(loader, desc="eval", leave=False):
            h, lid, y = h.to(device, non_blocking=True), lid.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(h, lid)
            B, D, E = logits.shape
            tgt = torch.zeros(B, D, E, device=device)
            tgt.scatter_(2, y, 1.0)
            loss = criterion(logits, tgt)
            total_loss += loss * B
            total_samples += B
            for d in range(D):
                for k in predict_topk:
                    pred = logits[:, d, :].topk(k, dim=-1).indices
                    matched = (pred.unsqueeze(1) == y[:, d, :].unsqueeze(2)).any(dim=2)
                    coverage[d][k] += matched.all(dim=1).sum()
    avg_loss = (total_loss / max(1, total_samples)).item()
    cov_out = {d: {k: (coverage[d][k] / max(1, total_samples)).item()
                   for k in predict_topk}
               for d in range(lookahead_depth)}
    return avg_loss, cov_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True,
                   help="e.g. mixtral_8x7b — used only for output filename.")
    p.add_argument("--npz_paths", nargs="+", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--lookahead_depth", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--max_tokens_per_layer", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32768)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_layers_mlp", type=int, default=3)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    out_dir = args.out_dir or f"{args.model_key}_multi_logs"
    os.makedirs(out_dir, exist_ok=True)

    dataset = MultiNpzDataset(
        npz_paths=args.npz_paths,
        lookahead_depth=args.lookahead_depth,
        top_k=args.top_k,
        max_tokens_per_layer=args.max_tokens_per_layer,
        shuffle=True,
    )

    N = len(dataset)
    train_size = int(N * args.train_fraction)
    val_size = int(N * args.val_fraction)
    test_size = N - train_size - val_size
    indices = list(range(N))
    train_ds = Subset(dataset, indices[:train_size])
    val_ds = Subset(dataset, indices[train_size: train_size + val_size])
    test_ds = Subset(dataset, indices[train_size + val_size:])
    print(f"splits: train={train_size:,}  val={val_size:,}  test={test_size:,}")

    np.save(os.path.join(out_dir, f"test_indices_d{args.lookahead_depth}.npy"),
            np.array(indices[train_size + val_size:]))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True,
                            multiprocessing_context="fork")

    model = GlobalMultiStepPredictor(
        d_model=dataset.hidden_size,
        num_experts=dataset.num_experts,
        num_layers_total=dataset.num_layers,
        lookahead_depth=args.lookahead_depth,
        layer_embed_dim=32,
        hidden_dim=args.hidden_dim,
        num_layers_mlp=args.num_layers_mlp,
    )

    if args.num_gpus > 1 and torch.cuda.device_count() >= args.num_gpus:
        print(f"DataParallel across {args.num_gpus} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
        main_device = "cuda:0"
    else:
        main_device = args.device
    model = model.to(main_device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    predict_topk = (args.top_k, args.top_k + 2, args.top_k + 4)

    best_val = float("inf")
    best_path = os.path.join(
        out_dir,
        f"{args.model_key}_multi_predictor_topk{args.top_k}_d{args.lookahead_depth}.pt",
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.zeros(1, device=main_device)
        total_samples = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        last_loss_value = 0.0
        for step, (h, lid, y) in enumerate(pbar):
            h, lid, y = h.to(main_device, non_blocking=True), lid.to(main_device, non_blocking=True), y.to(main_device, non_blocking=True)
            opt.zero_grad()
            logits = model(h, lid)
            B, D, E = logits.shape
            tgt = torch.zeros(B, D, E, device=main_device)
            tgt.scatter_(2, y, 1.0)
            loss = criterion(logits, tgt)
            loss.backward()
            opt.step()
            total_loss += loss.detach() * B
            total_samples += B
            if step % 50 == 0:
                last_loss_value = loss.item()
                pbar.set_postfix(loss=f"{last_loss_value:.4f}")

        val_loss, val_cov = evaluate(model, val_loader, main_device,
                                     args.lookahead_depth, args.top_k, predict_topk)
        train_loss_avg = (total_loss / max(1, total_samples)).item()
        msg = (f"epoch {epoch:02d} | train_loss={train_loss_avg:.4f} "
               f"| val_loss={val_loss:.4f}")
        for d in range(args.lookahead_depth):
            msg += f" | d+{d+1}_cov@{args.top_k}={val_cov[d][args.top_k]:.3f}"
        print(msg)

        if val_loss < best_val:
            best_val = val_loss
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_path)
            print(f"  -> saved best to {best_path}")

    # Final evaluation on the held-out test split
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             multiprocessing_context="fork")
    test_loss, test_cov = evaluate(model, test_loader, main_device,
                                   args.lookahead_depth, args.top_k, predict_topk)
    print("\n" + "=" * 70 + f"\nFINAL TEST RESULTS — {args.model_key}\n" + "=" * 70)
    print(f"test_loss = {test_loss:.4f}")
    for d in range(args.lookahead_depth):
        cov = test_cov[d]
        print(f"  d+{d+1}: " + "  ".join(f"@{k}={cov[k]:.4f}" for k in predict_topk))
    print(f"\ncheckpoint: {best_path}")
    print(f"test indices: {os.path.join(out_dir, f'test_indices_d{args.lookahead_depth}.npy')}")


if __name__ == "__main__":
    main()

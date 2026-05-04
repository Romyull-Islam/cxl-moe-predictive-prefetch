# expert_predictor_topk.py
import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm


# ----------------- Global multi-layer dataset -----------------


class GlobalMultiStepDataset(Dataset):
    def __init__(self, npz_path: str, lookahead_depth: int, top_k: int = 2, max_tokens_per_layer: int = None, shuffle: bool = True):
        assert os.path.exists(npz_path), f"NPZ not found: {npz_path}"
        self.lookahead_depth = lookahead_depth
        self.top_k = top_k
        meta = np.load(npz_path, mmap_mode="r")
        self.num_layers = int(meta["num_layers"][0])

        print("Preloading to RAM (float32)...")
        self.H_cache = {}
        self.E_cache = {}
        for l in range(self.num_layers):
            if f"H_layer{l}" in meta.files:
                self.H_cache[l] = torch.from_numpy(meta[f"H_layer{l}"]).to(torch.float32)
                # PRE-SLICE THE EXPERTS: This saves massive time in __getitem__
                self.E_cache[l] = torch.from_numpy(meta[f"E_layer{l}"][:, :top_k]).to(torch.long)

        samples = []
        for l in range(self.num_layers - lookahead_depth):
            if l in self.H_cache:
                N = self.H_cache[l].shape[0]
                idx = np.random.choice(N, size=max_tokens_per_layer, replace=False) if max_tokens_per_layer and max_tokens_per_layer < N else np.arange(N)
                for j in idx: samples.append((l, int(j)))
        
        if shuffle: np.random.default_rng(42).shuffle(samples)
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        l, j = self.samples[idx]
        h = self.H_cache[l][j]
        # FAST: Stack already-sliced expert indices
        y = torch.stack([self.E_cache[l + d][j] for d in range(1, self.lookahead_depth + 1)])
        return h, torch.tensor(l, dtype=torch.long), y


# ----------------- Models -----------------


class GlobalMultiStepPredictor(nn.Module):
    """
    One global predictor shared across all layers.

    Input:
      - hidden state h (d_model)
      - layer id (embedded)

    Output:
      - logits for lookahead_depth future layers: [B, D, num_experts]
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_layers_total: int,
        lookahead_depth: int,
        layer_embed_dim: int = 32,
        hidden_dim: int = 512,
        num_layers_mlp: int = 2,
    ):
        super().__init__()
        self.lookahead_depth = lookahead_depth

        self.layer_emb = nn.Embedding(num_layers_total, layer_embed_dim)

        in_dim = d_model + layer_embed_dim
        layers = []
        for _ in range(num_layers_mlp - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.heads = nn.ModuleList(
            [nn.Linear(in_dim, num_experts) for _ in range(lookahead_depth)]
        )

    def forward(self, h, layer_ids):
        """
        h: [B, d_model]
        layer_ids: [B]
        returns logits: [B, D, num_experts]
        """
        layer_vec = self.layer_emb(layer_ids)          # [B, layer_embed_dim]
        x = torch.cat([h, layer_vec], dim=-1)          # [B, d_model + layer_embed_dim]
        z = self.trunk(x)                              # [B, hidden_dim]
        logits_per_step = [head(z) for head in self.heads]  # list of [B, num_experts]
        logits = torch.stack(logits_per_step, dim=1)   # [B, D, num_experts]
        return logits


# ----------------- Visualization helpers -----------------


def plot_expert_frequency(npz_path: str, save_path: str = None):
    data = np.load(npz_path, mmap_mode="r")
    freq = data["freq"]  # [num_layers, num_experts]
    num_layers, num_experts = freq.shape

    global_freq = freq.sum(axis=0)

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(num_experts), global_freq)
    plt.xlabel("Expert index")
    plt.ylabel("Total activations")
    plt.title("Global expert activation counts")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved expert frequency plot to {save_path}")
    else:
        plt.show()


def plot_expert_heatmap(npz_path: str, save_path: str = None, log_scale: bool = True):
    data = np.load(npz_path, mmap_mode="r")
    freq = data["freq"]
    mat = np.log1p(freq) if log_scale else freq

    num_layers, num_experts = freq.shape

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Expert index")
    ax.set_ylabel("Layer index")
    title = "Layer–Expert Active Token Heatmap (log scale)" if log_scale \
        else "Layer–Expert Active Token Heatmap"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log(1 + activations)" if log_scale else "activations")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved expert heatmap to {save_path}")
    else:
        plt.show()


def summarize_tokens_per_layer(npz_path: str):
    data = np.load(npz_path, mmap_mode="r")
    hidden_size = int(data["hidden_size"][0])
    num_layers = int(data["num_layers"][0])

    print(f"hidden_size={hidden_size}, num_layers={num_layers}")
    for l in range(num_layers):
        key = f"H_layer{l}"
        if key not in data.files:
            continue
        N, d = data[key].shape
        print(f"Layer {l}: {N} tokens, dim={d}")


# ----------------- Training / Evaluation -----------------


def train_global_predictor(
    npz_path: str,
    lookahead_depth: int,
    top_k: int,
    max_tokens_per_layer: int,
    batch_size: int,
    lr: float,
    epochs: int,
    hidden_dim: int,
    num_layers_mlp: int,
    train_fraction: float,
    val_fraction: float,
    device: str,
    num_gpus: int = 1,
):
    data = np.load(npz_path, mmap_mode="r")
    hidden_size = int(data["hidden_size"][0])
    num_experts = int(data["num_experts_per_layer"][0])
    num_layers = int(data["num_layers"][0])

    print(f"Loaded metadata: hidden_size={hidden_size}, experts={num_experts}, layers={num_layers}")

    # DYNAMIC EVALUATION: Set evaluation thresholds based on the requested top_k
    predict_topk = (top_k, top_k + 2, top_k + 4)

    dataset = GlobalMultiStepDataset(
        npz_path=npz_path,
        lookahead_depth=lookahead_depth,
        top_k=top_k,
        max_tokens_per_layer=max_tokens_per_layer,
        shuffle=True,
    )

    N = len(dataset)
    train_size = int(N * train_fraction)
    val_size = int(N * val_fraction)
    test_size = N - train_size - val_size
    
    indices = list(range(N))
    train_ds = Subset(dataset, indices[:train_size])
    val_ds = Subset(dataset, indices[train_size:train_size + val_size])
    test_ds = Subset(dataset, indices[train_size + val_size:])
    
    print(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    out_dir = os.path.dirname(npz_path)
    np.save(os.path.join(out_dir, f"test_indices_d{lookahead_depth}.npy"), indices[train_size + val_size:])

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True,
        multiprocessing_context='fork'
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = GlobalMultiStepPredictor(
        d_model=hidden_size, num_experts=num_experts, num_layers_total=num_layers,
        lookahead_depth=lookahead_depth, hidden_dim=hidden_dim, num_layers_mlp=num_layers_mlp
    )
    
    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        print(f"\nUsing {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        main_device = 'cuda:0'
    else:
        main_device = device

    model = model.to(main_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    def evaluate(loader, desc="Validation"):
        model.eval()
        total_loss, total_samples = 0.0, 0
        coverage = {d: {k: 0 for k in predict_topk} for d in range(lookahead_depth)}

        with torch.no_grad():
            for h, layer_ids, y in tqdm(loader, desc=desc, leave=False):
                h, layer_ids, y = h.to(main_device), layer_ids.to(main_device), y.to(main_device)
                logits = model(h, layer_ids)
                B, D, E = logits.shape

                targets = torch.zeros(B, D, E, device=main_device)
                targets.scatter_(2, y, 1.0)
                
                loss = criterion(logits, targets)
                total_loss += loss.item() * B
                total_samples += B

                # Calculate Coverage
                for d in range(D):
                    for k in predict_topk:
                        topk_preds = logits[:, d, :].topk(k, dim=-1).indices 
                        true_experts = y[:, d, :]
                        # Check if all true experts for this layer are in the predicted top-k set
                        matched = (topk_preds.unsqueeze(1) == true_experts.unsqueeze(2)).any(dim=2)
                        coverage[d][k] += matched.all(dim=1).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        coverage_rate = {d: {k: coverage[d][k]/total_samples for k in predict_topk} for d in range(lookahead_depth)}
        return avg_loss, coverage_rate

    print("\n" + "="*80 + f"\nTRAINING START (Target: Top-{top_k})\n" + "="*80)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_samples = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for h, layer_ids, y in pbar:
            h, layer_ids, y = h.to(main_device), layer_ids.to(main_device), y.to(main_device)
            optimizer.zero_grad()
            
            logits = model(h, layer_ids)
            B, D, E = logits.shape
            
            targets = torch.zeros(B, D, E, device=main_device)
            targets.scatter_(2, y, 1.0) 

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_samples += B
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss, val_coverage = evaluate(val_loader)
        msg = f"Epoch {epoch:02d} | train_loss={total_loss/total_samples:.4f} | val_loss={val_loss:.4f}"
        # DYNAMIC LOGGING: Log the coverage for the actual top_k being used
        for d in range(lookahead_depth):
            msg += f" | d+{d+1}_cov@{top_k}={val_coverage[d][top_k]:.3f}"
        print(msg)

    return model, test_ds

# ----------------- Main CLI -----------------


def main():
    parser = argparse.ArgumentParser(
        description="Train global expert predictor from MoE logs "
                    "(any layer -> next D layers, top-k experts)."
    )
    parser.add_argument(
        "--npz_path", type=str, required=True,
        help="Path to *_all_layers_raw.npz for one model+benchmark."
    )
    parser.add_argument(
        "--lookahead_depth", type=int, default=4,
        help="Number of future layers to predict (e.g., 4 for prefetching)."
    )
    parser.add_argument(
        "--top_k", type=int, default=2,
        help="Number of experts per token (2 for Mixtral)."
    )
    parser.add_argument(
        "--max_tokens_per_layer", type=int, default=100_000,
        help="Max tokens to use per layer when building global dataset."
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--hidden_dim", type=int, default=1024,
        help="Hidden dim of MLP trunk."
    )
    parser.add_argument(
        "--num_layers_mlp", type=int, default=3,
        help="Number of linear layers in MLP (including output)."
    )
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use (1-8 for DGX). Batch size should scale accordingly."
    )
    parser.add_argument(
        "--plot_freq", action="store_true",
        help="If set, plot global expert frequency (histogram) and exit."
    )
    parser.add_argument(
        "--plot_heatmap", action="store_true",
        help="If set, plot layer-expert activation heatmap from freq and exit."
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help="If set, print per-layer token counts and exit."
    )

    args = parser.parse_args()

    if args.plot_freq:
        base = os.path.splitext(args.npz_path)[0]
        save_path = base + "_freq.png"
        plot_expert_frequency(args.npz_path, save_path=save_path)
        return

    if args.plot_heatmap:
        base = os.path.splitext(args.npz_path)[0]
        save_path = base + "_freq_heatmap.png"
        plot_expert_heatmap(args.npz_path, save_path=save_path, log_scale=True)
        return

    if args.summarize:
        summarize_tokens_per_layer(args.npz_path)
        return

    model, test_ds = train_global_predictor(
        npz_path=args.npz_path,
        lookahead_depth=args.lookahead_depth,
        top_k=args.top_k,
        max_tokens_per_layer=args.max_tokens_per_layer,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers_mlp=args.num_layers_mlp,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        device=args.device,
        num_gpus=args.num_gpus,
    )

    # Save global predictor (unwrap DataParallel if needed)
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    out_dir = os.path.dirname(args.npz_path)
    base = os.path.splitext(os.path.basename(args.npz_path))[0]
    model_name = f"{base}_predictor_topk{args.top_k}_d{args.lookahead_depth}.pt"
    save_path = os.path.join(out_dir, model_name)
    torch.save(model_state, save_path)
    print(f"\nSaved global predictor to {save_path}")
    print(f"\nTest set has been held out. Use test_predictor.py to evaluate on unseen data.")


if __name__ == "__main__":
    main()

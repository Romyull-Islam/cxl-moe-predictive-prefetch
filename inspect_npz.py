"""Print token counts and shapes from each model's trace NPZ."""
import os
import numpy as np

PATHS = {
    "mixtral_8x7b":     "mixtral_8x7b_wikitext_logs/mixtral_8x7b_wikitext_all_layers_raw.npz",
    "deepseek_moe_16b": "deepseek_moe_16b_wikitext_logs/deepseek_moe_16b_wikitext_all_layers_raw.npz",
    "qwen1_5_moe_a2_7b":"qwen1_5_moe_a2_7b_wikitext_logs/qwen1_5_moe_a2_7b_wikitext_all_layers_raw.npz",
}

WIKITEXT_TOTAL_LINES = 1_801_350  # WikiText-103 train, non-empty after filtering varies

for name, path in PATHS.items():
    if not os.path.exists(path):
        print(f"\n[{name}] MISSING: {path}")
        continue
    print(f"\n{'='*72}\n[{name}]\n  path = {path}\n  size = {os.path.getsize(path)/1024**3:.2f} GB")
    data = np.load(path, mmap_mode="r")
    hidden_size = int(data["hidden_size"][0])
    num_layers = int(data["num_layers"][0])
    num_experts = int(data["num_experts_per_layer"][0])
    top_k = int(data["top_k"][0])
    print(f"  num_layers={num_layers}  num_experts={num_experts}  top_k={top_k}  hidden_size={hidden_size}")

    per_layer = []
    for l in range(num_layers):
        k = f"H_layer{l}"
        if k in data.files:
            per_layer.append(data[k].shape[0])
    if per_layer:
        total = sum(per_layer)
        print(f"  layers with data: {len(per_layer)} / {num_layers}")
        print(f"  tokens per layer: min={min(per_layer):,}  max={max(per_layer):,}  mean={int(np.mean(per_layer)):,}")
        print(f"  total samples (sum over layers) = {total:,}")
        print(f"  tokens per layer ≈ # WikiText examples × seq_len; this is NOT the full WikiText-103 corpus")

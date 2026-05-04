import argparse
import json
import os
import time
import numpy as np
import torch
from torch import nn

from expert_predictor_topk import GlobalMultiStepPredictor


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model, dtype_bytes):
    return count_params(model) * dtype_bytes / (1024 ** 2)


def make_model(npz_path, lookahead_depth, hidden_dim=1024, num_layers_mlp=3):
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
        hidden_dim=hidden_dim,
        num_layers_mlp=num_layers_mlp,
    )
    return model, hidden_size, num_experts, num_layers


def time_forward(model, h, layer_ids, n_warmup, n_iters, device):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(h, layer_ids)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(h, layer_ids)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    per_iter_ms = total_ms / n_iters
    per_token_us = (per_iter_ms * 1000.0) / h.shape[0]
    return per_iter_ms, per_token_us


def bench_one(model_key, npz_path, ckpt_path, lookahead_depth, batch_sizes,
              n_warmup, n_iters, results):
    print(f"\n{'=' * 70}\n{model_key}\n{'=' * 70}")
    model, hidden_size, num_experts, num_layers = make_model(npz_path, lookahead_depth)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"loaded weights from {ckpt_path}")
    else:
        print("(no checkpoint — random init; latency unaffected)")

    n_params = count_params(model)
    fp32_mb = model_size_mb(model, 4)
    fp16_mb = model_size_mb(model, 2)
    int8_mb = model_size_mb(model, 1)
    print(f"params={n_params/1e6:.2f}M  fp32={fp32_mb:.2f}MB  fp16={fp16_mb:.2f}MB  int8≈{int8_mb:.2f}MB")

    rec = {
        "model": model_key,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "params_M": n_params / 1e6,
        "fp32_MB": fp32_mb,
        "fp16_MB": fp16_mb,
        "int8_MB": int8_mb,
        "latency": {},
    }

    for bs in batch_sizes:
        rec["latency"][bs] = {}

        # FP32 GPU
        if torch.cuda.is_available():
            m_gpu = model.to("cuda").float()
            h = torch.randn(bs, hidden_size, device="cuda", dtype=torch.float32)
            lid = torch.randint(0, num_layers, (bs,), device="cuda")
            iter_ms, tok_us = time_forward(m_gpu, h, lid, n_warmup, n_iters, "cuda")
            rec["latency"][bs]["fp32_gpu"] = {"iter_ms": iter_ms, "per_token_us": tok_us}
            print(f"bs={bs:>6}  fp32 GPU  iter={iter_ms:7.3f}ms  per-token={tok_us:7.3f}µs")

            # FP16 GPU
            m_h = model.to("cuda").half()
            h16 = h.half()
            iter_ms, tok_us = time_forward(m_h, h16, lid, n_warmup, n_iters, "cuda")
            rec["latency"][bs]["fp16_gpu"] = {"iter_ms": iter_ms, "per_token_us": tok_us}
            print(f"bs={bs:>6}  fp16 GPU  iter={iter_ms:7.3f}ms  per-token={tok_us:7.3f}µs")
            model = model.float().cpu()

        # FP32 CPU
        m_cpu = model.to("cpu").float()
        h = torch.randn(bs, hidden_size, dtype=torch.float32)
        lid = torch.randint(0, num_layers, (bs,))
        iter_ms, tok_us = time_forward(m_cpu, h, lid, n_warmup, n_iters, "cpu")
        rec["latency"][bs]["fp32_cpu"] = {"iter_ms": iter_ms, "per_token_us": tok_us}
        print(f"bs={bs:>6}  fp32 CPU  iter={iter_ms:7.3f}ms  per-token={tok_us:7.3f}µs")

        # INT8 dynamic-quantized CPU (Linear layers only — what this model is made of)
        m_int8 = torch.quantization.quantize_dynamic(
            m_cpu, {nn.Linear}, dtype=torch.qint8
        )
        iter_ms, tok_us = time_forward(m_int8, h, lid, n_warmup, n_iters, "cpu")
        rec["latency"][bs]["int8_cpu_dynamic"] = {"iter_ms": iter_ms, "per_token_us": tok_us}
        print(f"bs={bs:>6}  int8 CPU  iter={iter_ms:7.3f}ms  per-token={tok_us:7.3f}µs")

    results.append(rec)


MODELS = {
    "mixtral_8x7b": {
        "dir": "mixtral_8x7b_wikitext_logs",
        "npz": "mixtral_8x7b_wikitext_all_layers_raw.npz",
        "ckpt": "mixtral_8x7b_wikitext_all_layers_raw_predictor_topk2_d4.pt",
    },
    "deepseek_moe_16b": {
        "dir": "deepseek_moe_16b_wikitext_logs",
        "npz": "deepseek_moe_16b_wikitext_all_layers_raw.npz",
        "ckpt": "deepseek_moe_16b_wikitext_all_layers_raw_predictor_topk6_d4.pt",
    },
    "qwen1_5_moe_a2_7b": {
        "dir": "qwen1_5_moe_a2_7b_wikitext_logs",
        "npz": "qwen1_5_moe_a2_7b_wikitext_all_layers_raw.npz",
        "ckpt": "qwen1_5_moe_a2_7b_wikitext_all_layers_raw_predictor_topk4_d4.pt",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lookahead_depth", type=int, default=4)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 32, 256])
    p.add_argument("--n_warmup", type=int, default=10)
    p.add_argument("--n_iters", type=int, default=200)
    p.add_argument("--out", type=str, default="results_latency.json")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = p.parse_args()

    results = []
    for key in args.models:
        cfg = MODELS[key]
        npz = os.path.join(cfg["dir"], cfg["npz"])
        ckpt = os.path.join(cfg["dir"], cfg["ckpt"])
        if not os.path.exists(npz):
            print(f"[skip] {key}: npz missing")
            continue
        bench_one(key, npz, ckpt, args.lookahead_depth, args.batch_sizes,
                  args.n_warmup, args.n_iters, results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLatency results saved → {args.out}")


if __name__ == "__main__":
    main()

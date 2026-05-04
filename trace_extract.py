"""
trace_extract.py — CLI version of moe_logging_updated.py.

Extracts per-layer (H_n, full router probs P_n, top-K selected experts E_n) for
one (model, benchmark) pair and saves them as a single .npz suitable for
expert_predictor_topk.py / train_predictor_multi.py.

Examples:
  python trace_extract.py --model mixtral_8x7b   --benchmark mmlu   --num_examples 2000
  python trace_extract.py --model deepseek_moe_16b --benchmark gsm8k --num_examples 2000
  python trace_extract.py --model qwen1_5_moe_a2_7b --benchmark mmlu --num_examples 2000
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from tqdm import tqdm


if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_len, layer_idx=None):
        return self.get_seq_length()
    DynamicCache.get_usable_length = get_usable_length


@dataclass
class ModelCfg:
    hf_name: str
    moe_layer_path: str
    moe_block_attr: str
    top_k: int
    trust_remote_code: bool = False


MODEL_ZOO = {
    "mixtral_8x7b": ModelCfg("mistralai/Mixtral-8x7B-v0.1", "model.layers", "block_sparse_moe", 2, False),
    "qwen1_5_moe_a2_7b": ModelCfg("Qwen/Qwen1.5-MoE-A2.7B", "model.layers", "mlp", 4, False),
    "deepseek_moe_16b": ModelCfg("deepseek-ai/deepseek-moe-16b-base", "model.layers", "mlp", 6, True),
}


def get_attr_by_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def get_moe_block(model, layer_idx, cfg):
    return getattr(get_attr_by_path(model, cfg.moe_layer_path)[layer_idx], cfg.moe_block_attr)


def load_texts(benchmark, n, mode="subset_fixed", fraction=0.05):
    if benchmark == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
    elif benchmark == "mmlu":
        # Use auxiliary_train (MMLU's training split) so the test split stays
        # held out for evaluation. auxiliary_train has ~99k examples.
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        texts = [
            ex["question"] + "\n" + "\n".join(
                f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(ex["choices"])
            )
            for ex in ds
        ]
    elif benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        texts = [ex["question"] for ex in ds]
    elif benchmark == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        texts = [ex["prompt"] for ex in ds]
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")

    if mode == "full":
        return texts
    if mode == "subset_fixed":
        return texts[:n]
    if mode == "subset_fraction":
        return texts[: max(10, int(len(texts) * fraction))]
    raise ValueError(f"unknown mode: {mode}")


def build_patched_forward(layer_idx, moe_block, gate_inputs, routing_full, expert_logs,
                          layer_num_experts, top_k):
    orig_forward = moe_block.forward

    def patched(self, hidden_states, *args, **kwargs):
        if hidden_states.dim() == 3:
            b, s, d = hidden_states.shape
            gate_input_3d = hidden_states
        else:
            s, d = hidden_states.shape
            b = 1
            gate_input_3d = hidden_states.unsqueeze(0)

        gate_inputs[layer_idx].append(
            gate_input_3d.detach().to("cpu", dtype=torch.float16).numpy()
        )

        gate_out = self.gate(gate_input_3d)

        if isinstance(gate_out, tuple):
            topk_idx, topk_weight = gate_out[0], gate_out[1]
            if topk_idx.dim() == 2:
                k = topk_idx.shape[1]
                topk_idx = topk_idx.view(b, s, k)
                topk_weight = topk_weight.view(b, s, k)
            local_E = self.gate.weight.shape[0] if hasattr(self.gate, "weight") else int(topk_idx.max().item() + 1)
            layer_num_experts[layer_idx] = local_E
            full_probs = torch.zeros(b, s, local_E, device=topk_weight.device, dtype=torch.float32)
            full_probs.scatter_(2, topk_idx, topk_weight.to(torch.float32))
            routing_full[layer_idx].append(full_probs.to("cpu", dtype=torch.float16).numpy())
            expert_logs[layer_idx].append(topk_idx.cpu().numpy())
        else:
            full_probs = torch.softmax(gate_out, dim=-1, dtype=torch.float32)
            local_E = full_probs.shape[-1]
            layer_num_experts[layer_idx] = local_E
            routing_full[layer_idx].append(full_probs.to("cpu", dtype=torch.float16).numpy())
            flat = full_probs.view(-1, local_E)
            _, sel = torch.topk(flat, top_k, dim=-1)
            expert_logs[layer_idx].append(sel.view(b, s, top_k).cpu().numpy())

        return orig_forward(hidden_states, *args, **kwargs)

    return patched, orig_forward


def flush_chunk(out_dir, chunk_id, num_layers, hidden_size, top_k, num_experts_per_layer,
                gate_inputs, routing_full, expert_logs, layer_num_experts):
    if all(len(gate_inputs[l]) == 0 for l in range(num_layers)):
        return None
    if num_experts_per_layer is None:
        inferred = [n for n in layer_num_experts if n is not None]
        num_experts_per_layer = max(inferred)
    freq = np.zeros((num_layers, num_experts_per_layer), dtype=np.int64)
    arrays = {}
    for l in range(num_layers):
        if not gate_inputs[l]:
            continue
        local_E = layer_num_experts[l] or num_experts_per_layer
        H = np.concatenate([a.reshape(-1, hidden_size) for a in gate_inputs[l]], 0).astype(np.float16)
        P = np.concatenate([a.reshape(-1, local_E) for a in routing_full[l]], 0).astype(np.float16)
        E = np.concatenate([a.reshape(-1, top_k) for a in expert_logs[l]], 0).astype(np.int16)
        arrays[f"H_layer{l}"] = H
        arrays[f"P_layer{l}"] = P
        arrays[f"E_layer{l}"] = E
        freq[l] += np.bincount(E.flatten(), minlength=num_experts_per_layer)
        gate_inputs[l].clear(); routing_full[l].clear(); expert_logs[l].clear()
    arrays["freq"] = freq
    arrays["hidden_size"] = np.array([hidden_size], np.int32)
    arrays["num_layers"] = np.array([num_layers], np.int32)
    arrays["num_experts_per_layer"] = np.array([num_experts_per_layer], np.int32)
    arrays["top_k"] = np.array([top_k], np.int32)
    path = os.path.join(out_dir, f"chunk{chunk_id}_raw.npz")
    np.savez_compressed(path, **arrays)
    return path, num_experts_per_layer


def merge_chunks(chunk_files, out_path, num_layers, hidden_size, top_k, num_experts_per_layer):
    print(f"merging {len(chunk_files)} chunks into {out_path}")
    global_freq = None
    merged = {}
    for path in chunk_files:
        data = np.load(path)
        f = data["freq"].astype(np.int64)
        global_freq = f if global_freq is None else global_freq + f
        for l in range(num_layers):
            for prefix in ("H_layer", "P_layer", "E_layer"):
                k = f"{prefix}{l}"
                if k in data.files:
                    merged.setdefault(k, []).append(data[k])
    final = {k: np.concatenate(v, 0) for k, v in merged.items()}
    final["freq"] = global_freq
    final["hidden_size"] = np.array([hidden_size], np.int32)
    final["num_layers"] = np.array([num_layers], np.int32)
    final["num_experts_per_layer"] = np.array([num_experts_per_layer], np.int32)
    final["top_k"] = np.array([top_k], np.int32)
    np.savez_compressed(out_path, **final)
    for p in chunk_files:
        try: os.remove(p)
        except OSError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_ZOO.keys()))
    ap.add_argument("--benchmark", required=True,
                    choices=["wikitext", "mmlu", "gsm8k", "humaneval"])
    ap.add_argument("--num_examples", type=int, default=2000)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=100, help="batches per chunk flush")
    ap.add_argument("--use_4bit", action="store_true", default=True)
    ap.add_argument("--device_map", type=str, default="auto")
    args = ap.parse_args()

    cfg = MODEL_ZOO[args.model]
    out_dir = f"{args.model}_{args.benchmark}_logs"
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"{args.model}_{args.benchmark}_all_layers_raw.npz")
    if os.path.exists(final_path):
        print(f"already exists: {final_path}  (delete to re-extract)"); return

    print(f"loading {cfg.hf_name} (4bit={args.use_4bit}, device_map={args.device_map})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_name, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    ) if args.use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_name,
        quantization_config=qcfg,
        torch_dtype=None if args.use_4bit else torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    top_k = cfg.top_k
    num_experts_per_layer = getattr(model.config, "num_local_experts", None) \
                            or getattr(model.config, "num_experts", None)

    gate_inputs = [[] for _ in range(num_layers)]
    routing_full = [[] for _ in range(num_layers)]
    expert_logs = [[] for _ in range(num_layers)]
    layer_num_experts = [None] * num_layers
    originals = {}

    for l in range(num_layers):
        moe = get_moe_block(model, l, cfg)
        if not hasattr(moe, "gate"):
            continue
        patched, orig = build_patched_forward(
            l, moe, gate_inputs, routing_full, expert_logs, layer_num_experts, top_k
        )
        originals[l] = orig
        moe.forward = patched.__get__(moe)

    texts = load_texts(args.benchmark, args.num_examples)
    print(f"running {len(texts)} examples (bench={args.benchmark}, batch={args.batch_size}, max_len={args.max_length})")

    chunk_files = []
    chunk_id = 0
    batches_in_chunk = 0
    n_tokens = 0

    for i in tqdm(range(0, len(texts), args.batch_size), desc="batches"):
        batch = texts[i : i + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=args.max_length, padding=True).to(model.device)
        with torch.no_grad():
            _ = model(**enc, use_cache=False)
        n_tokens += enc.attention_mask.sum().item()
        batches_in_chunk += 1
        if batches_in_chunk >= args.chunk_size:
            res = flush_chunk(out_dir, chunk_id, num_layers, hidden_size, top_k,
                              num_experts_per_layer, gate_inputs, routing_full,
                              expert_logs, layer_num_experts)
            if res:
                p, num_experts_per_layer = res
                chunk_files.append(p)
            chunk_id += 1
            batches_in_chunk = 0

    res = flush_chunk(out_dir, chunk_id, num_layers, hidden_size, top_k,
                      num_experts_per_layer, gate_inputs, routing_full,
                      expert_logs, layer_num_experts)
    if res:
        p, num_experts_per_layer = res
        chunk_files.append(p)

    if chunk_files:
        merge_chunks(chunk_files, final_path, num_layers, hidden_size, top_k,
                     num_experts_per_layer)
        print(f"saved -> {final_path}  ({n_tokens:,} tokens)")
    else:
        print("no data captured")

    for l, of in originals.items():
        get_moe_block(model, l, cfg).forward = of


if __name__ == "__main__":
    main()

"""
prefetch_real.py — REAL inference comparison: DRAM-offload baseline vs
predictor-driven prefetch, on real Qwen1.5-MoE / DeepSeek-MoE-16B in bf16.

Pipeline:
  1. Load model bf16 on GPU. Verify.
  2. Manually move all expert sub-modules to CPU pinned memory.
  3. Patch each expert's forward to:
       - check if on GPU; if not, sync-fault from CPU
       - count hits/misses
  4. Define an LRU expert cache with a configurable capacity.
  5. Run inference once with no predictor   = BASELINE (LRU only, sync faults)
  6. Run inference again with predictor + prefetch hooks = PREFETCHED
     - Predictor runs at every MoE block's pre-forward
     - Predicted experts get .to(GPU, non_blocking=True) on a separate stream
     - When the gate later asks, the expert is already there (cache hit)
  7. Wall-clock both runs with perf_counter + cuda.synchronize.
  8. Report tokens/sec for each, speedup, real cache hit rate.

Usage:
  python prefetch_real.py --model qwen1_5_moe_a2_7b --gpu 0 \\
      --cache_capacity 200 --num_examples 16 --max_length 256 \\
      --benchmark wikitext_test
"""

import argparse
import json
import os
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset

from expert_predictor_topk import GlobalMultiStepPredictor


if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_len, layer_idx=None):
        return self.get_seq_length()
    DynamicCache.get_usable_length = get_usable_length


MODELS = {
    "mixtral_8x7b": dict(
        hf="mistralai/Mixtral-8x7B-v0.1",
        moe_attr="block_sparse_moe",
        top_k=2,
        ckpt="mixtral_8x7b_multi_logs/mixtral_8x7b_multi_predictor_topk2_d4.pt",
        npz="mixtral_8x7b_wikitext_logs/mixtral_8x7b_wikitext_all_layers_raw.npz",
        hidden_dim=1024, num_layers_mlp=3,
        trust_remote_code=False,
    ),
    "qwen1_5_moe_a2_7b": dict(
        hf="Qwen/Qwen1.5-MoE-A2.7B",
        moe_attr="mlp",
        top_k=4,
        ckpt="qwen1_5_moe_a2_7b_multi_logs/qwen1_5_moe_a2_7b_multi_predictor_topk4_d4.pt",
        npz="qwen1_5_moe_a2_7b_wikitext_logs/qwen1_5_moe_a2_7b_wikitext_all_layers_raw.npz",
        hidden_dim=2048, num_layers_mlp=4,
        trust_remote_code=False,
    ),
    "deepseek_moe_16b": dict(
        hf="deepseek-ai/deepseek-moe-16b-base",
        moe_attr="mlp",
        top_k=6,
        ckpt="deepseek_moe_16b_multi_logs/deepseek_moe_16b_multi_predictor_topk6_d4.pt",
        npz="deepseek_moe_16b_wikitext_logs/deepseek_moe_16b_wikitext_all_layers_raw.npz",
        hidden_dim=2048, num_layers_mlp=4,
        trust_remote_code=True,
    ),
}


# ---------------- Real GPU expert cache ----------------

class RealExpertCache:
    """
    Manages real GPU<->CPU residency of expert nn.Module objects.
    All transfers are torch native (.to(device, non_blocking=True)).
    Cache is LRU; hits/misses are real (counted on actual device check).
    """
    def __init__(self, capacity, gpu_id):
        self.capacity = capacity
        self.gpu_str = f"cuda:{gpu_id}"
        self.gpu_dev = torch.device(self.gpu_str)
        self.cpu_dev = torch.device("cpu")
        self.gpu_modules = OrderedDict()    # key -> module (LRU; oldest first)
        self.cpu_modules = {}                # key -> module on CPU
        self.in_flight = {}                  # key -> CUDA event marking prefetch complete
        self.prefetch_stream = torch.cuda.Stream(device=gpu_id)
        # stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefetches_issued = 0
        self.prefetch_used = 0  # prefetch landed AND was demanded

    def register_offload(self, key, module):
        """Move a freshly-loaded module to CPU pinned memory and remember it."""
        module = module.to(self.cpu_dev)
        # Pinning all module params is helpful for non_blocking transfers.
        for p in module.parameters():
            try:
                p.data = p.data.pin_memory()
            except RuntimeError:
                pass
        for b in module.buffers():
            try:
                b.data = b.data.pin_memory()
            except RuntimeError:
                pass
        self.cpu_modules[key] = module

    def _evict_lru_if_needed(self):
        """Evict the LRU non-in-flight item if at capacity. Skip in-flight items."""
        while len(self.gpu_modules) >= self.capacity:
            evictable = None
            for k in self.gpu_modules:
                if k not in self.in_flight:
                    evictable = k
                    break
            if evictable is None:
                # Everything is in flight — give up evicting; cache will grow temporarily
                break
            mod = self.gpu_modules.pop(evictable)
            mod.to(self.cpu_dev, non_blocking=False)
            self.cpu_modules[evictable] = mod
            self.evictions += 1

    def access_sync(self, key):
        """
        Demand access (called from expert forward).
        Returns the module, guaranteed on GPU.
        """
        # In-flight prefetch: wait for it, then verify it's still in cache.
        if key in self.in_flight:
            self.in_flight[key].synchronize()
            del self.in_flight[key]
            if key in self.gpu_modules:
                self.gpu_modules.move_to_end(key)
                self.prefetch_used += 1
                self.hits += 1
                return self.gpu_modules[key]
            # otherwise: was prefetched but evicted before use; treat as cold miss
        if key in self.gpu_modules:
            self.gpu_modules.move_to_end(key)
            self.hits += 1
            return self.gpu_modules[key]
        # Cold miss — sync fault
        self.misses += 1
        self._evict_lru_if_needed()
        if key in self.cpu_modules:
            module = self.cpu_modules.pop(key)
        else:
            # Module isn't in either dict — must be a stale prefetch ref. Find it.
            # In practice this shouldn't happen with the fix above, but be safe.
            raise KeyError(f"expert {key} not found in CPU or GPU modules")
        module.to(self.gpu_dev, non_blocking=False)
        self.gpu_modules[key] = module
        return module

    def prefetch(self, key):
        """Async CPU->GPU copy on the prefetch stream. Idempotent."""
        if key in self.gpu_modules:
            self.gpu_modules.move_to_end(key)
            return
        if key in self.in_flight:
            return
        if key not in self.cpu_modules:
            return
        # Evict if needed (synchronously — cheap)
        self._evict_lru_if_needed()
        if len(self.gpu_modules) >= self.capacity:
            # Couldn't evict anything (all in flight). Skip this prefetch.
            return
        module = self.cpu_modules.pop(key)
        with torch.cuda.stream(self.prefetch_stream):
            module.to(self.gpu_dev, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.prefetch_stream)
        self.in_flight[key] = event
        self.gpu_modules[key] = module
        self.prefetches_issued += 1

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefetches_issued = 0
        self.prefetch_used = 0

    @property
    def hit_rate(self):
        denom = self.hits + self.misses
        return self.hits / denom if denom else 0.0


# ---------------- Patches ----------------

def patch_expert_module(expert_module, cache_key, cache):
    """Wrap an expert's forward to ensure GPU residency on call."""
    orig_forward = expert_module.forward

    def patched(*args, **kwargs):
        cache.access_sync(cache_key)   # ensures module is on GPU
        return orig_forward(*args, **kwargs)

    expert_module.forward = patched


def install_predictor_hooks(model, predictor, cfg, num_layers, gpu_id, cache,
                             enable_prefetch, prefetch_topk, lookahead=4):
    """
    Adds a pre-forward hook on each MoE block that runs the predictor on the
    incoming hidden states and prefetches predicted experts for the next
    `lookahead` layers.

    Returns a teardown function.
    """
    pred_dev = next(predictor.parameters()).device
    layers = model.model.layers
    handles = []

    if not enable_prefetch:
        return lambda: None

    def make_hook(L, moe):
        def pre_hook(module, inputs):
            hidden_states = inputs[0]
            if hidden_states.dim() == 3:
                h_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            else:
                h_flat = hidden_states
            # Predictor forward — small, fast
            lid = torch.full((h_flat.shape[0],), L, device=pred_dev, dtype=torch.long)
            with torch.no_grad():
                pred_logits = predictor(h_flat.to(pred_dev).half(), lid)  # [N,4,E]
            pred_probs = pred_logits.softmax(dim=-1)
            top_v, top_i = pred_probs.topk(prefetch_topk, dim=-1)  # [N,4,K]

            # Aggregate by target layer + expert id, taking max confidence across tokens
            agg = {}  # (target_layer, expert_id) -> max prob
            num_lookahead, K = top_i.shape[1], top_i.shape[2]
            top_i_cpu = top_i.cpu().numpy()
            top_v_cpu = top_v.cpu().numpy()
            for d_idx in range(num_lookahead):
                target_layer = L + d_idx + 1
                if target_layer >= num_layers:
                    continue
                for t in range(top_i_cpu.shape[0]):
                    for s in range(K):
                        e = int(top_i_cpu[t, d_idx, s])
                        p = float(top_v_cpu[t, d_idx, s])
                        key = (target_layer, e)
                        if key not in agg or agg[key] < p:
                            agg[key] = p
            # Issue prefetches in confidence order, with a budget cap
            ranked = sorted(agg.items(), key=lambda kv: -kv[1])
            budget = max(1, cache.capacity // 4)  # don't flood
            for (key, _) in ranked[:budget]:
                cache.prefetch(key)
            return None
        return pre_hook

    for L in range(num_layers):
        moe = getattr(layers[L], cfg["moe_attr"])
        if hasattr(moe, "experts") and hasattr(moe, "gate"):
            h = moe.register_forward_pre_hook(make_hook(L, moe))
            handles.append(h)

    def teardown():
        for h in handles:
            h.remove()
    return teardown


# ---------------- Setup ----------------

def offload_all_experts(model, cfg, cache):
    """Find every expert sub-module across all MoE layers and offload to CPU."""
    layers = model.model.layers
    n = 0
    for L in range(len(layers)):
        moe = getattr(layers[L], cfg["moe_attr"])
        if not hasattr(moe, "experts"):
            continue
        for e_idx, expert_mod in enumerate(moe.experts):
            key = (L, e_idx)
            cache.register_offload(key, expert_mod)
            patch_expert_module(expert_mod, key, cache)
            n += 1
    return n


def load_predictor(cfg, npz_path, gpu_id):
    meta = np.load(npz_path, mmap_mode="r")
    hidden_size = int(meta["hidden_size"][0])
    num_experts = int(meta["num_experts_per_layer"][0])
    num_layers = int(meta["num_layers"][0])
    pred = GlobalMultiStepPredictor(
        d_model=hidden_size, num_experts=num_experts,
        num_layers_total=num_layers, lookahead_depth=4,
        layer_embed_dim=32,
        hidden_dim=cfg["hidden_dim"], num_layers_mlp=cfg["num_layers_mlp"],
    )
    pred.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu"))
    return pred.to(f"cuda:{gpu_id}").half().eval()


def load_texts(benchmark, n):
    if benchmark == "wikitext_train":
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
    elif benchmark == "wikitext_test":
        ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
    elif benchmark == "mmlu":
        # Trace was extracted from auxiliary_train; eval uses the test split.
        ds = load_dataset("cais/mmlu", "all", split="test")
        texts = [ex["question"] + "\n" + "\n".join(
                   f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(ex["choices"]))
                 for ex in ds]
    elif benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        texts = [ex["question"] for ex in ds]
    else:
        raise ValueError(benchmark)
    return texts[:n]


# ---------------- Run ----------------

def run_inference_pass(model, tokenizer, texts, cfg, gpu_id,
                       max_length, batch_size, label):
    """Run inference on the given texts. Returns wall-clock seconds and tokens."""
    n_tokens = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True).to(f"cuda:{gpu_id}")
        with torch.no_grad():
            _ = model(**enc, use_cache=False)
        n_tokens += enc.attention_mask.sum().item()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"[{label}] {n_tokens} tokens in {dt:.2f}s = {n_tokens/dt:.1f} tok/s")
    return dt, n_tokens


def load_for_hf_offload(cfg, gpu_id, gpu_memory_gb, quantization="bf16",
                          offload_target="disk", offload_dir="./hf_offload_cache"):
    """
    Load with device_map='auto' + max_memory.

    offload_target='disk' (default): no CPU DRAM tier. Anything that doesn't fit
        on GPU is written to NVMe disk. accelerate pages from disk on demand.
        Matches our simulation's GPU+NVMe two-tier memory hierarchy.
    offload_target='cpu': overflow goes to CPU pinned memory (PCIe DRAM tier).
    """
    from transformers import BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(cfg["hf"], trust_remote_code=cfg["trust_remote_code"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    if offload_target == "disk":
        os.makedirs(offload_dir, exist_ok=True)
        # 4-bit/8-bit need a small CPU scratch budget for quantization state.
        # The actual expert overflow still goes to disk because experts are huge.
        cpu_scratch_gb = 8 if quantization in ("4bit", "8bit") else 0
        max_memory = {gpu_id: f"{gpu_memory_gb}GiB",
                      "cpu": f"{cpu_scratch_gb}GiB"}
        common = dict(
            device_map="auto", max_memory=max_memory,
            trust_remote_code=cfg["trust_remote_code"],
            offload_folder=offload_dir,
            offload_state_dict=True,
        )
    elif offload_target == "cpu":
        # bnb 4-bit/8-bit refuses any CPU/disk split. If the cap is large
        # enough to hold the quantized model, load everything on GPU
        # (max_memory is essentially descriptive at that point).
        common = dict(
            device_map={gpu_id: 0} if False else {"": gpu_id},
            trust_remote_code=cfg["trust_remote_code"],
        )
        max_memory = "(unconstrained — quantized model loads fully on GPU)"
    else:
        raise ValueError(f"unknown offload_target: {offload_target}")

    print(f"  loading {quantization} with device_map='auto', max_memory={max_memory}, "
          f"offload_target={offload_target}")
    if quantization == "4bit":
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(cfg["hf"], quantization_config=qcfg, **common)
    elif quantization == "8bit":
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(cfg["hf"], quantization_config=qcfg, **common)
    elif quantization == "bf16":
        model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.bfloat16, **common)
    elif quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.float16, **common)
    else:
        raise ValueError(quantization)
    return tok, model.eval()


def run_hf_offload_pass(args, cfg, texts, quantization="bf16"):
    """HF auto-offload baseline. accelerate handles all expert paging via hooks."""
    print(f"\n=== HF AUTO-OFFLOAD BASELINE ({quantization}, max_memory={args.gpu_memory_gb} GiB) ===")
    tok, model = load_for_hf_offload(cfg, args.gpu, args.gpu_memory_gb, quantization,
                                       offload_target=args.offload_target,
                                       offload_dir=args.offload_dir)
    try:
        if args.warmup:
            run_inference_pass(model, tok, texts[:1], cfg, args.gpu,
                               args.max_length, args.batch_size, "warmup-hf")
        dt, n = run_inference_pass(
            model, tok, texts, cfg, args.gpu, args.max_length, args.batch_size,
            f"HF_OFFLOAD_{quantization}")
        return dict(wall_seconds=dt, tokens=n, tokens_per_sec=n/dt,
                    quantization=quantization)
    finally:
        del model
        torch.cuda.empty_cache()


def run_hf_with_predictor_pass(args, cfg, texts, quantization="bf16", k_extra_list=(0, 2, 4)):
    """
    HF auto-offload + our predictor — sweeps multiple K values in ONE model load.

    For each K = top_k + k_extra in k_extra_list, install pre-forward hooks on
    every MoE block that:
      - run the predictor on the gate's incoming hidden state
      - call `_hf_hook.pre_forward()` early on predicted-expert modules
    accelerate's offload mechanism handles the actual CPU->GPU weight copy at
    any quantization (4-/8-bit/bf16/fp16). Then time real inference for each K.
    Returns a dict mapping K -> stats.
    """
    print(f"\n=== HF AUTO-OFFLOAD + OUR PREDICTOR ({quantization}, max_memory={args.gpu_memory_gb} GiB) ===")
    tok, model = load_for_hf_offload(cfg, args.gpu, args.gpu_memory_gb, quantization,
                                       offload_target=args.offload_target,
                                       offload_dir=args.offload_dir)
    try:
        predictor = load_predictor(cfg, cfg["npz"], args.gpu)
        pred_dev = next(predictor.parameters()).device
        num_layers = model.config.num_hidden_layers

        # Catalog expert modules per layer (do once)
        expert_modules = {}
        for L in range(num_layers):
            moe = getattr(model.model.layers[L], cfg["moe_attr"], None)
            if moe is None or not hasattr(moe, "experts"):
                continue
            for e_idx, exp in enumerate(moe.experts):
                expert_modules[(L, e_idx)] = exp

        prefetch_count = [0]
        skip_count = [0]
        # Separate CUDA stream so prefetch copies overlap with main-stream compute.
        prefetch_stream = torch.cuda.Stream(device=args.gpu)

        def install_hooks(K):
            handles = []
            already_prefetched = set()  # (target_layer, expert_id) prefetched this step
            def make_hook(L):
                def pre_hook(module, inputs):
                    hs = inputs[0]
                    if hs.dim() == 3:
                        h_flat = hs.reshape(-1, hs.shape[-1])
                    else:
                        h_flat = hs
                    lid = torch.full((h_flat.shape[0],), L, device=pred_dev, dtype=torch.long)
                    with torch.no_grad():
                        logits = predictor(h_flat.to(pred_dev).half(), lid)
                        pred_idx = logits.topk(K, dim=-1).indices.cpu().numpy()

                    # Aggregate predictions for layers L+1..L+4
                    keys_to_prefetch = set()
                    for d_idx in range(min(4, pred_idx.shape[1])):
                        target_layer = L + d_idx + 1
                        if target_layer >= num_layers:
                            continue
                        for t in range(pred_idx.shape[0]):
                            for s in range(K):
                                e = int(pred_idx[t, d_idx, s])
                                keys_to_prefetch.add((target_layer, e))

                    # Issue prefetches on a SEPARATE stream so copies overlap
                    # with the main stream's compute (the actual MoE forward).
                    main_stream = torch.cuda.current_stream(args.gpu)
                    with torch.cuda.stream(prefetch_stream):
                        # Wait for any pending main-stream work that might write to
                        # weight memory (paranoia — usually no-op).
                        prefetch_stream.wait_stream(main_stream)
                        for key in keys_to_prefetch:
                            if key in already_prefetched:
                                continue   # this layer-step already issued
                            exp = expert_modules.get(key)
                            if exp is None or not hasattr(exp, "_hf_hook"):
                                skip_count[0] += 1
                                continue
                            try:
                                exp._hf_hook.pre_forward(exp)
                                prefetch_count[0] += 1
                                already_prefetched.add(key)
                            except Exception:
                                skip_count[0] += 1
                    # Don't sync here — let copies proceed in parallel with
                    # whatever the main stream does next.
                    return None
                return pre_hook
            for L in range(num_layers):
                moe = getattr(model.model.layers[L], cfg["moe_attr"], None)
                if moe is None or not hasattr(moe, "gate"):
                    continue
                h = moe.register_forward_pre_hook(make_hook(L))
                handles.append(h)
            return handles

        results_by_K = {}
        for k_extra in k_extra_list:
            K = cfg["top_k"] + k_extra
            handles = install_hooks(K)
            if args.warmup:
                run_inference_pass(model, tok, texts[:1], cfg, args.gpu,
                                   args.max_length, args.batch_size,
                                   f"warmup-hfpred-K{K}")
            prefetch_count[0] = 0
            skip_count[0] = 0
            dt, n = run_inference_pass(
                model, tok, texts, cfg, args.gpu, args.max_length, args.batch_size,
                f"HF_OFFLOAD_PRED_{quantization}_K{K}")
            results_by_K[K] = dict(
                wall_seconds=dt, tokens=n, tokens_per_sec=n/dt,
                quantization=quantization,
                K=K, k_extra=k_extra,
                predictor_prefetch_calls=prefetch_count[0],
                predictor_skip_calls=skip_count[0],
            )
            for h in handles:
                h.remove()
        return results_by_K
    finally:
        del model
        torch.cuda.empty_cache()


def run(args):
    cfg = MODELS[args.model]
    gpu_id = args.gpu

    # Load shared resources
    texts = load_texts(args.benchmark, args.num_examples)
    print(f"loaded {len(texts)} examples from {args.benchmark}")

    results = {}

    # ---------- HF auto-offload (with/without predictor) ----------
    if args.mode in ("hf_offload", "hf_compare", "all"):
        hf_quant = args.quantization if args.mode != "all" else "bf16"
        hf_stats = run_hf_offload_pass(args, cfg, texts, quantization=hf_quant)
        results["hf_offload"] = hf_stats

    if args.mode in ("hf_with_predictor", "hf_compare"):
        # K-sweep parsing (default: top_k, top_k+2, top_k+4)
        k_extra_list = tuple(int(x) for x in args.predict_topk_extra_list.split(","))
        hfp_results = run_hf_with_predictor_pass(
            args, cfg, texts, quantization=args.quantization, k_extra_list=k_extra_list)
        results["hf_with_predictor_by_K"] = hfp_results

    if args.mode in ("hf_offload", "hf_with_predictor", "hf_compare"):
        speedup_summary = {}
        if args.mode == "hf_compare":
            base_wall = results["hf_offload"]["wall_seconds"]
            print("\n" + "=" * 100)
            print(f"REAL HF COMPARISON — {args.model} ({args.quantization}, {args.benchmark}, "
                  f"max_memory={args.gpu_memory_gb} GiB)")
            print("=" * 100)
            cols = [("HF baseline", results["hf_offload"])]
            for K, stats in sorted(results["hf_with_predictor_by_K"].items()):
                cols.append((f"HF+Pred K={K}", stats))
            header = f"{'Metric':<28}" + "".join(f"{c[0]:<18}" for c in cols)
            print(header)
            print("-" * len(header))
            for label, key, fmt in [
                ("Wall seconds", "wall_seconds", "{:.2f}"),
                ("Tokens/sec",   "tokens_per_sec", "{:.1f}"),
                ("Prefetch calls","predictor_prefetch_calls", "{:d}"),
            ]:
                line = f"{label:<28}"
                for c_label, stats in cols:
                    v = stats.get(key, "-")
                    line += f"{(fmt.format(v) if v != '-' else '-'):<18}"
                print(line)
            print()
            for K, stats in sorted(results["hf_with_predictor_by_K"].items()):
                sp = base_wall / stats["wall_seconds"]
                speedup_summary[K] = sp
                print(f"  Speedup K={K}: {sp:.2f}× (HF+predictor vs HF alone)")
            print("=" * 100)

        out_path = args.out or f"prefetch_real_{args.model}_{args.quantization}_{args.gpu_memory_gb}gb_{args.benchmark}_{args.mode}.json"
        with open(out_path, "w") as f:
            json.dump(dict(
                model=args.model, mode=args.mode,
                gpu_memory_gb=args.gpu_memory_gb,
                quantization=args.quantization,
                benchmark=args.benchmark,
                results=results,
                speedup_by_K={str(K): v for K, v in speedup_summary.items()},
            ), f, indent=2)
        print(f"\nSaved -> {out_path}")
        return

    # ---------- Our system: load fully, manual offload ----------
    print(f"\n[{args.model}] loading bf16 model fully on cuda:{gpu_id} for OUR system...")
    tok = AutoTokenizer.from_pretrained(cfg["hf"], trust_remote_code=cfg["trust_remote_code"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf"], torch_dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=cfg["trust_remote_code"],
    ).eval()
    num_layers = model.config.num_hidden_layers
    print(f"  loaded; num_layers={num_layers}")

    print(f"loading predictor checkpoint: {cfg['ckpt']}")
    predictor = load_predictor(cfg, cfg["npz"], gpu_id)

    # Cache + offload
    print(f"creating GPU expert cache (capacity={args.cache_capacity})...")
    cache = RealExpertCache(args.cache_capacity, gpu_id)
    n_experts = offload_all_experts(model, cfg, cache)
    print(f"  offloaded {n_experts} experts to CPU pinned memory")

    # Memory snapshot
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device=gpu_id)
    print(f"  GPU memory: {free/1024**3:.2f} / {total/1024**3:.2f} GiB free after offload")

    # Load test data
    print(f"loading benchmark: {args.benchmark}")
    texts = load_texts(args.benchmark, args.num_examples)
    print(f"  {len(texts)} examples")

    # ---------- Pass 1: BASELINE (LRU only, sync faults) ----------
    cache.reset_stats()
    teardown_baseline = install_predictor_hooks(
        model, predictor, cfg, num_layers, gpu_id, cache,
        enable_prefetch=False, prefetch_topk=cfg["top_k"] + args.predict_topk_extra)
    print(f"\n=== BASELINE: LRU only, sync faults on miss ===")
    # Warmup (one batch) to prime the cache and JIT
    if args.warmup:
        run_inference_pass(model, tok, texts[:1], cfg, gpu_id,
                           args.max_length, args.batch_size, "warmup-baseline")
        cache.reset_stats()
    base_dt, base_tokens = run_inference_pass(
        model, tok, texts, cfg, gpu_id, args.max_length, args.batch_size, "BASELINE")
    base_stats = dict(
        wall_seconds=base_dt, tokens=base_tokens,
        tokens_per_sec=base_tokens / base_dt,
        cache_hits=cache.hits, cache_misses=cache.misses,
        cache_evictions=cache.evictions, hit_rate=cache.hit_rate,
    )
    teardown_baseline()

    # ---------- Pass 2: PREFETCHED ----------
    cache.reset_stats()
    teardown_pref = install_predictor_hooks(
        model, predictor, cfg, num_layers, gpu_id, cache,
        enable_prefetch=True, prefetch_topk=cfg["top_k"] + args.predict_topk_extra)
    print(f"\n=== PREFETCHED: predictor + LRU + async prefetch ===")
    if args.warmup:
        run_inference_pass(model, tok, texts[:1], cfg, gpu_id,
                           args.max_length, args.batch_size, "warmup-prefetched")
        cache.reset_stats()
    pref_dt, pref_tokens = run_inference_pass(
        model, tok, texts, cfg, gpu_id, args.max_length, args.batch_size, "PREFETCHED")
    pref_stats = dict(
        wall_seconds=pref_dt, tokens=pref_tokens,
        tokens_per_sec=pref_tokens / pref_dt,
        cache_hits=cache.hits, cache_misses=cache.misses,
        cache_evictions=cache.evictions, hit_rate=cache.hit_rate,
        prefetches_issued=cache.prefetches_issued,
        prefetch_used=cache.prefetch_used,
    )
    teardown_pref()

    results["our_lru"] = base_stats
    results["our_prefetch"] = pref_stats

    # ---------- Report ----------
    print("\n" + "=" * 100)
    print(f"REAL WALL-CLOCK 3-WAY COMPARISON — {args.model} (bf16, {args.benchmark})")
    print("=" * 100)
    print(f"GPU cache capacity: {args.cache_capacity} / {n_experts} experts "
          f"({100*args.cache_capacity/max(1,n_experts):.1f}%)")
    if "hf_offload" in results:
        print(f"HF max_memory: {args.gpu_memory_gb} GiB on GPU + 200 GiB on CPU")
    print()

    cols = []
    if "hf_offload" in results:
        cols.append(("HF auto-offload", results["hf_offload"]))
    cols.append(("Our LRU only", results["our_lru"]))
    cols.append(("Our + Predictor", results["our_prefetch"]))

    header = f"{'Metric':<28}" + "".join(f"{c[0]:<22}" for c in cols)
    print(header)
    print("-" * len(header))

    for metric_label, key_in_stats, fmt in [
        ("Wall seconds",          "wall_seconds",       "{:.2f}"),
        ("Tokens/sec",            "tokens_per_sec",     "{:.1f}"),
        ("Cache hit rate",        "hit_rate",           "{:.4f}"),
        ("Cache misses",          "cache_misses",       "{:d}"),
        ("Cache evictions",       "cache_evictions",    "{:d}"),
        ("Prefetches issued",     "prefetches_issued",  "{:d}"),
        ("Prefetch hits (used)",  "prefetch_used",      "{:d}"),
    ]:
        line = f"{metric_label:<28}"
        for label, stats in cols:
            v = stats.get(key_in_stats, "-")
            if v == "-":
                line += f"{'-':<22}"
            else:
                line += f"{fmt.format(v):<22}"
        print(line)
    print()

    # Speedup ladder relative to HF (or LRU if HF wasn't run)
    base_for_speedup = (results["hf_offload"]["wall_seconds"]
                        if "hf_offload" in results else results["our_lru"]["wall_seconds"])
    base_label = "HF auto-offload" if "hf_offload" in results else "our LRU"
    print(f"Speedup vs {base_label}:")
    if "hf_offload" in results:
        print(f"  Our LRU only        : {results['hf_offload']['wall_seconds']/results['our_lru']['wall_seconds']:.2f}×")
    print(f"  Our + Predictor     : {base_for_speedup/results['our_prefetch']['wall_seconds']:.2f}×")
    if "hf_offload" in results:
        print(f"  Predictor over LRU  : {results['our_lru']['wall_seconds']/results['our_prefetch']['wall_seconds']:.2f}×")
    print("=" * 100)

    out = dict(
        model=args.model, quantization="bf16",
        cache_capacity=args.cache_capacity,
        gpu_memory_gb=args.gpu_memory_gb,
        total_experts=n_experts,
        benchmark=args.benchmark,
        prefetch_topk=cfg["top_k"] + args.predict_topk_extra,
        results=results,
        speedup_predictor_over_lru=results["our_lru"]["wall_seconds"]/results["our_prefetch"]["wall_seconds"],
    )
    if "hf_offload" in results:
        out["speedup_lru_over_hf"] = results["hf_offload"]["wall_seconds"]/results["our_lru"]["wall_seconds"]
        out["speedup_predictor_over_hf"] = results["hf_offload"]["wall_seconds"]/results["our_prefetch"]["wall_seconds"]
    out_path = args.out or f"prefetch_real_{args.model}_{args.cache_capacity}cap_{args.benchmark}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS.keys()), default="qwen1_5_moe_a2_7b")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cache_capacity", type=int, default=200,
                   help="Number of expert modules to keep on GPU at once.")
    p.add_argument("--num_examples", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--predict_topk_extra", type=int, default=4,
                   help="Used only by --mode all/ours (single-K). For hf_compare/hf_with_predictor, "
                        "use --predict_topk_extra_list instead.")
    p.add_argument("--predict_topk_extra_list", type=str, default="0,2,4",
                   help="Comma-separated list of k_extra values to sweep in hf_compare / "
                        "hf_with_predictor mode. Default sweeps top_k, top_k+2, top_k+4.")
    p.add_argument("--benchmark", default="wikitext_test",
                   choices=["wikitext_train", "wikitext_test", "mmlu", "gsm8k"])
    p.add_argument("--warmup", action="store_true", default=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--mode",
                   choices=["all", "hf_offload", "hf_with_predictor", "hf_compare", "ours"],
                   default="hf_compare",
                   help="hf_compare: HF baseline + HF+predictor (RECOMMENDED). "
                        "hf_offload: HF baseline only. "
                        "hf_with_predictor: HF + our predictor only. "
                        "all: HF baseline + our standalone LRU + our prefetch (older 3-way). "
                        "ours: only our standalone system (LRU + prefetched).")
    p.add_argument("--gpu_memory_gb", type=float, default=12.0,
                   help="HF max_memory cap on GPU. Used in --mode hf_offload or all.")
    p.add_argument("--quantization", default="bf16",
                   choices=["4bit", "8bit", "bf16", "fp16"],
                   help="For --mode hf_offload only. 'all' mode always uses bf16 "
                        "for the OUR-system passes (4-bit modules don't move cleanly "
                        "between CPU and GPU).")
    p.add_argument("--offload_target", default="disk", choices=["disk", "cpu"],
                   help="Where HF accelerate places overflow weights. 'disk' = NVMe "
                        "(matches our simulation's GPU+NVMe tier). 'cpu' = pinned DRAM.")
    p.add_argument("--offload_dir", default="./hf_offload_cache",
                   help="Disk directory accelerate writes offloaded weights to. "
                        "Should live on the NVMe SSD you want to measure.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

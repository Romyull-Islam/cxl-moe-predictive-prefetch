"""
prefetch_constrained.py — constrained-memory prefetch demo.

Run real Mixtral / DeepSeek / Qwen 4-bit inference, capture (predicted experts,
actual gate-selected experts) at every layer for every token, then post-hoc
simulate two systems on the same recorded trace:

  (1) BASELINE — fixed-size LRU expert cache on GPU; misses fault synchronously.
  (2) PREFETCHED — same LRU cache, but a PrefetchScheduler runs the predictor's
      output through a confidence-keyed priority queue and async-copies experts
      to the GPU on a separate CUDA stream during prior layers' compute.

Per-token latency model:
    layer_latency = T_layer_compute + sum(miss_count * T_expert_transfer)
    For the predictor path, also add T_predictor per layer.

T_layer_compute, T_predictor, and T_expert_transfer are all MEASURED on the
real model on this GPU. The LRU + scheduler logic is simulated in Python so we
can swap policies without re-running the LLM.

Headline output:
  - per-token latency: baseline vs prefetched
  - prefetch hit rate (fraction of expert demands served from cache)
  - cold-miss rate, eviction rate
  - "feasible on a {gb} GiB GPU?" verdict

Usage:
  python prefetch_constrained.py --model mixtral_8x7b --gpu 0 \
      --gpu_memory_gb 12 --predict_topk_extra 4 \
      --num_examples 32 --batch_size 1 --max_length 256
"""

import argparse
import heapq
import json
import os
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from tqdm import tqdm

from expert_predictor_topk import GlobalMultiStepPredictor


if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_len, layer_idx=None):
        return self.get_seq_length()
    DynamicCache.get_usable_length = get_usable_length

# DeepSeek's old modeling code reads `past_key_values.seen_tokens` which newer
# transformers DynamicCache replaced with `get_seq_length()`. Shim it.
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

# Same for `get_max_length()` — DynamicCache has no max (grows dynamically),
# so return None which tells the modeling code "unbounded".
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None


# Per-expert size scales with bits/param. Reference 4-bit values measured.
QUANT_BYTES_MULT = {"4bit": 1.0, "8bit": 2.0, "fp16": 4.0}

MODELS = {
    "mixtral_8x7b": dict(
        hf="mistralai/Mixtral-8x7B-v0.1",
        moe_attr="block_sparse_moe",
        top_k=2,
        ckpt="mixtral_8x7b_multi_logs/mixtral_8x7b_multi_predictor_topk2_d4.pt",
        npz="mixtral_8x7b_wikitext_logs/mixtral_8x7b_wikitext_all_layers_raw.npz",
        hidden_dim=1024, num_layers_mlp=3,
        trust_remote_code=False,
        approx_expert_bytes_4bit=88 * 1024 * 1024,
        shared_per_layer=0,
        non_expert_overhead_gb_4bit=2.5,
    ),
    "deepseek_moe_16b": dict(
        hf="deepseek-ai/deepseek-moe-16b-base",
        moe_attr="mlp",
        top_k=6,
        ckpt="deepseek_moe_16b_multi_logs/deepseek_moe_16b_multi_predictor_topk6_d4.pt",
        npz="deepseek_moe_16b_wikitext_logs/deepseek_moe_16b_wikitext_all_layers_raw.npz",
        hidden_dim=2048, num_layers_mlp=4,
        trust_remote_code=True,
        approx_expert_bytes_4bit=6 * 1024 * 1024,
        shared_per_layer=2,
        non_expert_overhead_gb_4bit=1.5,
    ),
    "qwen1_5_moe_a2_7b": dict(
        hf="Qwen/Qwen1.5-MoE-A2.7B",
        moe_attr="mlp",
        top_k=4,
        ckpt="qwen1_5_moe_a2_7b_multi_logs/qwen1_5_moe_a2_7b_multi_predictor_topk4_d4.pt",
        npz="qwen1_5_moe_a2_7b_wikitext_logs/qwen1_5_moe_a2_7b_wikitext_all_layers_raw.npz",
        hidden_dim=2048, num_layers_mlp=4,
        trust_remote_code=False,
        approx_expert_bytes_4bit=6 * 1024 * 1024,
        shared_per_layer=1,
        non_expert_overhead_gb_4bit=1.5,
    ),
}


def get_quantized_sizes(cfg, quantization):
    mult = QUANT_BYTES_MULT[quantization]
    return dict(
        approx_expert_bytes=int(cfg["approx_expert_bytes_4bit"] * mult),
        non_expert_overhead_gb=cfg["non_expert_overhead_gb_4bit"] * mult,
    )


# ---------------- LRU cache ----------------

class LRUExpertCache:
    """
    Fixed-capacity GPU expert cache.
    Keys are (layer_idx, expert_idx). LRU eviction on insert.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __contains__(self, key):
        return key in self.cache

    def touch(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return True
        return False

    def insert(self, key):
        """Insert key; if at capacity, evict LRU."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return None
        if len(self.cache) >= self.capacity:
            evicted, _ = self.cache.popitem(last=False)
            self.evictions += 1
        else:
            evicted = None
        self.cache[key] = True
        return evicted

    def access(self, key):
        """Record a demand access. Returns True on hit, False on cold miss
        (caller must do a sync fault and then insert)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        return False


# ---------------- Prefetch scheduler ----------------

class PrefetchScheduler:
    """
    Maintains a confidence-keyed priority queue of pending prefetches.
    upsert((target_layer, expert_id), confidence) — keeps highest confidence.
    pop_for_layer(L) — issue prefetches for predictions targeting layer <= L.

    In the simulator, "issuing" a prefetch means inserting into the LRU cache
    immediately at the moment we *would* call cudaMemcpyAsync. The transfer
    itself is modeled as a per-call latency that overlaps with subsequent
    layer compute up to the budget.
    """
    def __init__(self, cache, transfer_ms_per_expert, layer_compute_ms):
        self.cache = cache
        self.transfer_ms = transfer_ms_per_expert
        self.compute_ms = layer_compute_ms
        # priority dict: key=(target_layer, expert) -> max confidence seen
        self.pending = {}
        # successful prefetches per layer
        self.prefetched_per_layer = defaultdict(list)
        self.wasted_prefetches = 0
        self.total_prefetches = 0
        self.dedup_skipped = 0  # times we skipped because already in cache

    def upsert(self, target_layer, expert_id, confidence):
        key = (target_layer, expert_id)
        prev = self.pending.get(key, -1.0)
        if confidence > prev:
            self.pending[key] = confidence

    def issue_for(self, current_layer, lookahead, max_concurrent=8):
        """
        Run the worker: process the priority queue in confidence order.

        A prefetch issued at layer `current_layer` for target layer `target` is
        in cache by the time target arrives only if:
            (target - current_layer) * compute_ms_per_layer  >=  transfer_ms

        i.e. the layers of compute between now and the target cover the transfer
        cost. We model multiple parallel DMA channels with `max_concurrent`.
        """
        import math
        layer_steps_to_complete = max(1, math.ceil(self.transfer_ms / self.compute_ms))
        issued_count = 0
        candidates = sorted(self.pending.items(), key=lambda kv: -kv[1])

        for (target_layer, expert_id), conf in candidates:
            if not (current_layer + 1 <= target_layer <= current_layer + lookahead):
                continue
            layers_until_target = target_layer - current_layer
            # Only issue prefetches that have enough lookahead to complete in time.
            if layers_until_target < layer_steps_to_complete:
                # Will arrive late — skip; falls through to regular cache miss.
                continue
            if issued_count >= max_concurrent:
                break  # respect parallel-channel limit
            if (target_layer, expert_id) not in self.cache:
                self.cache.insert((target_layer, expert_id))
                self.prefetched_per_layer[target_layer].append(expert_id)
                self.total_prefetches += 1
                issued_count += 1
            else:
                self.cache.touch((target_layer, expert_id))
                self.dedup_skipped += 1
            del self.pending[(target_layer, expert_id)]

    def mark_used(self, target_layer, used_expert_set):
        used = set(used_expert_set)
        prefetched = set(self.prefetched_per_layer.get(target_layer, []))
        self.wasted_prefetches += len(prefetched - used)


# ---------------- Real-time measurements (one-shot) ----------------

def measure_expert_transfer_one(expert_module, gpu_id, n_iters=30):
    """Time CPU(pinned)->GPU async copy of all expert params, mean ms."""
    params = list(expert_module.parameters())
    cpu_pins = [p.detach().to("cpu", copy=True).pin_memory() for p in params]
    stream = torch.cuda.Stream(device=gpu_id)
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            s.record(stream)
            for cp in cpu_pins:
                _ = cp.to(f"cuda:{gpu_id}", non_blocking=True)
            e.record(stream)
        stream.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.mean(times)), sum(p.numel() * p.element_size() for p in params)


def get_actual_topk(gate_out, top_k):
    if isinstance(gate_out, tuple):
        idx = gate_out[0]
        if idx.dim() == 3:
            idx = idx.view(-1, idx.shape[-1])
        return idx[:, :top_k]
    if gate_out.dim() == 3:
        b, s, e = gate_out.shape
        gate_out = gate_out.view(b * s, e)
    return torch.softmax(gate_out, dim=-1, dtype=torch.float32).topk(top_k, dim=-1).indices


# ---------------- Trace capture ----------------

def patch_and_capture(model, predictor, cfg, num_layers, gpu_id, top_k_extra,
                      stats):
    """
    Patch each MoE block to capture per-step:
      - predicted experts (top_k + top_k_extra) for d+1..d+4
      - softmax confidences for those predictions
      - actual gate top-K experts
      - timing of predictor + layer compute
    """
    layers = model.model.layers
    originals = {}
    pred_dev = next(predictor.parameters()).device
    top_k = cfg["top_k"]
    K = top_k + top_k_extra

    # Per-token trace: list of dicts, indexed by token-step within batch.
    # We accumulate one entry per (call_idx, layer_idx) triple — every layer is
    # called once per forward, so token_id = which row we're on.
    call_idx = {"v": 0}

    def make_patched(L, orig_forward, moe):
        def patched(self, hidden_states, *args, **kwargs):
            if hidden_states.dim() == 3:
                b, s, d = hidden_states.shape
                h_flat = hidden_states.reshape(-1, d)
            else:
                h_flat = hidden_states

            # ---- predictor (timed)
            lid = torch.full((h_flat.shape[0],), L, device=pred_dev, dtype=torch.long)
            ps = torch.cuda.Event(enable_timing=True); pe = torch.cuda.Event(enable_timing=True)
            ps.record()
            with torch.no_grad():
                pred_logits = predictor(h_flat.to(pred_dev).half(), lid)  # [N,4,E]
            pe.record(); torch.cuda.synchronize()
            stats["pred_ms"][L].append(ps.elapsed_time(pe))

            # softmax probs for confidence
            pred_probs = pred_logits.softmax(dim=-1)  # [N,4,E]
            top_v, top_i = pred_probs.topk(K, dim=-1)  # values, indices [N,4,K]

            # ---- actual gate top-K
            with torch.no_grad():
                gate_in = hidden_states if hidden_states.dim() == 3 else hidden_states.unsqueeze(0)
                gate_out = moe.gate(gate_in)
            actual_topk = get_actual_topk(gate_out, top_k).to(pred_dev)  # [N, top_k]

            # ---- timed layer compute
            ms = torch.cuda.Event(enable_timing=True); me = torch.cuda.Event(enable_timing=True)
            ms.record()
            out = orig_forward(hidden_states, *args, **kwargs)
            me.record(); torch.cuda.synchronize()
            stats["moe_ms"][L].append(ms.elapsed_time(me))

            # ---- record trace per-token
            top_i_cpu = top_i.cpu().numpy()    # [N, 4, K]
            top_v_cpu = top_v.cpu().numpy()    # [N, 4, K]
            actual_cpu = actual_topk.cpu().numpy()  # [N, top_k]
            for t in range(top_i_cpu.shape[0]):
                stats["trace"].append(dict(
                    layer=L, token=call_idx["v"] + t,
                    pred_idx=top_i_cpu[t],   # [4, K]
                    pred_prob=top_v_cpu[t],  # [4, K]
                    actual=actual_cpu[t],    # [top_k]
                ))
            return out
        return patched

    # We want call_idx to advance once per token, not per layer. Use a per-batch
    # counter keyed on the first MoE layer to mark new tokens.
    first_moe_layer = None
    for L in range(num_layers):
        moe = getattr(layers[L], cfg["moe_attr"])
        if hasattr(moe, "gate"):
            if first_moe_layer is None:
                first_moe_layer = L
            originals[L] = moe.forward
            moe.forward = make_patched(L, moe.forward, moe).__get__(moe)

    return originals, first_moe_layer


# ---------------- Simulator: walk the trace under two policies ----------------

def simulate(trace, num_layers, top_k, lookahead, cache_capacity,
             transfer_ms, compute_ms_per_layer, predictor_ms,
             use_predictor, hot_experts=None, hot_layers=(),
             prefetch_topk=None):
    """
    Walk through token-by-token, layer-by-layer. Each token has num_layers
    entries in trace. Returns per-token latency stats and cache stats.
    """
    cache = LRUExpertCache(cache_capacity)
    sched = PrefetchScheduler(cache, transfer_ms, compute_ms_per_layer)

    # Prepopulate cache with hot experts for cold-start protection.
    if hot_experts is not None:
        for (L, e) in hot_experts:
            cache.insert((L, e))

    # Group trace entries by token.
    by_token = defaultdict(dict)
    for ent in trace:
        by_token[ent["token"]][ent["layer"]] = ent

    per_token_latency = []
    miss_log = []          # list of (token, layer, miss_count)
    expert_demands = 0
    expert_hits = 0

    for t in sorted(by_token.keys()):
        token_layers = by_token[t]
        token_latency = 0.0

        if use_predictor:
            sched.pending.clear()
            sched.prefetched_per_layer.clear()

        for L in sorted(token_layers.keys()):
            ent = token_layers[L]

            # 1) Predictor pushes new predictions (only if using prefetcher)
            if use_predictor:
                pred_idx = ent["pred_idx"]   # [4, K_total]
                pred_prob = ent["pred_prob"] # [4, K_total]
                # Honor prefetch_topk to limit how many candidates per (layer, d)
                K_push = prefetch_topk if prefetch_topk is not None else pred_idx.shape[1]
                K_push = min(K_push, pred_idx.shape[1])
                for d_idx in range(pred_idx.shape[0]):
                    target_layer = L + d_idx + 1
                    if target_layer >= num_layers:
                        continue
                    for slot in range(K_push):
                        e = int(pred_idx[d_idx, slot])
                        c = float(pred_prob[d_idx, slot])
                        sched.upsert(target_layer, e, c)
                # Issue prefetches for the lookahead window.
                sched.issue_for(L, lookahead)
                token_latency += predictor_ms

            # 2) Demand: actual experts at this layer
            actual = [int(x) for x in ent["actual"]]
            misses_here = 0
            for e in actual:
                expert_demands += 1
                key = (L, e)
                if cache.access(key):
                    expert_hits += 1
                else:
                    cache.insert(key)
                    misses_here += 1
            token_latency += compute_ms_per_layer + misses_here * transfer_ms
            miss_log.append((t, L, misses_here))

            # 3) Track wasted prefetches for this layer
            if use_predictor:
                sched.mark_used(L, actual)

        per_token_latency.append(token_latency)

    return dict(
        per_token_latency=per_token_latency,
        mean_latency_ms=float(np.mean(per_token_latency)),
        median_latency_ms=float(np.median(per_token_latency)),
        cache_hit_rate=expert_hits / max(1, expert_demands),
        cache_misses=cache.misses,
        cache_evictions=cache.evictions,
        wasted_prefetches=sched.wasted_prefetches,
        total_prefetches=sched.total_prefetches,
        dedup_skipped=sched.dedup_skipped,
        expert_demands=expert_demands,
        expert_hits=expert_hits,
    )


# ---------------- Hot-expert preload ----------------

def hot_expert_keys_from_npz(npz_path, fraction):
    """Return list of (layer, expert) pairs for the most active experts."""
    if not os.path.exists(npz_path):
        return []
    data = np.load(npz_path, mmap_mode="r")
    freq = data["freq"]   # [num_layers, num_experts]
    flat = freq.flatten()
    n_keep = max(1, int(fraction * flat.size))
    top = np.argpartition(-flat, n_keep)[:n_keep]
    keys = [(int(idx // freq.shape[1]), int(idx % freq.shape[1])) for idx in top]
    return keys


# ---------------- Driver ----------------

def run(args):
    cfg = MODELS[args.model]
    gpu_id = args.gpu
    quant = args.quantization
    qsizes = get_quantized_sizes(cfg, quant)

    print(f"[{args.model}] loading {quant} model on cuda:{gpu_id}...")
    tok = AutoTokenizer.from_pretrained(cfg["hf"], trust_remote_code=cfg["trust_remote_code"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    device_map_arg = {"": gpu_id} if not args.shard else "auto"
    if quant == "4bit":
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["hf"], quantization_config=qcfg, device_map=device_map_arg,
            trust_remote_code=cfg["trust_remote_code"],
        )
    elif quant == "8bit":
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["hf"], quantization_config=qcfg, device_map=device_map_arg,
            trust_remote_code=cfg["trust_remote_code"],
        )
    elif quant == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["hf"], torch_dtype=torch.float16, device_map=device_map_arg,
            trust_remote_code=cfg["trust_remote_code"],
        )
    else:
        raise ValueError(f"unknown quantization: {quant}")
    model = model.eval()
    num_layers = model.config.num_hidden_layers

    print(f"loading predictor checkpoint...")
    meta = np.load(cfg["npz"], mmap_mode="r")
    hidden_size = int(meta["hidden_size"][0])
    num_experts = int(meta["num_experts_per_layer"][0])
    predictor = GlobalMultiStepPredictor(
        d_model=hidden_size, num_experts=num_experts,
        num_layers_total=num_layers, lookahead_depth=4,
        layer_embed_dim=32,
        hidden_dim=cfg["hidden_dim"], num_layers_mlp=cfg["num_layers_mlp"],
    )
    predictor.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu"))
    predictor = predictor.to(f"cuda:{gpu_id}").half().eval()

    # Measure transfer time on one expert. Find the first layer that actually
    # has an `experts` ModuleList — DeepSeek-MoE's layer 0 is a dense MLP.
    moe0 = None
    for L in range(num_layers):
        block = getattr(model.model.layers[L], cfg["moe_attr"])
        if hasattr(block, "experts") and hasattr(block, "gate"):
            moe0 = block
            print(f"first MoE layer with experts: layer {L}")
            break
    if moe0 is None:
        raise RuntimeError(f"no MoE block with `.experts` found in {args.model}")
    expert0 = moe0.experts[0]
    transfer_ms, expert_bytes = measure_expert_transfer_one(expert0, gpu_id)
    print(f"per-expert CPU->GPU transfer (measured): {transfer_ms:.3f} ms ({expert_bytes/1024**2:.2f} MB)")
    if args.simulate_transfer_ms is not None:
        transfer_ms = args.simulate_transfer_ms
        print(f"  ** OVERRIDING with simulated transfer time: {transfer_ms:.3f} ms")
        print(f"     (models slower memory tier — CXL SSD, NVMe, network-attached, etc.)")

    stats = {
        "pred_ms": defaultdict(list),
        "moe_ms": defaultdict(list),
        "trace": [],
    }

    print("patching forwards...")
    originals, _ = patch_and_capture(model, predictor, cfg, num_layers, gpu_id,
                                     args.predict_topk_extra, stats)

    bench = args.benchmark
    if bench == "wikitext_train":
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
    elif bench == "wikitext_test":
        ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
    elif bench == "mmlu":
        # Trace was extracted from auxiliary_train; eval uses the test split.
        # Strictly disjoint by construction.
        ds = load_dataset("cais/mmlu", "all", split="test")
        texts = [ex["question"] + "\n" + "\n".join(
                    f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(ex["choices"]))
                 for ex in ds]
    elif bench == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        texts = [ex["question"] for ex in ds]
    else:
        raise ValueError(f"unknown benchmark: {bench}")
    texts = texts[: args.num_examples]
    print(f"benchmark: {bench}  ({len(texts)} examples)")

    # Run actual inference (this is the only source of trace data)
    print(f"running {len(texts)} examples (batch={args.batch_size}, max_len={args.max_length})...")
    n_tokens = 0
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i: i + args.batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=args.max_length, padding=True).to(f"cuda:{gpu_id}")
        with torch.no_grad():
            _ = model(**enc, use_cache=False)
        n_tokens += enc.attention_mask.sum().item()

    for L, of in originals.items():
        getattr(model.model.layers[L], cfg["moe_attr"]).forward = of

    print(f"\ncaptured {n_tokens} tokens × {num_layers} layers = {len(stats['trace'])} trace entries")

    # Aggregated real-time measurements
    pred_ms = float(np.mean([t for v in stats["pred_ms"].values() for t in v]))
    compute_ms = float(np.mean([t for v in stats["moe_ms"].values() for t in v]))
    print(f"real measured: predictor={pred_ms:.3f} ms/call, layer_compute={compute_ms:.3f} ms/layer, transfer={transfer_ms:.3f} ms/expert")

    # Cache budget — account for non-expert weights AND always-on shared experts
    # Allow CLI override of non_expert_overhead_gb. Setting to 0 means
    # "treat gpu_memory_gb as expert cache budget directly" — useful when
    # non-expert weights are assumed to live permanently on GPU outside the cap.
    if args.non_expert_overhead_gb is not None:
        overhead_gb = args.non_expert_overhead_gb
    else:
        overhead_gb = qsizes["non_expert_overhead_gb"]
    expert_bytes = qsizes["approx_expert_bytes"]
    shared_count = num_layers * cfg["shared_per_layer"]
    shared_bytes = shared_count * expert_bytes
    shared_gb = shared_bytes / 1024**3
    expert_budget_gb = max(0.1, args.gpu_memory_gb - overhead_gb - shared_gb)
    cache_capacity = int(expert_budget_gb * 1024**3 / expert_bytes)
    total_experts = num_layers * num_experts
    # Budget can go ≤0 when overhead + shared_experts > gpu_memory_gb (e.g.,
    # Mixtral 8bit at 4 GB cap). Clamp to at least 1 cache slot so the LRU
    # remains well-defined; this represents a "no useful cache" regime where
    # the prefetcher demonstrates its lower-bound behavior.
    cache_capacity = max(1, min(cache_capacity, total_experts))
    if cache_capacity == 1:
        print("  WARNING: budget exhausted by overhead+shared; cache reduced to 1 slot.")
    print(f"\nGPU budget breakdown ({args.gpu_memory_gb} GiB total, {quant}):")
    print(f"  - non-expert overhead     : {overhead_gb:.2f} GiB  (attn + embed + norms + predictor)")
    print(f"  - shared experts (pinned) : {shared_gb:.3f} GiB  ({shared_count} × {expert_bytes/1024**2:.1f} MB)")
    print(f"  = expert cache budget     : {expert_budget_gb:.2f} GiB")
    print(f"  cache capacity = {cache_capacity} / {total_experts} routed experts "
          f"({100*cache_capacity/total_experts:.1f}%)")
    print(f"  active experts/layer/token: {cfg['top_k']} routed + {cfg['shared_per_layer']} shared "
          f"= {cfg['top_k'] + cfg['shared_per_layer']} total")

    # Hot-expert preload (top 5%)
    hot = hot_expert_keys_from_npz(cfg["npz"], fraction=0.05)
    if hot:
        print(f"  hot-expert preload: {len(hot)} pairs")

    # ---- Baseline (LRU only) ----
    print("\nsimulating BASELINE (LRU only)...")
    base = simulate(stats["trace"], num_layers, cfg["top_k"], lookahead=4,
                    cache_capacity=cache_capacity,
                    transfer_ms=transfer_ms,
                    compute_ms_per_layer=compute_ms,
                    predictor_ms=pred_ms,
                    use_predictor=False,
                    hot_experts=hot)

    # ---- Sweep over prefetch_topk ∈ {top_k, top_k+2, top_k+4} ----
    pref_by_K = {}
    if args.policy in ("prefetched", "both"):
        K_levels = [cfg["top_k"], cfg["top_k"] + 2, cfg["top_k"] + 4]
        for K in K_levels:
            if K > cfg["top_k"] + args.predict_topk_extra:
                print(f"  skipping K={K} (trace only contains top-{cfg['top_k'] + args.predict_topk_extra})")
                continue
            print(f"simulating PREFETCHED with prefetch_topk={K}...")
            pref_by_K[K] = simulate(
                stats["trace"], num_layers, cfg["top_k"], lookahead=4,
                cache_capacity=cache_capacity,
                transfer_ms=transfer_ms,
                compute_ms_per_layer=compute_ms,
                predictor_ms=pred_ms,
                use_predictor=True,
                hot_experts=hot,
                prefetch_topk=K,
            )

    # ---- Report ----
    print("\n" + "=" * 100)
    print(f"PREFETCH-CONSTRAINED REPORT — {args.model}  [policy={args.policy}]")
    print("=" * 100)
    print(f"Tokens simulated: {len(base['per_token_latency'])}")
    print(f"GPU cache cap:    {cache_capacity}/{total_experts} experts "
          f"({100*cache_capacity/total_experts:.1f}%)")
    print(f"top_k={cfg['top_k']}, lookahead=4")
    print()

    if args.policy == "baseline":
        # Baseline-only single-column view
        print("BASELINE — LRU expert cache, synchronous fault on miss")
        print("-" * 60)
        print(f"  Per-token latency (mean) : {base['mean_latency_ms']:.2f} ms")
        print(f"  Per-token throughput     : {1000/base['mean_latency_ms']:.4f} tok/s")
        print(f"  Cache hit rate           : {base['cache_hit_rate']:.4f}")
        print(f"  Cache misses             : {base['cache_misses']:,}")
        print(f"  Cache evictions          : {base['cache_evictions']:,}")

    elif args.policy == "prefetched":
        # Prefetched-only view, with speedup vs (silently-computed) baseline
        for K, r in pref_by_K.items():
            sp = base['mean_latency_ms'] / r['mean_latency_ms']
            print(f"PREFETCHED (K={K}) — predictor + confidence PQ + LRU + hot preload")
            print("-" * 60)
            print(f"  Per-token latency (mean) : {r['mean_latency_ms']:.2f} ms")
            print(f"  Per-token throughput     : {1000/r['mean_latency_ms']:.4f} tok/s")
            print(f"  Cache hit rate           : {r['cache_hit_rate']:.4f}")
            print(f"  Cache misses             : {r['cache_misses']:,}")
            print(f"  Cache evictions          : {r['cache_evictions']:,}")
            print(f"  Total prefetches         : {r['total_prefetches']:,}")
            print(f"  Dedup skipped            : {r['dedup_skipped']:,}")
            print(f"  Wasted prefetches        : {r['wasted_prefetches']:,}")
            print(f"  Waste rate               : {r['wasted_prefetches']/max(1,r['total_prefetches']):.4f}")
            print(f"  Speedup vs baseline      : {sp:.2f}×")
            print()

    else:  # both — original side-by-side table
        headers = ["Baseline"] + [f"Pref@K={K}" for K in pref_by_K]
        rows = [
            ("Mean per-token latency (ms)", [base["mean_latency_ms"]] +
                [r["mean_latency_ms"] for r in pref_by_K.values()]),
            ("Cache hit rate",              [base["cache_hit_rate"]] +
                [r["cache_hit_rate"] for r in pref_by_K.values()]),
            ("Cache misses",                [base["cache_misses"]] +
                [r["cache_misses"] for r in pref_by_K.values()]),
            ("Cache evictions",             [base["cache_evictions"]] +
                [r["cache_evictions"] for r in pref_by_K.values()]),
            ("Total prefetches",            ["-"] +
                [r["total_prefetches"] for r in pref_by_K.values()]),
            ("Dedup skipped (already cached)", ["-"] +
                [r["dedup_skipped"] for r in pref_by_K.values()]),
            ("Wasted prefetches (unused)",  ["-"] +
                [r["wasted_prefetches"] for r in pref_by_K.values()]),
            ("Waste rate (waste / total)",  ["-"] +
                [r["wasted_prefetches"] / max(1, r["total_prefetches"])
                 for r in pref_by_K.values()]),
            ("Speedup vs baseline",         ["1.00×"] +
                [f"{base['mean_latency_ms']/r['mean_latency_ms']:.2f}×"
                 for r in pref_by_K.values()]),
        ]

        col_w = 18
        print(f"{'Metric':<34}" + "".join(f"{h:<{col_w}}" for h in headers))
        print("-" * (34 + col_w * len(headers)))
        for label, vals in rows:
            cells = []
            for v in vals:
                if isinstance(v, float):
                    cells.append(f"{v:<{col_w}.4f}")
                elif isinstance(v, int):
                    cells.append(f"{v:<{col_w}d}")
                else:
                    cells.append(f"{v:<{col_w}}")
            print(f"{label:<34}" + "".join(cells))
    print("=" * 100)

    out_path = args.out or f"prefetch_constrained_{args.model}_{args.gpu_memory_gb}gb.json"
    with open(out_path, "w") as f:
        json.dump(dict(
            model=args.model,
            gpu_memory_gb=args.gpu_memory_gb,
            cache_capacity=cache_capacity,
            total_experts=total_experts,
            tokens=len(base["per_token_latency"]),
            measured=dict(predictor_ms=pred_ms, layer_compute_ms=compute_ms,
                           transfer_ms=transfer_ms),
            baseline=base,
            prefetched_by_K={str(K): r for K, r in pref_by_K.items()},
            speedup_by_K={str(K): base["mean_latency_ms"] / r["mean_latency_ms"]
                           for K, r in pref_by_K.items()},
        ), f, indent=2, default=lambda o: list(o) if hasattr(o, "__iter__") else str(o))
    print(f"\nReport saved -> {out_path}")

    # ---- Optional: show one model.generate() output for the demo ----
    if args.show_generation:
        print("\n" + "=" * 100)
        print("MODEL OUTPUT (one demo generation, model.generate(), unpatched forwards)")
        print("=" * 100)
        prompt_text = texts[0] if texts else ""
        print(f"PROMPT:\n{prompt_text}\n")
        print("-" * 60)
        enc = tok(prompt_text, return_tensors="pt", truncation=True,
                  max_length=args.max_length).to(f"cuda:{gpu_id}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(**enc, max_new_tokens=args.gen_tokens,
                                      do_sample=False,
                                      pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        gen_dt = time.perf_counter() - t0
        new_tokens = int(out_ids.shape[1] - enc.input_ids.shape[1])
        gen_text = tok.decode(out_ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"OUTPUT ({new_tokens} new tokens in {gen_dt:.2f}s, "
              f"{new_tokens/max(gen_dt,1e-9):.2f} tok/s on this dev GPU):")
        print(gen_text)
        print("=" * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS.keys()), default="mixtral_8x7b")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpu_memory_gb", type=float, default=12.0,
                   help="Simulated GPU memory cap. Mixtral 4-bit fully loaded is ~24 GB.")
    p.add_argument("--predict_topk_extra", type=int, default=4,
                   help="Prefetch top_k + this many experts per layer (over-provision).")
    p.add_argument("--num_examples", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--simulate_transfer_ms", type=float, default=None,
                   help="Override measured CPU->GPU transfer time. Use to model "
                        "slower memory tiers (CXL: ~5 ms, NVMe SSD: ~50 ms).")
    p.add_argument("--non_expert_overhead_gb", type=float, default=None,
                   help="Override the non-expert overhead estimate. Set to 0 to "
                        "treat --gpu_memory_gb as pure expert cache budget "
                        "(non-expert weights assumed loaded outside the cap).")
    p.add_argument("--benchmark", default="wikitext_test",
                   choices=["wikitext_train", "wikitext_test", "mmlu", "gsm8k"],
                   help="Test data for the prefetch demo.")
    p.add_argument("--quantization", default="4bit",
                   choices=["4bit", "8bit", "fp16"],
                   help="Model quantization. fp16 needs more GPU memory; "
                        "Mixtral fp16 requires multi-GPU sharding.")
    p.add_argument("--shard", action="store_true",
                   help="Shard the LLM across all visible GPUs (device_map='auto'). "
                        "Required for Mixtral fp16.")
    p.add_argument("--policy", default="both",
                   choices=["baseline", "prefetched", "both"],
                   help="Which cache policy to report. "
                        "'baseline' = LRU only (no predictor); "
                        "'prefetched' = predictor + confidence-PQ + LRU + hot preload; "
                        "'both' (default) = side-by-side comparison with speedup.")
    p.add_argument("--show_generation", action="store_true",
                   help="After the simulator report, also run one model.generate() "
                        "on the first benchmark prompt and print the output text. "
                        "Adds ~10-30s; for demo recordings only.")
    p.add_argument("--gen_tokens", type=int, default=64,
                   help="Number of new tokens to generate when --show_generation is set.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

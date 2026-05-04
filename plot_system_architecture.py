"""
Render a single publication-quality system architecture diagram for the
predictive prefetcher.

Output: figures/system_architecture.png

The figure shows, in one frame:
  (1) the frozen MoE backbone with a tap at every layer,
  (2) the GlobalMultiStepPredictor (MLP trunk + layer embed + 4 heads),
  (3) the confidence-keyed priority queue / scheduler,
  (4) the LRU expert cache + hot-expert preload on GPU,
  (5) the NVMe SSD expert pool with async transfer stream.

The arrow style distinguishes synchronous compute path (solid) from
asynchronous prefetch path (dashed).
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})


# --- palette --------------------------------------------------
GPU_BG       = "#f1f5f9"
GPU_EDGE     = "#1e293b"

BACKBONE_FILL = "#dbeafe"
BACKBONE_EDGE = "#1e3a8a"
TAP_FILL      = "#fde68a"
TAP_EDGE      = "#92400e"

PRED_FILL    = "#dcfce7"
PRED_EDGE    = "#166534"

PQ_FILL      = "#fef3c7"
PQ_EDGE      = "#92400e"

CACHE_FILL   = "#fce7f3"
CACHE_EDGE   = "#9d174d"
HOT_FILL     = "#fee2e2"
HOT_EDGE     = "#991b1b"

NVME_FILL    = "#e5e7eb"
NVME_EDGE    = "#374151"

ARR_SYNC     = "#1f2937"
ARR_PRED     = "#166534"
ARR_PREFETCH = "#9d174d"


def rbox(ax, x, y, w, h, text, fc, ec, fontsize=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fc, edgecolor=ec, linewidth=1.5,
    ))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight)


def arrow(ax, x1, y1, x2, y2, color="#1f2937", lw=1.4, ls="-", ms=14):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, linestyle=ls,
    ))


# --- canvas ---------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 7.6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

# === Title ================================================
ax.text(0.2, 7.65, "System architecture: predictive expert prefetcher for MoE inference",
        fontsize=13, weight="bold", color="#0f172a")
ax.text(0.2, 7.30,
        "Predictor reads $h_l$ at layer $l$ and forecasts top-$K$ experts for layers $l{+}1\\ldots l{+}4$. "
        "Predictions feed a confidence-keyed PQ that issues async NVMe$\\to$GPU copies on a separate CUDA stream.",
        fontsize=9.2, color="#334155")


# === GPU box (encloses backbone + predictor + cache) =======
gpu_x, gpu_y, gpu_w, gpu_h = 0.25, 0.4, 9.6, 6.5
ax.add_patch(FancyBboxPatch(
    (gpu_x, gpu_y), gpu_w, gpu_h,
    boxstyle="round,pad=0.02,rounding_size=0.18",
    facecolor=GPU_BG, edgecolor=GPU_EDGE, linewidth=1.6,
))
ax.text(gpu_x + 0.20, gpu_y + gpu_h - 0.30,
        "GPU (X GiB unified pool)", fontsize=10, weight="bold", color=GPU_EDGE)


# === (1) Frozen MoE backbone (left column) =================
bb_x = gpu_x + 0.30
bb_y = gpu_y + 0.30
bb_w = 1.95
bb_h = gpu_h - 0.85
ax.add_patch(FancyBboxPatch(
    (bb_x, bb_y), bb_w, bb_h,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    facecolor="white", edgecolor=BACKBONE_EDGE, linewidth=1.4, linestyle="--",
))
ax.text(bb_x + bb_w / 2, bb_y + bb_h - 0.18,
        "Frozen MoE backbone", ha="center", fontsize=9.5, weight="bold",
        color=BACKBONE_EDGE)

# Stack of layer slots
layer_labels = ["…", r"layer $l\!+\!4$", r"layer $l\!+\!3$",
                r"layer $l\!+\!2$", r"layer $l\!+\!1$",
                r"layer $l$  (tap)", r"layer $l\!-\!1$"]
n_show = len(layer_labels)
slot_top = bb_y + bb_h - 0.45
slot_h = 0.42
for i, lbl in enumerate(layer_labels):
    yy = slot_top - i * slot_h
    is_tap = "tap" in lbl
    fc = TAP_FILL if is_tap else BACKBONE_FILL
    ec = TAP_EDGE if is_tap else BACKBONE_EDGE
    ax.add_patch(Rectangle((bb_x + 0.15, yy - 0.32), bb_w - 0.30, 0.32,
                           facecolor=fc, edgecolor=ec, linewidth=0.9))
    ax.text(bb_x + bb_w / 2, yy - 0.16, lbl,
            ha="center", va="center", fontsize=8.3,
            weight="bold" if is_tap else "normal")


# === (2) Predictor module (center) =========================
pred_x = 3.10
pred_y = 4.10
pred_w = 4.10
pred_h = 2.60
ax.add_patch(FancyBboxPatch(
    (pred_x, pred_y), pred_w, pred_h,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    facecolor=PRED_FILL, edgecolor=PRED_EDGE, linewidth=1.4,
))
ax.text(pred_x + pred_w / 2, pred_y + pred_h - 0.22,
        "GlobalMultiStepPredictor", ha="center",
        fontsize=10, weight="bold", color=PRED_EDGE)

# layer-id chip + layer embedding
emb_x = pred_x + 0.15
emb_y = pred_y + 0.55
ax.add_patch(Rectangle((emb_x, emb_y), 0.95, 0.45,
                       facecolor="white", edgecolor=PRED_EDGE, linewidth=1.0))
ax.text(emb_x + 0.475, emb_y + 0.225,
        r"$\mathrm{Embed}(l)$",
        ha="center", va="center", fontsize=8.5)

# h_l input chip
hl_x = pred_x + 0.15
hl_y = pred_y + 1.40
ax.add_patch(Rectangle((hl_x, hl_y), 0.95, 0.45,
                       facecolor="white", edgecolor=TAP_EDGE, linewidth=1.0))
ax.text(hl_x + 0.475, hl_y + 0.225,
        r"$h_l$",
        ha="center", va="center", fontsize=10, color=TAP_EDGE, weight="bold")

# concat
cat_x = pred_x + 1.30
cat_y = pred_y + 0.95
ax.add_patch(FancyBboxPatch(
    (cat_x, cat_y), 0.50, 0.55,
    boxstyle="round,pad=0.02,rounding_size=0.06",
    facecolor="white", edgecolor=PRED_EDGE, linewidth=1.0,
))
ax.text(cat_x + 0.25, cat_y + 0.275, "[ ; ]",
        ha="center", va="center", fontsize=10)
arrow(ax, hl_x + 0.95, hl_y + 0.225, cat_x, cat_y + 0.45,
      color=PRED_EDGE, lw=1.0, ms=10)
arrow(ax, emb_x + 0.95, emb_y + 0.225, cat_x, cat_y + 0.10,
      color=PRED_EDGE, lw=1.0, ms=10)

# trunk (3-4 FC + ReLU)
trunk_x = cat_x + 0.70
trunk_y = pred_y + 0.55
trunk_w = 1.05
trunk_h = 1.35
ax.add_patch(FancyBboxPatch(
    (trunk_x, trunk_y), trunk_w, trunk_h,
    boxstyle="round,pad=0.02,rounding_size=0.06",
    facecolor="white", edgecolor=PRED_EDGE, linewidth=1.0,
))
ax.text(trunk_x + trunk_w / 2, trunk_y + trunk_h - 0.15,
        "MLP trunk", ha="center", fontsize=8.5, weight="bold")
ax.text(trunk_x + trunk_w / 2, trunk_y + trunk_h / 2 - 0.05,
        "FC + ReLU\n× (3–4)\n[1024 / 2048]",
        ha="center", va="center", fontsize=8)
arrow(ax, cat_x + 0.50, cat_y + 0.275, trunk_x, trunk_y + trunk_h / 2,
      color=PRED_EDGE, lw=1.0, ms=10)

# 4 heads
heads_x = trunk_x + trunk_w + 0.20
head_w = 0.95
head_h = 0.42
head_gap = 0.07
for i in range(4):
    yy = trunk_y + trunk_h - 0.18 - i * (head_h + head_gap)
    ax.add_patch(Rectangle((heads_x, yy - head_h), head_w, head_h,
                           facecolor="white", edgecolor=PRED_EDGE, linewidth=1.0))
    ax.text(heads_x + head_w / 2, yy - head_h / 2,
            f"head $d{{=}}{i+1}$",
            ha="center", va="center", fontsize=8)
    arrow(ax, trunk_x + trunk_w, trunk_y + trunk_h / 2,
          heads_x, yy - head_h / 2,
          color=PRED_EDGE, lw=0.7, ms=8)

ax.text(heads_x + head_w / 2, trunk_y - 0.05,
        "logits over $E$ experts",
        ha="center", fontsize=7.5, color=PRED_EDGE, style="italic")


# === (3) Priority queue / scheduler =========================
pq_x = 7.45
pq_y = 4.50
pq_w = 2.10
pq_h = 1.85
ax.add_patch(FancyBboxPatch(
    (pq_x, pq_y), pq_w, pq_h,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    facecolor=PQ_FILL, edgecolor=PQ_EDGE, linewidth=1.4,
))
ax.text(pq_x + pq_w / 2, pq_y + pq_h - 0.22,
        "Confidence-keyed PQ\n+ scheduler", ha="center",
        fontsize=9.5, weight="bold", color=PQ_EDGE)

bullets = [
    r"key = $(L', e)$",
    r"priority = max softmax",
    r"feasibility check:",
    r"$(L'\!-\!L)\!\cdot\!T_c \!\geq\! T_x$",
]
for i, b in enumerate(bullets):
    ax.text(pq_x + 0.12, pq_y + pq_h - 0.55 - i * 0.27, "• " + b,
            ha="left", fontsize=8, color="#0f172a")


# === (4) LRU cache + hot preload (lower-center, on GPU) =====
cache_x = 3.10
cache_y = 1.20
cache_w = 4.10
cache_h = 2.60
ax.add_patch(FancyBboxPatch(
    (cache_x, cache_y), cache_w, cache_h,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    facecolor=CACHE_FILL, edgecolor=CACHE_EDGE, linewidth=1.4,
))
ax.text(cache_x + cache_w / 2, cache_y + cache_h - 0.22,
        "LRU expert cache  +  hot-expert preload", ha="center",
        fontsize=10, weight="bold", color=CACHE_EDGE)

# little expert tile grid — visualize as 4×6 squares with some highlighted
tile_w = 0.46
tile_h = 0.36
grid_x0 = cache_x + 0.30
grid_y0 = cache_y + 0.55
for r in range(3):
    for c in range(8):
        is_hot = (r == 0 and c in (1, 3, 6))
        is_evict = (r == 2 and c == 7)
        fc = HOT_FILL if is_hot else "white"
        ec = HOT_EDGE if is_hot else CACHE_EDGE
        if is_evict:
            fc = "#fee2e2"; ec = "#dc2626"
        ax.add_patch(Rectangle(
            (grid_x0 + c * tile_w, grid_y0 + r * tile_h),
            tile_w - 0.04, tile_h - 0.06,
            facecolor=fc, edgecolor=ec, linewidth=0.6,
        ))

ax.text(cache_x + 0.30, cache_y + 0.34,
        "■ hot preload  ■ resident  ■ recently evicted",
        fontsize=7.8, color="#0f172a")


# === (5) NVMe SSD pool (right) ==============================
nvme_x = 10.20
nvme_y = 0.85
nvme_w = 3.55
nvme_h = 5.25
ax.add_patch(FancyBboxPatch(
    (nvme_x, nvme_y), nvme_w, nvme_h,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    facecolor=NVME_FILL, edgecolor=NVME_EDGE, linewidth=1.6, linestyle="--",
))
ax.text(nvme_x + nvme_w / 2, nvme_y + nvme_h - 0.27,
        "NVMe SSD\n(expert pool)", ha="center",
        fontsize=10.5, weight="bold", color=NVME_EDGE)

# many tiny tiles to suggest a large pool
import numpy as np
np.random.seed(1)
n_rows, n_cols = 9, 7
tx0 = nvme_x + 0.20
ty0 = nvme_y + 0.55
ttw = (nvme_w - 0.40) / n_cols
tth = (nvme_h - 1.45) / n_rows
for r in range(n_rows):
    for c in range(n_cols):
        ax.add_patch(Rectangle(
            (tx0 + c * ttw, ty0 + r * tth),
            ttw - 0.05, tth - 0.05,
            facecolor="white", edgecolor=NVME_EDGE, linewidth=0.5,
        ))

ax.text(nvme_x + nvme_w / 2, nvme_y + 0.30,
        r"$T_x \approx 50\,\mathrm{ms}$ per expert read",
        ha="center", fontsize=8.5, color=NVME_EDGE, style="italic")


# ==== Arrows: data and control flow ========================
# (a) tap from layer l → predictor h_l input
tap_y = slot_top - 5 * slot_h - 0.16
arrow(ax, bb_x + bb_w, tap_y,
      hl_x, hl_y + 0.225,
      color=TAP_EDGE, lw=1.6)
ax.text((bb_x + bb_w + hl_x) / 2 - 0.10, tap_y + 0.18,
        r"$h_l$",
        fontsize=9, color=TAP_EDGE, weight="bold")

# (b) layer-id constant into embedding
ax.text(bb_x + bb_w + 0.15, emb_y + 0.5,
        r"$l$", fontsize=10, color=PRED_EDGE)
arrow(ax, bb_x + bb_w + 0.05, emb_y + 0.225,
      emb_x, emb_y + 0.225,
      color=PRED_EDGE, lw=1.0, ms=10)

# (c) heads → priority queue
arrow(ax, heads_x + head_w, pred_y + pred_h / 2,
      pq_x, pq_y + pq_h / 2,
      color=ARR_PRED, lw=1.6)
ax.text((heads_x + head_w + pq_x) / 2, pred_y + pred_h / 2 + 0.12,
        r"top-$K$ predictions",
        ha="center", fontsize=8, color=ARR_PRED)

# (d) PQ → NVMe (issue prefetch)
arrow(ax, pq_x + pq_w, pq_y + pq_h / 2,
      nvme_x, pq_y + pq_h / 2,
      color=ARR_PREFETCH, lw=1.6, ls="--")
ax.text((pq_x + pq_w + nvme_x) / 2, pq_y + pq_h / 2 + 0.18,
        "issue async copy",
        ha="center", fontsize=8, color=ARR_PREFETCH)

# (e) NVMe → cache (async transfer arriving)
arrow(ax, nvme_x, nvme_y + 1.80,
      cache_x + cache_w, cache_y + cache_h - 0.55,
      color=ARR_PREFETCH, lw=1.6, ls="--")
ax.text(nvme_x - 0.05, nvme_y + 2.10,
        "async stream\n(overlaps compute)",
        ha="right", fontsize=8, color=ARR_PREFETCH)

# (f) cache ↔ backbone (demand path)
arrow(ax, cache_x, cache_y + cache_h / 2,
      bb_x + bb_w, cache_y + cache_h / 2,
      color=ARR_SYNC, lw=1.4)
arrow(ax, bb_x + bb_w + 0.0, cache_y + cache_h / 2 - 0.30,
      cache_x, cache_y + cache_h / 2 - 0.30,
      color=ARR_SYNC, lw=1.4)
ax.text((bb_x + bb_w + cache_x) / 2, cache_y + cache_h / 2 + 0.18,
        "demand fetch",
        ha="center", fontsize=8, color=ARR_SYNC)
ax.text((bb_x + bb_w + cache_x) / 2, cache_y + cache_h / 2 - 0.50,
        "miss → sync NVMe load",
        ha="center", fontsize=8, color=ARR_SYNC)


# === Legend ===============================================
lg_x, lg_y = 0.30, 0.05
ax.add_patch(Rectangle((lg_x, lg_y), 13.5, 0.30,
                       facecolor="white", edgecolor="#cbd5e1", linewidth=0.6))
# sample arrows
ax.add_patch(FancyArrowPatch((lg_x + 0.1, lg_y + 0.15), (lg_x + 0.65, lg_y + 0.15),
                             arrowstyle="-|>", mutation_scale=10,
                             color=ARR_SYNC, linewidth=1.3))
ax.text(lg_x + 0.80, lg_y + 0.15, "synchronous compute / demand path",
        fontsize=8, va="center", color="#0f172a")

ax.add_patch(FancyArrowPatch((lg_x + 4.7, lg_y + 0.15), (lg_x + 5.25, lg_y + 0.15),
                             arrowstyle="-|>", mutation_scale=10,
                             color=ARR_PRED, linewidth=1.3))
ax.text(lg_x + 5.40, lg_y + 0.15, "predictor signal (per layer-token)",
        fontsize=8, va="center", color="#0f172a")

ax.add_patch(FancyArrowPatch((lg_x + 9.0, lg_y + 0.15), (lg_x + 9.55, lg_y + 0.15),
                             arrowstyle="-|>", mutation_scale=10,
                             color=ARR_PREFETCH, linewidth=1.3, linestyle="--"))
ax.text(lg_x + 9.70, lg_y + 0.15, "asynchronous prefetch (overlapped)",
        fontsize=8, va="center", color="#0f172a")


os.makedirs("figures", exist_ok=True)
out_path = "figures/system_architecture.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
print(f"saved -> {out_path}")
plt.close()

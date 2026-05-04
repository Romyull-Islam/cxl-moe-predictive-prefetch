"""
Render a single clear diagram of how the prefetcher works:
a layer-time pipeline with a compute stream and an async transfer
stream, showing predictions issued at layer l arriving in time to
serve layers l+1 .. l+4.

Output: figures/prefetcher_timeline.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})


# ----- colors --------------------------------------------------
COMPUTE_FILL = "#dbeafe"
COMPUTE_EDGE = "#1e3a8a"
TAP_FILL     = "#fde68a"
TAP_EDGE     = "#92400e"
PRED_FILL    = "#ecfdf5"
PRED_EDGE    = "#065f46"
PQ_FILL      = "#fef3c7"
PQ_EDGE      = "#92400e"
HIT_GREEN    = "#16a34a"
MISS_RED     = "#dc2626"
NVME_FILL    = "#f3f4f6"
NVME_EDGE    = "#374151"
TRANSFER_OK  = "#10b981"
TRANSFER_LATE = "#ef4444"


def rbox(ax, x, y, w, h, text, fc, ec, fontsize=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
    ))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight)


def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.4, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color,
    ))


# ----- canvas --------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 6.8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.0)
ax.axis("off")

# Layer column centers (the "time" axis)
LAYERS = ["l-1", "l", "l+1", "l+2", "l+3", "l+4"]
LX = [1.6, 3.4, 5.4, 7.4, 9.4, 11.4]    # x-center of each layer column
LW = 1.6                                  # column width

# ----- title --------------------------------------------------
ax.text(0.2, 6.55,
        "Predictive expert prefetching — one layer-time slice",
        fontsize=13, weight="bold", color="#111827")
ax.text(0.2, 6.20,
        "Prediction is issued at layer $l$; experts arrive on the transfer stream "
        "before they are demanded at layers $l{+}1..l{+}4$.",
        fontsize=9.5, color="#374151")


# ============ ROW 1 — COMPUTE STREAM ============
row1_y = 4.45
row1_h = 0.85
ax.text(0.15, row1_y + row1_h / 2,
        "GPU\ncompute\nstream",
        ha="left", va="center", fontsize=9, weight="bold", color=COMPUTE_EDGE)

for i, lbl in enumerate(LAYERS):
    is_tap = (lbl == "l")
    rbox(ax, LX[i] - LW / 2, row1_y, LW, row1_h,
         f"layer  {lbl}",
         TAP_FILL if is_tap else COMPUTE_FILL,
         TAP_EDGE if is_tap else COMPUTE_EDGE,
         fontsize=10, weight="bold" if is_tap else "normal")

# tiny "T_c" label between two layers
ax.annotate("", xy=(LX[3] - LW / 2 - 0.05, row1_y + 0.18),
            xytext=(LX[2] + LW / 2 + 0.05, row1_y + 0.18),
            arrowprops=dict(arrowstyle="<->", color="#6b7280", lw=1))
ax.text((LX[2] + LX[3]) / 2, row1_y - 0.05,
        r"$T_c$ (per-layer compute)",
        ha="center", va="top", fontsize=8.5, color="#6b7280")


# ============ TAP — predictor fires at layer l ============
# Arrow from layer l down to predictor box
pred_y = 2.75
pred_h = 0.75
pred_w = 1.7
pred_x = LX[1] - pred_w / 2
arrow(ax, LX[1], row1_y, LX[1], pred_y + pred_h, color=TAP_EDGE, lw=1.6)
ax.text(LX[1] + 0.12, (row1_y + pred_y + pred_h) / 2,
        r"$h_l$",
        fontsize=10, color=TAP_EDGE, va="center")

rbox(ax, pred_x, pred_y, pred_w, pred_h,
     "Predictor\n(MLP, 4-head)",
     PRED_FILL, PRED_EDGE, fontsize=9.5, weight="bold")


# ============ Priority queue ============
pq_y = pred_y - 0.05
pq_h = 0.85
pq_w = 2.6
pq_x = pred_x + pred_w + 0.55
rbox(ax, pq_x, pq_y, pq_w, pq_h,
     "Confidence-keyed\npriority queue",
     PQ_FILL, PQ_EDGE, fontsize=9, weight="bold")
arrow(ax, pred_x + pred_w, pred_y + pred_h / 2,
      pq_x, pq_y + pq_h / 2,
      color=PRED_EDGE, lw=1.4)

# small chip showing K predictions per d
chips_y = pq_y - 0.65
ax.text(pq_x + pq_w / 2, chips_y + 0.30,
        r"top-$K$ experts predicted for $l{+}1, l{+}2, l{+}3, l{+}4$",
        ha="center", fontsize=8.5, color=PQ_EDGE)


# ============ ROW 2 — TRANSFER STREAM ============
row2_y = 1.30
row2_h = 0.55
ax.text(0.15, row2_y + row2_h / 2,
        "NVMe → GPU\ntransfer stream",
        ha="left", va="center", fontsize=9, weight="bold", color=NVME_EDGE)

# Background lane for transfer stream
ax.add_patch(Rectangle((LX[1] - LW / 2 + 0.1, row2_y - 0.05),
                       LX[5] + LW / 2 - (LX[1] - LW / 2 + 0.1),
                       row2_h + 0.10,
                       facecolor=NVME_FILL, edgecolor=NVME_EDGE,
                       linewidth=0.8, linestyle="--"))

# Each prefetch is a colored bar that begins at "issue time" (just after l)
# and ends at "arrival time" (just before l+d). T_x = transfer latency.
prefetches = [
    # (target_layer_idx_in_LAYERS, label, color)
    (2, r"experts for $l{+}1$", TRANSFER_OK),
    (3, r"experts for $l{+}2$", TRANSFER_OK),
    (4, r"experts for $l{+}3$", TRANSFER_OK),
    (5, r"experts for $l{+}4$", TRANSFER_OK),
]

# Start (issue) just after layer l finishes its compute (right edge of layer l).
issue_x = LX[1] + LW / 2 + 0.05
bar_h = 0.12
gap = 0.04
for k, (tgt, lbl, col) in enumerate(prefetches):
    arrival_x = LX[tgt] - LW / 2 - 0.05
    yy = row2_y + row2_h - 0.07 - k * (bar_h + gap)
    ax.add_patch(Rectangle((issue_x, yy - bar_h),
                           arrival_x - issue_x, bar_h,
                           facecolor=col, edgecolor=col, alpha=0.85))
    ax.text(arrival_x + 0.07, yy - bar_h / 2, lbl,
            ha="left", va="center", fontsize=8, color="#111827")
    # hand-off arrow from end of bar UP to the consuming layer
    arrow(ax, arrival_x, yy - bar_h / 2,
          LX[tgt], row1_y - 0.02,
          color=HIT_GREEN, lw=1.0, ms=10)

# Issue arrows: from PQ down to issue point on transfer lane
arrow(ax, pq_x + pq_w / 2, pq_y,
      issue_x + 0.05, row2_y + row2_h - 0.05,
      color=PQ_EDGE, lw=1.2)


# ============ Feasibility annotation ============
# Show T_x window between l and l+1 on the transfer lane.
fb_y = row2_y - 0.45
ax.annotate("", xy=(LX[2] - LW / 2 - 0.05, fb_y),
            xytext=(issue_x, fb_y),
            arrowprops=dict(arrowstyle="<->", color="#6b7280", lw=1))
ax.text((issue_x + LX[2] - LW / 2) / 2, fb_y - 0.15,
        r"$T_x$ (transfer latency)",
        ha="center", va="top", fontsize=8.5, color="#6b7280")

# The feasibility box at far right
fbx_x, fbx_y, fbx_w, fbx_h = 11.0, 0.10, 2.85, 0.68
rbox(ax, fbx_x, fbx_y, fbx_w, fbx_h,
     r"Feasible iff $\;\Delta \cdot T_c \;\geq\; T_x$" "\n"
     r"(else hides only the latest layers)",
     "#f0f9ff", "#0c4a6e", fontsize=9)


# ============ ROW 3 — LRU cache state at consumer side ============
row3_y = 5.55
row3_h = 0.55
ax.text(0.15, row3_y + row3_h / 2,
        "LRU expert\ncache (GPU)",
        ha="left", va="center", fontsize=9, weight="bold", color=COMPUTE_EDGE)

for i, lbl in enumerate(LAYERS):
    if i < 1:
        continue
    # at layer l-1 / l: cache is just whatever was warm
    # at layers l+1..l+4: arrivals from the transfer stream produce hits
    is_consumer = i >= 2
    ec = HIT_GREEN if is_consumer else "#9ca3af"
    fc = "#dcfce7" if is_consumer else "#f3f4f6"
    txt = "HIT (prefetched)" if is_consumer else "warm / LRU"
    ax.add_patch(FancyBboxPatch(
        (LX[i] - LW / 2 + 0.05, row3_y), LW - 0.10, row3_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.0,
    ))
    ax.text(LX[i], row3_y + row3_h / 2, txt,
            ha="center", va="center", fontsize=8.5,
            color=HIT_GREEN if is_consumer else "#374151")


# ============ Legend ============
lg_x = 0.20
lg_y = 0.10
ax.add_patch(Rectangle((lg_x, lg_y), 4.7, 0.78,
                       facecolor="white", edgecolor="#d1d5db", linewidth=0.8))
ax.text(lg_x + 0.10, lg_y + 0.55,
        r"$T_c$: per-layer compute time     "
        r"$T_x$: expert transfer time     "
        r"$\Delta$: lookahead depth ($D{=}4$)",
        fontsize=8.5, color="#111827")
ax.text(lg_x + 0.10, lg_y + 0.22,
        r"Green bars = async copies on a separate CUDA stream "
        r"overlapping compute on the GPU.",
        fontsize=8.5, color="#111827")


os.makedirs("figures", exist_ok=True)
out_path = "figures/prefetcher_timeline.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
print(f"saved -> {out_path}")
plt.close()

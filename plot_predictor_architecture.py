"""
Render a publication-quality architecture diagram for the
GlobalMultiStepPredictor defined in expert_predictor_topk.py.

Output: figures/predictor_architecture.png

The diagram shows:
  (1) hidden state h_l tapped from MoE layer l of the frozen backbone
  (2) layer-id l passed through a learnable embedding
  (3) [h_l ; e_l] concatenation feeding a shared MLP trunk
  (4) D parallel linear heads producing expert logits for layers l+1 .. l+D
  (5) top-K@(l+d) feeding the prefetch scheduler / expert cache
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# --------------------------------------------------------------- styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

COL_BACKBONE = "#dbeafe"   # light blue
COL_BACKBONE_EDGE = "#1e3a8a"
COL_PREDICTOR = "#ecfdf5"  # light green
COL_PREDICTOR_EDGE = "#065f46"
COL_HEAD = "#fef3c7"       # light amber
COL_HEAD_EDGE = "#92400e"
COL_CACHE = "#fee2e2"      # light red
COL_CACHE_EDGE = "#991b1b"
COL_TENSOR = "#f3f4f6"
COL_TENSOR_EDGE = "#374151"


def box(ax, x, y, w, h, text, face, edge, fontsize=10, weight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight)


def tensor_chip(ax, x, y, w, h, text, fontsize=9):
    p = Rectangle((x, y), w, h, linewidth=1.0,
                  facecolor=COL_TENSOR, edgecolor=COL_TENSOR_EDGE)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, family="monospace")


def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.4, style="-|>"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color,
    )
    ax.add_patch(a)


# --------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.set_aspect("equal")
ax.axis("off")

# ============ (1) Frozen MoE backbone (left) ============
backbone_x, backbone_y = 0.3, 0.6
backbone_w, backbone_h = 2.2, 5.4
ax.add_patch(Rectangle((backbone_x, backbone_y), backbone_w, backbone_h,
                       facecolor=COL_BACKBONE, edgecolor=COL_BACKBONE_EDGE,
                       linewidth=1.4, linestyle="--"))
ax.text(backbone_x + backbone_w / 2, backbone_y + backbone_h + 0.18,
        "Frozen MoE backbone", ha="center", fontsize=10, weight="bold",
        color=COL_BACKBONE_EDGE)

layer_labels = [r"layer $l\!+\!4$", r"layer $l\!+\!3$",
                r"layer $l\!+\!2$", r"layer $l\!+\!1$",
                r"layer $l$ (tap)", r"layer $l\!-\!1$"]
n_show = len(layer_labels)
slot_h = (backbone_h - 0.4) / n_show
for i, lbl in enumerate(layer_labels):
    yy = backbone_y + 0.2 + (n_show - 1 - i) * slot_h
    is_tap = "tap" in lbl
    fc = "#fde68a" if is_tap else "#bfdbfe"
    ec = "#92400e" if is_tap else COL_BACKBONE_EDGE
    ax.add_patch(Rectangle((backbone_x + 0.2, yy), backbone_w - 0.4, slot_h - 0.1,
                           facecolor=fc, edgecolor=ec, linewidth=1.0))
    ax.text(backbone_x + backbone_w / 2, yy + (slot_h - 0.1) / 2, lbl,
            ha="center", va="center", fontsize=9)

# Tap arrow out of layer l
tap_y = backbone_y + 0.2 + (n_show - 1 - 4) * slot_h + (slot_h - 0.1) / 2
arrow(ax, backbone_x + backbone_w, tap_y, 3.4, tap_y, color="#92400e", lw=1.8)
ax.text(3.0, tap_y + 0.22, r"$h_l \in \mathbb{R}^{d_{\mathrm{model}}}$",
        fontsize=10, ha="center", color="#92400e")

# Layer-id input
lid_y = 1.3
ax.text(backbone_x + backbone_w + 0.55, lid_y + 0.22,
        r"layer id $l$", fontsize=10, ha="center")
arrow(ax, backbone_x + backbone_w + 0.05, lid_y, 3.4, lid_y, lw=1.4)


# ============ (2) Inputs & embedding ============
emb_x, emb_w, emb_h = 3.45, 1.4, 0.55
box(ax, emb_x, lid_y - emb_h / 2, emb_w, emb_h,
    r"Layer Embed" "\n" r"$\mathbb{R}^{|L|\times 32}$",
    COL_PREDICTOR, COL_PREDICTOR_EDGE, fontsize=8.5)

# tensor chips for h and e
tensor_chip(ax, emb_x + emb_w + 0.10, lid_y - 0.18, 0.55, 0.36,
            r"$\mathbf{e}_l$", fontsize=9)
tensor_chip(ax, emb_x + emb_w + 0.10, tap_y - 0.18, 0.55, 0.36,
            r"$\mathbf{h}_l$", fontsize=9)

# concat node
concat_x = emb_x + emb_w + 0.85
concat_y = (tap_y + lid_y) / 2
box(ax, concat_x, concat_y - 0.35, 0.65, 0.7,
    "concat", COL_PREDICTOR, COL_PREDICTOR_EDGE, fontsize=9)
arrow(ax, emb_x + emb_w + 0.65, tap_y, concat_x, concat_y + 0.18, lw=1.2)
arrow(ax, emb_x + emb_w + 0.65, lid_y, concat_x, concat_y - 0.18, lw=1.2)


# ============ (3) MLP trunk ============
trunk_x = concat_x + 0.95
trunk_y_top = 4.2
trunk_y_bot = 1.6
trunk_w = 1.7
ax.add_patch(FancyBboxPatch(
    (trunk_x, trunk_y_bot), trunk_w, trunk_y_top - trunk_y_bot,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    facecolor=COL_PREDICTOR, edgecolor=COL_PREDICTOR_EDGE, linewidth=1.6,
))
ax.text(trunk_x + trunk_w / 2, trunk_y_top - 0.25,
        "MLP trunk", ha="center", fontsize=10, weight="bold",
        color=COL_PREDICTOR_EDGE)

inner_layers = [
    r"Linear  $\to\;1024$",
    "ReLU",
    r"Linear  $\to\;1024$",
    "ReLU",
]
n_in = len(inner_layers)
inner_top = trunk_y_top - 0.55
inner_bot = trunk_y_bot + 0.20
for i, lbl in enumerate(inner_layers):
    yy = inner_top - (i + 0.5) * (inner_top - inner_bot) / n_in
    is_act = lbl == "ReLU"
    fc = "#ffffff" if not is_act else "#d1fae5"
    ax.add_patch(Rectangle(
        (trunk_x + 0.18, yy - 0.18), trunk_w - 0.36, 0.36,
        facecolor=fc, edgecolor=COL_PREDICTOR_EDGE, linewidth=0.9,
    ))
    ax.text(trunk_x + trunk_w / 2, yy, lbl,
            ha="center", va="center", fontsize=8.5)

arrow(ax, concat_x + 0.65, concat_y, trunk_x, concat_y, lw=1.4)

# tensor chip out of trunk
trunk_out_x = trunk_x + trunk_w + 0.15
tensor_chip(ax, trunk_out_x, concat_y - 0.22, 0.65, 0.44,
            r"$\mathbf{z}\in\mathbb{R}^{1024}$", fontsize=8.5)


# ============ (4) D parallel heads ============
heads_x = trunk_out_x + 0.95
head_w, head_h = 1.7, 0.62
head_gap = 0.18
head_top = 4.3
head_titles = [r"Head $d{=}1$", r"Head $d{=}2$",
               r"Head $d{=}3$", r"Head $d{=}4$"]
head_subs  = [r"$\to\,$logits @ $l\!+\!1$",
              r"$\to\,$logits @ $l\!+\!2$",
              r"$\to\,$logits @ $l\!+\!3$",
              r"$\to\,$logits @ $l\!+\!4$"]
for i, (t, s) in enumerate(zip(head_titles, head_subs)):
    yy = head_top - i * (head_h + head_gap)
    box(ax, heads_x, yy - head_h, head_w, head_h,
        f"{t}\n{s}", COL_HEAD, COL_HEAD_EDGE, fontsize=8.5)
    # connect from z chip to head
    arrow(ax, trunk_out_x + 0.65, concat_y,
          heads_x, yy - head_h / 2, lw=1.0, color="#6b7280")

ax.text(heads_x + head_w / 2,
        head_top + 0.25,
        "$D{=}4$ parallel linear heads",
        ha="center", fontsize=9, weight="bold", color=COL_HEAD_EDGE)


# ============ (5) prefetch / cache (right) ============
cache_x = heads_x + head_w + 0.7
cache_w = 2.2
cache_top = 4.8
cache_bot = 0.6
ax.add_patch(FancyBboxPatch(
    (cache_x, cache_bot), cache_w, cache_top - cache_bot,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    facecolor=COL_CACHE, edgecolor=COL_CACHE_EDGE, linewidth=1.6, linestyle="--",
))
ax.text(cache_x + cache_w / 2, cache_top - 0.25,
        "Prefetch scheduler\n+ LRU expert cache",
        ha="center", fontsize=9, weight="bold", color=COL_CACHE_EDGE)

bullets = [
    r"top-$K$ per future layer",
    r"confidence-keyed PQ",
    r"async copy on CUDA stream",
    r"feasible iff $\Delta\!\cdot\!T_c\!\geq\!T_x$",
]
for i, b in enumerate(bullets):
    yy = cache_top - 0.85 - i * 0.55
    ax.text(cache_x + 0.18, yy, "•  " + b,
            fontsize=8.5, ha="left", va="center", color="#1f2937")

# Arrows from each head into cache
for i in range(4):
    yy = head_top - i * (head_h + head_gap) - head_h / 2
    arrow(ax, heads_x + head_w, yy, cache_x, yy,
          color=COL_CACHE_EDGE, lw=1.0)


# ============ caption / footer ============
ax.text(0.3, 0.25,
        r"GlobalMultiStepPredictor — shared across all backbone layers; "
        r"trained on $(h_l, l) \to \{\text{top-}K\text{ experts at }l{+}1..l{+}4\}$ "
        r"with per-head BCE.",
        fontsize=9, color="#374151")


os.makedirs("figures", exist_ok=True)
out_path = "figures/predictor_architecture.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
print(f"saved -> {out_path}")
plt.close()

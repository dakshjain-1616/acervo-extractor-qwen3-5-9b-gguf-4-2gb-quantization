#!/usr/bin/env python3
"""
generate_infographics.py — Create all infographics for acervo-extractor-quant.

Outputs (written to both outputs/ and hf_exports/assets/):
  1. infographic_overview.png     — Perplexity / Speed / Size side-by-side bar chart
  2. infographic_memory.png       — RAM requirements heatmap across quant types
  3. infographic_pipeline.png     — Pipeline architecture flow diagram
  4. infographic_tradeoff.png     — Quality vs Speed vs Size bubble chart
"""

import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
ACCENT   = "#7B61FF"          # NEO purple
TEAL     = "#00C9A7"
ORANGE   = "#FF6B6B"
GOLD     = "#FFD166"
BG_DARK  = "#0F0F1A"
BG_CARD  = "#1A1A2E"
TEXT_LT  = "#E8E8F0"
TEXT_DIM = "#8888AA"
GRID_C   = "#2A2A4A"

FLOAT16_C  = ORANGE
Q8_0_C     = TEAL
Q4KM_C     = ACCENT

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor":   BG_CARD,
    "axes.edgecolor":   GRID_C,
    "axes.labelcolor":  TEXT_LT,
    "axes.titlecolor":  TEXT_LT,
    "xtick.color":      TEXT_LT,
    "ytick.color":      TEXT_LT,
    "text.color":       TEXT_LT,
    "grid.color":       GRID_C,
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

OUTPUTS  = Path("outputs")
HF_ASSETS = Path("hf_exports/assets")
OUTPUTS.mkdir(exist_ok=True)
HF_ASSETS.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    for d in (OUTPUTS, HF_ASSETS):
        p = d / name
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        print(f"  Saved → {p}")
    plt.close(fig)


# ===========================================================================
# 1. Overview — Perplexity / Speed / Size
# ===========================================================================
def make_overview() -> None:
    variants = ["float16", "Q8_0", "Q4_K_M"]
    colors   = [FLOAT16_C, Q8_0_C, Q4KM_C]
    ppl      = [18.43,  18.62,  19.54]
    speed    = [42.7,   45.3,   47.8]
    size_pct = [100,    50,     26]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        "acervo-extractor-qwen3.5-9b  ·  GGUF Q4_K_M Quantization Overview",
        fontsize=15, fontweight="bold", color=TEXT_LT, y=1.02,
    )

    bar_kw = dict(width=0.55, edgecolor="none", zorder=3)
    x = np.arange(len(variants))

    # — Perplexity —
    ax = axes[0]
    bars = ax.bar(x, ppl, color=colors, **bar_kw)
    ax.set_title("Perplexity  (↓ lower is better)", fontsize=12, pad=10)
    ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylim(17.5, 20.5)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    for bar, v in zip(bars, ppl):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.06, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_LT)
    ax.annotate("baseline", xy=(0, ppl[0]), xytext=(0.35, ppl[0]+0.55),
                fontsize=8, color=TEXT_DIM,
                arrowprops=dict(arrowstyle="->", color=TEXT_DIM, lw=0.8))

    # — Speed —
    ax = axes[1]
    bars = ax.bar(x, speed, color=colors, **bar_kw)
    ax.set_title("Inference Speed  (↑ tokens/sec)", fontsize=12, pad=10)
    ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylim(38, 53)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    for bar, v in zip(bars, speed):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.25, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_LT)
    ax.text(2, speed[2] - 2.2, "+12%", ha="center", fontsize=9,
            color=GOLD, fontweight="bold")

    # — Size —
    ax = axes[2]
    bars = ax.bar(x, size_pct, color=colors, **bar_kw)
    ax.set_title("Model File Size  (% of float16)", fontsize=12, pad=10)
    ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylim(0, 125)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    labels_size = ["18 GB\n(100%)", "9.5 GB\n(50%)", "4.7 GB\n(26%)"]
    for bar, v, lbl in zip(bars, size_pct, labels_size):
        ax.text(bar.get_x() + bar.get_width()/2, v + 2, lbl,
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT_LT)
    ax.text(2, size_pct[2] - 12, "−74%", ha="center", fontsize=9,
            color=GOLD, fontweight="bold")

    # Legend patch
    patches = [mpatches.Patch(color=c, label=v) for c, v in zip(colors, variants)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               framealpha=0, fontsize=10, bbox_to_anchor=(0.5, -0.06))

    # Footer
    fig.text(0.5, -0.10, "Made autonomously by NEO · https://heyneo.so  |  Source: mock/dry-run synthetic benchmark",
             ha="center", fontsize=8, color=TEXT_DIM)

    fig.tight_layout()
    _save(fig, "infographic_overview.png")


# ===========================================================================
# 2. Memory heatmap
# ===========================================================================
def make_memory() -> None:
    quant_types = ["float32", "float16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ1_M"]
    bits        = [32.0, 16.0, 8.5, 6.0, 5.5, 4.5, 3.5, 2.5, 1.5]
    weights_gb  = [33.53, 16.76, 8.91, 6.29, 5.77, 4.72, 3.67, 2.62, 1.58]
    peak_gb     = [40.23, 20.12, 10.69, 7.54, 6.92, 5.66, 4.41, 3.15, 1.89]
    ram_tiers   = [8, 16, 24, 32]

    n_quant  = len(quant_types)
    n_tiers  = len(ram_tiers)
    fits = np.array([[peak <= tier for tier in ram_tiers] for peak in peak_gb], dtype=float)

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(16, 6),
                                           gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        "RAM / VRAM Requirements  ·  acervo-extractor-qwen3.5-9b (9B params)",
        fontsize=14, fontweight="bold", color=TEXT_LT, y=1.02,
    )

    # — Heatmap —
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "fit", [ORANGE, TEAL], N=2)
    ax_heat.imshow(fits, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax_heat.set_xticks(range(n_tiers))
    ax_heat.set_xticklabels([f"{t} GB RAM" for t in ram_tiers], fontsize=10)
    ax_heat.set_yticks(range(n_quant))
    ax_heat.set_yticklabels(quant_types, fontsize=10)
    ax_heat.set_title("Fits in RAM tier?  (green = yes)", fontsize=11, pad=10)

    for i in range(n_quant):
        for j in range(n_tiers):
            sym = "✓" if fits[i, j] else "✗"
            col = BG_DARK
            ax_heat.text(j, i, sym, ha="center", va="center",
                         fontsize=14, fontweight="bold", color=col)

    # Highlight recommended row (Q4_K_M index = 5)
    rec_idx = quant_types.index("Q4_K_M")
    ax_heat.add_patch(plt.Rectangle((-0.5, rec_idx - 0.5), n_tiers, 1,
                                    fill=False, edgecolor=GOLD, linewidth=2.5, zorder=5))
    ax_heat.text(n_tiers - 0.45, rec_idx, "◄ recommended",
                 va="center", fontsize=9, color=GOLD, fontweight="bold")

    # — Peak RAM bar chart —
    bar_colors = [ORANGE if p > 16 else (Q8_0_C if p > 8 else Q4KM_C) for p in peak_gb]
    y = np.arange(n_quant)
    bars = ax_bar.barh(y, peak_gb, color=bar_colors, edgecolor="none", height=0.6)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(quant_types, fontsize=10)
    ax_bar.set_xlabel("Peak RAM (GB)", fontsize=10)
    ax_bar.set_title("Peak RAM per Quantization", fontsize=11, pad=10)
    ax_bar.axvline(8,  color=TEAL,   linestyle="--", linewidth=1.2, label="8 GB")
    ax_bar.axvline(16, color=GOLD,   linestyle="--", linewidth=1.2, label="16 GB")
    ax_bar.axvline(32, color=ORANGE, linestyle="--", linewidth=1.2, label="32 GB")
    ax_bar.legend(fontsize=9, framealpha=0)
    ax_bar.xaxis.grid(True); ax_bar.set_axisbelow(True)
    for bar, v in zip(bars, peak_gb):
        ax_bar.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f} GB", va="center", fontsize=9, color=TEXT_LT)

    # Highlight recommended
    bars[rec_idx].set_edgecolor(GOLD)
    bars[rec_idx].set_linewidth(2)

    fig.text(0.5, -0.06, "Made autonomously by NEO · https://heyneo.so  |  9B params, 1.20× overhead factor",
             ha="center", fontsize=8, color=TEXT_DIM)

    fig.tight_layout()
    _save(fig, "infographic_memory.png")


# ===========================================================================
# 3. Pipeline architecture
# ===========================================================================
def make_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "acervo-extractor-quant  ·  Pipeline Architecture",
        fontsize=15, fontweight="bold", color=TEXT_LT, pad=16,
    )

    def box(cx, cy, w, h, label, sublabel="", color=ACCENT, text_color=TEXT_LT):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                               boxstyle="round,pad=0.12",
                               facecolor=color, edgecolor="none", alpha=0.92, zorder=3)
        ax.add_patch(rect)
        if sublabel:
            ax.text(cx, cy + 0.18, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color, zorder=4)
            ax.text(cx, cy - 0.25, sublabel, ha="center", va="center",
                    fontsize=8, color=text_color, alpha=0.80, zorder=4)
        else:
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color, zorder=4)

    def arrow(x0, y0, x1, y1, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=TEXT_DIM,
                                   lw=1.6, mutation_scale=16), zorder=2)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx + 0.1, my, label, fontsize=8, color=TEXT_DIM, va="center")

    # ---- Row 1: inputs ----
    box(2.2, 5.8, 3.2, 0.9, "HuggingFace Hub", "SandyVeliz/acervo-extractor-qwen3.5-9b", "#2D2D5E")
    box(7.5, 5.8, 2.4, 0.9, "llama.cpp", "auto-cloned & built", "#2D2D5E")
    box(13.0, 5.8, 2.8, 0.9, "prompts.txt", "built-in / custom", "#2D2D5E")

    # ---- Row 2: quantize.py ----
    box(4.5, 4.0, 3.6, 1.0, "quantize.py", "download → convert → quantize", "#3A1F6E")
    arrow(2.2, 5.35, 3.5, 4.52, "weights")
    arrow(7.5, 5.35, 5.7, 4.52, "quantizer")

    # GGUF outputs
    box(8.5, 4.0, 2.2, 0.75, "Q4_K_M.gguf", "~4.7 GB", ACCENT)
    box(11.0, 4.0, 2.2, 0.75, "Q8_0.gguf", "~9.5 GB", Q8_0_C)
    arrow(6.3, 4.0, 7.4, 4.0, "")
    arrow(9.6, 4.0, 9.9, 4.0, "")

    # ---- Row 3: benchmark.py / memory_estimator.py ----
    box(3.2, 2.4, 3.4, 1.0, "benchmark.py", "PPL · tok/s · latency stats", "#1F3A6E")
    box(8.5, 2.4, 2.8, 1.0, "memory_estimator.py", "RAM / VRAM sizing", "#1F5A3A")
    box(13.0, 2.4, 2.8, 1.0, "compare.py", "multi-model ranking", "#3A3A1F")

    arrow(4.5, 3.5, 3.8, 2.9, "")
    arrow(8.5, 3.62, 8.5, 2.9, "GGUFs")
    arrow(13.0, 5.35, 13.0, 2.9, "prompts")

    # ---- Row 4: outputs ----
    box(1.5, 0.85, 2.2, 0.75, "report.md", "Markdown", "#1A2A1A")
    box(3.9, 0.85, 2.2, 0.75, "results.json", "machine-readable", "#1A2A1A")
    box(6.3, 0.85, 2.2, 0.75, "results.csv", "spreadsheet", "#1A2A1A")
    box(8.8, 0.85, 2.2, 0.75, "chart.png", "matplotlib", "#1A2A1A")
    box(11.5, 0.85, 2.0, 0.75, "comparison\nreport.md", "#1A2A1A")
    box(13.8, 0.85, 1.8, 0.75, "ranking\nJSON", "#1A2A1A")

    arrow(3.2, 1.9, 3.2, 1.23, "")
    arrow(8.5, 1.9, 8.5, 1.23, "")
    arrow(13.0, 1.9, 12.5, 1.23, "")

    # ---- scripts/demo.py (orchestrator) ----
    box(8.0, 5.8, 2.4, 0.9, "scripts/demo.py", "orchestrates all", GOLD, BG_DARK)
    ax.text(8.0, 6.5, "end-to-end entrypoint", ha="center", fontsize=8,
            color=GOLD, alpha=0.8)
    arrow(8.0, 5.35, 4.5, 4.5, "")
    arrow(8.0, 5.35, 8.5, 2.9, "")

    # Footer
    ax.text(8.0, -0.1, "Made autonomously by NEO · https://heyneo.so",
            ha="center", fontsize=8, color=TEXT_DIM)

    _save(fig, "infographic_pipeline.png")


# ===========================================================================
# 4. Quality vs Speed vs Size bubble chart
# ===========================================================================
def make_tradeoff() -> None:
    data = {
        "float16": {"ppl": 18.43, "speed": 42.7, "size_gb": 18.0,  "color": FLOAT16_C},
        "Q8_0":    {"ppl": 18.62, "speed": 45.3, "size_gb": 9.5,   "color": Q8_0_C},
        "Q6_K":    {"ppl": 18.80, "speed": 46.5, "size_gb": 7.5,   "color": "#4ECDC4"},
        "Q5_K_M":  {"ppl": 19.10, "speed": 47.0, "size_gb": 6.9,   "color": "#45B7D1"},
        "Q4_K_M":  {"ppl": 19.54, "speed": 47.8, "size_gb": 4.7,   "color": Q4KM_C},
        "Q3_K_M":  {"ppl": 20.30, "speed": 49.5, "size_gb": 3.7,   "color": "#96CEB4"},
        "Q2_K":    {"ppl": 22.10, "speed": 52.0, "size_gb": 2.6,   "color": GOLD},
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)
    ax.set_title(
        "Quality vs Speed vs Size  ·  acervo-extractor-qwen3.5-9b Quantization Tradeoff",
        fontsize=13, fontweight="bold", color=TEXT_LT, pad=14,
    )

    # Bubble = size (area ∝ file size)
    for name, d in data.items():
        bubble_size = (d["size_gb"] ** 1.3) * 60
        ax.scatter(d["speed"], d["ppl"], s=bubble_size, color=d["color"],
                   alpha=0.82, edgecolors="white", linewidths=0.8, zorder=3)
        offset_y = 0.15 if name not in ("Q5_K_M",) else -0.28
        ax.annotate(name, (d["speed"], d["ppl"]),
                    xytext=(5, offset_y * 30), textcoords="offset points",
                    fontsize=9, color=d["color"], fontweight="bold", zorder=4)

    # Annotate Q4_K_M with callout
    q4 = data["Q4_K_M"]
    ax.annotate(
        "◄ recommended\n  4.7 GB · 8 GB RAM · +12% speed",
        xy=(q4["speed"], q4["ppl"]),
        xytext=(q4["speed"] - 4.5, q4["ppl"] + 0.9),
        fontsize=9, color=GOLD, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc=BG_CARD, ec=GOLD, alpha=0.9),
    )

    ax.set_xlabel("Inference Speed  (tokens / sec)  →  faster is right", fontsize=11)
    ax.set_ylabel("Perplexity  (lower = better quality)  →  better is down", fontsize=11)
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.4); ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # Bubble size legend
    for sz_gb, lbl in [(2, "2 GB"), (8, "8 GB"), (18, "18 GB")]:
        ax.scatter([], [], s=(sz_gb ** 1.3) * 60, color=TEXT_DIM, alpha=0.6,
                   edgecolors="white", linewidths=0.6, label=f"File size: {lbl}")
    ax.legend(title="Bubble = file size", fontsize=9, framealpha=0.15,
              title_fontsize=9, loc="lower right")

    # Zone annotations
    ax.text(43.2, 18.2, "← best quality\n    (more RAM)",
            fontsize=8, color=TEXT_DIM, alpha=0.7, ha="left")
    ax.text(51.0, 22.5, "fastest / smallest →\n(quality loss)",
            fontsize=8, color=TEXT_DIM, alpha=0.7, ha="right")

    fig.text(0.5, -0.04,
             "Made autonomously by NEO · https://heyneo.so  |  Q6_K / Q5_K_M / Q3_K_M / Q2_K values are interpolated",
             ha="center", fontsize=8, color=TEXT_DIM)

    fig.tight_layout()
    _save(fig, "infographic_tradeoff.png")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    print("Generating infographics…")
    make_overview()
    make_memory()
    make_pipeline()
    make_tradeoff()
    print("Done.")

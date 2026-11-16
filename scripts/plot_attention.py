#!/usr/bin/env python
"""
Publication-quality attention weight figures for IEEE Transactions double-column papers.

Supports:
  --mode single   : plot one model
  --mode combined : side-by-side Post AR vs Full AR

Usage:
  # Single model
  python scripts/plot_combined_attention.py --mode single \
      --data_dir experiments/attention_analysis/results/POMO_TSP100_Full_AR \
      --label "Full AR" \
      --output_dir experiments/attention_analysis/results/POMO_TSP100_Full_AR

  # Combined
  python scripts/plot_combined_attention.py --mode combined \
      --post_ar_dir experiments/attention_analysis/results/PO_TSP100_Post_AR \
      --full_ar_dir experiments/attention_analysis/results/POMO_TSP100_Full_AR \
      --output_dir experiments/attention_analysis/results/TSP100_Comparison
"""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Suppress tight_layout warnings caused by inset_axes (output is correct)
warnings.filterwarnings("ignore", message=".*tight_layout.*")


# ═══════════════════════════════════════════════════════════
# Colors — IEEE-appropriate, no gradients
# ═══════════════════════════════════════════════════════════

POST_BASE   = "#9ecae1"   # light blue (non-max bars)
POST_HIGH   = "#2c6fbb"   # dark blue (max bar)
POST_EDGE   = "#1b4f8a"

FULL_BASE   = "#f6c28b"   # light amber (non-max bars)
FULL_HIGH   = "#d4752a"   # dark amber (max bar)
FULL_EDGE   = "#9e5418"

ERR_COLOR   = "#444444"


# ═══════════════════════════════════════════════════════════
# IEEE style
# ═══════════════════════════════════════════════════════════

def set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "errorbar.capsize": 2.5,
        "patch.linewidth": 0.4,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.35,
    })


# ═══════════════════════════════════════════════════════════
# CSV reading (unchanged)
# ═══════════════════════════════════════════════════════════

def load_summary(data_dir, csv_name="post_ar_summary.csv"):
    """Try loading summary CSV. Falls back to full_ar_grouped_summary.csv."""
    paths_to_try = [
        os.path.join(data_dir, csv_name),
        os.path.join(data_dir, "full_ar_grouped_summary.csv"),
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return _read_csv_manual(p)
    raise FileNotFoundError(f"No summary CSV found in {data_dir}")


def _read_csv_manual(path):
    """Read CSV manually, robust to mixed types."""
    sources = []
    means = []
    stds = []
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        col_map = {h.strip(): i for i, h in enumerate(header)}
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            sources.append(parts[col_map["source"]])
            means.append(float(parts[col_map["mean_weight"]]))
            stds.append(float(parts[col_map["std_over_instances"]]))
    return sources, np.array(means), np.array(stds)


# ═══════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════

def draw_pretty_bars(ax, x, means, stds, base_color, highlight_color,
                     edge_color, y_lim, show_values=True):
    """Draw bars: max bar highlighted, rest uniform base color.

    Args:
        ax: matplotlib axis
        x: bar positions
        means, stds: data arrays
        base_color: color for non-max bars
        highlight_color: color for max bar
        edge_color: bar edge color
        y_lim: (y_min, y_max) for adaptive label offset
        show_values: whether to show value labels on bars
    """
    max_idx = np.argmax(means)
    colors = [highlight_color if i == max_idx else base_color for i in range(len(means))]

    ax.bar(x, means, width=0.65,
           color=colors, edgecolor=edge_color,
           linewidth=0.3, alpha=0.92, zorder=3)

    if stds is not None:
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    ecolor=ERR_COLOR, elinewidth=0.6,
                    capsize=2.5, capthick=0.6, zorder=4)

    if show_values:
        y_lower, y_upper = y_lim
        value_offset = 0.012 * (y_upper - y_lower)
        for xi, m, s in zip(x, means, stds if stds is not None else np.zeros_like(means)):
            label_y = m + s + value_offset
            if label_y > y_upper:
                label_y = y_upper - value_offset * 0.5
            ax.text(xi, label_y, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=6,
                    color="#333333")


def style_axis(ax, x_labels, y_label, y_lim, title=None):
    """Apply consistent, clean IEEE styling."""
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlim(-0.55, len(x_labels) - 0.45)

    ax.set_ylabel(y_label, labelpad=4)
    ax.set_ylim(y_lim)

    # Grid: major only, no minor
    ax.grid(axis="y", which="major", alpha=0.22, linewidth=0.35, zorder=0)

    # Y-axis major locator
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.ticklabel_format(axis="y", style="plain")  # no scientific notation

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    ax.tick_params(axis="both", direction="out", pad=2)

    if title:
        ax.set_title(title, pad=5, fontweight="normal")


def compute_y_limits(means_list, stds_list):
    """Compute shared (0-based) y-axis limits from multiple datasets.

    Returns (0.0, y_max) where y_max = min(1.0, raw_max * 1.12 + 0.02).
    """
    raw_max = 0.0
    for means, stds in zip(means_list, stds_list):
        raw_max = max(raw_max, float(np.max(means + stds)))
    y_max = min(1.0, raw_max * 1.12 + 0.02)
    return 0.0, y_max


# ═══════════════════════════════════════════════════════════
# Inset for small weights when one bar dominates
# ═══════════════════════════════════════════════════════════

def add_inset_if_needed(ax, x, means, stds, base_color, highlight_color,
                        edge_color, y_lim_main, source_labels=None):
    """Add upper-right inset for non-dominant sources when max >> second."""
    if len(means) <= 2:
        return

    sorted_vals = np.sort(means)[::-1]
    max_mean = sorted_vals[0]
    second_mean = sorted_vals[1]

    if max_mean <= 2.5 * second_mean:
        return

    # Build non-max data
    max_idx = np.argmax(means)
    mask = np.ones(len(means), dtype=bool)
    mask[max_idx] = False

    non_x = np.arange(mask.sum())
    non_means = means[mask]
    non_stds = stds[mask]
    non_labels = [source_labels[i] for i in range(len(means)) if mask[i]] if source_labels else None

    # Inset axis
    ax_ins = inset_axes(ax, width="42%", height="38%",
                        loc="upper right",
                        borderpad=0.6)
    ax_ins.bar(non_x, non_means, width=0.6,
               color=base_color, edgecolor=edge_color,
               linewidth=0.25, alpha=0.92, zorder=3)

    if non_stds is not None:
        ax_ins.errorbar(non_x, non_means, yerr=non_stds, fmt="none",
                        ecolor=ERR_COLOR, elinewidth=0.5,
                        capsize=2, capthick=0.5, zorder=4)

    # Inset styling
    ins_y_max = max(non_means + non_stds) * 1.25
    ins_y_max = max(ins_y_max, 0.01)
    ax_ins.set_ylim(0.0, ins_y_max)
    ax_ins.set_xlim(-0.5, len(non_x) - 0.5)

    if non_labels:
        ax_ins.set_xticks(non_x)
        ax_ins.set_xticklabels(non_labels, fontsize=5.5)
    ax_ins.set_title("Non-dominant sources", fontsize=6, pad=3, fontweight="normal")

    ax_ins.grid(axis="y", which="major", alpha=0.18, linewidth=0.3, zorder=0)
    ax_ins.tick_params(axis="both", labelsize=5.5, pad=1.5)
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)
    ax_ins.spines["left"].set_linewidth(0.4)
    ax_ins.spines["bottom"].set_linewidth(0.4)


# ═══════════════════════════════════════════════════════════
# Title helper
# ═══════════════════════════════════════════════════════════

def pick_title(label, prefix=""):
    """Pick a clean title based on label content."""
    lower = label.lower()
    if "full" in lower:
        return f"{prefix}Full AR: Final FF Fusion".lstrip()
    elif "post" in lower:
        return f"{prefix}Post AR: Post-Encoder Fusion".lstrip()
    return f"{prefix}{label}".lstrip()


# ═══════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════

def save_figures(fig, output_dir, dpi=300, suffix=""):
    """Save PNG (300 dpi), PDF (vector), TIFF (600 dpi)."""
    base = os.path.join(output_dir, f"attention_weights_tsp100{suffix}")
    paths = []
    for ext, d in [(".png", dpi), (".pdf", None), (".tiff", 600)]:
        p = base + ext
        kwargs = {"bbox_inches": "tight"}
        if d:
            kwargs["dpi"] = d
        fig.savefig(p, **kwargs)
        paths.append(p)
    for p in paths:
        print(f"  Saved: {p}")
    return paths


# ═══════════════════════════════════════════════════════════
# Single-model plot
# ═══════════════════════════════════════════════════════════

def plot_single(data_dir, label, output_dir, dpi=300):
    """Single-model IEEE plot with inset for dominant-bar case."""
    os.makedirs(output_dir, exist_ok=True)
    set_ieee_style()

    sources, means, stds = load_summary(data_dir)
    n = len(sources)

    x_labels = [s.replace("Layer ", "L") for s in sources]

    # Model-specific colors
    lower = label.lower()
    if "post" in lower:
        base_c, high_c, edge_c = POST_BASE, POST_HIGH, POST_EDGE
    else:
        base_c, high_c, edge_c = FULL_BASE, FULL_HIGH, FULL_EDGE

    # Y-axis: always from 0
    y_lim = (0.0, min(1.0, float(np.max(means + stds)) * 1.12 + 0.02))

    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    draw_pretty_bars(ax, np.arange(n), means, stds,
                     base_c, high_c, edge_c, y_lim, show_values=True)

    title = pick_title(label)
    style_axis(ax, x_labels, "Mean attention weight", y_lim, title=title)

    # Inset for dominant-bar case
    add_inset_if_needed(ax, np.arange(n), means, stds,
                        base_c, high_c, edge_c, y_lim,
                        source_labels=x_labels)

    fig.tight_layout(pad=0.4)
    save_figures(fig, output_dir, dpi)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Combined (side-by-side) plot
# ═══════════════════════════════════════════════════════════

def plot_combined(post_dir, full_dir, output_dir, dpi=300):
    """Side-by-side Post AR + Full AR for IEEE double-column."""
    os.makedirs(output_dir, exist_ok=True)
    set_ieee_style()

    post_src, post_m, post_s = load_summary(post_dir, "post_ar_summary.csv")
    full_src, full_m, full_s = load_summary(full_dir, "full_ar_grouped_summary.csv")

    x = np.arange(7)
    x_labels = ["Input", "L1", "L2", "L3", "L4", "L5", "L6"]

    # Shared y-axis: from 0
    y_lim = compute_y_limits([post_m, full_m], [post_s, full_s])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.8, 2.65),
        gridspec_kw={"wspace": 0.30},
    )

    # ── (a) Post AR ──
    draw_pretty_bars(ax1, x, post_m, post_s,
                     POST_BASE, POST_HIGH, POST_EDGE, y_lim, show_values=True)
    style_axis(ax1, x_labels, "Mean attention weight", y_lim,
               title="(a) Post AR")

    # ── (b) Full AR ──
    draw_pretty_bars(ax2, x, full_m, full_s,
                     FULL_BASE, FULL_HIGH, FULL_EDGE, y_lim, show_values=True)
    style_axis(ax2, x_labels, "Mean attention weight", y_lim,
               title="(b) Full AR")
    # Right panel: keep ticks, drop y-label
    ax2.set_ylabel("")
    ax2.tick_params(axis="y", labelleft=True)

    fig.tight_layout(pad=0.4)
    save_figures(fig, output_dir, dpi)
    plt.close(fig)


def plot_full_ar_13source(data_dir, output_dir, dpi=300):
    """Full AR 13-source breakdown figure for appendix."""
    os.makedirs(output_dir, exist_ok=True)
    set_ieee_style()

    csv_path = os.path.join(data_dir, "full_ar_13source_summary.csv")
    if not os.path.exists(csv_path):
        raw_path = os.path.join(data_dir, "full_ar_raw_final_ff_weights.npy")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Neither {csv_path} nor {raw_path} found. Run analysis first.")
        w = np.load(raw_path)
        names_13 = ['H0','M1','F1','M2','F2','M3','F3','M4','F4','M5','F5','M6','F6']
        gm = w.mean(axis=(0,1))
        pm = w.mean(axis=1)
        si = pm.std(axis=0)
        sources = names_13
        means = gm
        stds = si
    else:
        sources, means, stds = load_summary(data_dir, "full_ar_13source_summary.csv")

    means = np.array(means)
    stds = np.array(stds)
    n = len(sources)

    # Wider figure for appendix
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    x = np.arange(n)
    max_idx = np.argmax(means)

    colors = [FULL_BASE] * n
    colors[max_idx] = FULL_HIGH

    y_lim = (0.0, min(1.0, float(np.max(means + stds)) * 1.12 + 0.02))

    draw_pretty_bars(ax, x, means, stds, FULL_BASE, FULL_HIGH, FULL_EDGE,
                     y_lim, show_values=True)

    # X-axis labels with rotation for readability
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=6.5)
    ax.set_xlim(-0.6, n - 0.4)

    style_axis(ax, sources, "Mean attention weight", y_lim,
               title="Full AR: Final FF Fusion (13 sources)")
    # Override x labels set by style_axis
    ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=6.5)
    ax.set_xlim(-0.6, n - 0.4)

    # Inset for non-dominant sources
    add_inset_if_needed(ax, x, means, stds, FULL_BASE, FULL_HIGH, FULL_EDGE,
                        y_lim, source_labels=sources)

    fig.tight_layout(pad=0.4)
    save_figures(fig, output_dir, dpi, suffix="_13source")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# CLI (unchanged interface)
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["single", "combined", "full_ar_13source"])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="For --mode single: path to analysis results")
    parser.add_argument("--label", type=str, default="Model",
                        help="For --mode single: plot title label")
    parser.add_argument("--post_ar_dir", type=str, default=None)
    parser.add_argument("--full_ar_dir", type=str, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.mode == "single":
        if not args.data_dir:
            print("ERROR: --data_dir required for --mode single")
            sys.exit(1)
        plot_single(args.data_dir, args.label, args.output_dir, args.dpi)

    elif args.mode == "full_ar_13source":
        if not args.data_dir:
            print("ERROR: --data_dir required for --mode full_ar_13source")
            sys.exit(1)
        plot_full_ar_13source(args.data_dir, args.output_dir, args.dpi)

    elif args.mode == "combined":
        if not args.post_ar_dir or not args.full_ar_dir:
            print("ERROR: --post_ar_dir and --full_ar_dir required for --mode combined")
            sys.exit(1)
        plot_combined(args.post_ar_dir, args.full_ar_dir,
                      args.output_dir, args.dpi)


if __name__ == "__main__":
    main()

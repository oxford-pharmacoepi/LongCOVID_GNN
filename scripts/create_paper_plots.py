#!/usr/bin/env python3
"""
Publication-quality figures for the SEAL drug repurposing manuscript.

Generates 6 figures

    fig1  — Model tournament (SEAL vs Global GNNs vs Heuristics)
    fig2  — Edge-type ablation impact on Osteoporosis LOO
    fig3  — Node-feature ablation (DRNL sufficiency)
    fig4  — Disease complexity vs SEAL performance
    fig5  — Long COVID top-20 consensus drug candidates
    fig6  — Gene-configuration comparison (NARROW vs BROAD vs FULL)

Usage:
    uv run python scripts/create_paper_plots.py            # all figures
    uv run python scripts/create_paper_plots.py --figures 1 5   # specific
    uv run python scripts/create_paper_plots.py --format svg    # SVG output

Output:
    results/figures/fig{N}_{name}.{png,pdf,svg}

Data sources:
    results/ALL_RESULTS_SUMMARY.md — sections 1, 2, 7, 8, 11
    results/long_covid/FINAL_PREDICTIONS.md — 5-seed consensus

Dependencies:
    matplotlib >= 3.7, numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.patches as mpatches               # noqa: E402
import matplotlib.patheffects as pe                  # noqa: E402
import matplotlib.colors as mcolors                  # noqa: E402
import numpy as np                                   # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

OUT_DIR = Path("results/figures")

# ------------------------------------------------------------------
# Colour palette — restrained, journal-appropriate
# Inspired by Nature Reviews / Lancet colour schemes
# ------------------------------------------------------------------
PAL = {
    # Core model colours
    "seal":         "#1B65A6",   # strong, confident blue
    "seal_light":   "#8FBCDB",   # muted blue for secondary
    "gat":          "#D64550",   # muted crimson
    "sage":         "#6A4C93",   # plum / violet
    "heuristic":    "#2A9D8F",   # teal
    "neutral":      "#AAAAAA",   # warm grey
    "accent":       "#E9A820",   # warm amber
    "alert":        "#C94040",   # darker red for warnings

    # Drug class colours (fig 5) — distinguishable, accessible
    "antiviral":       "#1B65A6",
    "corticosteroid":  "#D64550",
    "anti_inflam":     "#E9A820",
    "immunotherapy":   "#6A4C93",
    "antimalarial":    "#2A9D8F",
    "antimicrobial":   "#3DA35D",
    "antibiotic":      "#8D775F",
    "hormone":         "#C45BAA",
    "other":           "#BBBBBB",

    # Gradient stops for ablation
    "good":         "#1B65A6",
    "ok":           "#E9A820",
    "bad":          "#D64550",
}

# Figure dimensions (inches) — standard single and double column
SINGLE_COL = 3.5          # ~89 mm
DOUBLE_COL = 7.2          # ~183 mm
FULL_PAGE  = 9.5           # wider for presentation


def apply_style() -> None:
    """Apply a Nature/Lancet-inspired matplotlib RC style."""
    plt.rcParams.update({
        # Typography — Helvetica Neue is available on macOS
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial",
                               "DejaVu Sans"],
        "font.size":          8,
        "axes.titlesize":     9,
        "axes.titleweight":   "bold",
        "axes.titlepad":      10,
        "axes.labelsize":     8,
        "axes.labelweight":   "medium",
        "axes.labelpad":      6,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "legend.framealpha":  1.0,
        "legend.edgecolor":   "#CCCCCC",
        "legend.fancybox":    False,
        "legend.borderpad":   0.5,
        # Figure
        "figure.facecolor":   "#FFFFFF",
        "axes.facecolor":     "#FFFFFF",
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "savefig.facecolor":  "#FFFFFF",
        # Grid & spines
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,          # off by default — added per-figure
        "grid.alpha":         0.3,
        "grid.color":         "#CCCCCC",
        "grid.linewidth":     0.4,
        "grid.linestyle":     "-",
        # Spines
        "axes.edgecolor":     "#444444",
        "axes.linewidth":     0.6,
        # Ticks
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Colours
        "text.color":         "#222222",
        "axes.labelcolor":    "#222222",
        "xtick.color":        "#444444",
        "ytick.color":        "#444444",
    })


def save(fig: plt.Figure, name: str, fmt: str = "png") -> None:
    """Save figure to OUT_DIR in requested format + PDF."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (fmt, "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
    print(f"  ✓ {name}")


def _panel_label(ax, label: str, x: float = -0.08, y: float = 1.08,
                 fontsize: int = 11):
    """Add a bold panel label (a, b, c, …) in the Nature style."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="right",
            color="#111111")


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Model Tournament
# ═══════════════════════════════════════════════════════════════════════

def fig1_tournament(fmt: str = "png") -> None:
    """
    Grouped bar chart comparing Hits@100 across three diseases for
    SEAL, GAT, SAGE-3L, and Adamic-Adar heuristic.

    Data: ALL_RESULTS_SUMMARY.md §1, FULL_TOURNAMENT_REPORT.md
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.8))

    diseases = [
        "Osteoporosis\n(27 drugs)",
        "Multiple Sclerosis\n(59 drugs)",
        "Depression\n(108 drugs)",
    ]
    seal   = [92.6, 44.1, 15.1]
    gat    = [25.9, 47.5, 34.3]
    sage3  = [66.7, 39.0, 21.3]
    aa     = [18.5,  8.5,  0.0]

    models = [
        ("SEAL (SAGE+JK)", seal,  PAL["seal"]),
        ("GAT (best global)", gat,   PAL["gat"]),
        ("SAGE-3L", sage3, PAL["sage"]),
        ("Adamic–Adar", aa, PAL["heuristic"]),
    ]

    x = np.arange(len(diseases))
    n = len(models)
    w = 0.18
    gap = 0.03

    for idx, (label, values, colour) in enumerate(models):
        offset = (idx - (n - 1) / 2) * (w + gap)
        bars = ax.bar(
            x + offset, values, w,
            label=label, color=colour,
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        # Value labels — compact
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f"{h:.0f}", ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color="#333333",
                )

    ax.set_ylabel("Hits@100 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=8)
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)

    # Legend — inside, upper right
    leg = ax.legend(
        loc="upper right", frameon=True,
        edgecolor="#CCCCCC", framealpha=1.0,
        handlelength=1.2, handletextpad=0.4,
        borderpad=0.4, labelspacing=0.3,
    )
    leg.get_frame().set_linewidth(0.5)

    # Complexity annotation arrow below
    ax.annotate(
        "", xy=(2.38, -7), xytext=(-0.38, -7),
        xycoords="data", textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color="#888888",
                        lw=0.8, mutation_scale=10),
        annotation_clip=False,
    )
    ax.text(
        1.0, -11.5,
        "<-- focused                    disease complexity"
        "                    broad -->",
        ha="center", fontsize=6.5, color="#888888",
        transform=ax.get_xaxis_transform(),
    )

    fig.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.16)
    save(fig, "fig1_tournament", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — Edge-Type Ablation  (lollipop chart)
# ═══════════════════════════════════════════════════════════════════════

def fig2_edge_ablation(fmt: str = "png") -> None:
    """
    Lollipop chart: each edge-type ablation → change in median rank.
    Horizontal orientation with coloured stems.

    Data: ALL_RESULTS_SUMMARY.md §7.A
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))

    configs = [
        "Full graph (baseline)",
        "No PPI edges (−75%)",
        "No disease similarity (−0.5%)",
        "No drug–gene / MoA (−0.9%)",
        "No disease–gene (−9%)",
    ]
    med_ranks = [29, 31, 40, 334, 553]

    # Colour gradient: good → bad
    norm = plt.Normalize(0, 600)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "perf", [PAL["good"], PAL["ok"], PAL["bad"]]
    )
    colours = [cmap(norm(v)) for v in med_ranks]

    y_pos = np.arange(len(configs))

    # Draw stems (horizontal lines from 0 to value)
    for i, (v, c) in enumerate(zip(med_ranks, colours)):
        ax.plot([0, v], [i, i], color=c, linewidth=1.8, zorder=2)
        ax.scatter(v, i, color=c, s=60, zorder=4, edgecolors="white",
                   linewidth=0.8)

    # Baseline reference
    ax.axvline(x=29, color=PAL["seal"], linestyle=":", alpha=0.4,
               linewidth=0.8, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs, fontsize=7.5)
    ax.set_xlabel("Median rank (lower = better)")
    ax.invert_yaxis()
    ax.set_xlim(-10, 680)

    # Value labels
    for i, v in enumerate(med_ranks):
        label = str(v)
        bold = v > 100
        if bold:
            label += "  << critical"
        ax.text(
            v + 14, i, label, va="center", fontsize=7,
            fontweight="bold" if bold else "medium",
            color=PAL["alert"] if bold else "#333333",
        )

    # Annotation
    ax.annotate(
        "Drug–gene & disease–gene edges\nare the core biological signal",
        xy=(553, 4), xytext=(420, 2.2),
        fontsize=6.5, color="#666666", ha="center",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.6),
    )

    plt.tight_layout()
    save(fig, "fig2_edge_ablation", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — Node Feature Ablation
# ═══════════════════════════════════════════════════════════════════════

def fig3_node_ablation(fmt: str = "png") -> None:
    """
    Paired panels: DRNL-only SEAL matches or beats full features.

    Data: ALL_RESULTS_SUMMARY.md §7.B
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    labels = [
        "DRNL +\nfeatures +\ntypes (49d)",
        "DRNL +\ntypes (3d)",
        "DRNL only\n(0d)",
    ]
    h20      = [40.7, 40.7, 44.4]
    medrank  = [29,   24,   22]

    # Progressive colouring: minimal → more vivid
    cols = [PAL["neutral"], PAL["seal_light"], PAL["seal"]]

    # ─── Panel a: Hits@20 ─────────────────────────────────────────
    ax = axes[0]
    bars_a = ax.bar(
        range(3), h20, color=cols,
        edgecolor="white", linewidth=0.6, width=0.52, zorder=3,
    )
    for i, bar in enumerate(bars_a):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{h20[i]:.1f}%",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color="#333333",
        )
    ax.set_ylabel("Hits@20 (%)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, 54)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _panel_label(ax, "a")

    # Highlight best
    ax.annotate(
        "Best", xy=(2, 44.4), xytext=(2, 50),
        ha="center", fontsize=7, fontweight="bold", color=PAL["seal"],
        arrowprops=dict(arrowstyle="->", color=PAL["seal"], lw=0.8),
    )

    # ─── Panel b: Median Rank ────────────────────────────────────
    ax = axes[1]
    bars_b = ax.bar(
        range(3), medrank, color=cols,
        edgecolor="white", linewidth=0.6, width=0.52, zorder=3,
    )
    for i, bar in enumerate(bars_b):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(medrank[i]),
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color="#333333",
        )
    ax.set_ylabel("Median rank (lower = better)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, 37)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _panel_label(ax, "b")

    ax.annotate(
        "Best", xy=(2, 22), xytext=(2, 27.5),
        ha="center", fontsize=7, fontweight="bold", color=PAL["seal"],
        arrowprops=dict(arrowstyle="->", color=PAL["seal"], lw=0.8),
    )

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_node_ablation", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — Disease Complexity vs Performance (bubble chart)
# ═══════════════════════════════════════════════════════════════════════

def fig4_disease_complexity(fmt: str = "png") -> None:
    """
    Bubble chart: n_true_drugs vs median_rank, bubble size ∝ Hits@100%.

    Data: ALL_RESULTS_SUMMARY.md §1, §8
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.0))

    # (n_true_drugs, median_rank, label, hits@100%, offset_xy)
    data = [
        (27,   24,   "Osteoporosis",                 92.6,  (8, -18)),
        (59,   130,  "Multiple Sclerosis",            44.1,  (8, 12)),
        (59,   206,  "Ank. Spondylitis\n(propagated)", 33.9,  (8, 12)),
        (44,   809,  "Neuropathic Pain",              None,  (8, -14)),
        (56,   354,  "Type 2 Diabetes",               None,  (8, 10)),
        (108,  399,  "Depression",                     15.1,  (8, -18)),
    ]

    for n, mr, name, h100, (dx, dy) in data:
        size = h100 * 3.5 if h100 else 80
        alpha = 0.80 if h100 else 0.40
        if h100 and h100 > 60:
            colour = PAL["seal"]
        elif h100 and h100 > 30:
            colour = PAL["accent"]
        elif h100:
            colour = PAL["alert"]
        else:
            colour = PAL["neutral"]

        ax.scatter(
            n, mr, s=size, c=colour, alpha=alpha, zorder=5,
            edgecolors="white", linewidth=1.5,
        )

        ax.annotate(
            name, (n, mr), xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5, ha="left", va="center",
            color="#333333",
            arrowprops=dict(
                arrowstyle="-", color="#BBBBBB",
                lw=0.5, connectionstyle="arc3,rad=0.1",
            ),
        )

    ax.set_xlabel("Number of true drugs for disease")
    ax.set_ylabel("SEAL median rank (lower = better)")
    ax.set_xlim(15, 120)
    ax.set_ylim(-40, 920)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)

    # Size legend
    for sz, lab in [
        (92 * 3.5, "H@100 ≈ 93%"),
        (44 * 3.5, "H@100 ≈ 44%"),
        (80,       "Not tested"),
    ]:
        ax.scatter([], [], s=sz, c=PAL["neutral"], alpha=0.5, label=lab,
                   edgecolors="white", linewidth=1)
    leg = ax.legend(
        loc="upper right", title="Bubble size = H@100",
        title_fontsize=7, frameon=True,
        edgecolor="#CCCCCC", handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    save(fig, "fig4_disease_complexity", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 5 — Long COVID Top-20 Predictions
# ═══════════════════════════════════════════════════════════════════════

def fig5_long_covid(fmt: str = "png") -> None:
    """
    Horizontal bar chart of the top 20 consensus drug predictions,
    colour-coded by therapeutic class.  Stars mark RCT drugs.
    Seed consistency shown inside bars.

    Data: ALL_RESULTS_SUMMARY.md §11
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5.5))

    # (name, score, class_key, is_rct, seeds_str)
    drugs = [
        ("Ribavirin",          0.622, "antiviral",      False, "5/5"),
        ("Nivolumab",          0.573, "immunotherapy",   False, "5/5"),
        ("Amphotericin B",     0.541, "antimicrobial",   False, "4/5"),
        ("Valacyclovir",       0.538, "antiviral",       False, "4/5"),
        ("Tafenoquine",        0.538, "antimalarial",    False, "3/5"),
        ("Spinosad",           0.529, "other",           False, "3/5"),
        ("CHEMBL499808",       0.528, "other",           False, "3/5"),
        ("Triclabendazole",    0.514, "other",           False, "3/5"),
        ("Mefloquine",         0.512, "antimalarial",    False, "3/5"),
        ("Acyclovir",          0.512, "antiviral",       False, "5/5"),
        ("Prednisone",         0.503, "corticosteroid",  False, "5/5"),
        ("Streptomycin",       0.503, "antibiotic",      False, "3/5"),
        ("Somatropin",         0.503, "hormone",         False, "5/5"),
        ("Amoxicillin",        0.493, "antibiotic",      False, "3/5"),
        ("Methylprednisolone", 0.491, "corticosteroid",  False, "5/5"),
        ("Tetracycline",       0.490, "antibiotic",      False, "3/5"),
        ("Rosuvastatin",       0.487, "anti_inflam",     False, "5/5"),
        ("Dexamethasone",      0.487, "corticosteroid",  True,  "4/5"),
        ("Sarilumab",          0.485, "anti_inflam",     False, "4/5"),
        ("Doxycycline",        0.485, "anti_inflam",     False, "5/5"),
    ]

    # Reverse for horizontal bar chart (top drug at top)
    drugs_rev = drugs[::-1]
    names = []
    for d in drugs_rev:
        n = d[0]
        if d[3]:
            n += "  *RCT*"
        names.append(n)
    scores  = [d[1] for d in drugs_rev]
    colours = [PAL.get(d[2], PAL["other"]) for d in drugs_rev]
    seeds   = [d[4] for d in drugs_rev]

    bars = ax.barh(
        range(20), scores, color=colours,
        edgecolor="white", linewidth=0.5, height=0.68, zorder=3,
        alpha=0.88,
    )

    ax.set_yticks(range(20))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Mean SEAL score (5-seed consensus, ≥3/5 seeds)")
    ax.set_xlim(0, 0.76)

    # Seed count & score labels
    for i, (bar, sd, sc) in enumerate(zip(bars, seeds, scores)):
        w = bar.get_width()
        # Seed count inside bar
        ax.text(
            w - 0.008, i, sd,
            va="center", ha="right", fontsize=5.5,
            color="white", fontweight="bold", alpha=0.85,
        )
        # Score outside bar
        ax.text(
            w + 0.006, i, f"{sc:.3f}",
            va="center", ha="left", fontsize=6,
            color="#666666",
        )

    # Legend — drug class
    class_labels = {
        "antiviral": "Antiviral", "corticosteroid": "Corticosteroid",
        "anti_inflam": "Anti-inflammatory", "immunotherapy": "Immunotherapy",
        "antimalarial": "Antimalarial", "antimicrobial": "Antimicrobial",
        "antibiotic": "Antibiotic", "hormone": "Hormone", "other": "Other",
    }
    seen = []
    handles = []
    for d in drugs:
        cls = d[2]
        if cls not in seen:
            seen.append(cls)
            handles.append(
                mpatches.Patch(
                    facecolor=PAL.get(cls, PAL["other"]),
                    label=class_labels.get(cls, cls),
                    alpha=0.88, edgecolor="white", linewidth=0.3,
                )
            )
    # RCT marker
    handles.append(
        plt.Line2D([0], [0], marker="*", color="none",
                   markerfacecolor=PAL["accent"], markersize=7,
                   label="RCT drug")
    )

    leg = ax.legend(
        handles=handles, loc="lower right", fontsize=6.5,
        frameon=True, title="Therapeutic class", title_fontsize=7,
        edgecolor="#CCCCCC", handlelength=1.0, handletextpad=0.4,
        borderpad=0.4, labelspacing=0.3, ncol=2,
    )
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    save(fig, "fig5_long_covid_predictions", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 6 — Gene Configuration Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig6_gene_config(fmt: str = "png") -> None:
    """
    Paired bar chart: NARROW / BROAD / FULL gene configs on
    (a) median RCT drug rank and (b) RCT drugs recovered in top 50.

    Data: ALL_RESULTS_SUMMARY.md §2
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    labels  = ["NARROW\n(8 genes)", "BROAD\n(39 genes)", "FULL\n(52 genes)"]
    med_rct = [236, 379, 474]
    top50   = [3.3, 2.0, 2.3]

    cols = [PAL["seal"], PAL["seal_light"], PAL["neutral"]]

    # ─── Panel a: Median RCT Rank ───────────────────────────────
    ax = axes[0]
    bars = ax.bar(
        range(3), med_rct, color=cols,
        edgecolor="white", linewidth=0.6, width=0.50, zorder=3,
    )
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            str(med_rct[i]),
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color="#333333",
        )
    ax.set_ylabel("Median RCT drug rank\n(lower = better)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 560)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _panel_label(ax, "a")

    ax.annotate(
        "Best", xy=(0, 236), xytext=(0.55, 420),
        ha="center", fontsize=7, fontweight="bold", color=PAL["seal"],
        arrowprops=dict(arrowstyle="->", color=PAL["seal"], lw=0.8),
    )

    # ─── Panel b: RCT in Top 50 ────────────────────────────────
    ax = axes[1]
    bars = ax.bar(
        range(3), top50, color=cols,
        edgecolor="white", linewidth=0.6, width=0.50, zorder=3,
    )
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{top50[i]:.1f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color="#333333",
        )
    ax.set_ylabel("Mean RCT drugs in top 50")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 4.5)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _panel_label(ax, "b")

    ax.annotate(
        "Best", xy=(0, 3.3), xytext=(0.55, 4.0),
        ha="center", fontsize=7, fontweight="bold", color=PAL["seal"],
        arrowprops=dict(arrowstyle="->", color=PAL["seal"], lw=0.8),
    )

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig6_gene_configs", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

ALL_FIGURES = {
    1: ("fig1_tournament",         fig1_tournament),
    2: ("fig2_edge_ablation",      fig2_edge_ablation),
    3: ("fig3_node_ablation",      fig3_node_ablation),
    4: ("fig4_disease_complexity",  fig4_disease_complexity),
    5: ("fig5_long_covid",         fig5_long_covid),
    6: ("fig6_gene_config",        fig6_gene_config),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--figures", nargs="+", type=int,
        default=list(ALL_FIGURES.keys()),
        help="Figure numbers to generate (default: all). E.g. --figures 1 5",
    )
    parser.add_argument(
        "--format", type=str, default="png",
        choices=["png", "svg", "pdf"],
        help="Primary output format (PDF always generated too)",
    )
    args = parser.parse_args()

    apply_style()

    print(f"Generating {len(args.figures)} figure(s) → {OUT_DIR}/")

    for num in args.figures:
        if num not in ALL_FIGURES:
            print(f"  ✗ Unknown figure number: {num}")
            continue
        name, func = ALL_FIGURES[num]
        func(fmt=args.format)

    print("Done.")


if __name__ == "__main__":
    main()

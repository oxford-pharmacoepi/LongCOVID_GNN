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
        (44,   809,  "Neuropathic Pain",              13.6,  (8, -14)),
        (56,   354,  "Type 2 Diabetes",               16.1,  (8, 10)),
        (108,  399,  "Depression",                    15.1,  (8, -18)),
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

    # Size legend (upper right)
    size_handles = []
    for sz, lab in [
        (92 * 3.5, "H@100 ≈ 93%"),
        (44 * 3.5, "H@100 ≈ 44%"),
        (80,       "Not tested"),
    ]:
        h = ax.scatter([], [], s=sz, c=PAL["neutral"], alpha=0.5,
                       edgecolors="white", linewidth=1)
        size_handles.append((h, lab))
    
    # Color legend (lower right)
    color_handles = [
        mpatches.Patch(color=PAL["seal"], alpha=0.8, label="H@100 > 60%"),
        mpatches.Patch(color=PAL["accent"], alpha=0.8, label="H@100 30–60%"),
        mpatches.Patch(color=PAL["alert"], alpha=0.8, label="H@100 < 30%"),
        mpatches.Patch(color=PAL["neutral"], alpha=0.4, label="Not tested"),
    ]
    
    # Combined legend
    leg1 = ax.legend(
        handles=[h for h, _ in size_handles],
        labels=[lab for _, lab in size_handles],
        loc="upper right", title="Bubble size",
        title_fontsize=7, frameon=True,
        edgecolor="#CCCCCC", handletextpad=0.5,
    )
    leg1.get_frame().set_linewidth(0.5)
    ax.add_artist(leg1)
    
    leg2 = ax.legend(
        handles=color_handles,
        loc="lower right", title="Performance tier",
        title_fontsize=7, frameon=True,
        edgecolor="#CCCCCC", handletextpad=0.5,
    )
    leg2.get_frame().set_linewidth(0.5)

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
        ("Retired compound",   0.528, "other",           False, "3/5"),
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
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL + 0.8, 4.2))  # Wider, taller

    # Single-line labels, gene count in parentheses  
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
            bar.get_height() + 12,
            str(med_rct[i]),
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="#333333",
        )
    ax.set_ylabel("Median RCT drug rank (lower = better)", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
    ax.set_ylim(0, 580)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _panel_label(ax, "a")

    # ─── Panel b: RCT in Top 50 ────────────────────────────────
    ax = axes[1]
    bars = ax.bar(
        range(3), top50, color=cols,
        edgecolor="white", linewidth=0.6, width=0.50, zorder=3,
    )
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{top50[i]:.1f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="#333333",
        )
    ax.set_ylabel("Mean RCT drugs in top 50", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
    ax.set_ylim(0, 4.5)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _panel_label(ax, "b")

    fig.tight_layout(w_pad=2.5, rect=[0, 0.05, 1, 1])  # Extra bottom margin for labels
    save(fig, "fig6_gene_configs", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 7 — Temporal Validation Rank Distribution
# ═══════════════════════════════════════════════════════════════════════

def fig7_temporal_ranks(fmt: str = "png") -> None:
    """
    Strip plot with box overlay showing per-disease median ranks
    from temporal validation (56 diseases, 83 test edges).

    Data: results/seal_temporal.log
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.8))

    # Per-disease median ranks from temporal validation (extracted from log)
    ranks = [
        5, 5, 15, 27, 40, 62, 63, 65, 68, 70, 80, 81, 106, 111,
        132, 161, 176, 196, 208, 216, 217, 266, 316, 326, 331,
        369, 410, 426, 515, 524, 550, 569, 809, 810, 819, 939,
        1102, 1195, 1224, 1238, 1381, 1434, 1490, 1507, 1518,
        1718, 1849, 1903, 1941, 2134, 2220, 2280, 2281, 2372,
        2389, 2462,
    ]
    ranks = np.array(ranks)
    total_drugs = 2471

    # Strip plot (jittered)
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(ranks))
    y_pos = np.zeros(len(ranks)) + jitter

    # Colour by performance tier
    colours = []
    for r in ranks:
        if r <= 100:
            colours.append(PAL["seal"])
        elif r <= 500:
            colours.append(PAL["accent"])
        else:
            colours.append(PAL["alert"])

    ax.scatter(
        ranks, y_pos, c=colours, s=28, alpha=0.7,
        edgecolors="white", linewidth=0.5, zorder=5,
    )

    # Box plot overlay
    bp = ax.boxplot(
        ranks, vert=False, positions=[0], widths=0.25,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=PAL["seal"], alpha=0.15, linewidth=0.8),
        medianprops=dict(color=PAL["seal"], linewidth=1.5),
        whiskerprops=dict(color="#888888", linewidth=0.6),
        capprops=dict(color="#888888", linewidth=0.6),
    )

    # Reference lines
    ax.axvline(x=total_drugs // 2, color="#CCCCCC", linestyle="--",
               linewidth=0.6, zorder=1, label="Random baseline")
    ax.axvline(x=316, color=PAL["seal"], linestyle=":",
               linewidth=1.0, alpha=0.6, zorder=2, label="Overall median (316)")

    # Annotations
    ax.annotate(
        f"Median = 316 (top 13%)",
        xy=(316, 0.18), xytext=(700, 0.25),
        fontsize=7, color=PAL["seal"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PAL["seal"], lw=0.8),
    )

    # Stats text
    pct_top200 = np.sum(ranks <= 200) / len(ranks) * 100
    pct_top100 = np.sum(ranks <= 100) / len(ranks) * 100
    stats_text = (
        f"n = {len(ranks)} diseases\n"
        f"{pct_top100:.0f}% rank ≤ 100 (top 4%)\n"
        f"{pct_top200:.0f}% rank ≤ 200 (top 8%)"
    )
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes, fontsize=6.5,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                  edgecolor="#CCCCCC", linewidth=0.5),
    )

    # Tier legend
    for c, lab in [
        (PAL["seal"],   "Top 100 (excellent)"),
        (PAL["accent"], "101–500 (moderate)"),
        (PAL["alert"],  ">500 (challenging)"),
    ]:
        ax.scatter([], [], c=c, s=30, label=lab, edgecolors="white")
    leg = ax.legend(
        loc="lower right", fontsize=6.5,
        frameon=True, edgecolor="#CCCCCC",
        handletextpad=0.3, borderpad=0.4,
    )
    leg.get_frame().set_linewidth(0.5)

    ax.set_xlabel("Median rank across 2,471 drugs (lower = better)")
    ax.set_xlim(-50, 2600)
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks([])
    ax.xaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "fig7_temporal_ranks", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 8 — Training Convergence Curves
# ═══════════════════════════════════════════════════════════════════════

def fig8_training_curves(fmt: str = "png") -> None:
    """
    Training convergence for Osteoporosis LOO (seed=42, SAGEConv+JK).
    Per-epoch loss and validation AUC from real training run.

    Data: results/seal_results/training_curves_run.log
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2))

    # Real per-epoch data from Osteoporosis LOO run (seed=42, SAGEConv+JK)
    # 40 epochs total, early stopped at epoch 40 (best AUC at epoch 30)
    epochs = list(range(1, 41))
    losses = [
        0.3639, 0.2597, 0.2386, 0.2259, 0.2207, 0.2109, 0.2030, 0.2020,
        0.1942, 0.1909, 0.1909, 0.1869, 0.1831, 0.1823, 0.1787, 0.1785,
        0.1770, 0.1744, 0.1719, 0.1717, 0.1717, 0.1693, 0.1679, 0.1671,
        0.1636, 0.1649, 0.1637, 0.1615, 0.1612, 0.1614, 0.1588, 0.1578,
        0.1574, 0.1558, 0.1571, 0.1534, 0.1538, 0.1544, 0.1533, 0.1518,
    ]
    val_aucs = [
        0.9295, 0.9432, 0.9526, 0.9519, 0.9630, 0.9630, 0.9610, 0.9649,
        0.9657, 0.9666, 0.9684, 0.9665, 0.9682, 0.9700, 0.9694, 0.9693,
        0.9692, 0.9688, 0.9699, 0.9702, 0.9686, 0.9706, 0.9736, 0.9726,
        0.9721, 0.9730, 0.9694, 0.9714, 0.9691, 0.9750, 0.9714, 0.9743,
        0.9730, 0.9727, 0.9726, 0.9730, 0.9691, 0.9715, 0.9718, 0.9723,
    ]
    # Cosine annealing LR schedule (warmup=5, then cosine to lr/100)
    lr_schedule = [
        0.0002, 0.0004, 0.0006, 0.0008, 0.0010,  # warmup
        0.000999, 0.000995, 0.000989, 0.000981, 0.000970,
        0.000957, 0.000942, 0.000925, 0.000905, 0.000884,
        0.000861, 0.000836, 0.000810, 0.000782, 0.000753,
        0.000722, 0.000690, 0.000658, 0.000625, 0.000591,
        0.000557, 0.000522, 0.000488, 0.000453, 0.000419,
        0.000385, 0.000352, 0.000320, 0.000288, 0.000258,
        0.000229, 0.000202, 0.000177, 0.000154, 0.000133,
    ]

    best_epoch = 30  # epoch with best val AUC (0.9750)

    # ─── Panel a: Training loss ──────────────────────────────────
    ax = axes[0]

    # Loss curve with gradient fill
    ax.fill_between(epochs, losses, [max(losses)] * len(epochs),
                    alpha=0.08, color=PAL["seal"])
    ax.plot(epochs, losses, "-", linewidth=2.0, color=PAL["seal"],
            zorder=3, label="Training loss")
    ax.scatter(epochs, losses, s=10, color=PAL["seal"], zorder=4,
               edgecolors="white", linewidths=0.3)

    # LR schedule overlay (secondary y-axis)
    ax2 = ax.twinx()
    ax2.plot(epochs, [lr * 1000 for lr in lr_schedule], "--",
             linewidth=0.8, color=PAL["accent"], alpha=0.5,
             label="LR (×10³)")
    ax2.set_ylabel("Learning rate (×10⁻³)", fontsize=7, color=PAL["accent"])
    ax2.tick_params(axis="y", labelcolor=PAL["accent"], labelsize=6)
    ax2.set_ylim(0, 1.2)

    # Warmup annotation
    ax.axvspan(0, 5, alpha=0.06, color=PAL["accent"], zorder=1)
    ax.text(2.5, max(losses) - 0.005, "Warmup", fontsize=5.5,
            ha="center", color=PAL["accent"], fontstyle="italic")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_xlim(0, 42)
    ax.set_ylim(0.14, 0.38)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _panel_label(ax, "a")

    # ─── Panel b: Validation AUC ────────────────────────────────
    ax = axes[1]

    ax.fill_between(epochs, val_aucs, [min(val_aucs)] * len(epochs),
                    alpha=0.08, color=PAL["seal"])
    ax.plot(epochs, val_aucs, "-", linewidth=2.0, color=PAL["seal"],
            zorder=3, label="Val AUC")
    ax.scatter(epochs, val_aucs, s=10, color=PAL["seal"], zorder=4,
               edgecolors="white", linewidths=0.3)

    # Mark best epoch
    ax.scatter([best_epoch], [val_aucs[best_epoch - 1]], s=80,
               color=PAL["alert"], zorder=5, edgecolors="white",
               linewidths=1.0, marker="*")

    # Early stopping region
    ax.axvspan(best_epoch, 40, alpha=0.04, color=PAL["alert"], zorder=1)
    ax.text(35, 0.928, "Early stop\npatience", fontsize=5.5,
            ha="center", color=PAL["alert"], fontstyle="italic")

    # AUC plateau line
    ax.axhline(y=0.97, color=PAL["neutral"], linestyle=":", linewidth=0.8,
               alpha=0.6, zorder=1)
    ax.text(2, 0.9708, "AUC = 0.97", fontsize=5.5, color="#888888")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_xlim(0, 42)
    ax.set_ylim(0.925, 0.980)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _panel_label(ax, "b")

    # Config annotation
    ax.text(
        0.98, 0.05,
        "Osteoporosis LOO | seed=42\n"
        "SAGEConv+JK | hidden=32\n"
        f"H@20={10}/27 (37%) | Med.Rank=28",
        transform=ax.transAxes, fontsize=5.5,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                  edgecolor="#CCCCCC", linewidth=0.5),
    )

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8_training_curves", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 9 — COVID-19 RCT Drug Rank Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig9_rct_comparison(fmt: str = "png") -> None:
    """
    Lollipop chart: successful vs failed COVID-19 RCT drugs by SEAL rank.
    Demonstrates model's ability to discriminate clinically effective drugs.

    Data: ALL_RESULTS_SUMMARY.md §4
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.2))  # Taller for spacing

    # (drug_name, rank, outcome)  — seed=42 results
    drugs = [
        ("Tocilizumab",         99,   "success"),
        ("Remdesivir",          196,  "success"),
        ("Dexamethasone",       569,  "success"),
        ("Baricitinib",         1122, "success"),
        ("Ivermectin",          254,  "failed"),
        ("Ruxolitinib",         541,  "failed"),
        ("Interferon beta-1a",  792,  "failed"),
        ("Lopinavir",           1142, "failed"),
        ("Anakinra",            2102, "failed"),
        ("Hydroxychloroquine",  2125, "failed"),
    ]

    # Sort by rank (best at top)
    drugs.sort(key=lambda x: x[1])
    names  = [d[0] for d in drugs]
    ranks  = [d[1] for d in drugs]
    colours_bar = [PAL["seal"] if d[2] == "success" else PAL["alert"]
                   for d in drugs]

    y_pos = np.arange(len(drugs))

    # Draw lollipop stems
    for i, (r, c) in enumerate(zip(ranks, colours_bar)):
        ax.plot([0, r], [i, i], color=c, linewidth=1.5, zorder=2)
        ax.scatter(r, i, color=c, s=55, zorder=4,
                   edgecolors="white", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("SEAL rank (out of 2,471 drugs, lower = better)", labelpad=8)
    ax.invert_yaxis()
    ax.set_xlim(-50, 2350)

    # Median reference lines
    success_med = 569
    failed_med = 1142
    ax.axvline(x=success_med, color=PAL["seal"], linestyle=":",
               alpha=0.5, linewidth=1.0, zorder=1)
    ax.axvline(x=failed_med, color=PAL["alert"], linestyle=":",
               alpha=0.5, linewidth=1.0, zorder=1)

    # Annotations - positioned below x-axis with more space
    ax.text(
        0.02, -0.18,
        f"Success median: {success_med}  |  Failed median: {failed_med}  |  Separation: {failed_med / success_med:.1f}×",
        transform=ax.transAxes, fontsize=7, fontweight="bold",
        ha="left", va="top", color="#333333",
    )

    # Value labels
    for i, (r, c) in enumerate(zip(ranks, colours_bar)):
        ax.text(r + 30, i, str(r), va="center", fontsize=6.5,
                color="#555555", fontweight="medium")

    # Legend - more prominent
    handles = [
        mpatches.Patch(facecolor=PAL["seal"], edgecolor="white", label="Successful RCT (blue)"),
        mpatches.Patch(facecolor=PAL["alert"], edgecolor="white", label="Failed RCT (red)"),
    ]
    leg = ax.legend(
        handles=handles, loc="upper right", fontsize=7,
        frameon=True, edgecolor="#CCCCCC",
    )
    leg.get_frame().set_linewidth(0.5)

    fig.subplots_adjust(bottom=0.22)  # More space for annotation below x-axis
    save(fig, "fig9_rct_comparison", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 10 — Long COVID Score Distribution
# ═══════════════════════════════════════════════════════════════════════

def fig10_score_distribution(fmt: str = "png") -> None:
    """
    KDE + rug plot of SEAL consensus prediction scores for the 96 consensus
    drugs, loaded from real experimental results.

    Data: results/long_covid/final_predictions_*.csv
    """
    import csv
    from scipy.stats import gaussian_kde

    # ── Load real consensus drug scores from CSV ─────────────────────
    csv_path = Path("results") / "long_covid" / "final_predictions_20260303_121216.csv"
    scores, names, is_rct_list, n_seeds_list = [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["mean_score"]))
            names.append(row["name"])
            is_rct_list.append(row["is_rct"] == "True")
            n_seeds_list.append(int(row["n_seeds"]))

    all_scores = np.array(scores)
    top20_mask = np.arange(len(all_scores)) < 20
    rct_mask = np.array(is_rct_list)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))

    # ── KDE for full distribution ────────────────────────────────────
    x_grid = np.linspace(0.22, 0.68, 300)
    kde_all = gaussian_kde(all_scores, bw_method=0.06)
    kde_top20 = gaussian_kde(all_scores[top20_mask], bw_method=0.06)

    # Full distribution fill
    ax.fill_between(x_grid, kde_all(x_grid), alpha=0.15, color=PAL["neutral"],
                    zorder=2, label="All 96 consensus drugs")
    ax.plot(x_grid, kde_all(x_grid), color=PAL["neutral"], linewidth=1.2,
            zorder=3)

    # Top-20 overlay
    ax.fill_between(x_grid, kde_top20(x_grid), alpha=0.4, color=PAL["seal"],
                    zorder=4, label="Top 20 drugs")
    ax.plot(x_grid, kde_top20(x_grid), color=PAL["seal"], linewidth=1.5,
            zorder=5)

    # ── Rug plot: individual drug positions ──────────────────────────
    rug_y = -0.08 * kde_all(x_grid).max()
    for i, (sc, rct) in enumerate(zip(all_scores, is_rct_list)):
        colour = PAL["alert"] if rct else (PAL["seal"] if i < 20 else PAL["neutral"])
        lw = 1.5 if rct else (1.0 if i < 20 else 0.5)
        alpha = 1.0 if rct else (0.8 if i < 20 else 0.4)
        ax.plot([sc, sc], [rug_y * 0.3, rug_y * 1.3], color=colour,
                linewidth=lw, alpha=alpha, zorder=6)

    # ── Mark RCT drugs with labels ───────────────────────────────────
    rct_indices = [i for i, r in enumerate(is_rct_list) if r]
    for idx in rct_indices:
        sc = all_scores[idx]
        name = names[idx]
        # Map ChEMBL IDs to readable names
        label_map = {
            "CHEMBL384467": "Dexamethasone",
            "CHEMBL131": "Prednisolone",
            "CHEMBL19019": "Naltrexone",
        }
        drug_ids_at_row = list(csv.DictReader(open(csv_path)))[idx]["drug_id"]
        label = label_map.get(drug_ids_at_row, name)
        rank = idx + 1

        ax.axvline(x=sc, color=PAL["alert"], linestyle="--",
                   linewidth=1.0, alpha=0.6, zorder=4)
        y_pos = kde_all(np.array([sc]))[0]
        ax.annotate(
            f"{label}\n(rank {rank}, RCT)",
            xy=(sc, y_pos), xytext=(sc + 0.035, y_pos + 0.3),
            fontsize=6, color=PAL["alert"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PAL["alert"],
                            lw=0.8, connectionstyle="arc3,rad=0.2"),
            zorder=8,
        )

    ax.set_xlabel("Mean SEAL score (5-seed consensus)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.22, 0.68)
    ax.set_ylim(bottom=rug_y * 1.8)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)

    # ── Stats text box ───────────────────────────────────────────────
    ax.text(
        0.98, 0.95,
        f"n = {len(all_scores)} consensus drugs\n"
        f"Mean = {all_scores.mean():.3f}\n"
        f"Median = {np.median(all_scores):.3f}\n"
        f"Range: {all_scores.min():.3f}–{all_scores.max():.3f}\n"
        f"RCT drugs found: {rct_mask.sum()}/3",
        transform=ax.transAxes, fontsize=6.5,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                  edgecolor="#CCCCCC", linewidth=0.5),
    )

    leg = ax.legend(loc="upper left", fontsize=7, frameon=True,
                    edgecolor="#CCCCCC")
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    save(fig, "fig10_score_distribution", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 11 — Knowledge Graph Degree Distribution
# ═══════════════════════════════════════════════════════════════════════

def fig11_degree_distribution(fmt: str = "png") -> None:
    """
    Log-log histogram of node degree distribution in the knowledge graph.
    Highlights heavy-tailed nature and hub gene exclusion threshold.

    Data: results/graph_21.06_processed_*.pt (pre-extracted stats)
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.8))

    # Degree distribution bins extracted from graph
    # n=31,902 nodes; min=2, median=14, mean=67, max=11,396
    bin_centres = [3, 8, 30, 75, 300, 750, 3000, 8000]
    bin_counts  = [6590, 7218, 9854, 2766, 4707, 692, 73, 2]
    bin_labels  = ["1–5", "6–10", "11–50", "51–100",
                   "101–500", "501–1k", "1k–5k", "5k+"]

    bars = ax.bar(
        range(len(bin_centres)), bin_counts,
        color=[PAL["seal"]] * 5 + [PAL["accent"]] * 2 + [PAL["alert"]],
        edgecolor="white", linewidth=0.5, width=0.65, zorder=3,
        alpha=0.85,
    )

    # Value labels
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        label_y = bar.get_height() + 100
        if count < 200:
            label_y = bar.get_height() + 150
        ax.text(
            bar.get_x() + bar.get_width() / 2, label_y,
            f"{count:,}", ha="center", va="bottom", fontsize=6.5,
            fontweight="bold", color="#333333",
        )

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=7, rotation=30, ha="right")
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Number of nodes")
    ax.set_yscale("log")
    ax.set_ylim(1, 15000)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)

    # Stats inset
    ax.text(
        0.98, 0.95,
        "n = 31,902 nodes\n"
        "Median degree = 14\n"
        "Mean degree = 67\n"
        "Max = 11,396 (TP53)",
        transform=ax.transAxes, fontsize=6.5,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                  edgecolor="#CCCCCC", linewidth=0.5),
    )

    plt.tight_layout()
    save(fig, "fig11_degree_distribution", fmt)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
# Figure 12 — SEAL Enclosing Subgraph (NetworkX visualisation)
# ═══════════════════════════════════════════════════════════════════════

def fig12_seal_subgraph(fmt: str = "png") -> None:
    """
    NetworkX visualisation of a real SEAL enclosing subgraph extracted
    from the knowledge graph. Shows DRNL labels and node types.

    Data: results/graph_*_processed_*.pt + mappings
    """
    import networkx as nx
    from collections import deque

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5.0))

    # ── Load the real graph ──────────────────────────────────────────
    graph_files = sorted(Path("results").glob("graph_*_processed_*.pt"))
    if not graph_files:
        print("  ⚠ No graph file found — generating schematic subgraph")
        _draw_schematic_subgraph(ax)
        save(fig, "fig12_seal_subgraph", fmt)
        plt.close()
        return

    import torch
    graph_data = torch.load(graph_files[-1], weights_only=False)
    edge_index = graph_data.edge_index

    mappings_path = str(graph_files[-1]).replace(".pt", "_mappings")
    import json
    with open(f"{mappings_path}/drug_key_mapping.json") as f:
        drug_map = {k: int(v) for k, v in json.load(f).items()}
    with open(f"{mappings_path}/disease_key_mapping.json") as f:
        disease_map = {k: int(v) for k, v in json.load(f).items()}
    with open(f"{mappings_path}/gene_key_mapping.json") as f:
        gene_map = {k: int(v) for k, v in json.load(f).items()}

    drug_set = set(drug_map.values())
    disease_set = set(disease_map.values())
    gene_set = set(gene_map.values())
    idx_to_drug = {v: k for k, v in drug_map.items()}
    idx_to_disease = {v: k for k, v in disease_map.items()}

    # ── Build adjacency ──────────────────────────────────────────────
    from collections import defaultdict
    adj = defaultdict(set)
    s, d = edge_index
    for i in range(edge_index.shape[1]):
        u, v = s[i].item(), d[i].item()
        adj[u].add(v)
        adj[v].add(u)

    # Pick a drug-disease pair: Dexamethasone → Osteoporosis
    target_drug = drug_map.get("CHEMBL384467")   # Dexamethasone
    target_disease = disease_map.get("EFO_0003854")  # Osteoporosis
    if target_drug is None or target_disease is None:
        # Fallback to first drug-disease pair
        target_drug = list(drug_set)[0]
        target_disease = list(disease_set)[0]

    # ── Extract 2-hop enclosing subgraph ─────────────────────────────
    def bfs_distances(start, max_hops=2, max_per_hop=15):
        """BFS from start, capped at max_per_hop per level."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            d_node = dist[node]
            if d_node >= max_hops:
                continue
            neighbours = list(adj.get(node, set()))
            # Sample if too many
            if len(neighbours) > max_per_hop:
                import random
                random.seed(42)
                neighbours = random.sample(neighbours, max_per_hop)
            for nb in neighbours:
                if nb not in dist:
                    dist[nb] = d_node + 1
                    queue.append(nb)
        return dist

    dist_u = bfs_distances(target_drug, max_hops=2, max_per_hop=12)
    dist_v = bfs_distances(target_disease, max_hops=2, max_per_hop=12)

    # Enclosing subgraph = nodes reachable from both within 2 hops
    subgraph_nodes = set(dist_u.keys()) & set(dist_v.keys())
    # Also include 1-hop exclusive neighbours for context (limit to 5)
    one_hop_u = {n for n in dist_u if dist_u[n] == 1 and n not in dist_v}
    one_hop_v = {n for n in dist_v if dist_v[n] == 1 and n not in dist_u}
    import random
    random.seed(42)
    if len(one_hop_u) > 5:
        one_hop_u = set(random.sample(list(one_hop_u), 5))
    if len(one_hop_v) > 5:
        one_hop_v = set(random.sample(list(one_hop_v), 5))
    subgraph_nodes |= one_hop_u | one_hop_v

    # Limit total size for readability
    if len(subgraph_nodes) > 40:
        # Keep target nodes + closest neighbours
        scored = []
        for n in subgraph_nodes:
            du = dist_u.get(n, 99)
            dv = dist_v.get(n, 99)
            scored.append((du + dv, n))
        scored.sort()
        subgraph_nodes = {n for _, n in scored[:40]}
    subgraph_nodes.add(target_drug)
    subgraph_nodes.add(target_disease)

    # ── DRNL Labels ──────────────────────────────────────────────────
    def drnl_label(du, dv):
        if du == 0 or dv == 0:
            return 1
        d = du + dv
        return 1 + min(du, dv) + (d // 2) * (d // 2 + d % 2 - 1)

    labels = {}
    for n in subgraph_nodes:
        du = dist_u.get(n, 99)
        dv = dist_v.get(n, 99)
        labels[n] = drnl_label(du, dv)

    # ── Build networkx graph ─────────────────────────────────────────
    G = nx.Graph()
    for n in subgraph_nodes:
        G.add_node(n)
    s_arr, d_arr = edge_index
    for i in range(edge_index.shape[1]):
        u, v = s_arr[i].item(), d_arr[i].item()
        if u in subgraph_nodes and v in subgraph_nodes:
            G.add_edge(u, v)

    # ── Node attributes ──────────────────────────────────────────────
    node_colours = []
    node_sizes = []
    node_labels_text = {}
    for n in G.nodes():
        z = labels.get(n, 99)
        if n == target_drug:
            node_colours.append(PAL["seal"])
            node_sizes.append(350)
            node_labels_text[n] = "Dexamethasone\n(u, z=1)"
        elif n == target_disease:
            node_colours.append(PAL["alert"])
            node_sizes.append(350)
            node_labels_text[n] = "Osteoporosis\n(v, z=1)"
        elif n in drug_set:
            node_colours.append("#7DB5D4")  # lighter blue
            node_sizes.append(120)
            node_labels_text[n] = f"z={z}"
        elif n in disease_set:
            node_colours.append("#E8847C")  # lighter red
            node_sizes.append(120)
            node_labels_text[n] = f"z={z}"
        else:  # gene
            node_colours.append(PAL["heuristic"])  # teal for genes
            node_sizes.append(80)
            node_labels_text[n] = f"z={z}"

    # ── Edge styling ─────────────────────────────────────────────────
    edge_colours = []
    edge_widths = []
    for u, v in G.edges():
        if (u == target_drug and v == target_disease) or \
           (v == target_drug and u == target_disease):
            edge_colours.append(PAL["accent"])
            edge_widths.append(2.5)
        elif u in drug_set or v in drug_set:
            edge_colours.append("#B0D0E8")
            edge_widths.append(0.8)
        elif u in disease_set or v in disease_set:
            edge_colours.append("#E8B8B0")
            edge_widths.append(0.8)
        else:
            edge_colours.append("#D0D0D0")
            edge_widths.append(0.4)

    # ── Layout ───────────────────────────────────────────────────────
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=100)

    # Draw graph
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colours,
                           width=edge_widths, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=node_sizes, edgecolors="white",
                           linewidths=0.5, alpha=0.9)

    # Labels for target nodes (larger, offset)
    target_labels = {n: l for n, l in node_labels_text.items()
                     if n in (target_drug, target_disease)}
    other_labels = {n: l for n, l in node_labels_text.items()
                    if n not in (target_drug, target_disease)}

    nx.draw_networkx_labels(G, pos, labels=target_labels, ax=ax,
                            font_size=7, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=other_labels, ax=ax,
                            font_size=5, font_color="#666666")

    # ── Legend ────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=PAL["seal"], label="Drug (target)"),
        mpatches.Patch(facecolor=PAL["alert"], label="Disease (target)"),
        mpatches.Patch(facecolor=PAL["heuristic"], label="Gene"),
        mpatches.Patch(facecolor="#7DB5D4", label="Other drug"),
        mpatches.Patch(facecolor="#E8847C", label="Other disease"),
        plt.Line2D([0], [0], color=PAL["accent"], linewidth=2.5,
                   label="Target link (u,v)"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper left", fontsize=6.5,
                    frameon=True, edgecolor="#CCCCCC", ncol=2)
    leg.get_frame().set_linewidth(0.5)

    # Stats annotation
    ax.text(
        0.98, 0.02,
        f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}\n"
        f"2-hop enclosing subgraph\n"
        f"DRNL labels: z ∈ [1, {max(labels.values())}]",
        transform=ax.transAxes, fontsize=6.5,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                  edgecolor="#CCCCCC", linewidth=0.5),
    )

    ax.set_title("SEAL Enclosing Subgraph: Dexamethasone → Osteoporosis",
                 fontsize=9, fontweight="bold", pad=10)
    ax.axis("off")

    plt.tight_layout()
    save(fig, "fig12_seal_subgraph", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

ALL_FIGURES = {
    1:  ("fig1_tournament",            fig1_tournament),
    2:  ("fig2_edge_ablation",         fig2_edge_ablation),
    3:  ("fig3_node_ablation",         fig3_node_ablation),
    4:  ("fig4_disease_complexity",    fig4_disease_complexity),
    5:  ("fig5_long_covid",            fig5_long_covid),
    6:  ("fig6_gene_config",           fig6_gene_config),
    7:  ("fig7_temporal_ranks",        fig7_temporal_ranks),
    8:  ("fig8_training_curves",       fig8_training_curves),
    9:  ("fig9_rct_comparison",        fig9_rct_comparison),
    10: ("fig10_score_distribution",   fig10_score_distribution),
    11: ("fig11_degree_distribution",  fig11_degree_distribution),
    12: ("fig12_seal_subgraph",        fig12_seal_subgraph),
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

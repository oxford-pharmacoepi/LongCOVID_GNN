#!/usr/bin/env python3
"""
Publication-quality figures for the SEAL drug repurposing manuscript.

Generates 6 figures covering the main experimental results:
    fig1  — Model tournament (SEAL vs Global GNNs vs Heuristics)
    fig2  — Edge-type ablation impact on Osteoporosis LOO
    fig3  — Node-feature ablation (DRNL sufficiency)
    fig4  — Disease complexity vs SEAL performance
    fig5  — Long COVID top-20 consensus drug candidates
    fig6  — Gene-configuration comparison (NARROW vs BROAD vs FULL)

Usage:
    uv run python scripts/create_paper_plots.py            # all figures
    uv run python scripts/create_paper_plots.py --figures 1 5   # specific
    uv run python scripts/create_paper_plots.py --dark          # dark theme
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
from matplotlib.collections import PatchCollection   # noqa: E402
import numpy as np                                   # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

OUT_DIR = Path("results/figures")

# ------------------------------------------------------------------
# Colour palettes
# ------------------------------------------------------------------
PALETTE_LIGHT = {
    "bg":          "#FFFFFF",
    "card":        "#F8FAFC",
    "text":        "#111827",
    "text_sec":    "#6B7280",
    "grid":        "#E5E7EB",
    "spine":       "#D1D5DB",
    # Model colours — vibrant yet professional
    "seal":        "#2563EB",      # blue-600
    "seal_grad":   "#7C3AED",      # violet-600 (gradient end)
    "seal_light":  "#BFDBFE",      # blue-200
    "gat":         "#DC2626",      # red-600
    "gat_light":   "#FECACA",      # red-200
    "sage":        "#7C3AED",      # violet-600
    "heuristic":   "#059669",      # emerald-600
    "neutral":     "#9CA3AF",      # grey-400
    "accent":      "#F59E0B",      # amber-500
    "success":     "#10B981",      # emerald-500
    "danger":      "#EF4444",      # red-500
    "bar_edge":    "#FFFFFF",
    # Drug class colours (fig 5)
    "antiviral":       "#2563EB",
    "corticosteroid":  "#DC2626",
    "anti_inflam":     "#F59E0B",
    "immunotherapy":   "#7C3AED",
    "antimalarial":    "#0891B2",
    "antimicrobial":   "#059669",
    "antibiotic":      "#78716C",
    "hormone":         "#DB2777",
    "other":           "#D1D5DB",
}

PALETTE_DARK = {
    "bg":          "#0F172A",
    "card":        "#1E293B",
    "text":        "#F1F5F9",
    "text_sec":    "#94A3B8",
    "grid":        "#334155",
    "spine":       "#475569",
    "seal":        "#60A5FA",
    "seal_grad":   "#A78BFA",
    "seal_light":  "#1E3A5F",
    "gat":         "#F87171",
    "gat_light":   "#7F1D1D",
    "sage":        "#A78BFA",
    "heuristic":   "#34D399",
    "neutral":     "#9CA3AF",
    "accent":      "#FBBF24",
    "success":     "#34D399",
    "danger":      "#F87171",
    "bar_edge":    "#1E293B",
    "antiviral":       "#60A5FA",
    "corticosteroid":  "#F87171",
    "anti_inflam":     "#FBBF24",
    "immunotherapy":   "#A78BFA",
    "antimalarial":    "#22D3EE",
    "antimicrobial":   "#34D399",
    "antibiotic":      "#A8A29E",
    "hormone":         "#F472B6",
    "other":           "#475569",
}

# Active palette — set by CLI flag
P = PALETTE_LIGHT


def apply_style() -> None:
    """Apply a consistent, premium matplotlib style."""
    plt.rcParams.update({
        # Typography
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Arial",
                               "DejaVu Sans"],
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.titlepad":      14,
        "axes.labelsize":     11,
        "axes.labelweight":   "medium",
        "axes.labelpad":      8,
        "xtick.labelsize":    9.5,
        "ytick.labelsize":    9.5,
        "legend.fontsize":    9,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   P["grid"],
        "legend.fancybox":    True,
        # Figure
        "figure.facecolor":   P["bg"],
        "axes.facecolor":     P["bg"],
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.25,
        "savefig.facecolor":  P["bg"],
        # Grid & spines
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.alpha":         0.35,
        "grid.color":         P["grid"],
        "grid.linewidth":     0.5,
        "grid.linestyle":     "--",
        # Colours
        "text.color":         P["text"],
        "axes.edgecolor":     P["spine"],
        "axes.labelcolor":    P["text"],
        "xtick.color":        P["text_sec"],
        "ytick.color":        P["text_sec"],
    })


def save(fig: plt.Figure, name: str, fmt: str = "png") -> None:
    """Save figure to OUT_DIR in requested format + PDF."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (fmt, "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
    print(f"  ✓ {name}")


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _shadow() -> list:
    """Return subtle text shadow path effects."""
    return [pe.withStroke(linewidth=3, foreground=P["bg"])]


def _rounded_bar(ax, x, height, width=0.6, colour="#333",
                 bottom=0, radius=0.08, **kwargs):
    """Draw a single bar with rounded top corners."""
    from matplotlib.patches import FancyBboxPatch
    left = x - width / 2
    rect = FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=colour, edgecolor=P["bar_edge"],
        linewidth=0.7, zorder=3, **kwargs,
    )
    ax.add_patch(rect)
    return rect


def _add_value_label(ax, x, y, text, colour=None, fontsize=9,
                     va="bottom", offset=0):
    """Add a styled value label above/below a bar."""
    colour = colour or P["text"]
    txt = ax.text(
        x, y + offset, text,
        ha="center", va=va, fontsize=fontsize,
        fontweight="bold", color=colour,
    )
    txt.set_path_effects(_shadow())
    return txt


def _gradient_fill(ax, x, y_bottom, y_top, colour_bottom, colour_top,
                   width=0.6, n_steps=50):
    """Fill a vertical region with a smooth gradient."""
    for i in range(n_steps):
        frac = i / n_steps
        y0 = y_bottom + frac * (y_top - y_bottom)
        y1 = y_bottom + (frac + 1 / n_steps) * (y_top - y_bottom)
        r = frac
        c0 = mcolors.to_rgb(colour_bottom)
        c1 = mcolors.to_rgb(colour_top)
        c = tuple(c0[j] * (1 - r) + c1[j] * r for j in range(3))
        ax.fill_between(
            [x - width / 2, x + width / 2], y0, y1,
            color=c, linewidth=0, zorder=2,
        )


def _subtitle(ax, text, y=-0.12, fontsize=9):
    """Add a descriptive subtitle below the axis."""
    ax.text(0.5, y, text, transform=ax.transAxes,
            ha="center", fontsize=fontsize, color=P["text_sec"],
            style="italic")


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Model Tournament
# ═══════════════════════════════════════════════════════════════════════

def fig1_tournament(fmt: str = "png") -> None:
    """
    Grouped bar chart comparing Hits@100 across three diseases for
    SEAL, GAT, SAGE-3L, and Adamic-Adar heuristic.

    Data: ALL_RESULTS_SUMMARY.md §1, FULL_TOURNAMENT_REPORT.md
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

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
        ("SEAL (SAGE+JK)", seal,  P["seal"]),
        ("GAT (best global)", gat,   P["gat"]),
        ("SAGE-3L", sage3, P["sage"]),
        ("Adamic-Adar", aa, P["heuristic"]),
    ]

    x = np.arange(len(diseases))
    n = len(models)
    w = 0.17
    gap = 0.03

    for idx, (label, values, colour) in enumerate(models):
        offset = (idx - (n - 1) / 2) * (w + gap)
        bars = ax.bar(
            x + offset, values, w,
            label=label, color=colour,
            edgecolor="white", linewidth=0.7, zorder=3,
            alpha=0.92,
        )
        # Value labels
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                _add_value_label(
                    ax, bar.get_x() + bar.get_width() / 2,
                    h, f"{h:.0f}%", fontsize=8, offset=1.2,
                )

    ax.set_ylabel("Hits@100 (%)")
    ax.set_title("Model Tournament — Drug Recovery Rate (Hits@100)")
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=10)
    ax.set_ylim(0, 110)

    # Legend
    ax.legend(
        loc="upper right", frameon=True, fancybox=True,
        edgecolor=P["grid"], framealpha=0.95,
    )

    # Complexity gradient arrow
    ax.annotate(
        "", xy=(2.42, -8), xytext=(-0.42, -8),
        xycoords="data", textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>", color=P["text_sec"],
            lw=1.2, mutation_scale=12,
        ),
        annotation_clip=False,
    )
    ax.text(
        1.0, -13,
        "← focused                  disease complexity                  broad →",
        ha="center", fontsize=8, color=P["text_sec"],
        transform=ax.get_xaxis_transform(),
    )

    # Highlight winner per disease
    for i, (s, g, sa, a) in enumerate(zip(seal, gat, sage3, aa)):
        best = max(s, g, sa, a)
        if best == s:
            ax.plot(i - 1.5 * (w + gap), best + 5, "▾",
                    color=P["seal"], markersize=6, zorder=5)

    plt.tight_layout()
    save(fig, "fig1_tournament", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — Edge-Type Ablation
# ═══════════════════════════════════════════════════════════════════════

def fig2_edge_ablation(fmt: str = "png") -> None:
    """
    Horizontal bar chart showing how removing each edge type affects
    the median rank in Osteoporosis LOO validation.

    Data: ALL_RESULTS_SUMMARY.md §7.A
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    configs = [
        "Full graph (baseline)",
        "No PPI edges (−75%)",
        "No disease similarity (−0.5%)",
        "No drug–gene / MoA (−0.9%)",
        "No disease–gene (−9%)",
    ]
    med_ranks = [29, 31, 40, 334, 553]

    # Gradient: good (teal) → bad (red)
    norm = plt.Normalize(0, 553)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "perf",
        [P["seal"], P["seal_light"], P["accent"], P["danger"]],
    )
    colours = [cmap(norm(v)) for v in med_ranks]

    bars = ax.barh(
        range(len(configs)), med_ranks, color=colours,
        edgecolor="white", linewidth=0.8, height=0.6, zorder=3,
    )

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)
    ax.set_xlabel("Median Rank (lower = better)")
    ax.set_title("Edge-Type Ablation — Osteoporosis LOO")
    ax.invert_yaxis()
    ax.set_xlim(0, 700)

    # Value labels
    for i, (bar, v) in enumerate(zip(bars, med_ranks)):
        c = P["danger"] if v > 100 else P["text"]
        x_pos = v + 12
        label = str(v)
        if v > 100:
            label += "  ▲ critical"
        txt = ax.text(x_pos, i, label, va="center", fontsize=10,
                      fontweight="bold", color=c)
        txt.set_path_effects(_shadow())

    # Reference line at baseline
    ax.axvline(x=29, color=P["seal"], linestyle=":", alpha=0.4,
               linewidth=1.2, zorder=1, label="Baseline (29)")

    # Annotations
    ax.annotate(
        "Drug–gene & disease–gene edges\nare the core biological signal",
        xy=(553, 4), xytext=(480, 2.5),
        fontsize=8, color=P["text_sec"], ha="center",
        arrowprops=dict(arrowstyle="->", color=P["text_sec"], lw=0.8),
    )

    plt.tight_layout()
    save(fig, "fig2_edge_ablation", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — Node Feature Ablation
# ═══════════════════════════════════════════════════════════════════════

def fig3_node_ablation(fmt: str = "png") -> None:
    """
    Paired panels showing that DRNL-only SEAL matches or beats
    the full-feature model on Hits@20 and Median Rank.

    Data: ALL_RESULTS_SUMMARY.md §7.B
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = [
        "DRNL + features\n+ types (49-dim)",
        "DRNL + types\n(3-dim)",
        "DRNL only\n(0-dim)",
    ]
    h20        = [40.7, 40.7, 44.4]
    medrank    = [29,   24,   22]

    # Colour progression: more minimal = more vibrant
    cols = [P["neutral"], P["seal_light"], P["seal"]]
    edge_cols = ["#D1D5DB", "#93C5FD", "#2563EB"]

    # ─── Panel A: Hits@20 ────────────────────────────────────────
    ax = axes[0]
    bars_a = ax.bar(
        range(3), h20, color=cols, edgecolor=edge_cols,
        linewidth=1.5, width=0.55, zorder=3,
    )
    for i, bar in enumerate(bars_a):
        _add_value_label(
            ax, bar.get_x() + bar.get_width() / 2,
            bar.get_height(), f"{h20[i]:.1f}%",
            fontsize=10, offset=0.8,
        )
    ax.set_ylabel("Hits@20 (%)")
    ax.set_title("a)  Drug Recovery Rate")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 56)

    # "Best" badge
    ax.annotate(
        "★ Best", xy=(2, 44.4), xytext=(2, 51),
        ha="center", fontsize=8.5, fontweight="bold", color=P["seal"],
        arrowprops=dict(arrowstyle="->", color=P["seal"], lw=1.2),
    )

    # ─── Panel B: Median Rank ────────────────────────────────────
    ax = axes[1]
    bars_b = ax.bar(
        range(3), medrank, color=cols, edgecolor=edge_cols,
        linewidth=1.5, width=0.55, zorder=3,
    )
    for i, bar in enumerate(bars_b):
        _add_value_label(
            ax, bar.get_x() + bar.get_width() / 2,
            bar.get_height(), str(medrank[i]),
            fontsize=10, offset=0.5,
        )
    ax.set_ylabel("Median Rank (lower = better)")
    ax.set_title("b)  Ranking Quality")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 38)

    ax.annotate(
        "★ Best", xy=(2, 22), xytext=(2, 28),
        ha="center", fontsize=8.5, fontweight="bold", color=P["seal"],
        arrowprops=dict(arrowstyle="->", color=P["seal"], lw=1.2),
    )

    fig.suptitle(
        "Node Feature Ablation — DRNL Structural Labels Suffice",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save(fig, "fig3_node_ablation", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — Disease Complexity vs Performance
# ═══════════════════════════════════════════════════════════════════════

def fig4_disease_complexity(fmt: str = "png") -> None:
    """
    Bubble chart: n_true_drugs vs median_rank, bubble size = H@100%.

    Data: ALL_RESULTS_SUMMARY.md §1, §8
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # (n_true_drugs, median_rank, name, hits_at_100_pct)
    data = [
        (27,   24,   "Osteoporosis",          92.6),
        (59,   130,  "Multiple\nSclerosis",   44.1),
        (59,   206,  "Ank. Spondylitis\n(propagated)", 33.9),
        (44,   809,  "Neuropathic\nPain",     None),
        (56,   354,  "Type 2\nDiabetes",      None),
        (108,  399,  "Depression",            15.1),
    ]

    for n, mr, name, h100 in data:
        size = h100 * 4 if h100 else 100
        alpha = 0.82 if h100 else 0.45
        if h100 and h100 > 60:
            colour = P["seal"]
        elif h100 and h100 > 30:
            colour = P["accent"]
        elif h100:
            colour = P["danger"]
        else:
            colour = P["neutral"]

        ax.scatter(
            n, mr, s=size, c=colour, alpha=alpha, zorder=5,
            edgecolors="white", linewidth=2,
        )

        # Smart label positioning
        ha, va, dx, dy = "left", "bottom", 6, 18
        if name.startswith("Osteo"):
            dx, dy, va = 6, -30, "top"
        elif name.startswith("Dep"):
            dx, dy, va = 6, -30, "top"
        elif name.startswith("Type"):
            dx, dy = 6, 12
        elif name.startswith("Neuro"):
            dx, dy, ha = 6, -20, "left"
            va = "top"

        ax.annotate(
            name, (n, mr), xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5, ha=ha, va=va,
            color=P["text"], fontweight="medium",
            arrowprops=dict(
                arrowstyle="-", color=P["grid"],
                lw=0.6, connectionstyle="arc3,rad=0.1",
            ),
        )

    ax.set_xlabel("Number of True Drugs for Disease")
    ax.set_ylabel("SEAL Median Rank (lower = better)")
    ax.set_title("Disease Complexity vs SEAL LOO Performance")
    ax.set_xlim(15, 120)
    ax.set_ylim(-60, 920)

    # Bubble size legend
    for sz, lab in [
        (92 * 4, "H@100 ≈ 93%"),
        (44 * 4, "H@100 ≈ 44%"),
        (100,    "Not tested"),
    ]:
        ax.scatter([], [], s=sz, c=P["neutral"], alpha=0.5, label=lab,
                   edgecolors="white", linewidth=1.5)
    ax.legend(
        loc="upper right", title="Bubble size = H@100",
        title_fontsize=8.5, frameon=True, fancybox=True,
        edgecolor=P["grid"],
    )

    plt.tight_layout()
    save(fig, "fig4_disease_complexity", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 5 — Long COVID Top-20 Predictions
# ═══════════════════════════════════════════════════════════════════════

def fig5_long_covid(fmt: str = "png") -> None:
    """
    Horizontal bar chart of the top 20 consensus drug predictions,
    colour-coded by therapeutic class. Stars mark RCT drugs.
    Seed consistency shown as text inside bars.

    Data: ALL_RESULTS_SUMMARY.md §11
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))

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
            n += "  ⭐"
        names.append(n)
    scores  = [d[1] for d in drugs_rev]
    colours = [P.get(d[2], P["other"]) for d in drugs_rev]
    seeds   = [d[4] for d in drugs_rev]

    bars = ax.barh(
        range(20), scores, color=colours,
        edgecolor="white", linewidth=0.7, height=0.72, zorder=3,
        alpha=0.9,
    )

    ax.set_yticks(range(20))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("Mean SEAL Score (5-seed consensus, ≥3/5 seeds)")
    ax.set_title(
        "Long COVID — Top 20 Predicted Drug Candidates",
        pad=16,
    )
    ax.set_xlim(0, 0.78)

    # Seed count & score inside/beside bars
    for i, (bar, sd, sc) in enumerate(zip(bars, seeds, scores)):
        w = bar.get_width()
        # Seed count inside bar (right-aligned)
        ax.text(
            w - 0.012, i, sd,
            va="center", ha="right", fontsize=7.5,
            color="white", fontweight="bold", alpha=0.9,
        )
        # Score to the right
        txt = ax.text(
            w + 0.008, i, f"{sc:.3f}",
            va="center", ha="left", fontsize=8,
            color=P["text_sec"], fontweight="medium",
        )
        txt.set_path_effects(_shadow())

    # Legend by drug class (de-duplicated, in order)
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
                    color=P.get(cls, P["other"]),
                    label=class_labels.get(cls, cls),
                    alpha=0.9,
                )
            )
    # Add RCT marker
    handles.append(
        plt.Line2D([0], [0], marker="*", color="none",
                   markerfacecolor=P["accent"], markersize=10,
                   label="RCT drug")
    )
    ax.legend(
        handles=handles, loc="lower right", fontsize=8,
        frameon=True, fancybox=True,
        title="Drug Class", title_fontsize=9,
        edgecolor=P["grid"],
    )

    plt.tight_layout()
    save(fig, "fig5_long_covid_predictions", fmt)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Figure 6 — Gene Configuration Comparison
# ═══════════════════════════════════════════════════════════════════════

def fig6_gene_config(fmt: str = "png") -> None:
    """
    Paired bar chart comparing NARROW / BROAD / FULL gene configs on
    (a) median RCT drug rank and (b) RCT drugs recovered in top 50.

    Data: ALL_RESULTS_SUMMARY.md §2
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = ["NARROW\n(8 genes)", "BROAD\n(39 genes)", "FULL\n(52 genes)"]
    med_rct = [236, 379, 474]
    top50   = [3.3, 2.0, 2.3]

    cols       = [P["seal"], P["seal_light"], P["neutral"]]
    edge_cols  = ["#2563EB", "#93C5FD", "#D1D5DB"]

    # ─── Panel A: Median RCT Rank ────────────────────────────────
    ax = axes[0]
    bars = ax.bar(
        range(3), med_rct, color=cols, edgecolor=edge_cols,
        linewidth=1.5, width=0.5, zorder=3,
    )
    for i, bar in enumerate(bars):
        _add_value_label(
            ax, bar.get_x() + bar.get_width() / 2,
            bar.get_height(), str(med_rct[i]),
            fontsize=10, offset=8,
        )
    ax.set_ylabel("Median RCT Drug Rank (lower = better)")
    ax.set_title("a)  Ranking Quality")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 570)

    ax.annotate(
        "★ Best", xy=(0, 236), xytext=(0.6, 440),
        ha="center", fontsize=9, fontweight="bold", color=P["seal"],
        arrowprops=dict(arrowstyle="->", color=P["seal"], lw=1.2),
    )

    # ─── Panel B: RCT in Top 50 ─────────────────────────────────
    ax = axes[1]
    bars = ax.bar(
        range(3), top50, color=cols, edgecolor=edge_cols,
        linewidth=1.5, width=0.5, zorder=3,
    )
    for i, bar in enumerate(bars):
        _add_value_label(
            ax, bar.get_x() + bar.get_width() / 2,
            bar.get_height(), f"{top50[i]:.1f}",
            fontsize=10, offset=0.06,
        )
    ax.set_ylabel("Mean RCT Drugs in Top 50")
    ax.set_title("b)  Top-50 Recovery (avg 3 seeds)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 4.8)

    ax.annotate(
        "★ Best", xy=(0, 3.3), xytext=(0.6, 4.2),
        ha="center", fontsize=9, fontweight="bold", color=P["seal"],
        arrowprops=dict(arrowstyle="->", color=P["seal"], lw=1.2),
    )

    fig.suptitle(
        "Gene Configuration — Fewer, More Specific Genes Perform Best",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
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
    parser.add_argument(
        "--dark", action="store_true",
        help="Use dark colour palette (for presentations)",
    )
    args = parser.parse_args()

    # Apply theme
    global P
    if args.dark:
        P = PALETTE_DARK
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

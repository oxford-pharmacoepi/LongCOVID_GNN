#!/usr/bin/env python3
"""
Aggregate Long COVID SEAL results across multiple seeds.

Loads JSON result files from results/long_covid/, groups by configuration,
averages scores across seeds, and produces a final ranked table with drug
names, mean rank, mean score ± SD, RCT status, and mechanism info.

Usage:
    uv run python scripts/seal/aggregate_lc_results.py
    uv run python scripts/seal/aggregate_lc_results.py --seeds 7 42 123 456 789
    uv run python scripts/seal/aggregate_lc_results.py --include-gat
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "long_covid"
CHEMBL_CACHE = RESULTS_DIR / "chembl_cache.json"
MEDICINES_FILE = PROJECT_ROOT / "medicines_tested.txt"


def load_chembl_cache() -> dict:
    """Load the ChEMBL name/info cache."""
    if CHEMBL_CACHE.exists():
        with open(CHEMBL_CACHE) as f:
            return json.load(f)
    return {}


def load_rct_drugs() -> dict:
    """Load RCT drugs from medicines_tested.txt. Returns {chembl_id: name}."""
    rct = {}
    if MEDICINES_FILE.exists():
        with open(MEDICINES_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("drug_id") or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    rct[parts[0]] = parts[1]
    return rct


def find_result_files(
    conv_type: str = "sage",
    gene_categories: str = "gwas",
    seeds: list[int] | None = None,
    use_jk: bool | None = None,
) -> list[Path]:
    """Find result JSON files matching the specified configuration."""
    matches = []
    for f in sorted(RESULTS_DIR.glob("seal_long_covid_*.json")):
        try:
            data = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue

        cfg = data.get("config", {})

        # Match conv type
        if cfg.get("conv_type") != conv_type:
            continue

        # Match gene categories
        file_cats = cfg.get("gene_categories", "unknown")
        if file_cats != gene_categories:
            continue

        # Match JK if specified
        if use_jk is not None and cfg.get("use_jk") != use_jk:
            continue

        # Match seeds if specified
        if seeds is not None and cfg.get("seed") not in seeds:
            continue

        # Must have max_drug_per_gene set (hub gene cap)
        if cfg.get("max_drug_per_gene") is None:
            continue

        matches.append(f)

    return matches


def aggregate_scores(files: list[Path]) -> dict:
    """
    Aggregate drug scores across multiple seed runs.

    Returns a dict of {drug_id: {"scores": [...], "ranks": [...]}}.
    """
    drug_data = defaultdict(lambda: {"scores": [], "ranks": []})
    configs = []

    for f in files:
        data = json.load(open(f))
        configs.append(data["config"])

        for pred in data["top_predictions"]:
            drug_id = pred["drug_id"]
            drug_data[drug_id]["scores"].append(pred["score"])
            drug_data[drug_id]["ranks"].append(pred["rank"])

    return dict(drug_data), configs


def build_final_table(
    drug_data: dict,
    configs: list[dict],
    chembl_cache: dict,
    rct_drugs: dict,
    gat_data: dict | None = None,
) -> list[dict]:
    """Build the final ranked table."""
    # Compute mean score for each drug and sort
    ranked = []
    for drug_id, info in drug_data.items():
        scores = info["scores"]
        ranks = info["ranks"]
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        mean_rank = np.mean(ranks)

        # Drug name from cache
        cache_entry = chembl_cache.get(drug_id)
        if cache_entry and isinstance(cache_entry, dict):
            name = cache_entry.get("pref_name", drug_id)
        else:
            name = drug_id

        # RCT status
        is_rct = drug_id in rct_drugs
        rct_name = rct_drugs.get(drug_id, "")

        # GAT comparison
        gat_rank = None
        gat_score = None
        if gat_data and drug_id in gat_data:
            gat_rank = gat_data[drug_id].get("rank")
            gat_score = gat_data[drug_id].get("score")

        ranked.append({
            "drug_id": drug_id,
            "name": name if name else drug_id,
            "mean_score": round(mean_score, 6),
            "std_score": round(std_score, 6),
            "mean_rank": round(mean_rank, 1),
            "n_seeds": len(scores),
            "is_rct": is_rct,
            "rct_name": rct_name,
            "gat_rank": gat_rank,
            "gat_score": round(gat_score, 6) if gat_score is not None else None,
        })

    # Sort by mean score descending
    ranked.sort(key=lambda x: x["mean_score"], reverse=True)

    # Assign final rank
    for i, entry in enumerate(ranked, 1):
        entry["final_rank"] = i

    return ranked


def find_gat_results(seeds: list[int] | None = None) -> dict | None:
    """Find and load GAT Long COVID results for comparison."""
    files = find_result_files(
        conv_type="gat",
        gene_categories="gwas",
        seeds=seeds,
        use_jk=False,
    )
    if not files:
        return None

    # Use the most recent GAT result
    f = files[-1]
    data = json.load(open(f))
    gat_data = {}
    for pred in data["top_predictions"]:
        gat_data[pred["drug_id"]] = {
            "rank": pred["rank"],
            "score": pred["score"],
        }
    print(f"  Loaded GAT results from {f.name} (seed={data['config']['seed']})")
    return gat_data


def write_markdown(
    ranked: list[dict],
    configs: list[dict],
    gat_data: dict | None,
    drug_data: dict,
    output_path: Path,
    top_n: int = 50,
):
    """Write the final predictions as a Markdown file."""
    seeds = sorted(set(c["seed"] for c in configs))
    n_seeds = len(seeds)

    lines = []
    lines.append("# Long COVID Drug Repurposing — Final Predictions")
    lines.append("")
    lines.append(f"**Compiled:** {datetime.now().strftime('%d %B %Y')}")
    lines.append(f"**Model:** SEAL (SAGEConv + JK, hidden=32, hops=2, mean+max pooling)")
    lines.append(f"**Gene config:** NARROW (8 GWAS lead signal genes)")
    lines.append(f"**Seeds:** {seeds}")
    lines.append(f"**Hub gene cap:** max_drug_per_gene=5")
    val_aucs = [c.get("best_val_auc", 0) for c in configs]
    lines.append(f"**Val AUC:** {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Top N table
    lines.append(f"## Top {top_n} Predicted Drug Candidates")
    lines.append("")

    if gat_data:
        lines.append(
            f"| Rank | Drug | ChEMBL ID | SEAL Score (mean ± SD) | "
            f"GAT Rank | RCT? |"
        )
        lines.append("|:---:|:---|:---|:---:|:---:|:---:|")
    else:
        lines.append(
            f"| Rank | Drug | ChEMBL ID | SEAL Score (mean ± SD) | RCT? |"
        )
        lines.append("|:---:|:---|:---|:---:|:---:|")

    for entry in ranked[:top_n]:
        rct_flag = "⭐" if entry["is_rct"] else ""
        score_str = f"{entry['mean_score']:.4f} ± {entry['std_score']:.4f}"
        if entry["n_seeds"] == 1:
            score_str = f"{entry['mean_score']:.4f}"

        if gat_data:
            gat_rank_str = str(entry["gat_rank"]) if entry["gat_rank"] else "—"
            lines.append(
                f"| {entry['final_rank']} | **{entry['name']}** | "
                f"{entry['drug_id']} | {score_str} | "
                f"{gat_rank_str} | {rct_flag} |"
            )
        else:
            lines.append(
                f"| {entry['final_rank']} | **{entry['name']}** | "
                f"{entry['drug_id']} | {score_str} | {rct_flag} |"
            )

    lines.append("")

    # RCT drug section
    lines.append("---")
    lines.append("")
    lines.append("## Known Long COVID RCT Drugs — Rankings")
    lines.append("")
    lines.append("| Drug | ChEMBL ID | SEAL Rank | SEAL Score | In Top 50? |")
    lines.append("|:---|:---|:---:|:---:|:---:|")

    rct_entries = [e for e in ranked if e["is_rct"]]
    rct_entries.sort(key=lambda x: x["mean_rank"])
    for entry in rct_entries:
        in_top = "✅" if entry["final_rank"] <= 50 else ""
        lines.append(
            f"| {entry['rct_name'] or entry['name']} | {entry['drug_id']} | "
            f"{entry['final_rank']} | {entry['mean_score']:.4f} | {in_top} |"
        )

    # Also list RCT drugs NOT in the ranked list
    ranked_ids = {e["drug_id"] for e in ranked}
    rct_drugs = load_rct_drugs()
    missing_rct = {k: v for k, v in rct_drugs.items() if k not in ranked_ids}
    if missing_rct:
        lines.append("")
        lines.append("**RCT drugs not in top 100 or not in graph:**")
        for drug_id, name in missing_rct.items():
            lines.append(f"- {name} ({drug_id})")

    lines.append("")

    # GAT comparison section
    if gat_data:
        lines.append("---")
        lines.append("")
        lines.append("## SEAL vs GAT Top-50 Overlap")
        lines.append("")

        seal_top50 = {e["drug_id"] for e in ranked[:50]}
        gat_top50 = {did for did, info in gat_data.items() if info["rank"] <= 50}
        overlap = seal_top50 & gat_top50
        seal_only = seal_top50 - gat_top50
        gat_only = gat_top50 - seal_top50

        lines.append(f"- **SEAL top 50:** {len(seal_top50)} drugs")
        lines.append(f"- **GAT top 50:** {len(gat_top50)} drugs")
        lines.append(f"- **Overlap (high-confidence):** {len(overlap)} drugs")
        lines.append(f"- **SEAL-only:** {len(seal_only)} drugs")
        lines.append(f"- **GAT-only:** {len(gat_only)} drugs")
        lines.append("")

        if overlap:
            chembl_cache = load_chembl_cache()
            lines.append("### High-Confidence Candidates (in both SEAL and GAT top 50)")
            lines.append("")
            lines.append("| Drug | ChEMBL ID | SEAL Rank | GAT Rank |")
            lines.append("|:---|:---|:---:|:---:|")
            overlap_details = []
            for drug_id in overlap:
                seal_entry = next(e for e in ranked if e["drug_id"] == drug_id)
                gat_rank = gat_data[drug_id]["rank"]
                overlap_details.append((seal_entry["final_rank"], drug_id,
                                        seal_entry["name"], gat_rank))
            overlap_details.sort()
            for seal_rank, drug_id, name, gat_rank in overlap_details:
                lines.append(f"| {name} | {drug_id} | {seal_rank} | {gat_rank} |")
            lines.append("")

    # Seed stability
    lines.append("---")
    lines.append("")
    lines.append("## Seed Stability (Top 20)")
    lines.append("")
    lines.append("| Rank | Drug | Ranks Across Seeds | Score SD |")
    lines.append("|:---:|:---|:---|:---:|")
    for entry in ranked[:20]:
        ranks_str = ", ".join(str(r) for r in sorted(
            drug_data[entry["drug_id"]]["ranks"]
        )) if entry["drug_id"] in drug_data else "—"
        lines.append(
            f"| {entry['final_rank']} | {entry['name']} | "
            f"{ranks_str} | {entry['std_score']:.4f} |"
        )
    lines.append("")

    # Config summary
    lines.append("---")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- **Seeds:** {seeds}")
    lines.append(f"- **Conv:** {configs[0].get('conv_type', 'sage')}")
    lines.append(f"- **Hidden:** {configs[0].get('hidden', 32)}")
    lines.append(f"- **Hops:** {configs[0].get('hops', 2)}")
    lines.append(f"- **Use JK:** {configs[0].get('use_jk', True)}")
    lines.append(f"- **Neg strategy:** {configs[0].get('neg_strategy', 'mixed')}")
    lines.append(f"- **Neg ratio:** 1:{configs[0].get('neg_ratio', 3)}")
    lines.append(f"- **Gene categories:** {configs[0].get('gene_categories', 'gwas')}")
    lines.append(f"- **Max drug per gene:** {configs[0].get('max_drug_per_gene', 5)}")
    lines.append(f"- **Genes connected:** {configs[0].get('gwas_genes_connected', '?')}")
    lines.append(f"- **Hub genes excluded:** {configs[0].get('gwas_genes_excluded_hubs', '?')}")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Written to {output_path}")


def write_csv(ranked: list[dict], output_path: Path):
    """Write the final predictions as a CSV."""
    import csv

    fields = [
        "final_rank", "drug_id", "name", "mean_score", "std_score",
        "mean_rank", "n_seeds", "is_rct", "gat_rank", "gat_score",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)

    print(f"  Written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Long COVID SEAL results across seeds"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Specific seeds to include (default: auto-detect)"
    )
    parser.add_argument(
        "--include-gat", action="store_true", default=False,
        help="Include GAT comparison in output"
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of top predictions in the table (default: 50)"
    )
    parser.add_argument(
        "--min-seeds", type=int, default=1,
        help="Minimum number of seeds a drug must appear in to be included (default: 1). "
             "E.g. --min-seeds 3 with 5 seeds gives a 3/5 consensus filter."
    )
    parser.add_argument(
        "--conv", type=str, default="sage",
        help="Conv type to aggregate (default: sage)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AGGREGATE LONG COVID SEAL RESULTS")
    print("=" * 60)

    # Find matching SEAL result files
    files = find_result_files(
        conv_type=args.conv,
        gene_categories="gwas",
        seeds=args.seeds,
        use_jk=True if args.conv == "sage" else None,
    )

    if not files:
        print("\n  ERROR: No matching result files found!")
        print("  Looking for: conv=sage, gene_categories=gwas, use_jk=True, max_drug_per_gene=set")
        print(f"  In directory: {RESULTS_DIR}")
        sys.exit(1)

    print(f"\n  Found {len(files)} result files:")
    for f in files:
        data = json.load(open(f))
        seed = data["config"]["seed"]
        auc = data["config"].get("best_val_auc", 0)
        print(f"    {f.name}  (seed={seed}, AUC={auc:.4f})")

    # Aggregate
    print("\n  Aggregating scores across seeds...")
    drug_data, configs = aggregate_scores(files)
    print(f"  {len(drug_data)} unique drugs across {len(files)} runs")

    # Load supporting data
    chembl_cache = load_chembl_cache()
    rct_drugs = load_rct_drugs()
    print(f"  ChEMBL cache: {len(chembl_cache)} entries")
    print(f"  RCT drugs: {len(rct_drugs)} entries")

    # Load GAT results if requested
    gat_data = None
    if args.include_gat:
        print("\n  Loading GAT results for comparison...")
        gat_data = find_gat_results()
        if gat_data:
            print(f"  GAT: {len(gat_data)} drugs ranked")
        else:
            print("  WARNING: No GAT results found — run run_lc_gat_comparison.sh first")

    # Build final table
    ranked = build_final_table(drug_data, configs, chembl_cache, rct_drugs, gat_data)

    # Apply consensus filter
    if args.min_seeds > 1:
        before = len(ranked)
        ranked = [e for e in ranked if e["n_seeds"] >= args.min_seeds]
        # Re-rank after filtering
        for i, entry in enumerate(ranked, 1):
            entry["final_rank"] = i
        print(f"\n  Consensus filter: kept {len(ranked)}/{before} drugs (≥{args.min_seeds}/{len(set(c['seed'] for c in configs))} seeds)")

    # Write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = RESULTS_DIR / "FINAL_PREDICTIONS.md"
    write_markdown(ranked, configs, gat_data, drug_data, md_path, top_n=args.top_n)

    csv_path = RESULTS_DIR / f"final_predictions_{timestamp}.csv"
    write_csv(ranked, csv_path)

    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    seeds = sorted(set(c["seed"] for c in configs))
    print(f"  Seeds: {seeds}")
    if args.min_seeds > 1:
        print(f"  Consensus: ≥{args.min_seeds}/{len(seeds)} seeds")
    print(f"  Drugs ranked: {len(ranked)}")
    print(f"  Top drug: {ranked[0]['name']} ({ranked[0]['drug_id']}) "
          f"score={ranked[0]['mean_score']:.4f}")

    rct_in_table = [e for e in ranked if e["is_rct"]]
    rct_in_top50 = [e for e in rct_in_table if e["final_rank"] <= 50]
    print(f"  RCT drugs found: {len(rct_in_table)}/{len(rct_drugs)}")
    print(f"  RCT drugs in top 50: {len(rct_in_top50)}")
    if rct_in_table:
        rct_ranks = [e["final_rank"] for e in rct_in_table]
        print(f"  RCT median rank: {np.median(rct_ranks):.0f}")

    if gat_data:
        seal_top50 = {e["drug_id"] for e in ranked[:50]}
        gat_top50 = {did for did, info in gat_data.items() if info["rank"] <= 50}
        overlap = seal_top50 & gat_top50
        print(f"  SEAL-GAT top-50 overlap: {len(overlap)} drugs")


if __name__ == "__main__":
    main()

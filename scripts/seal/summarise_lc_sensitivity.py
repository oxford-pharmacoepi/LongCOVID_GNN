#!/usr/bin/env python3
"""Summarise Long COVID sensitivity runs from a manifest TSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDICINES_FILE = PROJECT_ROOT / "medicines_tested.txt"


def load_trial_drugs() -> dict[str, str]:
    trial_drugs: dict[str, str] = {}
    with open(MEDICINES_FILE) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("drug_id"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                trial_drugs[parts[0]] = parts[1]
    return trial_drugs


def load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def summarise_run(entry: dict[str, str], trial_drugs: dict[str, str]) -> dict[str, object]:
    data = json.load(open(entry["json_path"]))
    cfg = data.get("config", {})
    rct_ranks = data.get("rct_drug_ranks", {})

    ranked: list[tuple[str, str, int]] = []
    missing: list[tuple[str, str]] = []
    for drug_id, drug_name in trial_drugs.items():
        info = rct_ranks.get(drug_id)
        if info is None or info.get("rank") is None:
            missing.append((drug_id, drug_name))
            continue
        ranked.append((drug_id, drug_name, int(info["rank"])))

    ranked.sort(key=lambda item: item[2])
    ranks = [rank for _, _, rank in ranked]

    return {
        "label": entry["label"],
        "seed": cfg.get("seed"),
        "gene_categories": cfg.get("gene_categories"),
        "max_drug_per_gene": cfg.get("max_drug_per_gene"),
        "genes_connected": cfg.get("gwas_genes_connected"),
        "genes_excluded_hubs": cfg.get("gwas_genes_excluded_hubs"),
        "val_auc": cfg.get("best_val_auc"),
        "ranked_count": len(ranked),
        "missing_count": len(missing),
        "median_rank": ranks[len(ranks) // 2] if ranks else None,
        "top20": sum(rank <= 20 for rank in ranks),
        "top50": sum(rank <= 50 for rank in ranks),
        "top100": sum(rank <= 100 for rank in ranks),
        "best_ranked": ranked[:8],
        "json_path": entry["json_path"],
        "log_path": entry["log_path"],
    }


def write_markdown(output: Path, rows: list[dict[str, object]]) -> None:
    by_label = {row["label"]: row for row in rows}
    gene_rows = [
        by_label[label]
        for label in ["gene_narrow", "gene_broad", "gene_full"]
        if label in by_label
    ]
    hub_rows = [
        by_label[label]
        for label in ["hub_cap3", "hub_cap5", "hub_cap10", "hub_none"]
        if label in by_label
    ]

    lines: list[str] = []
    lines.append("# Long COVID Method/Input Sensitivity Summary")
    lines.append("")
    if rows:
        lines.append(f"Seed: {rows[0]['seed']}")
        lines.append("")

    lines.append("## Gene-Configuration Comparison")
    lines.append("")
    lines.append("| Run | Gene categories | Genes | Hub genes excluded | Val AUC | Ranked trial drugs | Median rank | Top 20 | Top 50 | Top 100 |")
    lines.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in gene_rows:
        lines.append(
            "| {label} | {gene_categories} | {genes_connected} | {genes_excluded_hubs} | {val_auc:.4f} | {ranked_count}/26 | {median_rank} | {top20}/26 | {top50}/26 | {top100}/26 |".format(
                **row
            )
        )
    lines.append("")

    lines.append("## Hub-Threshold Sensitivity")
    lines.append("")
    lines.append("| Run | Max drug per gene | Genes | Hub genes excluded | Val AUC | Ranked trial drugs | Median rank | Top 20 | Top 50 | Top 100 |")
    lines.append("|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in hub_rows:
        max_label = row["max_drug_per_gene"] if row["max_drug_per_gene"] is not None else "none"
        lines.append(
            "| {label} | {max_label} | {genes_connected} | {genes_excluded_hubs} | {val_auc:.4f} | {ranked_count}/26 | {median_rank} | {top20}/26 | {top50}/26 | {top100}/26 |".format(
                max_label=max_label,
                **row,
            )
        )
    lines.append("")

    lines.append("## Top Ranked Trial Drugs")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['label']}")
        lines.append("")
        lines.append(f"- JSON: {row['json_path']}")
        lines.append(f"- Log: {row['log_path']}")
        top_list = row["best_ranked"]
        if top_list:
            lines.append("- Best-ranked trial drugs:")
            for _, name, rank in top_list:
                lines.append(f"  - {name}: rank {rank}")
        else:
            lines.append("- No rankable trial drugs found.")
        lines.append("")

    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise Long COVID sensitivity runs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trial_drugs = load_trial_drugs()
    manifest_rows = load_manifest(args.manifest)
    summaries = [summarise_run(entry, trial_drugs) for entry in manifest_rows]
    write_markdown(args.output, summaries)
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()

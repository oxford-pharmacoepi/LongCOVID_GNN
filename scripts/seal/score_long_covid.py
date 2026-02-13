#!/usr/bin/env python3
"""
Score Drugs for Long COVID using SEAL

Uses a trained global SEAL model to rank all drugs in the graph
for discovery of Long COVID repurposing candidates.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models_seal import SEALDataset, SEALModel


def score_long_covid(
    target_disease: str = "MONDO_0100320",
    model_path: str = "results/seal_global_model.pt",
    top_k: int = 50,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Score all drugs for *target_disease* using a trained SEAL model."""

    # Resolve model path (try v3, v2, base)
    if not os.path.exists(model_path):
        for suffix in ["_v3.pt", "_v2.pt"]:
            alt = model_path.replace(".pt", suffix)
            if os.path.exists(alt):
                model_path = alt
                break
        else:
            raise FileNotFoundError(f"Trained model not found at {model_path}")

    # ── 1. Load model ───────────────────────────────────────────────────
    print(f"Loading SEAL model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=False)
    cfg = checkpoint["model_config"]
    num_hops = checkpoint.get("num_hops", 2)

    model = SEALModel(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        pooling=cfg.get("pooling", "sort"),
        k=cfg.get("k", 30),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── 2. Load graph and mappings ──────────────────────────────────────
    print("Finding latest graph...")
    graph_files = sorted(glob.glob("results/graph_*_processed_*.pt"))
    graph_path = graph_files[-1]
    graph_data = torch.load(graph_path, weights_only=False)

    mappings_path = graph_path.replace(".pt", "_mappings")
    with open(f"{mappings_path}/drug_key_mapping.json") as f:
        drug_mapping = json.load(f)
    with open(f"{mappings_path}/disease_key_mapping.json") as f:
        disease_mapping = json.load(f)

    target_idx = disease_mapping.get(target_disease)
    if target_idx is None:
        raise ValueError(f"Disease {target_disease} not found in graph mapping")
    target_idx = int(target_idx)

    # ── 3. Score all drugs ──────────────────────────────────────────────
    all_drug_ids = list(drug_mapping.keys())
    all_drug_indices = [int(v) for v in drug_mapping.values()]
    eval_pairs = [(idx, target_idx) for idx in all_drug_indices]

    print(f"Preparing to score {len(eval_pairs)} drugs against {target_disease}...")
    dataset = SEALDataset(
        root="results/seal_tmp",
        pairs=eval_pairs,
        labels=[0] * len(eval_pairs),
        edge_index=graph_data.edge_index,
        node_features=graph_data.x,
        num_hops=num_hops,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Scoring candidates...")
    scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            if batch.x.shape[0] == 0:
                scores.extend([0.0] * batch.y.shape[0])
                continue
            out = torch.sigmoid(model(batch))
            scores.extend(out.tolist())

    # ── 4. Build results ────────────────────────────────────────────────
    results_df = pd.DataFrame({
        "drug_id": all_drug_ids[:len(scores)],
        "score": scores,
        "probability": scores,
    })
    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1
    results_df["drug_name"] = results_df["drug_id"]
    results_df["confidence"] = results_df["probability"].apply(
        lambda p: "High" if p > 0.7 else "Medium" if p > 0.5 else "Low"
    )
    results_df = results_df[["rank", "drug_id", "drug_name", "probability", "score", "confidence"]]

    # ── 5. Save ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/long_covid", exist_ok=True)
    save_csv = f"results/long_covid/long_covid_seal_predictions_{timestamp}.csv"
    latest_csv = "results/long_covid/long_covid_seal_latest.csv"
    results_df.to_csv(save_csv, index=False)
    results_df.to_csv(latest_csv, index=False)

    print(f"\nTop {top_k} Candidates for Long COVID:")
    print("=" * 60)
    print(results_df.head(top_k).to_string(index=False))
    print(f"\nFull results saved to {save_csv}")
    print(f"Latest results available at {latest_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank drugs for Long COVID using SEAL")
    parser.add_argument("--disease", type=str, default="MONDO_0100320", help="Target disease ID")
    parser.add_argument("--model", type=str, default="results/seal_global_model.pt")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    score_long_covid(
        target_disease=args.disease,
        model_path=args.model,
        top_k=args.top_k,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

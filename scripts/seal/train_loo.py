#!/usr/bin/env python3
"""
SEAL Leave-One-Out Validation for Drug-Disease Link Prediction

Trains a SEAL model on all drug-disease edges *except* those involving
a held-out target disease, then ranks all drugs for that disease.
This mirrors the Global GNN LOO workflow in ``leave_one_out_validation.py``
but uses subgraph-based classification instead of full-graph embeddings.
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.models_seal import SEALDataset, SEALModel


def train_seal_leave_one_out(
    target_disease: str = "EFO_0003854",
    num_hops: int = 2,
    hidden_channels: int = 32,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    num_workers: int = 0,
    limit: Optional[int] = None,
):
    """Train SEAL with leave-one-out validation for *target_disease*."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── 1. Load graph and mappings ──────────────────────────────────────
    print("Finding graph...")
    graph_files = sorted(glob.glob("results/graph_*_processed_*.pt"))
    if not graph_files:
        raise FileNotFoundError("No processed graph found in results/")
    graph_path = graph_files[-1]
    print(f"  Using: {graph_path}")

    graph_data = torch.load(graph_path, weights_only=False)
    edge_index = graph_data.edge_index
    node_features = graph_data.x

    mappings_path = graph_path.replace(".pt", "_mappings")
    with open(f"{mappings_path}/drug_key_mapping.json") as f:
        drug_mapping = {k: int(v) for k, v in json.load(f).items()}
    with open(f"{mappings_path}/disease_key_mapping.json") as f:
        disease_mapping = {k: int(v) for k, v in json.load(f).items()}

    idx_to_drug = {v: k for k, v in drug_mapping.items()}

    target_idx = disease_mapping.get(target_disease)
    if target_idx is None:
        raise ValueError(f"Disease {target_disease} not found in mapping")
    print(f"\nTarget disease: {target_disease} (idx={target_idx})")

    # ── 2. Extract drug-disease edges ───────────────────────────────────
    drug_indices = set(drug_mapping.values())
    disease_indices = set(disease_mapping.values())

    print("Extracting drug-disease edges...")
    drug_disease_edges = set()
    src, dst = edge_index
    for i in range(edge_index.shape[1]):
        u, v = src[i].item(), dst[i].item()
        if u in drug_indices and v in disease_indices:
            drug_disease_edges.add((u, v))
        elif v in drug_indices and u in disease_indices:
            drug_disease_edges.add((v, u))

    drug_disease_edges = list(drug_disease_edges)
    print(f"  Found {len(drug_disease_edges)} drug-disease edges")

    # Split: edges for target disease are test; everything else is train
    test_edges = [e for e in drug_disease_edges if e[1] == target_idx]
    train_edges = [e for e in drug_disease_edges if e[1] != target_idx]

    if limit and limit < len(train_edges):
        print(f"  Limiting training to {limit} positive edges (sampled from {len(train_edges)})...")
        train_edges = random.sample(train_edges, limit)

    print(f"  Train edges: {len(train_edges):,}")
    print(f"  Test edges:  {len(test_edges):,}")

    # ── 3. Negative sampling ────────────────────────────────────────────
    print("Sampling negative edges...")
    all_drug_list = list(drug_indices)
    all_disease_list = list(disease_indices)
    positive_set = set(drug_disease_edges)
    neg_train_edges = []

    while len(neg_train_edges) < len(train_edges):
        drug = random.choice(all_drug_list)
        disease = random.choice(all_disease_list)
        if (drug, disease) not in positive_set and disease != target_idx:
            neg_train_edges.append((drug, disease))
            positive_set.add((drug, disease))

    print(f"  Sampled {len(neg_train_edges)} negative edges")

    # ── 3b. Create training-only edge index ──
    print("\nCreating training-only edge index for subgraph extraction...")
    src, dst = edge_index
    
    # Efficiently identify drug nodes
    is_drug_node = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
    is_drug_node[list(drug_indices)] = True
    
    # We only want to remove DRUG-DISEASE edges involving the target disease.
    leakage_mask = ((is_drug_node[src]) & (dst == target_idx)) | \
                   ((is_drug_node[dst]) & (src == target_idx))
                   
    train_edge_index = edge_index[:, ~leakage_mask]
    
    print(f"  Full graph edges: {edge_index.shape[1]:,}")
    print(f"  Target disease drug-links removed: {leakage_mask.sum().item():,}")
    print(f"  Edges kept for context (incl. disease-gene): {train_edge_index.shape[1]:,}")

    # ── 4. Training dataset ─────────────────────────────────────────────
    print("\nInitialising SEAL training dataset...")
    train_pairs = train_edges + neg_train_edges
    train_labels = [1] * len(train_edges) + [0] * len(neg_train_edges)

    train_dataset = SEALDataset(
        root="results/seal_tmp",
        pairs=train_pairs,
        labels=train_labels,
        edge_index=train_edge_index,  # Use training-only edges!
        node_features=node_features,
        num_hops=num_hops,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    # ── 5. Model setup ──────────────────────────────────────────────────
    first_batch = next(iter(train_loader))
    in_channels = first_batch.x.shape[1]

    model = SEALModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout_rate=0.5,
        pooling="sort",
        k=30,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # ── 6. Training loop ────────────────────────────────────────────────
    print(f"\nTraining SEAL model for {epochs} epochs...", flush=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            optimiser.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}", flush=True)

    # ── 7. Ranking evaluation ───────────────────────────────────────────
    print(f"\nEvaluating: ranking all {len(all_drug_list)} drugs for {target_disease}...",
          flush=True)
    model.eval()

    eval_pairs = [(d, target_idx) for d in all_drug_list]
    eval_dataset = SEALDataset(
        root="results/seal_tmp",
        pairs=eval_pairs,
        labels=[0] * len(eval_pairs),
        edge_index=train_edge_index,  # Use training-only edges for fair evaluation
        node_features=node_features,
        num_hops=num_hops,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    scores = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Scoring drugs", leave=True):
            out = model(batch)
            scores.extend(torch.sigmoid(out).tolist())

    drug_scores = [(all_drug_list[i], s) for i, s in enumerate(scores) if i < len(all_drug_list)]
    drug_scores.sort(key=lambda x: x[1], reverse=True)

    # ── 8. Metrics ──────────────────────────────────────────────────────
    true_drugs = {d for d, _ in test_edges}
    rank_map = {d: r for r, (d, _) in enumerate(drug_scores, 1)}
    ranks = [rank_map[d] for d in true_drugs if d in rank_map]

    hits_at_10 = sum(1 for r in ranks if r <= 10)
    hits_at_20 = sum(1 for r in ranks if r <= 20)
    hits_at_50 = sum(1 for r in ranks if r <= 50)
    median_rank = float(torch.tensor(ranks, dtype=torch.float).median()) if ranks else 9999.0

    print(f"\nTop 20 Predictions:")
    print(f"{'Rank':<6} {'Drug ID':<20} {'Score':<10} {'True?'}")
    print("-" * 60)
    for rank, (drug_idx, score) in enumerate(drug_scores[:20], 1):
        drug_id = idx_to_drug.get(drug_idx, "Unknown")
        mark = "✓ True" if drug_idx in true_drugs else ""
        print(f"{rank:<6} {drug_id:<20} {score:<10.4f} {mark}")

    n = len(true_drugs)
    print(f"\n{'=' * 60}")
    print(f"SEAL PERFORMANCE SUMMARY FOR {target_disease}")
    print(f"{'=' * 60}")
    print(f"  Test Edges (True Positives): {n}")
    if n > 0:
        print(f"  Hits@10: {hits_at_10} / {n} ({hits_at_10 / n * 100:.1f}%)")
        print(f"  Hits@20: {hits_at_20} / {n} ({hits_at_20 / n * 100:.1f}%)")
        print(f"  Hits@50: {hits_at_50} / {n} ({hits_at_50 / n * 100:.1f}%)")
    else:
        print(f"  Hits@K: No test edges available")
    print(f"  Median Rank: {median_rank:.1f}")
    print(f"{'=' * 60}\n")

    return model, median_rank, hits_at_20


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEAL Leave-One-Out Validation for Drug-Disease Link Prediction"
    )
    parser.add_argument("--target-disease", type=str, default="EFO_0003854",
                        help="Target disease EFO/MONDO ID")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--hops", type=int, default=2, help="Subgraph extraction hops")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit training edges")
    parser.add_argument("--hidden", type=int, default=32, help="Model hidden channels")
    parser.add_argument("--layers", type=int, default=3, help="Number of GNN layers")
    args = parser.parse_args()

    train_seal_leave_one_out(
        target_disease=args.target_disease,
        num_hops=args.hops,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        epochs=args.epochs,
        batch_size=32,
        lr=0.001,
        seed=args.seed,
        num_workers=args.workers,
        limit=args.limit,
    )

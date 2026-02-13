#!/usr/bin/env python3
"""
Global SEAL Training Script

Trains a SEAL model on ALL available drug-disease links using
temporal or full-data splits. Uses SEALDataset for memory-efficient
on-the-fly subgraph extraction.

Features:
    1. Comprehensive training on all drug-disease links
    2. Memory-efficient via SEALDataset (on-the-fly extraction)
    3. Temporal split support (train/val/test from graph object)
    4. Sampled ranking evaluation (Hits@K, MRR)
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.models_seal import SEALDataset, SEALModel
from src.negative_sampling import MixedNegativeSampler


def _to_edge_list(edge_index: torch.Tensor):
    """Convert an edge_index tensor to a list of (src, dst) tuples."""
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        return [tuple(row) for row in edge_index.tolist()]
    elif edge_index.shape[0] == 2:
        return list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    raise ValueError(f"Unexpected edge_index shape: {edge_index.shape}")


def evaluate(model, loader, criterion, extraction_edge_index, node_features,
             train_pos_edges, val_pos_edges, drug_indices, num_hops,
             desc="Evaluating", rank_eval=False):
    """
    Evaluate the model on a given loader.

    If *rank_eval* is True, additionally computes sampled ranking
    metrics (Hits@10, Hits@20, MRR) by ranking each positive edge
    against 100 random negatives.
    """
    model.eval()
    all_probs, all_targets = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if batch.num_nodes == 0:
                continue
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(out).cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    try:
        auc = roc_auc_score(all_targets, all_probs)
        ap = average_precision_score(all_targets, all_probs)
    except Exception:
        auc, ap = 0.0, 0.0

    metrics = {"loss": total_loss / max(len(loader), 1), "auc": auc, "ap": ap}

    # Sampled ranking evaluation
    if rank_eval:
        print("  Calculating sampled ranking metrics (1 positive vs 100 random negatives)...")
        pos_pairs = [p for p, l in zip(loader.dataset.pairs, loader.dataset.labels) if l == 1]
        if len(pos_pairs) > 100:
            pos_pairs = random.sample(pos_pairs, 100)

        hits_at_10, hits_at_20, mrr, total_ranked = 0, 0, 0.0, 0
        pos_set = set(train_pos_edges) | set(val_pos_edges)

        for src, dst in tqdm(pos_pairs, desc="Ranking (sampled)"):
            neg_drugs = []
            while len(neg_drugs) < 100:
                d_idx = random.choice(list(drug_indices))
                if (d_idx, dst) not in pos_set:
                    neg_drugs.append(d_idx)

            rank_dataset = SEALDataset(
                root="results/seal_tmp_rank",
                pairs=[(src, dst)] + [(d, dst) for d in neg_drugs],
                labels=[1] + [0] * 100,
                edge_index=extraction_edge_index,
                node_features=node_features,
                num_hops=num_hops,
            )
            rank_loader = DataLoader(rank_dataset, batch_size=101, num_workers=0)

            with torch.no_grad():
                first_batch = next(iter(rank_loader))
                scores = torch.sigmoid(model(first_batch)).cpu().numpy()

            rank = int(np.sum(scores[1:] >= scores[0])) + 1
            if rank <= 10:
                hits_at_10 += 1
            if rank <= 20:
                hits_at_20 += 1
            mrr += 1.0 / rank
            total_ranked += 1

        if total_ranked > 0:
            metrics["hits@10"] = hits_at_10 / total_ranked
            metrics["hits@20"] = hits_at_20 / total_ranked
            metrics["mrr"] = mrr / total_ranked

    return metrics


def train_seal_global(
    num_hops: int = 2,
    hidden_channels: int = 32,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    exclude_disease: str = "MONDO_0100320",
    num_workers: int = 0,
    limit: int = 0,
    split: str = "temporal",
    val_ratio: int = 10,
    test_ratio: int = 10,
):
    """
    Train a global SEAL model.

    Args:
        split: 'temporal' uses standard 2021/2023/2024 splits.
               'all' uses ALL data (minus *exclude_disease*) for a final model.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 1. Load graph and mappings ──────────────────────────────────────
    print("Finding latest graph...")
    graph_files = sorted(glob.glob("results/graph_*_processed_*.pt"))
    if not graph_files:
        raise FileNotFoundError("No processed graph found in results/")
    graph_path = graph_files[-1]
    print(f"  Using: {graph_path}")

    graph_data = torch.load(graph_path, weights_only=False)
    full_edge_index = graph_data.edge_index
    node_features = graph_data.x

    mappings_path = graph_path.replace(".pt", "_mappings")
    with open(f"{mappings_path}/drug_key_mapping.json") as f:
        drug_mapping = json.load(f)
    with open(f"{mappings_path}/disease_key_mapping.json") as f:
        disease_mapping = json.load(f)

    drug_indices = {int(v) for v in drug_mapping.values()}
    disease_indices = {int(v) for v in disease_mapping.values()}

    # ── 2. Define positive edges based on split ─────────────────────────
    train_pos_edges, val_pos_edges, test_pos_edges = [], [], []

    if split == "temporal":
        print("Using TEMPORAL splits (2021 / 2023 / 2024)...")
        if not hasattr(graph_data, "train_edge_index"):
            raise ValueError("Graph has no temporal splits. Use split='all'.")

        train_pos_edges = _to_edge_list(graph_data.train_edge_index)
        val_pos_edges = _to_edge_list(graph_data.val_edge_index)
        test_pos_edges = _to_edge_list(graph_data.test_edge_index)
        print(f"  Train edges: {len(train_pos_edges):,}")
        print(f"  Val edges:   {len(val_pos_edges):,}")
        print(f"  Test edges:  {len(test_pos_edges):,}")

        # Use training edges only for subgraph extraction (avoid leakage)
        tei = graph_data.train_edge_index
        if tei.shape[0] != 2 and tei.shape[1] == 2:
            print("  Transposing train_edge_index ([E, 2] → [2, E])")
            extraction_edge_index = tei.t()
        else:
            extraction_edge_index = tei

    else:
        print("Using ALL data (repurposing mode)...")
        target_idx = disease_mapping.get(exclude_disease)
        if target_idx is not None:
            target_idx = int(target_idx)
            print(f"  Excluding links for {exclude_disease} (idx={target_idx})")

        src, dst = full_edge_index
        for i in range(full_edge_index.shape[1]):
            u, v = src[i].item(), dst[i].item()
            is_drug_disease = (u in drug_indices and v in disease_indices) or \
                              (v in drug_indices and u in disease_indices)
            if is_drug_disease and u != target_idx and v != target_idx:
                train_pos_edges.append((u, v))
        train_pos_edges = list(set(train_pos_edges))
        extraction_edge_index = full_edge_index

    # Debug limit
    if limit > 0:
        print(f"  DEBUG: Limiting train edges to {limit}")
        train_pos_edges = train_pos_edges[:limit]
        val_pos_edges = val_pos_edges[:limit // 10] if val_pos_edges else []
        test_pos_edges = test_pos_edges[:limit // 10] if test_pos_edges else []

    # ── 3. Negative sampling ────────────────────────────────────────────
    print(f"\nPreparing training data ({len(train_pos_edges)} positives)...")
    print("Generating negative samples for training...")
    pool = [
        (random.choice(list(drug_indices)), random.choice(list(disease_indices)))
        for _ in range(len(train_pos_edges) * 5)
    ]

    sampler = MixedNegativeSampler(
        strategy_weights={"hard": 0.6, "random": 0.4},
        seed=seed,
        min_cn_threshold=1,
        max_cn_threshold=10,
    )
    train_neg_edges = sampler.sample(
        positive_edges=set(train_pos_edges),
        all_possible_pairs=pool,
        num_samples=len(train_pos_edges),
        edge_index=extraction_edge_index,
    )

    train_pairs = train_pos_edges + train_neg_edges
    train_labels = [1] * len(train_pos_edges) + [0] * len(train_neg_edges)

    train_dataset = SEALDataset(
        root="results/seal_tmp",
        pairs=train_pairs,
        labels=train_labels,
        edge_index=extraction_edge_index,
        node_features=node_features,
        num_hops=num_hops,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    # ── 4. Model setup ──────────────────────────────────────────────────
    try:
        first_batch = next(iter(train_loader))
        in_channels = first_batch.x.shape[1]
    except Exception:
        in_channels = 100 + (node_features.shape[1] if node_features is not None else 0)

    model = SEALModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        pooling="sort",
        k=30,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # ── 5. Validation loader ────────────────────────────────────────────
    val_loader = None
    if split == "temporal" and val_pos_edges:
        num_val_neg = len(val_pos_edges) * val_ratio
        print(f"\nPreparing validation data ({len(val_pos_edges)} positives)...")
        print(f"  Sampling {num_val_neg} random negatives (1:{val_ratio})...")
        val_neg_edges = []
        pos_set = set(train_pos_edges) | set(val_pos_edges)
        while len(val_neg_edges) < num_val_neg:
            u = random.choice(list(drug_indices))
            v = random.choice(list(disease_indices))
            if (u, v) not in pos_set:
                val_neg_edges.append((u, v))

        val_dataset = SEALDataset(
            root="results/seal_tmp_val",
            pairs=val_pos_edges + val_neg_edges,
            labels=[1] * len(val_pos_edges) + [0] * len(val_neg_edges),
            edge_index=extraction_edge_index,
            node_features=node_features,
            num_hops=num_hops,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    # ── 6. Training loop ────────────────────────────────────────────────
    config = Config()
    primary_metric = getattr(config, "primary_metric", "auc").lower()
    is_ranking = "hits" in primary_metric or "mrr" in primary_metric

    print(f"\nTraining global SEAL classifier ({split} mode) for {epochs} epochs...")
    print(f"Optimising for: {primary_metric}")

    best_val_score = -1.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in pbar:
            if batch.num_nodes == 0:
                continue
            optimiser.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / len(train_loader):.4f}"})

        # Validation
        if val_loader:
            val_metrics = evaluate(
                model, val_loader, criterion, extraction_edge_index,
                node_features, train_pos_edges, val_pos_edges, drug_indices,
                num_hops, desc=f"Epoch {epoch + 1} [Val]", rank_eval=is_ranking,
            )

            metric_key = primary_metric.replace("_at_", "@")
            current_score = val_metrics.get(metric_key, val_metrics.get("auc", 0))
            val_str = f" | Val {primary_metric.upper()}: {current_score:.4f}"
            if "auc" in val_metrics and primary_metric != "auc":
                val_str += f" | Val AUC: {val_metrics['auc']:.4f}"

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} Summary: Train Loss: {avg_loss:.4f}"
                  f" | Val Loss: {val_metrics['loss']:.4f}{val_str}")

            if current_score >= best_val_score:
                best_val_score = current_score
                print(f"  ✓ New best {primary_metric}! Saving model...")
                torch.save(model.state_dict(), "results/seal_global_model_best.pt")

        # Checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": in_channels,
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
                "pooling": "sort",
                "k": 30,
            },
            "num_hops": num_hops,
            "epoch": epoch + 1,
            "split": split,
        }
        torch.save(checkpoint, "results/seal_global_model_latest.pt")

    # ── 7. Final model save ─────────────────────────────────────────────
    save_path = "results/seal_global_model_v3.pt"
    torch.save(checkpoint, save_path)
    print(f"\nModel saved to {save_path}")

    # ── 8. Test evaluation ──────────────────────────────────────────────
    if split == "temporal" and test_pos_edges:
        print("\nLoading best model for test evaluation...")
        best_path = "results/seal_global_model_best.pt"
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=True))

        num_test_neg = len(test_pos_edges) * test_ratio
        print(f"Preparing test data ({len(test_pos_edges)} positives)...")
        print(f"  Sampling {num_test_neg} random negatives (1:{test_ratio})...")
        test_neg_edges = []
        all_pos = set(train_pos_edges) | set(val_pos_edges) | set(test_pos_edges)
        while len(test_neg_edges) < num_test_neg:
            u = random.choice(list(drug_indices))
            v = random.choice(list(disease_indices))
            if (u, v) not in all_pos:
                test_neg_edges.append((u, v))

        # For test evaluation, use train + val edges for subgraph extraction
        # This simulates real-world deployment where all historical data is available
        print("  Creating train+val edge index for test subgraph extraction...")
        train_val_edge_index = torch.cat([
            graph_data.train_edge_index if graph_data.train_edge_index.shape[0] == 2 
            else graph_data.train_edge_index.t(),
            graph_data.val_edge_index if graph_data.val_edge_index.shape[0] == 2 
            else graph_data.val_edge_index.t()
        ], dim=1)
        print(f"  Train edges: {graph_data.train_edge_index.shape[1]}")
        print(f"  Val edges: {graph_data.val_edge_index.shape[1]}")
        print(f"  Train+Val edges for test: {train_val_edge_index.shape[1]}")

        test_dataset = SEALDataset(
            root="results/seal_tmp_test",
            pairs=test_pos_edges + test_neg_edges,
            labels=[1] * len(test_pos_edges) + [0] * len(test_neg_edges),
            edge_index=train_val_edge_index,  # Use train+val for realistic test setting
            node_features=node_features,
            num_hops=num_hops,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        test_metrics = evaluate(
            model, test_loader, criterion, train_val_edge_index,
            node_features, train_pos_edges, val_pos_edges, drug_indices,
            num_hops, desc="Testing", rank_eval=True,
        )
        print("\nFinal Test Results:")
        print(f"  AUC:     {test_metrics['auc']:.4f}")
        print(f"  AP:      {test_metrics['ap']:.4f}")
        if "hits@10" in test_metrics:
            print(f"  Hits@10: {test_metrics['hits@10']:.4f}")
            print(f"  Hits@20: {test_metrics['hits@20']:.4f}")
            print(f"  MRR:     {test_metrics['mrr']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global SEAL Training")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--hops", type=int, default=2, help="Extraction hops")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0=serial)")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for edges (0=all)")
    parser.add_argument("--split", type=str, default="temporal", choices=["temporal", "all"])
    parser.add_argument("--val-ratio", type=int, default=10, help="Neg:pos ratio for validation")
    parser.add_argument("--test-ratio", type=int, default=10, help="Neg:pos ratio for test")
    args = parser.parse_args()

    train_seal_global(
        epochs=args.epochs,
        num_hops=args.hops,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.workers,
        limit=args.limit,
        split=args.split,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

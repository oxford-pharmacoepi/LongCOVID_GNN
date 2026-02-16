#!/usr/bin/env python3
"""
SEAL Leave-One-Out Validation for Drug-Disease Link Prediction

Trains a SEAL model on all drug-disease edges *except* those involving
a held-out target disease, then ranks all drugs for that disease.
This mirrors the Global GNN LOO workflow in ``leave_one_out_validation.py``
but uses subgraph-based classification instead of full-graph embeddings.
"""

import argparse
from collections import defaultdict
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models_seal import SEALDataset, SEALModel, MAX_Z


# ═══════════════════════════════════════════════════════════════════════════
# Negative sampling strategies
# ═══════════════════════════════════════════════════════════════════════════

def _build_adjacency(edge_index: torch.Tensor) -> dict:
    """Build adjacency set from edge_index for fast common-neighbour lookup."""
    adj = defaultdict(set)
    src, dst = edge_index
    for i in range(edge_index.shape[1]):
        u, v = src[i].item(), dst[i].item()
        adj[u].add(v)
        adj[v].add(u)
    return adj


def sample_negatives_random(
    positive_set: set,
    all_drug_list: list,
    all_disease_list: list,
    num_samples: int,
    exclude_disease: int = None,
    future_positives: set = None,
) -> list:
    """Uniform random negative sampling."""
    future_positives = future_positives or set()
    neg_edges = []
    forbidden = positive_set | future_positives
    attempts = 0
    max_attempts = num_samples * 20

    while len(neg_edges) < num_samples and attempts < max_attempts:
        drug = random.choice(all_drug_list)
        disease = random.choice(all_disease_list)
        if disease == exclude_disease:
            attempts += 1
            continue
        if (drug, disease) not in forbidden:
            neg_edges.append((drug, disease))
            forbidden.add((drug, disease))
        attempts += 1

    return neg_edges


def sample_negatives_hard(
    positive_set: set,
    all_drug_list: list,
    all_disease_list: list,
    num_samples: int,
    edge_index: torch.Tensor,
    exclude_disease: int = None,
    future_positives: set = None,
    min_cn: int = 1,
    max_cn: int = None,
) -> list:
    """Hard negative sampling: select pairs with high common-neighbour count.
    
    Pairs with many common neighbours but no direct edge are the hardest
    to classify and produce the most informative gradients.
    """
    future_positives = future_positives or set()
    forbidden = positive_set | future_positives
    adj = _build_adjacency(edge_index)

    drug_set = set(all_drug_list)
    disease_set = set(all_disease_list)

    # Score all candidate negative pairs by common-neighbour count
    print("    Computing common neighbours for hard negatives...")
    candidates = []
    for drug in tqdm(all_drug_list, desc="    Hard neg scan", leave=False):
        drug_nb = adj.get(drug, set())
        for disease in all_disease_list:
            if disease == exclude_disease:
                continue
            if (drug, disease) in forbidden:
                continue
            cn = len(drug_nb & adj.get(disease, set()))
            if cn >= min_cn:
                if max_cn is None or cn <= max_cn:
                    candidates.append((drug, disease, cn))

    # Sort by CN descending (hardest first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    print(f"    Found {len(candidates)} hard negative candidates (CN >= {min_cn})")

    hard = [(d, dis) for d, dis, _ in candidates[:num_samples]]

    # If not enough hard negatives, pad with random
    if len(hard) < num_samples:
        shortfall = num_samples - len(hard)
        print(f"    Padding with {shortfall} random negatives (not enough hard)")
        hard_set = set(hard)
        extras = sample_negatives_random(
            forbidden | hard_set, all_drug_list, all_disease_list,
            shortfall, exclude_disease, future_positives,
        )
        hard.extend(extras)

    return hard


def sample_negatives_mixed(
    positive_set: set,
    all_drug_list: list,
    all_disease_list: list,
    num_samples: int,
    edge_index: torch.Tensor,
    exclude_disease: int = None,
    future_positives: set = None,
    hard_ratio: float = 0.5,
) -> list:
    """Mixed sampling: a proportion of hard negatives, rest random."""
    n_hard = int(num_samples * hard_ratio)
    n_random = num_samples - n_hard

    hard = sample_negatives_hard(
        positive_set, all_drug_list, all_disease_list, n_hard,
        edge_index, exclude_disease, future_positives,
    )

    hard_set = set(hard)
    rand = sample_negatives_random(
        positive_set | hard_set, all_drug_list, all_disease_list,
        n_random, exclude_disease, future_positives,
    )

    combined = hard + rand
    random.shuffle(combined)
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Training function
# ═══════════════════════════════════════════════════════════════════════════

def train_seal_leave_one_out(
    target_disease: str = "EFO_0003854",
    num_hops: int = 1,
    hidden_channels: int = 32,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    seed: int = 42,
    num_workers: int = 0,
    cache_root: str = "results/seal_tmp",
    neg_strategy: str = "mixed",
    neg_ratio: int = 3,
    hard_ratio: float = 0.5,
    pooling: str = "mean+max",
    sort_k: int = 30,
    conv_type: str = "sage",
    max_z: int = MAX_Z,
    dropout: float = 0.5,
    max_nodes_per_hop: int = 200,
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
    print(f"  Train edges: {len(train_edges)}")
    print(f"  Test edges:  {len(test_edges)}")

    if len(test_edges) == 0:
        print(f"\n  WARNING: No test edges for {target_disease}. "
              "LOO validation will have no ground truth to evaluate against.")

    # ── 3. Negative sampling ────────────────────────────────────────────
    all_drug_list = list(drug_indices)
    all_disease_list = list(disease_indices)
    positive_set = set(drug_disease_edges)

    # Pollution control: don't sample held-out test edges as negatives
    future_positives = set(test_edges)
    num_neg = len(train_edges) * neg_ratio

    print(f"\nSampling {num_neg} negative edges (strategy={neg_strategy}, ratio=1:{neg_ratio})...")

    if neg_strategy == "hard":
        neg_train_edges = sample_negatives_hard(
            positive_set, all_drug_list, all_disease_list, num_neg,
            edge_index, exclude_disease=target_idx,
            future_positives=future_positives,
        )
    elif neg_strategy == "mixed":
        neg_train_edges = sample_negatives_mixed(
            positive_set, all_drug_list, all_disease_list, num_neg,
            edge_index, exclude_disease=target_idx,
            future_positives=future_positives, hard_ratio=hard_ratio,
        )
    else:  # random
        neg_train_edges = sample_negatives_random(
            positive_set, all_drug_list, all_disease_list, num_neg,
            exclude_disease=target_idx, future_positives=future_positives,
        )

    print(f"  Sampled {len(neg_train_edges)} negative edges")

    # ── 3b. Create training-only edge index (CRITICAL: avoid test leakage) ──
    # Only remove DRUG→target_disease edges (the ones we predict).
    # Keep gene-disease, gene-gene, etc. edges for structural context.
    print("\nCreating training-only edge index for subgraph extraction...")
    src, dst = edge_index
    is_drug = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
    is_drug[list(drug_indices)] = True
    leakage_mask = ((is_drug[src]) & (dst == target_idx)) | \
                   ((is_drug[dst]) & (src == target_idx))
    train_edge_index = edge_index[:, ~leakage_mask]
    print(f"  Full graph edges: {edge_index.shape[1]}")
    print(f"  Drug-disease edges removed for target: {leakage_mask.sum().item()}")
    print(f"  Edges kept (incl. gene-disease context): {train_edge_index.shape[1]}")

    # ── 3c. Build adjacency dict for fast subgraph extraction ──
    print("Building adjacency dict for fast neighbour lookups...")
    adj_dict = defaultdict(set)
    s, d = train_edge_index
    for i in range(train_edge_index.shape[1]):
        u, v = s[i].item(), d[i].item()
        adj_dict[u].add(v)
        adj_dict[v].add(u)
    print(f"  Adjacency built for {len(adj_dict)} nodes")

    # ── 4. Training dataset ─────────────────────────────────────────────
    print("\nInitialising SEAL training dataset...")
    train_pairs = train_edges + neg_train_edges
    train_labels = [1] * len(train_edges) + [0] * len(neg_train_edges)

    train_dataset = SEALDataset(
        root=f"{cache_root}/{target_disease}",
        pairs=train_pairs,
        labels=train_labels,
        edge_index=train_edge_index,
        node_features=node_features,
        num_hops=num_hops,
        max_z=max_z,
        max_nodes_per_hop=max_nodes_per_hop,
        adj_dict=adj_dict,
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
        dropout_rate=dropout,
        pooling=pooling,
        k=sort_k,
        conv_type=conv_type,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr / 100)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"  Conv type: {conv_type}, Layers: {num_layers}, Hidden: {hidden_channels}")
    print(f"  Pooling: {pooling}" + (f" (k={sort_k})" if pooling == "sort" else ""))
    print(f"  Dropout: {dropout}, LR: {lr}, Weight Decay: {weight_decay}")
    print(f"  max_z: {max_z}, Feature dim: {in_channels}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ── 6. Training loop ────────────────────────────────────────────────
    print(f"\nTraining SEAL model for {epochs} epochs...", flush=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimiser.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}", flush=True)

    # ── 7. Ranking evaluation ───────────────────────────────────────────
    print(f"\nEvaluating: ranking all {len(all_drug_list)} drugs for {target_disease}...",
          flush=True)
    model.eval()

    eval_pairs = [(d, target_idx) for d in all_drug_list]
    eval_dataset = SEALDataset(
        root=f"{cache_root}/{target_disease}",
        pairs=eval_pairs,
        labels=[0] * len(eval_pairs),
        edge_index=train_edge_index,
        node_features=node_features,
        num_hops=num_hops,
        max_z=max_z,
        max_nodes_per_hop=max_nodes_per_hop,
        adj_dict=adj_dict,
        use_cache=False,
        save_cache=False,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    all_scores = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Scoring"):
            out = torch.sigmoid(model(batch))
            all_scores.extend(out.tolist())

    drug_scores = [(all_drug_list[i], s) for i, s in enumerate(all_scores) if i < len(all_drug_list)]
    drug_scores.sort(key=lambda x: x[1], reverse=True)

    # ── 8. Metrics ──────────────────────────────────────────────────────
    true_drugs = {d for d, _ in test_edges}
    rank_map = {d: r for r, (d, _) in enumerate(drug_scores, 1)}
    ranks = [rank_map[d] for d in true_drugs if d in rank_map]

    hits_at_10 = sum(1 for r in ranks if r <= 10)
    hits_at_20 = sum(1 for r in ranks if r <= 20)
    hits_at_50 = sum(1 for r in ranks if r <= 50)
    median_rank = float(torch.tensor(ranks, dtype=torch.float).median()) if ranks else 9999.0
    mean_rank = float(torch.tensor(ranks, dtype=torch.float).mean()) if ranks else 9999.0

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
    print(f"  Config: {conv_type} | {pooling} | hidden={hidden_channels} | layers={num_layers}")
    print(f"  Sampling: {neg_strategy} | ratio=1:{neg_ratio}" +
          (f" | hard_ratio={hard_ratio}" if neg_strategy == "mixed" else ""))
    if n > 0:
        print(f"  Hits@10: {hits_at_10} / {n} ({hits_at_10 / n * 100:.1f}%)")
        print(f"  Hits@20: {hits_at_20} / {n} ({hits_at_20 / n * 100:.1f}%)")
        print(f"  Hits@50: {hits_at_50} / {n} ({hits_at_50 / n * 100:.1f}%)")
    else:
        print(f"  Hits@K: No test edges available")
    print(f"  Median Rank: {median_rank:.1f}")
    print(f"  Mean Rank: {mean_rank:.1f}")
    print(f"{'=' * 60}\n")

    return model, median_rank, hits_at_20


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEAL Leave-One-Out Validation for Drug-Disease Link Prediction"
    )
    # Target & graph
    parser.add_argument("--target-disease", type=str, default="EFO_0003854",
                        help="Target disease EFO/MONDO ID")
    parser.add_argument("--cachedir", type=str, default="results/seal_tmp",
                        help="Root directory for SEAL subgraph cache")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")

    # Architecture
    parser.add_argument("--hidden", type=int, default=32, help="Hidden channels")
    parser.add_argument("--layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--hops", type=int, default=1, help="Subgraph extraction hops")
    parser.add_argument("--max-nodes-per-hop", type=int, default=200,
                        help="Max neighbours sampled per hop (caps subgraph size)")
    parser.add_argument("--pooling", type=str, default="mean+max",
                        choices=["sort", "mean", "max", "mean+max"],
                        help="Graph-level pooling method")
    parser.add_argument("--sort-k", type=int, default=30,
                        help="k for SortAggregation (only used if --pooling=sort)")
    parser.add_argument("--conv", type=str, default="sage",
                        choices=["sage", "gcn", "gin"],
                        help="GNN convolution type")
    parser.add_argument("--max-z", type=int, default=MAX_Z,
                        help="Max DRNL label (controls one-hot dimensionality)")

    # Negative sampling
    parser.add_argument("--neg-strategy", type=str, default="mixed",
                        choices=["random", "hard", "mixed"],
                        help="Negative sampling strategy")
    parser.add_argument("--neg-ratio", type=int, default=3,
                        help="Ratio of negatives to positives (e.g. 3 means 1:3)")
    parser.add_argument("--hard-ratio", type=float, default=0.5,
                        help="Proportion of hard negatives in mixed mode (0.0-1.0)")

    args = parser.parse_args()

    train_seal_leave_one_out(
        target_disease=args.target_disease,
        num_hops=args.hops,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.workers,
        cache_root=args.cachedir,
        neg_strategy=args.neg_strategy,
        neg_ratio=args.neg_ratio,
        hard_ratio=args.hard_ratio,
        pooling=args.pooling,
        sort_k=args.sort_k,
        conv_type=args.conv,
        max_z=args.max_z,
        dropout=args.dropout,
        max_nodes_per_hop=args.max_nodes_per_hop,
    )

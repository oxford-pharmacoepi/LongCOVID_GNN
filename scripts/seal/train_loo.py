#!/usr/bin/env python3
"""
SEAL Leave-One-Out Validation for Drug-Disease Link Prediction

Trains a SEAL model on all drug-disease edges *except* those involving
a held-out target disease, then ranks all drugs for that disease.
This mirrors the Global GNN LOO workflow in ``leave_one_out_validation.py``
but uses subgraph-based classification instead of full-graph embeddings.
"""

import argparse
import copy
from collections import defaultdict
from datetime import datetime, timezone
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models_seal import SEALDataset, SEALModel, MAX_Z
from src.training.tracker import ExperimentTracker


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
    num_hops: int = 2,
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
    val_ratio: float = 0.1,
    patience: int = 10,
    pos_weight: float = None,
    grad_clip: float = 1.0,
    warmup_epochs: int = 5,
    use_jk: bool = False,
    use_node_types: bool = True,
    mlflow_tracking: bool = True,
):
    """Train SEAL with leave-one-out validation for *target_disease*."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── MLflow setup ────────────────────────────────────────────────────
    tracker = None
    if mlflow_tracking:
        try:
            tracker = ExperimentTracker(
                experiment_name=f"SEAL-{target_disease}",
            )
            run_name = f"seal_{conv_type}_{hidden_channels}h_{num_hops}hop_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            tracker.start_run(run_name=run_name)
            print(f"MLflow tracking enabled: SEAL-{target_disease} / {run_name}")
        except Exception as e:
            print(f"MLflow init failed ({e}), continuing without tracking")
            tracker = None

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
    with open(f"{mappings_path}/gene_key_mapping.json") as f:
        gene_mapping = {k: int(v) for k, v in json.load(f).items()}

    idx_to_drug = {v: k for k, v in drug_mapping.items()}

    # ── 1b. Build node-type features (drug=0, gene=1, disease=2) ──
    if use_node_types:
        print("Building node-type features...")
        node_type = torch.zeros(graph_data.num_nodes, dtype=torch.long)
        for idx in drug_mapping.values():
            node_type[idx] = 0
        for idx in gene_mapping.values():
            node_type[idx] = 1
        for idx in disease_mapping.values():
            node_type[idx] = 2
        # One-hot encode (3 types)
        node_type_features = torch.nn.functional.one_hot(node_type, num_classes=3).float()
        # Append to existing node features
        if node_features is not None:
            node_features = torch.cat([node_features.float(), node_type_features], dim=1)
        else:
            node_features = node_type_features
        print(f"  Node features: {node_features.shape[1]} dims "
              f"(original + 3 node-type indicators)")

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
    all_train_edges = [e for e in drug_disease_edges if e[1] != target_idx]

    if len(test_edges) == 0:
        print(f"\n  WARNING: No test edges for {target_disease}. "
              "LOO validation will have no ground truth to evaluate against.")

    # ── 2b. Validation split ────────────────────────────────────────────
    if val_ratio > 0 and len(all_train_edges) > 20:
        n_val = max(1, int(len(all_train_edges) * val_ratio))
        random.shuffle(all_train_edges)
        val_edges = all_train_edges[:n_val]
        train_edges = all_train_edges[n_val:]
        print(f"  Train edges: {len(train_edges)}")
        print(f"  Val edges:   {len(val_edges)}")
        print(f"  Test edges:  {len(test_edges)}")
    else:
        train_edges = all_train_edges
        val_edges = []
        print(f"  Train edges: {len(train_edges)}")
        print(f"  Test edges:  {len(test_edges)}")

    # ── 3. Negative sampling ────────────────────────────────────────────
    all_drug_list = list(drug_indices)
    all_disease_list = list(disease_indices)
    positive_set = set(drug_disease_edges)

    # Pollution control: don't sample held-out test edges as negatives
    future_positives = set(test_edges) | set(val_edges)
    num_neg = len(train_edges) * neg_ratio

    print(f"\nSampling {num_neg} negative edges "
          f"(strategy={neg_strategy}, ratio=1:{neg_ratio})...")

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

    # Validation negatives (same strategy and ratio as training)
    if val_edges:
        num_val_neg = len(val_edges) * neg_ratio
        exclude_set = positive_set | set(neg_train_edges)
        if neg_strategy == "hard":
            neg_val_edges = sample_negatives_hard(
                exclude_set, all_drug_list, all_disease_list, num_val_neg,
                edge_index, exclude_disease=target_idx,
                future_positives=future_positives,
            )
        elif neg_strategy == "mixed":
            neg_val_edges = sample_negatives_mixed(
                exclude_set, all_drug_list, all_disease_list, num_val_neg,
                edge_index, exclude_disease=target_idx,
                future_positives=future_positives, hard_ratio=hard_ratio,
            )
        else:  # random
            neg_val_edges = sample_negatives_random(
                exclude_set, all_drug_list, all_disease_list, num_val_neg,
                exclude_disease=target_idx, future_positives=future_positives,
            )
        print(f"  Val negatives: {len(neg_val_edges)}")

    # ── 3b. Create training-only edge index ──
    # Only remove drug-disease edges
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

    # Validation dataset
    val_loader = None
    if val_edges:
        print("Initialising SEAL validation dataset...")
        val_pairs = val_edges + neg_val_edges
        val_labels = [1] * len(val_edges) + [0] * len(neg_val_edges)
        val_dataset = SEALDataset(
            root=f"{cache_root}/{target_disease}",
            pairs=val_pairs,
            labels=val_labels,
            edge_index=train_edge_index,
            node_features=node_features,
            num_hops=num_hops,
            max_z=max_z,
            max_nodes_per_hop=max_nodes_per_hop,
            adj_dict=adj_dict,
            use_cache=False,
            save_cache=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )

    # Pre-cache subgraphs in parallel (speeds up first run significantly)
    print("\nPre-caching subgraphs...")
    train_dataset.precache_parallel(num_workers=num_workers if num_workers > 0 else 0)

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
        use_jk=use_jk,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss with optional positive weighting
    if pos_weight is not None:
        pw = torch.tensor([pos_weight])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"  Using pos_weight={pos_weight}")
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # Cosine annealing LR scheduler (with warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, epochs - warmup_epochs), eta_min=lr / 100,
    )

    print(f"\nModel: {model.__class__.__name__}")
    print(f"  Conv type: {conv_type}, Layers: {num_layers}, Hidden: {hidden_channels}")
    print(f"  Pooling: {pooling}" + (f" (k={sort_k})" if pooling == "sort" else ""))
    print(f"  JKNet: {use_jk}, Node types: {use_node_types}")
    print(f"  Dropout: {dropout}, LR: {lr}, Weight Decay: {weight_decay}")
    print(f"  Warmup: {warmup_epochs} epochs, Grad clip: {grad_clip}")
    print(f"  max_z: {max_z}, Feature dim: {in_channels}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    if val_loader:
        print(f"  Early stopping: patience={patience}")

    # ── MLflow: log parameters ──────────────────────────────────────────
    if tracker:
        # Graph metadata
        tracker.log_param("graph_path", graph_path)
        tracker.log_metric("graph_num_nodes", graph_data.num_nodes)
        tracker.log_metric("graph_num_edges", edge_index.shape[1])
        tracker.log_metric("graph_num_features", node_features.shape[1] if node_features is not None else 0)
        tracker.log_param("target_disease", target_disease)
        # Model architecture
        tracker.log_param("conv_type", conv_type)
        tracker.log_param("hidden_channels", hidden_channels)
        tracker.log_param("num_layers", num_layers)
        tracker.log_param("pooling", pooling)
        tracker.log_param("use_jk", use_jk)
        tracker.log_param("use_node_types", use_node_types)
        tracker.log_param("max_z", max_z)
        tracker.log_param("in_channels", in_channels)
        tracker.log_metric("total_parameters", total_params)
        # Training hyperparameters
        tracker.log_param("epochs", epochs)
        tracker.log_param("batch_size", batch_size)
        tracker.log_param("lr", lr)
        tracker.log_param("weight_decay", weight_decay)
        tracker.log_param("dropout", dropout)
        tracker.log_param("grad_clip", grad_clip)
        tracker.log_param("warmup_epochs", warmup_epochs)
        tracker.log_param("pos_weight", pos_weight)
        tracker.log_param("seed", seed)
        # Negative sampling
        tracker.log_param("neg_strategy", neg_strategy)
        tracker.log_param("neg_ratio", neg_ratio)
        tracker.log_param("hard_ratio", hard_ratio if neg_strategy == "mixed" else None)
        # Subgraph configuration
        tracker.log_param("num_hops", num_hops)
        tracker.log_param("max_nodes_per_hop", max_nodes_per_hop)
        # Data splits
        tracker.log_param("val_ratio", val_ratio)
        tracker.log_param("patience", patience)
        tracker.log_metric("train_edges", len(train_edges))
        tracker.log_metric("val_edges", len(val_edges) if val_edges else 0)
        tracker.log_metric("test_edges", len(test_edges))
        tracker.log_metric("neg_train_edges", len(neg_train_edges))

    # ── 6. Training loop with early stopping ────────────────────────────
    print(f"\nTraining SEAL model for up to {epochs} epochs...", flush=True)
    best_val_auc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimiser.param_groups:
                pg['lr'] = warmup_lr

        for batch in train_loader:
            optimiser.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            total_loss += loss.item()

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        current_lr = optimiser.param_groups[0]['lr']

        # ── Validation step ──
        val_auc = None
        if val_loader and (epoch + 1) % 2 == 0:
            model.eval()
            val_preds, val_labels_list = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = torch.sigmoid(model(batch))
                    val_preds.extend(out.tolist())
                    val_labels_list.extend(batch.y.tolist())
            try:
                val_auc = roc_auc_score(val_labels_list, val_preds)
            except ValueError:
                val_auc = 0.5  # degenerate case

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 2  # checked every 2 epochs

        if (epoch + 1) % 5 == 0:
            status = f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
            if val_auc is not None:
                status += f", Val AUC: {val_auc:.4f} (best: {best_val_auc:.4f})"
            print(status, flush=True)

            # Log epoch metrics to MLflow
            if tracker:
                tracker.log_training_metrics(
                    epoch + 1, avg_loss,
                    val_metrics={"val_auc": val_auc} if val_auc is not None else None,
                )

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"\n  Early stopping at epoch {epoch + 1} "
                  f"(no val improvement for {patience} epochs)")
            break

    # Restore best model if we used validation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Restored best model (Val AUC: {best_val_auc:.4f})")

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
    hits_at_100 = sum(1 for r in ranks if r <= 100)
    median_rank = float(torch.tensor(ranks, dtype=torch.float).median()) if ranks else 9999.0
    mean_rank = float(torch.tensor(ranks, dtype=torch.float).mean()) if ranks else 9999.0
    mrr = float(torch.tensor([1.0 / r for r in ranks]).mean()) if ranks else 0.0

    n = len(true_drugs)
    total_drugs = len(all_drug_list)

    print(f"\nTop 20 Predictions:")
    print(f"{'Rank':<6} {'Drug ID':<20} {'Score':<10} {'True?'}")
    print("-" * 60)
    top20_list = []
    for rank, (drug_idx, score) in enumerate(drug_scores[:20], 1):
        drug_id = idx_to_drug.get(drug_idx, "Unknown")
        is_true = drug_idx in true_drugs
        mark = "✓ True" if is_true else ""
        print(f"{rank:<6} {drug_id:<20} {score:<10.4f} {mark}")
        top20_list.append({"rank": rank, "drug_id": drug_id, "score": round(score, 4), "true": is_true})

    print(f"\n{'=' * 60}")
    print(f"SEAL PERFORMANCE SUMMARY FOR {target_disease}")
    print(f"{'=' * 60}")
    print(f"  Test Edges (True Positives): {n}")
    print(f"  Total Drugs Ranked: {total_drugs}")
    print(f"  Config: {conv_type} | {pooling} | hidden={hidden_channels} | "
          f"layers={num_layers} | hops={num_hops}")
    print(f"  Sampling: {neg_strategy} | ratio=1:{neg_ratio}" +
          (f" | hard_ratio={hard_ratio}" if neg_strategy == "mixed" else ""))
    if val_edges:
        print(f"  Best Val AUC: {best_val_auc:.4f}")
    if n > 0:
        print(f"  Hits@10:  {hits_at_10} / {n} ({hits_at_10 / n * 100:.1f}%)")
        print(f"  Hits@20:  {hits_at_20} / {n} ({hits_at_20 / n * 100:.1f}%)")
        print(f"  Hits@50:  {hits_at_50} / {n} ({hits_at_50 / n * 100:.1f}%)")
        print(f"  Hits@100: {hits_at_100} / {n} ({hits_at_100 / n * 100:.1f}%)")
    else:
        print(f"  Hits@K: No test edges available")
    print(f"  Median Rank: {median_rank:.1f}")
    print(f"  Mean Rank: {mean_rank:.1f}")
    print(f"  MRR: {mrr:.4f}")
    print(f"{'=' * 60}\n")

    # ── 9. Save results to file ─────────────────────────────────────────
    results_dir = Path(cache_root).parent / "seal_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results_json = {
        "target_disease": target_disease,
        "timestamp": timestamp,
        "config": {
            "conv_type": conv_type, "pooling": pooling,
            "hidden": hidden_channels, "layers": num_layers,
            "hops": num_hops, "epochs": epochs,
            "neg_strategy": neg_strategy, "neg_ratio": neg_ratio,
            "hard_ratio": hard_ratio if neg_strategy == "mixed" else None,
            "use_jk": use_jk, "dropout": dropout,
            "seed": seed, "max_nodes_per_hop": max_nodes_per_hop,
        },
        "metrics": {
            "best_val_auc": round(best_val_auc, 4) if val_edges else None,
            "hits_at_10": hits_at_10, "hits_at_20": hits_at_20,
            "hits_at_50": hits_at_50, "hits_at_100": hits_at_100,
            "total_true": n, "total_drugs": total_drugs,
            "median_rank": round(median_rank, 1),
            "mean_rank": round(mean_rank, 1),
            "mrr": round(mrr, 4),
        },
        "top20": top20_list,
        "all_ranks": sorted(ranks),
    }

    json_path = results_dir / f"seal_{target_disease}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {json_path}")

    # ── MLflow: log final results ───────────────────────────────────────
    if tracker:
        tracker.log_metric("best_val_auc", best_val_auc)
        tracker.log_metric("hits_at_10", hits_at_10)
        tracker.log_metric("hits_at_20", hits_at_20)
        tracker.log_metric("hits_at_50", hits_at_50)
        tracker.log_metric("hits_at_100", hits_at_100)
        tracker.log_metric("median_rank", median_rank)
        tracker.log_metric("mean_rank", mean_rank)
        tracker.log_metric("mrr", mrr)
        tracker.log_metric("total_true", n)
        tracker.log_metric("total_drugs", total_drugs)
        tracker.log_metric("final_epoch", epoch + 1)
        # Log results JSON as artifact
        tracker.log_artifact(str(json_path), "seal_results")
        tracker.end_run()
        print(f"MLflow run logged to: SEAL-{target_disease}")

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
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="Positive class weight for BCE loss")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm (0 to disable)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="LR warmup epochs before cosine decay")

    # Architecture
    parser.add_argument("--hidden", type=int, default=32, help="Hidden channels")
    parser.add_argument("--layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--hops", type=int, default=2, help="Subgraph extraction hops")
    parser.add_argument("--max-nodes-per-hop", type=int, default=200,
                        help="Max neighbours sampled per hop (caps subgraph size)")
    parser.add_argument("--pooling", type=str, default="mean+max",
                        choices=["sort", "mean", "max", "mean+max"],
                        help="Graph-level pooling method")
    parser.add_argument("--sort-k", type=int, default=30,
                        help="k for SortAggregation (only used if --pooling=sort)")
    parser.add_argument("--conv", type=str, default="sage",
                        choices=["sage", "gcn", "gin", "gat"],
                        help="GNN convolution type")
    parser.add_argument("--max-z", type=int, default=MAX_Z,
                        help="Max DRNL label (controls one-hot dimensionality)")
    parser.add_argument("--use-jk", action="store_true",
                        help="Use JKNet-style layer concatenation")
    parser.add_argument("--no-node-types", action="store_true",
                        help="Disable node-type features")

    # Negative sampling
    parser.add_argument("--neg-strategy", type=str, default="mixed",
                        choices=["random", "hard", "mixed"],
                        help="Negative sampling strategy")
    parser.add_argument("--neg-ratio", type=int, default=3,
                        help="Ratio of negatives to positives (e.g. 3 means 1:3)")
    parser.add_argument("--hard-ratio", type=float, default=0.5,
                        help="Proportion of hard negatives in mixed mode (0.0-1.0)")

    # Validation
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of train edges to hold out for validation")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 to disable)")

    # MLflow
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow experiment tracking")

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
        val_ratio=args.val_ratio,
        patience=args.patience,
        pos_weight=args.pos_weight,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        use_jk=args.use_jk,
        use_node_types=not args.no_node_types,
        mlflow_tracking=not args.no_mlflow,
    )

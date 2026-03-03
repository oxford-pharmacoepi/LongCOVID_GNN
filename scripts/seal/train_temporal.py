#!/usr/bin/env python3
"""
SEAL Temporal Validation

Trains SEAL on the graph's temporal train split (2021 drug-disease edges)
and evaluates on the val (2023) and test (2024) splits — edges that were
added in later OpenTargets releases.

This validates whether the model can predict *future* drug-disease links
from structural patterns in the earlier graph.
"""

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models_seal import SEALDataset, SEALModel, MAX_Z


def train_seal_temporal(
    num_hops: int = 2,
    hidden_channels: int = 32,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    seed: int = 42,
    num_workers: int = 0,
    cache_root: str = "results/seal_tmp/temporal",
    neg_ratio: int = 3,
    pooling: str = "mean+max",
    sort_k: int = 30,
    conv_type: str = "sage",
    max_z: int = MAX_Z,
    dropout: float = 0.5,
    max_nodes_per_hop: int = 200,
    patience: int = 10,
    grad_clip: float = 1.0,
    warmup_epochs: int = 5,
    use_jk: bool = False,
    use_node_types: bool = True,
):
    """Train SEAL on temporal train split, evaluate on val/test splits."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 70)
    print("SEAL TEMPORAL VALIDATION")
    print("=" * 70)

    # ── 1. Load graph ────────────────────────────────────────────────────
    print("\nFinding graph...")
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
    idx_to_disease = {v: k for k, v in disease_mapping.items()}
    drug_indices = set(drug_mapping.values())
    disease_indices = set(disease_mapping.values())

    # ── 1b. Node-type features ───────────────────────────────────────────
    if use_node_types:
        print("Building node-type features...")
        node_type = torch.zeros(graph_data.num_nodes, dtype=torch.long)
        for idx in drug_mapping.values():
            node_type[idx] = 0
        for idx in gene_mapping.values():
            node_type[idx] = 1
        for idx in disease_mapping.values():
            node_type[idx] = 2
        node_type_features = F.one_hot(node_type, num_classes=3).float()
        if node_features is not None:
            node_features = torch.cat([node_features.float(), node_type_features], dim=1)
        else:
            node_features = node_type_features

    # ── 2. Extract temporal splits ───────────────────────────────────────
    print("\nExtracting temporal splits...")

    # Positive train edges (drug-disease only)
    train_pos = []
    for i in range(graph_data.train_edge_index.shape[0]):
        s, d = graph_data.train_edge_index[i].tolist()
        if graph_data.train_edge_label[i].item() == 1:
            if (s in drug_indices and d in disease_indices):
                train_pos.append((s, d))
            elif (d in drug_indices and s in disease_indices):
                train_pos.append((d, s))  # normalise: drug first

    val_pos = []
    for i in range(graph_data.val_edge_index.shape[0]):
        s, d = graph_data.val_edge_index[i].tolist()
        if graph_data.val_edge_label[i].item() == 1:
            if (s in drug_indices and d in disease_indices):
                val_pos.append((s, d))
            elif (d in drug_indices and s in disease_indices):
                val_pos.append((d, s))

    test_pos = []
    for i in range(graph_data.test_edge_index.shape[0]):
        s, d = graph_data.test_edge_index[i].tolist()
        if graph_data.test_edge_label[i].item() == 1:
            if (s in drug_indices and d in disease_indices):
                test_pos.append((s, d))
            elif (d in drug_indices and s in disease_indices):
                test_pos.append((d, s))

    print(f"  Train positives (drug-disease): {len(train_pos)}")
    print(f"  Val positives (drug-disease):   {len(val_pos)}")
    print(f"  Test positives (drug-disease):  {len(test_pos)}")

    # ── 3. Create training edge index ────────────────────────────────────
    # For temporal validation, the training graph should NOT contain
    # val/test positive edges. Check if they're already excluded from edge_index.
    positive_set = set(train_pos)
    val_test_set = set(val_pos + test_pos)

    # Verify val/test edges are in the full edge_index
    s_ei, d_ei = edge_index
    val_test_in_graph = 0
    for a, b in val_test_set:
        mask = ((s_ei == a) & (d_ei == b)) | ((s_ei == b) & (d_ei == a))
        if mask.any():
            val_test_in_graph += 1
    print(f"  Val/test edges found in full edge_index: {val_test_in_graph}/{len(val_test_set)}")

    # Remove val+test positive edges from edge_index for training
    train_edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
    for a, b in val_test_set:
        mask = ((s_ei == a) & (d_ei == b)) | ((s_ei == b) & (d_ei == a))
        train_edge_mask &= ~mask
    train_edge_index = edge_index[:, train_edge_mask]
    print(f"  Full graph edges: {edge_index.shape[1]}")
    print(f"  Training graph edges: {train_edge_index.shape[1]} "
          f"(removed {edge_index.shape[1] - train_edge_index.shape[1]})")

    # ── 4. Negative sampling ────────────────────────────────────────────
    all_drug_list = list(drug_indices)
    all_disease_list = list(disease_indices)
    all_positive = positive_set | val_test_set

    print(f"\nSampling negatives...")

    def sample_random_negatives(n, exclude):
        negs = []
        while len(negs) < n:
            drug = random.choice(all_drug_list)
            dis = random.choice(all_disease_list)
            pair = (drug, dis)
            if pair not in exclude and pair not in negs:
                negs.append(pair)
        return negs

    neg_train = sample_random_negatives(len(train_pos) * neg_ratio, all_positive)
    neg_val = sample_random_negatives(len(val_pos) * neg_ratio, all_positive | set(neg_train))
    neg_test = sample_random_negatives(len(test_pos) * neg_ratio, all_positive | set(neg_train) | set(neg_val))

    print(f"  Train: {len(train_pos)} pos + {len(neg_train)} neg")
    print(f"  Val:   {len(val_pos)} pos + {len(neg_val)} neg")
    print(f"  Test:  {len(test_pos)} pos + {len(neg_test)} neg")

    # ── 5. Build adjacency dict ──────────────────────────────────────────
    print("\nBuilding adjacency dict...")
    adj_dict = defaultdict(set)
    s, d = train_edge_index
    for i in range(train_edge_index.shape[1]):
        u, v = s[i].item(), d[i].item()
        adj_dict[u].add(v)
        adj_dict[v].add(u)

    # ── 6. Datasets ──────────────────────────────────────────────────────
    print("\nInitialising datasets...")
    train_pairs = train_pos + neg_train
    train_labels = [1] * len(train_pos) + [0] * len(neg_train)

    train_dataset = SEALDataset(
        root=f"{cache_root}/train",
        pairs=train_pairs,
        labels=train_labels,
        edge_index=train_edge_index,
        node_features=node_features,
        num_hops=num_hops,
        max_z=max_z,
        max_nodes_per_hop=max_nodes_per_hop,
        adj_dict=adj_dict,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_pairs = val_pos + neg_val
    val_labels = [1] * len(val_pos) + [0] * len(neg_val)

    val_dataset = SEALDataset(
        root=f"{cache_root}/val",
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_pairs = test_pos + neg_test
    test_labels = [1] * len(test_pos) + [0] * len(neg_test)

    test_dataset = SEALDataset(
        root=f"{cache_root}/test",
        pairs=test_pairs,
        labels=test_labels,
        edge_index=train_edge_index,
        node_features=node_features,
        num_hops=num_hops,
        max_z=max_z,
        max_nodes_per_hop=max_nodes_per_hop,
        adj_dict=adj_dict,
        use_cache=False,
        save_cache=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Pre-cache training subgraphs
    print("\nPre-caching training subgraphs...")
    train_dataset.precache_parallel(num_workers=num_workers if num_workers > 0 else 0)

    # ── 7. Model ──────────────────────────────────────────────────────────
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
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, epochs - warmup_epochs), eta_min=lr / 100,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model.__class__.__name__}")
    print(f"  Conv: {conv_type}, Layers: {num_layers}, Hidden: {hidden_channels}")
    print(f"  Pooling: {pooling}, JK: {use_jk}, Params: {total_params:,}")
    print(f"  Feature dim: {in_channels}")

    # ── 8. Training ───────────────────────────────────────────────────────
    print(f"\nTraining for up to {epochs} epochs...")
    best_val_auc = 0.0
    best_model_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimiser.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_dataset)

        # LR schedule
        if epoch >= warmup_epochs:
            scheduler.step()
        current_lr = optimiser.param_groups[0]['lr']

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_preds, val_labels_list = [], []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch)
                    val_preds.extend(torch.sigmoid(logits).tolist())
                    val_labels_list.extend(batch.y.tolist())
            val_auc = roc_auc_score(val_labels_list, val_preds)

            improved = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
                improved = " *"
            else:
                no_improve += 5

            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Val AUC: {val_auc:.4f} (best: {best_val_auc:.4f}){improved}")

            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        elif (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Restored best model (Val AUC: {best_val_auc:.4f})")

    # ── 9. Test evaluation ────────────────────────────────────────────────
    print(f"\nEvaluating on temporal test set ({len(test_pos)} pos + {len(neg_test)} neg)...")
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            logits = model(batch)
            test_preds.extend(torch.sigmoid(logits).tolist())
            test_labels_list.extend(batch.y.tolist())

    test_auc = roc_auc_score(test_labels_list, test_preds)

    # Compute ranking metrics on test positives
    # Score all (drug, disease) for each unique disease in test set
    test_diseases = set(d for _, d in test_pos)
    print(f"\nRanking evaluation across {len(test_diseases)} test diseases...")

    all_ranks = []
    per_disease_results = {}

    for disease_idx in sorted(test_diseases):
        disease_id = idx_to_disease.get(disease_idx, str(disease_idx))
        true_drugs_for_disease = {s for s, d in test_pos if d == disease_idx}

        # Score all drugs for this disease
        eval_pairs = [(drug, disease_idx) for drug in all_drug_list]
        eval_dataset = SEALDataset(
            root=f"{cache_root}/eval_{disease_idx}",
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
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        scores = []
        with torch.no_grad():
            for batch in eval_loader:
                logits = model(batch)
                scores.extend(torch.sigmoid(logits).tolist())

        # Rank
        drug_scores = list(zip(all_drug_list, scores))
        drug_scores.sort(key=lambda x: x[1], reverse=True)
        rank_map = {d: r + 1 for r, (d, _) in enumerate(drug_scores)}

        disease_ranks = []
        for drug_idx in true_drugs_for_disease:
            rank = rank_map.get(drug_idx, len(all_drug_list))
            disease_ranks.append(rank)
            all_ranks.append(rank)

        median_r = sorted(disease_ranks)[len(disease_ranks) // 2]
        hits10 = sum(1 for r in disease_ranks if r <= 10)
        hits50 = sum(1 for r in disease_ranks if r <= 50)
        per_disease_results[disease_id] = {
            "n_true": len(true_drugs_for_disease),
            "median_rank": median_r,
            "hits_at_10": hits10,
            "hits_at_50": hits50,
            "ranks": sorted(disease_ranks),
        }
        print(f"  {disease_id}: {len(true_drugs_for_disease)} drugs, "
              f"median_rank={median_r}, hits@50={hits50}")

    # Global metrics
    total_true = len(all_ranks)
    median_rank = sorted(all_ranks)[total_true // 2] if all_ranks else 0
    mean_rank = sum(all_ranks) / total_true if all_ranks else 0
    hits_at_10 = sum(1 for r in all_ranks if r <= 10)
    hits_at_20 = sum(1 for r in all_ranks if r <= 20)
    hits_at_50 = sum(1 for r in all_ranks if r <= 50)
    hits_at_100 = sum(1 for r in all_ranks if r <= 100)
    mrr = sum(1.0 / r for r in all_ranks) / total_true if all_ranks else 0

    print(f"\n{'=' * 70}")
    print(f"TEMPORAL VALIDATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Test positives: {total_true}")
    print(f"  Total drugs: {len(all_drug_list)}")
    print(f"  Diseases in test: {len(test_diseases)}")
    print(f"  Hits@10:  {hits_at_10} / {total_true} ({100*hits_at_10/total_true:.1f}%)")
    print(f"  Hits@20:  {hits_at_20} / {total_true} ({100*hits_at_20/total_true:.1f}%)")
    print(f"  Hits@50:  {hits_at_50} / {total_true} ({100*hits_at_50/total_true:.1f}%)")
    print(f"  Hits@100: {hits_at_100} / {total_true} ({100*hits_at_100/total_true:.1f}%)")
    print(f"  Median Rank: {median_rank}")
    print(f"  Mean Rank: {mean_rank:.1f}")
    print(f"  MRR: {mrr:.4f}")
    print(f"{'=' * 70}")

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "type": "temporal_validation",
        "seed": seed,
        "config": {
            "conv_type": conv_type, "hidden": hidden_channels,
            "hops": num_hops, "use_jk": use_jk, "pooling": pooling,
            "epochs": epochs, "neg_ratio": neg_ratio, "max_nodes_per_hop": max_nodes_per_hop,
        },
        "splits": {
            "train_pos": len(train_pos), "val_pos": len(val_pos), "test_pos": len(test_pos),
        },
        "metrics": {
            "test_auc": round(test_auc, 4),
            "best_val_auc": round(best_val_auc, 4),
            "hits_at_10": hits_at_10, "hits_at_20": hits_at_20,
            "hits_at_50": hits_at_50, "hits_at_100": hits_at_100,
            "median_rank": median_rank, "mean_rank": round(mean_rank, 1),
            "mrr": round(mrr, 4), "total_true": total_true,
            "total_drugs": len(all_drug_list), "test_diseases": len(test_diseases),
        },
        "per_disease": per_disease_results,
    }

    results_dir = Path("results/seal_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"seal_temporal_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    return test_auc, median_rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEAL Temporal Validation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--conv", type=str, default="sage")
    parser.add_argument("--pooling", type=str, default="mean+max")
    parser.add_argument("--max-nodes-per-hop", type=int, default=200)
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use-jk", action="store_true")
    parser.add_argument("--no-node-types", action="store_true")

    args = parser.parse_args()

    train_seal_temporal(
        num_hops=args.hops,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        neg_ratio=args.neg_ratio,
        pooling=args.pooling,
        conv_type=args.conv,
        dropout=0.5,
        max_nodes_per_hop=args.max_nodes_per_hop,
        patience=args.patience,
        use_jk=args.use_jk,
        use_node_types=not args.no_node_types,
    )

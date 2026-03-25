#!/usr/bin/env python3
"""
NCN (Neural Common Neighbor) Leave-One-Out Evaluation

Full-graph GNN that scores edges by aggregating representations of their
common neighbors.  Supports edge-type ablation, node-feature ablation,
and failed-RCT analysis — matching SEAL's train_loo.py feature set.

Reference:
  Wang, X., Yang, H., & Zhang, M. (2024).
  Neural Common Neighbor with Completion for Link Prediction. ICLR 2024.
  https://arxiv.org/abs/2302.00890
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Model components
# ═══════════════════════════════════════════════════════════════════════════

def count_common_neighbors(adj_scipy: sp.csr_matrix,
                           src: torch.Tensor,
                           dst: torch.Tensor) -> torch.Tensor:
    """Count common neighbors for each edge using scipy sparse."""
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    rows_src = adj_scipy[src_np]
    rows_dst = adj_scipy[dst_np]
    cn_counts = np.array(rows_src.multiply(rows_dst).sum(axis=1)).flatten()
    return torch.tensor(cn_counts, dtype=torch.float32)


class DropEdge(nn.Module):
    """Randomly drop edges from adjacency during training (edge dropout).

    Matches DropAdj from the official NCN codebase.
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return edge_index
        mask = torch.rand(edge_index.shape[1], device=edge_index.device) > self.p
        return edge_index[:, mask]


class GCNEncoder(nn.Module):
    """GCN encoder matching the paper's implementation.

    Supports residual connections, Jumping Knowledge (JK), input projection,
    and LayerNorm — matching the puregcn+res+jk configuration from the
    official GraphPKU/NeuralCommonNeighbor codebase.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=1,
                 dropout=0.3, xdp=0.0, use_ln=True, use_res=True,
                 use_jk=False):
        super().__init__()
        self.dropout = dropout
        self.use_res = use_res
        self.use_jk = use_jk

        # Input projection (maps features → hidden_channels)
        self.xemb = nn.Sequential(
            nn.Dropout(xdp),
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
        )

        if use_jk and num_layers > 0:
            self.jk_params = nn.Parameter(torch.randn(num_layers))

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        lnfn = lambda dim: nn.LayerNorm(dim) if use_ln else nn.Identity()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.lns.append(nn.Sequential(
                lnfn(hidden_channels),
                nn.Dropout(dropout),
                nn.ReLU(),
            ))

    def forward(self, x, edge_index):
        x = self.xemb(x)
        jk_xs = []
        for conv, ln in zip(self.convs, self.lns):
            x1 = ln(conv(x, edge_index))
            if self.use_res:
                x = x1 + x  # Residual connection
            else:
                x = x1
            if self.use_jk:
                jk_xs.append(x)
        if self.use_jk and jk_xs:
            stacked = torch.stack(jk_xs, dim=0)  # (layers, nodes, hidden)
            weights = self.jk_params.reshape(-1, 1, 1)
            x = (stacked * weights).sum(dim=0)
        return x


class NCNPredictor(nn.Module):
    """Neural Common Neighbor link predictor.

    Architecture matches SCNLinkPredictor from the official
    GraphPKU/NeuralCommonNeighbor codebase (3-layer MLPs).
    """
    def __init__(self, in_channels, hidden_channels, dropout=0.3,
                 use_ln=True, beta=1.0, edge_dropout=0.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta]))
        self.drop_edge = DropEdge(edge_dropout)
        lnfn = lambda dim: nn.LayerNorm(dim) if use_ln else nn.Identity()
        # 3-layer xcnlin matching official code:
        # Linear(1→h) → Dropout → ReLU → Linear(h→h) → LN → Dropout → ReLU → Linear(h→h)
        self.xcnlin = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        # 2-layer xijlin matching official code:
        # Linear(in→h) → LN → Dropout → ReLU → Linear(h→h)
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            lnfn(hidden_channels), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        # Final scoring MLP:
        # Linear(h→h) → LN → Dropout → ReLU → Linear(h→1)
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, h, cn_counts, src_idx, dst_idx):
        xi = h[src_idx]
        xj = h[dst_idx]
        xij = self.xijlin(xi * xj)
        xcn = self.xcnlin(cn_counts)
        score = self.lin(self.beta * xcn + xij)
        return score.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# LOO evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_ncn_loo(
    target_disease: str = "EFO_0003854",
    epochs: int = 200,
    hidden: int = 64,
    num_gnn_layers: int = 1,
    dropout: float = 0.3,
    xdp: float = 0.0,
    gnn_lr: float = 0.0003,
    pred_lr: float = 0.0003,
    edge_dropout: float = 0.3,
    batch_size: int = 4096,
    maskinput: bool = True,
    neg_ratio: int = 3,
    patience: int = 20,
    seed: int = 42,
    beta: float = 1.0,
    use_res: bool = True,
    use_jk: bool = False,
    exclude_edge_types: list = None,
    no_node_features: bool = False,
    no_node_types: bool = False,
    failed_rcts_file: str = None,
):
    """Run NCN LOO evaluation matching the SEAL protocol."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load graph ────────────────────────────────────────────────────
    print("Finding graph...")
    graph_files = sorted(glob.glob("results/graph_*.pt"))
    if not graph_files:
        raise FileNotFoundError("No graph found in results/")
    graph_path = graph_files[-1]
    print(f"  Using: {graph_path}")

    graph_data = torch.load(graph_path, weights_only=False)
    edge_index = graph_data.edge_index
    node_features = graph_data.x.float()
    num_nodes = graph_data.num_nodes

    mappings_path = graph_path.replace(".pt", "_mappings")
    with open(f"{mappings_path}/drug_key_mapping.json") as f:
        drug_mapping = {k: int(v) for k, v in json.load(f).items()}
    with open(f"{mappings_path}/disease_key_mapping.json") as f:
        disease_mapping = {k: int(v) for k, v in json.load(f).items()}
    with open(f"{mappings_path}/gene_key_mapping.json") as f:
        gene_mapping = {k: int(v) for k, v in json.load(f).items()}

    idx_to_drug = {v: k for k, v in drug_mapping.items()}

    # ── 1b. Edge-type ablation ───────────────────────────────────────────
    if exclude_edge_types:
        print(f"\nEdge-type ablation: excluding {exclude_edge_types}")
        src_ei, dst_ei = edge_index
        drug_set = set(drug_mapping.values())
        gene_set = set(gene_mapping.values())
        disease_set = set(disease_mapping.values())

        is_drug_t = torch.zeros(num_nodes, dtype=torch.bool)
        is_gene_t = torch.zeros(num_nodes, dtype=torch.bool)
        is_disease_t = torch.zeros(num_nodes, dtype=torch.bool)
        is_drug_t[list(drug_set)] = True
        is_gene_t[list(gene_set)] = True
        is_disease_t[list(disease_set)] = True

        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        before = edge_index.shape[1]

        for etype in exclude_edge_types:
            etype = etype.lower().replace('_', '-').replace(' ', '-')
            if etype in ('ppi', 'gene-gene'):
                mask = is_gene_t[src_ei] & is_gene_t[dst_ei]
                keep_mask &= ~mask
                print(f"    Removing gene-gene (PPI): {mask.sum().item()} edges")
            elif etype in ('drug-gene', 'moa'):
                mask = (is_drug_t[src_ei] & is_gene_t[dst_ei]) | \
                       (is_gene_t[src_ei] & is_drug_t[dst_ei])
                keep_mask &= ~mask
                print(f"    Removing drug-gene (MoA): {mask.sum().item()} edges")
            elif etype in ('disease-gene', 'dg'):
                mask = (is_disease_t[src_ei] & is_gene_t[dst_ei]) | \
                       (is_gene_t[src_ei] & is_disease_t[dst_ei])
                keep_mask &= ~mask
                print(f"    Removing disease-gene: {mask.sum().item()} edges")
            elif etype in ('disease-similarity', 'dd'):
                mask = is_disease_t[src_ei] & is_disease_t[dst_ei]
                keep_mask &= ~mask
                print(f"    Removing disease-disease: {mask.sum().item()} edges")
            else:
                print(f"    WARNING: unknown edge type '{etype}', skipping")

        edge_index = edge_index[:, keep_mask]
        print(f"  Edges: {before} → {edge_index.shape[1]} "
              f"(removed {before - edge_index.shape[1]})")

    # ── 1c. Node features ────────────────────────────────────────────────
    use_node_types = not no_node_types
    if no_node_features:
        node_features = None
        print("Node feature ablation: using ONLY node-type indicators" if use_node_types
              else "Node feature ablation: NO features at all (CN counts only)")

    if use_node_types:
        node_type = torch.zeros(num_nodes, dtype=torch.long)
        for idx in drug_mapping.values():
            node_type[idx] = 0
        for idx in gene_mapping.values():
            node_type[idx] = 1
        for idx in disease_mapping.values():
            node_type[idx] = 2
        node_type_features = torch.nn.functional.one_hot(node_type, num_classes=3).float()
        if node_features is not None and not no_node_features:
            node_features = torch.cat([node_features.float(), node_type_features], dim=1)
        else:
            node_features = node_type_features
        print(f"  Node features: {node_features.shape[1]} dims")
    elif no_node_features:
        # No node types and no node features — use identity-like embedding
        node_features = torch.ones(num_nodes, 1)
        print("  Node features: 1 dim (constant)")

    # ── 2. Extract drug-disease edges and build LOO split ────────────────
    drug_indices = set(drug_mapping.values())
    disease_indices = set(disease_mapping.values())

    target_idx = disease_mapping.get(target_disease)
    if target_idx is None:
        raise ValueError(f"Disease {target_disease} not found")
    print(f"\nTarget disease: {target_disease} (idx={target_idx})")

    drug_disease_edges = set()
    src, dst = edge_index
    for i in range(edge_index.shape[1]):
        u, v = src[i].item(), dst[i].item()
        if u in drug_indices and v in disease_indices:
            drug_disease_edges.add((u, v))
        elif v in drug_indices and u in disease_indices:
            drug_disease_edges.add((v, u))

    test_edges = [(d, dis) for d, dis in drug_disease_edges if dis == target_idx]
    train_dd_edges = [(d, dis) for d, dis in drug_disease_edges if dis != target_idx]

    print(f"  Total drug-disease edges: {len(drug_disease_edges)}")
    print(f"  Test edges (held out): {len(test_edges)}")
    print(f"  Train drug-disease edges: {len(train_dd_edges)}")

    if not test_edges:
        print(f"ERROR: No test edges for {target_disease}")
        return

    # ── 3. Build training graph (remove test edges) ──────────────────────
    test_edge_set = set()
    for d, dis in test_edges:
        test_edge_set.add((d, dis))
        test_edge_set.add((dis, d))

    keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
    for i in range(edge_index.shape[1]):
        u, v = src[i].item(), dst[i].item()
        if (u, v) in test_edge_set:
            keep_mask[i] = False

    train_edge_index = edge_index[:, keep_mask].to(device)
    print(f"  Training edges (all types): {train_edge_index.shape[1]}")

    # Build scipy adjacency for CN computation
    train_ei_cpu = train_edge_index.cpu()
    adj_scipy = sp.csr_matrix(
        (np.ones(train_ei_cpu.shape[1]),
         (train_ei_cpu[0].numpy(), train_ei_cpu[1].numpy())),
        shape=(num_nodes, num_nodes)
    )
    adj_scipy = adj_scipy + adj_scipy.T
    adj_scipy[adj_scipy > 1] = 1

    # ── 4. Build train/val splits ────────────────────────────────────────
    random.shuffle(train_dd_edges)
    val_size = max(1, len(train_dd_edges) // 10)
    val_edges_list = train_dd_edges[:val_size]
    actual_train_edges = train_dd_edges[val_size:]

    all_drug_list = sorted(drug_indices)
    all_disease_list = [d for d in disease_indices if d != target_idx]
    dd_set = set(drug_disease_edges)

    def sample_negatives(num_samples):
        negs = []
        seen = set(dd_set)
        attempts = 0
        while len(negs) < num_samples and attempts < num_samples * 20:
            d = random.choice(all_drug_list)
            dis = random.choice(all_disease_list)
            if (d, dis) not in seen:
                negs.append((d, dis))
                seen.add((d, dis))
            attempts += 1
        return negs

    neg_train = sample_negatives(len(actual_train_edges) * neg_ratio)
    neg_val = sample_negatives(len(val_edges_list) * neg_ratio)

    # Precompute CN counts
    def precompute_cn(edge_list):
        if not edge_list:
            return torch.zeros(0, 1)
        src_t = torch.tensor([e[0] for e in edge_list])
        dst_t = torch.tensor([e[1] for e in edge_list])
        cn = count_common_neighbors(adj_scipy, src_t, dst_t)
        return cn.reshape(-1, 1)

    print("Precomputing common neighbor counts...")
    cn_train_pos = precompute_cn(actual_train_edges).to(device)
    cn_train_neg = precompute_cn(neg_train).to(device)
    cn_val_pos = precompute_cn(val_edges_list).to(device)
    cn_val_neg = precompute_cn(neg_val).to(device)

    # Edge tensors
    train_pos_src = torch.tensor([e[0] for e in actual_train_edges], device=device)
    train_pos_dst = torch.tensor([e[1] for e in actual_train_edges], device=device)
    train_neg_src = torch.tensor([e[0] for e in neg_train], device=device)
    train_neg_dst = torch.tensor([e[1] for e in neg_train], device=device)
    val_pos_src = torch.tensor([e[0] for e in val_edges_list], device=device)
    val_pos_dst = torch.tensor([e[1] for e in val_edges_list], device=device)
    val_neg_src = torch.tensor([e[0] for e in neg_val], device=device)
    val_neg_dst = torch.tensor([e[1] for e in neg_val], device=device)

    x = node_features.to(device)

    # ── 5. Build model ───────────────────────────────────────────────────
    model = GCNEncoder(
        x.shape[1], hidden, num_gnn_layers, dropout,
        xdp=xdp, use_res=use_res, use_jk=use_jk
    ).to(device)
    predictor = NCNPredictor(hidden, hidden, dropout, beta=beta,
                             edge_dropout=edge_dropout).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': gnn_lr},
        {'params': predictor.parameters(), 'lr': pred_lr},
    ])

    total_params = sum(p.numel() for p in model.parameters()) + \
                   sum(p.numel() for p in predictor.parameters())
    print(f"\n  Model params: {total_params:,}")
    print(f"  Hidden: {hidden}, GNN layers: {num_gnn_layers}")

    # ── 6. Training ──────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    # Precompute hash-encoded edge index for efficient maskinput
    N = num_nodes  # for hash encoding: edge_hash = src * N + dst
    ei_hash = train_edge_index[0].long() * N + train_edge_index[1].long()

    print(f"\nTraining NCN for {epochs} epochs (maskinput={maskinput})...")
    for epoch in range(1, epochs + 1):
        model.train()
        predictor.train()

        n_train = len(actual_train_edges)
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()

            # maskinput: remove current batch's positive edges from adj
            # This prevents the GCN from seeing edges it's being asked to score
            if maskinput:
                batch_src = train_pos_src[idx]
                batch_dst = train_pos_dst[idx]
                # Hash both directions of batch edges
                remove_hash = torch.cat([
                    batch_src.long() * N + batch_dst.long(),
                    batch_dst.long() * N + batch_src.long(),
                ])
                # Fast vectorised membership test
                keep = ~torch.isin(ei_hash, remove_hash)
                masked_ei = train_edge_index[:, keep]
            else:
                masked_ei = train_edge_index

            # Apply edge dropout
            masked_ei = predictor.drop_edge(masked_ei)

            h = model(x, masked_ei)

            pos_score = predictor(h, cn_train_pos[idx],
                                  train_pos_src[idx], train_pos_dst[idx])
            pos_loss = -F.logsigmoid(pos_score).mean()

            neg_idx = torch.randint(0, len(neg_train), (len(idx),), device=device)
            neg_score = predictor(h, cn_train_neg[neg_idx],
                                  train_neg_src[neg_idx], train_neg_dst[neg_idx])
            neg_loss = -F.logsigmoid(-neg_score).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        predictor.eval()
        with torch.no_grad():
            h = model(x, train_edge_index)
            vp = predictor(h, cn_val_pos, val_pos_src, val_pos_dst).cpu()
            vn = predictor(h, cn_val_neg, val_neg_src, val_neg_dst).cpu()
            labels = torch.cat([torch.ones(vp.size(0)), torch.zeros(vn.size(0))])
            scores = torch.cat([vp, vn])
            val_auc = roc_auc_score(labels.numpy(), scores.numpy())

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {
                'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'pred': {k: v.cpu().clone() for k, v in predictor.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # ── 7. Restore best model and rank all drugs ─────────────────────────
    model.load_state_dict(best_state['model'])
    predictor.load_state_dict(best_state['pred'])
    model = model.to(device)
    predictor = predictor.to(device)
    model.eval()
    predictor.eval()

    print(f"\nBest model: epoch {best_epoch}, val AUC = {best_val_auc:.4f}")
    print("Ranking all drugs for target disease...")

    target_t = torch.tensor([target_idx], device=device)

    all_scores = []
    with torch.no_grad():
        h = model(x, train_edge_index)
        for i in range(0, len(all_drug_list), batch_size):
            batch = all_drug_list[i:i + batch_size]
            src_t = torch.tensor(batch, device=device)
            dst_t = target_t.expand(len(batch))
            cn = count_common_neighbors(
                adj_scipy,
                torch.tensor(batch),
                torch.tensor([target_idx] * len(batch))
            ).reshape(-1, 1).to(device)
            scores = predictor(h, cn, src_t, dst_t)
            all_scores.extend(zip(batch, scores.cpu().tolist()))

    drug_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)

    # ── 8. Metrics ───────────────────────────────────────────────────────
    true_drugs = {d for d, _ in test_edges}
    rank_map = {d: r for r, (d, _) in enumerate(drug_scores, 1)}
    ranks = [rank_map[d] for d in true_drugs if d in rank_map]

    n = len(true_drugs)
    total_drugs = len(all_drug_list)

    hits_at_10 = sum(1 for r in ranks if r <= 10)
    hits_at_20 = sum(1 for r in ranks if r <= 20)
    hits_at_50 = sum(1 for r in ranks if r <= 50)
    hits_at_100 = sum(1 for r in ranks if r <= 100)
    median_rank = float(torch.tensor(ranks, dtype=torch.float).median()) if ranks else 9999.0
    mean_rank = float(torch.tensor(ranks, dtype=torch.float).mean()) if ranks else 9999.0
    mrr = float(torch.tensor([1.0 / r for r in ranks]).mean()) if ranks else 0.0

    bottom_10 = sum(1 for r in ranks if r > total_drugs - 10)
    bottom_20 = sum(1 for r in ranks if r > total_drugs - 20)
    bottom_50 = sum(1 for r in ranks if r > total_drugs - 50)
    bottom_100 = sum(1 for r in ranks if r > total_drugs - 100)

    # Build config string for printing
    ablation_str = ""
    if exclude_edge_types:
        ablation_str += f" | excl={','.join(exclude_edge_types)}"
    if no_node_features:
        ablation_str += " | no_feat"
    if no_node_types:
        ablation_str += " | no_types"

    print(f"\n{'=' * 60}")
    print(f"NCN PERFORMANCE SUMMARY FOR {target_disease}")
    print(f"{'=' * 60}")
    print(f"  Test Edges (True Positives): {n}")
    print(f"  Total Drugs Ranked: {total_drugs}")
    print(f"  Config: hidden={hidden} | layers={num_gnn_layers} | "
          f"lr={gnn_lr}/{pred_lr} | beta={beta}{ablation_str}")
    print(f"  Best Val AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    if n > 0:
        print(f"  --- Top-K (higher is better) ---")
        print(f"  Hits@10:  {hits_at_10} / {n} ({hits_at_10 / n * 100:.1f}%)")
        print(f"  Hits@20:  {hits_at_20} / {n} ({hits_at_20 / n * 100:.1f}%)")
        print(f"  Hits@50:  {hits_at_50} / {n} ({hits_at_50 / n * 100:.1f}%)")
        print(f"  Hits@100: {hits_at_100} / {n} ({hits_at_100 / n * 100:.1f}%)")
        print(f"  Median Rank: {median_rank:.1f}")
        print(f"  Mean Rank: {mean_rank:.1f}")
        print(f"  MRR: {mrr:.4f}")
        print(f"  --- Bottom-K (false negatives, lower is better) ---")
        print(f"  Bottom@10:  {bottom_10} / {n} ({bottom_10 / n * 100:.1f}%)")
        print(f"  Bottom@20:  {bottom_20} / {n} ({bottom_20 / n * 100:.1f}%)")
        print(f"  Bottom@50:  {bottom_50} / {n} ({bottom_50 / n * 100:.1f}%)")
        print(f"  Bottom@100: {bottom_100} / {n} ({bottom_100 / n * 100:.1f}%)")
    print(f"{'=' * 60}")

    print(f"\nTop 20 NCN Predictions:")
    print(f"{'Rank':<6} {'Drug ID':<20} {'Score':<10} {'True?'}")
    print("-" * 60)
    top20_list = []
    for rank, (drug_idx, score) in enumerate(drug_scores[:20], 1):
        drug_id = idx_to_drug.get(drug_idx, "Unknown")
        is_true = drug_idx in true_drugs
        mark = "✓ True" if is_true else ""
        print(f"{rank:<6} {drug_id:<20} {score:<10.4f} {mark}")
        top20_list.append({"rank": rank, "drug_id": drug_id,
                           "score": round(score, 4), "true": is_true})

    # ── 8b. Failed RCT analysis ──────────────────────────────────────────
    rct_results = None
    if failed_rcts_file and os.path.exists(failed_rcts_file):
        failed_drugs = []
        success_drugs = []
        with open(failed_rcts_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if line.startswith('# ') and '\t' in line:
                        parts = line[2:].split('\t')
                        if len(parts) >= 5 and parts[4].strip() == 'success':
                            if parts[0].strip() == target_disease:
                                success_drugs.append(parts[1:5])
                    continue
                parts = line.split('\t')
                if len(parts) >= 5 and parts[0] == target_disease:
                    failed_drugs.append(tuple(parts[1:5]))

        if failed_drugs or success_drugs:
            print(f"\n{'=' * 60}")
            print(f"FAILED RCT ANALYSIS FOR {target_disease}")
            print(f"{'=' * 60}")
            rct_results = {"failed": [], "success": []}

            if failed_drugs:
                print(f"\n  Expected FAILURES (should rank LOW):")
                print(f"  {'Drug':<25s} {'Rank':>6} {'Score':>8} {'Trial':<20s} {'Outcome'}")
                print(f"  {'-' * 75}")
                failed_ranks = []
                for drug_id, name, trial, outcome in failed_drugs:
                    drug_idx = drug_mapping.get(drug_id)
                    if drug_idx is not None:
                        drug_idx = int(drug_idx)
                        rank = rank_map.get(drug_idx, total_drugs)
                        score = dict(drug_scores).get(drug_idx, 0.0)
                        print(f"  {name:<25s} {rank:>6} {score:>8.4f} {trial:<20s} {outcome}")
                        failed_ranks.append(rank)
                        rct_results["failed"].append({
                            "drug_id": drug_id, "name": name, "trial": trial,
                            "outcome": outcome, "rank": rank, "score": round(score, 4)
                        })
                    else:
                        print(f"  {name:<25s} {'N/A':>6} {'N/A':>8} {trial:<20s} (not in graph)")
                if failed_ranks:
                    print(f"\n  Failed drug median rank: {sorted(failed_ranks)[len(failed_ranks)//2]}")
                    print(f"  Failed drug mean rank:   {sum(failed_ranks)/len(failed_ranks):.1f}")

            if success_drugs:
                print(f"\n  Expected SUCCESSES (should rank HIGH):")
                print(f"  {'Drug':<25s} {'Rank':>6} {'Score':>8} {'Trial'}")
                print(f"  {'-' * 55}")
                success_ranks = []
                for drug_id, name, trial, outcome in success_drugs:
                    drug_idx = drug_mapping.get(drug_id)
                    if drug_idx is not None:
                        drug_idx = int(drug_idx)
                        rank = rank_map.get(drug_idx, total_drugs)
                        score = dict(drug_scores).get(drug_idx, 0.0)
                        print(f"  {name:<25s} {rank:>6} {score:>8.4f} {trial}")
                        success_ranks.append(rank)
                        rct_results["success"].append({
                            "drug_id": drug_id, "name": name, "trial": trial,
                            "rank": rank, "score": round(score, 4)
                        })
                    else:
                        print(f"  {name:<25s} {'N/A':>6} {'N/A':>8} {trial} (not in graph)")
                if success_ranks:
                    print(f"\n  Success drug median rank: {sorted(success_ranks)[len(success_ranks)//2]}")

            if 'failed_ranks' in dir() and failed_ranks and 'success_ranks' in dir() and success_ranks:
                mean_fail = sum(failed_ranks) / len(failed_ranks)
                mean_succ = sum(success_ranks) / len(success_ranks)
                print(f"\n  Rank separation ratio: {mean_fail / mean_succ:.1f}x "
                      f"(>1 means failed drugs rank lower = good)")

            print(f"\n{'=' * 60}\n")

    # ── 9. Save results ──────────────────────────────────────────────────
    results_dir = Path("results/ncnc_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results_json = {
        "model": "NCN",
        "reference": "Wang, X., Yang, H., & Zhang, M. (2024). Neural Common "
                     "Neighbor with Completion for Link Prediction. ICLR 2024.",
        "target_disease": target_disease,
        "timestamp": timestamp,
        "config": {
            "hidden": hidden, "gnn_layers": num_gnn_layers,
            "gnn_lr": gnn_lr, "pred_lr": pred_lr,
            "dropout": dropout, "beta": beta,
            "epochs": epochs, "batch_size": batch_size,
            "neg_ratio": neg_ratio, "seed": seed,
            "exclude_edge_types": exclude_edge_types,
            "no_node_features": no_node_features,
            "no_node_types": no_node_types,
        },
        "metrics": {
            "best_val_auc": round(best_val_auc, 4),
            "best_epoch": best_epoch,
            "hits_at_10": hits_at_10, "hits_at_20": hits_at_20,
            "hits_at_50": hits_at_50, "hits_at_100": hits_at_100,
            "bottom_10": bottom_10, "bottom_20": bottom_20,
            "bottom_50": bottom_50, "bottom_100": bottom_100,
            "total_true": n, "total_drugs": total_drugs,
            "median_rank": round(median_rank, 1),
            "mean_rank": round(mean_rank, 1),
            "mrr": round(mrr, 4),
        },
        "top20": top20_list,
        "all_ranks": sorted(ranks),
        "rct_analysis": rct_results,
    }

    json_path = results_dir / f"ncn_{target_disease}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    csv_path = results_dir / f"ncn_{target_disease}_{timestamp}_rankings.csv"
    with open(csv_path, "w") as f:
        f.write("rank,drug_idx,drug_id,score,is_true\n")
        for rank, (drug_idx, score) in enumerate(drug_scores, 1):
            drug_id = idx_to_drug.get(drug_idx, f"idx_{drug_idx}")
            is_true = drug_idx in true_drugs
            f.write(f"{rank},{drug_idx},{drug_id},{score:.6f},{is_true}\n")
    print(f"Full rankings CSV: {csv_path}")

    return results_json


def main():
    parser = argparse.ArgumentParser(
        description="NCN Leave-One-Out Evaluation (Wang et al., ICLR 2024)")
    # Disease
    parser.add_argument("--target-disease", type=str, default="EFO_0003854",
                        help="EFO/MONDO disease ID (default: Osteoporosis)")
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    # Model architecture
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--xdp", type=float, default=0.3,
                        help="Input feature dropout (paper default: 0.3)")
    parser.add_argument("--edge-dropout", type=float, default=0.3,
                        help="Edge dropout rate (paper default: 0.3)")
    parser.add_argument("--gnn-lr", type=float, default=0.0003,
                        help="GNN learning rate (paper default: 0.0003)")
    parser.add_argument("--pred-lr", type=float, default=0.0003,
                        help="Predictor learning rate (paper default: 0.0003)")
    parser.add_argument("--no-maskinput", action="store_true",
                        help="Disable target-link removal during training")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for CN features vs node pair features")
    parser.add_argument("--use-res", action="store_true", default=True,
                        help="Use residual connections in GCN (default: True)")
    parser.add_argument("--no-res", action="store_true",
                        help="Disable residual connections")
    parser.add_argument("--use-jk", action="store_true",
                        help="Use Jumping Knowledge connections")
    # Ablation flags
    parser.add_argument("--exclude-edge-types", type=str, nargs="+", default=None,
                        help="Edge types to remove: ppi, drug-gene, disease-gene, "
                             "disease-similarity")
    parser.add_argument("--no-node-features", action="store_true",
                        help="Disable original node features (use node types only)")
    parser.add_argument("--no-node-types", action="store_true",
                        help="Disable node-type indicators")
    # RCT validation
    parser.add_argument("--failed-rcts", type=str, default=None,
                        help="Path to failed_rcts.txt for RCT separation analysis")
    args = parser.parse_args()

    run_ncn_loo(
        target_disease=args.target_disease,
        epochs=args.epochs,
        hidden=args.hidden,
        num_gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        xdp=args.xdp,
        gnn_lr=args.gnn_lr,
        pred_lr=args.pred_lr,
        edge_dropout=args.edge_dropout,
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        patience=args.patience,
        seed=args.seed,
        beta=args.beta,
        use_res=not args.no_res,
        use_jk=args.use_jk,
        maskinput=not args.no_maskinput,
        exclude_edge_types=args.exclude_edge_types,
        no_node_features=args.no_node_features,
        no_node_types=args.no_node_types,
        failed_rcts_file=args.failed_rcts,
    )


if __name__ == "__main__":
    main()

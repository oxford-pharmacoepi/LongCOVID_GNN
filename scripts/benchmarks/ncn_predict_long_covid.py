#!/usr/bin/env python3
"""
NCN Drug Predictions for Long COVID (Cold-Start)

Trains NCN on ALL drug-disease edges (Long COVID has no known drug
associations to hold out), then scores every drug against Long COVID.
GWAS gene connections are wired into the graph before training.

Mirrors SEAL's predict_long_covid.py architecture and evaluation.

Reference:
  Wang, X., Yang, H., & Zhang, M. (2023).
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import model classes from ncnc_loo.py (same directory)
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))
from ncnc_loo import GCNEncoder, NCNPredictor, count_common_neighbors


# ═══════════════════════════════════════════════════════════════════════════
# Gene loading (same logic as SEAL's predict_long_covid.py)
# ═══════════════════════════════════════════════════════════════════════════

_CATEGORY_MAP = {
    "GWAS Lead Signals": "gwas",
    "Combinatorial OR": "combinatorial_or",
    "Prioritised Category 1": "cat1",
    "Prioritised Category 2": "cat2",
    "Prioritised Category 3": "cat3",
    "Selected Causal Drivers": "causal",
    "Combinatorial (symptom": "combinatorial",
    "User's Own Genes": "own",
}

ALL_CATEGORIES = {"gwas", "combinatorial_or", "cat1", "cat2", "cat3",
                  "causal", "combinatorial", "own"}


def _header_to_category(header: str) -> str:
    for prefix, cat in _CATEGORY_MAP.items():
        if prefix in header:
            return cat
    return "other"


def load_gwas_genes(path: str = "gwas_genes_long_covid.txt") -> list:
    genes = []
    current_category = "other"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# ==="):
                current_category = _header_to_category(line)
                continue
            if line.startswith("#"):
                continue
            gene_id = line.split()[0]
            genes.append((gene_id, current_category))
    return genes


def main():
    parser = argparse.ArgumentParser(
        description="NCN predictions for Long COVID (cold-start)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gnn-lr", type=float, default=0.003)
    parser.add_argument("--pred-lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--use-jk", action="store_true")
    parser.add_argument("--no-res", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--gwas-file", type=str, default="gwas_genes_long_covid.txt")
    parser.add_argument("--gene-categories", type=str, default="all",
                        help="Comma-separated: gwas, own, cat1, cat2, cat3, "
                             "causal, combinatorial, combinatorial_or, or 'all'")
    parser.add_argument("--max-drug-per-gene", type=int, default=None,
                        help="Exclude hub genes with more than N drug connections")
    parser.add_argument("--rct-drugs", type=str, nargs="*", default=None,
                        help="CHEMBL IDs of RCT drugs to look up in rankings")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load graph ────────────────────────────────────────────────────
    print("=" * 60)
    print("NCN DRUG PREDICTIONS FOR LONG COVID")
    print("=" * 60)

    graph_files = sorted(glob.glob("results/graph_*_processed_*.pt"))
    graph_path = graph_files[-1]
    print(f"Loading graph: {graph_path}")
    graph_data = torch.load(graph_path, weights_only=False)

    mappings_path = graph_path.replace(".pt", "_mappings")
    with open(f"{mappings_path}/drug_key_mapping.json") as f:
        drug_mapping = {k: int(v) for k, v in json.load(f).items()}
    with open(f"{mappings_path}/disease_key_mapping.json") as f:
        disease_mapping = {k: int(v) for k, v in json.load(f).items()}
    with open(f"{mappings_path}/gene_key_mapping.json") as f:
        gene_mapping = {k: int(v) for k, v in json.load(f).items()}

    idx_to_drug = {v: k for k, v in drug_mapping.items()}

    lc_id = "MONDO_0100320"
    lc_idx = int(disease_mapping[lc_id])
    print(f"Long COVID: {lc_id} (idx={lc_idx})")

    # ── 2. Wire GWAS genes to Long COVID ─────────────────────────────────
    gwas_genes_raw = load_gwas_genes(args.gwas_file)
    print(f"\nLoaded {len(gwas_genes_raw)} GWAS genes from {args.gwas_file}")

    if args.gene_categories.lower() == "all":
        allowed_cats = ALL_CATEGORIES
    else:
        allowed_cats = {c.strip() for c in args.gene_categories.split(",")}
    gwas_genes_raw = [(g, c) for g, c in gwas_genes_raw if c in allowed_cats]
    print(f"  Categories {sorted(allowed_cats)}: {len(gwas_genes_raw)} genes")

    # Build adjacency for hub detection
    drug_idx_set = set(drug_mapping.values())
    adj_check = None
    if args.max_drug_per_gene is not None:
        adj_check = defaultdict(set)
        src_a, dst_a = graph_data.edge_index
        for u, v in zip(src_a.tolist(), dst_a.tolist()):
            adj_check[u].add(v)
            adj_check[v].add(u)

    edge_index = graph_data.edge_index.clone()
    new_edges = []
    connected_genes = []
    excluded_hub_genes = []

    for gene_id, category in gwas_genes_raw:
        gene_idx = gene_mapping.get(gene_id)
        if gene_idx is None:
            continue
        gene_idx = int(gene_idx)

        if args.max_drug_per_gene is not None:
            drug_connections = len(adj_check.get(gene_idx, set()) & drug_idx_set)
            if drug_connections > args.max_drug_per_gene:
                excluded_hub_genes.append((gene_id, category, drug_connections))
                continue

        new_edges.append([lc_idx, gene_idx])
        new_edges.append([gene_idx, lc_idx])
        connected_genes.append((gene_id, category))

    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, new_edge_tensor], dim=1)
        print(f"  Wired {len(connected_genes)} genes → Long COVID "
              f"({len(new_edges)} edges)")
    if excluded_hub_genes:
        print(f"  Excluded {len(excluded_hub_genes)} hub genes "
              f"(>{args.max_drug_per_gene} drug connections)")

    num_nodes = graph_data.num_nodes
    node_features = graph_data.x.float()

    # Add node type features
    node_type = torch.zeros(num_nodes, dtype=torch.long)
    for idx in drug_mapping.values():
        node_type[idx] = 0
    for idx in gene_mapping.values():
        node_type[idx] = 1
    for idx in disease_mapping.values():
        node_type[idx] = 2
    node_type_features = torch.nn.functional.one_hot(node_type, 3).float()
    node_features = torch.cat([node_features, node_type_features], dim=1)

    # ── 3. Extract ALL drug-disease edges for training ───────────────────
    drug_indices = set(drug_mapping.values())
    disease_indices = set(disease_mapping.values())

    drug_disease_edges = set()
    src, dst = edge_index
    for i in range(edge_index.shape[1]):
        u, v = src[i].item(), dst[i].item()
        if u in drug_indices and v in disease_indices:
            drug_disease_edges.add((u, v))
        elif v in drug_indices and u in disease_indices:
            drug_disease_edges.add((v, u))

    # For LC cold-start: train on ALL drug-disease edges
    all_dd_list = list(drug_disease_edges)
    random.shuffle(all_dd_list)
    val_size = max(1, len(all_dd_list) // 10)
    val_edges = all_dd_list[:val_size]
    train_edges = all_dd_list[val_size:]

    print(f"\n  Total drug-disease edges: {len(drug_disease_edges)}")
    print(f"  Train: {len(train_edges)}, Val: {len(val_edges)}")

    # ── 4. Build adjacency and CN counts ─────────────────────────────────
    train_edge_index = edge_index.to(device)
    train_ei_cpu = train_edge_index.cpu()
    adj_scipy = sp.csr_matrix(
        (np.ones(train_ei_cpu.shape[1]),
         (train_ei_cpu[0].numpy(), train_ei_cpu[1].numpy())),
        shape=(num_nodes, num_nodes)
    )
    adj_scipy = adj_scipy + adj_scipy.T
    adj_scipy[adj_scipy > 1] = 1

    all_drug_list = sorted(drug_indices)
    all_disease_list = list(disease_indices)
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

    neg_train = sample_negatives(len(train_edges) * args.neg_ratio)
    neg_val = sample_negatives(len(val_edges) * args.neg_ratio)

    def precompute_cn(edge_list):
        if not edge_list:
            return torch.zeros(0, 1)
        src_t = torch.tensor([e[0] for e in edge_list])
        dst_t = torch.tensor([e[1] for e in edge_list])
        cn = count_common_neighbors(adj_scipy, src_t, dst_t)
        return cn.reshape(-1, 1)

    print("Precomputing common neighbors...")
    cn_train_pos = precompute_cn(train_edges).to(device)
    cn_train_neg = precompute_cn(neg_train).to(device)
    cn_val_pos = precompute_cn(val_edges).to(device)
    cn_val_neg = precompute_cn(neg_val).to(device)

    tps = torch.tensor([e[0] for e in train_edges], device=device)
    tpd = torch.tensor([e[1] for e in train_edges], device=device)
    tns = torch.tensor([e[0] for e in neg_train], device=device)
    tnd = torch.tensor([e[1] for e in neg_train], device=device)
    vps = torch.tensor([e[0] for e in val_edges], device=device)
    vpd = torch.tensor([e[1] for e in val_edges], device=device)
    vns = torch.tensor([e[0] for e in neg_val], device=device)
    vnd = torch.tensor([e[1] for e in neg_val], device=device)

    x = node_features.to(device)

    # ── 5. Model ─────────────────────────────────────────────────────────
    model = GCNEncoder(
        x.shape[1], args.hidden, args.gnn_layers, args.dropout,
        use_res=not args.no_res, use_jk=args.use_jk
    ).to(device)
    predictor = NCNPredictor(
        args.hidden, args.hidden, args.dropout, beta=args.beta
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.gnn_lr},
        {'params': predictor.parameters(), 'lr': args.pred_lr},
    ])

    total_params = sum(p.numel() for p in model.parameters()) + \
                   sum(p.numel() for p in predictor.parameters())
    print(f"\n  Model params: {total_params:,}")

    # ── 6. Training ──────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_epoch = 0
    best_state = None
    no_improve = 0
    bs = args.batch_size

    print(f"\nTraining NCN for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        predictor.train()
        n_train = len(train_edges)
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            h = model(x, train_edge_index)
            pos_score = predictor(h, cn_train_pos[idx], tps[idx], tpd[idx])
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_idx = torch.randint(0, len(neg_train), (len(idx),), device=device)
            neg_score = predictor(h, cn_train_neg[neg_idx], tns[neg_idx], tnd[neg_idx])
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss = pos_loss + neg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        predictor.eval()
        with torch.no_grad():
            h = model(x, train_edge_index)
            vp = predictor(h, cn_val_pos, vps, vpd).cpu()
            vn = predictor(h, cn_val_neg, vns, vnd).cpu()
            labels = torch.cat([torch.ones(vp.size(0)), torch.zeros(vn.size(0))])
            scores = torch.cat([vp, vn])
            val_auc = roc_auc_score(labels.numpy(), scores.numpy())

        avg_loss = total_loss / max(n_batches, 1)
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
        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # ── 7. Rank all drugs against Long COVID ─────────────────────────────
    model.load_state_dict(best_state['model'])
    predictor.load_state_dict(best_state['pred'])
    model = model.to(device)
    predictor = predictor.to(device)
    model.eval()
    predictor.eval()

    print(f"\nBest model: epoch {best_epoch}, val AUC = {best_val_auc:.4f}")
    print("Ranking all drugs against Long COVID...")

    all_scores = []
    with torch.no_grad():
        h = model(x, train_edge_index)
        for i in range(0, len(all_drug_list), bs):
            batch = all_drug_list[i:i + bs]
            src_t = torch.tensor(batch, device=device)
            dst_t = torch.tensor([lc_idx] * len(batch), device=device)
            cn = count_common_neighbors(
                adj_scipy, torch.tensor(batch),
                torch.tensor([lc_idx] * len(batch))
            ).reshape(-1, 1).to(device)
            s = predictor(h, cn, src_t, dst_t)
            all_scores.extend(zip(batch, s.cpu().tolist()))

    drug_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)

    # ── 8. Output ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"NCN LONG COVID TOP-{args.top_k} PREDICTIONS (seed={args.seed})")
    print(f"{'=' * 60}")
    print(f"  Val AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"  Gene config: categories={args.gene_categories}, "
          f"hub_cap={args.max_drug_per_gene}, "
          f"wired={len(connected_genes)}")
    print(f"\n{'Rank':<6} {'Drug ID':<20} {'Score':<10}")
    print("-" * 40)
    for rank, (drug_idx, score) in enumerate(drug_scores[:args.top_k], 1):
        drug_id = idx_to_drug.get(drug_idx, "Unknown")
        print(f"{rank:<6} {drug_id:<20} {score:<10.4f}")

    # Check RCT drugs if provided
    rank_map = {d: r for r, (d, _) in enumerate(drug_scores, 1)}
    if args.rct_drugs:
        print(f"\n{'=' * 60}")
        print("RCT DRUG RANKINGS")
        print(f"{'=' * 60}")
        for chembl_id in args.rct_drugs:
            drug_idx = drug_mapping.get(chembl_id)
            if drug_idx is not None:
                rank = rank_map.get(int(drug_idx), "N/A")
                score = dict(drug_scores).get(int(drug_idx), 0.0)
                print(f"  {chembl_id}: rank={rank}, score={score:.4f}")
            else:
                print(f"  {chembl_id}: not in graph")

    # Load trial drugs for automatic check
    trial_drugs_path = "medicines_tested.txt"
    trial_drugs = {}
    if os.path.exists(trial_drugs_path):
        with open(trial_drugs_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    trial_drugs[parts[0]] = parts[1] if len(parts) > 1 else parts[0]

    if trial_drugs:
        print(f"\n{'=' * 60}")
        print(f"CLINICAL TRIAL DRUG RANKINGS ({len(trial_drugs)} drugs)")
        print(f"{'=' * 60}")
        found_in_top50 = 0
        trial_ranks = []
        print(f"  {'Drug':<25s} {'Rank':>6} {'Score':>8}")
        print(f"  {'-' * 45}")
        for chembl_id, name in sorted(trial_drugs.items()):
            drug_idx = drug_mapping.get(chembl_id)
            if drug_idx is not None:
                rank = rank_map.get(int(drug_idx), len(all_drug_list))
                score = dict(drug_scores).get(int(drug_idx), 0.0)
                in50 = " ★" if rank <= 50 else ""
                print(f"  {name:<25s} {rank:>6} {score:>8.4f}{in50}")
                trial_ranks.append(rank)
                if rank <= 50:
                    found_in_top50 += 1
        print(f"\n  Trial drugs in top 50: {found_in_top50}/{len(trial_ranks)}")
        if trial_ranks:
            print(f"  Median rank: {sorted(trial_ranks)[len(trial_ranks)//2]}")

    # ── 9. Save results ──────────────────────────────────────────────────
    results_dir = Path("results/ncn_long_covid")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    gene_config = args.gene_categories.replace(",", "_")
    hub_str = f"_hub{args.max_drug_per_gene}" if args.max_drug_per_gene else ""

    results = {
        "model": "NCN",
        "target": "Long COVID (MONDO_0100320)",
        "timestamp": timestamp,
        "seed": args.seed,
        "val_auc": round(best_val_auc, 4),
        "best_epoch": best_epoch,
        "gene_config": {
            "categories": args.gene_categories,
            "max_drug_per_gene": args.max_drug_per_gene,
            "wired_genes": len(connected_genes),
            "excluded_hub_genes": len(excluded_hub_genes),
        },
        "config": {
            "hidden": args.hidden, "gnn_layers": args.gnn_layers,
            "dropout": args.dropout, "beta": args.beta,
            "gnn_lr": args.gnn_lr, "pred_lr": args.pred_lr,
            "epochs": args.epochs, "batch_size": args.batch_size,
        },
        "top_drugs": [
            {"rank": r, "drug_id": idx_to_drug.get(d, f"idx_{d}"),
             "score": round(s, 4)}
            for r, (d, s) in enumerate(drug_scores[:100], 1)
        ],
        "trial_drug_ranks": {
            name: rank_map.get(int(drug_mapping.get(cid, -1)), -1)
            for cid, name in trial_drugs.items()
            if drug_mapping.get(cid) is not None
        } if trial_drugs else {},
    }

    fname = f"ncn_lc_{gene_config}{hub_str}_seed{args.seed}_{timestamp}"
    json_path = results_dir / f"{fname}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    csv_path = results_dir / f"{fname}_rankings.csv"
    with open(csv_path, "w") as f:
        f.write("rank,drug_idx,drug_id,score\n")
        for rank, (drug_idx, score) in enumerate(drug_scores, 1):
            drug_id = idx_to_drug.get(drug_idx, f"idx_{drug_idx}")
            f.write(f"{rank},{drug_idx},{drug_id},{score:.6f}\n")
    print(f"Full rankings: {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SEAL Drug Predictions for Long COVID

Trains a SEAL model on ALL drug-disease edges in the graph (Long COVID has
no known drug associations to hold out), then scores every drug against
Long COVID.  Before training, the GWAS gene connections for Long COVID are
wired into the graph so the model can learn meaningful subgraph structure.

Usage:
    uv run python scripts/seal/predict_long_covid.py --conv sage --use-jk --hidden 32
"""

import argparse
import glob
import json
import os
from collections import defaultdict
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models_seal import SEALDataset, SEALModel

# Import negative sampling from train_loo (same directory)
SEAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SEAL_DIR))
from train_loo import sample_negatives_mixed, sample_negatives_random


# Category labels used in gwas_genes_long_covid.txt
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

# Valid category names for --gene-categories
ALL_CATEGORIES = {"gwas", "combinatorial_or", "cat1", "cat2", "cat3",
                  "causal", "combinatorial", "own"}


def _header_to_category(header: str) -> str:
    """Map a section header from the gene file to a short category label."""
    for prefix, cat in _CATEGORY_MAP.items():
        if prefix in header:
            return cat
    return "other"


def load_gwas_genes(path: str = "gwas_genes_long_covid.txt") -> list[tuple[str, str]]:
    """Read GWAS gene ENSG IDs from file, returning (gene_id, category) tuples."""
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
            gene_id = line.split()[0]  # handle inline comments
            genes.append((gene_id, current_category))
    return genes


def main():
    parser = argparse.ArgumentParser(description="SEAL predictions for Long COVID")
    parser.add_argument("--conv", type=str, default="sage", choices=["sage", "gat", "gin"])
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--pooling", type=str, default="mean+max")
    parser.add_argument("--use-jk", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--neg-strategy", type=str, default="mixed")
    parser.add_argument("--hard-ratio", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=50, help="Number of top predictions to show")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nodes-per-hop", type=int, default=200)
    parser.add_argument("--gwas-file", type=str, default="gwas_genes_long_covid.txt")
    parser.add_argument("--rct-drugs", type=str, nargs="*", default=None,
                        help="CHEMBL IDs of RCT drugs to look up in rankings")
    parser.add_argument("--max-drug-per-gene", type=int, default=None,
                        help="Exclude GWAS genes with more than N drug connections "
                             "(prevents hub genes from dominating). None = no cap.")
    parser.add_argument("--gene-categories", type=str, default="all",
                        help="Comma-separated gene categories to include, or 'all'. "
                             "Categories: gwas, own, cat1, cat2, cat3, causal, "
                             "combinatorial, combinatorial_or")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. Load graph and mappings ---
    print("=" * 60)
    print("SEAL DRUG PREDICTIONS FOR LONG COVID")
    print("=" * 60)

    graph_files = sorted(glob.glob("results/graph_*_processed_*.pt"))
    graph_path = graph_files[-1]
    print(f"Loading graph: {graph_path}")
    graph_data = torch.load(graph_path, weights_only=False)

    mappings_path = graph_path.replace(".pt", "_mappings")
    with open(f"{mappings_path}/drug_key_mapping.json") as f:
        drug_mapping = json.load(f)
    with open(f"{mappings_path}/disease_key_mapping.json") as f:
        disease_mapping = json.load(f)

    # Gene mapping
    gene_mapping = {}
    gene_map_file = f"{mappings_path}/gene_key_mapping.json"
    if os.path.exists(gene_map_file):
        with open(gene_map_file) as f:
            gene_mapping = json.load(f)

    lc_id = "MONDO_0100320"
    lc_idx = int(disease_mapping[lc_id])
    print(f"Long COVID: {lc_id} (idx={lc_idx})")

    # --- 2. Wire up GWAS genes (with optional filtering) ---
    gwas_genes_raw = load_gwas_genes(args.gwas_file)
    print(f"\nLoaded {len(gwas_genes_raw)} GWAS genes from {args.gwas_file}")

    # Category filter
    if args.gene_categories.lower() == "all":
        allowed_cats = ALL_CATEGORIES
    else:
        allowed_cats = {c.strip() for c in args.gene_categories.split(",")}
        unknown = allowed_cats - ALL_CATEGORIES
        if unknown:
            print(f"  WARNING: Unknown categories ignored: {unknown}")
            allowed_cats -= unknown
    before = len(gwas_genes_raw)
    gwas_genes_raw = [(g, c) for g, c in gwas_genes_raw if c in allowed_cats]
    print(f"  Gene categories {sorted(allowed_cats)}: {before} → {len(gwas_genes_raw)} genes")

    # Build adjacency for hub-gene detection (if cap is active)
    drug_idx_set_early = set(int(v) for v in drug_mapping.values())
    adj_early = None
    if args.max_drug_per_gene is not None:
        adj_early = defaultdict(set)
        src_a, dst_a = graph_data.edge_index
        for u, v in zip(src_a.tolist(), dst_a.tolist()):
            adj_early[u].add(v)
            adj_early[v].add(u)

    edge_index = graph_data.edge_index.clone()
    new_edges = []
    connected_genes = []
    missing_genes = []
    excluded_hub_genes = []

    for gene_id, category in gwas_genes_raw:
        gene_idx = gene_mapping.get(gene_id)
        if gene_idx is None:
            missing_genes.append(gene_id)
            continue
        gene_idx = int(gene_idx)

        # Hub gene cap: skip genes with too many drug connections
        if args.max_drug_per_gene is not None:
            drug_connections = len(adj_early.get(gene_idx, set()) & drug_idx_set_early)
            if drug_connections > args.max_drug_per_gene:
                excluded_hub_genes.append((gene_id, category, drug_connections))
                continue

        new_edges.append([lc_idx, gene_idx])
        new_edges.append([gene_idx, lc_idx])
        connected_genes.append((gene_id, category))

    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, new_edge_tensor], dim=1)
        print(f"  Wired {len(connected_genes)} genes to Long COVID ({len(new_edges)} edges)")
        if missing_genes:
            print(f"  {len(missing_genes)} genes not in graph: {missing_genes[:5]}")
        if excluded_hub_genes:
            print(f"  Excluded {len(excluded_hub_genes)} hub genes (>{args.max_drug_per_gene} drug connections):")
            for gid, cat, dc in excluded_hub_genes:
                comment = [l for l in open(args.gwas_file) if gid in l]
                sym = comment[0].split('#')[1].split('|')[0].strip() if comment else gid
                print(f"    {sym} ({gid}): {dc} drug connections [{cat}]")
    else:
        print("  WARNING: No GWAS genes found in graph!")

    # Count Long COVID's new degree
    lc_neighbors = set()
    for i in range(edge_index.shape[1]):
        s, d = edge_index[0, i].item(), edge_index[1, i].item()
        if s == lc_idx:
            lc_neighbors.add(d)
        elif d == lc_idx:
            lc_neighbors.add(s)
    print(f"  Long COVID now has {len(lc_neighbors)} neighbours")

    # --- 3. Build node features ---
    node_features = graph_data.x.clone()
    num_nodes = node_features.shape[0]
    drug_indices = set(int(v) for v in drug_mapping.values())
    disease_indices = set(int(v) for v in disease_mapping.values())
    gene_indices = set(int(v) for v in gene_mapping.values()) if gene_mapping else set()

    is_drug = torch.zeros(num_nodes, 1)
    is_disease = torch.zeros(num_nodes, 1)
    is_gene = torch.zeros(num_nodes, 1)
    for idx in drug_indices:
        is_drug[idx] = 1.0
    for idx in disease_indices:
        is_disease[idx] = 1.0
    for idx in gene_indices:
        is_gene[idx] = 1.0
    node_features = torch.cat([node_features, is_drug, is_disease, is_gene], dim=1)
    feat_dim = node_features.shape[1]
    print(f"  Node features: {feat_dim} dims")

    # --- 4. Extract drug-disease edges for training ---
    drug_idx_set = set(int(v) for v in drug_mapping.values())
    disease_idx_set = set(int(v) for v in disease_mapping.values())

    dd_edges = set()
    for i in range(edge_index.shape[1]):
        s, d = edge_index[0, i].item(), edge_index[1, i].item()
        if s in drug_idx_set and d in disease_idx_set:
            dd_edges.add((s, d))
        elif s in disease_idx_set and d in drug_idx_set:
            dd_edges.add((d, s))  # normalise as (drug,disease)

    dd_edges = list(dd_edges)
    print(f"\nDrug-disease edges for training: {len(dd_edges)}")

    # Long COVID has 0 drug edges — all dd_edges are training data
    lc_drug_edges = [(d, dis) for d, dis in dd_edges if dis == lc_idx]
    train_edges = [(d, dis) for d, dis in dd_edges if dis != lc_idx]
    print(f"  Long COVID drug edges (held out): {len(lc_drug_edges)}")
    print(f"  Training edges: {len(train_edges)}")

    # --- 5. Train/Val split ---
    np.random.shuffle(train_edges)
    val_size = int(0.1 * len(train_edges))
    val_edges = train_edges[:val_size]
    actual_train = train_edges[val_size:]
    print(f"  Train split: {len(actual_train)}, Val split: {val_size}")

    # --- 6. Build adjacency dict (before neg sampling so we can reuse it) ---
    print("\nBuilding adjacency dict...")
    adj_dict = defaultdict(set)
    src_arr = edge_index[0].tolist()
    dst_arr = edge_index[1].tolist()
    for u, v in zip(src_arr, dst_arr):
        adj_dict[u].add(v)
        adj_dict[v].add(u)
    print(f"  Adjacency built for {len(adj_dict)} nodes")

    # --- 7. Sample negatives ---
    print(f"\nSampling negatives (strategy={args.neg_strategy}, ratio=1:{args.neg_ratio})...")

    all_pos_set = set(dd_edges) | set((d, s) for s, d in dd_edges)
    all_drug_list = list(drug_idx_set)
    all_disease_list = list(disease_idx_set)
    lc_drug_edges_set = set(lc_drug_edges)

    n_train_neg = args.neg_ratio * len(actual_train)
    if args.neg_strategy == "mixed":
        # Fast hard-neg sampling using pre-built adj_dict
        # Only check diseases that share common neighbours with each drug
        n_hard = int(n_train_neg * args.hard_ratio)
        n_random = n_train_neg - n_hard
        forbidden = all_pos_set | lc_drug_edges_set

        print("    Computing common neighbours for hard negatives (fast)...")
        candidates = []
        disease_idx_set_local = disease_idx_set - {lc_idx}
        for drug in tqdm(all_drug_list, desc="    Hard neg scan", leave=False):
            drug_nb = adj_dict.get(drug, set())
            if not drug_nb:
                continue
            # Only check diseases reachable via shared neighbours
            candidate_diseases = set()
            for nb in drug_nb:
                for nb2 in adj_dict.get(nb, set()):
                    if nb2 in disease_idx_set_local:
                        candidate_diseases.add(nb2)
            for disease in candidate_diseases:
                if (drug, disease) in forbidden:
                    continue
                cn = len(drug_nb & adj_dict.get(disease, set()))
                if cn >= 1:
                    candidates.append((drug, disease, cn))

        candidates.sort(key=lambda x: x[2], reverse=True)
        print(f"    Found {len(candidates)} hard negative candidates (CN >= 1)")
        hard_negs = [(d, dis) for d, dis, _ in candidates[:n_hard]]

        # Pad with random if needed
        hard_set = set(hard_negs)
        rand_negs = sample_negatives_random(
            positive_set=all_pos_set | hard_set,
            all_drug_list=all_drug_list,
            all_disease_list=all_disease_list,
            num_samples=n_random,
            exclude_disease=lc_idx,
            future_positives=lc_drug_edges_set,
        )
        neg_edges = hard_negs + rand_negs
        random.shuffle(neg_edges)
    else:
        neg_edges = sample_negatives_random(
            positive_set=all_pos_set,
            all_drug_list=all_drug_list,
            all_disease_list=all_disease_list,
            num_samples=n_train_neg,
            exclude_disease=lc_idx,
            future_positives=lc_drug_edges_set,
        )
    print(f"  Sampled {len(neg_edges)} negative edges")

    n_val_neg = 3 * len(val_edges)
    val_neg = sample_negatives_random(
        positive_set=all_pos_set | set(neg_edges),
        all_drug_list=all_drug_list,
        all_disease_list=all_disease_list,
        num_samples=n_val_neg,
        exclude_disease=lc_idx,
        future_positives=lc_drug_edges_set,
    )

    # --- 8. Create datasets ---
    train_pairs = [(s, d) for s, d in actual_train] + neg_edges
    train_labels = [1] * len(actual_train) + [0] * len(neg_edges)

    val_pairs = [(s, d) for s, d in val_edges] + val_neg
    val_labels = [1] * len(val_edges) + [0] * len(val_neg)

    print(f"\nCreating SEAL datasets...")
    print(f"  Train: {len(train_pairs)} pairs ({len(actual_train)} pos, {len(neg_edges)} neg)")
    print(f"  Val: {len(val_pairs)} pairs ({len(val_edges)} pos, {len(val_neg)} neg)")

    max_z = 50

    train_dataset = SEALDataset(
        root="results/seal_tmp/long_covid",
        pairs=train_pairs, labels=train_labels,
        edge_index=edge_index, node_features=node_features,
        num_hops=args.hops, adj_dict=adj_dict,
        max_nodes_per_hop=args.max_nodes_per_hop, max_z=max_z,
    )
    val_dataset = SEALDataset(
        root="results/seal_tmp/long_covid",
        pairs=val_pairs, labels=val_labels,
        edge_index=edge_index, node_features=node_features,
        num_hops=args.hops, adj_dict=adj_dict,
        max_nodes_per_hop=args.max_nodes_per_hop, max_z=max_z,
    )

    # Pre-cache subgraphs in parallel
    print("\nPre-caching subgraphs...")
    train_dataset.precache_parallel(num_workers=0)
    val_dataset.precache_parallel(num_workers=0)

    # --- 9. Train ---
    # Determine in_channels from actual data (safest approach)
    first_batch = train_dataset[0]
    in_channels = first_batch.x.shape[1]
    model = SEALModel(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_layers=3,
        conv_type=args.conv,
        pooling=args.pooling,
        use_jk=args.use_jk,
    )
    print(f"\nModel: {model.__class__.__name__}")
    print(f"  Conv: {args.conv}, Hidden: {args.hidden}, JK: {args.use_jk}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    best_val_auc = 0.0
    best_state = None
    patience = 10
    no_improve = 0

    print(f"\nTraining for up to {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimiser.zero_grad()
            out = model(batch).squeeze()
            loss = criterion(out, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(train_dataset)

        if epoch % 5 == 0:
            # Validate
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = torch.sigmoid(model(batch).squeeze())
                    all_preds.extend(out.tolist())
                    all_labels.extend(batch.y.tolist())
            val_auc = roc_auc_score(all_labels, all_preds)

            improved = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience * 5:
                print(f"  Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f} (best: {best_val_auc:.4f})")
                print(f"  Early stopping at epoch {epoch}")
                break
            print(f"  Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f} (best: {best_val_auc:.4f})")
        scheduler.step()

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Best Val AUC: {best_val_auc:.4f}")

    # --- 10. Score ALL drugs against Long COVID ---
    print(f"\nScoring all {len(drug_mapping)} drugs against Long COVID...")
    all_drug_ids = list(drug_mapping.keys())
    all_drug_idx = [int(v) for v in drug_mapping.values()]
    eval_pairs = [(didx, lc_idx) for didx in all_drug_idx]

    eval_dataset = SEALDataset(
        root="results/seal_tmp/long_covid",
        pairs=eval_pairs, labels=[0] * len(eval_pairs),
        edge_index=edge_index, node_features=node_features,
        num_hops=args.hops, adj_dict=adj_dict,
        max_nodes_per_hop=args.max_nodes_per_hop, max_z=max_z,
    )
    eval_dataset.precache_parallel(num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0)

    scores = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Scoring"):
            out = torch.sigmoid(model(batch).squeeze())
            if out.dim() == 0:
                scores.append(out.item())
            else:
                scores.extend(out.tolist())

    # Build ranked results
    drug_scores = list(zip(all_drug_ids, scores))
    drug_scores.sort(key=lambda x: x[1], reverse=True)

    # --- 11. Resolve CHEMBL names ---
    print(f"\nLooking up drug names from ChEMBL...")
    chembl_names = {}
    try:
        name_file = f"{mappings_path}/drug_names.json"
        if os.path.exists(name_file):
            with open(name_file) as f:
                chembl_names = json.load(f)
    except Exception:
        pass

    # If no names file, try to build from raw data
    if not chembl_names:
        for raw_dir in ["raw_data", "processed_data"]:
            mol_file = f"{raw_dir}/molecule.json"
            if os.path.exists(mol_file):
                try:
                    with open(mol_file) as f:
                        for line in f:
                            mol = json.loads(line.strip())
                            cid = mol.get("id", "")
                            name = mol.get("name", mol.get("prefName", ""))
                            if cid and name:
                                chembl_names[cid] = name
                except Exception:
                    pass
                break

    # --- 12. Print top-K ---
    print(f"\n{'='*70}")
    print(f"TOP {args.top_k} DRUG CANDIDATES FOR LONG COVID")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Drug ID':<20} {'Name':<35} {'Score':<10}")
    print("-" * 70)
    for i, (drug_id, score) in enumerate(drug_scores[:args.top_k], 1):
        name = chembl_names.get(drug_id, "")
        print(f"{i:<6} {drug_id:<20} {name:<35} {score:.4f}")

    # --- 13. Look up RCT drugs ---
    rct_drugs = args.rct_drugs or []
    rct_names = {}  # ChEMBL ID -> drug name from file
    # Default: read from medicines_tested.txt
    if not rct_drugs:
        rct_file = os.path.join(PROJECT_ROOT, "medicines_tested.txt")
        if os.path.exists(rct_file):
            with open(rct_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("drug_id") or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        rct_drugs.append(parts[0])
                        rct_names[parts[0]] = parts[1]
            print(f"\n  Loaded {len(rct_drugs)} RCT drugs from {rct_file}")
        else:
            print(f"\n  WARNING: {rct_file} not found, no RCT drugs to check")

    if rct_drugs:
        drug_rank = {drug_id: rank + 1 for rank, (drug_id, _) in enumerate(drug_scores)}
        drug_score_map = {drug_id: score for drug_id, score in drug_scores}

        print(f"\n{'='*70}")
        print("LONG COVID RCT DRUGS — WHERE DO THEY RANK?")
        print(f"{'='*70}")
        print(f"{'Rank':<8} {'Drug ID':<20} {'Name':<30} {'Score':<10}")
        print("-" * 70)
        for rct_id in rct_drugs:
            rank = drug_rank.get(rct_id, "N/A")
            score = drug_score_map.get(rct_id, 0.0)
            name = chembl_names.get(rct_id, "") or rct_names.get(rct_id, "")
            flag = "⭐" if isinstance(rank, int) and rank <= 50 else ""
            print(f"{rank:<8} {rct_id:<20} {name:<30} {score:.4f} {flag}")

    # --- 14. Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/long_covid", exist_ok=True)

    results = {
        "timestamp": timestamp,
        "disease": lc_id,
        "disease_name": "Long COVID (Post-COVID-19 condition)",
        "config": {
            "conv_type": args.conv,
            "hidden": args.hidden,
            "hops": args.hops,
            "use_jk": args.use_jk,
            "pooling": args.pooling,
            "epochs": args.epochs,
            "neg_strategy": args.neg_strategy,
            "neg_ratio": args.neg_ratio,
            "hard_ratio": args.hard_ratio,
            "seed": args.seed,
            "gene_categories": args.gene_categories,
            "max_drug_per_gene": args.max_drug_per_gene,
            "gwas_genes_connected": len(connected_genes),
            "gwas_genes_excluded_hubs": len(excluded_hub_genes),
            "gwas_genes_missing": len(missing_genes),
            "best_val_auc": best_val_auc,
        },
        "total_drugs_ranked": len(drug_scores),
        "top_predictions": [
            {"rank": i + 1, "drug_id": did, "name": chembl_names.get(did, ""), "score": round(s, 6)}
            for i, (did, s) in enumerate(drug_scores[:100])
        ],
        "rct_drug_ranks": {
            rct_id: {"rank": drug_rank.get(rct_id), "score": round(drug_score_map.get(rct_id, 0.0), 6),
                      "name": chembl_names.get(rct_id, "")}
            for rct_id in rct_drugs
        } if rct_drugs else {},
    }

    save_path = f"results/long_covid/seal_long_covid_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()

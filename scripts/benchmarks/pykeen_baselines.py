#!/usr/bin/env python3
"""
KG Embedding Baselines (TransE, DistMult, RotatE) via PyKEEN.

Evaluates knowledge graph embedding methods on the same Osteoporosis LOO
task used in the SEAL tournament, enabling direct comparison.

Usage:
    uv run python scripts/benchmarks/pykeen_baselines.py

The script:
  1. Loads the SAME saved graph (results/graph_*.pt) that SEAL uses
  2. Extracts drug-disease edges the same way SEAL does (from edge_index)
  3. Converts to PyKEEN triple format (head, relation, tail)
  4. Trains TransE, DistMult, and RotatE
  5. Evaluates using the same LOO protocol: hold out all Osteoporosis
     drug edges, train on remaining, rank held-out drugs
"""

import os
import sys
import json
import time
import glob
import torch
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def build_triples_from_saved_graph():
    """
    Build triples from the SAME saved graph that SEAL uses.

    Loads results/graph_*.pt and its companion mappings directory,
    exactly matching the SEAL LOO evaluation protocol.

    Returns:
        triples: np.ndarray of shape (N, 3) with string IDs
        key_mappings: dict of {node_type: {string_id: global_idx}}
        drug_indices: set of global drug indices
        disease_indices: set of global disease indices
    """
    # Find graph file (same as SEAL train_loo.py line 224)
    graph_files = sorted(glob.glob(str(project_root / "results" / "graph_*.pt")))
    if not graph_files:
        raise FileNotFoundError("No graph found in results/")
    graph_path = graph_files[-1]
    mappings_path = graph_path.replace(".pt", "_mappings")
    print(f"Loading graph: {graph_path}")

    graph_data = torch.load(graph_path, weights_only=False)
    edge_index = graph_data.edge_index
    print(f"  Edge index: {edge_index.shape[1]} edges (undirected)")

    # Load mappings
    mapping_files = {
        'drug': 'drug_key_mapping.json',
        'drug_type': 'drug_type_key_mapping.json',
        'gene': 'gene_key_mapping.json',
        'reactome': 'reactome_key_mapping.json',
        'disease': 'disease_key_mapping.json',
        'therapeutic_area': 'therapeutic_area_key_mapping.json',
    }

    key_mappings = {}
    idx_to_id = {}
    idx_to_type = {}

    for name, fname in mapping_files.items():
        fpath = os.path.join(mappings_path, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                mapping = json.load(f)
            key_mappings[name] = mapping
            for string_id, global_idx in mapping.items():
                idx_to_id[global_idx] = f"{name}:{string_id}"
                idx_to_type[global_idx] = name

    drug_indices = set(key_mappings.get('drug', {}).values())
    disease_indices = set(key_mappings.get('disease', {}).values())
    gene_indices = set(key_mappings.get('gene', {}).values())

    print(f"  Nodes: {len(idx_to_id)} total "
          f"({len(drug_indices)} drugs, {len(disease_indices)} diseases, "
          f"{len(gene_indices)} genes)")

    # Convert edge_index to triples, inferring relation type from node types
    # Deduplicate (undirected graph has both directions)
    seen_edges = set()
    all_triples = []

    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()

        # Canonical ordering for deduplication
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        src_type = idx_to_type.get(u, 'unknown')
        dst_type = idx_to_type.get(v, 'unknown')
        src_id = idx_to_id.get(u, f"node:{u}")
        dst_id = idx_to_id.get(v, f"node:{v}")

        # Determine relation type from node types
        type_pair = tuple(sorted([src_type, dst_type]))
        rel_type_map = {
            ('disease', 'drug'): 'drug-disease',
            ('drug', 'drug_type'): 'drug-drugtype',
            ('drug', 'gene'): 'drug-gene',
            ('disease', 'gene'): 'disease-gene',
            ('gene', 'reactome'): 'gene-reactome',
            ('disease', 'therapeutic_area'): 'disease-therapeutic',
            ('gene', 'gene'): 'gene-gene',
            ('disease', 'disease'): 'disease-disease',
        }
        rel = rel_type_map.get(type_pair, f"{src_type}-{dst_type}")

        # For drug-disease: ensure drug is head, disease is tail
        if rel == 'drug-disease':
            if src_type == 'drug':
                all_triples.append((src_id, rel, dst_id))
            else:
                all_triples.append((dst_id, rel, src_id))
        else:
            all_triples.append((src_id, rel, dst_id))

    triples_array = np.array(all_triples, dtype=str)

    # Count by relation type
    rel_counts = Counter(t[1] for t in all_triples)
    print("  Triples by relation type (deduplicated):")
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel}: {count}")
    print(f"  Total deduplicated triples: {len(triples_array)}")

    return triples_array, key_mappings, drug_indices, disease_indices


def run_loo_evaluation(triples, key_mappings,
                       disease_id="EFO_0003854",  # Osteoporosis
                       models_to_run=("TransE", "DistMult", "RotatE"),
                       epochs=100, embedding_dim=64):
    """
    Run LOO evaluation: hold out all drug-disease edges for the target disease,
    train KGE models, then rank all drugs for that disease.
    """
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    disease_node = f"disease:{disease_id}"

    # Identify held-out edges (drug-disease edges for target disease)
    held_out_mask = []
    held_out_drugs = set()

    for i, (h, r, t) in enumerate(triples):
        if r == 'drug-disease' and t == disease_node:
            held_out_mask.append(i)
            held_out_drugs.add(h)

    held_out_mask = set(held_out_mask)
    print(f"\nLOO for {disease_id}:")
    print(f"  Held-out drug-disease edges: {len(held_out_mask)}")
    print(f"  Unique held-out drugs: {len(held_out_drugs)}")
    for d in sorted(held_out_drugs):
        print(f"    {d}")

    # Split: training = all triples minus held-out
    train_triples = np.array([t for i, t in enumerate(triples) if i not in held_out_mask])
    test_triples = triples[list(held_out_mask)]

    print(f"  Training triples: {len(train_triples)}")
    print(f"  Test triples: {len(test_triples)}")

    # Create PyKEEN factories
    train_tf = TriplesFactory.from_labeled_triples(train_triples)
    test_tf = TriplesFactory.from_labeled_triples(
        test_triples,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )

    # Get all drug node IDs for ranking
    all_drug_ids = list(key_mappings['drug'].keys())
    all_drug_nodes = [f"drug:{d}" for d in all_drug_ids]

    results = {}

    for model_name in models_to_run:
        print(f"\n  Training {model_name} (dim={embedding_dim}, epochs={epochs})...")
        t0 = time.time()

        result = pipeline(
            training=train_tf,
            testing=test_tf,
            model=model_name,
            model_kwargs={'embedding_dim': embedding_dim},
            training_kwargs={
                'num_epochs': epochs,
                'use_tqdm_batch': False,
            },
            evaluation_kwargs={'batch_size': 256},
            random_seed=42,
            device='cpu',
        )

        elapsed = time.time() - t0
        print(f"    Trained in {elapsed:.1f}s")

        # Get test metrics using flat dict for compatibility
        try:
            flat_metrics = result.metric_results.to_flat_dict()
            mr = flat_metrics.get('both.realistic.arithmetic_mean_rank', -1)
            mrr = flat_metrics.get('both.realistic.inverse_harmonic_mean_rank', -1)
        except Exception:
            mr = mrr = -1

        # Custom ranking: score all (drug, drug-disease, disease) triples
        model = result.model
        model.eval()

        drug_ranks = []

        rel_id = train_tf.relation_to_id.get('drug-disease')
        disease_ent_id = train_tf.entity_to_id.get(disease_node)
        total_drugs = 0

        if rel_id is not None and disease_ent_id is not None:
            scores_list = []
            drug_node_list = []

            for drug_node in all_drug_nodes:
                drug_ent_id = train_tf.entity_to_id.get(drug_node)
                if drug_ent_id is None:
                    continue
                drug_node_list.append(drug_node)

                h_tensor = torch.tensor([drug_ent_id], dtype=torch.long)
                r_tensor = torch.tensor([rel_id], dtype=torch.long)
                t_tensor = torch.tensor([disease_ent_id], dtype=torch.long)

                with torch.no_grad():
                    score = model.score_hrt(
                        torch.stack([h_tensor, r_tensor, t_tensor], dim=1)
                    ).item()
                scores_list.append(score)

            scored = sorted(zip(drug_node_list, scores_list), key=lambda x: -x[1])
            ranks = {drug: rank + 1 for rank, (drug, _) in enumerate(scored)}
            total_drugs = len(scored)

            for drug in held_out_drugs:
                if drug in ranks:
                    drug_ranks.append(ranks[drug])

            median_rank = np.median(drug_ranks) if drug_ranks else -1
            mean_rank = np.mean(drug_ranks) if drug_ranks else -1

            h10_c = sum(1 for r in drug_ranks if r <= 10) / len(drug_ranks) if drug_ranks else 0
            h20_c = sum(1 for r in drug_ranks if r <= 20) / len(drug_ranks) if drug_ranks else 0
            h50_c = sum(1 for r in drug_ranks if r <= 50) / len(drug_ranks) if drug_ranks else 0
            h100_c = sum(1 for r in drug_ranks if r <= 100) / len(drug_ranks) if drug_ranks else 0
        else:
            median_rank = mean_rank = -1
            h10_c = h20_c = h50_c = h100_c = 0.0

        results[model_name] = {
            'pykeen_mr': mr, 'pykeen_mrr': mrr,
            'median_rank': median_rank, 'mean_rank': mean_rank,
            'h10': h10_c, 'h20': h20_c, 'h50': h50_c, 'h100': h100_c,
            'total_drugs': total_drugs, 'held_out_drugs': len(held_out_drugs),
            'time_s': elapsed,
        }

        print(f"    PyKEEN MR: {mr:.1f}, MRR: {mrr:.4f}")
        print(f"    Custom LOO (median rank): {median_rank:.0f}/{total_drugs}")
        print(f"    H@10: {h10_c:.1%}, H@20: {h20_c:.1%}, "
              f"H@50: {h50_c:.1%}, H@100: {h100_c:.1%}")

    return results


def main():
    """Run PyKEEN baselines."""
    # Build triples from saved graph
    triples, key_mappings, drug_indices, disease_indices = build_triples_from_saved_graph()

    # Run LOO evaluation on Osteoporosis
    results = run_loo_evaluation(
        triples, key_mappings,
        disease_id="EFO_0003854",
        models_to_run=("TransE", "DistMult", "RotatE"),
        epochs=100,
        embedding_dim=64,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("KG EMBEDDING BASELINES — Osteoporosis LOO Summary")
    print("=" * 80)

    seal_ref = {
        'median_rank': 24, 'h10': 0.30, 'h20': 0.41,
        'h50': 0.78, 'h100': 0.96,
    }

    print(f"\n{'Model':<15} {'Med.Rank':>10} {'H@10':>8} {'H@20':>8} "
          f"{'H@50':>8} {'H@100':>8} {'Time':>8}")
    print("-" * 75)

    for model_name, r in results.items():
        print(f"{model_name:<15} {r['median_rank']:>10.0f} {r['h10']:>8.1%} "
              f"{r['h20']:>8.1%} {r['h50']:>8.1%} {r['h100']:>8.1%} "
              f"{r['time_s']:>7.0f}s")

    print(f"{'SEAL (ref)':<15} {seal_ref['median_rank']:>10} "
          f"{seal_ref['h10']:>8.1%} {seal_ref['h20']:>8.1%} "
          f"{seal_ref['h50']:>8.1%} {seal_ref['h100']:>8.1%}")

    # Save results
    results_dir = project_root / "results" / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "pykeen_baselines.json"

    serialisable = {}
    for k, v in results.items():
        serialisable[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                          for kk, vv in v.items()}

    with open(results_file, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()

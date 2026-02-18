#!/usr/bin/env python3
"""
Heuristic Baselines for Link Prediction

Implements classic link prediction heuristics to establish a performance floor
for comparison with GNN-based methods.

Heuristics:
- Common Neighbors (CN): Count of shared neighbors
- Jaccard Coefficient: Intersection over union of neighbors
- Adamic-Adar (AA): Sum of 1/log(degree) for shared neighbors

Usage:
    uv run scripts/heuristic_baselines.py --target-node EFO_0003854
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config


class HeuristicBaseline:
    """Heuristic link prediction baselines."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.graph = None
        self.adj_list = None
        self.degrees = None
        
        # Mappings
        self.drug_key_mapping = {}
        self.disease_key_mapping = {}
        self.idx_to_drug = {}
        self.idx_to_disease = {}
        
    def load_graph(self):
        """Load graph and build adjacency structures."""
        results_path = Path(self.config.paths['results'])
        
        # Find latest graph
        graph_files = list(results_path.glob("graph_*.pt"))
        if not graph_files:
            raise FileNotFoundError(f"No graph files found in {results_path}")
        
        latest_graph = max(graph_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading graph: {latest_graph}")
        
        self.graph = torch.load(latest_graph, weights_only=False)
        print(f"  Nodes: {self.graph.num_nodes:,}")
        print(f"  Edges: {self.graph.edge_index.shape[1]:,}")
        
        # Load mappings
        mappings_dir = str(latest_graph).replace('.pt', '_mappings')
        if os.path.isdir(mappings_dir):
            print(f"Loading mappings from: {mappings_dir}")
            with open(f"{mappings_dir}/drug_key_mapping.json") as f:
                self.drug_key_mapping = json.load(f)
            with open(f"{mappings_dir}/disease_key_mapping.json") as f:
                self.disease_key_mapping = json.load(f)
        else:
            # Fallback to processed_data
            mappings_path = Path('processed_data/mappings')
            print(f"Loading mappings from: {mappings_path}")
            with open(mappings_path / 'drug_key_mapping.json') as f:
                self.drug_key_mapping = json.load(f)
            with open(mappings_path / 'disease_key_mapping.json') as f:
                self.disease_key_mapping = json.load(f)
        
        self.idx_to_drug = {v: k for k, v in self.drug_key_mapping.items()}
        self.idx_to_disease = {v: k for k, v in self.disease_key_mapping.items()}
        
        # Build adjacency list
        self._build_adjacency()
        
    def _build_adjacency(self):
        """Build adjacency list and degree dictionary."""
        print("Building adjacency structures...")
        
        edge_index = self.graph.edge_index
        num_nodes = self.graph.num_nodes
        
        self.adj_list = defaultdict(set)
        self.degrees = np.zeros(num_nodes, dtype=np.int32)
        
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        
        for s, d in zip(src, dst):
            self.adj_list[s].add(d)
            self.degrees[s] += 1
            
        print(f"  Built adjacency for {len(self.adj_list):,} nodes")
        
    def common_neighbors(self, u: int, v: int) -> int:
        """Count shared neighbors between u and v."""
        return len(self.adj_list[u] & self.adj_list[v])
    
    def jaccard_coefficient(self, u: int, v: int) -> float:
        """Jaccard similarity: |intersection| / |union|."""
        intersection = self.adj_list[u] & self.adj_list[v]
        union = self.adj_list[u] | self.adj_list[v]
        return len(intersection) / len(union) if union else 0.0
    
    def adamic_adar(self, u: int, v: int) -> float:
        """Adamic-Adar: sum of 1/log(degree) for shared neighbors."""
        shared = self.adj_list[u] & self.adj_list[v]
        score = 0.0
        for w in shared:
            deg = self.degrees[w]
            if deg > 1:
                score += 1.0 / np.log(deg)
        return score
    
    def get_true_drug_neighbors(self, disease_idx: int) -> set:
        """Get drugs connected to this disease."""
        drug_indices = set(self.drug_key_mapping.values())
        neighbors = self.adj_list[disease_idx]
        return neighbors & drug_indices
    
    def evaluate_loo(self, target_node: str, k: int = 20, tracker=None):
        """
        Leave-One-Out evaluation for a disease.
        
        Ranks all drugs for the disease using each heuristic.
        Reports standardised metrics matching SEAL output format.
        """
        import datetime as dt
        
        # Resolve node
        if target_node in self.disease_key_mapping:
            disease_idx = self.disease_key_mapping[target_node]
            disease_name = target_node
        else:
            print(f"Error: Disease '{target_node}' not found")
            return
        
        print(f"\n{'='*70}")
        print(f"Heuristic Baselines for: {disease_name}")
        print(f"{'='*70}")
        
        # Get true drug connections
        true_drugs = self.get_true_drug_neighbors(disease_idx)
        print(f"True drug connections: {len(true_drugs)}")
        
        if len(true_drugs) == 0:
            print("No drug connections found for this disease.")
            return
        
        # Score all drugs with each heuristic
        drug_indices = list(self.drug_key_mapping.values())
        
        heuristics = {
            'Common Neighbors': self.common_neighbors,
            'Jaccard': self.jaccard_coefficient,
            'Adamic-Adar': self.adamic_adar,
        }
        
        results = {}
        
        for heuristic_name, heuristic_fn in heuristics.items():
            print(f"\n--- {heuristic_name} ---")
            
            # Score all drugs
            scores = []
            for drug_idx in tqdm(drug_indices, desc=f"Scoring ({heuristic_name})", leave=False):
                score = heuristic_fn(drug_idx, disease_idx)
                scores.append((drug_idx, score))
            
            # Sort by score descending
            sorted_drugs = sorted(scores, key=lambda x: x[1], reverse=True)
            total_drugs = len(sorted_drugs)
            
            # Build rank lookup
            rank_map = {d: r for r, (d, _) in enumerate(sorted_drugs, 1)}
            
            # Build top-20 list for display and JSON
            print(f"\nTop 20 predictions:")
            print(f"{'Rank':<6} {'Score':<12} {'Drug ID':<20} {'True?'}")
            print("-" * 55)
            
            top20_list = []
            for rank, (drug_idx, score) in enumerate(sorted_drugs[:20], 1):
                drug_id = self.idx_to_drug.get(drug_idx, "Unknown")
                is_true = drug_idx in true_drugs
                mark = "✓" if is_true else ""
                top20_list.append({
                    "rank": rank, "drug_id": drug_id,
                    "score": round(score, 6), "is_true": is_true,
                })
                print(f"{rank:<6} {score:<12.4f} {drug_id:<20} {mark}")
            
            # Get ranks for all true drugs
            ranks = []
            for drug_idx in true_drugs:
                ranks.append(rank_map.get(drug_idx, len(drug_indices) + 1))
            
            ranks = np.array(ranks)
            n = len(ranks)
            mean_rank = float(np.mean(ranks))
            median_rank = float(np.median(ranks))
            
            # Standardised metrics (matching SEAL output)
            hits_at_10 = int(np.sum(ranks <= 10))
            hits_at_20 = int(np.sum(ranks <= 20))
            hits_at_50 = int(np.sum(ranks <= 50))
            hits_at_100 = int(np.sum(ranks <= 100))
            mrr = float(np.mean(1.0 / ranks)) if n > 0 else 0.0
            
            # Display standardised summary
            print(f"\n{'=' * 60}")
            print(f"HEURISTIC ({heuristic_name}) SUMMARY FOR {disease_name}")
            print(f"{'=' * 60}")
            print(f"  Test Edges (True Positives): {n}")
            print(f"  Total Drugs Ranked: {total_drugs}")
            if n > 0:
                print(f"  Hits@10:  {hits_at_10} / {n} ({hits_at_10 / n * 100:.1f}%)")
                print(f"  Hits@20:  {hits_at_20} / {n} ({hits_at_20 / n * 100:.1f}%)")
                print(f"  Hits@50:  {hits_at_50} / {n} ({hits_at_50 / n * 100:.1f}%)")
                print(f"  Hits@100: {hits_at_100} / {n} ({hits_at_100 / n * 100:.1f}%)")
            print(f"  Median Rank: {median_rank:.1f}")
            print(f"  Mean Rank: {mean_rank:.1f}")
            print(f"  MRR: {mrr:.4f}")
            print(f"{'=' * 60}")
            
            # Show all true drug ranks
            print(f"\nTrue drug ranks:")
            for drug_idx in true_drugs:
                rank = rank_map.get(drug_idx, len(drug_indices) + 1)
                drug_id = self.idx_to_drug.get(drug_idx, "Unknown")
                score = next((s for d, s in scores if d == drug_idx), 0)
                print(f"  {drug_id:<20} Rank: {rank:<6} Score: {score:.4f}")
            
            # Save standardised JSON (matching SEAL format)
            heuristic_key = heuristic_name.lower().replace(' ', '_')
            results_dir = Path("results/heuristic_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results_json = {
                "method": f"heuristic_{heuristic_key}",
                "target_disease": disease_name,
                "timestamp": timestamp,
                "config": {
                    "heuristic": heuristic_name,
                },
                "metrics": {
                    "hits_at_10": hits_at_10, "hits_at_20": hits_at_20,
                    "hits_at_50": hits_at_50, "hits_at_100": hits_at_100,
                    "total_true": n, "total_drugs": total_drugs,
                    "median_rank": round(median_rank, 1),
                    "mean_rank": round(mean_rank, 1),
                    "mrr": round(mrr, 4),
                },
                "top20": top20_list,
                "all_ranks": sorted(ranks.tolist()),
            }
            
            json_path = results_dir / f"heuristic_{heuristic_key}_{disease_name}_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(results_json, f, indent=2)
            print(f"Results saved to: {json_path}")
            
            # MLflow tracking
            if tracker:
                tracker.log_metric(f"{heuristic_key}_hits_at_10", hits_at_10)
                tracker.log_metric(f"{heuristic_key}_hits_at_20", hits_at_20)
                tracker.log_metric(f"{heuristic_key}_hits_at_50", hits_at_50)
                tracker.log_metric(f"{heuristic_key}_hits_at_100", hits_at_100)
                tracker.log_metric(f"{heuristic_key}_median_rank", median_rank)
                tracker.log_metric(f"{heuristic_key}_mean_rank", mean_rank)
                tracker.log_metric(f"{heuristic_key}_mrr", mrr)
                tracker.log_artifact(str(json_path))
            
            results[heuristic_name] = {
                'hits_at_10': hits_at_10,
                'hits_at_20': hits_at_20,
                'hits_at_50': hits_at_50,
                'hits_at_100': hits_at_100,
                'mean_rank': mean_rank,
                'median_rank': median_rank,
                'mrr': mrr,
                'total_true': n,
                'total_drugs': total_drugs,
            }
        
        # Summary comparison
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Heuristic':<20} {'Hits@10':<10} {'Hits@20':<10} {'Hits@50':<10} {'MRR':<10} {'Med.Rank':<10}")
        print("-" * 70)
        
        for name, m in results.items():
            print(f"{name:<20} {m['hits_at_10']:<10} {m['hits_at_20']:<10} "
                  f"{m['hits_at_50']:<10} {m['mrr']:<10.4f} {m['median_rank']:<10.1f}")
        
        return results


def main():
    import datetime as dt
    
    parser = argparse.ArgumentParser(description="Heuristic baselines for link prediction")
    parser.add_argument('--target-node', type=str, required=True,
                        help='Disease ID to evaluate (e.g., EFO_0003854)')
    parser.add_argument('--k', type=int, default=20,
                        help='K for Hits@K metric (default: 20)')
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow experiment tracking")
    
    args = parser.parse_args()
    
    # MLflow tracking
    tracker = None
    if not args.no_mlflow:
        from src.training.tracker import ExperimentTracker
        tracker = ExperimentTracker(
            experiment_name=f"Heuristic-{args.target_node}",
        )
        run_name = f"heuristic_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tracker.start_run(run_name=run_name)
        print(f"MLflow tracking enabled: Heuristic-{args.target_node} / {run_name}")
        tracker.log_param("method", "heuristic")
        tracker.log_param("target_disease", args.target_node)
    
    baseline = HeuristicBaseline()
    baseline.load_graph()
    baseline.evaluate_loo(args.target_node, k=args.k, tracker=tracker)
    
    if tracker:
        tracker.end_run()
        print("Ended MLflow run.")


if __name__ == "__main__":
    main()


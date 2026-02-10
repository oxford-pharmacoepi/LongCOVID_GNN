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
    
    def evaluate_loo(self, target_node: str, k: int = 20):
        """
        Leave-One-Out evaluation for a disease.
        
        Ranks all drugs for the disease using each heuristic.
        Reports Hits@K and Mean Rank.
        """
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
            
            # Calculate metrics
            hits = 0
            ranks = []
            
            # Build rank lookup
            rank_map = {d: r for r, (d, _) in enumerate(sorted_drugs, 1)}
            
            print(f"\nTop {k} predictions:")
            print(f"{'Rank':<6} {'Score':<12} {'Drug ID':<20} {'True?'}")
            print("-" * 55)
            
            for rank, (drug_idx, score) in enumerate(sorted_drugs[:k], 1):
                drug_id = self.idx_to_drug.get(drug_idx, "Unknown")
                is_true = drug_idx in true_drugs
                mark = "✓" if is_true else ""
                if is_true:
                    hits += 1
                print(f"{rank:<6} {score:<12.4f} {drug_id:<20} {mark}")
            
            # Get ranks for all true drugs
            for drug_idx in true_drugs:
                ranks.append(rank_map.get(drug_idx, len(drug_indices) + 1))
            
            ranks = np.array(ranks)
            mean_rank = np.mean(ranks)
            median_rank = np.median(ranks)
            
            print(f"\nMetrics:")
            print(f"  Hits@{k}: {hits} / {len(true_drugs)}")
            print(f"  Mean Rank: {mean_rank:.1f}")
            print(f"  Median Rank: {median_rank:.1f}")
            
            # Show all true drug ranks
            print(f"\nTrue drug ranks:")
            for drug_idx in true_drugs:
                rank = rank_map.get(drug_idx, len(drug_indices) + 1)
                drug_id = self.idx_to_drug.get(drug_idx, "Unknown")
                score = next((s for d, s in scores if d == drug_idx), 0)
                print(f"  {drug_id:<20} Rank: {rank:<6} Score: {score:.4f}")
            
            results[heuristic_name] = {
                'hits_at_k': hits,
                'mean_rank': mean_rank,
                'median_rank': median_rank,
                'k': k,
                'total_true': len(true_drugs)
            }
        
        # Summary comparison
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Heuristic':<20} {'Hits@'+str(k):<12} {'Mean Rank':<12} {'Median Rank':<12}")
        print("-" * 60)
        
        for name, metrics in results.items():
            hits_str = f"{metrics['hits_at_k']}/{metrics['total_true']}"
            print(f"{name:<20} {hits_str:<12} {metrics['mean_rank']:<12.1f} {metrics['median_rank']:<12.1f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Heuristic baselines for link prediction")
    parser.add_argument('--target-node', type=str, required=True,
                        help='Disease ID to evaluate (e.g., EFO_0003854)')
    parser.add_argument('--k', type=int, default=20,
                        help='K for Hits@K metric (default: 20)')
    
    args = parser.parse_args()
    
    baseline = HeuristicBaseline()
    baseline.load_graph()
    baseline.evaluate_loo(args.target_node, k=args.k)


if __name__ == "__main__":
    main()

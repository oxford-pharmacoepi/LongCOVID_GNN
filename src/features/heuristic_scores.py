"""
Heuristic score computation for edge features.

Computes Common Neighbors, Adamic-Adar, and Jaccard coefficients
that can be used as edge features during GNN training.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional, Set
from tqdm import tqdm


class HeuristicScorer:
    """Compute heuristic link prediction scores for drug-disease pairs."""
    
    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        """
        Initialise scorer with graph structure.
        
        Args:
            edge_index: [2, E] edge index tensor
            num_nodes: Total number of nodes
        """
        self.num_nodes = num_nodes
        self.adj_list = defaultdict(set)
        self.degrees = np.zeros(num_nodes, dtype=np.int32)
        
        # Build adjacency
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        
        for s, d in zip(src, dst):
            self.adj_list[s].add(d)
            self.degrees[s] += 1
            
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
    
    def compute_all_scores(self, u: int, v: int) -> Tuple[float, float, float]:
        """
        Compute all heuristic scores for a pair.
        
        Returns:
            (common_neighbors, adamic_adar, jaccard)
        """
        cn = self.common_neighbors(u, v)
        aa = self.adamic_adar(u, v)
        jc = self.jaccard_coefficient(u, v)
        return cn, aa, jc
    
    def compute_edge_heuristic_features(
        self, 
        edges: torch.Tensor,
        normalise: bool = True
    ) -> torch.Tensor:
        """
        Compute heuristic scores for a batch of edges.
        
        Args:
            edges: [2, E] edge tensor with src, dst node indices
            normalise: Whether to normalise scores to [0, 1] range
            
        Returns:
            [E, 3] tensor of (CN, AA, Jaccard) features
        """
        num_edges = edges.shape[1]
        features = torch.zeros(num_edges, 3, dtype=torch.float32)
        
        src = edges[0].numpy()
        dst = edges[1].numpy()
        
        # Compute scores for each edge
        for i, (u, v) in enumerate(zip(src, dst)):
            cn, aa, jc = self.compute_all_scores(u, v)
            features[i, 0] = cn
            features[i, 1] = aa
            features[i, 2] = jc
            
        if normalise:
            # Normalise each feature column
            for col in range(3):
                max_val = features[:, col].max()
                if max_val > 0:
                    features[:, col] = features[:, col] / max_val
                    
        return features
    
    def compute_batch_heuristic_features(
        self,
        edges: torch.Tensor,
        batch_size: int = 10000,
        normalise: bool = True,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Compute heuristic features for large edge sets in batches.
        
        Args:
            edges: [2, E] edge tensor
            batch_size: Number of edges to process at once
            normalise: Whether to normalise at the end
            show_progress: Show progress bar
            
        Returns:
            [E, 3] tensor of heuristic features
        """
        num_edges = edges.shape[1]
        features = torch.zeros(num_edges, 3, dtype=torch.float32)
        
        src = edges[0].numpy()
        dst = edges[1].numpy()
        
        iterator = range(0, num_edges, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing heuristic features")
            
        for start in iterator:
            end = min(start + batch_size, num_edges)
            for i in range(start, end):
                cn, aa, jc = self.compute_all_scores(src[i], dst[i])
                features[i, 0] = cn
                features[i, 1] = aa
                features[i, 2] = jc
                
        if normalise:
            # Normalise each feature column (avoiding division by zero)
            for col in range(3):
                max_val = features[:, col].max()
                if max_val > 0:
                    features[:, col] = features[:, col] / max_val
                    
        return features


def compute_heuristic_edge_features(
    graph,
    train_edges: torch.Tensor,
    normalise: bool = False,
    show_progress: bool = True
) -> torch.Tensor:
    """
    Convenience function to compute heuristic features for training edges.
    
    Args:
        graph: PyG Data object with edge_index
        train_edges: [2, E] edges to compute features for
        normalise: Whether to normalise features
        show_progress: Show progress bar
        
    Returns:
        [E, 3] tensor with (CN, AA, Jaccard) normalised features
    """
    scorer = HeuristicScorer(graph.edge_index, graph.num_nodes)
    return scorer.compute_batch_heuristic_features(
        train_edges, 
        normalise=normalise,
        show_progress=show_progress
    )

"""
Negative Sampling Strategies for Knowledge Graph Link Prediction

This module provides different negative sampling strategies for training GNN models
on drug-disease prediction tasks. The goal is to generate informative negative samples
that help the model learn meaningful patterns rather than trivial distinctions.

Strategies implemented:
1. Random sampling (baseline)
2. Hard negative sampling (common neighbors ≥ threshold)
3. Degree-matched sampling
4. Feature-similarity sampling
5. Mixed strategy sampling
"""

import torch
import numpy as np
import random
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional
from tqdm import tqdm


class NegativeSampler:
    """
    Base class for negative sampling strategies.
    All sampling strategies should inherit from this class.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialise sampler.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def sample(self, 
               positive_edges: Set[Tuple[int, int]], 
               all_possible_pairs: List[Tuple[int, int]], 
               num_samples: int,
               **kwargs) -> List[Tuple[int, int]]:
        """
        Sample negative edges.
        
        Args:
            positive_edges: Set of positive (src, dst) pairs
            all_possible_pairs: List of all possible (src, dst) pairs
            num_samples: Number of negative samples to generate
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            List of negative (src, dst) pairs
        """
        raise NotImplementedError("Subclasses must implement sample()")
    
    def get_name(self) -> str:
        """Return the name of the sampling strategy."""
        return self.__class__.__name__


class RandomNegativeSampler(NegativeSampler):
    """
    Baseline: Random negative sampling.
    Samples uniformly from all non-positive pairs.
    """
    
    def sample(self, 
               positive_edges: Set[Tuple[int, int]], 
               all_possible_pairs: List[Tuple[int, int]], 
               num_samples: int,
               **kwargs) -> List[Tuple[int, int]]:
        """Sample random negatives from non-positive pairs."""
        
        # Filter out positive edges
        negative_candidates = list(set(all_possible_pairs) - positive_edges)
        
        # Sample
        if len(negative_candidates) < num_samples:
            print(f"Warning: Only {len(negative_candidates)} negative candidates available, requested {num_samples}")
            return negative_candidates
        
        return random.sample(negative_candidates, num_samples)


class HardNegativeSampler(NegativeSampler):
    """
    Hard negative sampling based on common neighbors with random fallback.
    Ranks all negative candidates by their common neighbor count and selects
    the top N hardest negatives (highest common neighbors).
    
    If not enough hard negatives exist, supplements with random negatives.
    """
    
    def __init__(self, seed: int = 42, fallback_to_random: bool = True):
        """
        Initialise hard negative sampler.
        
        Args:
            seed: Random seed
            fallback_to_random: If True, supplement with random negatives when hard negatives run out
        """
        super().__init__(seed)
        self.fallback_to_random = fallback_to_random
    
    def sample(self, 
               positive_edges: Set[Tuple[int, int]], 
               all_possible_pairs: List[Tuple[int, int]], 
               num_samples: int,
               edge_index: Optional[torch.Tensor] = None,
               **kwargs) -> List[Tuple[int, int]]:
        """
        Sample the hardest negatives by ranking all candidates by common neighbors.
        Falls back to random sampling if not enough hard negatives exist.
        
        Args:
            positive_edges: Set of positive edges
            all_possible_pairs: All possible pairs
            num_samples: Number of samples needed
            edge_index: Graph edge index for computing common neighbors
            
        Returns:
            List of hard negative samples (supplemented with random if needed)
        """
        
        if edge_index is None:
            raise ValueError("edge_index required for hard negative sampling")
        
        print(f"Sampling hard negatives (target: {num_samples:,})...")
        
        # Build adjacency list
        adj_list = defaultdict(set)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].add(dst)
            adj_list[dst].add(src)
        
        # Compute common neighbors for all negative candidates
        candidate_pool = list(set(all_possible_pairs) - positive_edges)
        
        print(f"Computing common neighbors for {len(candidate_pool):,} negative candidates...")
        cn_scores = []
        
        for src, dst in tqdm(candidate_pool, desc="Computing common neighbors"):
            # Compute common neighbors
            src_neighbors = adj_list[src]
            dst_neighbors = adj_list[dst]
            common_neighbors = src_neighbors & dst_neighbors
            cn_count = len(common_neighbors)
            
            cn_scores.append((cn_count, src, dst))
        
        # Sort by common neighbor count (descending - hardest first)
        cn_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Report statistics
        if cn_scores:
            max_cn = cn_scores[0][0]
            min_cn = cn_scores[-1][0]
            avg_cn = np.mean([score[0] for score in cn_scores])
            print(f"\nCommon neighbor statistics:")
            print(f"  Min: {min_cn}, Max: {max_cn}, Avg: {avg_cn:.2f}")
            
            # Show distribution of top candidates
            top_k = min(10, len(cn_scores))
            print(f"\nTop {top_k} hardest negatives have CN counts: {[s[0] for s in cn_scores[:top_k]]}")
            
            # Print distribution: count how many edges have each CN count
            from collections import Counter
            cn_distribution = Counter([score[0] for score in cn_scores])
            print(f"\nDistribution of edges by common neighbor count:")
            print(f"{'CN':>5} | {'Count':>8} | {'%':>6}")
            print("-" * 25)
            # Sort by CN count (descending) and show top 20
            for cn_count in sorted(cn_distribution.keys(), reverse=True)[:50]:
                count = cn_distribution[cn_count]
                percentage = (count / len(cn_scores)) * 100
                print(f"{cn_count:5d} | {count:8,d} | {percentage:5.2f}%")
        
        # Check if we have enough hard negatives
        num_available = len(cn_scores)
        
        if num_available >= num_samples:
            # Enough hard negatives available
            hard_negatives = [(src, dst) for _, src, dst in cn_scores[:num_samples]]
            
            # Report the CN range of selected samples
            selected_cn_min = cn_scores[num_samples - 1][0]
            selected_cn_max = cn_scores[0][0]
            
            print(f"\n✓ Selected {len(hard_negatives):,} hard negatives")
            print(f"  CN range: [{selected_cn_min} - {selected_cn_max}]")
            
            return hard_negatives
        
        else:
            # Not enough hard negatives - need to supplement with random
            print(f"\n Only {num_available:,} hard negatives available (need {num_samples:,})")
            
            if not self.fallback_to_random:
                print(f"  Fallback disabled - returning all {num_available:,} available hard negatives only")
                return [(src, dst) for _, src, dst in cn_scores]
            
            # Use all hard negatives
            hard_negatives = [(src, dst) for _, src, dst in cn_scores]
            num_hard = len(hard_negatives)
            
            # Supplement with random negatives
            deficit = num_samples - num_hard
            print(f"  Using all {num_hard:,} hard negatives")
            print(f"  Supplementing with {deficit:,} random negatives")
            
            # Get random negatives from remaining candidates (exclude already selected hard negatives)
            used_negatives = set(hard_negatives)
            random_candidates = [pair for pair in candidate_pool if pair not in used_negatives]
            
            if len(random_candidates) < deficit:
                print(f"  Warning: Only {len(random_candidates):,} random candidates available")
                deficit = len(random_candidates)
            
            random_negatives = random.sample(random_candidates, deficit)
            
            # Combine hard + random
            all_negatives = hard_negatives + random_negatives
            
            # Shuffle to mix hard and random
            random.shuffle(all_negatives)
            
            print(f"\n Selected {len(all_negatives):,} total negatives:")
            print(f"  - {num_hard:,} hard negatives ({100*num_hard/len(all_negatives):.1f}%)")
            print(f"  - {len(random_negatives):,} random negatives ({100*len(random_negatives)/len(all_negatives):.1f}%)")
            
            return all_negatives


class DegreeMatchedNegativeSampler(NegativeSampler):
    """
    Degree-matched negative sampling.
    Samples negatives where source and target nodes have similar degrees
    to those in positive samples.
    """
    
    def __init__(self, degree_tolerance: float = 0.3, seed: int = 42):
        """
        Initialise degree-matched sampler.
        
        Args:
            degree_tolerance: Relative tolerance for degree matching (0.3 = ±30%)
            seed: Random seed
        """
        super().__init__(seed)
        self.degree_tolerance = degree_tolerance
    
    def sample(self, 
               positive_edges: Set[Tuple[int, int]], 
               all_possible_pairs: List[Tuple[int, int]], 
               num_samples: int,
               edge_index: Optional[torch.Tensor] = None,
               **kwargs) -> List[Tuple[int, int]]:
        """
        Sample negatives with degree distribution matching positives.
        
        Args:
            positive_edges: Set of positive edges
            all_possible_pairs: All possible pairs
            num_samples: Number of samples needed
            edge_index: Graph edge index for computing degrees
            
        Returns:
            List of degree-matched negative samples
        """
        
        if edge_index is None:
            raise ValueError("edge_index required for degree-matched sampling")
        
        print(f"Sampling degree-matched negatives (tolerance={self.degree_tolerance})...")
        
        # Compute node degrees
        degrees = defaultdict(int)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            degrees[src] += 1
            degrees[dst] += 1
        
        # Compute degree statistics for positive edges
        pos_degrees_src = [degrees[src] for src, _ in positive_edges]
        pos_degrees_dst = [degrees[dst] for _, dst in positive_edges]
        
        mean_deg_src = np.mean(pos_degrees_src)
        mean_deg_dst = np.mean(pos_degrees_dst)
        std_deg_src = np.std(pos_degrees_src)
        std_deg_dst = np.std(pos_degrees_dst)
        
        print(f"Positive edge degrees: src={mean_deg_src:.1f}±{std_deg_src:.1f}, "
              f"dst={mean_deg_dst:.1f}±{std_deg_dst:.1f}")
        
        # Find negative candidates with similar degrees
        negative_candidates = []
        candidate_pool = list(set(all_possible_pairs) - positive_edges)
        random.shuffle(candidate_pool)
        
        for src, dst in tqdm(candidate_pool, desc="Finding degree-matched negatives"):
            if len(negative_candidates) >= num_samples * 2:
                break
            
            deg_src = degrees[src]
            deg_dst = degrees[dst]
            
            # Check if degrees are within tolerance
            src_match = abs(deg_src - mean_deg_src) <= mean_deg_src * self.degree_tolerance
            dst_match = abs(deg_dst - mean_deg_dst) <= mean_deg_dst * self.degree_tolerance
            
            if src_match and dst_match:
                negative_candidates.append((src, dst))
        
        # Sample from candidates
        if len(negative_candidates) < num_samples:
            print(f"Warning: Only found {len(negative_candidates)} degree-matched negatives")
            return negative_candidates
        
        return random.sample(negative_candidates, num_samples)


class FeatureSimilarityNegativeSampler(NegativeSampler):
    """
    Feature-similarity based negative sampling.
    Samples negatives where nodes have similar features to positive pairs.
    """
    
    def __init__(self, similarity_threshold: float = 0.5, seed: int = 42):
        """
        Initialise feature-similarity sampler.
        
        Args:
            similarity_threshold: Minimum cosine similarity for node features
            seed: Random seed
        """
        super().__init__(seed)
        self.similarity_threshold = similarity_threshold
    
    def sample(self, 
               positive_edges: Set[Tuple[int, int]], 
               all_possible_pairs: List[Tuple[int, int]], 
               num_samples: int,
               node_features: Optional[torch.Tensor] = None,
               **kwargs) -> List[Tuple[int, int]]:
        """
        Sample negatives with similar node features to positives.
        
        Args:
            positive_edges: Set of positive edges
            all_possible_pairs: All possible pairs
            num_samples: Number of samples needed
            node_features: Node feature matrix
            
        Returns:
            List of feature-similar negative samples
        """
        
        if node_features is None:
            raise ValueError("node_features required for feature-similarity sampling")
        
        print(f"Sampling feature-similar negatives (threshold={self.similarity_threshold})...")
        
        # Compute average feature profile for positive pairs
        pos_features_src = []
        pos_features_dst = []
        
        for src, dst in list(positive_edges)[:1000]:  # Sample for efficiency
            pos_features_src.append(node_features[src])
            pos_features_dst.append(node_features[dst])
        
        avg_features_src = torch.stack(pos_features_src).mean(dim=0)
        avg_features_dst = torch.stack(pos_features_dst).mean(dim=0)
        
        # Normalise for cosine similarity
        avg_features_src = avg_features_src / (avg_features_src.norm() + 1e-8)
        avg_features_dst = avg_features_dst / (avg_features_dst.norm() + 1e-8)
        
        # Find similar negatives
        negative_candidates = []
        candidate_pool = list(set(all_possible_pairs) - positive_edges)
        random.shuffle(candidate_pool)
        
        for src, dst in tqdm(candidate_pool, desc="Finding feature-similar negatives"):
            if len(negative_candidates) >= num_samples * 2:
                break
            
            # Compute similarities
            feat_src = node_features[src]
            feat_dst = node_features[dst]
            
            feat_src_norm = feat_src / (feat_src.norm() + 1e-8)
            feat_dst_norm = feat_dst / (feat_dst.norm() + 1e-8)
            
            sim_src = torch.dot(feat_src_norm, avg_features_src).item()
            sim_dst = torch.dot(feat_dst_norm, avg_features_dst).item()
            
            if sim_src >= self.similarity_threshold and sim_dst >= self.similarity_threshold:
                negative_candidates.append((src, dst))
        
        # Sample
        if len(negative_candidates) < num_samples:
            print(f"Warning: Only found {len(negative_candidates)} feature-similar negatives")
            return negative_candidates
        
        return random.sample(negative_candidates, num_samples)


class MixedNegativeSampler(NegativeSampler):
    """
    Mixed strategy: Combine multiple sampling strategies.
    Useful for getting diverse negative samples.
    """
    
    def __init__(self, 
                 strategy_weights: Dict[str, float] = None,
                 degree_tolerance: float = 0.3,
                 similarity_threshold: float = 0.5,
                 seed: int = 42):
        """
        Initialise mixed sampler.
        
        Args:
            strategy_weights: Dict mapping strategy names to their weights
                Example: {'hard': 0.5, 'degree_matched': 0.3, 'random': 0.2}
            degree_tolerance: Passed to DegreeMatchedNegativeSampler
            similarity_threshold: Passed to FeatureSimilarityNegativeSampler
            seed: Random seed
        """
        super().__init__(seed)
        
        if strategy_weights is None:
            strategy_weights = {
                'hard': 0.6,
                'degree_matched': 0.3,
                'random': 0.1
            }
        
        self.strategy_weights = strategy_weights
        
        # Initialise sub-samplers
        self.samplers = {
            'hard': HardNegativeSampler(seed=seed),
            'degree_matched': DegreeMatchedNegativeSampler(degree_tolerance=degree_tolerance, seed=seed),
            'random': RandomNegativeSampler(seed=seed),
            'feature_similar': FeatureSimilarityNegativeSampler(similarity_threshold=similarity_threshold, seed=seed)
        }
    
    def sample(self, 
               positive_edges: Set[Tuple[int, int]], 
               all_possible_pairs: List[Tuple[int, int]], 
               num_samples: int,
               **kwargs) -> List[Tuple[int, int]]:
        """
        Sample negatives using mixed strategy.
        
        Args:
            positive_edges: Set of positive edges
            all_possible_pairs: All possible pairs
            num_samples: Number of samples needed
            **kwargs: Passed to individual samplers
            
        Returns:
            List of mixed negative samples
        """
        
        print(f"Sampling with mixed strategy: {self.strategy_weights}")
        
        all_negatives = []
        
        # Sample from each strategy according to weights
        for strategy_name, weight in self.strategy_weights.items():
            if weight <= 0:
                continue
            
            n_samples = int(num_samples * weight)
            if n_samples == 0:
                continue
            
            sampler = self.samplers[strategy_name]
            
            try:
                negatives = sampler.sample(
                    positive_edges, 
                    all_possible_pairs, 
                    n_samples,
                    **kwargs
                )
                all_negatives.extend(negatives)
                print(f"  {strategy_name}: sampled {len(negatives)}/{n_samples}")
            except Exception as e:
                print(f"  {strategy_name}: failed ({e})")
        
        # If we don't have enough, fill with random
        if len(all_negatives) < num_samples:
            deficit = num_samples - len(all_negatives)
            print(f"Filling {deficit} samples with random negatives")
            
            random_sampler = RandomNegativeSampler(seed=self.seed)
            used_negatives = set(all_negatives)
            additional = random_sampler.sample(
                positive_edges | used_negatives,
                all_possible_pairs,
                deficit,
                **kwargs
            )
            all_negatives.extend(additional)
        
        # Shuffle and return
        random.shuffle(all_negatives)
        return all_negatives[:num_samples]


def get_sampler(strategy: str = 'random', **kwargs) -> NegativeSampler:
    """
    Factory function to get a negative sampler.
    
    Args:
        strategy: Name of sampling strategy
            - 'random': Random sampling (default, baseline)
            - 'hard': Hard negatives with common neighbors
            - 'degree_matched': Degree-matched negatives
            - 'feature_similar': Feature-similar negatives
            - 'mixed': Mixed strategy
        **kwargs: Additional arguments for the sampler
        
    Returns:
        NegativeSampler instance
    """
    
    samplers = {
        'random': RandomNegativeSampler,
        'hard': HardNegativeSampler,
        'degree_matched': DegreeMatchedNegativeSampler,
        'feature_similar': FeatureSimilarityNegativeSampler,
        'mixed': MixedNegativeSampler,
    }
    
    if strategy not in samplers:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(samplers.keys())}")
    
    return samplers[strategy](**kwargs)

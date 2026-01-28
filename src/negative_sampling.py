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
6. Temporal-aware sampling (prevents leakage from future positive edges)
"""

import torch
import numpy as np
import random
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional
from tqdm import tqdm


def validate_temporal_consistency(train_negatives: Set[Tuple[int, int]], 
                                   val_positives: Set[Tuple[int, int]], 
                                   test_positives: Set[Tuple[int, int]]) -> Dict[str, int]:
    """
    Validate that negative edges don't appear as positive edges in future time periods.
    
    Args:
        train_negatives: Set of negative edges used in training
        val_positives: Set of positive edges in validation set
        test_positives: Set of positive edges in test set
        
    Returns:
        Dictionary with statistics about temporal leakage
    """
    print("\n" + "="*80)
    print("TEMPORAL CONSISTENCY VALIDATION")
    print("="*80)
    
    # Check for leakage
    val_leakage = train_negatives & val_positives
    test_leakage = train_negatives & test_positives
    total_leakage = val_leakage | test_leakage
    
    stats = {
        'total_train_negatives': len(train_negatives),
        'val_positives': len(val_positives),
        'test_positives': len(test_positives),
        'val_leakage_count': len(val_leakage),
        'test_leakage_count': len(test_leakage),
        'total_leakage_count': len(total_leakage),
        'leakage_percentage': (len(total_leakage) / len(train_negatives) * 100) if train_negatives else 0
    }
    
    print(f"\nTraining negatives: {stats['total_train_negatives']:,}")
    print(f"Validation positives: {stats['val_positives']:,}")
    print(f"Test positives: {stats['test_positives']:,}")
    print(f"\n  LEAKAGE DETECTED:")
    print(f"  - Validation leakage: {stats['val_leakage_count']:,} edges ({stats['val_leakage_count']/len(val_positives)*100:.2f}% of val positives)")
    print(f"  - Test leakage: {stats['test_leakage_count']:,} edges ({stats['test_leakage_count']/len(test_positives)*100:.2f}% of test positives)")
    print(f"  - Total unique leakage: {stats['total_leakage_count']:,} edges ({stats['leakage_percentage']:.2f}% of train negatives)")
    
    if len(total_leakage) > 0:
        print(f"\n  WARNING: {len(total_leakage)} training negatives appear as future positives!")
        print("   This causes temporal leakage and unreliable evaluation.")
        print("   Recommendation: Exclude these edges from negative sampling.")
    else:
        print(f"\nNo temporal leakage detected - training negatives are clean!")
    
    print("="*80 + "\n")
    
    return stats


def filter_negatives_by_future_positives(negative_candidates: List[Tuple[int, int]],
                                          future_positives: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Remove negative candidates that appear as positives in future time periods.
    
    Args:
        negative_candidates: List of candidate negative edges
        future_positives: Set of positive edges from validation and/or test sets
        
    Returns:
        Filtered list of negative candidates
    """
    original_count = len(negative_candidates)
    filtered = [edge for edge in negative_candidates if edge not in future_positives]
    removed_count = original_count - len(filtered)
    
    if removed_count > 0:
        print(f"  Filtered out {removed_count:,} negative candidates that are future positives ({removed_count/original_count*100:.2f}%)")
    
    return filtered


class NegativeSampler:
    """
    Base class for negative sampling strategies.
    All sampling strategies should inherit from this class.
    """
    
    def __init__(self, seed: int = 42, future_positives: Optional[Set[Tuple[int, int]]] = None, **kwargs):
        """
        Initialise sampler.
        
        Args:
            seed: Random seed for reproducibility
            future_positives: Set of edges that are positive in validation/test sets
                             (to prevent temporal leakage)
            **kwargs: Additional arguments ignored by this sampler
        """
        self.seed = seed
        self.future_positives = future_positives or set()
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
    
    def _filter_candidates(self, candidates: List[Tuple[int, int]], positive_edges: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Filter candidates to remove positives and future positives.
        
        Args:
            candidates: List of candidate edges
            positive_edges: Set of current positive edges
            
        Returns:
            Filtered list of valid negative candidates
        """
        # Remove current positives
        valid = set(candidates) - positive_edges
        
        # Remove future positives (temporal leakage prevention)
        if self.future_positives:
            valid = valid - self.future_positives
        
        return list(valid)


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
        
        # Filter out positive edges and future positives
        negative_candidates = self._filter_candidates(all_possible_pairs, positive_edges)
        
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
    
    def __init__(self, 
                 seed: int = 42, 
                 fallback_to_random: bool = True, 
                 future_positives: Optional[Set[Tuple[int, int]]] = None,
                 max_cn_threshold: Optional[int] = None,
                 min_cn_threshold: int = 1,
                 **kwargs):
        """
        Initialise hard negative sampler.
        
        Args:
            seed: Random seed
            fallback_to_random: If True, supplement with random negatives when hard negatives run out
            future_positives: Set of future positive edges to exclude
            max_cn_threshold: Maximum common neighbors to consider (prevents sampling potential positives)
                             If None, no upper limit. Recommended: 5-10 for drug-disease networks
            min_cn_threshold: Minimum common neighbors to consider as "hard" (default: 1)
        """
        super().__init__(seed, future_positives, **kwargs)
        self.fallback_to_random = fallback_to_random
        self.max_cn_threshold = max_cn_threshold
        self.min_cn_threshold = min_cn_threshold
    
    def sample(self, 
               positive_edges: Set[Tuple[int, int]], 
               all_possible_pairs: List[Tuple[int, int]], 
               num_samples: int,
               edge_index: Optional[torch.Tensor] = None,
               **kwargs) -> List[Tuple[int, int]]:
        """
        Sample the hardest negatives by ranking all candidates by common neighbors.
        Falls back to random sampling if not enough hard negatives exist.
        
        Filters out negatives with too many common neighbors (likely unlabeled positives).
        
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
        if self.max_cn_threshold is not None:
            print(f"  CN range filter: [{self.min_cn_threshold}, {self.max_cn_threshold}]")
        else:
            print(f"  CN minimum threshold: {self.min_cn_threshold}")
        
        # Build adjacency list
        adj_list = defaultdict(set)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].add(dst)
            adj_list[dst].add(src)
        
        # Compute common neighbors for all negative candidates
        candidate_pool = self._filter_candidates(all_possible_pairs, positive_edges)
        
        print(f"Computing common neighbors for {len(candidate_pool):,} negative candidates...")
        cn_scores = []
        
        for src, dst in tqdm(candidate_pool, desc="Computing common neighbors"):
            # Compute common neighbors
            src_neighbors = adj_list[src]
            dst_neighbors = adj_list[dst]
            common_neighbors = src_neighbors & dst_neighbors
            cn_count = len(common_neighbors)
            
            # Filter by CN thresholds
            if cn_count < self.min_cn_threshold:
                continue  # Too easy
            if self.max_cn_threshold is not None and cn_count > self.max_cn_threshold:
                continue  # Too hard (likely unlabeled positive)
            
            cn_scores.append((cn_count, src, dst))
        
        # Sort by common neighbor count (descending - hardest first)
        cn_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Report statistics
        if cn_scores:
            max_cn = cn_scores[0][0]
            min_cn = cn_scores[-1][0]
            avg_cn = np.mean([score[0] for score in cn_scores])
            print(f"\nCommon neighbor statistics (after filtering):")
            print(f"  Min: {min_cn}, Max: {max_cn}, Avg: {avg_cn:.2f}")
            print(f"  Total candidates in range: {len(cn_scores):,}")
            
            # Show distribution of top candidates
            top_k = min(10, len(cn_scores))
            print(f"\nTop {top_k} hardest negatives have CN counts: {[s[0] for s in cn_scores[:top_k]]}")
            
            # Print distribution
            from collections import Counter
            cn_distribution = Counter([score[0] for score in cn_scores])
            print(f"\nDistribution of edges by common neighbor count:")
            print(f"{'CN':>5} | {'Count':>8} | {'%':>8}")
            print("-" * 27)
            for cn_count in sorted(cn_distribution.keys(), reverse=True)[:50]:
                count = cn_distribution[cn_count]
                percentage = (count / len(cn_scores)) * 100
                if percentage < 0.01:
                    print(f"{cn_count:5d} | {count:8,d} | {percentage:7.4f}%")
                elif percentage < 1.0:
                    print(f"{cn_count:5d} | {count:8,d} | {percentage:7.3f}%")
                else:
                    print(f"{cn_count:5d} | {count:8,d} | {percentage:7.2f}%")
        
        # Check if we have enough hard negatives
        num_available = len(cn_scores)
        
        if num_available >= num_samples:
            # Enough hard negatives available
            hard_negatives = [(src, dst) for _, src, dst in cn_scores[:num_samples]]
            
            # Report the CN range of selected samples
            selected_cn_min = cn_scores[num_samples - 1][0]
            selected_cn_max = cn_scores[0][0]
            
            print(f"\nSelected {len(hard_negatives):,} hard negatives")
            print(f"  CN range: [{selected_cn_min} - {selected_cn_max}]")
            
            return hard_negatives
        
        else:
            # Not enough hard negatives - need to supplement with random
            print(f"\n  Only {num_available:,} hard negatives available (need {num_samples:,})")
            
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
            
            print(f"\nSelected {len(all_negatives):,} total negatives:")
            print(f"  - {num_hard:,} hard negatives ({100*num_hard/len(all_negatives):.1f}%)")
            print(f"  - {len(random_negatives):,} random negatives ({100*len(random_negatives)/len(all_negatives):.1f}%)")
            
            return all_negatives


class DegreeMatchedNegativeSampler(NegativeSampler):
    """
    Degree-matched negative sampling.
    Samples negatives where source and target nodes have similar degrees
    to those in positive samples.
    """
    
    def __init__(self, degree_tolerance: float = 0.3, seed: int = 42, future_positives: Optional[Set[Tuple[int, int]]] = None, **kwargs):
        """
        Initialise degree-matched sampler.
        
        Args:
            degree_tolerance: Relative tolerance for degree matching (0.3 = ±30%)
            seed: Random seed
            future_positives: Set of future positive edges to exclude
        """
        super().__init__(seed, future_positives, **kwargs)
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
        candidate_pool = self._filter_candidates(all_possible_pairs, positive_edges)
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
    
    def __init__(self, similarity_threshold: float = 0.5, seed: int = 42, future_positives: Optional[Set[Tuple[int, int]]] = None, **kwargs):
        """
        Initialise feature-similarity sampler.
        
        Args:
            similarity_threshold: Minimum cosine similarity for node features
            seed: Random seed
            future_positives: Set of future positive edges to exclude
        """
        super().__init__(seed, future_positives, **kwargs)
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
        candidate_pool = self._filter_candidates(all_possible_pairs, positive_edges)
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
    Mixed strategy: Combine multiple sampling strategies with adaptive weighting.
    Useful for getting diverse negative samples.
    """
    
    def __init__(self, 
                 strategy_weights: Dict[str, float] = None,
                 degree_tolerance: float = 0.3,
                 similarity_threshold: float = 0.5,
                 seed: int = 42,
                 future_positives: Optional[Set[Tuple[int, int]]] = None,
                 adaptive: bool = False,
                 max_cn_threshold: Optional[int] = None,
                 min_cn_threshold: int = 1,
                 **kwargs):
        """
        Initialise mixed sampler.
        
        Args:
            strategy_weights: Dict mapping strategy names to their weights
                Example: {'hard': 0.5, 'degree_matched': 0.3, 'random': 0.2}
                If None, uses recommended defaults
            degree_tolerance: Passed to DegreeMatchedNegativeSampler
            similarity_threshold: Passed to FeatureSimilarityNegativeSampler
            seed: Random seed
            future_positives: Set of future positive edges to exclude
            adaptive: If True, enables curriculum learning
            max_cn_threshold: Maximum common neighbors for hard negatives (prevents hidden positives)
                             Recommended: 5-10 for drug-disease networks
            min_cn_threshold: Minimum common neighbors for hard negatives
        """
        super().__init__(seed, future_positives, **kwargs)
        
        if strategy_weights is None:
            strategy_weights = {
                'hard': 0.5,           # Moderate hard negatives
                'degree_matched': 0.3, # Structural similarity control
                'random': 0.2          # Random for diversity
            }
        
        self.strategy_weights = strategy_weights
        self.adaptive = adaptive
        self.training_progress = 0.0  # For curriculum learning (0 = start, 1 = end)
        
        # Initialise sub-samplers with CN thresholds to avoid hidden positives
        self.samplers = {
            'hard': HardNegativeSampler(
                seed=seed, 
                future_positives=future_positives,
                max_cn_threshold=max_cn_threshold,  
                min_cn_threshold=min_cn_threshold,
                fallback_to_random=True
            ),
            'degree_matched': DegreeMatchedNegativeSampler(
                degree_tolerance=degree_tolerance, 
                seed=seed, 
                future_positives=future_positives
            ),
            'random': RandomNegativeSampler(seed=seed, future_positives=future_positives),
            'feature_similar': FeatureSimilarityNegativeSampler(
                similarity_threshold=similarity_threshold, 
                seed=seed, 
                future_positives=future_positives
            )
        }
    
    def set_training_progress(self, epoch: int, total_epochs: int):
        """
        Update training progress for curriculum learning.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs
        """
        self.training_progress = epoch / max(total_epochs - 1, 1)
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """
        Get adaptive weights based on training progress (curriculum learning).
        
        Strategy:
        - Early training (0-30%): More random negatives (easier to learn)
        - Mid training (30-70%): Balanced mix
        - Late training (70-100%): More hard negatives (fine-tuning)
        
        Returns:
            Adjusted strategy weights
        """
        if not self.adaptive:
            return self.strategy_weights
        
        progress = self.training_progress
        
        # Curriculum schedule
        if progress < 0.3:
            # Early stage: Start with easier negatives
            weights = {
                'hard': 0.2,
                'degree_matched': 0.3,
                'random': 0.5
            }
        elif progress < 0.7:
            # Mid stage: Balanced mix (use defaults)
            weights = self.strategy_weights.copy()
        else:
            # Late stage: Focus on hard negatives for fine-tuning
            weights = {
                'hard': 0.6,
                'degree_matched': 0.3,
                'random': 0.1
            }
        
        # Normalise to sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def sample(self, 
               positive_edges: Set[Tuple[int, int]], 
               all_possible_pairs: List[Tuple[int, int]], 
               num_samples: int,
               **kwargs) -> List[Tuple[int, int]]:
        """
        Sample negatives using mixed strategy with optional curriculum learning.
        
        Args:
            positive_edges: Set of positive edges
            all_possible_pairs: All possible pairs
            num_samples: Number of samples needed
            **kwargs: Passed to individual samplers
            
        Returns:
            List of mixed negative samples
        """
        
        # Get current weights (adaptive or fixed)
        current_weights = self.get_adaptive_weights()
        
        if self.adaptive:
            print(f"Mixed strategy (adaptive, progress={self.training_progress:.2f}): {current_weights}")
        else:
            print(f"Mixed strategy (fixed): {current_weights}")
        
        # Consolidate unique negatives from all strategies
        all_negatives_set = set()
        
        # Sample from each strategy according to weights
        for strategy_name, weight in current_weights.items():
            if weight <= 0:
                continue
            
            n_samples = int(num_samples * weight)
            if n_samples == 0:
                continue
            
            if strategy_name not in self.samplers:
                continue
            
            sampler = self.samplers[strategy_name]
            
            try:
                negatives = sampler.sample(
                    positive_edges, 
                    all_possible_pairs, 
                    n_samples,
                    **kwargs
                )
                original_len = len(all_negatives_set)
                all_negatives_set.update(negatives)
                new_len = len(all_negatives_set)
                print(f"  {strategy_name}: added {new_len - original_len} unique negatives (requested {n_samples})")
            except Exception as e:
                import traceback
                print(f"  {strategy_name}: failed ({e})")
                # print(traceback.format_exc())
        
        # If we don't have enough unique samples, fill with random
        if len(all_negatives_set) < num_samples:
            deficit = num_samples - len(all_negatives_set)
            print(f"Filling {deficit} samples with additional random negatives to reach target {num_samples}...")
            
            random_sampler = RandomNegativeSampler(seed=self.seed, future_positives=self.future_positives)
            
            # Continue sampling until target is reached or we run out of candidates
            max_attempts = 3
            for _ in range(max_attempts):
                additional = random_sampler.sample(
                    positive_edges | all_negatives_set,
                    all_possible_pairs,
                    deficit,
                    **kwargs
                )
                all_negatives_set.update(additional)
                deficit = num_samples - len(all_negatives_set)
                if deficit <= 0:
                    break
        
        # Shuffle and return exact number
        all_negatives_list = list(all_negatives_set)
        random.shuffle(all_negatives_list)
        return all_negatives_list[:num_samples]


def get_sampler(strategy: str = 'random', future_positives: Optional[Set[Tuple[int, int]]] = None, **kwargs) -> NegativeSampler:
    """
    Factory function to get a negative sampler.
    
    Args:
        strategy: Name of sampling strategy
            - 'random': Random sampling (default, baseline)
            - 'hard': Hard negatives with common neighbors
            - 'degree_matched': Degree-matched negatives
            - 'feature_similar': Feature-similar negatives
            - 'mixed': Mixed strategy
        future_positives: Set of edges that are positive in validation/test sets
                         (to prevent temporal leakage)
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
    
    return samplers[strategy](future_positives=future_positives, **kwargs)

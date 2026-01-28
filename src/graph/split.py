"""
Data splitting and sampling strategy.
"""

import torch
import random
import numpy as np
from typing import Optional, Set, Tuple
from src.utils.edge_utils import generate_pairs
from src.negative_sampling import get_sampler

class DataSplitter:
    """Handles train/validation/test splitting and negative sampling."""
    
    def __init__(self, config, processor):
        """
        Initialize data splitter.
        
        Args:
            config: Configuration object
            processor: DataProcessor instance
        """
        self.config = config
        self.processor = processor
        
    def create_splits(self, mappings, all_edges, all_features, edge_type_mask=None):
        """
        Create training, validation, test splits with negative sampling.
        
        Args:
            mappings: Node mappings dictionary
            all_edges: Tensor of all edges [2, num_edges]
            all_features: Feature matrix for all nodes
            edge_type_mask: Dictionary mapping edge types to slices (optional)
            
        Returns:
            Dictionary containing split tensors (train_edges, train_labels, etc.)
        """
        print("Creating train/validation/test splits...")
        print(f"Negative sampling strategy: {self.config.negative_sampling_strategy}")
        print(f"Pos:Neg ratio: 1:{self.config.train_neg_ratio} (training), 1:{self.config.pos_neg_ratio} (val/test)")
        
        # We need drug-disease edges specifically for splitting
        # If we have edge_type_mask, we can extract them. But typically GraphBuilder has specific access.
        # In this refactor, we rely on the caller (GraphBuilder) to provide the right context or 
        # we regenerate the set of edges we care about.
        
        # Problem: how to identify drug-disease edges from 'all_edges' if it's concatenated?
        # GraphBuilder knows `self.edge_builder.edges['molecule_disease']`.
        # So we should probably pass `drug_disease_edges` explicitly.
        # But for now, let's assume we can reconstruct or filter them.
        pass

    def create_splits_from_edges(self, drug_disease_edges, all_edge_index, all_features, mappings):
        """
        Create splits using specific drug-disease edges as the positive set.
        
        Args:
            drug_disease_edges: Tensor of drug-disease edges [2, num_edges]
            all_edge_index: Tensor of ALL edges in the graph (for negative sampling context)
            all_features: Feature matrix for all nodes
            mappings: Node mappings dict
            
        Returns:
            Dictionary of split tensors
        """
        # Extract training edges 
        train_edges_set = set(zip(
            drug_disease_edges[0].tolist(),
            drug_disease_edges[1].tolist()
        ))
        
        print(f"Training edges: {len(train_edges_set)}")
        
        # Generate all possible drug-disease pairs
        all_possible_pairs = generate_pairs(
            mappings['approved_drugs_list'],
            mappings['disease_list'],
            mappings['drug_key_mapping'],
            mappings['disease_key_mapping']
        )
        
        print(f"Total possible drug-disease pairs: {len(all_possible_pairs)}")
        
        # Generate validation and test positive splits using temporal data
        try:
            new_val_edges_set, new_test_edges_set = self.processor.generate_validation_test_splits(
                self.config, mappings, train_edges_set
            )
            print(f"Generated temporal splits from OpenTargets data")
        except Exception as e:
            print(f"Warning: Could not generate temporal splits: {e}")
            print("Creating synthetic splits...")
            
            # Fallback: create synthetic splits
            not_linked = list(set(all_possible_pairs) - train_edges_set)
            
            random.seed(self.config.seed)
            val_size = min(len(train_edges_set) // 10, len(not_linked) // 2)
            test_size = min(len(train_edges_set) // 10, len(not_linked) - val_size)
            
            new_val_edges_set = set(random.sample(not_linked, val_size))
            remaining_not_linked = list(set(not_linked) - new_val_edges_set)
            new_test_edges_set = set(random.sample(remaining_not_linked, test_size))
        
        print(f"Validation positive edges: {len(new_val_edges_set)}")
        print(f"Test positive edges: {len(new_test_edges_set)}")
        
        # Calculate total negatives needed
        train_true_pairs = list(train_edges_set)
        val_true_pairs = list(new_val_edges_set)
        test_true_pairs = list(new_test_edges_set)
        
        num_train_negatives = len(train_true_pairs) * self.config.train_neg_ratio
        num_val_negatives = len(val_true_pairs) * self.config.pos_neg_ratio
        num_test_negatives = len(test_true_pairs) * self.config.pos_neg_ratio
        total_negatives_needed = num_train_negatives + num_val_negatives + num_test_negatives
        
        print(f"\nNegatives needed:")
        print(f"  Training: {num_train_negatives} (1:{self.config.train_neg_ratio} ratio)")
        print(f"  Validation: {num_val_negatives} (1:{self.config.pos_neg_ratio} ratio)")
        print(f"  Test: {num_test_negatives} (1:{self.config.pos_neg_ratio} ratio)")
        print(f"  TOTAL: {total_negatives_needed}")
        
        # Sample ALL negatives at once
        all_positive_edges = train_edges_set | new_val_edges_set | new_test_edges_set
        
        print(f"\nSampling ALL {total_negatives_needed} negatives at once...")
        sampler = self._create_sampler(future_positives=new_val_edges_set | new_test_edges_set)
        all_negative_pairs = sampler.sample(
            positive_edges=all_positive_edges,
            all_possible_pairs=all_possible_pairs,
            num_samples=total_negatives_needed,
            edge_index=all_edge_index,
            node_features=all_features
        )
        print(f"Sampled {len(all_negative_pairs)} unique negatives")
        
        # Randomly split negatives
        random.seed(self.config.seed)
        shuffled_negatives = list(all_negative_pairs)
        random.shuffle(shuffled_negatives)
        
        # Distribute negatives proportionally to maintain equal ratios across splits
        actual_total_negatives = len(shuffled_negatives)
        
        if actual_total_negatives < total_negatives_needed and actual_total_negatives > 0:
            print(f"Warning: Only {actual_total_negatives} negatives available (target: {total_negatives_needed})")
            print("Distributing negatives proportionally across splits...")
            
            # Simple proportional allocation
            train_share = num_train_negatives / total_negatives_needed
            val_share = num_val_negatives / total_negatives_needed
            
            n_train = int(actual_total_negatives * train_share)
            n_val = int(actual_total_negatives * val_share)
            
            train_false_pairs = shuffled_negatives[:n_train]
            val_false_pairs = shuffled_negatives[n_train:n_train + n_val]
            test_false_pairs = shuffled_negatives[n_train + n_val:]
        else:
            # Standard exact allocation
            train_false_pairs = shuffled_negatives[:num_train_negatives]
            val_false_pairs = shuffled_negatives[num_train_negatives:num_train_negatives + num_val_negatives]
            test_false_pairs = shuffled_negatives[num_train_negatives + num_val_negatives:num_train_negatives + num_val_negatives + num_test_negatives]
            
        # Final Summary Statistics
        print("\n" + "="*50)
        print("FINAL DATASET SPLIT SUMMARY")
        print("="*50)
        print(f"{'Split':<12} | {'Positive':<10} | {'Negative':<10} | {'Ratio':<6}")
        print("-" * 50)
        
        def print_row(name, pos, neg):
            ratio = f"1:{neg/pos:.1f}" if pos > 0 else "N/A"
            print(f"{name:<12} | {pos:<10,} | {neg:<10,} | {ratio:<6}")
            
        print_row("Training", len(train_true_pairs), len(train_false_pairs))
        print_row("Validation", len(val_true_pairs), len(val_false_pairs))
        print_row("Test", len(test_true_pairs), len(test_false_pairs))
        print("="*50 + "\n")
        
        # Verify splits logic
        self._verify_splits(train_false_pairs, val_false_pairs, test_false_pairs, all_positive_edges)
        
        # Create tensors
        train_labels = [1] * len(train_true_pairs) + [0] * len(train_false_pairs)
        val_labels = [1] * len(val_true_pairs) + [0] * len(val_false_pairs)
        test_labels = [1] * len(test_true_pairs) + [0] * len(test_false_pairs)
        
        return {
            'train': {
                'edge_index': torch.tensor(train_true_pairs + train_false_pairs, dtype=torch.long),
                'label': torch.tensor(train_labels, dtype=torch.long)
            },
            'val': {
                'edge_index': torch.tensor(val_true_pairs + val_false_pairs, dtype=torch.long),
                'label': torch.tensor(val_labels, dtype=torch.long)
            },
            'test': {
                'edge_index': torch.tensor(test_true_pairs + test_false_pairs, dtype=torch.long),
                'label': torch.tensor(test_labels, dtype=torch.long)
            }
        }
        
    def _create_sampler(self, future_positives: Optional[Set[Tuple[int, int]]] = None):
        """Create negative sampler."""
        strategy = self.config.negative_sampling_strategy
        params = self.config.neg_sampling_params or {}
        
        return get_sampler(
            strategy=strategy,
            future_positives=future_positives,
            seed=self.config.seed,
            **params
        )
        
    def _verify_splits(self, train_neg, val_neg, test_neg, all_pos):
        """Verify split integrity."""
        train_neg_set = set(train_neg)
        val_neg_set = set(val_neg)
        test_neg_set = set(test_neg)
        
        overlap = (train_neg_set | val_neg_set | test_neg_set) & all_pos
        if overlap:
            print(f"WARNING: {len(overlap)} negatives overlap with positives!")
        else:
            print("Split verification passed: No negative/positive overlap.")

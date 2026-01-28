"""
Edge feature extraction module for GNN pipeline.
Handles extraction of edge features from OpenTargets datasets.
"""

import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_moa_features(moa_df: pd.DataFrame, 
                         drug_key_mapping: Dict[str, int],
                         gene_key_mapping: Dict[str, int],
                         existing_drug_gene_edges: torch.Tensor) -> torch.Tensor:
    """
    Extract mechanism of action features for drug-gene edges.
    
    Args:
        moa_df: DataFrame with mechanismOfAction data
        drug_key_mapping: Mapping from drug IDs to node indices
        gene_key_mapping: Mapping from gene IDs to node indices
        existing_drug_gene_edges: Tensor of shape [2, num_edges] with drug-gene edges
        
    Returns:
        Tensor of shape [num_edges, 6] with action type features
    """
    print("Extracting mechanism of action edge features...")
    
    if moa_df.empty:
        print("  Warning: No mechanismOfAction data available")
        num_edges = existing_drug_gene_edges.shape[1]
        return torch.zeros((num_edges, 6), dtype=torch.float32)
    
    # Define action type categories
    ACTION_TYPES = {
        'inhibitor': 0,
        'antagonist': 1,
        'agonist': 2,
        'activator': 3,
        'modulator': 4,
        'other': 5
    }
    
    # Build drug-gene to action type mapping
    # Key: (drug_idx, gene_idx), Value: action_type_index
    edge_to_action = {}
    
    print(f"  Processing {len(moa_df)} mechanism of action records...")
    
    # Iterate through mechanismOfAction records
    for _, row in moa_df.iterrows():
        # Get drug ID(s) - handle numpy arrays from parquet
        drug_ids = row.get('chemblIds', [])
        if drug_ids is None:
            drug_ids = []
        elif isinstance(drug_ids, str):
            drug_ids = [drug_ids]
        elif hasattr(drug_ids, '__iter__'):
            # Handle numpy arrays, lists, or any iterable
            drug_ids = list(drug_ids)
        else:
            drug_ids = []
        
        # Get target/gene ID(s) - handle numpy arrays from parquet
        target_ids = row.get('targets', [])
        if target_ids is None:
            target_ids = []
        elif isinstance(target_ids, str):
            target_ids = [target_ids]
        elif hasattr(target_ids, '__iter__'):
            # Handle numpy arrays, lists, or any iterable
            target_ids = list(target_ids)
        else:
            target_ids = []
        
        # Get action type
        action_type_raw = row.get('actionType', 'other')
        if pd.isna(action_type_raw):
            action_type_raw = 'other'
        
        # Normalise action type to our categories
        action_type = normalise_action_type(action_type_raw, ACTION_TYPES)
        
        # Create mappings for all drug-gene pairs
        for drug_id in drug_ids:
            if drug_id not in drug_key_mapping:
                continue
            
            drug_idx = drug_key_mapping[drug_id]
            
            for target_id in target_ids:
                if target_id not in gene_key_mapping:
                    continue
                
                gene_idx = gene_key_mapping[target_id]
                edge_key = (drug_idx, gene_idx)
                
                # Store action type (if multiple, keep the first one)
                if edge_key not in edge_to_action:
                    edge_to_action[edge_key] = action_type
    
    print(f"  Found action types for {len(edge_to_action)} drug-gene pairs")
    
    # Create feature matrix for existing edges
    num_edges = existing_drug_gene_edges.shape[1]
    edge_features = torch.zeros((num_edges, 6), dtype=torch.float32)
    
    edges_with_features = 0
    
    for i in range(num_edges):
        drug_idx = existing_drug_gene_edges[0, i].item()
        gene_idx = existing_drug_gene_edges[1, i].item()
        edge_key = (drug_idx, gene_idx)
        
        if edge_key in edge_to_action:
            action_idx = edge_to_action[edge_key]
            edge_features[i, action_idx] = 1.0
            edges_with_features += 1
        else:
            # No mechanismOfAction data for this edge: use 'other' category
            edge_features[i, ACTION_TYPES['other']] = 1.0
    
    print(f"  Created edge features: {edge_features.shape}")
    print(f"  Edges with mechanismOfAction data: {edges_with_features}/{num_edges} ({100*edges_with_features/num_edges:.1f}%)")
    
    # Print distribution of action types
    action_counts = edge_features.sum(dim=0)
    print(f"  Action type distribution:")
    for action_name, action_idx in ACTION_TYPES.items():
        count = int(action_counts[action_idx].item())
        print(f"    {action_name:12s}: {count:5d} ({100*count/num_edges:.1f}%)")
    
    return edge_features


def normalise_action_type(action_type_raw: str, action_types_dict: Dict[str, int]) -> int:
    """
    Normalise raw action type string to one of our categories.
    
    Args:
        action_type_raw: Raw action type from mechanismOfAction
        action_types_dict: Dictionary mapping action names to indices
        
    Returns:
        Index of the normalised action type
    """
    action_lower = str(action_type_raw).lower().strip()
    
    # Direct matches
    if action_lower in action_types_dict:
        return action_types_dict[action_lower]
    
    # Partial matches for common variations
    if 'inhibit' in action_lower or 'blocker' in action_lower:
        return action_types_dict['inhibitor']
    elif 'antagonist' in action_lower or 'inverse agonist' in action_lower:
        return action_types_dict['antagonist']
    elif 'agonist' in action_lower:
        return action_types_dict['agonist']
    elif 'activat' in action_lower or 'inducer' in action_lower:
        return action_types_dict['activator']
    elif 'modulat' in action_lower or 'regulator' in action_lower:
        return action_types_dict['modulator']
    else:
        return action_types_dict['other']


def pad_edge_features_to_match_all_edges(drug_gene_edge_features: torch.Tensor,
                                         num_drug_gene_edges: int,
                                         total_edges: int) -> torch.Tensor:
    """
    Pad drug-gene edge features with zeros for other edge types.
    
    Since only drug-gene edges have mechanismOfAction features, we need to
    add zero features for all other edge types (drug-drugtype, gene-reactome, etc.)
    
    Args:
        drug_gene_edge_features: Features for drug-gene edges [num_drug_gene_edges, feature_dim]
        num_drug_gene_edges: Number of drug-gene edges
        total_edges: Total number of edges in the graph
        
    Returns:
        Padded edge features [total_edges, feature_dim]
    """
    if drug_gene_edge_features.shape[0] != num_drug_gene_edges:
        raise ValueError(f"Feature shape mismatch: expected {num_drug_gene_edges} drug-gene edges, "
                        f"got {drug_gene_edge_features.shape[0]}")
    
    feature_dim = drug_gene_edge_features.shape[1]
    
    # Create tensor with zeros for all edges
    all_edge_features = torch.zeros((total_edges, feature_dim), dtype=torch.float32)
    
    # The drug-gene edges will be at a specific position in the concatenated edge tensor
    # This function should be called with the correct slice of edges
    
    return all_edge_features


def create_edge_type_mask(edge_counts: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    """
    Create masks to identify which edges belong to which type.
    
    Args:
        edge_counts: Dictionary with edge type names and their counts
        
    Returns:
        Dictionary mapping edge type to (start_idx, end_idx) tuple
    """
    edge_type_slices = {}
    current_idx = 0
    
    for edge_type, count in edge_counts.items():
        edge_type_slices[edge_type] = (current_idx, current_idx + count)
        current_idx += count
    
    return edge_type_slices

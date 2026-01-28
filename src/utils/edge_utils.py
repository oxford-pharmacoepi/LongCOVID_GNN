"""
Utilities for edge extraction and tensor generation.
"""

import torch

def get_indices_from_keys(key_list, index_mapping):
    """Get indices from keys using mapping dictionary."""
    return [index_mapping[key] for key in key_list if key in index_mapping]


def generate_pairs(source_list, target_list, source_mapping, target_mapping, return_set=False, return_tensor=False):
    """Generate all possible edge combinations from 2 lists."""
    edges = []
    for source_id in source_list:
        for target_id in target_list:
            if source_id in source_mapping and target_id in target_mapping:
                edges.append((source_mapping[source_id], target_mapping[target_id]))
    
    if return_set:
        return set(edges)
    elif return_tensor: 
        edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index_tensor
    else: 
        return edges


def generate_tensor(source_list, target_list, source_mapping, target_mapping):
    """Generate tensor for edge combinations from parallel lists."""
    edges = []
    for i in range(len(source_list)):
        if source_list[i] in source_mapping and target_list[i] in target_mapping:
            edges.append((source_mapping[source_list[i]], target_mapping[target_list[i]]))
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index_tensor


def extract_edges(table, source_mapping, target_mapping, return_edge_list=False, return_edge_set=False, debug=False):
    """Extract edges from a PyArrow table."""
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    missing_sources = set()
    missing_targets = set()
    
    for i in range(len(source)):
        source_id = source[i].as_py()
        target_list = targets.slice(i, 1).to_pylist()[0]
        
        # Ensure target_list is actually a list
        if not isinstance(target_list, list):
            target_list = [target_list]
        
        # Track missing mappings
        if source_id not in source_mapping:
            missing_sources.add(source_id)
        
        # Create edges for each target
        for target_id in target_list:
            if target_id not in target_mapping:
                missing_targets.add(target_id)
            
            if source_id in source_mapping and target_id in target_mapping:
                edges.append((source_mapping[source_id], target_mapping[target_id]))
    
    # Debug output if requested
    if debug and (missing_sources or missing_targets):
        print(f"        [DEBUG] Missing mappings:")
        print(f"          Sources not in mapping: {len(missing_sources)}")
        if len(missing_sources) <= 10:
            print(f"            Examples: {list(missing_sources)[:10]}")
        else:
            print(f"            Examples: {list(missing_sources)[:10]}")
        print(f"          Targets not in mapping: {len(missing_targets)}")
        if len(missing_targets) <= 10:
            print(f"            Examples: {list(missing_targets)[:10]}")
        else:
            print(f"            Examples: {list(missing_targets)[:10]}")

    if return_edge_list:
        return edges
    elif return_edge_set:
        return set(edges)
    else:
        unique_edges = list(set(edges))  # Deduplicate
        edge_index_tensor = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        return edge_index_tensor


def extract_edges_no_mapping(table, return_edge_list=False, return_edge_set=False):
    """Extract edges without index mapping (for debugging)."""
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    for i in range(len(source)):
        source_id = source[i].as_py()
        target_list = targets.slice(i, 1).to_pylist()[0]
        
        if not isinstance(target_list, list):
            target_list = [target_list]
        
        for target_id in target_list:
            edges.append(f"{source_id} -> {target_id}")
    
    if return_edge_list:
        return edges
    elif return_edge_set:
        return set(edges)
    else:
        return edges

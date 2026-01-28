"""
Graph analysis and construction utilities.
"""

import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import to_networkx
import networkx as nx
from collections import defaultdict
from .edge_utils import extract_edges

def standard_graph_analysis(graph):
    """Perform standard graph analysis and statistics."""
    
    # Convert to NetworkX for analysis
    G = to_networkx(graph, to_undirected=True)
    
    print("=== STANDARD GRAPH ANALYSIS ===")
    
    # Basic stats
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    print(f"Density: {nx.density(G):.4f}")
    
    # Degree stats
    degrees = [d for n, d in G.degree()]
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Degree std: {np.std(degrees):.2f}")
    
    # Connectivity
    is_connected = nx.is_connected(G)
    print(f"Connected: {is_connected}")
    
    if is_connected:
        # Path analysis for connected graphs
        avg_path = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
        print(f"Average path length: {avg_path:.2f}")
        print(f"Diameter: {diameter}")
    else:
        # Component analysis for disconnected graphs
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len)
        print(f"Connected components: {len(components)}")
        print(f"Largest component: {len(largest_cc)} nodes ({len(largest_cc)/G.number_of_nodes()*100:.1f}%)")
    
    # Clustering
    clustering = nx.average_clustering(G)
    print(f"Average clustering: {clustering:.4f}")
    
    # Assortativity
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"Degree assortativity: {assortativity:.4f}")
    
    # Centrality summary
    betweenness = list(nx.betweenness_centrality(G).values())
    print(f"Average betweenness centrality: {np.mean(betweenness):.4f}")
    
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': np.mean(degrees),
        'max_degree': max(degrees),
        'is_connected': is_connected,
        'avg_clustering': clustering,
        'assortativity': assortativity,
        'avg_betweenness': np.mean(betweenness)
    }


def generate_edge_list(source_list, target_list, source_mapping, target_mapping):
    """Generate edge list from parallel source and target lists."""
    edges = []
    for i in range(len(source_list)):
        source_id = source_list[i]
        target_id = target_list[i]
        if source_id in source_mapping and target_id in target_mapping:
            edges.append((source_mapping[source_id], target_mapping[target_id]))
    return edges


def as_negative_sampling(filtered_molecule_table, associations_table, score_column, 
                        drug_key_mapping, disease_key_mapping, return_list=False, return_set=False):
    """Association score-based negative sampling."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import polars as pl
    from tqdm import tqdm
    
    # Table with Molecule and linked targets
    MLT = filtered_molecule_table.select(['id', 'linkedTargets.rows']).drop_null()
    
    # Table with Disease and linked targets and association scores
    DT = associations_table.select(['diseaseId', 'targetId', score_column])
    
    # Convert to pandas DataFrames for processing
    df_DT = DT.to_pandas()
    df_MLT = MLT.to_pandas()
    
    # Explode the linkedTargets to create molecule-target pairs
    df_MLT_exploded = df_MLT.explode('linkedTargets.rows').reset_index(drop=True)
    df_MLT_exploded.rename(columns={'linkedTargets.rows': 'targetId'}, inplace=True)
    
    MLT_exploded = pa.Table.from_pandas(df_MLT_exploded)
    
    # Memory management for large files
    if len(DT) > 1000000:
        print("Processing large dataset in chunks...")
        # Use Polars for efficient processing
        pl_MLT_exploded = pl.from_pandas(df_MLT_exploded)
        pl_DT = pl.from_pandas(df_DT)
        
        final_df = pl.DataFrame()
        
        for i in tqdm(range(0, len(DT), 1000000), desc="Processing chunks"):
            chunk_size = min(1000000, len(DT) - i)
            t_DT = DT.slice(i, chunk_size)
            
            pl_t_DT = pl.from_arrow(t_DT)
            joined_chunk = pl_MLT_exploded.join(pl_t_DT, on='targetId')
            sorted_chunk = joined_chunk.sort(score_column)
            
            final_df = pl.concat([final_df, sorted_chunk], how='vertical')
        
        MTD_table = final_df.to_arrow()
    else:
        # Simple join for smaller datasets
        MTD_table = MLT_exploded.join(DT, 'targetId').combine_chunks().sort_by(score_column)
    
    # Filter for low association scores (negative samples)
    expr = pc.field(score_column) <= pc.scalar(0.01)
    filtered_MTD = MTD_table.filter(expr)
    
    # Create negative sample pairs
    negative_sample_table = filtered_MTD.drop_columns(['targetId', score_column]).drop_null()
    
    if len(negative_sample_table) == 0:
        print("Warning: No negative samples found with current threshold")
        return [] if return_list else set() if return_set else torch.empty((2, 0), dtype=torch.long)
    
    mlist = negative_sample_table.column('id').combine_chunks().to_pylist()
    dlist = negative_sample_table.column('diseaseId').combine_chunks().to_pylist()
    
    # Create edge list
    ng_list = generate_edge_list(mlist, dlist, drug_key_mapping, disease_key_mapping)
    
    if return_list:
        return ng_list
    elif return_set:   
        return set(ng_list)
    else: 
        return torch.tensor(ng_list, dtype=torch.long).t().contiguous()


def find_repurposing_edges(table1, table2, column_name, source_mapping, target_mapping):
    """Find potential drug repurposing edges."""
    import pyarrow.compute as pc
    
    # Create filter mask
    filter_mask = pc.is_in(table2.column('id'), value_set=table1.column('id'))
    filtered_table = table2.filter(filter_mask)
    
    # Create new edge list
    new_edge_list = []
    for i in range(len(filtered_table)):
        row = filtered_table.slice(i, 1)
        drug_id = row.column('id').combine_chunks()[0].as_py()
        linked_items = row.column(column_name).combine_chunks()[0].as_py()

        for item in linked_items:
            if drug_id in source_mapping and item in target_mapping:
                new_edge_list.append((source_mapping[drug_id], target_mapping[item]))
    
    return new_edge_list


def create_disease_similarity_edges_from_ancestors(disease_table, disease_key_mapping, 
                                                   max_children_per_parent=10, min_shared_ancestors=1):
    """
    Create disease-disease similarity edges based on shared direct parent ancestor.
    """
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    
    print(f"    Building disease similarity network from shared direct parent...")
    
    # Extract disease IDs and their ancestors
    disease_df = disease_table.select(['id', 'ancestors']).to_pandas()
    
    # Build mapping: ancestor -> list of diseases that have this as direct parent
    ancestor_to_diseases = defaultdict(set)
    
    for _, row in disease_df.iterrows():
        disease_id = row['id']
        ancestors = row['ancestors']
        
        # Skip if disease is not in our node mapping
        if disease_id not in disease_key_mapping:
            continue
        
        # Handle None or empty ancestors
        if ancestors is None:
            continue
        
        # Convert numpy array to list if needed
        if isinstance(ancestors, np.ndarray):
            ancestors = ancestors.tolist()
        
        # Skip if not a list or empty
        if not isinstance(ancestors, list) or len(ancestors) == 0:
            continue
        
        # Use only the first ancestor (direct parent), index 0
        if len(ancestors) > 0:
            direct_parent = ancestors[0]
            ancestor_to_diseases[direct_parent].add(disease_id)
    
    print(f"    Found {len(ancestor_to_diseases)} unique direct parents")
    
    # Build disease-disease edges based on shared ancestors
    edges = []
    shared_ancestor_counts = defaultdict(int)
    
    # For each ancestor, connect all diseases that share it
    # Only use parents that are specific enough
    for ancestor_id, disease_set in ancestor_to_diseases.items():
        if len(disease_set) > max_children_per_parent:
            print(f"    Skipping extremely common parent {ancestor_id} ({len(disease_set)} diseases)")
            continue
            
        disease_list = list(disease_set)
        
        # Create edges between all pairs of diseases that share this ancestor
        for i in range(len(disease_list)):
            for j in range(i + 1, len(disease_list)):
                disease_1 = disease_list[i]
                disease_2 = disease_list[j]
                
                # Track how many ancestors these diseases share
                shared_ancestor_counts[(disease_1, disease_2)] += 1
    
    # Create edges for disease pairs that share at least min_shared_ancestors
    for (disease_1, disease_2), count in shared_ancestor_counts.items():
        if count >= min_shared_ancestors:
            idx1 = disease_key_mapping[disease_1]
            idx2 = disease_key_mapping[disease_2]
            
            # Add bidirectional edges
            edges.append((idx1, idx2))
            edges.append((idx2, idx1))
    
    print(f"    Created {len(edges)} disease-disease similarity edges (threshold={min_shared_ancestors})")
    print(f"    Unique disease pairs: {len(edges) // 2}")
    
    return edges





def custom_edges(disease_similarity_network, disease_similarity_max_children, disease_similarity_min_shared,
                trial_edges, filtered_disease_table, filtered_molecule_table,
                disease_key_mapping, drug_key_mapping):
    """Generate custom edges for enhanced graph connectivity."""
    
    print("\n" + "="*80)
    print("CUSTOM_EDGES FUNCTION - CREATING DISEASE SIMILARITY NETWORK")
    print("="*80)
    
    custom_edges = []

    # Disease similarity network edges (shared ancestors)
    if disease_similarity_network:
        print("\n>>> Processing Disease Similarity Network...")
        try:
            similarity_edges = create_disease_similarity_edges_from_ancestors(
                filtered_disease_table, 
                disease_key_mapping,
                max_children_per_parent=disease_similarity_max_children,
                min_shared_ancestors=disease_similarity_min_shared
            )
            custom_edges.extend(similarity_edges)
        except Exception as e:
            print(f"        ERROR creating similarity edges: {e}")
            

    # Trial edges
    if trial_edges:
        print("\n>>> Processing Trial Edges...")
        molecule_trial_table = filtered_molecule_table.select(['id', 'linkedDiseases.rows']).flatten()
        trial_edges_list = extract_edges(molecule_trial_table, drug_key_mapping, disease_key_mapping, return_edge_list=True, debug=False)
        print(f"    Extracted {len(trial_edges_list)} trial edges")
        custom_edges.extend(trial_edges_list)


    # Handle empty case : return tensor with shape [2, 0]
    print(f"\n>>> FINAL RESULT: {len(custom_edges)} total custom edges")
    print("="*80 + "\n")
    
    if len(custom_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    custom_edge_tensor = torch.tensor(custom_edges, dtype=torch.long).t().contiguous()
    return custom_edge_tensor

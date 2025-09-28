"""
Shared utility functions for drug-disease prediction pipeline.
Common functions used across all modules.
"""

import random
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import math


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_full_reproducibility(seed=42):
    """Enable full reproducibility with deterministic algorithms."""
    set_seed(seed)
    torch.use_deterministic_algorithms(True)


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


def extract_edges(table, source_mapping, target_mapping, return_edge_list=False, return_edge_set=False):
    """Extract edges from a PyArrow table."""
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    for i in range(len(source)):
        source_id = source[i].as_py()
        target_list = targets.slice(i, 1).to_pylist()[0]
        
        # Ensure target_list is actually a list
        if not isinstance(target_list, list):
            target_list = [target_list]
        
        # Create edges for each target
        for target_id in target_list:
            if source_id in source_mapping and target_id in target_mapping:
                edges.append((source_mapping[source_id], target_mapping[target_id]))

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


# Feature Engineering Functions
def boolean_encode(boolean_array, pad_length):
    """Encode boolean arrays with padding."""
    boolean_series = pd.Series(boolean_array.to_pandas()).astype("float")
    boolean_array_filled = boolean_series.fillna(-1).to_numpy().reshape(-1, 1)
    tensor = torch.from_numpy(boolean_array_filled.astype(np.int64))

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor


def normalize(array, pad_length):
    """Normalize arrays with padding."""
    df = array.to_pandas().to_numpy().reshape(-1, 1)
    df = pd.DataFrame(df)
    df.fillna(-1, inplace=True)
    standardized = (df - df.mean()) / df.std()
    tensor = torch.from_numpy(standardized.to_numpy())

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor


def cat_encode(array, pad_length):
    """Encode categorical variables with padding."""
    uni = array.unique().to_pandas()
    unidict = {uni[i]: i for i in range(len(uni))}
    
    tensor = torch.tensor([unidict[i] for i in array.to_pandas()], dtype=torch.int32)

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor


def pad_feature_matrix(matrix, pad_size, pad_value=-1):
    """Pad feature matrix to specified size."""
    if matrix.size(1) < pad_size:
        padding = torch.ones(matrix.size(0), pad_size - matrix.size(1)) * pad_value
        matrix = torch.cat([matrix, padding], dim=1)
    return matrix


def align_features(matrix, feature_columns, global_feature_columns):
    """Align feature matrices to global feature columns."""
    aligned_matrix = torch.zeros(matrix.size(0), len(global_feature_columns)) - 1  
    for idx, col in enumerate(feature_columns):
        global_idx = global_feature_columns.index(col)
        aligned_matrix[:, global_idx] = matrix[:, idx]
    return aligned_matrix


def word_embeddings(array):
    """Generate word embeddings using BioBERT (optional function)."""
    # This requires transformers library and is computationally expensive
    # Can be used for text-based features if needed
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    
    array = [text if text is not None else "" for text in array.to_pylist()]
    batch_size = 32
    embeddings_list = []
    
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    
    for i in tqdm(range(0, len(array), batch_size), desc="Processing batches"):
        batch_texts = array[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            output = model(**encoded_input)
        
        embeddings = output.last_hidden_state.mean(dim=1)
        embeddings_list.append(embeddings)

    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    return embeddings_tensor


def one_hot_encode(node_type, num_node_types):
    """Create one-hot encoding for node types."""
    feature_vector = torch.zeros(num_node_types)
    feature_vector[node_type] = 1
    return feature_vector.unsqueeze(0)


def randomize_tensor(tensor):
    """Randomize tensor columns for shuffling edges."""
    tensor = tensor[:, torch.randperm(tensor.size(1))]
    return tensor


# Evaluation and Statistical Functions
def calculate_bootstrap_ci(y_true, y_pred_proba, y_pred_binary, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for classification metrics."""
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Storage for bootstrap results
    bootstrap_metrics = {
        'sensitivity': [], 'specificity': [], 'precision': [], 
        'f1': [], 'auc': [], 'apr': []
    }
    
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    
    for i in range(n_bootstrap):
        # Bootstrap sample indices
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Bootstrap samples
        boot_y_true = y_true[boot_indices]
        boot_y_pred_proba = y_pred_proba[boot_indices] 
        boot_y_pred_binary = y_pred_binary[boot_indices]
        
        # Calculate metrics for this bootstrap sample
        # Confusion matrix elements
        tp = np.sum(boot_y_pred_binary * boot_y_true)
        fp = np.sum(boot_y_pred_binary * (1 - boot_y_true))
        fn = np.sum((1 - boot_y_pred_binary) * boot_y_true)
        tn = np.sum((1 - boot_y_pred_binary) * (1 - boot_y_true))
        
        # Classification metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Store results
        bootstrap_metrics['sensitivity'].append(sensitivity)
        bootstrap_metrics['specificity'].append(specificity) 
        bootstrap_metrics['precision'].append(precision)
        bootstrap_metrics['f1'].append(f1)
        
        # AUC and APR (only if we have both classes)
        if len(np.unique(boot_y_true)) == 2:
            try:
                auc_score = roc_auc_score(boot_y_true, boot_y_pred_proba)
                apr_score = average_precision_score(boot_y_true, boot_y_pred_proba)
                bootstrap_metrics['auc'].append(auc_score)
                bootstrap_metrics['apr'].append(apr_score)
            except:
                # Skip this iteration if AUC/APR calculation fails
                bootstrap_metrics['auc'].append(np.nan)
                bootstrap_metrics['apr'].append(np.nan)
        else:
            bootstrap_metrics['auc'].append(np.nan)
            bootstrap_metrics['apr'].append(np.nan)
    
    # Calculate confidence intervals
    ci_results = {}
    for metric, values in bootstrap_metrics.items():
        # Remove NaN values
        clean_values = [v for v in values if not np.isnan(v)]
        
        if len(clean_values) > 0:
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            ci_lower = np.percentile(clean_values, lower_percentile)
            ci_upper = np.percentile(clean_values, upper_percentile)
            mean_val = np.mean(clean_values)
            
            ci_results[metric] = {
                'mean': mean_val,
                'ci_lower': ci_lower, 
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower
            }
        else:
            ci_results[metric] = {
                'mean': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan, 
                'ci_width': np.nan
            }
    
    return ci_results


def calculate_metrics(y_true, y_prob, y_pred):
    """Calculate comprehensive evaluation metrics."""
    from sklearn.metrics import confusion_matrix
    
    # Basic metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Advanced metrics
    auc_score = roc_auc_score(y_true, y_prob)
    apr_score = average_precision_score(y_true, y_prob)
    
    # Additional metrics
    ppv = precision  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Calculate bootstrap confidence intervals
    ci_results = calculate_bootstrap_ci(y_true, y_prob, y_pred, n_bootstrap=1000)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc_score,
        'apr': apr_score,
        'ppv': ppv,
        'npv': npv,
        'confusion_matrix': {
            'TP': int(tp), 'FP': int(fp), 
            'TN': int(tn), 'FN': int(fn)
        },
        'ci_results': ci_results
    }


# Graph Analysis Functions
def standard_graph_analysis(graph):
    """Perform standard graph analysis and statistics."""
    from torch_geometric.utils import to_networkx
    import networkx as nx
    
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


# Negative Sampling Functions  
def as_negative_sampling(filtered_molecule_table, associations_table, score_column, 
                        drug_key_mapping, disease_key_mapping, return_list=False, return_set=False):
    """Association score-based negative sampling."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import polars as pl
    import logging
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


def generate_edge_list(source_list, target_list, source_mapping, target_mapping):
    """Generate edge list from parallel source and target lists."""
    edges = []
    for i in range(len(source_list)):
        source_id = source_list[i]
        target_id = target_list[i]
        if source_id in source_mapping and target_id in target_mapping:
            edges.append((source_mapping[source_id], target_mapping[target_id]))
    return edges


# Utility Functions for Explainer Integration
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


def custom_edges(disease_similarity_network, trial_edges, molecule_similarity_network,
                filtered_disease_table, filtered_molecule_table,
                disease_key_mapping, drug_key_mapping):
    """Generate custom edges for enhanced graph connectivity."""
    
    custom_edges = []

    # Disease similarity network edges
    if disease_similarity_network:
        disease_descendants_table = filtered_disease_table.select(['id', 'descendants']).flatten()
        disease_children_table = filtered_disease_table.select(['id', 'children']).flatten()
        disease_ancestors_table = filtered_disease_table.select(['id', 'ancestors']).flatten()

        custom_edges.extend(extract_edges(disease_descendants_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(disease_children_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(disease_ancestors_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))

    # Trial edges
    if trial_edges:
        molecule_trial_table = filtered_molecule_table.select(['id', 'linkedDiseases']).flatten()
        custom_edges.extend(extract_edges(molecule_trial_table, drug_key_mapping, disease_key_mapping, return_edge_list=True))

    # Molecule similarity network edges
    if molecule_similarity_network:
        molecule_parents_table = filtered_molecule_table.select(['id', 'parentId']).flatten()
        molecule_children_table = filtered_molecule_table.select(['id', 'childChemblIds']).flatten()

        custom_edges.extend(extract_edges(molecule_parents_table, drug_key_mapping, drug_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(molecule_children_table, drug_key_mapping, drug_key_mapping, return_edge_list=True))

    custom_edge_tensor = torch.tensor(custom_edges, dtype=torch.long).t().contiguous()
    return custom_edge_tensor

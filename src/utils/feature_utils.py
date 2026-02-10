"""
Feature engineering utility functions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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


def normalise(array, pad_length):
    """Normalise arrays with padding."""
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


def one_hot_encode_categorical(values_list, unique_values=None):
    """
    Create one-hot encoding for categorical values.
    
    Args:
        values_list: List of categorical values
        unique_values: Optional list of unique values for consistent encoding
    
    Returns:
        Tensor of one-hot encoded features
    """
    if unique_values is None:
        unique_values = sorted(list(set(values_list)))
    
    value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
    
    encoded = []
    for val in values_list:
        one_hot = torch.zeros(len(unique_values))
        if val in value_to_idx:
            one_hot[value_to_idx[val]] = 1.0
        encoded.append(one_hot)
    
    return torch.stack(encoded)


def extract_biotype_features(gene_table, gene_list, gene_key_mapping, version=21.06):
    """
    Extract bioType features from gene table.
    
    Args:
        gene_table: PyArrow table with gene data
        gene_list: List of gene IDs
        gene_key_mapping: Mapping from gene ID to index
        version: OpenTargets version number
    
    Returns:
        Tensor of one-hot encoded bioType features aligned to gene_list order
    """
    import pyarrow as pa
    
    # Get bioType column (name differs by version)
    if version in [21.04, 21.06]:
        biotype_col_name = 'bioType'
    else:
        biotype_col_name = 'biotype'
    
    # Extract gene IDs and bioTypes
    gene_df = gene_table.select(['id', biotype_col_name]).to_pandas()
    
    # Create mapping from gene ID to bioType
    gene_to_biotype = dict(zip(gene_df['id'], gene_df[biotype_col_name]))
    
    # Get unique bioTypes for consistent encoding
    unique_biotypes = sorted(list(set(gene_to_biotype.values())))
    biotype_to_idx = {bt: idx for idx, bt in enumerate(unique_biotypes)}
    
    # Create feature matrix aligned to gene_list order
    biotype_features = []
    for gene_id in gene_list:
        biotype = gene_to_biotype.get(gene_id, 'unknown')
        one_hot = torch.zeros(len(unique_biotypes))
        if biotype in biotype_to_idx:
            one_hot[biotype_to_idx[biotype]] = 1.0
        biotype_features.append(one_hot)
    
    return torch.stack(biotype_features)

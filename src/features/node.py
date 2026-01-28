"""
Node feature builder strategy.
"""

import torch
import pyarrow as pa
from src.utils.edge_utils import get_indices_from_keys
from src.utils.feature_utils import boolean_encode, normalize, extract_biotype_features

class NodeFeatureBuilder:
    """Builder for node features across different entity types."""
    
    def __init__(self, config, processor):
        """
        Initialize feature builder.
        
        Args:
            config: Configuration object
            processor: DataProcessor instance for loading external data
        """
        self.config = config
        self.processor = processor
        
    def create_features(self, mappings, molecule_df):
        """
        Create aggregated feature matrix for all nodes.
        
        Args:
            mappings: Dictionary containing node mappings
            molecule_df: Pandas DataFrame containing molecule data
            
        Returns:
            torch.Tensor: Concatenated feature matrix for all nodes
        """
        print("Creating node features...")
        
        # Get node indices
        drug_indices = torch.tensor(get_indices_from_keys(
            mappings['approved_drugs_list'], mappings['drug_key_mapping']
        ), dtype=torch.long)
        
        drug_type_indices = torch.tensor(get_indices_from_keys(
            mappings['drug_type_list'], mappings['drug_type_key_mapping']
        ), dtype=torch.long)
        
        gene_indices = torch.tensor(get_indices_from_keys(
            mappings['gene_list'], mappings['gene_key_mapping']
        ), dtype=torch.long)
        
        reactome_indices = torch.tensor(get_indices_from_keys(
            mappings['reactome_list'], mappings['reactome_key_mapping']
        ), dtype=torch.long)
        
        disease_indices = torch.tensor(get_indices_from_keys(
            mappings['disease_list'], mappings['disease_key_mapping']
        ), dtype=torch.long)
        
        therapeutic_area_indices = torch.tensor(get_indices_from_keys(
            mappings['therapeutic_area_list'], mappings['therapeutic_area_key_mapping']
        ), dtype=torch.long)
        
        # Create drug features from molecule table
        molecule_table = pa.Table.from_pandas(molecule_df)
        
        blackBoxWarning = molecule_table.column('blackBoxWarning').combine_chunks()
        blackBoxWarning_vector = boolean_encode(blackBoxWarning, drug_indices)
        
        yearOfFirstApproval = molecule_table.column('yearOfFirstApproval').combine_chunks()
        yearOfFirstApproval_vector = normalize(yearOfFirstApproval, drug_indices)
        
        # Create one-hot encodings for node types
        node_type_vectors = {
            'drug': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'drug_type': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            'gene': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'reactome': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            'disease': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            'therapeutic_area': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        }
        
        # Load gene data to extract bioType features
        print("Extracting gene bioType features...")
        gene_table = self.processor.load_gene_data(
            self.config.paths['targets'], 
            self.config.training_version
        )
        
        # Extract bioType features for genes
        gene_biotype_features = extract_biotype_features(
            gene_table, 
            mappings['gene_list'], 
            mappings['gene_key_mapping'],
            self.config.training_version
        )
        
        print(f"Gene bioType features shape: {gene_biotype_features.shape}")
        
        # Create feature matrices for each node type
        # Drug features: node_type (6) + blackBoxWarning (1) + yearOfFirstApproval (1) = 8 features
        drug_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['drug']], dtype=torch.float32).repeat(len(drug_indices), 1),
            blackBoxWarning_vector,
            yearOfFirstApproval_vector
        ], dim=1)
        
        # Drug type features: node_type (6) + padding (2) = 8 features
        drug_type_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['drug_type']], dtype=torch.float32).repeat(len(drug_type_indices), 1),
            torch.zeros(len(drug_type_indices), 2)  # Zero padding instead of -1
        ], dim=1)
        
        # Gene features: node_type (6) + bioType features (variable) = 6 + biotype_dim
        num_biotype_features = gene_biotype_features.shape[1]
        gene_node_type = torch.tensor([node_type_vectors['gene']], dtype=torch.float32).repeat(len(gene_indices), 1)
        gene_feature_matrix = torch.cat([gene_node_type, gene_biotype_features], dim=1)
        
        # Determine final target dimension
        drug_feature_dim = drug_feature_matrix.shape[1]  # Should be 8
        gene_feature_dim = gene_feature_matrix.shape[1]  # 6 + num_biotypes
        target_dim = max(drug_feature_dim, gene_feature_dim)
        
        # Pad all feature matrices to target dimension
        if drug_feature_matrix.shape[1] < target_dim:
            padding_size = target_dim - drug_feature_matrix.shape[1]
            drug_feature_matrix = torch.cat([
                drug_feature_matrix,
                torch.zeros(len(drug_indices), padding_size)
            ], dim=1)
        
        if gene_feature_matrix.shape[1] < target_dim:
            padding_size = target_dim - gene_feature_matrix.shape[1]
            gene_feature_matrix = torch.cat([
                gene_feature_matrix,
                torch.zeros(len(gene_indices), padding_size)
            ], dim=1)
        
        # Recreate drug_type features with correct target dimension
        drug_type_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['drug_type']], dtype=torch.float32).repeat(len(drug_type_indices), 1),
            torch.zeros(len(drug_type_indices), target_dim - 6)
        ], dim=1)
        
        # Reactome, disease, and therapeutic area features: node_type (6) + padding
        reactome_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['reactome']], dtype=torch.float32).repeat(len(reactome_indices), 1),
            torch.zeros(len(reactome_indices), target_dim - 6)
        ], dim=1)
        
        disease_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['disease']], dtype=torch.float32).repeat(len(disease_indices), 1),
            torch.zeros(len(disease_indices), target_dim - 6)
        ], dim=1)
        
        therapeutic_area_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['therapeutic_area']], dtype=torch.float32).repeat(len(therapeutic_area_indices), 1),
            torch.zeros(len(therapeutic_area_indices), target_dim - 6)
        ], dim=1)
        
        # Combine all features
        all_features = torch.cat([
            drug_feature_matrix,
            drug_type_feature_matrix,
            gene_feature_matrix,
            reactome_feature_matrix,
            disease_feature_matrix,
            therapeutic_area_feature_matrix
        ], dim=0)
        
        print(f"Created feature matrix: {all_features.shape}")
        
        return all_features

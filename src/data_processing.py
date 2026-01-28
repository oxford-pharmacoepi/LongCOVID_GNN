"""
Data processing module - Compatibility wrapper.
Delegates to new modular classes in src/data/ while maintaining backward compatibility.
"""

import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path

# Import new modular classes
from .data import (
    OpenTargetsLoader,
    IdMapper,
    NodeIndexMapper,
    MoleculeFilter,
    AssociationFilter,
    DataStorage
)
from .utils import extract_edges


class DataProcessor:
    """Compatibility wrapper that delegates to new modular classes.
    
    This maintains backward compatibility while using the refactored SOLID architecture.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize new modular components
        self.loader = OpenTargetsLoader(config)
        self.id_mapper = IdMapper()
        self.node_mapper = NodeIndexMapper()
        self.molecule_filter = MoleculeFilter()
        self.association_filter = AssociationFilter()
        self.storage = DataStorage()
        
        # Maintain backward compatibility
        self.redundant_id_mapping = self.id_mapper.redundant_id_mapping
    
    # Delegate ID mapping methods to IdMapper
    def resolve_mapping(self, entity_id, mapping_dict):
        return self.id_mapper.resolve_mapping(entity_id, mapping_dict)
    
    def safe_list_conversion(self, value):
        return self.id_mapper.safe_list_conversion(value)
    
    def update_approved_indications(self, disease_list, mapping_dict):
        return self.id_mapper.update_approved_indications(disease_list, mapping_dict)
    
    # Delegate loading methods to OpenTargetsLoader
    def load_indication_data(self, path):
        return self.loader.load_indication_data(path)
    
    def load_molecule_data(self, path):
        return self.loader.load_molecule_data(path)
    
    def load_disease_data(self, path):
        return self.loader.load_disease_data(path)
    
    def load_gene_data(self, path, version):
        return self.loader.load_gene_data(path, version)
    
    def load_associations_data(self, path, version):
        return self.loader.load_associations_data(path, version)
    
    def load_mechanism_of_action(self, path):
        return self.loader.load_mechanism_of_action(path)
    
    def load_drug_warnings(self, path):
        return self.loader.load_drug_warnings(path)
    
    def load_interaction(self, path):
        return self.loader.load_interaction(path)
    
    def load_known_drugs_aggregated(self, path):
        return self.loader.load_known_drugs_aggregated(path)
    
    # Delegate filtering methods
    def filter_linked_molecules(self, molecule_df, indication_df, known_drugs_df=None):
        return self.molecule_filter.filter_linked_molecules(molecule_df, indication_df, known_drugs_df)
    
    def filter_associations_by_genes_and_diseases(self, associations_table, gene_ids, disease_ids, score_column, threshold=0.1):
        return self.association_filter.filter_by_genes_and_diseases(
            associations_table, gene_ids, disease_ids, score_column, threshold
        )
    
    # Delegate storage methods
    def save_processed_data(self, data_dict, output_dir):
        return self.storage.save_processed_data(data_dict, output_dir)
    
    def load_processed_data(self, data_dir):
        return self.storage.load_processed_data(data_dir)
    
    def save_mappings(self, mappings, output_dir):
        return self.storage.save_mappings(mappings, output_dir)
    
    def load_mappings(self, mappings_path):
        return self.storage.load_mappings(mappings_path)
    
    # Delegate node mapping creation
    def create_node_mappings(self, molecule_df, disease_df, gene_table, version):
        return self.node_mapper.create_node_mappings(molecule_df, disease_df, gene_table, version)
    
    # Methods that combine multiple components
    def apply_id_mappings(self, molecule_df, indication_df):
        """Apply redundant ID mappings to clean the data."""
        print("Applying ID mappings for data consistency...")
        
        # EXACT COPY from original: only apply drug mappings to indication_df, NOT molecule_df
        # Apply drug ID mappings to indication_df only
        drug_mappings = self.id_mapper.redundant_id_mapping['drug_mappings']
        id_to_parentid_mapping = {
            k: self.id_mapper.resolve_mapping(v, drug_mappings) 
            for k, v in drug_mappings.items()
        }
        
        # Update drug IDs in indication_df only
        indication_df['id'] = indication_df['id'].apply(
            lambda x: self.id_mapper.resolve_mapping(x, id_to_parentid_mapping) 
            if x in id_to_parentid_mapping else x
        )
        
        # Apply disease mappings to indications
        indication_df = self.id_mapper.apply_disease_mappings(indication_df)
        
        return molecule_df, indication_df
    
    def create_gene_reactome_mapping(self, gene_table, version):
        """Create gene-reactome pathway mappings based on version."""
        print("Creating gene-reactome mappings...")
        
        if version >= 22.04:
            pathway_column = 'pathways'
        else:
            pathway_column = 'pathways'
        
        if pathway_column not in gene_table.column_names:
            print(f"Warning: {pathway_column} column not found in gene table")
            return pa.table({'geneId': [], 'reactomeId': []})
        
        # Extract gene-pathway pairs
        gene_ids = []
        pathway_ids = []
        
        for i in range(len(gene_table)):
            gene_id = gene_table['id'][i].as_py()
            pathways = gene_table[pathway_column][i].as_py()
            
            if pathways:
                if isinstance(pathways, str):
                    import ast
                    try:
                        pathways = ast.literal_eval(pathways)
                    except:
                        continue
                
                if isinstance(pathways, list):
                    for pathway in pathways:
                        gene_ids.append(gene_id)
                        pathway_ids.append(pathway)
        
        return pa.table({'geneId': gene_ids, 'reactomeId': pathway_ids})
    
    def generate_validation_test_splits(self, config, mappings, train_edges_set):
        """Generate validation and test edge splits using different OpenTargets versions."""
        print("Generating validation and test edge splits...")
        
        val_edges = []
        test_edges = []
        
        # Load validation version data (23.06)
        if config.validation_version:
            val_path = config.paths.get('validation_indications', 
                                       f"raw_data/{config.validation_version}/indication")
            if Path(val_path).exists():
                val_indication_table = self.load_indication_data(val_path)
                val_edges_set = extract_edges(
                    val_indication_table.select(['id', 'approvedIndications']).flatten(),
                    mappings['drug_key_mapping'],
                    mappings['disease_key_mapping'],
                    return_edge_set=True
                )
                # Only keep new edges not in training
                val_edges = list(val_edges_set - train_edges_set)
        
        # Load test version data (24.06)
        if config.test_version:
            test_path = config.paths.get('test_indications',
                                        f"raw_data/{config.test_version}/indication")
            if Path(test_path).exists():
                test_indication_table = self.load_indication_data(test_path)
                test_edges_set = extract_edges(
                    test_indication_table.select(['id', 'approvedIndications']).flatten(),
                    mappings['drug_key_mapping'],
                    mappings['disease_key_mapping'],
                    return_edge_set=True
                )
                # Only keep new edges not in training or validation
                all_seen = train_edges_set | set(val_edges)
                test_edges = list(test_edges_set - all_seen)
        
        print(f"Validation set: {len(val_edges)} new edges")
        print(f"Test set: {len(test_edges)} new edges")
        print("Generated temporal splits from OpenTargets data")
        
        return val_edges, test_edges


# Helper functions for backward compatibility
def detect_data_mode(config, force_mode=None):
    """Detect whether to use raw or processed data."""
    if force_mode:
        return force_mode
    
    processed_path = config.paths.get('processed_path', 'processed_data/')
    tables_path = Path(processed_path) / 'tables'
    mappings_path = Path(processed_path) / 'mappings'
    
    if tables_path.exists() and mappings_path.exists():
        return 'processed'
    else:
        return 'raw'


def create_full_dataset(config):
    """Create full dataset from raw data - compatibility function."""
    processor = DataProcessor(config)
    
    # Load raw data
    indication_table = processor.load_indication_data(config.paths['indications'])
    molecule_df = processor.load_molecule_data(config.paths['molecules'])
    disease_df = processor.load_disease_data(config.paths['diseases'])
    gene_table = processor.load_gene_data(config.paths['genes'], config.training_version)
    
    # Apply ID mappings
    molecule_df, indication_table = processor.apply_id_mappings(molecule_df, indication_table)
    
    # Filter molecules
    known_drugs_df = processor.load_known_drugs_aggregated(config.paths.get('known_drugs'))
    molecule_df = processor.filter_linked_molecules(molecule_df, indication_table, known_drugs_df)
    
    # Create node mappings
    mappings = processor.create_node_mappings(molecule_df, disease_df, gene_table, config.training_version)
    
    return {
        'molecule_df': molecule_df,
        'indication_table': indication_table,
        'disease_df': disease_df,
        'gene_table': gene_table,
        'known_drugs_df': known_drugs_df,
        'mappings': mappings
    }

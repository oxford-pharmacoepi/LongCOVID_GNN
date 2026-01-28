#!/usr/bin/env python3
"""
Graph Creation Script for Drug-Disease Prediction
Creates knowledge graph from OpenTargets data with validation/test splits.
"""

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
import datetime as dt
import argparse
import json
import os
from pathlib import Path
import random
from typing import Optional, Set, Tuple
from src.negative_sampling import get_sampler
import pandas as pd
import numpy as np

# Import from shared modules
from src.utils import (
    set_seed, enable_full_reproducibility, get_indices_from_keys,
    generate_pairs, extract_edges, boolean_encode, normalize, 
    cat_encode, pad_feature_matrix, align_features, standard_graph_analysis,
    custom_edges
)
from src.data_processing import DataProcessor, detect_data_mode, create_full_dataset
from src.config import get_config, create_custom_config
from src.mlflow_tracker import ExperimentTracker
from src.edge_features import extract_moa_features 


class GraphBuilder:
    """Main graph builder using shared modules."""
    
    def __init__(self, config, force_mode=None, tracker=None):
        self.config = config
        self.processor = DataProcessor(config)
        self.mappings = None
        self.data_mode = detect_data_mode(config, force_mode)
        self.tracker = tracker
        
    def load_or_create_data(self):
        """Load existing processed data or create from raw data."""
        print(f"Data mode: {self.data_mode}")
        
        if self.tracker:
            self.tracker.log_dict_as_json({'data_mode': self.data_mode}, 'data_mode.json')
        
        if self.data_mode == "processed":
            self.load_preprocessed_data()
        else:
            self.create_from_raw_data()
    
    def load_preprocessed_data(self):
        """Load pre-processed data."""
        print("Loading pre-processed data...")
        
        # Load mappings
        mappings_path = f"{self.config.paths['processed']}mappings/"
        self.mappings = self.processor.load_mappings(mappings_path)
        
        # Load processed tables
        processed_dir = f"{self.config.paths['processed']}tables/"
        self.molecule_df = pd.read_csv(f"{processed_dir}processed_molecules.csv")
        self.indication_df = pd.read_csv(f"{processed_dir}processed_indications.csv")
        self.disease_df = pd.read_csv(f"{processed_dir}processed_diseases.csv")
        self.known_drugs_df = pd.read_csv(f"{processed_dir}processed_known_drugs.csv") if os.path.exists(f"{processed_dir}processed_known_drugs.csv") else pd.DataFrame()
        
        # Handle list columns in indication_df
        if 'approvedIndications' in self.indication_df.columns:
            self.indication_df['approvedIndications'] = self.indication_df['approvedIndications'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
        
        # Handle list columns in molecule_df
        if 'linkedDiseases.rows' in self.molecule_df.columns:
            self.molecule_df['linkedDiseases.rows'] = self.molecule_df['linkedDiseases.rows'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
        
        # Handle list columns in disease_df (for disease-disease edges)
        def parse_numpy_array_string(value):
            """Parse numpy array string format: "['item1' 'item2' 'item3']" """
            if not isinstance(value, str):
                return value if isinstance(value, list) else []
            
            # Remove brackets and quotes, then split by whitespace
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                # Remove outer brackets
                inner = value[1:-1].strip()
                if not inner:
                    return []
                
                # Split by whitespace and remove quotes
                items = []
                current_item = ""
                in_quotes = False
                
                for char in inner:
                    if char == "'" or char == '"':
                        in_quotes = not in_quotes
                    elif char in (' ', '\n', '\t') and not in_quotes:
                        if current_item:
                            items.append(current_item)
                            current_item = ""
                    else:
                        current_item += char
                
                # Add last item
                if current_item:
                    items.append(current_item)
                
                return items
            
            return []
        
        list_columns = ['ancestors', 'descendants', 'children', 'therapeuticAreas']
        for col in list_columns:
            if col in self.disease_df.columns:
                self.disease_df[col] = self.disease_df[col].apply(parse_numpy_array_string)
        
        print("Pre-processed data loaded successfully")
    
    def create_from_raw_data(self):
        """Create data from raw OpenTargets files."""
        print("Processing raw OpenTargets data...")
        
        # Load raw data
        indication_table = self.processor.load_indication_data(self.config.paths['indication'])
        molecule_table = self.processor.load_molecule_data(self.config.paths['molecule'])
        disease_table = self.processor.load_disease_data(self.config.paths['diseases'])
        gene_table = self.processor.load_gene_data(self.config.paths['targets'], self.config.training_version)
        associations_table, score_column = self.processor.load_associations_data(
            self.config.paths['associations'], self.config.training_version
        )
        # Load other drug-disease datasets
        self.known_drugs_df = self.processor.load_known_drugs_aggregated(self.config.paths['knownDrugsAggregated']) if 'knownDrugsAggregated' in self.config.paths else None

        # Convert to dataframes for processing
        indication_df = indication_table.to_pandas()
        molecule_df = molecule_table.to_pandas()
        disease_df = disease_table.to_pandas()
        
        # Apply ID mappings and filtering
        self.molecule_df, self.indication_df = self.processor.apply_id_mappings(molecule_df, indication_df)
        self.molecule_df = self.processor.filter_linked_molecules(self.molecule_df, self.indication_df, self.known_drugs_df)
        self.disease_df = disease_df
        
        # Create node mappings
        self.mappings = self.processor.create_node_mappings(
            self.molecule_df, self.disease_df, gene_table, self.config.training_version
        )
        
        # Save processed data for future use
        processed_data = {
            'processed_molecules': self.molecule_df,
            'processed_indications': self.indication_df,
            'processed_diseases': self.disease_df,
            'processed_genes': gene_table.to_pandas(),
            'processed_associations': associations_table.to_pandas(),
            'processed_known_drugs': self.known_drugs_df if isinstance(self.known_drugs_df, pd.DataFrame) else pd.DataFrame()
        }
        
        self.processor.save_processed_data(processed_data, f"{self.config.paths['processed']}tables/")
        self.processor.save_mappings(self.mappings, f"{self.config.paths['processed']}mappings/")
        
        print("Raw data processing completed")
    
    def create_node_features(self):
        """Create node feature matrices."""
        print("Creating node features...")
        
        # Get node indices
        drug_indices = torch.tensor(get_indices_from_keys(
            self.mappings['approved_drugs_list'], self.mappings['drug_key_mapping']
        ), dtype=torch.long)
        
        drug_type_indices = torch.tensor(get_indices_from_keys(
            self.mappings['drug_type_list'], self.mappings['drug_type_key_mapping']
        ), dtype=torch.long)
        
        gene_indices = torch.tensor(get_indices_from_keys(
            self.mappings['gene_list'], self.mappings['gene_key_mapping']
        ), dtype=torch.long)
        
        reactome_indices = torch.tensor(get_indices_from_keys(
            self.mappings['reactome_list'], self.mappings['reactome_key_mapping']
        ), dtype=torch.long)
        
        disease_indices = torch.tensor(get_indices_from_keys(
            self.mappings['disease_list'], self.mappings['disease_key_mapping']
        ), dtype=torch.long)
        
        therapeutic_area_indices = torch.tensor(get_indices_from_keys(
            self.mappings['therapeutic_area_list'], self.mappings['therapeutic_area_key_mapping']
        ), dtype=torch.long)
        
        # Create drug features from molecule table
        import pyarrow as pa
        from src.utils import extract_biotype_features
        
        molecule_table = pa.Table.from_pandas(self.molecule_df)
        
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
            self.mappings['gene_list'], 
            self.mappings['gene_key_mapping'],
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
        self.all_features = torch.cat([
            drug_feature_matrix,
            drug_type_feature_matrix,
            gene_feature_matrix,
            reactome_feature_matrix,
            disease_feature_matrix,
            therapeutic_area_feature_matrix
        ], dim=0)
        
        print(f"Created feature matrix: {self.all_features.shape}")
        print(f"  - Drug features: {drug_feature_matrix.shape}")
        print(f"  - Drug type features: {drug_type_feature_matrix.shape}")
        print(f"  - Gene features (with bioType): {gene_feature_matrix.shape}")
        print(f"  - Reactome features: {reactome_feature_matrix.shape}")
        print(f"  - Disease features: {disease_feature_matrix.shape}")
        print(f"  - Therapeutic area features: {therapeutic_area_feature_matrix.shape}")
    
    def create_edges(self):
        """Create edge indices and edge features."""
        print("Creating graph edges...")
        
        if self.data_mode == "processed":
            # Load all edge types from processed files
            edge_dir = f"{self.config.paths['processed']}edges/"
            
            print(f"Loading edges from {edge_dir}")
            
            # Load all 6 edge types from saved files
            self.molecule_drugType_edges = torch.load(f"{edge_dir}1_molecule_drugType_edges.pt")
            self.molecule_disease_edges = torch.load(f"{edge_dir}2_molecule_disease_edges.pt")
            self.molecule_gene_edges = torch.load(f"{edge_dir}3_molecule_gene_edges.pt")
            self.gene_reactome_edges = torch.load(f"{edge_dir}4_gene_reactome_edges.pt")
            self.disease_therapeutic_edges = torch.load(f"{edge_dir}5_disease_therapeutic_edges.pt")
            self.disease_gene_edges = torch.load(f"{edge_dir}6_disease_gene_edges.pt")
            
            print(f"Loaded edge types:")
            print(f"  Drug-DrugType: {self.molecule_drugType_edges.shape}")
            print(f"  Drug-Disease: {self.molecule_disease_edges.shape}")
            print(f"  Drug-Gene: {self.molecule_gene_edges.shape}")
            print(f"  Gene-Reactome: {self.gene_reactome_edges.shape}")
            print(f"  Disease-Therapeutic: {self.disease_therapeutic_edges.shape}")
            print(f"  Disease-Gene: {self.disease_gene_edges.shape}")
            
            # Extract custom edges (disease-disease, trial edges, molecule-molecule) if enabled
            print("\nChecking for custom edge types...")
            disease_similarity_enabled = self.config.network_config.get('disease_similarity_network', False)
            trial_edges_enabled = self.config.network_config.get('trial_edges', False)
            molecule_similarity_enabled = self.config.network_config.get('molecule_similarity_network', False)
            
            if disease_similarity_enabled or trial_edges_enabled or molecule_similarity_enabled:
                print(f"  Disease similarity network: {disease_similarity_enabled}")
                print(f"  Trial edges: {trial_edges_enabled}")
                print(f"  Molecule similarity network: {molecule_similarity_enabled}")
                
                # Convert DataFrames to PyArrow tables for custom edge extraction
                import pyarrow as pa
                
                # For disease table, explicitly set list column types
                disease_schema = None
                if disease_similarity_enabled:
                    # Build schema with explicit list types for disease columns
                    disease_schema_fields = []
                    for col in self.disease_df.columns:
                        if col in ['ancestors', 'descendants', 'children', 'therapeuticAreas']:
                            # Explicitly mark as list of strings
                            disease_schema_fields.append(pa.field(col, pa.list_(pa.string())))
                        else:
                            # Infer type from pandas dtype
                            dtype = self.disease_df[col].dtype
                            if dtype == 'object':
                                pa_type = pa.string()
                            elif dtype == 'int64':
                                pa_type = pa.int64()
                            elif dtype == 'float64':
                                pa_type = pa.float64()
                            elif dtype == 'bool':
                                pa_type = pa.bool_()
                            else:
                                pa_type = pa.string()  # fallback
                            disease_schema_fields.append(pa.field(col, pa_type))
                    
                    disease_schema = pa.schema(disease_schema_fields)
                    print(f"\n  Building disease table with explicit schema:")
                    print(f"    Total columns: {len(disease_schema_fields)}")
                    print(f"    List columns: {[f.name for f in disease_schema_fields if pa.types.is_list(f.type)]}")
                
                disease_table = pa.Table.from_pandas(self.disease_df, schema=disease_schema)
                molecule_table = pa.Table.from_pandas(self.molecule_df)
                
                if disease_similarity_enabled:
                    print(f"\n  Disease table schema verification:")
                    for col in ['id', 'ancestors', 'descendants', 'children']:
                        if col in disease_table.column_names:
                            col_type = disease_table.schema.field(col).type
                            print(f"    {col}: {col_type}")
                            # Quick sanity check - sample first few values
                            col_data = disease_table.column(col).combine_chunks()
                            if col in ['ancestors', 'descendants', 'children']:
                                non_null = sum(1 for i in range(len(col_data)) if col_data[i].as_py() and len(col_data[i].as_py()) > 0)
                                print(f"      Non-empty lists: {non_null}/{len(col_data)}")
                                # Show first non-empty example
                                for i in range(min(5, len(col_data))):
                                    val = col_data[i].as_py()
                                    if val and len(val) > 0:
                                        print(f"      Example: {val[:3]}...")
                                        break
                            else:
                                # Show first few IDs
                                print(f"      Sample values: {[col_data[i].as_py() for i in range(min(3, len(col_data)))]}")
                
                # Call custom_edges() function
                self.custom_edge_tensor = custom_edges(
                    disease_similarity_network=disease_similarity_enabled,
                    trial_edges=trial_edges_enabled,
                    molecule_similarity_network=molecule_similarity_enabled,
                    filtered_disease_table=disease_table,
                    filtered_molecule_table=molecule_table,
                    disease_key_mapping=self.mappings['disease_key_mapping'],
                    drug_key_mapping=self.mappings['drug_key_mapping']
                )
                
                print(f"  Custom edges created: {self.custom_edge_tensor.shape}")
            else:
                print("  No custom edges enabled - skipping")
                self.custom_edge_tensor = torch.empty((2, 0), dtype=torch.long)
            
        else:
            # Extract edges from raw data
            import pyarrow as pa
            
            # Convert disease table to PyArrow with explicit schema for list columns
            disease_similarity_enabled = self.config.network_config.get('disease_similarity_network', False)
            
            disease_schema = None
            if disease_similarity_enabled:
                # Build schema with explicit list types for disease columns
                disease_schema_fields = []
                for col in self.disease_df.columns:
                    if col in ['ancestors', 'descendants', 'children', 'therapeuticAreas']:
                        # Explicitly mark as list of strings
                        disease_schema_fields.append(pa.field(col, pa.list_(pa.string())))
                    else:
                        # Infer type from pandas dtype
                        dtype = self.disease_df[col].dtype
                        if dtype == 'object':
                            pa_type = pa.string()
                        elif dtype == 'int64':
                            pa_type = pa.int64()
                        elif dtype == 'float64':
                            pa_type = pa.float64()
                        elif dtype == 'bool':
                            pa_type = pa.bool_()
                        else:
                            pa_type = pa.string()  # fallback
                        disease_schema_fields.append(pa.field(col, pa_type))
                
                disease_schema = pa.schema(disease_schema_fields)
            
            disease_table = pa.Table.from_pandas(self.disease_df, schema=disease_schema)
            molecule_table = pa.Table.from_pandas(self.molecule_df)
            indication_table = pa.Table.from_pandas(self.indication_df)
            
            # Extract different edge types
            molecule_drugType_table = molecule_table.select(['id', 'drugType']).drop_null().flatten()
            self.molecule_drugType_edges = extract_edges(
                molecule_drugType_table, 
                self.mappings['drug_key_mapping'], 
                self.mappings['drug_type_key_mapping']
            )
            
            # Extract molecule-disease edges from multiple sources for completeness
            print("Extracting drug-disease edges from multiple sources...")
            
            # 1. Approved Indications
            ind_table = indication_table.select(['id', 'approvedIndications']).flatten()
            print(f"DEBUG: indication_table rows: {len(ind_table)}")
            if len(ind_table) > 0:
                print(f"DEBUG: Sample indication IDs: {ind_table['id'].slice(0,5).to_pylist()}")
                print(f"DEBUG: Sample approvedIndications: {ind_table['approvedIndications'].slice(0,5).to_pylist()}")
            
            ind_edges = extract_edges(ind_table, self.mappings['drug_key_mapping'], self.mappings['disease_key_mapping'], return_edge_set=True, debug=True)
            print(f"  - Edges from indications: {len(ind_edges)}")
            
            # 2. Known Drugs (Clinical Trials Phase 3 and 4)
            known_edges = set()
            if hasattr(self, 'known_drugs_df') and self.known_drugs_df is not None and not self.known_drugs_df.empty:
                valid_known = self.known_drugs_df[self.known_drugs_df['phase'] >= 3]
                # Convert to table for extract_edges
                valid_known_table = pa.Table.from_pandas(valid_known[['drugId', 'diseaseId']])
                known_edges = extract_edges(valid_known_table, self.mappings['drug_key_mapping'], self.mappings['disease_key_mapping'], return_edge_set=True)
                print(f"  - Edges from clinical trials (Ph 3/4): {len(known_edges)}")
            
            # 3. Pre-linked diseases from molecule metadata
            meta_edges = set()
            if 'linkedDiseases.rows' in self.molecule_df.columns:
                meta_table = pa.Table.from_pandas(self.molecule_df[['id', 'linkedDiseases.rows']])
                meta_edges = extract_edges(meta_table, self.mappings['drug_key_mapping'], self.mappings['disease_key_mapping'], return_edge_set=True)
                print(f"  - Edges from molecule metadata: {len(meta_edges)}")
            
            # Merge all edges
            all_md_edges = ind_edges | known_edges | meta_edges
            self.molecule_disease_edges = torch.tensor(list(all_md_edges), dtype=torch.long).t().contiguous()
            print(f"  Total unique drug-disease edges: {self.molecule_disease_edges.shape[1]}")
            
            molecule_gene_table = molecule_table.select(['id', 'linkedTargets.rows']).drop_null().flatten()
            self.molecule_gene_edges = extract_edges(
                molecule_gene_table,
                self.mappings['drug_key_mapping'],
                self.mappings['gene_key_mapping']
            )
            
            # Extract Gene-Reactome edges from gene data
            print("Extracting Gene-Reactome edges...")
            gene_reactome_table = self.processor.create_gene_reactome_mapping(
                self.processor.load_gene_data(self.config.paths['targets'], self.config.training_version),
                self.config.training_version
            )
            self.gene_reactome_edges = extract_edges(
                gene_reactome_table,
                self.mappings['gene_key_mapping'],
                self.mappings['reactome_key_mapping']
            )
            
            # Extract Disease-Therapeutic edges from disease data
            print("Extracting Disease-Therapeutic edges...")
            disease_therapeutic_table = disease_table.select(['id', 'therapeuticAreas']).flatten()
            self.disease_therapeutic_edges = extract_edges(
                disease_therapeutic_table,
                self.mappings['disease_key_mapping'],
                self.mappings['therapeutic_area_key_mapping']
            )
            
            # Extract Disease-Gene edges from associations data
            print("Extracting Disease-Gene edges...")
            associations_table, score_column = self.processor.load_associations_data(
                self.config.paths['associations'], self.config.training_version
            )
            # Filter associations by genes and diseases that are in our mappings
            filtered_associations = self.processor.filter_associations_by_genes_and_diseases(
                associations_table,
                list(self.mappings['gene_key_mapping'].keys()),
                list(self.mappings['disease_key_mapping'].keys()),
                score_column
            )
            # Create edges from filtered associations
            self.disease_gene_edges = extract_edges(
                filtered_associations.select(['diseaseId', 'targetId']),
                self.mappings['disease_key_mapping'],
                self.mappings['gene_key_mapping']
            )
            
            print(f"Extracted edge types:")
            print(f"  Drug-DrugType: {self.molecule_drugType_edges.shape}")
            print(f"  Drug-Disease: {self.molecule_disease_edges.shape}")
            print(f"  Drug-Gene: {self.molecule_gene_edges.shape}")
            print(f"  Gene-Reactome: {self.gene_reactome_edges.shape}")
            print(f"  Disease-Therapeutic: {self.disease_therapeutic_edges.shape}")
            print(f"  Disease-Gene: {self.disease_gene_edges.shape}")
            
            # Extract custom edges (disease-disease, trial edges, molecule-molecule) if enabled
            print("\nChecking for custom edge types...")
            disease_similarity_enabled = self.config.network_config.get('disease_similarity_network', False)
            trial_edges_enabled = self.config.network_config.get('trial_edges', False)
            molecule_similarity_enabled = self.config.network_config.get('molecule_similarity_network', False)
            
            if disease_similarity_enabled or trial_edges_enabled or molecule_similarity_enabled:
                print(f"  Disease similarity network: {disease_similarity_enabled}")
                print(f"  Trial edges: {trial_edges_enabled}")
                print(f"  Molecule similarity network: {molecule_similarity_enabled}")
                
                # Call custom_edges() function
                self.custom_edge_tensor = custom_edges(
                    disease_similarity_network=disease_similarity_enabled,
                    trial_edges=trial_edges_enabled,
                    molecule_similarity_network=molecule_similarity_enabled,
                    filtered_disease_table=disease_table,
                    filtered_molecule_table=molecule_table,
                    disease_key_mapping=self.mappings['disease_key_mapping'],
                    drug_key_mapping=self.mappings['drug_key_mapping']
                )
                
                print(f"  Custom edges created: {self.custom_edge_tensor.shape}")
            else:
                print("  No custom edges enabled - skipping")
                self.custom_edge_tensor = torch.empty((2, 0), dtype=torch.long)
        
        # Combine all edges
        all_edges = [
            self.molecule_drugType_edges,
            self.molecule_disease_edges,
            self.molecule_gene_edges,
            self.gene_reactome_edges,
            self.disease_therapeutic_edges,
            self.disease_gene_edges
        ]
        
        # Add custom edges if they exist
        if hasattr(self, 'custom_edge_tensor') and self.custom_edge_tensor.size(1) > 0:
            all_edges.append(self.custom_edge_tensor)
        
        # Filter out empty tensors
        non_empty_edges = [e for e in all_edges if e.size(1) > 0]
        
        if non_empty_edges:
            self.all_edge_index = torch.cat(non_empty_edges, dim=1)
        else:
            self.all_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"Created edges: {self.all_edge_index.size(1)} total")
        
        # Extract mechanismOfAction edge features for drug-gene edges
        print("\n" + "="*80)
        print("EXTRACTING EDGE FEATURES")
        print("="*80)
        self.create_edge_features()
        
        # Save edge tensors for future use (only when processing raw data)
        if self.data_mode == "raw":
            edge_dir = f"{self.config.paths['processed']}edges/"
            os.makedirs(edge_dir, exist_ok=True)
            
            torch.save(self.molecule_drugType_edges, f"{edge_dir}1_molecule_drugType_edges.pt")
            torch.save(self.molecule_disease_edges, f"{edge_dir}2_molecule_disease_edges.pt")
            torch.save(self.molecule_gene_edges, f"{edge_dir}3_molecule_gene_edges.pt")
            torch.save(self.gene_reactome_edges, f"{edge_dir}4_gene_reactome_edges.pt")
            torch.save(self.disease_therapeutic_edges, f"{edge_dir}5_disease_therapeutic_edges.pt")
            torch.save(self.disease_gene_edges, f"{edge_dir}6_disease_gene_edges.pt")
    
    def create_edge_features(self):
        """Extract edge features from mechanismOfAction data."""
        import pandas as pd
        import os
        
        # Try to load mechanismOfAction data
        try:
            moa_path = self.config.paths.get('mechanismOfAction', None)
            
            if moa_path and os.path.exists(moa_path):
                # Load mechanismOfAction data
                moa_df = self.processor.load_mechanism_of_action(moa_path)
                
                # Extract features for drug-gene edges
                self.drug_gene_edge_features = extract_moa_features(
                    moa_df,
                    self.mappings['drug_key_mapping'],
                    self.mappings['gene_key_mapping'],
                    self.molecule_gene_edges
                )
                
                # Create padded edge features for all edges
                # Since only drug-gene edges have features, we pad with zeros for other edge types
                num_drug_gene_edges = self.molecule_gene_edges.shape[1]
                total_edges = self.all_edge_index.shape[1]
                feature_dim = self.drug_gene_edge_features.shape[1]
                
                # Create edge feature tensor for all edges
                self.all_edge_features = torch.zeros((total_edges, feature_dim), dtype=torch.float32)
                
                # Find where drug-gene edges are in the concatenated edge tensor
                # Order: drugType, disease, gene, reactome, therapeutic, disease-gene
                edge_offset = (self.molecule_drugType_edges.shape[1] + 
                              self.molecule_disease_edges.shape[1])
                
                # Copy drug-gene edge features to the correct position
                self.all_edge_features[edge_offset:edge_offset + num_drug_gene_edges] = self.drug_gene_edge_features
                
                print(f"\n✓ Created edge feature matrix: {self.all_edge_features.shape}")
                print(f"  - Drug-gene edges with features: {num_drug_gene_edges}")
                print(f"  - Total edges: {total_edges}")
                print(f"  - Feature dimension: {feature_dim}")
                
            else:
                print(f"  Warning: mechanismOfAction data not found at {moa_path}")
                print(f"  Skipping edge features - graph will use structure only")
                self.all_edge_features = None
                
        except Exception as e:
            print(f"  Error loading edge features: {e}")
            print(f"  Skipping edge features - graph will use structure only")
            self.all_edge_features = None
    
    def create_train_val_test_splits(self):
        """Create training, validation, and test splits with negative sampling."""
        print("Creating train/validation/test splits...")
        print(f"Negative sampling strategy: {self.config.negative_sampling_strategy}")
        print(f"Pos:Neg ratio: 1:{self.config.train_neg_ratio} (training), 1:{self.config.pos_neg_ratio} (val/test)")

        # Extract training edges 
        train_edges_set = set(zip(
            self.molecule_disease_edges[0].tolist(),
            self.molecule_disease_edges[1].tolist()
        ))
        
        print(f"Training edges: {len(train_edges_set)}")
        
        # Generate all possible drug-disease pairs
        from src.utils import generate_pairs
        all_possible_pairs = generate_pairs(
            self.mappings['approved_drugs_list'],
            self.mappings['disease_list'],
            self.mappings['drug_key_mapping'],
            self.mappings['disease_key_mapping']
        )
        
        print(f"Total possible drug-disease pairs: {len(all_possible_pairs)}")
        
        # Generate validation and test positive splits using temporal data
        try:
            new_val_edges_set, new_test_edges_set = self.processor.generate_validation_test_splits(
                self.config, self.mappings, train_edges_set
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
        
        print("\n" + "="*80)
        print(f"NEGATIVE SAMPLING ({self.config.negative_sampling_strategy.upper()})")
        print("="*80)
        
        # Sample ALL negatives at once
        all_positive_edges = train_edges_set | new_val_edges_set | new_test_edges_set
        
        print(f"\n✨ Sampling ALL {total_negatives_needed} negatives at once...")
        sampler = self._create_sampler(future_positives=new_val_edges_set | new_test_edges_set)
        all_negative_pairs = sampler.sample(
            positive_edges=all_positive_edges,
            all_possible_pairs=all_possible_pairs,
            num_samples=total_negatives_needed,
            edge_index=self.all_edge_index,
            node_features=self.all_features
        )
        print(f"✓ Sampled {len(all_negative_pairs)} total negatives")
        
        # Are there duplicates in the sampled negatives?
        negative_set = set(all_negative_pairs)
        num_duplicates = len(all_negative_pairs) - len(negative_set)
        if num_duplicates > 0:
            print(f"   WARNING: Sampler returned {num_duplicates} DUPLICATE negatives!")
            print(f"   This means the sampler's sample() method is not returning unique samples.")
            print(f"   Deduplicating now...")
            all_negative_pairs = list(negative_set)
            print(f"   After deduplication: {len(all_negative_pairs)} unique negatives")
        
        # Now split the negatives randomly across train/val/test
        print(f"\n Randomly splitting negatives across train/val/test...")
        random.seed(self.config.seed)
        shuffled_negatives = list(all_negative_pairs)
        random.shuffle(shuffled_negatives)
        
        # Split into train/val/test
        train_false_pairs = shuffled_negatives[:num_train_negatives]
        val_false_pairs = shuffled_negatives[num_train_negatives:num_train_negatives + num_val_negatives]
        test_false_pairs = shuffled_negatives[num_train_negatives + num_val_negatives:]
        
        print(f" Train negatives: {len(train_false_pairs)}")
        print(f" Val negatives: {len(val_false_pairs)}")
        print(f" Test negatives: {len(test_false_pairs)}")
        
        # Verify splits
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        
        train_neg_set = set(train_false_pairs)
        val_neg_set = set(val_false_pairs)
        test_neg_set = set(test_false_pairs)
        all_pos_set = train_edges_set | new_val_edges_set | new_test_edges_set
        
        # Check: No negatives overlap with any positives
        neg_pos_overlap = (train_neg_set | val_neg_set | test_neg_set) & all_pos_set
        print(f"Negatives ∩ Positives: {len(neg_pos_overlap)} (should be 0)")
        
        # Check: No overlap between negative splits (by design they shouldn't overlap)
        train_val_overlap = train_neg_set & val_neg_set
        train_test_overlap = train_neg_set & test_neg_set
        val_test_overlap = val_neg_set & test_neg_set
        print(f"Train negatives ∩ Val negatives: {len(train_val_overlap)} (should be 0)")
        print(f"Train negatives ∩ Test negatives: {len(train_test_overlap)} (should be 0)")
        print(f"Val negatives ∩ Test negatives: {len(val_test_overlap)} (should be 0)")
        
        if neg_pos_overlap or train_val_overlap or train_test_overlap or val_test_overlap:
            print("\n VERIFICATION FAILED!")
        else:
            print("\n ALL CHECKS PASSED!")
        
        print("="*80 + "\n")
        
        # Create training tensors
        train_labels = [1] * len(train_true_pairs) + [0] * len(train_false_pairs)
        self.train_edge_tensor = torch.tensor(train_true_pairs + train_false_pairs, dtype=torch.long)
        self.train_label_tensor = torch.tensor(train_labels, dtype=torch.long)
        
        # Create validation tensors
        val_labels = [1] * len(val_true_pairs) + [0] * len(val_false_pairs)
        self.val_edge_tensor = torch.tensor(val_true_pairs + val_false_pairs, dtype=torch.long)
        self.val_label_tensor = torch.tensor(val_labels, dtype=torch.long)
        
        # Create test tensors
        test_labels = [1] * len(test_true_pairs) + [0] * len(test_false_pairs)
        self.test_edge_tensor = torch.tensor(test_true_pairs + test_false_pairs, dtype=torch.long)
        self.test_label_tensor = torch.tensor(test_labels, dtype=torch.long)
        
        print("SPLIT SUMMARY")
        print("="*80)
        print(f"Training: {len(train_true_pairs)} positive, {len(train_false_pairs)} negative (ratio 1:{self.config.train_neg_ratio})")
        print(f"Validation: {len(val_true_pairs)} positive, {len(val_false_pairs)} negative (ratio 1:{self.config.pos_neg_ratio})")
        print(f"Test: {len(test_true_pairs)} positive, {len(test_false_pairs)} negative (ratio 1:{self.config.pos_neg_ratio})")
        print(f"Negative sampling strategy: {self.config.negative_sampling_strategy}")
        print("="*80 + "\n")
    
    def _create_sampler(self, future_positives: Optional[Set[Tuple[int, int]]] = None):
        """
        Create a negative sampler instance based on the configured strategy.
        
        Args:
            future_positives: Set of edges that should not be sampled as negatives
                             (e.g., validation/test positives when sampling train negatives)
        
        Returns:
            NegativeSampler instance
        """
        strategy = self.config.negative_sampling_strategy
        params = self.config.neg_sampling_params or {}
        
        return get_sampler(
            strategy=strategy,
            future_positives=future_positives,
            seed=self.config.seed,
            **params
        )
    
    def build_graph(self):
        """Build the final graph object."""
        print("Building final graph...")
        
        # Create metadata
        node_info = {
            "Drugs": len(self.mappings['approved_drugs_list']),
            "Drug_Types": len(self.mappings['drug_type_list']),
            "Genes": len(self.mappings['gene_list']),
            "Reactome_Pathways": len(self.mappings['reactome_list']),
            "Diseases": len(self.mappings['disease_list']),
            "Therapeutic_Areas": len(self.mappings['therapeutic_area_list'])
        }
        
        edge_info = {
            "Drug-DrugType": int(self.molecule_drugType_edges.size(1)),
            "Drug-Disease": int(self.molecule_disease_edges.size(1)),
            "Drug-Gene": int(self.molecule_gene_edges.size(1)),
            "Gene-Reactome": int(self.gene_reactome_edges.size(1)),
            "Disease-Therapeutic": int(self.disease_therapeutic_edges.size(1)),
            "Disease-Gene": int(self.disease_gene_edges.size(1))
        }
        
        # Add custom edges to metadata if they exist
        if hasattr(self, 'custom_edge_tensor') and self.custom_edge_tensor.size(1) > 0:
            # Count custom edges by type based on network config
            num_custom_edges = int(self.custom_edge_tensor.size(1))
            
            if self.config.network_config.get('disease_similarity_network', False):
                edge_info["Disease-Disease"] = num_custom_edges
                print(f"  Added Disease-Disease edges to metadata: {num_custom_edges}")
            
            if self.config.network_config.get('molecule_similarity_network', False):
                edge_info["Drug-Drug"] = num_custom_edges
                print(f"  Added Drug-Drug edges to metadata: {num_custom_edges}")
            
            if self.config.network_config.get('trial_edges', False):
                edge_info["Trial-Edges"] = num_custom_edges
                print(f"  Added Trial edges to metadata: {num_custom_edges}")
        
        metadata = {
            "node_info": node_info,
            "edge_info": edge_info,
            "data_mode": self.data_mode,
            "graph_creation_config": {
                # Only store config relevant to graph creation
                "training_version": self.config.training_version,
                "validation_version": self.config.validation_version,
                "test_version": self.config.test_version,
                "negative_sampling_strategy": self.config.negative_sampling_strategy,
                "pos_neg_ratio": self.config.pos_neg_ratio,
                "neg_sampling_params": self.config.neg_sampling_params,
                "network_config": self.config.network_config,
                "seed": self.config.seed,
                "edge_features_enabled": self.all_edge_features is not None, 
                "edge_feature_dim": self.all_edge_features.shape[1] if self.all_edge_features is not None else 0,
            },
            "creation_timestamp": dt.datetime.now().isoformat(),
            "total_nodes": sum(node_info.values()),
            "total_edges": sum(edge_info.values())
        }
        
        # Create graph with all splits stored (train/val/test)
        graph = Data(
            x=self.all_features,
            edge_index=self.all_edge_index,
            train_edge_index=self.train_edge_tensor,
            train_edge_label=self.train_label_tensor,
            val_edge_index=self.val_edge_tensor,
            val_edge_label=self.val_label_tensor,
            test_edge_index=self.test_edge_tensor,
            test_edge_label=self.test_label_tensor,
            edge_attr=self.all_edge_features, 
            metadata=metadata
        )
        
        # Convert to undirected
        graph = T.ToUndirected()(graph)
        
        print(f"Graph created: {graph.x.size(0):,} nodes, {graph.edge_index.size(1):,} edges")
        return graph


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create drug-disease prediction graph')
    parser.add_argument('--output-dir', type=str, default='results/', help='Output directory')
    parser.add_argument('--analyze', action='store_true', help='Run graph analysis')
    parser.add_argument('--force-mode', type=str, choices=['raw', 'processed'], 
                        help='Force specific data processing mode (raw or processed)')
    parser.add_argument('--experiment-name', type=str, default='graph_creation',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Load configuration from config.py
    config = get_config()
    
    # Update output path if specified
    if args.output_dir:
        config.update_paths(results=args.output_dir)
    
    # Set reproducibility
    enable_full_reproducibility(config.seed)  # Use config.seed instead of hardcoded 42
    
    # Initialise MLflow tracker
    tracker = ExperimentTracker(experiment_name=args.experiment_name)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"graph_{config.training_version}_{timestamp}"
    
    try:
        tracker.start_run(run_name=run_name)
        
        # Log configuration
        tracker.log_config(config)
        
        # Log additional graph creation parameters
        tracker.log_param("force_mode", args.force_mode if args.force_mode else "auto")
        tracker.log_param("analyze_graph", args.analyze)
        tracker.log_param("output_dir", args.output_dir)
        
        # Create graph
        builder = GraphBuilder(config, args.force_mode, tracker)
        builder.load_or_create_data()
        builder.create_node_features()
        builder.create_edges()
        builder.create_train_val_test_splits()
        
        graph = builder.build_graph()
        
        # Log graph metadata
        tracker.log_graph_metadata(graph)
        
        # Save graph
        graph_filename = f"graph_{config.training_version}_{builder.data_mode}_{timestamp}.pt"
        graph_path = os.path.join(config.paths['results'], graph_filename)
        
        torch.save(graph, graph_path)
        
        # Log graph artifact
        tracker.log_artifact(graph_path, "graphs")
        
        # Run analysis if requested
        if args.analyze:
            print("\nRunning graph analysis...")
            analysis_results = standard_graph_analysis(graph)
            
            # Save analysis results
            analysis_path = graph_path.replace('.pt', '_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Log analysis artifact
            tracker.log_artifact(analysis_path, "analysis")
            
            # Log key analysis metrics
            if 'degree_stats' in analysis_results:
                for key, value in analysis_results['degree_stats'].items():
                    if isinstance(value, (int, float)):
                        tracker.log_metric(f"graph_degree_{key}", value)
            
            print(f"Analysis saved to: {analysis_path}")
        
        print(f"\nGraph creation completed!")
        print(f"Graph saved to: {graph_path}")
        print(f"Nodes: {graph.x.size(0):,}")
        print(f"Edges: {graph.edge_index.size(1):,}")
        print(f"Features: {graph.x.size(1)}")
        print(f"\nMLflow tracking URI: {tracker.experiment_name}")
        print(f"Run ID: {tracker.run_id}")
        
        tracker.end_run()
        return graph_path
        
    except Exception as e:
        print(f"Error during graph creation: {e}")
        tracker.end_run()
        raise


if __name__ == "__main__":
    main()

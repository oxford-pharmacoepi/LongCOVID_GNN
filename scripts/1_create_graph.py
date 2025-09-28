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

# Import from shared modules
from src.utils import (
    set_seed, enable_full_reproducibility, get_indices_from_keys,
    generate_pairs, extract_edges, boolean_encode, normalize, 
    cat_encode, pad_feature_matrix, align_features, standard_graph_analysis
)
from src.data_processing import DataProcessor, detect_data_mode, create_full_dataset
from src.config import get_config, create_custom_config


class GraphBuilder:
    """Main graph builder using shared modules."""
    
    def __init__(self, config):
        self.config = config
        self.processor = DataProcessor(config)
        self.mappings = None
        self.data_mode = detect_data_mode(config)
        
    def load_or_create_data(self):
        """Load existing processed data or create from raw data."""
        print(f"Data mode: {self.data_mode}")
        
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
        import pandas as pd
        
        processed_dir = f"{self.config.paths['processed']}tables/"
        self.molecule_df = pd.read_csv(f"{processed_dir}processed_molecules.csv")
        self.indication_df = pd.read_csv(f"{processed_dir}processed_indications.csv")
        self.disease_df = pd.read_csv(f"{processed_dir}processed_diseases.csv")
        
        # Handle list columns
        if 'approvedIndications' in self.indication_df.columns:
            self.indication_df['approvedIndications'] = self.indication_df['approvedIndications'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
        
        print("Pre-processed data loaded successfully")
    
    def create_from_raw_data(self):
        """Create data from raw OpenTargets files."""
        print("Processing raw OpenTargets data...")
        
        # Load raw data
        indication_table = self.processor.load_indication_data(self.config.paths['indication'])
        molecule_table = self.processor.load_molecule_data(self.config.paths['molecule'])
        disease_table = self.processor.load_disease_data(self.config.paths['disease'])
        gene_table = self.processor.load_gene_data(self.config.paths['gene'], self.config.training_version)
        associations_table, score_column = self.processor.load_associations_data(
            self.config.paths['associations'], self.config.training_version
        )
        
        # Convert to dataframes for processing
        indication_df = indication_table.to_pandas()
        molecule_df = molecule_table.to_pandas()
        disease_df = disease_table.to_pandas()
        
        # Apply ID mappings and filtering
        self.molecule_df, self.indication_df = self.processor.apply_id_mappings(molecule_df, indication_df)
        self.molecule_df = self.processor.filter_linked_molecules(self.molecule_df, self.indication_df)
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
            'processed_associations': associations_table.to_pandas()
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
        
        # Create drug features
        import pyarrow as pa
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
        
        # Create feature matrices for each node type
        drug_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['drug']], dtype=torch.float32).repeat(len(drug_indices), 1),
            blackBoxWarning_vector,
            yearOfFirstApproval_vector
        ], dim=1)
        
        drug_type_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['drug_type']], dtype=torch.float32).repeat(len(drug_type_indices), 1),
            torch.ones(len(drug_type_indices), 2) * -1
        ], dim=1)
        
        gene_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['gene']], dtype=torch.float32).repeat(len(gene_indices), 1),
            torch.ones(len(gene_indices), 2) * -1
        ], dim=1)
        
        reactome_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['reactome']], dtype=torch.float32).repeat(len(reactome_indices), 1),
            torch.ones(len(reactome_indices), 2) * -1
        ], dim=1)
        
        disease_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['disease']], dtype=torch.float32).repeat(len(disease_indices), 1),
            torch.ones(len(disease_indices), 2) * -1
        ], dim=1)
        
        therapeutic_area_feature_matrix = torch.cat([
            torch.tensor([node_type_vectors['therapeutic_area']], dtype=torch.float32).repeat(len(therapeutic_area_indices), 1),
            torch.ones(len(therapeutic_area_indices), 2) * -1
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
    
    def create_edges(self):
        """Create edge indices."""
        print("Creating graph edges...")
        
        # Convert dataframes to PyArrow for edge extraction
        import pyarrow as pa
        
        molecule_table = pa.Table.from_pandas(self.molecule_df)
        indication_table = pa.Table.from_pandas(self.indication_df)
        
        # Extract different edge types
        molecule_drugType_table = molecule_table.select(['id', 'drugType']).drop_null().flatten()
        self.molecule_drugType_edges = extract_edges(
            molecule_drugType_table, 
            self.mappings['drug_key_mapping'], 
            self.mappings['drug_type_key_mapping']
        )
        
        molecule_disease_table = indication_table.select(['id', 'approvedIndications']).flatten()
        self.molecule_disease_edges = extract_edges(
            molecule_disease_table,
            self.mappings['drug_key_mapping'],
            self.mappings['disease_key_mapping']
        )
        
        molecule_gene_table = molecule_table.select(['id', 'linkedTargets.rows']).drop_null().flatten()
        self.molecule_gene_edges = extract_edges(
            molecule_gene_table,
            self.mappings['drug_key_mapping'],
            self.mappings['gene_key_mapping']
        )
        
        # For other edge types, create minimal edges or load from processed data
        # This is simplified - in practice you'd extract from the full dataset
        self.gene_reactome_edges = torch.empty((2, 0), dtype=torch.long)
        self.disease_therapeutic_edges = torch.empty((2, 0), dtype=torch.long)
        self.disease_gene_edges = torch.empty((2, 0), dtype=torch.long)
        
        # Combine all edges
        all_edges = [
            self.molecule_drugType_edges,
            self.molecule_disease_edges,
            self.molecule_gene_edges,
            self.gene_reactome_edges,
            self.disease_therapeutic_edges,
            self.disease_gene_edges
        ]
        
        # Filter out empty tensors
        non_empty_edges = [e for e in all_edges if e.size(1) > 0]
        
        if non_empty_edges:
            self.all_edge_index = torch.cat(non_empty_edges, dim=1)
        else:
            self.all_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"Created edges: {self.all_edge_index.size(1)} total")
        
        # Save edge tensors for future use
        if self.data_mode == "raw":
            edge_dir = f"{self.config.paths['processed']}edges/"
            os.makedirs(edge_dir, exist_ok=True)
            
            torch.save(self.molecule_drugType_edges, f"{edge_dir}1_molecule_drugType_edges.pt")
            torch.save(self.molecule_disease_edges, f"{edge_dir}2_molecule_disease_edges.pt")
            torch.save(self.molecule_gene_edges, f"{edge_dir}3_molecule_gene_edges.pt")
            torch.save(self.gene_reactome_edges, f"{edge_dir}4_gene_reactome_edges.pt")
            torch.save(self.disease_therapeutic_edges, f"{edge_dir}5_disease_therapeutic_edges.pt")
            torch.save(self.disease_gene_edges, f"{edge_dir}6_disease_gene_edges.pt")
    
    def create_train_val_test_splits(self):
        """Create training, validation, and test splits."""
        print("Creating train/validation/test splits...")
        
        # Extract training edges
        train_edges_set = set(zip(
            self.molecule_disease_edges[0].tolist(),
            self.molecule_disease_edges[1].tolist()
        ))
        
        # Generate validation and test splits using temporal data
        try:
            new_val_edges_set, new_test_edges_set = self.processor.generate_validation_test_splits(
                self.config, self.mappings, train_edges_set
            )
        except Exception as e:
            print(f"Warning: Could not generate temporal splits: {e}")
            print("Creating synthetic splits...")
            
            # Fallback: create synthetic splits
            all_pairs = generate_pairs(
                self.mappings['approved_drugs_list'],
                self.mappings['disease_list'],
                self.mappings['drug_key_mapping'],
                self.mappings['disease_key_mapping']
            )
            
            not_linked = list(set(all_pairs) - train_edges_set)
            
            # Sample validation and test edges
            import random
            random.seed(42)
            val_size = min(len(train_edges_set) // 10, len(not_linked) // 2)
            test_size = min(len(train_edges_set) // 10, len(not_linked) - val_size)
            
            new_val_edges_set = set(random.sample(not_linked, val_size))
            remaining_not_linked = list(set(not_linked) - new_val_edges_set)
            new_test_edges_set = set(random.sample(remaining_not_linked, test_size))
        
        # Create validation tensors
        val_true_pairs = list(new_val_edges_set)
        val_false_pairs = random.sample(
            list(set(all_pairs) - train_edges_set - new_val_edges_set), 
            len(val_true_pairs)
        )
        
        val_labels = [1] * len(val_true_pairs) + [0] * len(val_false_pairs)
        self.val_edge_tensor = torch.tensor(val_true_pairs + val_false_pairs, dtype=torch.long)
        self.val_label_tensor = torch.tensor(val_labels, dtype=torch.long)
        
        # Create test tensors
        test_true_pairs = list(new_test_edges_set)
        test_false_pairs = random.sample(
            list(set(all_pairs) - train_edges_set - new_val_edges_set - new_test_edges_set),
            len(test_true_pairs)
        )
        
        test_labels = [1] * len(test_true_pairs) + [0] * len(test_false_pairs)
        self.test_edge_tensor = torch.tensor(test_true_pairs + test_false_pairs, dtype=torch.long)
        self.test_label_tensor = torch.tensor(test_labels, dtype=torch.long)
        
        print(f"Validation: {len(val_true_pairs)} positive, {len(val_false_pairs)} negative")
        print(f"Test: {len(test_true_pairs)} positive, {len(test_false_pairs)} negative")
    
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
        
        metadata = {
            "node_info": node_info,
            "edge_info": edge_info,
            "data_mode": self.data_mode,
            "config": self.config.__dict__,
            "creation_timestamp": dt.datetime.now().isoformat(),
            "total_nodes": sum(node_info.values()),
            "total_edges": sum(edge_info.values())
        }
        
        # Create graph
        graph = Data(
            x=self.all_features,
            edge_index=self.all_edge_index,
            val_edge_index=self.val_edge_tensor,
            val_edge_label=self.val_label_tensor,
            test_edge_index=self.test_edge_tensor,
            test_edge_label=self.test_label_tensor,
            metadata=metadata
        )
        
        # Convert to undirected
        graph = T.ToUndirected()(graph)
        
        print(f"Graph created: {graph.x.size(0):,} nodes, {graph.edge_index.size(1):,} edges")
        return graph


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create drug-disease prediction graph')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--output-dir', type=str, default='results/', help='Output directory')
    parser.add_argument('--analyze', action='store_true', help='Run graph analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = create_custom_config(**config_dict)
    else:
        config = get_config()
    
    # Update output path
    config.update_paths(results=args.output_dir)
    
    # Set reproducibility
    enable_full_reproducibility(42)
    
    # Create graph
    builder = GraphBuilder(config)
    builder.load_or_create_data()
    builder.create_node_features()
    builder.create_edges()
    builder.create_train_val_test_splits()
    
    graph = builder.build_graph()
    
    # Save graph
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_filename = f"graph_{config.training_version}_{builder.data_mode}_{timestamp}.pt"
    graph_path = os.path.join(config.paths['results'], graph_filename)
    
    torch.save(graph, graph_path)
    
    # Run analysis if requested
    if args.analyze:
        print("\nRunning graph analysis...")
        analysis_results = standard_graph_analysis(graph)
        
        # Save analysis results
        analysis_path = graph_path.replace('.pt', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"Analysis saved to: {analysis_path}")
    
    print(f"\nGraph creation completed!")
    print(f"Graph saved to: {graph_path}")
    print(f"Nodes: {graph.x.size(0):,}")
    print(f"Edges: {graph.edge_index.size(1):,}")
    print(f"Features: {graph.x.size(1)}")
    
    return graph_path


if __name__ == "__main__":
    main()

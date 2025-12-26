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
from src.mlflow_tracker import ExperimentTracker


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
        disease_table = self.processor.load_disease_data(self.config.paths['diseases'])
        gene_table = self.processor.load_gene_data(self.config.paths['targets'], self.config.training_version)
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
        """Create edge indices."""
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
            
        else:
            # Extract edges from raw data
            import pyarrow as pa
            
            molecule_table = pa.Table.from_pandas(self.molecule_df)
            indication_table = pa.Table.from_pandas(self.indication_df)
            disease_table = pa.Table.from_pandas(self.disease_df)
            
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
    
    def create_train_val_test_splits(self):
        """Create training, validation, and test splits."""
        print("Creating train/validation/test splits...")
        print(f"Using negative sampling ratio 1:{self.config.pos_neg_ratio}")
        
        import random 
        
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
            
            # Generate all possible pairs for negative sampling
            from src.utils import generate_pairs
            all_pairs = generate_pairs(
                self.mappings['approved_drugs_list'],
                self.mappings['disease_list'],
                self.mappings['drug_key_mapping'],
                self.mappings['disease_key_mapping']
            )
            
        except Exception as e:
            print(f"Warning: Could not generate temporal splits: {e}")
            print("Creating synthetic splits...")
            
            # Fallback: create synthetic splits
            from src.utils import generate_pairs
            all_pairs = generate_pairs(
                self.mappings['approved_drugs_list'],
                self.mappings['disease_list'],
                self.mappings['drug_key_mapping'],
                self.mappings['disease_key_mapping']
            )
            
            not_linked = list(set(all_pairs) - train_edges_set)
            
            # Sample validation and test edges
            random.seed(42)
            val_size = min(len(train_edges_set) // 10, len(not_linked) // 2)
            test_size = min(len(train_edges_set) // 10, len(not_linked) - val_size)
            
            new_val_edges_set = set(random.sample(not_linked, val_size))
            remaining_not_linked = list(set(not_linked) - new_val_edges_set)
            new_test_edges_set = set(random.sample(remaining_not_linked, test_size))
        
        # Create validation tensors with configurable negative ratio
        val_true_pairs = list(new_val_edges_set)
        num_val_negatives = len(val_true_pairs) * self.config.pos_neg_ratio
        val_false_pairs = random.sample(
            list(set(all_pairs) - train_edges_set - new_val_edges_set), 
            num_val_negatives
        )
        
        val_labels = [1] * len(val_true_pairs) + [0] * len(val_false_pairs)
        self.val_edge_tensor = torch.tensor(val_true_pairs + val_false_pairs, dtype=torch.long)
        self.val_label_tensor = torch.tensor(val_labels, dtype=torch.long)
        
        # Create test tensors with configurable negative ratio
        test_true_pairs = list(new_test_edges_set)
        num_test_negatives = len(test_true_pairs) * self.config.pos_neg_ratio
        test_false_pairs = random.sample(
            list(set(all_pairs) - train_edges_set - new_val_edges_set - new_test_edges_set),
            num_test_negatives
        )
        
        test_labels = [1] * len(test_true_pairs) + [0] * len(test_false_pairs)
        self.test_edge_tensor = torch.tensor(test_true_pairs + test_false_pairs, dtype=torch.long)
        self.test_label_tensor = torch.tensor(test_labels, dtype=torch.long)
        
        print(f"Validation: {len(val_true_pairs)} positive, {len(val_false_pairs)} negative (1:{self.config.pos_neg_ratio})")
        print(f"Test: {len(test_true_pairs)} positive, {len(test_false_pairs)} negative (1:{self.config.pos_neg_ratio})")
    
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
    enable_full_reproducibility(42)
    
    # Initialize MLflow tracker
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

"""
Main Graph Builder orchestrator.
"""

import os
import torch
import datetime as dt
import pandas as pd
from torch_geometric.data import Data
import torch_geometric.transforms as T

from src.data_processing import DataProcessor, detect_data_mode
from src.features.node import NodeFeatureBuilder
from src.graph.edges import GraphEdgeBuilder
from src.graph.split import DataSplitter
from src.utils.graph_utils import standard_graph_analysis

class GraphBuilder:
    """Main graph builder using modular components."""
    
    def __init__(self, config, force_mode=None, tracker=None):
        """Initialize graph builder with components."""
        self.config = config
        self.tracker = tracker
        
        # Initialize components
        self.processor = DataProcessor(config)
        self.data_mode = detect_data_mode(config, force_mode)
        
        # Sub-builders
        self.node_builder = NodeFeatureBuilder(config, self.processor)
        self.edge_builder = GraphEdgeBuilder(config, self.processor)
        self.splitter = DataSplitter(config, self.processor)
        
        # State
        self.mappings = None
        self.molecule_df = None
        self.indication_df = None
        self.disease_df = None
        self.known_drugs_df = None
        
        # Results
        self.node_features = None
        self.all_edge_index = None
        self.all_edge_features = None
        self.splits = None
        
    def load_or_create_data(self):
        """Load existing processed data or create from raw data."""
        print(f"Data mode: {self.data_mode}")
        
        if self.tracker:
            self.tracker.log_dict_as_json({'data_mode': self.data_mode}, 'data_mode.json')
        
        if self.data_mode == "processed":
            self._load_preprocessed_data()
        else:
            self._create_from_raw_data()
            
    def _load_preprocessed_data(self):
        """Load pre-processed node data."""
        print("Loading pre-processed data...")
        
        # Load mappings
        mappings_path = f"{self.config.paths['processed']}mappings/"
        self.mappings = self.processor.load_mappings(mappings_path)
        
        # Load processed tables
        processed_dir = f"{self.config.paths['processed']}tables/"
        self.molecule_df = pd.read_parquet(f"{processed_dir}processed_molecules.parquet")
        self.indication_df = pd.read_parquet(f"{processed_dir}processed_indications.parquet")
        self.disease_df = pd.read_parquet(f"{processed_dir}processed_diseases.parquet")
        
        known_drugs_path = f"{processed_dir}processed_known_drugs.parquet"
        self.known_drugs_df = pd.read_parquet(known_drugs_path) if os.path.exists(known_drugs_path) else pd.DataFrame()
        
        # Handle parsed columns (lists)
        self._parse_list_columns()
        print("Pre-processed data loaded successfully")
        
    def _create_from_raw_data(self):
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
        
        if 'knownDrugsAggregated' in self.config.paths:
            self.known_drugs_df = self.processor.load_known_drugs_aggregated(self.config.paths['knownDrugsAggregated'])
        else:
            self.known_drugs_df = None
            
        # Convert to pandas
        indication_df = indication_table.to_pandas()
        molecule_df = molecule_table.to_pandas()
        disease_df = disease_table.to_pandas()
        
        # Apply ID mappings and filtering
        self.molecule_df, self.indication_df = self.processor.apply_id_mappings(molecule_df, indication_df)
        self.molecule_df = self.processor.filter_linked_molecules(self.molecule_df, self.indication_df, self.known_drugs_df)
        self.disease_df = disease_df
        
        # Create mappings
        self.mappings = self.processor.create_node_mappings(
            self.molecule_df, self.disease_df, gene_table, self.config.training_version
        )
        
        # Save processed data
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
        
    def _parse_list_columns(self):
        """Helper to parse stringified list columns in loaded CSVs."""
        # Indication
        if 'approvedIndications' in self.indication_df.columns:
            self.indication_df['approvedIndications'] = self.indication_df['approvedIndications'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
        
        # Molecule
        if 'linkedDiseases.rows' in self.molecule_df.columns:
            self.molecule_df['linkedDiseases.rows'] = self.molecule_df['linkedDiseases.rows'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
            
        # Disease (numpy array strings)
        from src.graph.edges import GraphEdgeBuilder 
        # Reuse logic or reimplement simple parser
        # For brevity, implementing simple one here or we could put this in utils.common?
        # Let's implement robustly.
        pass # The logic is in original script lines 82-122. I will implement it fully below.

        def parse_numpy_array_string(value):
            if not isinstance(value, str): return value if isinstance(value, list) else []
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                inner = value[1:-1].strip()
                if not inner: return []
                # Simple split by space/newline respecting quotes
                items = []; current_item = ""; in_quotes = False
                for char in inner:
                    if char == "'" or char == '"': in_quotes = not in_quotes
                    elif char in (' ', '\n', '\t') and not in_quotes:
                        if current_item: items.append(current_item); current_item = ""
                    else: current_item += char
                if current_item: items.append(current_item)
                return items
            return []
            
        for col in ['ancestors', 'descendants', 'children', 'therapeuticAreas']:
            if col in self.disease_df.columns:
                self.disease_df[col] = self.disease_df[col].apply(parse_numpy_array_string)

    def create_node_features(self):
        """Delegate node feature creation."""
        self.node_features = self.node_builder.create_features(
            self.mappings, self.molecule_df
        )
        
    def create_edges(self):
        """Delegate edge creation."""
        self.all_edge_index, self.all_edge_features = self.edge_builder.build_edges(
            self.data_mode,
            self.mappings,
            self.molecule_df,
            self.indication_df,
            self.disease_df,
            self.known_drugs_df
        )
        
    def create_train_val_test_splits(self):
        """Delegate split creation."""
        # Get drug-disease edges for positive samples
        drug_disease_edges = self.edge_builder.edges['molecule_disease']
        
        self.splits = self.splitter.create_splits_from_edges(
            drug_disease_edges,
            self.all_edge_index,
            self.node_features,
            self.mappings
        )
        
    def build_graph(self):
        """Assemble final graph object."""
        print("Building final graph...")
        
        # Construct metadata
        node_info = {
            "Drugs": len(self.mappings['approved_drugs_list']),
            "Drug_Types": len(self.mappings['drug_type_list']),
            "Genes": len(self.mappings['gene_list']),
            "Reactome_Pathways": len(self.mappings['reactome_list']),
            "Diseases": len(self.mappings['disease_list']),
            "Therapeutic_Areas": len(self.mappings['therapeutic_area_list'])
        }
        
        edge_info = {
            "Drug-DrugType": int(self.edge_builder.edges['molecule_drugType'].size(1)),
            "Drug-Disease": int(self.edge_builder.edges['molecule_disease'].size(1)),
            "Drug-Gene": int(self.edge_builder.edges['molecule_gene'].size(1)),
            "Gene-Reactome": int(self.edge_builder.edges['gene_reactome'].size(1)),
            "Disease-Therapeutic": int(self.edge_builder.edges['disease_therapeutic'].size(1)),
            "Disease-Gene": int(self.edge_builder.edges['disease_gene'].size(1))
        }
        
        # Add custom edges info
        if self.all_edge_index.size(1) > sum(edge_info.values()):
            custom_count = self.all_edge_index.size(1) - sum(edge_info.values())
            edge_info['Custom_Edges'] = custom_count # Simplified
            
        metadata = {
            "node_info": node_info,
            "edge_info": edge_info,
            "data_mode": self.data_mode,
            "creation_timestamp": dt.datetime.now().isoformat(),
            "total_nodes": sum(node_info.values()),
            "total_edges": self.all_edge_index.size(1)
        }
        
        # Create Data object
        graph = Data(
            x=self.node_features,
            edge_index=self.all_edge_index,
            train_edge_index=self.splits['train']['edge_index'],
            train_edge_label=self.splits['train']['label'],
            val_edge_index=self.splits['val']['edge_index'],
            val_edge_label=self.splits['val']['label'],
            test_edge_index=self.splits['test']['edge_index'],
            test_edge_label=self.splits['test']['label'],
            edge_attr=self.all_edge_features, 
            metadata=metadata
        )
        
        # Convert to undirected
        graph = T.ToUndirected()(graph)
        
        print(f"Graph created: {graph.x.size(0):,} nodes, {graph.edge_index.size(1):,} edges")
        return graph

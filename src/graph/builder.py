"""
Main Graph Builder orchestrator.
"""

import os
import torch
import datetime as dt
import pandas as pd
import numpy as np
import ast
import re
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import remove_isolated_nodes, subgraph, dropout_adj

from src.data_processing import DataProcessor, detect_data_mode
from src.features.node import NodeFeatureBuilder
from src.graph.edges import GraphEdgeBuilder
from src.graph.split import DataSplitter
from src.utils.graph_utils import standard_graph_analysis

class GraphBuilder:
    """Main graph builder using modular components."""
    
    def __init__(self, config, force_mode=None, tracker=None):
        """Initialise graph builder with components."""
        self.config = config
        self.tracker = tracker
        
        # Initialise components
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
        """Parse stringified list/numpy array columns back into lists."""
        
        def safe_parse(val):
            if val is None:
                return []
            if isinstance(val, (list, np.ndarray)):
                return [str(x) for x in val] if hasattr(val, '__iter__') else []
            if not isinstance(val, str):
                return []
            val = val.strip()
            if not val or val == 'nan':
                return []
            if val.startswith('[') and val.endswith(']'):
                try:
                    res = ast.literal_eval(val)
                    if isinstance(res, list):
                        return [str(x) for x in res]
                    return [str(res)]
                except:
                    inner = val[1:-1].strip()
                    if not inner: return []
                    if ',' not in inner:
                        return [item.strip(" '\"") for item in re.split(r'\s+', inner) if item.strip()]
                    else:
                        return [item.strip(" '\"") for item in inner.split(',') if item.strip()]
            return [val]

        # Keep only necessary columns to avoid Arrow issues with unused metadata
        mol_needed = ['id', 'drugType', 'blackBoxWarning', 'yearOfFirstApproval', 'parentId', 'childChemblIds', 'linkedTargets.rows', 'linkedDiseases.rows']
        self.molecule_df = self.molecule_df[[c for c in mol_needed if c in self.molecule_df.columns]]
        
        dis_needed = ['id', 'name', 'description', 'therapeuticAreas', 'parents', 'children', 'ancestors', 'descendants']
        self.disease_df = self.disease_df[[c for c in dis_needed if c in self.disease_df.columns]]

        # Molecule columns
        if 'blackBoxWarning' in self.molecule_df.columns:
             self.molecule_df['blackBoxWarning'] = self.molecule_df['blackBoxWarning'].apply(lambda x: x if isinstance(x, bool) else (str(x).lower() == 'true'))
        
        list_cols = {
            'molecule_df': ['linkedTargets.rows', 'linkedDiseases.rows', 'tradeNames', 'synonyms', 'urls'],
            'indication_df': ['approvedIndications'], # Added indication_df here
            'disease_df': ['therapeuticAreas', 'parents', 'children', 'ancestors', 'descendants', 'synonyms']
        }
        
        for df_name, cols in list_cols.items():
            df = getattr(self, df_name)
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].apply(safe_parse)

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
        
        # Prune graph to ensure connectivity (remove isolated nodes)
        graph = self._prune_graph(graph)
        
        # We don't save mappings to 'processed' here anymore because it might 
        # break the ability to re-run different graph configs.
        # The script calling the builder should handle saving result mappings.
        
        print(f"Graph created: {graph.x.size(0):,} nodes, {graph.edge_index.size(1):,} edges")
        return graph

    def _prune_graph(self, graph):
        """
        Prune graph to remove isolated nodes and ensure a single connected component if possible.
        Updates metadata counts accordingly.
        """
        print("\nPruning graph (removing isolated nodes)...")
        num_nodes_before = graph.x.size(0)
        
        # 1. Identify isolated nodes
        edge_index = graph.edge_index
        node_indices = torch.unique(edge_index)
        
        mask = torch.zeros(num_nodes_before, dtype=torch.bool)
        mask[node_indices] = True
        
        # 2. Re-map indices for all edge tensors
        # PyG remove_isolated_nodes only handles the main edge_index
        # We need to handle train/val/test too
        
        # Get the mapping from old index to new index
        old_to_new = torch.full((num_nodes_before,), -1, dtype=torch.long)
        new_nodes_count = mask.sum().item()
        old_to_new[mask] = torch.arange(new_nodes_count)
        
        # Filter and remap labels
        def filter_and_remap(edge_tensor, labels):
            # edge_tensor is [N, 2]
            src = edge_tensor[:, 0]
            dst = edge_tensor[:, 1]
            
            # Keep only pairs where both nodes exist in the pruned graph
            valid_mask = mask[src] & mask[dst]
            
            new_edge_tensor = torch.stack([
                old_to_new[src[valid_mask]],
                old_to_new[dst[valid_mask]]
            ], dim=1)
            
            new_labels = labels[valid_mask]
            return new_edge_tensor, new_labels
            
        print(f"  Filtering {graph.train_edge_index.size(0):,} training pairs...")
        graph.train_edge_index, graph.train_edge_label = filter_and_remap(
            graph.train_edge_index, graph.train_edge_label
        )
        
        print(f"  Filtering {graph.val_edge_index.size(0):,} validation pairs...")
        graph.val_edge_index, graph.val_edge_label = filter_and_remap(
            graph.val_edge_index, graph.val_edge_label
        )
        
        print(f"  Filtering {graph.test_edge_index.size(0):,} test pairs...")
        graph.test_edge_index, graph.test_edge_label = filter_and_remap(
            graph.test_edge_index, graph.test_edge_label
        )
        
        # Update node features and metadata
        # We must know how many of each type were removed to update metadata accurately
        # Order: Drugs, DrugType, Genes, Reactome, Disease, Therapeutic
        node_info = graph.metadata['node_info']
        new_node_info = {}
        
        current_idx = 0
        for type_name, count in node_info.items():
            type_mask = torch.zeros(num_nodes_before, dtype=torch.bool)
            type_mask[current_idx:current_idx + count] = True
            
            # Nodes of this type that were kept
            kept_count = (type_mask & mask).sum().item()
            new_node_info[type_name] = kept_count
            current_idx += count
            if count != kept_count:
                print(f"  Removed {count - kept_count} isolated {type_name}")
        
        graph.metadata['node_info'] = new_node_info
        graph.metadata['total_nodes'] = new_nodes_count
        graph.x = graph.x[mask]
        
        # Remap main edge_index
        graph.edge_index = old_to_new[graph.edge_index]
        
        # Update mappings in self.mappings to stay in sync with pruned indices
        if hasattr(self, 'mappings') and self.mappings:
            print("  Syncing mappings with pruned indices...")
            for map_name, mapping in self.mappings.items():
                if not isinstance(mapping, dict):
                    continue
                
                new_mapping = {}
                count_removed = 0
                for key, old_idx in mapping.items():
                    # Ensure old_idx is within range
                    if old_idx >= num_nodes_before:
                        continue
                        
                    new_idx = old_to_new[old_idx].item()
                    if new_idx != -1:
                        new_mapping[key] = new_idx
                    else:
                        count_removed += 1
                
                self.mappings[map_name] = new_mapping
                
                # Also synchronise the corresponding list if it exists
                list_name = map_name.replace('_key_mapping', '_list')
                if list_name in self.mappings:
                    # Sort the keys that were kept by their new global indices
                    # and create the list in that order
                    kept_keys_with_indices = []
                    for key, ni in new_mapping.items():
                        kept_keys_with_indices.append((key, ni))
                    
                    # Sort by index
                    kept_keys_with_indices.sort(key=lambda x: x[1])
                    
                    # New list is just the keys in that order
                    new_list = [x[0] for x in kept_keys_with_indices]
                    self.mappings[list_name] = new_list
                    print(f"    - Updated {list_name}: filtered to {len(new_list)} items")
                
                if count_removed > 0:
                    print(f"    - Updated {map_name}: removed {count_removed} keys")
        
        print(f"  Dataset pruned from {num_nodes_before:,} to {new_nodes_count:,} nodes.")
        
        # 3. Strictly ensure one component if requested (largest component)
        # Note: We already remove isolated nodes above. 
        # For now, isolated nodes is usually enough to connect everyone who has edges.
        
        return graph

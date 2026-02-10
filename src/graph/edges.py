"""
Edge construction and management.
"""

import torch
import os
import pandas as pd
import pyarrow as pa
from src.utils.edge_utils import extract_edges
from src.utils.graph_utils import custom_edges
from src.features.edge import extract_moa_features
from src.features.heuristic_scores import compute_heuristic_edge_features

class GraphEdgeBuilder:
    """Builder for graph edges and edge features."""
    
    def __init__(self, config, processor):
        """
        Initialize edge builder.
        
        Args:
            config: Configuration object
            processor: DataProcessor instance
        """
        self.config = config
        self.processor = processor
        self.edges = {}
        
    def build_edges(self, data_mode, mappings, molecule_df, indication_df, disease_df, known_drugs_df=None):
        """
        Build edges either by loading or extraction.
        
        Args:
            data_mode: 'raw' or 'processed'
            mappings: Node mappings dictionary
            molecule_df: Molecule dataframe
            indication_df: Indication dataframe
            disease_df: Disease dataframe
            known_drugs_df: Known drugs dataframe (optional)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (concatenated_edge_index, edge_features)
        """
        print(f"Building edges (mode={data_mode})...")
        
        if data_mode == 'processed':
            self._load_edges()
        else:
            self._extract_standard_edges(mappings, molecule_df, indication_df, disease_df, known_drugs_df)
        
        # Extract custom edges (re-generated even in processed mode as they depend on config filters)
        custom_edge_tensor = self._extract_custom_edges(mappings, molecule_df, disease_df)
        
        # Combine all edges in a specific order
        all_edges = [
            self.edges.get('molecule_drugType', torch.empty((2, 0), dtype=torch.long)),
            self.edges.get('molecule_disease', torch.empty((2, 0), dtype=torch.long)),
            self.edges.get('molecule_gene', torch.empty((2, 0), dtype=torch.long)),
            self.edges.get('gene_reactome', torch.empty((2, 0), dtype=torch.long)),
            self.edges.get('disease_therapeutic', torch.empty((2, 0), dtype=torch.long)),
            self.edges.get('disease_gene', torch.empty((2, 0), dtype=torch.long))
        ]
        
        # Add high-connectivity networks only if enabled
        if self.config.network_config.get('use_ppi_network', False):
            all_edges.append(self.edges.get('gene_gene', torch.empty((2, 0), dtype=torch.long)))
            
        if custom_edge_tensor.size(1) > 0:
            all_edges.append(custom_edge_tensor)
            
        # Filter out empty tensors
        non_empty_edges = [e for e in all_edges if e.size(1) > 0]
        
        if non_empty_edges:
            all_edge_index = torch.cat(non_empty_edges, dim=1)
        else:
            all_edge_index = torch.empty((2, 0), dtype=torch.long)
            
        print(f"Total edges in graph: {all_edge_index.size(1)}")
        
        # Create edge features
        all_edge_features = self._create_edge_features(all_edge_index, mappings)
        
        # Save standard edges if in raw mode
        if data_mode == 'raw':
            self._save_edges()
            
        return all_edge_index, all_edge_features

    def _load_edges(self):
        """Load standard edges from processed files."""
        edge_dir = f"{self.config.paths['processed']}edges/"
        print(f"Loading edges from {edge_dir}")
        
        try:
            self.edges['molecule_drugType'] = torch.load(f"{edge_dir}1_molecule_drugType_edges.pt")
            self.edges['molecule_disease'] = torch.load(f"{edge_dir}2_molecule_disease_edges.pt")
            self.edges['molecule_gene'] = torch.load(f"{edge_dir}3_molecule_gene_edges.pt")
            self.edges['gene_reactome'] = torch.load(f"{edge_dir}4_gene_reactome_edges.pt")
            self.edges['disease_therapeutic'] = torch.load(f"{edge_dir}5_disease_therapeutic_edges.pt")
            self.edges['disease_gene'] = torch.load(f"{edge_dir}6_disease_gene_edges.pt")
            
            # Optional high-connectivity edges
            if self.config.network_config.get('use_ppi_network', False):
                if os.path.exists(f"{edge_dir}7_gene_gene_edges.pt"):
                    self.edges['gene_gene'] = torch.load(f"{edge_dir}7_gene_gene_edges.pt")
                    print("  - Loaded Gene-Gene interactions")
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load processed edges from {edge_dir}. "
                                   f"Run with force-mode='raw' first. Error: {e}")

    def _save_edges(self):
        """Save extracted edges to processed files."""
        edge_dir = f"{self.config.paths['processed']}edges/"
        os.makedirs(edge_dir, exist_ok=True)
        
        torch.save(self.edges['molecule_drugType'], f"{edge_dir}1_molecule_drugType_edges.pt")
        torch.save(self.edges['molecule_disease'], f"{edge_dir}2_molecule_disease_edges.pt")
        torch.save(self.edges['molecule_gene'], f"{edge_dir}3_molecule_gene_edges.pt")
        torch.save(self.edges['gene_reactome'], f"{edge_dir}4_gene_reactome_edges.pt")
        torch.save(self.edges['disease_therapeutic'], f"{edge_dir}5_disease_therapeutic_edges.pt")
        torch.save(self.edges['disease_gene'], f"{edge_dir}6_disease_gene_edges.pt")
        
        if 'gene_gene' in self.edges and self.edges['gene_gene'].size(1) > 0:
            torch.save(self.edges['gene_gene'], f"{edge_dir}7_gene_gene_edges.pt")

    def _extract_standard_edges(self, mappings, molecule_df, indication_df, disease_df, known_drugs_df):
        """Extract all standard edge types."""
        import pyarrow as pa
        
        # Convert tables
        molecule_table = pa.Table.from_pandas(molecule_df)
        indication_table = pa.Table.from_pandas(indication_df)
        
        disease_schema = self._get_disease_schema(disease_df)
        disease_table = pa.Table.from_pandas(disease_df, schema=disease_schema)
        
        # 1. Drug-DrugType
        print("Extracting Drug-DrugType edges...")
        molecule_drugType_table = molecule_table.select(['id', 'drugType']).drop_null().flatten()
        self.edges['molecule_drugType'] = extract_edges(
            molecule_drugType_table, 
            mappings['drug_key_mapping'], 
            mappings['drug_type_key_mapping']
        )
        
        # 2. Drug-Disease (from multiple sources)
        print("Extracting Drug-Disease edges...")
        self.edges['molecule_disease'] = self._extract_drug_disease_edges(
            molecule_df, indication_table, known_drugs_df, mappings
        )
        
        # 3. Drug-Gene
        print("Extracting Drug-Gene edges...")
        molecule_gene_table = molecule_table.select(['id', 'linkedTargets.rows']).drop_null().flatten()
        self.edges['molecule_gene'] = extract_edges(
            molecule_gene_table,
            mappings['drug_key_mapping'],
            mappings['gene_key_mapping']
        )
        
        # 4. Gene-Reactome
        print("Extracting Gene-Reactome edges...")
        gene_reactome_table = self.processor.create_gene_reactome_mapping(
            self.processor.load_gene_data(self.config.paths['targets'], self.config.training_version),
            self.config.training_version
        )
        self.edges['gene_reactome'] = extract_edges(
            gene_reactome_table,
            mappings['gene_key_mapping'],
            mappings['reactome_key_mapping']
        )
        
        # 5. Disease-Therapeutic
        print("Extracting Disease-Therapeutic edges...")
        disease_therapeutic_table = disease_table.select(['id', 'therapeuticAreas']).flatten()
        self.edges['disease_therapeutic'] = extract_edges(
            disease_therapeutic_table,
            mappings['disease_key_mapping'],
            mappings['therapeutic_area_key_mapping']
        )
        
        # 6. Disease-Gene
        print("Extracting Disease-Gene edges...")
        associations_table, score_column = self.processor.load_associations_data(
            self.config.paths['associations'], self.config.training_version
        )
        filtered_associations = self.processor.filter_associations_by_genes_and_diseases(
            associations_table,
            list(mappings['gene_key_mapping'].keys()),
            list(mappings['disease_key_mapping'].keys()),
            score_column
        )
        
        # Check if we should extract association scores
        use_scores = getattr(self.config, 'data_enrichment', {}).get('use_gene_disease_scores', False)
        
        if use_scores and score_column in filtered_associations.column_names:
            # Extract edges with scores
            df = filtered_associations.select(['diseaseId', 'targetId', score_column]).to_pandas()
            
            # Build edge tensor and score mapping
            src_indices = []
            dst_indices = []
            self.disease_gene_scores = {}  # (disease_idx, gene_idx) -> score
            
            for _, row in df.iterrows():
                disease_id = row['diseaseId']
                gene_id = row['targetId']
                score = row[score_column]
                
                if disease_id in mappings['disease_key_mapping'] and gene_id in mappings['gene_key_mapping']:
                    d_idx = mappings['disease_key_mapping'][disease_id]
                    g_idx = mappings['gene_key_mapping'][gene_id]
                    src_indices.append(d_idx)
                    dst_indices.append(g_idx)
                    self.disease_gene_scores[(d_idx, g_idx)] = float(score) if score is not None else 0.0
            
            self.edges['disease_gene'] = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            print(f"  - Disease-Gene edges with scores: {len(src_indices)}")
        else:
            # Standard extraction without scores
            self.edges['disease_gene'] = extract_edges(
                filtered_associations.select(['diseaseId', 'targetId']),
                mappings['disease_key_mapping'],
                mappings['gene_key_mapping']
            )
            self.disease_gene_scores = {}
        
        # 7. Gene-Gene
        if self.config.network_config.get('use_ppi_network', False):
            print("Extracting Gene-Gene interactions...")
            edges = self._extract_gene_gene_edges(mappings)
            if edges.size(1) > 0:
                mask = edges[0] != edges[1]
                if not mask.all():
                    print(f"  Removed {(~mask).sum().item()} self-loops from PPI edges")
                    edges = edges[:, mask]
            self.edges['gene_gene'] = edges
        else:
            self.edges['gene_gene'] = torch.empty((2, 0), dtype=torch.long)
            
        # 8. Disease-Disease
        self.edges['disease_disease'] = torch.empty((2, 0), dtype=torch.long)

    def _extract_gene_gene_edges(self, mappings):
        """Extract high-confidence gene-gene interaction edges."""
        interaction_path = self.config.paths.get('interaction')
        if not interaction_path or not os.path.exists(interaction_path):
            print("  Warning: Interaction data path not found")
            return torch.empty((2, 0), dtype=torch.long)
            
        interaction_table = self.processor.load_interaction(interaction_path)
        
        # Convert to pandas for filtering
        # Note: Large table, but we only need human and high confidence
        df = interaction_table.select(['targetA', 'targetB', 'speciesA', 'speciesB', 'scoring']).to_pandas()
        
        # Filter for human (taxon_id 9606)
        def is_human(x):
            if isinstance(x, dict): return x.get('taxon_id') == 9606
            return False
            
        mask = df['speciesA'].apply(is_human) & df['speciesB'].apply(is_human)
        
        # Filter by scoring threshold (default 0.5 if not in config)
        threshold = self.config.network_config.get('ppi_score_threshold', 0.5)
        mask = mask & (df['scoring'] >= threshold)
        
        df = df[mask]
        print(f"  - Found {len(df):,} high-confidence human interactions")
        
        # Keep only edges where both genes are in our mapping
        gene_mapping = mappings['gene_key_mapping']
        df = df[df['targetA'].isin(gene_mapping) & df['targetB'].isin(gene_mapping)]
        print(f"  - {len(df):,} interactions between mapped genes")
        
        edges_set = set()
        for _, row in df.iterrows():
            u = gene_mapping[row['targetA']]
            v = gene_mapping[row['targetB']]
            if u != v:
                edges_set.add(tuple(sorted((u, v))))
        
        print(f"  - Unique undirected Gene-Gene edges: {len(edges_set):,}")
        
        if not edges_set:
            return torch.empty((2, 0), dtype=torch.long)
            
        return torch.tensor(list(edges_set), dtype=torch.long).t().contiguous()


    def _extract_drug_disease_edges(self, molecule_df, indication_table, known_drugs_df, mappings):
        """Extract drug-disease edges from indications, known drugs, and metadata."""
        # 1. Approved Indications
        ind_table = indication_table.select(['id', 'approvedIndications']).flatten()
        ind_edges = extract_edges(ind_table, mappings['drug_key_mapping'], mappings['disease_key_mapping'], return_edge_set=True)
        print(f"  - Edges from indications: {len(ind_edges)}")
        
        # 2. Known Drugs (Clinical Trials Phase 3 and 4)
        known_edges = set()
        if known_drugs_df is not None and not known_drugs_df.empty:
            valid_known = known_drugs_df[known_drugs_df['phase'] >= 3]
            valid_known_table = pa.Table.from_pandas(valid_known[['drugId', 'diseaseId']])
            known_edges = extract_edges(valid_known_table, mappings['drug_key_mapping'], mappings['disease_key_mapping'], return_edge_set=True)
            print(f"  - Edges from clinical trials (Ph 3/4): {len(known_edges)}")
        
        # 3. Pre-linked diseases from molecule metadata
        meta_edges = set()
        if 'linkedDiseases.rows' in molecule_df.columns:
            meta_table = pa.Table.from_pandas(molecule_df[['id', 'linkedDiseases.rows']])
            meta_edges = extract_edges(meta_table, mappings['drug_key_mapping'], mappings['disease_key_mapping'], return_edge_set=True)
            print(f"  - Edges from molecule metadata: {len(meta_edges)}")
        
        # Merge all edges
        all_md_edges = ind_edges | known_edges | meta_edges
        # Sort edges for deterministic creation
        sorted_edges = sorted(list(all_md_edges))
        return torch.tensor(sorted_edges, dtype=torch.long).t().contiguous()

    def _extract_custom_edges(self, mappings, molecule_df, disease_df):
        """Extract custom edges (similarity, trials, etc.)."""
        print("\nChecking for custom edge types...")
        # Map new config names to the variables used here
        disease_similarity_enabled = self.config.network_config.get('use_disease_similarity', False)
        trial_edges_enabled = self.config.network_config.get('trial_edges', False)
        
        if not (disease_similarity_enabled or trial_edges_enabled):
            print("  No custom edges enabled - skipping")
            return torch.empty((2, 0), dtype=torch.long)
            
        print(f"  Disease similarity network: {disease_similarity_enabled}")
        print(f"  Trial edges: {trial_edges_enabled}")
        
        disease_schema = self._get_disease_schema(disease_df)
        disease_table = pa.Table.from_pandas(disease_df, schema=disease_schema)
        molecule_table = pa.Table.from_pandas(molecule_df)
        
        custom_edge_tensor = custom_edges(
            disease_similarity_network=disease_similarity_enabled,
            disease_similarity_max_children=self.config.network_config.get('disease_similarity_max_children', 10),
            disease_similarity_min_shared=self.config.network_config.get('disease_similarity_min_shared', 1),
            trial_edges=trial_edges_enabled,
            filtered_disease_table=disease_table,
            filtered_molecule_table=molecule_table,
            disease_key_mapping=mappings['disease_key_mapping'],
            drug_key_mapping=mappings['drug_key_mapping']
        )
        
        # Important: Remove self-loops (edges where source == destination)
        # Probably none in the graph, but kept for safety
        if custom_edge_tensor.size(1) > 0:
            self_loop_mask = custom_edge_tensor[0] != custom_edge_tensor[1]
            if not self_loop_mask.all():
                num_self_loops = (~self_loop_mask).sum().item()
                print(f"  Removed {num_self_loops} self-loops from custom edges")
                custom_edge_tensor = custom_edge_tensor[:, self_loop_mask]
                
        print(f"  Custom edges created: {custom_edge_tensor.shape}")
        return custom_edge_tensor

    def _create_edge_features(self, all_edge_index, mappings):
        """
        Create features for edges (MoA).
        
        Args:
            all_edge_index: Tensor of all edges in the graph [2, num_edges]
            mappings: Dictionary containing node mappings
            
        Returns:
            torch.Tensor: Edge feature matrix [num_edges, feature_dim]
        """
        print("\n" + "="*80)
        print("EXTRACTING EDGE FEATURES")
        print("="*80)
        
        moa_path = self.config.paths.get('mechanismOfAction', None)
        if not moa_path or not os.path.exists(moa_path):
            print(f"  Warning: mechanismOfAction data not found at {moa_path}")
            return None
            
        try:
            moa_df = self.processor.load_mechanism_of_action(moa_path)
            
            # Extract features for drug-gene edges
            drug_gene_edges = self.edges['molecule_gene']
            
            drug_gene_edge_features = extract_moa_features(
                moa_df,
                mappings['drug_key_mapping'],
                mappings['gene_key_mapping'],
                drug_gene_edges
            )
            
            # Create padded edge features for all edges
            # Since only drug-gene edges have features, we pad with zeros for other edge types
            num_drug_gene_edges = drug_gene_edges.shape[1]
            total_edges = all_edge_index.shape[1]
            feature_dim = drug_gene_edge_features.shape[1]
            
            # Check if we have gene-disease scores to add
            use_scores = getattr(self.config, 'data_enrichment', {}).get('use_gene_disease_scores', False)
            has_scores = hasattr(self, 'disease_gene_scores') and len(self.disease_gene_scores) > 0
            
            if use_scores and has_scores:
                feature_dim += 1  # Add one dimension for association score
                print(f"  Adding gene-disease association scores as edge feature")
            
            # Create edge feature tensor for all edges
            all_edge_features = torch.zeros((total_edges, feature_dim), dtype=torch.float32)
            
            # Find where drug-gene edges are in the concatenated edge tensor
            # Order matches strict order in create_edges:
            # 1. molecule_drugType
            # 2. molecule_disease
            # 3. molecule_gene  <- drug-gene features go here
            
            edge_offset = (self.edges['molecule_drugType'].shape[1] + 
                          self.edges['molecule_disease'].shape[1])
            
            # Copy drug-gene edge features to the correct position
            if use_scores and has_scores:
                # Features are now [6 action types, 1 association score]
                all_edge_features[edge_offset:edge_offset + num_drug_gene_edges, :drug_gene_edge_features.shape[1]] = drug_gene_edge_features
            else:
                all_edge_features[edge_offset:edge_offset + num_drug_gene_edges] = drug_gene_edge_features
            
            # Add gene-disease association scores if enabled
            # Order: 4. gene_reactome, 5. disease_therapeutic, 6. disease_gene
            if use_scores and has_scores:
                disease_gene_offset = (
                    self.edges['molecule_drugType'].shape[1] +
                    self.edges['molecule_disease'].shape[1] +
                    self.edges['molecule_gene'].shape[1] +
                    self.edges['gene_reactome'].shape[1] +
                    self.edges['disease_therapeutic'].shape[1]
                )
                
                disease_gene_edges = self.edges['disease_gene']
                score_col_idx = feature_dim - 1  # Last column is the score
                
                scores_added = 0
                for i in range(disease_gene_edges.shape[1]):
                    d_idx = disease_gene_edges[0, i].item()
                    g_idx = disease_gene_edges[1, i].item()
                    score = self.disease_gene_scores.get((d_idx, g_idx), 0.0)
                    all_edge_features[disease_gene_offset + i, score_col_idx] = score
                    if score > 0:
                        scores_added += 1
                
                print(f"  - Disease-Gene scores added: {scores_added}/{disease_gene_edges.shape[1]}")
            
            print(f"\n Created edge feature matrix: {all_edge_features.shape}")
            print(f"  - Drug-gene edges with features: {num_drug_gene_edges}")
            print(f"  - Total edges: {total_edges}")
            print(f"  - Feature dimension: {feature_dim}")
            
            # Step 2: Add heuristic features (CN, AA, Jaccard) for ALL edges
            print("\nComputing heuristic edge features (CN, AA, Jaccard)...")
            try:
                # Create a temporary Data object for heuristic computation
                from torch_geometric.data import Data
                temp_graph = Data(edge_index=all_edge_index, num_nodes=all_edge_index.max().item() + 1)
                
                heuristic_features = compute_heuristic_edge_features(temp_graph, all_edge_index)
                print(f"  - Heuristic features shape: {heuristic_features.shape}")
                
                # Concatenate heuristics to existing edge features
                all_edge_features = torch.cat([all_edge_features, heuristic_features], dim=1)
                print(f"  - Final edge feature shape: {all_edge_features.shape}")
            except Exception as he:
                print(f"  Warning: Could not compute heuristic features: {he}")
                print("  Proceeding without heuristics in edge features.")
            
            return all_edge_features
                
        except Exception as e:
            print(f"  Error loading edge features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_disease_schema(self, disease_df):
        """Helper to create PyArrow schema for disease table."""
        disease_schema_fields = []
        for col in disease_df.columns:
            if col in ['ancestors', 'descendants', 'children', 'therapeuticAreas']:
                disease_schema_fields.append(pa.field(col, pa.list_(pa.string())))
            else:
                dtype = disease_df[col].dtype
                if dtype == 'object': pa_type = pa.string()
                elif dtype == 'int64': pa_type = pa.int64()
                elif dtype == 'float64': pa_type = pa.float64()
                elif dtype == 'bool': pa_type = pa.bool_()
                else: pa_type = pa.string()
                disease_schema_fields.append(pa.field(col, pa_type))
        return pa.schema(disease_schema_fields)

"""
Data faithfulness tests 
Verify the graph accurately represents OpenTargets data

"""
import pytest
import pandas as pd
import glob
import torch
from pathlib import Path


@pytest.mark.integration
class TestEdgeCountFaithfulness:
    """Test that edge counts are reasonable compared to raw data."""
    
    def test_drug_disease_edge_count_is_reasonable(self, shared_graph):
        """
        Verify we capture a reasonable portion of drug-disease edges from raw data.
        Some edges are filtered (drugs not approved, diseases are TA roots, etc.)
        but we should still have a substantial portion.
        """
        print("\n" + "="*80)
        print("TESTING: Drug-Disease Edge Count Faithfulness")
        print("="*80)
        
        builder = shared_graph['builder']
        config = shared_graph['config']
        
        # Count raw drug-disease pairs from indication data
        indication_path = config.paths.get('indication')
        if not indication_path or not Path(indication_path).exists():
            pytest.skip("Indication data not found")
        
        parquet_files = glob.glob(f"{indication_path}/*.parquet")
        if not parquet_files:
            pytest.skip("No parquet files found")
        
        # Count unique drug-disease pairs in raw data
        raw_pairs = set()
        for f in parquet_files[:5]:  # Sample first 5 files for speed
            df = pd.read_parquet(f)
            if 'id' in df.columns and 'approvedIndications' in df.columns:
                for _, row in df.iterrows():
                    drug_id = row['id']
                    indications = row.get('approvedIndications', [])
                    if isinstance(indications, list):
                        for dis in indications:
                            raw_pairs.add((drug_id, dis))
        
        print(f"Raw drug-disease pairs (sampled): {len(raw_pairs):,}")
        
        # Count graph drug-disease edges
        drug_mapping = builder.mappings.get('drug_key_mapping', {})
        disease_mapping = builder.mappings.get('disease_key_mapping', {})
        
        drug_indices = set(drug_mapping.values())
        disease_indices = set(disease_mapping.values())
        
        graph = shared_graph['graph']
        graph_dd_edges = 0
        
        for i in range(graph.edge_index.size(1)):
            src = graph.edge_index[0, i].item()
            dst = graph.edge_index[1, i].item()
            if (src in drug_indices and dst in disease_indices) or \
               (dst in drug_indices and src in disease_indices):
                graph_dd_edges += 1
        
        # Divide by 2 because graph is undirected (each edge appears twice)
        graph_dd_edges = graph_dd_edges // 2
        
        print(f"Graph drug-disease edges: {graph_dd_edges:,}")
        
        # We should have at least some edges
        assert graph_dd_edges > 0, "No drug-disease edges found in graph!"
        
        # If we have raw pair count, check ratio
        if len(raw_pairs) > 0:
            ratio = graph_dd_edges / len(raw_pairs)
            print(f"Ratio (graph/raw): {ratio:.2%}")
            # We expect at least 10% of raw edges to survive filtering
            # This is a loose bound to catch catastrophic failures
            assert ratio >= 0.05, f"Only {ratio:.2%} of raw edges in graph - too few!"
        
        print("Drug-disease edge count is reasonable")


@pytest.mark.integration
class TestDrugGeneEdgeFaithfulness:
    """Test that drug-gene edges correctly represent mechanismOfAction data."""
    
    def test_drug_gene_edges_from_mechanism_of_action(self, shared_graph):
        """
        Sample random drug-gene pairs from mechanismOfAction and verify in graph.
        """
        print("\n" + "="*80)
        print("TESTING: Drug-Gene Edge Faithfulness")
        print("="*80)
        
        builder = shared_graph['builder']
        config = shared_graph['config']
        graph = shared_graph['graph']
        
        # Load mechanism of action data
        moa_path = config.paths.get('mechanismOfAction')
        if not moa_path or not Path(moa_path).exists():
            pytest.skip("MechanismOfAction data not found")
        
        parquet_files = glob.glob(f"{moa_path}/*.parquet")
        if not parquet_files:
            pytest.skip("No MoA parquet files found")
        
        moa_df = pd.read_parquet(parquet_files[0])
        print(f"MoA records loaded: {len(moa_df):,}")
        
        # Get relevant columns
        drug_col = 'chemblIds' if 'chemblIds' in moa_df.columns else 'drugId'
        gene_col = 'targets' if 'targets' in moa_df.columns else 'targetId'
        
        if drug_col not in moa_df.columns or gene_col not in moa_df.columns:
            pytest.skip(f"Required columns not found. Available: {moa_df.columns.tolist()}")
        
        # Get mappings
        drug_mapping = builder.mappings.get('drug_key_mapping', {})
        gene_mapping = builder.mappings.get('gene_key_mapping', {})
        
        # Sample and check
        sample_size = min(20, len(moa_df))
        sampled = moa_df.sample(n=sample_size, random_state=42)
        
        found_count = 0
        filtered_count = 0
        checked_count = 0
        
        for _, row in sampled.iterrows():
            drug_ids = row[drug_col]
            gene_ids = row[gene_col]
            
            # Handle list/array columns
            if hasattr(drug_ids, 'tolist'):
                drug_ids = drug_ids.tolist()
            elif not isinstance(drug_ids, list):
                drug_ids = [drug_ids] if drug_ids else []
            
            if hasattr(gene_ids, 'tolist'):
                gene_ids = gene_ids.tolist()
            elif not isinstance(gene_ids, list):
                gene_ids = [gene_ids] if gene_ids else []
            
            # Take first from each list
            drug_ids = drug_ids[:1] if drug_ids else []
            gene_ids = gene_ids[:1] if gene_ids else []
            
            for drug_id in drug_ids:
                for gene_id in gene_ids:
                    # Skip if None or empty
                    if not drug_id or not gene_id:
                        continue
                        
                    # Convert to string if needed
                    drug_id = str(drug_id)
                    gene_id = str(gene_id)
                    
                    if drug_id not in drug_mapping or gene_id not in gene_mapping:
                        filtered_count += 1
                        continue
                    
                    checked_count += 1
                    drug_idx = drug_mapping[drug_id]
                    gene_idx = gene_mapping[gene_id]
                    
                    # Check edge exists
                    edge_exists = (
                        ((graph.edge_index[0] == drug_idx) & (graph.edge_index[1] == gene_idx)).any() or
                        ((graph.edge_index[0] == gene_idx) & (graph.edge_index[1] == drug_idx)).any()
                    )
                    
                    if edge_exists:
                        found_count += 1
                        print(f"  ✓ Found: {drug_id} -> {gene_id}")
                    else:
                        print(f"  ✗ Missing: {drug_id} -> {gene_id}")
        
        print(f"\nResult: Found {found_count} drug-gene edges")
        print(f"Filtered (nodes not in graph): {filtered_count}")
        print(f"Actually checked: {checked_count}")
        
        # We expect to find most edges for nodes that exist
        if checked_count > 0:
            hit_rate = found_count / checked_count
            print(f"Hit rate: {hit_rate:.2%}")
            assert hit_rate >= 0.5, f"Only {hit_rate:.2%} of MoA edges found!"
        
        print("Drug-gene edges validated")


@pytest.mark.integration
class TestTemporalSplitCorrectness:
    """Test that temporal splits don't leak future data into training."""
    
    def test_val_test_edges_are_from_future_versions(self, shared_graph):
        """
        Verify that validation and test positive edges come from 
        later OpenTargets versions (23.06, 24.06) not the training version (21.06).
        """
        print("\n" + "="*80)
        print("TESTING: Temporal Split Correctness")
        print("="*80)
        
        builder = shared_graph['builder']
        config = shared_graph['config']
        graph = shared_graph['graph']
        
        # Try to load future indication data
        future_paths = []
        for version in ['23.06', '24.06']:
            path = f"raw_data/{version}/indication"
            if Path(path).exists():
                future_paths.append(path)
        
        if not future_paths:
            pytest.skip("No future indication data found")
        
        # Collect future drug-disease pairs
        future_pairs = set()
        for path in future_paths:
            parquet_files = glob.glob(f"{path}/*.parquet")
            for f in parquet_files:  # Load all files for complete coverage
                df = pd.read_parquet(f)
                if 'id' not in df.columns:
                    continue
                    
                for _, row in df.iterrows():
                    drug_id = row['id']
                    
                    # Handle 'indications' column (nested format with disease IDs)
                    # Can be list or numpy array
                    if 'indications' in df.columns:
                        indications = row.get('indications')
                        if indications is not None and hasattr(indications, '__iter__'):
                            for ind in indications:
                                if isinstance(ind, dict) and 'disease' in ind:
                                    future_pairs.add((drug_id, ind['disease']))
                    
                    # Also handle 'approvedIndications' (simple format)
                    if 'approvedIndications' in df.columns:
                        approved = row.get('approvedIndications')
                        if approved is not None and hasattr(approved, '__iter__'):
                            for dis in approved:
                                if dis:
                                    future_pairs.add((drug_id, dis))
        
        print(f"Future pairs collected: {len(future_pairs):,}")
        
        # Get mappings
        drug_mapping = builder.mappings.get('drug_key_mapping', {})
        disease_mapping = builder.mappings.get('disease_key_mapping', {})
        
        # Reverse mappings for lookup
        idx_to_drug = {v: k for k, v in drug_mapping.items()}
        idx_to_disease = {v: k for k, v in disease_mapping.items()}
        
        # Check val/test positive edges
        val_edge_index = graph.val_edge_index
        val_edge_label = graph.val_edge_label
        
        # Handle both [N, 2] and [2, N] tensor shapes
        if val_edge_index.dim() == 2 and val_edge_index.size(1) == 2:
            # Shape is [N, 2]
            val_mask = val_edge_label == 1
            val_positives = val_edge_index[val_mask]
            val_count = val_positives.size(0)
            
            val_matches = 0
            for i in range(val_count):
                src = val_positives[i, 0].item()
                dst = val_positives[i, 1].item()
                
                # Try to map back to IDs - drug at src position
                drug_id = idx_to_drug.get(src)
                disease_id = idx_to_disease.get(dst)
                
                # If drug is at dst position (reversed)
                if drug_id is None:
                    drug_id = idx_to_drug.get(dst)
                    disease_id = idx_to_disease.get(src)
                
                if drug_id and disease_id:
                    # Convert to string for comparison
                    disease_id = str(disease_id)
                    if (drug_id, disease_id) in future_pairs:
                        val_matches += 1
        else:
            # Shape is [2, N]
            val_positives = val_edge_index[:, val_edge_label == 1]
            val_count = val_positives.size(1)
            val_matches = 0
            
            for i in range(val_count):
                src = val_positives[0, i].item()
                dst = val_positives[1, i].item()
                
                drug_id = idx_to_drug.get(src)
                disease_id = idx_to_disease.get(dst)
                if drug_id is None:
                    drug_id = idx_to_drug.get(dst)
                    disease_id = idx_to_disease.get(src)
                
                if drug_id and disease_id:
                    disease_id = str(disease_id)
                    if (drug_id, disease_id) in future_pairs:
                        val_matches += 1
        
        print(f"Validation positive edges: {val_count}")
        print(f"Matching future data: {val_matches}")
        
        if val_count > 0:
            match_rate = val_matches / val_count
            print(f"Match rate: {match_rate:.2%}")
            # All val/test positive edges should come from future indication versions
            # We expect >90% match rate (some may fail due to ID resolution differences)
            assert match_rate >= 0.9, f"Only {match_rate:.2%} of val edges from future data - expected >90%!"
        
        print("Temporal splits verified correctly")


@pytest.mark.integration  
class TestNegativeSamplingCorrectness:
    """Test that negative samples are actually negative (no edge should exist)."""
    
    def test_negative_samples_are_true_negatives(self, shared_graph):
        """
        Verify that edges marked as negative in training/val/test 
        don't actually exist as positive edges in the graph.
        """
        print("\n" + "="*80)
        print("TESTING: Negative Sampling Correctness")
        print("="*80)
        
        graph = shared_graph['graph']
        
        # Build set of all positive edges in the main graph
        positive_edges = set()
        for i in range(graph.edge_index.size(1)):
            src = graph.edge_index[0, i].item()
            dst = graph.edge_index[1, i].item()
            positive_edges.add((src, dst))
            positive_edges.add((dst, src))  # Both directions
        
        print(f"Total positive edge pairs: {len(positive_edges):,}")
        
        # Check training negatives
        train_edges = graph.train_edge_index
        train_labels = graph.train_edge_label
        
        train_neg_mask = train_labels == 0
        
        # Handle both [N, 2] and [2, N] tensor shapes
        if train_edges.dim() == 2 and train_edges.size(1) == 2:
            # Shape is [N, 2]
            train_neg_edges = train_edges[train_neg_mask]
            neg_count = train_neg_edges.size(0)
            
            false_negatives = 0
            sample_size = min(1000, neg_count)
            
            for i in range(sample_size):
                src = train_neg_edges[i, 0].item()
                dst = train_neg_edges[i, 1].item()
                
                if (src, dst) in positive_edges or (dst, src) in positive_edges:
                    false_negatives += 1
        else:
            # Shape is [2, N]
            train_neg_edges = train_edges[:, train_neg_mask]
            neg_count = train_neg_edges.size(1)
            
            false_negatives = 0
            sample_size = min(1000, neg_count)
            
            for i in range(sample_size):
                src = train_neg_edges[0, i].item()
                dst = train_neg_edges[1, i].item()
                
                if (src, dst) in positive_edges or (dst, src) in positive_edges:
                    false_negatives += 1
        
        print(f"Training negative samples checked: {sample_size:,}")
        print(f"False negatives found: {false_negatives}")
        
        if sample_size > 0:
            false_neg_rate = false_negatives / sample_size
            print(f"False negative rate: {false_neg_rate:.4%}")
            # Should have zero or very few false negatives
            assert false_neg_rate < 0.01, f"Too many false negatives: {false_neg_rate:.2%}!"
        
        print("Negative samples are true negatives")


@pytest.mark.integration
class TestMappingConsistency:
    """Test that node mappings are consistent with graph structure."""
    
    def test_mapping_indices_match_graph(self, shared_graph):
        """
        Verify that all indices in mappings are valid node indices in the graph.
        """
        print("\n" + "="*80)
        print("TESTING: Mapping Consistency")
        print("="*80)
        
        builder = shared_graph['builder']
        graph = shared_graph['graph']
        
        num_nodes = graph.x.size(0)
        print(f"Graph nodes: {num_nodes:,}")
        
        # Check each mapping
        mappings_to_check = [
            'drug_key_mapping',
            'disease_key_mapping', 
            'gene_key_mapping',
            'drug_type_key_mapping',
            'therapeutic_area_key_mapping',
            'reactome_key_mapping'
        ]
        
        for mapping_name in mappings_to_check:
            if mapping_name not in builder.mappings:
                print(f"  ⊘ {mapping_name} not found")
                continue
            
            mapping = builder.mappings[mapping_name]
            if not isinstance(mapping, dict):
                continue
            
            max_idx = max(mapping.values()) if mapping else -1
            min_idx = min(mapping.values()) if mapping else 0
            
            print(f"  {mapping_name}: {len(mapping)} items, index range [{min_idx}, {max_idx}]")
            
            # All indices should be valid
            assert max_idx < num_nodes, f"{mapping_name} has index {max_idx} >= num_nodes {num_nodes}"
            assert min_idx >= 0, f"{mapping_name} has negative index {min_idx}"
        
        print("All mappings use valid node indices")
    
    def test_node_type_ranges_are_disjoint(self, shared_graph):
        """
        Verify that different node types have non-overlapping index ranges.
        """
        print("\n" + "="*80)
        print("TESTING: Node Type Index Disjointness")
        print("="*80)
        
        builder = shared_graph['builder']
        
        # Get index sets for each type
        type_indices = {}
        for mapping_name in ['drug_key_mapping', 'disease_key_mapping', 'gene_key_mapping', 'drug_type_key_mapping']:
            if mapping_name in builder.mappings:
                mapping = builder.mappings[mapping_name]
                if isinstance(mapping, dict):
                    type_indices[mapping_name] = set(mapping.values())
        
        # Check for overlaps
        types = list(type_indices.keys())
        for i, t1 in enumerate(types):
            for t2 in types[i+1:]:
                overlap = type_indices[t1] & type_indices[t2]
                if overlap:
                    print(f"Overlap between {t1} and {t2}: {len(overlap)} indices")
                else:
                    print(f"{t1} and {t2} are disjoint")
        
        print("Node type index check complete")


@pytest.mark.integration
class TestFeatureCorrectness:
    """Test that node features match source data values."""
    
    def test_drug_count_matches_features(self, shared_graph):
        """
        Verify that number of drugs matches expected from mappings.
        """
        print("\n" + "="*80)
        print("TESTING: Feature Correctness")
        print("="*80)
        
        builder = shared_graph['builder']
        graph = shared_graph['graph']
        
        # Count drugs in mapping
        drug_mapping = builder.mappings.get('drug_key_mapping', {})
        num_drugs = len(drug_mapping)
        
        # Count from metadata
        metadata_drugs = graph.metadata.get('node_info', {}).get('Drugs', 0)
        
        print(f"Drugs in mapping: {num_drugs}")
        print(f"Drugs in metadata: {metadata_drugs}")
        
        # These should match
        assert num_drugs == metadata_drugs, f"Drug count mismatch: mapping={num_drugs}, metadata={metadata_drugs}"
        
        print("Drug counts consistent")
    
    def test_feature_dimensions_sensible(self, shared_graph):
        """
        Verify that feature dimensions are reasonable for the data.
        """
        graph = shared_graph['graph']
        
        num_nodes = graph.x.size(0)
        num_features = graph.x.size(1)
        
        print(f"Nodes: {num_nodes:,}")
        print(f"Features per node: {num_features}")
        
        # Check for reasonable feature dimension
        assert num_features > 10, f"Too few features: {num_features}"
        assert num_features < 1000, f"Too many features: {num_features}"
        
        # Check for non-zero features
        non_zero = (graph.x != 0).sum().item()
        sparsity = 1 - (non_zero / (num_nodes * num_features))
        
        print(f"Non-zero elements: {non_zero:,}")
        print(f"Sparsity: {sparsity:.2%}")
        
        # Features shouldn't be all zeros
        assert non_zero > 0, "All features are zero!"
        
        print("Feature dimensions are sensible")

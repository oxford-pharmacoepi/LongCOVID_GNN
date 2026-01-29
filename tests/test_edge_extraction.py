"""
Integration tests for edge extraction with real data validation.
"""
import pytest
import pandas as pd
import glob
from pathlib import Path


@pytest.mark.integration
class TestEdgeExtractionFromRawData:
    """Test edge extraction against actual raw data files."""
    
    def test_known_edges_from_raw_data(self, shared_graph):
        """
        Automated validation: Sample random drug-disease pairs from raw parquet files
        and verify they exist in the constructed graph.
        
        NOTE: Not all edges from raw data will be in the graph due to filtering:
        - Drugs must be in approved list or have Phase 3+ trials
        - Diseases must not be therapeutic area roots
        - Genes must be protein-coding
        This test verifies the pipeline is working, not that all edges are preserved.
        """
        print("\n" + "="*80)
        print("TESTING: Known edges from raw data")
        print("="*80)
        
        graph = shared_graph['graph']
        builder = shared_graph['builder']
        config = shared_graph['config']
        
        # Load indication data from raw files
        indication_path = config.paths.get('indication')
        if not indication_path or not Path(indication_path).exists():
            pytest.skip("Indication data not found")
        
        parquet_files = glob.glob(f"{indication_path}/*.parquet")
        if not parquet_files:
            pytest.skip("No parquet files found")
        
        # Read a sample of indication data
        indication_df = pd.read_parquet(parquet_files[0])
        
        # Check if we have the right columns
        if 'diseaseFromSource' not in indication_df.columns:
            # Try knownDrugsAggregated format
            known_drugs_path = config.paths.get('knownDrugsAggregated')
            if known_drugs_path and Path(known_drugs_path).exists():
                parquet_files = glob.glob(f"{known_drugs_path}/*.parquet")
                if parquet_files:
                    indication_df = pd.read_parquet(parquet_files[0])
        
        if 'diseaseFromSource' not in indication_df.columns and 'diseaseId' not in indication_df.columns:
            pytest.skip("Cannot find suitable drug-disease data")
        
        # Sample 10 random drug-disease pairs
        disease_col = 'diseaseFromSource' if 'diseaseFromSource' in indication_df.columns else 'diseaseId'
        drug_col = 'id' if 'id' in indication_df.columns else 'drugId'
        
        sample_size = min(10, len(indication_df))
        sampled_pairs = indication_df.sample(n=sample_size, random_state=42)[[drug_col, disease_col]]
        
        print(f"\nSampled {sample_size} drug-disease pairs from raw data:")
        for idx, row in sampled_pairs.iterrows():
            print(f"  - {row[drug_col]} -> {row[disease_col]}")
        
        # Get mappings
        drug_mapping = builder.mappings.get('drug_key_mapping', {})
        disease_mapping = builder.mappings.get('disease_key_mapping', {})
        
        # Check if sampled pairs exist in graph
        found_count = 0
        filtered_count = 0
        
        for _, row in sampled_pairs.iterrows():
            drug_id = row[drug_col]
            disease_id = row[disease_col]
            
            # Check if nodes exist in mappings (survived filtering)
            drug_in_graph = drug_id in drug_mapping
            disease_in_graph = disease_id in disease_mapping
            
            if not drug_in_graph or not disease_in_graph:
                filtered_count += 1
                reason = []
                if not drug_in_graph:
                    reason.append("drug filtered")
                if not disease_in_graph:
                    reason.append("disease filtered")
                print(f"  ⊘ Filtered out: {drug_id} -> {disease_id} ({', '.join(reason)})")
                continue
            
            drug_idx = drug_mapping[drug_id]
            disease_idx = disease_mapping[disease_id]
            
            # Check if edge exists
            edge_exists = (
                ((graph.edge_index[0] == drug_idx) & (graph.edge_index[1] == disease_idx)).any() or
                ((graph.edge_index[0] == disease_idx) & (graph.edge_index[1] == drug_idx)).any()
            )
            if edge_exists:
                found_count += 1
                print(f"Found edge: {drug_id} -> {disease_id}")
            else:
                print(f"Nodes exist but edge missing: {drug_id} -> {disease_id}")
        
        print(f"\nResult: Found {found_count}/{sample_size} sampled edges in graph")
        print(f"Filtered: {filtered_count}/{sample_size} (nodes removed during filtering)")
        print(f"Expected in graph: {sample_size - filtered_count}")
        
        # Adjusted expectation: we expect to find edges for nodes that survived filtering
        expected_in_graph = sample_size - filtered_count
        if expected_in_graph > 0:
            # We should find most edges for nodes that survived filtering
            assert found_count >= expected_in_graph * 0.7, \
                f"Only found {found_count}/{expected_in_graph} edges for nodes in graph"
        else:
            print("All sampled edges were filtered out - this is expected behavior")
    
    def test_self_loop_removal(self, shared_graph):
        """Ensure no self-loops exist in any edge type."""
        graph = shared_graph['graph']
        
        # Check for self-loops
        self_loops = (graph.edge_index[0] == graph.edge_index[1]).sum().item()
        
        print(f"\nTotal edges: {graph.edge_index.size(1):,}")
        print(f"Self-loops: {self_loops}")
        
        assert self_loops == 0, f"Found {self_loops} self-loops"
        print("No self-loops found")

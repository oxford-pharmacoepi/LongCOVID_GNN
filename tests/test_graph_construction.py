"""
Integration tests for graph construction.
"""
import pytest


@pytest.mark.integration
class TestGraphConstruction:
    """Test graph construction end-to-end."""
    
    def test_graph_builder_creates_valid_graph(self, shared_graph):
        """Test that GraphBuilder creates a valid graph structure."""
        graph = shared_graph['graph']
        
        # Check graph structure
        assert graph.x is not None
        assert graph.edge_index is not None
        assert graph.x.size(0) > 0
        assert graph.edge_index.size(1) > 0
        
        print(f"\nGraph created: {graph.x.size(0):,} nodes, {graph.edge_index.size(1):,} edges")
    
    def test_train_val_test_splits_exist(self, shared_graph):
        """Test that train/val/test splits exist and have no overlap."""
        graph = shared_graph['graph']
        
        # Check splits exist
        assert hasattr(graph, 'train_edge_index')
        assert hasattr(graph, 'val_edge_index')
        assert hasattr(graph, 'test_edge_index')
        
        train_size = graph.train_edge_index.size(1)
        val_size = graph.val_edge_index.size(1)
        test_size = graph.test_edge_index.size(1)
        
        print(f"\nSplit sizes - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")
        
        # Check no overlap between splits
        train_set = set((graph.train_edge_index[0, i].item(), graph.train_edge_index[1, i].item())
                       for i in range(train_size))
        val_set = set((graph.val_edge_index[0, i].item(), graph.val_edge_index[1, i].item())
                     for i in range(val_size))
        test_set = set((graph.test_edge_index[0, i].item(), graph.test_edge_index[1, i].item())
                      for i in range(test_size))
        
        assert len(train_set & val_set) == 0, "Train and val sets overlap"
        assert len(train_set & test_set) == 0, "Train and test sets overlap"
        assert len(val_set & test_set) == 0, "Val and test sets overlap"
        
        print("No overlap between train/val/test splits")
    
    def test_graph_metadata_present(self, shared_graph):
        """Test that graph metadata is populated."""
        graph = shared_graph['graph']
        
        assert hasattr(graph, 'metadata')
        assert 'node_info' in graph.metadata
        assert 'edge_info' in graph.metadata
        assert 'total_nodes' in graph.metadata
        assert 'total_edges' in graph.metadata
        
        print(f"\nMetadata present: {graph.metadata['total_nodes']:,} nodes, {graph.metadata['total_edges']:,} edges")

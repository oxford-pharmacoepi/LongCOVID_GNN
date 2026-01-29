"""
Integration tests - Simplified to work with processed data
"""
import pytest
import torch
from pathlib import Path
from src.config import get_config


@pytest.mark.integration
class TestProcessedGraphValidation:
    """Test validation of pre-processed graph data."""
    
    def test_processed_graph_loads(self):
        """Test that processed graph data can be loaded."""
        config = get_config()
        
        # Check if processed data exists
        processed_path = Path("processed_data")
        if not processed_path.exists():
            pytest.skip("No processed data found - run scripts/1_create_graph.py first")
        
        # Load processed graph
        graph_path = processed_path / "graph.pt"
        if not graph_path.exists():
            pytest.skip("No graph.pt found - run scripts/1_create_graph.py first")
        
        graph = torch.load(graph_path, weights_only=False)
        
        # Basic validation
        assert hasattr(graph, 'x'), "Graph missing node features"
        assert hasattr(graph, 'edge_index'), "Graph missing edge index"
        assert graph.x.size(0) > 0, "No nodes in graph"
        assert graph.edge_index.size(1) > 0, "No edges in graph"
        
        print(f"Loaded graph: {graph.x.size(0):,} nodes, {graph.edge_index.size(1):,} edges")
    
    def test_no_self_loops(self):
        """Test that graph has no self-loops."""
        processed_path = Path("processed_data/graph.pt")
        if not processed_path.exists():
            pytest.skip("No processed graph found")
        
        graph = torch.load(processed_path, weights_only=False)
        
        self_loops = (graph.edge_index[0] == graph.edge_index[1]).sum().item()
        
        print(f"Total edges: {graph.edge_index.size(1):,}")
        print(f"Self-loops: {self_loops}")
        
        assert self_loops == 0, f"Found {self_loops} self-loops"
        print("No self-loops found")
    
    def test_edge_indices_valid(self):
        """Test that all edge indices are within valid range."""
        processed_path = Path("processed_data/graph.pt")
        if not processed_path.exists():
            pytest.skip("No processed graph found")
        
        graph = torch.load(processed_path, weights_only=False)
        
        num_nodes = graph.x.size(0)
        max_idx = graph.edge_index.max().item()
        min_idx = graph.edge_index.min().item()
        
        print(f"Nodes: {num_nodes:,}")
        print(f"Edge index range: [{min_idx}, {max_idx}]")
        
        assert min_idx >= 0, "Negative edge indices found"
        assert max_idx < num_nodes, f"Edge index {max_idx} >= num_nodes {num_nodes}"
        
        print("All edge indices valid")
    
    def test_graph_is_undirected(self):
        """Test that graph is undirected (symmetric edges)."""
        processed_path = Path("processed_data/graph.pt")
        if not processed_path.exists():
            pytest.skip("No processed graph found")
        
        graph = torch.load(processed_path, weights_only=False)
        
        # Build edge set
        edges = set()
        for i in range(graph.edge_index.size(1)):
            edges.add((graph.edge_index[0, i].item(), graph.edge_index[1, i].item()))
        
        # Check symmetry
        asymmetric = 0
        for u, v in list(edges)[:1000]:  # Sample 1000 edges
            if (v, u) not in edges:
                asymmetric += 1
        
        print(f"Checked 1000 edges for symmetry")
        print(f"Asymmetric: {asymmetric}")
        
        assert asymmetric == 0, f"Found {asymmetric} asymmetric edges"
        print("Graph is undirected")
    
    def test_splits_exist(self):
        """Test that train/val/test splits exist."""
        processed_path = Path("processed_data/graph.pt")
        if not processed_path.exists():
            pytest.skip("No processed graph found")
        
        graph = torch.load(processed_path, weights_only=False)
        
        assert hasattr(graph, 'train_edge_index'), "Missing train split"
        assert hasattr(graph, 'val_edge_index'), "Missing val split"
        assert hasattr(graph, 'test_edge_index'), "Missing test split"
        
        train_size = graph.train_edge_index.size(1)
        val_size = graph.val_edge_index.size(1)
        test_size = graph.test_edge_index.size(1)
        
        print(f"Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")
        print("All splits present")

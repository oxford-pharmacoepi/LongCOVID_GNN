"""
Data validation tests 
Testing graph invariants and properties.

"""
import pytest


@pytest.mark.integration
class TestDataInvariants:
    """Test that graph data satisfies required invariants."""
    
    def test_all_edge_indices_within_node_range(self, shared_graph):
        """Test that all edge indices are valid (< num_nodes)."""
        graph = shared_graph['graph']
        num_nodes = graph.x.size(0)
        
        # Check all edge indices
        assert graph.edge_index.max() < num_nodes, "Edge indices out of range"
        assert graph.edge_index.min() >= 0, "Negative edge indices found"
        
        print(f"\nAll edge indices in valid range [0, {num_nodes})")
    
    def test_no_duplicate_edges(self, shared_graph):
        """Test that there are no duplicate edges in the graph."""
        graph = shared_graph['graph']
        
        # Convert edges to set
        edges = set()
        duplicates = 0
        
        for i in range(graph.edge_index.size(1)):
            edge = (graph.edge_index[0, i].item(), graph.edge_index[1, i].item())
            if edge in edges:
                duplicates += 1
            edges.add(edge)
        
        print(f"\nTotal edges: {graph.edge_index.size(1):,}")
        print(f"Unique edges: {len(edges):,}")
        print(f"Duplicates: {duplicates}")
        
        assert duplicates == 0, f"Found {duplicates} duplicate edges"
        print("No duplicate edges found")
    
    def test_bidirectional_edges_symmetric(self, shared_graph):
        """Test that if edge (u,v) exists, edge (v,u) also exists (for undirected graph)."""
        graph = shared_graph['graph']
        
        # Build edge set
        edges = set()
        for i in range(graph.edge_index.size(1)):
            edges.add((graph.edge_index[0, i].item(), graph.edge_index[1, i].item()))
        
        # Check symmetry
        asymmetric = 0
        for u, v in edges:
            if (v, u) not in edges:
                asymmetric += 1
        
        print(f"\nTotal edges: {len(edges):,}")
        print(f"Asymmetric edges: {asymmetric}")
        
        # Graph should be symmetric after ToUndirected()
        assert asymmetric == 0, f"Found {asymmetric} asymmetric edges in undirected graph"
        print("Graph is fully symmetric")
    
    def test_feature_dimensions_match(self, shared_graph):
        """Test that node features match node count."""
        graph = shared_graph['graph']
        
        num_nodes = graph.edge_index.max().item() + 1
        feature_rows = graph.x.size(0)
        
        print(f"\nMax node index: {num_nodes - 1}")
        print(f"Feature matrix rows: {feature_rows}")
        
        assert feature_rows >= num_nodes, "Feature matrix too small for graph"
        print("Feature dimensions match graph structure")
    
    def test_edge_features_present(self, shared_graph):
        """Test that edge features exist and match edge count."""
        graph = shared_graph['graph']
        
        assert hasattr(graph, 'edge_attr'), "Graph missing edge features"
        assert graph.edge_attr is not None, "Edge features are None"
        
        num_edges = graph.edge_index.size(1)
        num_edge_features = graph.edge_attr.size(0)
        
        print(f"\nEdges: {num_edges:,}")
        print(f"Edge features: {num_edge_features:,}")
        print(f"Feature dimension: {graph.edge_attr.size(1)}")
        
        assert num_edge_features == num_edges, f"Edge feature count mismatch: {num_edge_features} != {num_edges}"
        print("Edge features match edge count")

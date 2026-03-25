"""
Expanded tests for src/models_seal.py — SEALDataset class.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from torch_geometric.data import Data

from src.models_seal import SEALDataset, extract_enclosing_subgraph, MAX_Z


@pytest.fixture
def graph_data():
    """10-node connected graph."""
    torch.manual_seed(42)
    edge_list = []
    for i in range(9):
        edge_list += [[i, i + 1], [i + 1, i]]
    edge_list += [[0, 5], [5, 0], [3, 7], [7, 3]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    node_features = torch.randn(10, 8)
    return edge_index, node_features


class TestSEALDataset:
    def test_len(self, graph_data):
        edge_index, node_features = graph_data
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SEALDataset(
                root=tmpdir,
                pairs=[(0, 1), (2, 5), (4, 9)],
                labels=[1, 0, 1],
                edge_index=edge_index,
                node_features=node_features,
                num_hops=1,
                use_cache=False,
                save_cache=False,
            )
            assert len(ds) == 3

    def test_getitem_returns_data(self, graph_data):
        edge_index, node_features = graph_data
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SEALDataset(
                root=tmpdir,
                pairs=[(0, 1)],
                labels=[1],
                edge_index=edge_index,
                node_features=node_features,
                num_hops=2,
                use_cache=False,
                save_cache=False,
            )
            data = ds[0]
            assert isinstance(data, Data)
            assert data.x is not None
            assert data.y is not None
            assert data.y.item() == 1.0

    def test_feature_dim(self, graph_data):
        edge_index, node_features = graph_data
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SEALDataset(
                root=tmpdir,
                pairs=[(0, 1)],
                labels=[1],
                edge_index=edge_index,
                node_features=node_features,
                num_hops=2,
                use_cache=False,
                save_cache=False,
            )
            data = ds[0]
            # Feature dim = node_features (8) + z_one_hot (MAX_Z)
            assert data.x.shape[1] == node_features.shape[1] + MAX_Z

    def test_caching(self, graph_data):
        edge_index, node_features = graph_data
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SEALDataset(
                root=tmpdir,
                pairs=[(0, 1), (2, 3)],
                labels=[1, 0],
                edge_index=edge_index,
                node_features=node_features,
                num_hops=2,
                use_cache=True,
                save_cache=True,
            )
            # First access creates cache
            data1 = ds[0]
            # Check cache file exists
            cache_files = list(Path(tmpdir).rglob("subgraph_*.pt"))
            assert len(cache_files) >= 1

            # Second access should use cache
            data2 = ds[0]
            assert data2.x is not None

    def test_no_node_features(self, graph_data):
        """When node_features=None, x should be z_one_hot only."""
        edge_index, _ = graph_data
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SEALDataset(
                root=tmpdir,
                pairs=[(0, 1)],
                labels=[1],
                edge_index=edge_index,
                node_features=None,
                num_hops=2,
                use_cache=False,
                save_cache=False,
            )
            data = ds[0]
            assert data.x.shape[1] == MAX_Z

    def test_batch_loading(self, graph_data):
        """Test that SEALDataset works with PyG DataLoader."""
        from torch_geometric.loader import DataLoader
        edge_index, node_features = graph_data
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SEALDataset(
                root=tmpdir,
                pairs=[(0, 1), (2, 5), (4, 9)],
                labels=[1, 0, 1],
                edge_index=edge_index,
                node_features=node_features,
                num_hops=2,
                use_cache=False,
                save_cache=False,
            )
            loader = DataLoader(ds, batch_size=3)
            batch = next(iter(loader))
            assert batch.x is not None
            assert batch.y.shape[0] == 3

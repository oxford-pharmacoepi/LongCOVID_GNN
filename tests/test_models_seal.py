"""
Unit tests for src/models_seal.py

Tests DRNL labeling, subgraph extraction, and SEALModel forward pass.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch

from src.models_seal import (
    drnl_node_labeling,
    extract_enclosing_subgraph,
    SEALModel,
    MAX_Z,
)


# ── fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def simple_graph():
    """6-node path graph: 0-1-2-3-4-5 with back-edges."""
    edge_list = []
    for i in range(5):
        edge_list += [[i, i + 1], [i + 1, i]]
    # Add cross-edge
    edge_list += [[0, 3], [3, 0]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.randn(6, 8)
    return Data(x=x, edge_index=edge_index, num_nodes=6)


@pytest.fixture
def triangle_graph():
    """3-node complete graph (triangle)."""
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2],
         [1, 2, 0, 2, 0, 1]], dtype=torch.long
    )
    x = torch.randn(3, 8)
    return Data(x=x, edge_index=edge_index, num_nodes=3)


# ── drnl_node_labeling ──────────────────────────────────────────────
# Signature: drnl_node_labeling(num_nodes, src_node, dst_node, edge_index, max_z=50)
class TestDRNLNodeLabeling:
    def test_source_target_labeled_one(self, simple_graph):
        """Source and target should get label 1."""
        labels = drnl_node_labeling(6, 0, 1, simple_graph.edge_index)
        assert labels[0] == 1
        assert labels[1] == 1

    def test_output_length(self, simple_graph):
        labels = drnl_node_labeling(6, 0, 1, simple_graph.edge_index)
        assert len(labels) == 6

    def test_labels_non_negative(self, simple_graph):
        labels = drnl_node_labeling(6, 0, 5, simple_graph.edge_index)
        assert all(l >= 0 for l in labels)

    def test_triangle(self, triangle_graph):
        labels = drnl_node_labeling(3, 0, 1, triangle_graph.edge_index)
        assert labels[0] == 1
        assert labels[1] == 1

    def test_max_z_parameter(self, simple_graph):
        labels = drnl_node_labeling(6, 0, 1, simple_graph.edge_index, max_z=10)
        assert labels.shape[0] == 6


# ── extract_enclosing_subgraph ──────────────────────────────────────
# Returns (Data, rel_src, rel_dst) — Data has z, nodes, edge_index but NOT x
class TestExtractEnclosingSubgraph:
    def test_returns_tuple(self, simple_graph):
        result = extract_enclosing_subgraph(
            0, 1, simple_graph.edge_index,
            node_features=simple_graph.x, num_hops=2
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        sub_data, rel_src, rel_dst = result
        assert isinstance(sub_data, Data)

    def test_has_z_and_nodes(self, simple_graph):
        sub_data, _, _ = extract_enclosing_subgraph(
            0, 1, simple_graph.edge_index,
            node_features=simple_graph.x, num_hops=2
        )
        assert hasattr(sub_data, 'z')
        assert hasattr(sub_data, 'nodes')
        assert sub_data.z.shape[0] == sub_data.num_nodes

    def test_subgraph_smaller(self, simple_graph):
        sub_data, _, _ = extract_enclosing_subgraph(
            0, 1, simple_graph.edge_index,
            node_features=simple_graph.x, num_hops=1
        )
        assert sub_data.num_nodes <= simple_graph.num_nodes

    def test_rel_indices_valid(self, simple_graph):
        sub_data, rel_src, rel_dst = extract_enclosing_subgraph(
            0, 1, simple_graph.edge_index,
            node_features=simple_graph.x, num_hops=2
        )
        assert 0 <= rel_src < sub_data.num_nodes
        assert 0 <= rel_dst < sub_data.num_nodes


# ── SEALModel ────────────────────────────────────────────────────────
class TestSEALModel:
    def _make_batch(self, simple_graph, node_features):
        """Create a small batch with full features for SEAL."""
        batch_data = []
        for src, dst in [(0, 1), (0, 5), (2, 3)]:
            sub_data, _, _ = extract_enclosing_subgraph(
                src, dst, simple_graph.edge_index,
                node_features=node_features, num_hops=2
            )
            # Construct x from z + real features (mimic SEALDataset behaviour)
            n = sub_data.num_nodes
            z_one_hot = torch.nn.functional.one_hot(
                sub_data.z.clamp(0, MAX_Z), num_classes=MAX_Z + 1
            ).float()
            # Use node features from parent graph
            x_feat = node_features[sub_data.nodes]
            sub_data.x = torch.cat([z_one_hot, x_feat], dim=1)
            batch_data.append(sub_data)
        return Batch.from_data_list(batch_data)

    def test_forward_shape(self, simple_graph):
        batch = self._make_batch(simple_graph, simple_graph.x)
        in_channels = batch.x.size(1)  # z_onehot (51) + features (8) = 59
        model = SEALModel(
            in_channels=in_channels,
            hidden_channels=16,
            num_layers=2,
            pooling='mean',  # Use mean pooling for simplicity
        )
        model.eval()
        out = model(batch)
        assert out.dim() == 1
        assert out.shape[0] == 3

    def test_all_conv_types(self, simple_graph):
        batch = self._make_batch(simple_graph, simple_graph.x)
        in_channels = batch.x.size(1)
        for conv_type in ['gcn', 'sage', 'gin']:
            model = SEALModel(
                in_channels=in_channels,
                hidden_channels=16,
                num_layers=2,
                conv_type=conv_type,
                pooling='mean',
            )
            model.eval()
            out = model(batch)
            assert out.shape[0] == 3, f"Failed for conv_type={conv_type}"

    def test_pool_types(self, simple_graph):
        batch = self._make_batch(simple_graph, simple_graph.x)
        in_channels = batch.x.size(1)
        for pooling in ['mean', 'mean+max']:
            model = SEALModel(
                in_channels=in_channels,
                hidden_channels=16,
                num_layers=2,
                pooling=pooling,
            )
            model.eval()
            out = model(batch)
            assert out.shape[0] == 3, f"Failed for pooling={pooling}"

    def test_gradient_flow(self, simple_graph):
        batch = self._make_batch(simple_graph, simple_graph.x)
        in_channels = batch.x.size(1)
        model = SEALModel(
            in_channels=in_channels,
            hidden_channels=16,
            num_layers=2,
            pooling='mean',
        )
        out = model(batch)
        loss = out.sum()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_sort_pooling(self, simple_graph):
        batch = self._make_batch(simple_graph, simple_graph.x)
        in_channels = batch.x.size(1)
        model = SEALModel(
            in_channels=in_channels,
            hidden_channels=16,
            num_layers=2,
            pooling='sort',
            k=5,
        )
        model.eval()
        out = model(batch)
        assert out.shape[0] == 3

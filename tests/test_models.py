"""
Unit tests for src/models.py

Tests GNN encoders and LinkPredictor decoder with synthetic graphs.
"""

import pytest
import torch
from torch_geometric.data import Data

from src.models import (
    GCNModel,
    TransformerModel,
    SAGEModel,
    GATModel,
    LinkPredictor,
    MODEL_CLASSES,
)


# ── fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def small_graph():
    """Small synthetic graph for forward-pass tests."""
    torch.manual_seed(42)
    num_nodes = 20
    num_features = 16
    x = torch.randn(num_nodes, num_features)
    # Create a simple connected graph
    edge_list = []
    for i in range(num_nodes - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    # Add a few extra edges
    edge_list += [[0, 5], [5, 0], [3, 10], [10, 3]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return x, edge_index


@pytest.fixture
def edge_pairs():
    """Edge pairs for link prediction (shape [2, num_edges])."""
    return torch.tensor([[0, 1, 2, 3], [5, 10, 15, 19]], dtype=torch.long)


# ── GCNModel ─────────────────────────────────────────────────────────
class TestGCNModel:
    def test_forward_shape(self, small_graph):
        x, edge_index = small_graph
        model = GCNModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2, dropout_rate=0.1
        )
        model.eval()
        out = model(x, edge_index)
        assert out.shape == (20, 16)

    def test_gradient_flow(self, small_graph):
        x, edge_index = small_graph
        model = GCNModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2
        )
        out = model(x, edge_index)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_different_num_layers(self, small_graph):
        x, edge_index = small_graph
        for n_layers in [1, 2, 3, 4]:
            model = GCNModel(
                in_channels=16, hidden_channels=32,
                out_channels=8, num_layers=n_layers
            )
            model.eval()
            out = model(x, edge_index)
            assert out.shape == (20, 8)


# ── TransformerModel ─────────────────────────────────────────────────
class TestTransformerModel:
    def test_forward_shape(self, small_graph):
        x, edge_index = small_graph
        model = TransformerModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2,
            heads=2, concat=False
        )
        model.eval()
        out = model(x, edge_index)
        assert out.shape == (20, 16)

    def test_concat_mode(self, small_graph):
        x, edge_index = small_graph
        model = TransformerModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2,
            heads=2, concat=True
        )
        model.eval()
        out = model(x, edge_index)
        assert out.shape == (20, 16)

    def test_gradient_flow(self, small_graph):
        x, edge_index = small_graph
        model = TransformerModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2
        )
        out = model(x, edge_index)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ── SAGEModel ────────────────────────────────────────────────────────
class TestSAGEModel:
    def test_forward_shape(self, small_graph):
        x, edge_index = small_graph
        model = SAGEModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2
        )
        model.eval()
        out = model(x, edge_index)
        assert out.shape == (20, 16)

    def test_gradient_flow(self, small_graph):
        x, edge_index = small_graph
        model = SAGEModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2
        )
        out = model(x, edge_index)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ── GATModel ─────────────────────────────────────────────────────────
class TestGATModel:
    def test_forward_shape(self, small_graph):
        x, edge_index = small_graph
        model = GATModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2,
            heads=2, concat=False
        )
        model.eval()
        out = model(x, edge_index)
        assert out.shape == (20, 16)

    def test_concat_mode(self, small_graph):
        x, edge_index = small_graph
        model = GATModel(
            in_channels=16, hidden_channels=32,
            out_channels=16, num_layers=2,
            heads=2, concat=True
        )
        model.eval()
        out = model(x, edge_index)
        assert out.shape == (20, 16)


# ── LinkPredictor ────────────────────────────────────────────────────
class TestLinkPredictor:
    def _make_predictor(self, decoder_type='dot', out_channels=16):
        """Helper to create encoder + predictor."""
        encoder = GCNModel(in_channels=16, hidden_channels=32, out_channels=out_channels, num_layers=2)
        predictor = LinkPredictor(
            encoder=encoder,
            hidden_channels=out_channels,
            decoder_type=decoder_type,
        )
        return predictor

    def test_dot_decoder(self, small_graph, edge_pairs):
        x, edge_index = small_graph
        predictor = self._make_predictor('dot')
        predictor.eval()
        z = predictor.encode(x, edge_index)
        scores = predictor.decode(z, edge_pairs)
        assert scores.shape == (4,)

    def test_mlp_decoder(self, small_graph, edge_pairs):
        x, edge_index = small_graph
        predictor = self._make_predictor('mlp')
        predictor.eval()
        z = predictor.encode(x, edge_index)
        scores = predictor.decode(z, edge_pairs)
        assert scores.shape == (4,)

    def test_mlp_interaction_decoder(self, small_graph, edge_pairs):
        x, edge_index = small_graph
        predictor = self._make_predictor('mlp_interaction')
        predictor.eval()
        z = predictor.encode(x, edge_index)
        scores = predictor.decode(z, edge_pairs)
        assert scores.shape == (4,)

    def test_mlp_neighbor_decoder(self, small_graph, edge_pairs):
        x, edge_index = small_graph
        predictor = self._make_predictor('mlp_neighbor')
        predictor.eval()
        z = predictor.encode(x, edge_index)
        heuristics = torch.randn(4, 3)
        scores = predictor.decode(z, edge_pairs, heuristic_features=heuristics)
        assert scores.shape == (4,)

    def test_predict_proba(self, small_graph, edge_pairs):
        x, edge_index = small_graph
        predictor = self._make_predictor('dot')
        predictor.eval()
        z = predictor.encode(x, edge_index)
        probs = predictor.predict_proba(z, edge_pairs)
        assert probs.shape == (4,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_flow(self, small_graph, edge_pairs):
        x, edge_index = small_graph
        predictor = self._make_predictor('mlp')
        z = predictor.encode(x, edge_index)
        scores = predictor.decode(z, edge_pairs)
        loss = scores.sum()
        loss.backward()
        assert any(p.grad is not None for p in predictor.parameters())

    def test_mlp_heuristic_alias(self, small_graph, edge_pairs):
        """mlp_heuristic should alias to mlp_neighbor."""
        x, edge_index = small_graph
        encoder = GCNModel(in_channels=16, hidden_channels=32, out_channels=16, num_layers=2)
        predictor = LinkPredictor(
            encoder=encoder, hidden_channels=16,
            decoder_type='mlp_heuristic'
        )
        predictor.eval()
        z = predictor.encode(x, edge_index)
        scores = predictor.decode(z, edge_pairs)
        assert scores.shape == (4,)


# ── MODEL_CLASSES ────────────────────────────────────────────────────
class TestModelClasses:
    def test_all_keys_present(self):
        for name in ['GCN', 'Transformer', 'SAGE', 'GAT']:
            assert name in MODEL_CLASSES

    def test_all_values_are_classes(self):
        for cls in MODEL_CLASSES.values():
            assert isinstance(cls, type)

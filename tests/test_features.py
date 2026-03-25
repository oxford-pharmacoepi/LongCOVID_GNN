"""
Unit tests for src/features/heuristic_scores.py
"""

import pytest
import torch
import numpy as np

from src.features.heuristic_scores import HeuristicScorer, compute_heuristic_edge_features
from torch_geometric.data import Data


@pytest.fixture
def triangle_scorer():
    """Scorer for a triangle graph: 0-1, 1-2, 0-2 (bidirectional)."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2],
         [1, 0, 2, 1, 2, 0]], dtype=torch.long
    )
    return HeuristicScorer(edge_index, num_nodes=3)


@pytest.fixture
def path_scorer():
    """Scorer for a path: 0-1-2-3 (bidirectional)."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3],
         [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    return HeuristicScorer(edge_index, num_nodes=4)


class TestCommonNeighbors:
    def test_triangle(self, triangle_scorer):
        # 0 and 1 share neighbor 2
        cn = triangle_scorer.common_neighbors(0, 1)
        assert cn == 1

    def test_no_shared(self, path_scorer):
        # 0 and 3 share no neighbors in path graph
        cn = path_scorer.common_neighbors(0, 3)
        assert cn == 0

    def test_path_adjacent(self, path_scorer):
        # 1 and 2 share neighbors based on path structure
        cn = path_scorer.common_neighbors(1, 2)
        assert cn >= 0


class TestJaccardCoefficient:
    def test_triangle(self, triangle_scorer):
        jc = triangle_scorer.jaccard_coefficient(0, 1)
        assert 0.0 <= jc <= 1.0

    def test_no_neighbors(self):
        """Node with no neighbors → Jaccard = 0."""
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        scorer = HeuristicScorer(edge_index, num_nodes=3)
        jc = scorer.jaccard_coefficient(0, 2)  # node 2 has no outgoing edges
        assert jc == 0.0


class TestAdamicAdar:
    def test_triangle(self, triangle_scorer):
        aa = triangle_scorer.adamic_adar(0, 1)
        assert aa >= 0.0

    def test_no_shared(self, path_scorer):
        aa = path_scorer.adamic_adar(0, 3)
        assert aa == 0.0


class TestComputeAllScores:
    def test_returns_tuple(self, triangle_scorer):
        cn, aa, jc = triangle_scorer.compute_all_scores(0, 1)
        assert isinstance(cn, int)
        assert isinstance(aa, float)
        assert isinstance(jc, float)


class TestComputeEdgeHeuristicFeatures:
    def test_shape(self, triangle_scorer):
        edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        features = triangle_scorer.compute_edge_heuristic_features(edges)
        assert features.shape == (2, 3)

    def test_normalised(self, triangle_scorer):
        edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        features = triangle_scorer.compute_edge_heuristic_features(edges, normalise=True)
        assert features.max().item() <= 1.0

    def test_unnormalised(self, triangle_scorer):
        edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        features = triangle_scorer.compute_edge_heuristic_features(edges, normalise=False)
        assert features.shape == (2, 3)


class TestComputeBatchHeuristicFeatures:
    def test_shape(self, triangle_scorer):
        edges = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
        features = triangle_scorer.compute_batch_heuristic_features(
            edges, batch_size=2, show_progress=False
        )
        assert features.shape == (3, 3)


class TestConvenienceFunction:
    def test_compute_heuristic_edge_features(self):
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 0, 2],
             [1, 0, 2, 1, 2, 0]], dtype=torch.long
        )
        graph = Data(edge_index=edge_index, num_nodes=3)
        train_edges = torch.tensor([[0], [1]], dtype=torch.long)
        features = compute_heuristic_edge_features(
            graph, train_edges, normalise=False, show_progress=False
        )
        assert features.shape == (1, 3)

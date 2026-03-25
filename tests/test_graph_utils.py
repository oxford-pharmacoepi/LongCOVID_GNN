"""
Unit tests for src/utils/graph_utils.py (expanded)

Covers graph analysis, edge list generation, disease similarity,
and custom_edges function.
"""

import pytest
import torch
import numpy as np
import pyarrow as pa
from torch_geometric.data import Data
from collections import defaultdict

from src.utils.graph_utils import (
    generate_edge_list,
    standard_graph_analysis,
    custom_edges,
    create_disease_similarity_edges_from_ancestors,
)


@pytest.fixture
def tiny_graph():
    """5-node graph for analysis tests."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 0, 4],
         [1, 0, 2, 1, 3, 2, 4, 3, 4, 0]], dtype=torch.long
    )
    x = torch.randn(5, 4)
    return Data(x=x, edge_index=edge_index, num_nodes=5)


# ── generate_edge_list ───────────────────────────────────────────────
class TestGenerateEdgeList:
    def test_basic(self):
        edges = generate_edge_list(['A', 'B'], ['X', 'Y'], {'A': 0, 'B': 1}, {'X': 10, 'Y': 11})
        assert (0, 10) in edges
        assert (1, 11) in edges

    def test_missing_keys(self):
        edges = generate_edge_list(['A', 'MISSING'], ['X', 'Y'], {'A': 0}, {'X': 10, 'Y': 11})
        assert len(edges) == 1

    def test_empty(self):
        edges = generate_edge_list([], [], {}, {})
        assert edges == []

    def test_single(self):
        edges = generate_edge_list(['A'], ['X'], {'A': 0}, {'X': 10})
        assert edges == [(0, 10)]


# ── standard_graph_analysis ──────────────────────────────────────────
class TestStandardGraphAnalysis:
    def test_returns_dict(self, tiny_graph):
        result = standard_graph_analysis(tiny_graph)
        assert isinstance(result, dict)

    def test_correct_node_count(self, tiny_graph):
        result = standard_graph_analysis(tiny_graph)
        assert result['num_nodes'] == 5

    def test_density_in_range(self, tiny_graph):
        result = standard_graph_analysis(tiny_graph)
        assert 0.0 <= result['density'] <= 1.0

    def test_connected_graph(self, tiny_graph):
        result = standard_graph_analysis(tiny_graph)
        assert result['is_connected'] is True

    def test_disconnected_graph(self):
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
        )
        x = torch.randn(4, 2)
        graph = Data(x=x, edge_index=edge_index, num_nodes=4)
        result = standard_graph_analysis(graph)
        assert result['is_connected'] is False

    def test_avg_clustering(self, tiny_graph):
        result = standard_graph_analysis(tiny_graph)
        assert 'avg_clustering' in result
        assert result['avg_clustering'] >= 0.0

    def test_avg_betweenness(self, tiny_graph):
        result = standard_graph_analysis(tiny_graph)
        assert 'avg_betweenness' in result


# ── create_disease_similarity_edges_from_ancestors ───────────────────
class TestDiseaseSimilarityEdges:
    @pytest.fixture
    def disease_table(self):
        """Diseases with shared ancestors."""
        return pa.table({
            'id': ['D1', 'D2', 'D3', 'D4'],
            'ancestors': [['P1'], ['P1'], ['P2'], ['P2']],
        })

    def test_shared_parent(self, disease_table):
        mapping = {'D1': 0, 'D2': 1, 'D3': 2, 'D4': 3}
        edges = create_disease_similarity_edges_from_ancestors(
            disease_table, mapping, max_children_per_parent=10, min_shared_ancestors=1
        )
        assert len(edges) > 0
        # D1 and D2 share P1 → bidirectional edges
        assert (0, 1) in edges or (1, 0) in edges

    def test_no_shared(self):
        table = pa.table({
            'id': ['D1', 'D2'],
            'ancestors': [['P1'], ['P2']],
        })
        mapping = {'D1': 0, 'D2': 1}
        edges = create_disease_similarity_edges_from_ancestors(
            table, mapping, max_children_per_parent=10, min_shared_ancestors=1
        )
        # D1 and D2 have different parents → no edges
        assert (0, 1) not in edges

    def test_max_children_filter(self):
        """Parents with too many children should be skipped."""
        table = pa.table({
            'id': [f'D{i}' for i in range(15)],
            'ancestors': [['P1'] for _ in range(15)],
        })
        mapping = {f'D{i}': i for i in range(15)}
        edges = create_disease_similarity_edges_from_ancestors(
            table, mapping, max_children_per_parent=10, min_shared_ancestors=1
        )
        # P1 has 15 children > 10 → should be skipped
        assert len(edges) == 0


# ── custom_edges ─────────────────────────────────────────────────────
class TestCustomEdges:
    @pytest.fixture
    def tables_and_mappings(self):
        disease_table = pa.table({
            'id': ['D1', 'D2'],
            'ancestors': [['P1'], ['P1']],
        })
        molecule_table = pa.table({
            'id': ['M1'],
            'linkedDiseases.rows': [['D1', 'D2']],
        })
        disease_mapping = {'D1': 0, 'D2': 1}
        drug_mapping = {'M1': 10}
        return disease_table, molecule_table, disease_mapping, drug_mapping

    def test_no_custom_edges(self, tables_and_mappings):
        dt, mt, dm, drm = tables_and_mappings
        result = custom_edges(
            disease_similarity_network=False,
            disease_similarity_max_children=10,
            disease_similarity_min_shared=1,
            trial_edges=False,
            filtered_disease_table=dt,
            filtered_molecule_table=mt,
            disease_key_mapping=dm,
            drug_key_mapping=drm
        )
        assert result.shape[0] == 2
        assert result.shape[1] == 0

    def test_disease_similarity_only(self, tables_and_mappings):
        dt, mt, dm, drm = tables_and_mappings
        result = custom_edges(
            disease_similarity_network=True,
            disease_similarity_max_children=10,
            disease_similarity_min_shared=1,
            trial_edges=False,
            filtered_disease_table=dt,
            filtered_molecule_table=mt,
            disease_key_mapping=dm,
            drug_key_mapping=drm
        )
        assert result.shape[0] == 2
        # D1 and D2 share ancestor P1 → should create edges
        assert result.shape[1] > 0

    def test_trial_edges_only(self, tables_and_mappings):
        dt, mt, dm, drm = tables_and_mappings
        result = custom_edges(
            disease_similarity_network=False,
            disease_similarity_max_children=10,
            disease_similarity_min_shared=1,
            trial_edges=True,
            filtered_disease_table=dt,
            filtered_molecule_table=mt,
            disease_key_mapping=dm,
            drug_key_mapping=drm
        )
        assert result.shape[0] == 2
        assert result.shape[1] > 0

"""
Tests for src/utils/graph_utils.py — as_negative_sampling, find_repurposing_edges.
Uses mock PyArrow tables.
"""

import pytest
import torch
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd

from src.utils.graph_utils import (
    generate_edge_list,
    as_negative_sampling,
    find_repurposing_edges,
)


class TestAsNegativeSampling:
    @pytest.fixture
    def tables_and_mappings(self):
        """Create minimal PyArrow tables for as_negative_sampling."""
        # Molecule table with linked targets
        molecule_table = pa.table({
            'id': ['M1', 'M2'],
            'linkedTargets.rows': [['G1', 'G2'], ['G1']],
        })
        
        # Associations table with scores
        associations_table = pa.table({
            'diseaseId': ['D1', 'D2', 'D1', 'D2', 'D3'],
            'targetId': ['G1', 'G1', 'G2', 'G2', 'G1'],
            'score': [0.001, 0.9, 0.005, 0.001, 0.001],
        })
        
        drug_mapping = {'M1': 0, 'M2': 1}
        disease_mapping = {'D1': 10, 'D2': 11, 'D3': 12}
        
        return molecule_table, associations_table, drug_mapping, disease_mapping

    def test_returns_tensor(self, tables_and_mappings):
        mol, assoc, drug_map, disease_map = tables_and_mappings
        result = as_negative_sampling(
            mol, assoc, 'score', drug_map, disease_map
        )
        assert isinstance(result, torch.Tensor)
        if result.numel() > 0:
            assert result.shape[0] == 2

    def test_return_list(self, tables_and_mappings):
        mol, assoc, drug_map, disease_map = tables_and_mappings
        result = as_negative_sampling(
            mol, assoc, 'score', drug_map, disease_map, return_list=True
        )
        assert isinstance(result, list)

    def test_return_set(self, tables_and_mappings):
        mol, assoc, drug_map, disease_map = tables_and_mappings
        result = as_negative_sampling(
            mol, assoc, 'score', drug_map, disease_map, return_set=True
        )
        assert isinstance(result, set)

    def test_no_low_scores(self):
        """When all scores are high, should return empty."""
        mol = pa.table({
            'id': ['M1'],
            'linkedTargets.rows': [['G1']],
        })
        assoc = pa.table({
            'diseaseId': ['D1'],
            'targetId': ['G1'],
            'score': [0.9],  # All high scores
        })
        result = as_negative_sampling(
            mol, assoc, 'score', {'M1': 0}, {'D1': 10}
        )
        # Should be empty since no scores <= 0.01
        assert result.numel() == 0


class TestFindRepurposingEdges:
    def test_basic(self):
        table1 = pa.table({'id': ['M1', 'M2']})
        table2 = pa.table({
            'id': ['M1', 'M2', 'M3'],
            'linkedDiseases': [['D1', 'D2'], ['D3'], ['D4']],
        })
        drug_mapping = {'M1': 0, 'M2': 1}
        disease_mapping = {'D1': 10, 'D2': 11, 'D3': 12, 'D4': 13}
        
        result = find_repurposing_edges(
            table1, table2, 'linkedDiseases', drug_mapping, disease_mapping
        )
        assert isinstance(result, list)
        # M1 linked to D1, D2 → 2 edges; M2 linked to D3 → 1 edge
        assert len(result) == 3

    def test_no_overlap(self):
        table1 = pa.table({'id': ['M1']})
        table2 = pa.table({
            'id': ['M99'],
            'linkedDiseases': [['D1']],
        })
        result = find_repurposing_edges(
            table1, table2, 'linkedDiseases', {'M1': 0}, {'D1': 10}
        )
        assert len(result) == 0

    def test_missing_mappings(self):
        table1 = pa.table({'id': ['M1']})
        table2 = pa.table({
            'id': ['M1'],
            'linkedDiseases': [['D_UNKNOWN']],
        })
        result = find_repurposing_edges(
            table1, table2, 'linkedDiseases', {'M1': 0}, {'D1': 10}
        )
        # D_UNKNOWN not in disease_mapping → no edges
        assert len(result) == 0

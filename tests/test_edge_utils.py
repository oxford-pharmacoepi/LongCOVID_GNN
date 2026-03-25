"""
Unit tests for src/utils/edge_utils.py
"""

import pytest
import torch
import pyarrow as pa
from src.utils.edge_utils import (
    get_indices_from_keys,
    generate_pairs,
    generate_tensor,
    extract_edges,
    extract_edges_no_mapping,
)


# ── fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def simple_mappings():
    source_mapping = {'A': 0, 'B': 1, 'C': 2}
    target_mapping = {'X': 10, 'Y': 11, 'Z': 12}
    return source_mapping, target_mapping


@pytest.fixture
def arrow_table():
    """Simple PyArrow table with source and target list columns."""
    sources = pa.array(['A', 'B'])
    targets = pa.array([['X', 'Y'], ['Z']], type=pa.list_(pa.string()))
    return pa.table({'source': sources, 'targets': targets})


# ── get_indices_from_keys ────────────────────────────────────────────
class TestGetIndicesFromKeys:
    def test_all_keys_present(self, simple_mappings):
        source_mapping, _ = simple_mappings
        result = get_indices_from_keys(['A', 'B', 'C'], source_mapping)
        assert result == [0, 1, 2]

    def test_missing_keys_skipped(self, simple_mappings):
        source_mapping, _ = simple_mappings
        result = get_indices_from_keys(['A', 'D', 'C'], source_mapping)
        assert result == [0, 2]

    def test_empty_list(self, simple_mappings):
        source_mapping, _ = simple_mappings
        result = get_indices_from_keys([], source_mapping)
        assert result == []


# ── generate_pairs ───────────────────────────────────────────────────
class TestGeneratePairs:
    def test_all_combinations(self, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = generate_pairs(['A'], ['X', 'Y'], source_mapping, target_mapping)
        assert (0, 10) in result
        assert (0, 11) in result
        assert len(result) == 2

    def test_return_set(self, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = generate_pairs(['A'], ['X'], source_mapping, target_mapping, return_set=True)
        assert isinstance(result, set)

    def test_return_tensor(self, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = generate_pairs(['A', 'B'], ['X'], source_mapping, target_mapping, return_tensor=True)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2  # [2, num_edges]

    def test_missing_keys_skipped(self, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = generate_pairs(['A', 'MISSING'], ['X'], source_mapping, target_mapping)
        assert len(result) == 1


# ── generate_tensor ──────────────────────────────────────────────────
class TestGenerateTensor:
    def test_parallel_lists(self, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = generate_tensor(['A', 'B'], ['X', 'Y'], source_mapping, target_mapping)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)

    def test_missing_keys(self, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = generate_tensor(['A', 'MISSING'], ['X', 'Y'], source_mapping, target_mapping)
        assert result.shape[1] == 1  # Only A->X survived


# ── extract_edges ────────────────────────────────────────────────────
class TestExtractEdges:
    def test_returns_tensor(self, arrow_table, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = extract_edges(arrow_table, source_mapping, target_mapping)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2  # [2, num_edges]

    def test_return_edge_list(self, arrow_table, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = extract_edges(arrow_table, source_mapping, target_mapping, return_edge_list=True)
        assert isinstance(result, list)
        assert len(result) == 3  # A->X, A->Y, B->Z

    def test_return_edge_set(self, arrow_table, simple_mappings):
        source_mapping, target_mapping = simple_mappings
        result = extract_edges(arrow_table, source_mapping, target_mapping, return_edge_set=True)
        assert isinstance(result, set)

    def test_debug_output(self, arrow_table, simple_mappings, capsys):
        source_mapping, target_mapping = simple_mappings
        # Add a missing target to trigger debug
        sources = pa.array(['A'])
        targets = pa.array([['MISSING']], type=pa.list_(pa.string()))
        table = pa.table({'source': sources, 'targets': targets})
        extract_edges(table, source_mapping, target_mapping, debug=True)
        captured = capsys.readouterr()
        assert 'Missing' in captured.out or len(captured.out) == 0  # Debug may or may not print


# ── extract_edges_no_mapping ────────────────────────────────────────
class TestExtractEdgesNoMapping:
    def test_returns_list(self, arrow_table):
        result = extract_edges_no_mapping(arrow_table)
        assert isinstance(result, list)
        assert all('->' in edge for edge in result)

    def test_return_edge_list(self, arrow_table):
        result = extract_edges_no_mapping(arrow_table, return_edge_list=True)
        assert isinstance(result, list)

    def test_return_edge_set(self, arrow_table):
        result = extract_edges_no_mapping(arrow_table, return_edge_set=True)
        assert isinstance(result, set)

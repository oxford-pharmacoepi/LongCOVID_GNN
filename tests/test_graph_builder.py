"""
Tests for src/graph/builder.py — GraphBuilder._prune_graph and _parse_list_columns.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from unittest.mock import patch, MagicMock

from src.graph.builder import GraphBuilder
from src.config import Config


@pytest.fixture
def mock_builder():
    """Create a GraphBuilder with mocked __init__."""
    with patch.object(GraphBuilder, '__init__', lambda self, *a, **kw: None):
        builder = GraphBuilder.__new__(GraphBuilder)
        builder.config = Config()
        builder.tracker = None
        builder.mappings = None
        return builder


class TestPruneGraph:
    def test_removes_isolated(self, mock_builder):
        """Node 3 has no edges → should be removed."""
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        
        train_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        train_labels = torch.tensor([1.0, 0.0])
        val_edges = torch.tensor([[0, 2]], dtype=torch.long)
        val_labels = torch.tensor([1.0])
        test_edges = torch.tensor([[1, 0]], dtype=torch.long)
        test_labels = torch.tensor([0.0])
        
        graph = Data(
            x=x, edge_index=edge_index,
            train_edge_index=train_edges, train_edge_label=train_labels,
            val_edge_index=val_edges, val_edge_label=val_labels,
            test_edge_index=test_edges, test_edge_label=test_labels,
            metadata={
                'node_info': {'Type1': 2, 'Type2': 2},
                'total_nodes': 4, 'total_edges': 4,
            }
        )
        
        mock_builder.mappings = {
            'drug_key_mapping': {f'D{i}': i for i in range(4)},
            'approved_drugs_list': [f'D{i}' for i in range(4)],
        }
        
        result = mock_builder._prune_graph(graph)
        assert result.x.shape[0] == 3  # Node 3 removed
        assert result.metadata['total_nodes'] == 3

    def test_no_isolated(self, mock_builder):
        """All nodes connected → nothing removed."""
        x = torch.randn(3, 3)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        
        graph = Data(
            x=x, edge_index=edge_index,
            train_edge_index=torch.tensor([[0, 1]], dtype=torch.long),
            train_edge_label=torch.tensor([1.0]),
            val_edge_index=torch.tensor([[0, 2]], dtype=torch.long),
            val_edge_label=torch.tensor([1.0]),
            test_edge_index=torch.tensor([[1, 2]], dtype=torch.long),
            test_edge_label=torch.tensor([0.0]),
            metadata={
                'node_info': {'Type1': 3},
                'total_nodes': 3, 'total_edges': 4,
            }
        )
        mock_builder.mappings = None
        
        result = mock_builder._prune_graph(graph)
        assert result.x.shape[0] == 3


class TestParseListColumns:
    def test_parse_list_columns(self, mock_builder):
        mock_builder.molecule_df = pd.DataFrame({
            'id': ['M1'],
            'drugType': ['SmallMolecule'],
            'blackBoxWarning': ['true'],
            'yearOfFirstApproval': [2010],
            'parentId': [None],
            'childChemblIds': [None],
            'linkedTargets.rows': ["['G1', 'G2']"],
            'linkedDiseases.rows': ["['D1']"],
        })
        mock_builder.indication_df = pd.DataFrame({
            'id': ['M1'],
            'approvedIndications': ["['D1']"],
        })
        mock_builder.disease_df = pd.DataFrame({
            'id': ['D1'],
            'name': ['Disease1'],
            'description': ['d1'],
            'therapeuticAreas': ["['TA1']"],
            'ancestors': ["['A1']"],
            'descendants': ['[]'],
            'children': [None],
        })
        
        mock_builder._parse_list_columns()
        
        # Stringified lists should now be actual lists
        assert isinstance(mock_builder.molecule_df.loc[0, 'linkedTargets.rows'], list)
        assert isinstance(mock_builder.disease_df.loc[0, 'therapeuticAreas'], list)

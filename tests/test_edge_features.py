"""
Tests for src/features/edge.py — MoA feature extraction and edge type masks.
"""

import pytest
import torch
import pandas as pd
import numpy as np

from src.features.edge import (
    extract_moa_features,
    normalise_action_type,
    pad_edge_features_to_match_all_edges,
    create_edge_type_mask,
)


ACTION_TYPES = {
    'inhibitor': 0, 'antagonist': 1, 'agonist': 2,
    'activator': 3, 'modulator': 4, 'other': 5,
}


class TestNormaliseActionType:
    def test_direct_match(self):
        assert normalise_action_type('inhibitor', ACTION_TYPES) == 0

    def test_partial_inhibitor(self):
        assert normalise_action_type('ENZYME INHIBITOR', ACTION_TYPES) == 0

    def test_blocker(self):
        assert normalise_action_type('channel blocker', ACTION_TYPES) == 0

    def test_antagonist(self):
        assert normalise_action_type('ANTAGONIST', ACTION_TYPES) == 1

    def test_inverse_agonist(self):
        assert normalise_action_type('inverse agonist', ACTION_TYPES) == 1

    def test_agonist(self):
        assert normalise_action_type('partial agonist', ACTION_TYPES) == 2

    def test_activator(self):
        assert normalise_action_type('ACTIVATOR', ACTION_TYPES) == 3

    def test_inducer(self):
        assert normalise_action_type('inducer', ACTION_TYPES) == 3

    def test_modulator(self):
        assert normalise_action_type('allosteric modulator', ACTION_TYPES) == 4

    def test_regulator(self):
        assert normalise_action_type('negative regulator', ACTION_TYPES) == 4

    def test_other(self):
        assert normalise_action_type('unknown mechanism', ACTION_TYPES) == 5


class TestExtractMoaFeatures:
    def test_empty_moa(self):
        edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        result = extract_moa_features(
            pd.DataFrame(), {'M1': 0, 'M2': 1}, {'G1': 2, 'G2': 3}, edges
        )
        assert result.shape == (2, 6)

    def test_with_moa_data(self):
        edges = torch.tensor([[0, 0], [2, 3]], dtype=torch.long)
        moa_df = pd.DataFrame({
            'chemblIds': [['M1']],
            'targets': [['G1']],
            'actionType': ['inhibitor'],
        })
        result = extract_moa_features(
            moa_df, {'M1': 0}, {'G1': 2, 'G2': 3}, edges
        )
        assert result.shape == (2, 6)
        assert result[0, 0] == 1.0  # inhibitor

    def test_string_drug_id(self):
        edges = torch.tensor([[0], [2]], dtype=torch.long)
        moa_df = pd.DataFrame({
            'chemblIds': ['M1'],  # string, not list
            'targets': [['G1']],
            'actionType': ['agonist'],
        })
        result = extract_moa_features(
            moa_df, {'M1': 0}, {'G1': 2}, edges
        )
        assert result.shape == (1, 6)

    def test_none_drug_ids(self):
        edges = torch.tensor([[0], [2]], dtype=torch.long)
        moa_df = pd.DataFrame({
            'chemblIds': [None],
            'targets': [['G1']],
            'actionType': ['agonist'],
        })
        result = extract_moa_features(
            moa_df, {'M1': 0}, {'G1': 2}, edges
        )
        assert result.shape == (1, 6)

    def test_none_targets(self):
        edges = torch.tensor([[0], [2]], dtype=torch.long)
        moa_df = pd.DataFrame({
            'chemblIds': [['M1']],
            'targets': [None],
            'actionType': ['agonist'],
        })
        result = extract_moa_features(
            moa_df, {'M1': 0}, {'G1': 2}, edges
        )
        assert result.shape == (1, 6)

    def test_nan_action_type(self):
        edges = torch.tensor([[0], [2]], dtype=torch.long)
        moa_df = pd.DataFrame({
            'chemblIds': [['M1']],
            'targets': [['G1']],
            'actionType': [float('nan')],
        })
        result = extract_moa_features(
            moa_df, {'M1': 0}, {'G1': 2}, edges
        )
        assert result.shape == (1, 6)


class TestPadEdgeFeatures:
    def test_basic(self):
        features = torch.randn(10, 6)
        result = pad_edge_features_to_match_all_edges(features, 10, 50)
        assert result.shape == (50, 6)

    def test_shape_mismatch_raises(self):
        features = torch.randn(10, 6)
        with pytest.raises(ValueError):
            pad_edge_features_to_match_all_edges(features, 20, 50)


class TestCreateEdgeTypeMask:
    def test_basic(self):
        counts = {'drug_gene': 100, 'drug_disease': 200, 'gene_reactome': 50}
        result = create_edge_type_mask(counts)
        assert result['drug_gene'] == (0, 100)
        assert result['drug_disease'] == (100, 300)
        assert result['gene_reactome'] == (300, 350)

    def test_empty(self):
        assert create_edge_type_mask({}) == {}

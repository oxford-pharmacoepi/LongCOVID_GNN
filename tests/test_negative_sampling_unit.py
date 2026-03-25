"""
Additional unit tests for src/negative_sampling.py

Covers the sampler classes that are not reached by integration tests.
"""

import pytest
import torch
import numpy as np

from src.negative_sampling import (
    RandomNegativeSampler,
    HardNegativeSampler,
    DegreeMatchedNegativeSampler,
    FeatureSimilarityNegativeSampler,
    MixedNegativeSampler,
    get_sampler,
    validate_temporal_consistency,
    filter_negatives_by_future_positives,
)


# ── fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def setup():
    """Common test data."""
    torch.manual_seed(42)
    positive_edges = {(0, 10), (1, 11), (2, 12)}
    all_possible_pairs = [(i, j) for i in range(5) for j in range(10, 15)]
    edge_index = torch.tensor(
        [[0, 10, 1, 11, 2, 12],
         [10, 0, 11, 1, 12, 2]], dtype=torch.long
    )
    node_features = torch.randn(15, 8)
    return positive_edges, all_possible_pairs, edge_index, node_features


# ── RandomNegativeSampler ────────────────────────────────────────────
class TestRandomNegativeSampler:
    def test_sample_count(self, setup):
        pos, pairs, ei, nf = setup
        sampler = RandomNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=5,
            edge_index=ei,
            node_features=nf
        )
        assert len(negatives) == 5

    def test_no_overlap_with_positives(self, setup):
        pos, pairs, ei, nf = setup
        sampler = RandomNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=10,
            edge_index=ei,
            node_features=nf
        )
        assert len(set(negatives) & pos) == 0

    def test_returns_list(self, setup):
        pos, pairs, ei, nf = setup
        sampler = RandomNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=3,
            edge_index=ei,
            node_features=nf
        )
        assert isinstance(negatives, (list, set))


# ── HardNegativeSampler ─────────────────────────────────────────────
class TestHardNegativeSampler:
    def test_sample_count(self, setup):
        pos, pairs, ei, nf = setup
        sampler = HardNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=5,
            edge_index=ei,
            node_features=nf
        )
        assert len(negatives) >= 1

    def test_no_overlap(self, setup):
        pos, pairs, ei, nf = setup
        sampler = HardNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=5,
            edge_index=ei,
            node_features=nf
        )
        assert len(set(negatives) & pos) == 0


# ── DegreeMatchedNegativeSampler ─────────────────────────────────────
class TestDegreeMatchedNegativeSampler:
    def test_sample_count(self, setup):
        pos, pairs, ei, nf = setup
        sampler = DegreeMatchedNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=5,
            edge_index=ei,
            node_features=nf
        )
        assert len(negatives) >= 1

    def test_no_overlap(self, setup):
        pos, pairs, ei, nf = setup
        sampler = DegreeMatchedNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=5,
            edge_index=ei,
            node_features=nf
        )
        assert len(set(negatives) & pos) == 0


# ── FeatureSimilarityNegativeSampler ─────────────────────────────────
class TestFeatureSimilarityNegativeSampler:
    def test_sample_count(self, setup):
        pos, pairs, ei, nf = setup
        sampler = FeatureSimilarityNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=5,
            edge_index=ei,
            node_features=nf
        )
        assert len(negatives) >= 1

    def test_no_overlap(self, setup):
        pos, pairs, ei, nf = setup
        sampler = FeatureSimilarityNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=5,
            edge_index=ei,
            node_features=nf
        )
        assert len(set(negatives) & pos) == 0


# ── MixedNegativeSampler ────────────────────────────────────────────
class TestMixedNegativeSampler:
    def test_sample_count(self, setup):
        pos, pairs, ei, nf = setup
        sampler = MixedNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=10,
            edge_index=ei,
            node_features=nf
        )
        assert len(negatives) >= 1

    def test_no_overlap(self, setup):
        pos, pairs, ei, nf = setup
        sampler = MixedNegativeSampler(seed=42)
        negatives = sampler.sample(
            positive_edges=pos,
            all_possible_pairs=pairs,
            num_samples=10,
            edge_index=ei,
            node_features=nf
        )
        assert len(set(negatives) & pos) == 0


# ── validate_temporal_consistency ────────────────────────────────────
class TestValidateTemporalConsistency:
    def test_no_leakage(self):
        train_negs = {(0, 10), (1, 11)}
        val_pos = {(2, 12)}
        test_pos = {(3, 13)}
        result = validate_temporal_consistency(train_negs, val_pos, test_pos)
        assert result['total_leakage_count'] == 0

    def test_with_leakage(self):
        train_negs = {(0, 10), (2, 12)}  # (2,12) overlaps with val_pos
        val_pos = {(2, 12), (3, 13)}
        test_pos = {(4, 14)}
        result = validate_temporal_consistency(train_negs, val_pos, test_pos)
        assert result['total_leakage_count'] >= 1


# ── filter_negatives_by_future_positives ─────────────────────────────
class TestFilterNegatives:
    def test_filters_correctly(self):
        negatives = [(0, 10), (1, 11), (2, 12)]
        future_positives = {(2, 12)}
        result = filter_negatives_by_future_positives(negatives, future_positives)
        assert (2, 12) not in result
        assert len(result) == 2


# ── get_sampler factory ──────────────────────────────────────────────
class TestGetSampler:
    @pytest.mark.parametrize("strategy", [
        "random", "hard", "degree_matched", "feature_similar", "mixed"
    ])
    def test_factory(self, strategy):
        sampler = get_sampler(strategy=strategy, seed=42)
        assert sampler is not None

    def test_invalid_strategy(self):
        with pytest.raises((ValueError, KeyError)):
            get_sampler(strategy="nonexistent")

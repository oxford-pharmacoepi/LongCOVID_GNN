"""
Unit tests for src/utils/eval_utils.py

Tests evaluation metrics with known-answer inputs.
"""

import pytest
import numpy as np
from src.utils.eval_utils import (
    calculate_metrics,
    calculate_recall_at_k,
    calculate_hits_at_k,
    calculate_precision_at_k,
    calculate_ndcg_at_k,
    calculate_bootstrap_ci,
    calculate_ranking_metrics,
)


# ── fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def perfect_predictions():
    """Labels and scores where positive/negative are perfectly separated."""
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    preds = np.array([1, 1, 1, 0, 0, 0])
    return labels, scores, preds


@pytest.fixture
def mixed_predictions():
    """Realistic labels, scores, preds with some errors."""
    labels = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.6, 0.7, 0.2, 0.5, 0.3, 0.1, 0.8])
    preds = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    return labels, scores, preds


# ── calculate_metrics ────────────────────────────────────────────────
class TestCalculateMetrics:
    def test_perfect_auc(self, perfect_predictions):
        labels, scores, preds = perfect_predictions
        metrics = calculate_metrics(labels, scores, preds)
        assert metrics['auc'] == pytest.approx(1.0)

    def test_returns_dict(self, mixed_predictions):
        labels, scores, preds = mixed_predictions
        metrics = calculate_metrics(labels, scores, preds)
        assert isinstance(metrics, dict)
        assert 'auc' in metrics
        assert 'apr' in metrics  # average precision → 'apr'

    def test_metrics_in_range(self, mixed_predictions):
        labels, scores, preds = mixed_predictions
        metrics = calculate_metrics(labels, scores, preds)
        for key in ['auc', 'apr', 'accuracy', 'precision', 'recall', 'f1']:
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]}"

    def test_confusion_matrix_present(self, mixed_predictions):
        labels, scores, preds = mixed_predictions
        metrics = calculate_metrics(labels, scores, preds)
        assert 'confusion_matrix' in metrics
        cm = metrics['confusion_matrix']
        assert 'TP' in cm and 'FP' in cm and 'TN' in cm and 'FN' in cm


# ── calculate_recall_at_k ───────────────────────────────────────────
class TestRecallAtK:
    def test_returns_dict(self, perfect_predictions):
        labels, scores, _ = perfect_predictions
        result = calculate_recall_at_k(labels, scores, k_values=[3])
        assert isinstance(result, dict)
        assert 'recall@3' in result

    def test_perfect_recall(self, perfect_predictions):
        labels, scores, _ = perfect_predictions
        result = calculate_recall_at_k(labels, scores, k_values=[3])
        assert result['recall@3'] == pytest.approx(1.0)

    def test_recall_increases_with_k(self, mixed_predictions):
        labels, scores, _ = mixed_predictions
        result = calculate_recall_at_k(labels, scores, k_values=[1, 4])
        assert result['recall@4'] >= result['recall@1']


# ── calculate_hits_at_k ─────────────────────────────────────────────
class TestHitsAtK:
    def test_perfect_hits(self, perfect_predictions):
        labels, scores, _ = perfect_predictions
        result = calculate_hits_at_k(labels, scores, k_values=[3])
        assert result['hits_at_3'] == pytest.approx(1.0)

    def test_hit_at_1(self, perfect_predictions):
        labels, scores, _ = perfect_predictions
        result = calculate_hits_at_k(labels, scores, k_values=[1])
        assert result['hits_at_1'] == pytest.approx(1.0)

    def test_range(self, mixed_predictions):
        labels, scores, _ = mixed_predictions
        result = calculate_hits_at_k(labels, scores, k_values=[3])
        assert 0.0 <= result['hits_at_3'] <= 1.0


# ── calculate_precision_at_k ────────────────────────────────────────
class TestPrecisionAtK:
    def test_perfect_precision(self, perfect_predictions):
        labels, scores, _ = perfect_predictions
        result = calculate_precision_at_k(labels, scores, k_values=[3])
        assert result['precision@3'] == pytest.approx(1.0)

    def test_range(self, mixed_predictions):
        labels, scores, _ = mixed_predictions
        result = calculate_precision_at_k(labels, scores, k_values=[3])
        assert 0.0 <= result['precision@3'] <= 1.0


# ── calculate_ndcg_at_k ─────────────────────────────────────────────
class TestNDCGAtK:
    def test_perfect_ndcg(self, perfect_predictions):
        labels, scores, _ = perfect_predictions
        result = calculate_ndcg_at_k(labels, scores, k_values=[3])
        assert result['ndcg@3'] == pytest.approx(1.0)

    def test_range(self, mixed_predictions):
        labels, scores, _ = mixed_predictions
        result = calculate_ndcg_at_k(labels, scores, k_values=[3])
        assert 0.0 <= result['ndcg@3'] <= 1.0


# ── calculate_ranking_metrics ────────────────────────────────────────
class TestRankingMetrics:
    def test_returns_all_metric_types(self, mixed_predictions):
        labels, scores, _ = mixed_predictions
        result = calculate_ranking_metrics(labels, scores, k_values=[3])
        assert 'recall@3' in result
        assert 'precision@3' in result
        assert 'hits_at_3' in result
        assert 'ndcg@3' in result


# ── calculate_bootstrap_ci ──────────────────────────────────────────
class TestBootstrapCI:
    def test_returns_dict(self, perfect_predictions):
        labels, scores, preds = perfect_predictions
        result = calculate_bootstrap_ci(labels, scores, preds, n_bootstrap=50)
        assert isinstance(result, dict)

    def test_ci_contains_expected_keys(self, mixed_predictions):
        labels, scores, preds = mixed_predictions
        result = calculate_bootstrap_ci(labels, scores, preds, n_bootstrap=100)
        assert 'auc' in result
        info = result['auc']
        assert 'ci_lower' in info
        assert 'ci_upper' in info
        assert 'mean' in info

    def test_ci_bounds(self, mixed_predictions):
        labels, scores, preds = mixed_predictions
        result = calculate_bootstrap_ci(labels, scores, preds, n_bootstrap=100)
        for metric_name, info in result.items():
            if not np.isnan(info['mean']):
                assert info['ci_lower'] <= info['ci_upper']

"""
Expanded tests for src/training/losses.py

Covers RankingAwareBCELoss, GroupedRankingAwareLoss, ConfidenceWeightedBCELoss
with confidences, and get_loss_function factory for all types.
"""

import pytest
import torch
from src.training.losses import (
    StandardBCEWithLogitsLoss,
    WeightedBCEWithLogitsLoss,
    ConfidenceWeightedBCELoss,
    FocalLoss,
    BalancedFocalLoss,
    RankingAwareBCELoss,
    GroupedRankingAwareLoss,
    get_loss_function,
)


@pytest.fixture
def batch():
    torch.manual_seed(42)
    logits = torch.randn(32)
    targets = torch.cat([torch.ones(16), torch.zeros(16)])
    return logits, targets


# ── Standard BCE ─────────────────────────────────────────────────────
class TestStandardBCE:
    def test_loss_scalar(self, batch):
        logits, targets = batch
        loss_fn = StandardBCEWithLogitsLoss()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0 and loss.item() > 0

    def test_kwargs_ignored(self, batch):
        logits, targets = batch
        loss_fn = StandardBCEWithLogitsLoss()
        loss = loss_fn(logits, targets, extra_arg=True)
        assert loss.dim() == 0


# ── Weighted BCE ─────────────────────────────────────────────────────
class TestWeightedBCE:
    def test_auto_weight(self, batch):
        logits, targets = batch
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=None)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_explicit_weight(self, batch):
        logits, targets = batch
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=5.0)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0


# ── Confidence-Weighted BCE ──────────────────────────────────────────
class TestConfidenceWeightedBCE:
    def test_no_confidences(self, batch):
        logits, targets = batch
        loss_fn = ConfidenceWeightedBCELoss()
        loss = loss_fn(logits, targets, sample_confidences=None)
        assert loss.dim() == 0

    def test_with_confidences(self, batch):
        logits, targets = batch
        confidences = torch.rand(32)
        loss_fn = ConfidenceWeightedBCELoss()
        loss = loss_fn(logits, targets, sample_confidences=confidences)
        assert loss.dim() == 0

    def test_custom_weights(self, batch):
        logits, targets = batch
        confidences = torch.rand(32)
        loss_fn = ConfidenceWeightedBCELoss(pos_weight=5.0, min_neg_weight=0.2, max_neg_weight=0.8)
        loss = loss_fn(logits, targets, sample_confidences=confidences)
        assert loss.dim() == 0


# ── Focal Loss ───────────────────────────────────────────────────────
class TestFocalLoss:
    def test_default(self, batch):
        logits, targets = batch
        loss_fn = FocalLoss()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_custom_gamma(self, batch):
        logits, targets = batch
        loss_fn = FocalLoss(gamma=5.0)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0


# ── Balanced Focal Loss ──────────────────────────────────────────────
class TestBalancedFocalLoss:
    def test_default(self, batch):
        logits, targets = batch
        loss_fn = BalancedFocalLoss()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_custom_params(self, batch):
        logits, targets = batch
        loss_fn = BalancedFocalLoss(alpha=0.5, gamma=3.0, pos_weight=10.0)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0


# ── RankingAwareBCELoss ──────────────────────────────────────────────
class TestRankingAwareBCE:
    def test_default(self, batch):
        logits, targets = batch
        loss_fn = RankingAwareBCELoss()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_custom_margin(self, batch):
        logits, targets = batch
        loss_fn = RankingAwareBCELoss(margin=1.0, ranking_weight=0.5, variance_weight=0.2)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_all_positive(self):
        logits = torch.randn(10)
        targets = torch.ones(10)
        loss_fn = RankingAwareBCELoss()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_explicit_pos_weight(self, batch):
        logits, targets = batch
        loss_fn = RankingAwareBCELoss(pos_weight=5.0)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0


# ── GroupedRankingAwareLoss ──────────────────────────────────────────
class TestGroupedRankingLoss:
    def test_no_groups(self, batch):
        """Without groups, falls back to global ranking."""
        logits, targets = batch
        loss_fn = GroupedRankingAwareLoss()
        loss = loss_fn(logits, targets, groups=None)
        assert loss.dim() == 0

    def test_with_groups(self, batch):
        logits, targets = batch
        groups = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)])
        loss_fn = GroupedRankingAwareLoss()
        loss = loss_fn(logits, targets, groups=groups)
        assert loss.dim() == 0

    def test_single_group(self, batch):
        logits, targets = batch
        groups = torch.zeros(32, dtype=torch.long)
        loss_fn = GroupedRankingAwareLoss()
        loss = loss_fn(logits, targets, groups=groups)
        assert loss.dim() == 0


# ── get_loss_function factory ────────────────────────────────────────
class TestGetLossFunction:
    @pytest.mark.parametrize("loss_type", [
        'standard_bce', 'bce', 'weighted_bce', 'confidence_weighted',
        'focal', 'balanced_focal', 'ranking_aware_bce', 'grouped_ranking_bce'
    ])
    def test_factory(self, loss_type):
        loss_fn = get_loss_function(loss_type)
        assert loss_fn is not None

    def test_factory_with_kwargs(self):
        loss_fn = get_loss_function('focal', alpha=0.7, gamma=3.0)
        assert loss_fn is not None

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            get_loss_function('nonexistent')

    def test_filtering_irrelevant_kwargs(self):
        """Extra kwargs should be filtered out."""
        loss_fn = get_loss_function('standard_bce', irrelevant_param=99)
        assert loss_fn is not None

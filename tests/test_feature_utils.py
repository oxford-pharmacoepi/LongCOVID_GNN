"""
Tests for src/utils/feature_utils.py — PyArrow-based encoding functions.
These require pyarrow/pandas arrays.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import pyarrow as pa

from src.utils.feature_utils import (
    boolean_encode,
    normalise,
    pad_feature_matrix,
    align_features,
    one_hot_encode,
    one_hot_encode_categorical,
)


class TestBooleanEncode:
    def test_basic(self):
        arr = pa.array([True, False, None])
        pad = [0] * 5  # pad to length 5
        result = boolean_encode(arr, pad)
        assert result.shape == (5, 1)
        assert result[0, 0] == 1  # True
        assert result[1, 0] == 0  # False
        assert result[2, 0] == -1  # None → -1

    def test_no_padding_needed(self):
        arr = pa.array([True, False])
        pad = [0, 1]  # same length
        result = boolean_encode(arr, pad)
        assert result.shape == (2, 1)


class TestNormalise:
    def test_basic(self):
        arr = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pad = [0] * 7  # pad to length 7
        result = normalise(arr, pad)
        assert result.shape == (7, 1)

    def test_no_padding(self):
        arr = pa.array([10.0, 20.0])
        pad = [0, 1]
        result = normalise(arr, pad)
        assert result.shape == (2, 1)


class TestPadFeatureMatrixExpanded:
    def test_pad_needed(self):
        m = torch.randn(3, 4)
        result = pad_feature_matrix(m, 8, pad_value=-1)
        assert result.shape == (3, 8)
        assert (result[:, 4:] == -1).all()

    def test_no_pad(self):
        m = torch.randn(3, 10)
        result = pad_feature_matrix(m, 10)
        assert result.shape == (3, 10)

    def test_larger_already(self):
        m = torch.randn(3, 15)
        result = pad_feature_matrix(m, 10)
        assert result.shape == (3, 15)


class TestAlignFeaturesExpanded:
    def test_subset(self):
        m = torch.tensor([[1.0, 2.0]])
        result = align_features(m, ['B', 'C'], ['A', 'B', 'C', 'D'])
        assert result.shape == (1, 4)
        assert result[0, 0] == -1  # A not in feature_columns
        assert result[0, 1] == 1.0  # B
        assert result[0, 2] == 2.0  # C
        assert result[0, 3] == -1  # D not in feature_columns


class TestOneHotEncodeExpanded:
    def test_various(self):
        for i in range(5):
            result = one_hot_encode(i, 5)
            assert result.sum() == 1.0
            assert result[0, i] == 1.0


class TestOneHotEncodeCategoricalExpanded:
    def test_all_same(self):
        result = one_hot_encode_categorical(['A', 'A', 'A'])
        assert result.shape == (3, 1)
        assert (result == 1.0).all()

    def test_empty(self):
        with pytest.raises(RuntimeError):
            one_hot_encode_categorical([])

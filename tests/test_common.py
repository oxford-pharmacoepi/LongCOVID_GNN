"""
Unit tests for src/utils/common.py
"""

import pytest
import torch
import numpy as np
import random

from src.utils.common import set_seed, enable_full_reproducibility


class TestSetSeed:
    def test_deterministic_torch(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_deterministic_numpy(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_random(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b


class TestEnableFullReproducibility:
    def test_deterministic(self):
        enable_full_reproducibility(42)
        a = torch.randn(5)
        enable_full_reproducibility(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

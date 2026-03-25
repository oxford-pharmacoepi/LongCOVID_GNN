"""
Unit tests for src/graph/split.py
"""

import pytest
import torch
import random

from src.graph.split import DataSplitter


class TestVerifySplits:
    """Test the _verify_splits method directly."""

    def test_no_overlap(self, capsys):
        splitter = DataSplitter.__new__(DataSplitter)
        train_neg = [(0, 5), (1, 6)]
        val_neg = [(2, 7)]
        test_neg = [(3, 8)]
        all_pos = {(0, 10), (1, 11)}
        
        splitter._verify_splits(train_neg, val_neg, test_neg, all_pos)
        captured = capsys.readouterr()
        assert 'passed' in captured.out.lower()

    def test_overlap_detected(self, capsys):
        splitter = DataSplitter.__new__(DataSplitter)
        train_neg = [(0, 10)]  # overlaps with positives
        val_neg = []
        test_neg = []
        all_pos = {(0, 10)}
        
        splitter._verify_splits(train_neg, val_neg, test_neg, all_pos)
        captured = capsys.readouterr()
        assert 'WARNING' in captured.out or 'overlap' in captured.out.lower()

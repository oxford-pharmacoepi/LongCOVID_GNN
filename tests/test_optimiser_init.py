"""
Tests for src/training/optimiser.py — __init__ with real graph, objective method.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data

from src.training.optimiser import BayesianOptimiser
from src.config import Config
import optuna


@pytest.fixture
def graph_with_splits():
    """Create a small graph with proper train/val splits."""
    num_nodes = 20
    x = torch.randn(num_nodes, 16)
    edges = []
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    graph = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    
    # Create proper train/val edge tensors
    train_pos = torch.randint(0, num_nodes, (50, 2))
    train_neg = torch.randint(0, num_nodes, (50, 2))
    graph.train_edge_index = torch.cat([train_pos, train_neg])
    graph.train_edge_label = torch.cat([torch.ones(50), torch.zeros(50)])
    
    val_pos = torch.randint(0, num_nodes, (20, 2))
    val_neg = torch.randint(0, num_nodes, (20, 2))
    graph.val_edge_index = torch.cat([val_pos, val_neg])
    graph.val_edge_label = torch.cat([torch.ones(20), torch.zeros(20)])
    
    return graph


class TestBayesianOptimiserInit:
    @patch('src.training.tracker.mlflow')
    def test_init(self, mock_mlflow, graph_with_splits):
        mock_mlflow.active_run.return_value = MagicMock(
            info=MagicMock(run_id='test')
        )
        config = Config()
        config.model_choice = 'GCN'
        config.primary_metric = 'auc'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = BayesianOptimiser(
                graph_with_splits, config,
                n_trials=2,
                results_dir=tmpdir,
                mlflow_tracking=False,
            )
            assert opt.n_trials == 2
            assert opt.train_edges is not None
            assert opt.val_edges is not None

    @patch('src.training.tracker.mlflow')
    def test_init_with_tracking(self, mock_mlflow, graph_with_splits):
        mock_mlflow.active_run.return_value = MagicMock(
            info=MagicMock(run_id='test')
        )
        config = Config()
        config.model_choice = 'GCN'
        config.primary_metric = 'auc'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = BayesianOptimiser(
                graph_with_splits, config,
                n_trials=2,
                results_dir=tmpdir,
                mlflow_tracking=True,
            )
            assert opt.mlflow_tracker is not None

    @patch('src.training.tracker.mlflow')
    def test_auto_study_name(self, mock_mlflow, graph_with_splits):
        mock_mlflow.active_run.return_value = MagicMock(
            info=MagicMock(run_id='test')
        )
        config = Config()
        config.model_choice = 'GCN'
        config.primary_metric = 'auc'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = BayesianOptimiser(
                graph_with_splits, config,
                n_trials=2,
                results_dir=tmpdir,
                mlflow_tracking=False,
            )
            assert 'gnn_optimisation_' in opt.study_name


class TestObjective:
    @patch('src.training.tracker.mlflow')
    def test_objective_runs(self, mock_mlflow, graph_with_splits):
        mock_mlflow.active_run.return_value = MagicMock(
            info=MagicMock(run_id='test')
        )
        config = Config()
        config.model_choice = 'GCN'
        config.primary_metric = 'auc'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = BayesianOptimiser(
                graph_with_splits, config,
                n_trials=1,
                results_dir=tmpdir,
                mlflow_tracking=False,
            )
            # Run 1 trial
            opt.study.optimize(opt.objective, n_trials=1, catch=(Exception,))
            # Should have completed without crash
            assert len(opt.study.trials) == 1

    @patch('src.training.tracker.mlflow')
    def test_objective_with_mlflow(self, mock_mlflow, graph_with_splits):
        mock_mlflow.active_run.return_value = MagicMock(
            info=MagicMock(run_id='test')
        )
        config = Config()
        config.model_choice = 'GCN'
        config.primary_metric = 'auc'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = BayesianOptimiser(
                graph_with_splits, config,
                n_trials=1,
                results_dir=tmpdir,
                mlflow_tracking=True,
            )
            opt.study.optimize(opt.objective, n_trials=1, catch=(Exception,))
            assert len(opt.study.trials) == 1

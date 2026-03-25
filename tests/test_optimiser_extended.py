"""
Tests for src/training/optimiser.py — optimise, save, and export methods.
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from torch_geometric.data import Data

from src.training.optimiser import BayesianOptimiser
from src.config import Config
import optuna


@pytest.fixture
def mock_optimiser():
    """Create a BayesianOptimiser with mocked __init__."""
    config = Config()
    config.model_choice = 'GCN'
    config.primary_metric = 'auc'

    # Create a small synthetic graph
    num_nodes = 20
    x = torch.randn(num_nodes, 16)
    edge_index = torch.tensor(
        [[i, i+1, i+1, i] for i in range(num_nodes-1)], dtype=torch.long
    ).reshape(-1, 2).t().contiguous()
    
    graph = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    graph.train_edge_index = torch.randint(0, num_nodes, (100, 2))
    graph.train_edge_label = torch.cat([torch.ones(50), torch.zeros(50)])
    graph.val_edge_index = torch.randint(0, num_nodes, (40, 2))
    graph.val_edge_label = torch.cat([torch.ones(20), torch.zeros(20)])

    with patch.object(BayesianOptimiser, '__init__', lambda self, *a, **kw: None):
        opt = BayesianOptimiser.__new__(BayesianOptimiser)
        opt.graph = graph
        opt.config = config
        opt.n_trials = 2
        opt.device = 'cpu'
        opt.mlflow_tracker = None
        opt.mlflow_tracking = False
        opt.results_dir = Path(tempfile.mkdtemp())
        opt.study_name = 'test_study'
        opt.train_edges = graph.train_edge_index
        opt.train_labels = graph.train_edge_label
        opt.val_edges = graph.val_edge_index
        opt.val_labels = graph.val_edge_label
        opt.study = optuna.create_study(
            study_name='test_study',
            direction='maximize',
        )
        
    return opt


class TestSaveResults:
    def test_saves_json_and_csv(self, mock_optimiser):
        # Create a fake trial
        mock_optimiser.study.optimize(
            lambda trial: trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            n_trials=2,
        )
        best_trial = mock_optimiser.study.best_trial
        results = mock_optimiser._save_results(best_trial)
        
        assert isinstance(results, dict)
        assert 'best_value' in results
        assert 'best_params' in results
        
        # Check files created
        json_files = list(mock_optimiser.results_dir.glob("best_params_*.json"))
        csv_files = list(mock_optimiser.results_dir.glob("optimisation_history_*.csv"))
        assert len(json_files) >= 1
        assert len(csv_files) >= 1
        
        # Validate JSON content
        with open(json_files[0]) as f:
            saved = json.load(f)
        assert 'best_value' in saved


class TestCreateVisualisations:
    def test_creates_plot(self, mock_optimiser):
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Run some trials first
        mock_optimiser.study.optimize(
            lambda trial: trial.suggest_float('lr', 1e-4, 1e-2, log=True) * -1 + 1,
            n_trials=3,
        )
        mock_optimiser._create_visualisations()
        
        # Check plot file
        png_files = list(mock_optimiser.results_dir.glob("optimisation_plots_*.png"))
        assert len(png_files) >= 1


class TestOptimise:
    def test_optimise_flow(self, mock_optimiser):
        """Test the full optimise flow with mocked objective."""
        import matplotlib
        matplotlib.use('Agg')
        
        # Replace objective with a simple function
        mock_optimiser.objective = lambda trial: trial.suggest_float('x', 0, 1)
        
        results = mock_optimiser.optimise()
        assert isinstance(results, dict)
        assert 'best_value' in results


class TestGetBestConfig:
    def test_returns_params(self, mock_optimiser):
        mock_optimiser.study.optimize(
            lambda trial: trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            n_trials=2,
        )
        params = mock_optimiser.get_best_config()
        assert isinstance(params, dict)
        assert 'lr' in params


class TestApplyBestParams:
    def test_updates_config(self, mock_optimiser):
        mock_optimiser.study.optimize(
            lambda trial: trial.suggest_int('hidden_channels', 32, 128),
            n_trials=2,
        )
        config = mock_optimiser.apply_best_params_to_config()
        assert 'hidden_channels' in config.model_config

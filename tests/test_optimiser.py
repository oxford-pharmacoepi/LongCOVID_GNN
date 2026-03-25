"""
Unit tests for src/training/optimiser.py

Tests the BayesianOptimiser helper methods with mocked dependencies.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from torch_geometric.data import Data

from src.training.optimiser import BayesianOptimiser
from src.config import Config


# We need to mock the heavy __init__ since it requires a real graph with splits
@pytest.fixture
def mock_optimiser():
    """Create a BayesianOptimiser with mocked __init__."""
    config = Config()
    config.model_choice = 'GCN'
    config.primary_metric = 'auc'

    # Create a small synthetic graph with train/val splits
    num_nodes = 20
    x = torch.randn(num_nodes, 16)
    edge_index = torch.tensor(
        [[i, i+1, i+1, i] for i in range(num_nodes-1)], dtype=torch.long
    ).reshape(-1, 2).t().contiguous()
    
    graph = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    # Add train/val splits
    graph.train_edge_index = torch.randint(0, num_nodes, (100, 2))
    graph.train_edge_label = torch.cat([torch.ones(50), torch.zeros(50)])
    graph.val_edge_index = torch.randint(0, num_nodes, (40, 2))
    graph.val_edge_label = torch.cat([torch.ones(20), torch.zeros(20)])

    with patch.object(BayesianOptimiser, '__init__', lambda self, *a, **kw: None):
        opt = BayesianOptimiser.__new__(BayesianOptimiser)
        opt.graph = graph
        opt.config = config
        opt.n_trials = 3
        opt.device = 'cpu'
        opt.mlflow_tracker = None
        opt.mlflow_tracking = False
        opt.results_dir = '/tmp/test_optum'
        opt.study_name = 'test'
        opt.train_edges = graph.train_edge_index
        opt.train_labels = graph.train_edge_label
        opt.val_edges = graph.val_edge_index
        opt.val_labels = graph.val_edge_label
        
    return opt


class TestDefineSearchSpace:
    def test_returns_dict(self, mock_optimiser):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        params = mock_optimiser.define_search_space(trial)
        assert isinstance(params, dict)
        assert 'learning_rate' in params
        assert 'hidden_channels' in params
        assert 'num_layers' in params
        assert 'dropout_rate' in params

    def test_transformer_params(self, mock_optimiser):
        import optuna
        mock_optimiser.config.model_choice = 'Transformer'
        study = optuna.create_study()
        trial = study.ask()
        params = mock_optimiser.define_search_space(trial)
        assert 'heads' in params
        assert 'concat' in params

    def test_gcn_no_heads(self, mock_optimiser):
        import optuna
        mock_optimiser.config.model_choice = 'GCN'
        study = optuna.create_study()
        trial = study.ask()
        params = mock_optimiser.define_search_space(trial)
        assert 'heads' not in params


class TestCreateModel:
    def test_gcn(self, mock_optimiser):
        params = {
            'hidden_channels': 32,
            'out_channels': 16,
            'num_layers': 2,
            'dropout_rate': 0.3,
        }
        mock_optimiser.config.model_choice = 'GCN'
        model = mock_optimiser._create_model(params)
        assert model is not None

    def test_transformer(self, mock_optimiser):
        params = {
            'hidden_channels': 32,
            'out_channels': 16,
            'num_layers': 2,
            'dropout_rate': 0.3,
            'heads': 2,
            'concat': False,
        }
        mock_optimiser.config.model_choice = 'Transformer'
        model = mock_optimiser._create_model(params)
        assert model is not None

    def test_sage(self, mock_optimiser):
        params = {
            'hidden_channels': 32,
            'out_channels': 16,
            'num_layers': 2,
            'dropout_rate': 0.3,
        }
        mock_optimiser.config.model_choice = 'SAGE'
        model = mock_optimiser._create_model(params)
        assert model is not None

    def test_default_model(self, mock_optimiser):
        params = {
            'hidden_channels': 32,
            'out_channels': 16,
            'num_layers': 2,
            'dropout_rate': 0.3,
            'heads': 2,
            'concat': False,
        }
        mock_optimiser.config.model_choice = 'UnknownModel'
        model = mock_optimiser._create_model(params)
        assert model is not None  # defaults to Transformer


class TestTrainEpoch:
    def test_returns_float(self, mock_optimiser):
        from src.models import GCNModel
        model = GCNModel(in_channels=16, hidden_channels=32, out_channels=16, num_layers=2)
        model.train()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = mock_optimiser._train_epoch(model, optimiser, batch_size=64)
        assert isinstance(loss, float)
        assert loss >= 0.0


class TestValidateEpoch:
    def test_returns_float(self, mock_optimiser):
        from src.models import GCNModel
        model = GCNModel(in_channels=16, hidden_channels=32, out_channels=16, num_layers=2)
        model.eval()
        metric = mock_optimiser._validate_epoch(model, batch_size=64)
        assert isinstance(metric, float)

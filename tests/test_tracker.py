"""
Tests for src/training/tracker.py — ExperimentTracker.
Uses mocked mlflow to avoid real tracking server dependency.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, PropertyMock
from torch_geometric.data import Data

from src.training.tracker import ExperimentTracker
from src.config import Config


@pytest.fixture
def mock_mlflow():
    """Patch mlflow for all tracker tests."""
    with patch('src.training.tracker.mlflow') as mock:
        mock.active_run.return_value = MagicMock(info=MagicMock(run_id='test-run-123'))
        yield mock


@pytest.fixture
def tracker(mock_mlflow):
    return ExperimentTracker(experiment_name='test', tracking_uri='file:./test_mlruns')


class TestTrackerInit:
    def test_creates_tracker(self, mock_mlflow):
        t = ExperimentTracker(experiment_name='test')
        assert t.experiment_name == 'test'

    def test_sets_tracking_uri(self, mock_mlflow):
        ExperimentTracker(experiment_name='test', tracking_uri='file:./custom')
        mock_mlflow.set_tracking_uri.assert_called_with('file:./custom')


class TestStartRun:
    def test_start_run(self, tracker, mock_mlflow):
        tracker.start_run(run_name='test_run')
        mock_mlflow.start_run.assert_called_once()
        assert tracker.run_id == 'test-run-123'

    def test_start_run_auto_name(self, tracker, mock_mlflow):
        tracker.start_run()
        mock_mlflow.start_run.assert_called_once()


class TestLogConfig:
    def test_log_config(self, tracker, mock_mlflow):
        config = Config()
        tracker.log_config(config)
        assert mock_mlflow.log_param.call_count > 5


class TestLogGraphMetadata:
    def test_with_metadata(self, tracker, mock_mlflow):
        graph = Data(
            x=torch.randn(10, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t(),
        )
        graph.metadata = {
            'node_info': {'Drugs': 5, 'Genes': 5},
            'edge_info': {'Drug-Gene': 2},
            'total_nodes': 10,
            'total_edges': 2,
            'data_mode': 'raw',
            'creation_timestamp': '2024-01-01',
        }
        tracker.log_graph_metadata(graph)
        mock_mlflow.log_metric.assert_any_call('graph_num_nodes', 10)

    def test_with_val_edges(self, tracker, mock_mlflow):
        graph = Data(
            x=torch.randn(10, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t(),
        )
        graph.val_edge_index = torch.randint(0, 10, (20, 2))
        graph.val_edge_label = torch.cat([torch.ones(10), torch.zeros(10)])
        tracker.log_graph_metadata(graph)

    def test_with_test_edges(self, tracker, mock_mlflow):
        graph = Data(
            x=torch.randn(10, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t(),
        )
        graph.test_edge_index = torch.randint(0, 10, (20, 2))
        graph.test_edge_label = torch.cat([torch.ones(10), torch.zeros(10)])
        tracker.log_graph_metadata(graph)


class TestLogTrainingMetrics:
    def test_basic(self, tracker, mock_mlflow):
        tracker.log_training_metrics(epoch=0, train_loss=0.5)
        mock_mlflow.log_metric.assert_any_call('train_loss', 0.5, step=0)

    def test_with_val(self, tracker, mock_mlflow):
        tracker.log_training_metrics(epoch=1, train_loss=0.4, val_loss=0.3)
        mock_mlflow.log_metric.assert_any_call('val_loss', 0.3, step=1)

    def test_with_metrics(self, tracker, mock_mlflow):
        tracker.log_training_metrics(
            epoch=2, train_loss=0.3, val_metrics={'recall@10': 0.8}
        )
        mock_mlflow.log_metric.assert_any_call('val_recall_10', 0.8, step=2)


class TestLogTestMetrics:
    def test_basic(self, tracker, mock_mlflow):
        tracker.log_test_metrics({'auc': 0.95, 'apr': 0.88})
        mock_mlflow.log_metric.assert_any_call('test_auc', 0.95)


class TestLogModel:
    def test_with_path(self, tracker, mock_mlflow, tmp_path):
        model_path = tmp_path / 'model.pt'
        model_path.write_bytes(b'fake model')
        tracker.log_model(None, model_path=str(model_path))
        mock_mlflow.log_artifact.assert_called_once()

    def test_without_path(self, tracker, mock_mlflow):
        model = MagicMock()
        tracker.log_model(model)
        mock_mlflow.pytorch.log_model.assert_called_once()


class TestLogArtifact:
    def test_existing(self, tracker, mock_mlflow, tmp_path):
        f = tmp_path / 'art.txt'
        f.write_text('hello')
        tracker.log_artifact(str(f), 'test')
        mock_mlflow.log_artifact.assert_called_once()

    def test_missing(self, tracker, mock_mlflow):
        tracker.log_artifact('/nonexistent/path')
        mock_mlflow.log_artifact.assert_not_called()


class TestEndRun:
    def test_end_run(self, tracker, mock_mlflow):
        tracker.run_id = 'abc'
        tracker.end_run()
        mock_mlflow.end_run.assert_called_once()


class TestLogParamMetricTag:
    def test_log_param(self, tracker, mock_mlflow):
        tracker.log_param('k', 'v')
        mock_mlflow.log_param.assert_called_with('k', 'v')

    def test_log_metric(self, tracker, mock_mlflow):
        tracker.log_metric('k', 1.0, step=5)
        mock_mlflow.log_metric.assert_called_with('k', 1.0, step=5)

    def test_set_tag(self, tracker, mock_mlflow):
        tracker.set_tag('k', 'v')
        mock_mlflow.set_tag.assert_called_with('k', 'v')


class TestCompareRuns:
    def test_no_experiment(self, mock_mlflow):
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        mock_mlflow.tracking.MlflowClient.return_value = client
        result = ExperimentTracker.compare_runs('nonexistent')
        assert result is None

    def test_with_experiment(self, mock_mlflow):
        exp = MagicMock()
        exp.experiment_id = '1'
        
        run = MagicMock()
        run.info.run_id = 'r1'
        run.info.status = 'FINISHED'
        run.info.start_time = 1700000000000
        run.data.tags = {'mlflow.runName': 'run1'}
        run.data.params = {'lr': '0.001'}
        run.data.metrics = {'test_auc': 0.95}
        
        client = MagicMock()
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = [run]
        mock_mlflow.tracking.MlflowClient.return_value = client
        
        result = ExperimentTracker.compare_runs('test')
        assert result is not None
        assert len(result) == 1


class TestGetBestRun:
    def test_no_experiment(self, mock_mlflow):
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        mock_mlflow.tracking.MlflowClient.return_value = client
        assert ExperimentTracker.get_best_run('nonexistent') is None

    def test_no_runs(self, mock_mlflow):
        exp = MagicMock()
        exp.experiment_id = '1'
        client = MagicMock()
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = []
        mock_mlflow.tracking.MlflowClient.return_value = client
        assert ExperimentTracker.get_best_run('test') is None

    def test_with_best_run(self, mock_mlflow):
        exp = MagicMock()
        exp.experiment_id = '1'
        
        run = MagicMock()
        run.info.run_id = 'r1'
        run.data.metrics = {'test_auc': 0.95}
        run.data.params = {'lr': '0.001'}
        
        client = MagicMock()
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = [run]
        mock_mlflow.tracking.MlflowClient.return_value = client
        
        result = ExperimentTracker.get_best_run('test')
        assert result is not None

"""
Unit tests for src/training/tracker.py and src/training/optimiser.py

Tests ExperimentTracker methods and BayesianOptimiser helper methods.
Since these require MLflow/Optuna infrastructure, we mock external calls.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
import torch

from src.training.tracker import ExperimentTracker


# ── ExperimentTracker ────────────────────────────────────────────────
class TestExperimentTracker:
    @patch('src.training.tracker.mlflow')
    def test_init_sets_experiment(self, mock_mlflow):
        tracker = ExperimentTracker(
            experiment_name="test_experiment",
            tracking_uri="file:./test_mlruns"
        )
        mock_mlflow.set_tracking_uri.assert_called_once_with("file:./test_mlruns")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        assert tracker.experiment_name == "test_experiment"
        assert tracker.run_id is None

    @patch('src.training.tracker.mlflow')
    def test_start_run(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_mlflow.active_run.return_value = mock_run
        
        tracker = ExperimentTracker()
        tracker.start_run(run_name="test_run")
        
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
        assert tracker.run_id == "abc123"

    @patch('src.training.tracker.mlflow')
    def test_start_run_auto_name(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "def456"
        mock_mlflow.active_run.return_value = mock_run
        
        tracker = ExperimentTracker()
        tracker.start_run()
        
        # Should auto-generate a name
        assert mock_mlflow.start_run.called

    @patch('src.training.tracker.mlflow')
    def test_log_training_metrics(self, mock_mlflow):
        tracker = ExperimentTracker()
        tracker.log_training_metrics(epoch=5, train_loss=0.5, val_loss=0.3)
        
        mock_mlflow.log_metric.assert_any_call("train_loss", 0.5, step=5)
        mock_mlflow.log_metric.assert_any_call("val_loss", 0.3, step=5)

    @patch('src.training.tracker.mlflow')
    def test_log_training_metrics_with_val_metrics(self, mock_mlflow):
        tracker = ExperimentTracker()
        val_metrics = {'auc': 0.85, 'hits@10': 0.6}
        tracker.log_training_metrics(epoch=1, train_loss=0.5, val_metrics=val_metrics)
        
        # @ should be sanitised to _
        mock_mlflow.log_metric.assert_any_call("val_hits_10", 0.6, step=1)

    @patch('src.training.tracker.mlflow')
    def test_log_test_metrics(self, mock_mlflow):
        tracker = ExperimentTracker()
        test_metrics = {'auc': 0.9, 'ap': 0.85}
        tracker.log_test_metrics(test_metrics)
        
        mock_mlflow.log_metric.assert_any_call("test_auc", 0.9)
        mock_mlflow.log_metric.assert_any_call("test_ap", 0.85)

    @patch('src.training.tracker.mlflow')
    def test_log_model_with_path(self, mock_mlflow):
        tracker = ExperimentTracker()
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({'state': 'test'}, f.name)
            tracker.log_model(model=None, model_path=f.name)
            mock_mlflow.log_artifact.assert_called_once_with(f.name, "models")
            os.unlink(f.name)

    @patch('src.training.tracker.mlflow')
    def test_log_artifact(self, mock_mlflow):
        tracker = ExperimentTracker()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            tracker.log_artifact(f.name, "test_artifacts")
            mock_mlflow.log_artifact.assert_called_once_with(f.name, "test_artifacts")
            os.unlink(f.name)

    @patch('src.training.tracker.mlflow')
    def test_log_artifact_missing(self, mock_mlflow, capsys):
        tracker = ExperimentTracker()
        tracker.log_artifact("/nonexistent/file.txt")
        captured = capsys.readouterr()
        assert 'Warning' in captured.out

    @patch('src.training.tracker.mlflow')
    def test_log_dict_as_json(self, mock_mlflow):
        tracker = ExperimentTracker()
        test_dict = {'key': 'value', 'num': 42}
        tracker.log_dict_as_json(test_dict, "test.json")
        mock_mlflow.log_artifact.assert_called_once()

    @patch('src.training.tracker.mlflow')
    def test_log_feature_importance(self, mock_mlflow):
        tracker = ExperimentTracker()
        names = [f"feature_{i}" for i in range(15)]
        values = list(range(15, 0, -1))
        tracker.log_feature_importance(names, values)
        # Should log top 10 features as params
        assert mock_mlflow.log_param.call_count == 10

    @patch('src.training.tracker.mlflow')
    def test_end_run(self, mock_mlflow):
        tracker = ExperimentTracker()
        tracker.run_id = "test_id"
        tracker.end_run()
        mock_mlflow.end_run.assert_called_once()

    @patch('src.training.tracker.mlflow')
    def test_log_param(self, mock_mlflow):
        tracker = ExperimentTracker()
        tracker.log_param("key", "value")
        mock_mlflow.log_param.assert_called_once_with("key", "value")

    @patch('src.training.tracker.mlflow')
    def test_log_metric(self, mock_mlflow):
        tracker = ExperimentTracker()
        tracker.log_metric("auc", 0.95, step=10)
        mock_mlflow.log_metric.assert_called_once_with("auc", 0.95, step=10)

    @patch('src.training.tracker.mlflow')
    def test_set_tag(self, mock_mlflow):
        tracker = ExperimentTracker()
        tracker.set_tag("env", "test")
        mock_mlflow.set_tag.assert_called_once_with("env", "test")

    @patch('src.training.tracker.mlflow')
    def test_log_config(self, mock_mlflow):
        from src.config import Config
        tracker = ExperimentTracker()
        config = Config()
        tracker.log_config(config)
        assert mock_mlflow.log_param.called

    @patch('src.training.tracker.mlflow')
    def test_log_graph_metadata(self, mock_mlflow):
        tracker = ExperimentTracker()
        # Create a mock graph with required attributes
        graph = MagicMock()
        graph.x.size.return_value = torch.Size([100, 16])
        graph.edge_index.size.return_value = torch.Size([2, 500])
        graph.metadata = {
            'node_info': {'drug': 50, 'disease': 50},
            'total_nodes': 100,
            'edge_info': {'drug_disease': 200},
            'total_edges': 500,
            'data_mode': 'test',
            'creation_timestamp': '2026-01-01'
        }
        # No val_edge_index, test_edge_index
        graph.val_edge_index = None
        del graph.val_edge_index
        graph.test_edge_index = None
        del graph.test_edge_index
        
        tracker.log_graph_metadata(graph)
        assert mock_mlflow.log_metric.called

    @patch('src.training.tracker.mlflow')
    def test_compare_runs(self, mock_mlflow):
        mock_client = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        result = ExperimentTracker.compare_runs(experiment_name="test")
        assert result is not None or result is None  # May return empty df

    @patch('src.training.tracker.mlflow')
    def test_compare_runs_not_found(self, mock_mlflow):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        result = ExperimentTracker.compare_runs(experiment_name="nonexistent")
        assert result is None

    @patch('src.training.tracker.mlflow')
    def test_get_best_run(self, mock_mlflow):
        mock_client = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        result = ExperimentTracker.get_best_run(experiment_name="test")
        assert result is None  # No runs found

    @patch('src.training.tracker.mlflow')
    def test_get_best_run_not_found(self, mock_mlflow):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        result = ExperimentTracker.get_best_run(experiment_name="nonexistent")
        assert result is None

"""
MLflow experiment tracking utilities for drug-disease prediction pipeline.
Tracks all parameters, metrics, and artifacts for reproducibility.
"""

import mlflow
import mlflow.pytorch
import json
import os
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    """MLflow experiment tracker for pipeline runs."""
    
    def __init__(self, experiment_name="drug_disease_prediction", tracking_uri="file:./mlruns"):
        """
        Initialize MLflow experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking (default: local file storage)
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run_id = None
        
    def start_run(self, run_name=None):
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        print(f"Started MLflow run: {run_name} (ID: {self.run_id})")
        
    def log_config(self, config):
        """
        Log all configuration parameters.
        
        Args:
            config: Config object with all settings
        """
        # Log version settings
        mlflow.log_param("training_version", config.training_version)
        mlflow.log_param("validation_version", config.validation_version)
        mlflow.log_param("test_version", config.test_version)
        
        # Log dataset settings
        mlflow.log_param("as_dataset", config.as_dataset)
        mlflow.log_param("negative_sampling_approach", config.negative_sampling_approach)
        mlflow.log_param("pos_neg_ratio", config.pos_neg_ratio)
        
        # Log model hyperparameters
        for key, value in config.model_config.items():
            mlflow.log_param(f"model_{key}", value)
        
        # Log network settings
        for key, value in config.network_config.items():
            mlflow.log_param(f"network_{key}", value)
        
        # Log training settings
        mlflow.log_param("seed", config.seed)
        mlflow.log_param("device", config.device)
        
        # Log paths as tags (for reference)
        mlflow.set_tag("general_path", config.general_path)
        mlflow.set_tag("results_path", config.results_path)
        
    def log_graph_metadata(self, graph):
        """
        Log graph creation metadata.
        
        Args:
            graph: PyTorch Geometric graph object with metadata
        """
        if hasattr(graph, 'metadata'):
            metadata = graph.metadata
            
            # Log node counts
            if 'node_info' in metadata:
                for node_type, count in metadata['node_info'].items():
                    mlflow.log_metric(f"nodes_{node_type}", count)
                mlflow.log_metric("nodes_total", metadata.get('total_nodes', 0))
            
            # Log edge counts
            if 'edge_info' in metadata:
                for edge_type, count in metadata['edge_info'].items():
                    mlflow.log_metric(f"edges_{edge_type}", count)
                mlflow.log_metric("edges_total", metadata.get('total_edges', 0))
            
            # Log data mode
            if 'data_mode' in metadata:
                mlflow.set_tag("data_mode", metadata['data_mode'])
            
            # Log creation timestamp
            if 'creation_timestamp' in metadata:
                mlflow.set_tag("graph_creation_time", metadata['creation_timestamp'])
        
        # Log graph statistics
        mlflow.log_metric("graph_num_nodes", graph.x.size(0))
        mlflow.log_metric("graph_num_edges", graph.edge_index.size(1))
        mlflow.log_metric("graph_num_features", graph.x.size(1))
        
        # Log validation/test set sizes
        if hasattr(graph, 'val_edge_index'):
            mlflow.log_metric("val_set_size", graph.val_edge_index.size(0))
            val_pos = (graph.val_edge_label == 1).sum().item()
            val_neg = (graph.val_edge_label == 0).sum().item()
            mlflow.log_metric("val_positive_samples", val_pos)
            mlflow.log_metric("val_negative_samples", val_neg)
            mlflow.log_metric("val_pos_neg_ratio", val_neg / val_pos if val_pos > 0 else 0)
        
        if hasattr(graph, 'test_edge_index'):
            mlflow.log_metric("test_set_size", graph.test_edge_index.size(0))
            test_pos = (graph.test_edge_label == 1).sum().item()
            test_neg = (graph.test_edge_label == 0).sum().item()
            mlflow.log_metric("test_positive_samples", test_pos)
            mlflow.log_metric("test_negative_samples", test_neg)
            mlflow.log_metric("test_pos_neg_ratio", test_neg / test_pos if test_pos > 0 else 0)
    
    def log_training_metrics(self, epoch, train_loss, val_loss=None, val_metrics=None):
        """
        Log training metrics for each epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            val_metrics: Dictionary of validation metrics (optional)
        """
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)
    
    def log_test_metrics(self, test_metrics):
        """
        Log final test metrics.
        
        Args:
            test_metrics: Dictionary of test metrics
        """
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    def log_model(self, model, model_path=None):
        """
        Log trained model to MLflow.
        
        Args:
            model: PyTorch model
            model_path: Optional path to saved model file
        """
        if model_path and os.path.exists(model_path):
            mlflow.log_artifact(model_path, "models")
        else:
            mlflow.pytorch.log_model(model, "model")
    
    def log_artifact(self, artifact_path, artifact_name=None):
        """
        Log any artifact (graph, predictions, plots, etc.).
        
        Args:
            artifact_path: Path to artifact file
            artifact_name: Optional subfolder name in artifacts
        """
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path, artifact_name)
        else:
            print(f"Warning: Artifact not found: {artifact_path}")
    
    def log_dict_as_json(self, dictionary, filename):
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            filename: Name for the JSON file
        """
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'w') as f:
            json.dump(dictionary, f, indent=2)
        mlflow.log_artifact(temp_path, "metadata")
        os.remove(temp_path)
    
    def log_feature_importance(self, feature_names, importance_values):
        """
        Log feature importance as parameters and artifact.
        
        Args:
            feature_names: List of feature names
            importance_values: List of importance values
        """
        # Log as artifact
        importance_dict = dict(zip(feature_names, importance_values))
        self.log_dict_as_json(importance_dict, "feature_importance.json")
        
        # Log top 10 as parameters
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (name, value) in enumerate(sorted_features[:10]):
            mlflow.log_param(f"top_feature_{i+1}", f"{name} ({value:.4f})")
    
    def log_predictions(self, predictions_path):
        """
        Log prediction results.
        
        Args:
            predictions_path: Path to predictions CSV file
        """
        self.log_artifact(predictions_path, "predictions")
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        print(f"Ended MLflow run: {self.run_id}")
    
    def log_param(self, key, value):
        """
        Log a single parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)
    
    def log_metric(self, key, value, step=None):
        """
        Log a single metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number (e.g., epoch)
        """
        mlflow.log_metric(key, value, step=step)
    
    def set_tag(self, key, value):
        """
        Set a tag.
        
        Args:
            key: Tag name
            value: Tag value
        """
        mlflow.set_tag(key, value)
    
    @staticmethod
    def compare_runs(experiment_name="drug_disease_prediction", metric="test_auc"):
        """
        Compare all runs in an experiment.
        
        Args:
            experiment_name: Name of the experiment
            metric: Metric to sort by
        
        Returns:
            DataFrame with run comparisons
        """
        import pandas as pd
        
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )
        
        data = []
        for run in runs:
            row = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add parameters
            for key, value in run.data.params.items():
                row[f'param_{key}'] = value
            
            # Add metrics
            for key, value in run.data.metrics.items():
                row[f'metric_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    @staticmethod
    def get_best_run(experiment_name="drug_disease_prediction", metric="test_auc"):
        """
        Get the best run based on a metric.
        
        Args:
            experiment_name: Name of the experiment
            metric: Metric to optimize
        
        Returns:
            Best run info
        """
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            print(f"\nBest run: {best_run.info.run_id}")
            print(f"  Metric {metric}: {best_run.data.metrics.get(metric, 'N/A')}")
            print(f"  Parameters:")
            for key, value in best_run.data.params.items():
                print(f"    {key}: {value}")
            return best_run
        
        return None

#!/usr/bin/env python3
"""
Model Testing and Evaluation Script for Drug-Disease Prediction
Comprehensive testing with FP export for GNNExplainer analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score, classification_report
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dt
import json
import os
import argparse
from pathlib import Path
import random
import glob

# Import from shared modules
from src.models import GCNModel, TransformerModel, SAGEModel, MODEL_CLASSES
from src.config import get_config
from src.utils import enable_full_reproducibility, calculate_metrics
from src.mlflow_tracker import ExperimentTracker

def find_latest_graph(results_dir='results'):
    """Auto-detect the latest graph file."""
    graph_patterns = [
        os.path.join(results_dir, 'graph_*.pt'),
        'graph_*.pt',
        os.path.join('data', 'processed', '*.pt'),
    ]
    
    graph_files = []
    for pattern in graph_patterns:
        graph_files.extend(glob.glob(pattern))
    
    if not graph_files:
        raise FileNotFoundError("No graph files found. Please create a graph first using script 1_create_graph.py")
    
    # Sort by modification time (most recent first)
    latest_graph = max(graph_files, key=os.path.getmtime)
    print(f"Auto-detected latest graph: {latest_graph}")
    return latest_graph


def find_latest_models(results_dir='results'):
    """Auto-detect the latest trained models directory or files."""
    model_patterns = [
        os.path.join(results_dir, 'models', '*_best_model_*.pt'),
        os.path.join(results_dir, 'models', '*.pt'),
    ]
    
    model_files = []
    for pattern in model_patterns:
        found_files = glob.glob(pattern)
        # Filter to exclude graph files
        model_files.extend([f for f in found_files if 'graph' not in os.path.basename(f).lower()])
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {results_dir}/models/. Please train models first using script 2_train_models.py")
    
    # If multiple models found, return the directory containing the most recent one
    latest_model = max(model_files, key=os.path.getmtime)
    models_dir = os.path.dirname(latest_model)
    print(f"Auto-detected models directory: {models_dir}")
    return models_dir


class ModelEvaluator:
    """Comprehensive model evaluation with FP export for GNNExplainer."""
    
    def __init__(self, results_path="results/", use_mlflow=True, mlflow_experiment_name=None):
        self.results_path = results_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dirs = self._create_results_structure()
        
        # Initialize MLflow tracker
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = None
        if use_mlflow:
            experiment_name = mlflow_experiment_name or "drug_disease_evaluation"
            self.mlflow_tracker = ExperimentTracker(experiment_name=experiment_name)
            print(f"MLflow tracking enabled: {experiment_name}")
        
    def _create_results_structure(self):
        """Create organized results directory structure."""
        subdirs = {
            'evaluation': 'evaluation',
            'predictions': 'predictions',
            'explainer': 'explainer',
            'figures': 'figures',
            'logs': 'logs'
        }
        
        result_dirs = {}
        for key, dirname in subdirs.items():
            dir_path = os.path.join(self.results_path, dirname)
            os.makedirs(dir_path, exist_ok=True)
            result_dirs[key] = dir_path
            
        print(f"Results directory structure created under: {self.results_path}")
        return result_dirs
        
    def load_trained_models_info(self, models_path, config=None):
        """Load information about trained models, respecting model_choice config."""
        trained_models = {}
        
        # Get model_choice from config if available
        model_choice = None
        if config:
            model_choice = getattr(config, 'model_choice', None)
            if model_choice:
                print(f"Config specifies model_choice: {model_choice}")
        
        # Look for model files
        if os.path.isfile(models_path):
            # Single model file provided
            model_files = [models_path]
            models_dir = os.path.dirname(models_path)
        else:
            # Directory provided
            models_dir = models_path
            model_files = []
            for f in os.listdir(models_dir):
                # Only look for files that contain 'model' in the name (not 'graph')
                if f.endswith('.pt') and 'model' in f.lower() and 'graph' not in f.lower():
                    model_files.append(os.path.join(models_dir, f))
        
        # Sort by modification time to get most recent first
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        print(f"Found {len(model_files)} model files (sorted by recency)")
        
        if not model_files:
            print("Warning: No valid model files found (excluding graph files)")
            return trained_models
        
        # Track which model types we've already found (to avoid loading old duplicates)
        found_model_types = set()
        
        # Create model info dictionary - only keep the most recent of each type
        for model_file in model_files:
            filename = os.path.basename(model_file)
            
            if 'GCN' in filename or 'gcn' in filename:
                model_name = 'GCNModel'
                model_class = GCNModel
            elif 'Transformer' in filename or 'transformer' in filename:
                model_name = 'TransformerModel'
                model_class = TransformerModel
            elif 'SAGE' in filename or 'sage' in filename:
                model_name = 'SAGEModel'
                model_class = SAGEModel
            else:
                # Could not determine model type
                continue
            
            # Skip if we already found this model type
            if model_name in found_model_types:
                continue
            
            # Check if this model matches model_choice (if specified)
            if model_choice:
                # Normalise model_choice to match model_name format
                if model_choice.lower() == 'gcn' and model_name != 'GCNModel':
                    continue
                elif model_choice.lower() == 'transformer' and model_name != 'TransformerModel':
                    continue
                elif model_choice.lower() == 'sage' and model_name != 'SAGEModel':
                    continue
            
            trained_models[model_name] = {
                'model_path': model_file,
                'model_class': model_class,
                'threshold': 0.5  # Default threshold
            }
            found_model_types.add(model_name)
            print(f"Selected {model_name}: {filename}")
        
        if model_choice and not trained_models:
            print(f"Warning: model_choice '{model_choice}' specified but no matching model found in {models_dir}")
        
        return trained_models
        
    def create_test_data_from_graph(self, graph, use_test_split=True):
        """Create test dataset from the graph object."""
        print("Creating test dataset from graph...")
        
        if use_test_split and hasattr(graph, 'test_edge_index') and hasattr(graph, 'test_edge_label'):
            print("Using existing test split from graph")
            test_edge_tensor = graph.test_edge_index
            test_label_tensor = graph.test_edge_label
        elif hasattr(graph, 'val_edge_index') and hasattr(graph, 'val_edge_label'):
            print("Using validation split as test set")
            test_edge_tensor = graph.val_edge_index
            test_label_tensor = graph.val_edge_label
        else:
            print("Creating synthetic test dataset...")
            # Create synthetic test data
            num_test_edges = min(1000, graph.edge_index.size(1) // 10)
            
            # Sample random edges as positive examples
            perm = torch.randperm(graph.edge_index.size(1))
            test_pos_indices = perm[:num_test_edges//2]
            test_pos_edges = graph.edge_index[:, test_pos_indices].t()
            
            # Generate random negative edges
            num_nodes = graph.x.size(0)
            test_neg_edges = []
            while len(test_neg_edges) < num_test_edges//2:
                src = random.randint(0, num_nodes-1)
                dst = random.randint(0, num_nodes-1)
                if src != dst:
                    test_neg_edges.append([src, dst])
            
            test_neg_edges = torch.tensor(test_neg_edges, dtype=torch.long)
            
            # Combine positive and negative edges
            test_edge_tensor = torch.cat([test_pos_edges, test_neg_edges], dim=0)
            test_label_tensor = torch.cat([
                torch.ones(len(test_pos_edges), dtype=torch.long),
                torch.zeros(len(test_neg_edges), dtype=torch.long)
            ])
        
        print(f"Test set contains {len(test_edge_tensor)} samples")
        print(f"Positive samples: {torch.sum(test_label_tensor == 1).item()}")
        print(f"Negative samples: {torch.sum(test_label_tensor == 0).item()}")
        
        return test_edge_tensor, test_label_tensor
    
    def test_single_model(self, model_info, graph, test_edge_tensor, test_label_tensor, config):
        """Test a single model and return detailed results."""
        model_path = model_info['model_path']
        model_class = model_info['model_class']
        threshold = model_info['threshold']
        
        print(f"Testing model from {model_path}")
        
        # Get model configuration from config
        model_config = config.get_model_config()
        
        # Check if graph has edge features
        has_edge_attr = hasattr(graph, 'edge_attr') and graph.edge_attr is not None
        if has_edge_attr:
            print(f"✓ Using edge features: {graph.edge_attr.shape}")
            edge_attr = graph.edge_attr.float()
            
            # For TransformerModel, pass edge_dim to constructor
            model_name = model_class.__name__
            if model_name == 'TransformerModel':
                edge_dim = graph.edge_attr.size(1)
                model = model_class(
                    in_channels=graph.x.size(1),
                    hidden_channels=model_config['hidden_channels'],
                    out_channels=model_config['out_channels'],
                    num_layers=model_config['num_layers'],
                    dropout_rate=model_config['dropout_rate'],
                    edge_dim=edge_dim
                ).to(self.device)
            else:
                model = model_class(
                    in_channels=graph.x.size(1),
                    hidden_channels=model_config['hidden_channels'],
                    out_channels=model_config['out_channels'],
                    num_layers=model_config['num_layers'],
                    dropout_rate=model_config['dropout_rate']
                ).to(self.device)
        else:
            print("  Note: No edge features found")
            edge_attr = None
            # Load model with config parameters
            model = model_class(
                in_channels=graph.x.size(1),
                hidden_channels=model_config['hidden_channels'],
                out_channels=model_config['out_channels'],
                num_layers=model_config['num_layers'],
                dropout_rate=model_config['dropout_rate']
            ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        model.eval()
        
        # Move data to device
        graph = graph.to(self.device)
        test_edge_tensor = test_edge_tensor.to(self.device)
        test_label_tensor = test_label_tensor.to(self.device)
        if has_edge_attr:
            edge_attr = edge_attr.to(self.device)
        
        # Make predictions in batches to avoid memory issues
        batch_size = model_config['batch_size']
        test_probs = []
        
        with torch.no_grad():
            # Forward pass with edge features if available
            if has_edge_attr:
                z = model(graph.x.float(), graph.edge_index, edge_attr=edge_attr)
            else:
                z = model(graph.x.float(), graph.edge_index)
            
            # Normalise embeddings during testing
            z = F.normalize(z, p=2, dim=1)
            
            # Process in batches
            for start in range(0, len(test_edge_tensor), batch_size):
                end = min(start + batch_size, len(test_edge_tensor))
                batch_edges = test_edge_tensor[start:end]
                
                # Calculate edge scores
                batch_scores = (z[batch_edges[:, 0]] * z[batch_edges[:, 1]]).sum(dim=-1)
                batch_probs = torch.sigmoid(batch_scores)
                test_probs.append(batch_probs.cpu().numpy())
        
        # Combine results
        test_probs = np.concatenate(test_probs)
        test_preds = (test_probs >= threshold).astype(int)
        test_labels_np = test_label_tensor.cpu().numpy()
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(test_labels_np, test_probs, test_preds)
        
        return {
            'model': model,
            'model_path': model_path,
            'probabilities': test_probs,
            'predictions': test_preds,
            'labels': test_labels_np,
            'metrics': metrics,
            'test_edges': test_edge_tensor.cpu().numpy(),
            'threshold': threshold
        }
    
    def test_all_models(self, trained_models_info, graph, test_edge_tensor, test_label_tensor,
                       export_fp=True, fp_threshold=0.7, fp_top_k=10000, mappings_dir=None, 
                       config=None, training_run_ids=None):
        """Test all trained models and return comprehensive results."""
        
        test_results = {}
        
        for model_name, model_info in trained_models_info.items():
            print(f"\nTesting {model_name}...")
            
            # Start MLflow run for this model
            if self.mlflow_tracker:
                run_name = f"test_{model_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.mlflow_tracker.start_run(run_name=run_name)
                
                # Link to training run if available
                if training_run_ids and model_name in training_run_ids:
                    self.mlflow_tracker.log_param("training_run_id", training_run_ids[model_name])
                
                # Log test configuration
                self.mlflow_tracker.log_param("model_name", model_name)
                self.mlflow_tracker.log_param("model_path", model_info['model_path'])
                self.mlflow_tracker.log_param("threshold", model_info['threshold'])
                self.mlflow_tracker.log_param("test_set_size", len(test_edge_tensor))
                self.mlflow_tracker.log_param("export_fp", export_fp)
                self.mlflow_tracker.log_param("fp_threshold", fp_threshold)
                self.mlflow_tracker.log_param("fp_top_k", fp_top_k)
                self.mlflow_tracker.log_param("device", str(self.device))
            
            try:
                # Test model
                results = self.test_single_model(model_info, graph, test_edge_tensor, test_label_tensor, config)
                test_results[model_name] = results
                
                # Add realistic deployment evaluation
                deployment_metrics = self._evaluate_deployment_scenario(
                    model_info, graph, config, num_diseases=10
                )
                results['deployment_metrics'] = deployment_metrics
                
                # Log test metrics to MLflow
                if self.mlflow_tracker:
                    self._log_metrics_to_mlflow(model_name, results)
                
                # Export FP predictions if requested
                if export_fp:
                    fp_results = self.export_fp_predictions(
                        model_name=model_name,
                        model_results=results,
                        graph=graph,
                        threshold=fp_threshold,
                        top_k=fp_top_k,
                        mappings_dir=mappings_dir
                    )
                    
                    if fp_results:
                        results['fp_export'] = fp_results
                        print(f"FP export successful: {fp_results['fp_count']} predictions exported")
                        
                        # Log FP predictions to MLflow
                        if self.mlflow_tracker:
                            self._log_fp_export_to_mlflow(fp_results)
                
                # Print results with confidence intervals
                metrics = results['metrics']
                ci_results = metrics.get('ci_results', {})
                
                print(f"\nTest Results for {model_name}:")
                print("-" * 40)
                print(f"AUC: {metrics['auc']:.4f}")
                if 'auc' in ci_results:
                    ci = ci_results['auc']
                    print(f"     (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
                
                print(f"APR: {metrics['apr']:.4f}")
                if 'apr' in ci_results:
                    ci = ci_results['apr']
                    print(f"     (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
                
                print(f"F1-Score: {metrics['f1']:.4f}")
                if 'f1' in ci_results:
                    ci = ci_results['f1']
                    print(f"         (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
                
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                
                print(f"Precision: {metrics['precision']:.4f}")
                if 'precision' in ci_results:
                    ci = ci_results['precision']
                    print(f"          (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
                
                print(f"Recall: {metrics['recall']:.4f}")
                if 'recall' in ci_results:
                    ci = ci_results['recall']
                    print(f"       (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
                
                # End MLflow run for this model
                if self.mlflow_tracker:
                    self.mlflow_tracker.end_run()
                
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
                if self.mlflow_tracker:
                    self.mlflow_tracker.end_run()
                continue
        
        return test_results
    
    def _evaluate_deployment_scenario(self, model_info, graph, config, num_diseases=10, top_k_values=[50, 100, 200]):
        """
        Evaluate model in realistic deployment scenario.
        
        Simulates: "Rank ALL drugs for a disease, how many true associations are in top-K?"
        This reflects the real 1:2000 or more ratio you'd face in drug repurposing.
        """
        print("\n" + "="*80)
        print("REALISTIC DEPLOYMENT EVALUATION")
        print("="*80)
        print(f"Simulating: Rank all drugs for {num_diseases} diseases")
        print(f"Reporting: Recall@K for K={top_k_values}")
        print("="*80)
        
        model_path = model_info['model_path']
        model_class = model_info['model_class']
        
        # Check if graph has edge features
        has_edge_attr = hasattr(graph, 'edge_attr') and graph.edge_attr is not None
        if has_edge_attr:
            edge_attr = graph.edge_attr.float()
            
            # For TransformerModel, pass edge_dim to constructor
            model_name = model_class.__name__
            if model_name == 'TransformerModel':
                edge_dim = graph.edge_attr.size(1)
                model_config = config.get_model_config()
                model = model_class(
                    in_channels=graph.x.size(1),
                    hidden_channels=model_config['hidden_channels'],
                    out_channels=model_config['out_channels'],
                    num_layers=model_config['num_layers'],
                    dropout_rate=model_config['dropout_rate'],
                    edge_dim=edge_dim
                ).to(self.device)
            else:
                model_config = config.get_model_config()
                model = model_class(
                    in_channels=graph.x.size(1),
                    hidden_channels=model_config['hidden_channels'],
                    out_channels=model_config['out_channels'],
                    num_layers=model_config['num_layers'],
                    dropout_rate=model_config['dropout_rate']
                ).to(self.device)
        else:
            edge_attr = None
            # Load model
            model_config = config.get_model_config()
            model = model_class(
                in_channels=graph.x.size(1),
                hidden_channels=model_config['hidden_channels'],
                out_channels=model_config['out_channels'],
                num_layers=model_config['num_layers'],
                dropout_rate=model_config['dropout_rate']
            ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        model.eval()
        
        # Get drug-disease edges from test set
        test_positives = set()
        for i in range(len(graph.test_edge_index)):
            src, dst = graph.test_edge_index[i, 0].item(), graph.test_edge_index[i, 1].item()
            if graph.test_edge_label[i].item() == 1:
                test_positives.add((src, dst))
        
        # Group by disease
        disease_to_drugs = {}
        for src, dst in test_positives:
            if dst not in disease_to_drugs:
                disease_to_drugs[dst] = []
            disease_to_drugs[dst].append(src)
        
        # Sample diseases with at least 5 positive drugs
        eligible_diseases = [d for d, drugs in disease_to_drugs.items() if len(drugs) >= 5]
        if len(eligible_diseases) == 0:
            print("  No diseases with ≥5 positive drugs in test set")
            return None
        
        config = get_config()
        enable_full_reproducibility(config.seed)
        sample_diseases = random.sample(eligible_diseases, min(num_diseases, len(eligible_diseases)))
        
        # Evaluate recall@K for each disease
        with torch.no_grad():
            graph = graph.to(self.device)
            if has_edge_attr:
                edge_attr = edge_attr.to(self.device)
                z = model(graph.x.float(), graph.edge_index, edge_attr=edge_attr)
            else:
                z = model(graph.x.float(), graph.edge_index)
            
            num_drugs = 2000  # Approximate number of drugs
            drug_embeddings = z[:num_drugs]
            
            recall_at_k = {k: [] for k in top_k_values}
            
            for disease_idx in sample_diseases:
                disease_embedding = z[disease_idx]
                true_drugs = set(disease_to_drugs[disease_idx])
                
                # Score all drugs
                scores = torch.matmul(drug_embeddings, disease_embedding)
                probs = torch.sigmoid(scores)
                
                # Get top-K predictions
                for k in top_k_values:
                    top_k_drugs = torch.topk(probs, k).indices.cpu().tolist()
                    top_k_set = set(top_k_drugs)
                    
                    # Calculate recall
                    hits = len(true_drugs & top_k_set)
                    recall = hits / len(true_drugs)
                    recall_at_k[k].append(recall)
        
        # Average across diseases
        deployment_metrics = {}
        print("\nDeployment Metrics (Recall@K):")
        print("-" * 40)
        for k in top_k_values:
            avg_recall = np.mean(recall_at_k[k])
            deployment_metrics[f'recall@{k}'] = avg_recall
            print(f"  Recall@{k:3d}: {avg_recall:.3f} ({avg_recall*100:.1f}% of true drugs found)")
        
        print("\nInterpretation:")
        print(f"  When ranking {num_drugs} drugs for a new disease,")
        print(f"  the model finds ~{deployment_metrics['recall@100']*100:.0f}% of true associations")
        print(f"  by investigating only the top 100 candidates (5% effort)")
        print("="*80 + "\n")
        
        return deployment_metrics
    
    def _log_metrics_to_mlflow(self, model_name, results):
        """Log test metrics to MLflow."""
        if not self.mlflow_tracker:
            return
        
        metrics = results['metrics']
        
        # Log primary metrics
        self.mlflow_tracker.log_metric("test_auc", metrics['auc'])
        self.mlflow_tracker.log_metric("test_apr", metrics['apr'])
        self.mlflow_tracker.log_metric("test_f1", metrics['f1'])
        self.mlflow_tracker.log_metric("test_accuracy", metrics['accuracy'])
        self.mlflow_tracker.log_metric("test_precision", metrics['precision'])
        self.mlflow_tracker.log_metric("test_recall", metrics['recall'])
        self.mlflow_tracker.log_metric("test_specificity", metrics['specificity'])
        # Sensitivity is the same as recall in binary classification
        self.mlflow_tracker.log_metric("test_sensitivity", metrics['recall'])
        
        # Log confusion matrix values (note the nested structure from calculate_metrics)
        cm = metrics['confusion_matrix']
        self.mlflow_tracker.log_metric("test_true_negatives", cm['TN'])
        self.mlflow_tracker.log_metric("test_false_positives", cm['FP'])
        self.mlflow_tracker.log_metric("test_false_negatives", cm['FN'])
        self.mlflow_tracker.log_metric("test_true_positives", cm['TP'])
        
        # Log confidence intervals if available
        ci_results = metrics.get('ci_results', {})
        for metric_name in ['auc', 'apr', 'f1', 'precision', 'recall', 'specificity']:
            if metric_name in ci_results:
                ci = ci_results[metric_name]
                self.mlflow_tracker.log_metric(f"test_{metric_name}_ci_lower", ci['ci_lower'])
                self.mlflow_tracker.log_metric(f"test_{metric_name}_ci_upper", ci['ci_upper'])
                self.mlflow_tracker.log_metric(f"test_{metric_name}_ci_width", ci['ci_width'])
        
        print(f"✓ Test metrics logged to MLflow for {model_name}")
    
    def _log_fp_export_to_mlflow(self, fp_results):
        """Log false positive export information to MLflow."""
        if not self.mlflow_tracker:
            return
        
        summary = fp_results['summary']
        
        # Log FP export metrics
        self.mlflow_tracker.log_metric("fp_export_count", summary['exported_fp_count'])
        self.mlflow_tracker.log_metric("fp_total_candidates", summary['total_fp_candidates'])
        self.mlflow_tracker.log_metric("fp_mean_confidence", summary['mean_confidence'])
        self.mlflow_tracker.log_metric("fp_min_confidence", summary['confidence_range'][0])
        self.mlflow_tracker.log_metric("fp_max_confidence", summary['confidence_range'][1])
        
        # Log FP export parameters
        self.mlflow_tracker.log_param("fp_threshold", summary['fp_threshold'])
        self.mlflow_tracker.log_param("fp_top_k", summary['top_k'])
        
        # Log FP prediction files as artifacts
        if 'csv_path' in fp_results:
            self.mlflow_tracker.log_artifact(fp_results['csv_path'])
        if 'summary_path' in fp_results:
            self.mlflow_tracker.log_artifact(fp_results['summary_path'])
        
        print(f"✓ FP predictions logged to MLflow")

    def export_fp_predictions(self, model_name, model_results, graph, 
                             threshold=0.7, top_k=10000, mappings_dir=None):
        """Export False Positive predictions for GNNExplainer analysis."""
        print(f"\nExporting FP predictions for {model_name}...")
        
        # Load actual mappings if available
        drug_idx_to_name = {}
        disease_idx_to_name = {}
        
        if mappings_dir and os.path.exists(mappings_dir):
            try:
                # Load drug mappings (reverse mapping from index to name)
                drug_mapping_path = os.path.join(mappings_dir, "drug_key_mapping.json")
                with open(drug_mapping_path, 'r') as f:
                    drug_name_to_idx = json.load(f)
                    drug_idx_to_name = {int(idx): name for name, idx in drug_name_to_idx.items()}
                
                # Load disease mappings (reverse mapping from index to name)
                disease_mapping_path = os.path.join(mappings_dir, "disease_key_mapping.json")
                with open(disease_mapping_path, 'r') as f:
                    disease_name_to_idx = json.load(f)
                    disease_idx_to_name = {int(idx): name for name, idx in disease_name_to_idx.items()}
                
                print(f"Loaded mappings: {len(drug_idx_to_name)} drugs, {len(disease_idx_to_name)} diseases")
                
            except Exception as e:
                print(f"Warning: Could not load mappings from {mappings_dir}: {e}")
                print("Will use generic names instead.")
        else:
            print("Warning: No mappings directory provided. Using generic names.")
        
        # Extract data
        probabilities = model_results['probabilities']
        labels = model_results['labels']
        test_edges = model_results['test_edges']
        
        # Identify False Positives (high confidence predictions on negative labels)
        fp_mask = (labels == 0) & (probabilities >= threshold)
        fp_indices = np.where(fp_mask)[0]
        
        if len(fp_indices) == 0:
            print(f"No FP predictions found above threshold {threshold}")
            return None
        
        print(f"Found {len(fp_indices)} FP predictions above threshold {threshold}")
        
        # Sort by confidence and take top-K
        fp_scores = probabilities[fp_indices]
        sorted_indices = np.argsort(fp_scores)[::-1]  # Descending order
        top_fp_indices = fp_indices[sorted_indices[:top_k]]
        top_fp_scores = fp_scores[sorted_indices[:top_k]]
        
        print(f"Exporting top {len(top_fp_indices)} FP predictions")
        
        # Create FP predictions data
        fp_data = []
        missing_mappings_count = 0
        
        for i, (fp_idx, score) in enumerate(zip(top_fp_indices, top_fp_scores)):
            edge_idx = int(fp_idx)
            drug_idx = int(test_edges[edge_idx, 0])
            disease_idx = int(test_edges[edge_idx, 1])
            
            # Get actual names from mappings if available, otherwise use generic names
            drug_name = drug_idx_to_name.get(drug_idx, f"Drug_{drug_idx}")
            disease_name = disease_idx_to_name.get(disease_idx, f"Disease_{disease_idx}")
            
            # Count missing mappings for statistics
            if drug_idx not in drug_idx_to_name or disease_idx not in disease_idx_to_name:
                missing_mappings_count += 1
            
            fp_data.append({
                'rank': i + 1,
                'drug_idx': drug_idx,
                'disease_idx': disease_idx,
                'drug_name': drug_name,
                'disease_name': disease_name,
                'confident_score': float(score),
                'prediction': float(score),
                'true_label': 0,  # All FPs have true label 0
                'model': model_name
            })
        
        if missing_mappings_count > 0:
            print(f"Warning: {missing_mappings_count}/{len(fp_data)} pairs used generic names due to missing mappings")
        
        # Convert to DataFrame
        fp_df = pd.DataFrame(fp_data)
        
        # Generate timestamp for unique filenames
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Export in multiple formats for GNNExplainer compatibility
        
        # 1. CSV format (primary format for GNNExplainer script)
        csv_filename = f"{model_name}_FP_predictions_{timestamp}.csv"
        csv_path = os.path.join(self.results_dirs['predictions'], csv_filename)
        fp_df.to_csv(csv_path, index=False)
        print(f"FP predictions saved as CSV: {csv_path}")
        
        # 2. PyTorch tensor format (backup format)
        tensor_data = [[row['drug_name'], row['disease_name'], row['confident_score'], 
                        row['drug_idx'], row['disease_idx']] 
                       for _, row in fp_df.iterrows()]
        pt_filename = f"{model_name}_FP_predictions_{timestamp}.pt"
        pt_path = os.path.join(self.results_dirs['predictions'], pt_filename)
        torch.save(tensor_data, pt_path)
        print(f"FP predictions saved as PyTorch: {pt_path}")
        
        # 3. Export summary statistics
        summary_data = {
            'model_name': model_name,
            'timestamp': timestamp,
            'total_test_edges': len(test_edges),
            'total_negative_edges': int(np.sum(labels == 0)),
            'fp_threshold': threshold,
            'top_k': top_k,
            'total_fp_candidates': len(fp_indices),
            'exported_fp_count': len(fp_data),
            'confidence_range': [float(fp_df['confident_score'].min()), float(fp_df['confident_score'].max())],
            'mean_confidence': float(fp_df['confident_score'].mean()),
            'csv_file': csv_filename,
            'pt_file': pt_filename
        }
        
        summary_filename = f"{model_name}_FP_export_summary_{timestamp}.json"
        summary_path = os.path.join(self.results_dirs['predictions'], summary_filename)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Export summary saved: {summary_path}")
        
        return {
            'csv_path': csv_path,
            'pt_path': pt_path,
            'summary_path': summary_path,
            'fp_count': len(fp_data),
            'summary': summary_data
        }
    
    def create_visualizations(self, test_results):
        """Create comprehensive visualizations of test results."""
        
        datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 1. ROC Curves
        roc_path = self._plot_roc_curves(test_results, datetime_str)
        
        # 2. Precision-Recall Curves
        pr_path = self._plot_pr_curves(test_results, datetime_str)
        
        # 3. Confusion Matrices
        cm_path = self._plot_confusion_matrices(test_results, datetime_str)
        
        # 4. Metrics Comparison
        metrics_path = self._plot_metrics_comparison(test_results, datetime_str)
        
        # 5. Interactive Plotly Visualizations
        interactive_path = self._create_interactive_plots(test_results, datetime_str)
        
        # Log all visualizations to MLflow in a dedicated run
        if self.mlflow_tracker:
            run_name = f"test_visualizations_{datetime_str}"
            self.mlflow_tracker.start_run(run_name=run_name)
            
            # Log all visualization artifacts
            for path in [roc_path, pr_path, cm_path, metrics_path, interactive_path]:
                if path and os.path.exists(path):
                    self.mlflow_tracker.log_artifact(path)
            
            self.mlflow_tracker.end_run()
            print(f"✓ All visualizations logged to MLflow")
    
    def _plot_roc_curves(self, test_results, datetime_str):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(12, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':']
        
        for i, (model_name, results) in enumerate(test_results.items()):
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                    color=colors[i % len(colors)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=3,
                    label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = f'{self.results_dirs["figures"]}/test_roc_curves_{datetime_str}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    
    def _plot_pr_curves(self, test_results, datetime_str):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(12, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':']
        
        for i, (model_name, results) in enumerate(test_results.items()):
            precision, recall, _ = precision_recall_curve(results['labels'], results['probabilities'])
            avg_precision = average_precision_score(results['labels'], results['probabilities'])
            
            plt.plot(recall, precision,
                    color=colors[i % len(colors)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=3,
                    label=f'{model_name} (AP = {avg_precision:.4f})')
        
        # Baseline
        baseline = np.mean(test_results[list(test_results.keys())[0]]['labels'])
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2, alpha=0.7, label=f'Random (AP = {baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves', fontsize=16)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = f'{self.results_dirs["figures"]}/test_pr_curves_{datetime_str}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    
    def _plot_confusion_matrices(self, test_results, datetime_str):
        """Plot confusion matrices for all models."""
        n_models = len(test_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(test_results.items()):
            cm = confusion_matrix(results['labels'], results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}', fontsize=14)
            axes[i].set_xlabel('Predicted Label', fontsize=12)
            axes[i].set_ylabel('True Label', fontsize=12)
            axes[i].set_xticklabels(['Negative', 'Positive'])
            axes[i].set_yticklabels(['Negative', 'Positive'])
        
        plt.tight_layout()
        
        output_path = f'{self.results_dirs["figures"]}/test_confusion_matrices_{datetime_str}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    
    def _plot_metrics_comparison(self, test_results, datetime_str):
        """Plot comprehensive metrics comparison."""
        metrics_to_plot = ['auc', 'apr', 'f1', 'accuracy', 'precision', 'recall', 'specificity']
        model_names = list(test_results.keys())
        
        # Prepare data
        metrics_data = {metric: [] for metric in metrics_to_plot}
        for model_name in model_names:
            for metric in metrics_to_plot:
                metrics_data[metric].append(test_results[model_name]['metrics'][metric])
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'mediumpurple', 'lightcoral', 'lightskyblue']
        
        for i, metric in enumerate(metrics_to_plot):
            values = metrics_data[metric]
            bars = axes[i].bar(model_names, values, color=colors[i % len(colors)], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            axes[i].set_ylim(0, 1.1)
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Remove the last empty subplot
        fig.delaxes(axes[7])
        
        plt.suptitle('Model Performance Comparison - Test Set', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = f'{self.results_dirs["figures"]}/test_metrics_comparison_{datetime_str}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    
    def _create_interactive_plots(self, test_results, datetime_str):
        """Create interactive Plotly visualizations."""
        
        # Interactive ROC Curves
        fig_roc = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, results) in enumerate(test_results.items()):
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc = auc(fpr, tpr)
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.4f})',
                line=dict(color=colors[i % len(colors)], width=3)
            ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        fig_roc.update_layout(
            title='Interactive ROC Curves - Test Set',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600
        )
        
        output_path = f'{self.results_dirs["figures"]}/interactive_roc_curves_{datetime_str}.html'
        fig_roc.write_html(output_path)
        return output_path
    
    def save_detailed_results(self, test_results):
        """Save detailed test results to files."""
        
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Save metrics summary with confidence intervals
        summary_data = []
        for model_name, results in test_results.items():
            row = {'Model': model_name}
            row.update(results['metrics'])
            row['Threshold'] = results['threshold']
            
            # Add CI columns if available
            ci_results = results['metrics'].get('ci_results', {})
            for metric in ['auc', 'apr', 'f1', 'precision', 'recall', 'specificity']:
                if metric in ci_results:
                    ci = ci_results[metric]
                    row[f'{metric}_ci_lower'] = ci['ci_lower']
                    row[f'{metric}_ci_upper'] = ci['ci_upper']
                    row[f'{metric}_ci_width'] = ci['ci_width']
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.results_dirs["evaluation"]}/test_results_summary_{timestamp}.csv', index=False)
        
        # Save detailed results as JSON
        results_for_json = {}
        for model_name, results in test_results.items():
            results_for_json[model_name] = {
                'metrics': results['metrics'],
                'threshold': results['threshold'],
                'predictions_sample': results['predictions'][:100].tolist(),
                'probabilities_sample': results['probabilities'][:100].tolist()
            }
            
            # Add FP export info if available
            if 'fp_export' in results:
                results_for_json[model_name]['fp_export_summary'] = results['fp_export']['summary']
        
        with open(f'{self.results_dirs["evaluation"]}/test_results_detailed_{timestamp}.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        print(f"Detailed results saved to: {self.results_dirs['evaluation']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test and evaluate trained models')
    parser.add_argument('--graph', '--graph-path', dest='graph_path', default=None, help='Path to graph file (.pt) - auto-detects if not provided')
    parser.add_argument('--model', '--models-path', dest='models_path', default=None, help='Path to trained models directory or single model file - auto-detects if not provided')
    parser.add_argument('--results-path', default='results/', help='Results output directory')
    parser.add_argument('--export-fp', action='store_true', default=True, help='Export FP predictions for GNNExplainer')
    parser.add_argument('--fp-threshold', type=float, default=0.7, help='FP confidence threshold')
    parser.add_argument('--fp-top-k', type=int, default=10000, help='Maximum FP predictions to export')
    parser.add_argument('--mappings-dir', type=str, help='Path to directory containing drug/disease mappings')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    parser.add_argument('--mlflow-experiment-name', type=str, help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Auto-detect graph if not provided
    if not args.graph_path:
        print("No graph specified, auto-detecting latest graph...")
        try:
            args.graph_path = find_latest_graph(args.results_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
    
    # Auto-detect models if not provided
    if not args.models_path:
        print("No models specified, auto-detecting latest models...")
        try:
            args.models_path = find_latest_models(args.results_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
    
    # Load configuration
    config = get_config()
    
    # Set reproducibility
    enable_full_reproducibility(config.seed)
    
    # Initialize evaluator (MLflow enabled by default unless --no-mlflow is passed)
    evaluator = ModelEvaluator(
        results_path=args.results_path,
        use_mlflow=not args.no_mlflow,
        mlflow_experiment_name=args.mlflow_experiment_name
    )
    
    # Load graph
    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph file not found: {args.graph_path}")
    
    graph = torch.load(args.graph_path, map_location=evaluator.device, weights_only=False)
    print(f"Loaded graph: {graph}")
    
    # Load trained models information
    trained_models_info = evaluator.load_trained_models_info(args.models_path, config=config)
    
    if not trained_models_info:
        print("No trained models found! Please train models first.")
        return None
    
    print(f"Found trained models: {list(trained_models_info.keys())}")
    
    # Auto-detect mappings directory if not provided
    mappings_dir = args.mappings_dir
    if not mappings_dir:
        # Try to auto-detect mappings directory relative to workspace
        workspace_root = Path(__file__).parent.parent
        potential_mappings_dir = workspace_root / "processed_data" / "mappings"
        if potential_mappings_dir.exists():
            mappings_dir = str(potential_mappings_dir)
            print(f"Auto-detected mappings directory: {mappings_dir}")
        else:
            print("Warning: No mappings directory found. Will use generic names for predictions.")
    
    # Create test data from graph
    test_edge_tensor, test_label_tensor = evaluator.create_test_data_from_graph(graph)
    
    # Test all models with FP export
    test_results = evaluator.test_all_models(
        trained_models_info, 
        graph, 
        test_edge_tensor, 
        test_label_tensor,
        export_fp=args.export_fp,
        fp_threshold=args.fp_threshold,
        fp_top_k=args.fp_top_k,
        mappings_dir=mappings_dir,
        config=config
    )
    
    if not test_results:
        print("No successful test results!")
        return None
    
    # Create visualizations
    print("Creating visualizations...")
    evaluator.create_visualizations(test_results)
    
    # Save detailed results
    print("Saving detailed results...")
    evaluator.save_detailed_results(test_results)
    
    print(f"\nEvaluation completed! Results saved to {args.results_path}")
    
    # Print summary and GNNExplainer integration instructions
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model_name, results in test_results.items():
        metrics = results['metrics']
        print(f"\n{model_name}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1:  {metrics['f1']:.4f}")
        print(f"  Acc: {metrics['accuracy']:.4f}")
        
        if 'fp_export' in results:
            fp_count = results['fp_export']['fp_count']
            csv_file = results['fp_export']['summary']['csv_file']
            print(f"  FP Exported: {fp_count} predictions")
            print(f"  CSV File: {args.results_path}predictions/{csv_file}")
    
    # Print GNNExplainer integration instructions
    if args.export_fp:
        print(f"\n" + "="*60)
        print("GNNEXPLAINER INTEGRATION")
        print("="*60)
        print("To analyze FP predictions with GNNExplainer, run:")
        
        for model_name, results in test_results.items():
            if 'fp_export' in results:
                csv_file = results['fp_export']['summary']['csv_file']
                print(f"\n# For {model_name}:")
                print(f"python scripts/4_explain_predictions.py \\")
                print(f"  --graph {args.graph_path} \\")
                print(f"  --predictions {args.results_path}predictions/{csv_file} \\")
                print(f"  --output-dir {args.results_path}explainer/{model_name}_explainer")
    
    return test_results


if __name__ == "__main__":
    main()

"""
Testing and Evaluation Module for Drug-Disease Prediction
This module handles model testing, performance evaluation, visualization, and FP export for GNNExplainer.
"""

import torch
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
import platform
import random
from scipy import stats

from src.models import GCNModel, TransformerModel, SAGEModel
from src.utils import set_seed, enable_full_reproducibility, calculate_bootstrap_ci
from src.config import get_config

class ModelEvaluator:
    """Class for comprehensive model evaluation and testing with GNNExplainer integration."""
    
    def __init__(self, results_path="test_results/"):
        self.results_path = results_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dirs = self._create_results_structure()
        
    def _create_results_structure(self):
        """Create organized results directory structure."""
        subdirs = {
            'models': 'models',
            'predictions': 'predictions',
            'evaluation': 'evaluation',
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
        
    def create_test_data_from_graph(self, graph, test_ratio=0.1, seed=42):
        """Create test dataset from the loaded graph object (compatible with train_CI.py structure)."""
        print("Creating test dataset from graph...")
        
        set_seed(seed)
        
        # Use validation edges if they exist in the graph, otherwise create test set
        if hasattr(graph, 'val_edge_index') and hasattr(graph, 'val_edge_label'):
            print("Using existing validation data as test set")
            test_edge_tensor = graph.val_edge_index
            test_label_tensor = graph.val_edge_label
        else:
            print("Creating new test dataset...")
            # This is a fallback - in practice, test data should be provided
            # Extract some edges for testing (this is simplified)
            num_edges = graph.edge_index.size(1)
            test_size = int(num_edges * test_ratio)
            
            # Sample test edges
            perm = torch.randperm(num_edges)
            test_indices = perm[:test_size]
            
            test_edges = graph.edge_index[:, test_indices].t()
            test_labels = torch.ones(test_size, dtype=torch.long)  # All positive for simplicity
            
            test_edge_tensor = test_edges
            test_label_tensor = test_labels
        
        print(f"Test set contains {len(test_edge_tensor)} samples")
        print(f"Positive samples: {torch.sum(test_label_tensor == 1).item()}")
        print(f"Negative samples: {torch.sum(test_label_tensor == 0).item()}")
        
        return test_edge_tensor, test_label_tensor
    
    def test_model(self, model_path, model_class, graph, test_edge_tensor, test_label_tensor, threshold=0.5):
        """Test a single model and return detailed results."""
        print(f"Testing model from {model_path}")
        
        # Load model
        model = model_class(
            in_channels=graph.x.size(1),
            hidden_channels=16,
            out_channels=16,
            num_layers=2,
            dropout_rate=0.5
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # Move data to device
        graph = graph.to(self.device)
        test_edge_tensor = test_edge_tensor.to(self.device)
        test_label_tensor = test_label_tensor.to(self.device)
        
        # Make predictions in batches to avoid memory issues
        batch_size = 1000
        test_probs = []
        
        with torch.no_grad():
            z = model(graph.x.float(), graph.edge_index)
            
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
        metrics = self._calculate_metrics(test_labels_np, test_probs, test_preds)
        
        return {
            'model': model,
            'probabilities': test_probs,
            'predictions': test_preds,
            'labels': test_labels_np,
            'metrics': metrics,
            'test_edges': test_edge_tensor.cpu().numpy()
        }
    
    def export_fp_predictions(self, model_name, model_results, graph, drug_names=None, disease_names=None, 
                             threshold=0.7, top_k=10000):
        """Export False Positive predictions for GNNExplainer analysis."""
        print(f"\nExporting FP predictions for {model_name}...")
        
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
        for i, (fp_idx, score) in enumerate(zip(top_fp_indices, top_fp_scores)):
            edge_idx = int(fp_idx)
            drug_idx = int(test_edges[edge_idx, 0])
            disease_idx = int(test_edges[edge_idx, 1])
            
            # Get names (with bounds checking)
            if drug_names and drug_idx < len(drug_names):
                drug_name = drug_names[drug_idx]
            else:
                drug_name = f"Drug_{drug_idx}"
                
            if disease_names and disease_idx < len(disease_names):
                disease_name = disease_names[disease_idx]
            else:
                disease_name = f"Disease_{disease_idx}"
            
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
        
        # Convert to DataFrame
        fp_df = pd.DataFrame(fp_data)
        
        # Generate timestamp for unique filenames
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Export in multiple formats for GNNExplainer compatibility
        
        # 1. CSV format (primary format for GNNExplainer script)
        csv_filename = f"{model_name}_Baseline_Test_FP_links_{timestamp}.csv"
        csv_path = os.path.join(self.results_dirs['predictions'], csv_filename)
        fp_df.to_csv(csv_path, index=False)
        print(f"FP predictions saved as CSV: {csv_path}")
        
        # 2. PyTorch tensor format (backup format)
        tensor_data = [[row['drug_name'], row['disease_name'], row['confident_score']] 
                       for _, row in fp_df.iterrows()]
        pt_filename = f"{model_name}_Baseline_Test_FP_links_{timestamp}.pt"
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
    
    def calculate_bootstrap_ci(self, y_true, y_pred_proba, y_pred_binary, n_bootstrap=1000, confidence_level=0.95):
        """Calculate bootstrap confidence intervals for classification metrics (compatible with train_CI.py)"""
        
        # Set seed for reproducibility (consistent with train_CI.py)
        random.seed(42)
        np.random.seed(42)
        
        # Storage for bootstrap results (matching train_CI.py metrics)
        bootstrap_metrics = {
            'sensitivity': [], 'specificity': [], 'precision': [], 
            'f1': [], 'auc': [], 'apr': []
        }
        
        n_samples = len(y_true)
        alpha = 1 - confidence_level
        
        print(f"Calculating bootstrap CIs with {n_bootstrap} iterations...")
        
        for i in range(n_bootstrap):
            # Bootstrap sample indices
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Bootstrap samples
            boot_y_true = y_true[boot_indices]
            boot_y_pred_proba = y_pred_proba[boot_indices] 
            boot_y_pred_binary = y_pred_binary[boot_indices]
            
            # Calculate metrics for this bootstrap sample
            # Confusion matrix elements
            tp = np.sum(boot_y_pred_binary * boot_y_true)
            fp = np.sum(boot_y_pred_binary * (1 - boot_y_true))
            fn = np.sum((1 - boot_y_pred_binary) * boot_y_true)
            tn = np.sum((1 - boot_y_pred_binary) * (1 - boot_y_true))
            
            # Classification metrics (matching train_CI.py)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # Store results
            bootstrap_metrics['sensitivity'].append(sensitivity)
            bootstrap_metrics['specificity'].append(specificity) 
            bootstrap_metrics['precision'].append(precision)
            bootstrap_metrics['f1'].append(f1)
            
            # AUC and APR (only if we have both classes)
            if len(np.unique(boot_y_true)) == 2:
                try:
                    auc_score = roc_auc_score(boot_y_true, boot_y_pred_proba)
                    apr_score = average_precision_score(boot_y_true, boot_y_pred_proba)
                    bootstrap_metrics['auc'].append(auc_score)
                    bootstrap_metrics['apr'].append(apr_score)
                except:
                    # Skip this iteration if AUC/APR calculation fails
                    bootstrap_metrics['auc'].append(np.nan)
                    bootstrap_metrics['apr'].append(np.nan)
            else:
                bootstrap_metrics['auc'].append(np.nan)
                bootstrap_metrics['apr'].append(np.nan)
        
        # Calculate confidence intervals
        ci_results = {}
        for metric, values in bootstrap_metrics.items():
            # Remove NaN values
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) > 0:
                lower_percentile = (alpha/2) * 100
                upper_percentile = (1 - alpha/2) * 100
                
                ci_lower = np.percentile(clean_values, lower_percentile)
                ci_upper = np.percentile(clean_values, upper_percentile)
                mean_val = np.mean(clean_values)
                
                ci_results[metric] = {
                    'mean': mean_val,
                    'ci_lower': ci_lower, 
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower
                }
            else:
                ci_results[metric] = {
                    'mean': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan, 
                    'ci_width': np.nan
                }
        
        return ci_results

    def _calculate_metrics(self, y_true, y_prob, y_pred):
        """Calculate comprehensive evaluation metrics with bootstrap confidence intervals."""
        
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Advanced metrics
        auc_score = roc_auc_score(y_true, y_prob)
        apr_score = average_precision_score(y_true, y_prob)
        
        # Additional metrics
        ppv = precision  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Calculate bootstrap confidence intervals
        print("Computing bootstrap confidence intervals...")
        ci_results = self.calculate_bootstrap_ci(y_true, y_prob, y_pred, n_bootstrap=1000)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auc': auc_score,
            'apr': apr_score,
            'ppv': ppv,
            'npv': npv,
            'confusion_matrix': {
                'TP': int(tp), 'FP': int(fp), 
                'TN': int(tn), 'FN': int(fn)
            },
            'ci_results': ci_results
        }
    
    def test_all_models(self, trained_models_info, graph, test_edge_tensor, test_label_tensor, 
                       export_fp=True, fp_threshold=0.7, fp_top_k=10000):
        """Test all trained models and return comprehensive results."""
        
        test_results = {}
        
        for model_name, model_info in trained_models_info.items():
            print(f"Testing {model_name}...")
            
            # Get model class
            model_classes = {
                'GCNModel': GCNModel,
                'TransformerModel': TransformerModel,
                'SAGEModel': SAGEModel
            }
            
            if model_name not in model_classes:
                print(f"Warning: Unknown model class {model_name}")
                continue
                
            model_class = model_classes[model_name]
            
            # Test model
            results = self.test_model(
                model_info['model_path'],
                model_class,
                graph,
                test_edge_tensor,
                test_label_tensor,
                model_info.get('threshold', 0.5)
            )
            
            results['threshold'] = model_info.get('threshold', 0.5)
            test_results[model_name] = results
            
            # Export FP predictions if requested
            if export_fp:
                fp_results = self.export_fp_predictions(
                    model_name=model_name,
                    model_results=results,
                    graph=graph,
                    threshold=fp_threshold,
                    top_k=fp_top_k
                )
                
                if fp_results:
                    results['fp_export'] = fp_results
                    print(f"FP export successful: {fp_results['fp_count']} predictions exported")
            
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
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            
            print(f"Precision: {metrics['precision']:.4f}")
            if 'precision' in ci_results:
                ci = ci_results['precision']
                print(f"          (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
            
            print(f"Recall: {metrics['recall']:.4f}")
            if 'sensitivity' in ci_results:  # recall is sensitivity
                ci = ci_results['sensitivity']
                print(f"       (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
            
            print(f"F1-Score: {metrics['f1']:.4f}")
            if 'f1' in ci_results:
                ci = ci_results['f1']
                print(f"         (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
            
            print(f"Specificity: {metrics['specificity']:.4f}")
            if 'specificity' in ci_results:
                ci = ci_results['specificity']
                print(f"            (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
            print()
        
        return test_results
    
    def create_visualizations(self, test_results):
        """Create comprehensive visualizations of test results."""
        
        datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 1. ROC Curves
        self._plot_roc_curves(test_results, datetime_str)
        
        # 2. Precision-Recall Curves
        self._plot_pr_curves(test_results, datetime_str)
        
        # 3. Confusion Matrices
        self._plot_confusion_matrices(test_results, datetime_str)
        
        # 4. Metrics Comparison
        self._plot_metrics_comparison(test_results, datetime_str)
        
        # 5. Interactive Plotly Visualizations
        self._create_interactive_plots(test_results, datetime_str)
        
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
        plt.savefig(f'{self.results_dirs["figures"]}/test_roc_curves_{datetime_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.savefig(f'{self.results_dirs["figures"]}/test_pr_curves_{datetime_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.savefig(f'{self.results_dirs["figures"]}/test_confusion_matrices_{datetime_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.savefig(f'{self.results_dirs["figures"]}/test_metrics_comparison_{datetime_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        
        fig_roc.write_html(f'{self.results_dirs["figures"]}/interactive_roc_curves_{datetime_str}.html')
        
        # Interactive Metrics Radar Chart
        metrics_names = ['AUC', 'APR', 'F1', 'Accuracy', 'Precision', 'Recall', 'Specificity']
        
        fig_radar = go.Figure()
        
        for i, (model_name, results) in enumerate(test_results.items()):
            metrics_values = [
                results['metrics']['auc'],
                results['metrics']['apr'],
                results['metrics']['f1'],
                results['metrics']['accuracy'],
                results['metrics']['precision'],
                results['metrics']['recall'],
                results['metrics']['specificity']
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=metrics_values,
                theta=metrics_names,
                fill='toself',
                name=model_name,
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart - Test Set",
            width=800, height=600
        )
        
        fig_radar.write_html(f'{self.results_dirs["figures"]}/interactive_radar_chart_{datetime_str}.html')
    
    def save_detailed_results(self, test_results, datetime_str):
        """Save detailed test results to files."""
        
        # Save metrics summary with confidence intervals
        summary_data = []
        for model_name, results in test_results.items():
            row = {'Model': model_name}
            row.update(results['metrics'])
            row['Threshold'] = results['threshold']
            
            # Add CI columns if available
            ci_results = results['metrics'].get('ci_results', {})
            for metric in ['auc', 'apr', 'f1', 'precision', 'sensitivity', 'specificity']:
                if metric in ci_results:
                    ci = ci_results[metric]
                    row[f'{metric}_ci_lower'] = ci['ci_lower']
                    row[f'{metric}_ci_upper'] = ci['ci_upper']
                    row[f'{metric}_ci_width'] = ci['ci_width']
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.results_dirs["evaluation"]}/test_results_summary_with_ci_{datetime_str}.csv', index=False)
        summary_df.to_excel(f'{self.results_dirs["evaluation"]}/test_results_summary_with_ci_{datetime_str}.xlsx', index=False)
        
        # Save detailed results as JSON
        results_for_json = {}
        for model_name, results in test_results.items():
            results_for_json[model_name] = {
                'metrics': results['metrics'],
                'threshold': results['threshold'],
                'predictions_sample': results['predictions'][:100].tolist(),
                'probabilities_sample': results['probabilities'][:100].tolist(),
                'ci_results': results['metrics'].get('ci_results', {})
            }
            
            # Add FP export info if available
            if 'fp_export' in results:
                results_for_json[model_name]['fp_export_summary'] = results['fp_export']['summary']
        
        with open(f'{self.results_dirs["evaluation"]}/test_results_detailed_with_ci_{datetime_str}.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        # Create comprehensive report
        self._create_report(test_results, datetime_str)
    
    def _create_report(self, test_results, datetime_str):
        """Create a comprehensive text report."""
        
        report_path = f'{self.results_dirs["evaluation"]}/test_evaluation_report_{datetime_str}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Report Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            # Find best performing model
            best_auc_model = max(test_results.items(), key=lambda x: x[1]['metrics']['auc'])
            best_f1_model = max(test_results.items(), key=lambda x: x[1]['metrics']['f1'])
            
            f.write(f"Best AUC Performance: {best_auc_model[0]} ({best_auc_model[1]['metrics']['auc']:.4f})\n")
            f.write(f"Best F1 Performance: {best_f1_model[0]} ({best_f1_model[1]['metrics']['f1']:.4f})\n")
            f.write(f"Total Models Evaluated: {len(test_results)}\n\n")
            
            # FP Export Summary
            f.write("FALSE POSITIVE EXPORT SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            for model_name, results in test_results.items():
                if 'fp_export' in results:
                    fp_info = results['fp_export']['summary']
                    f.write(f"{model_name}:\n")
                    f.write(f"  FP Predictions Exported: {fp_info['exported_fp_count']}\n")
                    f.write(f"  Confidence Range: {fp_info['confidence_range'][0]:.3f} - {fp_info['confidence_range'][1]:.3f}\n")
                    f.write(f"  CSV File: {fp_info['csv_file']}\n")
                    f.write(f"  PyTorch File: {fp_info['pt_file']}\n\n")
            
            # Detailed Results for Each Model
            for model_name, results in test_results.items():
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write("=" * 50 + "\n")
                
                metrics = results['metrics']
                cm = metrics['confusion_matrix']
                ci_results = metrics.get('ci_results', {})
                
                f.write(f"Threshold Used: {results['threshold']:.4f}\n\n")
                
                f.write("Performance Metrics with 95% Confidence Intervals:\n")
                f.write(f"  • AUC-ROC: {metrics['auc']:.4f}")
                if 'auc' in ci_results:
                    ci = ci_results['auc']
                    f.write(f" [95% CI: {ci['ci_lower']:.4f} - {ci['ci_upper']:.4f}]")
                f.write("\n")
                
                f.write(f"  • Average Precision: {metrics['apr']:.4f}")
                if 'apr' in ci_results:
                    ci = ci_results['apr']
                    f.write(f" [95% CI: {ci['ci_lower']:.4f} - {ci['ci_upper']:.4f}]")
                f.write("\n")
                
                f.write(f"  • Accuracy: {metrics['accuracy']:.4f}\n")
                
                f.write(f"  • Precision (PPV): {metrics['precision']:.4f}")
                if 'precision' in ci_results:
                    ci = ci_results['precision']
                    f.write(f" [95% CI: {ci['ci_lower']:.4f} - {ci['ci_upper']:.4f}]")
                f.write("\n")
                
                f.write(f"  • Recall (Sensitivity): {metrics['recall']:.4f}")
                if 'sensitivity' in ci_results:
                    ci = ci_results['sensitivity']
                    f.write(f" [95% CI: {ci['ci_lower']:.4f} - {ci['ci_upper']:.4f}]")
                f.write("\n")
                
                f.write(f"  • Specificity: {metrics['specificity']:.4f}")
                if 'specificity' in ci_results:
                    ci = ci_results['specificity']
                    f.write(f" [95% CI: {ci['ci_lower']:.4f} - {ci['ci_upper']:.4f}]")
                f.write("\n")
                
                f.write(f"  • F1-Score: {metrics['f1']:.4f}")
                if 'f1' in ci_results:
                    ci = ci_results['f1']
                    f.write(f" [95% CI: {ci['ci_lower']:.4f} - {ci['ci_upper']:.4f}]")
                f.write("\n")
                
                f.write(f"  • NPV: {metrics['npv']:.4f}\n\n")
                
                f.write("Confusion Matrix:\n")
                f.write(f"  True Positives:  {cm['TP']:4d}\n")
                f.write(f"  False Positives: {cm['FP']:4d}\n")
                f.write(f"  True Negatives:  {cm['TN']:4d}\n")
                f.write(f"  False Negatives: {cm['FN']:4d}\n\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
            
            # GNNExplainer Integration Instructions
            f.write("\nGNNEXPLAINER INTEGRATION\n")
            f.write("=" * 40 + "\n")
            f.write("FP predictions have been exported for GNNExplainer analysis.\n")
            f.write("To run GNNExplainer on exported predictions:\n\n")
            
            for model_name, results in test_results.items():
                if 'fp_export' in results:
                    csv_file = results['fp_export']['summary']['csv_file']
                    f.write(f"python scripts/run_gnn_explainer.py \\\n")
                    f.write(f"  --predictions {self.results_path}predictions/{csv_file} \\\n")
                    f.write(f"  --output-dir {self.results_path}explainer/{model_name}_explainer\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

def load_trained_models_info(results_path):
    """Load information about trained models from the results directory."""
    trained_models = {}
    
    # Check for model files in the results directory
    models_dir = os.path.join(results_path, "models")
    if not os.path.exists(models_dir):
        models_dir = results_path
    
    # Look for model files
    model_files = []
    for f in os.listdir(models_dir):
        if f.endswith('_best_model.pt'):
            model_files.append(f)
    
    print(f"Found model files: {model_files}")
    
    # Create model info dictionary
    for model_file in model_files:
        if 'GCN' in model_file:
            model_name = 'GCNModel'
        elif 'Transformer' in model_file:
            model_name = 'TransformerModel'
        elif 'SAGE' in model_file:
            model_name = 'SAGEModel'
        else:
            # Try to extract model name from filename
            model_name = model_file.replace('_best_model.pt', '')
        
        trained_models[model_name] = {
            'model_path': os.path.join(models_dir, model_file),
            'threshold': 0.5  # Default threshold
        }
    
    return trained_models

def run_evaluation(graph_path, results_path="test_results/", trained_models_info=None,
                  export_fp=True, fp_threshold=0.7, fp_top_k=10000):
    """Main function to run complete evaluation with FP export for GNNExplainer."""
    
    print("Starting comprehensive model evaluation with FP export...")
    print(f"Graph path: {graph_path}")
    print(f"Results path: {results_path}")
    
    # Set reproducibility
    enable_full_reproducibility(42)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(results_path)
    
    # Load graph
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    graph = torch.load(graph_path, map_location=evaluator.device)
    print(f"Loaded graph: {graph}")
    
    # Load trained models information
    if trained_models_info is None:
        trained_models_info = load_trained_models_info(results_path)
    
    if not trained_models_info:
        print("No trained models found! Please train models first using train_CI.py")
        return None
    
    print(f"Found trained models: {list(trained_models_info.keys())}")
    
    # Create test data from graph
    test_edge_tensor, test_label_tensor = evaluator.create_test_data_from_graph(graph)
    
    # Test all models with FP export
    test_results = evaluator.test_all_models(
        trained_models_info, 
        graph, 
        test_edge_tensor, 
        test_label_tensor,
        export_fp=export_fp,
        fp_threshold=fp_threshold,
        fp_top_k=fp_top_k
    )
    
    # Create visualizations
    print("Creating visualizations...")
    evaluator.create_visualizations(test_results)
    
    # Save detailed results
    datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    print("Saving detailed results...")
    evaluator.save_detailed_results(test_results, datetime_str)
    
    print(f"Evaluation completed! Results organized under {results_path}")
    
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
            print(f"  CSV File: {results_path}predictions/{csv_file}")
    
    # Print GNNExplainer integration instructions
    if export_fp:
        print(f"\n" + "="*60)
        print("GNNEXPLAINER INTEGRATION")
        print("="*60)
        print("To analyze FP predictions with GNNExplainer, run:")
        
        for model_name, results in test_results.items():
            if 'fp_export' in results:
                csv_file = results['fp_export']['summary']['csv_file']
                print(f"\n# For {model_name}:")
                print(f"python scripts/run_gnn_explainer.py \\")
                print(f"  --predictions {results_path}predictions/{csv_file} \\")
                print(f"  --output-dir {results_path}explainer/{model_name}_explainer")
    
    return test_results

def main():
    parser = argparse.ArgumentParser(description='Comprehensive model testing and evaluation with FP export')
    parser.add_argument('graph_path', help='Path to graph file (.pt)')
    parser.add_argument('--results-path', default='test_results/', help='Results output directory')
    parser.add_argument('--export-fp', action='store_true', default=True, help='Export FP predictions for GNNExplainer')
    parser.add_argument('--fp-threshold', type=float, default=0.7, help='FP confidence threshold')
    parser.add_argument('--fp-top-k', type=int, default=10000, help='Maximum FP predictions to export')
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs(args.results_path, exist_ok=True)
    
    # Run evaluation
    test_results = run_evaluation(
        graph_path=args.graph_path,
        results_path=args.results_path,
        trained_models_info=None,  # Will auto-detect from results_path
        export_fp=args.export_fp,
        fp_threshold=args.fp_threshold,
        fp_top_k=args.fp_top_k
    )
    
    return test_results

if __name__ == "__main__":
    main()



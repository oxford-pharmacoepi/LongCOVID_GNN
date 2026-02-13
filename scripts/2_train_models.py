#!/usr/bin/env python3
"""
Model Training Script for Drug-Disease Prediction
Trains GNN models with early stopping and comprehensive evaluation.
Supports optional Bayesian hyperparameter optimisation before training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime as dt
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import glob

# Import from shared modules
from src.models import GCNModel, TransformerModel, SAGEModel, MODEL_CLASSES, LinkPredictor
from src.features.heuristic_scores import compute_heuristic_edge_features
from src.utils.common import set_seed, enable_full_reproducibility
from src.utils.eval_utils import calculate_metrics
from src.utils.edge_utils import generate_pairs
from src.config import get_config, create_custom_config
from src.training.tracker import ExperimentTracker


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


class ModelTrainer:
    """Comprehensive model trainer with early stopping and validation."""
    
    def __init__(self, config, tracker=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = config.get_model_config()
        self.tracker = tracker
        
        print(f"Using device: {self.device}")
        
    def prepare_training_data(self, graph):
        """
        Extract pre-computed training edges from graph.
        
        Training edges (with positive and negative samples) are already computed
        in script 1 using the chosen negative sampling strategy, so we just extract them.
        """
        print("Extracting pre-computed training edges from graph...")
        
        # Check if graph has pre-computed training edges
        if hasattr(graph, 'train_edge_index') and hasattr(graph, 'train_edge_label'):
            train_edge_tensor = graph.train_edge_index
            train_label_tensor = graph.train_edge_label
            
            # Split into positive and negative edges based on labels
            pos_mask = train_label_tensor == 1
            neg_mask = train_label_tensor == 0
            
            pos_edge_index = train_edge_tensor[pos_mask].T
            neg_edge_index = train_edge_tensor[neg_mask].T
            
            # Get negative sampling strategy from graph metadata
            neg_strategy = 'unknown'
            if hasattr(graph, 'metadata') and 'graph_creation_config' in graph.metadata:
                neg_strategy = graph.metadata['graph_creation_config'].get('negative_sampling_strategy', 'unknown')
            
            print(f"✓ Extracted {pos_edge_index.size(1)} positive training edges")
            print(f"✓ Extracted {neg_edge_index.size(1)} negative training edges")
            print(f"  Training ratio: 1:{neg_edge_index.size(1) // pos_edge_index.size(1)}")
            print(f"  Negative sampling strategy: {neg_strategy}")
            
            return pos_edge_index, neg_edge_index
        
        else:
            print("ERROR: No pre-computed training edges found in graph!")
            print("The graph was created with an old version of script 1.")
            print("Please regenerate the graph using: python scripts/1_create_graph.py")
            raise ValueError("Graph must have train_edge_index and train_edge_label attributes")
    
    def prepare_training_data_synthetic(self, graph):
        """OLD SYNTHETIC METHOD - should not be used!"""
        print("WARNING: Using synthetic training data - this will give poor results!")
        
        # Extract positive edges from graph metadata
        if hasattr(graph, 'metadata') and 'edge_info' in graph.metadata:
            edge_info = graph.metadata['edge_info']
            drug_disease_edges = edge_info.get('Drug-Disease', 0)
            print(f"Found {drug_disease_edges} drug-disease edges in metadata")
        
        # Find drug-disease edges in the edge index
        # This is simplified - in practice you'd need proper node type identification
        pos_edges = []
        
        # For now, create synthetic positive edges
        # In a real implementation, you'd extract actual drug-disease edges
        num_drugs = 1000  # Approximate number of drugs
        num_diseases = 500  # Approximate number of diseases
        disease_offset = num_drugs + 100  # Approximate offset for disease nodes
        
        for i in range(min(500, graph.edge_index.size(1))):  # Sample some edges
            src, dst = graph.edge_index[:, i]
            if src < num_drugs and dst >= disease_offset:
                pos_edges.append((int(src), int(dst)))
        
        pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).T
        print(f"Extracted {pos_edge_index.size(1)} positive drug-disease edges")
        
        # Generate negative edges using seed from config
        all_drug_nodes = list(range(num_drugs))
        all_disease_nodes = list(range(disease_offset, disease_offset + num_diseases))
        
        # Create all possible drug-disease pairs
        all_possible_pairs = []
        for drug in all_drug_nodes[:100]:  # Limit for efficiency
            for disease in all_disease_nodes[:50]:
                all_possible_pairs.append((drug, disease))
        
        # Remove existing positive pairs
        pos_set = set(pos_edges)
        negative_candidates = list(set(all_possible_pairs) - pos_set)
        
        # Sample negative edges using seed from config
        rng = random.Random(self.config.seed)
        num_neg_samples = min(len(pos_edges), len(negative_candidates))
        neg_edges = rng.sample(negative_candidates, num_neg_samples)
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).T
        
        print(f"Generated {neg_edge_index.size(1)} negative edges")
        
        return pos_edge_index, neg_edge_index
    
    def train_single_model(self, model_class, model_name, graph, pos_edge_index, neg_edge_index, 
                          val_edge_tensor, val_label_tensor, results_path):
        """Train a single model with early stopping."""
        
        print(f"\nTraining {model_name}")
        print("-" * 50)
        
        # Get primary metric from config
        primary_metric = self.config.primary_metric if hasattr(self.config, 'primary_metric') else 'auc'
        print(f"Using {primary_metric.upper()} as primary metric for model selection")
        
        # ------------------------------------------------------------------
        # Phase 1: Compute Heuristics (Add to features)
        # ------------------------------------------------------------------
        print("Computing heuristic feature scores for Training & Validation...")
        # Move graph to CPU for heuristic calculation (networkx/numpy based)
        graph_cpu = graph.cpu()
        
        # 1. Training Heuristics
        # Combine pos and neg edges for batch computation: [2, N]
        train_edges_all = torch.cat([pos_edge_index.cpu(), neg_edge_index.cpu()], dim=1)
        train_heuristics = compute_heuristic_edge_features(graph_cpu, train_edges_all)
        
        # Split back
        n_pos = pos_edge_index.size(1)
        train_pos_h = train_heuristics[:n_pos]
        train_neg_h = train_heuristics[n_pos:]
        
        print(f"Computed heuristics for {n_pos} positive and {neg_edge_index.size(1)} negative training edges.")
        
        # 2. Validation Heuristics
        val_edges_T = val_edge_tensor.cpu().T
        val_heuristics = compute_heuristic_edge_features(graph_cpu, val_edges_T)
        print(f"Computed heuristics for {val_edge_tensor.size(0)} validation edges.")
        
        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        # Phase 2: Initialise Encoder and LinkPredictor
        # ------------------------------------------------------------------
        # Determine if we need edge dimensions for the encoder
        has_edge_attr = hasattr(graph, 'edge_attr') and graph.edge_attr is not None
        edge_dim = graph.edge_attr.size(1) if has_edge_attr else None
        
        # Initialise Encoder
        if model_name in ['TransformerModel', 'GATModel']:
            encoder = model_class(
                in_channels=graph.x.size(1),
                hidden_channels=self.model_config['hidden_channels'],
                out_channels=self.model_config['out_channels'],
                num_layers=self.model_config['num_layers'],
                dropout_rate=self.model_config['dropout_rate'],
                edge_dim=edge_dim
            )
        else:
            encoder = model_class(
                in_channels=graph.x.size(1),
                hidden_channels=self.model_config['hidden_channels'],
                out_channels=self.model_config['out_channels'],
                num_layers=self.model_config['num_layers'],
                dropout_rate=self.model_config['dropout_rate']
            )
            
        # Wrap in LinkPredictor
        # Use 'mlp_neighbor' or 'mlp_interaction' based on config
        decoder_type = self.model_config.get('decoder_type', 'mlp_neighbor')
        
        model = LinkPredictor(
            encoder=encoder,
            hidden_channels=self.model_config['out_channels'],
            decoder_type=decoder_type
        ).to(self.device)
        
        if has_edge_attr:
            print(f" Using edge features with {model_name} (edge_dim={edge_dim})")
            edge_attr = graph.edge_attr.float()
        else:
            edge_attr = None
        
        # Use lower learning rate for SAGE model (more stable)
        lr = self.model_config.get('lr', 0.001)
        if model_name == 'SAGEModel':
            lr = lr * 0.1  # 10x lower learning rate for SAGE
            print(f"Using reduced learning rate for SAGE: {lr:.6f}")
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr,
            weight_decay=self.model_config.get('weight_decay', 0.0)
        )
        
        # Get loss function from config
        from src.training.losses import get_loss_function
        loss_config = self.config.get_loss_config()
        loss_function = get_loss_function(
            loss_type=loss_config['loss_function'],
            **loss_config['params']
        )
        
        print(f"Using loss function: {loss_config['loss_function']}")
        
        # Log loss function to MLflow
        if self.tracker:
            self.tracker.log_param(f"{model_name}_loss_function", loss_config['loss_function'])
            for key, value in loss_config['params'].items():
                if value is not None:
                    self.tracker.log_param(f"{model_name}_loss_{key}", value)
        
        # Move data to device
        graph = graph.to(self.device)
        pos_edge_index = pos_edge_index.to(self.device)
        neg_edge_index = neg_edge_index.to(self.device)
        train_pos_h = train_pos_h.to(self.device)
        train_neg_h = train_neg_h.to(self.device)
        val_heuristics = val_heuristics.to(self.device)
        val_edge_tensor = val_edge_tensor.to(self.device)
        val_label_tensor = val_label_tensor.to(self.device)
        if has_edge_attr:
            edge_attr = edge_attr.to(self.device)
        
        # Training tracking - track both loss and primary metric
        best_val_loss = float('inf')
        best_val_metric = -1.0  # For AUC, APR, F1, etc. (higher is better)
        counter = 0
        best_threshold = 0.5
        train_losses = []
        val_losses = []
        val_metrics_history = []  # Track primary metric over time
        
        # Create models directory and set model save path
        models_dir = os.path.join(results_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = os.path.join(models_dir, f"{model_name}_best_model_{timestamp}.pt")
        
        print(f"Training for {self.model_config['num_epochs']} epochs with patience {self.model_config['patience']}")
        
        # Use tqdm with explicit settings for better compatibility
        import sys
        for epoch in tqdm(range(self.model_config['num_epochs']), 
                         desc=f'Training {model_name}',
                         file=sys.stdout,
                         ncols=80,
                         mininterval=1.0):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            # Forward pass with edge features if available
            if has_edge_attr:
                z = model(graph.x.float(), graph.edge_index, edge_attr=edge_attr)
            else:
                z = model(graph.x.float(), graph.edge_index)
            
            # Normalise embeddings to unit sphere (prevents saturation)
            z = F.normalize(z, p=2, dim=1)
            
            # Compute edge scores using LinkPredictor
            # It handles dot product + heuristic injection + boosting
            pos_scores = model.decode(z, pos_edge_index, heuristic_features=train_pos_h)
            neg_scores = model.decode(z, neg_edge_index, heuristic_features=train_neg_h)
            
            # Compute loss using the custom loss function
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores)
            ])
            loss = loss_function(all_scores, all_labels)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Log to MLflow (every 10 epochs to avoid clutter)
            if self.tracker and epoch % 10 == 0:
                self.tracker.log_training_metrics(epoch, loss.item())
            
            # Validation phase (every 5 epochs)
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # Forward pass with edge features if available
                    if has_edge_attr:
                        z = model(graph.x.float(), graph.edge_index, edge_attr=edge_attr)
                    else:
                        z = model(graph.x.float(), graph.edge_index)
                    
                    # Normalise embeddings in validation too
                    z = F.normalize(z, p=2, dim=1)
                    
                    # LinkPredictor decode. val_edge_tensor is [N, 2] -> edge_index [2, N]
                    val_edge_index = val_edge_tensor.T
                    val_scores = model.decode(z, val_edge_index, heuristic_features=val_heuristics)
                    val_loss = loss_function(val_scores, val_label_tensor.float())
                    val_probs = torch.sigmoid(val_scores)
                    
                    # Calculate validation metrics
                    val_preds = (val_probs >= 0.5).float()
                    val_metrics = calculate_metrics(
                        val_label_tensor.cpu().numpy(),
                        val_probs.cpu().numpy(),
                        val_preds.cpu().numpy()
                    )
                    
                    # Get the primary metric value
                    # We use recall@k as a more sensitive optimisation proxy if hits_at_k is requested.
                    pm_lower = primary_metric.lower()
                    if pm_lower.startswith('hits_at_'):
                        k_val = pm_lower.split('_')[-1]
                        eval_metric = f'recall@{k_val}'
                        print(f"  Note: Using {eval_metric} instead of {pm_lower} for more sensitive optimisation")
                    else:
                        eval_metric = pm_lower

                    current_metric = val_metrics.get(eval_metric)
                    if current_metric is None:
                        # Try mapping hits_at_20 to hits@20 or vice versa
                        alt_metric = eval_metric.replace('_at_', '@') if '_at_' in eval_metric else eval_metric.replace('@', '_at_')
                        current_metric = val_metrics.get(alt_metric)
                    
                    if current_metric is None:
                        # Final fallback
                        current_metric = val_metrics.get(pm_lower, 0.0)
                    
                    val_threshold = val_probs.mean().item()
                    
                    val_losses.append(val_loss.item())
                    val_metrics_history.append(current_metric)
                    
                    # Log validation metrics to MLflow
                    if self.tracker:
                        self.tracker.log_training_metrics(
                            epoch, 
                            loss.item(), 
                            val_loss.item(), 
                            {primary_metric: current_metric, 'threshold': val_threshold}
                        )
                    
                    # Check for improvement based on primary metric
                    if current_metric >= best_val_metric:
                        best_val_loss = val_loss.item()
                        best_val_metric = current_metric
                        best_threshold = val_threshold
                        counter = 0
                        
                        # Save best model
                        torch.save(model.state_dict(), model_path)
                        
                        if epoch % 50 == 0:
                            print(f"Epoch {epoch+1}: New best {primary_metric.upper()}: {best_val_metric:.4f}, Loss: {best_val_loss:.4f}")
                    else:
                        counter += 1
                    
                    # Early stopping
                    if counter >= self.model_config['patience']:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model.eval()
        
        # Final validation with comprehensive metrics
        with torch.no_grad():
            # Forward pass with edge features if available
            if has_edge_attr:
                z = model(graph.x.float(), graph.edge_index, edge_attr=edge_attr)
            else:
                z = model(graph.x.float(), graph.edge_index)
            
            # Normalise embeddings (must match training validation!)
            z = F.normalize(z, p=2, dim=1)
            
            val_edge_index = val_edge_tensor.T
            val_scores = model.decode(z, val_edge_index, heuristic_features=val_heuristics)
            val_probs = torch.sigmoid(val_scores)
            val_preds = (val_probs >= best_threshold).float()
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(
                val_label_tensor.cpu().numpy(),
                val_probs.cpu().numpy(),
                val_preds.cpu().numpy()
            )
        
        # Create training plots
        self._create_training_plots(model_name, train_losses, val_losses, val_metrics_history, 
                                    results_path, timestamp, primary_metric)
        
        # Log plots and model to MLflow
        if self.tracker:
            plot_path = os.path.join(results_path, f"{model_name}_training_curves_{timestamp}.png")
            if os.path.exists(plot_path):
                self.tracker.log_artifact(plot_path, f"training_plots/{model_name}")
            self.tracker.log_model(model, model_path)
            
            # Log final metrics (sanitise metric name for MLflow)
            sanitised_metric = primary_metric.replace('@', '_')
            self.tracker.log_metric(f"{model_name}_final_val_{sanitised_metric}", best_val_metric)
            self.tracker.log_metric(f"{model_name}_final_val_loss", best_val_loss)
            self.tracker.log_metric(f"{model_name}_best_threshold", best_threshold)
            self.tracker.log_metric(f"{model_name}_total_epochs", len(train_losses))
        
        print(f"Training completed for {model_name}")
        print(f"Best validation {primary_metric.upper()}: {best_val_metric:.4f}")
        print(f"Model saved to: {model_path}")
        
        return {
            'model_path': model_path,
            'model_class': model_class,
            'threshold': best_threshold,
            'validation_metric': best_val_metric,  # Changed from validation_auc
            'validation_metrics': metrics,
            'primary_metric': primary_metric,  # Store which metric was used
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_metrics': val_metrics_history  # Changed from val_aucs
            }
        }
    
    def _create_training_plots(self, model_name, train_losses, val_losses, val_metrics, 
                              results_path, timestamp, metric_name='auc'):
        """Create training visualisation plots."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Training loss
        axes[0].plot(train_losses, label='Training Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Validation loss (every 5 epochs)
        val_epochs = list(range(0, len(train_losses), 5))[:len(val_losses)]
        axes[1].plot(val_epochs, val_losses, label='Validation Loss', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'{model_name} - Validation Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Validation primary metric
        axes[2].plot(val_epochs, val_metrics, label=f'Validation {metric_name.upper()}', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel(metric_name.upper())
        axes[2].set_title(f'{model_name} - Validation {metric_name.upper()}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(results_path, f"{model_name}_training_curves_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {plot_path}")
    
    def train_all_models(self, graph_path, results_path):
        """Train all model types and compare results."""
        
        print("Loading graph and preparing data...")
        
        # Load graph
        graph = torch.load(graph_path, map_location=self.device, weights_only=False)
        print(f"Loaded graph: {graph}")
        
        # Log graph information to MLflow
        if self.tracker:
            self.tracker.log_param("graph_path", graph_path)
            self.tracker.log_metric("graph_num_nodes", graph.x.size(0))
            self.tracker.log_metric("graph_num_edges", graph.edge_index.size(1))
            self.tracker.log_metric("graph_num_features", graph.x.size(1))
        
        # Prepare training data
        pos_edge_index, neg_edge_index = self.prepare_training_data(graph)
        
        # Log training data stats
        if self.tracker:
            self.tracker.log_metric("train_pos_edges", pos_edge_index.size(1))
            self.tracker.log_metric("train_neg_edges", neg_edge_index.size(1))
            self.tracker.log_metric("train_pos_neg_ratio", neg_edge_index.size(1) / pos_edge_index.size(1))
        
        # Extract validation data from graph
        val_edge_tensor = graph.val_edge_index if hasattr(graph, 'val_edge_index') else torch.empty((0, 2), dtype=torch.long)
        val_label_tensor = graph.val_edge_label if hasattr(graph, 'val_edge_label') else torch.empty(0, dtype=torch.long)
        
        if val_edge_tensor.size(0) == 0:
            print("No validation data found in graph, creating synthetic validation set...")
            # Create synthetic validation data
            val_pos_size = min(50, pos_edge_index.size(1) // 10)
            val_neg_size = min(50, neg_edge_index.size(1) // 10)
            
            val_pos_edges = pos_edge_index[:, :val_pos_size].T
            val_neg_edges = neg_edge_index[:, :val_neg_size].T
            
            val_edge_tensor = torch.cat([val_pos_edges, val_neg_edges], dim=0)
            val_label_tensor = torch.cat([
                torch.ones(val_pos_size, dtype=torch.long),
                torch.zeros(val_neg_size, dtype=torch.long)
            ])
        
        print(f"Validation set: {val_edge_tensor.size(0)} samples")
        
        # DIAGNOSTIC: Print validation set composition
        print("\n" + "="*60)
        print("VALIDATION SET DIAGNOSTICS:")
        print("="*60)
        num_pos = (val_label_tensor == 1).sum().item()
        num_neg = (val_label_tensor == 0).sum().item()
        print(f"Positive samples: {num_pos}")
        print(f"Negative samples: {num_neg}")
        print(f"Pos:Neg ratio: 1:{num_neg/num_pos if num_pos > 0 else 0:.2f}")
        print(f"Class balance: {num_pos/(num_pos + num_neg)*100:.1f}% positive")
        
        # Check if validation set is too small
        if val_edge_tensor.size(0) < 1000:
            print(f"\n  WARNING: Validation set is very small ({val_edge_tensor.size(0)} samples)")
            print("   This may lead to unreliable metrics!")
        
        if num_neg < 100:
            print(f"\n  WARNING: Very few negative samples ({num_neg})")
            print("   Precision@K metrics will be inflated!")
            print("   Recommendation: Increase negative samples in validation set")
        
        if num_pos < 100:
            print(f"\n  WARNING: Fewer than 100 positive samples ({num_pos})")
            print("   Recall@100 will be capped by number of positives!")
        
        print("="*60 + "\n")
        
        # Models to train - supports single model selection
        all_models = {
            'GCNModel': GCNModel,
            'TransformerModel': TransformerModel,
            'SAGEModel': SAGEModel
        }
        
        # Determine which models to train based on --model argument
        if hasattr(self.config, 'model_choice') and self.config.model_choice != 'all':
            model_choice = self.config.model_choice
            # Map simple names to full names
            model_map = {
                'GCN': 'GCNModel',
                'Transformer': 'TransformerModel',
                'SAGE': 'SAGEModel',
                'GCNModel': 'GCNModel',
                'TransformerModel': 'TransformerModel',
                'SAGEModel': 'SAGEModel'
            }
            model_name = model_map.get(model_choice, 'TransformerModel')
            models_to_train = {model_name: all_models[model_name]}
            print(f"\nTraining only {model_name} (as specified)")
        else:
            models_to_train = all_models
            print(f"\nTraining all {len(models_to_train)} models")
        
        # Train all models
        training_results = {}
        
        for model_name, model_class in models_to_train.items():
            # Set seed for reproducibility
            config = get_config()
            enable_full_reproducibility(config.seed)
            
            try:
                result = self.train_single_model(
                    model_class, model_name, graph, pos_edge_index, neg_edge_index,
                    val_edge_tensor, val_label_tensor, results_path
                )
                training_results[model_name] = result
                
                # Print validation results
                metrics = result['validation_metrics']
                print(f"\n{model_name} Validation Results:")
                print(f"  AUC: {metrics['auc']:.4f}")
                print(f"  APR: {metrics['apr']:.4f}")
                print(f"  F1:  {metrics['f1']:.4f}")
                print(f"  Acc: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Create comparison visualisation
        self._create_model_comparison(training_results, results_path)
        
        # Log comparison plot to MLflow
        if self.tracker:
            timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            comparison_path = os.path.join(results_path, f"model_comparison_{timestamp}.png")
            if os.path.exists(comparison_path):
                self.tracker.log_artifact(comparison_path, "comparison")
        
        # Save training summary
        self._save_training_summary(training_results, results_path)
        
        # Log summary to MLflow
        if self.tracker:
            summary_path = os.path.join(results_path, f"training_summary_{timestamp}.json")
            if os.path.exists(summary_path):
                self.tracker.log_artifact(summary_path, "summaries")
        
        print(f"\nTraining completed for all models!")
        print(f"Results saved to: {results_path}")
        
        return training_results
    
    def _create_model_comparison(self, training_results, results_path):
        """Create model comparison visualisation."""
        
        if not training_results:
            return
        
        model_names = list(training_results.keys())
        metrics_names = ['auc', 'apr', 'f1', 'accuracy', 'precision', 'recall']
        
        # Prepare data for plotting
        metrics_data = {}
        for metric in metrics_names:
            metrics_data[metric] = []
            for model_name in model_names:
                if 'validation_metrics' in training_results[model_name]:
                    value = training_results[model_name]['validation_metrics'].get(metric, 0)
                    metrics_data[metric].append(value)
                else:
                    metrics_data[metric].append(0)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['steelblue', 'coral', 'lightgreen']
        
        for i, metric in enumerate(metrics_names):
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
        
        plt.suptitle('Model Comparison - Validation Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        plot_path = os.path.join(results_path, f"model_comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plot saved to: {plot_path}")
    
    def _save_training_summary(self, training_results, results_path):
        """Save comprehensive training summary."""
        
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Create summary data
        summary_data = {
            'timestamp': timestamp,
            'training_config': self.model_config,
            'device': str(self.device),
            'models_trained': len(training_results),
            'model_results': {}
        }
        
        for model_name, result in training_results.items():
            summary_data['model_results'][model_name] = {
                'model_path': result['model_path'],
                'threshold': result['threshold'],
                'validation_metric': result['validation_metric'],
                'validation_metrics': result['validation_metrics']
            }
        
        # Save as JSON
        summary_path = os.path.join(results_path, f"training_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Save as text report
        report_path = os.path.join(results_path, f"training_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write("DRUG-DISEASE PREDICTION MODEL TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training completed: {timestamp}\n")
            f.write(f"Device used: {self.device}\n")
            f.write(f"Models trained: {len(training_results)}\n\n")
            
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.model_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("MODEL RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for model_name, result in training_results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Model Path: {result['model_path']}\n")
                f.write(f"  Threshold: {result['threshold']:.4f}\n")
                f.write(f"  Validation Metric ({result['primary_metric']}): {result['validation_metric']:.4f}\n")
                
                if 'validation_metrics' in result:
                    metrics = result['validation_metrics']
                    f.write(f"  Validation Metrics:\n")
                    f.write(f"    AUC: {metrics.get('auc', 0):.4f}\n")
                    f.write(f"    APR: {metrics.get('apr', 0):.4f}\n")
                    f.write(f"    F1:  {metrics.get('f1', 0):.4f}\n")
                    f.write(f"    Accuracy: {metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"    Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"    Recall: {metrics.get('recall', 0):.4f}\n")
            
            # Model ranking
            f.write(f"\nMODEL RANKING BY VALIDATION {self.config.primary_metric.upper()}:\n")
            f.write("-" * 40 + "\n")
            
            sorted_models = sorted(training_results.items(), 
                                 key=lambda x: x[1]['validation_metric'], reverse=True)
            for i, (model_name, result) in enumerate(sorted_models):
                f.write(f"{i+1}. {model_name}: {result['validation_metric']:.4f}\n")
        
        print(f"Training summary saved to: {summary_path}")
        print(f"Training report saved to: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train GNN models for drug-disease prediction')
    parser.add_argument('--graph', type=str, help='Path to graph file (.pt) - auto-detects latest if not provided')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['GCN', 'Transformer', 'SAGE', 'all'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides config)')
    parser.add_argument('--output-dir', type=str, default='results/', help='Output directory')
    
    # Bayesian optimisation options
    parser.add_argument('--optimise-first', action='store_true',
                       help='Run Bayesian hyperparameter optimisation before training')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimisation trials (default: 50, only used with --optimise-first)')
    
    args = parser.parse_args()
    
    # Auto-detect graph if not provided
    if not args.graph:
        args.graph = find_latest_graph(args.output_dir)
    
    # FORCE RELOAD CONFIG MODULE TO PICK UP ANY CHANGES
    import importlib
    import src.config
    importlib.reload(src.config)
    
    # Load configuration from config.py (freshly reloaded)
    config = get_config()
    
    # Set model choice for training (overrides config default)
    if args.model != 'all':
        config.model_choice = args.model
        
    # Set epochs for training (overrides config default)
    if args.epochs:
        config.model_config['num_epochs'] = args.epochs
        
    # Set output directory
    if args.output_dir:
        config.results_path = args.output_dir
    
    # Bayesian Optimisation (if requested)
    if args.optimise_first:
        print("\n" + "="*80)
        print("BAYESIAN HYPERPARAMETER OPTIMISATION")
        print("="*80)
        print(f"Running optimisation with {args.n_trials} trials before training...")
        print("="*80 + "\n")
        
        try:
            from src.bayesian_optimiser import BayesianOptimiser
            
            # Load graph for optimisation
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            graph = torch.load(args.graph, map_location=device, weights_only=False)
            
            # Run optimisation
            optimiser = BayesianOptimiser(
                graph=graph,
                config=config,
                n_trials=args.n_trials,
                device=device,
                mlflow_tracking=True,
                results_dir=os.path.join(args.output_dir, 'bayesian_optimisation')
            )
            
            results = optimiser.optimise()
            
            # Apply best hyperparameters to config
            best_params = results['best_params']
            config.model_config.update(best_params)
            
            print("\n" + "="*80)
            print("OPTIMISATION COMPLETE - Applying best hyperparameters")
            print("="*80)
            print(f"Best {config.primary_metric.upper()}: {results['best_value']:.4f}")
            print("\nOptimised hyperparameters:")
            for key, value in best_params.items():
                print(f"  {key:20s}: {value}")
            print("="*80 + "\n")
            
            # Save optimised config
            opt_config_path = os.path.join(args.output_dir, 'bayesian_optimisation', 'applied_config.json')
            os.makedirs(os.path.dirname(opt_config_path), exist_ok=True)
            with open(opt_config_path, 'w') as f:
                json.dump({
                    'model_config': config.model_config,
                    'best_params': best_params,
                    'optimisation_results': {
                        'best_value': results['best_value'],
                        'n_trials': args.n_trials,
                        'model_choice': config.model_choice,
                        'primary_metric': config.primary_metric
                    }
                }, f, indent=2)
            
            print(f"Optimised config saved to: {opt_config_path}\n")
            
        except ImportError:
            print("ERROR: Bayesian optimiser not available. Please install optuna:")
            print("  pip install optuna>=3.5.0")
            return None
        except Exception as e:
            print(f"ERROR during optimisation: {e}")
            print("Continuing with default hyperparameters...")
    
    # Print config to verify it's correct
    print("\n" + "="*60)
    print("CONFIGURATION FOR TRAINING:")
    print("="*60)
    print(f"Loss function: {config.loss_function}")
    print(f"Primary metric: {config.primary_metric}")
    print(f"Negative sampling strategy: {config.negative_sampling_strategy}")
    print(f"Pos:Neg ratio: 1:{config.pos_neg_ratio}")
    
    # Print model config
    print("\nModel hyperparameters:")
    model_cfg = config.get_model_config()
    for key, value in model_cfg.items():
        print(f"  {key:20s}: {value}")
    print("="*60 + "\n")
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set reproducibility
    enable_full_reproducibility(config.seed)  
    
    # Initialise MLflow tracker
    tracker = ExperimentTracker(experiment_name='model_training')
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"training_{timestamp}"
    
    try:
        tracker.start_run(run_name=run_name)
        
        # Log configuration
        tracker.log_config(config)
        
        # Log additional parameters
        tracker.log_param("graph_path", args.graph)
        tracker.log_param("output_dir", args.output_dir)
        tracker.log_param("model_to_train", args.model)
        tracker.log_param("optimised_hyperparams", args.optimise_first)
        if args.optimise_first:
            tracker.log_param("optimisation_trials", args.n_trials)
        
        # Auto-detect graph if not provided
        graph_path = args.graph or find_latest_graph()
        
        # Initialise trainer
        trainer = ModelTrainer(config, tracker)
        
        # Train models
        training_results = trainer.train_all_models(graph_path, args.output_dir)
        
        if training_results:
            print("\nTRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            
            # Print summary with ranking metrics
            for model_name, result in training_results.items():
                val_metrics = result.get('validation_metrics', {})
                ranking_metrics = val_metrics.get('ranking_metrics', {})
                
                # Get recall@k and precision@k
                k_value = config.recall_k if hasattr(config, 'recall_k') else 100
                recall_at_k = ranking_metrics.get(f'recall@{k_value}', 0.0)
                precision_at_k = ranking_metrics.get(f'precision@{k_value}', 0.0)
                
                print(f"{model_name}:")
                print(f"  {result['primary_metric'].upper()} = {result['validation_metric']:.4f}")
                print(f"  Recall@{k_value} = {recall_at_k:.4f}")
                print(f"  Precision@{k_value} = {precision_at_k:.4f}")
            
            best_model = max(training_results.items(), key=lambda x: x[1]['validation_metric'])
            print(f"\nBest model: {best_model[0]} ({best_model[1]['primary_metric'].upper()} = {best_model[1]['validation_metric']:.4f})")
            
            # Log best model info (sanitise metric name for MLflow)
            tracker.log_param("best_model", best_model[0])
            sanitised_metric = best_model[1]['primary_metric'].replace('@', '_')
            tracker.log_metric(f"best_model_{sanitised_metric}", best_model[1]['validation_metric'])
            
            print(f"\nMLflow tracking URI: {tracker.experiment_name}")
            print(f"Run ID: {tracker.run_id}")
            
            tracker.end_run()
            return training_results
        else:
            print("Training failed!")
            tracker.end_run()
            return None
            
    except Exception as e:
        print(f"Error during training: {e}")
        tracker.end_run()
        raise


if __name__ == "__main__":
    main()

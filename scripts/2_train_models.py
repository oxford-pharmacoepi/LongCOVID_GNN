#!/usr/bin/env python3
"""
Model Training Script for Drug-Disease Prediction
Trains GNN models with early stopping and comprehensive evaluation.
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

# Import from shared modules
from src.models import GCNModel, TransformerModel, SAGEModel, MODEL_CLASSES
from src.utils import set_seed, enable_full_reproducibility, calculate_metrics, generate_pairs
from src.config import get_config, create_custom_config


class ModelTrainer:
    """Comprehensive model trainer with early stopping and validation."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = config.get_model_config()
        
        print(f"Using device: {self.device}")
        
    def prepare_training_data(self, graph):
        """Prepare positive and negative edges for training."""
        print("Preparing training data...")
        
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
        
        # Generate negative edges
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
        
        # Sample negative edges
        random.seed(42)
        num_neg_samples = min(len(pos_edges), len(negative_candidates))
        neg_edges = random.sample(negative_candidates, num_neg_samples)
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).T
        
        print(f"Generated {neg_edge_index.size(1)} negative edges")
        
        return pos_edge_index, neg_edge_index
    
    def train_single_model(self, model_class, model_name, graph, pos_edge_index, neg_edge_index, 
                          val_edge_tensor, val_label_tensor, results_path):
        """Train a single model with early stopping."""
        
        print(f"\nTraining {model_name}")
        print("-" * 50)
        
        # Initialize model
        model = model_class(
            in_channels=graph.x.size(1),
            hidden_channels=self.model_config['hidden_channels'],
            out_channels=self.model_config['out_channels'],
            num_layers=self.model_config['num_layers'],
            dropout_rate=self.model_config['dropout_rate']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config['learning_rate'])
        loss_function = torch.nn.BCEWithLogitsLoss()
        
        # Move data to device
        graph = graph.to(self.device)
        pos_edge_index = pos_edge_index.to(self.device)
        neg_edge_index = neg_edge_index.to(self.device)
        val_edge_tensor = val_edge_tensor.to(self.device)
        val_label_tensor = val_label_tensor.to(self.device)
        
        # Training tracking
        best_val_loss = float('inf')
        best_val_auc = 0.0
        counter = 0
        best_threshold = 0.5
        train_losses = []
        val_losses = []
        val_aucs = []
        
        # Model save path
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = os.path.join(results_path, f"{model_name}_best_model_{timestamp}.pt")
        
        print(f"Training for {self.model_config['num_epochs']} epochs with patience {self.model_config['patience']}")
        
        for epoch in tqdm(range(self.model_config['num_epochs']), desc=f'Training {model_name}'):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            z = model(graph.x.float(), graph.edge_index)
            
            # Compute edge scores
            pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
            neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
            
            # Compute loss
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            loss = pos_loss + neg_loss
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation phase (every 5 epochs)
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    z = model(graph.x.float(), graph.edge_index)
                    val_scores = (z[val_edge_tensor[:, 0]] * z[val_edge_tensor[:, 1]]).sum(dim=-1)
                    val_loss = loss_function(val_scores, val_label_tensor.float())
                    val_probs = torch.sigmoid(val_scores)
                    
                    # Calculate validation AUC
                    from sklearn.metrics import roc_auc_score
                    try:
                        val_auc = roc_auc_score(val_label_tensor.cpu().numpy(), val_probs.cpu().numpy())
                    except ValueError:
                        val_auc = 0.5  # Default if calculation fails
                    
                    val_threshold = val_probs.mean().item()
                    
                    val_losses.append(val_loss.item())
                    val_aucs.append(val_auc)
                    
                    # Check for improvement
                    if val_auc > best_val_auc:
                        best_val_loss = val_loss.item()
                        best_val_auc = val_auc
                        best_threshold = val_threshold
                        counter = 0
                        
                        # Save best model
                        torch.save(model.state_dict(), model_path)
                        
                        if epoch % 50 == 0:
                            print(f"Epoch {epoch+1}: New best AUC: {best_val_auc:.4f}, Loss: {best_val_loss:.4f}")
                    else:
                        counter += 1
                    
                    # Early stopping
                    if counter >= self.model_config['patience']:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Final validation
        with torch.no_grad():
            z = model(graph.x.float(), graph.edge_index)
            val_scores = (z[val_edge_tensor[:, 0]] * z[val_edge_tensor[:, 1]]).sum(dim=-1)
            val_probs = torch.sigmoid(val_scores)
            val_preds = (val_probs >= best_threshold).float()
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(
                val_label_tensor.cpu().numpy(),
                val_probs.cpu().numpy(),
                val_preds.cpu().numpy()
            )
        
        # Create training plots
        self._create_training_plots(model_name, train_losses, val_losses, val_aucs, results_path, timestamp)
        
        print(f"Training completed for {model_name}")
        print(f"Best validation AUC: {best_val_auc:.4f}")
        print(f"Model saved to: {model_path}")
        
        return {
            'model_path': model_path,
            'model_class': model_class,
            'threshold': best_threshold,
            'validation_auc': best_val_auc,
            'validation_metrics': metrics,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_aucs': val_aucs
            }
        }
    
    def _create_training_plots(self, model_name, train_losses, val_losses, val_aucs, results_path, timestamp):
        """Create training visualization plots."""
        
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
        
        # Validation AUC
        axes[2].plot(val_epochs, val_aucs, label='Validation AUC', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title(f'{model_name} - Validation AUC')
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
        graph = torch.load(graph_path, map_location=self.device)
        print(f"Loaded graph: {graph}")
        
        # Prepare training data
        pos_edge_index, neg_edge_index = self.prepare_training_data(graph)
        
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
        
        # Models to train
        models_to_train = {
            'GCNModel': GCNModel,
            'TransformerModel': TransformerModel,
            'SAGEModel': SAGEModel
        }
        
        # Train all models
        training_results = {}
        
        for model_name, model_class in models_to_train.items():
            # Set seed for reproducibility
            set_seed(42)
            
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
        
        # Create comparison visualization
        self._create_model_comparison(training_results, results_path)
        
        # Save training summary
        self._save_training_summary(training_results, results_path)
        
        print(f"\nTraining completed for all models!")
        print(f"Results saved to: {results_path}")
        
        return training_results
    
    def _create_model_comparison(self, training_results, results_path):
        """Create model comparison visualization."""
        
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
                'validation_auc': result['validation_auc'],
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
                f.write(f"  Validation AUC: {result['validation_auc']:.4f}\n")
                
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
            f.write(f"\nMODEL RANKING BY VALIDATION AUC:\n")
            f.write("-" * 40 + "\n")
            
            sorted_models = sorted(training_results.items(), 
                                 key=lambda x: x[1]['validation_auc'], reverse=True)
            for i, (model_name, result) in enumerate(sorted_models):
                f.write(f"{i+1}. {model_name}: {result['validation_auc']:.4f}\n")
        
        print(f"Training summary saved to: {summary_path}")
        print(f"Training report saved to: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train drug-disease prediction models')
    parser.add_argument('graph_path', help='Path to graph file (.pt)')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--results-path', type=str, default='results/models/', help='Results output directory')
    parser.add_argument('--models', nargs='+', choices=['GCN', 'Transformer', 'SAGE', 'all'], 
                       default=['all'], help='Models to train')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_path, exist_ok=True)
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = create_custom_config(**config_dict)
    else:
        config = get_config()
    
    # Set reproducibility
    enable_full_reproducibility(42)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train models
    training_results = trainer.train_all_models(args.graph_path, args.results_path)
    
    if training_results:
        print("\nTRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Print summary
        for model_name, result in training_results.items():
            print(f"{model_name}: AUC = {result['validation_auc']:.4f}")
        
        best_model = max(training_results.items(), key=lambda x: x[1]['validation_auc'])
        print(f"\nBest model: {best_model[0]} (AUC = {best_model[1]['validation_auc']:.4f})")
        
        return training_results
    else:
        print("Training failed!")
        return None


if __name__ == "__main__":
    main()

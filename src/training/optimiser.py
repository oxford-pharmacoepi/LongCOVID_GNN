#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimisation Module

A clean, modular system for optimising GNN hyperparameters using Optuna.

Date: January 2026
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import numpy as np

from src.models import GCNModel, TransformerModel, SAGEModel, MODEL_CLASSES
from src.utils.common import enable_full_reproducibility
from src.utils.eval_utils import calculate_metrics
from src.training.tracker import ExperimentTracker


class BayesianOptimiser:
    """
    Bayesian hyperparameter optimiser for GNN models.
    
    Uses Optuna with TPE sampler for efficient hyperparameter search.
    Integrates with existing GNN pipeline.
    
    Features:
    - Multi-model support (GCN, Transformer, SAGE)
    - Automatic early stopping with pruning
    - MLflow integration for tracking
    - Visualisation of optimisation history
    - Export of best hyperparameters
    """
    
    def __init__(
        self,
        graph,
        config,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        device: Optional[str] = None,
        mlflow_tracking: bool = True,
        results_dir: str = "results/bayesian_optimisation"
    ):
        """
        Initialise Bayesian optimiser.
        
        Args:
            graph: PyTorch Geometric graph object with train/val/test splits
            config: Config object from src.config
            n_trials: Number of optimisation trials
            study_name: Name for the Optuna study (auto-generated if None)
            storage: Optuna storage backend (None = in-memory)
            device: Device to use (None = auto-detect)
            mlflow_tracking: Enable MLflow tracking
            results_dir: Directory to save results
        """
        self.graph = graph
        self.config = config
        self.n_trials = n_trials
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlflow_tracking = mlflow_tracking
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create study name if not provided
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"gnn_optimisation_{timestamp}"
        self.study_name = study_name
        
        # Initialise MLflow tracker
        self.mlflow_tracker = None
        if mlflow_tracking:
            self.mlflow_tracker = ExperimentTracker(
                experiment_name="Bayesian_Hyperparameter_Optimisation"
            )
        
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # Maximise validation metric
            sampler=TPESampler(seed=config.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            storage=storage,
            load_if_exists=True
        )
        
        # Prepare data
        self.graph = self.graph.to(self.device)
        self._prepare_validation_data()
        
        print(f"Bayesian Optimiser initialised:")
        print(f"  Study name: {study_name}")
        print(f"  N trials: {n_trials}")
        print(f"  Device: {self.device}")
        print(f"  Results dir: {self.results_dir}")
    
    def _prepare_validation_data(self):
        """Prepare validation data from graph."""
        if hasattr(self.graph, 'val_edge_index') and hasattr(self.graph, 'val_edge_label'):
            self.val_edges = self.graph.val_edge_index.to(self.device)
            self.val_labels = self.graph.val_edge_label.to(self.device)
            print(f"  Validation set: {len(self.val_labels)} samples")
        else:
            raise ValueError("Graph must have val_edge_index and val_edge_label attributes")
        
        if hasattr(self.graph, 'train_edge_index') and hasattr(self.graph, 'train_edge_label'):
            self.train_edges = self.graph.train_edge_index.to(self.device)
            self.train_labels = self.graph.train_edge_label.to(self.device)
            print(f"  Training set: {len(self.train_labels)} samples")
        else:
            raise ValueError("Graph must have train_edge_index and train_edge_label attributes")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space.
        
        Override this method to customise the search space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            # Learning rate 
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 5e-3, log=True),
            
            # Weight decay 
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
            
            # Hidden channels
            'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64, 128]),
            
            # Output channels 
            'out_channels': trial.suggest_categorical('out_channels', [16, 32, 64]),
            
            # Number of layers
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            
            # Dropout rate 
            'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.6),
            
            # Batch size 
            'batch_size': trial.suggest_categorical('batch_size', [1024, 2048]),
        }
        
        # Model-specific parameters
        model_choice = self.config.model_choice
        if model_choice == 'Transformer':
            # Heads 
            params['heads'] = trial.suggest_categorical('heads', [2, 4])
            
            params['concat'] = trial.suggest_categorical('concat', [True, False])
        
        return params
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimisation.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation metric score
        """
        # Sample hyperparameters
        params = self.define_search_space(trial)
        
        # Log trial to MLflow
        if self.mlflow_tracker:
            run_name = f"trial_{trial.number:04d}"
            self.mlflow_tracker.start_run(run_name=run_name)
            
            # Log hyperparameters
            for key, value in params.items():
                self.mlflow_tracker.log_param(key, value)
            self.mlflow_tracker.log_param("trial_number", trial.number)
        
        try:
            # Initialise model
            model = self._create_model(params)
            
            # Train model
            best_val_metric = self._train_model(model, params, trial)
            
            # Check for NaN or invalid values
            if best_val_metric is None or np.isnan(best_val_metric) or np.isinf(best_val_metric):
                raise ValueError(f"Invalid metric value: {best_val_metric}")
            
            # Log results
            if self.mlflow_tracker:
                self.mlflow_tracker.log_metric(f"best_val_{self.config.primary_metric}", best_val_metric)
                self.mlflow_tracker.end_run()
            
            return best_val_metric
            
        except optuna.TrialPruned:
            # Re-raise pruning exceptions
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()
            raise
            
        except Exception as e:
            # Print full error with traceback
            import traceback
            print(f"\nTrial {trial.number} failed with error:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print(f"  Hyperparameters: {params}")
            print(f"  Traceback:")
            traceback.print_exc()
            
            if self.mlflow_tracker:
                self.mlflow_tracker.log_param("error", str(e))
                self.mlflow_tracker.end_run()
            
            # Prune the trial
            raise optuna.TrialPruned()
    
    def _create_model(self, params: Dict[str, Any]):
        """Create model with given hyperparameters."""
        model_choice = self.config.model_choice
        
        if model_choice == 'GCN':
            model_class = GCNModel
        elif model_choice == 'Transformer':
            model_class = TransformerModel
        elif model_choice == 'SAGE':
            model_class = SAGEModel
        else:
            # Default to Transformer
            model_class = TransformerModel
        
        # Create model with trial hyperparameters
        # Only pass heads and concat to Transformer
        if model_choice == 'Transformer':
            model = model_class(
                in_channels=self.graph.x.size(1),
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                dropout_rate=params['dropout_rate'],
                heads=params.get('heads', 4),
                concat=params.get('concat', False)
            ).to(self.device)
        else:
            # GCN and SAGE don't use heads/concat
            model = model_class(
                in_channels=self.graph.x.size(1),
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                dropout_rate=params['dropout_rate']
            ).to(self.device)
        
        return model
    
    def _train_model(self, model, params: Dict[str, Any], trial: optuna.Trial) -> float:
        """
        Train model and return best validation metric.
        
        Args:
            model: Model to train
            params: Hyperparameters
            trial: Optuna trial for pruning
            
        Returns:
            Best validation metric
        """
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        batch_size = params['batch_size']
        patience = 10
        best_val_metric = 0.0
        patience_counter = 0
        
        # Training loop (max 100 epochs for hyperparameter search)
        max_epochs = min(100, self.config.model_config.get('num_epochs', 100))
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = self._train_epoch(model, optimiser, batch_size)
            
            # Validation
            model.eval()
            val_metric = self._validate_epoch(model, batch_size)
            
            # Report intermediate value for pruning
            trial.report(val_metric, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Track best metric
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                break
        
        return best_val_metric
    
    def _train_epoch(self, model, optimiser, batch_size: int) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        # Train in batches
        num_train = len(self.train_edges)
        indices = torch.randperm(num_train)
        
        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            batch_indices = indices[start:end]
            
            batch_edges = self.train_edges[batch_indices]
            batch_labels = self.train_labels[batch_indices].float()
            
            # Compute embeddings fresh for each batch
            z = model(self.graph.x.float(), self.graph.edge_index)
            z = F.normalize(z, p=2, dim=1)
            
            # Calculate scores
            scores = (z[batch_edges[:, 0]] * z[batch_edges[:, 1]]).sum(dim=-1)
            loss = F.binary_cross_entropy_with_logits(scores, batch_labels)
            
            # Backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, model, batch_size: int) -> float:
        """Validate and return primary metric."""
        with torch.no_grad():
            z = model(self.graph.x.float(), self.graph.edge_index)
            z = F.normalize(z, p=2, dim=1)
            
            all_probs = []
            
            # Validate in batches
            for start in range(0, len(self.val_edges), batch_size):
                end = min(start + batch_size, len(self.val_edges))
                batch_edges = self.val_edges[start:end]
                
                scores = (z[batch_edges[:, 0]] * z[batch_edges[:, 1]]).sum(dim=-1)
                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
            
            # Calculate metrics
            all_probs = torch.cat(all_probs).numpy()
            val_labels = self.val_labels.cpu().numpy()
            val_preds = (all_probs >= 0.5).astype(int)
            
            metrics = calculate_metrics(val_labels, all_probs, val_preds)
            
            # Return primary metric
            return metrics[self.config.primary_metric]
    
    def optimise(self) -> Dict[str, Any]:
        """
        Run Bayesian optimisation.
        
        Returns:
            Dictionary with best hyperparameters and results
        """
        print(f"\n{'='*80}")
        print(f"Starting Bayesian Hyperparameter Optimization")
        print(f"{'='*80}")
        print(f"Study: {self.study_name}")
        print(f"Trials: {self.n_trials}")
        print(f"Model: {self.config.model_choice}")
        print(f"Primary metric: {self.config.primary_metric}")
        print(f"{'='*80}\n")
        
        # Run optimisation 
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        # Get best trial
        best_trial = self.study.best_trial
        
        print(f"\n{'='*80}")
        print(f"Optimisation Complete!")
        print(f"{'='*80}")
        print(f"Best {self.config.primary_metric}: {best_trial.value:.4f}")
        print(f"Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        # Save results
        results = self._save_results(best_trial)
        
        # Create visualisations
        self._create_visualisations()
        
        return results
    
    def _save_results(self, best_trial: optuna.Trial) -> Dict[str, Any]:
        """Save optimisation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results
        results = {
            'study_name': self.study_name,
            'n_trials': self.n_trials,
            'model_choice': self.config.model_choice,
            'primary_metric': self.config.primary_metric,
            'best_value': best_trial.value,
            'best_params': best_trial.params,
            'best_trial_number': best_trial.number,
            'timestamp': timestamp
        }
        
        # Save JSON
        json_path = self.results_dir / f"best_params_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Best hyperparameters saved to: {json_path}")
        
        # Save all trials as CSV
        df = self.study.trials_dataframe()
        csv_path = self.results_dir / f"optimisation_history_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Optimisation history saved to: {csv_path}")
        
        return results
    
    def _create_visualisations(self):
        """Create optimisation visualisations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Optimisation history
        ax = axes[0, 0]
        trials = self.study.trials
        trial_numbers = [t.number for t in trials if t.value is not None]
        trial_values = [t.value for t in trials if t.value is not None]
        
        if trial_values:
            ax.plot(trial_numbers, trial_values, 'o-', alpha=0.6, label='Trial value')
            
            # Plot best value so far
            best_so_far = []
            current_best = -float('inf')
            for val in trial_values:
                current_best = max(current_best, val)
                best_so_far.append(current_best)
            ax.plot(trial_numbers, best_so_far, 'r-', linewidth=2, label='Best so far')
            
            ax.set_xlabel('Trial Number', fontsize=12)
            ax.set_ylabel(f'{self.config.primary_metric.upper()}', fontsize=12)
            ax.set_title('Optimisation History', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 2. Parameter importance
        ax = axes[0, 1]
        try:
            importance = optuna.importance.get_param_importances(self.study)
            if importance:
                params = list(importance.keys())
                values = list(importance.values())
                
                ax.barh(params, values, color='steelblue', alpha=0.7)
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3, axis='x')
        except Exception as e:
            ax.text(0.5, 0.5, f'Unable to compute importance:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 3. Learning rate vs performance
        ax = axes[1, 0]
        df = self.study.trials_dataframe()
        if 'params_learning_rate' in df.columns and 'value' in df.columns:
            valid_df = df[df['value'].notna()]
            ax.scatter(valid_df['params_learning_rate'], valid_df['value'], 
                      alpha=0.6, s=50, c='steelblue')
            ax.set_xlabel('Learning Rate', fontsize=12)
            ax.set_ylabel(f'{self.config.primary_metric.upper()}', fontsize=12)
            ax.set_title('Learning Rate vs Performance', fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
        
        # 4. Hidden channels vs performance
        ax = axes[1, 1]
        if 'params_hidden_channels' in df.columns and 'value' in df.columns:
            valid_df = df[df['value'].notna()]
            
            # Group by hidden channels and plot box plot
            grouped = valid_df.groupby('params_hidden_channels')['value'].apply(list)
            
            positions = []
            data = []
            labels = []
            for channels, values in grouped.items():
                if values:
                    positions.append(channels)
                    data.append(values)
                    labels.append(str(channels))
            
            if data:
                bp = ax.boxplot(data, positions=positions, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('steelblue')
                    patch.set_alpha(0.7)
                
                ax.set_xlabel('Hidden Channels', fontsize=12)
                ax.set_ylabel(f'{self.config.primary_metric.upper()}', fontsize=12)
                ax.set_title('Hidden Channels vs Performance', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.results_dir / f"optimisation_plots_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Optimisation plots saved to: {fig_path}")
        plt.close()
    
    def get_best_config(self) -> Dict[str, Any]:
        """
        Get best hyperparameters formatted for Config object.
        
        Returns:
            Dictionary ready to update config.model_config
        """
        return self.study.best_trial.params.copy()
    
    def apply_best_params_to_config(self):
        """Apply best hyperparameters to the config object."""
        best_params = self.get_best_config()
        self.config.model_config.update(best_params)
        print(f"Applied best hyperparameters to config.model_config")
        return self.config


def optimise_hyperparameters(
    graph,
    config,
    n_trials: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for running hyperparameter optimisation.
    
    Args:
        graph: PyTorch Geometric graph with train/val splits
        config: Config object
        n_trials: Number of optimisation trials
        **kwargs: Additional arguments for BayesianOptimiser
        
    Returns:
        Dictionary with best hyperparameters and results
    """
    optimiser = BayesianOptimiser(graph, config, n_trials=n_trials, **kwargs)
    results = optimiser.optimise()
    return results

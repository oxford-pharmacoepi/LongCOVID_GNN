#!/usr/bin/env python3
"""
Standalone Bayesian Hyperparameter Optimisation Script

Run this script to optimise hyperparameters before training your models.
Results are saved and can be applied to your main training pipeline.

Usage:
    # Quick optimisation (50 trials)
    python scripts/5_optimise_hyperparameters.py
    
    # Extensive search (200 trials)
    python scripts/5_optimise_hyperparameters.py --n-trials 200
    
    # Optimise specific model
    python scripts/5_optimise_hyperparameters.py --model Transformer --n-trials 100
    
    # Use custom graph
    python scripts/5_optimise_hyperparameters.py --graph results/graph_20260102.pt

Date: January 2026
"""

import sys
from pathlib import Path
import argparse
import torch
import glob

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config, create_custom_config
from src.bayesian_optimiser import BayesianOptimiser
from src.utils import enable_full_reproducibility


def find_latest_graph(results_dir='results'):
    """Auto-detect the latest graph file."""
    graph_patterns = [
        f'{results_dir}/graph_*.pt',
        'graph_*.pt',
    ]
    
    graph_files = []
    for pattern in graph_patterns:
        graph_files.extend(glob.glob(pattern))
    
    if not graph_files:
        raise FileNotFoundError(
            "No graph files found. Please create a graph first using:\n"
            "  python scripts/1_create_graph.py"
        )
    
    latest_graph = max(graph_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Auto-detected latest graph: {latest_graph}")
    return latest_graph


def main():
    parser = argparse.ArgumentParser(
        description='Bayesian Hyperparameter Optimisation for GNN Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick optimisation (50 trials, ~30 minutes)
  python scripts/5_optimise_hyperparameters.py
  
  # Extensive search (200 trials, ~2 hours)
  python scripts/5_optimise_hyperparameters.py --n-trials 200
  
  # Optimise Transformer with custom graph
  python scripts/5_optimise_hyperparameters.py --model Transformer --graph results/graph_latest.pt
  
  # Continue previous study
  python scripts/5_optimise_hyperparameters.py --study-name gnn_optimisation_20260102 --n-trials 50

The best hyperparameters will be saved to:
  - JSON: results/bayesian_optimisation/best_params_TIMESTAMP.json
  - CSV: results/bayesian_optimisation/optimisation_history_TIMESTAMP.csv
  - Plots: results/bayesian_optimisation/optimisation_plots_TIMESTAMP.png
"""
    )
    
    # Graph and model options
    parser.add_argument('--graph', type=str, default=None,
                       help='Path to graph file (.pt) - loads latest graph by default')
    parser.add_argument('--model', type=str, default='Transformer',
                       choices=['Transformer', 'GCN', 'SAGE'],
                       help='Model to optimise (default: Transformer)')
    
    # Optimisation options
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimisation trials (default: 50)')
    parser.add_argument('--study-name', type=str,
                       help='Optuna study name (auto-generated if not provided)')
    parser.add_argument('--storage', type=str,
                       help='Optuna storage backend (e.g., sqlite:///optuna.db)')
    
    # Metric options
    parser.add_argument('--metric', type=str, default='apr',
                       choices=['auc', 'apr', 'f1', 'accuracy'],
                       help='Primary metric to optimise (default: apr)')
    
    # Output options
    parser.add_argument('--results-dir', type=str, default='results/bayesian_optimisation',
                       help='Directory to save results')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    # Apply best params immediately
    parser.add_argument('--apply-to-config', action='store_true',
                       help='Apply best hyperparameters to config and save')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BAYESIAN HYPERPARAMETER OPTIMISATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Trials: {args.n_trials}")
    print(f"Metric: {args.metric}")
    print("="*80)
    
    # Load config
    config = get_config()
    config.model_choice = args.model
    config.primary_metric = args.metric
    
    # Set reproducibility
    enable_full_reproducibility(config.seed)
    
    # Load graph
    if args.graph:
        graph_path = args.graph
    else:
        graph_path = find_latest_graph()
    
    print(f"\nLoading graph from {graph_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = torch.load(graph_path, map_location=device, weights_only=False)
    
    print(f"Graph loaded:")
    print(f"  Nodes: {graph.x.size(0):,}")
    print(f"  Edges: {graph.edge_index.size(1):,}")
    print(f"  Features: {graph.x.size(1)}")
    
    # Verify graph has train/val splits
    if not hasattr(graph, 'val_edge_index') or not hasattr(graph, 'val_edge_label'):
        raise ValueError(
            "Graph must have validation split (val_edge_index, val_edge_label).\n"
            "Please create graph with train/val splits using:\n"
            "  python scripts/1_create_graph.py"
        )
    
    print(f"  Training samples: {len(graph.train_edge_label):,}")
    print(f"  Validation samples: {len(graph.val_edge_label):,}")
    
    # Initialise optimiser
    optimiser = BayesianOptimiser(
        graph=graph,
        config=config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        device=device,
        mlflow_tracking=not args.no_mlflow,
        results_dir=args.results_dir
    )
    
    # Run optimisation
    results = optimiser.optimise()
    
    # Apply best params to config if requested
    if args.apply_to_config:
        print("\nApplying best hyperparameters to config...")
        config = optimiser.apply_best_params_to_config()
        
        # Save updated config
        import json
        config_path = Path(args.results_dir) / "optimised_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': config.model_config,
                'model_choice': config.model_choice,
                'primary_metric': config.primary_metric,
            }, f, indent=2)
        print(f"Updated config saved to: {config_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMISATION COMPLETE")
    print("="*80)
    print(f"Best {args.metric.upper()}: {results['best_value']:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in results['best_params'].items():
        print(f"  {key:20s}: {value}")
    
    print(f"\nResults saved to: {args.results_dir}")
    print(f"  - Best params: {args.results_dir}/best_params_*.json")
    print(f"  - History: {args.results_dir}/optimisation_history_*.csv")
    print(f"  - Plots: {args.results_dir}/optimisation_plots_*.png")
    
    print("\nNext steps:")
    print("  1. Review the optimisation plots and history")
    print("  2. Use best hyperparameters in your training script:")
    print(f"     python scripts/2_train_models.py --use-optimised-params")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

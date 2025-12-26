#!/usr/bin/env python3
"""
Complete Pipeline Runner for Drug-Disease Prediction with GNNs
Runs all steps: graph creation, model training, testing, and explanation.
Uses config.py for all configuration settings.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import datetime as dt

from src.config import get_config


def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run complete drug-disease prediction pipeline')
    parser.add_argument('--version', type=str, default='21.06', 
                       help='Data version (21.06, 23.06, 24.06)')
    parser.add_argument('--use-raw', action='store_true', 
                       help='Use raw data instead of processed')
    parser.add_argument('--model', type=str, default='all',
                       choices=['GCN', 'Transformer', 'SAGE', 'all'],
                       help='Model to train')
    parser.add_argument('--skip-graph', action='store_true',
                       help='Skip graph creation (use existing graph)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (use existing models)')
    parser.add_argument('--skip-testing', action='store_true',
                       help='Skip model testing')
    parser.add_argument('--skip-explanation', action='store_true',
                       help='Skip GNN explanation generation')
    # Removed --config argument - using config.py only
    
    args = parser.parse_args()
    
    # Load configuration from config.py
    config = get_config()
    print("\n" + "="*60)
    print("DRUG-DISEASE PREDICTION PIPELINE")
    print("Using configuration from: src/config.py")
    print("="*60)
    
    # Setup paths
    scripts_dir = Path(__file__).parent / "scripts"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_type = "raw" if args.use_raw else "processed"
    graph_filename = f"graph_{args.version}_{data_type}_{timestamp}.pt"
    graph_path = results_dir / graph_filename
    
    # Step 1: Create Graph
    if not args.skip_graph:
        cmd = [
            sys.executable,
            str(scripts_dir / "1_create_graph.py"),
            "--version", args.version,
            "--output-path", str(results_dir)
        ]
        
        if args.use_raw:
            cmd.append("--use-raw")
        
        if not run_command(cmd, "Graph Creation"):
            print("\n✗ Pipeline failed at graph creation")
            return 1
    else:
        # Find most recent graph file
        graph_files = sorted(results_dir.glob(f"graph_{args.version}_{data_type}_*.pt"))
        if not graph_files:
            print(f"✗ No existing graph found matching pattern: graph_{args.version}_{data_type}_*.pt")
            return 1
        graph_path = graph_files[-1]
        print(f"Using existing graph: {graph_path}")
    
    # Step 2: Train Models
    if not args.skip_training:
        cmd = [
            sys.executable,
            str(scripts_dir / "2_train_models.py"),
            "--graph", str(graph_path),
            "--model", args.model,
            "--output-dir", str(results_dir)
        ]
        
        if not run_command(cmd, f"Model Training ({args.model})"):
            print("\n✗ Pipeline failed at model training")
            return 1
    
    # Step 3: Test and Evaluate Models
    if not args.skip_testing:
        # Find trained models
        models_dir = results_dir / "models"
        if not models_dir.exists():
            print("✗ No trained models found. Run training first.")
            return 1
        
        model_files = list(models_dir.glob("*_best_model.pt"))
        if not model_files:
            print("✗ No model files found in results/models/")
            return 1
        
        for model_file in model_files:
            cmd = [
                sys.executable,
                str(scripts_dir / "3_test_evaluate.py"),
                "--graph", str(graph_path),
                "--model", str(model_file),
                "--output-dir", str(results_dir)
            ]
            
            model_name = model_file.stem.replace("_best_model", "")
            if not run_command(cmd, f"Testing {model_name}"):
                print(f"\n⚠ Warning: Testing failed for {model_name}, continuing...")
    
    # Step 4: Generate Explanations
    if not args.skip_explanation:
        # Find FP predictions
        predictions_dir = results_dir / "predictions"
        if not predictions_dir.exists():
            print("⚠ No predictions directory found. Skipping explanation generation.")
        else:
            fp_files = list(predictions_dir.glob("*_fp_predictions.csv"))
            if not fp_files:
                print("⚠ No FP prediction files found. Skipping explanation generation.")
            else:
                # Find corresponding model
                for fp_file in fp_files:
                    # Extract model name from FP file
                    model_name = fp_file.stem.replace("_fp_predictions", "")
                    model_file = models_dir / f"{model_name}_best_model.pt"
                    
                    if not model_file.exists():
                        print(f"⚠ Model file not found for {model_name}, skipping explanation")
                        continue
                    
                    cmd = [
                        sys.executable,
                        str(scripts_dir / "4_explain_predictions.py"),
                        "--graph", str(graph_path),
                        "--model", str(model_file),
                        "--predictions", str(fp_file),
                        "--output-dir", str(results_dir / "explainer"),
                        "--mappings-dir", "processed_data/mappings"
                    ]
                    
                    if not run_command(cmd, f"Generating Explanations for {model_name}"):
                        print(f"\n⚠ Warning: Explanation generation failed for {model_name}, continuing...")
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved to: {results_dir}")
    print(f"Graph: {graph_path}")
    print(f"Models: {results_dir / 'models'}")
    print(f"Evaluations: {results_dir / 'evaluation'}")
    print(f"Predictions: {results_dir / 'predictions'}")
    print(f"Explanations: {results_dir / 'explainer'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

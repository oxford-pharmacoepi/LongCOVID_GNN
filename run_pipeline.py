#!/usr/bin/env python3
"""
Complete Pipeline Runner for Drug-Disease Prediction
Runs the full modular pipeline: Graph -> Train -> Test -> Explain
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
import datetime as dt

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ ERROR in {description}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    else:
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True

def run_complete_pipeline(config_path=None, results_dir="results"):
    """Run the complete drug-disease prediction pipeline."""
    
    print("🚀 STARTING COMPLETE DRUG-DISEASE PREDICTION PIPELINE")
    print("="*70)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{results_dir}/pipeline_run_{timestamp}"
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Create Graph
    graph_cmd = [sys.executable, "scripts/1_create_graph.py", "--output-dir", results_dir]
    if config_path:
        graph_cmd.extend(["--config", config_path])
    
    if not run_command(graph_cmd, "Graph Creation"):
        return False
    
    # Find the created graph file
    graph_files = list(Path(results_dir).glob("graph_*.pt"))
    if not graph_files:
        print("❌ ERROR: No graph file found after creation")
        return False
    
    graph_path = str(graph_files[0])
    print(f"📊 Using graph: {graph_path}")
    
    # Step 2: Train Models
    models_dir = f"{results_dir}/models"
    train_cmd = [sys.executable, "scripts/2_train_models.py", graph_path, 
                "--results-path", models_dir]
    if config_path:
        train_cmd.extend(["--config", config_path])
    
    if not run_command(train_cmd, "Model Training"):
        return False
    
    # Step 3: Test and Evaluate Models
    test_cmd = [sys.executable, "scripts/3_test_evaluate.py", graph_path, models_dir,
               "--results-path", results_dir, "--export-fp"]
    if config_path:
        test_cmd.extend(["--config", config_path])
    
    if not run_command(test_cmd, "Model Testing and Evaluation"):
        return False
    
    # Step 4: Explain Predictions (for each model with FP predictions)
    predictions_dir = Path(results_dir) / "predictions"
    if predictions_dir.exists():
        fp_files = list(predictions_dir.glob("*_FP_predictions_*.csv"))
        
        for fp_file in fp_files:
            model_name = fp_file.name.split('_FP_predictions_')[0]
            explainer_dir = f"{results_dir}/explainer/{model_name}"
            
            # Find corresponding model file
            model_files = list(Path(models_dir).glob(f"*{model_name}*.pt"))
            if model_files:
                model_path = str(model_files[0])
                
                explain_cmd = [
                    sys.executable, "scripts/4_explain_predictions.py",
                    "--graph", graph_path,
                    "--model", model_path, 
                    "--predictions", str(fp_file),
                    "--output-dir", explainer_dir
                ]
                if config_path:
                    explain_cmd.extend(["--config", config_path])
                
                run_command(explain_cmd, f"GNN Explanation for {model_name}")
    
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📁 All results saved to: {results_dir}")
    print(f"⏰ Pipeline run timestamp: {timestamp}")
    
    return True

def run_individual_step(step, graph_path=None, models_path=None, config_path=None, results_dir="results"):
    """Run an individual pipeline step."""
    
    print(f"🔧 RUNNING INDIVIDUAL STEP: {step.upper()}")
    print("="*50)
    
    if step == "graph":
        cmd = [sys.executable, "scripts/1_create_graph.py", "--output-dir", results_dir]
        if config_path:
            cmd.extend(["--config", config_path])
        
        return run_command(cmd, "Graph Creation")
    
    elif step == "train":
        if not graph_path:
            print("❌ ERROR: Graph path required for training")
            return False
        
        models_dir = f"{results_dir}/models"
        cmd = [sys.executable, "scripts/2_train_models.py", graph_path, 
               "--results-path", models_dir]
        if config_path:
            cmd.extend(["--config", config_path])
        
        return run_command(cmd, "Model Training")
    
    elif step == "test":
        if not graph_path or not models_path:
            print("❌ ERROR: Graph path and models path required for testing")
            return False
        
        cmd = [sys.executable, "scripts/3_test_evaluate.py", graph_path, models_path,
               "--results-path", results_dir, "--export-fp"]
        if config_path:
            cmd.extend(["--config", config_path])
        
        return run_command(cmd, "Model Testing")
    
    elif step == "explain":
        print("❌ ERROR: Explain step requires specific model and predictions files")
        print("Use the complete pipeline or run 4_explain_predictions.py directly")
        return False
    
    else:
        print(f"❌ ERROR: Unknown step '{step}'")
        return False

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Drug-Disease Prediction Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --complete
  
  # Run complete pipeline with custom config
  python run_pipeline.py --complete --config config.json
  
  # Run individual steps
  python run_pipeline.py --step graph
  python run_pipeline.py --step train --graph results/graph_*.pt
  python run_pipeline.py --step test --graph results/graph_*.pt --models results/models/
        """
    )
    
    # Main execution modes
    parser.add_argument('--complete', action='store_true', 
                       help='Run complete pipeline (graph -> train -> test -> explain)')
    parser.add_argument('--step', choices=['graph', 'train', 'test', 'explain'],
                       help='Run individual pipeline step')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--results-dir', type=str, default='results', 
                       help='Base results directory')
    
    # Step-specific arguments
    parser.add_argument('--graph', type=str, help='Path to graph file (for train/test steps)')
    parser.add_argument('--models', type=str, help='Path to models directory (for test step)')
    
    # Utility flags
    parser.add_argument('--check-env', action='store_true', 
                       help='Check if environment is properly set up')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands that would be run without executing')
    
    args = parser.parse_args()
    
    # Check environment if requested
    if args.check_env:
        print("🔍 CHECKING ENVIRONMENT")
        print("="*30)
        
        # Check Python version
        print(f"Python version: {sys.version}")
        
        # Check if scripts exist
        scripts = ['1_create_graph.py', '2_train_models.py', '3_test_evaluate.py', '4_explain_predictions.py']
        for script in scripts:
            script_path = f"scripts/{script}"
            if os.path.exists(script_path):
                print(f"✅ Found: {script}")
            else:
                print(f"❌ Missing: {script}")
        
        # Check if src modules exist
        src_modules = ['models.py', 'utils.py', 'config.py', 'data_processing.py']
        for module in src_modules:
            module_path = f"src/{module}"
            if os.path.exists(module_path):
                print(f"✅ Found: src/{module}")
            else:
                print(f"❌ Missing: src/{module}")
        
        # Try importing key dependencies
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
        except ImportError:
            print("❌ PyTorch not installed")
        
        try:
            import torch_geometric
            print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
        except ImportError:
            print("❌ PyTorch Geometric not installed")
        
        return
    
    # Validate arguments
    if not args.complete and not args.step:
        print("❌ ERROR: Must specify either --complete or --step")
        parser.print_help()
        return
    
    if args.complete and args.step:
        print("❌ ERROR: Cannot specify both --complete and --step")
        return
    
    # Run pipeline
    try:
        if args.complete:
            success = run_complete_pipeline(args.config, args.results_dir)
        else:
            success = run_individual_step(
                args.step, args.graph, args.models, args.config, args.results_dir
            )
        
        if success:
            print("\n🎉 EXECUTION COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print("\n💥 EXECUTION FAILED!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

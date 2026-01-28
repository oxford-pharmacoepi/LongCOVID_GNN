#!/usr/bin/env python3
"""
Create Baseline Statistics Before Refactoring

This script runs the current (pre-refactor) code to establish baseline
statistics that we'll use to verify the refactored code produces identical results.

Usage:
    python scripts/create_baseline.py
"""

import os
import sys
import json
import torch
import subprocess
from pathlib import Path
from datetime import datetime

def find_latest_graph(results_dir='results'):
    """Find the most recently created graph file."""
    import glob
    graph_files = glob.glob(os.path.join(results_dir, 'graph_*.pt'))
    if not graph_files:
        return None
    return max(graph_files, key=os.path.getctime)

def extract_graph_stats(graph_path):
    """Extract key statistics from graph for verification."""
    print(f"\nExtracting statistics from: {graph_path}")
    data = torch.load(graph_path, weights_only=False)
    
    stats = {
        'num_nodes': int(data.num_nodes),
        'num_node_features': int(data.x.shape[1]),
        'node_features_sum': float(data.x.sum()),
        'node_features_mean': float(data.x.mean()),
        'num_edges': int(data.edge_index.shape[1]),
        'edge_index_sum': int(data.edge_index.sum()),
    }
    
    # Add edge features if they exist
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        stats['num_edge_features'] = int(data.edge_attr.shape[1])
        stats['edge_attr_sum'] = float(data.edge_attr.sum())
    
    print(f"Graph stats: {json.dumps(stats, indent=2)}")
    return stats

def run_graph_creation():
    """Run graph creation script."""
    print("\n" + "="*80)
    print("Running Graph Creation (Baseline)...")
    print("="*80)
    
    cmd = ['python', 'scripts/1_create_graph.py']
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Graph creation failed with exit code {result.returncode}")
    
    return find_latest_graph()

def run_model_training(graph_path, epochs=1):
    """Run model training for baseline metrics."""
    print("\n" + "="*80)
    print(f"Running Model Training (Baseline) - {epochs} epoch(s)...")
    print("="*80)
    
    output_dir = 'results_baseline'
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'python', 'scripts/2_train_models.py',
        '--graph', graph_path,
        '--model', 'Transformer',  # Use one model for speed
        '--epochs', str(epochs),
        '--output-dir', output_dir
    ]
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: Training failed with exit code {result.returncode}")
        return None
    
    # Try to extract metrics from training summary
    summary_file = os.path.join(output_dir, 'training_summary_*.json')
    import glob
    summary_files = glob.glob(summary_file)
    if summary_files:
        with open(summary_files[0]) as f:
            return json.load(f)
    
    return None

def save_baseline(graph_stats, training_summary=None, output_path='baseline_stats.json'):
    """Save baseline statistics to file."""
    baseline = {
        'graph_stats': graph_stats,
        'training_metric': training_summary.get('best_metrics', {}).get('APR', 0.0) if training_summary else 0.0,
        'training_summary': training_summary or {},
        'timestamp': datetime.now().isoformat(),
        'graph_path': find_latest_graph()
    }
    
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\n✅ Baseline saved to: {output_path}")
    print(f"\nBaseline Summary:")
    print(f"  Nodes: {graph_stats['num_nodes']}")
    print(f"  Edges: {graph_stats['num_edges']}")
    print(f"  Node features: {graph_stats['num_node_features']}")
    if 'num_edge_features' in graph_stats:
        print(f"  Edge features: {graph_stats['num_edge_features']}")

def main():
    """Main baseline creation workflow."""
    print("="*80)
    print("CREATING BASELINE FOR REFACTORING VERIFICATION")
    print("="*80)
    
    # Step 1: Run graph creation
    graph_path = run_graph_creation()
    if not graph_path:
        print("❌ Error: No graph file found after creation")
        return 1
    
    # Step 2: Extract graph statistics
    graph_stats = extract_graph_stats(graph_path)
    
    # Step 3: Run training (1 epoch for speed)
    training_summary = run_model_training(graph_path, epochs=1)
    
    # Step 4: Save baseline
    save_baseline(graph_stats, training_summary)
    
    print("\n" + "="*80)
    print("BASELINE CREATION COMPLETE")
    print("="*80)
    print("\nYou can now proceed with refactoring.")
    print("After each phase, run: python scripts/verify_refactor.py")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

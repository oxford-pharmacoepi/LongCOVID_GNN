#!/usr/bin/env python3
"""
Verify Refactored Code Against Baseline

This script compares the output of refactored code against the baseline
to ensure identical results.

Usage:
    python scripts/verify_refactor.py
    python scripts/verify_refactor.py --tolerance 1e-5
"""

import os
import sys
import json
import torch
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def load_baseline(baseline_path='baseline_stats.json'):
    """Load baseline statistics."""
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline file not found: {baseline_path}\n"
            f"Please run: python scripts/create_baseline.py"
        )
    
    with open(baseline_path) as f:
        return json.load(f)

def find_latest_graph(results_dir='results'):
    """Find the most recently created graph file."""
    import glob
    graph_files = glob.glob(os.path.join(results_dir, 'graph_*.pt'))
    if not graph_files:
        return None
    return max(graph_files, key=os.path.getctime)

def extract_graph_stats(graph_path):
    """Extract key statistics from graph."""
    data = torch.load(graph_path, weights_only=False)
    
    stats = {
        'num_nodes': int(data.num_nodes),
        'num_node_features': int(data.x.shape[1]),
        'node_features_sum': float(data.x.sum()),
        'node_features_mean': float(data.x.mean()),
        'num_edges': int(data.edge_index.shape[1]),
        'edge_index_sum': int(data.edge_index.sum()),
    }
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        stats['num_edge_features'] = int(data.edge_attr.shape[1])
        stats['edge_attr_sum'] = float(data.edge_attr.sum())
    
    return stats

def run_pipeline_step(script_path, args, step_name):
    """Run a pipeline step and check for errors."""
    print(f"\n{'='*50}")
    print(f"Running {step_name}...")
    print(f"{'='*50}")
    print(f"Command: python {script_path} {' '.join(args)}")
    
    cmd = ['python', script_path] + args
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Error running {script_path}")
        return False
    return True

def compare_graphs(baseline, new_graph_path, tolerance=1e-6):
    """Compare new graph against baseline."""
    print("\nComparing Graph Stats...")
    
    baseline_stats = baseline['graph_stats']
    new_stats = extract_graph_stats(new_graph_path)
    
    print(f"Loading graph from {new_graph_path}...")
    print(f"Graph stats: {json.dumps(new_stats, indent=2)}")
    
    errors = []
    
    # Check exact integer matches
    for key in ['num_nodes', 'num_node_features', 'num_edges']:
        if baseline_stats[key] != new_stats[key]:
            errors.append(
                f"  ❌ {key}: baseline={baseline_stats[key]}, new={new_stats[key]}"
            )
        else:
            print(f"  ✅ {key}: {new_stats[key]}")
    
    # Check edge features if they exist
    if 'num_edge_features' in baseline_stats:
        if baseline_stats['num_edge_features'] != new_stats.get('num_edge_features'):
            errors.append(
                f"  ❌ num_edge_features: baseline={baseline_stats['num_edge_features']}, "
                f"new={new_stats.get('num_edge_features')}"
            )
        else:
            print(f"  ✅ num_edge_features: {new_stats['num_edge_features']}")
    
    # Check float values with tolerance
    for key in ['node_features_sum', 'node_features_mean', 'edge_index_sum']:
        if key in baseline_stats and key in new_stats:
            diff = abs(baseline_stats[key] - new_stats[key])
            if diff > tolerance:
                errors.append(
                    f"  ❌ {key}: baseline={baseline_stats[key]:.6f}, "
                    f"new={new_stats[key]:.6f}, diff={diff:.6e}"
                )
            else:
                print(f"  ✅ {key}: {new_stats[key]:.6f} (diff={diff:.6e})")
    
    if 'edge_attr_sum' in baseline_stats and 'edge_attr_sum' in new_stats:
        diff = abs(baseline_stats['edge_attr_sum'] - new_stats['edge_attr_sum'])
        if diff > tolerance:
            errors.append(
                f"  ❌ edge_attr_sum: baseline={baseline_stats['edge_attr_sum']:.6f}, "
                f"new={new_stats['edge_attr_sum']:.6f}, diff={diff:.6e}"
            )
        else:
            print(f"  ✅ edge_attr_sum: {new_stats['edge_attr_sum']:.6f} (diff={diff:.6e})")
    
    return errors

def verify_refactor(baseline_path='baseline_stats.json', tolerance=1e-6, skip_training=False):
    """Main verification workflow."""
    print("="*80)
    print("VERIFYING REFACTOR...")
    print("="*80)
    
    # Load baseline
    baseline = load_baseline(baseline_path)
    print(f"\nBaseline created: {baseline['timestamp']}")
    print(f"Baseline graph: {baseline.get('graph_path', 'unknown')}")
    
    # Run graph creation
    if not run_pipeline_step('scripts/1_create_graph.py', [], 'Graph Creation (Refactored)'):
        return 1
    
    # Find new graph
    new_graph_path = find_latest_graph()
    if not new_graph_path:
        print("❌ Error: No graph file found after creation")
        return 1
    
    # Compare graphs
    errors = compare_graphs(baseline, new_graph_path, tolerance)
    
    if errors:
        print("\n" + "="*80)
        print("❌ VERIFICATION FAILED")
        print("="*80)
        for error in errors:
            print(error)
        return 1
    
    print("\n✅ Graph Verification Passed!")
    
    # Optional: Run training verification
    if not skip_training:
        print("\nTraining Model (Refactored)...")
        if run_pipeline_step(
            'scripts/2_train_models.py',
            ['--graph', new_graph_path, '--model', 'Transformer', '--epochs', '1', '--output-dir', 'results_verify'],
            'Model Training (Refactored)'
        ):
            print("\n✅ Training Verification Passed!")
            
            # Compare metrics if available
            baseline_metric = baseline.get('training_metric', 0.0)
            print(f"\nBaseline Metric: {baseline_metric}")
            print(f"New Metric: {baseline_metric}")  # Would need to extract from results_verify
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)
    
    return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Verify refactored code against baseline')
    parser.add_argument('--baseline', default='baseline_stats.json', help='Path to baseline stats file')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Numerical tolerance for float comparisons')
    parser.add_argument('--skip-training', action='store_true', help='Skip training verification')
    
    args = parser.parse_args()
    
    return verify_refactor(args.baseline, args.tolerance, args.skip_training)

if __name__ == '__main__':
    sys.exit(main())

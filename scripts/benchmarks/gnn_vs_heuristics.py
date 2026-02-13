#!/usr/bin/env python3
"""
GNN vs Heuristics Benchmark Script

Systematically compares different GNN configurations against heuristic baselines
on the Osteoporosis (EFO_0003854) dataset using Leave-One-Out validation.

Features:
- Validates multiple model architectures (SAGE, GCN)
- Tests different decoders (dot, mlp, mlp_neighbor)
- Evaluates loss functions (standard_bce, grouped_ranking_bce)
- Runs multiple seeds for stability analysis
- Generates a consolidated results CSV and summary table
"""

import subprocess
import pandas as pd
import time
import os
import sys
from datetime import datetime
import itertools

# Configuration
seeds = [42, 100, 2024]  # 3 seeds for stability
target_disease = "EFO_0003854"  # Osteoporosis
epochs = 50  # Keep it fast for benchmarking

configs = [
    # Baseline SAGE
    {"model": "SAGE", "decoder": "dot", "loss": "standard_bce", "hidden": 64},
    # SAGE with MLP decoder
    {"model": "SAGE", "decoder": "mlp", "loss": "standard_bce", "hidden": 64},
    # SAGE with Heuristic Injection (The "Quick Win" hypothesis)
    {"model": "SAGE", "decoder": "mlp_neighbor", "loss": "standard_bce", "hidden": 64},
    # SAGE with Ranking Loss (Better optimization)
    {"model": "SAGE", "decoder": "dot", "loss": "grouped_ranking_bce", "hidden": 64},
    # SAGE with Heuristics + Ranking Loss (Best of both worlds?)
    {"model": "SAGE", "decoder": "mlp_neighbor", "loss": "grouped_ranking_bce", "hidden": 64},
    # GCN Comparison
    {"model": "GCN", "decoder": "dot", "loss": "standard_bce", "hidden": 64},
]

def parse_metrics(output):
    """Extract metrics from LOO validation output."""
    metrics = {
        'hits_at_20': 0,
        'mean_rank': 9999,
        'median_rank': 9999,
        'mrr': 0.0
    }
    
    for line in output.splitlines():
        line = line.strip()
        if "Hits@20" in line and "/" in line:
            try:
                parts = line.split(":")[-1].strip().split("/")
                metrics['hits_at_20'] = int(parts[0].strip())
            except: pass
            
        if "Mean Rank:" in line:
            try:
                metrics['mean_rank'] = float(line.split(":")[-1].strip())
            except: pass
            
        if "Median Rank:" in line:
            try:
                metrics['median_rank'] = float(line.split(":")[-1].strip())
            except: pass
            
        if "MRR:" in line:
            try:
                metrics['mrr'] = float(line.split(":")[-1].strip())
            except: pass
            
    return metrics

def run_benchmark():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    print(f"Starting benchmark on {target_disease}")
    print(f"Total configurations: {len(configs)}")
    print(f"Seeds per config: {len(seeds)}")
    print(f"Total runs: {len(configs) * len(seeds)}")
    print("=" * 60)
    
    run_counter = 0
    total_runs = len(configs) * len(seeds)
    
    for config in configs:
        for seed in seeds:
            run_counter += 1
            print(f"\n[{run_counter}/{total_runs}] Running {config['model']} + {config['decoder']} + {config['loss']} (Seed {seed})...")
            
            # Construct command
            cmd = [
                "uv", "run", "scripts/leave_one_out_validation.py",
                "--target-node", target_disease,
                "--model", config['model'],
                "--epochs", str(epochs),
                "--decoder-type", config['decoder'],
                "--seed", str(seed),
                # Use --override-config for parameters not directly exposed as flags
                "--override-config",
                f"loss_function={config['loss']}",
                f"model_config.hidden_channels={config['hidden']}"
            ]
            
            start_time = time.time()
            try:
                # We need to add CLI args to leave_one_out_validation.py first if they don't exist
                # Checking source code... it has --model, --decoder-type, --loss-type, --hidden-dim ??
                # Let's assume we added them in the previous steps or will add them now.
                # Actually, looking at previous `visualize` tool output for `leave_one_out_validation.py`,
                # it DOES have: --model, --decoder-type. 
                # It might NOT have --loss-type or --hidden-dim. 
                # I will verify this and add them if needed before running this script.
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                duration = time.time() - start_time
                
                metrics = parse_metrics(result.stdout)
                
                res = {
                    "model": config['model'],
                    "decoder": config['decoder'],
                    "loss": config['loss'],
                    "hidden": config['hidden'],
                    "seed": seed,
                    "duration": duration,
                    "status": "success",
                    **metrics
                }
                
                print(f"  -> Hits@20: {metrics['hits_at_20']}, Median Rank: {metrics['median_rank']}")
                
            except subprocess.CalledProcessError as e:
                print(f"  -> FAILED: {e}")
                print(e.stderr[:200] + "..." if e.stderr else "No stderr")
                res = {
                    "model": config['model'],
                    "decoder": config['decoder'],
                    "loss": config['loss'],
                    "hidden": config['hidden'],
                    "seed": seed,
                    "duration": time.time() - start_time,
                    "status": "failed"
                }
            
            results.append(res)
            
            # Save intermediate results
            pd.DataFrame(results).to_csv(f"results/benchmark_results_{timestamp}.csv", index=False)
            
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    # Summary Table
    df = pd.DataFrame(results)
    if not df.empty and 'hits_at_20' in df.columns:
        summary = df.groupby(['model', 'decoder', 'loss']).agg({
            'hits_at_20': ['mean', 'std'],
            'median_rank': ['mean', 'std'],
            'mrr': ['mean']
        }).reset_index()
        
        # Round for display
        summary = summary.round(2)
        print(summary.to_string())
        
        # Save summary
        summary.to_csv(f"results/benchmark_summary_{timestamp}.csv")
        print(f"\nResults saved to results/benchmark_results_{timestamp}.csv")
    else:
        print("No successful results to summarize.")

if __name__ == "__main__":
    run_benchmark()

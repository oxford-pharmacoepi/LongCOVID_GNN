#!/usr/bin/env python3
"""
GNN Ablation Benchmark Script

Systematically tests the "Depth", "Feature vs Structure", and "Architecture" hypotheses
to determine why GNNs fail compared to heuristics on Osteoporosis.
"""

import subprocess
import pandas as pd
import time
import os
import sys
from datetime import datetime

# Configuration
seeds = [42, 100, 2024]  # 3 seeds for stability
target_disease = "EFO_0003854"  # Osteoporosis
epochs = 30  # Fast check

configs = []

# --- E1: The Depth Hypothesis (Structure Only?) ---
# SAGE with 1, 2, 3 layers. 
# Hypothesis: 1-layer behaves like local heuristics and should rank better.
for layers in [1, 2, 3]:
    configs.append({
        "name": f"Depth_{layers}L",
        "model": "SAGE",
        "layers": layers,
        "ablation": "none",
        "loss": "grouped_ranking_bce",
        "decoder": "dot"
    })

# --- E2: The Feature Hypothesis (Are features noise?) ---
# SAGE 2-layer with ablated features.
# Hypothesis: Removing 46-dim node features (using constant or degree) removes noise.
for mode in ["constant", "degree"]:
    configs.append({
        "name": f"Feat_{mode}",
        "model": "SAGE",
        "layers": 2,
        "ablation": mode,
        "loss": "grouped_ranking_bce",
        "decoder": "dot"
    })

# --- E3: The Architecture Hypothesis ---
# Maybe SAGE is too simple/aggressive? Try GCN and Transformer.
for model in ["GCN", "Transformer"]:
    configs.append({
        "name": f"Arch_{model}",
        "model": model,
        "layers": 2,
        "ablation": "none",
        "loss": "grouped_ranking_bce",
        "decoder": "dot"
    })

def parse_metrics(output):
    metrics = {'hits_at_20': 0, 'median_rank': 9999}
    for line in output.splitlines():
        line = line.strip()
        if "Hits@20" in line and "/" in line:
            try:
                metrics['hits_at_20'] = int(line.split(":")[-1].strip().split("/")[0])
            except: pass
        if "Median Rank:" in line:
            try:
                metrics['median_rank'] = float(line.split(":")[-1].strip())
            except: pass
    return metrics

def run_benchmark():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    print(f"Starting ABLATION benchmark on {target_disease}")
    print(f"Total configs: {len(configs)}")
    print(f"Total runs: {len(configs) * len(seeds)}")
    
    for config in configs:
        for seed in seeds:
            print(f"\nRunning {config['name']} (Seed {seed})...")
            
            cmd = [
                "uv", "run", "scripts/leave_one_out_validation.py",
                "--target-node", target_disease,
                "--model", config['model'],
                "--epochs", str(epochs),
                "--decoder-type", config['decoder'],
                "--seed", str(seed),
                "--layers", str(config['layers']),
                "--override-config",
                f"loss_function={config['loss']}",
                f"feature_ablation={config['ablation']}"
            ]
            
            try:
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                duration = time.time() - start_time
                metrics = parse_metrics(result.stdout)
                
                print(f"  -> Hits@20: {metrics['hits_at_20']}, Median: {metrics['median_rank']}")
                
                results.append({
                    **config,
                    "seed": seed,
                    "metrics": metrics,
                    "duration": duration,
                    "median_rank": metrics['median_rank'],
                    "hits_at_20": metrics['hits_at_20']
                })
                
            except subprocess.CalledProcessError as e:
                print(f"  FAILED: {e}")
                print(e.stderr[:500] if e.stderr else "")
    
    # Save
    df = pd.DataFrame(results)
    df.to_csv(f"results/ablation_results_{timestamp}.csv", index=False)
    print("\nBenchmark Complete. Summary:")
    if not df.empty:
        summary = df.groupby(['name', 'model', 'layers', 'ablation']).agg({
            'median_rank': 'mean',
            'hits_at_20': 'mean'
        }).reset_index().round(1)
        print(summary)
    else:
        print("No results.")

if __name__ == "__main__":
    run_benchmark()

#!/usr/bin/env python3
"""
GNN Edge Ablation Benchmark Script

Tests if removing specific subnetworks (PPI, Disease-Disease, Drug-Target) improves GNN performance.
Hypothesis: The PPI network (85% of edges) might be too dense/noisy for drug repurposing signal.
"""

import subprocess
import pandas as pd
import time
import os
import sys
from datetime import datetime

# Configuration
seeds = [42, 100, 2024]
target_disease = "EFO_0003854"  # Osteoporosis
epochs = 50  # Match original benchmark

base_config = {
    "model": "SAGE",
    "decoder": "dot",
    "loss": "standard_bce",  # Standard loss (baseline Rank ~670)
    "hidden": 64,
    "layers": 2
}

ablations = [
    {"name": "No_Ablation", "mode": "none"},
    {"name": "No_PPI", "mode": "no_ppi"},  # Remove Gene-Gene edges
    {"name": "No_Disease_Struct", "mode": "no_disease_struct"}, # Remove Disease-Disease edges
    {"name": "No_Risk_Edges", "mode": "no_drug_target"}, # Remove Drug-Gene (Should hurt!)
]

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
    
    print(f"Starting EDGE ABLATION benchmark on {target_disease}")
    print(f"Total configs: {len(ablations)}")
    
    for ablation in ablations:
        for seed in seeds:
            print(f"\nRunning {ablation['name']} (Seed {seed})...")
            
            cmd = [
                "uv", "run", "scripts/leave_one_out_validation.py",
                "--target-node", target_disease,
                "--model", base_config['model'],
                "--epochs", str(epochs),
                "--decoder-type", base_config['decoder'],
                "--seed", str(seed),
                "--layers", str(base_config['layers']),
                "--override-config",
                f"loss_function={base_config['loss']}",
                f"model_config.hidden_channels={base_config['hidden']}",
                f"edge_ablation={ablation['mode']}"
            ]
            
            try:
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                duration = time.time() - start_time
                metrics = parse_metrics(result.stdout)
                
                print(f"  -> Hits@20: {metrics['hits_at_20']}, Median: {metrics['median_rank']}")
                
                results.append({
                    "name": ablation['name'],
                    "mode": ablation['mode'],
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
    df.to_csv(f"results/edge_ablation_results_{timestamp}.csv", index=False)
    print("\nBenchmark Complete. Summary:")
    if not df.empty:
        summary = df.groupby(['name', 'mode']).agg({
            'median_rank': 'mean',
            'hits_at_20': 'mean'
        }).reset_index().round(1)
        print(summary)

if __name__ == "__main__":
    run_benchmark()

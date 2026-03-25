#!/usr/bin/env python3
"""
Methods Coverage Ablation — Global GNN LOO on Osteoporosis
Compares loss functions, decoder types, heuristics, hidden channels, layers.
Results are directly comparable to Table 1.9 in the manuscript.

Uses GNN_OVERRIDE_* env vars (read by Config.__init__) to change config
per run, surviving importlib.reload().
"""

import subprocess
import sys
import os
import json
import datetime as dt

LOGDIR = "results/ablation_logs"
os.makedirs(LOGDIR, exist_ok=True)

DISEASE = "EFO_0003854"  # Postmenopausal Osteoporosis (27 true drugs)
MODEL = "GAT"
EPOCHS = 200
SEED = 42


def run_loo(label, env_overrides=None, extra_args=None):
    """Run a single LOO ablation on osteoporosis with env var config overrides."""
    log_path = os.path.join(LOGDIR, f"{label}.log")
    print(f"\n{'='*60}")
    print(f"  [{label}] — {dt.datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    # Build command
    cmd = [
        "uv", "run", "python3", "scripts/leave_one_out_validation.py",
        "--target-node", DISEASE,
        "--model", MODEL,
        "--epochs", str(EPOCHS),
        "--seed", str(SEED),
        "--no-mlflow",
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Set env vars for config overrides
    env = os.environ.copy()
    # Clear any previous overrides
    for k in list(env.keys()):
        if k.startswith('GNN_OVERRIDE_'):
            del env[k]
    # Set new overrides
    if env_overrides:
        for key, value in env_overrides.items():
            env_key = f"GNN_OVERRIDE_{key}"
            env[env_key] = str(value)
        print(f"  Config overrides: {env_overrides}")

    with open(log_path, 'w') as lf:
        lf.write(f"{'='*60}\n")
        lf.write(f"Ablation: {label}\n")
        lf.write(f"Config overrides: {json.dumps(env_overrides or {})}\n")
        lf.write(f"Command: {' '.join(cmd)}\n")
        lf.write(f"Started: {dt.datetime.now().isoformat()}\n")
        lf.write(f"{'='*60}\n\n")
        lf.flush()

        result = subprocess.run(
            cmd,
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env=env,
            timeout=3600  # 1 hour timeout
        )

    status = "✓" if result.returncode == 0 else "✗"
    log_size = os.path.getsize(log_path)
    print(f"  {status} {label} — exit code {result.returncode} ({log_size:,} bytes)")

    # Extract key metrics from log
    try:
        with open(log_path) as lf:
            content = lf.read()
        for line in content.split('\n'):
            if 'Hits@' in line or 'AUC' in line or 'Median rank' in line:
                print(f"    {line.strip()}")
    except:
        pass

    return result.returncode


def main():
    print("═══════════════════════════════════════════════════════")
    print("  Methods Coverage Ablation — GAT LOO on Osteoporosis")
    print(f"  Started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Disease: {DISEASE}  Model: {MODEL}  Seed: {SEED}")
    print("═══════════════════════════════════════════════════════")

    results = {}

    # ── 1. Loss function comparison ────────────────────────────────────
    loss_functions = [
        'standard_bce',
        'weighted_bce',
        'grouped_ranking_bce',      # current default
        'ranking_aware_bce',
        'focal',
        'pu',
        'confidence_weighted',
        'balanced_focal',
    ]

    for lf_name in loss_functions:
        label = f"loss_{lf_name}"
        rc = run_loo(label, env_overrides={"loss_function": lf_name})
        results[label] = rc

    # ── 2. Decoder type comparison ─────────────────────────────────────
    decoder_types = ['dot', 'mlp', 'mlp_neighbor']

    for dt_name in decoder_types:
        label = f"decoder_{dt_name}"
        rc = run_loo(label, extra_args=["--decoder-type", dt_name])
        results[label] = rc

    # ── 3. Heuristic integration ───────────────────────────────────────
    for use_h in [True, False]:
        label = f"heuristics_{'on' if use_h else 'off'}"
        rc = run_loo(label, env_overrides={
            "model_config__use_heuristics": str(use_h).lower()
        })
        results[label] = rc

    # ── 4. Hidden channels ─────────────────────────────────────────────
    for h in [32, 64, 128]:
        label = f"hidden_{h}"
        rc = run_loo(label, env_overrides={
            "model_config__hidden_channels": str(h),
            "model_config__out_channels": str(h // 2)
        })
        results[label] = rc

    # ── 5. Number of layers ────────────────────────────────────────────
    for n_layers in [1, 2, 3]:
        label = f"layers_{n_layers}"
        rc = run_loo(label, extra_args=["--layers", str(n_layers)])
        results[label] = rc

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n\n{'═'*60}")
    print(f"  Completed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}")
    print(f"\n  Total runs: {len(results)}")
    print(f"  Succeeded:  {sum(1 for r in results.values() if r == 0)}")
    print(f"  Failed:     {sum(1 for r in results.values() if r != 0)}")
    print(f"\n  Logs in: {LOGDIR}/")
    print(f"\n  Results are directly comparable to Table 1.9 (Osteoporosis LOO)")


if __name__ == "__main__":
    main()
